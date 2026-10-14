"""Scenario sampling and dataset construction.

This module is the single entry point that turns ``configs/env_default.yaml``
into a reproducible dataset of episodes:

    master_seed
      └── scenario_seed = derive_seed(master_seed, "scenario", i)
            ├── env_seed   = derive_seed(scenario_seed, "env", j)    (layout-free rollout seed)
            ├── agent_seed = derive_seed(scenario_seed, "agent", j)  (policy RNG)
            └── goal_seed  = derive_seed(scenario_seed, "goal", j)   (hidden goal assignment)

Every seed is derived with :mod:`hashlib` (never the salted builtin ``hash``),
so ``scenario_id`` and every episode are reproducible across processes and
machines (M2 tests 3e/3f/3h).

Two properties that M2 requires and that ad-hoc test harnesses previously got
wrong are enforced here:

* every scenario carries ``num_candidate_targets`` (K = 4 by default) targets,
  and the focal agent's hidden goal is **re-assigned per episode**, so goal
  recognition is a non-degenerate inference problem;
* obstacles are validated for connectivity, so a path exists between every
  start and every target.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.policy import DEFAULT_BEHAVIOR_WEIGHTS, StochasticPolicy
from ..environment.gridworld import GridWorld
from ..environment.pathfinding import all_cells_connected, cost_map
from ..features.observations import (
    build_condition_a_observation,
    build_condition_b_observation,
)
from .generator import generate_episode
from .scenarios import (
    ScenarioConfig,
    assert_no_scenario_leakage,
    compute_scenario_id,
    split_dataset,
)

SCHEMA_VERSION = "1.0.0"

Position = Tuple[int, int]

DEFAULT_MAX_GENERATION_ATTEMPTS = 200


def derive_seed(master_seed: int, *parts: Any) -> int:
    """Deterministic 63-bit seed derivation via SHA-256 (no builtin hash)."""
    payload = "|".join([str(master_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) >> 1  # 63-bit, always a valid non-negative seed


def config_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DatasetConfig:
    """Typed view of ``configs/env_default.yaml``."""

    grid_width: int = 16
    grid_height: int = 16
    obstacle_density: float = 0.15
    num_candidate_targets: int = 4
    max_steps: int = 50
    deadlock_limit: int = 10
    num_agents: int = 2
    min_dwell_steps: int = 4
    blocked_detour_threshold: int = 2
    temperature: float = 1.0
    persistence_bonus: float = 1.5
    detour_boost: float = 2.0
    wait_bias: float = 0.0
    behavior_weights: Tuple[Tuple[str, float], ...] = tuple(
        sorted(DEFAULT_BEHAVIOR_WEIGHTS.items())
    )
    policy_sensing_radius: int = 3
    min_target_separation: int = 6
    min_start_target_distance: int = 5
    min_start_separation: int = 2
    peer_spawn_radius: int = 4
    peer_avoidance_weight: float = 0.25
    # "open" | "bottleneck" | "crossing" | "mixed" (mixed rotates by scenario index)
    scenario_family: str = "mixed"
    families: Tuple[str, ...] = ("open", "bottleneck", "crossing")
    num_scenarios: int = 100
    episodes_per_scenario: int = 20
    seed: int = 42
    schema_version: str = SCHEMA_VERSION

    # ---------------- construction ----------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DatasetConfig":
        env = dict(raw.get("env", {}))
        agents = dict(raw.get("agents", {}))
        focal = dict(agents.get("focal_agent", {}))
        intention = dict(focal.get("intention", {}))
        weights = dict(focal.get("behavior_weights", {}) or DEFAULT_BEHAVIOR_WEIGHTS)
        others = dict(agents.get("other_agents", {}))
        dataset = dict(raw.get("dataset", {}))
        return cls(
            grid_width=int(env.get("grid_width", cls.grid_width)),
            grid_height=int(env.get("grid_height", cls.grid_height)),
            obstacle_density=float(env.get("obstacle_density", cls.obstacle_density)),
            num_candidate_targets=int(
                env.get("num_candidate_targets", cls.num_candidate_targets)
            ),
            max_steps=int(env.get("max_steps", cls.max_steps)),
            deadlock_limit=int(env.get("deadlock_limit", cls.deadlock_limit)),
            num_agents=int(others.get("count", 1)) + 1,
            peer_spawn_radius=int(env.get("peer_spawn_radius", cls.peer_spawn_radius)),
            peer_avoidance_weight=float(
                env.get("peer_avoidance_weight", cls.peer_avoidance_weight)
            ),
            scenario_family=str(env.get("scenario_family", cls.scenario_family)),
            min_dwell_steps=int(intention.get("min_dwell_steps", cls.min_dwell_steps)),
            blocked_detour_threshold=int(
                intention.get("blocked_detour_threshold", cls.blocked_detour_threshold)
            ),
            behavior_weights=tuple(sorted((k, float(v)) for k, v in weights.items())),
            policy_sensing_radius=int(
                focal.get("policy_sensing_radius", cls.policy_sensing_radius)
            ),
            num_scenarios=int(dataset.get("num_scenarios", cls.num_scenarios)),
            episodes_per_scenario=int(
                dataset.get("episodes_per_scenario", cls.episodes_per_scenario)
            ),
            seed=int(dataset.get("seed", cls.seed)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        import yaml  # local import: keeps the module importable without pyyaml

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    # ---------------- canonicalisation ----------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "obstacle_density": self.obstacle_density,
            "num_candidate_targets": self.num_candidate_targets,
            "max_steps": self.max_steps,
            "deadlock_limit": self.deadlock_limit,
            "num_agents": self.num_agents,
            "min_dwell_steps": self.min_dwell_steps,
            "blocked_detour_threshold": self.blocked_detour_threshold,
            "temperature": self.temperature,
            "persistence_bonus": self.persistence_bonus,
            "detour_boost": self.detour_boost,
            "wait_bias": self.wait_bias,
            "behavior_weights": {k: v for k, v in self.behavior_weights},
            "policy_sensing_radius": self.policy_sensing_radius,
            "min_target_separation": self.min_target_separation,
            "min_start_target_distance": self.min_start_target_distance,
            "min_start_separation": self.min_start_separation,
            "peer_spawn_radius": self.peer_spawn_radius,
            "peer_avoidance_weight": self.peer_avoidance_weight,
            "scenario_family": self.scenario_family,
            "families": list(self.families),
            "schema_version": self.schema_version,
        }

    def config_hash(self) -> str:
        return config_hash(self.to_dict())

    @property
    def behavior_weights_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.behavior_weights}


@dataclass(frozen=True)
class ScenarioSpec:
    """A fully specified, reproducible map layout."""

    scenario_index: int
    scenario_seed: int
    config: ScenarioConfig
    cfg_hash: str

    @property
    def scenario_id(self) -> str:
        return compute_scenario_id(self.config)

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def obstacles(self) -> List[Position]:
        return [tuple(o) for o in self.config.obstacles]

    @property
    def targets(self) -> List[Position]:
        return [tuple(t) for t in self.config.targets]

    @property
    def starts(self) -> List[Position]:
        return [tuple(s) for s in self.config.agent_starts]

    @property
    def num_agents(self) -> int:
        return self.config.num_agents

    def grid(self) -> GridWorld:
        return GridWorld(
            width=self.width,
            height=self.height,
            obstacles=self.obstacles,
            targets=self.targets,
            agent_starts=self.starts,
        )

    def policies(self, agent_seed: int, cfg: DatasetConfig) -> Dict[int, StochasticPolicy]:
        out: Dict[int, StochasticPolicy] = {}
        for a in range(self.num_agents):
            out[a] = StochasticPolicy(
                rng=random.Random(derive_seed(agent_seed, "agent", a)),
                min_dwell_steps=cfg.min_dwell_steps,
                blocked_detour_threshold=cfg.blocked_detour_threshold,
                temperature=cfg.temperature,
                persistence_bonus=cfg.persistence_bonus,
                detour_boost=cfg.detour_boost,
                wait_bias=cfg.wait_bias,
                behavior_weights=cfg.behavior_weights_dict,
            )
        return out

    def spec_dict(self) -> Dict[str, Any]:
        return {
            "scenario_index": self.scenario_index,
            "scenario_seed": self.scenario_seed,
            "scenario_id": self.scenario_id,
            "config_hash": self.cfg_hash,
            "grid_dimensions": [self.width, self.height],
            "static_obstacles": [list(o) for o in self.obstacles],
            "candidate_targets": [list(t) for t in self.targets],
            "agent_starts": [list(s) for s in self.starts],
            "num_agents": self.num_agents,
            # full structural configuration: required so the scenario_id can be
            # recomputed from the serialized record alone (M2 test 3e)
            "agent_configs": [dict(c) for c in self.config.agent_configs],
            "env_params": dict(self.config.env_params),
        }


# --------------------------------------------------------------------------
# scenario sampling
# --------------------------------------------------------------------------
def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def structural_obstacles(
    rng: random.Random, cfg: DatasetConfig, family: str
) -> List[Position]:
    """Layout structure that forces agents to share space (RQ2 measurability).

    ``open``       no structure; interactions come only from clustered spawns.
    ``bottleneck`` a wall across the grid with a single 1-cell gap, so every
                   agent crossing sides must queue through the same cell.
    ``crossing``   a plus-shaped wall whose only opening is the centre cell, so
                   all four quadrants exchange traffic through one cell.

    Connectivity is still validated by the caller; the free ring around the
    structure keeps the space connected by construction.
    """
    w, h = cfg.grid_width, cfg.grid_height
    if family == "open":
        return []
    if family == "bottleneck":
        row = rng.randint(4, max(4, h - 5))
        gap = rng.randint(2, max(2, w - 3))
        return [(x, row) for x in range(1, w - 1) if x != gap]
    if family == "crossing":
        cx, cy = w // 2, h // 2
        vertical = [(cx, y) for y in range(2, h - 2) if y != cy]
        horizontal = [(x, cy) for x in range(2, w - 2) if x != cx]
        return sorted(set(vertical) | set(horizontal))
    raise ValueError(f"unknown scenario family {family!r}")


def resolve_family(cfg: DatasetConfig, scenario_index: int) -> str:
    if cfg.scenario_family != "mixed":
        return cfg.scenario_family
    return cfg.families[scenario_index % len(cfg.families)]


def _sample_obstacles(
    rng: random.Random,
    cfg: DatasetConfig,
    interior_margin: int = 1,
    family: str = "open",
) -> List[Position]:
    """Random obstacle field that keeps the free space fully connected.

    The structural walls for ``family`` are always kept; only the random rubble
    is grown cell by cell under a connectivity check.
    """
    cells = [
        (x, y)
        for y in range(cfg.grid_height)
        for x in range(cfg.grid_width)
        if interior_margin <= x < cfg.grid_width - interior_margin
        and interior_margin <= y < cfg.grid_height - interior_margin
    ]
    structure = structural_obstacles(rng, cfg, family)
    structure_set = set(structure)
    cells = [c for c in cells if c not in structure_set]
    target_count = int(round(cfg.obstacle_density * cfg.grid_width * cfg.grid_height))
    rng.shuffle(cells)
    obstacles: List[Position] = list(structure)
    for cell in cells:
        if len(obstacles) - len(structure) >= target_count:
            break
        candidate = obstacles + [cell]
        if all_cells_connected(cfg.grid_width, cfg.grid_height, candidate, [cell]):
            # cheap full-connectivity check from a fixed anchor cell
            if _free_space_connected(cfg, candidate):
                obstacles.append(cell)
    return sorted(obstacles)


def _free_space_connected(cfg: DatasetConfig, obstacles: Sequence[Position]) -> bool:
    blocked = set(obstacles)
    anchor = None
    for y in range(cfg.grid_height):
        for x in range(cfg.grid_width):
            if (x, y) not in blocked:
                anchor = (x, y)
                break
        if anchor is not None:
            break
    if anchor is None:
        return False
    from ..environment.pathfinding import connected_component, free_cells

    return len(connected_component(cfg.grid_width, cfg.grid_height, blocked, anchor)) == len(
        free_cells(cfg.grid_width, cfg.grid_height, blocked)
    )


def _sample_targets(
    rng: random.Random, cfg: DatasetConfig, obstacles: Sequence[Position]
) -> Optional[List[Position]]:
    from ..environment.pathfinding import free_cells

    cells = free_cells(cfg.grid_width, cfg.grid_height, obstacles)
    rng.shuffle(cells)
    chosen: List[Position] = []
    for cell in cells:
        if all(_manhattan(cell, c) >= cfg.min_target_separation for c in chosen):
            chosen.append(cell)
        if len(chosen) == cfg.num_candidate_targets:
            break
    if len(chosen) != cfg.num_candidate_targets:
        return None
    return sorted(chosen)


def _sample_starts(
    rng: random.Random,
    cfg: DatasetConfig,
    obstacles: Sequence[Position],
    targets: Sequence[Position],
) -> Optional[List[Position]]:
    from ..environment.pathfinding import cost_map, free_cells

    cells = free_cells(cfg.grid_width, cfg.grid_height, obstacles)
    # start cells must be able to reach every target
    reach = {t: cost_map(cfg.grid_width, cfg.grid_height, obstacles, t) for t in targets}
    usable = [c for c in cells if all(c in m for m in reach.values())]
    far = [
        c
        for c in usable
        if min(_manhattan(c, t) for t in targets) >= cfg.min_start_target_distance
    ]
    pool = far or usable
    rng.shuffle(pool)
    if not pool:
        return None
    # The focal agent spawns first; peers spawn inside ``peer_spawn_radius`` of
    # it when possible, so agents genuinely share corridors and the symmetric
    # collision / blocking mechanics (and therefore RQ2) have something to
    # measure. Falls back to the unclustered pool when the neighbourhood is too
    # crowded by obstacles.
    focal = pool[0]
    chosen: List[Position] = [focal]
    near = [
        c
        for c in pool[1:]
        if 0 < _manhattan(c, focal) <= cfg.peer_spawn_radius
    ]
    rng.shuffle(near)
    for cell in near + pool[1:]:
        if cell in chosen:
            continue
        if all(_manhattan(cell, c) >= cfg.min_start_separation for c in chosen):
            chosen.append(cell)
        if len(chosen) == cfg.num_agents:
            break
    if len(chosen) != cfg.num_agents:
        return None
    return sorted(chosen)


def sample_scenario(cfg: DatasetConfig, scenario_index: int, master_seed: int) -> ScenarioSpec:
    """Deterministically sample a validated scenario layout."""
    scenario_seed = derive_seed(master_seed, "scenario", scenario_index)
    rng = random.Random(scenario_seed)
    cfg_hash = cfg.config_hash()
    family = resolve_family(cfg, scenario_index)

    for _ in range(DEFAULT_MAX_GENERATION_ATTEMPTS):
        obstacles = _sample_obstacles(rng, cfg, family=family)
        targets = _sample_targets(rng, cfg, obstacles)
        if targets is None:
            continue
        starts = _sample_starts(rng, cfg, obstacles, targets)
        if starts is None:
            continue
        if not all_cells_connected(
            cfg.grid_width, cfg.grid_height, obstacles, list(targets) + list(starts)
        ):
            continue
        config = ScenarioConfig(
            width=cfg.grid_width,
            height=cfg.grid_height,
            obstacles=tuple(obstacles),
            targets=tuple(targets),
            agent_starts=tuple(starts),
            num_agents=cfg.num_agents,
            agent_configs=tuple(
                [
                    {
                        "planner": "stochastic_policy",
                        "policy_sensing_radius": cfg.policy_sensing_radius,
                        "intention": {
                            "min_dwell_steps": cfg.min_dwell_steps,
                            "stochastic": True,
                            "blocked_detour_threshold": cfg.blocked_detour_threshold,
                        },
                        "behavior_weights": cfg.behavior_weights_dict,
                    }
                ]
                * cfg.num_agents
            ),
            env_params=(
                ("max_steps", cfg.max_steps),
                ("deadlock_limit", cfg.deadlock_limit),
                ("collision_rule", "symmetric"),
                ("scenario_family", family),
            ),
        )
        return ScenarioSpec(
            scenario_index=scenario_index,
            scenario_seed=scenario_seed,
            config=config,
            cfg_hash=cfg_hash,
        )
    raise RuntimeError(
        f"could not sample a valid scenario for index {scenario_index} "
        f"under density={cfg.obstacle_density}, K={cfg.num_candidate_targets}"
    )


# --------------------------------------------------------------------------
# episode / dataset construction
# --------------------------------------------------------------------------
def _assign_goals(
    rng: random.Random, spec: ScenarioSpec, focal_agent_id: int = 0
) -> Dict[int, Position]:
    """Hidden goal assignment: every agent's goal is drawn independently and
    uniformly over the K targets.

    Independence is a validity requirement, not a convenience. If peer goals
    were drawn *without* replacement (the M1 scheme), then with N agents and K
    targets an observer who infers the peers' goals from their trajectories can
    recover the focal goal by elimination — exactly, once N = K. That would make
    RQ2's "interaction-visible" arm win by a counting argument instead of by
    explaining peer-induced perturbations, and would push Condition A toward
    trivial accuracy. With independent draws the peers' destinations carry no
    information about the focal goal, so any Condition A advantage has to come
    from interaction, which is the thing under test. Peers may share a target;
    that also raises congestion, which is what RQ2 needs to be measurable.
    """
    k = len(spec.targets)
    goals: Dict[int, Position] = {}
    for a in range(spec.num_agents):
        goals[a] = spec.targets[rng.randrange(k)]
    return goals


def generate_episode_record(
    spec: ScenarioSpec,
    cfg: DatasetConfig,
    episode_index: int,
    master_seed: int,
    focal_agent_id: int = 0,
) -> Dict[str, Any]:
    """Generate one serialized episode with observable + privileged sections."""
    seeds = {
        "master_seed": master_seed,
        "scenario_seed": spec.scenario_seed,
        "env_seed": derive_seed(spec.scenario_seed, "env", episode_index),
        "agent_seed": derive_seed(spec.scenario_seed, "agent", episode_index),
        "goal_seed": derive_seed(spec.scenario_seed, "goal", episode_index),
    }
    return _generate_with_seeds(spec, cfg, episode_index, seeds, master_seed, focal_agent_id)


def _generate_with_seeds(
    spec: ScenarioSpec,
    cfg: DatasetConfig,
    episode_index: int,
    seeds: Dict[str, int],
    master_seed: int,
    focal_agent_id: int = 0,
) -> Dict[str, Any]:
    env_seed = seeds["env_seed"]
    agent_seed = seeds["agent_seed"]
    goal_seed = seeds["goal_seed"]

    goal_rng = random.Random(goal_seed)
    goals = _assign_goals(goal_rng, spec, focal_agent_id)
    starts = {a: spec.starts[a] for a in range(spec.num_agents)}

    episode = generate_episode(
        scenario_id=spec.scenario_id,
        grid=spec.grid(),
        agents=spec.policies(agent_seed, cfg),
        goals=goals,
        starts=starts,
        max_steps=cfg.max_steps,
        deadlock_limit=cfg.deadlock_limit,
        focal_agent_id=focal_agent_id,
        seed_info=dict(seeds),
        policy_sensing_radius=cfg.policy_sensing_radius,
        peer_avoidance_weight=cfg.peer_avoidance_weight,
    )

    focal_goal = tuple(goals[focal_agent_id])
    focal_goal_id = spec.targets.index(focal_goal)
    focal_steps = [s for s in episode["steps"] if s["agent_id"] == focal_agent_id]
    total_steps = len(focal_steps)

    record: Dict[str, Any] = {
        "schema_version": cfg.schema_version,
        "scenario_id": spec.scenario_id,
        "scenario_index": spec.scenario_index,
        "episode_id": f"{spec.scenario_id}_ep{episode_index:04d}",
        "episode_index": episode_index,
        "config_hash": spec.cfg_hash,
        "scenario_spec": spec.spec_dict(),
        "seeds": episode["seeds"],
        "grid_dimensions": [spec.width, spec.height],
        "static_obstacles": [list(o) for o in spec.obstacles],
        "candidate_targets": [
            {"target_id": i, "position": list(t)} for i, t in enumerate(spec.targets)
        ],
        "num_agents": spec.num_agents,
        "focal_agent_id": focal_agent_id,
        "total_steps": total_steps,
        "termination_reason": episode["termination_reason"],
        "steps": episode["steps"],
        "privileged_ground_truth": {
            "focal_goal_id": focal_goal_id,
            "focal_goal_position": list(focal_goal),
            "focal_initial_intention": focal_steps[0]["intention"] if focal_steps else None,
            "step_intentions": [s["intention"] for s in focal_steps],
            "intended_actions": [s["intended_action"] for s in focal_steps],
            "other_agent_goals": [
                {"agent_id": a, "goal_id": spec.targets.index(tuple(g)), "goal_position": list(g)}
                for a, g in sorted(goals.items())
                if a != focal_agent_id
            ],
            "planner_type": "stochastic_policy",
        },
    }
    return record


def generate_dataset(
    cfg: DatasetConfig,
    master_seed: Optional[int] = None,
    num_scenarios: Optional[int] = None,
    episodes_per_scenario: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate the full dataset deterministically from ``master_seed``."""
    seed = cfg.seed if master_seed is None else int(master_seed)
    n_scen = cfg.num_scenarios if num_scenarios is None else int(num_scenarios)
    n_ep = cfg.episodes_per_scenario if episodes_per_scenario is None else int(episodes_per_scenario)

    records: List[Dict[str, Any]] = []
    for i in range(n_scen):
        spec = sample_scenario(cfg, i, seed)
        for j in range(n_ep):
            records.append(generate_episode_record(spec, cfg, j, seed))
    return records


def reconstruct_episode_record(
    record: Dict[str, Any], cfg: Optional[DatasetConfig] = None
) -> Dict[str, Any]:
    """Rebuild an episode from its serialized metadata alone (M2 test 3e).

    Requires ``scenario_spec`` (layout), ``seeds`` (env/agent/goal) and
    ``config_hash``. Raises if the supplied config does not hash to the recorded
    ``config_hash`` — a mismatch means the episode is not addressable under the
    configuration being used.
    """
    if cfg is None:
        raise ValueError("reconstruction requires the DatasetConfig used for generation")
    if cfg.config_hash() != record.get("config_hash"):
        raise ValueError(
            f"config_hash mismatch: recorded {record.get('config_hash')} != {cfg.config_hash()}"
        )
    spec_dict = record["scenario_spec"]
    for key in ("scenario_index", "scenario_seed", "scenario_id", "config_hash"):
        if key not in spec_dict:
            raise ValueError(f"scenario_spec is missing required metadata field {key!r}")
    seeds = record["seeds"]
    for key in ("env_seed", "agent_seed", "goal_seed"):
        if key not in seeds:
            raise ValueError(f"episode is missing required seed metadata {key!r}")

    config = ScenarioConfig(
        width=spec_dict["grid_dimensions"][0],
        height=spec_dict["grid_dimensions"][1],
        obstacles=tuple(tuple(o) for o in spec_dict["static_obstacles"]),
        targets=tuple(tuple(t) for t in spec_dict["candidate_targets"]),
        agent_starts=tuple(tuple(s) for s in spec_dict["agent_starts"]),
        num_agents=spec_dict["num_agents"],
        agent_configs=tuple(dict(c) for c in spec_dict["agent_configs"]),
        env_params=tuple(sorted(spec_dict["env_params"].items())),
    )
    spec = ScenarioSpec(
        scenario_index=spec_dict["scenario_index"],
        scenario_seed=spec_dict["scenario_seed"],
        config=config,
        cfg_hash=spec_dict["config_hash"],
    )
    if spec.scenario_id != record["scenario_id"]:
        raise ValueError("reconstructed scenario_id does not match the serialized record")
    return _generate_with_seeds(
        spec,
        cfg,
        episode_index=record["episode_index"],
        seeds=seeds,
        master_seed=seeds.get("master_seed", cfg.seed),
        focal_agent_id=record["focal_agent_id"],
    )


# --------------------------------------------------------------------------
# observable / privileged projections, splitting, hashing
# --------------------------------------------------------------------------
PRIVILEGED_TOP_LEVEL_KEYS = (
    "privileged_ground_truth",
    "scenario_spec",
    "seeds",
    "config_hash",
    "scenario_index",
    "episode_index",
    # DATASET_SCHEMA.md §4 lists termination_reason as evaluation-only
    "termination_reason",
)


def observable_record(record: Dict[str, Any], masked: bool = False) -> Dict[str, Any]:
    """Condition A (``masked=False``) or Condition B (``masked=True``) view."""
    condition_a = build_condition_a_observation(record, record)
    return build_condition_b_observation(condition_a) if masked else condition_a


def privileged_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "episode_id": record["episode_id"],
        "scenario_id": record["scenario_id"],
        "termination_reason": record["termination_reason"],
        "privileged_ground_truth": record["privileged_ground_truth"],
    }


def strip_privileged(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove every privileged top-level section (defence in depth)."""
    return {k: v for k, v in record.items() if k not in PRIVILEGED_TOP_LEVEL_KEYS}


def split_records(
    records: List[Dict[str, Any]],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scenario-level split + leakage assertion (fails loudly on overlap)."""
    splits = split_dataset(records, train=train, val=val, test=test, seed=seed)
    assert_no_scenario_leakage(splits)
    return splits


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def dataset_hash(records: Sequence[Dict[str, Any]]) -> str:
    """SHA-256 over the canonical serialization of every episode."""
    hasher = hashlib.sha256()
    for record in records:
        hasher.update(canonical_json(record).encode("utf-8"))
    return hasher.hexdigest()
