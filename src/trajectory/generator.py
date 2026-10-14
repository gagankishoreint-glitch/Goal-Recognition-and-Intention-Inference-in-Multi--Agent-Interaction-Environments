"""Trajectory / episode generation with strict observable-privileged split.

``generate_episode`` is the single rollout primitive. It consumes an explicit
seed-controlled policy per agent and never touches global RNG state, which is
what the M2 reproducibility tests (3a-3g) assert.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..agents.policy import StochasticPolicy
from ..environment.gridworld import GridWorld

Position = tuple


def generate_episode(
    scenario_id: str,
    grid: GridWorld,
    agents: Dict[int, StochasticPolicy],
    goals: Dict[int, Position],
    starts: Dict[int, Position],
    max_steps: int = 50,
    deadlock_limit: int = 10,
    focal_agent_id: int = 0,
    seed_info: Optional[Dict[str, int]] = None,
    policy_sensing_radius: int = 3,
    peer_avoidance_weight: Optional[float] = None,
) -> Dict[str, Any]:
    positions = {a: tuple(p) for a, p in starts.items()}
    for p in agents.values():
        p.reset()
    intention_histories: Dict[int, List[str]] = {a: [] for a in positions}
    blocked_counts: Dict[int, int] = {a: 0 for a in positions}
    blocked_streak: Dict[int, int] = {a: 0 for a in positions}
    episode_steps: List[Dict[str, Any]] = []
    termination = "MAX_STEPS"

    obstacles = sorted(grid.obstacles)
    bounds = (grid.width, grid.height)

    for t in range(max_steps):
        intended: Dict[int, str] = {}
        chosen_intentions: Dict[int, str] = {}
        # Deterministic agent iteration order: no agent-ID priority and no
        # dependence on dict/PYTHONHASHSEED ordering.
        for a in sorted(agents):
            policy = agents[a]
            block_count = blocked_counts[a]
            intention = policy.select_intention(
                goal=tuple(goals[a]),
                current=positions[a],
                history=intention_histories[a],
                blocked_count=block_count,
                obstacles=obstacles,
                bounds=bounds,
            )
            peers = [positions[o] for o in sorted(positions) if o != a]
            action = policy.select_action(
                intention=intention,
                goal=tuple(goals[a]),
                current=positions[a],
                peers=peers,
                policy_sensing_radius=policy_sensing_radius,
                peer_avoidance_weight=peer_avoidance_weight,
                obstacles=obstacles,
                bounds=bounds,
            )
            intended[a] = action
            chosen_intentions[a] = intention
            intention_histories[a].append(intention)

        prev_positions = dict(positions)
        new_positions, realized, blocked = grid.step(positions, intended)
        destinations = {
            a: grid.destination(prev_positions[a], intended[a]) for a in sorted(positions)
        }
        blocked_reasons = {
            a: grid.block_reason(
                prev_positions[a], intended[a], prev_positions, a, destinations
            )
            for a in sorted(positions)
        }
        for a in sorted(positions):
            blocked_counts[a] = blocked_counts[a] + 1 if blocked[a] else 0
            blocked_streak[a] = blocked_streak[a] + 1 if blocked[a] else 0

        for a in sorted(positions):
            episode_steps.append(
                {
                    "step": t,
                    "agent_id": a,
                    # observable: the cell the agent occupied when the
                    # intention/action for this step was chosen
                    "position_before": list(prev_positions[a]),
                    "position": list(new_positions[a]),
                    "realized_action": realized[a],
                    "blocked": bool(blocked[a]),
                    # privileged:
                    "goal": list(goals[a]),
                    "intention": chosen_intentions[a],
                    "intended_action": intended[a],
                    # privileged diagnostic: derived from the intended action
                    "blocked_reason": blocked_reasons[a],
                }
            )
        positions = new_positions

        focal_pos = positions[focal_agent_id]
        if grid.goal_reached(focal_pos, tuple(goals[focal_agent_id])):
            termination = "GOAL_REACHED"
            break
        if all(blocked_streak[a] >= deadlock_limit for a in positions):
            termination = "DEADLOCK_LIMIT"
            break

    return {
        "episode_id": f"{scenario_id}_{seed_info.get('agent_seed', 0) if seed_info else 0}",
        "scenario_id": scenario_id,
        "seeds": seed_info or {},
        "steps": episode_steps,
        "termination_reason": termination,
        "focal_agent_id": focal_agent_id,
        "goals": {a: list(goals[a]) for a in sorted(goals)},
    }


def focal_steps(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Focal-agent steps only, in time order."""
    focal = episode["focal_agent_id"]
    return [s for s in episode["steps"] if s["agent_id"] == focal]


def episode_prefix(episode: Dict[str, Any], n_steps: int) -> Dict[str, Any]:
    """Truncate an episode to its first ``n_steps`` transitions.

    Only observable/privileged fields of the retained steps survive; nothing is
    recomputed, so a prefix is a strict projection of the full episode.
    """
    keep = max(0, int(n_steps))
    steps = [s for s in episode["steps"] if s["step"] < keep]
    out = dict(episode)
    out["steps"] = steps
    out["total_steps"] = len({s["step"] for s in steps})
    return out


# --------------------------------------------------------------------------
# CLI: python -m src.trajectory.generator  (see REPRODUCIBILITY.md §2.1)
# --------------------------------------------------------------------------
def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.trajectory.generator",
        description="Generate the goal-recognition dataset and write it to disk.",
    )
    parser.add_argument("--config", default="configs/env_default.yaml")
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--num_scenarios", type=int, default=None)
    parser.add_argument("--episodes_per_scenario", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="master seed")
    parser.add_argument("--split_seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    import json

    from .dataset import DatasetConfig, generate_dataset, split_records
    from .serialize import write_dataset

    args = _build_arg_parser().parse_args(argv)
    cfg = DatasetConfig.from_yaml(args.config)
    master_seed = cfg.seed if args.seed is None else int(args.seed)
    records = generate_dataset(
        cfg,
        master_seed=master_seed,
        num_scenarios=args.num_scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
    )
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    splits = split_records(records, seed=args.split_seed, **ratios)
    metadata = write_dataset(
        records,
        splits,
        out_dir=args.output_dir,
        master_seed=master_seed,
        config_hash=cfg.config_hash(),
        split_seed=args.split_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "master_seed": master_seed,
                "config_hash": cfg.config_hash(),
                "episodes": metadata["episodes"],
                "splits": {
                    name: {"scenarios": info["scenarios"], "episodes": info["episodes"]}
                    for name, info in metadata["splits"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys as _sys

    raise SystemExit(main(_sys.argv[1:]))
