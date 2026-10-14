"""Dataset tests: schema, scenario sampling, splitting, isolation, determinism."""

from __future__ import annotations

from typing import Dict, Tuple

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.pathfinding import all_cells_connected
from src.features.observations import (
    InformationIsolationError,
    build_condition_a_observation,
    build_condition_b_observation,
    validate_batch,
)
from src.trajectory.dataset import (
    DatasetConfig,
    dataset_hash,
    derive_seed,
    generate_dataset,
    observable_record,
    privileged_record,
    reconstruct_episode_record,
    sample_scenario,
    split_records,
    strip_privileged,
)
from src.trajectory.scenarios import assert_no_scenario_leakage

SMALL = dict(num_scenarios=4, episodes_per_scenario=3)


@pytest.fixture(scope="module")
def cfg() -> DatasetConfig:
    return DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))


@pytest.fixture(scope="module")
def records(cfg):
    return generate_dataset(cfg, master_seed=4242, **SMALL)


def test_config_round_trips_from_yaml(cfg: DatasetConfig) -> None:
    assert cfg.grid_width == 16 and cfg.grid_height == 16
    assert cfg.num_candidate_targets == 4
    assert cfg.max_steps == 50 and cfg.deadlock_limit == 10
    assert cfg.num_agents == 4  # focal agent + other_agents.count (RQ2 densest arm)
    # The interaction parameters that make RQ2 measurable. They are pinned here
    # because a silent revert to the M1 defaults drops the focal peer-blocked
    # rate from ~7% back to ~2.5%, which is what made the question untestable.
    assert cfg.scenario_family == "mixed"
    assert cfg.peer_spawn_radius == 2
    assert cfg.peer_avoidance_weight == 0.05
    assert cfg.policy_sensing_radius == 3
    # ...and they are part of the structural hash, so they cannot drift quietly.
    for key, delta in (
        ("peer_spawn_radius", 1),
        ("policy_sensing_radius", 1),
        ("peer_avoidance_weight", 0.05),
    ):
        moved = DatasetConfig(**{**cfg.__dict__, key: cfg.__dict__[key] + delta})
        assert moved.config_hash() != cfg.config_hash(), f"{key} is not in the config hash"
    weights = dict(cfg.behavior_weights)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert max(weights.values()) <= 0.60, "a dominant intention class trips the triviality flag"
    # the hash is stable and sensitive to the policy configuration
    assert cfg.config_hash() == cfg.config_hash()
    other = DatasetConfig(**{**cfg.__dict__, "min_dwell_steps": cfg.min_dwell_steps + 1})
    assert other.config_hash() != cfg.config_hash()


def test_seed_derivation_is_deterministic_and_distinct() -> None:
    assert derive_seed(42, "scenario", 0) == derive_seed(42, "scenario", 0)
    assert derive_seed(42, "scenario", 0) != derive_seed(42, "scenario", 1)
    assert derive_seed(42, "scenario", 0) != derive_seed(43, "scenario", 0)
    assert 0 <= derive_seed(42, "agent", 0) < 2 ** 63


def test_scenarios_have_k_targets_and_connected_layouts(cfg: DatasetConfig) -> None:
    for index in range(4):
        spec = sample_scenario(cfg, index, 99)
        assert len(spec.targets) == cfg.num_candidate_targets
        assert len(set(spec.targets)) == cfg.num_candidate_targets
        assert len(spec.starts) == cfg.num_agents
        assert len(set(spec.starts)) == cfg.num_agents
        assert all_cells_connected(
            spec.width, spec.height, spec.obstacles, list(spec.targets) + list(spec.starts)
        )
        # targets are separated and starts are not on top of a target
        for i, a in enumerate(spec.targets):
            for b in spec.targets[i + 1 :]:
                assert abs(a[0] - b[0]) + abs(a[1] - b[1]) >= cfg.min_target_separation
        for s in spec.starts:
            assert s not in spec.targets
            assert s not in spec.obstacles
        assert spec.scenario_id == sample_scenario(cfg, index, 99).scenario_id


def test_scenario_ids_differ_across_layouts(cfg: DatasetConfig) -> None:
    ids = {sample_scenario(cfg, i, 7).scenario_id for i in range(6)}
    assert len(ids) == 6


def test_episode_records_carry_required_metadata(cfg: DatasetConfig, records) -> None:
    assert len(records) == SMALL["num_scenarios"] * SMALL["episodes_per_scenario"]
    required_top = {
        "schema_version",
        "scenario_id",
        "episode_id",
        "config_hash",
        "scenario_spec",
        "seeds",
        "grid_dimensions",
        "static_obstacles",
        "candidate_targets",
        "num_agents",
        "focal_agent_id",
        "total_steps",
        "termination_reason",
        "steps",
        "privileged_ground_truth",
    }
    for record in records:
        assert required_top <= set(record), f"missing {required_top - set(record)}"
        assert record["schema_version"] == "1.0.0"
        assert {"env_seed", "agent_seed", "goal_seed", "scenario_seed", "master_seed"} <= set(
            record["seeds"]
        )
        assert len(record["candidate_targets"]) == cfg.num_candidate_targets
        assert record["termination_reason"] in {"GOAL_REACHED", "MAX_STEPS", "DEADLOCK_LIMIT"}
        # JSON serializable (schema is a storage contract)
        json.dumps(record, sort_keys=True)


def test_hidden_goal_varies_across_episodes(cfg: DatasetConfig, records) -> None:
    """Regression guard: the old harness froze the focal goal to targets[0]."""
    goals = {r["privileged_ground_truth"]["focal_goal_id"] for r in records}
    assert len(goals) >= 2, f"degenerate goal distribution: {goals}"
    for record in records:
        gid = record["privileged_ground_truth"]["focal_goal_id"]
        assert record["candidate_targets"][gid]["position"] == record["privileged_ground_truth"][
            "focal_goal_position"
        ]
    for record in records:
        focal = record["privileged_ground_truth"]["focal_goal_id"]
        for peer in record["privileged_ground_truth"]["other_agent_goals"]:
            assert 0 <= peer["goal_id"] < len(record["candidate_targets"])
            assert peer["agent_id"] != record["focal_agent_id"]


def test_peer_goals_carry_no_information_about_the_focal_goal(cfg: DatasetConfig) -> None:
    """Guards the elimination shortcut that would trivialise RQ2's contrast.

    If peer goals were drawn without replacement from the remaining targets, an
    observer that recovers the peers' goals from their trajectories could infer
    the focal goal by counting — exactly, once N = K. Peer goals are therefore
    independent draws, and this test pins that: peers sometimes share the focal
    goal, and the sharing rate matches 1/K.
    """
    import collections

    recs = generate_dataset(cfg, master_seed=20261008, num_scenarios=25, episodes_per_scenario=8)
    k = cfg.num_candidate_targets
    pairs = 0
    shared = 0
    focal_given_peer: Dict[Tuple[int, int], int] = collections.defaultdict(int)
    for record in recs:
        gt = record["privileged_ground_truth"]
        focal = gt["focal_goal_id"]
        for peer in gt["other_agent_goals"]:
            pairs += 1
            shared += int(peer["goal_id"] == focal)
            focal_given_peer[(peer["goal_id"], focal)] += 1
    rate = shared / pairs
    assert pairs > 200
    # 1/K = 0.25 for the shipped K = 4; the window is generous but well clear of
    # both the exclusion scheme (rate = 0) and a degenerate one (rate = 1).
    assert abs(rate - 1.0 / k) < 0.07, f"peer/focal sharing rate {rate:.3f} is not ~1/K"
    assert shared > 0, "peer goals never coincide with the focal goal: elimination shortcut is back"
    # conditional independence in the other direction: the focal goal stays
    # close to uniform whatever goal a peer was heading for
    for peer_goal in range(k):
        col = {f: focal_given_peer.get((peer_goal, f), 0) for f in range(k)}
        total = sum(col.values())
        assert total > 20, f"peer goal {peer_goal} too rare to check ({total} samples)"
        assert max(col.values()) / total < 0.45, f"focal goal not uniform given peer goal {peer_goal}"


def test_steps_are_observable_plus_privileged_and_time_ordered(cfg: DatasetConfig, records) -> None:
    for record in records:
        steps = [s for s in record["steps"] if s["agent_id"] == record["focal_agent_id"]]
        assert [s["step"] for s in steps] == list(range(len(steps)))
        assert len(steps) == record["total_steps"]
        for s in steps:
            assert {"position", "position_before", "realized_action", "blocked"} <= set(s)
            assert {"goal", "intention", "intended_action", "blocked_reason"} <= set(s)
            assert s["realized_action"] in {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}
            assert s["intention"] in {"DIRECT", "DETOUR_LEFT", "DETOUR_RIGHT", "WAIT"}
            assert s["blocked_reason"] in {"none", "obstacle", "boundary", "occupied", "conflict"}
            x, y = s["position"]
            assert 0 <= x < record["grid_dimensions"][0]
            assert 0 <= y < record["grid_dimensions"][1]
            assert tuple(s["position"]) not in {tuple(o) for o in record["static_obstacles"]}
        # the privileged intention list matches the per-step privileged field
        assert record["privileged_ground_truth"]["step_intentions"] == [
            s["intention"] for s in steps
        ]


def test_split_is_scenario_level_and_leak_free(cfg: DatasetConfig, records) -> None:
    splits = split_records(records, seed=7)
    ids = {name: {r["scenario_id"] for r in rows} for name, rows in splits.items()}
    for a in ids:
        for b in ids:
            if a < b:
                assert not (ids[a] & ids[b])
    all_ids = sorted(r["episode_id"] for rows in splits.values() for r in rows)
    assert all_ids == sorted(r["episode_id"] for r in records)

    # a constructed leak must be rejected
    shared = records[0]["scenario_id"]
    with pytest.raises(AssertionError):
        assert_no_scenario_leakage(
            {
                "train": [r for r in records if r["scenario_id"] == shared],
                "test": [r for r in records if r["scenario_id"] == shared],
            }
        )


def test_observable_projection_hides_privileged_state(cfg: DatasetConfig, records) -> None:
    for record in records:
        a = observable_record(record, masked=False)
        b = observable_record(record, masked=True)
        validate_batch(a)
        validate_batch(b)
        assert set(b) < set(a)
        assert "peer_agents_trajectory" not in b
        for entry in b["focal_trajectory"]:
            assert "blocked" not in entry
        with pytest.raises(InformationIsolationError):
            validate_batch({**a, "focal_goal_id": 1})
        with pytest.raises(InformationIsolationError):
            validate_batch({**a, "step_intentions": ["DIRECT"]})
        # the privileged payload is disjoint from the observable view (the two
        # shared identifiers, episode_id and scenario_id, are observable by
        # DATASET_SCHEMA §4)
        priv = privileged_record(record)
        assert set(priv) - {"episode_id", "scenario_id"} == {"termination_reason", "privileged_ground_truth"}
        assert (set(priv) - {"episode_id", "scenario_id"}) & set(a) == set()


def test_dataset_is_reproducible_and_addressable(cfg: DatasetConfig, records) -> None:
    again = generate_dataset(cfg, master_seed=4242, **SMALL)
    assert dataset_hash(again) == dataset_hash(records)
    assert again == records
    different = generate_dataset(cfg, master_seed=4243, **SMALL)
    assert dataset_hash(different) != dataset_hash(records)
    for record in records[:4]:
        rebuilt = reconstruct_episode_record(record, cfg)
        assert rebuilt["steps"] == record["steps"]
    with pytest.raises(ValueError):
        reconstruct_episode_record(
            records[0], DatasetConfig(**{**cfg.__dict__, "max_steps": cfg.max_steps + 1})
        )


def test_strip_privileged_removes_every_privileged_section(cfg: DatasetConfig, records) -> None:
    for record in records:
        stripped = strip_privileged(record)
        for key in (
            "privileged_ground_truth",
            "scenario_spec",
            "seeds",
            "config_hash",
            "termination_reason",
        ):
            assert key not in stripped
        assert stripped["scenario_id"] == record["scenario_id"]


def test_observations_survive_a_json_round_trip(cfg: DatasetConfig, records) -> None:
    record = records[0]
    a = observable_record(record, masked=False)
    reloaded = json.loads(json.dumps(a, sort_keys=True))
    assert reloaded == a
    validate_batch(reloaded)
    assert build_condition_b_observation(reloaded) == observable_record(record, masked=True)
    assert build_condition_a_observation(record, record) == a
