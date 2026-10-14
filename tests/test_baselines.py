"""Baseline tests: distribution contracts, CPU-only path, and the ordering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.intentions import ALL_INTENTIONS
from src.evaluation.baselines import (
    BayesianBeliefBaseline,
    DistanceHeuristicBaseline,
    UniformBaseline,
    belief_trajectory,
    observation_stream,
)
from src.trajectory.dataset import generate_dataset, observable_record, split_records

TARGETS = [(1, 1), (1, 14), (14, 1), (14, 14)]
BOUNDS = (16, 16)


@pytest.fixture(scope="module")
def small_split():
    from verification.m2.dataset_m2 import verification_config

    records = generate_dataset(
        verification_config(), master_seed=2024, num_scenarios=6, episodes_per_scenario=3
    )
    return split_records(records, seed=3)


def is_distribution(values) -> bool:
    arr = np.asarray(list(values), dtype=float)
    return bool(arr.min() >= 0.0 and abs(arr.sum() - 1.0) < 1e-9)


def test_uniform_baseline_is_uniform_and_fairly_random() -> None:
    baseline = UniformBaseline(TARGETS, seed=0)
    dist = baseline.goal_distribution()
    assert len(dist) == 4
    assert all(abs(p - 0.25) < 1e-5 for p in dist)
    assert is_distribution(dist)
    # the seeded jitter makes top-1 a fair guess instead of a constant class
    baseline.reset()
    picks = [int(np.argmax(baseline.goal_distribution())) for _ in range(4000)]
    counts = np.bincount(picks, minlength=4) / 4000
    assert counts.max() - counts.min() < 0.06, counts
    assert len(set(picks)) == 4
    # reset restarts the same stream: reproducible
    baseline.reset()
    assert [int(np.argmax(baseline.goal_distribution())) for _ in range(5)] == picks[:5]


def test_distance_baseline_is_a_distribution_and_prefers_the_nearest_target() -> None:
    baseline = DistanceHeuristicBaseline(TARGETS, beta=0.5, bounds=BOUNDS, distance_metric="manhattan")
    assert is_distribution(baseline.goal_distribution())  # prior before any update
    baseline.update((0, 0), (2, 2))
    dist = baseline.goal_distribution()
    assert is_distribution(dist)
    assert int(np.argmax(dist)) == 0, "closest target must receive the most mass"
    baseline.update((2, 2), (13, 13))
    assert int(np.argmax(baseline.goal_distribution())) == 3
    assert is_distribution(baseline.intention_distribution())


def test_distance_baseline_uses_shortest_paths_around_obstacles() -> None:
    """Manhattan and shortest-path metrics must disagree when a wall intervenes."""
    wall = [(8, y) for y in range(0, 15)]  # vertical wall with a gap at y = 15
    targets = [(2, 7), (14, 7)]
    manhattan = DistanceHeuristicBaseline(
        targets, beta=0.5, obstacles=wall, bounds=BOUNDS, distance_metric="manhattan"
    )
    shortest = DistanceHeuristicBaseline(
        targets, beta=0.5, obstacles=wall, bounds=BOUNDS, distance_metric="shortest_path"
    )
    manhattan.update((7, 7), (7, 7))
    shortest.update((7, 7), (7, 7))
    # the left target is reachable without touching the wall: both metrics agree
    assert manhattan.cost((7, 7), targets[0]) == 5
    assert shortest.cost((7, 7), targets[0]) == 5
    # the right target is 7 cells away in Manhattan but the wall forces a
    # detour through the gap at y = 15
    assert manhattan.cost((7, 7), targets[1]) == 7
    assert shortest.cost((7, 7), targets[1]) == 23
    # so the two metrics rank the goals differently once the wall matters
    assert manhattan.goal_distribution() != shortest.goal_distribution()
    with pytest.raises(ValueError):
        DistanceHeuristicBaseline(targets, distance_metric="euclidean")


def test_bayesian_belief_is_a_normalized_joint() -> None:
    baseline = BayesianBeliefBaseline(TARGETS, bounds=BOUNDS, persist=0.85)
    assert is_distribution(baseline.goal_distribution())
    assert is_distribution(baseline.intention_distribution())
    assert len(baseline.belief) == len(TARGETS) * len(ALL_INTENTIONS)
    baseline.update((2, 2), (3, 2), "RIGHT", False)
    assert is_distribution(baseline.goal_distribution())
    assert abs(sum(baseline.belief.values()) - 1.0) < 1e-9


def test_bayesian_filter_converges_towards_the_true_goal(small_split) -> None:
    """The posterior must move towards the ground-truth goal on real episodes."""
    improvements = 0
    total = 0
    for record in small_split["test"] + small_split["train"]:
        targets = [tuple(t["position"]) for t in record["candidate_targets"]]
        baseline = BayesianBeliefBaseline(
            targets,
            obstacles=[tuple(o) for o in record["static_obstacles"]],
            bounds=tuple(record["grid_dimensions"]),
        )
        traj = belief_trajectory(baseline, record, record["focal_agent_id"])
        if len(traj) < 6:
            continue
        true_goal = record["privileged_ground_truth"]["focal_goal_id"]
        start_p = traj[0][true_goal]
        end_p = traj[-1][true_goal]
        total += 1
        improvements += int(end_p > start_p)
    assert total >= 5
    assert improvements / total >= 0.8, f"posterior improved on only {improvements}/{total} episodes"


def test_blocked_flag_changes_the_likelihood() -> None:
    """Condition A (blocked observable) and Condition B differ in the emission."""
    kwargs = dict(obstacles=[], bounds=BOUNDS, persist=0.85, block_rate=0.2)
    a = BayesianBeliefBaseline(TARGETS, **kwargs)
    b = BayesianBeliefBaseline(TARGETS, **kwargs)
    a.update((2, 2), (2, 2), "WAIT", True)
    b.update((2, 2), (2, 2), "WAIT", False)
    assert a.goal_distribution() != b.goal_distribution()
    # a blocked WAIT is evidence *against* the WAIT intention
    wait_idx = ALL_INTENTIONS.index("WAIT")
    assert a.intention_distribution()[wait_idx] < b.intention_distribution()[wait_idx]


def test_baselines_run_on_a_condition_b_projection(small_split) -> None:
    """The masked observation must be sufficient to run every baseline."""
    for record in small_split["test"]:
        masked = observable_record(record, masked=True)
        assert "peer_agents_trajectory" not in masked
        steps = masked["focal_trajectory"]
        for entry in steps:
            assert "blocked" not in entry
        baseline = BayesianBeliefBaseline(
            [tuple(t["position"]) for t in masked["candidate_targets"]],
            obstacles=[tuple(o) for o in masked["static_obstacles"]],
            bounds=tuple(masked["grid_dimensions"]),
        )
        for prev, cur in zip(steps, steps[1:]):
            baseline.update(tuple(prev["position"]), tuple(cur["position"]), cur["realized_action"], None)
        assert is_distribution(baseline.goal_distribution())


def test_baseline_ordering_wiring() -> None:
    """random < distance < Bayesian at an informative prefix fraction.

    This is a wiring check over a full small dataset, not a generalization
    claim: the held-out ordering gate lives in verification/m2 §7.
    """
    from verification.m2.dataset_m2 import verification_config

    records = generate_dataset(
        verification_config(), master_seed=777, num_scenarios=10, episodes_per_scenario=4
    )
    scores = {"random": [], "distance": [], "bayesian": []}
    for record in records:
        targets = [tuple(t["position"]) for t in record["candidate_targets"]]
        obstacles = [tuple(o) for o in record["static_obstacles"]]
        bounds = tuple(record["grid_dimensions"])
        true_goal = record["privileged_ground_truth"]["focal_goal_id"]
        baselines = {
            "random": UniformBaseline(targets, seed=1),
            "distance": DistanceHeuristicBaseline(targets, beta=0.5, obstacles=obstacles, bounds=bounds),
            "bayesian": BayesianBeliefBaseline(targets, obstacles=obstacles, bounds=bounds),
        }
        for name, baseline in baselines.items():
            traj = belief_trajectory(baseline, record, record["focal_agent_id"])
            idx = max(1, int(0.5 * (len(traj) - 1)))
            scores[name].append(int(np.argmax(traj[idx]) == true_goal))
    acc = {k: float(np.mean(v)) for k, v in scores.items() if v}
    assert acc["random"] < acc["distance"] < acc["bayesian"], acc


def test_observation_stream_yields_one_transition_per_step(small_split) -> None:
    for record in small_split["train"][:3]:
        focal = [s for s in record["steps"] if s["agent_id"] == record["focal_agent_id"]]
        stream = list(observation_stream(record, record["focal_agent_id"]))
        assert len(stream) == len(focal) - 1
        for (prev_pos, pos, action, blocked), step in zip(stream, focal[1:]):
            assert pos == tuple(step["position"])
            assert action == step["realized_action"]
            assert blocked == bool(step["blocked"])


def test_cpu_only_execution_path() -> None:
    """No GPU dependency may be imported by the baseline / metrics stack."""
    for module in list(sys.modules):
        if module.split(".")[0] in {"torch", "cupy", "jax"}:
            pytest.skip(f"{module} already imported by the test session")
    import importlib

    for name in ("src.evaluation.baselines", "src.evaluation.metrics", "src.features.tensorize"):
        importlib.import_module(name)
    assert "torch" not in sys.modules
    assert "cupy" not in sys.modules
