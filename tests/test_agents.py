"""Agent policy tests: stochasticity, reproducibility, and intention semantics.

These tests enforce the MILESTONE_1_ACCEPTANCE "Agent policy" rows: the
intention must depend on goal, history and previous intention, must persist for
a minimum dwell, must respond to blocking, and must be seed-controlled.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.intentions import ALL_INTENTIONS, Intention
from src.agents.policy import (
    DEFAULT_BEHAVIOR_WEIGHTS,
    StochasticPolicy,
    action_distribution,
    intention_distribution,
)

BOUNDS = (16, 16)
WEIGHTS = {"DIRECT": 0.34, "DETOUR_LEFT": 0.24, "DETOUR_RIGHT": 0.24, "WAIT": 0.18}


def make_policy(seed: int = 0, **kwargs) -> StochasticPolicy:
    params = dict(
        min_dwell_steps=4,
        blocked_detour_threshold=2,
        temperature=1.0,
        persistence_bonus=1.5,
        detour_boost=2.0,
        behavior_weights=WEIGHTS,
    )
    params.update(kwargs)
    return StochasticPolicy(rng=random.Random(seed), **params)


def total_close_to_one(dist) -> bool:
    return abs(sum(dist.values()) - 1.0) < 1e-9 and all(v >= 0.0 for v in dist.values())


def test_intention_distribution_is_a_valid_categorical() -> None:
    dist = intention_distribution(
        behavior_weights=WEIGHTS,
        goal=(10, 10),
        current=(2, 2),
        prev_intention=None,
        history=[],
        blocked_count=0,
        bounds=BOUNDS,
    )
    assert set(dist) == set(ALL_INTENTIONS)
    assert total_close_to_one(dist)


def test_behavior_weights_drive_the_policy() -> None:
    """The config must actually shape the intention distribution."""
    direct_heavy = {"DIRECT": 0.90, "DETOUR_LEFT": 0.04, "DETOUR_RIGHT": 0.04, "WAIT": 0.02}
    wait_heavy = {"DIRECT": 0.10, "DETOUR_LEFT": 0.10, "DETOUR_RIGHT": 0.10, "WAIT": 0.70}
    kwargs = dict(
        goal=(12, 3),
        current=(4, 4),
        prev_intention=None,
        history=[],
        blocked_count=0,
        bounds=BOUNDS,
    )
    a = intention_distribution(behavior_weights=direct_heavy, **kwargs)
    b = intention_distribution(behavior_weights=wait_heavy, **kwargs)
    assert a[Intention.DIRECT.value] > b[Intention.DIRECT.value]
    assert b[Intention.WAIT.value] > a[Intention.WAIT.value]


def test_intention_depends_on_the_hidden_goal() -> None:
    kwargs = dict(
        behavior_weights=WEIGHTS,
        current=(8, 8),
        prev_intention=None,
        history=[],
        blocked_count=0,
        bounds=BOUNDS,
    )
    left = intention_distribution(goal=(2, 8), **kwargs)
    right = intention_distribution(goal=(14, 8), **kwargs)
    assert left != right, "identical distributions for different goals"
    # the lateral preference flips with the goal side
    assert left[Intention.DETOUR_LEFT.value] != right[Intention.DETOUR_LEFT.value]


def test_intention_depends_on_history() -> None:
    kwargs = dict(
        behavior_weights=WEIGHTS,
        goal=(12, 12),
        current=(4, 4),
        prev_intention=None,
        blocked_count=0,
        bounds=BOUNDS,
    )
    fresh = intention_distribution(history=[], **kwargs)
    repeated = intention_distribution(
        history=[Intention.DETOUR_LEFT.value] * 3, **kwargs
    )
    assert repeated[Intention.DETOUR_LEFT.value] < fresh[Intention.DETOUR_LEFT.value]
    assert repeated[Intention.DETOUR_RIGHT.value] > fresh[Intention.DETOUR_RIGHT.value]


def test_intention_depends_on_previous_intention() -> None:
    kwargs = dict(
        behavior_weights=WEIGHTS,
        goal=(12, 12),
        current=(4, 4),
        history=[],
        blocked_count=0,
        bounds=BOUNDS,
    )
    from_direct = intention_distribution(prev_intention=Intention.DIRECT.value, **kwargs)
    from_wait = intention_distribution(prev_intention=Intention.WAIT.value, **kwargs)
    assert from_direct[Intention.DIRECT.value] > from_wait[Intention.DIRECT.value]
    assert from_wait[Intention.WAIT.value] > from_direct[Intention.WAIT.value]


def test_minimum_dwell_enforces_persistence() -> None:
    policy = make_policy(seed=7, min_dwell_steps=4)
    chosen = [
        policy.select_intention(
            goal=(14, 14),
            current=(2 + t, 2),
            history=[],
            blocked_count=0,
            bounds=BOUNDS,
        )
        for t in range(12)
    ]
    runs, cur = [], 1
    for a, b in zip(chosen, chosen[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    # every completed run must respect the dwell (the last run may be truncated)
    assert all(r >= 4 for r in runs[:-1]), f"dwell violated: {runs}"
    assert min(runs[:-1]) >= 4 if len(runs) > 1 else True


def test_blocked_streak_boosts_detours_and_breaks_dwell() -> None:
    kwargs = dict(
        behavior_weights=WEIGHTS,
        goal=(12, 12),
        current=(4, 4),
        prev_intention=Intention.DIRECT.value,
        history=[],
        bounds=BOUNDS,
    )
    calm = intention_distribution(blocked_count=0, **kwargs)
    stuck = intention_distribution(blocked_count=2, **kwargs)
    assert stuck[Intention.DIRECT.value] < calm[Intention.DIRECT.value]
    assert stuck[Intention.DETOUR_LEFT.value] > calm[Intention.DETOUR_LEFT.value]
    assert stuck[Intention.DETOUR_RIGHT.value] > calm[Intention.DETOUR_RIGHT.value]

    # and the dwell is interrupted so the agent can escape a deadlock
    policy = make_policy(seed=3, min_dwell_steps=5)
    first = policy.select_intention(
        goal=(12, 12), current=(4, 4), history=[], blocked_count=0, bounds=BOUNDS
    )
    forced = policy.select_intention(
        goal=(12, 12), current=(4, 4), history=[first], blocked_count=3, bounds=BOUNDS
    )
    assert forced in (Intention.DETOUR_LEFT.value, Intention.DETOUR_RIGHT.value)


def test_detour_left_and_right_are_behaviourally_distinct() -> None:
    """Regression guard: the two detour labels used to share one action model."""
    goal, current = (12, 8), (4, 8)
    left = action_distribution(Intention.DETOUR_LEFT.value, goal, current, bounds=BOUNDS)
    right = action_distribution(Intention.DETOUR_RIGHT.value, goal, current, bounds=BOUNDS)
    assert total_close_to_one(left) and total_close_to_one(right)
    assert left != right, "DETOUR_LEFT and DETOUR_RIGHT are observationally identical"
    # they are mirror images: swapping UP/DOWN mirrors the distribution
    mirrored = {
        "UP": right["DOWN"],
        "DOWN": right["UP"],
        "LEFT": right["LEFT"],
        "RIGHT": right["RIGHT"],
        "WAIT": right["WAIT"],
    }
    assert all(abs(left[a] - mirrored[a]) < 1e-9 for a in left), (left, mirrored)


def test_action_distribution_prefers_goal_progress_under_direct() -> None:
    dist = action_distribution(Intention.DIRECT.value, (10, 4), (4, 4), bounds=BOUNDS)
    assert total_close_to_one(dist)
    assert dist["RIGHT"] > dist["LEFT"], "DIRECT should move towards the goal"
    wait_only = action_distribution(Intention.WAIT.value, (10, 4), (4, 4), bounds=BOUNDS)
    assert wait_only["WAIT"] > 0.8


def test_action_distribution_respects_obstacles_and_bounds() -> None:
    obstacles = [(5, 4)]
    dist = action_distribution(
        Intention.DIRECT.value, (10, 4), (4, 4), obstacles=obstacles, bounds=BOUNDS
    )
    assert dist.get("RIGHT", 0.0) == 0.0, "an obstacle cell must receive no mass"
    corner = action_distribution(Intention.DIRECT.value, (0, 0), (0, 0), bounds=(4, 4))
    assert total_close_to_one(corner)
    # fully surrounded: every neighbour is an obstacle, so WAIT is the only option
    trapped = action_distribution(
        Intention.DIRECT.value,
        (5, 5),
        (1, 1),
        obstacles=[(0, 1), (2, 1), (1, 0), (1, 2)],
        bounds=(8, 8),
    )
    assert trapped == {"WAIT": 1.0}
    # partially blocked: the free directions keep their mass, the blocked one is zero
    partial = action_distribution(
        Intention.DIRECT.value, (5, 5), (1, 1), obstacles=[(0, 1), (1, 0)], bounds=(8, 8)
    )
    assert partial.get("LEFT", 0.0) == 0.0 and partial.get("UP", 0.0) == 0.0
    assert partial["RIGHT"] > 0.0 and partial["DOWN"] > 0.0


def test_same_seed_reproduces_the_rollout_exactly() -> None:
    def rollout(seed: int) -> list:
        policy = make_policy(seed=seed)
        out = []
        history = []
        pos = (2, 2)
        for _ in range(40):
            intention = policy.select_intention(
                goal=(13, 11), current=pos, history=history, blocked_count=0, bounds=BOUNDS
            )
            action = policy.select_action(
                intention=intention,
                goal=(13, 11),
                current=pos,
                peers=[(3, 3)],
                policy_sensing_radius=3,
                bounds=BOUNDS,
            )
            history.append(intention)
            out.append((intention, action))
        return out

    assert rollout(11) == rollout(11)
    assert rollout(11) != rollout(12), "different agent seeds produced identical rollouts"
    # the policy draws from more than one intention and more than one action
    seq = rollout(11)
    assert len({i for i, _ in seq}) >= 2
    assert len({a for _, a in seq}) >= 2


def test_policy_reset_clears_state() -> None:
    policy = make_policy(seed=5)
    policy.select_intention(goal=(10, 10), current=(1, 1), history=[], blocked_count=0, bounds=BOUNDS)
    assert policy.state.prev_intention is not None
    policy.reset()
    assert policy.state.prev_intention is None
    assert policy.state.dwell_remaining == 0
    assert policy.state.blocked_count == 0


def test_default_behavior_weights_are_a_distribution() -> None:
    assert abs(sum(DEFAULT_BEHAVIOR_WEIGHTS.values()) - 1.0) < 1e-9
    assert max(DEFAULT_BEHAVIOR_WEIGHTS.values()) <= 0.60


def test_peer_avoidance_reweights_moves_into_occupied_cells() -> None:
    """A peer standing on the greedy next cell makes that move less likely."""

    def sample(peers, n=500):
        policy = make_policy(seed=1)
        return [
            policy.select_action(
                intention=Intention.DIRECT.value,
                goal=(10, 4),
                current=(4, 4),
                peers=peers,
                policy_sensing_radius=3,
                bounds=BOUNDS,
            )
            for _ in range(n)
        ]

    free = sample([])
    crowded = sample([(5, 4)])
    assert free.count("RIGHT") / len(free) > crowded.count("RIGHT") / len(crowded)
    # the base model still prefers the greedy direction when nobody is there
    base = action_distribution(Intention.DIRECT.value, (10, 4), (4, 4), bounds=BOUNDS)
    assert base["RIGHT"] > base["LEFT"]
