"""Termination-condition tests: GOAL_REACHED, MAX_STEPS, DEADLOCK_LIMIT.

Uses deterministic stub policies so each condition is reached by construction
rather than by sampling luck.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.gridworld import GridWorld
from src.trajectory.generator import episode_prefix, generate_episode

Position = Tuple[int, int]


class StubPolicy:
    """Duck-typed stand-in for StochasticPolicy with a fixed action."""

    def __init__(self, action: str, intention: str = "DIRECT"):
        self.action = action
        self.intention = intention
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_intention(self, goal, current, history, blocked_count, **kwargs) -> str:
        return self.intention

    def select_action(self, intention, goal, current, peers=(), **kwargs) -> str:
        return self.action


def rollout(
    width: int,
    height: int,
    obstacles: Sequence[Position],
    starts: Dict[int, Position],
    goals: Dict[int, Position],
    policies: Dict[int, StubPolicy],
    max_steps: int = 50,
    deadlock_limit: int = 10,
):
    grid = GridWorld(width, height, list(obstacles), sorted(goals.values()), [starts[i] for i in sorted(starts)])
    return generate_episode(
        scenario_id="test",
        grid=grid,
        agents=policies,
        goals=goals,
        starts=starts,
        max_steps=max_steps,
        deadlock_limit=deadlock_limit,
        seed_info={"env_seed": 0, "agent_seed": 0, "goal_seed": 0},
    )


def test_goal_reached_terminates_the_episode() -> None:
    episode = rollout(
        width=8,
        height=8,
        obstacles=[],
        starts={0: (2, 2)},
        goals={0: (5, 2)},
        policies={0: StubPolicy("RIGHT")},
    )
    assert episode["termination_reason"] == "GOAL_REACHED"
    focal = [s for s in episode["steps"] if s["agent_id"] == 0]
    assert len(focal) == 3, "three RIGHT moves reach the goal"
    assert tuple(focal[-1]["position"]) == (5, 2)


def test_goal_reached_takes_priority_over_max_steps() -> None:
    episode = rollout(
        width=8,
        height=8,
        obstacles=[],
        starts={0: (2, 2)},
        goals={0: (3, 2)},
        policies={0: StubPolicy("RIGHT")},
        max_steps=50,
    )
    assert episode["termination_reason"] == "GOAL_REACHED"
    assert len([s for s in episode["steps"] if s["agent_id"] == 0]) == 1


def test_max_steps_terminates_when_the_goal_is_never_reached() -> None:
    episode = rollout(
        width=8,
        height=8,
        obstacles=[],
        starts={0: (0, 0)},
        goals={0: (7, 7)},
        policies={0: StubPolicy("WAIT")},
        max_steps=6,
    )
    assert episode["termination_reason"] == "MAX_STEPS"
    focal = [s for s in episode["steps"] if s["agent_id"] == 0]
    assert len(focal) == 6
    assert [s["step"] for s in focal] == list(range(6))


def test_deadlock_limit_terminates_mutual_blocking() -> None:
    """Two agents permanently proposing each other's cell are both blocked."""
    episode = rollout(
        width=8,
        height=8,
        obstacles=[],
        starts={0: (3, 3), 1: (4, 3)},
        goals={0: (4, 3), 1: (3, 3)},
        policies={0: StubPolicy("RIGHT"), 1: StubPolicy("LEFT")},
        max_steps=50,
        deadlock_limit=4,
    )
    assert episode["termination_reason"] == "DEADLOCK_LIMIT"
    focal = [s for s in episode["steps"] if s["agent_id"] == 0]
    assert len(focal) == 4, "deadlock is declared after deadlock_limit blocked steps"
    assert all(s["blocked"] for s in focal)
    assert all(s["blocked_reason"] == "occupied" for s in focal)
    # nobody ever moved
    assert all(tuple(s["position"]) == (3, 3) for s in focal)


def test_deadlock_requires_every_agent_to_be_blocked() -> None:
    """A single blocked agent must not end the episode."""
    episode = rollout(
        width=8,
        height=8,
        obstacles=[],
        starts={0: (0, 0), 1: (6, 6)},
        goals={0: (0, 1), 1: (6, 5)},
        # agent 0 walks into a wall forever, agent 1 keeps moving
        policies={0: StubPolicy("LEFT"), 1: StubPolicy("UP")},
        max_steps=5,
        deadlock_limit=4,
    )
    # agent 0 is blocked on the boundary for all 5 steps, but agent 1 keeps
    # moving, so the "all agents blocked" condition never holds
    assert episode["termination_reason"] == "MAX_STEPS"
    peer = [s for s in episode["steps"] if s["agent_id"] == 1]
    assert not any(s["blocked"] for s in peer)
    focal = [s for s in episode["steps"] if s["agent_id"] == 0]
    assert all(s["blocked"] and s["blocked_reason"] == "boundary" for s in focal)


def test_prefix_truncation_is_a_strict_projection() -> None:
    episode = rollout(
        width=8,
        height=8,
        obstacles=[],
        starts={0: (0, 0)},
        goals={0: (6, 0)},
        policies={0: StubPolicy("RIGHT")},
        max_steps=10,
    )
    total = len([s for s in episode["steps"] if s["agent_id"] == 0])
    prefix = episode_prefix(episode, 3)
    assert len([s for s in prefix["steps"] if s["agent_id"] == 0]) == 3
    assert prefix["steps"] == episode["steps"][:3]
    assert prefix["total_steps"] == 3
    # truncating beyond the episode length is a no-op, and zero yields nothing
    assert episode_prefix(episode, 10 ** 6)["steps"] == episode["steps"]
    assert episode_prefix(episode, 0)["steps"] == []
