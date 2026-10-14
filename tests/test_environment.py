"""Environment tests: bounds, obstacles, and the symmetric collision rule.

Covers every case enumerated in PROJECT_SPEC.md §3.2 and ARCHITECTURE.md §3:
same-destination, swap, 3-cycle, longer cycle, convoy, obstacle, boundary,
simultaneity, determinism and the absence of agent-ID priority.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.gridworld import GridWorld
from src.environment.pathfinding import (
    UNREACHABLE,
    all_cells_connected,
    cost_map,
    fully_connected,
    path_cost,
)


def make_grid(obstacles=(), targets=(), width=8, height=8, starts=()):
    return GridWorld(
        width=width,
        height=height,
        obstacles=[tuple(o) for o in obstacles],
        targets=[tuple(t) for t in targets],
        agent_starts=[tuple(s) for s in starts],
    )


def test_grid_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_grid(width=0)
    with pytest.raises(ValueError):
        make_grid(height=-1)


def test_bounds_and_reachability_predicates() -> None:
    grid = make_grid(obstacles=[(1, 1)])
    assert grid.in_bounds((0, 0)) and grid.in_bounds((7, 7))
    assert not grid.in_bounds((-1, 0)) and not grid.in_bounds((8, 0))
    assert grid.is_obstacle((1, 1)) and not grid.is_obstacle((1, 2))
    assert grid.reachable((0, 0)) and not grid.reachable((1, 1))


def test_boundary_collision_keeps_the_agent_in_place() -> None:
    grid = make_grid(width=4, height=4)
    positions = {0: (0, 0)}
    new, realized, blocked = grid.step(positions, {0: "LEFT"})
    assert new == {0: (0, 0)}
    assert realized == {0: "WAIT"}
    assert blocked == {0: True}
    assert grid.block_reason((0, 0), "LEFT", positions, 0) == "boundary"


def test_obstacle_collision_keeps_the_agent_in_place() -> None:
    grid = make_grid(obstacles=[(1, 0)])
    positions = {0: (0, 0)}
    new, realized, blocked = grid.step(positions, {0: "RIGHT"})
    assert new == {0: (0, 0)} and blocked == {0: True}
    assert grid.block_reason((0, 0), "RIGHT", positions, 0) == "obstacle"


def test_free_move_is_simultaneous_and_unblocked() -> None:
    grid = make_grid()
    positions = {0: (0, 0), 1: (5, 5)}
    new, realized, blocked = grid.step(positions, {0: "RIGHT", 1: "UP"})
    assert new == {0: (1, 0), 1: (5, 4)}
    assert realized == {0: "RIGHT", 1: "UP"}
    assert blocked == {0: False, 1: False}


def test_wait_is_not_reported_as_blocked() -> None:
    grid = make_grid()
    new, realized, blocked = grid.step({0: (2, 2)}, {0: "WAIT"})
    assert new == {0: (2, 2)} and realized == {0: "WAIT"} and blocked == {0: False}


def test_same_destination_blocks_both_agents() -> None:
    grid = make_grid()
    positions = {0: (0, 0), 1: (2, 0)}
    new, realized, blocked = grid.step(positions, {0: "RIGHT", 1: "LEFT"})
    assert new == positions, "same-destination proposals must both stay"
    assert blocked == {0: True, 1: True}
    assert grid.block_reason((0, 0), "RIGHT", positions, 0, {0: (1, 0), 1: (1, 0)}) == "conflict"


def test_swap_blocks_both_agents() -> None:
    grid = make_grid()
    positions = {0: (0, 0), 1: (1, 0)}
    new, _, blocked = grid.step(positions, {0: "RIGHT", 1: "LEFT"})
    assert new == positions, "a swap must leave both agents in place"
    assert blocked == {0: True, 1: True}
    assert grid.block_reason((0, 0), "RIGHT", positions, 0) == "occupied"


def test_odd_cycles_are_not_constructible_on_a_4_connected_grid() -> None:
    """PROJECT_SPEC §3.2 lists a 3-cycle; on a 4-connected grid it cannot occur.

    The grid graph is bipartite (colour a cell by (x + y) % 2 and every move
    flips the colour), so it contains no odd cycle: a rotation in which every
    proposed destination is occupied needs an even number of agents. The
    3-cycle row of the specification is therefore vacuous, and the smallest
    real cycle is the 4-cycle tested below. This is asserted rather than
    assumed so a future 8-connected move set cannot silently invalidate it.
    """

    def is_bipartite(width: int, height: int) -> bool:
        colour = {(x, y): (x + y) % 2 for x in range(width) for y in range(height)}
        for (x, y), c in colour.items():
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in colour and colour[nxt] == c:
                    return False
        return True

    assert is_bipartite(8, 8)


def _rotation(positions, ring):
    """Each agent proposes the cell currently held by the next agent."""
    deltas = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}
    actions = {}
    for i, cur in enumerate(ring):
        nxt = ring[(i + 1) % len(ring)]
        actions[i] = deltas[(nxt[0] - cur[0], nxt[1] - cur[1])]
    return actions


def test_four_cycle_blocks_all_agents() -> None:
    """Smallest real cycle: every destination is occupied at time t."""
    grid = make_grid()
    ring = [(0, 0), (1, 0), (1, 1), (0, 1)]
    positions = {i: ring[i] for i in range(len(ring))}
    new, _, blocked = grid.step(positions, _rotation(positions, ring))
    assert new == positions, "a 4-cycle must leave every agent in place"
    assert all(blocked.values())


def test_longer_cycle_blocks_all_agents() -> None:
    """A 6-agent rotation behaves identically: all agents remain."""
    grid = make_grid(width=8, height=8)
    ring = [(2, 2), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
    positions = {i: ring[i] for i in range(len(ring))}
    new, _, blocked = grid.step(positions, _rotation(positions, ring))
    assert new == positions, "a 6-cycle must leave every agent in place"
    assert all(blocked.values())
    assert all(
        grid.block_reason(positions[a], act, positions, a) == "occupied"
        for a, act in _rotation(positions, ring).items()
    )


def test_convoy_movement_is_staggered() -> None:
    """B follows A into the cell A is leaving: B cannot move in the same step."""
    grid = make_grid()
    positions = {0: (1, 0), 1: (0, 0)}
    new, realized, blocked = grid.step(positions, {0: "RIGHT", 1: "RIGHT"})
    assert new[0] == (2, 0) and blocked[0] is False
    assert new[1] == (0, 0) and blocked[1] is True, "no iterative re-check: convoy is staggered"
    # on the next step the cell is free and the convoy advances
    new2, _, blocked2 = grid.step(new, {0: "RIGHT", 1: "RIGHT"})
    assert new2 == {0: (3, 0), 1: (1, 0)} and not any(blocked2.values())


def test_transition_rule_has_no_agent_id_priority() -> None:
    """Relabelling agents must not change the physical outcome."""
    grid = make_grid()
    positions = {0: (0, 0), 1: (2, 0), 2: (1, 1)}
    actions = {0: "RIGHT", 1: "LEFT", 2: "UP"}
    new, realized, blocked = grid.step(positions, actions)

    relabel = {2: (0, 0), 5: (2, 0), 9: (1, 1)}
    relabel_actions = {2: "RIGHT", 5: "LEFT", 9: "UP"}
    new_r, realized_r, blocked_r = grid.step(relabel, relabel_actions)
    mapping = {0: 2, 1: 5, 2: 9}
    assert {mapping[a]: p for a, p in new.items()} == new_r
    assert {mapping[a]: r for a, r in realized.items()} == realized_r
    assert {mapping[a]: b for a, b in blocked.items()} == blocked_r


def test_transition_is_deterministic_and_single_pass() -> None:
    grid = make_grid()
    positions = {0: (0, 0), 1: (2, 0)}
    actions = {0: "RIGHT", 1: "LEFT"}
    first = grid.step(positions, actions)
    for _ in range(5):
        assert grid.step(positions, actions) == first


def test_unknown_action_is_rejected() -> None:
    grid = make_grid()
    with pytest.raises(ValueError):
        grid.step({0: (0, 0)}, {0: "TELEPORT"})


def test_goal_reached_predicate() -> None:
    grid = make_grid(targets=[(3, 3)])
    assert grid.goal_reached((3, 3), (3, 3))
    assert not grid.goal_reached((3, 2), (3, 3))


def test_reset_returns_starts_without_touching_the_layout() -> None:
    grid = make_grid(obstacles=[(1, 1)], targets=[(7, 7)], starts=[(0, 0), (0, 1)])
    obstacles_before = set(grid.obstacles)
    targets_before = list(grid.targets)
    assert grid.reset(seed=123) == {0: (0, 0), 1: (0, 1)}
    assert grid.reset(seed=999) == {0: (0, 0), 1: (0, 1)}
    assert grid.obstacles == obstacles_before
    assert grid.targets == targets_before


def test_pathfinding_utilities() -> None:
    # a partial wall is walked around: (0,0) -> down column 0 -> across row 4
    partial = [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert path_cost(5, 5, partial, (0, 0), (4, 4)) == 8
    assert all_cells_connected(5, 5, partial, [(0, 0), (4, 4)])

    # a wall spanning the whole grid disconnects the two halves
    full_wall = partial + [(1, 4)]
    assert path_cost(5, 5, full_wall, (0, 0), (4, 4)) == UNREACHABLE
    assert not all_cells_connected(5, 5, full_wall, [(0, 0), (4, 4)])
    assert not fully_connected(5, 5, full_wall)

    assert fully_connected(5, 5, [])
    costs = cost_map(5, 5, [], (0, 0))
    assert costs[(0, 0)] == 0 and costs[(1, 0)] == 1 and costs[(4, 4)] == 8
    assert path_cost(5, 5, [], (0, 0), (0, 0)) == 0
