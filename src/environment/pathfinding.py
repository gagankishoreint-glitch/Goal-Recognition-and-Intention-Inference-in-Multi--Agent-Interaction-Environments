"""Shortest-path utilities for the 4-connected grid.

Used by (a) the scenario sampler's connectivity validation, (b) the
distance-heuristic baseline, and (c) the Bayesian cost-difference baseline.
Pure Python / BFS: no third-party graph library and no GPU dependency, so the
CPU-only execution path required by REPRODUCIBILITY.md is trivially satisfied.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Position = Tuple[int, int]

MOVE_DELTAS: Tuple[Tuple[str, Position], ...] = (
    ("UP", (0, -1)),
    ("DOWN", (0, 1)),
    ("LEFT", (-1, 0)),
    ("RIGHT", (1, 0)),
)

UNREACHABLE = -1


def in_bounds(pos: Position, width: int, height: int) -> bool:
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def neighbors(
    pos: Position,
    width: int,
    height: int,
    obstacles: Optional[Iterable[Position]] = None,
) -> List[Position]:
    """Free 4-connected neighbours of ``pos`` (sorted for determinism)."""
    blocked: Set[Position] = set(obstacles or ())
    out: List[Position] = []
    for _, (dx, dy) in MOVE_DELTAS:
        nxt = (pos[0] + dx, pos[1] + dy)
        if in_bounds(nxt, width, height) and nxt not in blocked:
            out.append(nxt)
    return sorted(out)


def cost_map(
    width: int,
    height: int,
    obstacles: Iterable[Position],
    source: Position,
) -> Dict[Position, int]:
    """BFS distance from ``source`` to every reachable free cell.

    Unreachable cells are absent from the returned mapping.
    """
    blocked = set(obstacles)
    dist: Dict[Position, int] = {source: 0}
    queue: deque = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt in neighbors(cur, width, height, blocked):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def path_cost(
    width: int,
    height: int,
    obstacles: Iterable[Position],
    start: Position,
    goal: Position,
) -> int:
    """Shortest obstacle-avoiding path length, or ``UNREACHABLE`` (-1)."""
    if start == goal:
        return 0
    return cost_map(width, height, obstacles, goal).get(start, UNREACHABLE)


def free_cells(width: int, height: int, obstacles: Iterable[Position]) -> List[Position]:
    blocked = set(obstacles)
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in blocked
    ]


def connected_component(
    width: int, height: int, obstacles: Iterable[Position], start: Position
) -> Set[Position]:
    blocked = set(obstacles)
    seen = {start}
    queue: deque = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in neighbors(cur, width, height, blocked):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def all_cells_connected(
    width: int,
    height: int,
    obstacles: Iterable[Position],
    cells: Sequence[Position],
) -> bool:
    """True iff every cell in ``cells`` lies in one connected free component."""
    cells = [c for c in cells]
    if not cells:
        return True
    component = connected_component(width, height, obstacles, cells[0])
    return all(c in component for c in cells)


def fully_connected(width: int, height: int, obstacles: Iterable[Position]) -> bool:
    """True iff the whole free space is a single connected component."""
    cells = free_cells(width, height, obstacles)
    if not cells:
        return False
    return len(connected_component(width, height, obstacles, cells[0])) == len(cells)
