"""GridWorld environment with single-pass symmetric collision resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Position = Tuple[int, int]

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
_DELTAS: Dict[str, Position] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}


@dataclass
class StepResult:
    positions: Dict[int, Position]
    realized_actions: Dict[int, str]
    blocked: Dict[int, bool]
    terminated: bool = False
    reason: Optional[str] = None


class GridWorld:
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: List[Position],
        targets: List[Position],
        agent_starts: Optional[List[Position]] = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("grid dimensions must be positive")
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.targets = list(targets)
        self.agent_starts = [tuple(s) for s in (agent_starts or [])]

    def reset(self, seed: Optional[int] = None) -> Dict[int, Position]:
        """Return the deterministic initial positions.

        The layout is static: ``reset`` never regenerates obstacles or targets,
        and the ``seed`` argument is accepted so callers can thread the
        environment seed through a single interface (M2 test 3i asserts that a
        reset leaves the static scenario untouched).
        """
        return {i: pos for i, pos in enumerate(self.agent_starts)}

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, pos: Position) -> bool:
        return pos in self.obstacles

    def reachable(self, pos: Position) -> bool:
        return self.in_bounds(pos) and not self.is_obstacle(pos)

    def destination(self, pos: Position, action: str) -> Position:
        if action not in _DELTAS:
            raise ValueError(f"unknown action {action}")
        dx, dy = _DELTAS[action]
        return (pos[0] + dx, pos[1] + dy)

    def step(
        self,
        positions: Dict[int, Position],
        intended_actions: Dict[int, str],
    ) -> Tuple[Dict[int, Position], Dict[int, str], Dict[int, bool]]:
        """Single-pass simultaneous transition.

        A proposal is valid iff:
          - destination inside bounds,
          - destination is not an obstacle,
          - destination is not occupied at time t,
          - no other agent proposes the same destination.
        Invalid proposals keep the agent in place. WAIT is always valid
        unless another agent also proposes the current cell.
        """
        destinations: Dict[int, Position] = {}
        for agent, action in intended_actions.items():
            destinations[agent] = self.destination(positions[agent], action)

        counts: Dict[Position, int] = {}
        for d in destinations.values():
            counts[d] = counts.get(d, 0) + 1

        new_positions: Dict[int, Position] = {}
        realized: Dict[int, str] = {}
        blocked: Dict[int, bool] = {}
        for agent, action in intended_actions.items():
            dest = destinations[agent]
            valid = (
                self.in_bounds(dest)
                and not self.is_obstacle(dest)
                and dest not in positions.values()
                and counts[dest] == 1
            )
            if valid:
                new_positions[agent] = dest
                realized[agent] = action
                blocked[agent] = False
            else:
                new_positions[agent] = positions[agent]
                realized[agent] = "WAIT" if action != "WAIT" else "WAIT"
                blocked[agent] = action != "WAIT"
        return new_positions, realized, blocked

    def block_reason(
        self,
        pos: Position,
        action: str,
        occupied: Dict[int, Position],
        agent: Optional[int] = None,
        destinations: Optional[Dict[int, Position]] = None,
    ) -> str:
        """Why a proposal did not move (privileged diagnostic, never observable).

        ``destinations`` enables the same-destination / conflict distinction; when
        it is omitted the check degrades to occupancy only.
        """
        if action == "WAIT":
            return "none"
        dest = self.destination(pos, action)
        if not self.in_bounds(dest):
            return "boundary"
        if self.is_obstacle(dest):
            return "obstacle"
        if dest in occupied.values():
            return "occupied"
        if destinations is not None:
            others = [a for a, d in destinations.items() if d == dest and a != agent]
            if others:
                return "conflict"
        return "none"

    def goal_reached(self, pos: Position, goal: Position) -> bool:
        return pos == goal
