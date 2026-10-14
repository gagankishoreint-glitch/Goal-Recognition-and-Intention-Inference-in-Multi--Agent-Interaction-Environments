"""Stateful stochastic agent policy.

Generative model
----------------
    P(I_t | G, S_t, H_t, I_{t-1}) = softmax(U(G, S_t, H_t, I_{t-1}))
    P(A_t | I_t, G, S_t)          = softmax(Q(I_t, G, S_t))

Design requirements enforced here (see PROJECT_SPEC.md §1.1 and §3.3):

* the base utilities come from ``behavior_weights`` in the config, so the
  configuration actually drives the policy;
* intention selection is a multi-factor softmax (base behaviour weights,
  goal-relative lateral preference, local obstacle clearance, recent history,
  previous-intention persistence, blocked-step boost) and therefore is *not*
  reducible to a single deterministic geometric threshold;
* minimum dwell enforces temporal commitment;
* ``DETOUR_LEFT`` and ``DETOUR_RIGHT`` are mirror-asymmetric *behaviours*
  (opposite lateral drift relative to the goal vector), so the two labels are
  observationally distinguishable instead of being two names for one behaviour.

The action model is exposed as the module-level :func:`action_distribution` so
that the Bayesian baseline can score observations with exactly the model that
generated them (a well-specified likelihood), while :meth:`select_action`
samples from the same distribution.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .intentions import ALL_INTENTIONS, Intention

Position = Tuple[int, int]

DELTAS: Dict[str, Position] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}
_MOVE_ACTIONS: Tuple[str, ...] = ("DOWN", "LEFT", "RIGHT", "UP")  # sorted: deterministic

DEFAULT_BEHAVIOR_WEIGHTS: Dict[str, float] = {
    Intention.DIRECT.value: 0.45,
    Intention.DETOUR_LEFT.value: 0.20,
    Intention.DETOUR_RIGHT.value: 0.20,
    Intention.WAIT.value: 0.15,
}

# --- intention utility weights -------------------------------------------
SIDE_PREFERENCE = 0.60        # goal-relative lateral preference (left vs right)
CLEAR_PATH_BONUS = 0.40       # DIRECT bonus when the greedy step is unobstructed
BLOCKED_PATH_PENALTY = 0.60   # DIRECT penalty when every greedy step is blocked
HISTORY_REPEAT_PENALTY = 1.5  # repeating the same detour without progress
HISTORY_SWITCH_BONUS = 0.8
WAIT_REPEAT_PENALTY = 1.0
DEFAULT_WAIT_BIAS = 0.0        # additive logit on the WAIT intention

# --- action model weights ------------------------------------------------
GOAL_PROGRESS_WEIGHT = 2.0    # rationality of goal progress under DIRECT
DETOUR_SIDE_WEIGHT = 1.6      # lateral drift strength under a detour intention
DETOUR_PROGRESS_WEIGHT = 0.6  # residual goal progress kept during a detour
WAIT_MOVE_MASS = 0.15         # WAIT intention still jitters occasionally
MOVE_FLOOR = 0.02             # floor mass on WAIT under a moving intention
PEER_AVOIDANCE_WEIGHT = 0.25  # multiplicative down-weight on moves into a peer-occupied cell


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _destination(pos: Position, action: str) -> Position:
    dx, dy = DELTAS[action]
    return (pos[0] + dx, pos[1] + dy)


def _valid_moves(
    current: Position,
    obstacles: Optional[Sequence[Position]] = None,
    bounds: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Move actions whose destination is free (sorted → deterministic order)."""
    blocked = set(obstacles or ())
    out: List[str] = []
    for action in _MOVE_ACTIONS:
        nxt = _destination(current, action)
        if bounds is not None:
            w, h = bounds
            if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                continue
        if nxt in blocked:
            continue
        out.append(action)
    return out


def _greedy_moves(goal: Position, current: Position, moves: Sequence[str]) -> List[str]:
    d0 = _manhattan(current, goal)
    return [a for a in moves if _manhattan(_destination(current, a), goal) < d0]


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0.0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _side_unit_vector(goal: Position, current: Position, sign: int) -> Tuple[float, float]:
    """Unit vector perpendicular to the goal direction.

    ``sign=+1`` is the counter-clockwise perpendicular (``DETOUR_LEFT``),
    ``sign=-1`` its mirror (``DETOUR_RIGHT``). Returns the zero vector when the
    agent already stands on the goal, which makes both detours symmetric.
    """
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return (0.0, 0.0)
    ux, uy = dx / norm, dy / norm
    return (sign * -uy, sign * ux)


def action_distribution(
    intention: str,
    goal: Position,
    current: Position,
    obstacles: Optional[Sequence[Position]] = None,
    bounds: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """P(A | I, G, s) — the shared action model of generator and observer."""
    moves = _valid_moves(current, obstacles, bounds)
    if not moves:
        return {"WAIT": 1.0}

    if intention == Intention.WAIT.value:
        weights = {a: WAIT_MOVE_MASS / len(moves) for a in moves}
        weights["WAIT"] = 1.0 - WAIT_MOVE_MASS
        return _normalize(weights)

    d0 = _manhattan(current, goal)
    side = _side_unit_vector(goal, current, +1 if intention == Intention.DETOUR_LEFT.value else -1)
    weights: Dict[str, float] = {}
    for action in moves:
        nxt = _destination(current, action)
        progress = d0 - _manhattan(nxt, goal)
        if intention == Intention.DIRECT.value:
            score = GOAL_PROGRESS_WEIGHT * progress
        else:  # DETOUR_LEFT / DETOUR_RIGHT
            dx, dy = DELTAS[action]
            lateral = dx * side[0] + dy * side[1]
            score = DETOUR_SIDE_WEIGHT * lateral + DETOUR_PROGRESS_WEIGHT * progress
        weights[action] = math.exp(score)
    weights["WAIT"] = MOVE_FLOOR * sum(weights.values())
    return _normalize(weights)


def intention_utilities(
    behavior_weights: Dict[str, float],
    goal: Position,
    current: Position,
    prev_intention: Optional[str],
    history: Sequence[str],
    blocked_count: int,
    obstacles: Optional[Sequence[Position]] = None,
    bounds: Optional[Tuple[int, int]] = None,
    min_dwell_steps: int = 4,
    blocked_detour_threshold: int = 2,
    persistence_bonus: float = 1.5,
    detour_boost: float = 2.0,
    wait_bias: float = DEFAULT_WAIT_BIAS,
) -> Dict[str, float]:
    """Utility U(G, S_t, H_t, I_{t-1}) before the softmax.

    Five independent factors contribute, so no single geometric threshold
    determines the intention label.
    """
    names = list(behavior_weights) or ALL_INTENTIONS
    utilities: Dict[str, float] = {
        name: math.log(max(behavior_weights.get(name, 1e-6), 1e-6)) for name in names
    }

    dx = goal[0] - current[0]
    dy = goal[1] - current[1]

    # (1) goal-relative lateral preference: the dominant axis of the goal
    # offset decides which detour side is favoured.
    if dx or dy:
        sign = 1 if (abs(dx) >= abs(dy) and dx > 0) or (abs(dx) < abs(dy) and dy > 0) else -1
        utilities[Intention.DETOUR_LEFT.value] += SIDE_PREFERENCE * sign
        utilities[Intention.DETOUR_RIGHT.value] -= SIDE_PREFERENCE * sign

    # (2) local obstacle clearance: DIRECT is attractive only when a greedy
    # step towards the goal is actually available.
    moves = _valid_moves(current, obstacles, bounds)
    if dx or dy:
        if _greedy_moves(goal, current, moves):
            utilities[Intention.DIRECT.value] += CLEAR_PATH_BONUS
        else:
            utilities[Intention.DIRECT.value] -= BLOCKED_PATH_PENALTY
            utilities[Intention.DETOUR_LEFT.value] += 0.5 * BLOCKED_PATH_PENALTY
            utilities[Intention.DETOUR_RIGHT.value] += 0.5 * BLOCKED_PATH_PENALTY

    # (3) behavioural history: repeating a detour or waiting is discouraged.
    recent = list(history)[-3:]
    if recent.count(Intention.DETOUR_LEFT.value) >= 2:
        utilities[Intention.DETOUR_LEFT.value] -= HISTORY_REPEAT_PENALTY
        utilities[Intention.DETOUR_RIGHT.value] += HISTORY_SWITCH_BONUS
    if recent.count(Intention.DETOUR_RIGHT.value) >= 2:
        utilities[Intention.DETOUR_RIGHT.value] -= HISTORY_REPEAT_PENALTY
        utilities[Intention.DETOUR_LEFT.value] += HISTORY_SWITCH_BONUS
    if recent.count(Intention.WAIT.value) >= 2:
        utilities[Intention.WAIT.value] -= WAIT_REPEAT_PENALTY
        utilities[Intention.DIRECT.value] += HISTORY_SWITCH_BONUS

    # (3b) explicit WAIT logit offset from the configuration.
    utilities[Intention.WAIT.value] += wait_bias

    # (4) persistence in the previous intention (temporal commitment).
    if prev_intention is not None and prev_intention in utilities:
        utilities[prev_intention] += persistence_bonus

    # (5) blocked-step response: detours become attractive after a deadlock
    # streak (PROJECT_SPEC.md §3.3).
    if blocked_count >= blocked_detour_threshold:
        utilities[Intention.DIRECT.value] -= detour_boost
        utilities[Intention.DETOUR_LEFT.value] += detour_boost
        utilities[Intention.DETOUR_RIGHT.value] += detour_boost
        utilities[Intention.WAIT.value] += 0.3

    return utilities


def intention_distribution(
    behavior_weights: Dict[str, float],
    goal: Position,
    current: Position,
    prev_intention: Optional[str],
    history: Sequence[str],
    blocked_count: int,
    obstacles: Optional[Sequence[Position]] = None,
    bounds: Optional[Tuple[int, int]] = None,
    temperature: float = 1.0,
    **kwargs,
) -> Dict[str, float]:
    """softmax(U / temperature)."""
    utilities = intention_utilities(
        behavior_weights=behavior_weights,
        goal=goal,
        current=current,
        prev_intention=prev_intention,
        history=history,
        blocked_count=blocked_count,
        obstacles=obstacles,
        bounds=bounds,
        **kwargs,
    )
    scaled = {k: math.exp(v / max(temperature, 1e-6)) for k, v in utilities.items()}
    return _normalize(scaled)


def peer_influence(
    intention: str,
    goal: Position,
    current: Position,
    peers: Sequence[Position],
    policy_sensing_radius: int = 1,
    obstacles: Optional[Sequence[Position]] = None,
    bounds: Optional[Tuple[int, int]] = None,
    avoidance_weight: Optional[float] = None,
) -> float:
    """Total-variation distance between the action model with and without peers.

    ``blocked`` only counts moves the environment actually denied; this measures
    the wider question RQ2 asks — how often a peer's presence *changes what the
    agent does*. 0.0 means the peer is irrelevant at this step.
    """
    base = action_distribution(intention, goal, current, obstacles, bounds)
    weight = PEER_AVOIDANCE_WEIGHT if avoidance_weight is None else avoidance_weight
    if intention == Intention.WAIT.value or not peers or not policy_sensing_radius:
        return 0.0
    near = [p for p in peers if _manhattan(p, current) <= policy_sensing_radius]
    if not near:
        return 0.0
    adjusted = dict(base)
    for action, prob in base.items():
        if action == "WAIT":
            continue
        nxt = _destination(current, action)
        if any(_manhattan(nxt, p) == 0 for p in near):
            adjusted[action] = prob * weight
    adjusted = _normalize(adjusted)
    return 0.5 * sum(abs(adjusted.get(a, 0.0) - base.get(a, 0.0)) for a in set(base) | set(adjusted))


@dataclass
class PolicyState:
    prev_intention: Optional[str] = None
    dwell_remaining: int = 0
    blocked_count: int = 0
    last_positions: List[Position] = field(default_factory=list)


class StochasticPolicy:
    """Seed-controlled stochastic policy with persistent intentions."""

    def __init__(
        self,
        rng: random.Random,
        min_dwell_steps: int = 4,
        blocked_detour_threshold: int = 2,
        temperature: float = 1.0,
        persistence_bonus: float = 1.5,
        detour_boost: float = 2.0,
        behavior_weights: Optional[Dict[str, float]] = None,
        wait_bias: float = DEFAULT_WAIT_BIAS,
    ):
        self.rng = rng
        self.min_dwell_steps = max(1, int(min_dwell_steps))
        self.blocked_detour_threshold = blocked_detour_threshold
        self.temperature = temperature
        self.persistence_bonus = persistence_bonus
        self.detour_boost = detour_boost
        self.wait_bias = wait_bias
        self.behavior_weights = dict(behavior_weights or DEFAULT_BEHAVIOR_WEIGHTS)
        self.state = PolicyState()

    def reset(self) -> None:
        self.state = PolicyState()

    # ---------------- intention ----------------
    def intention_distribution(
        self,
        goal: Position,
        current: Position,
        blocked_count: int,
        prev_intention: Optional[str],
        dwell_remaining: int,
        history: Sequence[str],
        obstacles: Optional[Sequence[Position]] = None,
        bounds: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        return intention_distribution(
            behavior_weights=self.behavior_weights,
            goal=goal,
            current=current,
            prev_intention=prev_intention,
            history=history,
            blocked_count=blocked_count,
            obstacles=obstacles,
            bounds=bounds,
            temperature=self.temperature,
            min_dwell_steps=self.min_dwell_steps,
            blocked_detour_threshold=self.blocked_detour_threshold,
            persistence_bonus=self.persistence_bonus,
            detour_boost=self.detour_boost,
            wait_bias=self.wait_bias,
        )

    def select_intention(
        self,
        goal: Position,
        current: Position,
        history: Sequence[str],
        blocked_count: int,
        obstacles: Optional[Sequence[Position]] = None,
        bounds: Optional[Tuple[int, int]] = None,
    ) -> str:
        s = self.state
        interrupted = (
            blocked_count >= self.blocked_detour_threshold
            or (s.prev_intention is not None and _manhattan(current, goal) == 0)
        )
        if s.dwell_remaining > 0 and s.prev_intention is not None and not interrupted:
            s.dwell_remaining -= 1
            s.blocked_count = blocked_count
            return s.prev_intention

        dist = self.intention_distribution(
            goal=goal,
            current=current,
            blocked_count=blocked_count,
            prev_intention=s.prev_intention,
            dwell_remaining=s.dwell_remaining,
            history=history,
            obstacles=obstacles,
            bounds=bounds,
        )
        # Sorted categorical sampling: iteration order independent of dict
        # insertion order and therefore of PYTHONHASHSEED.
        r = self.rng.random()
        acc = 0.0
        chosen = sorted(dist)[-1]
        for name in sorted(dist):
            acc += dist[name]
            if r <= acc:
                chosen = name
                break
        s.prev_intention = chosen
        s.dwell_remaining = self.min_dwell_steps - 1
        s.blocked_count = blocked_count
        return chosen

    # ---------------- action ----------------
    def action_distribution(
        self,
        intention: str,
        goal: Position,
        current: Position,
        obstacles: Optional[Sequence[Position]] = None,
        bounds: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        return action_distribution(intention, goal, current, obstacles, bounds)

    def select_action(
        self,
        intention: str,
        goal: Position,
        current: Position,
        peers: Sequence[Position] = (),
        policy_sensing_radius: int = 3,
        obstacles: Optional[Sequence[Position]] = None,
        bounds: Optional[Tuple[int, int]] = None,
        peer_avoidance_weight: Optional[float] = None,
    ) -> str:
        """Sample A_t ~ P(A | I_t, G, s). Peer avoidance is a local, sensing
        radius-bounded re-weighting of the same distribution."""
        dist = action_distribution(intention, goal, current, obstacles, bounds)
        weight = (
            PEER_AVOIDANCE_WEIGHT if peer_avoidance_weight is None else peer_avoidance_weight
        )
        if peers and policy_sensing_radius and intention != Intention.WAIT.value:
            near = [p for p in peers if _manhattan(p, current) <= policy_sensing_radius]
            if near:
                adjusted = dict(dist)
                for action, prob in dist.items():
                    if action == "WAIT":
                        continue
                    nxt = _destination(current, action)
                    if any(_manhattan(nxt, p) == 0 for p in near):
                        adjusted[action] = prob * weight
                dist = _normalize(adjusted)
        r = self.rng.random()
        acc = 0.0
        chosen = sorted(dist)[-1]
        for action in sorted(dist):
            acc += dist[action]
            if r <= acc:
                chosen = action
                break
        return chosen
