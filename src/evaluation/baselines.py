"""Goal-recognition baselines (CPU-only, closed form, no training).

All three baselines answer the same question — ``P(G | O_{1:t})`` over the K
candidate targets — so their accuracy / NLL / Brier scores are comparable:

``UniformBaseline``
    The random baseline: constant uniform distribution. This is the floor any
    goal recognizer must beat.

``DistanceHeuristicBaseline``
    Markovian softmax over the negative path cost from the *current* observed
    position to each candidate target. ``distance_metric`` selects
    ``shortest_path`` (obstacle-aware BFS, the default) or ``manhattan``.

``BayesianBeliefBaseline``
    Recursive filter over the joint belief ``b_t(G, I_t)`` (PROJECT_SPEC §2.3).
    Its emission model is the *same* action model that generated the data
    (:func:`src.agents.policy.action_distribution`), so the likelihood is
    well-specified rather than a hand-tuned guess:

        P(A_t | I_t, G, s_{t-1}) = action_distribution(I_t, G, s_{t-1})

    combined with the observable ``blocked`` flag when it is available
    (Condition A) and with a block-rate mixture when it is not (Condition B).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents.intentions import ALL_INTENTIONS
from ..agents.policy import action_distribution as policy_action_distribution
from ..environment.pathfinding import UNREACHABLE, cost_map

Position = Tuple[int, int]

# Unreachable cells are scored with this finite penalty instead of -inf so that
# the softmax stays normalizable while still strongly disfavoured.
UNREACHABLE_COST = 64


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _softmax(scores: Sequence[float]) -> List[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class UniformBaseline:
    """Random / chance baseline: uniform over candidate targets.

    A uniform vector has a degenerate ``argmax`` (always class 0), which would
    report the *class frequency of target 0* instead of chance accuracy. The
    probabilities therefore carry an infinitesimal seeded jitter so that the
    top-1 prediction is a fair random guess (expected accuracy 1/K) while the
    probabilities used for NLL / Brier remain uniform to within 1e-6.
    """

    name = "random"

    JITTER = 1e-6

    def __init__(self, targets: Sequence[Position], seed: int = 0, **_: Any):
        self.targets = [tuple(t) for t in targets]
        self.k = len(self.targets)
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None  # constant distribution: nothing to update

    def goal_distribution(self) -> List[float]:
        jittered = [1.0 / self.k + self.rng.random() * self.JITTER for _ in range(self.k)]
        total = sum(jittered)
        return [v / total for v in jittered]

    def intention_distribution(self) -> List[float]:
        n = len(ALL_INTENTIONS)
        return [1.0 / n] * n


class DistanceHeuristicBaseline:
    """Softmax over negative path cost from the latest observed position."""

    name = "distance"

    def __init__(
        self,
        targets: Sequence[Position],
        beta: float = 0.5,
        obstacles: Optional[Sequence[Position]] = None,
        bounds: Optional[Tuple[int, int]] = None,
        distance_metric: str = "shortest_path",
        **_: Any,
    ):
        self.targets = [tuple(t) for t in targets]
        self.k = len(self.targets)
        self.beta = beta
        self.obstacles = [tuple(o) for o in (obstacles or [])]
        self.bounds = tuple(bounds) if bounds else None
        if distance_metric not in ("shortest_path", "manhattan"):
            raise ValueError(f"unknown distance_metric {distance_metric!r}")
        self.distance_metric = distance_metric
        self._cost_maps: Dict[Position, Dict[Position, int]] = {}
        if self.distance_metric == "shortest_path" and self.bounds:
            w, h = self.bounds
            for t in self.targets:
                self._cost_maps[t] = cost_map(w, h, self.obstacles, t)
        self._position: Optional[Position] = None

    def reset(self) -> None:
        self._position = None

    def cost(self, position: Position, target: Position) -> float:
        if self.distance_metric == "manhattan" or not self.bounds:
            return float(_manhattan(position, target))
        c = self._cost_maps[target].get(position, UNREACHABLE)
        return float(UNREACHABLE_COST if c == UNREACHABLE else c)

    def update(
        self,
        prev_pos: Position,
        pos: Position,
        action: Optional[str] = None,
        blocked: Optional[bool] = None,
    ) -> None:
        """The heuristic is Markovian: only the newest position matters."""
        self._position = tuple(pos)

    def goal_distribution(self) -> List[float]:
        if self._position is None:
            return [1.0 / self.k] * self.k
        return _softmax([-self.beta * self.cost(self._position, t) for t in self.targets])

    def intention_distribution(self) -> List[float]:
        # The heuristic is goal-marginal and carries no intention model; report
        # the induced mixture P(I) = sum_g P(I | g) P(g) with a uniform P(I | g)
        # so that intention NLL stays comparable across baselines.
        n = len(ALL_INTENTIONS)
        return [1.0 / n] * n


class BayesianBeliefBaseline:
    """Recursive joint belief over (goal, intention).

    ``persist`` is the intention self-transition probability; the emission is
    the generative action model. Both are configuration knobs, not fitted
    quantities, which keeps the baseline training-free.
    """

    name = "bayesian"

    def __init__(
        self,
        targets: Sequence[Position],
        obstacles: Optional[Sequence[Position]] = None,
        bounds: Optional[Tuple[int, int]] = None,
        intentions: Sequence[str] = ALL_INTENTIONS,
        persist: float = 0.85,
        beta: float = 1.0,
        block_rate: float = 0.05,
        floor: float = 1e-9,
        **_: Any,
    ):
        self.targets = [tuple(t) for t in targets]
        self.k = len(self.targets)
        self.obstacles = [tuple(o) for o in (obstacles or [])]
        self.bounds = tuple(bounds) if bounds else None
        self.intentions = list(intentions)
        self.persist = persist
        self.beta = beta
        self.block_rate = block_rate
        self.floor = floor
        self.belief: Dict[Tuple[int, str], float] = {}
        self.reset()

    # ---------------- belief state ----------------
    def reset(self) -> None:
        p = 1.0 / (self.k * len(self.intentions))
        self.belief = {(g, i): p for g in range(self.k) for i in self.intentions}

    def _transition(self, i_prev: str, i: str) -> float:
        if i == i_prev:
            return self.persist
        return (1.0 - self.persist) / max(len(self.intentions) - 1, 1)

    def _emission(
        self,
        action: str,
        prev_pos: Position,
        goal: Position,
        intention: str,
        blocked: Optional[bool],
    ) -> float:
        """P(observed realized action | I, G, s_{t-1})."""
        model = policy_action_distribution(
            intention, goal, prev_pos, self.obstacles, self.bounds
        )
        if blocked is True:
            # realized action is WAIT and only tells us the intent was a move
            mass = 1.0 - model.get("WAIT", 0.0)
            return max(mass, self.floor)
        if blocked is False:
            return max(model.get(action, 0.0), self.floor)
        # blocked unobservable (Condition B): mixture over blocked / not blocked
        p_wait = model.get("WAIT", 0.0)
        if action == "WAIT":
            value = (1.0 - self.block_rate) * p_wait + self.block_rate * (1.0 - p_wait)
        else:
            value = (1.0 - self.block_rate) * model.get(action, 0.0)
        return max(value, self.floor)

    def update(
        self,
        prev_pos: Position,
        pos: Optional[Position] = None,
        action: Optional[str] = None,
        blocked: Optional[bool] = None,
    ) -> Dict[Tuple[int, str], float]:
        """One recursive Bayes update; returns the normalized belief."""
        new: Dict[Tuple[int, str], float] = {}
        for (g, i_prev), b in self.belief.items():
            if b <= 0.0:
                continue
            goal = self.targets[g]
            for i in self.intentions:
                p_trans = self._transition(i_prev, i)
                e = self._emission(action, prev_pos, goal, i, blocked) ** self.beta
                key = (g, i)
                new[key] = new.get(key, 0.0) + b * p_trans * e
        total = sum(new.values())
        if total <= 0.0:
            self.reset()
            return dict(self.belief)
        self.belief = {k: v / total for k, v in new.items()}
        return dict(self.belief)

    # ---------------- read-outs ----------------
    def goal_distribution(self) -> List[float]:
        out = [0.0] * self.k
        for (g, _), v in self.belief.items():
            out[g] += v
        return out

    def intention_distribution(self) -> List[float]:
        out = [0.0] * len(self.intentions)
        index = {i: n for n, i in enumerate(self.intentions)}
        for (_, i), v in self.belief.items():
            out[index[i]] += v
        return out


def focal_step_sequence(episode: Dict[str, Any], focal_agent_id: int = 0) -> List[Dict[str, Any]]:
    return sorted(
        (s for s in episode["steps"] if s["agent_id"] == focal_agent_id),
        key=lambda s: s["step"],
    )


def observation_stream(episode: Dict[str, Any], focal_agent_id: int = 0):
    """Yield ``(prev_pos, pos, realized_action, blocked)`` per transition.

    ``blocked`` is part of the Condition A observable schema and ``None`` when
    the episode was projected to Condition B.
    """
    steps = focal_step_sequence(episode, focal_agent_id)
    for prev, cur in zip(steps, steps[1:]):
        yield (
            tuple(prev["position"]),
            tuple(cur["position"]),
            cur["realized_action"],
            bool(cur["blocked"]) if "blocked" in cur else None,
        )


def belief_trajectory(
    baseline: Any,
    episode: Dict[str, Any],
    focal_agent_id: int = 0,
    use_blocked: bool = True,
    max_steps: Optional[int] = None,
) -> List[List[float]]:
    """Goal distribution after every observed transition (index 0 = t=1)."""
    baseline.reset()
    out: List[List[float]] = [list(baseline.goal_distribution())]
    for n, (prev_pos, pos, action, blocked) in enumerate(
        observation_stream(episode, focal_agent_id), start=1
    ):
        if max_steps is not None and n > max_steps:
            break
        baseline.update(prev_pos, pos, action, blocked if use_blocked else None)
        out.append(list(baseline.goal_distribution()))
    return out
