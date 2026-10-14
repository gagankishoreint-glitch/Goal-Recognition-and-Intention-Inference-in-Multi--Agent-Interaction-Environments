"""Time-step and label contract: what a model may see, and what it must infer.

This module is the single implementation of the contract written in
``TIMING_CONTRACT.md``. Every supervised sample used by a probe, a baseline or
(M3) a sequence model is built here, so the contract cannot drift from the code.

Index convention (0-based, matching ``src.trajectory.generator``)
-----------------------------------------------------------------
At step ``t``:

1. the agent occupies ``position_before[t]`` (the start cell when ``t = 0``,
   otherwise ``position[t-1]``);
2. the policy chooses the hidden intention ``I_t`` and an intended action;
3. the environment transitions and the step record stores ``position[t]``
   (post-transition), ``realized_action[t]`` (the action that produced the
   transition) and ``blocked[t]``.

Therefore the observation available when ``I_t`` is chosen is

    O_t = ( focal cell before the step,
            peer cells at the same instant,
            realized_action[t-1],
            blocked[t-1],
            static map M, candidate targets )

and ``O_t`` excludes ``A_t = realized_action[t]``, ``I_t``, ``position[t]`` and
everything after ``t``. For ``t = 0`` there is no ``t-1``; the previous-behaviour
features are an explicit "no previous step" flag rather than a silent zero that
could be confused with a real ``WAIT``.

Tasks
-----
``goal``       label ``G`` (episode-level, constant) from ``O_{1:t}``
``intention``  label ``I_t`` from ``O_{1:t}``
``action``     label ``A_t`` — the action *generated at* step ``t`` — from
               ``O_{1:t}``. In PROJECT_SPEC's 1-based notation this is
               ``P(A_{t+1} | O_{1:t})``.

The leak-proof property is tested by mutation, not by inspection: see
``tests/test_timing_contract.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .observations import validate_batch

Position = Tuple[int, int]

ACTIONS: Tuple[str, ...] = ("DOWN", "LEFT", "RIGHT", "UP", "WAIT")
TASKS: Tuple[str, ...] = ("goal", "intention", "action")

# The only step-record keys a frame may read. Anything else — in particular
# `realized_action`, `blocked` and `position` of the current step — is off
# limits, and `tests/test_timing_contract.py` asserts this allowlist.
FRAME_ALLOWED_STEP_KEYS = frozenset({"step", "position_before", "agent_id"})
FRAME_ALLOWED_PREVIOUS_KEYS = frozenset({"realized_action", "blocked"})

NO_PREVIOUS_STEP: Tuple[float, ...] = (0.0,) * len(ACTIONS) + (0.0, 1.0)


class TimingContractViolation(ValueError):
    pass


def focal_steps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    focal_id = record["focal_agent_id"]
    return sorted(
        (s for s in record["steps"] if s["agent_id"] == focal_id), key=lambda s: s["step"]
    )


def peer_cells_at(record: Dict[str, Any], t: int) -> List[Position]:
    """Peer cells at the instant step ``t`` begins (pre-transition)."""
    out = [
        tuple(s["position_before"])
        for s in record["steps"]
        if s["step"] == t and s["agent_id"] != record["focal_agent_id"]
    ]
    return sorted(out)


def observable_frame(record: Dict[str, Any], t: int) -> Dict[str, Any]:
    """Build ``O_t`` — the observation available when ``I_t`` is chosen."""
    steps = focal_steps(record)
    if not 0 <= t < len(steps):
        raise TimingContractViolation(f"step {t} outside [0, {len(steps)})")
    current = steps[t]
    frame: Dict[str, Any] = {
        "t": t,
        "focal_cell": tuple(current["position_before"]),
        "peer_cells": peer_cells_at(record, t),
        "previous_action": None,
        "previous_blocked": None,
        "grid_dimensions": tuple(record["grid_dimensions"]),
        "obstacles": [tuple(o) for o in record["static_obstacles"]],
        "targets": [tuple(c["position"]) for c in record["candidate_targets"]],
    }
    if t > 0:
        frame["previous_action"] = steps[t - 1]["realized_action"]
        frame["previous_blocked"] = bool(steps[t - 1]["blocked"])
    return frame


def frame_geometry_features(frame: Dict[str, Any], max_peers: int = 3) -> List[float]:
    """P1-class features: current-step observable geometry only."""
    w, h = frame["grid_dimensions"]
    x, y = frame["focal_cell"]
    feats: List[float] = [x / max(w - 1, 1), y / max(h - 1, 1)]

    for tx, ty in frame["targets"]:
        dx, dy = tx - x, ty - y
        feats.extend(
            [dx / w, dy / h, float(abs(dx) + abs(dy)) / (w + h), float(np.hypot(dx, dy)) / (w + h)]
        )

    for i in range(max_peers):
        if i < len(frame["peer_cells"]):
            px, py = frame["peer_cells"][i]
            dx, dy = px - x, py - y
            feats.extend([dx / w, dy / h, float(abs(dx) + abs(dy)) / (w + h), 1.0])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    blocked = set(frame["obstacles"])
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        feats.append(1.0 if (not (0 <= nx < w and 0 <= ny < h)) or (nx, ny) in blocked else 0.0)
    feats.append(len(frame["obstacles"]) / float(w * h))
    return feats


def frame_interaction_features(frame: Dict[str, Any]) -> List[float]:
    """Interaction channels: how peers constrain the focal agent's options.

    These are the features that RQ2 turns on and off. Three are per candidate
    target (so a peer can *discriminate* between goals instead of merely adding
    noise) and four are global. All are computable from ``O_t`` alone: peer cells
    at the same instant, the focal cell, the map, and the previous step's blocked
    flag.

    Every channel here must be specific to *where the peers are*. An earlier
    version also carried "is a peer collinear with the focal cell and target g",
    which was rejected by the shuffled-peer control in
    ``experiments/multi_agent/run_interaction_contrast.py``: the number of
    collinear cells grows with focal-to-target distance, so with peers relocated
    at random the feature degenerates into a distance-to-goal proxy and scores as
    well as the real thing. It is not an interaction feature and it is gone.
    """
    w, h = frame["grid_dimensions"]
    x, y = frame["focal_cell"]
    peers = set(frame["peer_cells"])
    out: List[float] = []
    for tx, ty in frame["targets"]:
        greedy = [
            (x + dx, y + dy)
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
            if abs(x + dx - tx) + abs(y + dy - ty) < abs(x - tx) + abs(y - ty)
        ]
        held = [c for c in greedy if c in peers]
        out.append(1.0 if held else 0.0)
        out.append(1.0 if greedy and len(held) == len(greedy) else 0.0)
        out.append(1.0 if (tx, ty) in peers else 0.0)  # a peer is camping on target g
    adjacent = sum(1 for px, py in frame["peer_cells"] if abs(px - x) + abs(py - y) == 1)
    # 0.0 when no peer is visible: a masked frame is all-zero in the peer
    # channels rather than carrying a synthetic "far away" distance.
    nearest = min((abs(px - x) + abs(py - y) for px, py in frame["peer_cells"]), default=0)
    prev_blocked_adjacent = (
        1.0
        if frame["previous_blocked"]
        and any(abs(px - x) + abs(py - y) <= 1 for px, py in frame["peer_cells"])
        else 0.0
    )
    out += [
        adjacent / 4.0,
        float(nearest) / (w + h),
        prev_blocked_adjacent,
        len(frame["peer_cells"]) / 4.0,
    ]
    return out


def frame_behavior_features(frame: Dict[str, Any]) -> List[float]:
    """P2-class additions: previous realized action (one-hot) and blocked flag."""
    if frame["previous_action"] is None:
        return list(NO_PREVIOUS_STEP)
    onehot = [1.0 if frame["previous_action"] == a else 0.0 for a in ACTIONS]
    return onehot + [1.0 if frame["previous_blocked"] else 0.0, 0.0]


def frame_features(
    frame: Dict[str, Any],
    include_behavior: bool = True,
    include_interaction: bool = True,
    peer_visible: bool = True,
    blocked_visible: bool = True,
) -> List[float]:
    """Full feature vector for one frame.

    ``include_interaction``, ``peer_visible`` and ``blocked_visible`` implement
    the RQ2 masking experiment. Masked channels are **zeroed, not removed**, so
    every condition has identical dimensionality and identical model capacity;
    only information differs. The three switches are independent because they
    are separate claims: peer *positions*, the derived interaction features, and
    the previous step's blocked flag. Conflating them (an earlier version of
    this function dropped the blocked flag together with the peers) makes it
    impossible to say which one a measured gap came from.
    """
    frame = dict(frame)
    if not peer_visible:
        # Emptying ``peer_cells`` is enough to zero the peer slots — the geometry
        # builder always emits ``max_peers`` slots and pads absent peers with
        # zeros. Nothing else in the geometry vector is peer-derived, so walls,
        # obstacle density, target geometry and the previous action stay
        # bit-identical between conditions.
        frame["peer_cells"] = []
    if not blocked_visible:
        frame["previous_blocked"] = None
    feats = frame_geometry_features(frame)
    interaction = frame_interaction_features(dict(frame))
    if include_behavior:
        feats += frame_behavior_features(frame)
    feats += interaction if include_interaction else [0.0] * len(interaction)
    return feats


def frame_width(record: Dict[str, Any], include_behavior: bool = True) -> int:
    return len(frame_features(observable_frame(record, 0), include_behavior))


def stacked_features(
    record: Dict[str, Any],
    t: int,
    window: int,
    include_behavior: bool = True,
    include_interaction: bool = True,
    peer_visible: bool = True,
    blocked_visible: bool = True,
) -> List[float]:
    """Frames ``t, t-1, ..., t-window+1`` concatenated, zero-padded at the start."""
    width = frame_width(record, include_behavior)
    out: List[float] = []
    for k in range(window):
        idx = t - k
        out += (
            frame_features(
                observable_frame(record, idx),
                include_behavior,
                include_interaction,
                peer_visible,
                blocked_visible,
            )
            if idx >= 0
            else [0.0] * width
        )
    return out


def labels(record: Dict[str, Any]) -> Dict[str, Any]:
    """Privileged labels, aligned to the frame index ``t``."""
    steps = focal_steps(record)
    return {
        "goal": int(record["privileged_ground_truth"]["focal_goal_id"]),
        "intentions": [s["intention"] for s in steps],
        "actions": [s["realized_action"] for s in steps],
        "length": len(steps),
    }


def leaky_frame_features(record: Dict[str, Any], t: int) -> List[float]:
    """DELIBERATELY WRONG: includes the outcome of step ``t``.

    Exists only so ``tests/test_timing_contract.py`` can prove the leakage test
    actually fires. Never import this from a probe or a model.
    """
    steps = focal_steps(record)
    feats = frame_geometry_features(observable_frame(record, t))
    feats += [1.0 if steps[t]["realized_action"] == a else 0.0 for a in ACTIONS]
    feats += [float(steps[t]["position"][0]), float(steps[t]["position"][1])]
    return feats


def supervised_dataset(
    records: Sequence[Dict[str, Any]],
    task: str,
    window: int = 1,
    include_behavior: bool = True,
    leaky: bool = False,
    include_interaction: bool = True,
    peer_visible: bool = True,
    blocked_visible: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Feature matrix, integer labels and the scenario id of every sample."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    xs: List[List[float]] = []
    ys: List[int] = []
    sids: List[str] = []
    action_index = {a: i for i, a in enumerate(ACTIONS)}
    from ..agents.intentions import ALL_INTENTIONS

    intention_index = {i: n for n, i in enumerate(ALL_INTENTIONS)}
    for record in records:
        lab = labels(record)
        for t in range(lab["length"]):
            if leaky:
                x = leaky_frame_features(record, t)
            elif window > 1:
                x = stacked_features(
                    record, t, window, include_behavior, include_interaction,
                    peer_visible, blocked_visible,
                )
            else:
                x = frame_features(
                    observable_frame(record, t),
                    include_behavior,
                    include_interaction,
                    peer_visible,
                    blocked_visible,
                )
            if task == "goal":
                y = lab["goal"]
            elif task == "intention":
                y = intention_index[lab["intentions"][t]]
            else:
                y = action_index[lab["actions"][t]]
            xs.append(x)
            ys.append(y)
            sids.append(record["scenario_id"])
    if not xs:
        return np.zeros((0, 1)), np.zeros(0, dtype=int), []
    width = max(len(x) for x in xs)
    xs = [x + [0.0] * (width - len(x)) for x in xs]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int), sids


def assert_observable_only(observation: Dict[str, Any]) -> None:
    """Guard for model-facing batches (delegates to the isolation guard)."""
    validate_batch(observation)
