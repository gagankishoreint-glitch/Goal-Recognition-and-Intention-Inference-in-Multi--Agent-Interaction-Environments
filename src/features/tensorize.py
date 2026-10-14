"""Turn an *observable* record into a numeric tensor for model input.

``episode_tensor`` is the last stage of the isolation pipeline: it accepts only
the observable projection (validated by :func:`validate_batch`), so a privileged
field cannot reach a model input tensor even by accident.

Condition B is a strict projection of Condition A: the same function is used and
the peer-derived column block is simply absent, which is what makes the
Condition A vs Condition B comparison (RQ2) an observation-mask comparison
rather than a change of feature pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .observations import validate_batch

ACTION_ORDER = ("DOWN", "LEFT", "RIGHT", "UP", "WAIT")


def _normalize_pair(x: int, y: int, dims: Sequence[int]) -> List[float]:
    w = max(dims[0] - 1, 1)
    h = max(dims[1] - 1, 1)
    return [x / w, y / h]


def feature_names(observation: Dict[str, Any]) -> List[str]:
    """Column names of ``episode_tensor`` (used by ablations and audits)."""
    dims = observation["grid_dimensions"]
    names = ["focal_x", "focal_y", "focal_prev_x", "focal_prev_y"]
    names += [f"action_{a}" for a in ACTION_ORDER]
    if not observation.get("interaction_mask", False):
        names.append("blocked")
    for target in observation["candidate_targets"]:
        tid = target["target_id"]
        names += [
            f"target_{tid}_rel_x",
            f"target_{tid}_rel_y",
            f"target_{tid}_manhattan",
            f"target_{tid}_euclidean",
        ]
    if not observation.get("interaction_mask", False):
        peers = sorted({e["agent_id"] for e in observation.get("peer_agents_trajectory", [])})
        for pid in peers:
            names += [f"peer_{pid}_rel_x", f"peer_{pid}_rel_y", f"peer_{pid}_manhattan", f"peer_{pid}_present"]
    return names


def episode_tensor(observation: Dict[str, Any]) -> np.ndarray:
    """``[T, F]`` float array of observable features, one row per step."""
    validate_batch(observation)
    dims = observation["grid_dimensions"]
    w = max(dims[0] - 1, 1)
    h = max(dims[1] - 1, 1)
    span = float(np.hypot(dims[0], dims[1]))
    masked = bool(observation.get("interaction_mask", False))

    peers_by_step: Dict[int, Dict[int, List[int]]] = {}
    if not masked:
        for entry in observation.get("peer_agents_trajectory", []):
            peers_by_step.setdefault(entry["step"], {})[entry["agent_id"]] = list(entry["position"])
    peer_ids = sorted({pid for by_step in peers_by_step.values() for pid in by_step})

    rows: List[List[float]] = []
    for entry in observation["focal_trajectory"]:
        pos = list(entry["position"])
        prev = list(entry.get("position_before", pos))
        row: List[float] = _normalize_pair(pos[0], pos[1], dims) + _normalize_pair(
            prev[0], prev[1], dims
        )
        row += [1.0 if entry["realized_action"] == a else 0.0 for a in ACTION_ORDER]
        if not masked:
            row.append(1.0 if entry.get("blocked") else 0.0)
        for target in observation["candidate_targets"]:
            tx, ty = target["position"]
            dx, dy = tx - pos[0], ty - pos[1]
            row += [dx / w, dy / h, float(abs(dx) + abs(dy)) / (w + h), float(np.hypot(dx, dy)) / span]
        if not masked:
            step_peers = peers_by_step.get(entry["step"], {})
            for pid in peer_ids:
                if pid in step_peers:
                    px, py = step_peers[pid]
                    dx, dy = px - pos[0], py - pos[1]
                    row += [dx / w, dy / h, float(abs(dx) + abs(dy)) / (w + h), 1.0]
                else:
                    row += [0.0, 0.0, 0.0, 0.0]
        rows.append(row)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)


def batch_tensor(observations: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Stack episodes into ``[B, T, F]`` with zero padding to the longest one."""
    mats = [episode_tensor(o) for o in observations]
    if not mats:
        return np.zeros((0, 0, 0), dtype=float)
    width = max(m.shape[1] for m in mats)
    length = max(m.shape[0] for m in mats)
    out = np.zeros((len(mats), length, width), dtype=float)
    for i, m in enumerate(mats):
        out[i, : m.shape[0], : m.shape[1]] = m
    return out
