"""Intention triviality probe.

Shallow single-step classifier uses ONLY current-step observable
information (position, velocity). Compared against a majority baseline
and a history-aware reference (windowed features) on a held-out
scenario split.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _current_features(step: Dict[str, Any], grid_w: int, grid_h: int) -> List[float]:
    x, y = step["position"]
    return [x / grid_w, y / grid_h]


def _window_features(
    steps: List[Dict[str, Any]], idx: int, grid_w: int, grid_h: int, window: int = 4
) -> List[float]:
    feats: List[float] = []
    start = max(0, idx - window + 1)
    segment = steps[start : idx + 1]
    for s in segment:
        x, y = s["position"]
        feats.extend([x / grid_w, y / grid_h])
    while len(segment) < window:
        feats.extend([0.0, 0.0]) if len(feats) < 2 * window else None
        segment = segment + [segment[0]] if segment else [{"position": (0, 0)}]
        if 2 * len(segment) > len(feats):
            break
    # pad to fixed length
    while len(feats) < 2 * window:
        feats.append(0.0)
    return feats[: 2 * window]


def _focal_steps(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    focal = episode["focal_agent_id"]
    return [s for s in episode["steps"] if s["agent_id"] == focal]


def evaluate_triviality(
    train_episodes: List[Dict[str, Any]],
    test_episodes: List[Dict[str, Any]],
    grid_w: int = 16,
    grid_h: int = 16,
) -> Dict[str, Dict[str, float]]:
    X_tr, y_tr = [], []
    X_tr_hist, y_tr_hist = [], []
    for ep in train_episodes:
        steps = _focal_steps(ep)
        for i, s in enumerate(steps):
            X_tr.append(_current_features(s, grid_w, grid_h))
            y_tr.append(s["intention"])
            X_tr_hist.append(_window_features(steps, i, grid_w, grid_h))
            y_tr_hist.append(s["intention"])

    X_te, y_te = [], []
    X_te_hist, y_te_hist = [], []
    for ep in test_episodes:
        steps = _focal_steps(ep)
        for i, s in enumerate(steps):
            X_te.append(_current_features(s, grid_w, grid_h))
            y_te.append(s["intention"])
            X_te_hist.append(_window_features(steps, i, grid_w, grid_h))
            y_te_hist.append(s["intention"])

    # majority baseline
    majority = max(set(y_tr), key=y_tr.count)
    maj_acc = sum(1 for v in y_te if v == majority) / max(len(y_te), 1)
    maj_f1 = f1_score(y_te, [majority] * len(y_te), average="macro")

    shallow = LogisticRegression(max_iter=500)
    shallow.fit(X_tr, y_tr)
    pred = shallow.predict(X_te)
    shallow_acc = accuracy_score(y_te, pred)
    shallow_f1 = f1_score(y_te, pred, average="macro")

    hist = LogisticRegression(max_iter=500)
    hist.fit(X_tr_hist, y_tr_hist)
    pred_h = hist.predict(X_te_hist)
    hist_acc = accuracy_score(y_te_hist, pred_h)
    hist_f1 = f1_score(y_te_hist, pred_h, average="macro")

    return {
        "majority": {"accuracy": maj_acc, "macro_f1": maj_f1},
        "shallow_single_step": {"accuracy": shallow_acc, "macro_f1": shallow_f1},
        "history_aware": {"accuracy": hist_acc, "macro_f1": hist_f1},
    }


def interpret(result: Dict[str, Dict[str, float]]) -> str:
    s = result["shallow_single_step"]["accuracy"]
    h = result["history_aware"]["accuracy"]
    m = result["majority"]["accuracy"]
    if s >= h - 0.05 and s > m + 0.05:
        return (
            "POTENTIALLY TRIVIAL: shallow single-step classifier matches the "
            "history-aware model; revisit the intention generator."
        )
    return (
        "NON-TRIVIAL: history-aware model provides meaningful additional "
        "information over the shallow single-step baseline."
    )
