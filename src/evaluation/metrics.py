"""Metrics suite for goal recognition (EXPERIMENT_PLAN.md §1.2).

Everything here is plain NumPy / scikit-learn and runs on CPU only.

Confidence intervals use a **scenario-cluster bootstrap**: test scenarios are
resampled with replacement and the metric is recomputed over all episodes of
the sampled scenarios. Episodes from one scenario share a map layout and are
correlated, so resampling episodes independently would understate variance.
Because the resampled point sets always contain the empirical distribution in
expectation, the reported interval brackets the point estimate — unlike a
retraining bootstrap, which can place the point estimate outside its own CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

EPS = 1e-12


def to_matrix(distributions: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(distributions, dtype=float)
    if m.ndim == 1:
        m = m[None, :]
    return m


def normalize(m: np.ndarray) -> np.ndarray:
    totals = m.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return m / totals


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    if probs.shape[0] == 0:
        return float("nan")
    k = min(k, probs.shape[1])
    top = np.argsort(-probs, axis=1)[:, :k]
    hits = (top == labels[:, None]).any(axis=1)
    return float(hits.mean())


def macro_f1_from_predictions(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over the classes present in ``labels`` or ``pred``.

    Vectorized NumPy equivalent of
    ``sklearn.metrics.f1_score(labels, pred, labels=present, average="macro",
    zero_division=0)``; it is used inside the bootstrap where scikit-learn's
    per-call overhead dominates. ``tests/test_metrics.py`` asserts the two agree.
    """
    if pred.shape[0] == 0:
        return float("nan")
    conf = np.bincount(
        pred * n_classes + labels, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)
    tp = np.diag(conf).astype(float)
    support_true = conf.sum(axis=1).astype(float)   # per true class
    support_pred = conf.sum(axis=0).astype(float)   # per predicted class
    fp = support_pred - tp
    fn = support_true - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(
            precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0
        )
    present = np.unique(np.concatenate([np.unique(labels), np.unique(pred)]))
    if present.size == 0:
        return 0.0
    return float(f1[present].mean())


def macro_f1(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    if probs.shape[0] == 0:
        return float("nan")
    return macro_f1_from_predictions(probs.argmax(axis=1), labels, n_classes)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    if probs.shape[0] == 0:
        return float("nan")
    p = np.clip(probs[np.arange(probs.shape[0]), labels], EPS, 1.0)
    return float(-np.log(p).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    if probs.shape[0] == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ntcd_episode(
    belief_sequence: Sequence[Sequence[float]],
    true_goal: int,
    confidence: float = 0.7,
) -> float:
    """Normalized time-to-correct-decision for one episode.

    ``t*`` is the first step at which the prediction is correct *and* stays
    correct with ``P(G*) >= confidence`` for every later step. If no such step
    exists the episode is never-correct and NTCD = 1.
    """
    seq = to_matrix(belief_sequence)
    T = seq.shape[0]
    if T == 0:
        return 1.0
    for t in range(T):
        tail = seq[t:]
        if not (tail.argmax(axis=1) == true_goal).all():
            continue
        if (tail[:, true_goal] >= confidence).all():
            return float(t) / max(T - 1, 1)
    return 1.0


def ntcd(
    belief_sequences: Sequence[Sequence[Sequence[float]]],
    true_goals: Sequence[int],
    confidence: float = 0.7,
) -> float:
    if len(belief_sequences) == 0:
        return float("nan")
    values = [
        ntcd_episode(seq, int(g), confidence) for seq, g in zip(belief_sequences, true_goals)
    ]
    return float(np.mean(values))


def convergence_failure_rate(
    belief_sequences: Sequence[Sequence[Sequence[float]]],
    true_goals: Sequence[int],
    confidence: float = 0.7,
) -> float:
    """Fraction of episodes that never reach a stable correct decision."""
    if len(belief_sequences) == 0:
        return float("nan")
    values = [
        ntcd_episode(seq, int(g), confidence) >= 1.0
        for seq, g in zip(belief_sequences, true_goals)
    ]
    return float(np.mean(values))


@dataclass
class MetricBundle:
    """Point estimates plus optional scenario-cluster bootstrap intervals."""

    name: str
    accuracy: float
    top2_accuracy: float
    macro_f1: float
    nll: float
    brier: float
    n_episodes: int
    n_scenarios: int
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        out = {
            "name": self.name,
            "accuracy": self.accuracy,
            "top2_accuracy": self.top2_accuracy,
            "macro_f1": self.macro_f1,
            "nll": self.nll,
            "brier": self.brier,
            "n_episodes": self.n_episodes,
            "n_scenarios": self.n_scenarios,
        }
        for k, (lo, hi) in self.ci.items():
            out[f"{k}_ci95"] = [lo, hi]
        return out


def _metric_vector(
    probs: np.ndarray, labels: np.ndarray, n_classes: int
) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == labels).mean()),
        "top2_accuracy": top_k_accuracy(probs, labels, 2),
        "macro_f1": macro_f1_from_predictions(pred, labels, n_classes),
        "nll": nll(probs, labels),
        "brier": brier_score(probs, labels),
    }


def scenario_bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    scenario_ids: Sequence[str],
    n_classes: int,
    num_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Percentile CI of each metric under scenario-level resampling."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(scenario_ids)
    unique = np.unique(ids)
    if len(unique) < 2:
        return {}
    groups = {u: np.flatnonzero(ids == u) for u in unique}
    alpha = (1.0 - level) / 2.0
    samples: Dict[str, List[float]] = {
        k: [] for k in ("accuracy", "top2_accuracy", "macro_f1", "nll", "brier")
    }
    for _ in range(num_resamples):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[u] for u in draw])
        values = _metric_vector(probs[idx], labels[idx], n_classes)
        for k, v in values.items():
            if not np.isnan(v):
                samples[k].append(v)
    out: Dict[str, Tuple[float, float]] = {}
    for k, vals in samples.items():
        if vals:
            arr = np.asarray(vals)
            out[k] = (float(np.percentile(arr, 100 * alpha)), float(np.percentile(arr, 100 * (1 - alpha))))
    return out


def evaluate_goal_predictions(
    name: str,
    distributions: Sequence[Sequence[float]],
    labels: Sequence[int],
    scenario_ids: Sequence[str],
    num_resamples: int = 2000,
    seed: int = 42,
) -> MetricBundle:
    probs = normalize(to_matrix(distributions))
    y = np.asarray(labels, dtype=int)
    n_classes = probs.shape[1]
    point = _metric_vector(probs, y, n_classes)
    ci = scenario_bootstrap_ci(
        probs, y, scenario_ids, n_classes, num_resamples=num_resamples, seed=seed
    )
    return MetricBundle(
        name=name,
        n_episodes=int(probs.shape[0]),
        n_scenarios=len(set(scenario_ids)),
        ci=ci,
        **point,
    )


def paired_gap_ci(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    scenario_ids: Sequence[str],
    metric: str = "macro_f1",
    num_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Scenario-cluster bootstrap CI for the paired gap ``metric(a) - metric(b)``."""
    n_classes = probs_a.shape[1]
    ids = np.asarray(scenario_ids)
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    groups = {u: np.flatnonzero(ids == u) for u in unique}

    def value(pa: np.ndarray, pb: np.ndarray, idx: np.ndarray) -> float:
        return _metric_vector(pa[idx], labels[idx], n_classes)[metric] - _metric_vector(
            pb[idx], labels[idx], n_classes
        )[metric]

    point = value(probs_a, probs_b, np.arange(len(labels)))
    draws = []
    for _ in range(num_resamples):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[u] for u in draw])
        draws.append(value(probs_a, probs_b, idx))
    arr = np.asarray(draws)
    alpha = (1.0 - level) / 2.0
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci_lower": float(np.percentile(arr, 100 * alpha)),
        "ci_upper": float(np.percentile(arr, 100 * (1 - alpha))),
        "num_resamples": int(num_resamples),
        "n_scenarios": int(len(unique)),
    }
