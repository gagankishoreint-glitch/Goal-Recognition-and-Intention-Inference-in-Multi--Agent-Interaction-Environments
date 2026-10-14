"""Metric tests: corner cases, NTCD semantics, and the cluster bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (
    brier_score,
    convergence_failure_rate,
    evaluate_goal_predictions,
    macro_f1_from_predictions,
    normalize,
    nll,
    ntcd,
    ntcd_episode,
    paired_gap_ci,
    scenario_bootstrap_ci,
    to_matrix,
    top_k_accuracy,
)

K = 4


def onehot(index: int, k: int = K) -> np.ndarray:
    v = np.zeros(k)
    v[index] = 1.0
    return v


def uniform(k: int = K) -> np.ndarray:
    return np.full(k, 1.0 / k)


def test_to_matrix_and_normalize() -> None:
    assert to_matrix([0.25, 0.25, 0.25, 0.25]).shape == (1, 4)
    assert to_matrix([[1.0, 1.0], [2.0, 2.0]]).shape == (2, 2)
    out = normalize(np.array([[1.0, 3.0]]))
    assert np.allclose(out, [[0.25, 0.75]])
    # a zero row does not produce NaNs
    assert np.allclose(normalize(np.array([[0.0, 0.0]])), [[0.0, 0.0]])


def test_top_k_accuracy() -> None:
    probs = np.array([onehot(0), onehot(1), onehot(2)])
    labels = np.array([0, 1, 3])
    assert top_k_accuracy(probs, labels, 1) == pytest.approx(2 / 3)
    two = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    # top-1 misses both; top-2 recovers both (1 is second for row 0, 3 first for row 1)
    assert top_k_accuracy(two, np.array([1, 0]), 1) == 0.0
    assert top_k_accuracy(two, np.array([1, 3]), 2) == 1.0
    assert np.isnan(top_k_accuracy(np.zeros((0, K)), np.zeros(0, dtype=int), 1))


def test_nll_and_brier_corner_cases() -> None:
    probs = np.array([uniform() for _ in range(5)])
    labels = np.array([0, 1, 2, 3, 0])
    assert nll(probs, labels) == pytest.approx(np.log(K))
    perfect = np.array([onehot(l) for l in labels])
    assert nll(perfect, labels) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(perfect, labels) == pytest.approx(0.0, abs=1e-9)
    expected_brier = (1 - 1 / K) ** 2 + (K - 1) * (1 / K) ** 2
    assert brier_score(probs, labels) == pytest.approx(expected_brier)
    # a confident wrong answer is worse than a uniform guess
    wrong = np.array([onehot(1)])
    assert nll(wrong, np.array([0])) > nll(np.array([uniform()]), np.array([0]))


def test_macro_f1_matches_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(4, 40))
        k = int(rng.integers(2, 6))
        y = rng.integers(0, k, n)
        p = rng.integers(0, k, n)
        present = sorted(set(y.tolist()) | set(p.tolist()))
        expected = f1_score(y, p, labels=present, average="macro", zero_division=0)
        assert macro_f1_from_predictions(p, y, k) == pytest.approx(expected, abs=1e-12)
    # a perfect classifier scores 1.0 and a constant classifier does not
    y = np.array([0, 1, 2, 3])
    assert macro_f1_from_predictions(y, y, 4) == pytest.approx(1.0)
    # a constant classifier only scores on the class it always predicts:
    # f1 = [0.4, 0, 0, 0] -> macro 0.1
    assert macro_f1_from_predictions(np.zeros(4, dtype=int), y, 4) == pytest.approx(0.1)


def _belief(n_steps: int, goal: int, prob: float, flip_at: int = None, other: int = None) -> list:
    """Belief trajectory that predicts ``goal`` with mass ``prob``."""
    other = 1 if other is None or other == goal else other
    seq = []
    for t in range(n_steps):
        v = np.full(K, (1.0 - prob) / (K - 1))
        v[goal] = prob
        if flip_at is not None and t >= flip_at:
            v = np.full(K, (1.0 - prob) / (K - 1))
            v[other] = prob
        seq.append(v.tolist())
    return seq


def test_ntcd_semantics() -> None:
    # correct and confident from the first step -> NTCD = 0
    assert ntcd_episode(_belief(10, 2, 0.9), 2, 0.7) == pytest.approx(0.0)
    # never correct -> NTCD = 1
    assert ntcd_episode(_belief(10, 2, 0.9), 3, 0.7) == pytest.approx(1.0)
    # correct but never confident enough -> never-correct treatment
    assert ntcd_episode(_belief(10, 2, 0.4), 2, 0.7) == pytest.approx(1.0)
    # correct early, then wrong: the stability requirement makes it never-correct
    seq = _belief(10, 2, 0.9, flip_at=5)
    assert ntcd_episode(seq, 2, 0.7) == pytest.approx(1.0)
    # becomes correct and stays correct at step 4 of 10 -> 4 / (T - 1)
    late = []
    for t in range(10):
        v = np.full(K, (1 - 0.3) / (K - 1))
        v[0 if t < 4 else 2] = 0.3 if t < 4 else 0.9
        late.append(v.tolist())
    assert ntcd_episode(late, 2, 0.7) == pytest.approx(4 / 9)
    assert ntcd([_belief(10, 2, 0.9), _belief(10, 2, 0.9)], [2, 3], 0.7) == pytest.approx(0.5)
    assert convergence_failure_rate([_belief(10, 2, 0.9), _belief(10, 2, 0.9)], [2, 3], 0.7) == 0.5


def test_scenario_bootstrap_needs_multiple_clusters() -> None:
    probs = np.array([uniform() for _ in range(10)])
    labels = np.arange(10) % K
    assert scenario_bootstrap_ci(probs, labels, ["s0"] * 10, K, num_resamples=100) == {}


def test_cluster_bootstrap_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(3)
    n_scenarios, per_scenario = 20, 8
    probs, labels, sids = [], [], []
    for s in range(n_scenarios):
        for _ in range(per_scenario):
            g = int(rng.integers(0, K))
            # an imperfect predictor: a quarter of the samples put the mass on a
            # wrong class, so the bootstrap interval is non-degenerate
            predicted = g if rng.random() > 0.25 else int((g + 1) % K)
            p = np.full(K, 0.05)
            p[predicted] = 0.85
            probs.append(p / p.sum())
            labels.append(g)
            sids.append(f"s{s}")
    probs = np.array(probs)
    labels = np.array(labels)
    ci = scenario_bootstrap_ci(probs, labels, sids, K, num_resamples=500, seed=0)
    point = {
        "accuracy": top_k_accuracy(probs, labels, 1),
        "top2_accuracy": top_k_accuracy(probs, labels, 2),
        "macro_f1": macro_f1_from_predictions(probs.argmax(axis=1), labels, K),
        "nll": nll(probs, labels),
        "brier": brier_score(probs, labels),
    }
    assert set(point) == set(ci), (sorted(point), sorted(ci))
    for key, (lo, hi) in ci.items():
        assert lo <= point[key] <= hi, f"{key}: {point[key]} outside [{lo}, {hi}]"
        assert lo < hi, f"{key}: degenerate interval"


def test_paired_gap_of_a_model_with_itself_is_zero() -> None:
    rng = np.random.default_rng(5)
    probs = rng.dirichlet(np.ones(K), size=60)
    labels = rng.integers(0, K, 60)
    sids = [f"s{i % 10}" for i in range(60)]
    gap = paired_gap_ci(probs, probs, labels, sids, metric="macro_f1", num_resamples=200, seed=1)
    assert gap["point_estimate"] == pytest.approx(0.0, abs=1e-12)
    assert gap["ci_lower"] <= 0.0 <= gap["ci_upper"]
    assert gap["n_scenarios"] == 10


def test_evaluate_goal_predictions_bundle() -> None:
    rng = np.random.default_rng(7)
    n = 40
    probs = rng.dirichlet(np.ones(K) * 2, size=n)
    labels = probs.argmax(axis=1)
    sids = [f"s{i % 8}" for i in range(n)]
    bundle = evaluate_goal_predictions("m", probs, labels, sids, num_resamples=200, seed=2)
    assert bundle.n_episodes == n
    assert bundle.n_scenarios == 8
    assert bundle.accuracy == pytest.approx(1.0)
    assert bundle.macro_f1 == pytest.approx(1.0)
    assert bundle.nll < 1.0
    d = bundle.as_dict()
    assert {"name", "accuracy", "top2_accuracy", "macro_f1", "nll", "brier"} <= set(d)
    assert "accuracy_ci95" in d and len(d["accuracy_ci95"]) == 2
