"""Correctness and learning checks for the NumPy multi-task GRU.

The BPTT implementation is verified against finite differences, and the model is
checked to learn the two tasks above chance on the real data — both are
preconditions for reporting any learned-model number in the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.timing import frame_features, observable_frame
from src.models.gru import MultiTaskGRU, numerical_gradient_check
from src.trajectory.dataset import DatasetConfig, generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sequence(record, window_len=None):
    from src.features.timing import focal_steps, labels

    steps = focal_steps(record)
    X = np.asarray([frame_features(observable_frame(record, t)) for t in range(len(steps))])
    lab = labels(record)
    intentions = np.asarray(
        [{"DIRECT": 0, "DETOUR_LEFT": 1, "DETOUR_RIGHT": 2, "WAIT": 3}[i]
         for i in lab["intentions"]], dtype=int
    )
    if window_len is not None:
        X = X[:window_len]
        intentions = intentions[:window_len]
    return X, int(lab["goal"]), intentions


def test_bptt_matches_finite_differences() -> None:
    rng = np.random.default_rng(0)
    model = MultiTaskGRU(input_dim=8, hidden=6, n_goals=3, n_intentions=4, seed=1)
    X = rng.normal(size=(7, 8))
    intentions = rng.integers(0, 4, size=7)
    worst = numerical_gradient_check(model, X, goal=1, intentions=intentions, alpha_index=4)
    assert worst < 1e-4, f"BPTT gradients wrong: max relative error {worst:.2e}"


def test_gru_learns_goal_and_intention_above_chance() -> None:
    cfg = DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))
    recs = generate_dataset(cfg, master_seed=20261008, num_scenarios=4, episodes_per_scenario=2)
    seqs = [_sequence(r) for r in recs]
    D = seqs[0][0].shape[1]
    model = MultiTaskGRU(input_dim=D, hidden=24, n_goals=4, n_intentions=4, seed=0)

    from src.models.gru import Adam

    opt = Adam(model.p, lr=5e-3)
    for epoch in range(6):
        for X, goal, intentions in seqs:
            caches, _ = model.forward(X)
            g, loss, _, _ = model.backward(caches, X, goal, intentions,
                                           alpha_index=len(X) - 1)
            opt.step(model.p, g)

    # evaluate on the same (small) set: should clear chance for both heads
    goal_correct = goal_total = 0
    intent_correct = intent_total = 0
    for X, goal, intentions in seqs:
        _, hs = model.forward(X)
        gp = np.exp(model.goal_logits(hs[-1]))
        gp /= gp.sum()
        goal_correct += int(gp.argmax() == goal)
        goal_total += 1
        ip = np.exp(model.intention_logits(hs))
        ip /= ip.sum(axis=1, keepdims=True)
        intent_correct += int((ip.argmax(axis=1) == intentions).sum())
        intent_total += len(intentions)
    assert goal_correct / goal_total > 0.25, "goal head at chance"
    assert intent_correct / intent_total > 0.25, "intention head at chance"
