"""Tests for the time-step / label contract (TIMING_CONTRACT.md).

The headline property is proved by mutation rather than inspection: the
observation at step ``t`` must not change when the *outcomes* of step ``t`` or of
any later step are altered. A canary test then proves the detector is sensitive
by showing that a deliberately leaky feature builder is caught.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.timing import (
    ACTIONS,
    FRAME_ALLOWED_PREVIOUS_KEYS,
    FRAME_ALLOWED_STEP_KEYS,
    TASKS,
    TimingContractViolation,
    focal_steps,
    frame_features,
    frame_width,
    labels,
    leaky_frame_features,
    observable_frame,
    stacked_features,
    supervised_dataset,
)

ACTIONS_SET = set(ACTIONS)


@pytest.fixture(scope="module")
def records():
    from verification.m2.dataset_m2 import build_splits

    splits = build_splits(num_scenarios=8, episodes_per_scenario=4)
    return splits["train"], splits["test"]


# --------------------------------------------------------------------------
# contract shape
# --------------------------------------------------------------------------
def test_frame_at_t_zero_has_no_previous_behaviour(records) -> None:
    train, _ = records
    for record in train[:5]:
        frame = observable_frame(record, 0)
        assert frame["previous_action"] is None
        assert frame["previous_blocked"] is None
        assert frame["focal_cell"] == tuple(focal_steps(record)[0]["position_before"])


def test_frame_carries_the_previous_step_outcome_only(records) -> None:
    train, _ = records
    for record in train[:5]:
        steps = focal_steps(record)
        for t in range(1, len(steps)):
            frame = observable_frame(record, t)
            assert frame["previous_action"] == steps[t - 1]["realized_action"]
            assert frame["previous_blocked"] == bool(steps[t - 1]["blocked"])
            # the frame carries no post-transition fields at all
            assert "realized_action" not in frame and "position" not in frame
            assert "blocked" not in frame and "intention" not in frame


def test_frame_out_of_range_raises(records) -> None:
    train, _ = records
    record = train[0]
    with pytest.raises(TimingContractViolation):
        observable_frame(record, len(focal_steps(record)))
    with pytest.raises(TimingContractViolation):
        observable_frame(record, -1)


# --------------------------------------------------------------------------
# the mutation proof
# --------------------------------------------------------------------------
def _mutate(record, from_step, fields):
    out = copy.deepcopy(record)
    for step in out["steps"]:
        if step["step"] >= from_step and step["agent_id"] == out["focal_agent_id"]:
            for field, value in fields.items():
                if field in step:
                    step[field] = value
    return out


def test_outcomes_of_step_t_and_later_do_not_change_the_observation(records) -> None:
    """The core no-leakage property, proved by mutation."""
    train, _ = records
    mutations = {
        "realized_action": "WAIT",
        "blocked": True,
        "position": [0, 0],
        "intended_action": "UP",
        "intention": "WAIT",
        "blocked_reason": "conflict",
        "goal": [0, 0],
    }
    for record in train[:6]:
        steps = focal_steps(record)
        for t in (0, 1, len(steps) // 2, len(steps) - 1):
            before = observable_frame(record, t)
            mutated = _mutate(record, t, mutations)
            after = observable_frame(mutated, t)
            assert before == after, f"O_{t} changed when step >= {t} outcomes were mutated"
            assert frame_features(before) == frame_features(after)


def test_frame_reads_only_the_allowlisted_step_keys(records) -> None:
    """Mechanically mutate every non-allowlisted key and require no change."""
    train, _ = records
    for record in train[:4]:
        steps = focal_steps(record)
        t = min(2, len(steps) - 1)
        base = observable_frame(record, t)
        candidate_keys = set(steps[t]) | set(steps[t - 1] if t else {})
        forbidden = candidate_keys - (
            FRAME_ALLOWED_STEP_KEYS | FRAME_ALLOWED_PREVIOUS_KEYS | {"agent_id"}
        )
        assert forbidden, "expected the step record to carry privileged keys to test against"
        for key in sorted(forbidden):
            mutated = copy.deepcopy(record)
            for step in mutated["steps"]:
                if step["agent_id"] == mutated["focal_agent_id"] and key in step:
                    step[key] = "SENTINEL" if isinstance(step[key], str) else [99, 99]
            assert observable_frame(mutated, t) == base, f"frame reads forbidden key {key!r}"


def test_frame_is_prefix_invariant(records) -> None:
    from src.trajectory.generator import episode_prefix

    train, _ = records
    for record in train[:6]:
        length = len(focal_steps(record))
        for t in range(length):
            truncated = episode_prefix(record, t + 1)
            assert observable_frame(truncated, t) == observable_frame(record, t)
            assert frame_features(observable_frame(truncated, t)) == frame_features(
                observable_frame(record, t)
            )


def test_features_are_independent_of_the_hidden_goal(records) -> None:
    """Relabelling the privileged goal must not move a single feature."""
    train, _ = records
    for record in train[:6]:
        base = [frame_features(observable_frame(record, t)) for t in range(len(focal_steps(record)))]
        relabelled = copy.deepcopy(record)
        relabelled["privileged_ground_truth"]["focal_goal_id"] = (
            relabelled["privileged_ground_truth"]["focal_goal_id"] + 1
        ) % len(record["candidate_targets"])
        for step in relabelled["steps"]:
            step["goal"] = [99, 99]
        moved = [
            frame_features(observable_frame(relabelled, t)) for t in range(len(focal_steps(relabelled)))
        ]
        assert base == moved


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def test_labels_align_with_frame_indices(records) -> None:
    train, _ = records
    for record in train[:6]:
        steps = focal_steps(record)
        lab = labels(record)
        assert lab["length"] == len(steps)
        assert lab["intentions"] == [s["intention"] for s in steps]
        assert lab["actions"] == [s["realized_action"] for s in steps]
        assert all(a in ACTIONS_SET for a in lab["actions"])
        assert lab["goal"] == record["privileged_ground_truth"]["focal_goal_id"]


def test_supervised_dataset_shapes_and_alignment(records) -> None:
    train, test = records
    for task in TASKS:
        x, y, sids = supervised_dataset(train, task, window=1)
        expected_rows = sum(len(focal_steps(r)) for r in train)
        assert x.shape[0] == expected_rows == len(y) == len(sids)
        assert set(sids) == {r["scenario_id"] for r in train}
        if task == "goal":
            assert y.max() < len(train[0]["candidate_targets"])
        elif task == "intention":
            assert y.max() < 4
        else:
            assert y.max() < len(ACTIONS)
    # windowed features are exactly window x frame_width
    width = frame_width(train[0], include_behavior=True)
    x3, _, _ = supervised_dataset(train[:2], "action", window=3)
    assert x3.shape[1] == 3 * width
    x1, _, _ = supervised_dataset(train[:2], "action", window=1)
    assert x1.shape[1] == width
    # P1-class geometry features are narrower (no previous-behaviour block)
    xg, _, _ = supervised_dataset(train[:2], "action", window=1, include_behavior=False)
    assert xg.shape[1] == frame_width(train[0], include_behavior=False) < width
    with pytest.raises(ValueError):
        supervised_dataset(train, "next_intention")


# --------------------------------------------------------------------------
# canary: the detector must be able to catch leakage
# --------------------------------------------------------------------------
def test_leakage_canary_detects_an_off_by_one_builder(records) -> None:
    """A leaky builder is caught; the contract builder is not near-perfect."""
    train, test = records
    x_leak_tr, y_leak_tr, _ = supervised_dataset(train, "action", window=1, leaky=True)
    x_leak_te, y_leak_te, _ = supervised_dataset(test, "action", window=1, leaky=True)
    leak = LogisticRegression(max_iter=1000, random_state=0).fit(x_leak_tr, y_leak_tr)
    leak_acc = float((leak.predict(x_leak_te) == y_leak_te).mean())

    x_tr, y_tr, _ = supervised_dataset(train, "action", window=1)
    x_te, y_te, _ = supervised_dataset(test, "action", window=1)
    clean = LogisticRegression(max_iter=1000, random_state=0).fit(x_tr, y_tr)
    clean_acc = float((clean.predict(x_te) == y_te).mean())

    assert leak_acc > 0.95, (
        f"the leaky builder should be trivially predictable (got {leak_acc:.3f}); "
        "the canary cannot prove anything if it cannot detect the leak"
    )
    assert clean_acc < leak_acc - 0.3, (
        f"contract features predict A_t at {clean_acc:.3f}, too close to the leaky "
        f"{leak_acc:.3f}: the observation is leaking the action it must predict"
    )


def test_intention_task_is_not_trivially_leaky(records) -> None:
    """History helps (contract), but a leaky window is dramatically better."""
    train, test = records
    x_tr, y_tr, _ = supervised_dataset(train, "intention", window=1)
    x_te, y_te, _ = supervised_dataset(test, "intention", window=1)
    clean = LogisticRegression(max_iter=1000, random_state=0).fit(x_tr, y_tr)
    clean_acc = float((clean.predict(x_te) == y_te).mean())
    assert clean_acc < 0.95, f"single-frame intention accuracy {clean_acc:.3f} is implausibly high"
