#!/usr/bin/env python3
"""Section 6 — intention triviality probe (frozen protocol).

Probes (M2_VERIFICATION_GUIDE.md §6)
------------------------------------
P0  majority class, no features.
P1  geometry only, current step: focal position, per-target relative vectors
    and distances, per-peer relative vectors and distances, local map
    occupancy. No actions, no blocked flags, no history.
P2  P1 + realized_action_{t-1} (one-hot) + blocked_{t-1}.
P3  P2 stacked over a window of k=5 steps (shallow history reference, *not* a
    sequence model).

Timing convention: ``I_t`` generates ``A_t``, so ``A_t`` is never an input to a
probe predicting ``I_t``. Every sample predicting ``I_t`` is built from the cell
the agent occupied *before* the step (``position_before``), the peer cells at
the same instant, and the previous step's realized action / blocked flag.

Statistics use a **scenario-cluster bootstrap** over the held-out test
scenarios with the probe weights frozen, so the reported interval brackets the
point estimate. (The earlier revision bootstrapped *retrainings*, which is why
its interval [0.0043, 0.1895] did not contain its own point estimate -0.0061.)

Gate: if the 95% CI for the P3 - P1 macro-F1 gap contains zero, intention labels
are geometrically trivial under this protocol → RED (STOP).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.intentions import ALL_INTENTIONS  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    macro_f1_from_predictions,
    normalize,
    paired_gap_ci,
    scenario_bootstrap_ci,
    to_matrix,
    top_k_accuracy,
)
from verification.m2.dataset_m2 import build_splits, splits_summary  # noqa: E402

PROBE_SEED = 42
WINDOW = 5
NUM_RESAMPLES = 2000
GATE_METRIC = "macro_f1"
MAX_CLASS_FLAG = 0.60

REPORT_PATH = REPO_ROOT / "verification" / "m2" / "triviality.md"
STATS_PATH = REPO_ROOT / "verification" / "m2" / "label_stats.md"
RESULTS_JSON = REPO_ROOT / "verification" / "m2" / "triviality.json"


# --------------------------------------------------------------------------
# sample construction
#
# Feature construction is delegated to `src.features.timing`, the single
# implementation of TIMING_CONTRACT.md, so the frozen probe and the contract
# cannot drift apart. P1 = geometry only; P2 = P1 + realized_action_{t-1} +
# blocked_{t-1}; P3 = P2 stacked over a window of k=WINDOW steps.
# --------------------------------------------------------------------------
from src.features.timing import frame_features, observable_frame, stacked_features  # noqa: E402


def build_samples(
    records: List[Dict[str, Any]], probe: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Feature matrix, integer labels and the scenario id of every sample."""
    xs: List[List[float]] = []
    ys: List[int] = []
    sids: List[str] = []
    for record in records:
        intentions = record["privileged_ground_truth"]["step_intentions"]
        for t in range(len(intentions)):
            if probe == "P1":
                x = frame_features(observable_frame(record, t), include_behavior=False)
            elif probe == "P2":
                x = frame_features(observable_frame(record, t), include_behavior=True)
            elif probe == "P3":
                x = stacked_features(record, t, WINDOW, include_behavior=True)
            else:
                raise ValueError(f"unknown probe {probe}")
            xs.append(x)
            ys.append(ALL_INTENTIONS.index(intentions[t]))
            sids.append(record["scenario_id"])
    if not xs:
        return np.zeros((0, 1)), np.zeros(0, dtype=int), []
    width = max(len(x) for x in xs)
    xs = [x + [0.0] * (width - len(x)) for x in xs]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int), sids


def fit_probe(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, random_state=PROBE_SEED, C=1.0)
    model.fit(x, y)
    return model


def evaluate_probe(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scenario_ids: Sequence[str],
    num_resamples: int = NUM_RESAMPLES,
) -> Dict[str, Any]:
    if name == "P0":
        majority = int(np.bincount(y_train, minlength=len(ALL_INTENTIONS)).argmax())
        pred = np.full(y_test.shape, majority, dtype=int)
        probs = np.zeros((len(y_test), len(ALL_INTENTIONS)))
        probs[:, majority] = 1.0
    else:
        model = fit_probe(x_train, y_train)
        pred = model.predict(x_test)
        probs = normalize(model.predict_proba(x_test))
    n_classes = len(ALL_INTENTIONS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=list(range(n_classes)), zero_division=0
    )
    metrics = {
        "accuracy": float((pred == y_test).mean()),
        "macro_f1": macro_f1_from_predictions(pred, y_test, n_classes),
        "n_samples": int(len(y_test)),
        "n_scenarios": len(set(scenario_ids)),
        "per_intention": {
            ALL_INTENTIONS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
    }
    if name != "P0":
        metrics["ci95"] = {
            k: [float(v[0]), float(v[1])]
            for k, v in scenario_bootstrap_ci(
                probs, y_test, scenario_ids, n_classes, num_resamples=num_resamples, seed=PROBE_SEED
            ).items()
            if k in ("accuracy", "macro_f1")
        }
    return metrics


# --------------------------------------------------------------------------
# label statistics required by the guide
# --------------------------------------------------------------------------
def _run_lengths(labels: Sequence[str]) -> List[int]:
    runs: List[int] = []
    if not labels:
        return runs
    cur = 1
    for a, b in zip(labels, labels[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def label_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    intention_counts: Dict[str, int] = {i: 0 for i in ALL_INTENTIONS}
    goal_counts: Dict[int, int] = {}
    dwells: List[int] = []
    switches: List[int] = []
    blocked_total = 0
    steps_total = 0
    reasons: Dict[str, int] = {}
    for record in records:
        labels = record["privileged_ground_truth"]["step_intentions"]
        for label in labels:
            intention_counts[label] += 1
        gid = record["privileged_ground_truth"]["focal_goal_id"]
        goal_counts[gid] = goal_counts.get(gid, 0) + 1
        dwells.extend(_run_lengths(labels))
        switches.append(sum(1 for a, b in zip(labels, labels[1:]) if a != b))
        for s in record["steps"]:
            if s["agent_id"] != record["focal_agent_id"]:
                continue
            steps_total += 1
            if s["blocked"]:
                blocked_total += 1
            reasons[s["blocked_reason"]] = reasons.get(s["blocked_reason"], 0) + 1
    total = sum(intention_counts.values()) or 1
    peer_blocks = reasons.get("occupied", 0) + reasons.get("conflict", 0)
    obstacle_blocks = reasons.get("obstacle", 0) + reasons.get("boundary", 0)
    return {
        "episodes": len(records),
        "scenarios": len({r["scenario_id"] for r in records}),
        "intention_distribution": {k: v / total for k, v in sorted(intention_counts.items())},
        "max_intention_share": max(intention_counts.values()) / total if total else 0.0,
        "goal_distribution": {
            str(k): goal_counts[k] / max(len(records), 1) for k in sorted(goal_counts)
        },
        "dwell_mean": float(np.mean(dwells)) if dwells else 0.0,
        "dwell_median": float(np.median(dwells)) if dwells else 0.0,
        "switches_mean": float(np.mean(switches)) if switches else 0.0,
        "switches_median": float(np.median(switches)) if switches else 0.0,
        "fraction_episodes_with_switch": float(np.mean([s >= 1 for s in switches])) if switches else 0.0,
        "blocked_step_rate": blocked_total / steps_total if steps_total else 0.0,
        "blocked_by_peer": peer_blocks / steps_total if steps_total else 0.0,
        "blocked_by_obstacle_or_boundary": obstacle_blocks / steps_total if steps_total else 0.0,
        "steps": steps_total,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_section6(
    write_report: bool = True, num_resamples: int = NUM_RESAMPLES
) -> Dict[str, Any]:
    splits = build_splits()
    train, test = splits["train"], splits["test"]

    results: Dict[str, Any] = {"probes": {}, "dataset": splits_summary(splits)}
    predictions: Dict[str, np.ndarray] = {}
    probas: Dict[str, np.ndarray] = {}
    _, y_train, _ = build_samples(train, "P1")

    for probe in ("P1", "P2", "P3"):
        x_tr, y_tr, _ = build_samples(train, probe)
        x_te, y_te, sids = build_samples(test, probe)
        results["probes"][probe] = evaluate_probe(
            probe, x_tr, y_tr, x_te, y_te, sids, num_resamples=num_resamples
        )
        model = fit_probe(x_tr, y_tr)
        predictions[probe] = model.predict(x_te)
        probas[probe] = normalize(model.predict_proba(x_te))
        results["feature_dim"] = {**results.get("feature_dim", {}), probe: int(x_te.shape[1])}

    x_te_p1, y_te, sids = build_samples(test, "P1")
    # P0 uses the *training* majority class, evaluated on the test samples
    results["probes"]["P0"] = evaluate_probe(
        "P0", np.zeros((1, 1)), y_train, x_te_p1, y_te, sids, num_resamples=num_resamples
    )

    gap = paired_gap_ci(
        probas["P3"],
        probas["P1"],
        y_te,
        sids,
        metric=GATE_METRIC,
        num_resamples=num_resamples,
        seed=PROBE_SEED,
    )
    acc_gap = paired_gap_ci(
        probas["P3"], probas["P1"], y_te, sids, metric="accuracy",
        num_resamples=num_resamples, seed=PROBE_SEED,
    )
    results["gaps"] = {
        "accuracy": acc_gap,
        GATE_METRIC: gap,
    }
    ci_contains_zero = gap["ci_lower"] <= 0.0 <= gap["ci_upper"]
    results["gate"] = {
        "criterion": "95% scenario-cluster CI of the P3-P1 macro-F1 gap contains zero",
        "ci_contains_zero": bool(ci_contains_zero),
        "verdict": "TRIVIAL (STOP)" if ci_contains_zero else "NON-TRIVIAL (CONTINUE)",
        "status": "RED (STOP)" if ci_contains_zero else "GREEN (CONTINUE)",
    }
    results["label_stats"] = {
        "train": label_statistics(train),
        "test": label_statistics(test),
    }
    results["flags"] = {
        "intention_class_over_60pct": {
            split: results["label_stats"][split]["max_intention_share"] > MAX_CLASS_FLAG
            for split in ("train", "test")
        },
        "median_switches_zero": {
            split: results["label_stats"][split]["switches_median"] == 0
            for split in ("train", "test")
        },
    }

    if write_report:
        REPORT_PATH.write_text(render_report(results), encoding="utf-8")
        STATS_PATH.write_text(render_label_stats(results), encoding="utf-8")
        RESULTS_JSON.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def render_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Intention Triviality Probe Results (Section 6)")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "Shallow probes predicting `I_t` from the observation available *before* the step: "
        "`position_before_t`, peer cells at the same instant, per-target relative geometry and "
        "local map occupancy. `A_t` (which `I_t` generates) is never an input. Trained on training "
        "scenarios only, evaluated on held-out test scenarios, Condition A observable fields, "
        f"fixed probe seed {PROBE_SEED}, one model family (multinomial logistic regression)."
    )
    lines.append("")
    lines.append("- P0: majority class, no features")
    lines.append("- P1: current-step geometry only")
    lines.append("- P2: P1 + realized_action_{t-1} + blocked_{t-1}")
    lines.append(f"- P3: P2 stacked over a window of k={WINDOW} steps (shallow history reference)")
    lines.append("")
    ds = payload["dataset"]["test"]
    lines.append(
        f"Held-out test set: {ds['episodes']} episodes / {ds['scenarios']} scenarios."
    )
    lines.append("")
    lines.append("## Probe Performance")
    lines.append("")
    lines.append("| probe | features | Accuracy | Acc 95% CI | Macro-F1 | Macro-F1 95% CI | samples |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("P0", "P1", "P2", "P3"):
        p = payload["probes"][name]
        ci = p.get("ci95", {})
        dim = payload.get("feature_dim", {}).get(name, 0)

        def fmt_ci(key: str) -> str:
            # P0 is a constant predictor: resampling cannot move it
            if key not in ci:
                return "n/a (constant)"
            return f"[{ci[key][0]:.4f}, {ci[key][1]:.4f}]"

        lines.append(
            f"| {name} | {dim} | {p['accuracy']:.4f} | {fmt_ci('accuracy')} | "
            f"{p['macro_f1']:.4f} | {fmt_ci('macro_f1')} | {p['n_samples']} |"
        )
    lines.append("")
    lines.append("## Per-Intention Precision / Recall / F1")
    lines.append("")
    lines.append("| probe | intention | precision | recall | F1 | support |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("P0", "P1", "P2", "P3"):
        for intent, row in payload["probes"][name]["per_intention"].items():
            lines.append(
                f"| {name} | {intent} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['support']} |"
            )
    lines.append("")
    lines.append("## P3 - P1 Gaps (scenario-cluster bootstrap)")
    lines.append("")
    for metric in ("accuracy", GATE_METRIC):
        g = payload["gaps"][metric]
        lines.append(
            f"- {metric}: point estimate {g['point_estimate']:.4f}, "
            f"95% CI [{g['ci_lower']:.4f}, {g['ci_upper']:.4f}] "
            f"({g['num_resamples']} resamples over {g['n_scenarios']} test scenarios)"
        )
    gap = payload["gaps"][GATE_METRIC]
    inside = gap["ci_lower"] <= gap["point_estimate"] <= gap["ci_upper"]
    lines.append(
        f"- point estimate inside its own CI: {'yes' if inside else 'no'} "
        "(a sanity property of the cluster bootstrap)"
    )
    lines.append("")
    lines.append("## Triviality Gate")
    lines.append("")
    gate = payload["gate"]
    lines.append(f"- Criterion: {gate['criterion']}")
    lines.append(f"- CI contains zero: {'YES' if gate['ci_contains_zero'] else 'NO'}")
    lines.append(f"- Verdict: intention labels are **{gate['verdict']}**")
    lines.append(f"- GATE STATUS: {gate['status']}")
    lines.append("")
    lines.append("## Flags (reported, not silently modified)")
    lines.append("")
    for split in ("train", "test"):
        stats = payload["label_stats"][split]
        lines.append(
            f"- {split}: max intention share {stats['max_intention_share']:.2%} "
            f"({'FLAG >60%' if payload['flags']['intention_class_over_60pct'][split] else 'ok'}), "
            f"median switches/episode {stats['switches_median']:.2f} "
            f"({'FLAG =0' if payload['flags']['median_switches_zero'][split] else 'ok'})"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_label_stats(payload: Dict[str, Any]) -> str:
    lines: List[str] = ["# Label Statistics (Section 6)", ""]
    for split in ("train", "test"):
        s = payload["label_stats"][split]
        lines.append(f"## {split} split")
        lines.append("")
        lines.append(f"- episodes: {s['episodes']} / scenarios: {s['scenarios']} / focal steps: {s['steps']}")
        lines.append("- intention distribution:")
        for k, v in s["intention_distribution"].items():
            lines.append(f"  - {k}: {v:.2%}")
        lines.append(f"- maximum intention share: {s['max_intention_share']:.2%} (flag threshold 60%)")
        lines.append("- goal distribution (focal hidden goal):")
        for k, v in s["goal_distribution"].items():
            lines.append(f"  - target {k}: {v:.2%}")
        lines.append(f"- intention dwell: mean {s['dwell_mean']:.2f}, median {s['dwell_median']:.2f}")
        lines.append(
            f"- switches per episode: mean {s['switches_mean']:.2f}, median {s['switches_median']:.2f}"
        )
        lines.append(f"- episodes with >=1 switch: {s['fraction_episodes_with_switch']:.2%}")
        lines.append(f"- blocked-step rate: {s['blocked_step_rate']:.2%}")
        lines.append(
            f"  - attributable to peer interaction (occupied/conflict): {s['blocked_by_peer']:.2%}"
        )
        lines.append(
            f"  - attributable to obstacle/boundary: {s['blocked_by_obstacle_or_boundary']:.2%}"
        )
        lines.append("")
    lines.append("## Triviality Gate Assessment")
    lines.append("")
    gap = payload["gaps"][GATE_METRIC]
    lines.append(
        f"- 95% CI for P3-P1 Macro-F1 gap: [{gap['ci_lower']:.4f}, {gap['ci_upper']:.4f}], "
        f"point estimate {gap['point_estimate']:.4f}"
    )
    lines.append(f"- CI contains zero: {'YES' if payload['gate']['ci_contains_zero'] else 'NO'}")
    lines.append(f"- Gate status: {payload['gate']['status']}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------
def test_section6_history_beats_geometry() -> None:
    payload = run_section6(write_report=False, num_resamples=400)
    gap = payload["gaps"][GATE_METRIC]
    assert not (gap["ci_lower"] <= 0.0 <= gap["ci_upper"]), (
        "P3-P1 macro-F1 CI contains zero: intention labels are geometrically trivial"
    )
    assert gap["point_estimate"] > 0.0


def test_section6_point_estimate_inside_ci() -> None:
    """Guard against the retraining-bootstrap defect of the earlier revision."""
    payload = run_section6(write_report=False, num_resamples=400)
    gap = payload["gaps"][GATE_METRIC]
    assert gap["ci_lower"] <= gap["point_estimate"] <= gap["ci_upper"]


def test_section6_no_intention_class_dominates() -> None:
    payload = run_section6(write_report=False, num_resamples=1)
    for split in ("train", "test"):
        share = payload["label_stats"][split]["max_intention_share"]
        assert share <= MAX_CLASS_FLAG, f"{split} split has a dominant intention class: {share:.2%}"
        assert payload["label_stats"][split]["switches_median"] > 0


if __name__ == "__main__":
    payload = run_section6()
    print("=== Section 6: Intention Triviality Probe ===")
    for name in ("P0", "P1", "P2", "P3"):
        p = payload["probes"][name]
        print(f"{name}: accuracy={p['accuracy']:.4f} macro_f1={p['macro_f1']:.4f}")
    gap = payload["gaps"][GATE_METRIC]
    print(
        f"P3-P1 macro-F1 gap: {gap['point_estimate']:.4f} "
        f"CI [{gap['ci_lower']:.4f}, {gap['ci_upper']:.4f}]"
    )
    print("GATE STATUS:", payload["gate"]["status"])
    sys.exit(0 if payload["gate"]["status"].startswith("GREEN") else 1)
