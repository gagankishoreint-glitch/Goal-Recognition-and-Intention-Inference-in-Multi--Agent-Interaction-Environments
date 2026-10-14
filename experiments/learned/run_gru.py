#!/usr/bin/env python3
"""The learned sequence model (reviewer #4) and its RQ2 interaction contrast.

Trains the NumPy multi-task GRU (``src/models/gru.py``) on the train split over
the timing-contract features, then compares it against the random / distance /
Bayesian baselines on the held-out test scenarios (same splits, same label
contract, same metrics), and runs the interaction contrast the pre-study left
open: does a *sequence* model use peer trajectories over time? That is measured
as the paired gap Condition A (full) minus Condition D (peer channels replaced
by Gaussian noise, ``TIMING_CONTRACT.md`` §8) on the same trained model.

Run:
    python experiments/learned/run_gru.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines import (  # noqa: E402
    BayesianBeliefBaseline,
    DistanceHeuristicBaseline,
    UniformBaseline,
    belief_trajectory,
)
from src.evaluation.metrics import (  # noqa: E402
    evaluate_goal_predictions,
    macro_f1_from_predictions,
    normalize,
    paired_gap_ci,
    to_matrix,
)
from src.features.timing import focal_steps, frame_features, labels, observable_frame  # noqa: E402
from src.models.gru import Adam, MultiTaskGRU, softmax  # noqa: E402
from verification.m2.dataset_m2 import build_splits, verification_config  # noqa: E402

HIDDEN = 48
EPOCHS = 8
LR = 3e-3
ALPHAS: Tuple[float, ...] = (0.30, 0.50, 0.75, 1.00)
NUM_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
INTENTION_INDEX = {"DIRECT": 0, "DETOUR_LEFT": 1, "DETOUR_RIGHT": 2, "WAIT": 3}

OUT_DIR = REPO_ROOT / "experiments" / "learned"

K = 4
PEER_S, PEER_E = 2 + 4 * K, 2 + 4 * K + 12
TAIL = 3 * K + 4


def sequence_for(record: Dict[str, Any], mask: str, seed_offset: int = 0) -> Dict[str, Any]:
    """X (T, D), goal, intentions, scenario_id for one episode under a mask.

    mask in {"A", "D"}: A keeps peer + interaction + blocked channels; D keeps the
    blocked flag but substitutes Gaussian noise for the peer slots and the
    interaction block (the §8 protocol).
    """
    steps = focal_steps(record)
    T = len(steps)
    rows = []
    for t in range(T):
        frame = observable_frame(record, t)
        x = frame_features(
            frame,
            include_behavior=True,
            include_interaction=(mask == "A"),
            peer_visible=(mask == "A"),
            blocked_visible=True,
        )
        if mask == "D":
            rng = np.random.default_rng(hash(record["episode_id"]) % (2 ** 31) + t + seed_offset)
            x[PEER_S:PEER_E] = rng.standard_normal(PEER_E - PEER_S)
            x[len(x) - TAIL:] = rng.standard_normal(TAIL)
        rows.append(x)
    lab = labels(record)
    return {
        "X": np.asarray(rows, dtype=float),
        "goal": int(lab["goal"]),
        "intentions": np.asarray([INTENTION_INDEX[i] for i in lab["intentions"]], dtype=int),
        "scenario_id": record["scenario_id"],
        "record": record,
    }


def alpha_index(T: int, alpha: float) -> int:
    return int(min(T - 1, max(0, round(alpha * (T - 1)))))


def gru_goal_probs(model: MultiTaskGRU, seqs, alpha: float) -> np.ndarray:
    rows = []
    for s in seqs:
        _, hs = model.forward(s["X"])
        idx = alpha_index(len(hs), alpha)
        rows.append(softmax(model.goal_logits(hs[idx])))
    return np.asarray(rows)


def gru_intention_probs(model: MultiTaskGRU, seqs) -> np.ndarray:
    rows = []
    for s in seqs:
        _, hs = model.forward(s["X"])
        rows.append(softmax(model.intention_logits(hs), axis=1))
    return rows


def baseline_goal_probs(name: str, seqs, alpha: float) -> np.ndarray:
    rows = []
    for s in seqs:
        rec = s["record"]
        targets = [tuple(t["position"]) for t in rec["candidate_targets"]]
        obstacles = [tuple(o) for o in rec["static_obstacles"]]
        bounds = tuple(rec["grid_dimensions"])
        if name == "random":
            baseline = UniformBaseline(targets, seed=1)
        elif name == "distance":
            baseline = DistanceHeuristicBaseline(targets, obstacles=obstacles, bounds=bounds)
        else:
            baseline = BayesianBeliefBaseline(targets, obstacles=obstacles, bounds=bounds)
        traj = belief_trajectory(baseline, rec, rec["focal_agent_id"], use_blocked=True)
        last = len(traj) - 1
        idx = int(min(last, max(0, round(alpha * last))))
        rows.append(traj[idx])
    return normalize(to_matrix(rows))


def main() -> int:
    t0 = time.time()
    cfg = verification_config()
    splits = build_splits()
    train = [sequence_for(r, "A") for r in splits["train"]]
    test_a = [sequence_for(r, "A") for r in splits["test"]]
    test_d = [sequence_for(r, "D") for r in splits["test"]]
    D = train[0]["X"].shape[1]

    model = MultiTaskGRU(input_dim=D, hidden=HIDDEN, n_goals=K, n_intentions=4, seed=0)
    opt = Adam(model.p, lr=LR)
    for epoch in range(EPOCHS):
        order = np.random.default_rng(epoch).permutation(len(train))
        tot = 0.0
        for i in order:
            s = train[i]
            caches, _ = model.forward(s["X"])
            g, loss, _, _ = model.backward(caches, s["X"], s["goal"], s["intentions"],
                                           alpha_index(len(s["X"]), 0.75))
            opt.step(model.p, g)
            tot += loss
        print(f"epoch {epoch}: mean loss {tot / len(train):.4f}", flush=True)

    results: Dict[str, Any] = {
        "config": {
            "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR, "input_dim": D,
            "alphas": list(ALPHAS), "num_resamples": NUM_RESAMPLES,
        },
        "goal_by_alpha": {},
        "intention": {},
        "interaction_contrast": {},
    }

    test_labels = np.asarray([s["goal"] for s in test_a])
    test_sids = [s["scenario_id"] for s in test_a]

    for alpha in ALPHAS:
        entry: Dict[str, Any] = {}
        probs_gru = gru_goal_probs(model, test_a, alpha)
        entry["gru"] = evaluate_goal_predictions(
            "gru", probs_gru, test_labels, test_sids,
            num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED,
        ).as_dict()
        for name in ("random", "distance", "bayesian"):
            probs = baseline_goal_probs(name, test_a, alpha)
            entry[name] = evaluate_goal_predictions(
                name, probs, test_labels, test_sids,
                num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED,
            ).as_dict()
        results["goal_by_alpha"][f"{alpha:.2f}"] = entry

    # intention per-step on the test split
    intent_rows = []
    intent_y = []
    intent_sid = []
    for s in test_a:
        probs = softmax(model.intention_logits(model.forward(s["X"])[1]), axis=1)
        for t in range(len(s["intentions"])):
            intent_rows.append(probs[t])
            intent_y.append(s["intentions"][t])
            intent_sid.append(s["scenario_id"])
    intent_y = np.asarray(intent_y)
    intent_probs = np.asarray(intent_rows)
    intent_pred = intent_probs.argmax(axis=1)
    results["intention"]["gru"] = {
        "accuracy": float((intent_pred == intent_y).mean()),
        "macro_f1": macro_f1_from_predictions(intent_pred, intent_y, 4),
        "n_samples": int(len(intent_y)),
    }

    # interaction contrast for the GRU: same model, A vs D at test time
    for task in ("goal", "intention"):
        if task == "goal":
            pa = gru_goal_probs(model, test_a, 0.75)
            pd = gru_goal_probs(model, test_d, 0.75)
            y = test_labels
            sids = test_sids
        else:
            pa_rows, pd_rows, y_rows, sid_rows = [], [], [], []
            for sa, sd in zip(test_a, test_d):
                ha = model.forward(sa["X"])[1]
                hd = model.forward(sd["X"])[1]
                ia = softmax(model.intention_logits(ha), axis=1)
                idn = softmax(model.intention_logits(hd), axis=1)
                for t in range(len(sa["intentions"])):
                    pa_rows.append(ia[t]); pd_rows.append(idn[t])
                    y_rows.append(sa["intentions"][t]); sid_rows.append(sa["scenario_id"])
            pa, pd = np.asarray(pa_rows), np.asarray(pd_rows)
            y, sids = np.asarray(y_rows), sid_rows
        gap = paired_gap_ci(pa, pd, y, sids, metric="accuracy",
                            num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED)
        gap_f1 = paired_gap_ci(pa, pd, y, sids, metric="macro_f1",
                               num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED)
        results["interaction_contrast"][task] = {
            "accuracy_A": float((pa.argmax(-1) == y).mean()),
            "accuracy_D": float((pd.argmax(-1) == y).mean()),
            "gap_accuracy": gap,
            "gap_macro_f1": gap_f1,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gru_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (OUT_DIR / "gru_results.md").write_text(render(results))
    print(render(results))
    print(f"[{time.time() - t0:.1f}s]")
    return 0


def _sig(g):
    return "significant" if (g["ci_lower"] > 0 or g["ci_upper"] < 0) else "not significant"


def render(r: Dict[str, Any]) -> str:
    L = ["# Multi-Task GRU Results", "",
         f"GRU hidden {r['config']['hidden']}, {r['config']['epochs']} epochs, Adam lr "
         f"{r['config']['lr']}; input {r['config']['input_dim']} timing-contract channels per "
         "frame; goal head read at the alpha-truncated hidden state.", "",
         "## Goal recognition vs baselines (test scenarios)", "",
         "| alpha | model | accuracy | macro-F1 | NLL |", "|---|---|---|---|---|"]
    for alpha, entry in r["goal_by_alpha"].items():
        for name in ("gru", "bayesian", "distance", "random"):
            e = entry[name]
            L.append(f"| {alpha} | {name} | {e['accuracy']:.4f} | {e['macro_f1']:.4f} | {e['nll']:.4f} |")
    L += ["", "## Intention inference (per step, test scenarios)", "",
          "| model | accuracy | macro-F1 | n |", "|---|---|---|---|"]
    gi = r["intention"]["gru"]
    L.append(f"| gru | {gi['accuracy']:.4f} | {gi['macro_f1']:.4f} | {gi['n_samples']} |")
    L += ["", "## RQ2 interaction contrast for the GRU (A full vs D noise-masked)", "",
          "| task | acc A | acc D | acc gap (A-D) | 95% CI | macro-F1 gap |",
          "|---|---|---|---|---|---|"]
    for task, c in r["interaction_contrast"].items():
        g, gf = c["gap_accuracy"], c["gap_macro_f1"]
        L.append(
            f"| {task} | {c['accuracy_A']:.4f} | {c['accuracy_D']:.4f} | "
            f"{g['point_estimate']:+.4f} ({_sig(g)}) | "
            f"[{g['ci_lower']:+.4f}, {g['ci_upper']:+.4f}] | {gf['point_estimate']:+.4f} |"
        )
    L += ["",
          "A gap counts as supported only when its 95% CI excludes zero. The masked arm uses "
          "Gaussian noise substitution, never zeroing (TIMING_CONTRACT.md §8).", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
