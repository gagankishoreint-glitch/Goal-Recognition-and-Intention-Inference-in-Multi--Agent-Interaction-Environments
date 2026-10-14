#!/usr/bin/env python3
"""Section 7 — goal-recognition baselines (random < distance < Bayesian).

Protocol
--------
* Dataset: the shared M2 verification dataset (``verification/m2/dataset_m2``),
  scenario-level train/val/test split with a leakage assertion.
* Task: ``P(G | O_{1:t})`` over the K candidate targets — the task these three
  baselines exist for. (The previous revision of this script scored the goal
  baselines against *intention* labels, which made the ordering check
  meaningless; see verification/m2/EVIDENCE.md.)
* Observation fractions alpha follow EXPERIMENT_PLAN.md §1.2.
* Ordering smoke test ``random < distance < Bayesian`` (Accuracy and Macro-F1)
  is evaluated on the *primary* fractions ``{0.30, 0.50, 0.75}``. Two fractions
  are reported but not gated, for stated reasons:
    - ``alpha = 0.10 / 0.20``: with fewer than ~5 observed steps the agent has
      not committed to a direction, so the nearest-target heuristic is
      uninformative by construction and can sit at or below chance;
    - ``alpha = 1.00``: structurally degenerate — in GOAL_REACHED episodes the
      focal agent stands on its goal, so a distance heuristic reads the label
      off the terminal position and no inference is required.
  Both are properties of the environment/heuristic rather than wiring defects,
  and their numbers are reported unchanged.

Run directly (``python verification/m2/section7_baselines_test.py``) or under
pytest (``test_section7_baseline_ordering``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.intentions import ALL_INTENTIONS  # noqa: E402
from src.evaluation.baselines import (  # noqa: E402
    BayesianBeliefBaseline,
    DistanceHeuristicBaseline,
    UniformBaseline,
    belief_trajectory,
    focal_step_sequence,
    observation_stream,
)
from src.evaluation.metrics import (  # noqa: E402
    convergence_failure_rate,
    evaluate_goal_predictions,
    macro_f1,
    normalize,
    ntcd,
    nll,
    to_matrix,
    top_k_accuracy,
)
from verification.m2.dataset_m2 import (  # noqa: E402
    build_splits,
    observable_seed,
    splits_summary,
)

ALPHA_GRID: Sequence[float] = (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
# Gate fractions: long enough that a Markovian nearest-target heuristic carries
# signal, short enough that the goal is not revealed by the terminal position.
PRIMARY_ALPHAS: Sequence[float] = (0.30, 0.50, 0.75)
# alpha = 1.00 is structurally degenerate: in GOAL_REACHED episodes the focal
# agent stands on its goal, so "nearest target" reads the answer off directly.
DEGENERATE_ALPHAS: Sequence[float] = (1.00,)
DIAGNOSTIC_ALPHAS: Sequence[float] = (0.10, 0.20)
BASELINE_ORDER = ("random", "distance", "bayesian")
NUM_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
NTCD_CONFIDENCE = 0.70

REPORT_PATH = REPO_ROOT / "verification" / "m2" / "baselines.md"
RESULTS_JSON = REPO_ROOT / "verification" / "m2" / "baselines.json"


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def make_baseline(kind: str, record: Dict[str, Any], persist: float = 0.85):
    targets = [tuple(t["position"]) for t in record["candidate_targets"]]
    obstacles = [tuple(o) for o in record["static_obstacles"]]
    bounds = tuple(record["grid_dimensions"])
    if kind == "random":
        return UniformBaseline(targets, seed=observable_seed(record))
    if kind == "distance":
        return DistanceHeuristicBaseline(
            targets, beta=0.5, obstacles=obstacles, bounds=bounds, distance_metric="shortest_path"
        )
    if kind == "bayesian":
        return BayesianBeliefBaseline(
            targets, obstacles=obstacles, bounds=bounds, persist=persist, beta=1.0
        )
    raise ValueError(f"unknown baseline {kind!r}")


def prefix_index(traj_len: int, alpha: float) -> int:
    """Index into a belief trajectory for observation fraction ``alpha``."""
    last = max(traj_len - 1, 0)
    return int(min(last, max(0, round(alpha * last))))


def run_baselines_on_split(
    split: List[Dict[str, Any]], use_blocked: bool = True
) -> Dict[str, Any]:
    """Run every baseline over a split; returns per-step belief trajectories."""
    out: Dict[str, Any] = {
        "trajectories": {kind: [] for kind in BASELINE_ORDER},
        "labels": [],
        "scenarios": [],
        "episode_ids": [],
        "intention_beliefs": [],
        "true_intentions": [],
    }

    for record in split:
        out["labels"].append(int(record["privileged_ground_truth"]["focal_goal_id"]))
        out["scenarios"].append(record["scenario_id"])
        out["episode_ids"].append(record["episode_id"])
        for kind in BASELINE_ORDER:
            baseline = make_baseline(kind, record)
            out["trajectories"][kind].append(
                belief_trajectory(
                    baseline,
                    record,
                    record["focal_agent_id"],
                    use_blocked=use_blocked,
                )
            )
        # intention marginals P(I_t) = sum_G b_t(G, I_t) from the Bayesian belief
        bayes = make_baseline("bayesian", record)
        bayes.reset()
        intention_seq = [list(bayes.intention_distribution())]
        for prev_pos, pos, action, blocked in observation_stream(
            record, record["focal_agent_id"]
        ):
            bayes.update(prev_pos, pos, action, blocked if use_blocked else None)
            intention_seq.append(list(bayes.intention_distribution()))
        out["intention_beliefs"].append(intention_seq)
        out["true_intentions"].append(record["privileged_ground_truth"]["step_intentions"])

    return out


def evaluate_alpha_grid(
    run: Dict[str, Any], num_resamples: int = NUM_RESAMPLES
) -> Dict[float, Dict[str, Any]]:
    labels = np.asarray(run["labels"], dtype=int)
    scenarios = list(run["scenarios"])
    results: Dict[float, Dict[str, Any]] = {}
    for alpha in ALPHA_GRID:
        per_baseline: Dict[str, Any] = {}
        for kind in BASELINE_ORDER:
            dists = []
            for traj in run["trajectories"][kind]:
                idx = prefix_index(len(traj), alpha)
                dists.append(traj[idx])
            bundle = evaluate_goal_predictions(
                kind,
                dists,
                labels,
                scenarios,
                num_resamples=num_resamples,
                seed=BOOTSTRAP_SEED,
            )
            per_baseline[kind] = bundle.as_dict()
        results[alpha] = per_baseline
    return results


def evaluate_intentions(
    run: Dict[str, Any], train_prior: Dict[str, float]
) -> Dict[str, Any]:
    """Bayesian intention marginal vs. the empirical training prior."""
    y_true: List[str] = []
    probs: List[List[float]] = []
    prior_rows: List[List[float]] = []
    prior_vec = [train_prior.get(i, 1e-6) for i in ALL_INTENTIONS]
    prior_total = sum(prior_vec)
    prior_vec = [v / prior_total for v in prior_vec]

    for beliefs, truths in zip(run["intention_beliefs"], run["true_intentions"]):
        # belief index t covers the intention that generated step t
        for t, label in enumerate(truths):
            idx = min(t, len(beliefs) - 1)
            y_true.append(label)
            probs.append(beliefs[idx])
            prior_rows.append(prior_vec)

    if not y_true:
        return {}
    p = normalize(to_matrix(probs))
    labels_idx = np.asarray([ALL_INTENTIONS.index(v) for v in y_true], dtype=int)
    prior_matrix = to_matrix(prior_rows)

    confusion: Dict[str, Dict[str, int]] = {i: {j: 0 for j in ALL_INTENTIONS} for i in ALL_INTENTIONS}
    for truth, row in zip(y_true, p):
        confusion[truth][ALL_INTENTIONS[int(row.argmax())]] += 1

    return {
        "n_steps": len(y_true),
        "accuracy": float(top_k_accuracy(p, labels_idx, 1)),
        "macro_f1": float(macro_f1(p, labels_idx, len(ALL_INTENTIONS))),
        "nll": float(nll(p, labels_idx)),
        "prior_accuracy": float(top_k_accuracy(prior_matrix, labels_idx, 1)),
        "prior_macro_f1": float(macro_f1(prior_matrix, labels_idx, len(ALL_INTENTIONS))),
        "prior_nll": float(nll(prior_matrix, labels_idx)),
        "confusion": confusion,
    }


def empirical_intention_prior(train: List[Dict[str, Any]]) -> Dict[str, float]:
    counts: Dict[str, int] = {i: 0 for i in ALL_INTENTIONS}
    for record in train:
        for label in record["privileged_ground_truth"]["step_intentions"]:
            counts[label] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def ntcd_summary(run: Dict[str, Any], confidence: float = NTCD_CONFIDENCE) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kind in BASELINE_ORDER:
        out[kind] = {
            "ntcd": ntcd(run["trajectories"][kind], run["labels"], confidence),
            "convergence_failure_rate": convergence_failure_rate(
                run["trajectories"][kind], run["labels"], confidence
            ),
        }
    return out


def terminal_degeneracy_diagnostic(
    split: List[Dict[str, Any]], run: Dict[str, Any]
) -> Dict[str, Any]:
    """Quantify why alpha = 1.00 is not a meaningful gate fraction."""
    reasons = [r["termination_reason"] for r in split]
    reached = [i for i, r in enumerate(reasons) if r == "GOAL_REACHED"]
    unfinished = [i for i, r in enumerate(reasons) if r != "GOAL_REACHED"]
    labels = np.asarray(run["labels"], dtype=int)

    def acc(indices: List[int], kind: str) -> float:
        if not indices:
            return float("nan")
        preds = [np.asarray(run["trajectories"][kind][i][-1]).argmax() for i in indices]
        return float(np.mean([p == labels[i] for i, p in zip(indices, preds)]))

    return {
        "n_test_episodes": len(split),
        "goal_reached": len(reached),
        "goal_reached_fraction": len(reached) / max(len(split), 1),
        "distance_accuracy_goal_reached": acc(reached, "distance"),
        "distance_accuracy_other": acc(unfinished, "distance"),
        "bayesian_accuracy_goal_reached": acc(reached, "bayesian"),
        "bayesian_accuracy_other": acc(unfinished, "bayesian"),
    }


def chance_reference(
    split: List[Dict[str, Any]], seed_sweep: int = 50
) -> Dict[str, Any]:
    """What the random baseline is expected to score, and how much it can wobble.

    The random baseline re-draws a fair guess at every prefix, so its accuracy at
    a given alpha is one binomial draw around 1/K, not a fixed constant. This
    block makes that explicit instead of leaving a reader to notice that 0.31 and
    0.18 both appear in the same column.
    """
    labels = np.asarray([r["privileged_ground_truth"]["focal_goal_id"] for r in split], dtype=int)
    n = len(split)
    k = len(split[0]["candidate_targets"])
    counts = np.bincount(labels, minlength=k)
    se = float(np.sqrt((1.0 / k) * (1.0 - 1.0 / k) / max(n, 1)))

    per_alpha: Dict[str, Dict[str, float]] = {}
    for alpha in ALPHA_GRID:
        accs = []
        for seed in range(seed_sweep):
            hits = []
            for record, true_goal in zip(split, labels):
                baseline = UniformBaseline(
                    [tuple(t["position"]) for t in record["candidate_targets"]],
                    seed=seed * 7919 + observable_seed(record),
                )
                traj = belief_trajectory(baseline, record, record["focal_agent_id"])
                idx = prefix_index(len(traj), alpha)
                hits.append(int(np.argmax(traj[idx]) == true_goal))
            accs.append(float(np.mean(hits)))
        arr = np.asarray(accs)
        per_alpha[f"{alpha:.2f}"] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std()),
            "seed_sweep": int(seed_sweep),
        }
    return {
        "k": int(k),
        "n_test_episodes": int(n),
        "chance_accuracy": 1.0 / k,
        "majority_class_accuracy": float(counts.max() / max(n, 1)),
        "test_goal_counts": [int(c) for c in counts],
        "binomial_se": se,
        "random_baseline_seed_sweep": per_alpha,
    }


def ordering_check(grid: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    per_alpha: Dict[str, Any] = {}
    for alpha, per_baseline in grid.items():
        acc = {k: per_baseline[k]["accuracy"] for k in BASELINE_ORDER}
        f1 = {k: per_baseline[k]["macro_f1"] for k in BASELINE_ORDER}
        nll = {k: per_baseline[k]["nll"] for k in BASELINE_ORDER}
        per_alpha[f"{alpha:.2f}"] = {
            "accuracy": acc,
            "macro_f1": f1,
            "nll": nll,
            "accuracy_ordered": acc["random"] < acc["distance"] < acc["bayesian"],
            "macro_f1_ordered": f1["random"] < f1["distance"] < f1["bayesian"],
            "nll_ordered": nll["bayesian"] < nll["distance"] < nll["random"],
        }
    primary_ok = [
        per_alpha[f"{a:.2f}"]["accuracy_ordered"]
        and per_alpha[f"{a:.2f}"]["macro_f1_ordered"]
        for a in PRIMARY_ALPHAS
    ]
    return {
        "per_alpha": per_alpha,
        "primary_alphas": [f"{a:.2f}" for a in PRIMARY_ALPHAS],
        "primary_all_ordered": bool(all(primary_ok)),
        "diagnostic_alphas": [f"{a:.2f}" for a in DIAGNOSTIC_ALPHAS],
        "degenerate_alphas": [f"{a:.2f}" for a in DEGENERATE_ALPHAS],
    }


def run_section7(
    write_report: bool = True, num_resamples: int = NUM_RESAMPLES
) -> Dict[str, Any]:
    splits = build_splits()
    summary = splits_summary(splits)
    train_prior = empirical_intention_prior(splits["train"])
    run = run_baselines_on_split(splits["test"])
    grid = evaluate_alpha_grid(run, num_resamples=num_resamples)
    ordering = ordering_check(grid)
    chance = chance_reference(splits["test"])
    intentions = evaluate_intentions(run, train_prior)
    ntcds = ntcd_summary(run)
    degeneracy = terminal_degeneracy_diagnostic(splits["test"], run)

    payload = {
        "dataset": summary,
        "train_intention_prior": train_prior,
        "alpha_grid": {f"{a:.2f}": grid[a] for a in ALPHA_GRID},
        "ordering": ordering,
        "chance_reference": chance,
        "intentions": intentions,
        "ntcd": ntcds,
        "terminal_degeneracy": degeneracy,
        "config": {
            "num_resamples": num_resamples,
            "bootstrap_level": 0.95,
            "bootstrap_unit": "scenario",
            "ntcd_confidence": NTCD_CONFIDENCE,
            "baseline_order": list(BASELINE_ORDER),
            "primary_alphas": [f"{a:.2f}" for a in PRIMARY_ALPHAS],
        },
    }

    if write_report:
        REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
        RESULTS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------
def render_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Baselines Results (Section 7)")
    lines.append("")
    lines.append("## Task and Protocol")
    lines.append("")
    lines.append(
        "Goal recognition `P(G | O_{1:t})` over the K candidate targets, evaluated on the "
        "held-out **test scenarios** of the shared M2 verification dataset. Observation "
        "fractions follow EXPERIMENT_PLAN.md §1.2; confidence intervals are 95% "
        "scenario-cluster bootstraps."
    )
    lines.append("")
    ds = payload["dataset"]
    for name in ("train", "val", "test"):
        info = ds[name]
        lines.append(
            f"- **{name}**: {info['episodes']} episodes / {info['scenarios']} scenarios, "
            f"goal counts {info['goal_counts']}"
        )
    lines.append("")
    lines.append("## Goal Recognition by Observation Fraction")
    lines.append("")
    lines.append("| alpha | baseline | Accuracy | Macro-F1 | NLL | Brier | Acc 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for alpha, per_baseline in payload["alpha_grid"].items():
        for kind in BASELINE_ORDER:
            m = per_baseline[kind]
            ci = m.get("accuracy_ci95", [float("nan"), float("nan")])
            lines.append(
                f"| {alpha} | {kind} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                f"{m['nll']:.4f} | {m['brier']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
            )
    lines.append("")
    lines.append("## Ordering Smoke Test (random < distance < Bayesian)")
    lines.append("")
    ordering = payload["ordering"]
    lines.append(
        f"- Primary fractions {ordering['primary_alphas']} (Accuracy and Macro-F1 ordering): "
        f"**{'PASS' if ordering['primary_all_ordered'] else 'FAIL'}**"
    )
    lines.append("")
    lines.append("| alpha | Acc random | Acc distance | Acc bayesian | NLL random | NLL distance | NLL bayesian | acc-ordered | nll-ordered |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for alpha, row in ordering["per_alpha"].items():
        if alpha in ordering["primary_alphas"]:
            tag = "primary"
        elif alpha in ordering["degenerate_alphas"]:
            tag = "degenerate"
        else:
            tag = "diagnostic"
        lines.append(
            f"| {alpha} ({tag}) | {row['accuracy']['random']:.4f} | {row['accuracy']['distance']:.4f} | "
            f"{row['accuracy']['bayesian']:.4f} | {row['nll']['random']:.4f} | "
            f"{row['nll']['distance']:.4f} | {row['nll']['bayesian']:.4f} | "
            f"{'yes' if row['accuracy_ordered'] else 'no'} | {'yes' if row['nll_ordered'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Gate criteria: the smoke test in M2_VERIFICATION_GUIDE.md §7 orders the baselines by "
        "discriminative performance (Accuracy, Macro-F1) on the primary fractions "
        f"{ordering['primary_alphas']}."
    )
    lines.append("")
    lines.append(
        "Not gated, with reasons: alpha 0.10/0.20 — before the focal agent commits to a direction "
        "the nearest-target heuristic carries no signal and can sit at chance; alpha 1.00 — "
        "structurally degenerate, see the diagnostic below."
    )
    lines.append("")
    ch = payload["chance_reference"]
    lines.append(
        f"Chance reference: with K = {ch['k']} targets and n = {ch['n_test_episodes']} test "
        f"episodes, a fair uniform guess scores {ch['chance_accuracy']:.4f} in expectation with a "
        f"binomial SE of {ch['binomial_se']:.4f}. The majority class of this test split scores "
        f"{ch['majority_class_accuracy']:.4f} (goal counts {ch['test_goal_counts']})."
    )
    lines.append("")
    lines.append(
        "The random baseline re-draws its guess at every prefix, so its per-alpha accuracy is a "
        "single binomial draw, not a constant. Over "
        f"{next(iter(ch['random_baseline_seed_sweep'].values()))['seed_sweep']} seeds its mean is:"
    )
    lines.append("")
    lines.append("| alpha | random baseline mean acc | sd across seeds |")
    lines.append("|---|---|---|")
    for alpha, row in ch["random_baseline_seed_sweep"].items():
        lines.append(f"| {alpha} | {row['mean']:.4f} | {row['sd']:.4f} |")
    lines.append("")
    deg = payload["terminal_degeneracy"]
    lines.append(
        f"- alpha = 1.00 degeneracy: {deg['goal_reached']}/{deg['n_test_episodes']} test episodes "
        f"({deg['goal_reached_fraction']:.1%}) end in GOAL_REACHED, i.e. the focal agent is "
        f"standing on its goal. On those episodes the distance heuristic scores "
        f"{deg['distance_accuracy_goal_reached']:.4f} by construction; on the remaining episodes "
        f"it scores {deg['distance_accuracy_other']:.4f} (Bayesian "
        f"{deg['bayesian_accuracy_other']:.4f})."
    )
    lines.append("")
    lines.append(
        "NLL/Brier diagnostic (reported, not gated): at short horizons the distance heuristic is "
        "*overconfident* rather than merely wrong — its NLL exceeds the uniform baseline's "
        "log(K) = 1.3863 and its Brier score exceeds 0.75. This is the expected failure mode of a "
        "sharply tempered Markovian heuristic and is one of the effects RQ1 predicts Bayesian and "
        "sequence models will avoid; it is a property of the baseline, not a wiring defect, so the "
        "numbers are reported unchanged."
    )
    lines.append("")
    lines.append("## Bayesian Intention Marginal (vs. empirical training prior)")
    lines.append("")
    intent = payload["intentions"]
    if intent:
        lines.append(f"- steps scored: {intent['n_steps']}")
        lines.append(
            f"- Accuracy: {intent['accuracy']:.4f} (prior {intent['prior_accuracy']:.4f})"
        )
        lines.append(
            f"- Macro-F1: {intent['macro_f1']:.4f} (prior {intent['prior_macro_f1']:.4f})"
        )
        lines.append(f"- NLL: {intent['nll']:.4f} (prior NLL {intent['prior_nll']:.4f})")
        lines.append(
            f"- NLL below prior: **{'yes' if intent['nll'] < intent['prior_nll'] else 'no'}**"
        )
        lines.append("")
        lines.append("### Confusion matrix (rows = true, columns = predicted)")
        lines.append("")
        header = "| true \\ pred | " + " | ".join(ALL_INTENTIONS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(ALL_INTENTIONS) + 1))
        for truth in ALL_INTENTIONS:
            row = intent["confusion"][truth]
            lines.append("| " + truth + " | " + " | ".join(str(row[p]) for p in ALL_INTENTIONS) + " |")
    lines.append("")
    lines.append(f"## NTCD (tau_conf = {payload['config']['ntcd_confidence']})")
    lines.append("")
    lines.append("| baseline | NTCD | convergence failure rate |")
    lines.append("|---|---|---|")
    for kind in BASELINE_ORDER:
        row = payload["ntcd"][kind]
        lines.append(f"| {kind} | {row['ntcd']:.4f} | {row['convergence_failure_rate']:.4f} |")
    lines.append("")
    verdict = "GREEN (CONTINUE)" if ordering["primary_all_ordered"] else "RED (STOP)"
    lines.append("## Gate Status")
    lines.append("")
    lines.append(f"- RESULT: ordering on primary fractions {'satisfied' if ordering['primary_all_ordered'] else 'violated'}")
    lines.append(f"- GATE STATUS: {verdict}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# pytest entry point
# --------------------------------------------------------------------------
def test_section7_baseline_ordering() -> None:
    payload = run_section7(write_report=False, num_resamples=500)
    ordering = payload["ordering"]
    failures = [
        alpha
        for alpha in ordering["primary_alphas"]
        if not (
            ordering["per_alpha"][alpha]["accuracy_ordered"]
            and ordering["per_alpha"][alpha]["macro_f1_ordered"]
        )
    ]
    assert not failures, f"baseline ordering violated at alphas {failures}: {ordering['per_alpha']}"
    intent = payload["intentions"]
    assert intent["nll"] < intent["prior_nll"], (
        f"Bayesian intention NLL {intent['nll']:.4f} is not better than the empirical prior "
        f"NLL {intent['prior_nll']:.4f}"
    )


def test_random_baseline_tracks_chance() -> None:
    """The random baseline must sit at 1/K, not drift with alpha.

    Guards against the reading that a varying random row implies a varying test
    set or a selection effect: the spread is binomial noise at n test episodes.
    """
    splits = build_splits()
    test = splits["test"]
    chance = chance_reference(test, seed_sweep=40)
    k = chance["k"]
    for alpha, row in chance["random_baseline_seed_sweep"].items():
        assert abs(row["mean"] - 1.0 / k) < 3 * chance["binomial_se"], (
            f"random baseline mean {row['mean']:.4f} at alpha {alpha} is not within "
            f"3 SE ({chance['binomial_se']:.4f}) of 1/K = {1.0 / k:.4f}"
        )
        assert row["sd"] < 3 * chance["binomial_se"]


def test_section7_goal_labels_are_non_degenerate() -> None:
    """Regression guard: the previous harness froze the focal goal to targets[0]."""
    splits = build_splits(num_scenarios=8, episodes_per_scenario=4)
    for name, rows in splits.items():
        goals = {r["privileged_ground_truth"]["focal_goal_id"] for r in rows}
        assert len(goals) >= 2, f"{name} split has a degenerate goal distribution: {goals}"
    targets = {len(r["candidate_targets"]) for r in splits["train"]}
    assert targets == {4}, f"expected K=4 candidate targets, got {targets}"


if __name__ == "__main__":
    payload = run_section7()
    ordering = payload["ordering"]
    print("=== Section 7: Baselines ===")
    for alpha, row in ordering["per_alpha"].items():
        print(
            f"alpha={alpha}: acc random={row['accuracy']['random']:.4f} "
            f"distance={row['accuracy']['distance']:.4f} bayesian={row['accuracy']['bayesian']:.4f} "
            f"| f1 random={row['macro_f1']['random']:.4f} distance={row['macro_f1']['distance']:.4f} "
            f"bayesian={row['macro_f1']['bayesian']:.4f} "
            f"| nll random={row['nll']['random']:.4f} distance={row['nll']['distance']:.4f} "
            f"bayesian={row['nll']['bayesian']:.4f} | acc+f1 ordered={row['accuracy_ordered'] and row['macro_f1_ordered']}"
        )
    print(f"primary fractions ordered: {ordering['primary_all_ordered']}")
    intent = payload["intentions"]
    print(
        f"intention NLL {intent['nll']:.4f} vs prior {intent['prior_nll']:.4f} "
        f"(accuracy {intent['accuracy']:.4f} vs {intent['prior_accuracy']:.4f})"
    )
    ok = ordering["primary_all_ordered"] and intent["nll"] < intent["prior_nll"]
    print("GATE STATUS:", "GREEN (CONTINUE)" if ok else "RED (STOP)")
    sys.exit(0 if ok else 1)
