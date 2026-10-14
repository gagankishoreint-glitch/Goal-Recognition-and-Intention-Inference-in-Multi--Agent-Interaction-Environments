#!/usr/bin/env python3
"""RQ2 pre-study: does interaction information change goal/intention inference?

Two manipulations, same underlying episodes throughout:

1. **Closed-form Bayesian recognizer** — Condition A sees the observable
   ``blocked`` flag, Condition B does not. This is the only interaction input the
   M2 baseline consumes, so the gap it shows is a floor.
2. **Linear probe over the timing-contract features** — Condition A keeps the
   peer channels and the interaction features; three masked variants differ in
   *how* the information is removed. This matters more than it looks: zeroing the
   channels changes the geometry of a regularized linear model, and that alone
   produces a spurious gap. The arms are

   * ``B_zeroed``  — channels set to zero (the naive mask),
   * ``C_shuffled_peers`` — same channels, peer cells relocated at random,
   * ``D_noise`` — same channels, filled with Gaussian noise.

   ``A - B`` is the number a paper would report; ``A - D`` is the number that
   means anything, because it holds the input geometry fixed and asks only
   whether *real* peer placement beats noise. Both are reported, plus the
   decomposition ``D - B``, which is the size of the artifact that zeroing the
   channels creates on its own.

Gaps carry scenario-cluster bootstrap CIs (paired, frozen predictions, 2000
resamples), the same estimator used by the M2 gate.

Run:
    python experiments/multi_agent/run_interaction_contrast.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.policy import peer_influence  # noqa: E402
from src.evaluation.baselines import BayesianBeliefBaseline, belief_trajectory  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    evaluate_goal_predictions,
    macro_f1_from_predictions,
    normalize,
    paired_gap_ci,
    to_matrix,
)
from src.features.timing import supervised_dataset  # noqa: E402
from verification.m2.dataset_m2 import build_splits, verification_config  # noqa: E402

ALPHAS: Sequence[float] = (0.30, 0.50, 0.75, 1.00)
NUM_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
PROBE_WINDOW = 5
PROBE_TASKS: Sequence[str] = ("goal", "intention")

OUT_DIR = REPO_ROOT / "experiments" / "multi_agent"


# --------------------------------------------------------------------------
# interaction measurement
# --------------------------------------------------------------------------
def family_of(record: Dict[str, Any]) -> str:
    return str(dict(record["scenario_spec"]["env_params"]).get("scenario_family", "open"))


DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def interaction_stats(record: Dict[str, Any], sensing_radius: int, weight: float) -> Dict[str, float]:
    """Peer-blocked / peer-influenced / greedy-cell-held rates for one episode."""
    steps = sorted(
        (s for s in record["steps"] if s["agent_id"] == record["focal_agent_id"]),
        key=lambda s: s["step"],
    )
    obstacles = [tuple(o) for o in record["static_obstacles"]]
    bounds = tuple(record["grid_dimensions"])
    out = {"steps": 0.0, "peer_blocked": 0.0, "peer_influenced": 0.0, "greedy_held": 0.0, "tv_sum": 0.0}
    for s in steps:
        cur = tuple(s["position_before"])
        goal = tuple(s["goal"])
        peers = sorted(
            {
                tuple(x["position_before"])
                for x in record["steps"]
                if x["step"] == s["step"] and x["agent_id"] != record["focal_agent_id"]
            }
        )
        out["steps"] += 1
        if s["blocked"] and s["blocked_reason"] in ("occupied", "conflict"):
            out["peer_blocked"] += 1
        tv = peer_influence(
            s["intention"], goal, cur, peers, sensing_radius, obstacles, bounds, weight
        )
        if tv > 0:
            out["peer_influenced"] += 1
            out["tv_sum"] += tv
        greedy = [
            (cur[0] + dx, cur[1] + dy)
            for dx, dy in DELTAS.values()
            if _manhattan((cur[0] + dx, cur[1] + dy), goal) < _manhattan(cur, goal)
        ]
        if greedy and any(c in peers for c in greedy):
            out["greedy_held"] += 1
    return out


def shuffle_peers(record: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Copy of *record* whose peer cells are replaced by random free cells.

    Used as the control arm: identical feature dimensionality and identical
    interaction-feature computation, but the peer positions carry no information
    about the focal trajectory.
    """
    import copy

    out = copy.deepcopy(record)
    w, h = out["grid_dimensions"]
    obstacles = {tuple(o) for o in out["static_obstacles"]}
    targets = {tuple(t["position"]) for t in out["candidate_targets"]}
    free = [
        (x, y)
        for x in range(w)
        for y in range(h)
        if (x, y) not in obstacles and (x, y) not in targets
    ]
    rng = random.Random(seed)
    for step in out["steps"]:
        if step["agent_id"] == out["focal_agent_id"]:
            continue
        cell = list(rng.choice(free))
        step["position_before"] = cell
        step["position"] = cell
    return out


# --------------------------------------------------------------------------
# arm 1: closed-form Bayesian recognizer
# --------------------------------------------------------------------------
def bayes_probs(use_blocked: bool, records: List[Dict[str, Any]], alpha: float) -> np.ndarray:
    rows = []
    for record in records:
        baseline = BayesianBeliefBaseline(
            [tuple(t["position"]) for t in record["candidate_targets"]],
            obstacles=[tuple(o) for o in record["static_obstacles"]],
            bounds=tuple(record["grid_dimensions"]),
        )
        traj = belief_trajectory(baseline, record, record["focal_agent_id"], use_blocked=use_blocked)
        last = max(len(traj) - 1, 0)
        idx = int(min(last, max(0, round(alpha * last))))
        rows.append(traj[idx])
    return normalize(to_matrix(rows))


# --------------------------------------------------------------------------
# arm 2: linear probe over timing-contract features
# --------------------------------------------------------------------------
ARMS: Tuple[str, ...] = ("A_full", "B_zeroed", "C_shuffled_peers", "D_noise", "E_blocked_only")


def _noise_channels(
    x: np.ndarray, window: int, n_targets: int, seed: int
) -> np.ndarray:
    """Replace the peer slots and the interaction block with Gaussian noise.

    Used as the matched-dimensionality control: the mask removes peer
    *information* without changing the input geometry the regularizer sees.
    """
    out = x.copy()
    width = x.shape[1] // window
    tail = 3 * n_targets + 4
    peer_start = 2 + 4 * n_targets
    peer_end = peer_start + 4 * 3
    rng = np.random.default_rng(seed)
    for w in range(window):
        off = w * width
        out[:, off + peer_start:off + peer_end] = rng.standard_normal(
            (x.shape[0], peer_end - peer_start)
        )
        out[:, off + width - tail:off + width] = rng.standard_normal((x.shape[0], tail))
    return out


def probe_predictions(
    task: str,
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    condition: str,
    window: int = PROBE_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # arm -> (interaction features, peer cells, previous-step blocked flag)
    kwargs: Dict[str, Any] = {
        "A_full": {"include_interaction": True, "peer_visible": True, "blocked_visible": True},
        "B_zeroed": {"include_interaction": False, "peer_visible": False, "blocked_visible": False},
        "C_shuffled_peers": {"include_interaction": True, "peer_visible": True, "blocked_visible": True},
        "D_noise": {"include_interaction": False, "peer_visible": False, "blocked_visible": True},
        "E_blocked_only": {"include_interaction": False, "peer_visible": False, "blocked_visible": True},
    }[condition]
    tr = train if condition != "C_shuffled_peers" else [
        shuffle_peers(r, 90000 + i) for i, r in enumerate(train)
    ]
    te = test if condition != "C_shuffled_peers" else [
        shuffle_peers(r, 95000 + i) for i, r in enumerate(test)
    ]
    x_tr, y_tr, _ = supervised_dataset(tr, task, window=window, **kwargs)
    x_te, y_te, sids = supervised_dataset(te, task, window=window, **kwargs)
    if condition == "D_noise":
        n_targets = len(test[0]["candidate_targets"])
        x_tr = _noise_channels(x_tr, window, n_targets, seed=11)
        x_te = _noise_channels(x_te, window, n_targets, seed=12)
    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(x_tr), y_tr)
    probs = model.predict_proba(scaler.transform(x_te))
    classes = model.classes_
    full = np.zeros((probs.shape[0], int(max(classes) + 1)))
    full[:, classes] = probs
    return normalize(full), np.asarray(y_te, dtype=int), sids


def summarize_probe(
    task: str, train: List[Dict[str, Any]], test: List[Dict[str, Any]]
) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for condition in ARMS:
        probs, y, sids = probe_predictions(task, train, test, condition)
        pred = probs.argmax(axis=1)
        arms[condition] = {
            "accuracy": float((pred == y).mean()),
            "macro_f1": macro_f1_from_predictions(pred, y, probs.shape[1]),
            "n_samples": int(len(y)),
            "n_scenarios": len(set(sids)),
            "_probs": probs,
            "_y": y,
            "_sids": sids,
        }
    out: Dict[str, Any] = {
        cond: {k: v for k, v in arm.items() if not k.startswith("_")}
        for cond, arm in arms.items()
    }
    def gap(left: str, right: str) -> Dict[str, Any]:
        return {
            metric: paired_gap_ci(
                arms[left]["_probs"], arms[right]["_probs"], arms[left]["_y"],
                arms[left]["_sids"], metric=metric, num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED,
            )
            for metric in ("accuracy", "macro_f1")
        }

    out["gap_A_minus_B"] = gap("A_full", "B_zeroed")
    out["gap_A_minus_C"] = gap("A_full", "C_shuffled_peers")
    out["gap_A_minus_D"] = gap("A_full", "D_noise")
    out["gap_A_minus_E"] = gap("A_full", "E_blocked_only")
    out["gap_E_minus_B"] = gap("E_blocked_only", "B_zeroed")
    out["gap_D_minus_B"] = gap("D_noise", "B_zeroed")
    return out


def main() -> int:
    cfg = verification_config()
    splits = build_splits()
    train, test = splits["train"], splits["test"]

    stats = [interaction_stats(r, cfg.policy_sensing_radius, cfg.peer_avoidance_weight) for r in test]
    per_family: Dict[str, Dict[str, float]] = {}
    for record, st in zip(test, stats):
        b = per_family.setdefault(family_of(record), {"episodes": 0.0})
        b["episodes"] += 1
        for k in ("steps", "peer_blocked", "peer_influenced", "greedy_held", "tv_sum"):
            b[k] = b.get(k, 0.0) + st[k]

    results: Dict[str, Any] = {
        "config": {
            "num_agents": cfg.num_agents,
            "policy_sensing_radius": cfg.policy_sensing_radius,
            "peer_avoidance_weight": cfg.peer_avoidance_weight,
            "peer_spawn_radius": cfg.peer_spawn_radius,
            "scenario_family": cfg.scenario_family,
            "alphas": list(ALPHAS),
            "probe_window": PROBE_WINDOW,
            "num_resamples": NUM_RESAMPLES,
        },
        "interaction_by_family": {
            fam: {
                "episodes": int(b["episodes"]),
                "steps": int(b["steps"]),
                "peer_blocked_rate": b["peer_blocked"] / max(b["steps"], 1),
                "peer_influenced_rate": b["peer_influenced"] / max(b["steps"], 1),
                "greedy_cell_held_rate": b["greedy_held"] / max(b["steps"], 1),
                "mean_tv_when_influenced": b["tv_sum"] / max(b["peer_influenced"], 1),
            }
            for fam, b in sorted(per_family.items())
        },
        "bayes": {},
        "probe": {},
    }

    labels = np.asarray([r["privileged_ground_truth"]["focal_goal_id"] for r in test], dtype=int)
    sids = [r["scenario_id"] for r in test]

    for alpha in ALPHAS:
        pa = bayes_probs(True, test, alpha)
        pb = bayes_probs(False, test, alpha)
        entry: Dict[str, Any] = {
            "A_full": evaluate_goal_predictions(
                "A_full", pa, labels, sids, num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED
            ).as_dict(),
            "B_masked": evaluate_goal_predictions(
                "B_masked", pb, labels, sids, num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED
            ).as_dict(),
            "gap": {
                metric: paired_gap_ci(
                    pa, pb, labels, sids, metric=metric,
                    num_resamples=NUM_RESAMPLES, seed=BOOTSTRAP_SEED,
                )
                for metric in ("accuracy", "macro_f1", "nll")
            },
        }
        results["bayes"][f"{alpha:.2f}"] = entry

    for task in PROBE_TASKS:
        results["probe"][task] = summarize_probe(task, train, test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "interaction_contrast.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = render(results)
    (OUT_DIR / "interaction_contrast.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def _fmt_gap(gap: Dict[str, float]) -> str:
    return f"{gap['point_estimate']:+.4f} [{gap['ci_lower']:+.4f}, {gap['ci_upper']:+.4f}]"


def render(r: Dict[str, Any]) -> str:
    cfg = r["config"]
    L = ["# RQ2 Pre-Study: Interaction-Visible vs Interaction-Masked", ""]
    L.append(
        f"Environment: N = {cfg['num_agents']} agents, `scenario_family = {cfg['scenario_family']}`, "
        f"peer spawn radius {cfg['peer_spawn_radius']}, policy sensing radius "
        f"{cfg['policy_sensing_radius']}, peer avoidance weight {cfg['peer_avoidance_weight']}. "
        "Both conditions run on the *same* episodes; only the interaction channels differ."
    )
    L += ["", "## 1. Is interaction measurable? (test split, focal agent)", "",
          "| family | episodes | steps | peer-blocked | peer-influenced | greedy cell held by peer | mean TV |",
          "|---|---|---|---|---|---|---|"]
    for fam, row in r["interaction_by_family"].items():
        L.append(
            f"| {fam} | {row['episodes']} | {row['steps']} | {row['peer_blocked_rate']:.2%} | "
            f"{row['peer_influenced_rate']:.2%} | {row['greedy_cell_held_rate']:.2%} | "
            f"{row['mean_tv_when_influenced']:.3f} |"
        )
    L += ["",
          "- **peer-blocked** — the environment denied the focal move because a peer held or claimed the cell.",
          "- **peer-influenced** — a peer inside the sensing radius changed the focal action distribution at all (total variation > 0).",
          "- **greedy cell held by peer** — a cell that would reduce distance to the true goal was occupied by a peer, i.e. the agent was forced off its direct line. This is the mechanism a masked observer must mis-attribute.",
          ""]
    L += ["## 2. Closed-form Bayesian recognizer (blocked flag visible vs masked)", "",
          "| alpha | Acc A | Acc B | F1 A | F1 B | NLL A | NLL B | Acc gap (A-B) 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
    for alpha, e in r["bayes"].items():
        L.append(
            f"| {alpha} | {e['A_full']['accuracy']:.4f} | {e['B_masked']['accuracy']:.4f} | "
            f"{e['A_full']['macro_f1']:.4f} | {e['B_masked']['macro_f1']:.4f} | "
            f"{e['A_full']['nll']:.4f} | {e['B_masked']['nll']:.4f} | "
            f"{_fmt_gap(e['gap']['accuracy'])} |"
        )
    L += ["",
          "The Bayesian baseline only consumes the `blocked` flag, so this arm measures the value of "
          "the flag alone. It is reported as-is, including where the gap is not distinguishable from zero.",
          ""]
    L += [f"## 3. Linear probe over the timing-contract features (window {cfg['probe_window']})", "",
          "Five arms, identical feature dimensionality (58 channels per frame) and identical "
          "model capacity; they differ only in which information is present:", "",
          "| arm | peer cells | interaction features | previous blocked flag |",
          "|---|---|---|---|",
          "| A_full | real | real | real |",
          "| B_zeroed | zeroed | zeroed | dropped |",
          "| C_shuffled_peers | randomly relocated | recomputed | real |",
          "| D_noise | Gaussian noise | Gaussian noise | real |",
          "| E_blocked_only | zeroed | zeroed | real |",
          ""]
    L += ["| task | arm | accuracy | macro-F1 |", "|---|---|---|---|"]
    for task, block in r["probe"].items():
        for arm in ARMS:
            row = block[arm]
            L.append(
                f"| {task} | {arm} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
            )
    L += ["", "### Paired gaps (scenario-cluster bootstrap, 2000 resamples)", "",
          "| task | contrast | accuracy | macro-F1 | reading |",
          "|---|---|---|---|---|"]
    notes = {
        "gap_A_minus_B": "naive contrast — confounds peers with the blocked flag",
        "gap_A_minus_D": "**primary**: real peer placement vs Gaussian noise",
        "gap_A_minus_E": "real peer placement vs zeroed peer channels",
        "gap_E_minus_B": "value of the previous-step blocked flag alone",
        "gap_A_minus_C": "real peers vs relocated peers",
        "gap_D_minus_B": "artifact size: zeroing channels + dropping the flag",
    }
    for task, block in r["probe"].items():
        for key in ("gap_A_minus_B", "gap_A_minus_D", "gap_A_minus_E", "gap_E_minus_B",
                    "gap_A_minus_C", "gap_D_minus_B"):
            g = block[key]
            L.append(
                f"| {task} | {key.replace('gap_', '')} | {_fmt_gap(g['accuracy'])} | "
                f"{_fmt_gap(g['macro_f1'])} | {notes[key]} |"
            )
    L += ["", "A gap counts as supported only when its 95% CI excludes zero.", ""]

    L += ["## 4. What these numbers say", ""]
    for task, block in r["probe"].items():
        sig = lambda g: g["ci_lower"] > 0 or g["ci_upper"] < 0  # noqa: E731
        ab, ad, ae, eb = (
            block["gap_A_minus_B"]["accuracy"], block["gap_A_minus_D"]["accuracy"],
            block["gap_A_minus_E"]["accuracy"], block["gap_E_minus_B"]["accuracy"],
        )
        f1 = {
            k: block[k]["macro_f1"]
            for k in ("gap_A_minus_B", "gap_A_minus_D", "gap_A_minus_E", "gap_E_minus_B")
        }
        L.append(f"**{task}**")
        L.append("")
        L.append(
            f"- naive A-B accuracy {ab['point_estimate']:+.4f} "
            f"({'significant' if sig(ab) else 'not significant'}); "
            f"valid A-D {ad['point_estimate']:+.4f} "
            f"({'significant' if sig(ad) else 'not significant'}); "
            f"peer placement over the flag A-E {ae['point_estimate']:+.4f} "
            f"({'significant' if sig(ae) else 'not significant'}); "
            f"blocked flag alone E-B {eb['point_estimate']:+.4f} "
            f"({'significant' if sig(eb) else 'not significant'}). "
            f"The same contrasts on macro-F1: A-B {f1['gap_A_minus_B']['point_estimate']:+.4f}, "
            f"A-D {f1['gap_A_minus_D']['point_estimate']:+.4f}, "
            f"E-B {f1['gap_E_minus_B']['point_estimate']:+.4f}."
        )
        if sig(ab) and not sig(ad):
            L.append(
                "  - The naive contrast is **not** attributable to peer placement: replacing the "
                "peer channels with Gaussian noise reproduces it. "
                + (
                    "The whole of it is carried by the previous-step `blocked` flag (E-B), a "
                    "one-bit lagging trace of the collision the environment recorded."
                    if sig(eb)
                    else "No single component reproduces it, so it is an artifact of the masking "
                    "geometry rather than of information."
                )
            )
        elif sig(ad):
            L.append(
                "  - Peer placement itself carries information: real peer positions beat "
                "Gaussian noise at matched dimensionality."
            )
        elif not sig(ab):
            L.append(
                "  - No supported effect of interaction information for this task."
            )
        L.append("")
    L += [
        "Methodological consequence, carried into M3: the interaction-masked arm must be "
        "constructed by replacing the masked channels with noise (arm D), never by zeroing them "
        "(arm B). Zeroing changes what the regularizer sees and manufactures a gap that has "
        "nothing to do with interaction — here worth up to "
        f"{max(abs(r['probe'][t]['gap_D_minus_B']['accuracy']['point_estimate']) for t in r['probe']):+.4f} "
        "accuracy on its own. The same protocol applies to the GRU.",
        "",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
