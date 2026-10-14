#!/usr/bin/env python3
"""Render a demo animation from a deterministically regenerated episode.

READ-ONLY with respect to generation: it never writes to ``data/``, never edits
``configs/``, and never touches a result JSON. It rebuilds the verification
dataset in memory from the frozen master seed (the same call the gate uses, so
the config hash ``410df64cf3f7f224`` and dataset sha256 ``a452ac66fca4f3f0…`` are
untouched), picks a crossing episode, runs the committed Bayesian baseline over
it for the belief panel, and writes ``figures/demo_crossing.gif``.

    python verification/make_animation.py            # full verification dataset
    python verification/make_animation.py --fast     # tiny dataset (used by tests)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.overview_figure import (  # noqa: E402
    GOAL_COLORS,
    belief_series,
    interaction_events,
    peer_cells_at,
    select_showcase,
)
from verification.m2.dataset_m2 import build_records, verification_config  # noqa: E402

OUT = REPO_ROOT / "figures" / "demo_crossing.gif"


def choose_episode(records, sensing_radius: int, weight: float):
    """Prefer a crossing episode with the most peer-blocked focal steps."""
    crossing = [
        r
        for r in records
        if dict(r["scenario_spec"]["env_params"]).get("scenario_family") == "crossing"
    ]
    pool = crossing or records
    return select_showcase(pool, sensing_radius, weight)


def render(record, events, out: Path, fps: int = 6) -> Path:
    from src.visualization.overview_figure import focal_steps

    steps = focal_steps(record)
    T = len(steps)
    belief = belief_series(record, use_blocked=True)
    K = belief.shape[1]
    true_goal = int(record["privileged_ground_truth"]["focal_goal_id"])

    fig, (axw, axb) = plt.subplots(1, 2, figsize=(11, 5.2))
    w, h = record["grid_dimensions"]

    def base_world():
        axw.set_xlim(-0.5, w - 0.5)
        axw.set_ylim(h - 0.5, -0.5)
        axw.set_aspect("equal")
        axw.set_facecolor("#fbfbfb")
        axw.grid(True, color="#e6e6e6", linewidth=0.5)
        for ox, oy in record["static_obstacles"]:
            axw.add_patch(
                plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="#b0b0b0", edgecolor="none")
            )
        for i, t in enumerate(record["candidate_targets"]):
            x, y = t["position"]
            axw.plot(x, y, marker="*", markersize=20 if i == true_goal else 13,
                     color=GOAL_COLORS[i % len(GOAL_COLORS)], markeredgecolor="black",
                     markeredgewidth=0.7, zorder=5)
            axw.annotate(f"G{i}", (x, y), textcoords="offset points", xytext=(6, 6),
                         fontsize=9, color=GOAL_COLORS[i % len(GOAL_COLORS)], fontweight="bold")

    def frame(t: int):
        axw.clear()
        axb.clear()
        base_world()
        # peer paths up to t
        for agent_id in sorted({s["agent_id"] for s in record["steps"]}):
            if agent_id == record["focal_agent_id"]:
                continue
            path = sorted((s for s in record["steps"] if s["agent_id"] == agent_id),
                          key=lambda s: s["step"])
            upto = path[: t + 1]
            xs = [s["position_before"][0] for s in upto]
            ys = [s["position_before"][1] for s in upto]
            axw.plot(xs, ys, "--", color="#c0c0c0", linewidth=1.0, zorder=2)
            axw.plot(xs[-1], ys[-1], "o", markersize=6, markerfacecolor="white",
                     markeredgecolor="#888888", zorder=3)
        # focal path up to t
        fx = [s["position_before"][0] for s in steps[: t + 1]]
        fy = [s["position_before"][1] for s in steps[: t + 1]]
        axw.plot(fx, fy, "-", color="black", linewidth=1.8, zorder=4)
        axw.plot(fx[-1], fy[-1], "s", markersize=9, color="black", zorder=6)
        if events["blocked"][t]:
            axw.plot(fx[-1], fy[-1], "o", markersize=15, markerfacecolor="none",
                     markeredgecolor="#d62728", markeredgewidth=2.2, zorder=7)
        elif events["perturbed"][t]:
            axw.plot(fx[-1], fy[-1], "o", markersize=11, markerfacecolor="none",
                     markeredgecolor="#1f77b4", markeredgewidth=1.4, zorder=6)
        axw.set_title(f"t = {t}/{T - 1}  (red ring = peer-blocked)")
        axw.set_xticks(range(0, w, 4))
        axw.set_yticks(range(0, h, 4))

        b = belief[min(t, len(belief) - 1)]
        axb.bar(range(K), b, color=[GOAL_COLORS[i % len(GOAL_COLORS)] for i in range(K)])
        axb.set_ylim(0, 1)
        axb.set_xticks(range(K))
        axb.set_xticklabels([f"G{i}" for i in range(K)])
        axb.set_title(f"P(G | O_1:t)  true = G{true_goal}")

        return []

    anim = FuncAnimation(fig, frame, frames=T, interval=1000 / fps, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="tiny dataset for tests")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    cfg = verification_config()
    if args.fast:
        records = build_records(num_scenarios=6, episodes_per_scenario=2)
    else:
        records = build_records()
    record, events = choose_episode(records, cfg.policy_sensing_radius, cfg.peer_avoidance_weight)
    out = render(record, events, Path(args.out), fps=args.fps)
    family = dict(record["scenario_spec"]["env_params"]).get("scenario_family")
    print(
        f"wrote {out} ({out.stat().st_size} bytes) | episode {record['episode_id']} "
        f"({family}, {sum(events['blocked'])} peer-blocked, {record['termination_reason']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
