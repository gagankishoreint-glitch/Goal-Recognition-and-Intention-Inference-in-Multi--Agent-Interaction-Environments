"""Report figures, generated from the dataset and the committed result artifacts.

The overview figure is the one the report needs: environment, interaction,
inference and the RQ2 result in a single panel set.

    (a) scenario layout        what the observer is given
    (b) trajectories           where interaction actually happens
    (c) belief evolution       how the posterior over goals moves
    (d) RQ2 forest plot        what interaction information is worth

Nothing here is hardcoded. Panels (a)-(c) are drawn from generated episodes;
panel (d) reads ``experiments/multi_agent/interaction_contrast.json``. If the
numbers change, the figure changes with them.

Agents do not communicate in this environment — there is no message channel to
draw. What is visualised is blocking, perturbation and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

Position = Tuple[int, int]

DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
GOAL_COLORS = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")

CONTRAST_ORDER: Tuple[Tuple[str, str], ...] = (
    ("gap_A_minus_B", "A - B\n(naive zero-mask)"),
    ("gap_D_minus_B", "D - B\n(noise vs zeroed)"),
    ("gap_A_minus_D", "A - D\n(real peers vs noise)"),
    ("gap_E_minus_B", "E - B\n(blocked flag alone)"),
)


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def focal_steps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(
        (s for s in record["steps"] if s["agent_id"] == record["focal_agent_id"]),
        key=lambda s: s["step"],
    )


def peer_cells_at(record: Dict[str, Any], t: int) -> List[Position]:
    return sorted(
        {
            tuple(s["position_before"])
            for s in record["steps"]
            if s["step"] == t and s["agent_id"] != record["focal_agent_id"]
        }
    )


def interaction_events(
    record: Dict[str, Any], sensing_radius: int, avoidance_weight: float
) -> Dict[str, List[Any]]:
    """Per-step interaction markers for the focal agent.

    ``blocked``       the environment denied the move because a peer held/claimed it
    ``perturbed``     a nearby peer changed the action distribution at all
    ``line_held``     a peer stood on a cell that shortens the path to the true goal
    """
    from ..agents.policy import peer_influence

    obstacles = [tuple(o) for o in record["static_obstacles"]]
    bounds = tuple(record["grid_dimensions"])
    blocked: List[bool] = []
    perturbed: List[bool] = []
    line_held: List[bool] = []
    for s in focal_steps(record):
        cur = tuple(s["position_before"])
        goal = tuple(s["goal"])
        peers = peer_cells_at(record, s["step"])
        blocked.append(
            bool(s["blocked"]) and s["blocked_reason"] in ("occupied", "conflict")
        )
        perturbed.append(
            peer_influence(
                s["intention"], goal, cur, peers, sensing_radius, obstacles, bounds,
                avoidance_weight,
            )
            > 0.0
        )
        greedy = [
            (cur[0] + dx, cur[1] + dy)
            for dx, dy in DELTAS.values()
            if _manhattan((cur[0] + dx, cur[1] + dy), goal) < _manhattan(cur, goal)
        ]
        line_held.append(bool(greedy) and any(c in peers for c in greedy))
    return {"blocked": blocked, "perturbed": perturbed, "line_held": line_held}


def select_showcase(
    records: Sequence[Dict[str, Any]], sensing_radius: int, avoidance_weight: float
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """The episode with the most peer-blocked focal steps (longest wins ties)."""
    best = None
    best_events: Dict[str, List[Any]] = {}
    for record in records:
        events = interaction_events(record, sensing_radius, avoidance_weight)
        key = (sum(events["blocked"]), len(events["blocked"]))
        if best is None or key > best[0]:
            best = (key, record)
            best_events = events
    return best[1], best_events


def belief_series(record: Dict[str, Any], use_blocked: bool = True) -> np.ndarray:
    """P(G = g | O_{1:t}) after every observed transition; row 0 is the prior."""
    from ..evaluation.baselines import BayesianBeliefBaseline, belief_trajectory

    baseline = BayesianBeliefBaseline(
        [tuple(t["position"]) for t in record["candidate_targets"]],
        obstacles=[tuple(o) for o in record["static_obstacles"]],
        bounds=tuple(record["grid_dimensions"]),
    )
    traj = belief_trajectory(
        baseline, record, record["focal_agent_id"], use_blocked=use_blocked
    )
    return np.asarray(traj, dtype=float)


def _draw_grid(ax, record: Dict[str, Any]) -> None:
    w, h = record["grid_dimensions"]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # y down, so (0, 0) is the top-left cell
    ax.set_xticks(range(0, w, 2))
    ax.set_yticks(range(0, h, 2))
    ax.set_aspect("equal")
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_facecolor("#fbfbfb")
    for ox, oy in record["static_obstacles"]:
        ax.add_patch(
            plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, facecolor="#b0b0b0", edgecolor="none")
        )


def _draw_targets(ax, record: Dict[str, Any], true_goal: Optional[int] = None) -> None:
    for i, target in enumerate(record["candidate_targets"]):
        x, y = target["position"]
        color = GOAL_COLORS[i % len(GOAL_COLORS)]
        ax.plot(
            x, y, marker="*", markersize=22 if i == true_goal else 15,
            color=color, markeredgecolor="black", markeredgewidth=0.8, zorder=5,
        )
        ax.annotate(
            f"G{i}", (x, y), textcoords="offset points", xytext=(7, 7),
            fontsize=9, color=color, fontweight="bold",
        )


def panel_layout(ax, record: Dict[str, Any], family: str) -> None:
    _draw_grid(ax, record)
    _draw_targets(ax, record)
    steps = focal_steps(record)
    fx, fy = steps[0]["position_before"]
    ax.plot(fx, fy, "o", markersize=11, color="black", zorder=6)
    ax.annotate("focal", (fx, fy), textcoords="offset points", xytext=(-30, -12), fontsize=9)
    for n, cell in enumerate(peer_cells_at(record, 0), start=1):
        ax.plot(cell[0], cell[1], "s", markersize=8, markerfacecolor="white",
                markeredgecolor="#555555", zorder=6)
        ax.annotate(f"P{n}", cell, textcoords="offset points", xytext=(6, -10), fontsize=8)
    n_obs = len(record["static_obstacles"])
    n_agents = len({s["agent_id"] for s in record["steps"]})
    ax.set_title(
        f"(a) Scenario layout — {family}, N={n_agents} agents, "
        f"{n_obs} obstacles",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def panel_trajectory(
    ax, record: Dict[str, Any], events: Dict[str, List[Any]], family: str
) -> None:
    _draw_grid(ax, record)
    true_goal = int(record["privileged_ground_truth"]["focal_goal_id"])
    _draw_targets(ax, record, true_goal=true_goal)

    for agent_id in sorted({s["agent_id"] for s in record["steps"]}):
        if agent_id == record["focal_agent_id"]:
            continue
        path = sorted(
            (s for s in record["steps"] if s["agent_id"] == agent_id), key=lambda s: s["step"]
        )
        xs = [s["position_before"][0] for s in path] + [path[-1]["position"][0]]
        ys = [s["position_before"][1] for s in path] + [path[-1]["position"][1]]
        ax.plot(xs, ys, "-", color="#c8c8c8", linewidth=1.2, zorder=2)
        ax.plot(xs[0], ys[0], "s", markersize=6, markerfacecolor="white",
                markeredgecolor="#999999", zorder=3)

    steps = focal_steps(record)
    xs = [s["position_before"][0] for s in steps] + [steps[-1]["position"][0]]
    ys = [s["position_before"][1] for s in steps] + [steps[-1]["position"][1]]
    ax.plot(xs, ys, "-", color="black", linewidth=1.8, zorder=4, label="focal path")
    for i, s in enumerate(steps):
        cx, cy = s["position_before"]
        if events["blocked"][i]:
            ax.plot(cx, cy, "x", color="#d62728", markersize=11, markeredgewidth=2.4, zorder=7)
        elif events["line_held"][i]:
            ax.plot(cx, cy, "o", markerfacecolor="none", markeredgecolor="#ff7f0e",
                    markersize=10, markeredgewidth=2.0, zorder=6)
        elif events["perturbed"][i]:
            ax.plot(cx, cy, "o", markerfacecolor="none", markeredgecolor="#1f77b4",
                    markersize=6, markeredgewidth=1.2, zorder=5)
    ax.plot(xs[0], ys[0], "o", markersize=11, color="black", zorder=8)
    n = len(steps)
    ax.set_title(
        f"(b) Trajectories — {n} steps, {sum(events['blocked'])} peer-blocked, "
        f"{sum(events['perturbed'])} peer-perturbed",
        fontsize=10,
    )
    handles = [
        plt.Line2D([], [], color="black", linewidth=1.8, label="focal path"),
        plt.Line2D([], [], color="#c8c8c8", linewidth=1.2, label="peer paths"),
        plt.Line2D([], [], color="#d62728", marker="x", linestyle="", markersize=9,
                   markeredgewidth=2.2, label="peer-blocked step"),
        plt.Line2D([], [], color="#ff7f0e", marker="o", linestyle="", markersize=8,
                   markerfacecolor="none", markeredgewidth=1.8, label="peer on the direct line"),
        plt.Line2D([], [], color="#1f77b4", marker="o", linestyle="", markersize=6,
                   markerfacecolor="none", label="action distribution perturbed"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def panel_belief(ax, record: Dict[str, Any], alpha: float = 0.75) -> None:
    probs = belief_series(record, use_blocked=True)
    masked = belief_series(record, use_blocked=False)
    true_goal = int(record["privileged_ground_truth"]["focal_goal_id"])
    t = np.arange(probs.shape[0])
    k = probs.shape[1]
    for g in range(k):
        is_true = g == true_goal
        ax.plot(
            t, probs[:, g],
            color=GOAL_COLORS[g % len(GOAL_COLORS)],
            linewidth=2.4 if is_true else 1.1,
            linestyle="-" if is_true else "--",
            alpha=1.0 if is_true else 0.75,
            label=f"G{g}" + (" (true)" if is_true else ""),
        )
    ax.plot(t, masked[:, true_goal], ":", color="black", linewidth=1.4,
            label="true goal, interaction masked")
    ax.axhline(1.0 / k, color="#888888", linewidth=0.9, linestyle="-.")
    ax.annotate(
        f"prior 1/K = {1.0 / k:.2f}", (0.02, 1.0 / k + 0.02), fontsize=7, color="#666666"
    )
    cut = int(min(probs.shape[0] - 1, round(alpha * (probs.shape[0] - 1))))
    ax.axvline(cut, color="#444444", linewidth=1.0, linestyle=":")
    ax.annotate(
        f"alpha={alpha:.2f} (t={cut})\nargmax G{int(probs[cut].argmax())}",
        (cut + 0.4, 0.05), fontsize=7, color="#444444",
    )
    ax.set_title(
        f"(c) Posterior over goals, P(G | O_1:t) — episode ends "
        f"{record['termination_reason'].replace('_', ' ').title()}",
        fontsize=10,
    )
    ax.set_xlabel("transitions observed  t")
    ax.set_ylabel("P(G = g)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)


def panel_forest(ax, contrast: Dict[str, Any], task: str = "intention") -> None:
    block = contrast["probe"][task]
    ypos: List[float] = []
    for i, (key, label) in enumerate(CONTRAST_ORDER):
        gap = block[key]["accuracy"]
        y = len(CONTRAST_ORDER) - 1 - i
        ypos.append(y)
        lo, hi = gap["ci_lower"], gap["ci_upper"]
        point = gap["point_estimate"]
        significant = lo > 0 or hi < 0
        color = "#1a7f37" if significant else "#999999"
        ax.plot([lo, hi], [y, y], "-", color=color, linewidth=2.0)
        ax.plot([point], [y], "o", color=color, markersize=8 if significant else 6)
        ax.annotate(
            f"{point:+.4f}  [{lo:+.4f}, {hi:+.4f}]",
            (hi, y), textcoords="offset points", xytext=(6, -3), fontsize=7.5, color=color,
        )
        ax.annotate(
            "CI excludes 0" if significant else "not supported",
            (lo, y), textcoords="offset points", xytext=(-6, 8), fontsize=6.5,
            color=color, ha="right",
        )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(ypos)
    # ticks were added top-to-bottom in CONTRAST_ORDER, so labels follow the same
    # order (an earlier reversal put every value next to the wrong contrast)
    ax.set_yticklabels([label for _, label in CONTRAST_ORDER], fontsize=8)
    span = max(
        abs(block[k]["accuracy"][b])
        for k, _ in CONTRAST_ORDER
        for b in ("ci_lower", "ci_upper", "point_estimate")
    )
    ax.set_xlim(-span - 0.06, span + 0.115)
    ax.set_title(
        f"(d) RQ2 contrast, {task} accuracy — paired scenario-cluster bootstrap",
        fontsize=10,
    )
    ax.set_xlabel("accuracy difference (Condition A - comparator)")
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.6)


def build_overview_figure(
    records: Sequence[Dict[str, Any]],
    contrast: Dict[str, Any],
    out_path: str | Path,
    sensing_radius: int = 3,
    avoidance_weight: float = 0.05,
    alpha: float = 0.75,
    showcase: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render the four-panel overview figure and write it to ``out_path``."""
    import json

    if isinstance(contrast, (str, Path)):
        contrast = json.loads(Path(contrast).read_text(encoding="utf-8"))
    record = showcase or select_showcase(records, sensing_radius, avoidance_weight)[0]
    events = interaction_events(record, sensing_radius, avoidance_weight)
    family = dict(record["scenario_spec"]["env_params"]).get("scenario_family", "open")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    fig.patch.set_facecolor("white")
    panel_layout(axes[0][0], record, family)
    panel_trajectory(axes[0][1], record, events, family)
    panel_belief(axes[1][0], record, alpha=alpha)
    panel_forest(axes[1][1], contrast)
    fig.suptitle(
        "Goal recognition from multi-agent interaction: environment, interaction, "
        "inference and the RQ2 result",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out
