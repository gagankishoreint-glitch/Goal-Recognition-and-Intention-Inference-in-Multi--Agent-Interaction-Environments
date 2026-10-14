#!/usr/bin/env python3
"""Render the report figures from the verification dataset and committed results.

    python verification/make_figures.py

Writes ``figures/fig1_overview.png``. Deterministic: same dataset hash, same
episode selection, same figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.overview_figure import (  # noqa: E402
    build_overview_figure,
    interaction_events,
    select_showcase,
)
from verification.m2.dataset_m2 import build_splits, verification_config  # noqa: E402

CONTRAST_PATH = REPO_ROOT / "experiments" / "multi_agent" / "interaction_contrast.json"
OUT_PATH = REPO_ROOT / "figures" / "fig1_overview.png"


def main() -> int:
    cfg = verification_config()
    splits = build_splits()
    test = splits["test"]
    record, events = select_showcase(
        test, cfg.policy_sensing_radius, cfg.peer_avoidance_weight
    )
    contrast = json.loads(CONTRAST_PATH.read_text(encoding="utf-8"))
    out = build_overview_figure(
        test,
        contrast,
        OUT_PATH,
        sensing_radius=cfg.policy_sensing_radius,
        avoidance_weight=cfg.peer_avoidance_weight,
        showcase=record,
    )
    family = dict(record["scenario_spec"]["env_params"])["scenario_family"]
    agents = len({s["agent_id"] for s in record["steps"]})
    # the fingerprint of the full verification dataset, as committed
    fingerprint = [
        line
        for line in (REPO_ROOT / "verification" / "m2" / "dataset_hash.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("- dataset sha256:")
    ]
    print(f"{fingerprint[0].lstrip('- ') if fingerprint else 'dataset hash unavailable'}")
    print(f"showcase drawn from the test split: {len(test)} episodes / "
          f"{len({r['scenario_id'] for r in test})} scenarios")
    print(
        f"showcase: {record['episode_id']} ({family}, {agents} agents, "
        f"{record['termination_reason']}, {len([s for s in record['steps'] if s['agent_id'] == record['focal_agent_id']])} steps)"
    )
    print(
        f"interaction markers: {sum(events['blocked'])} peer-blocked, "
        f"{sum(events['line_held'])} peer-on-direct-line, "
        f"{sum(events['perturbed'])} perturbed"
    )
    print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
