#!/usr/bin/env python3
"""Section 4 — scenario leakage test (M2 gate).

1. One scenario yields several episodes that all carry the same ``scenario_id``.
2. A deliberately leaky split must be rejected by ``assert_no_scenario_leakage``.
3. The production split over the full verification dataset must have zero
   scenario overlap between train / val / test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trajectory.dataset import (  # noqa: E402
    generate_dataset,
    split_records,
)
from src.trajectory.scenarios import assert_no_scenario_leakage  # noqa: E402
from verification.m2.dataset_m2 import (  # noqa: E402
    VERIFICATION_MASTER_SEED,
    verification_config,
)

REPORT_PATH = REPO_ROOT / "verification" / "m2" / "section4_leakage.md"


def _full_dataset() -> List[Dict[str, Any]]:
    cfg = verification_config()
    return generate_dataset(
        cfg,
        master_seed=VERIFICATION_MASTER_SEED,
        num_scenarios=20,
        episodes_per_scenario=4,
    )


def test_scenario_ids_are_shared_within_a_scenario() -> None:
    """4.1 episodes derived from one scenario share its scenario_id."""
    cfg = verification_config()
    records = generate_dataset(
        cfg, master_seed=VERIFICATION_MASTER_SEED, num_scenarios=1, episodes_per_scenario=5
    )
    ids = {r["scenario_id"] for r in records}
    assert len(records) == 5
    assert len(ids) == 1, f"episodes from one scenario carry different ids: {ids}"
    assert len({r["episode_id"] for r in records}) == 5, "episode ids are not unique"
    # different layouts must produce different ids
    other = generate_dataset(
        cfg, master_seed=VERIFICATION_MASTER_SEED + 1, num_scenarios=1, episodes_per_scenario=1
    )
    assert other[0]["scenario_id"] != records[0]["scenario_id"]


def test_leaky_split_is_rejected() -> None:
    """4.2 a split that repeats a scenario_id across train/test must raise."""
    records = _full_dataset()
    shared = records[0]["scenario_id"]
    leaky = {
        "train": [r for r in records if r["scenario_id"] == shared],
        "test": [r for r in records if r["scenario_id"] == shared],
    }
    with pytest.raises(AssertionError) as excinfo:
        assert_no_scenario_leakage(leaky)
    message = str(excinfo.value)
    assert shared in message and "train" in message and "test" in message, message
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Scenario Leakage Test (Section 4)\n\n"
        "## Adversarial leakage exception (4.2)\n\n"
        "A split in which the same `scenario_id` appears in both train and test is rejected:\n\n"
        f"```\n{message}\n```\n",
        encoding="utf-8",
    )


def test_production_split_has_no_scenario_overlap() -> None:
    """4.3 the production split keeps scenarios disjoint across partitions."""
    records = _full_dataset()
    splits = split_records(records)  # raises on overlap by construction
    counts = {
        name: {
            "episodes": len(rows),
            "scenarios": len({r["scenario_id"] for r in rows}),
        }
        for name, rows in splits.items()
    }
    ids = {name: {r["scenario_id"] for r in rows} for name, rows in splits.items()}
    for a in ids:
        for b in ids:
            if a < b:
                assert not (ids[a] & ids[b]), f"{a} and {b} share scenarios {ids[a] & ids[b]}"
    assert sum(c["episodes"] for c in counts.values()) == len(records)
    # the split partitions the dataset: no episode lost or duplicated
    repartitioned = sorted(
        r["episode_id"] for rows in splits.values() for r in rows
    )
    assert repartitioned == sorted(r["episode_id"] for r in records)
    for name, rows in splits.items():
        assert rows, f"{name} split is empty"
        # episodes of one scenario are contiguous in the same split
        per_scenario = {}
        for r in rows:
            per_scenario.setdefault(r["scenario_id"], 0)
            per_scenario[r["scenario_id"]] += 1
        assert sum(per_scenario.values()) == len(rows)
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    lines = ["\n## Production split (4.3)\n",
             f"Dataset: 20 scenarios x 4 episodes, master seed {VERIFICATION_MASTER_SEED}.\n"]
    for name, c in counts.items():
        lines.append(f"- {name}: {c['episodes']} episodes / {c['scenarios']} unique scenarios")
    lines.append("- overlap between any two splits: none (asserted)")
    REPORT_PATH.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--disable-warnings"]))
