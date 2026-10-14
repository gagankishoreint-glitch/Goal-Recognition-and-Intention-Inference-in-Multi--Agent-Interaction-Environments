"""Shared verification dataset for the M2 gate (Sections 4, 6, 7).

Every M2 section must evaluate the *same* underlying episodes — that is what
makes the Condition A / Condition B comparison (RQ2) and the triviality /
baseline gates comparable. This module is the single source of truth: the
dataset is rebuilt deterministically from ``VERIFICATION_MASTER_SEED`` and the
checked-in ``configs/env_default.yaml`` instead of being cached to disk, so
rebuilding it is itself a reproducibility check.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trajectory.dataset import (  # noqa: E402
    DatasetConfig,
    dataset_hash,
    generate_dataset,
    observable_record,
    split_records,
)

VERIFICATION_MASTER_SEED = 20261008
VERIFICATION_SCENARIOS = 60
VERIFICATION_EPISODES_PER_SCENARIO = 8
VERIFICATION_SPLIT_SEED = 42

CONFIG_PATH = REPO_ROOT / "configs" / "env_default.yaml"


def verification_config() -> DatasetConfig:
    return DatasetConfig.from_yaml(str(CONFIG_PATH))


def build_records(
    num_scenarios: int = VERIFICATION_SCENARIOS,
    episodes_per_scenario: int = VERIFICATION_EPISODES_PER_SCENARIO,
    master_seed: int = VERIFICATION_MASTER_SEED,
) -> List[Dict[str, Any]]:
    return generate_dataset(
        verification_config(),
        master_seed=master_seed,
        num_scenarios=num_scenarios,
        episodes_per_scenario=episodes_per_scenario,
    )


def build_splits(
    num_scenarios: int = VERIFICATION_SCENARIOS,
    episodes_per_scenario: int = VERIFICATION_EPISODES_PER_SCENARIO,
    master_seed: int = VERIFICATION_MASTER_SEED,
    split_seed: int = VERIFICATION_SPLIT_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    records = build_records(num_scenarios, episodes_per_scenario, master_seed)
    return split_records(records, seed=split_seed)


def splits_summary(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, rows in splits.items():
        goals: Dict[int, int] = {}
        intentions: Dict[str, int] = {}
        for r in rows:
            gid = r["privileged_ground_truth"]["focal_goal_id"]
            goals[gid] = goals.get(gid, 0) + 1
            for step in r["privileged_ground_truth"]["step_intentions"]:
                intentions[step] = intentions.get(step, 0) + 1
        out[name] = {
            "episodes": len(rows),
            "scenarios": len({r["scenario_id"] for r in rows}),
            "goal_counts": {str(k): goals[k] for k in sorted(goals)},
            "intention_counts": {k: intentions[k] for k in sorted(intentions)},
        }
    return out


def observable_seed(record: Dict[str, Any]) -> int:
    """Deterministic per-episode seed derived from *observable* identity only."""
    digest = hashlib.sha256(record["episode_id"].encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


__all__ = [
    "VERIFICATION_MASTER_SEED",
    "VERIFICATION_SCENARIOS",
    "VERIFICATION_EPISODES_PER_SCENARIO",
    "verification_config",
    "build_records",
    "build_splits",
    "splits_summary",
    "observable_seed",
    "dataset_hash",
    "observable_record",
]
