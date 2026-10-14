"""Scenario configuration, deterministic scenario_id, and level splitting."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class ScenarioConfig:
    width: int
    height: int
    obstacles: tuple
    targets: tuple
    agent_starts: tuple
    num_agents: int
    agent_configs: tuple
    env_params: tuple = ()

    def canonical(self) -> str:
        payload = {
            "width": self.width,
            "height": self.height,
            "obstacles": sorted([list(o) for o in self.obstacles]),
            "targets": [list(t) for t in self.targets],
            "agent_starts": [list(a) for a in self.agent_starts],
            "num_agents": self.num_agents,
            "agent_configs": list(self.agent_configs),
            "env_params": dict(self.env_params),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def scenario_id(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]


def compute_scenario_id(config: ScenarioConfig) -> str:
    return config.scenario_id()


def split_dataset(
    records: List[Dict[str, Any]],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split at the scenario level, before episode sampling."""
    assert abs(train + val + test - 1.0) < 1e-6
    scenario_ids = sorted({r["scenario_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(scenario_ids)
    n = len(scenario_ids)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    train_ids = set(scenario_ids[:n_train])
    val_ids = set(scenario_ids[n_train : n_train + n_val])
    test_ids = set(scenario_ids[n_train + n_val :])
    out = {"train": [], "val": [], "test": []}
    for r in records:
        sid = r["scenario_id"]
        if sid in train_ids:
            out["train"].append(r)
        elif sid in val_ids:
            out["val"].append(r)
        else:
            out["test"].append(r)
    return out


def assert_no_scenario_leakage(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    seen = {}
    for name, rows in splits.items():
        for r in rows:
            sid = r["scenario_id"]
            if sid in seen and seen[sid] != name:
                raise AssertionError(
                    f"scenario {sid} appears in both {seen[sid]} and {name}"
                )
            seen[sid] = name
