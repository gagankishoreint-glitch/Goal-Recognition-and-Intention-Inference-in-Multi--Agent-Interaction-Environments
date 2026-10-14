#!/usr/bin/env python3
"""Section 5 — end-to-end information isolation test (M2 gate).

Pipeline exercised (production code, not a stand-in):

    scenario sampling -> policy reset -> trajectory generation
      -> observable / privileged separation
      -> Condition A construction -> Condition B projection
      -> JSON serialization -> reload -> batch validation -> model input tensor

Assertions:
* no privileged field in a Condition A batch;
* no privileged field and no peer-derived field in a Condition B batch;
* Condition B schema is a subset of the Condition A schema;
* injecting a privileged field makes the loader fail loudly;
* no serialized observable file contains privileged key names (grep);
* observations up to t do not depend on the suffix after t;
* a model input tensor cannot carry privileged columns.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.observations import (  # noqa: E402
    CONDITION_B_ALLOWED_FIELDS,
    InformationIsolationError,
    OBSERVABLE_FIELDS,
    assert_condition_b_projection,
    assert_prefix_invariance,
    build_condition_a_observation,
    build_condition_b_observation,
    validate_batch,
)
from src.features.tensorize import episode_tensor, feature_names  # noqa: E402
from src.trajectory.dataset import (  # noqa: E402
    dataset_hash,
    generate_dataset,
    observable_record,
    strip_privileged,
)
from verification.m2.dataset_m2 import (  # noqa: E402
    VERIFICATION_MASTER_SEED,
    verification_config,
)

PRIVILEGED_KEY_PATTERNS = (
    "intention",
    "goal",
    "intended_action",
    "planner_internal_state",
    "seed_state",
    "blocked_reason",
    "privileged",
    "scenario_spec",
    "config_hash",
)

REPORT_PATH = REPO_ROOT / "verification" / "m2" / "section5_isolation.md"


@pytest.fixture(scope="module")
def records() -> List[Dict[str, Any]]:
    return generate_dataset(
        verification_config(),
        master_seed=VERIFICATION_MASTER_SEED,
        num_scenarios=4,
        episodes_per_scenario=3,
    )


def _privileged_keys(obj: Any, path: str = "") -> List[str]:
    """Recursively collect keys matching a privileged pattern."""
    hits: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            for pattern in PRIVILEGED_KEY_PATTERNS:
                if pattern in lowered:
                    hits.append(f"{path}/{key}")
            hits.extend(_privileged_keys(value, f"{path}/{key}"))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(_privileged_keys(value, f"{path}[{i}]"))
    return hits


def test_condition_a_carries_no_privileged_fields(records: List[Dict[str, Any]]) -> None:
    for record in records:
        obs = observable_record(record, masked=False)
        validate_batch(obs)  # must not raise
        leaked = _privileged_keys(obs)
        assert not leaked, f"Condition A leaked privileged keys: {leaked[:5]}"
        assert set(obs) <= OBSERVABLE_FIELDS


def test_condition_b_is_a_peer_free_projection(records: List[Dict[str, Any]]) -> None:
    for record in records:
        a = observable_record(record, masked=False)
        b = observable_record(record, masked=True)
        assert_condition_b_projection(a, b)  # raises on any deviation
        assert set(b) <= set(a), "Condition B schema is not a subset of Condition A"
        assert set(b) <= CONDITION_B_ALLOWED_FIELDS
        assert b["interaction_mask"] is True and a["interaction_mask"] is False
        assert "peer_agents_trajectory" not in b
        leaked = _privileged_keys(b)
        assert not leaked, f"Condition B leaked privileged keys: {leaked[:5]}"
        for entry in b["focal_trajectory"]:
            assert "blocked" not in entry, "Condition B must not expose the blocked flag"
            assert set(entry) <= {"step", "position", "position_before", "realized_action"}
        # identical underlying episodes: same focal path, only the mask differs
        assert [e["position"] for e in a["focal_trajectory"]] == [
            e["position"] for e in b["focal_trajectory"]
        ]


def test_privileged_field_injection_fails_loudly(records: List[Dict[str, Any]]) -> None:
    obs = observable_record(records[0], masked=False)
    for field in ("focal_goal_id", "step_intentions", "intended_action_history", "planner_internal_state"):
        injected = dict(obs)
        injected[field] = 0
        with pytest.raises(InformationIsolationError):
            validate_batch(injected)


def test_serialized_observable_files_are_clean(records: List[Dict[str, Any]]) -> None:
    """No cached/serialized observable artifact contains privileged key names."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "observable.json"
        payload = [observable_record(r, masked=False) for r in records] + [
            observable_record(r, masked=True) for r in records
        ]
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        found = sorted({p for p in PRIVILEGED_KEY_PATTERNS if re.search(rf'"{p}[a-z_]*"\s*:', text)})
        assert not found, f"serialized observations contain privileged keys: {found}"
        reloaded = json.loads(text)
        for obs in reloaded:
            validate_batch(obs)
    assert dataset_hash(records) == dataset_hash(records)  # hash is stable across calls


def test_prefix_observations_ignore_the_suffix(records: List[Dict[str, Any]]) -> None:
    """Observations up to t must not depend on what happens after t."""
    from src.trajectory.generator import episode_prefix

    for record in records:
        total = record["total_steps"]
        if total < 3:
            continue
        cut = max(2, total // 2)
        truncated = episode_prefix(record, cut)
        assert_prefix_invariance(record, truncated, cut - 1)
        full_obs = observable_record(record, masked=False)["focal_trajectory"][:cut]
        trunc_obs = observable_record(truncated, masked=False)["focal_trajectory"]
        assert full_obs == trunc_obs, "prefix observation changed with the suffix"


def test_model_input_tensor_has_no_privileged_columns(records: List[Dict[str, Any]]) -> None:
    for record in records:
        a = observable_record(record, masked=False)
        b = observable_record(record, masked=True)
        ta, tb = episode_tensor(a), episode_tensor(b)
        assert ta.shape[0] == tb.shape[0] == len(a["focal_trajectory"])
        assert tb.shape[1] < ta.shape[1], "Condition B must drop peer/blocked columns"
        names_a, names_b = feature_names(a), feature_names(b)
        assert len(names_a) == ta.shape[1] and len(names_b) == tb.shape[1]
        assert not any(n.startswith("peer_") for n in names_b)
        assert "blocked" not in names_b and "blocked" in names_a
        for name in names_a + names_b:
            lowered = name.lower()
            assert not any(p in lowered for p in PRIVILEGED_KEY_PATTERNS), name
        with pytest.raises(InformationIsolationError):
            episode_tensor({**a, "focal_intention_history": ["DIRECT"]})


def test_privileged_projection_is_separate(records: List[Dict[str, Any]]) -> None:
    for record in records:
        stripped = strip_privileged(record)
        for key in ("privileged_ground_truth", "scenario_spec", "seeds", "config_hash"):
            assert key not in stripped
        assert _privileged_keys(observable_record(record, masked=False)) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--disable-warnings"]))
