#!/usr/bin/env python3
"""Validate a generated dataset directory (REPRODUCIBILITY.md §2.2).

Checks, per scenario file:
* ``schema_version`` matches the expected version;
* no privileged field appears inside ``observations``;
* positions stay inside the grid and off the obstacles;
* ``scenario_id`` is consistent between the file and every episode;

and across the directory:
* no ``scenario_id`` appears in more than one split;
* every file hash recorded in ``metadata.json`` matches the file on disk;
* every episode deserializes into a valid observable batch.

Exit code 0 = valid, 1 = violations found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..features.observations import validate_batch
from ..trajectory.serialize import file_sha256, iter_observations, validate_scenario_file

SPLIT_NAMES = ("train", "val", "test")


def validate_data_dir(data_dir: str | Path, schema_version: str = "1.0.0") -> List[str]:
    root = Path(data_dir)
    errors: List[str] = []
    if not root.exists():
        return [f"data directory {root} does not exist"]

    metadata_path = root / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != schema_version:
            errors.append(
                f"metadata.json schema_version {metadata.get('schema_version')!r} != {schema_version!r}"
            )

    seen: Dict[str, str] = {}
    for split in SPLIT_NAMES:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("scenario_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            errors.extend(f"{split}/{path.name}: {e}" for e in validate_scenario_file(payload))
            scenario_id = payload.get("scenario_id")
            if scenario_id in seen and seen[scenario_id] != split:
                errors.append(
                    f"{split}/{path.name}: scenario {scenario_id} also appears in {seen[scenario_id]}"
                )
            seen[scenario_id] = split
            recorded = (
                metadata.get("splits", {}).get(split, {}).get("files", {}).get(path.name, {})
            )
            if recorded and recorded.get("sha256") != file_sha256(path):
                errors.append(f"{split}/{path.name}: sha256 does not match metadata.json")
            for observation in iter_observations(payload, masked=False):
                try:
                    validate_batch(observation)
                except Exception as exc:  # noqa: BLE001 - report as a violation
                    errors.append(f"{split}/{path.name}: observable batch rejected: {exc}")
            for observation in iter_observations(payload, masked=True):
                if "peer_agents_trajectory" in observation:
                    errors.append(f"{split}/{path.name}: Condition B still exposes peer trajectories")
                for entry in observation["focal_trajectory"]:
                    if "blocked" in entry:
                        errors.append(f"{split}/{path.name}: Condition B exposes the blocked flag")
    declared = {
        sid
        for info in metadata.get("splits", {}).values()
        for sid in info.get("scenario_ids", [])
    }
    missing = declared - set(seen)
    if missing:
        errors.append(f"metadata.json declares scenarios that are not on disk: {sorted(missing)[:5]}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.validate_dataset",
        description="Validate a generated dataset directory.",
    )
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--schema_version", default="1.0.0")
    args = parser.parse_args(argv)
    errors = validate_data_dir(args.data_dir, args.schema_version)
    if errors:
        print(f"INVALID: {len(errors)} violation(s)")
        for error in errors[:50]:
            print(f"  - {error}")
        return 1
    print(f"VALID: {args.data_dir} passes schema, isolation, bounds and split-disjointness checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
