"""Serialization of the dataset to the on-disk schema (DATASET_SCHEMA.md).

On disk a scenario file holds *both* sections: the observable ``observations``
list and the quarantined ``privileged_ground_truth`` block. The loader
(:func:`iter_observations`) is what enforces the split — it rebuilds the
Condition A / Condition B observation dicts through
``src.features.observations``, so a model-facing batch can only ever contain
observable fields. ``validate_scenario_file`` audits a file for schema
violations and privileged leakage inside ``observations``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..features.observations import (
    build_condition_a_observation,
    build_condition_b_observation,
)

SCHEMA_VERSION = "1.0.0"

PRIVILEGED_EPISODE_KEYS = (
    "privileged_ground_truth",
    "seeds",
    "scenario_spec",
    "config_hash",
    "scenario_index",
    "episode_index",
    "termination_reason",
)

PRIVILEGED_STEP_KEY_PATTERNS = (
    "intention",
    "goal",
    "intended_action",
    "blocked_reason",
    "planner_internal_state",
    "seed_state",
)


def _velocity(before: Sequence[int], after: Sequence[int]) -> List[int]:
    return [int(after[0] - before[0]), int(after[1] - before[1])]


def episode_to_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an in-memory episode record to the documented episode format."""
    focal_id = record["focal_agent_id"]
    observations: List[Dict[str, Any]] = []
    steps_by_index: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for step in record["steps"]:
        steps_by_index.setdefault(step["step"], {})[step["agent_id"]] = step
    for t in sorted(steps_by_index):
        by_agent = steps_by_index[t]
        focal = by_agent[focal_id]
        before = list(focal.get("position_before", focal["position"]))
        observations.append(
            {
                "step": t,
                "focal_agent_state": {
                    "position": list(focal["position"]),
                    "position_before": before,
                    "velocity": _velocity(before, focal["position"]),
                },
                "other_agents_state": [
                    {
                        "agent_id": agent_id,
                        "position": list(step["position"]),
                        "position_before": list(
                            step.get("position_before", step["position"])
                        ),
                        "velocity": _velocity(
                            step.get("position_before", step["position"]), step["position"]
                        ),
                        # peer realized actions are observable (PROJECT_SPEC §4.2)
                        "realized_action": step["realized_action"],
                        "blocked": bool(step["blocked"]),
                    }
                    for agent_id, step in sorted(by_agent.items())
                    if agent_id != focal_id
                ],
                "realized_action": focal["realized_action"],
                "blocked": bool(focal["blocked"]),
            }
        )
    return {
        "episode_id": record["episode_id"],
        "episode_index": record["episode_index"],
        "scenario_id": record["scenario_id"],
        "seeds": record["seeds"],
        "num_agents": record["num_agents"],
        "focal_agent_id": focal_id,
        "total_steps": record["total_steps"],
        "termination_reason": record["termination_reason"],
        "privileged_ground_truth": record["privileged_ground_truth"],
        "observations": observations,
    }


def scenario_file_payload(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one scenario file payload from that scenario's episode records."""
    if not records:
        raise ValueError("cannot serialize an empty scenario")
    head = records[0]
    if any(r["scenario_id"] != head["scenario_id"] for r in records):
        raise ValueError("records belong to more than one scenario")
    return {
        "schema_version": head.get("schema_version", SCHEMA_VERSION),
        "scenario_id": head["scenario_id"],
        "scenario_index": head["scenario_index"],
        "config_hash": head["config_hash"],
        "scenario_spec": head["scenario_spec"],
        "grid_dimensions": head["grid_dimensions"],
        "static_obstacles": head["static_obstacles"],
        "candidate_targets": head["candidate_targets"],
        "episodes": [episode_to_schema(r) for r in sorted(records, key=lambda r: r["episode_index"])],
    }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_dataset(
    records: Sequence[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
    out_dir: str | Path,
    master_seed: int,
    config_hash: str,
    split_seed: int,
) -> Dict[str, Any]:
    """Write ``data/{train,val,test}/scenario_*.json`` plus ``metadata.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "master_seed": int(master_seed),
        "config_hash": config_hash,
        "split_seed": int(split_seed),
        "episodes": len(records),
        "splits": {},
    }
    for split_name, rows in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        by_scenario: Dict[str, List[Dict[str, Any]]] = {}
        for record in rows:
            by_scenario.setdefault(record["scenario_id"], []).append(record)
        files: Dict[str, Any] = {}
        for scenario_id in sorted(by_scenario):
            payload = scenario_file_payload(by_scenario[scenario_id])
            path = split_dir / f"scenario_{scenario_id}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            files[path.name] = {
                "episodes": len(payload["episodes"]),
                "sha256": file_sha256(path),
            }
        metadata["splits"][split_name] = {
            "scenarios": len(by_scenario),
            "episodes": len(rows),
            "scenario_ids": sorted(by_scenario),
            "files": files,
        }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def load_scenario_file(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _focal_and_peer_steps(payload: Dict[str, Any], episode: Dict[str, Any]):
    focal_traj = []
    peer_traj = []
    for obs in episode["observations"]:
        focal_traj.append(
            {
                "step": obs["step"],
                "position": list(obs["focal_agent_state"]["position"]),
                "position_before": list(obs["focal_agent_state"]["position_before"]),
                "realized_action": obs["realized_action"],
                "blocked": bool(obs["blocked"]),
            }
        )
        for peer in obs["other_agents_state"]:
            peer_traj.append(
                {
                    "agent_id": peer["agent_id"],
                    "step": obs["step"],
                    "position": list(peer["position"]),
                    "position_before": list(peer["position_before"]),
                    "realized_action": peer["realized_action"],
                }
            )
    return focal_traj, peer_traj


def iter_observations(
    payload: Dict[str, Any], masked: bool = False
) -> List[Dict[str, Any]]:
    """Loader-side projection: observable Condition A / Condition B batches."""
    out: List[Dict[str, Any]] = []
    for episode in payload["episodes"]:
        focal_traj, peer_traj = _focal_and_peer_steps(payload, episode)
        condition_a = {
            "scenario_id": payload["scenario_id"],
            "grid_dimensions": list(payload["grid_dimensions"]),
            "static_obstacles": payload["static_obstacles"],
            "candidate_targets": payload["candidate_targets"],
            "focal_trajectory": focal_traj,
            "peer_agents_trajectory": peer_traj,
            "interaction_mask": False,
        }
        out.append(build_condition_b_observation(condition_a) if masked else condition_a)
    return out


def _privileged_hits(obj: Any, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if any(pattern in lowered for pattern in PRIVILEGED_STEP_KEY_PATTERNS):
                hits.append(f"{path}/{key}")
            hits.extend(_privileged_hits(value, f"{path}/{key}"))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(_privileged_hits(value, f"{path}[{i}]"))
    return hits


def validate_scenario_file(payload: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable violations (empty means valid)."""
    errors: List[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version {payload.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    for key in ("scenario_id", "grid_dimensions", "static_obstacles", "candidate_targets", "episodes"):
        if key not in payload:
            errors.append(f"missing top-level key {key!r}")
    if errors:
        return errors

    w, h = payload["grid_dimensions"]
    obstacles = {tuple(o) for o in payload["static_obstacles"]}
    for episode in payload["episodes"]:
        for key in ("episode_id", "scenario_id", "observations", "privileged_ground_truth"):
            if key not in episode:
                errors.append(f"{episode.get('episode_id', '?')}: missing {key!r}")
        if episode.get("scenario_id") != payload["scenario_id"]:
            errors.append(f"{episode.get('episode_id')}: scenario_id mismatch")
        if len(episode.get("observations", [])) != episode.get("total_steps"):
            errors.append(
                f"{episode.get('episode_id')}: {len(episode.get('observations', []))} observations "
                f"but total_steps={episode.get('total_steps')}"
            )
        for obs in episode.get("observations", []):
            hits = _privileged_hits(obs)
            if hits:
                errors.append(f"{episode.get('episode_id')} step {obs.get('step')}: privileged {hits[:3]}")
            state = obs.get("focal_agent_state", {})
            pos = state.get("position")
            if not pos or not (0 <= pos[0] < w and 0 <= pos[1] < h):
                errors.append(f"{episode.get('episode_id')} step {obs.get('step')}: position out of bounds")
            elif tuple(pos) in obstacles:
                errors.append(f"{episode.get('episode_id')} step {obs.get('step')}: position inside an obstacle")
            if obs.get("realized_action") not in {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}:
                errors.append(f"{episode.get('episode_id')} step {obs.get('step')}: bad realized_action")
    return errors
