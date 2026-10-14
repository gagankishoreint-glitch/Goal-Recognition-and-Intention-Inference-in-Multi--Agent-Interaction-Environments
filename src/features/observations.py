"""Observation construction: Condition A (full) and Condition B (masked
projection of A via an explicit allowlist)."""

from __future__ import annotations

from typing import Any, Dict, List

# Condition B allowlist: only fields whose semantics do not depend on
# another agent.
CONDITION_B_ALLOWED_FIELDS = {
    "scenario_id",
    "grid_dimensions",
    "static_obstacles",
    "candidate_targets",
    "focal_trajectory",
    "interaction_mask",
}

# Focal trajectory entry fields allowed in Condition B (no blocked, since
# blocked may reveal peer interaction; realized action kept).
_CONDITION_B_FOCAL_ENTRY_FIELDS = {
    "step",
    "position",
    "position_before",
    "realized_action",
}


def _focal_and_peer_trajectories(episode: Dict[str, Any]):
    focal_id = episode["focal_agent_id"]
    focal_traj = []
    peer_traj = []
    for s in episode["steps"]:
        if s["agent_id"] == focal_id:
            focal_traj.append(
                {
                    "step": s["step"],
                    "position": list(s["position"]),
                    "position_before": list(s.get("position_before", s["position"])),
                    "realized_action": s["realized_action"],
                    "blocked": bool(s["blocked"]),
                }
            )
        else:
            peer_traj.append(
                {
                    "agent_id": s["agent_id"],
                    "step": s["step"],
                    "position": list(s["position"]),
                    "position_before": list(s.get("position_before", s["position"])),
                    "realized_action": s["realized_action"],
                }
            )
    return focal_traj, peer_traj


def build_condition_a_observation(
    episode: Dict[str, Any], scenario: Dict[str, Any]
) -> Dict[str, Any]:
    focal_traj, peer_traj = _focal_and_peer_trajectories(episode)
    return {
        "scenario_id": episode["scenario_id"],
        "grid_dimensions": list(scenario["grid_dimensions"]),
        "static_obstacles": scenario["static_obstacles"],
        "candidate_targets": scenario["candidate_targets"],
        "focal_trajectory": focal_traj,
        "peer_agents_trajectory": peer_traj,
        "interaction_mask": False,
    }


def build_condition_b_observation(condition_a: Dict[str, Any]) -> Dict[str, Any]:
    """Condition B is a projection of Condition A, never an independent
    feature pipeline."""
    projected: Dict[str, Any] = {}
    for key, value in condition_a.items():
        if key not in CONDITION_B_ALLOWED_FIELDS:
            continue
        if key == "focal_trajectory":
            projected[key] = [
                {k: v for k, v in entry.items() if k in _CONDITION_B_FOCAL_ENTRY_FIELDS}
                for entry in value
            ]
        else:
            projected[key] = value
    projected["interaction_mask"] = True
    return projected


# ---------------- information-isolation guard ----------------

OBSERVABLE_FIELDS = set(CONDITION_B_ALLOWED_FIELDS) | {"peer_agents_trajectory"}

_PRIVILEGED_SUBSTRINGS = (
    "goal",
    "intention",
    "future",
    "intended_action",
    "privileged",
    "policy_state",
)


class InformationIsolationError(ValueError):
    pass


def validate_batch(batch: Dict[str, Any]) -> None:
    keys = set(batch.keys())
    extra = keys - OBSERVABLE_FIELDS
    if extra:
        raise InformationIsolationError(
            f"batch contains non-observable fields: {sorted(extra)}"
        )
    # defensive: reject any key whose semantics are privileged even if aliased
    for key in keys:
        lowered = key.lower()
        for bad in _PRIVILEGED_SUBSTRINGS:
            if bad in lowered and key not in OBSERVABLE_FIELDS:
                raise InformationIsolationError(
                    f"privileged field leaked into batch: {key}"
                )


def _entry_semantic_depends_on_peer(entry: Dict[str, Any]) -> bool:
    for key in entry:
        k = key.lower()
        if k.startswith("peer") or k.startswith("rel_") or "collision" in k or "blocked" in k:
            return True
    return False


def assert_condition_b_projection(condition_a: Dict[str, Any], condition_b: Dict[str, Any]) -> None:
    expected = build_condition_b_observation(condition_a)
    if condition_b != expected:
        raise AssertionError("Condition B is not the projection of Condition A")
    for key, value in condition_b.items():
        if key not in CONDITION_B_ALLOWED_FIELDS:
            raise AssertionError(f"Condition B field not in allowlist: {key}")
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and _entry_semantic_depends_on_peer(entry):
                    raise AssertionError(f"Condition B field depends on peer: {key}")


def assert_prefix_invariance(
    episode_a: Dict[str, Any], episode_b: Dict[str, Any], t: int
) -> None:
    """Observations up to t must not depend on the suffix after t."""
    steps_a = [s for s in episode_a["steps"] if s["step"] <= t]
    steps_b = [s for s in episode_b["steps"] if s["step"] <= t]
    obs_a = [
        {k: v for k, v in s.items() if k not in ("goal", "intention", "intended_action")}
        for s in steps_a
    ]
    obs_b = [
        {k: v for k, v in s.items() if k not in ("goal", "intention", "intended_action")}
        for s in steps_b
    ]
    if obs_a != obs_b:
        raise AssertionError("prefix observation differs across suffixes")
