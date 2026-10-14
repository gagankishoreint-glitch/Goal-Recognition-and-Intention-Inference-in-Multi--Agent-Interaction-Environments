#!/usr/bin/env python3
"""Generate deterministic episodes and Bayesian goal-belief replays for the UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines import BayesianBeliefBaseline  # noqa: E402
from src.trajectory.dataset import DatasetConfig, generate_dataset  # noqa: E402


def _episode_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build one replay; only observable movement is passed to the recognizer."""
    focal_id = int(record["focal_agent_id"])
    targets = [tuple(item["position"]) for item in record["candidate_targets"]]
    bounds = tuple(record["grid_dimensions"])
    obstacles = [tuple(cell) for cell in record["static_obstacles"]]
    recognizer = BayesianBeliefBaseline(targets, obstacles=obstacles, bounds=bounds)

    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for transition in record["steps"]:
        by_step.setdefault(int(transition["step"]), []).append(transition)

    first_step = by_step[min(by_step)] if by_step else []
    initial_agents = [
        {
            "id": int(item["agent_id"]),
            "x": int(item["position_before"][0]),
            "y": int(item["position_before"][1]),
            "action": None,
            "blocked": False,
        }
        for item in sorted(first_step, key=lambda item: item["agent_id"])
    ]
    frames: List[Dict[str, Any]] = [
        {
            "step": 0,
            "agents": initial_agents,
            "posterior": recognizer.goal_distribution(),
            "policy_states": {
                str(item["agent_id"]): item["intention"] for item in first_step
            },
        }
    ]

    for step_index in sorted(by_step):
        transitions = sorted(by_step[step_index], key=lambda item: item["agent_id"])
        focal_transition = next(item for item in transitions if item["agent_id"] == focal_id)
        recognizer.update(
            tuple(focal_transition["position_before"]),
            tuple(focal_transition["position"]),
            focal_transition["realized_action"],
            bool(focal_transition["blocked"]),
        )
        frames.append(
            {
                "step": step_index + 1,
                "agents": [
                    {
                        "id": int(item["agent_id"]),
                        "x": int(item["position"][0]),
                        "y": int(item["position"][1]),
                        "action": item["realized_action"],
                        "blocked": bool(item["blocked"]),
                    }
                    for item in transitions
                ],
                "posterior": recognizer.goal_distribution(),
                "policy_states": {
                    str(item["agent_id"]): item["intention"] for item in transitions
                },
            }
        )

    family = dict(record["scenario_spec"]["env_params"]).get("scenario_family", "mixed")
    true_goal_id = int(record["privileged_ground_truth"]["focal_goal_id"])
    return {
        "episode_id": record["episode_id"],
        "scenario_family": family,
        "termination_reason": record["termination_reason"],
        "grid": {"width": int(bounds[0]), "height": int(bounds[1])},
        "obstacles": [list(cell) for cell in record["static_obstacles"]],
        "goals": [
            {"id": int(item["target_id"]), "position": list(item["position"])}
            for item in record["candidate_targets"]
        ],
        "focal_agent_id": focal_id,
        "frames": frames,
        "final_reveal": {"true_goal_id": true_goal_id},
    }


def build_demo_payload(seed: int = 42) -> Dict[str, Any]:
    """Return reproducible examples across maps, destinations, and trajectories."""
    config = DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))
    records = generate_dataset(
        config,
        master_seed=seed,
        num_scenarios=8,
        episodes_per_scenario=2,
    )
    episodes = [_episode_payload(record) for record in records]
    default_index = max(
        range(len(records)),
        key=lambda index: (
            sum(
                bool(step["blocked"])
                for step in records[index]["steps"]
                if step["agent_id"] == records[index]["focal_agent_id"]
            ),
            len(
                {
                    step["realized_action"]
                    for step in records[index]["steps"]
                    if step["agent_id"] == records[index]["focal_agent_id"]
                }
            ),
            len(episodes[index]["frames"]),
        ),
    )
    return {
        "schema_version": 2,
        "seed": seed,
        "episodes": episodes,
        "default_episode_id": episodes[default_index]["episode_id"],
    }


def write_demo(path: Path, seed: int = 42) -> Path:
    payload = build_demo_payload(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "demo_episode.json",
        help="output JSON path (default: frontend/demo_episode.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = write_demo(args.out, args.seed)
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        f"Wrote {output} | {len(payload['episodes'])} reproducible episodes "
        f"| default {payload['default_episode_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
