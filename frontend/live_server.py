#!/usr/bin/env python3
"""Train small CPU models and serve a live multi-agent simulation to the UI."""

from __future__ import annotations

import json
import argparse
import random
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.intentions import ALL_INTENTIONS  # noqa: E402
from src.evaluation.baselines import BayesianBeliefBaseline  # noqa: E402
from src.environment.gridworld import ACTIONS  # noqa: E402
from src.features.timing import frame_features  # noqa: E402
from src.trajectory.dataset import (  # noqa: E402
    DatasetConfig,
    derive_seed,
    sample_scenario,
)
from frontend.live_models import (  # noqa: E402
    ACTION_INDEX,
    TrainedModels,
    policy_features,
    train_models,
)


class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stage = "Waiting to train"
        self.current = 0
        self.total = 1
        self.models: Optional[TrainedModels] = None
        self.error: Optional[str] = None

    def report(self, stage: str, current: int, total: int) -> None:
        with self.lock:
            self.stage, self.current, self.total = stage, current, total

    def train(self) -> None:
        try:
            self.report("Generating synthetic training episodes", 0, 1)
            models = train_models(self.report)
            with self.lock:
                self.models = models
                self.stage = "Models ready"
                self.current = self.total = 1
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.stage = "Training failed"

    def public_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "ready": self.models is not None,
                "stage": self.stage,
                "current": self.current,
                "total": self.total,
                "error": self.error,
                "metrics": self.models.metrics if self.models is not None else None,
            }


class LiveSimulation:
    """One goal-conditioned policy rollout with online neural/Bayesian beliefs."""

    def __init__(self, models: TrainedModels, seed: int) -> None:
        self.models = models
        self.seed = int(seed)
        self.config = DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))
        self.spec = sample_scenario(self.config, self.seed % 10000, self.seed)
        self.grid = self.spec.grid()
        self.bounds = (self.spec.width, self.spec.height)
        self.obstacles = self.spec.obstacles
        self.targets = self.spec.targets
        self.focal_id = 0
        self.positions = {agent: position for agent, position in enumerate(self.spec.starts)}
        goal_rng = random.Random(derive_seed(self.seed, "live-goals"))
        self.goals = {
            agent: self.targets[goal_rng.randrange(len(self.targets))]
            for agent in self.positions
        }
        self.agent_rngs = {
            agent: random.Random(derive_seed(self.seed, "live-neural-agent", agent))
            for agent in self.positions
        }
        self.previous_intentions: Dict[int, Optional[str]] = {
            agent: None for agent in self.positions
        }
        self.dwell_remaining: Dict[int, int] = {
            agent: 0 for agent in self.positions
        }
        self.blocked_counts: Dict[int, int] = {agent: 0 for agent in self.positions}
        self.blocked_streaks: Dict[int, int] = {agent: 0 for agent in self.positions}
        self.previous_action: Optional[str] = None
        self.previous_blocked: Optional[bool] = None
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[str] = None
        self.gru_hidden: Optional[torch.Tensor] = None
        self.recognizer = BayesianBeliefBaseline(
            self.targets, obstacles=self.obstacles, bounds=self.bounds
        )
        self.recognizer.reset()
        self.neural_goal: list[float] = []
        self.neural_intention: list[float] = []
        self._infer_current_observation()
        self.last_decisions: Dict[int, Dict[str, Any]] = {}
        self.last_blocked: Dict[int, bool] = {agent: False for agent in self.positions}

    def _observer_frame(self) -> Dict[str, Any]:
        return {
            "t": self.step_count,
            "focal_cell": tuple(self.positions[self.focal_id]),
            "peer_cells": [
                tuple(position)
                for agent, position in sorted(self.positions.items())
                if agent != self.focal_id
            ],
            "previous_action": self.previous_action,
            "previous_blocked": self.previous_blocked,
            "grid_dimensions": self.bounds,
            "obstacles": self.obstacles,
            "targets": self.targets,
        }

    def _infer_current_observation(self) -> None:
        features = frame_features(
            self._observer_frame(),
            include_behavior=True,
            include_interaction=True,
            peer_visible=True,
            blocked_visible=True,
        )
        if len(features) != 58:
            raise RuntimeError(f"Expected 58 observer features, received {len(features)}")
        sequence = torch.tensor(features, dtype=torch.float32).view(1, 1, -1)
        with torch.inference_mode():
            goal_logits, intention_logits, self.gru_hidden = self.models.goal_recognizer(
                sequence, self.gru_hidden
            )
        self.neural_goal = torch.softmax(goal_logits[0, 0], dim=-1).tolist()
        self.neural_intention = torch.softmax(intention_logits[0, 0], dim=-1).tolist()

    def _frame(self) -> Dict[str, Any]:
        agents = []
        for agent, position in sorted(self.positions.items()):
            decision = self.last_decisions.get(agent, {})
            agents.append(
                {
                    "id": agent,
                    "x": position[0],
                    "y": position[1],
                    "goal_id": self.targets.index(self.goals[agent]),
                    "last_action": decision.get("action"),
                    "intention": decision.get("intention"),
                    "intention_probabilities": decision.get(
                        "intention_probabilities", [0.0] * len(ALL_INTENTIONS)
                    ),
                    "action_probabilities": decision.get(
                        "action_probabilities", [0.0] * len(ACTION_INDEX)
                    ),
                    "blocked": self.last_blocked.get(agent, False),
                }
            )
        frame: Dict[str, Any] = {
            "step": self.step_count,
            "agents": agents,
            "neural_goal": self.neural_goal,
            "neural_intention": self.neural_intention,
            "bayesian_goal": self.recognizer.goal_distribution(),
            "done": self.done,
            "termination_reason": self.termination_reason,
        }
        if self.done:
            frame["final_reveal"] = {
                "focal_goal_id": self.targets.index(self.goals[self.focal_id]),
                "agent_goal_ids": {
                    str(agent): self.targets.index(goal)
                    for agent, goal in sorted(self.goals.items())
                },
            }
        return frame

    def public_state(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "max_steps": self.config.max_steps,
            "scenario_family": dict(self.spec.config.env_params).get(
                "scenario_family", "mixed"
            ),
            "grid": {"width": self.spec.width, "height": self.spec.height},
            "obstacles": [list(cell) for cell in self.obstacles],
            "goals": [
                {"id": index, "position": list(target)}
                for index, target in enumerate(self.targets)
            ],
            "focal_agent_id": self.focal_id,
            "frame": self._frame(),
        }

    def step(self) -> Dict[str, Any]:
        if self.done:
            return self.public_state()

        intended: Dict[int, str] = {}
        decisions: Dict[int, Dict[str, Any]] = {}
        for agent in sorted(self.positions):
            current = self.positions[agent]
            goal = self.goals[agent]
            peers = [
                self.positions[other]
                for other in sorted(self.positions)
                if other != agent
            ]
            features = policy_features(
                self.spec.width,
                self.spec.height,
                self.obstacles,
                current,
                goal,
                peers,
                self.previous_intentions[agent],
                self.blocked_counts[agent],
                self.dwell_remaining[agent],
            )
            with torch.inference_mode():
                intention_logits, action_logits = self.models.action_policy(
                    torch.from_numpy(features).view(1, -1)
                )
                intention_probabilities = torch.softmax(
                    intention_logits[0], dim=-1
                ).tolist()
            interrupted = (
                self.blocked_counts[agent] >= self.config.blocked_detour_threshold
                or (self.previous_intentions[agent] is not None and current == goal)
            )
            if (
                self.dwell_remaining[agent] > 0
                and self.previous_intentions[agent] is not None
                and not interrupted
            ):
                intention = self.previous_intentions[agent]
                self.dwell_remaining[agent] -= 1
            else:
                draw = self.agent_rngs[agent].random()
                cumulative = 0.0
                intention_index = len(ALL_INTENTIONS) - 1
                for index, probability in enumerate(intention_probabilities):
                    cumulative += probability
                    if draw <= cumulative:
                        intention_index = index
                        break
                intention = ALL_INTENTIONS[intention_index]
                self.previous_intentions[agent] = intention
                self.dwell_remaining[agent] = self.config.min_dwell_steps - 1

            intention_index = ALL_INTENTIONS.index(intention)
            with torch.inference_mode():
                conditional_logits = action_logits.reshape(
                    1, len(ALL_INTENTIONS), len(ACTIONS)
                )[0, intention_index]
                probabilities = torch.softmax(conditional_logits, dim=-1).tolist()
            draw = self.agent_rngs[agent].random()
            cumulative = 0.0
            action = ACTIONS[-1]
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if draw <= cumulative:
                    action = ACTIONS[index]
                    break
            intended[agent] = action
            decisions[agent] = {
                "action": action,
                "intention": intention,
                "intention_probabilities": intention_probabilities,
                "action_probabilities": probabilities,
            }

        previous_positions = dict(self.positions)
        next_positions, realized, blocked = self.grid.step(self.positions, intended)
        self.positions = next_positions
        self.last_decisions = decisions
        self.last_blocked = dict(blocked)
        for agent in sorted(self.positions):
            self.blocked_counts[agent] = (
                self.blocked_counts[agent] + 1 if blocked[agent] else 0
            )
            self.blocked_streaks[agent] = (
                self.blocked_streaks[agent] + 1 if blocked[agent] else 0
            )

        focal_previous = previous_positions[self.focal_id]
        focal_current = self.positions[self.focal_id]
        self.previous_action = realized[self.focal_id]
        self.previous_blocked = bool(blocked[self.focal_id])
        self.recognizer.update(
            focal_previous,
            focal_current,
            realized[self.focal_id],
            bool(blocked[self.focal_id]),
        )
        self.step_count += 1

        if self.positions[self.focal_id] == self.goals[self.focal_id]:
            self.done = True
            self.termination_reason = "GOAL_REACHED"
        elif all(
            self.blocked_streaks[agent] >= self.config.deadlock_limit
            for agent in self.positions
        ):
            self.done = True
            self.termination_reason = "DEADLOCK_LIMIT"
        elif self.step_count >= self.config.max_steps:
            self.done = True
            self.termination_reason = "MAX_STEPS"

        self._infer_current_observation()
        return self.public_state()


class LiveApplication:
    def __init__(self) -> None:
        self.training = TrainingState()
        self.simulation: Optional[LiveSimulation] = None
        self.lock = threading.RLock()
        self.seed = 42

    def start_training(self) -> None:
        threading.Thread(target=self.training.train, daemon=True).start()

    def new_simulation(self, seed: Optional[int] = None) -> Dict[str, Any]:
        with self.training.lock:
            models = self.training.models
        if models is None:
            raise RuntimeError("Neural models are not ready yet")
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
            else:
                self.seed += 1
            self.simulation = LiveSimulation(models, self.seed)
            return self.simulation.public_state()

    def step(self) -> Dict[str, Any]:
        with self.lock:
            if self.simulation is None:
                raise RuntimeError("Start a simulation first")
            return self.simulation.step()


APP = LiveApplication()


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _json_response(self, status: int, value: Dict[str, Any]) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Request JSON must be an object")
        return value

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._json_response(200, APP.training.public_status())
            return
        if self.path == "/api/state":
            with APP.lock:
                if APP.simulation is None:
                    self._json_response(200, {"simulation": None})
                else:
                    self._json_response(200, {"simulation": APP.simulation.public_state()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            body = self._read_json()
            if self.path == "/api/new":
                state = APP.new_simulation(body.get("seed"))
                self._json_response(200, state)
            elif self.path == "/api/step":
                self._json_response(200, APP.step())
            else:
                self._json_response(404, {"error": "Unknown API endpoint"})
        except (ValueError, RuntimeError) as exc:
            self._json_response(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._json_response(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    APP.start_training()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), RequestHandler)
    print(f"Live goal-recognition sandbox: http://localhost:{args.port}")
    print("Training the two CPU models in the background; the page reports progress.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live sandbox.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
