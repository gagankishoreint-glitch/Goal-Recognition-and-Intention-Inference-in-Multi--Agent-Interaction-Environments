"""Small CPU neural models trained and exercised by the live sandbox."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.agents.intentions import ALL_INTENTIONS
from src.environment.gridworld import ACTIONS
from src.evaluation.baselines import BayesianBeliefBaseline, belief_trajectory
from src.features.timing import frame_features, observable_frame
from src.trajectory.dataset import DatasetConfig, generate_dataset, split_records

POLICY_HIDDEN = 1024
POLICY_BOTTLENECK = 128
RECOGNIZER_HIDDEN = 548
RECOGNIZER_INPUT = 58
INTENTION_INDEX = {label: index for index, label in enumerate(ALL_INTENTIONS)}
ACTION_INDEX = {label: index for index, label in enumerate(ACTIONS)}
TRAINING_SCENARIOS = 64
EPISODES_PER_SCENARIO = 3


class NeuralActionPolicy(nn.Module):
    """MLP predicting intention and intention-conditioned action distributions."""

    def __init__(self, grid_width: int, grid_height: int) -> None:
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        input_dim = 4 * grid_width * grid_height + len(ALL_INTENTIONS) + 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, POLICY_HIDDEN),
            nn.GELU(),
            nn.Linear(POLICY_HIDDEN, POLICY_BOTTLENECK),
            nn.GELU(),
        )
        self.intention_head = nn.Linear(POLICY_BOTTLENECK, len(ALL_INTENTIONS))
        self.action_head = nn.Linear(
            POLICY_BOTTLENECK, len(ALL_INTENTIONS) * len(ACTIONS)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return self.intention_head(encoded), self.action_head(encoded)


class NeuralGoalRecognizer(nn.Module):
    """GRU yielding episode-goal and per-frame intention predictions."""

    def __init__(self, input_dim: int = RECOGNIZER_INPUT, n_goals: int = 4) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, RECOGNIZER_HIDDEN, batch_first=True)
        self.goal_head = nn.Linear(RECOGNIZER_HIDDEN, n_goals)
        self.intention_head = nn.Linear(RECOGNIZER_HIDDEN, len(ALL_INTENTIONS))

    def forward(
        self, sequence: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, hidden_out = self.gru(sequence, hidden)
        return self.goal_head(encoded), self.intention_head(encoded), hidden_out


def policy_features(
    width: int,
    height: int,
    obstacles: Sequence[Tuple[int, int]],
    current: Tuple[int, int],
    goal: Tuple[int, int],
    peers: Sequence[Tuple[int, int]],
    previous_intention: Optional[str],
    blocked_count: int = 0,
    dwell_remaining: int = 0,
) -> np.ndarray:
    """Encode map, agent state and compact policy history for the acting agent."""
    if width != 16 or height != 16:
        raise ValueError("The live neural action policy currently requires a 16x16 grid")
    planes = np.zeros((4, height, width), dtype=np.float32)
    for x, y in obstacles:
        planes[0, y, x] = 1.0
    x, y = current
    planes[1, y, x] = 1.0
    for x, y in peers:
        planes[2, y, x] = 1.0
    x, y = goal
    planes[3, y, x] = 1.0
    previous_intention_one_hot = np.zeros(len(ALL_INTENTIONS), dtype=np.float32)
    if previous_intention is not None:
        previous_intention_one_hot[INTENTION_INDEX[previous_intention]] = 1.0
    temporal_context = np.asarray(
        [min(blocked_count, 10) / 10.0, min(dwell_remaining, 4) / 4.0],
        dtype=np.float32,
    )
    return np.concatenate((planes.ravel(), previous_intention_one_hot, temporal_context))


@dataclass
class TrainedModels:
    action_policy: NeuralActionPolicy
    goal_recognizer: NeuralGoalRecognizer
    metrics: Dict[str, Any]


def _policy_examples(
    records: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    config: DatasetConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    intention_labels: List[int] = []
    action_labels: List[int] = []
    for record in records:
        agent_ids = {int(item["agent_id"]) for item in record["steps"]}
        previous: Dict[int, Optional[str]] = {agent: None for agent in agent_ids}
        blocked_counts = {agent: 0 for agent in agent_ids}
        dwell_remaining = {agent: 0 for agent in agent_ids}
        by_step: Dict[int, List[Dict[str, Any]]] = {}
        for item in record["steps"]:
            by_step.setdefault(int(item["step"]), []).append(item)
        for step_index in sorted(by_step):
            transitions = by_step[step_index]
            for step in sorted(transitions, key=lambda item: item["agent_id"]):
                agent = int(step["agent_id"])
                goal = tuple(step["goal"])
                current = tuple(step["position_before"])
                peers = [
                    tuple(other["position_before"])
                    for other in transitions
                    if other["agent_id"] != agent
                ]
                xs.append(
                    policy_features(
                        width,
                        height,
                        [tuple(cell) for cell in record["static_obstacles"]],
                        current,
                        goal,
                        peers,
                        previous[agent],
                        blocked_counts[agent],
                        dwell_remaining[agent],
                    )
                )
                intention_labels.append(INTENTION_INDEX[step["intention"]])
                action_labels.append(ACTION_INDEX[step["intended_action"]])

                interrupted = (
                    blocked_counts[agent] >= config.blocked_detour_threshold
                    or current == goal
                )
                if dwell_remaining[agent] > 0 and previous[agent] is not None and not interrupted:
                    dwell_remaining[agent] -= 1
                else:
                    previous[agent] = step["intention"]
                    dwell_remaining[agent] = config.min_dwell_steps - 1
                blocked_counts[agent] = (
                    blocked_counts[agent] + 1 if step["blocked"] else 0
                )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(intention_labels, dtype=np.int64),
        np.asarray(action_labels, dtype=np.int64),
    )


def _recognizer_examples(
    records: Sequence[Dict[str, Any]],
) -> List[Tuple[np.ndarray, int, np.ndarray]]:
    samples = []
    for record in records:
        steps = sorted(
            (
                step
                for step in record["steps"]
                if step["agent_id"] == record["focal_agent_id"]
            ),
            key=lambda step: step["step"],
        )
        sequence = np.asarray(
            [
                frame_features(
                    observable_frame(record, index),
                    include_behavior=True,
                    include_interaction=True,
                    peer_visible=True,
                    blocked_visible=True,
                )
                for index in range(len(steps))
            ],
            dtype=np.float32,
        )
        intention_ids = np.asarray(
            [INTENTION_INDEX[step["intention"]] for step in steps], dtype=np.int64
        )
        goal_id = int(record["privileged_ground_truth"]["focal_goal_id"])
        samples.append((sequence, goal_id, intention_ids))
    return samples


def _evaluate_action(
    model: NeuralActionPolicy,
    xs: np.ndarray,
    intention_labels: np.ndarray,
    action_labels: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    with torch.inference_mode():
        intention_logits, action_logits = model(torch.from_numpy(xs))
        predicted_intentions = intention_logits.argmax(dim=-1)
        conditional_actions = action_logits.reshape(
            -1, len(ALL_INTENTIONS), len(ACTIONS)
        )[torch.arange(len(xs)), torch.from_numpy(intention_labels)]
        chosen_intentions = action_logits.reshape(
            -1, len(ALL_INTENTIONS), len(ACTIONS)
        )[torch.arange(len(xs)), predicted_intentions]
        return {
            "intention_accuracy": float(
                (predicted_intentions == torch.from_numpy(intention_labels))
                .float()
                .mean()
            ),
            "action_accuracy_given_expert_intention": float(
                (conditional_actions.argmax(dim=-1) == torch.from_numpy(action_labels))
                .float()
                .mean()
            ),
            "action_accuracy_using_model_intention": float(
                (chosen_intentions.argmax(dim=-1) == torch.from_numpy(action_labels))
                .float()
                .mean()
            ),
        }


def _evaluate_recognizer(
    model: NeuralGoalRecognizer,
    samples: Sequence[Tuple[np.ndarray, int, np.ndarray]],
) -> Dict[str, float]:
    model.eval()
    goal_correct = 0
    goal_count = 0
    intention_correct = 0
    intention_count = 0
    with torch.inference_mode():
        for sequence, goal_id, intention_ids in samples:
            goal_logits, intention_logits, _ = model(
                torch.from_numpy(sequence).unsqueeze(0)
            )
            goal_correct += int(goal_logits[0, -1].argmax().item() == goal_id)
            intention_correct += int(
                (intention_logits[0].argmax(dim=-1).numpy() == intention_ids).sum()
            )
            goal_count += 1
            intention_count += len(intention_ids)
    return {
        "goal_accuracy": goal_correct / max(goal_count, 1),
        "intention_accuracy": intention_correct / max(intention_count, 1),
    }


def _evaluate_bayesian_goal(records: Sequence[Dict[str, Any]]) -> float:
    correct = 0
    for record in records:
        targets = [tuple(target["position"]) for target in record["candidate_targets"]]
        model = BayesianBeliefBaseline(
            targets,
            obstacles=[tuple(cell) for cell in record["static_obstacles"]],
            bounds=tuple(record["grid_dimensions"]),
        )
        final_belief = belief_trajectory(
            model, record, focal_agent_id=record["focal_agent_id"], use_blocked=True
        )[-1]
        correct += int(
            int(np.argmax(final_belief))
            == int(record["privileged_ground_truth"]["focal_goal_id"])
        )
    return correct / max(len(records), 1)


def train_models(
    progress: Callable[[str, int, int], None] | None = None,
) -> TrainedModels:
    """Train on generated episodes with held-out scenarios, entirely on CPU."""
    def report(stage: str, current: int, total: int) -> None:
        if progress is not None:
            progress(stage, current, total)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.manual_seed(20261015)
    random.seed(20261015)
    config = DatasetConfig.from_yaml("configs/env_default.yaml")
    records = generate_dataset(
        config,
        master_seed=20261015,
        num_scenarios=TRAINING_SCENARIOS,
        episodes_per_scenario=EPISODES_PER_SCENARIO,
    )
    splits = split_records(records, train=0.7, val=0.15, test=0.15, seed=42)
    if not splits["train"] or not splits["test"]:
        raise RuntimeError("Synthetic training split must have train and test episodes")

    width, height = config.grid_width, config.grid_height
    model_policy = NeuralActionPolicy(width, height)
    x_train, intention_train, action_train = _policy_examples(
        splits["train"], width, height, config
    )
    x_test, intention_test, action_test = _policy_examples(
        splits["test"], width, height, config
    )
    policy_x = torch.from_numpy(x_train)
    policy_intentions = torch.from_numpy(intention_train)
    policy_actions = torch.from_numpy(action_train)
    optimizer = torch.optim.AdamW(model_policy.parameters(), lr=0.001, weight_decay=1e-4)
    policy_epochs = 8
    model_policy.train()
    for epoch in range(policy_epochs):
        permutation = torch.randperm(len(policy_x))
        for start in range(0, len(permutation), 128):
            indices = permutation[start : start + 128]
            intention_logits, action_logits = model_policy(policy_x[indices])
            action_logits = action_logits.reshape(
                -1, len(ALL_INTENTIONS), len(ACTIONS)
            )
            rows = torch.arange(len(indices))
            action_loss = nn.functional.cross_entropy(
                action_logits[rows, policy_intentions[indices]],
                policy_actions[indices],
            )
            intention_loss = nn.functional.cross_entropy(
                intention_logits, policy_intentions[indices]
            )
            loss = intention_loss + action_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        report("Training neural action policy", epoch + 1, policy_epochs)

    model_recognizer = NeuralGoalRecognizer(n_goals=config.num_candidate_targets)
    train_sequences = _recognizer_examples(splits["train"])
    test_sequences = _recognizer_examples(splits["test"])
    optimizer_gru = torch.optim.AdamW(
        model_recognizer.parameters(), lr=0.001, weight_decay=1e-4
    )
    recognizer_epochs = 8
    model_recognizer.train()
    for epoch in range(recognizer_epochs):
        order = torch.randperm(len(train_sequences)).tolist()
        for index in order:
            sequence, goal_id, intention_ids = train_sequences[index]
            x = torch.from_numpy(sequence).unsqueeze(0)
            goal_logits, intention_logits, _ = model_recognizer(x)
            goal_loss = nn.functional.cross_entropy(
                goal_logits[:, -1, :], torch.tensor([goal_id])
            )
            intention_loss = nn.functional.cross_entropy(
                intention_logits.reshape(-1, len(ALL_INTENTIONS)),
                torch.from_numpy(intention_ids),
            )
            loss = goal_loss + 0.5 * intention_loss
            optimizer_gru.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_recognizer.parameters(), 1.0)
            optimizer_gru.step()
        report("Training neural goal recognizer", epoch + 1, recognizer_epochs)

    policy_metrics = _evaluate_action(
        model_policy, x_test, intention_test, action_test
    )
    recognizer_metrics = _evaluate_recognizer(model_recognizer, test_sequences)
    return TrainedModels(
        action_policy=model_policy.eval(),
        goal_recognizer=model_recognizer.eval(),
        metrics={
            "device": "CPU",
            "training_scenarios": TRAINING_SCENARIOS,
            "training_episodes": len(splits["train"]),
            "test_scenarios": len({record["scenario_id"] for record in splits["test"]}),
            "test_episodes": len(splits["test"]),
            "action_policy_parameters": sum(
                parameter.numel() for parameter in model_policy.parameters()
            ),
            "goal_recognizer_parameters": sum(
                parameter.numel() for parameter in model_recognizer.parameters()
            ),
            **{
                f"action_policy_{name}": value
                for name, value in policy_metrics.items()
            },
            **recognizer_metrics,
            "bayesian_test_goal_accuracy": _evaluate_bayesian_goal(splits["test"]),
            "training_description": (
                "Supervised imitation on synthetic episodes; test episodes use "
                "held-out map scenarios. Accuracy is a demo diagnostic, not a benchmark."
            ),
        },
    )
