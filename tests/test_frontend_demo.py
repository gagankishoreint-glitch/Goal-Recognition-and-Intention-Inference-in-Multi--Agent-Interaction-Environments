"""Contract tests for the browser demo episode exporter."""

from __future__ import annotations

import json
from pathlib import Path

from frontend.generate_demo import build_demo_payload, write_demo

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "frontend"


def test_demo_exporter_writes_browser_payload(tmp_path: Path) -> None:
    output = write_demo(tmp_path / "demo_episode.json", seed=42)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert len(payload["episodes"]) >= 10
    assert payload["default_episode_id"] in {
        episode["episode_id"] for episode in payload["episodes"]
    }
    assert {episode["scenario_family"] for episode in payload["episodes"]} >= {
        "open",
        "bottleneck",
        "crossing",
    }
    assert len(
        {
            episode["final_reveal"]["true_goal_id"]
            for episode in payload["episodes"]
        }
    ) > 1

    for episode in payload["episodes"]:
        assert set(episode["grid"]) == {"width", "height"}
        assert episode["goals"]
        assert episode["frames"]
        assert episode["frames"][0]["step"] == 0
        assert len(episode["frames"]) > 1
        assert len(episode["frames"][-1]["agents"]) >= 2
        assert episode["final_reveal"]["true_goal_id"] in {
            goal["id"] for goal in episode["goals"]
        }
        goal_count = len(episode["goals"])
        for frame in episode["frames"]:
            assert len(frame["posterior"]) == goal_count
            assert all(0.0 <= probability <= 1.0 for probability in frame["posterior"])
            assert abs(sum(frame["posterior"]) - 1.0) < 1e-9
            assert frame["policy_states"]
            for agent in frame["agents"]:
                assert {"id", "x", "y", "action", "blocked"} <= agent.keys()
                assert 0 <= agent["x"] < episode["grid"]["width"]
                assert 0 <= agent["y"] < episode["grid"]["height"]
            assert not {"intention", "goal", "intended_action"} & frame.keys()
        assert "privileged_ground_truth" not in episode


def test_demo_generation_is_deterministic() -> None:
    assert build_demo_payload(seed=19) == build_demo_payload(seed=19)


def test_ui_has_required_controls_and_fetches_exported_contract() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "world",
        "play",
        "step",
        "reset",
        "new-world",
        "gru-beliefs",
        "bayes-beliefs",
        "agent-list",
    ):
        assert f'id="{element_id}"' in html
    assert 'api("/api/status")' in html
    assert 'api("/api/step"' in html
    assert "softmax(W hₜ + b)" in html
    assert "Assigned destinations are simulator truth" in html
    assert ".layout { display:grid; grid-template-columns:minmax(0,1fr)" in html
    assert ".side { display:grid; grid-template-columns:repeat(2,minmax(0,1fr))" in html
    for element_id in (
        "world-label",
        "step-counter",
        "episode-report",
        "agent-status-rows",
    ):
        assert f'id="{element_id}"' in html
    assert "Distance (Manhattan)" in html
    assert "assigned goal (simulator truth)" in html
    assert "inferred (model estimate)" in html
    assert "baseline = chance level for that label space" in html
    assert "no agent reached its goal within the horizon" in html


def test_live_models_are_around_one_million_parameters() -> None:
    import torch
    from frontend.live_models import NeuralActionPolicy, NeuralGoalRecognizer

    actor = NeuralActionPolicy(16, 16)
    recognizer = NeuralGoalRecognizer(input_dim=58, n_goals=4)
    actor_params = sum(parameter.numel() for parameter in actor.parameters())
    recognizer_params = sum(parameter.numel() for parameter in recognizer.parameters())
    assert 1_000_000 <= actor_params <= 1_300_000
    assert 1_000_000 <= recognizer_params <= 1_010_000
    intention_logits, action_logits = actor(torch.zeros(2, 1030))
    assert intention_logits.shape == (2, 4)
    assert action_logits.shape == (2, 20)


def test_live_simulation_uses_neural_action_and_streaming_beliefs() -> None:
    from frontend.live_models import NeuralActionPolicy, NeuralGoalRecognizer, TrainedModels
    from frontend.live_server import LiveSimulation

    models = TrainedModels(
        action_policy=NeuralActionPolicy(16, 16).eval(),
        goal_recognizer=NeuralGoalRecognizer(input_dim=58, n_goals=4).eval(),
        metrics={},
    )
    simulation = LiveSimulation(models, seed=42)
    initial = simulation.public_state()
    assert initial["max_steps"] == simulation.config.max_steps
    assert initial["frame"]["step"] == 0
    assert abs(sum(initial["frame"]["neural_goal"]) - 1.0) < 1e-6
    assert abs(sum(initial["frame"]["bayesian_goal"]) - 1.0) < 1e-9
    assert "final_reveal" not in initial["frame"]

    next_state = simulation.step()
    assert next_state["frame"]["step"] == 1
    assert abs(sum(next_state["frame"]["neural_goal"]) - 1.0) < 1e-6
    focal = next(
        agent
        for agent in next_state["frame"]["agents"]
        if agent["id"] == next_state["focal_agent_id"]
    )
    assert focal["intention"] in {
        "DIRECT",
        "DETOUR_LEFT",
        "DETOUR_RIGHT",
        "WAIT",
    }
    assert len(focal["intention_probabilities"]) == 4
    assert abs(sum(focal["intention_probabilities"]) - 1.0) < 1e-6
    assert len(focal["action_probabilities"]) == 5
    assert abs(sum(focal["action_probabilities"]) - 1.0) < 1e-6
    assert "final_reveal" not in next_state["frame"]
