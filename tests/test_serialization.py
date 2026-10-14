"""Serialization / on-disk schema tests, including the documented CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.validate_dataset import validate_data_dir
from src.features.observations import InformationIsolationError, validate_batch
from src.features.tensorize import episode_tensor
from src.trajectory.dataset import DatasetConfig, generate_dataset, split_records
from src.trajectory.serialize import (
    episode_to_schema,
    iter_observations,
    scenario_file_payload,
    validate_scenario_file,
    write_dataset,
)


@pytest.fixture(scope="module")
def cfg() -> DatasetConfig:
    return DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))


@pytest.fixture(scope="module")
def records(cfg):
    return generate_dataset(cfg, master_seed=31337, num_scenarios=4, episodes_per_scenario=3)


@pytest.fixture()
def written(tmp_path, cfg, records):
    splits = split_records(records, seed=11)
    metadata = write_dataset(
        records,
        splits,
        out_dir=tmp_path,
        master_seed=31337,
        config_hash=cfg.config_hash(),
        split_seed=11,
    )
    return tmp_path, metadata, splits


def test_written_layout_matches_the_documented_schema(written) -> None:
    root, metadata, splits = written
    assert (root / "metadata.json").exists()
    for split in ("train", "val", "test"):
        assert (root / split).is_dir()
        files = list((root / split).glob("scenario_*.json"))
        assert len(files) == metadata["splits"][split]["scenarios"] == len(
            {r["scenario_id"] for r in splits[split]}
        )
    payload = json.loads(sorted((root / "train").glob("scenario_*.json"))[0].read_text())
    assert {
        "schema_version",
        "scenario_id",
        "grid_dimensions",
        "static_obstacles",
        "candidate_targets",
        "episodes",
    } <= set(payload)
    episode = payload["episodes"][0]
    assert {"episode_id", "observations", "privileged_ground_truth", "total_steps"} <= set(episode)
    obs = episode["observations"][0]
    assert {"step", "focal_agent_state", "other_agents_state", "realized_action", "blocked"} <= set(obs)
    assert {"position", "position_before", "velocity"} <= set(obs["focal_agent_state"])
    assert validate_scenario_file(payload) == []


def test_velocity_is_the_position_delta(records) -> None:
    schema = episode_to_schema(records[0])
    for obs in schema["observations"]:
        state = obs["focal_agent_state"]
        assert state["velocity"] == [
            state["position"][0] - state["position_before"][0],
            state["position"][1] - state["position_before"][1],
        ]
        for peer in obs["other_agents_state"]:
            assert peer["velocity"] == [
                peer["position"][0] - peer["position_before"][0],
                peer["position"][1] - peer["position_before"][1],
            ]
            assert peer["realized_action"] in {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}


def test_loader_projection_is_observable_only(written) -> None:
    root, _, _ = written
    path = sorted((root / "train").glob("scenario_*.json"))[0]
    payload = json.loads(path.read_text())
    for masked in (False, True):
        for observation in iter_observations(payload, masked=masked):
            validate_batch(observation)
            assert episode_tensor(observation).shape[0] == len(observation["focal_trajectory"])
            if masked:
                assert "peer_agents_trajectory" not in observation
                assert all("blocked" not in e for e in observation["focal_trajectory"])
        with pytest.raises(InformationIsolationError):
            validate_batch({**iter_observations(payload)[0], "focal_goal_id": 0})


def test_no_privileged_key_inside_observations(records) -> None:
    payload = scenario_file_payload([r for r in records if r["scenario_id"] == records[0]["scenario_id"]])
    text = json.dumps({"observations": [e["observations"] for e in payload["episodes"]]})
    for pattern in ("intention", "goal", "intended_action", "blocked_reason"):
        assert f'"{pattern}' not in text, pattern
    assert validate_scenario_file(payload) == []
    # the privileged block is present at the episode level, and only there
    assert "privileged_ground_truth" in payload["episodes"][0]


def test_validator_rejects_injected_privileged_fields(written) -> None:
    root, _, _ = written
    path = sorted((root / "train").glob("scenario_*.json"))[0]
    payload = json.loads(path.read_text())
    payload["episodes"][0]["observations"][0]["intention"] = "DIRECT"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_data_dir(root)
    assert any("privileged" in e for e in errors), errors


def test_validator_rejects_out_of_bounds_positions(written) -> None:
    root, _, _ = written
    path = sorted((root / "train").glob("scenario_*.json"))[0]
    payload = json.loads(path.read_text())
    payload["episodes"][0]["observations"][0]["focal_agent_state"]["position"] = [999, 999]
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_data_dir(root)
    assert any("out of bounds" in e for e in errors), errors


def test_validator_rejects_scenario_overlap_across_splits(written) -> None:
    root, _, _ = written
    train_file = sorted((root / "train").glob("scenario_*.json"))[0]
    duplicate = root / "test" / train_file.name
    duplicate.write_text(train_file.read_text(), encoding="utf-8")
    errors = validate_data_dir(root)
    assert any("also appears in" in e for e in errors), errors


def test_validator_detects_a_broken_checksum(written) -> None:
    root, _, _ = written
    path = sorted((root / "train").glob("scenario_*.json"))[0]
    payload = json.loads(path.read_text())
    payload["episodes"][0]["total_steps"] = 1  # schema violation, keeps the hash check meaningful
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_data_dir(root)
    assert any("sha256" in e for e in errors), errors


def test_generator_cli_round_trip(tmp_path, cfg) -> None:
    """`python -m src.trajectory.generator` then `python -m src.evaluation.validate_dataset`."""
    out = tmp_path / "data"
    generate = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.trajectory.generator",
            "--config",
            str(REPO_ROOT / "configs" / "env_default.yaml"),
            "--output_dir",
            str(out),
            "--num_scenarios",
            "4",
            "--episodes_per_scenario",
            "2",
            "--seed",
            "99",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert generate.returncode == 0, generate.stderr
    summary = json.loads(generate.stdout)
    assert summary["episodes"] == 8
    assert summary["config_hash"] == cfg.config_hash()

    validate = subprocess.run(
        [sys.executable, "-m", "src.evaluation.validate_dataset", "--data_dir", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
    assert "VALID" in validate.stdout

    # a tampered file must flip the validator to exit code 1
    target = sorted((out / "train").glob("scenario_*.json"))[0]
    payload = json.loads(target.read_text())
    payload["episodes"][0]["observations"][0]["focal_agent_state"]["goal"] = [1, 1]
    target.write_text(json.dumps(payload), encoding="utf-8")
    validate_again = subprocess.run(
        [sys.executable, "-m", "src.evaluation.validate_dataset", "--data_dir", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert validate_again.returncode == 1
    assert "INVALID" in validate_again.stdout


def test_generation_is_deterministic_on_disk(tmp_path, cfg) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        records = generate_dataset(cfg, master_seed=555, num_scenarios=2, episodes_per_scenario=2)
        write_dataset(
            records,
            split_records(records, seed=3),
            out_dir=out,
            master_seed=555,
            config_hash=cfg.config_hash(),
            split_seed=3,
        )
    files_a = {p.name: p.read_text() for p in a.rglob("*.json")}
    files_b = {p.name: p.read_text() for p in b.rglob("*.json")}
    assert files_a == files_b
