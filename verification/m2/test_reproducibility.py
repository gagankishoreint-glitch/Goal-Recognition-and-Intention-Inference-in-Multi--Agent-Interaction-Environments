"""Section 3 — reproducibility suite for the M2 gate.

Every test runs against the *production* pipeline
(``src.trajectory.dataset`` → ``src.trajectory.generator`` →
``src.environment.gridworld`` / ``src.agents.policy``) on the shared M2
verification dataset, not against a hand-written stand-in scenario. Test ids
3a-3i match the evidence table in M2_VERIFICATION_GUIDE.md.
"""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.gridworld import GridWorld  # noqa: E402
from src.trajectory.dataset import (  # noqa: E402
    DatasetConfig,
    dataset_hash,
    derive_seed,
    generate_dataset,
    generate_episode_record,
    reconstruct_episode_record,
    sample_scenario,
)
from src.trajectory.generator import generate_episode  # noqa: E402
from verification.m2.dataset_m2 import (  # noqa: E402
    VERIFICATION_MASTER_SEED,
    verification_config,
)

SMALL_SCENARIOS = 3
SMALL_EPISODES = 3


@pytest.fixture(scope="module")
def cfg() -> DatasetConfig:
    return verification_config()


@pytest.fixture(scope="module")
def small_dataset(cfg: DatasetConfig) -> List[Dict[str, Any]]:
    return generate_dataset(
        cfg,
        master_seed=VERIFICATION_MASTER_SEED,
        num_scenarios=SMALL_SCENARIOS,
        episodes_per_scenario=SMALL_EPISODES,
    )


def test_3a_single_episode_determinism_same_seed(cfg: DatasetConfig) -> None:
    """3a. same seed + same config → identical rollout."""
    spec = sample_scenario(cfg, 0, VERIFICATION_MASTER_SEED)
    a = generate_episode_record(spec, cfg, 0, VERIFICATION_MASTER_SEED)
    b = generate_episode_record(spec, cfg, 0, VERIFICATION_MASTER_SEED)
    assert a == b, "identical seeds and config produced different episodes"
    assert a["steps"], "episode has no steps"


def test_3b_single_episode_determinism_different_runs() -> None:
    """3b. determinism across separate interpreter runs (subprocess)."""
    script = (
        "import sys, json;"
        "sys.path.insert(0, %r);"
        "from src.trajectory.dataset import generate_dataset;"
        "from verification.m2.dataset_m2 import verification_config, VERIFICATION_MASTER_SEED;"
        "recs = generate_dataset(verification_config(), master_seed=VERIFICATION_MASTER_SEED,"
        " num_scenarios=2, episodes_per_scenario=2);"
        "print(json.dumps(recs, sort_keys=True))" % str(REPO_ROOT)
    )
    env = dict(os.environ, PYTHONHASHSEED="0")
    first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    second = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout, "two interpreter runs produced different datasets"


def test_3c_episode_does_not_mutate_static_scenario(cfg: DatasetConfig) -> None:
    """3c. running an episode leaves the static scenario untouched."""
    spec = sample_scenario(cfg, 1, VERIFICATION_MASTER_SEED)
    grid = spec.grid()
    snapshot = {
        "scenario_id": spec.scenario_id,
        "width": grid.width,
        "height": grid.height,
        "obstacles": copy.deepcopy(grid.obstacles),
        "targets": copy.deepcopy(grid.targets),
        "agent_starts": copy.deepcopy(grid.agent_starts),
    }
    for episode_index in range(SMALL_EPISODES):
        generate_episode_record(spec, cfg, episode_index, VERIFICATION_MASTER_SEED)
    assert grid.width == snapshot["width"]
    assert grid.height == snapshot["height"]
    assert grid.obstacles == snapshot["obstacles"]
    assert grid.targets == snapshot["targets"]
    assert grid.agent_starts == snapshot["agent_starts"]
    assert spec.scenario_id == snapshot["scenario_id"]


def test_3d_different_agent_seed_rng_consumed(cfg: DatasetConfig) -> None:
    """3d. the policy RNG is consumed by a rollout and is seed-dependent."""
    spec = sample_scenario(cfg, 2, VERIFICATION_MASTER_SEED)
    agent_seed = derive_seed(spec.scenario_seed, "agent", 0)
    policies = spec.policies(agent_seed, cfg)
    state_before = policies[0].rng.getstate()

    episode = generate_episode(
        scenario_id=spec.scenario_id,
        grid=spec.grid(),
        agents=policies,
        goals={a: spec.targets[a % len(spec.targets)] for a in range(spec.num_agents)},
        starts={a: spec.starts[a] for a in range(spec.num_agents)},
        max_steps=cfg.max_steps,
        deadlock_limit=cfg.deadlock_limit,
        seed_info={"env_seed": 0, "agent_seed": agent_seed, "goal_seed": 0},
        policy_sensing_radius=cfg.policy_sensing_radius,
    )
    assert episode["steps"], "rollout produced no steps"
    assert policies[0].rng.getstate() != state_before, "policy RNG state was not consumed"

    other = spec.policies(derive_seed(spec.scenario_seed, "agent", 1), cfg)[0]
    assert other.rng.getstate() != state_before, "different agent seeds share an RNG stream"

    a = generate_dataset(cfg, master_seed=VERIFICATION_MASTER_SEED, num_scenarios=2, episodes_per_scenario=6)
    b = generate_dataset(cfg, master_seed=VERIFICATION_MASTER_SEED + 1, num_scenarios=2, episodes_per_scenario=6)
    assert dataset_hash(a) != dataset_hash(b), "different master seeds produced identical datasets"


def test_3e_episode_addressability(cfg: DatasetConfig, small_dataset: List[Dict[str, Any]]) -> None:
    """3e. every episode is reconstructible from its serialized metadata."""
    required_spec = {"scenario_spec", "seeds", "config_hash"}
    for record in small_dataset:
        missing = required_spec - set(record)
        assert not missing, f"episode {record['episode_id']} is missing metadata {missing}"
        assert {"env_seed", "agent_seed", "goal_seed"} <= set(record["seeds"])
        rebuilt = reconstruct_episode_record(record, cfg)
        assert rebuilt["scenario_id"] == record["scenario_id"]
        assert rebuilt["episode_id"] == record["episode_id"]
        assert rebuilt["steps"] == record["steps"]
        assert rebuilt["privileged_ground_truth"] == record["privileged_ground_truth"]
        assert rebuilt["termination_reason"] == record["termination_reason"]

    # and reconstruction must fail loudly under a different configuration
    mutated = DatasetConfig(**{**cfg.__dict__, "min_dwell_steps": cfg.min_dwell_steps + 1})
    with pytest.raises(ValueError, match="config_hash mismatch"):
        reconstruct_episode_record(small_dataset[0], mutated)


def test_3f_dataset_reproducibility_from_master_seed(cfg: DatasetConfig) -> None:
    """3f. the whole dataset is reproducible from the master seed."""
    a = generate_dataset(
        cfg, master_seed=VERIFICATION_MASTER_SEED, num_scenarios=4, episodes_per_scenario=3
    )
    b = generate_dataset(
        cfg, master_seed=VERIFICATION_MASTER_SEED, num_scenarios=4, episodes_per_scenario=3
    )
    assert len(a) == len(b) == 12
    assert dataset_hash(a) == dataset_hash(b)
    for ep_a, ep_b in zip(a, b):
        assert ep_a == ep_b


def test_3g_implementation_diagnostics_no_global_random_seed(cfg: DatasetConfig) -> None:
    """3g. generation does not touch global random / numpy random state."""
    np_before = np.random.get_state()
    py_before = random.getstate()
    generate_dataset(cfg, master_seed=VERIFICATION_MASTER_SEED, num_scenarios=2, episodes_per_scenario=2)
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    assert np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert py_before == random.getstate(), "global python random state was modified"

    src_files = sorted((REPO_ROOT / "src").rglob("*.py"))
    offenders = []
    for path in src_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("#", 1)[0]
            for pattern in ("np.random.seed(", "numpy.random.seed(", "random.seed(", "torch.manual_seed("):
                if pattern in stripped:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {pattern}")
    assert not offenders, f"global seeding found in src/: {offenders}"


def test_3h_implementation_diagnostics_no_builtin_hash(cfg: DatasetConfig) -> None:
    """3h. ids/seeds come from hashlib and are PYTHONHASHSEED independent."""
    script = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "from src.trajectory.dataset import sample_scenario, derive_seed;"
        "from verification.m2.dataset_m2 import verification_config, VERIFICATION_MASTER_SEED;"
        "spec = sample_scenario(verification_config(), 0, VERIFICATION_MASTER_SEED);"
        "print(spec.scenario_id, derive_seed(VERIFICATION_MASTER_SEED, 'agent', 0))" % str(REPO_ROOT)
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"scenario id / derived seed depend on PYTHONHASHSEED: {outputs}"

    src_files = sorted((REPO_ROOT / "src").rglob("*.py"))
    offenders = []
    for path in src_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("#", 1)[0].strip()
            if stripped.startswith("hash(") or " hash(" in stripped or "=hash(" in stripped.replace(" ", "="):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")
    assert not offenders, f"builtin hash() used in src/: {offenders}"


def test_3i_reset_does_not_affect_static_scenario(cfg: DatasetConfig) -> None:
    """3i. env.reset() and policy.reset() leave the static scenario unchanged."""
    spec = sample_scenario(cfg, 0, VERIFICATION_MASTER_SEED)
    grid = spec.grid()
    policies = spec.policies(derive_seed(spec.scenario_seed, "agent", 0), cfg)
    before = {
        "scenario_id": spec.scenario_id,
        "width": grid.width,
        "height": grid.height,
        "obstacles": copy.deepcopy(grid.obstacles),
        "targets": copy.deepcopy(grid.targets),
        "agent_starts": copy.deepcopy(grid.agent_starts),
    }
    for policy in policies.values():
        policy.select_intention(
            goal=spec.targets[0],
            current=spec.starts[0],
            history=["DIRECT"],
            blocked_count=0,
            obstacles=spec.obstacles,
            bounds=(grid.width, grid.height),
        )
    positions = grid.reset(seed=derive_seed(spec.scenario_seed, "env", 0))
    for policy in policies.values():
        policy.reset()
    assert positions == {i: s for i, s in enumerate(spec.starts)}
    assert grid.width == before["width"]
    assert grid.height == before["height"]
    assert grid.obstacles == before["obstacles"]
    assert grid.targets == before["targets"]
    assert grid.agent_starts == before["agent_starts"]
    assert spec.scenario_id == before["scenario_id"]
    assert all(p.state.prev_intention is None for p in policies.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--disable-warnings"]))
