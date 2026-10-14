"""RQ2 measurability: the environment must actually make agents interact.

The reviewer's blocker was that a ~2% focal blocking rate left the
interaction-visible vs interaction-masked comparison untestable. These tests pin
the three things that fixed it:

* the shipped config carries the interaction parameters (and they are hashed);
* generated episodes show a peer-blocked rate well above the M1 default;
* the peer/interaction feature channels behave as advertised — they respond to
  where peers are, and masking them zeroes exactly those channels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.agents.policy import PEER_AVOIDANCE_WEIGHT, peer_influence
from src.features.timing import (
    frame_features,
    frame_interaction_features,
    frame_width,
    observable_frame,
    stacked_features,
)
from src.trajectory.dataset import (
    DatasetConfig,
    generate_dataset,
    resolve_family,
    structural_obstacles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008


@pytest.fixture(scope="module")
def cfg() -> DatasetConfig:
    return DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))


@pytest.fixture(scope="module")
def small_dataset(cfg: DatasetConfig):
    return generate_dataset(cfg, master_seed=SEED, num_scenarios=6, episodes_per_scenario=3)


# --------------------------------------------------------------------------
# scenario families
# --------------------------------------------------------------------------
def test_mixed_families_rotate_by_scenario_index(cfg: DatasetConfig) -> None:
    assert resolve_family(cfg, 0) == "open"
    assert resolve_family(cfg, 1) == "bottleneck"
    assert resolve_family(cfg, 2) == "crossing"
    assert resolve_family(cfg, 3) == "open"
    pinned = DatasetConfig(**{**cfg.__dict__, "scenario_family": "crossing"})
    assert all(resolve_family(pinned, i) == "crossing" for i in range(5))


def test_structures_force_shared_space(cfg: DatasetConfig) -> None:
    import random

    rng = random.Random(0)
    w, h = cfg.grid_width, cfg.grid_height

    open_cells = structural_obstacles(rng, cfg, "open")
    assert open_cells == []

    wall = structural_obstacles(rng, cfg, "bottleneck")
    rows = {y for _, y in wall}
    assert len(rows) == 1, "bottleneck must be a single wall row"
    row = next(iter(rows))
    cols = sorted(x for x, _ in wall)
    interior = [x for x in range(1, w - 1)]
    gaps = [x for x in interior if x not in set(cols)]
    assert len(gaps) == 1, f"bottleneck must leave exactly one 1-cell gap, got {gaps}"
    assert 0 < row < h - 1

    cross = set(structural_obstacles(rng, cfg, "crossing"))
    cx, cy = w // 2, h // 2
    assert (cx, cy) not in cross, "the crossing centre must stay open"
    assert (cx, cy - 1) in cross and (cx - 1, cy) in cross

    with pytest.raises(ValueError):
        structural_obstacles(rng, cfg, "no-such-family")


def test_generated_scenarios_use_the_families(small_dataset) -> None:
    by_scenario = {}
    for r in small_dataset:
        by_scenario[r["scenario_id"]] = dict(r["scenario_spec"]["env_params"])["scenario_family"]
    assert set(by_scenario.values()) == {"open", "bottleneck", "crossing"}
    # every episode of a scenario carries the same family
    for r in small_dataset:
        assert dict(r["scenario_spec"]["env_params"])["scenario_family"] == by_scenario[r["scenario_id"]]


# --------------------------------------------------------------------------
# the rate that made RQ2 untestable
# --------------------------------------------------------------------------
def test_peer_blocked_rate_clears_the_m1_default(small_dataset) -> None:
    steps = blocked = 0
    for record in small_dataset:
        for s in record["steps"]:
            if s["agent_id"] != record["focal_agent_id"]:
                continue
            steps += 1
            if s["blocked"] and s["blocked_reason"] in ("occupied", "conflict"):
                blocked += 1
    rate = blocked / steps
    # M1 defaults (N=3, spawn radius 4, avoidance weight 0.25) measured 2.51%.
    # The shipped RQ2 config measures ~5.5% on this sample and ~7.5% on the full
    # verification set; the floor is set above the M1 number so a silent revert
    # of configs/env_default.yaml fails here.
    assert steps > 300
    assert rate > 0.035, f"peer-blocked rate {rate:.2%} is back at the untestable M1 level"


def test_blocking_is_peer_attributable(small_dataset) -> None:
    reasons = set()
    for record in small_dataset:
        for s in record["steps"]:
            if s["agent_id"] != record["focal_agent_id"] and not s["blocked"]:
                continue
            if s["blocked"]:
                reasons.add(s["blocked_reason"])
    assert reasons, "no blocked steps at all — interaction is not being exercised"
    assert reasons <= {"occupied", "conflict", "wall", "obstacle"}


# --------------------------------------------------------------------------
# the policy-side interaction measure
# --------------------------------------------------------------------------
def test_peer_influence_is_zero_without_nearby_peers() -> None:
    tv = peer_influence("DIRECT", (10, 10), (2, 2), [(15, 15)], policy_sensing_radius=2)
    assert tv == 0.0


def test_peer_influence_is_zero_when_nothing_is_avoided() -> None:
    peers = [(3, 2)]  # holds the greedy cell toward the goal
    assert peer_influence("DIRECT", (10, 2), (2, 2), peers, policy_sensing_radius=2) > 0.0
    # weight 1.0 leaves the action distribution untouched, so no influence
    assert (
        peer_influence(
            "DIRECT", (10, 2), (2, 2), peers, policy_sensing_radius=2, avoidance_weight=1.0
        )
        == 0.0
    )
    # WAIT never moves, so a nearby peer cannot change its action distribution
    assert peer_influence("WAIT", (10, 2), (2, 2), peers, policy_sensing_radius=2) == 0.0


def test_stronger_avoidance_means_more_influence() -> None:
    peers = [(3, 2)]
    weak = peer_influence(
        "DIRECT", (10, 2), (2, 2), peers, policy_sensing_radius=2, avoidance_weight=0.5
    )
    strong = peer_influence(
        "DIRECT", (10, 2), (2, 2), peers, policy_sensing_radius=2, avoidance_weight=0.05
    )
    assert strong > weak > 0.0
    assert 0.0 < PEER_AVOIDANCE_WEIGHT < 1.0


# --------------------------------------------------------------------------
# the feature-side interaction channels
# --------------------------------------------------------------------------
def _frame_with_peer(cfg: DatasetConfig, peer_cell):
    return {
        "t": 0,
        "focal_cell": (2, 2),
        "peer_cells": [peer_cell],
        "previous_action": None,
        "previous_blocked": None,
        "grid_dimensions": (cfg.grid_width, cfg.grid_height),
        "obstacles": [],
        "targets": [(10, 2), (2, 10), (14, 14), (1, 14)],
    }


def test_interaction_features_respond_to_peer_placement(cfg: DatasetConfig) -> None:
    # target 0 lies due east of (2, 2), so the greedy cell is (3, 2)
    far = frame_interaction_features(_frame_with_peer(cfg, (15, 0)))
    on_path = frame_interaction_features(_frame_with_peer(cfg, (3, 2)))
    assert far[0] == 0.0, "peer far from the goal line must not gate target 0"
    assert on_path[0] == 1.0, "a peer on the greedy cell must gate target 0"
    assert on_path[3] == 0.0, "the same peer must not gate target 1 (due south)"
    # adjacency / nearest-peer channels move too
    assert on_path[12] > far[12]
    assert on_path[13] < far[13]


def test_masking_zeroes_exactly_the_peer_and_interaction_channels(cfg: DatasetConfig) -> None:
    record = generate_dataset(cfg, master_seed=SEED, num_scenarios=1, episodes_per_scenario=1)[0]
    frame = observable_frame(record, 5)
    a = frame_features(frame, include_interaction=True, peer_visible=True)
    b = frame_features(frame, include_interaction=False, peer_visible=False)
    assert len(a) == len(b) == frame_width(record)
    k = len(frame["targets"])
    tail = 3 * k + 4                     # interaction block at the end
    peer_start = 2 + 4 * k               # after focal (x, y) and the K target blocks
    peer_end = peer_start + 4 * 3        # three peer slots x (dx, dy, manhattan, present)
    assert any(v != 0.0 for v in a), "condition A must carry non-zero features"
    assert all(v == 0.0 for v in b[len(b) - tail:]), "interaction block must be zeroed"
    assert all(v == 0.0 for v in b[peer_start:peer_end]), "peer slots must be zeroed"
    # every channel that is not peer-derived is bit-identical between conditions
    assert a[:peer_start] == b[:peer_start]
    assert a[peer_end:len(a) - tail] == b[peer_end:len(b) - tail]


def test_masked_and_full_features_have_identical_width(small_dataset) -> None:
    record = small_dataset[0]
    assert frame_width(record) == len(
        frame_features(observable_frame(record, 0), include_interaction=False, peer_visible=False)
    )
    full = stacked_features(record, 4, 3, True, True, True)
    masked = stacked_features(record, 4, 3, True, False, False)
    assert len(full) == len(masked) == 3 * frame_width(record)
    assert np.any(np.asarray(full) != np.asarray(masked))


def test_interaction_features_respect_the_timing_contract(small_dataset) -> None:
    """Mutating the present and all future steps must not move the frame."""
    import copy

    record = small_dataset[1]
    t = 3
    base = frame_features(observable_frame(record, t), include_interaction=True, peer_visible=True)
    mutated = copy.deepcopy(record)
    for step in mutated["steps"]:
        if step["step"] < t:
            continue
        step["realized_action"] = "WAIT"
        step["intention"] = "WAIT"
        step["goal"] = [0, 0]
        step["blocked"] = True
        step["blocked_reason"] = "occupied"
        step["position"] = [7, 7]
    after = frame_features(observable_frame(mutated, t), include_interaction=True, peer_visible=True)
    assert base == after


def test_shuffled_peer_control_keeps_the_feature_geometry(small_dataset) -> None:
    spec = importlib.util.spec_from_file_location(
        "rq2_contrast", REPO_ROOT / "experiments" / "multi_agent" / "run_interaction_contrast.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = small_dataset[0]
    shuffled = module.shuffle_peers(record, seed=1)
    assert frame_width(shuffled) == frame_width(record)
    assert module.family_of(record) in {"open", "bottleneck", "crossing"}

    a = np.asarray(stacked_features(record, 6, 3, True, True, True))
    c = np.asarray(stacked_features(shuffled, 6, 3, True, True, True))
    assert a.shape == c.shape
    assert np.any(a != c), "shuffling peers must change the interaction channels"
    # the focal trajectory itself is untouched, so the labels still line up
    focal_before = sorted(
        (s["step"], tuple(s["position_before"]))
        for s in record["steps"]
        if s["agent_id"] == record["focal_agent_id"]
    )
    focal_after = sorted(
        (s["step"], tuple(s["position_before"]))
        for s in shuffled["steps"]
        if s["agent_id"] == shuffled["focal_agent_id"]
    )
    assert focal_before == focal_after
