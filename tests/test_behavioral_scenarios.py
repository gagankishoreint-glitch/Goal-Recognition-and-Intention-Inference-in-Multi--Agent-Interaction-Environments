"""Behavioral assertions on the committed scenario families.

These test EXISTING behaviour — they change nothing about generation. The
verification dataset is rebuilt deterministically from the frozen master seed, so
every assertion below is stable across runs (it is a property of the committed
code, not of a random draw). Measured on the full 480-episode set:
open mean 32.6 steps / 6.17% peer-blocked, bottleneck 38.3 / 7.61%,
crossing 40.6 / 13.11%; peer/focal goal sharing 0.237 vs 1/K = 0.25.

If any of these ever flips, that is a finding about the generator, not a test to
silently relax.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from src.trajectory.dataset import DatasetConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records():
    from verification.m2.dataset_m2 import build_records

    return build_records()


def _family_stats(records):
    agg = collections.defaultdict(lambda: [0, 0, 0])  # episodes, steps, blocked
    for r in records:
        fam = dict(r["scenario_spec"]["env_params"])["scenario_family"]
        a = agg[fam]
        a[0] += 1
        foc = sorted(
            (s for s in r["steps"] if s["agent_id"] == r["focal_agent_id"]),
            key=lambda s: s["step"],
        )
        a[1] += len(foc)
        a[2] += sum(
            1 for s in foc if s["blocked"] and s["blocked_reason"] in ("occupied", "conflict")
        )
    return {
        fam: {
            "episodes": v[0],
            "mean_steps": v[1] / v[0],
            "blocked_rate": v[2] / v[1],
        }
        for fam, v in agg.items()
    }


def test_all_three_families_present(records) -> None:
    fams = {dict(r["scenario_spec"]["env_params"])["scenario_family"] for r in records}
    assert fams == {"open", "bottleneck", "crossing"}


def test_bottleneck_and_crossing_are_slower_than_open(records) -> None:
    st = _family_stats(records)
    assert st["bottleneck"]["mean_steps"] > st["open"]["mean_steps"], st
    assert st["crossing"]["mean_steps"] > st["open"]["mean_steps"], st


def test_crossing_is_more_contested_than_open(records) -> None:
    st = _family_stats(records)
    assert st["crossing"]["blocked_rate"] > st["open"]["blocked_rate"], st
    # crossing should at least double the open blocked rate on this set
    assert st["crossing"]["blocked_rate"] > 1.5 * st["open"]["blocked_rate"], st


def test_all_trajectories_stay_in_bounds(records) -> None:
    for r in records:
        w, h = r["grid_dimensions"]
        for s in r["steps"]:
            x, y = s["position"]
            assert 0 <= x < w and 0 <= y < h, (r["episode_id"], s["step"], (x, y))
            xb, yb = s["position_before"]
            assert 0 <= xb < w and 0 <= yb < h, (r["episode_id"], s["step"], (xb, yb))


def test_peer_goals_remain_independent_of_focal(records) -> None:
    k = None
    pairs = shared = 0
    for r in records:
        gt = r["privileged_ground_truth"]
        k = len(r["candidate_targets"])
        for p in gt["other_agent_goals"]:
            pairs += 1
            shared += int(p["goal_id"] == gt["focal_goal_id"])
    rate = shared / pairs
    assert pairs > 500
    # independent draws give 1/K sharing; the window excludes the old
    # without-replacement scheme (rate 0) and a degenerate one (rate 1)
    assert abs(rate - 1.0 / k) < 0.05, (rate, 1.0 / k)
