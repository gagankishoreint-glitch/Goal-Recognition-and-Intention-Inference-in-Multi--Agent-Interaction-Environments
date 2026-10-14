"""The report figure builder must run end-to-end and be driven by real data.

The figure is a derived artifact: every panel is computed from generated
episodes or from a result dictionary. This test exercises the builder on a small
dataset with a synthetic (but schema-correct) contrast block so that the figure
pipeline cannot rot silently between commits. It does not assert aesthetic
properties — those live in ``figures/fig1_overview.png``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trajectory.dataset import DatasetConfig, generate_dataset
from src.visualization.overview_figure import (
    build_overview_figure,
    interaction_events,
    select_showcase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> DatasetConfig:
    return DatasetConfig.from_yaml(str(REPO_ROOT / "configs" / "env_default.yaml"))


@pytest.fixture(scope="module")
def records(cfg: DatasetConfig):
    return generate_dataset(cfg, master_seed=20261008, num_scenarios=3, episodes_per_scenario=2)


def _contrast_block() -> dict:
    def gap(point, lo, hi):
        return {
            "accuracy": {
                "point_estimate": point,
                "ci_lower": lo,
                "ci_upper": hi,
                "metric": "accuracy",
            }
        }

    return {
        "probe": {
            "intention": {
                "gap_A_minus_B": gap(0.0616, 0.0203, 0.0995),
                "gap_D_minus_B": gap(0.0588, 0.0254, 0.0890),
                "gap_A_minus_D": gap(0.0027, -0.0101, 0.0142),
                "gap_E_minus_B": gap(0.0643, 0.0286, 0.0979),
            }
        }
    }


def test_select_showcase_prefers_most_interaction(records, cfg) -> None:
    record, events = select_showcase(
        records, cfg.policy_sensing_radius, cfg.peer_avoidance_weight
    )
    # the showcase has the most peer-blocked steps of any episode in the batch
    counts = [
        sum(interaction_events(r, cfg.policy_sensing_radius, cfg.peer_avoidance_weight)["blocked"])
        for r in records
    ]
    assert sum(events["blocked"]) == max(counts)
    assert len(events["blocked"]) > 0
    assert len(events["perturbed"]) == len(events["line_held"]) == len(events["blocked"])


def test_build_overview_figure_writes_a_four_panel_image(records, tmp_path) -> None:
    out = tmp_path / "fig.png"
    result = build_overview_figure(
        records,
        _contrast_block(),
        out,
        sensing_radius=3,
        avoidance_weight=0.05,
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 10_000, "figure must be a real image"


def test_belief_series_converges_toward_the_true_goal(records) -> None:
    from src.visualization.overview_figure import belief_series

    for record in records:
        probs = belief_series(record, use_blocked=True)
        assert abs(probs.sum(axis=1) - 1).max() < 1e-6, "posterior must stay normalized"
        true = int(record["privileged_ground_truth"]["focal_goal_id"])
        assert probs.shape[1] == len(record["candidate_targets"])
        # a well-specified Bayesian update should put the true goal no lower than
        # uniform by the end of a reasonably long episode
        if len(probs) > 20:
            assert probs[-1, true] >= 1.0 / probs.shape[1] - 1e-9
