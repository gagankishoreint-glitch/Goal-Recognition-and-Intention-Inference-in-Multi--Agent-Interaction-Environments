# Frozen snapshot of the pre-fix M2 gate (2026-10-07)

This directory preserves the artifacts from the M2 run that stopped the gate
**RED** at Section 7. It is kept for provenance and must not be read as the
current state — see `../EVIDENCE.md` for the current gate.

What was wrong with that run, and what replaced it:

| Defect in this snapshot | Fix |
|---|---|
| The focal goal was frozen to `targets[0]` for every episode, so the test-split goal distribution was a single class (see `label_stats.md`) | Per-episode hidden goal assignment from a dedicated `goal_seed` (`src/trajectory/dataset.py`), asserted by `tests/test_dataset.py::test_hidden_goal_varies_across_episodes` |
| Only 2 candidate targets on an 8x8 grid, versus K=4 on 16x16 in `configs/env_default.yaml` | Scenarios are sampled from the config with connectivity validation |
| Section 7 scored the *goal-recognition* baselines against *intention* labels, with a hardcoded intention distribution for the distance baseline | Section 7 now scores `P(G | O_{1:t})` against `focal_goal_id` |
| The Bayesian emission was hand-tuned and misspecified (NLL 1.2944 > prior 1.0350) | The emission is the generative action model shared with the policy |
| The Section 6 bootstrap resampled *retrainings*, producing a CI that did not contain its own point estimate (-0.0061 vs [0.0043, 0.1895]) | Scenario-cluster bootstrap over frozen predictions (`src/evaluation/metrics.py`), bracketing asserted in `tests/test_metrics.py` |
| `DETOUR_LEFT` and `DETOUR_RIGHT` produced identical action distributions | Mirror-asymmetric lateral drift; regression-tested in `tests/test_agents.py` |
| `behavior_weights` in the config were never read by the policy | Base utilities are `log(behavior_weights)` |
