# Archived diagnostics (superseded)

These scripts were the ad-hoc investigation harnesses used while the M2 §7
baseline gate was red. They hardcode a degenerate 8x8 scenario with two targets
and a focal goal frozen to `targets[0]`, and they score the goal-recognition
baselines against *intention* labels, so their numbers are not comparable to
anything in the current pipeline. They are kept for provenance only.

Replaced by:

- `verification/m2/dataset_m2.py` — the shared verification dataset builder
- `verification/m2/section6_triviality_test.py`
- `verification/m2/section7_baselines_test.py`
- `verification/m2/test_reproducibility.py`

Also archived here:

- `triviality_probe_exploratory.py` — the first exploratory intention probe
  (formerly `src/evaluation/triviality.py`), superseded by the frozen Section 6
  protocol.

Do not import from this directory.
