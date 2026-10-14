# M2 Verification Gate Evidence

Generated 2026-10-09T09:29:09+00:00 by `python verification/m2/run_gate.py`.

## Environment

- Python 3.11.2 (Linux x86_64)
- CPU-only execution path; no torch/CUDA import in the baseline or metrics stack (asserted by `tests/test_baselines.py::test_cpu_only_execution_path`)

## Verification dataset

- master seed 20261008, 60 scenarios x 8 episodes = 480 episodes
- config hash `410df64cf3f7f224`, dataset sha256 `a452ac66fca4f3f0167c5c6870f1fe01af235c951dfc164bb8de072aa91d8373`
- splits (scenario level): train 336ep/42scen, val 72ep/9scen, test 72ep/9scen

## Evidence table

| Layer | Test / artifact | Command | Result | Status |
|---|---|---|---|---|
| test-suite | Complete test suite (tests/ + verification/) | `pytest -rs` | 141 passed in 160.70s (0:02:40) | PASS |
| single-episode | 3a-3d determinism, mutation, RNG consumption | `pytest verification/m2/test_reproducibility.py` | 9 passed | PASS |
| addressability | 3e episode reconstructible from metadata | `pytest verification/m2/test_reproducibility.py::test_3e_episode_addressability` | reconstruction bit-identical; config_hash mismatch raises | PASS |
| dataset | 3f dataset reproducible from master seed | `pytest verification/m2/test_reproducibility.py::test_3f_dataset_reproducibility_from_master_seed` | sha256 a452ac66fca4f3f0... | PASS |
| implementation | 3g/3h no global seeding, no builtin hash | `pytest verification/m2/test_reproducibility.py -k 3g or 3h` | PYTHONHASHSEED independent | PASS |
| isolation | Section 4 scenario leakage | `pytest verification/m2/section4_leakage_test.py` | leaky split rejected; production split disjoint | PASS |
| isolation | Section 5 end-to-end isolation | `pytest verification/m2/section5_isolation_test.py` | 7 passed | PASS |
| triviality | Section 6 intention triviality probe | `python verification/m2/section6_triviality_test.py` | NON-TRIVIAL (CONTINUE) | PASS |
| baseline | Section 7 goal-recognition baselines | `python verification/m2/section7_baselines_test.py` | ordering satisfied | PASS |

## Section 6 — intention triviality

- P0 majority: acc 0.3678, macro-F1 0.1344
- P1 geometry-only: acc 0.3582, macro-F1 0.2028
- P2 geometry + previous behaviour: acc 0.4157, macro-F1 0.3929
- P3 stacked history (k=5): acc 0.4711, macro-F1 0.4512
- P3 - P1 macro-F1 gap: 0.2485, 95% scenario-cluster CI [0.2001, 0.2957]
- verdict: NON-TRIVIAL (CONTINUE)

## Section 7 — goal-recognition baselines

- alpha 0.10: accuracy random 0.3194 / distance 0.2361 / Bayesian 0.4583; macro-F1 0.3160 / 0.2208 / 0.4472
- alpha 0.20: accuracy random 0.3611 / distance 0.2500 / Bayesian 0.4444; macro-F1 0.3591 / 0.2320 / 0.4341
- alpha 0.30: accuracy random 0.2361 / distance 0.2639 / Bayesian 0.4861; macro-F1 0.2283 / 0.2500 / 0.4743
- alpha 0.50: accuracy random 0.2222 / distance 0.3472 / Bayesian 0.6667; macro-F1 0.1943 / 0.3256 / 0.6725
- alpha 0.75: accuracy random 0.2778 / distance 0.4861 / Bayesian 0.7778; macro-F1 0.2792 / 0.4751 / 0.7722
- alpha 1.00: accuracy random 0.1806 / distance 0.5417 / Bayesian 0.8889; macro-F1 0.1788 / 0.5274 / 0.8923
- gate fractions ['0.30', '0.50', '0.75']: ordering satisfied
- alpha = 1.00 excluded as structurally degenerate (28/72 test episodes end in GOAL_REACHED)
- Bayesian intention marginal: NLL 0.9475 vs empirical prior 1.3124; accuracy 0.5618 vs 0.3678

## Gate status

- test suite: GREEN
- Section 6: GREEN (CONTINUE)
- Section 7: GREEN (CONTINUE)
- **OVERALL: GREEN (CONTINUE)**

