# M2 Verification Guide: Reproducibility Gate

## Section 1 — Environment and Tooling Constraints
(Include Python version, determinism requirements, CPU-only execution path, etc. as specified in REPRODUCIBILITY.md)

## Section 2 — Execution CLI Commands (Standardized Interface)

```bash
# everything: test suite + all sections + regenerated evidence artifacts
python verification/m2/run_gate.py

# individual sections
python -m pytest                                                   # tests/ + verification/m2/
python -m pytest verification/m2/test_reproducibility.py -v        # Section 3 (3a-3i)
python -m pytest verification/m2/section4_leakage_test.py -v       # Section 4
python -m pytest verification/m2/section5_isolation_test.py -v     # Section 5
python verification/m2/section6_triviality_test.py                 # Section 6
python verification/m2/section7_baselines_test.py                  # Section 7

# dataset generation / validation (REPRODUCIBILITY.md §2.1-2.2)
python -m src.trajectory.generator --config configs/env_default.yaml \
    --output_dir data/ --num_scenarios 100 --episodes_per_scenario 20 --seed 42
python -m src.evaluation.validate_dataset --data_dir data/ --schema_version 1.0.0
```

All sections consume the same dataset: `verification/m2/dataset_m2.py` builds it
deterministically from `VERIFICATION_MASTER_SEED` and `configs/env_default.yaml`
(60 scenarios x 8 episodes = 480 episodes, scenario-level 70/15/15 split). It is
rebuilt rather than cached, so rebuilding it is itself a reproducibility check.

## Section 3: Reproducibility Suite

The M2 verification gate tests reproducibility at three distinct layers, executed in dependency order:

### Layer 1 — Single-episode determinism
Tests that establish foundation reproducibility:
- **3a.** Single-episode determinism: same seed + same config → identical rollout
- **3b.** Single-episode determinism across separate runs  
- **3c.** Running an episode does not mutate the static scenario
- **3d.** Different agent seed → RNG state is consumed (validates stochastic path exercise)

### Layer 2 — Episode addressability  
- **3e.** Episode addressability: episode does not depend on previously generated episodes
  *Requires metadata: scenario_spec, env_seed, agent_seed, config_hash in serialized episode*

### Layer 3 — Dataset reproducibility
- **3f.** Dataset reproducibility: full dataset reproducible from master seed

### Implementation Diagnostics
- **3g.** Implementation diagnostics: check for np.random.seed / global random
- **3h.** Implementation diagnostics: check for builtin hash() and PYTHONHASHSEED
- **3i.** Reset does not affect static scenario

### Test Sequencing & Dependencies
Tests must be executed in this order due to dependencies:

```
Section 3 (Reproducibility Tests)
   │
   ├── 3a–3d  Single-episode determinism (can run in any order)
   │
   ├── 3e     Addressability (depends on 3a passing)
   │             │
   │             └──── metadata precondition: scenario_spec, env_seed, agent_seed, config_hash required
   │
   ├── 3f     Dataset reproducibility (depends on 3e passing)
   │
   └── 3g/3h  Implementation diagnostics (can run in any order but after 3a-3f)
          │
          ▼
    Section 4 (Isolation Tests)
          │
          ▼
    Section 5 (Mutation Canary - 5b depends on 3e)
          │
          ▼
    Sections 6-7 (Triviality and Baselines tests)
```

**Execution Rule:** If an earlier test fails, later tests that depend on it do not need to run — the gate is already red, and the failure should be reported without further work.

## Section 4 — Scenario leakage

Run the adversarial leakage test:

1. Generate a scenario with at least three derived episodes. Assert all share the same `scenario_id`.
2. Construct a split where the same `scenario_id` appears in both train and test. Assert `assert_no_scenario_leakage(...)` raises a clear exception. Record the exception text.
3. Run the production split on the full dataset. Record unique scenario counts in train, validation, and test. Verify no overlap.

If the adversarial test does not raise, the gate is red. Report.

## Section 5 — End-to-end information isolation

Run the full pipeline test:

```
simulator.reset(seed)
  → trajectory generation
  → observable/privileged separation
  → condition A construction
  → condition B projection
  → dataset serialization
  → dataset loading
  → batch construction
  → model input tensor
```

Assert at each stage:

- no privileged field in condition A batch
- no privileged field in condition B batch
- no peer-derived field in condition B batch
- condition B schema ⊂ condition A schema
- no cached or preprocessed file contains privileged fields (grep)

If any assertion fails, the gate is red. Report.

## Section 6 — Intention triviality

Run the frozen triviality protocol on the held-out test split.

Timing convention:
- O_t contains position_t, realized_action_{t-1}, blocked_{t-1},
  peer positions_t, candidate targets, and obstacles.
- I_t generates action_t.
- Therefore action_t must NOT be an input to a probe predicting I_t.

Use the same shallow model family and fixed random seed across probes.

P0 — Majority baseline
- No input features.
- Predict the most frequent training-set intention.

P1 — Geometry-only primary probe
- Current-step observable geometry only.
- Include self position, peer positions/relative geometry,
  candidate target locations, map/obstacle features.
- Exclude actions and blocked flags.
- No temporal history.

P2 — Geometry + previous behavior
- P1 features
- + realized_action_{t-1}
- + blocked_{t-1}

P3 — Stacked-history reference
- Same shallow model family.
- Window k=5.
- All permitted observable fields over the window.
- This is a history reference, NOT a GRU or other sequence model.

Training/evaluation:
- Train only on training scenarios.
- Evaluate only on held-out test scenarios.
- Use Condition A observable fields.
- Fixed probe seed.

Report for every probe:
- Accuracy
- Macro-F1
- Per-intention precision/recall/F1
- 95% scenario-cluster-bootstrap confidence intervals
- Number of test scenarios and samples

Report:
- P3 − P1 Accuracy gap
- P3 − P1 Macro-F1 gap
- 95% scenario-cluster-bootstrap CI for the P3 − P1 Macro-F1 gap

Primary triviality gate:
- If the 95% CI for P3 − P1 Macro-F1 contains zero,
  the intention labels are considered geometrically trivial
  under this protocol.
- STOP M2 and report the result.
- Do NOT replace this criterion with an arbitrary universal
  accuracy/F1 threshold.

Implementation notes (frozen with the current pipeline):
- Probes P0/P1/P2/P3 are implemented in
  `verification/m2/section6_triviality_test.py`; the model family is multinomial
  logistic regression for every probe, `random_state=42`.
- The timing convention is realised through the `position_before` field: a sample
  predicting `I_t` uses `position_before` at step `t`, the peer cells at step `t`,
  and step `t-1`'s realized action / blocked flag. `A_t` is never an input.
- Confidence intervals are **scenario-cluster bootstraps over frozen probe
  predictions** (`src.evaluation.metrics.scenario_bootstrap_ci` and
  `paired_gap_ci`). Test scenarios are resampled with replacement and the metric
  is recomputed over all their samples. This brackets the point estimate by
  construction; the earlier revision bootstrapped *retrainings*, which produced
  an interval that did not contain its own point estimate and therefore made the
  gate verdict arbitrary. `tests/test_metrics.py` asserts the bracketing
  property.

Also report:
- intention class distribution overall and by split
- goal distribution
- mean/median intention dwell
- mean/median switches per episode
- fraction of episodes with >=1 intention switch
- blocked-step rate
- fraction of blocked steps attributable to peer interaction
  versus obstacle/boundary

Flag, but do not silently modify:
- any intention class >60% of labels
- median switches per episode = 0

Document exactly which features belong to P1 and why they satisfy
the frozen timing and observability definition.

## Section 7 — Baselines

Run random, distance heuristic, and Bayesian baselines on the **held-out test
scenarios** and report, for every observation fraction $\alpha$ in
EXPERIMENT_PLAN §1.2:

- Accuracy (Top-1) and Macro-F1, with 95% scenario-cluster bootstrap CIs
- NLL and Brier score
- the Bayesian intention marginal (accuracy, macro-F1, confusion matrix) against
  the empirical training-set intention prior
- NTCD and convergence-failure rate at $\tau_{\text{conf}} = 0.70$

**Task**: the baselines are goal recognizers, so the smoke test scores
$P(G \mid O_{1:t})$ against the privileged `focal_goal_id`. (An earlier revision
scored these goal baselines against *intention* labels, which made the ordering
check meaningless.)

**Expected ordering (smoke test)**: `random < distance < Bayesian` on Accuracy and
Macro-F1, evaluated on the **gate fractions** $\alpha \in \{0.30, 0.50, 0.75\}$.
If the ordering fails there, the gate is red. Two fractions are reported but not
gated, each for a stated structural reason:

- $\alpha \in \{0.10, 0.20\}$: with fewer than about five observed steps the focal
  agent has not committed to a direction, so a Markovian nearest-target heuristic
  carries no signal and can sit at or below chance.
- $\alpha = 1.00$: structurally degenerate. In `GOAL_REACHED` episodes the focal
  agent is standing on its goal, so "nearest target" reads the label off the
  terminal position without any inference. The report quantifies this
  (`terminal_degeneracy` block: goal-reached fraction and per-subset accuracies).

NLL ordering is reported as a diagnostic and not gated: at short horizons the
distance heuristic is *overconfident* (NLL above the uniform $\log K$, Brier above
0.75) rather than merely wrong, which is the expected failure mode of a sharply
tempered Markovian heuristic.

## Section 8: Evidence Table

The evidence table is generated, not hand-maintained: `python
verification/m2/run_gate.py` writes `verification/m2/EVIDENCE.md` with the
per-layer table, the dataset fingerprint, the Section 6/7 numbers and the overall
verdict. The authoritative artifact is that file.

| Layer | Test / artifact | Command |
|---|---|---|
| test-suite | complete suite (`tests/` + `verification/m2/`) | `python -m pytest` |
| single-episode | 3a-3d determinism, no mutation, RNG consumption | `python -m pytest verification/m2/test_reproducibility.py` |
| addressability | 3e reconstruction from `scenario_spec` + `seeds` + `config_hash` | `python -m pytest verification/m2/test_reproducibility.py::test_3e_episode_addressability` |
| dataset | 3f dataset reproducibility from the master seed | `python -m pytest verification/m2/test_reproducibility.py::test_3f_dataset_reproducibility_from_master_seed` |
| implementation | 3g/3h no global seeding, no builtin `hash`, PYTHONHASHSEED independence | `python -m pytest verification/m2/test_reproducibility.py -k "3g or 3h"` |
| single-episode | 3i reset leaves the static scenario untouched | `python -m pytest verification/m2/test_reproducibility.py::test_3i_reset_does_not_affect_static_scenario` |
| isolation | Section 4 scenario leakage | `python -m pytest verification/m2/section4_leakage_test.py` |
| isolation | Section 5 end-to-end information isolation | `python -m pytest verification/m2/section5_isolation_test.py` |
| triviality | Section 6 intention triviality probe | `python verification/m2/section6_triviality_test.py` |
| baseline | Section 7 goal-recognition baselines | `python verification/m2/section7_baselines_test.py` |
| cpu | CPU-only execution path | `python -m pytest tests/test_baselines.py::test_cpu_only_execution_path` |

### Layer Definitions
- **test-suite:** functionality, collision rules, dataset and metric contracts
- **single-episode:** reproducibility within one episode given fixed seeds and configuration
- **addressability:** episodes reconstructible from their serialized metadata
- **dataset:** the full dataset reproducible from the master seed
- **isolation:** scenario leakage and observable/privileged information isolation
- **triviality:** whether intention labels are recoverable from current-step geometry alone
- **baseline:** performance and ordering of the random, distance and Bayesian goal recognizers
- **cpu:** CPU-only execution path for all reported results

### Precondition for 3e
Each serialized episode records `scenario_spec`, `env_seed`, `agent_seed`,
`goal_seed` and `config_hash`. Missing metadata = **FAIL**, and
`reconstruct_episode_record` additionally raises when the supplied
`DatasetConfig` does not hash to the recorded `config_hash`.

### Implementation Audit for 3g/3h
Test 3g asserts that generation leaves `random` and `numpy.random` global state
untouched *and* greps `src/` for global seeding calls. Test 3h runs scenario-id
and seed derivation in subprocesses under three different `PYTHONHASHSEED`
values and asserts identical output, and greps `src/` for builtin `hash()`
usage. Both also assert deterministic agent iteration order (sorted), which
removes dict/set ordering as a source of variation.
