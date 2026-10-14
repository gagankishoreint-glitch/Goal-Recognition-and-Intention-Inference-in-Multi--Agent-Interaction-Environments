# Reproducibility Protocol and Environment Guide

## 1. Environment and Tooling Constraints

### 1.1 Python Version Compatibility
- Verified on **Python 3.11.2** (Linux x86_64) with numpy/scipy/scikit-learn/pyyaml/pytest; see `verification/m2/EVIDENCE.md` for the exact interpreter recorded with the gate evidence.
- The M2 stack (environment, dataset, baselines, metrics) needs only NumPy and scikit-learn; **torch is required from M3 onwards** and is not imported anywhere in the M2 path (asserted by `tests/test_baselines.py::test_cpu_only_execution_path`).
- *Constraint Note*: prebuilt wheels for PyTorch and some compiled scientific packages may be constrained on very new interpreters (e.g. 3.14). Use 3.10–3.12 in a virtual environment when initializing the M3 model stack.
- All models, baselines, and evaluation pipelines must have an explicit CPU execution path. CPU execution is mandatory for all reported experimental results. MPS/GPU acceleration may be used opportunistically during development but is **not** an experimental dependency, and development acceleration must never be substituted for reported experiments (`development acceleration ≠ reported experiment`), since MPS and CPU are not bit-identical.
- No CUDA-only kernels, GPU-only tensor ops, GPU-specific dataloading, or GPU-specific memory assumptions may be required. Dataset and model sizes must remain feasible on a CPU-only laptop.

### 1.2 Determinism and Random Seeding
All stochastic components are governed by an explicit seed hierarchy:
- **Master Seed** (dataset level): `--seed` on the generator CLI; every other seed is derived from it.
- **Scenario Seed**: `derive_seed(master_seed, "scenario", i)` — governs obstacle generation, target placement and agent spawn positions.
- **Environment Seed**: `derive_seed(scenario_seed, "env", j)`.
- **Agent Seed**: `derive_seed(scenario_seed, "agent", j)` — governs stochastic policy sampling (categorical intention selection and action sampling), with a per-agent stream `derive_seed(agent_seed, "agent", a)`.
- **Goal Seed**: `derive_seed(scenario_seed, "goal", j)` — governs the hidden goal assignment, kept separate so goal sampling cannot perturb the policy stream.
- **Run / Noise Seed**: governs observation jitter and sensor dropouts (RQ5, M3).
- **Model Seed**: governs weight initialization and training batch shuffling (M3).

All derivations use `hashlib.sha256`, never the salted builtin `hash`, so ids and
seeds are stable across processes and `PYTHONHASHSEED` values (M2 tests 3g/3h).
No module under `src/` calls `random.seed`, `np.random.seed` or
`torch.manual_seed`; generation leaves global RNG state untouched.

For a fixed environment seed, scenario, policy configuration, and agent seed, the rollout must be reproducible. M2 must include tests verifying: `same seed + same configuration → identical rollout`, and (for stochastic policies) `different seed → potentially different rollout` while preserving the same transition rules and policy distribution.

To ensure exact reproducibility across machines:
```python
import random
import numpy as np
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
```

---

## 2. Execution CLI Commands (Standardized Interface)

### 2.1 Dataset Generation
```bash
python3 -m src.trajectory.generator \
  --config configs/env_default.yaml \
  --output_dir data/ \
  --num_scenarios 100 \
  --episodes_per_scenario 20 \
  --seed 42
```

### 2.2 Dataset Verification & Split Validation
```bash
python3 -m src.evaluation.validate_dataset \
  --data_dir data/ \
  --schema_version 1.0.0
```

### 2.3 M2 Verification Gate (all sections + evidence artifacts)
```bash
python verification/m2/run_gate.py
```

### 2.4 Running Baselines (Random, Distance Heuristic, Bayesian)
Implemented as the M2 Section 7 protocol:
```bash
python verification/m2/section7_baselines_test.py     # writes verification/m2/baselines.md
```
The packaged `src.inference.run_baselines` CLI over a written dataset directory is
an M3 deliverable.

### 2.5 Training Sequence Models (M3)
```bash
python3 -m src.models.train \
  --config configs/model_gru.yaml \
  --data_dir data/ \
  --output_dir results/checkpoints/gru/ \
  --seed 42
```

### 2.6 Running Experiments & Generating Reports (M3)
```bash
python3 -m src.evaluation.run_experiments \
  --config configs/experiments.yaml \
  --checkpoints_dir results/checkpoints/ \
  --results_dir results/ \
  --seeds 42 43 44 45 46
```

---

## 3. Data Integrity & Checksum Verification
- `src.trajectory.serialize.write_dataset` writes `metadata.json` with the master seed, config hash, per-split scenario ids and the SHA-256 of every partition file.
- `python -m src.evaluation.validate_dataset` verifies that:
  1. no `scenario_id` in `train` appears in `val` or `test`;
  2. no privileged field appears inside `observations`, and every loaded batch passes `validate_batch` (and drops peer/blocked fields in Condition B);
  3. all trajectory lengths and discrete coordinates remain within valid grid bounds and off the obstacles;
  4. every file hash matches `metadata.json` (tampering is detected);
  5. `schema_version` matches the expected version.
- These checks are exercised end-to-end, including the negative cases, by `tests/test_serialization.py`.
