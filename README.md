# Goal Recognition in Multi-Agent Grid Environments: A Reproducible Baseline and Interaction-Sensitivity Study

*Report: [report/report.md](report/report.md) — references in [report/references.md](report/references.md).*

Research environment and benchmark for inferring an agent's **hidden terminal
goal**, its **latent short-horizon intentions**, and its **next action** from
partial behavioural observations in a multi-agent grid world.

- [PROJECT_SPEC.md](PROJECT_SPEC.md) — formal ontology, probabilistic formulation, environment dynamics, observer model
- [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) — RQ1–RQ5 and hypotheses
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — experiment matrix and statistical standards
- [ARCHITECTURE.md](ARCHITECTURE.md) — module contracts
- [DATASET_SCHEMA.md](DATASET_SCHEMA.md) — on-disk schema and observable/privileged split
- [MODEL_CARD.md](MODEL_CARD.md) — baseline and model specifications
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — seeding, CPU-only policy, CLI reference
- [verification/m2/](verification/m2/) — M2 verification gate and its evidence

## Status

| Milestone | State |
|---|---|
| M1 — specification and acceptance gate | documents complete, acceptance items tracked in [MILESTONE_1_ACCEPTANCE.md](MILESTONE_1_ACCEPTANCE.md) |
| M2 — environment, dataset, isolation, baselines | implemented and verified: gate **GREEN** ([EVIDENCE.md](verification/m2/EVIDENCE.md)) |
| M3 — sequence models (GRU multi-task), experiments | implemented and verified: NumPy GRU + interaction-sensitivity contrast ([gru_results.md](experiments/learned/gru_results.md)); IEEE report in [report/](report/) |

## Quickstart (CPU only)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy scikit-learn pyyaml pytest pandas   # torch only needed from M3

# run the whole verification suite (tests/ + verification/m2/)
python -m pytest

# regenerate every M2 verification artifact and the gate verdict
python verification/m2/run_gate.py

# generate a dataset and validate it (documented CLIs)
python -m src.trajectory.generator --config configs/env_default.yaml \
    --output_dir data/ --num_scenarios 100 --episodes_per_scenario 20 --seed 42
python -m src.evaluation.validate_dataset --data_dir data/ --schema_version 1.0.0
```

## Live neural sandbox

Install the project dependencies, including PyTorch, then start the local
simulation server from the project root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install numpy scipy scikit-learn pyyaml pytest pandas torch
python3 frontend/live_server.py
```

Open <http://localhost:8000>. The page reports progress while the server
generates synthetic episodes and trains two CPU models (no checkpoint is
downloaded or saved): an approximately 1.19-million-parameter neural policy
that predicts intentions and intention-conditioned actions, and an
approximately 1.00-million-parameter GRU that estimates the focal agent's
hidden goal. The policy is supervised to imitate the existing simulator policy;
this is a small generated-data demo, not a claim of real-world or
research-grade generalization. The simulator applies its existing intention
dwell and collision rules after the model's stochastic choices. A Bayesian
filter is shown as a non-neural goal-recognition comparison.

Each agent may observe its own assigned destination while acting. Destinations
are shown in the UI to explain the behavior, but are not provided to the GRU;
the observer's goal probabilities are estimates, not ground truth. The
simulator-only focal destination is revealed after the episode ends. Use **Step
once**, **Run**, **Pause**, **Reset same seed**, or **New world & goals** to
inspect the live rollout. If port 8000 is already occupied, start on another
port, for example `python3 frontend/live_server.py --port 8001`, then open
<http://localhost:8001>.

## Offline deterministic replay

The original static multi-episode replay remains available separately:

```bash
python3 frontend/generate_demo.py
python3 -m http.server 8000 --directory frontend
```

Open <http://localhost:8000>. It replays fixed deterministic examples and does
not run or train the live neural models.

## Layout

```text
src/environment/     grid world, symmetric single-pass collision rule, BFS path costs
src/agents/          intention vocabulary, stochastic policy (shared action model)
src/trajectory/      scenario sampling, episode generation, dataset build, serialization
src/features/        observable projections (Condition A/B), isolation guard, tensors
src/evaluation/      goal-recognition baselines, metrics suite, dataset validator
frontend/             live neural sandbox, training/server code, and offline replay exporter
configs/             environment, baseline, model and experiment configuration
tests/               unit/integration suite (environment, policy, dataset, baselines, metrics)
verification/m2/     M2 gate: reproducibility, leakage, isolation, triviality, baselines
```

## Two invariants worth knowing

1. **Observability is enforced, not assumed.** `validate_batch` rejects any batch
   carrying a privileged field, Condition B is a strict projection of Condition A,
   and the tensor builder is the last stage of that pipeline.
2. **Everything is reproducible from one master seed.** Seeds are derived with
   `hashlib` (never the salted builtin `hash`), episodes carry the metadata needed
   to reconstruct them bit-for-bit, and no code path touches global RNG state.
