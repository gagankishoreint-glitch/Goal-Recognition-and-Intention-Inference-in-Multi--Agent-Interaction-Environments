# Master Experiment Plan

## 1. Experimental Setup and Design

### 1.1 Dataset Partitioning Strategy
To avoid any data leakage from environment geometry:
- **Scenario Generation**: 100 unique map layouts (varied obstacle distributions and goal coordinates).
- **Split Distribution**:
  - **Train**: 70 scenarios (1,400 total episodes: 20 episodes per scenario).
  - **Validation**: 15 scenarios (300 episodes).
  - **Test**: 15 scenarios (300 episodes).
- **Stratification**: Scenarios are balanced across the 4 candidate goal targets (the focal goal is re-drawn per episode; observed goal counts per split are reported in `verification/m2/dataset_hash.txt`) and the 4 behavioral intention types `{DIRECT, DETOUR_LEFT, DETOUR_RIGHT, WAIT}`.

- **Split Verification**: A `split_dataset(...)` helper must perform the split before trajectory-level sampling, and a leakage assertion must verify no scenario/configuration appears in more than one split. The test set must measure generalization to unseen scenarios.
- **Interaction Experiment (RQ2)**: Condition A (full interaction observation) and Condition B (interaction-masked) must be evaluated on the same underlying episodes with only the mask changed.
- **RQ2 pre-study (executed, `experiments/multi_agent/interaction_contrast.md`)**: the environment was first made interaction-dense enough to measure — `N = 4` agents, `scenario_family = mixed` (rotating `open`/`bottleneck`/`crossing`), peer spawn radius 2, avoidance weight 0.05. On the test split the focal agent is peer-blocked on 6.5–20.3% of steps by family (10.45% bottleneck, 20.33% crossing, 6.47% open), a peer perturbs its action distribution on 16–25% of steps, and a peer holds a cell on its direct line to the goal on 4.2–8.2% of steps — against 2.51% peer-blocked under the M1 defaults. Measured outcome with a 5-frame logistic probe: goal recognition shows **no supported effect** of interaction information (A−D −0.0147, CI [−0.0494, +0.0201]); intention inference does, but the effect is carried entirely by the previous-step `blocked` flag (+0.0643 accuracy, CI [+0.0286, +0.0979]) and **not** by the peer positions (A−D +0.0027, CI [−0.0101, +0.0142]). The naive zero-mask contrast (+0.0616, CI [+0.0203, +0.0995]) is reproduced by Gaussian noise and must not be reported as an interaction effect. Consequence for Phase 3: the masked arm is built by noise substitution (`TIMING_CONTRACT.md` §8), the `blocked` flag is ablated separately, and the value of peer *trajectories over time* — which a 5-frame window cannot express — is the specific hypothesis the sequence model is there to test.

### 1.2 Evaluation Metrics Formulation
1. **Accuracy (Top-1)**:
   $$\text{Acc} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\hat{g}_i = g_i^*)$$
2. **Macro-Averaged F1-Score**:
   $$\text{Macro-F1} = \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \frac{2 \cdot P_g \cdot R_g}{P_g + R_g}$$
3. **Negative Log-Likelihood (NLL)**:
   $$\text{NLL} = -\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \log P(G = g_i^* \mid O_{1:t}^{(i)})$$
4. **Brier Score (Calibration)**:
   $$\text{BS} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \sum_{g \in \mathcal{G}} \left(P(G = g \mid O_{1:t}^{(i)}) - \mathbb{I}(g = g_i^*)\right)^2$$
5. **Normalized Time-to-Correct-Decision (NTCD)** (retained with explicit definition):
   $$\text{NTCD} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \frac{t_i^*}{T_i}$$
   where $t_i^*$ is the first timestep at which the prediction is correct and remains correct for all subsequent steps with $P(g_i^* \mid O_{1:t'}) \ge 0.7$ (stability requirement), and $t_i^* = T_i$ (i.e., $\text{NTCD}=1$) if no such timestep exists (never-correct treatment). Aggregation is the episode mean, reported with scenario-level bootstrap CIs.

---

## 2. Phase-by-Phase Experiment Matrix

| Phase | Target RQ | Models Evaluated | Conditions & Parameters | Primary Output |
|---|---|---|---|---|
| **Phase 1: Baselines** | RQ1 | Baseline A (Random), Baseline B (Heuristic), Baseline C (Bayesian), Model D (GRU) | Full trajectories ($\alpha = 1.0$), Clean observations | Accuracy, Macro-F1, NLL, Brier Score Table |
| **Phase 2: Early Recognition** | RQ3 | Heuristic B, Bayesian C, Model D (GRU) | $\alpha \in \{0.10, 0.20, 0.30, 0.50, 0.75, 1.00\}$ | Accuracy vs. Horizon Curves, NTCD Distribution |
| **Phase 3: Multi-Agent Context (RQ2)** | RQ2 | Multi-Task Model D | Condition A (full observation) vs Condition B (interaction-masked projection) on identical episodes; $N \in \{1,2,4\}$ | Differential Accuracy ($\Delta \text{Acc}$), Interaction F1 |
| **Phase 4: Intention & Action** | RQ4 | Multi-Task Model D | Intention $I \in \mathcal{I}$, Next action $A_{t+1}$ | Intention Confusion Matrix, Action Top-1/Top-2 Acc |
| **Phase 5: Noise & Missing Data** | RQ5 | Heuristic B, Bayesian C, Model D | Gaussian $\sigma \in \{0, 0.2, 0.5, 1.0\}$; Missing $p \in \{0, 0.1, 0.25, 0.5\}$ | Robustness Degradation Curves, ECE Score |
| **Phase 6: Model Ablations** | RQ1, RQ4 | Model D variants | - Without Velocity<br>- Without Goal Vectors<br>- Single vs. 2-layer GRU | Ablation Performance Delta Table |

---

### 2.1 Baseline Ordering Gate (M2 smoke test)
The `random < distance < Bayesian` ordering is checked on **Accuracy and
Macro-F1** at $\alpha \in \{0.30, 0.50, 0.75\}$. $\alpha = 1.00$ is excluded from the
gate because in `GOAL_REACHED` episodes the focal agent stands on its goal, so a
distance heuristic reads the label off the terminal position without inference;
$\alpha \in \{0.10, 0.20\}$ is reported but not gated because a Markovian
nearest-target heuristic has no signal before the agent commits to a direction.
All fractions are reported unchanged in `verification/m2/baselines.md`.

---

## 3. Reporting and Statistical Standards
- **Multi-Seed Execution**: All experiments are conducted across 5 fixed random seeds (`[42, 43, 44, 45, 46]`).
- **Reporting Format**: All metrics reported as $\text{Mean} \pm \text{Std}$ across the 5 seeds, with 95% bootstrap confidence intervals computed via scenario-level resampling (episodes generated from the same scenario are correlated and must not be resampled blindly at the episode level).
- **Significance Testing**: No Wilcoxon test over 5 seeds is used as the primary significance analysis. Statistical claims require the scenario-level bootstrap CIs; paired tests may only be reported as secondary diagnostics and never cited as justification for significance on their own.
- **Logging**: Results saved automatically to `results/` in both JSON format and publication-ready LaTeX tables (`results/tables/*.tex`) and vector PDF/PNG figures (`results/figures/*.png`).
