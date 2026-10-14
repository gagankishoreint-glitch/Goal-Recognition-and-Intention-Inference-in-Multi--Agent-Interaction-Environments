# Research Questions and Experimental Hypotheses

## RQ1: Goal Inference from Incomplete Behavioral Trajectories
- **Core Question**: How accurately and reliably can an agent's hidden terminal goal $G$ be inferred from partial behavioral observations $O_{1:t}$ compared to full-trajectory observations?
- **Hypothesis $H_1$**: Sequence-based neural models (GRU) and cost-difference Bayesian models will achieve significantly higher accuracy ($\ge 20\%$ margin) and lower cross-entropy loss than distance-based heuristics, particularly when obstacles induce non-linear or ambiguous paths.
- **Protocol**:
  - Evaluate models on fixed test trajectories truncated at observation fractions $\alpha = t/T \in \{0.10, 0.20, 0.30, 0.50, 0.75, 1.00\}$.
  - Maintain identical train/val/test splits across all model variants.
- **Independent Variables**: Observation fraction $\alpha$, Model architecture (Uniform, Heuristic, Bayesian, GRU).
- **Dependent Variables**: Accuracy, Macro-F1, Negative Log-Likelihood (NLL), Top-2 Accuracy, Brier Score.
- **Statistical Reporting**: All dependent variables are reported as mean $\pm$ standard deviation across 5 independent training seeds, with 95% bootstrap confidence intervals using **scenario-level resampling** (episodes from the same scenario are correlated). A paired Wilcoxon signed-rank test over 5 seeds is not used as the primary significance analysis; it may only be reported as a secondary, clearly-labeled diagnostic.

---

## RQ2: Value of Multi-Agent Behavioral Context
- **Core Question**: Does access to observable peer-agent behavior improve goal recognition for the focal agent relative to interaction-masked observation, under identical underlying episodes?
- **Hypothesis $H_2$**: In scenarios where focal detours are caused by collision avoidance or blocking (`DETOUR_LEFT`, `DETOUR_RIGHT`), Condition A will outperform Condition B by disambiguating interaction-caused detours from genuine changes in target heading.
- **Protocol**:
  - The primary comparison uses the **same underlying episodes** under two observation conditions:
    1. **Condition A — Full Interaction Observation**: $P(G_A \mid A_A, A_{-A}, E)$ — the recognizer receives focal behavior plus observable peer behavior.
    2. **Condition B — Interaction-Masked Observation**: $P(G_A \mid A_A, E)$ — the same episode with peer behavioral information masked.
  - The environment and trajectory are unchanged between conditions; only the interaction-information mask differs. We do NOT compare separately generated "interactive" vs. "non-interactive" environments, which would change the data-generating process.
  - Evaluate across agent density conditions: $N=1$ (control), $N=2$, and $N=4$.
  - Note: the data-generating policy may differ across $N$ conditions; the head-to-head Condition A vs. Condition B comparison at fixed $N$ is the primary claim.
- **Independent Variables**: Interaction-information mask (Condition A vs. B), Agent density $N$.
- **Dependent Variables**: Differential Accuracy ($\Delta \text{Acc} = \text{Acc}_A - \text{Acc}_B$), Macro-F1 on interaction-heavy trajectories.
- **Statistical Reporting**: mean $\pm$ std across 5 seeds; 95% scenario-level bootstrap CIs.

---

## RQ3: Early Goal Recognition Latency
- **Core Question**: How early along a trajectory can a goal be recognized with high confidence before the agent's path becomes trivial?
- **Hypothesis $H_3$**: Bayesian and GRU models will achieve the "early decision convergence threshold" (defined as continuous correct prediction with confidence $P(G^* \mid O_{1:t}) \ge 0.70$) within the first $35\%$ of trajectory length, whereas distance heuristics will require $>60\%$ of trajectory completion.
- **Protocol**:
  - Measure the earliest time step $t^*$ such that for all $t' \ge t^*$, $\arg\max_g P(G = g \mid O_{1:t'}) = G^*$ and $P(G^* \mid O_{1:t'}) \ge \tau_{\text{conf}}$ (for $\tau_{\text{conf}} \in \{0.5, 0.7, 0.9\}$).
  - Compute Normalized Time-to-Correct-Decision: $\text{NTCD} = t^* / T$, where $t^*$ is the first timestep at which the predicted goal is correct and remains correct for the rest of the episode with confidence $P(G^* \mid O_{1:t}) \ge \tau_{\text{conf}}$ (stability requirement). If no such $t^*$ exists, $\text{NTCD} = 1$. Aggregation: mean over episodes, reported per confidence threshold with scenario-level bootstrap CIs.
- **Independent Variables**: Model architecture, Ambiguity ratio of initial scenario layout.
- **Dependent Variables**: Mean NTCD, Convergence Failure Rate (fraction of episodes where convergence is not reached before $T$), Area under Early Accuracy Curve (AEAC).

---

## RQ4: Hierarchical Intention Inference and Action Prediction
- **Core Question**: Can the inference system accurately identify the agent's intermediate intention $I$ and predict its immediate next action $A_{t+1}$?
- **Hypothesis $H_4$**: Conditioning next-action prediction on the inferred goal and intention distribution ($P(A_{t+1} \mid O_{1:t}, \hat{G}, \hat{I})$) yields higher action prediction accuracy than an unconditioned marginal action predictor ($P(A_{t+1} \mid O_{1:t})$).
- **Protocol**:
  - Train multi-task GRU sequence model with joint heads for Goal $G$, Intention $I$, and Action $A_{t+1}$.
  - Evaluate intention classification performance across all intention classes (Macro-F1, Confusion Matrix).
  - Compare next-action accuracy with and without goal/intention conditioning.
- **Independent Variables**: Conditioning structure (unconditioned vs. goal-conditioned vs. joint goal+intention conditioned).
- **Dependent Variables**: Intention Macro-F1, Intention Accuracy, Action Top-1 Accuracy, Action Top-2 Accuracy.

---

## RQ5: Robustness to Observational Noise & Missing Data
- **Core Question**: How gracefully does recognition performance degrade under noisy coordinate readings and intermittent sensor packet drops?
- **Hypothesis $H_5$**: Sequence models with temporal memory and Bayesian cost-difference models will degrade significantly more gracefully under Gaussian coordinate jitter and missing steps than Markovian heuristics, which suffer immediate degradation upon corrupted single-step observations.
- **Protocol**:
  - Test pre-trained models on corrupted test sets across systematic perturbation levels:
    - Coordinate Noise: Gaussian $\sigma \in \{0.0, 0.2, 0.5, 1.0, 1.5\}$.
    - Missing Steps: Uniform dropout probability $p_{\text{missing}} \in \{0.0, 0.1, 0.25, 0.50\}$.
  - Evaluate zero-shot robustness (trained on clean data) and noise-augmented robustness (trained with noise).
- **Independent Variables**: Noise standard deviation $\sigma$, Dropout rate $p_{\text{missing}}$, Training augmentation regime.
- **Dependent Variables**: Degradation slope $d(\text{Acc})/d\sigma$, NLL, Calibration Error (ECE).
