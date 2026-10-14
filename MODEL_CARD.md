# Model Cards and Inference Specification

## 1. Overview
This document specifies the exact architectures, probabilistic formulations, input/output contracts, and assumptions for all models evaluated in the project.

---

## 2. Model Taxonomy

**Implementation status (M2)**: Baselines A, B and C are implemented in
`src/evaluation/baselines.py` and are scored by the Section 7 protocol on
$P(G \mid O_{1:t})$. Baseline C is implemented as the *recursive joint belief*
over $(G, I_t)$ of PROJECT_SPEC §2.3 with the generative action model as its
emission; the cost-difference likelihood $\Delta(O_{1:t}, g)$ described below is
specified but not yet implemented (M3). Model D and Model E are M3 deliverables.

### Baseline A: Uniform Random Baseline (`UniformBaseline`)
- **Description**: Assigns equal probability to all candidate goals regardless of observed trajectory history. Probabilities carry a seeded infinitesimal jitter so that top-1 accuracy is a fair random guess (expected $1/|\mathcal{G}|$) instead of a constant class.
- **Formulation**:
  $$P(G = g \mid O_{1:t}) = \frac{1}{|\mathcal{G}|}$$
- **Role**: Establishes the theoretical lower bound for accuracy and entropy baseline for Negative Log-Likelihood ($-\log(1/|\mathcal{G}|)$).

---

### Baseline B: Distance-Based Heuristic (`DistanceHeuristicBaseline`)
- **Description**: Computes the shortest path distance (Manhattan or obstacle-aware BFS) from the current observed agent position $p_t$ to each candidate goal $g \in \mathcal{G}$, then applies a softmax over negative distances. `distance_metric` and `inverse_temperature` come from `configs/baselines.yaml` (defaults: `shortest_path`, $\beta = 0.5$).
- **Formulation**:
  $$P(G = g \mid O_{1:t}) = \frac{\exp(-\beta \cdot d(p_t, g))}{\sum_{g' \in \mathcal{G}} \exp(-\beta \cdot d(p_t, g'))}$$
  where $\beta > 0$ is an inverse temperature parameter tuned on the validation set.
- **Assumptions**: Markovian; ignores observation history before $t$; does not account for multi-agent interactions or obstacle detours unless $d(\cdot)$ is shortest path through obstacles.

---

### Baseline C: Probabilistic Bayesian Goal Recognition (`BayesianBeliefBaseline`)
- **Implemented emission**: the recursive belief update
  $$b_t(G, I_t) \propto P(A_t \mid I_t, G, s_{t-1}) \sum_{I_{t-1}} P(I_t \mid I_{t-1})\, b_{t-1}(G, I_{t-1})$$
  where $P(A_t \mid I_t, G, s_{t-1})$ is the *same* action model that generated the
  data (`src.agents.policy.action_distribution`) and $P(I_t \mid I_{t-1})$ is a
  persistence transition with parameter `intention_persist` (0.85). When the
  `blocked` flag is observable (Condition A) it is used exactly: a blocked step
  contributes the mass of "intended a move"; when it is masked (Condition B) a
  `block_rate` mixture is used instead. Because the likelihood is the generative
  model, the filter is well-specified — its intention NLL sits well below the
  empirical prior (Section 7 report), which was not true of the earlier
  hand-tuned emission.
- **Specified alternative (not yet implemented)**: the cost-difference
  formulation below, grounded in the plan-recognition-as-planning framework (Ramírez & Geffner 2009, 2010; Baker et al. 2009). The likelihood of an observed sequence of actions given goal $g$ is modeled by comparing the optimal cost of reaching $g$ from the initial state vs. the optimal cost of reaching $g$ constrained to pass through the observed trajectory.
- **Formulation**:
  Let $C^*(s, g)$ be the optimal cost from state $s$ to goal $g$.
  The cost difference is:
  $$\Delta(O_{1:t}, g) = \text{Cost}(O_{1:t}) + C^*(p_t, g) - C^*(p_0, g)$$
  Assuming Boltzmann rational agent behavior:
  $$P(O_{1:t} \mid G = g) \propto \exp(-\beta \cdot \Delta(O_{1:t}, g))$$
  With prior $P(G = g)$, the posterior is:
  $$P(G = g \mid O_{1:t}) = \frac{P(G = g) \exp(-\beta \Delta(O_{1:t}, g))}{\sum_{g'} P(G = g') \exp(-\beta \Delta(O_{1:t}, g'))}$$
- **Multi-Agent Extension**: Incorporates interaction penalties if candidate path intersects other agents' predicted paths.

---

### Model D: Recurrent Neural Sequence Model (GRU / LSTM)
- **Architecture**:
  1. **Input Representation**:
     - *Self-Only Variant*: Feature vector $x_t \in \mathbb{R}^{d_{\text{self}}}$ containing focal position $(x, y)$, normalized coordinates, velocity $(\Delta x, \Delta y)$, action one-hot, and relative vector to each candidate goal $(g_k - p_t)$.
     - *Multi-Agent Variant*: Concatenated or attention-pooled feature vector $x_t \in \mathbb{R}^{d_{\text{multi}}}$ containing $x_t^{\text{self}}$ plus relative displacement vectors to all other agents $(p_{j, t} - p_{0, t})$, other agent velocities, and inter-agent distances.
  2. **Recurrent Encoder**:
     $$h_t = \text{GRU}(x_t, h_{t-1}), \quad h_t \in \mathbb{R}^{H}$$
  3. **Prediction Heads**:
     - **Goal Recognition Head**: Linear layer + Softmax over $|\mathcal{G}|$ goals:
       $$\hat{P}(G \mid O_{1:t}) = \text{Softmax}(W_G h_t + b_G)$$
     - **Intention Inference Head**: Conditioned on representation $h_t$, goal embedding $e_G$, and previous intention embedding $e_{I_{t-1}}$:
       $$\hat{P}(I_t \mid O_{1:t}, G, I_{t-1}) = \text{Softmax}(W_I [h_t; e_G; e_{I_{t-1}}] + b_I)$$
     - **Future Action Prediction Head** (hierarchical, conditioned on goal and intention):
       $$\hat{P}(A_{t+1} \mid O_{1:t}, G, I_t) = \text{Softmax}(W_A [h_t; e_G; e_I] + b_A)$$
       A marginal variant $\hat{P}(A_{t+1} \mid O_{1:t}) = \text{Softmax}(W_A h_t + b_A)$ is retained only for the unconditioned benchmark arm.

---

### Model E: Relational Transformer (Optional / Advanced)
- **Architecture**: Temporal multi-head self-attention over sequence time steps $1:t$, combined with pairwise spatial relational attention across agents.
- **Role**: Validates whether global temporal attention yields significant gain over recurrent inductive biases under long horizons.

---

## 3. Loss Functions and Training Protocol
- **Goal Recognition Loss**: Multiclass Cross-Entropy $\mathcal{L}_G = -\log P(G_{\text{true}} \mid O_{1:t})$
- **Intention Recognition Loss**: Multiclass Cross-Entropy $\mathcal{L}_I = -\log P(I_{\text{true}} \mid O_{1:t}, G_{\text{true}})$
- **Action Prediction Loss**: Multiclass Cross-Entropy $\mathcal{L}_A = -\log P(A_{t+1} \mid O_{1:t})$
- **Joint Objective**:
  $$\mathcal{L} = \mathcal{L}_G + \lambda_I \mathcal{L}_I + \lambda_A \mathcal{L}_A$$
- **Optimizer**: AdamW, learning rate $1\times 10^{-3}$, weight decay $1\times 10^{-4}$, cosine annealing schedule.
- **Early Stopping**: Monitored on validation Goal Recognition Macro-F1 with patience = 10 epochs.
- **Execution**: All reported results must use the documented CPU configuration. MPS/GPU may be used during development only and is never a substitute for reported numbers.
- **Bayesian belief formulation**: Baseline C maintains a recursive joint belief over $(G, I_t)$ per PROJECT_SPEC §2.3 rather than treating each timestep as an independent classification problem.
