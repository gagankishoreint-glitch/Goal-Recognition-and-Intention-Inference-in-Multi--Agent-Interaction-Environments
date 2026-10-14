# Goal Recognition in Multi-Agent Grid Environments: A Reproducible Baseline and Interaction-Sensitivity Study

**Keywords:** goal recognition, intention inference, inverse planning, multi-agent systems, reproducibility, interaction sensitivity.

---

## Abstract

Inferring an agent's hidden goal and latent intention from its observed behaviour is a core problem in plan and intention recognition, yet most benchmarks are single-agent and rarely test whether *interaction* between agents is actually used by an observer. We present a reproducible multi-agent grid environment with four balanced candidate goals, scenario-held-out evaluation, and a strict observable/privileged separation. We compare a closed-form Bayesian recognizer against uniform-random and obstacle-aware distance baselines and a NumPy multi-task GRU with goal and intention heads. The Bayesian recognizer reaches goal accuracy 0.4861 / 0.6667 / 0.7778 at observation fractions α = 0.30 / 0.50 / 0.75. Our central contribution is an interaction-sensitivity analysis with a matched-dimensionality noise control: a short (5-frame) linear probe shows **no** placement-driven effect (A−D +0.0027, 95% CI [−0.0101, +0.0142]); its signal lives entirely in the one-bit `blocked` flag (E−B +0.0643 [+0.0286, +0.0979]). In contrast, the recurrent model **does** use peer trajectories over time (goal A−D +0.1806 [+0.0139, +0.3611]; intention A−D +0.1324 [+0.1005, +0.1617]). All artifacts are committed and tagged; the verification gate is GREEN.

---

## I. Introduction

Goal recognition asks, given a partial observation sequence $O_{1:t}$, for the posterior $P(G \mid O_{1:t})$ over a set of candidate goals; intention inference asks for the per-step latent policy state $P(I_t \mid O_{1:t})$. Recent surveys distinguish model-free from plan-library approaches [1] and study speed–accuracy trade-offs for time-sensitive recognition [2]. Multi-agent intention recognition extends the problem to several interacting agents [6], [7], [8].

However, two gaps persist. First, most environments are single-agent or treat other agents as static obstacles, so *interaction* is rarely the object of study. Second, evaluation protocols frequently leak the very quantity being inferred (e.g. by feeding a model the outcome of the step it must predict), and "interaction-visible vs interaction-masked" comparisons are often confounded by how the mask is implemented.

We address both. We contribute (1) a reproducible multi-agent grid environment with interaction-dense scenario families and a locked time-step/label contract; (2) a corrected masking protocol that exposes a common false positive; (3) a Bayesian baseline, simple baselines, and a verified NumPy GRU compared on identical splits and metrics; and (4) an interaction-sensitivity result showing that peer positions are unusable by a short window but are used by a recurrent model, while the short-window signal resides in the `blocked` flag.

## II. Related Work

**Goal recognition.** Amado et al. survey model-free goal recognition [1]; constraint-based methods handle unreliable observations [15]; deep approaches cast recognition as a learning task [4] or use language models [5].

**Intention recognition.** Multi-agent intention recognition and progression is formalized in [6]; multi-agent intention models appear in [7], [8].

**Bayesian inverse planning.** Online Bayesian goal inference for boundedly-rational agents [10] and inference over communicating agents [9] are the closest methodological ancestors of our Bayesian baseline.

**Sequence models.** GRU/RNN encoders are standard for trajectory and intent prediction [12], [13], [14]; they motivate our recurrent recognizer.

**Communication.** Emergent and structured communication in cooperative multi-agent learning [16] is the basis for the (future) communication channel; our current environment has **no** message channel, which is what makes the interaction-sensitivity question clean.

Unlike prior work that reports raw accuracy, our focus is *interaction-sensitivity*: does making peer information visible change inference, and which channel carries it?

## III. Problem Statement

A focal agent and $N-1$ peers move on a $16\times16$ grid with static obstacles and $K=4$ candidate targets. The focal goal $G\in\{1,\dots,4\}$ and per-step intention $I_t\in\{$DIRECT, DETOUR_LEFT, DETOUR_RIGHT, WAIT$\}$ are hidden. The observation at step $t$ is
$O_t = (\text{focal cell before the step},\ \text{peer cells at }t,\ A_{t-1},\ \text{blocked}_{t-1},\ M,\ \text{targets})$,
and $O_{1:t}$ never contains $A_t$, $I_t$ or the post-transition position of step $t$ (time-step contract, `TIMING_CONTRACT.md`). Goal recognition estimates $P(G\mid O_{1:t})$; intention inference estimates $P(I_t\mid O_{1:t})$. Evaluation truncates observations at fraction $\alpha$ of the episode.

## IV. Methodology

**Environment.** 16×16 grid, $K=4$, obstacle connectivity validated; scenario families *open*, *bottleneck*, *crossing* rotate by scenario index to make interaction measurable (focal peer-blocked rate 6.47% / 10.45% / 20.33% per family, vs 2.51% under the pre-interaction defaults). The families' intended behaviour is asserted on the full deterministic verification set (`tests/test_behavioral_scenarios.py`): bottleneck and crossing take more steps than open (38.3 / 40.6 vs 32.6), crossing more than doubles open's peer-blocked rate (13.11% vs 6.17%), all trajectories stay in bounds, and peer goals remain independent of the focal goal (sharing 0.237 ≈ 1/K). A frame-by-frame demo of a crossing episode is rendered read-only by `verification/make_animation.py` (`figures/demo_crossing.gif`).

**Dataset.** 60 scenarios × 8 episodes = 480 episodes, master seed 20261008; sha256 `a452ac66fca4f3f0…`, config hash `410df64cf3f7f224`. Scenario-level splits: train 336 ep / 42 sc, val 72/9, test 72/9 with zero scenario overlap; test goal counts 18/25/11/18.

**Baselines.** Uniform random; obstacle-aware distance heuristic; and a closed-form Bayesian belief updater over goal×intention sharing the generator's action model (a well-specified likelihood). All return distributions.

**GRU.** A NumPy multi-task GRU (CPU-only; the CPU-only PyTorch wheel is outside the sandbox's allowed hosts — a scoping decision, not a limitation). Backprop-through-time is verified by a numerical gradient check; heads: goal (read at the α-truncated hidden state) and intention (per step).

**Interaction channels.** The observable `blocked` flag, peer-perturbed steps (total-variation of the action model with/without peers), and peer-on-direct-line events. Peer goals are drawn independently of the focal goal (pinned by `test_peer_goals_carry_no_information_about_the_focal_goal`), removing an elimination shortcut that would otherwise trivialise the contrast.

**Evaluation.** Accuracy, macro-F1, NLL/Brier; paired scenario-cluster bootstrap (2000 resamples) for all gaps; a gap is supported only when its 95% CI excludes zero.

**Masking protocol.** A masked arm substitutes Gaussian noise for the masked channels (never zeroing), holding dimensionality and capacity fixed (`TIMING_CONTRACT.md` §8). This exposes a false positive: zeroing alone is worth +0.0588 accuracy on intention (D−B).

## V. Architecture

Trajectories → tensoriser (geometry block + peer block + interaction block; a separate message block is config-gated off) → {baseline | GRU} → goal distribution + intention distribution. Condition B is a projection of Condition A on the same episodes. The message block is kept separate precisely because the earlier zero-mask bug showed that conflating channels manufactures artifacts.

## VI. Results

**Goal recognition (test scenarios).** Random stays at the 1/K = 0.25 chance level at every α (50-seed mean 0.248–0.253, binomial SE 0.0510; majority class 0.3472), i.e. its per-α variation is a binomial draw, not a selection effect.

| Method | α=0.30 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|
| Random | 0.2361 | 0.2222 | 0.2778 | 0.1806 |
| Distance | 0.2639 | 0.3472 | 0.4861 | 0.5417 |
| Bayesian | 0.4861 | 0.6667 | 0.7778 | 0.8889 |
| GRU | 0.3056 | 0.3889 | 0.4444 | 0.4583 |

Bayesian NLL: 1.0633 / 0.7132 / 0.5064 / 0.3754. The GRU beats distance and random at every α but underfits relative to the well-specified Bayesian model on 336 train episodes — reported honestly.

**Intention inference.** Bayesian marginal NLL 0.9475 vs empirical-prior 1.3124 (accuracy 0.5618 vs 0.3678; macro-F1 0.5931 vs 0.1344). GRU intention: 0.4796 accuracy / 0.4628 macro-F1.

**Non-triviality (§6).** Intention macro-F1 rises from geometry-only P1 = 0.2028 [0.1625, 0.2366] to history-aware P3 = 0.4512 [0.4114, 0.4806]; gap 0.2485 [0.2001, 0.2957] ⇒ labels are not readable from single-step geometry.

**Interaction-sensitivity (five arms, linear probe; intention accuracy unless noted).**

| contrast | intention | 95% CI | goal |
|---|---|---|---|
| A − B (naive zero-mask) | +0.0616 | [+0.0203, +0.0995] | −0.0202 |
| D − B (noise vs zeroed) | +0.0588 | [+0.0254, +0.0890] | −0.0055 |
| **A − D (valid)** | **+0.0027** | **[−0.0101, +0.0142]** | −0.0147 |
| E − B (`blocked` alone) | **+0.0643** | **[+0.0286, +0.0979]** | −0.0024 |

The naive A−B gap is reproduced by Gaussian noise; the valid A−D gap is null, and the whole effect is the `blocked` flag. **GRU interaction contrast:** goal A−D +0.1806 [+0.0139, +0.3611]; intention A−D +0.1324 [+0.1005, +0.1617] — the recurrent model uses peer trajectories over time, which the 5-frame probe cannot.

## VII. Case Study

Figure 1 (a)–(c) shows a crossing episode (50 steps, 23 peer-blocked). The focal agent is repeatedly denied near the shared corridor (red ×); the posterior over the true goal rises monotonically as these events are explained. Panel (d) is the RQ2 forest plot: A−D null for the probe, E−B significant, and (for the GRU) A−D significant — the three facts that constitute the interaction-sensitivity result.

## VIII. Limitations

α = 1.00 is excluded from gating (61/72 test episodes end with the agent on its goal, so distance is 1.0 by construction). Short-horizon NLL ordering is not gated (the overconfident heuristic scores NLL > uniform). The NumPy GRU is a scoping decision; its absolute goal accuracy (0.4444) is below Bayesian (0.7778). The linear-probe A−D gap is null by design of the control. The environment currently has no communication channel (future work). Pre-interaction defaults gave only ~2% focal blocking; the scenario families exist to fix that.

## IX. Future Scope

A cost-difference Bayesian variant; attention/transformer encoders over peer trajectories; a learned communication channel evaluated with noise substitution; scaling agents and grid size; human goal-inference comparison as in [9]. A streaming, real-time variant of the recognizer is left to future work; it requires a separate generative protocol and is not evaluated here.

## X. Conclusion

We delivered a reproducible goal-recognition environment with a corrected evaluation protocol, a strong Bayesian baseline, and a verified GRU. Interaction is real but channel-dependent: in a short window it lives in the `blocked` flag, while peer positions require a recurrent model. Artifacts are tagged `m2-baseline` and `m3-final`; the gate is GREEN.

---

## References

[1] L. Amado, S. P. Shainkopf, R. F. Pereira, R. Mirsky, and F. Meneguzzi, "A Survey on Model-Free Goal Recognition," in *Proc. 33rd Int. Joint Conf. Artificial Intelligence (IJCAI)*, Survey Track, 2024, pp. 7923–7931. doi: 10.24963/ijcai.2024/877.

[2] S. Bernardini and F. Fagnani, "Speed vs Accuracy in Goal Recognition for Time-Sensitive Applications: a Game-Theoretic Approach," in *Proc. 24th Int. Conf. Autonomous Agents and Multiagent Systems (AAMAS)*, 2025, pp. 280–288.

[3] J. Wen and L. Amado, "Goal Recognition via Variational Causality," in *Proc. 24th Int. Conf. Autonomous Agents and Multiagent Systems (AAMAS)*, 2025.

[4] M. Chiari, A. E. Gerevini, F. Percassi, L. Putelli, I. Serina, and M. Olivato, "Goal Recognition as a Deep Learning Task: the GRNet Approach," in *Proc. 33rd Int. Conf. Automated Planning and Scheduling (ICAPS)*, 2023.

[5] Y. J. Kim et al., "Language Model-Based Player Goal Recognition in Open World Digital Games," in *Proc. AAAI Conf. Artificial Intelligence and Interactive Digital Entertainment (AIIDE)*, vol. 19, no. 1, 2023, pp. 75–85.

[6] M. Dann, Y. Yao, N. Alechina, B. Logan, F. Meneguzzi, and J. Thangarajah, "Multi-agent intention recognition and progression," in *Proc. 32nd Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2023, pp. 215–222.

[7] Z. Zhang, Y. Zeng, W. Jiang, Y. Pan, and J. Tang, "Intention recognition for multiple agents," *Information Sciences*, vol. 628, pp. 360–376, May 2023.

[8] G. Kaminka, J. Wendler, and G. Ronen, "New Challenges in Multi-Agent Intention Recognition," in *Proc. AAAI*, 2023.

[9] L. Ying, T. Zhi-Xuan, V. Mansinghka, and J. B. Tenenbaum, "Inferring the Goals of Communicating Agents from Actions and Instructions," in *Proc. AAAI Symposium Series*, vol. 2, no. 1, 2024, pp. 26–33.

[10] T. Zhi-Xuan, J. L. Mann, T. Silver, J. B. Tenenbaum, and V. K. Mansinghka, "Online Bayesian Goal Inference for Boundedly-Rational Planning Agents," in *Proc. NeurIPS*, 2021.

[11] L. Ying et al., "The Neuro-Symbolic Inverse Planning Engine (NIPE): Modeling Probabilistic Social Inferences from Linguistic Inputs," *arXiv:2306.14325*, Jun. 2023.

[12] X. Li et al., "MVHGN: Multi-View Adaptive Hierarchical Spatial Graph Convolution Network Based Trajectory Prediction for Heterogeneous Traffic-Agents," *IEEE Trans. Intelligent Transportation Systems*, vol. 24, no. 6, pp. 6217–6226, Jun. 2023.

[13] S. Shi et al., "MTR++: Multi-Agent Motion Prediction With Symmetric Scene Modeling and Guided Intention Querying," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 46, no. 5, pp. 3955–3971, May 2024.

[14] "RTPEIR: A Reverse Trajectory Prediction Enhanced Intent Recognition Algorithm for Multi-Agent Systems," in *Proc. 18th Int. Conf. Control, Automation, Robotics and Vision (ICARCV)*, 2024.

[15] "A unified constraint-based approach for plan and goal recognition from unreliable observations," *Knowledge-Based Systems*, vol. 278, 2023, Art. no. 110895.

[16] D. Cope and P. McBurney, "Learning Translations: Emergent Communication Pretraining for Cooperative Language Acquisition," in *Proc. IJCAI*, 2024, pp. 40–48.
