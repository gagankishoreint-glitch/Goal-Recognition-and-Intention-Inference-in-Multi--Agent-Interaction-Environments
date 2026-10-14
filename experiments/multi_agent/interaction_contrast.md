# RQ2 Pre-Study: Interaction-Visible vs Interaction-Masked

Environment: N = 4 agents, `scenario_family = mixed`, peer spawn radius 2, policy sensing radius 3, peer avoidance weight 0.05. Both conditions run on the *same* episodes; only the interaction channels differ.

## 1. Is interaction measurable? (test split, focal agent)

| family | episodes | steps | peer-blocked | peer-influenced | greedy cell held by peer | mean TV |
|---|---|---|---|---|---|---|
| bottleneck | 16 | 756 | 10.45% | 16.40% | 4.23% | 0.131 |
| crossing | 24 | 1038 | 20.33% | 24.57% | 8.19% | 0.217 |
| open | 32 | 1129 | 6.47% | 16.47% | 4.78% | 0.121 |

- **peer-blocked** — the environment denied the focal move because a peer held or claimed the cell.
- **peer-influenced** — a peer inside the sensing radius changed the focal action distribution at all (total variation > 0).
- **greedy cell held by peer** — a cell that would reduce distance to the true goal was occupied by a peer, i.e. the agent was forced off its direct line. This is the mechanism a masked observer must mis-attribute.

## 2. Closed-form Bayesian recognizer (blocked flag visible vs masked)

| alpha | Acc A | Acc B | F1 A | F1 B | NLL A | NLL B | Acc gap (A-B) 95% CI |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.4861 | 0.4583 | 0.4743 | 0.4478 | 1.0633 | 1.0628 | +0.0278 [+0.0000, +0.0694] |
| 0.50 | 0.6667 | 0.6667 | 0.6725 | 0.6725 | 0.7132 | 0.7303 | +0.0000 [+0.0000, +0.0000] |
| 0.75 | 0.7778 | 0.7778 | 0.7722 | 0.7722 | 0.5064 | 0.5197 | +0.0000 [+0.0000, +0.0000] |
| 1.00 | 0.8889 | 0.8750 | 0.8923 | 0.8747 | 0.3754 | 0.3676 | +0.0139 [+0.0000, +0.0417] |

The Bayesian baseline only consumes the `blocked` flag, so this arm measures the value of the flag alone. It is reported as-is, including where the gap is not distinguishable from zero.

## 3. Linear probe over the timing-contract features (window 5)

Five arms, identical feature dimensionality (58 channels per frame) and identical model capacity; they differ only in which information is present:

| arm | peer cells | interaction features | previous blocked flag |
|---|---|---|---|
| A_full | real | real | real |
| B_zeroed | zeroed | zeroed | dropped |
| C_shuffled_peers | randomly relocated | recomputed | real |
| D_noise | Gaussian noise | Gaussian noise | real |
| E_blocked_only | zeroed | zeroed | real |

| task | arm | accuracy | macro-F1 |
|---|---|---|---|
| goal | A_full | 0.3243 | 0.3060 |
| goal | B_zeroed | 0.3445 | 0.3185 |
| goal | C_shuffled_peers | 0.3346 | 0.3078 |
| goal | D_noise | 0.3390 | 0.3138 |
| goal | E_blocked_only | 0.3421 | 0.3169 |
| intention | A_full | 0.4619 | 0.4462 |
| intention | B_zeroed | 0.4003 | 0.3207 |
| intention | C_shuffled_peers | 0.4680 | 0.4411 |
| intention | D_noise | 0.4591 | 0.4373 |
| intention | E_blocked_only | 0.4646 | 0.4357 |

### Paired gaps (scenario-cluster bootstrap, 2000 resamples)

| task | contrast | accuracy | macro-F1 | reading |
|---|---|---|---|---|
| goal | A_minus_B | -0.0202 [-0.0567, +0.0172] | -0.0125 [-0.0439, +0.0177] | naive contrast — confounds peers with the blocked flag |
| goal | A_minus_D | -0.0147 [-0.0494, +0.0201] | -0.0078 [-0.0374, +0.0180] | **primary**: real peer placement vs Gaussian noise |
| goal | A_minus_E | -0.0178 [-0.0588, +0.0238] | -0.0109 [-0.0472, +0.0223] | real peer placement vs zeroed peer channels |
| goal | E_minus_B | -0.0024 [-0.0108, +0.0055] | -0.0016 [-0.0086, +0.0077] | value of the previous-step blocked flag alone |
| goal | A_minus_C | -0.0103 [-0.0449, +0.0223] | -0.0018 [-0.0358, +0.0242] | real peers vs relocated peers |
| goal | D_minus_B | -0.0055 [-0.0130, +0.0000] | -0.0047 [-0.0124, +0.0047] | artifact size: zeroing channels + dropping the flag |
| intention | A_minus_B | +0.0616 [+0.0203, +0.0995] | +0.1255 [+0.0691, +0.1730] | naive contrast — confounds peers with the blocked flag |
| intention | A_minus_D | +0.0027 [-0.0101, +0.0142] | +0.0089 [-0.0124, +0.0255] | **primary**: real peer placement vs Gaussian noise |
| intention | A_minus_E | -0.0027 [-0.0127, +0.0062] | +0.0105 [-0.0041, +0.0222] | real peer placement vs zeroed peer channels |
| intention | E_minus_B | +0.0643 [+0.0286, +0.0979] | +0.1150 [+0.0685, +0.1562] | value of the previous-step blocked flag alone |
| intention | A_minus_C | -0.0062 [-0.0224, +0.0079] | +0.0050 [-0.0149, +0.0199] | real peers vs relocated peers |
| intention | D_minus_B | +0.0588 [+0.0254, +0.0890] | +0.1166 [+0.0780, +0.1536] | artifact size: zeroing channels + dropping the flag |

A gap counts as supported only when its 95% CI excludes zero.

## 4. What these numbers say

**goal**

- naive A-B accuracy -0.0202 (not significant); valid A-D -0.0147 (not significant); peer placement over the flag A-E -0.0178 (not significant); blocked flag alone E-B -0.0024 (not significant). The same contrasts on macro-F1: A-B -0.0125, A-D -0.0078, E-B -0.0016.
  - No supported effect of interaction information for this task.

**intention**

- naive A-B accuracy +0.0616 (significant); valid A-D +0.0027 (not significant); peer placement over the flag A-E -0.0027 (not significant); blocked flag alone E-B +0.0643 (significant). The same contrasts on macro-F1: A-B +0.1255, A-D +0.0089, E-B +0.1150.
  - The naive contrast is **not** attributable to peer placement: replacing the peer channels with Gaussian noise reproduces it. The whole of it is carried by the previous-step `blocked` flag (E-B), a one-bit lagging trace of the collision the environment recorded.

Methodological consequence, carried into M3: the interaction-masked arm must be constructed by replacing the masked channels with noise (arm D), never by zeroing them (arm B). Zeroing changes what the regularizer sees and manufactures a gap that has nothing to do with interaction — here worth up to +0.0588 accuracy on its own. The same protocol applies to the GRU.

