# Baselines Results (Section 7)

## Task and Protocol

Goal recognition `P(G | O_{1:t})` over the K candidate targets, evaluated on the held-out **test scenarios** of the shared M2 verification dataset. Observation fractions follow EXPERIMENT_PLAN.md §1.2; confidence intervals are 95% scenario-cluster bootstraps.

- **train**: 336 episodes / 42 scenarios, goal counts {'0': 83, '1': 80, '2': 72, '3': 101}
- **val**: 72 episodes / 9 scenarios, goal counts {'0': 17, '1': 23, '2': 14, '3': 18}
- **test**: 72 episodes / 9 scenarios, goal counts {'0': 18, '1': 25, '2': 11, '3': 18}

## Goal Recognition by Observation Fraction

| alpha | baseline | Accuracy | Macro-F1 | NLL | Brier | Acc 95% CI |
|---|---|---|---|---|---|---|
| 0.10 | random | 0.3194 | 0.3160 | 1.3863 | 0.7500 | [0.2361, 0.4167] |
| 0.10 | distance | 0.2361 | 0.2208 | 4.0617 | 1.2629 | [0.1250, 0.3611] |
| 0.10 | bayesian | 0.4583 | 0.4472 | 1.1858 | 0.6456 | [0.3333, 0.5833] |
| 0.20 | random | 0.3611 | 0.3591 | 1.3863 | 0.7500 | [0.2500, 0.4583] |
| 0.20 | distance | 0.2500 | 0.2320 | 3.7647 | 1.2402 | [0.1250, 0.3611] |
| 0.20 | bayesian | 0.4444 | 0.4341 | 1.0892 | 0.6067 | [0.3056, 0.5833] |
| 0.30 | random | 0.2361 | 0.2283 | 1.3863 | 0.7500 | [0.1528, 0.3056] |
| 0.30 | distance | 0.2639 | 0.2500 | 3.6598 | 1.2336 | [0.1250, 0.4028] |
| 0.30 | bayesian | 0.4861 | 0.4743 | 1.0633 | 0.6068 | [0.3611, 0.6111] |
| 0.50 | random | 0.2222 | 0.1943 | 1.3863 | 0.7500 | [0.1389, 0.3056] |
| 0.50 | distance | 0.3472 | 0.3256 | 2.9042 | 1.0730 | [0.2361, 0.4583] |
| 0.50 | bayesian | 0.6667 | 0.6725 | 0.7132 | 0.4109 | [0.4722, 0.8333] |
| 0.75 | random | 0.2778 | 0.2792 | 1.3863 | 0.7500 | [0.1944, 0.3889] |
| 0.75 | distance | 0.4861 | 0.4751 | 2.5538 | 0.8757 | [0.3750, 0.5972] |
| 0.75 | bayesian | 0.7778 | 0.7722 | 0.5064 | 0.2772 | [0.6389, 0.9028] |
| 1.00 | random | 0.1806 | 0.1788 | 1.3863 | 0.7500 | [0.0694, 0.2917] |
| 1.00 | distance | 0.5417 | 0.5274 | 2.0087 | 0.7211 | [0.4306, 0.6667] |
| 1.00 | bayesian | 0.8889 | 0.8923 | 0.3754 | 0.1895 | [0.8056, 0.9583] |

## Ordering Smoke Test (random < distance < Bayesian)

- Primary fractions ['0.30', '0.50', '0.75'] (Accuracy and Macro-F1 ordering): **PASS**

| alpha | Acc random | Acc distance | Acc bayesian | NLL random | NLL distance | NLL bayesian | acc-ordered | nll-ordered |
|---|---|---|---|---|---|---|---|---|
| 0.10 (diagnostic) | 0.3194 | 0.2361 | 0.4583 | 1.3863 | 4.0617 | 1.1858 | no | no |
| 0.20 (diagnostic) | 0.3611 | 0.2500 | 0.4444 | 1.3863 | 3.7647 | 1.0892 | no | no |
| 0.30 (primary) | 0.2361 | 0.2639 | 0.4861 | 1.3863 | 3.6598 | 1.0633 | yes | no |
| 0.50 (primary) | 0.2222 | 0.3472 | 0.6667 | 1.3863 | 2.9042 | 0.7132 | yes | no |
| 0.75 (primary) | 0.2778 | 0.4861 | 0.7778 | 1.3863 | 2.5538 | 0.5064 | yes | no |
| 1.00 (degenerate) | 0.1806 | 0.5417 | 0.8889 | 1.3863 | 2.0087 | 0.3754 | yes | no |

Gate criteria: the smoke test in M2_VERIFICATION_GUIDE.md §7 orders the baselines by discriminative performance (Accuracy, Macro-F1) on the primary fractions ['0.30', '0.50', '0.75'].

Not gated, with reasons: alpha 0.10/0.20 — before the focal agent commits to a direction the nearest-target heuristic carries no signal and can sit at chance; alpha 1.00 — structurally degenerate, see the diagnostic below.

Chance reference: with K = 4 targets and n = 72 test episodes, a fair uniform guess scores 0.2500 in expectation with a binomial SE of 0.0510. The majority class of this test split scores 0.3472 (goal counts [18, 25, 11, 18]).

The random baseline re-draws its guess at every prefix, so its per-alpha accuracy is a single binomial draw, not a constant. Over 50 seeds its mean is:

| alpha | random baseline mean acc | sd across seeds |
|---|---|---|
| 0.10 | 0.2569 | 0.0544 |
| 0.20 | 0.2531 | 0.0503 |
| 0.30 | 0.2514 | 0.0460 |
| 0.50 | 0.2506 | 0.0476 |
| 0.75 | 0.2508 | 0.0477 |
| 1.00 | 0.2367 | 0.0511 |

- alpha = 1.00 degeneracy: 28/72 test episodes (38.9%) end in GOAL_REACHED, i.e. the focal agent is standing on its goal. On those episodes the distance heuristic scores 1.0000 by construction; on the remaining episodes it scores 0.2500 (Bayesian 0.8182).

NLL/Brier diagnostic (reported, not gated): at short horizons the distance heuristic is *overconfident* rather than merely wrong — its NLL exceeds the uniform baseline's log(K) = 1.3863 and its Brier score exceeds 0.75. This is the expected failure mode of a sharply tempered Markovian heuristic and is one of the effects RQ1 predicts Bayesian and sequence models will avoid; it is a property of the baseline, not a wiring defect, so the numbers are reported unchanged.

## Bayesian Intention Marginal (vs. empirical training prior)

- steps scored: 2923
- Accuracy: 0.5618 (prior 0.3678)
- Macro-F1: 0.5931 (prior 0.1344)
- NLL: 0.9475 (prior NLL 1.3124)
- NLL below prior: **yes**

### Confusion matrix (rows = true, columns = predicted)

| true \ pred | DIRECT | DETOUR_LEFT | DETOUR_RIGHT | WAIT |
|---|---|---|---|---|
| DIRECT | 558 | 192 | 260 | 65 |
| DETOUR_LEFT | 184 | 350 | 137 | 28 |
| DETOUR_RIGHT | 246 | 113 | 484 | 27 |
| WAIT | 14 | 9 | 6 | 250 |

## NTCD (tau_conf = 0.7)

| baseline | NTCD | convergence failure rate |
|---|---|---|
| random | 1.0000 | 1.0000 |
| distance | 0.7245 | 0.4861 |
| bayesian | 0.7017 | 0.2778 |

## Gate Status

- RESULT: ordering on primary fractions satisfied
- GATE STATUS: GREEN (CONTINUE)

