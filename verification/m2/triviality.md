# Intention Triviality Probe Results (Section 6)

## Protocol

Shallow probes predicting `I_t` from the observation available *before* the step: `position_before_t`, peer cells at the same instant, per-target relative geometry and local map occupancy. `A_t` (which `I_t` generates) is never an input. Trained on training scenarios only, evaluated on held-out test scenarios, Condition A observable fields, fixed probe seed 42, one model family (multinomial logistic regression).

- P0: majority class, no features
- P1: current-step geometry only
- P2: P1 + realized_action_{t-1} + blocked_{t-1}
- P3: P2 stacked over a window of k=5 steps (shallow history reference)

Held-out test set: 72 episodes / 9 scenarios.

## Probe Performance

| probe | features | Accuracy | Acc 95% CI | Macro-F1 | Macro-F1 95% CI | samples |
|---|---|---|---|---|---|---|
| P0 | 0 | 0.3678 | n/a (constant) | 0.1344 | n/a (constant) | 2923 |
| P1 | 51 | 0.3582 | [0.3011, 0.4152] | 0.2028 | [0.1625, 0.2366] | 2923 |
| P2 | 58 | 0.4157 | [0.3610, 0.4684] | 0.3929 | [0.3517, 0.4236] | 2923 |
| P3 | 290 | 0.4711 | [0.4314, 0.5023] | 0.4512 | [0.4114, 0.4806] | 2923 |

## Per-Intention Precision / Recall / F1

| probe | intention | precision | recall | F1 | support |
|---|---|---|---|---|---|
| P0 | DIRECT | 0.3678 | 1.0000 | 0.5378 | 1075 |
| P0 | DETOUR_LEFT | 0.0000 | 0.0000 | 0.0000 | 699 |
| P0 | DETOUR_RIGHT | 0.0000 | 0.0000 | 0.0000 | 870 |
| P0 | WAIT | 0.0000 | 0.0000 | 0.0000 | 279 |
| P1 | DIRECT | 0.3654 | 0.8298 | 0.5074 | 1075 |
| P1 | DETOUR_LEFT | 0.2489 | 0.0801 | 0.1212 | 699 |
| P1 | DETOUR_RIGHT | 0.3984 | 0.1126 | 0.1756 | 870 |
| P1 | WAIT | 0.0909 | 0.0036 | 0.0069 | 279 |
| P2 | DIRECT | 0.4012 | 0.7479 | 0.5222 | 1075 |
| P2 | DETOUR_LEFT | 0.3055 | 0.1359 | 0.1881 | 699 |
| P2 | DETOUR_RIGHT | 0.4238 | 0.1471 | 0.2184 | 870 |
| P2 | WAIT | 0.6144 | 0.6738 | 0.6427 | 279 |
| P3 | DIRECT | 0.4736 | 0.7674 | 0.5857 | 1075 |
| P3 | DETOUR_LEFT | 0.3636 | 0.1831 | 0.2436 | 699 |
| P3 | DETOUR_RIGHT | 0.4536 | 0.2920 | 0.3552 | 870 |
| P3 | WAIT | 0.6320 | 0.6093 | 0.6204 | 279 |

## P3 - P1 Gaps (scenario-cluster bootstrap)

- accuracy: point estimate 0.1129, 95% CI [0.0693, 0.1577] (2000 resamples over 9 test scenarios)
- macro_f1: point estimate 0.2485, 95% CI [0.2001, 0.2957] (2000 resamples over 9 test scenarios)
- point estimate inside its own CI: yes (a sanity property of the cluster bootstrap)

## Triviality Gate

- Criterion: 95% scenario-cluster CI of the P3-P1 macro-F1 gap contains zero
- CI contains zero: NO
- Verdict: intention labels are **NON-TRIVIAL (CONTINUE)**
- GATE STATUS: GREEN (CONTINUE)

## Flags (reported, not silently modified)

- train: max intention share 43.16% (ok), median switches/episode 6.00 (ok)
- test: max intention share 36.78% (ok), median switches/episode 6.50 (ok)

