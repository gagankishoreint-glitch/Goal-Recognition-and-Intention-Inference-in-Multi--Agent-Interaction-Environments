# Baselines Results (Section 7)

## Baseline Performance Metrics

### Random Baseline
- Accuracy: 0.6447
- Macro-F1: 0.1960

### Distance Heuristic Baseline
- Accuracy: 0.6447
- Macro-F1: 0.1960

### Bayesian Belief Baseline
- Accuracy: 0.3816
- Macro-F1: 0.2304
- Confusion Matrix:
  - DETOUR_LEFT: {DETOUR_LEFT: 24, DETOUR_RIGHT: 5, DIRECT: 7, WAIT: 0}
  - DETOUR_RIGHT: {DETOUR_LEFT: 21, DETOUR_RIGHT: 3, DIRECT: 4, WAIT: 0}
  - DIRECT: {DETOUR_LEFT: 75, DETOUR_RIGHT: 32, DIRECT: 89, WAIT: 0}
  - WAIT: {DETOUR_LEFT: 14, DETOUR_RIGHT: 9, DIRECT: 21, WAIT: 0}
- Negative Log Likelihood (NLL): 1.2944
- Prior NLL (empirical): 1.0350
- Average Belief Entropy: 1.2105

## Prefix Results (Bayesian)
- 25%: Accuracy = 0.3158, Macro-F1 = 0.1899
- 50%: Accuracy = 0.3092, Macro-F1 = 0.1752
- 100%: Accuracy = 0.3816, Macro-F1 = 0.2304

## Diagnostic Findings
- Random vs Distance Predictions: Identical (100% match)
  - Reason: Intention distribution is independent of goal in the agent model (fixed behavior weights: DIRECT 0.6, DETOUR_LEFT 0.15, DETOUR_RIGHT 0.15, WAIT 0.1). Therefore, Distance heuristic's goal distribution does not affect intention prediction, resulting in identical predictions to Random.
  - Distance goal scores vary across episodes (mean: [0.722, 0.278], variance: ~0.002), confirming goal distribution is not uniform.
- Bayesian NLL > Prior NLL: Bayesian model's predictions are worse than the empirical prior in terms of likelihood, indicating potential misspecification.
- Prefix results show Bayesian accuracy increases with more data (0.3158 → 0.3816), suggesting the model learns slowly but remains below prior performance.

## Ordering Check (Smoke Test: random < distance < Bayesian)
- Accuracy: 0.6447 < 0.6447 < 0.3816 → False
  - Random < Distance: False (equal)
  - Distance < Bayesian: False (0.6447 > 0.3816)
- Macro-F1: 0.1960 < 0.1960 < 0.2304 → False
  - Random < Distance: False (equal)
  - Distance < Bayesian: True (0.1960 < 0.2304)
- Overall Ordering: FAILED

## Gate Status
- RESULT: Ordering constraint violated
- GATE STATUS: RED (STOP)
- Notes: The equality of Random and Distance baselines is expected given the agent's intention prior is independent of goal. The Bayesian baseline underperforms the prior, suggesting a likelihood misspecification.