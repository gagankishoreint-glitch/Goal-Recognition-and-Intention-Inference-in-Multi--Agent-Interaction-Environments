# Scenario Leakage Test (Section 4)

## Adversarial leakage exception (4.2)

A split in which the same `scenario_id` appears in both train and test is rejected:

```
scenario 9aed644818b77c6d appears in both train and test
```

## Production split (4.3)

Dataset: 20 scenarios x 4 episodes, master seed 20261008.

- train: 56 episodes / 14 unique scenarios
- val: 12 episodes / 3 unique scenarios
- test: 12 episodes / 3 unique scenarios
- overlap between any two splits: none (asserted)
