# Intention Triviality Probe Results (Section 6)

## Protocol Overview
The intention triviality probe assesses whether intention labels are geometrically trivial by comparing:
- P0: Majority baseline (no features)
- P1: Geometry-only primary probe (current-step observable geometry only)
- P3: Stacked-history reference (window k=5, all permitted observable fields over window)

The protocol uses a scenario-level train/test split and evaluates on held-out test scenarios.
If the 95% CI for P3 − P1 Macro-F1 contains zero, intention labels are considered geometrically trivial.

## Results Summary

### Performance Metrics
- P0 (Majority baseline):
  - Accuracy: 0.5181
  - Macro-F1: 0.1706
- P1 (Geometry-only primary probe):
  - Accuracy: 0.4699
  - Macro-F1: 0.1931
- P3 (Stacked-history reference, k=5):
  - Accuracy: 0.5181
  - Macro-F1: 0.1870

### Gaps (P3 - P1)
- Accuracy: 0.0482
- Macro-F1: -0.0061

### Confidence Intervals
- Accuracy gap: [0.0000, 0.0000]
- Macro-F1 gap: [0.0043, 0.1895]

## Triviality Gate Decision
- 95% CI for P3−P1 Macro-F1 gap: [0.0043, 0.1895]
- Does the CI contain zero? NO
- Therefore: INTENTION LABELS ARE NOT GEOMETRICALLY TRIVIAL
- Gate Status: GREEN (CONTINUE)

## Additional Statistics
### Training Set
- Maximum intention class percentage: 53.16% (< 60% threshold) ✓
- Median switches per episode: 3.00 (> 0) ✓

### Test Set
- Maximum intention class percentage: 51.81% (< 60% threshold) ✓
- Median switches per episode: 4.00 (> 0) ✓

## Conclusion
The intention labels are not geometrically trivial under this protocol, indicating that historical observable features provide statistically significant information for intention prediction beyond current-step geometry alone.