# Label Statistics for Intention Triviality Probe (Section 6)

## Training Set Statistics
- Intention distribution:
  - DIRECT: 53.16%
  - DETOUR_LEFT: 14.13%
  - WAIT: 20.82%
  - DETOUR_RIGHT: 11.90%
- Goal distribution:
  - (7, 7): 19.33%
  - (6, 6): 80.67%
- Intention dwell:
  - Mean: 5.85 steps
  - Median: 4.00 steps
- Switches per episode:
  - Mean: 3.00
  - Median: 3.00
- Fraction episodes with >=1 switch: 91.67%
- Blocked-step rate: 8.55%
- Fraction blocked steps (peer/obstacle): 0.00% / 0.00%

## Test Set Statistics
- Intention distribution:
  - DETOUR_RIGHT: 28.92%
  - DETOUR_LEFT: 14.46%
  - DIRECT: 51.81%
  - WAIT: 4.82%
- Goal distribution:
  - (6, 6): 100.0%
- Intention dwell:
  - Mean: 4.88 steps
  - Median: 4.00 steps
- Switches per episode:
  - Mean: 4.67
  - Median: 4.00
- Fraction episodes with >=1 switch: 100.00%
- Blocked-step rate: 15.66%
- Fraction blocked steps (peer/obstacle): 0.00% / 0.00%

## Triviality Gate Assessment
- Maximum intention class percentage: 53.16% (< 60% threshold) ✓
- Median switches per episode: 4.00 (> 0) ✓
- 95% CI for P3−P1 Macro-F1 gap: [0.0043, 0.1895] (does not contain zero) ✓
- Gate Status: GREEN (CONTINUE)
- Result: INTENTION LABELS ARE NOT GEOMETRICALLY TRIVIAL