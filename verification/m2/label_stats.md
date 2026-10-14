# Label Statistics (Section 6)

## train split

- episodes: 336 / scenarios: 42 / focal steps: 12469
- intention distribution:
  - DETOUR_LEFT: 22.82%
  - DETOUR_RIGHT: 22.54%
  - DIRECT: 43.16%
  - WAIT: 11.49%
- maximum intention share: 43.16% (flag threshold 60%)
- goal distribution (focal hidden goal):
  - target 0: 24.70%
  - target 1: 23.81%
  - target 2: 21.43%
  - target 3: 30.06%
- intention dwell: mean 5.65, median 4.00
- switches per episode: mean 5.57, median 6.00
- episodes with >=1 switch: 94.64%
- blocked-step rate: 8.93%
  - attributable to peer interaction (occupied/conflict): 8.93%
  - attributable to obstacle/boundary: 0.00%

## test split

- episodes: 72 / scenarios: 9 / focal steps: 2923
- intention distribution:
  - DETOUR_LEFT: 23.91%
  - DETOUR_RIGHT: 29.76%
  - DIRECT: 36.78%
  - WAIT: 9.54%
- maximum intention share: 36.78% (flag threshold 60%)
- goal distribution (focal hidden goal):
  - target 0: 25.00%
  - target 1: 34.72%
  - target 2: 15.28%
  - target 3: 25.00%
- intention dwell: mean 5.44, median 4.00
- switches per episode: mean 6.46, median 6.50
- episodes with >=1 switch: 93.06%
- blocked-step rate: 12.42%
  - attributable to peer interaction (occupied/conflict): 12.42%
  - attributable to obstacle/boundary: 0.00%

## Triviality Gate Assessment

- 95% CI for P3-P1 Macro-F1 gap: [0.2001, 0.2957], point estimate 0.2485
- CI contains zero: NO
- Gate status: GREEN (CONTINUE)

