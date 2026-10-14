# Multi-Task GRU Results

GRU hidden 48, 8 epochs, Adam lr 0.003; input 58 timing-contract channels per frame; goal head read at the alpha-truncated hidden state.

## Goal recognition vs baselines (test scenarios)

| alpha | model | accuracy | macro-F1 | NLL |
|---|---|---|---|---|
| 0.30 | gru | 0.3056 | 0.2802 | 1.7470 |
| 0.30 | bayesian | 0.4861 | 0.4743 | 1.0633 |
| 0.30 | distance | 0.2639 | 0.2500 | 3.6598 |
| 0.30 | random | 0.1667 | 0.1281 | 1.3863 |
| 0.50 | gru | 0.3889 | 0.3382 | 1.5351 |
| 0.50 | bayesian | 0.6667 | 0.6725 | 0.7132 |
| 0.50 | distance | 0.3472 | 0.3256 | 2.9042 |
| 0.50 | random | 0.2778 | 0.1900 | 1.3863 |
| 0.75 | gru | 0.4444 | 0.3993 | 1.2677 |
| 0.75 | bayesian | 0.7778 | 0.7722 | 0.5064 |
| 0.75 | distance | 0.4861 | 0.4751 | 2.5538 |
| 0.75 | random | 0.3194 | 0.1953 | 1.3863 |
| 1.00 | gru | 0.4583 | 0.3959 | 1.3499 |
| 1.00 | bayesian | 0.8889 | 0.8923 | 0.3754 |
| 1.00 | distance | 0.5417 | 0.5274 | 2.0087 |
| 1.00 | random | 0.2917 | 0.1812 | 1.3863 |

## Intention inference (per step, test scenarios)

| model | accuracy | macro-F1 | n |
|---|---|---|---|
| gru | 0.4796 | 0.4628 | 2923 |

## RQ2 interaction contrast for the GRU (A full vs D noise-masked)

| task | acc A | acc D | acc gap (A-D) | 95% CI | macro-F1 gap |
|---|---|---|---|---|---|
| goal | 0.4444 | 0.2639 | +0.1806 (significant) | [+0.0139, +0.3611] | +0.1876 |
| intention | 0.4796 | 0.3472 | +0.1324 (significant) | [+0.1005, +0.1617] | +0.1057 |

A gap counts as supported only when its 95% CI excludes zero. The masked arm uses Gaussian noise substitution, never zeroing (TIMING_CONTRACT.md §8).

