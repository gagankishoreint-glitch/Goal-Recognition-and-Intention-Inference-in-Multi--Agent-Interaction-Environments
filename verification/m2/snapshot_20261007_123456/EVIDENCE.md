# M2 Verification Gate Evidence

## Overview
This document summarizes the evidence for the M2 verification gate, which assesses the statistical detectiveness of intention predictions in the agent simulation framework.

## Section 4: Scenario Leakage Test
- Status: PASSED
- Evidence: No significant information leakage between scenarios detected
- Details: All scenario isolation tests passed

## Section 5: End-to-End Information Isolation Test
- Status: PASSED  
- Evidence: No end-to-end information isolation violations
- Details: Information flow controls are functioning correctly

## Section 6: Intention Triviality Probe (Frozen Version)
- Status: GREEN (CONTINUE)
- Evidence: Intention labels are NOT geometrically trivial
- Details:
  - Maximum intention class percentage: 53.16% (< 60% threshold)
  - Median switches per episode: 4.00 (> 0)
  - 95% CI for P3−P1 Macro-F1 gap: [0.0043, 0.1895] (does not contain zero)
  - Result: INTENTION LABELS ARE NOT GEOMETRICALLY TRIVIAL

## Section 7: Baselines Test
- Status: RED (STOP)
- Evidence: Ordering constraint violated (random < distance < Bayesian) but with nuanced interpretation
- Details:
  - Random Baseline: Accuracy = 0.6447, Macro-F1 = 0.1960
  - Distance Heuristic Baseline: Accuracy = 0.6447, Macro-F1 = 0.1960 (identical to Random)
  - Bayesian Belief Baseline: Accuracy = 0.3816, Macro-F1 = 0.2304, NLL = 1.2944 (Prior NLL = 1.0350)
  - Ordering Check:
    - Accuracy: 0.6447 < 0.6447 < 0.3816 → False
    - Macro-F1: 0.1960 < 0.1960 < 0.2304 → False
  - Result: Ordering constraint violated
  - Important Nuance:
    * The equality of Random and Distance baselines is **not** a wiring defect but reflects the agent's intention prior being independent of goal (fixed behavior weights). Therefore, Distance heuristic's goal variation does not affect intention prediction.
    * The Bayesian baseline underperforms the empirical prior in terms of likelihood (NLL higher than prior), indicating a potential likelihood misspecification.
    * Prefix results show Bayesian accuracy improves with more data but remains below prior performance.

## Overall M2 Verification Gate Status
- Section 4: PASSED
- Section 5: PASSED  
- Section 6: GREEN (CONTINUE)
- Section 7: RED (STOP)
- GATE STATUS: RED (STOP)
- Summary: M2 verification gate is NOT GREEN due to failure in Section 7 baselines test ordering constraint. Diagnostics indicate the failure stems from Bayesian likelihood underperformance rather than a simple wiring error.

## Evidence Files
- pytest.txt: Complete pytest output showing all section tests passed except where noted
- dataset_hash.txt: Hash of verification dataset for reproducibility
- label_stats.md: Label statistics from Section 6 triviality probe
- baselines.md: Baseline performance results from Section 7 (including diagnostic details)
- triviality.md: Detailed Section 6 triviality probe results
- EVIDENCE.md: This document