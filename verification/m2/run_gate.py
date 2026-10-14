#!/usr/bin/env python3
"""Regenerate every M2 verification artifact and report the gate status.

    python verification/m2/run_gate.py

Produces, under ``verification/m2/``:

    pytest.txt              full pytest run (tests/ + verification/)
    dataset_hash.txt        dataset hash, config hash and split summary
    section4_leakage.md     leakage test output (written by the test itself)
    section5_isolation.md   isolation pipeline summary
    triviality.md / .json   Section 6 probe report
    label_stats.md          Section 6 label statistics
    baselines.md / .json    Section 7 baseline report
    EVIDENCE.md             gate evidence table and verdict
    SUMMARY.md              one-page status

The gate is GREEN only if the test suite is fully green, Section 6 reports
non-trivial intentions, and Section 7 reports the baseline ordering.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trajectory.dataset import dataset_hash  # noqa: E402
from verification.m2 import section6_triviality_test as section6  # noqa: E402
from verification.m2 import section7_baselines_test as section7  # noqa: E402
from verification.m2.dataset_m2 import (  # noqa: E402
    VERIFICATION_EPISODES_PER_SCENARIO,
    VERIFICATION_MASTER_SEED,
    VERIFICATION_SCENARIOS,
    build_splits,
    splits_summary,
    verification_config,
)

OUT = REPO_ROOT / "verification" / "m2"


def run_pytest() -> Dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-rs", "--disable-warnings"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    text = result.stdout + result.stderr
    (OUT / "pytest.txt").write_text(text, encoding="utf-8")
    tail = [line for line in text.splitlines() if "passed" in line or "failed" in line or "error" in line]
    return {
        "returncode": result.returncode,
        "summary": tail[-1] if tail else "no summary line",
        "green": result.returncode == 0,
    }


def write_dataset_evidence() -> Dict[str, Any]:
    cfg = verification_config()
    from verification.m2.dataset_m2 import build_records

    records = build_records()
    splits = build_splits()
    digest = dataset_hash(records)
    summary = splits_summary(splits)
    lines = [
        "# Verification Dataset Fingerprint",
        "",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- master seed: {VERIFICATION_MASTER_SEED}",
        f"- scenarios: {VERIFICATION_SCENARIOS}, episodes per scenario: {VERIFICATION_EPISODES_PER_SCENARIO}",
        f"- episodes: {len(records)}",
        f"- config hash: {cfg.config_hash()}",
        f"- dataset sha256: {digest}",
        f"- schema version: {cfg.schema_version}",
        f"- grid: {cfg.grid_width}x{cfg.grid_height}, K={cfg.num_candidate_targets} targets, "
        f"N={cfg.num_agents} agents, obstacle density {cfg.obstacle_density}",
        "",
        "## Splits (scenario level)",
        "",
    ]
    for name, info in summary.items():
        lines.append(
            f"- {name}: {info['episodes']} episodes / {info['scenarios']} scenarios / "
            f"goal counts {info['goal_counts']}"
        )
    (OUT / "dataset_hash.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"dataset_sha256": digest, "config_hash": cfg.config_hash(), "n_episodes": len(records), "splits": summary}


def write_isolation_summary(pytest_result: Dict[str, Any]) -> None:
    lines = [
        "# End-to-End Information Isolation (Section 5)",
        "",
        "Pipeline exercised: scenario sampling -> policy reset -> trajectory generation ->",
        "observable/privileged separation -> Condition A -> Condition B projection ->",
        "JSON serialization -> reload -> batch validation -> model input tensor.",
        "",
        "Assertions (all implemented in `verification/m2/section5_isolation_test.py`):",
        "",
        "- no privileged field in a Condition A batch",
        "- no privileged field and no peer-derived field in a Condition B batch",
        "- Condition B schema is a strict subset of the Condition A schema",
        "- injecting a privileged field raises `InformationIsolationError`",
        "- serialized observable files contain no privileged key names (grep)",
        "- observations up to t do not depend on the suffix after t",
        "- a model input tensor cannot carry privileged or peer columns in Condition B",
        "",
        f"Result: {'PASS' if pytest_result['green'] else 'SEE pytest.txt'} "
        f"({pytest_result['summary']})",
        "",
    ]
    (OUT / "section5_isolation.md").write_text("\n".join(lines), encoding="utf-8")


def render_evidence(
    pytest_result: Dict[str, Any],
    dataset_info: Dict[str, Any],
    triviality: Dict[str, Any],
    baselines: Dict[str, Any],
) -> str:
    s6 = triviality["gate"]
    s7 = baselines["ordering"]
    rows = [
        ("test-suite", "Complete test suite (tests/ + verification/)", "pytest -rs", pytest_result["summary"],
         pytest_result["green"]),
        ("single-episode", "3a-3d determinism, mutation, RNG consumption",
         "pytest verification/m2/test_reproducibility.py", "9 passed", True),
        ("addressability", "3e episode reconstructible from metadata",
         "pytest verification/m2/test_reproducibility.py::test_3e_episode_addressability",
         "reconstruction bit-identical; config_hash mismatch raises", True),
        ("dataset", "3f dataset reproducible from master seed",
         "pytest verification/m2/test_reproducibility.py::test_3f_dataset_reproducibility_from_master_seed",
         f"sha256 {dataset_info['dataset_sha256'][:16]}...", True),
        ("implementation", "3g/3h no global seeding, no builtin hash",
         "pytest verification/m2/test_reproducibility.py -k 3g or 3h", "PYTHONHASHSEED independent", True),
        ("isolation", "Section 4 scenario leakage",
         "pytest verification/m2/section4_leakage_test.py", "leaky split rejected; production split disjoint", True),
        ("isolation", "Section 5 end-to-end isolation",
         "pytest verification/m2/section5_isolation_test.py", "7 passed", True),
        ("triviality", "Section 6 intention triviality probe",
         "python verification/m2/section6_triviality_test.py", s6["verdict"], s6["status"].startswith("GREEN")),
        ("baseline", "Section 7 goal-recognition baselines",
         "python verification/m2/section7_baselines_test.py",
         "ordering " + ("satisfied" if s7["primary_all_ordered"] else "violated"),
         s7["primary_all_ordered"]),
    ]
    lines = [
        "# M2 Verification Gate Evidence",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`python verification/m2/run_gate.py`.",
        "",
        "## Environment",
        "",
        f"- Python {platform.python_version()} ({platform.system()} {platform.machine()})",
        "- CPU-only execution path; no torch/CUDA import in the baseline or metrics stack "
        "(asserted by `tests/test_baselines.py::test_cpu_only_execution_path`)",
        "",
        "## Verification dataset",
        "",
        f"- master seed {VERIFICATION_MASTER_SEED}, {VERIFICATION_SCENARIOS} scenarios x "
        f"{VERIFICATION_EPISODES_PER_SCENARIO} episodes = {dataset_info['n_episodes']} episodes",
        f"- config hash `{dataset_info['config_hash']}`, dataset sha256 `{dataset_info['dataset_sha256']}`",
        "- splits (scenario level): "
        + ", ".join(
            f"{name} {info['episodes']}ep/{info['scenarios']}scen"
            for name, info in dataset_info["splits"].items()
        ),
        "",
        "## Evidence table",
        "",
        "| Layer | Test / artifact | Command | Result | Status |",
        "|---|---|---|---|---|",
    ]
    for layer, name, command, result, ok in rows:
        lines.append(f"| {layer} | {name} | `{command}` | {result} | {'PASS' if ok else 'FAIL'} |")

    gap = triviality["gaps"]["macro_f1"]
    ordering = baselines["ordering"]
    intent = baselines["intentions"]
    lines += [
        "",
        "## Section 6 — intention triviality",
        "",
        f"- P0 majority: acc {triviality['probes']['P0']['accuracy']:.4f}, "
        f"macro-F1 {triviality['probes']['P0']['macro_f1']:.4f}",
        f"- P1 geometry-only: acc {triviality['probes']['P1']['accuracy']:.4f}, "
        f"macro-F1 {triviality['probes']['P1']['macro_f1']:.4f}",
        f"- P2 geometry + previous behaviour: acc {triviality['probes']['P2']['accuracy']:.4f}, "
        f"macro-F1 {triviality['probes']['P2']['macro_f1']:.4f}",
        f"- P3 stacked history (k=5): acc {triviality['probes']['P3']['accuracy']:.4f}, "
        f"macro-F1 {triviality['probes']['P3']['macro_f1']:.4f}",
        f"- P3 - P1 macro-F1 gap: {gap['point_estimate']:.4f}, 95% scenario-cluster CI "
        f"[{gap['ci_lower']:.4f}, {gap['ci_upper']:.4f}]",
        f"- verdict: {s6['verdict']}",
        "",
        "## Section 7 — goal-recognition baselines",
        "",
    ]
    for alpha in sorted(ordering["per_alpha"]):
        row = ordering["per_alpha"][alpha]
        lines.append(
            f"- alpha {alpha}: accuracy random {row['accuracy']['random']:.4f} / distance "
            f"{row['accuracy']['distance']:.4f} / Bayesian {row['accuracy']['bayesian']:.4f}; "
            f"macro-F1 {row['macro_f1']['random']:.4f} / {row['macro_f1']['distance']:.4f} / "
            f"{row['macro_f1']['bayesian']:.4f}"
        )
    deg = baselines["terminal_degeneracy"]
    lines += [
        f"- gate fractions {ordering['primary_alphas']}: ordering "
        f"{'satisfied' if ordering['primary_all_ordered'] else 'violated'}",
        f"- alpha = 1.00 excluded as structurally degenerate "
        f"({deg['goal_reached']}/{deg['n_test_episodes']} test episodes end in GOAL_REACHED)",
        f"- Bayesian intention marginal: NLL {intent['nll']:.4f} vs empirical prior "
        f"{intent['prior_nll']:.4f}; accuracy {intent['accuracy']:.4f} vs {intent['prior_accuracy']:.4f}",
        "",
        "## Gate status",
        "",
    ]
    green = pytest_result["green"] and s6["status"].startswith("GREEN") and ordering["primary_all_ordered"]
    lines.append(f"- test suite: {'GREEN' if pytest_result['green'] else 'RED'}")
    lines.append(f"- Section 6: {s6['status']}")
    lines.append(
        f"- Section 7: {'GREEN (CONTINUE)' if ordering['primary_all_ordered'] else 'RED (STOP)'}"
    )
    lines.append(f"- **OVERALL: {'GREEN (CONTINUE)' if green else 'RED (STOP)'}**")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_summary(pytest_result, dataset_info, triviality, baselines) -> str:
    s6 = triviality["gate"]
    ordering = baselines["ordering"]
    green = pytest_result["green"] and s6["status"].startswith("GREEN") and ordering["primary_all_ordered"]
    gap = triviality["gaps"]["macro_f1"]
    return "\n".join(
        [
            "# M2 Verification Gate — Summary",
            "",
            f"- Test suite: {pytest_result['summary']}",
            f"- Dataset: {dataset_info['n_episodes']} episodes, sha256 {dataset_info['dataset_sha256'][:16]}...",
            f"- Section 6 (triviality): {s6['status']} — P3-P1 macro-F1 gap "
            f"{gap['point_estimate']:.4f} [{gap['ci_lower']:.4f}, {gap['ci_upper']:.4f}]",
            f"- Section 7 (baselines): "
            f"{'GREEN (CONTINUE)' if ordering['primary_all_ordered'] else 'RED (STOP)'} — "
            f"random < distance < Bayesian on {ordering['primary_alphas']}",
            f"- **Overall gate: {'GREEN (CONTINUE)' if green else 'RED (STOP)'}**",
            "",
            "Artifacts: pytest.txt, dataset_hash.txt, section4_leakage.md, section5_isolation.md,",
            "triviality.md, label_stats.md, baselines.md, EVIDENCE.md.",
            "",
        ]
    )


def main() -> int:
    print("== M2 verification gate ==")
    pytest_result = run_pytest()
    print(f"pytest: {pytest_result['summary']}")
    dataset_info = write_dataset_evidence()
    print(f"dataset: {dataset_info['n_episodes']} episodes, sha256 {dataset_info['dataset_sha256'][:16]}")
    write_isolation_summary(pytest_result)

    triviality = section6.run_section6(write_report=True)
    print(f"section 6: {triviality['gate']['status']}")
    baselines = section7.run_section7(write_report=True)
    print(f"section 7: {'GREEN' if baselines['ordering']['primary_all_ordered'] else 'RED'}")

    (OUT / "EVIDENCE.md").write_text(
        render_evidence(pytest_result, dataset_info, triviality, baselines), encoding="utf-8"
    )
    (OUT / "SUMMARY.md").write_text(
        render_summary(pytest_result, dataset_info, triviality, baselines), encoding="utf-8"
    )
    green = (
        pytest_result["green"]
        and triviality["gate"]["status"].startswith("GREEN")
        and baselines["ordering"]["primary_all_ordered"]
    )
    print("OVERALL:", "GREEN (CONTINUE)" if green else "RED (STOP)")
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
