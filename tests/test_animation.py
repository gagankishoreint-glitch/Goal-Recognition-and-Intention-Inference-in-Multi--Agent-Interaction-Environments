"""The demo animation renders from regenerated episodes without touching generation.

Runs the actual script with ``--fast`` (tiny dataset) and a scratch output, so the
suite exercises the real rendering path. The committed artifact
(``figures/demo_crossing.gif``) is produced by the default invocation and is
deliberately not rebuilt on every test run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "verification" / "make_animation.py"


def test_animation_script_renders_a_gif(tmp_path) -> None:
    out = tmp_path / "demo.gif"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--fast", "--out", str(out), "--fps", "4"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists(), "animation was not written"
    assert out.stat().st_size > 10_000, f"suspiciously small gif: {out.stat().st_size} bytes"
    # GIF magic header
    assert out.read_bytes()[:3] == b"GIF"


def test_committed_artifact_exists() -> None:
    gif = REPO_ROOT / "figures" / "demo_crossing.gif"
    assert gif.exists(), "run verification/make_animation.py to produce the committed demo"
    assert gif.stat().st_size > 10_000
