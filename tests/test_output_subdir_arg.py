from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_geometry  # noqa: E402

# This test guards against tonight's silent-contamination bug: run_geometry.py
# used to hardcode output_subdir from --mode, which didn't match the real
# committed baseline location (geometry/dsk256) and would
# have silently reprocessed/duplicated the whole committed dataset. The fix
# makes --output-subdir required with no default, so a missing destination
# fails loudly at argparse time instead of silently writing to the wrong place.
#
# Exercised in-process rather than via subprocess: a real subprocess re-import
# of numpy/pyarrow was unreliable tonight due to genuine OS thread/process
# contention from the concurrently-running 8-job HAMO batch on this shared
# login node (OpenBLAS pthread_create failures, not a code issue). The
# in-process check below gives the same guarantee without competing for that
# same constrained resource.


def test_output_subdir_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["run_geometry.py", "--data-root", "/tmp", "--metakernel", "/tmp/fake.tm", "--mode", "DSK256"],
    )

    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture), pytest.raises(SystemExit) as exc_info:
        run_geometry.main()

    assert exc_info.value.code == 2  # argparse's standard "bad usage" exit code
    stderr_text = stderr_capture.getvalue().lower()
    assert "output-subdir" in stderr_text or "output_subdir" in stderr_text
    assert "required" in stderr_text


def test_output_subdir_accepted_when_provided(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Providing --output-subdir must let argparse succeed. Exercises the
    real main() end-to-end (not a duplicated parser) against an empty data
    root, so it naturally hits the empty-worklist early return without ever
    needing SPICE — if argparse had rejected --output-subdir, this would
    raise SystemExit(2) before reaching that point.
    """
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_geometry.py",
            "--data-root", str(tmp_path),
            "--metakernel", str(tmp_path / "fake.tm"),
            "--mode", "DSK256",
            "--output-subdir", "geometry/dsk256",
        ],
    )

    run_geometry.main()  # no .IMG files under tmp_path -> empty worklist -> clean early return

    assert (tmp_path / "geometry/dsk256").is_dir()
