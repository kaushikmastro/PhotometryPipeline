from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "golden" / "build_disk_resolved_golden.py"

_spec = importlib.util.spec_from_file_location("build_disk_resolved_golden", SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

build_golden_query = _module.build_golden_query
build_disk_resolved_golden = _module.build_disk_resolved_golden

COMMITTED_INPUT = (
    PROJECT_ROOT / "data" / "silver" / "gaskell_dsk256_110825"
    / "DR_survey_gaskell_dsk256_110825.parquet"
)
COMMITTED_GOLDEN = {
    50.0: PROJECT_ROOT / "data" / "golden" / "survey_binned_dsk256_110825_range50.parquet",
    80.0: PROJECT_ROOT / "data" / "golden" / "survey_binned_dsk256_110825_range80.parquet",
}


# ---------------------------------------------------------------------------
# Fast: pure query-string construction, no DB / no real data needed
# ---------------------------------------------------------------------------


def test_build_golden_query_uses_banker_rounding_on_all_three_axes() -> None:
    query = build_golden_query("dummy.parquet", 50.0)

    for column in ("phase", "incidence", "emission"):
        assert f"({column}/5.0)" in query
        assert f"FLOOR({column}/5.0)" in query
        assert f"CEIL({column}/5.0)" in query
    assert "1e-9" in query


def test_build_golden_query_threshold_is_the_only_thing_that_changes() -> None:
    """The two committed variants (range50, range80) must differ ONLY in the
    incidence/emission threshold -- confirms this is genuinely one parametrized query,
    not two independently-drifting copies."""
    q50 = build_golden_query("x.parquet", 50.0)
    q80 = build_golden_query("x.parquet", 80.0)

    assert "incidence < 50.0" in q50 and "emission < 50.0" in q50
    assert "incidence < 80.0" in q80 and "emission < 80.0" in q80
    # strip the threshold text and confirm everything else is byte-identical
    assert q50.replace("50.0", "X") == q80.replace("80.0", "X")


def test_build_golden_query_triangle_law_and_validity_floor_present() -> None:
    query = build_golden_query("x.parquet", 50.0)

    assert "phase >= ABS(incidence - emission)" in query
    assert "phase <= (incidence + emission) + 0.1" in query
    assert "HAVING COUNT(*) >= 10" in query
    assert "STDDEV_SAMP(iof) IS NOT NULL" in query
    assert "image_id LIKE '%F1B%'" in query


def test_build_golden_query_respects_custom_bin_size_and_filter() -> None:
    query = build_golden_query("x.parquet", 50.0, bin_size_deg=10.0, image_id_like="%F1C%", min_pixels_per_bin=5)

    assert "(phase/10.0)" in query
    assert "image_id LIKE '%F1C%'" in query
    assert "HAVING COUNT(*) >= 5" in query


# ---------------------------------------------------------------------------
# Slow: real-data bit-for-bit reproduction of the committed golden parquets.
# The input silver-layer parquet is ~12GB / hundreds of millions of rows -- per
# CLAUDE.md Rule 1, running this (with --run-slow) must go through srun/sbatch, never
# directly on the login node.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("max_deg", [50.0, 80.0])
def test_reproduces_committed_golden_parquet_bit_for_bit(max_deg: float, tmp_path) -> None:
    if not COMMITTED_INPUT.exists():
        pytest.skip(f"real silver-layer input not present: {COMMITTED_INPUT}")
    committed_path = COMMITTED_GOLDEN[max_deg]
    if not committed_path.exists():
        pytest.skip(f"committed golden parquet not present: {committed_path}")

    output_path = tmp_path / f"regenerated_range{int(max_deg)}.parquet"
    regenerated = build_disk_resolved_golden(str(COMMITTED_INPUT), str(output_path), max_deg)
    committed = pd.read_parquet(committed_path)

    assert list(regenerated.columns) == list(committed.columns)
    assert len(regenerated) == len(committed), (
        f"bin count differs: regenerated={len(regenerated)} committed={len(committed)}"
    )

    regenerated_sorted = regenerated.sort_values(["alpha_grid", "i_grid", "e_grid"]).reset_index(drop=True)
    committed_sorted = committed.sort_values(["alpha_grid", "i_grid", "e_grid"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        regenerated_sorted, committed_sorted, check_exact=False, rtol=1e-10, atol=0.0
    )
