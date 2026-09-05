from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "geometry"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_geometry import REQUIRED_COLUMNS, discover_worklist  # noqa: E402

# NOTE: this file previously tested build_worklist(), a manifest-CSV-driven
# worklist builder that no longer exists in run_geometry.py (the pipeline now
# globs calibrated_raw_images/**/*.IMG directly, no manifest). These tests
# target the current discover_worklist(input_images, target_output_root)
# function instead, which is what compute_geometry() results are actually
# skip-checked against today.


def _touch_img(root: Path, phase: str, stem: str) -> Path:
    img_path = root / "calibrated_raw_images" / phase / f"{stem}.IMG"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"ok")
    return img_path


def _write_valid_parquet(path: Path, image_id: str = "TEST") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "image_id": [image_id],
            "pixel_x": [0],
            "pixel_y": [0],
            "iof": [0.1],
            "incidence": [10.0],
            "emission": [10.0],
            "phase": [10.0],
            "latitude": [0.0],
            "longitude": [0.0],
        }
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def _write_schema_mismatched_parquet(path: Path) -> None:
    """A real, readable parquet, but missing REQUIRED_COLUMNS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": ["TEST"], "some_other_column": [1]}).to_parquet(
        path, engine="pyarrow", index=False
    )


def _write_corrupted_file(path: Path) -> None:
    """Not a parquet file at all — simulates a truncated/killed-mid-write output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real parquet file")


def test_discover_worklist_includes_all_four_phases(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "geometry/dsk256"

    images = [
        _touch_img(data_root, "survey", "FC21SURVEY_0001"),
        _touch_img(data_root, "hamo", "FC21HAMO_0002"),
        _touch_img(data_root, "lamo", "FC21LAMO_0003"),
        _touch_img(data_root, "rc", "FC21RC_0004"),
    ]

    worklist = discover_worklist(sorted(images), output_root)

    assert len(worklist) == 4
    assert any("/survey/" in p for p in worklist)
    assert any("/hamo/" in p for p in worklist)
    assert any("/lamo/" in p for p in worklist)
    assert any("/rc/" in p for p in worklist)


def test_discover_worklist_skips_existing_valid_parquet(tmp_path: Path) -> None:
    """Item 4: an existing, schema-valid, readable parquet must be skipped."""
    data_root = tmp_path / "data"
    output_root = tmp_path / "geometry/dsk256"

    done_img = _touch_img(data_root, "lamo", "FC21LAMO_0100")
    pending_img = _touch_img(data_root, "rc", "FC21RC_0200")

    _write_valid_parquet(output_root / "lamo" / "FC21LAMO_0100_geometry.parquet")

    worklist = discover_worklist(sorted([done_img, pending_img]), output_root)

    assert worklist == [str(pending_img)]


def test_discover_worklist_reprocesses_corrupted_parquet(tmp_path: Path) -> None:
    """Item 4: a truncated/corrupted parquet must NOT be silently trusted —
    this is the exact behavior that caught the 3 corrupted hamo_allF1 files
    tonight (unreadable-as-parquet -> exception -> re-queued, not skipped).
    """
    data_root = tmp_path / "data"
    output_root = tmp_path / "geometry/dsk256"

    img = _touch_img(data_root, "hamo", "FC21HAMO_9999")
    _write_corrupted_file(output_root / "hamo" / "FC21HAMO_9999_geometry.parquet")

    worklist = discover_worklist([img], output_root)

    assert worklist == [str(img)]


def test_discover_worklist_reprocesses_schema_mismatched_parquet(tmp_path: Path) -> None:
    """Item 4: a real, readable parquet that's missing REQUIRED_COLUMNS must
    also be re-queued, not trusted just because it opens successfully.
    """
    data_root = tmp_path / "data"
    output_root = tmp_path / "geometry/dsk256"

    img = _touch_img(data_root, "survey", "FC21SURVEY_5555")
    _write_schema_mismatched_parquet(output_root / "survey" / "FC21SURVEY_5555_geometry.parquet")

    worklist = discover_worklist([img], output_root)

    assert worklist == [str(img)]


def test_discover_worklist_diff_exact_no_off_by_one_no_duplicates(tmp_path: Path) -> None:
    """Item 5: enumerate-all minus already-done must be exact for a known set —
    5 candidate images, 2 already done (one valid, one corrupted), expect
    exactly the 3 pending images plus the 1 corrupted one re-queued (4 total),
    with no duplicates.
    """
    data_root = tmp_path / "data"
    output_root = tmp_path / "geometry/dsk256"

    img_done_valid = _touch_img(data_root, "hamo", "FC21HAMO_0001")
    img_done_corrupt = _touch_img(data_root, "hamo", "FC21HAMO_0002")
    img_pending_a = _touch_img(data_root, "hamo", "FC21HAMO_0003")
    img_pending_b = _touch_img(data_root, "hamo", "FC21HAMO_0004")
    img_pending_c = _touch_img(data_root, "hamo", "FC21HAMO_0005")

    _write_valid_parquet(output_root / "hamo" / "FC21HAMO_0001_geometry.parquet")
    _write_corrupted_file(output_root / "hamo" / "FC21HAMO_0002_geometry.parquet")

    input_images = sorted([img_done_valid, img_done_corrupt, img_pending_a, img_pending_b, img_pending_c])
    worklist = discover_worklist(input_images, output_root)

    expected = sorted(str(p) for p in [img_done_corrupt, img_pending_a, img_pending_b, img_pending_c])
    assert sorted(worklist) == expected
    assert len(worklist) == len(set(worklist))  # no duplicates
    assert str(img_done_valid) not in worklist  # the one genuinely-done image is excluded


def test_discover_worklist_required_columns_constant_matches_geometry_engine_output() -> None:
    """Sanity check that REQUIRED_COLUMNS hasn't silently drifted from what
    compute_geometry() actually writes (the DataFrame columns built in
    geometry_engine.py's Stage 5)."""
    expected = {
        "image_id", "pixel_x", "pixel_y", "iof",
        "incidence", "emission", "phase", "latitude", "longitude",
    }
    assert REQUIRED_COLUMNS == expected
