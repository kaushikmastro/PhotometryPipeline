import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_geometry import build_worklist  # noqa: E402


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    manifest_path = path / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path


def _touch_img(root: Path, phase: str, stem: str) -> None:
    img_path = root / "01_calibrated_images" / phase / f"{stem}.IMG"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"ok")


def test_build_worklist_includes_all_four_phases(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    rows = [
        {
            "image_filename": "FC21SURVEY_0001.IMG",
            "phase_subdir": "survey",
            "file_specification_name": "/DATA/SURVEY/FC21SURVEY_0001.IMG",
        },
        {
            "image_filename": "FC21HAMO_0002.IMG",
            "phase_subdir": "hamo",
            "file_specification_name": "/DATA/HAMO/FC21HAMO_0002.IMG",
        },
        {
            "image_filename": "FC21LAMO_0003.IMG",
            "phase_subdir": "lamo",
            "file_specification_name": "/DATA/LAMO/FC21LAMO_0003.IMG",
        },
        {
            "image_filename": "FC21RC_0004.IMG",
            "phase_subdir": "rc",
            "file_specification_name": "/DATA/RC/FC21RC_0004.IMG",
        },
    ]
    manifest_path = _write_manifest(tmp_path, rows)

    _touch_img(data_root, "survey", "FC21SURVEY_0001")
    _touch_img(data_root, "hamo", "FC21HAMO_0002")
    _touch_img(data_root, "lamo", "FC21LAMO_0003")
    _touch_img(data_root, "rc", "FC21RC_0004")

    to_process, skipped = build_worklist(data_root, manifest_path)

    assert skipped == []
    assert len(to_process) == 4
    assert any("/survey/" in p for p in to_process)
    assert any("/hamo/" in p for p in to_process)
    assert any("/lamo/" in p for p in to_process)
    assert any("/rc/" in p for p in to_process)



def test_build_worklist_has_one_path_per_valid_image_no_double_count(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    rows = [
        {
            "image_filename": "FC21LAMO_0100.IMG",
            "phase_subdir": "lamo",
            "file_specification_name": "/DATA/LAMO/FC21LAMO_0100.IMG",
        },
        {
            "image_filename": "FC21RC_0200.IMG",
            "phase_subdir": "rc",
            "file_specification_name": "/DATA/RC/FC21RC_0200.IMG",
        },
    ]
    manifest_path = _write_manifest(tmp_path, rows)

    _touch_img(data_root, "lamo", "FC21LAMO_0100")
    _touch_img(data_root, "rc", "FC21RC_0200")

    # Mark one output as already complete so it goes to skipped, not queued.
    done_output = data_root / "04_geometry_tables" / "lamo" / "FC21LAMO_0100_geometry.parquet"
    done_output.parent.mkdir(parents=True, exist_ok=True)
    done_output.write_bytes(b"done")

    to_process, skipped = build_worklist(data_root, manifest_path)

    assert skipped == ["FC21LAMO_0100"]
    assert to_process == [str(data_root / "01_calibrated_images" / "rc" / "FC21RC_0200.IMG")]

    # Strong duplicate-path guard: one manifest row -> one queued path.
    assert len(to_process) == len(set(to_process))


def test_build_worklist_dedupes_duplicate_manifest_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    rows = [
        {
            "image_filename": "FC21LAMO_7777.IMG",
            "phase_subdir": "lamo",
            "file_specification_name": "/DATA/LAMO/FC21LAMO_7777.IMG",
        },
        {
            "image_filename": "FC21LAMO_7777.IMG",
            "phase_subdir": "lamo",
            "file_specification_name": "/DATA/LAMO/FC21LAMO_7777.IMG",
        },
    ]
    manifest_path = _write_manifest(tmp_path, rows)
    _touch_img(data_root, "lamo", "FC21LAMO_7777")

    to_process, skipped = build_worklist(data_root, manifest_path)

    assert skipped == []
    assert to_process == [str(data_root / "01_calibrated_images" / "lamo" / "FC21LAMO_7777.IMG")]
