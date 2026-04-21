#!/usr/bin/env python3
"""Pre-flight smoke test for Dawn Vesta ingestion and geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import spiceypy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.geometry_engine import GeometryEngine  # noqa: E402


def _manifest_column(df: pd.DataFrame) -> str:
    if "image_filename" in df.columns:
        return "image_filename"
    if "image_id" in df.columns:
        return "image_id"
    raise KeyError("Manifest must contain 'image_filename' or 'image_id'.")


def _normalize_stem(value: str) -> str:
    return Path(str(value).strip()).stem.upper()


def _first_pair_paths(first_value: str, image_dir: Path) -> tuple[Path, Path]:
    stem = _normalize_stem(first_value)
    img_path = image_dir / f"{stem}.IMG"
    lbl_path = image_dir / f"{stem}.LBL"
    return img_path, lbl_path


def gate_spice_clock(metakernel_path: Path) -> bool:
    print("[GATE 1] SPICE Clock / Leapseconds Check")
    try:
        spiceypy.furnsh(str(metakernel_path))
        spiceypy.utc2et("2011-08-11T12:00:00")
    except spiceypy.support_types.SpiceyError as exc:
        message = str(exc)
        if "SPICE(NOLEAPSECONDS)" in message:
            print("[FAIL] Gate 1: SPICE(NOLEAPSECONDS) - the .tls kernel is missing from the metakernel.")
        else:
            print(f"[FAIL] Gate 1: {message.splitlines()[0] if message else 'SPICE error'}")
        return False
    finally:
        spiceypy.kclear()

    print("[PASS] Gate 1: SPICE clock load and UTC->ET conversion succeeded.")
    return True


def gate_data_warehouse(manifest_path: Path, data_root: Path) -> tuple[bool, str | None]:
    print("[GATE 2] Data Warehouse / Path Check")
    df = pd.read_csv(manifest_path)
    if df.empty:
        print("[FAIL] Gate 2: survey_manifest.csv is empty.")
        return False, None

    col = _manifest_column(df)
    first_value = str(df.iloc[0][col])
    image_dir = data_root.resolve(strict=True) / "01_calibrated_images"
    img_path, lbl_path = _first_pair_paths(first_value, image_dir)

    missing: list[Path] = []
    if not img_path.exists():
        missing.append(img_path)
    if not lbl_path.exists():
        missing.append(lbl_path)

    if missing:
        for path in missing:
            print(f"[FAIL] Gate 2 missing: {path}")
        return False, str(img_path)

    print(f"[PASS] Gate 2: Found paired files at {img_path} and {lbl_path}")
    return True, str(img_path)


def gate_engine_ignition(data_root: Path, metakernel_path: Path, image_path: str) -> bool:
    print("[GATE 3] Engine Ignition / Single-Image Math")
    try:
        engine = GeometryEngine(
            data_root=str(data_root),
            metakernel_path=str(metakernel_path),
            body_fixed_frame="IAU_VESTA",
        )
        df = engine.compute_geometry(image_path)
        print(
            "[PASS] Gate 3: real compute_geometry succeeded "
            f"(rows={len(df)})."
        )
        return True
    except Exception as exc:
        print(f"[FAIL] Gate 3: {exc}")
        return False
    finally:
        spiceypy.kclear()


def main() -> int:
    manifest_path = PROJECT_ROOT / "configs" / "survey_manifest.csv"
    data_root = PROJECT_ROOT / "data"
    metakernel_path = data_root / "02_spice_kernels" / "dawn_dynamic.tm"

    gate1_ok = gate_spice_clock(metakernel_path)
    gate2_ok, first_image_path = gate_data_warehouse(manifest_path, data_root)
    gate3_ok = False
    if gate2_ok and first_image_path is not None:
        gate3_ok = gate_engine_ignition(data_root, metakernel_path, first_image_path)
    else:
        print("[FAIL] Gate 3: Skipped because Gate 2 did not pass.")

    print("\nSmoke test summary:")
    print(f"  Gate 1: {'PASS' if gate1_ok else 'FAIL'}")
    print(f"  Gate 2: {'PASS' if gate2_ok else 'FAIL'}")
    print(f"  Gate 3: {'PASS' if gate3_ok else 'FAIL'}")

    return 0 if gate1_ok and gate2_ok and gate3_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())