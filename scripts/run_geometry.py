#!/usr/bin/env python3
"""Batch geometry table generation for Dawn FC images."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import spiceypy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.geometry_engine import GeometryEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_ENGINE: GeometryEngine | None = None
FAILURE_LOG_PATH = PROJECT_ROOT / "logs" / "geometry_failure_log.jsonl"


def _resolve_manifest_column(df: pd.DataFrame) -> str:
    if "image_filename" in df.columns:
        return "image_filename"
    if "image_id" in df.columns:
        return "image_id"
    raise KeyError("Manifest must include 'image_filename' or 'image_id'.")


def _phase_from_file_spec(file_specification_name: str) -> str | None:
    path = str(file_specification_name).upper()
    if "SURVEY" in path:
        return "survey"
    if "HAMO" in path:
        return "hamo"
    if "LAMO" in path:
        return "lamo"
    if "_RC" in path or "/RC" in path:
        return "rc"
    return None


def _phase_subdir_for_row(row: pd.Series) -> str:
    phase = str(row.get("phase_subdir", "")).strip().lower()
    if phase in {"rc", "survey", "hamo", "lamo"}:
        return phase

    file_spec = str(
        row.get("file_specification_name", row.get("FILE_SPECIFICATION_NAME", ""))
    ).strip()
    phase_from_spec = _phase_from_file_spec(file_spec)
    return phase_from_spec or "survey"


def _init_worker(data_root: str, metakernel_path: str, body_fixed_frame: str = "IAU_VESTA") -> None:
    """Initializer for each worker process; loads SPICE and DTM once per worker."""
    global _ENGINE
    _ENGINE = GeometryEngine(
        data_root=data_root,
        metakernel_path=metakernel_path,
        body_fixed_frame=body_fixed_frame,
    )
    atexit.register(spiceypy.kclear)


def _process_one(image_file_path: str) -> dict[str, Any]:
    """Process one image; executed inside worker process."""
    global _ENGINE
    if _ENGINE is None:
        raise RuntimeError("Worker GeometryEngine is not initialized.")

    image_path = Path(image_file_path)
    image_id = image_path.stem
    t0 = time.time()
    try:
        df = _ENGINE.compute_geometry(str(image_path))
        elapsed = time.time() - t0
        return {
            "status": "ok",
            "image_id": image_id,
            "rows": int(len(df)),
            "seconds": elapsed,
        }
    except Exception as exc:  # pragma: no cover - exercised in HPC runtime
        elapsed = time.time() - t0
        error_type = "Exception"
        if isinstance(exc, spiceypy.support_types.SpiceyError):
            error_type = "SpiceyError"
        elif "SPICE(" in str(exc):
            error_type = "SpiceyErrorText"

        return {
            "status": "error",
            "image_id": image_id,
            "image_path": str(image_path),
            "rows": 0,
            "seconds": elapsed,
            "error_type": error_type,
            "error": str(exc),
        }


def _append_failure_record(result: dict[str, Any], log_path: Path) -> None:
    """Append one structured failure record to the dedicated failure log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "image_id": result.get("image_id", ""),
        "image_path": result.get("image_path", ""),
        "error_type": result.get("error_type", "Exception"),
        "error": result.get("error", "unknown"),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_worklist(data_root: Path, manifest_path: Path) -> tuple[list[str], list[str]]:
    """Return (to_process_paths, skipped_ids) based on parquet idempotency."""
    df = pd.read_csv(manifest_path)
    col = _resolve_manifest_column(df)
    valid_phases = {"survey", "hamo", "lamo", "rc"}

    image_dir = data_root / "01_calibrated_images"
    output_dir = data_root / "04_geometry_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    to_process: list[str] = []
    skipped: list[str] = []
    seen_paths: set[str] = set()

    for _, row in df.iterrows():
        raw = str(row.get(col, ""))
        stem = Path(raw).stem.upper()
        phase_subdir = _phase_subdir_for_row(row)

        if phase_subdir not in valid_phases:
            continue

        image_path = image_dir / phase_subdir / f"{stem}.IMG"

        output_phase_dir = output_dir / phase_subdir
        output_phase_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_phase_dir / f"{stem}_geometry.parquet"

        if os.path.exists(output_path):
            skipped.append(stem)
            continue

        if not image_path.exists():
            logging.warning("Skipping missing image file for %s (phase=%s)", stem, phase_subdir)
            continue

        image_path_str = str(image_path)
        if image_path_str in seen_paths:
            continue

        seen_paths.add(image_path_str)
        to_process.append(image_path_str)

    return to_process, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometry generation for all manifest images.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root data directory containing 01/02/03/04 subdirectories.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "survey_manifest.csv"),
        help="Path to survey manifest CSV.",
    )
    parser.add_argument(
        "--metakernel",
        type=str,
        required=True,
        help="Explicit metakernel path (.tm) to furnish for the entire run.",
    )
    parser.add_argument(
        "--body-fixed-frame",
        type=str,
        default="IAU_VESTA",
        help="Body-fixed reference frame for geometry calculations (default: IAU_VESTA). "
        "For high-resolution DSK models with localized frames, provide the DSK-native frame name.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        logging.error("Manifest not found: %s", manifest_path)
        return 2

    try:
        resolved = data_root.resolve(strict=True)
    except FileNotFoundError:
        logging.error("Data root does not exist: %s", data_root)
        return 3

    if not str(resolved).startswith("/scratch/"):
        logging.error("Refusing run: data root resolves to %s, expected /scratch/...", resolved)
        return 4

    to_process, skipped = build_worklist(data_root, manifest_path)
    logging.info(
        "Geometry batch summary: %d queued, %d already done", len(to_process), len(skipped)
    )

    if not to_process:
        logging.info("No missing geometry tables. Nothing to do.")
        return 0

    total = len(to_process)
    worker_count = max(1, min(int(args.workers), total))
    if worker_count != int(args.workers):
        logging.info(
            "Adjusting workers from %d to %d based on queued images=%d",
            int(args.workers),
            worker_count,
            total,
        )

    ok_count = 0
    err_count = 0
    t_start = time.time()

    try:
        with mp.Pool(
            processes=worker_count,
            initializer=_init_worker,
            initargs=(str(data_root), args.metakernel, args.body_fixed_frame),
        ) as pool:
            for idx, result in enumerate(
                pool.imap_unordered(_process_one, to_process, chunksize=1), start=1
            ):
                if result["status"] == "ok":
                    ok_count += 1
                    logging.info(
                        "[%d/%d] OK %s rows=%d time=%.1fs",
                        idx,
                        total,
                        result["image_id"],
                        result["rows"],
                        result["seconds"],
                    )
                else:
                    err_count += 1
                    logging.error(
                        "[%d/%d] FAIL %s time=%.1fs error=%s",
                        idx,
                        total,
                        result["image_id"],
                        result["seconds"],
                        result.get("error", "unknown"),
                    )
                    _append_failure_record(result, FAILURE_LOG_PATH)

                elapsed = time.time() - t_start
                rate = idx / elapsed if elapsed > 0 else 0.0
                remaining = (total - idx) / rate if rate > 0 else float("inf")
                if remaining != float("inf"):
                    logging.info(
                        "Progress %.1f%% (%d/%d), throughput=%.3f img/s, ETA=%.1f min",
                        100.0 * idx / total,
                        idx,
                        total,
                        rate,
                        remaining / 60.0,
                    )

        logging.info("Geometry batch completed: ok=%d fail=%d total=%d", ok_count, err_count, total)
        return 1 if err_count > 0 else 0
    finally:
        spiceypy.kclear()


if __name__ == "__main__":
    raise SystemExit(main())
