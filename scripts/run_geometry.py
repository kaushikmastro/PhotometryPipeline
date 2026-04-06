#!/usr/bin/env python3
"""Batch geometry table generation for Dawn FC images."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.geometry_engine import GeometryEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_ENGINE: GeometryEngine | None = None


def _resolve_manifest_column(df: pd.DataFrame) -> str:
    if "image_filename" in df.columns:
        return "image_filename"
    if "image_id" in df.columns:
        return "image_id"
    raise KeyError("Manifest must include 'image_filename' or 'image_id'.")


def _init_worker(data_root: str, metakernel_path: str | None) -> None:
    """Initializer for each worker process; loads SPICE and DTM once per worker."""
    global _ENGINE
    _ENGINE = GeometryEngine(data_root=data_root, metakernel_path=metakernel_path)


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
        return {
            "status": "error",
            "image_id": image_id,
            "rows": 0,
            "seconds": elapsed,
            "error": str(exc),
        }


def build_worklist(data_root: Path, manifest_path: Path) -> tuple[list[str], list[str]]:
    """Return (to_process_paths, skipped_ids) based on parquet idempotency."""
    df = pd.read_csv(manifest_path)
    col = _resolve_manifest_column(df)

    image_dir = data_root / "01_calibrated_images"
    output_dir = data_root / "04_geometry_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    to_process: list[str] = []
    skipped: list[str] = []

    for raw in df[col].astype(str):
        stem = Path(raw).stem.upper()
        image_path = image_dir / f"{stem}.IMG"
        output_path = output_dir / f"{stem}_geometry.parquet"

        if output_path.exists():
            skipped.append(stem)
            continue

        if not image_path.exists():
            logging.warning("Skipping missing image file: %s", image_path)
            continue

        to_process.append(str(image_path))

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
        default=None,
        help="Optional explicit metakernel path. Defaults to first *.tm in data/02_spice_kernels.",
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
    logging.info("Geometry batch summary: %d queued, %d already done", len(to_process), len(skipped))

    if not to_process:
        logging.info("No missing geometry tables. Nothing to do.")
        return 0

    total = len(to_process)
    worker_count = max(1, min(int(args.workers), total))
    if worker_count != int(args.workers):
        logging.info("Adjusting workers from %d to %d based on queued images=%d", int(args.workers), worker_count, total)

    ok_count = 0
    err_count = 0
    t_start = time.time()

    with mp.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(str(data_root), args.metakernel),
    ) as pool:
        for idx, result in enumerate(pool.imap_unordered(_process_one, to_process, chunksize=1), start=1):
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


if __name__ == "__main__":
    raise SystemExit(main())
