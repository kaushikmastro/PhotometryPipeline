"""
LAMO F1B geometry grind — mission-science Gaskell DSK256 (vesta_gaskell_256_110825.bds).

Input:  calibrated_raw_images/lamo/ — 4349 F1B images (CYCLE15-20 + Transfer-to-LAMO)
Output: geometry/dsk256/lamo/

DSK kernel must be referenced in dawn_dynamic.tm before this script runs.
Idempotent: skips parquets that already exist with valid schema.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hapke_mcmc_package.etl.geometry_engine import GeometryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_ROOT      = Path("/scratch/kaushim07/vesta_data")
METAKERNEL     = DATA_ROOT / "spice_kernels" / "dawn_dynamic.tm"
OUTPUT_SUBDIR  = "geometry/dsk256"
SURFACE_METHOD = "DSK/UNPRIORITIZED"
F_SOLAR        = 892.0
N_WORKERS      = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

REQUIRED_COLUMNS = {
    "image_id", "pixel_x", "pixel_y", "iof",
    "incidence", "emission", "phase", "latitude", "longitude",
}

_engine_worker: GeometryEngine | None = None


def _init_worker() -> None:
    global _engine_worker
    _engine_worker = GeometryEngine(
        data_root=str(DATA_ROOT),
        metakernel_path=str(METAKERNEL),
        surface_intercept_method=SURFACE_METHOD,
        output_subdir=OUTPUT_SUBDIR,
        f_solar=F_SOLAR,
    )


def _process_image(image_path: str) -> tuple[bool, str, str | None]:
    global _engine_worker
    try:
        _engine_worker.compute_geometry(image_path)
        return True, image_path, None
    except Exception as exc:
        logging.error("Failed: %s  error: %s", image_path, exc)
        return False, image_path, str(exc)


def main() -> None:
    output_root = DATA_ROOT / OUTPUT_SUBDIR / "lamo"
    output_root.mkdir(parents=True, exist_ok=True)

    lamo_f1b = sorted(
        (DATA_ROOT / "calibrated_raw_images" / "lamo").glob("*F1B*.IMG")
    )
    logging.info("LAMO F1B images found: %d", len(lamo_f1b))

    if not lamo_f1b:
        logging.error("No LAMO F1B images found under %s",
                      DATA_ROOT / "calibrated_raw_images" / "lamo")
        sys.exit(1)

    worklist: list[str] = []
    for img in lamo_f1b:
        out = output_root / f"{img.stem}_geometry.parquet"
        if out.exists():
            try:
                meta = pq.read_metadata(out)
                if REQUIRED_COLUMNS.issubset(set(meta.schema.names)):
                    pq.read_table(out, columns=["image_id"]).slice(0, 1)
                    continue
            except Exception:
                pass
        worklist.append(str(img))

    n_done = len(lamo_f1b) - len(worklist)
    logging.info("Already done: %d  To process: %d", n_done, len(worklist))

    if not worklist:
        logging.info("All LAMO F1B outputs verified. Nothing to do.")
        return

    with multiprocessing.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_process_image, worklist)

    failed = [r for r in results if not r[0]]
    logging.info("Finished. Successful: %d  Failed: %d",
                 len(results) - len(failed), len(failed))
    if failed:
        for _, path, err in failed:
            logging.error("FAILED: %s — %s", path, err)
        sys.exit(1)


if __name__ == "__main__":
    main()