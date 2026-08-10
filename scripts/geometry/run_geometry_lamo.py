"""
LAMO F1B geometry grind — preliminary Gaskell DSK256, f_solar=892.

Processes only LAMO F1B images (clear filter, disk-resolved).
Writes to data/04_geometry_tables_fast/lamo/ — same directory tree as
the Survey, RC, and HAMO tables already there.

CK coverage: precision FC2 CK (Jul 2011 – Jul 2012) + quicklook CKs
through Jan 6 2013 fully cover the LAMO epoch (DOY 11346–12167,
Dec 2011 – Jun 2012). No additional CK needed.

Input: /scratch/.../calibrated_raw_images/lamo/ (real Vesta LAMO data)
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from photometry_etl.etl.geometry_engine import GeometryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_ROOT      = Path("/scratch/kaushim07/vesta_data")
METAKERNEL     = DATA_ROOT / "spice_kernels" / "dawn_dynamic.tm"
OUTPUT_SUBDIR  = "04_geometry_tables_fast"
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
        logging.error(
            "No LAMO F1B images found under %s",
            DATA_ROOT / "calibrated_raw_images" / "lamo",
        )
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

    with multiprocessing.Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
    ) as pool:
        results = pool.map(_process_image, worklist)

    n_ok   = sum(1 for ok, _, _ in results if ok)
    n_fail = sum(1 for ok, _, _ in results if not ok)
    logging.info("Finished: %d succeeded, %d failed", n_ok, n_fail)
    if n_fail > 0:
        for ok, path, err in results:
            if not ok:
                logging.error("  FAILED: %s  %s", path, err)
        sys.exit(1)


if __name__ == "__main__":
    main()