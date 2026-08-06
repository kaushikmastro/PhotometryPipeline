"""
Survey F1B geometry grind — triaxial ELLIPSOID shape model.

Input:  calibrated_raw_images/survey/ — 844 genuine Survey F1B images
        (846 F1B *.IMG on disk; 1 approach image DOY 11123 excluded)
Output: geometry/ellipsoid/survey/

Uses SPICE surface_intercept_method="ELLIPSOID" (analytical, no DSK needed).
Body radii from PCK (pck00010.tpc): BODY2000004_RADII = (289, 280, 229) km
(DAWN-derived, Russell et al. 2011).

Idempotent: skips parquets that already exist with valid schema.
"""
from __future__ import annotations

import argparse
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
OUTPUT_SUBDIR  = "geometry/ellipsoid"
SURFACE_METHOD = "ELLIPSOID"
F_SOLAR        = 892.0
N_WORKERS      = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

REQUIRED_COLUMNS = {
    "image_id", "pixel_x", "pixel_y", "iof",
    "incidence", "emission", "phase", "latitude", "longitude",
}

_engine_worker: GeometryEngine | None = None


def is_survey_image(img_path: Path) -> bool:
    """Return False for the one approach image (DOY 11123); True for all Survey images."""
    stem = img_path.stem  # e.g. FC21B0001898_11123133516F1B
    parts = stem.split("_")
    if len(parts) < 2:
        return False
    yyddd = parts[1][:5]  # "11123" = year 2011, DOY 123 (approach)
    return yyddd != "11123"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Process the first image only; exits 0 on success")
    parser.add_argument("--image", type=str, default=None,
                        help="Process a single image file (overrides glob; for smoke testing)")
    args = parser.parse_args()

    output_root = DATA_ROOT / OUTPUT_SUBDIR / "survey"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.image:
        survey_f1b = [Path(args.image)]
        logging.info("--image override: single file: %s", args.image)
    else:
        all_f1b = sorted(
            (DATA_ROOT / "calibrated_raw_images" / "survey").glob("*F1B*.IMG")
        )
        logging.info("Survey F1B images found (pre-filter): %d", len(all_f1b))

        survey_f1b = [img for img in all_f1b if is_survey_image(img)]
        n_excluded = len(all_f1b) - len(survey_f1b)
        if n_excluded:
            logging.info("Excluded %d approach_image(s) (DOY 11123)", n_excluded)
        logging.info("Survey F1B images after DOY filter: %d", len(survey_f1b))

    if not survey_f1b:
        logging.error("No images to process")
        sys.exit(1)

    if args.smoke_test and not args.image:
        survey_f1b = survey_f1b[:1]
        logging.info("SMOKE TEST MODE: processing 1 image only: %s", survey_f1b[0])

    worklist: list[str] = []
    for img in survey_f1b:
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

    n_done = len(survey_f1b) - len(worklist)
    logging.info("Already done: %d  To process: %d", n_done, len(worklist))

    if not worklist:
        logging.info("All Survey F1B ellipsoid outputs verified. Nothing to do.")
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