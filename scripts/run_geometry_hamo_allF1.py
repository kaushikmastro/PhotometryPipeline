"""
HAMO all-F1-version-letter geometry grind — mission-science Gaskell DSK256
(vesta_gaskell_256_110825.bds).

Covers every PRODUCT_VERSION_ID under FILTER_NUMBER="1" as present on disk
for HAMO, not just F1B. Confirmed via primary PDS labels that FILTER_NUMBER="1"
for F1B, F1D, F1G (all genuine clear/broadband, differing only in
processing-version tag). HAMO's on-disk letters are B, C, D, E, F, G only
(no F1A/H/I present in this phase). F1I was checked and confirmed on the
LAMO product set, not HAMO — do not conflate the two phases' letter coverage.

Input:  01_calibrated_images/hamo/ — 5547 F1-any images (1089 F1B + 4458 other)
Output: 04_geometry_tables_dsk256_110825/hamo_allF1/  (SEPARATE from hamo/ —
        the validated F1B-only tables used in the committed Case 1 fit are
        left untouched)

DSK kernel must be referenced in dawn_dynamic.tm before this script runs.
Idempotent: skips parquets that already exist with valid schema. hamo_allF1/
is pre-seeded with hard copies of the 1089 validated F1B parquets from
hamo/ (see submit script), so the skip logic recognizes them as done and
only computes the ~4458 new non-B images. image_id (the output parquet
stem) is derived from the full filename stem, so copied F1B outputs are
byte-identical to what this script would have produced for those images.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hapke_mcmc_package.etl.geometry_engine import GeometryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_ROOT      = Path("/scratch/kaushim07/vesta_data")
METAKERNEL     = DATA_ROOT / "02_spice_kernels" / "dawn_dynamic.tm"
OUTPUT_SUBDIR  = "04_geometry_tables_dsk256_110825"
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
    output_root = DATA_ROOT / OUTPUT_SUBDIR / "hamo_allF1"
    output_root.mkdir(parents=True, exist_ok=True)

    hamo_allF1 = sorted(
        (DATA_ROOT / "01_calibrated_images" / "hamo").glob("*F1[A-Z].IMG")
    )
    logging.info("HAMO all-F1 images found: %d", len(hamo_allF1))

    if not hamo_allF1:
        logging.error("No HAMO F1-any images found under %s",
                      DATA_ROOT / "01_calibrated_images" / "hamo")
        sys.exit(1)

    worklist: list[str] = []
    for img in hamo_allF1:
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

    n_done = len(hamo_allF1) - len(worklist)
    logging.info("Already done: %d  To process: %d", n_done, len(worklist))

    if not worklist:
        logging.info("All HAMO all-F1 outputs verified. Nothing to do.")
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
