"""
Geometry re-run for shape-model comparison: vesta_gaskell_256_110825.bds
(Gaskell SPC, Q=256 downsampled from Q=512, provided to NAIF 2011-08-25)
vs the original pre-Dawn model (vesta_gaskell_256_PRELIM_preDawn.bds).

Processes ONLY Survey F1B images (845 images) — the exact input set
used in the main fit. Writes to data/04_geometry_tables_dsk256_110825/survey/
so the new geometry is isolated from the existing tables and does not
overwrite any prior results.

The metakernel (dawn_dynamic.tm) has been updated to load
vesta_gaskell_256_110825.bds before this script is run.
"""
import logging
import multiprocessing
from pathlib import Path
import sys
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hapke_mcmc_package.etl.geometry_engine import GeometryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_ROOT      = Path("/scratch/kaushim07/vesta_data")
METAKERNEL     = DATA_ROOT / "02_spice_kernels" / "dawn_dynamic.tm"
OUTPUT_SUBDIR  = "04_geometry_tables_dsk256_110825"
SURFACE_METHOD = "DSK/UNPRIORITIZED"
N_WORKERS      = int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 16))

REQUIRED_COLUMNS = {
    "image_id", "pixel_x", "pixel_y", "iof",
    "incidence", "emission", "phase", "latitude", "longitude"
}

_engine_worker = None


def _init_worker():
    global _engine_worker
    _engine_worker = GeometryEngine(
        data_root=str(DATA_ROOT),
        metakernel_path=str(METAKERNEL),
        surface_intercept_method=SURFACE_METHOD,
        output_subdir=OUTPUT_SUBDIR,
    )


def _process_image(image_path: str) -> tuple[bool, str, str | None]:
    global _engine_worker
    try:
        _engine_worker.compute_geometry(image_path)
        return True, image_path, None
    except Exception as exc:
        logging.error("Failed: %s  error: %s", image_path, exc)
        return False, image_path, str(exc)


def main():
    output_root = DATA_ROOT / OUTPUT_SUBDIR / "survey"
    output_root.mkdir(parents=True, exist_ok=True)

    # Only Survey F1B images — the exact set used in the main fit
    survey_f1b = sorted((DATA_ROOT / "01_calibrated_images" / "survey").glob("*F1B*.IMG"))
    logging.info("Survey F1B images found: %d", len(survey_f1b))

    worklist = []
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

    logging.info("Already done: %d  To process: %d", len(survey_f1b) - len(worklist), len(worklist))

    if not worklist:
        logging.info("All Survey F1B geometry complete. Nothing to do.")
        return

    with multiprocessing.Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
    ) as pool:
        results = pool.map(_process_image, worklist)

    ok = sum(1 for r in results if r[0])
    fail = [r for r in results if not r[0]]
    logging.info("Done. Success: %d  Failed: %d", ok, len(fail))
    for _, path, err in fail:
        logging.error("  FAILED: %s  — %s", path, err)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
