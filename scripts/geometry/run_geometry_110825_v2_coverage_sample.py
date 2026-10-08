"""
Disk-coverage validation sample: 2 F1B images each from RC, HAMO, LAMO (Survey already
has 2 regenerated from the earlier validation subset -- reused, not reprocessed).
Same DSK (vesta_gaskell_256_110825.bds) and intercept method as the committed
geometry/gaskell_dsk256_110825/ product. Writes to the SEPARATE
geometry/gaskell_dsk256_110825_v2/{rc,hamo,lamo}/ output -- nothing committed touched.

Purpose: Task 1 disk-coverage report (per mission phase, does any phase capture the
whole illuminated disk in a single frame?), not a phase-curve sweep -- images picked
for phase/mission-phase representation, not diversity within a phase.
"""
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from photometry_etl.etl.geometry_engine import GeometryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_ROOT      = Path("/scratch/kaushim07/vesta_data")
METAKERNEL     = DATA_ROOT / "spice_kernels" / "dawn_dynamic.tm"
OUTPUT_SUBDIR  = "geometry/gaskell_dsk256_110825_v2"
SURFACE_METHOD = "DSK/UNPRIORITIZED"

IMAGES_BY_PHASE = {
    "rc":    ["FC21B0002309_11181045100F1B", "FC21B0002465_11181103102F1B"],
    "hamo":  ["FC21B0006742_11246184031F1B", "FC21B0027624_12179233227F1B"],
    "lamo":  ["FC21B0014114_11312011632F1B", "FC21B0024628_12096152844F1B"],
}


def main():
    engine = GeometryEngine(
        data_root=str(DATA_ROOT),
        metakernel_path=str(METAKERNEL),
        surface_intercept_method=SURFACE_METHOD,
        output_subdir=OUTPUT_SUBDIR,
    )

    ok, failed = 0, []
    for phase_name, image_ids in IMAGES_BY_PHASE.items():
        output_root = DATA_ROOT / OUTPUT_SUBDIR / phase_name
        output_root.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            img_path = DATA_ROOT / "calibrated_raw_images" / phase_name / f"{image_id}.IMG"
            if not img_path.exists():
                failed.append((image_id, "raw image not found"))
                continue
            try:
                df = engine.compute_geometry(str(img_path))
                logging.info("OK: %s/%s (%d rows)", phase_name, image_id, len(df))
                ok += 1
            except Exception as exc:
                logging.error("FAILED: %s/%s  error: %s", phase_name, image_id, exc)
                failed.append((image_id, str(exc)))

    logging.info("Done. Success: %d  Failed: %d", ok, len(failed))
    for image_id, err in failed:
        logging.error("  FAILED: %s  — %s", image_id, err)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
