"""
Validation subset for the new per-pixel area schema (range_km, pixel_solid_angle_sr,
pixel_area_km2, projected_area_km2). Same DSK (vesta_gaskell_256_110825.bds) and
intercept method as the committed geometry/gaskell_dsk256_110825/survey/ product --
this is purely a schema regeneration for 6 hand-picked images, NOT a reprocessing of
the full committed set. Writes to a SEPARATE output_subdir so nothing committed is
touched or overwritten.

Image selection (from geometry/gaskell_dsk256_110825/survey/'s existing phase/emission
stats, queried via srun): spans phase angle 15-84 deg, plus one high-incidence
(mean_incidence ~76 deg) limb-heavy frame.
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

IMAGE_IDS = [
    "FC21B0005078_11230045329F1B",  # phase ~15 deg
    "FC21B0006187_11235143731F1B",  # phase ~35 deg
    "FC21B0005240_11230115842F1B",  # phase ~50 deg
    "FC21B0006362_11238050915F1B",  # phase ~70 deg
    "FC21B0006630_11239092544F1B",  # phase ~84 deg
    "FC21B0005650_11232170557F1B",  # phase ~37 deg, mean_incidence ~76 deg (limb-heavy)
]


def main():
    output_root = DATA_ROOT / OUTPUT_SUBDIR / "survey"
    output_root.mkdir(parents=True, exist_ok=True)

    engine = GeometryEngine(
        data_root=str(DATA_ROOT),
        metakernel_path=str(METAKERNEL),
        surface_intercept_method=SURFACE_METHOD,
        output_subdir=OUTPUT_SUBDIR,
    )

    ok, failed = 0, []
    for image_id in IMAGE_IDS:
        img_path = DATA_ROOT / "calibrated_raw_images" / "survey" / f"{image_id}.IMG"
        if not img_path.exists():
            failed.append((image_id, "raw image not found"))
            continue
        try:
            df = engine.compute_geometry(str(img_path))
            logging.info("OK: %s (%d rows)", image_id, len(df))
            ok += 1
        except Exception as exc:
            logging.error("FAILED: %s  error: %s", image_id, exc)
            failed.append((image_id, str(exc)))

    logging.info("Done. Success: %d  Failed: %d", ok, len(failed))
    for image_id, err in failed:
        logging.error("  FAILED: %s  — %s", image_id, err)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
