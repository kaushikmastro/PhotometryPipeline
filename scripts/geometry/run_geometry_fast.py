import argparse
import logging
import multiprocessing
from pathlib import Path
import sys
import pyarrow.parquet as pq

from photometry_etl.etl.geometry_engine import GeometryEngine

_engine_worker = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REQUIRED_COLUMNS = {
    "image_id", "pixel_x", "pixel_y", "iof", 
    "incidence", "emission", "phase", "latitude", "longitude"
}

def _init_worker(data_root: str, metakernel_path: str, mode: str, output_subdir: str):
    global _engine_worker
    spice_method = "DSK/UNPRIORITIZED" if mode == "DSK256" else "ELLIPSOID"
    _engine_worker = GeometryEngine(
        data_root=data_root,
        metakernel_path=metakernel_path,
        surface_intercept_method=spice_method,
        output_subdir=output_subdir
    )

def _process_image_worker(image_path: str) -> tuple[bool, str, str | None]:
    global _engine_worker
    try:
        _engine_worker.compute_geometry(image_path)
        return True, image_path, None
    except Exception as exc:
        return False, image_path, str(exc)

def main():
    parser = argparse.ArgumentParser(description="Dawn FC Fast-Track Geometry")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--metakernel", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    # Isolate this run completely to avoid filesystem race conditions
    output_subdir = "04_geometry_tables_fast"
    target_output_root = Path(args.data_root) / output_subdir
    target_output_root.mkdir(parents=True, exist_ok=True)

    logging.info("FAST-TRACK: Targeting RC and SURVEY phases only.")
    
    # Explicitly pull ONLY RC and Survey phases
    input_images = []
    for phase in ["rc", "survey"]:
        phase_dir = Path(args.data_root) / "calibrated_raw_images" / phase
        input_images.extend(sorted(list(phase_dir.glob("**/*.IMG"))))

    worklist = []
    for img in input_images:
        phase_subdir = GeometryEngine._phase_subdir_from_image_path(img)
        expected_parquet = target_output_root / phase_subdir / f"{img.stem}_geometry.parquet"
        
        if expected_parquet.exists():
            try:
                meta = pq.read_metadata(expected_parquet)
                if REQUIRED_COLUMNS.issubset(set(meta.schema.names)):
                    continue
            except Exception:
                pass
        worklist.append(str(img))

    logging.info("Discovery complete. Total target images: %d. Queued: %d", len(input_images), len(worklist))

    if not worklist:
        logging.info("All fast-track targets already completed.")
        return

    with multiprocessing.Pool(
        processes=args.workers, 
        initializer=_init_worker, 
        initargs=(args.data_root, args.metakernel, args.mode, output_subdir)
    ) as pool:
        pool.map(_process_image_worker, worklist)

    logging.info("Fast-track processing run finished successfully.")

if __name__ == "__main__":
    main()