import argparse
import logging
import multiprocessing
from pathlib import Path
import sys
import pyarrow.parquet as pq

# Absolute package path resolution following formal repository architecture standards
from hapke_mcmc_package.etl.geometry_engine import GeometryEngine

# Thread-isolated storage reference for processing node workers
_engine_worker = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# CANONICAL SCHEMA TARGET REQUIREMENTS
REQUIRED_COLUMNS = {
    "image_id", "pixel_x", "pixel_y", "iof", 
    "incidence", "emission", "phase", "latitude", "longitude"
}

# EXPLICIT TOPOGRAPHY MODE MAPPING
SURFACE_METHOD_MAPPING = {
    "ELLIPSOID": "ELLIPSOID",
    "DSK256": "DSK/UNPRIORITIZED"
}


def _init_worker(data_root: str, metakernel_path: str, mode: str, output_subdir: str):
    """Instantiates a core-isolated instance of GeometryEngine per process slot."""
    global _engine_worker
    
    spice_method = SURFACE_METHOD_MAPPING.get(mode, "ELLIPSOID")
    
    _engine_worker = GeometryEngine(
        data_root=data_root,
        metakernel_path=metakernel_path,
        surface_intercept_method=spice_method,
        output_subdir=output_subdir
    )


def discover_worklist(input_images: list[Path], target_output_root: Path) -> list[str]:
    """Given candidate images and the output root, return the subset that still
    needs geometry computed: skip images with an existing, schema-valid,
    readable parquet; re-queue anything missing, corrupted, truncated, or
    schema-mismatched.
    """
    worklist: list[str] = []

    # DEEP METADATA AND SEMANTIC INTEGRITY AUDITING LAYER
    for img in input_images:
        phase_subdir = GeometryEngine._phase_subdir_from_image_path(img)

        img_parts_lower = [part.lower() for part in img.parts]
        if not any(phase in img_parts_lower for phase in ("rc", "survey", "hamo", "lamo")):
            logging.warning("File path context lacks structural phase designations. Default routing to 'survey': %s", img)

        expected_parquet = target_output_root / phase_subdir / f"{img.stem}_geometry.parquet"

        if expected_parquet.exists():
            try:
                meta = pq.read_metadata(expected_parquet)
                existing_columns = set(meta.schema.names)

                if REQUIRED_COLUMNS.issubset(existing_columns):
                    # Native PyArrow slice read forces page evaluation without loading full tables into RAM
                    pq.read_table(expected_parquet, columns=["image_id"]).slice(0, 1)
                    continue
                else:
                    logging.warning("Schema mismatch verified on storage layer for file: %s. Queueing for recovery overwrite.", expected_parquet)
            except Exception as exc:
                logging.warning(
                    "Truncated, partial, or corrupted file trace intercepted at destination path: %s. "
                    "Error context: %s. Queueing for generation overwrite.", expected_parquet, exc
                )

        worklist.append(str(img))

    return worklist


def _process_image_worker(image_path: str) -> tuple[bool, str, str | None]:
    """
    Executes geometry tracking across worker threads. 
    Returns a structured status payload back to the main coordination loop.
    """
    global _engine_worker
    try:
        _engine_worker.compute_geometry(image_path)
        return True, image_path, None
    except Exception as exc:
        logging.error("Catastrophic error encountered during core tracking execution on target: %s", 
                      image_path, exc_info=True)
        return False, image_path, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Dawn FC Geometry ETL Pipeline Execution Wrapper")
    
    # Dual CLI Flag Support: Gracefully handles both hyphens and underscores for cross-tooling safety
    parser.add_argument(
        "--data-root", "--data_root", 
        dest="data_root", 
        type=str, 
        required=True, 
        help="Path to absolute data storage root"
    )
    parser.add_argument("--metakernel", type=str, required=True, help="Path to targeted fallback SPICE metakernel")
    parser.add_argument("--workers", type=int, default=4, help="Total execution process thread count pools")
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(SURFACE_METHOD_MAPPING.keys()),
        required=True,
        help="Topography calculation style option"
    )
    parser.add_argument(
        "--output-subdir", "--output_subdir",
        dest="output_subdir",
        type=str,
        required=True,
        help="Output directory name under data-root for geometry tables "
             "(e.g. geometry/dsk256 to match the committed mission-science "
             "baseline). No default: the destination must always be explicit."
    )
    parser.add_argument(
        "--image-list", "--image_list",
        dest="image_list",
        type=str,
        default=None,
        help="Optional path to a text file of explicit .IMG file paths (one per line) to "
             "restrict this run to. Still subject to the same skip-check as a full scan. "
             "If omitted, all *.IMG files under data-root are scanned (existing behavior)."
    )
    args = parser.parse_args()

    target_output_root = Path(args.data_root) / args.output_subdir
    target_output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Initializing multi-core processing array under configuration mode: %s", args.mode)
    logging.info("Destination directory mapped for structural output generation: %s", target_output_root)

    if args.image_list:
        with open(args.image_list) as f:
            input_images = sorted(Path(line.strip()) for line in f if line.strip())
        logging.info("Restricting run to explicit image list: %s (%d paths)", args.image_list, len(input_images))
    else:
        # Scans the calibrated data repository path exclusively to avoid tracing raw instrument inputs
        input_images = sorted(list(Path(args.data_root).glob("calibrated_raw_images/**/*.IMG")))

    worklist = discover_worklist(input_images, target_output_root)

    logging.info("Discovery audit complete. Total calibrated images located: %d. Active items passed to processing queue: %d",
                 len(input_images), len(worklist))

    if not worklist:
        logging.info("All outputs verified complete and schema-conforming on target layout. Pipeline finished.")
        return

    # Initialize pool execution context
    with multiprocessing.Pool(
        processes=args.workers, 
        initializer=_init_worker, 
        initargs=(args.data_root, args.metakernel, args.mode, args.output_subdir)
    ) as pool:
        results = pool.map(_process_image_worker, worklist)

    # PROCESS MONITORING STATUS INSPECTION
    failed_jobs = [item for item in results if not item[0]]
    successful_count = len(results) - len(failed_jobs)
    
    logging.info("ETL Tracking Sequence Finalized. Successful files: %d, Terminal task errors: %d", 
                 successful_count, len(failed_jobs))
    
    if failed_jobs:
        logging.critical("Pipeline processing run finished with system failures. Aborting cluster tracking state.")
        for _, path, error_msg in failed_jobs:
            logging.error("Target item failure summary -> File: %s | Trace context: %s", path, error_msg)
        
        sys.exit(1)


if __name__ == "__main__":
    main()