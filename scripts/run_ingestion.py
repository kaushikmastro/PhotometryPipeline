#!/usr/bin/env python3
"""SLURM entrypoint for validating and downloading Dawn Vesta ingestion data."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/kaushim07/photometry_mcmc_env")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.ingestion import DataManager  # noqa: E402


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    configure_logging()

    manifest_path = PROJECT_ROOT / "configs" / "survey_manifest.csv"
    data_root = PROJECT_ROOT / "data"
    dtm_dir = data_root / "03_dtm"
    geometry_tables_dir = data_root / "04_geometry_tables"

    if not manifest_path.exists():
        logging.error("Manifest not found: %s", manifest_path)
        return 2

    # Safety guard for HPC policy: data/ should resolve to /scratch.
    try:
        resolved_data_root = data_root.resolve(strict=True)
        if not str(resolved_data_root).startswith("/scratch/"):
            logging.error(
                "Refusing download: data/ resolves to %s, expected /scratch/...",
                resolved_data_root,
            )
            return 3
    except FileNotFoundError:
        logging.error("Data root does not exist: %s", data_root)
        return 4

    # Prepare next-phase ETL directories on scratch-backed storage.
    dtm_dir.mkdir(parents=True, exist_ok=True)
    geometry_tables_dir.mkdir(parents=True, exist_ok=True)

    manager = DataManager(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
    )

    logging.info("Step 1/3: Ensuring DTM foundation is present.")
    try:
        manager.ensure_dtm_foundation()
    except RuntimeError as exc:
        logging.critical("DTM foundation step failed: %s", exc)
        return 6

    if (
        not any(dtm_dir.glob("*.IMG"))
        or not (dtm_dir / "dawn_vesta_SPG20160901.lbl").exists()
        or not (dtm_dir / "dawn_vesta_SPG20160901.tpc").exists()
    ):
        print(
            "WARNING: DTM foundation is incomplete. Geometry Engine will fall back to Ellipsoid model, which is insufficient for Hapke Roughness."
        )

    if manager.validate_data_ready():
        logging.info("All required data already present. No download needed.")
        return 0

    logging.info("Step 2/3: Missing data detected. Starting SPICE/image download workflow.")
    if not any(manager.spice_dir.glob("*.tm")):
        logging.info("No local metakernel found. Starting SPICE kernel download workflow.")
        manager.download_spice_kernels()

    manager.download_missing_data()

    logging.info("Step 3/3: Re-validating data readiness.")
    if manager.validate_data_ready():
        logging.info("Download workflow completed successfully.")
        return 0

    logging.error("Download workflow finished with missing files still present.")
    return 5


if __name__ == "__main__":
    sys.exit(main())
