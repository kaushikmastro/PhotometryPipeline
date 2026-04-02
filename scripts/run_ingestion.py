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

from hapke_mcmc_package.etl.ingestion import DataManager


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    configure_logging()

    manifest_path = PROJECT_ROOT / "configs" / "survey_manifest.csv"
    data_root = PROJECT_ROOT / "data"

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

    manager = DataManager(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
    )

    if manager.validate_data_ready():
        logging.info("All required data already present. No download needed.")
        return 0

    logging.info("Missing data detected. Starting download workflow.")
    if not any(manager.spice_dir.glob("*.tm")):
        logging.info("No local metakernel found. Starting SPICE kernel download workflow.")
        manager.download_spice_kernels()

    manager.download_missing_data()

    if manager.validate_data_ready():
        logging.info("Download workflow completed successfully.")
        return 0

    logging.error("Download workflow finished with missing files still present.")
    return 5


if __name__ == "__main__":
    sys.exit(main())
