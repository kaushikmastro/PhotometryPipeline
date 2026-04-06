#!/usr/bin/env python3
"""
Update SPICE kernels: download reconstructed SPK kernels and generate dynamic metakernel.

This script:
1. Discovers and downloads reconstructed spacecraft SPK kernels
2. Ensures de421.bsp (planetary positions) is available
3. Generates a dynamic metakernel (dawn_dynamic.tm) that includes all available kernels
4. Reports kernel inventory and readiness status
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hapke_mcmc_package.etl.ingestion import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    data_root = "/home/kaushim07/photometry_mcmc_env/data"
    manifest_path = "/home/kaushim07/photometry_mcmc_env/data/manifest.csv"
    
    logger.info("=" * 70)
    logger.info("SPICE Kernel Update: Downloading Reconstructed SPK & Dynamic Metakernel")
    logger.info("=" * 70)
    
    dm = DataManager(manifest_path, data_root)
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"SPICE directory: {dm.spice_dir}")
    
    # Current kernel inventory
    logger.info("\nCurrent kernel inventory:")
    for ext, desc in [
        (".bsp", "Ephemeris (SPK)"),
        (".bc", "Attitude (CK)"),
        (".tm", "Metakernels"),
        (".tpc", "Bodies/parameters (PCK)"),
        (".tf", "Frame kernels"),
        (".ti", "Instrument kernels"),
        (".tls", "Leap seconds (LSK)"),
        (".tsc", "Spacecraft clock (SCLK)"),
    ]:
        files = sorted(dm.spice_dir.glob(f"*{ext}"))
        logger.info(f"  {desc:25s}: {len(files):3d} files")
    
    # Download kernels and generate dynamic metakernel
    logger.info("\nDownloading kernels...")
    success = dm.download_spice_kernels()
    
    if not success:
        logger.error("Kernel download encountered issues (see warnings above)")
    else:
        logger.info("Kernel download completed successfully")
    
    # New kernel inventory after download
    logger.info("\nKernel inventory after update:")
    for ext, desc in [
        (".bsp", "Ephemeris (SPK)"),
        (".bc", "Attitude (CK)"),
        (".tm", "Metakernels"),
        (".tpc", "Bodies/parameters (PCK)"),
        (".tf", "Frame kernels"),
        (".ti", "Instrument kernels"),
        (".tls", "Leap seconds (LSK)"),
        (".tsc", "Spacecraft clock (SCLK)"),
    ]:
        files = sorted(dm.spice_dir.glob(f"*{ext}"))
        logger.info(f"  {desc:25s}: {len(files):3d} files")
    
    # Verify dynamic metakernel exists
    dynamic_mk = dm.spice_dir / "dawn_dynamic.tm"
    if dynamic_mk.exists():
        mk_size = dynamic_mk.stat().st_size
        logger.info(f"\n✓ Dynamic metakernel created: {dynamic_mk.name} ({mk_size} bytes)")
        
        # Show first 20 lines of metakernel for audit.
        with open(dynamic_mk, 'r') as f:
            lines = f.readlines()
            logger.info(f"  First 20 lines:")
            for line in lines[:20]:
                logger.info(f"    {line.rstrip()}")
    else:
        logger.error("✗ Dynamic metakernel was not generated!")
        return 1
    
    # Verify de421.bsp
    de421 = dm.spice_dir / "de421.bsp"
    if de421.exists():
        size = de421.stat().st_size / (1024**2)
        logger.info(f"✓ Planetary ephemeris (de421.bsp): {size:.1f} MB")
    else:
        logger.warning("⚠ de421.bsp not present (may cause issues)")
    
    logger.info("\n" + "=" * 70)
    logger.info("SPICE kernel update complete!")
    logger.info("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
