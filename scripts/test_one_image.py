#!/usr/bin/env python3
"""Test geometry computation on exactly one image to verify lat/lon population."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hapke_mcmc_package.etl.geometry_engine import GeometryEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometry on exactly one image and validate lat/lon population."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Absolute path to the exact IMG file to process.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root that contains 01_calibrated_images and 02_spice_kernels.",
    )
    parser.add_argument(
        "--metakernel",
        type=Path,
        default=None,
        help="Optional explicit metakernel path. Defaults to <data_root>/02_spice_kernels/dawn_dynamic.tm",
    )
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.image is not None:
        image_path = Path(args.image).expanduser().resolve()
    else:
        # Backward-compatible default.
        image_path = (
            PROJECT_ROOT
            / "data"
            / "01_calibrated_images"
            / "survey"
            / "FC21B0003931_11223181258F1F.IMG"
        ).resolve()

    if args.data_root is not None:
        data_root = Path(args.data_root).expanduser().resolve()
    elif args.image is not None:
        # Infer /.../vesta_data from /.../vesta_data/01_calibrated_images/<phase>/<file>.IMG
        if len(image_path.parents) < 3:
            raise ValueError(f"Cannot infer data root from image path: {image_path}")
        data_root = image_path.parents[2]
    else:
        data_root = (PROJECT_ROOT / "data").resolve()

    if args.metakernel is not None:
        metakernel_path = Path(args.metakernel).expanduser().resolve()
    else:
        metakernel_path = data_root / "02_spice_kernels" / "dawn_dynamic.tm"

    return image_path, data_root, metakernel_path


def main() -> int:
    args = _parse_args()
    first_image, data_root, metakernel_path = _resolve_inputs(args)

    if not first_image.exists():
        logging.error(f"Test image not found: {first_image}")
        return 1
    
    if not metakernel_path.exists():
        logging.error(f"Metakernel not found: {metakernel_path}")
        return 2
    
    logging.info(f"Testing geometry computation on single image: {first_image}")
    logging.info(f"Using data root: {data_root}")
    logging.info(f"Using metakernel: {metakernel_path}")
    
    try:
        engine = GeometryEngine(
            data_root=str(data_root),
            metakernel_path=str(metakernel_path),
            body_fixed_frame="IAU_VESTA",
        )
        
        df = engine.compute_geometry(str(first_image))
        
        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        logging.info("\nData info:")
        df.info()
        
        # Check lat/lon population
        lat_nonnull = df['latitude'].notna().sum()
        lon_nonnull = df['longitude'].notna().sum()
        
        logging.info(f"\nLatitude non-null count: {lat_nonnull} / {len(df)}")
        logging.info(f"Longitude non-null count: {lon_nonnull} / {len(df)}")
        
        if lat_nonnull == 0:
            logging.error("FAIL: Latitude column is completely empty!")
            return 3
        if lon_nonnull == 0:
            logging.error("FAIL: Longitude column is completely empty!")
            return 4
        
        # Show sample values
        valid_mask = df['latitude'].notna()
        sample_df = df[valid_mask].head(3)[['image_id', 'pixel_x', 'pixel_y', 'latitude', 'longitude', 'incidence', 'emission', 'phase']]
        logging.info("\nSample rows with valid lat/lon:")
        logging.info(f"\n{sample_df}")
        
        # Check lat range
        lat_min = df['latitude'].min()
        lat_max = df['latitude'].max()
        lon_min = df['longitude'].min()
        lon_max = df['longitude'].max()
        
        logging.info(f"\nLatitude range: {lat_min:.2f}° to {lat_max:.2f}°")
        logging.info(f"Longitude range: {lon_min:.2f}° to {lon_max:.2f}°")
        
        if lat_min < -90 or lat_max > 90:
            logging.error(f"FAIL: Latitude out of valid range [-90, 90]: min={lat_min:.2f}, max={lat_max:.2f}")
            return 5
        
        logging.info("\nSUCCESS: Lat/lon populated correctly!")
        return 0
        
    except Exception as e:
        logging.error(f"Error during test: {e}", exc_info=True)
        return 10
    finally:
        import spiceypy
        spiceypy.kclear()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
