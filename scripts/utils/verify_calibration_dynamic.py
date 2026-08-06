import argparse
import logging
from pathlib import Path
import pandas as pd
from hapke_mcmc_package.etl.geometry_engine import GeometryEngine

def get_image_by_index(phase: str, index: int) -> Path:

    """Locate the N-th image in a given phase directory."""

    phase_dir = Path(f"data/calibrated_raw_images/{phase}")
    
    # Sort files to ensure index 0, 1, 2 are consistent
    images = sorted(list(phase_dir.glob("*.LBL")))
    if index >= len(images):
        raise IndexError(f"Phase {phase} only has {len(images)} images.")
    return images[index]

def main():
    parser = argparse.ArgumentParser(description="Validate calibration on a specific image.")
    parser.add_argument("--phase", required=True, choices=["rc", "hamo", "lamo", "survey"], help="Mission phase")
    parser.add_argument("--index", type=int, default=0, help="Index of the image to process (0-indexed)")
    args = parser.parse_args()

    # Engine setup
    engine = GeometryEngine(
        data_root="data",
        metakernel_path="data/spice_kernels/dawn_dynamic.tm",
        surface_intercept_method="DSK/UNPRIORITIZED",
    )

    img_path = get_image_by_index(args.phase, args.index)
    print(f"Processing: {img_path.name}")
    
    df = engine.compute_geometry(str(img_path))
    mean_iof = df['iof'].mean()
    
    print(f"\nResult for {args.phase.upper()} index {args.index}:")
    print(f"Mean I/F: {mean_iof:.4f}")
    
    # Physics sanity check
    if 0.15 <= mean_iof <= 0.25:
        print("STATUS: Physically Valid.")
    else:
        print("STATUS: Check F_solar/DSK geometry.")

if __name__ == "__main__":
    main()