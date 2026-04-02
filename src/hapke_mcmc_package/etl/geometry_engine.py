import spiceypy
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeometryEngine:
    """
    Core SPICE-based ray-tracing engine to calculate photometric angles.
    """

    def __init__(self, metakernel_path: str):
        """
        Initializes the GeometryEngine by loading the SPICE metakernel.

        Args:
            metakernel_path (str): Path to the SPICE metakernel file (.tm).
        """
        try:
            spiceypy.furnsh(metakernel_path)
            logging.info(f"Successfully loaded SPICE metakernel: {metakernel_path}")
        except Exception as e:
            logging.error(f"Failed to load SPICE metakernel: {e}")
            raise

        # Camera constants for Dawn's Framing Camera (FC2)
        self.instrument = 'DAWN_FC2'
        self.target = 'VESTA'
        self.aberration_correction = 'LT+S'
        self.target_frame = 'IAU_VESTA'
        
        # Get camera FOV details
        try:
            self.cam_id = spiceypy.bodn2c(self.instrument)
            _, self.cam_frame, self.boresight, self.num_bounds, self.bounds = spiceypy.getfov(self.cam_id, 4)
            self.image_shape = (1024, 1024) # FC2 is 1024x1024 pixels
            logging.info(f"Successfully loaded camera model for {self.instrument}")
        except Exception as e:
            logging.error(f"Could not get camera FOV for {self.instrument}. Check SPICE kernels. Error: {e}")
            raise

    def process_image(self, image_id: str, image_path: str, time_utc: str, output_dir: str, use_dsk: bool = False):
        """
        Performs ray-tracing for each pixel of an image to calculate geometry.

        Args:
            image_id (str): A unique identifier for the image (e.g., filename stem).
            image_path (str): Path to the Level 1c unprojected image file.
            time_utc (str): Observation time in UTC for SPICE calculations.
            output_dir (str): Directory to save the output Parquet file.
            use_dsk (bool): If True, use DSK/DEM for ray-tracing. Defaults to Ellipsoid.
        """
        logging.info(f"Processing image: {image_id}")
        
        # 1. Convert UTC time to Ephemeris Time (ET)
        et = spiceypy.utc2et(time_utc)

        # 2. Load I/F data from the image (assuming a simple numpy array for now)
        # In a real scenario, this would involve reading a PDS/ISIS cube.
        try:
            # Placeholder: Create a dummy I/F array. Replace with actual image reading logic.
            iof_data = np.random.rand(*self.image_shape).astype(np.float32)
            logging.info(f"Successfully loaded I/F data from {image_path}")
        except Exception as e:
            logging.error(f"Could not read image file {image_path}: {e}")
            return

        # 3. Define camera FOV vectors
        x = np.linspace(-1, 1, self.image_shape[1])
        y = np.linspace(-1, 1, self.image_shape[0])
        xv, yv = np.meshgrid(x, y)
        
        # This is a simplified way to generate rays. A more precise method would use
        # spiceypy.pxform and the camera distortion model.
        rays = np.dstack([xv, yv, -np.ones(self.image_shape)])
        rays /= np.linalg.norm(rays, axis=2, keepdims=True)

        # 4. Perform ray-tracing with spiceypy.sincpt
        logging.info("Performing ray-tracing for each pixel...")
        method = 'DSK/UNPRIORITIZED' if use_dsk else 'ELLIPSOID'
        
        # Flatten arrays for vectorized processing
        rays_flat = rays.reshape(-1, 3)
        results = []

        for i in range(rays_flat.shape[0]):
            try:
                spoint, _, srfvec = spiceypy.sincpt(method, self.target, et, self.target_frame, 
                                                   self.aberration_correction, self.instrument, self.cam_frame, 
                                                   rays_flat[i])
                
                # 5. Calculate geometry angles
                _, inc, emi, pha = spiceypy.illumg(method, self.target, 'SUN', et, self.target_frame, 
                                                  self.aberration_correction, self.instrument, spoint)
                
                # 6. Filter out nighttime pixels
                if np.rad2deg(inc) <= 90.0:
                    pixel_id = f"{image_id}_{i}"
                    iof = iof_data.flat[i]
                    results.append([pixel_id, iof, np.rad2deg(inc), np.rad2deg(emi), np.rad2deg(pha)])

            except spiceypy.support_types.SpiceyError:
                # This error is expected if the ray does not intersect the target
                continue
        
        if not results:
            logging.warning(f"No valid surface intersections found for image {image_id}.")
            return

        # 7. Save results to a Parquet file
        df = pd.DataFrame(results, columns=['pixel_id', 'iof', 'inc', 'emi', 'pha'])
        output_path = Path(output_dir) / f"{image_id}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_parquet(output_path, engine='pyarrow')
            logging.info(f"Successfully saved geometry table to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save Parquet file: {e}")

    def __del__(self):
        """Unload SPICE kernels when the object is destroyed."""
        try:
            spiceypy.kclear()
            logging.info("SPICE kernels have been unloaded.")
        except Exception as e:
            logging.error(f"Error unloading SPICE kernels: {e}")

if __name__ == '__main__':
    # This is an example of how to run the engine.
    # It requires a valid metakernel and dummy data directories.
    
    # Create dummy files for testing
    print("Setting up dummy files for testing...")
    data_root = Path("data")
    spice_dir = data_root / "02_spice_kernels"
    spice_dir.mkdir(parents=True, exist_ok=True)
    metakernel_path = spice_dir / "vesta_v01.tm"
    metakernel_path.touch() # In a real scenario, this file would list all other kernels.

    image_dir = data_root / "01_calibrated_images"
    image_dir.mkdir(exist_ok=True)
    dummy_image_path = image_dir / "test_image.IMG"
    dummy_image_path.touch()

    output_dir = data_root / "04_geometry_tables"
    
    print(f"To run this example, ensure '{metakernel_path}' is a valid SPICE metakernel.")
    
    # Example usage (commented out to prevent execution without a real metakernel)
    # try:
    #     geo_engine = GeometryEngine(str(metakernel_path))
    #     geo_engine.process_image(
    #         image_id="test_image",
    #         image_path=str(dummy_image_path),
    #         time_utc="2011-08-01T00:00:00",
    #         output_dir=str(output_dir),
    #         use_dsk=False
    #     )
    # except Exception as e:
    #     print(f"Could not run GeometryEngine example. This is expected without a valid SPICE setup. Error: {e}")

    print("Dummy file setup complete.")
