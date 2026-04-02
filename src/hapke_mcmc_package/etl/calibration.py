import numpy as np
import planetaryimage
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageCalibrator:
    """
    Handles the cleaning and standardization of Dawn FC2 Level 1c images.
    """

    def __init__(self):
        """
        Initializes the ImageCalibrator.
        """
        logging.info("ImageCalibrator initialized.")

    def clean_level1c_image(self, img_file_path: str) -> np.ndarray:
        """
        Reads a Dawn FC2 Level 1c PDS .IMG file, cleans invalid pixel values,
        and returns a standardized numpy array.

        Args:
            img_file_path (str): The file path to the PDS .IMG file.

        Returns:
            np.ndarray: A 2D numpy array of type float32 containing the I/F values.
                        Returns an empty array if cleaning fails.
        """
        logging.info(f"Cleaning image file: {img_file_path}")
        
        file_path = Path(img_file_path)
        if not file_path.exists():
            logging.error(f"Image file not found: {img_file_path}")
            return np.array([], dtype=np.float32)

        try:
            # 1. Use planetaryimage to open the PDS file
            pds_image = planetaryimage.PDS3Image.open(str(file_path))
            
            # 2. Extract the 2D data array and ensure it's float32
            # We create a copy to avoid modifying the original object's data
            image_data = pds_image.image.astype(np.float32)
            
            # 3. Identify and replace invalid pixel flags with np.nan
            # For PDS files, invalid data is often represented by large negative values.
            # A simple check for values less than 0 is a robust starting point.
            # More specific flags like `pds_image.label['IMAGE']['MISSING_CONSTANT']`
            # could be used if available and necessary.
            invalid_pixels_mask = image_data < 0
            
            num_invalid = np.sum(invalid_pixels_mask)
            if num_invalid > 0:
                logging.info(f"Found and replaced {num_invalid} invalid pixels.")
                image_data[invalid_pixels_mask] = np.nan
            
            logging.info(f"Successfully cleaned {img_file_path}. Image shape: {image_data.shape}")
            
            # 4. Return the clean numpy array
            return image_data

        except Exception as e:
            logging.error(f"Failed to read or clean PDS file {img_file_path}: {e}")
            return np.array([], dtype=np.float32)

if __name__ == '__main__':
    # This is an example of how to use the ImageCalibrator.
    # It requires a dummy PDS file to be created or a real one to be present.
    
    print("Setting up dummy files for testing...")
    # To run this, you would need a sample PDS .IMG file.
    # Since creating a valid PDS file from scratch is complex, we will
    # just illustrate the usage.
    
    dummy_img_path = "test_image.IMG"
    print(f"To run this example, place a valid PDS .IMG file at: {dummy_img_path}")

    # Example usage (commented out to prevent execution without a real PDS file)
    # try:
    #     # You would need to create a dummy PDS file for this to work.
    #     # For now, we just demonstrate the class usage pattern.
    #     calibrator = ImageCalibrator()
    #     # cleaned_data = calibrator.clean_level1c_image(dummy_img_path)
    #     # if cleaned_data.size > 0:
    #     #     print("Image cleaned successfully.")
    #     #     print(f"Shape: {cleaned_data.shape}")
    #     #     print(f"Data type: {cleaned_data.dtype}")
    #     #     # Check if NaNs are present (they should be if invalid pixels were in the dummy file)
    #     #     print(f"Contains NaNs: {np.isnan(cleaned_data).any()}")
    # except Exception as e:
    #     print(f"Could not run ImageCalibrator example. Error: {e}")

    print("ImageCalibrator class is defined and ready.")
