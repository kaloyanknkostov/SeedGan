import os

import cv2
import numpy as np

# --- Configuration ---
# Get the absolute path to this script file
# Note: __file__ only works when run as a script, not in some IDEs/notebooks
# If this fails, replace BASE_DIR with a hardcoded absolute path.
try:
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    BASE_DIR = os.path.dirname(script_dir)
except NameError:
    # Fallback for environments where __file__ is not defined
    print(
        "Warning: __file__ is not defined. Using current working directory as BASE_DIR."
    )
    print("Please ensure this is correct or hardcode your BASE_DIR path.")
    BASE_DIR = os.path.abspath(
        os.path.join(os.getcwd(), "..")
    )  # Assumes script is in a 'scripts' folder
    #
    # --- OR Hardcode your path like this: ---
    # BASE_DIR = "/home/kaloyan/Code/Thesis/SeedGan"
    # print(f"Using hardcoded BASE_DIR: {BASE_DIR}")


# Set input and output directories based on the base project path
# /home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled4x
INPUT_DIR = os.path.join(BASE_DIR, "data", "crops_with_background_scaled4x")

# /home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled_256x256_real-esrgan
OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "crops_with_background_scaled_256x256_real-esrgan"
)

# Define the target size (width, height)
TARGET_SIZE = (256, 256)

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# --- End Configuration ---


def main():
    """
    Reads images and resizes them all to a fixed 256x256 size,
    using the best interpolation method for upscaling vs. downscaling.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Reading images from: {INPUT_DIR}")
    print(f"Saving resized images to: {OUTPUT_DIR}")

    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        print(f"\n--- ERROR ---")
        print(f"Input directory not found at: {INPUT_DIR}")
        print(f"Please check the path and make sure the previous script ran.")
        return

    processed_count = 0

    # Iterate through all files in the input directory
    for image_filename in os.listdir(INPUT_DIR):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        try:
            # --- 1. Load the image ---
            image_path = os.path.join(INPUT_DIR, image_filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not read {image_filename}. Skipping.")
                continue

            # --- 2. Determine best interpolation method ---
            original_height, original_width = image.shape[:2]
            target_width, target_height = TARGET_SIZE

            if target_width < original_width or target_height < original_height:
                # We are shrinking the image
                interpolation = cv2.INTER_AREA
            else:
                # We are enlarging the image
                interpolation = cv2.INTER_CUBIC

            # --- 3. Resize the image ---
            resized_image = cv2.resize(image, TARGET_SIZE, interpolation=interpolation)

            # --- 4. Save the resized image ---
            output_path = os.path.join(OUTPUT_DIR, image_filename)
            cv2.imwrite(output_path, resized_image)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {image_filename}: {e}")

    print(f"\n--- Processing Complete ---")
    print(f"Resized {processed_count} images and saved them to {OUTPUT_DIR}")


# --- THIS IS THE MISSING PART ---
# This block tells Python to run the main() function
# when the script is executed directly.
if __name__ == "__main__":
    main()
