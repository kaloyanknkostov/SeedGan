import os

import cv2
import numpy as np

# --- Configuration ---
# Get the absolute path to this script file
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
BASE_DIR = os.path.dirname(script_dir)

# Set input and output directories based on the base project path
INPUT_DIR = os.path.join(BASE_DIR, "data", "crops_with_background")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "crops_with_no_background")

# --- IMPORTANT: HSV Green Range Tuning ---
# This is the most important part to get right.
# OpenCV HSV range: H:[0-179], S:[0-255], V:[0-255]
#
# * H (Hue): The color itself. Green is ~ 35-85.
# * S (Saturation): The "purity" of the color. 0 is gray, 255 is pure color.
# * V (Value): The "brightness" of the color. 0 is black, 255 is bright.

# We set a minimum Saturation (40) and Value (40) to avoid
# picking up dark/grayish pixels that might be tinted green.
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# --- End Configuration ---


def main():
    """
    Reads images, isolates green regions using HSV masking,
    and saves the result with a black background.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Reading crops from: {INPUT_DIR}")
    print(f"Saving masked crops to: {OUTPUT_DIR}")

    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        print(f"\n--- ERROR ---")
        print(f"Input directory not found at: {INPUT_DIR}")
        print(f"Did you run the first script (crop_class_0.py) successfully?")
        return

    processed_count = 0

    # Iterate through all files in the input directory
    for image_filename in os.listdir(INPUT_DIR):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        # --- 1. Load the image ---
        image_path = os.path.join(INPUT_DIR, image_filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  Error: Could not read image {image_path}. Skipping.")
            continue

        print(f"Processing {image_filename}...")

        # --- 2. Convert to HSV color space ---
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- 3. Create a mask for the green color ---
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Optional: You can add morphological operations here to clean up the mask
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- 4. Apply the mask to the original image ---
        # This operation keeps pixels from 'image' only where the 'mask' is non-zero
        result = cv2.bitwise_and(image, image, mask=mask)

        # --- 5. Save the resulting image ---
        output_path = os.path.join(OUTPUT_DIR, image_filename)
        # We use the original filename (which should be .png)
        cv2.imwrite(output_path, result)

        processed_count += 1

    print(f"\n--- Script Finished ---")
    print(f"Successfully processed and saved {processed_count} images.")


if __name__ == "__main__":
    main()
