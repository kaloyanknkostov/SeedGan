import os

import cv2
import numpy as np

# --- Configuration ---
# Use os.path.join for cross-platform compatibility
IMAGE_DIR = os.path.join("data", "original", "images")
LABEL_DIR = os.path.join("data", "original", "labels")
OUTPUT_DIR = os.path.join("data", "crops_with_background")
# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# --- End Configuration ---


def yolo_to_pixels(yolo_box, img_width, img_height):
    """
    Converts YOLO format (class, x_center, y_center, w, h)
    to pixel coordinates (x_min, y_min, x_max, y_max).
    """
    _, x_center_norm, y_center_norm, width_norm, height_norm = yolo_box

    # Calculate absolute pixel values
    w_abs = int(width_norm * img_width)
    h_abs = int(height_norm * img_height)
    x_min = int((x_center_norm * img_width) - (w_abs / 2))
    y_min = int((y_center_norm * img_height) - (h_abs / 2))
    x_max = x_min + w_abs
    y_max = y_min + h_abs

    # Clamp coordinates to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    return x_min, y_min, x_max, y_max


def main():
    """
    Main function to process images, find class 0 labels,
    and save the corresponding crops.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ensured at: {OUTPUT_DIR}")

    processed_images = 0
    total_crops_saved = 0

    # Iterate through all files in the image directory
    for image_filename in os.listdir(IMAGE_DIR):
        # Check if it's a valid image file
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        # --- 1. Find corresponding label file ---
        base_filename = os.path.splitext(image_filename)[0]
        label_filename = f"{base_filename}.txt"
        label_path = os.path.join(LABEL_DIR, label_filename)

        # --- 2. Check if label file exists ---
        if not os.path.exists(label_path):
            print(f"Skipping {image_filename}: No corresponding label file found.")
            continue

        # --- 3. Load the image ---
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  Error: Could not read image {image_path}. Skipping.")
            continue

        img_height, img_width, _ = image.shape
        print(f"\nProcessing {image_filename} (Size: {img_width}x{img_height})...")

        # --- 4. Read label file and find class 0 ---
        crops_per_image = 0
        try:
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    # Parse YOLO line
                    parts = line.split()
                    class_id = int(parts[0])

                    # --- 5. Filter for class_id 0 ---
                    if class_id == 0:
                        yolo_box = [float(p) for p in parts]

                        # --- 6. Convert to pixel coordinates ---
                        x_min, y_min, x_max, y_max = yolo_to_pixels(
                            yolo_box, img_width, img_height
                        )

                        # --- 7. Extract the crop ---
                        crop = image[y_min:y_max, x_min:x_max]

                        # Check if crop is valid
                        if crop.size == 0:
                            print(
                                f"  Warning: Generated an empty crop from box {yolo_box} in {label_filename}."
                            )
                            continue

                        # --- 8. Save the crop ---
                        # Create a unique filename for each crop
                        crop_filename = (
                            f"{base_filename}_class0_crop_{crops_per_image}.png"
                        )
                        output_path = os.path.join(OUTPUT_DIR, crop_filename)
                        cv2.imwrite(output_path, crop)

                        crops_per_image += 1
                        total_crops_saved += 1

        except Exception as e:
            print(f"  Error parsing file {label_path}: {e}")

        if crops_per_image > 0:
            print(
                f"  Successfully extracted and saved {crops_per_image} 'class 0' crops."
            )
            processed_images += 1
        else:
            print(f"  No 'class 0' objects found in this image.")

    print(f"\n--- Script Finished ---")
    print(f"Processed {processed_images} images that contained 'class 0' objects.")
    print(f"Saved a total of {total_crops_saved} crops to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
