# Project Overview

This project contains a suite of Python scripts for analyzing and augmenting images of crops, primarily for use in training machine learning models. The scripts handle tasks like analyzing object positions, stitching images together, and augmenting datasets by pasting new objects onto existing images.

## Available Scripts

This section details the scripts available in the project, their purpose, and how to run them.

---

### `analyze_middle.py`

**Purpose:** This script analyzes a directory of images to identify and count crop objects located on the horizontal middle line of an image.

**Usage:**
```bash
python analyze_middle.py /path/to/your/data_folder
```
*   The `data_folder` should contain `images/` and `labels/` subdirectories.

**Output:**
*   The total number of images analyzed.
*   The number of images with at least one crop on the middle line.
*   The total count of all crops found on the middle line.

---

### `stitch_images.py`

**Purpose:** This script creates new images by stitching together halves of other images from the dataset. It randomly pairs images to combine the left half of one with the right half of another, ensuring all images are used. It also processes the corresponding label files to match the new, stitched images.

**Usage:**
```bash
python stitch_images.py /path/to/your/data_folder
```
*   The `data_folder` should contain `images/` and `labels/` subdirectories.
*   The script will create an `output/` directory inside your `data_folder` for the results.

**Functionality:**
*   Prints the number of found images and labels.
*   Randomly shuffles and pairs images for stitching.
*   Adjusts bounding boxes for objects that cross the stitch line.
*   Saves new images to `output/images/` and new labels to `output/labels/`.

---

### `paste_crops.py`

**Purpose:** This is a powerful augmentation script that pastes images of individual crops onto your field images. It intelligently places the crops to avoid overlapping with existing objects and generates updated label files.

**Usage:**
```bash
python paste_crops.py /path/to/your/data_folder [OPTIONS]
```
*   The `data_folder` must contain `images/`, `labels/`, and `crops/` subdirectories.
*   The `crops/` directory should contain the individual crop images to be pasted.
*   The script will create an `output/` directory for the results.

**Options:**

*   `--use-distribution`: If this flag is present, the script will analyze the size distribution of objects in your original `labels/` directory. It then resizes the pasted crops to match this distribution, which can create more realistic variations in object size.
*   `--blur-edges`: If this flag is present, the script will apply a feathering/blurring effect to the edges of non-transparent crop images (e.g., JPGs). This helps the pasted crops blend more smoothly with the background image.

**Example Commands:**

```bash
# Basic usage
python paste_crops.py /path/to/data

# Use size distribution for scaling
python paste_crops.py /path/to/data --use-distribution

# Apply edge blurring for better blending
python paste_crops.py /path/to/data --blur-edges

# Use both features at once
python paste_crops.py /path/to/data --use-distribution --blur-edges
```

---
## Development Summary

This section documents the iterative process of building the scripts in this project.

### `stitch_images.py`
1.  **Initial Goal:** Create a script to stitch halves of images together and merge their corresponding labels.
2.  **Clarification:** A Q&A process was used to establish key details:
    *   Stitching should be done horizontally (left/right halves).
    *   Image pairing should be random, ensuring all images are used once for a left side and once for a right side.
    *   Bounding boxes crossing the center line should be cropped.
    *   The coordinate system for the new labels was confirmed to be the same as the originals, as the output image resolution is unchanged.
3.  **Debugging:**
    *   The initial script produced empty 0-byte label files.
    *   **Decision:** A step-by-step debugging process was initiated. It was first confirmed that the script's file-writing logic was correct.
    *   **Discovery:** Investigation revealed the input `labels/` directory was empty. After clarifying the correct path with the user, the script still produced empty files.
    *   **Root Cause:** The final issue was identified as a mismatch in the label format. The script was expecting space-separated values, but the input files were comma-separated.
4.  **Finalization:** The script was corrected to parse comma-separated input, which resolved the issue. A feature to print the number of found files was also added for better user feedback.

### `view_annotations.py`
1.  **Goal:** The user reported that the "preview window" was stuck in the corner of the screen and requested it be centered.
2.  **Investigation:** It was determined that `view_annotations.py` (not the other scripts) was responsible for creating this UI window.
3.  **Decision:** The best way to center the window in a cross-platform manner was to use the `tkinter` library to get the screen's dimensions.
4.  **Implementation:** The script was modified to:
    *   Get the screen width and height.
    *   Calculate the required `(x, y)` coordinates to center the image window.
    *   Use OpenCV's `moveWindow()` function to position the window before displaying it.

### `paste_crops.py`
1.  **Goal:** A new, complex data augmentation script was requested to paste crop images onto field images.
2.  **Requirement Scoping:** The request was broken down into core features through a detailed Q&A process:
    *   **Inputs:** A single data directory containing `images/`, `labels/`, and `crops/` folders was established.
    *   **Placement:** The script must read existing labels and place new crops randomly, ensuring no overlap.
    *   **Scaling:** A default scaling strategy (shrink by 20% until it fits) was defined.
    *   **Advanced Scaling:** A `--use-distribution` flag was proposed and accepted to allow scaling based on the statistical size distribution of original objects. The logic for preserving relative sizes was also clarified.
    *   **Blending:** To handle non-transparent (JPG) crops, a `--blur-edges` flag was proposed to apply a feathering effect for smoother blending.
3.  **Implementation:** The script was built from scratch, incorporating all the defined logic, command-line arguments, and error handling.
4.  **Refinement:** Based on user feedback, the strength of the blur effect was increased by adjusting the parameters in the mask-creation function.
