
import os
import argparse
import subprocess
import cv2
import tkinter as tk

def run_fzf(image_list):
    """
    Uses fzf to allow the user to select an image from a list.

    Args:
        image_list (list): A list of image filenames.

    Returns:
        str: The selected image filename, or None if fzf was cancelled.
    """
    fzf_input = "\n".join(image_list).encode('utf-8')
    try:
        # Run fzf as a subprocess
        process = subprocess.run(
            ['fzf', '--height', '40%', '--reverse'],
            input=fzf_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL  # Hide fzf messages
        )
        # Decode the selected output and remove trailing newline
        selected = process.stdout.decode('utf-8').strip()
        return selected if selected else None
    except FileNotFoundError:
        print("Error: 'fzf' command not found.")
        print("Please install fzf to use this script (e.g., 'sudo apt-get install fzf').")
        return None

def draw_annotations(image_path, label_path):
    """
    Draws bounding boxes from a label file onto an image and displays it.

    Args:
        image_path (str): The full path to the image file.
        label_path (str): The full path to the corresponding label file.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file at {image_path}")
        return
        
    img_height, img_width, _ = image.shape

    # Define colors for classes (BGR format for OpenCV)
    # Class 0 (Crop): Green
    # Class 1 (Weed): Red
    colors = {0: (0, 255, 0), 1: (0, 0, 255)}
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    class_id, x_center, y_center, width, height = map(float, parts)
                    class_id = int(class_id)

                    # Convert relative YOLO coordinates to absolute pixel coordinates
                    abs_x_center = x_center * img_width
                    abs_y_center = y_center * img_height
                    abs_width = width * img_width
                    abs_height = height * img_height

                    # Calculate top-left and bottom-right corners
                    x1 = int(abs_x_center - abs_width / 2)
                    y1 = int(abs_y_center - abs_height / 2)
                    x2 = int(abs_x_center + abs_width / 2)
                    y2 = int(abs_y_center + abs_height / 2)
                    
                    # Get color for the class, default to white if class is unknown
                    color = colors.get(class_id, (255, 255, 255))
                    
                    # Draw the rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add a label
                    label = "Crop" if class_id == 0 else "Weed"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed line in {label_path}: {line.strip()}")
                    continue

    # --- Centering Logic ---
    # Get screen dimensions using tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Calculate center position for the window
    x_pos = (screen_width - img_width) // 2
    y_pos = (screen_height - img_height) // 2
    
    window_name = f'Annotations for {os.path.basename(image_path)}'

    # Create a named window, move it, and then display the image
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, x_pos, y_pos)
    cv2.imshow(window_name, image)
    
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively view image annotations using fzf.")
    parser.add_argument("folder", help="Path to the data directory with 'images' and 'labels' subfolders.")
    args = parser.parse_args()

    images_dir = os.path.join(args.folder, 'images')
    labels_dir = os.path.join(args.folder, 'labels')

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"Error: 'images' and/or 'labels' directory not found in {args.folder}")
        exit(1)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found to display.")
        exit(0)

    # Loop to allow viewing multiple images
    while True:
        selected_image_name = run_fzf(image_files)

        if selected_image_name:
            image_path = os.path.join(images_dir, selected_image_name)
            label_name = os.path.splitext(selected_image_name)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            draw_annotations(image_path, label_path)
        else:
            print("No image selected. Exiting.")
            break
