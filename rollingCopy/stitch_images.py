
import os
import sys
import argparse
import random
from PIL import Image

def process_labels(label_path, side):
    """
    Processes a label file to filter and adjust bounding boxes based on the side of the image.

    Args:
        label_path (str): The path to the label file.
        side (str): 'left' or 'right', indicating which half of the image the labels belong to.

    Returns:
        list: A list of processed label strings.
    """
    processed_labels = []
    if not os.path.exists(label_path):
        return processed_labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 5:
                continue

            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            left_edge = x_center - width / 2
            right_edge = x_center + width / 2

            if side == 'left':
                # Keep objects fully in the left half
                if right_edge <= 0.5:
                    processed_labels.append(f"{class_id},{x_center},{y_center},{width},{height}")
                # Adjust objects crossing the midline
                elif left_edge < 0.5 and right_edge > 0.5:
                    new_width = 0.5 - left_edge
                    new_x_center = 0.5 - new_width / 2
                    processed_labels.append(f"{class_id},{new_x_center},{y_center},{new_width},{height}")
            
            elif side == 'right':
                # Keep objects fully in the right half
                if left_edge >= 0.5:
                    processed_labels.append(f"{class_id},{x_center},{y_center},{width},{height}")
                # Adjust objects crossing the midline
                elif left_edge < 0.5 and right_edge > 0.5:
                    new_width = right_edge - 0.5
                    new_x_center = 0.5 + new_width / 2
                    processed_labels.append(f"{class_id},{new_x_center},{y_center},{new_width},{height}")

    return processed_labels

def stitch_images_and_labels(data_folder):
    """
    Stitches images and processes corresponding labels.
    """
    images_dir = os.path.join(data_folder, 'images')
    labels_dir = os.path.join(data_folder, 'labels')
    output_images_dir = os.path.join(data_folder, 'output', 'images')
    output_labels_dir = os.path.join(data_folder, 'output', 'labels')

    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]

    print(f"Found {len(image_files)} images in {images_dir}")
    print(f"Found {len(label_files)} labels in {labels_dir}")
    
    if not image_files:
        print("No images found in the images directory.")
        return

    left_images = list(image_files)
    right_images = list(image_files)
    random.shuffle(left_images)
    random.shuffle(right_images)

    num_stitched = 0
    for i, (left_image_name, right_image_name) in enumerate(zip(left_images, right_images)):
        left_image_path = os.path.join(images_dir, left_image_name)
        right_image_path = os.path.join(images_dir, right_image_name)

        try:
            with Image.open(left_image_path) as img_left, Image.open(right_image_path) as img_right:
                if img_left.size != img_right.size:
                    print(f"Skipping pair: {left_image_name} and {right_image_name} have different sizes.")
                    continue

                width, height = img_left.size
                
                # Crop halves
                left_half = img_left.crop((0, 0, width / 2, height))
                right_half = img_right.crop((width / 2, 0, width, height))

                # Create new image and paste halves
                stitched_image = Image.new('RGB', (width, height))
                stitched_image.paste(left_half, (0, 0))
                stitched_image.paste(right_half, (int(width / 2), 0))

                # Save stitched image
                stitched_filename = f"stitched_{i}.jpg"
                stitched_image.save(os.path.join(output_images_dir, stitched_filename))

                # Process labels
                left_label_name = os.path.splitext(left_image_name)[0] + '.txt'
                right_label_name = os.path.splitext(right_image_name)[0] + '.txt'
                
                left_label_path = os.path.join(labels_dir, left_label_name)
                right_label_path = os.path.join(labels_dir, right_label_name)

                new_labels = []
                new_labels.extend(process_labels(left_label_path, 'left'))
                new_labels.extend(process_labels(right_label_path, 'right'))

                # Save new labels
                stitched_label_filename = f"stitched_{i}.txt"
                with open(os.path.join(output_labels_dir, stitched_label_filename), 'w') as f:
                    for label in new_labels:
                        f.write(label + '\n')
                
                num_stitched += 1

        except Exception as e:
            print(f"Error processing pair ({left_image_name}, {right_image_name}): {e}")

    print(f"Successfully stitched {num_stitched} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch image halves and merge corresponding labels.")
    parser.add_argument("data_folder", help="Path to the data folder containing images/ and labels/ subdirectories.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_folder):
        print(f"Error: Data folder not found at {args.data_folder}", file=sys.stderr)
        sys.exit(1)

    stitch_images_and_labels(args.data_folder)
