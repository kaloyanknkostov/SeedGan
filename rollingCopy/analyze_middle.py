
import os
import argparse

def analyze_middle_line(data_folder):
    """
    Analyzes images and labels to find how many crops lie on the horizontal middle line.

    Args:
        data_folder (str): Path to the directory containing 'images' and 'labels' subdirectories.
    """
    labels_dir = os.path.join(data_folder, 'labels')
    images_dir = os.path.join(data_folder, 'images')
    
    if not os.path.isdir(labels_dir):
        print(f"Error: 'labels' directory not found in {data_folder}")
        return
    if not os.path.isdir(images_dir):
        print(f"Error: 'images' directory not found in {data_folder}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    
    if total_images == 0:
        print("No images found in the 'images' directory.")
        return

    images_with_middle_crops = 0
    total_middle_crops = 0
    
    img_height = 1088
    middle_line_y = img_height / 2

    for image_file in image_files:
        label_filename = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(label_path):
            continue

        found_middle_crop_in_image = False
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split(','))
                    
                    if int(class_id) == 0:  # Class 0 is crop
                        y_center_abs = y_center * img_height
                        height_abs = height * img_height
                        
                        y_min = y_center_abs - (height_abs / 2)
                        y_max = y_center_abs + (height_abs / 2)

                        if y_min <= middle_line_y <= y_max:
                            total_middle_crops += 1
                            if not found_middle_crop_in_image:
                                images_with_middle_crops += 1
                                found_middle_crop_in_image = True
                except ValueError:
                    print(f"Warning: Could not parse line in {label_path}: {line.strip()}")
                    continue

    print(f"Analyzed {total_images} images.")
    print(f"Found {images_with_middle_crops} images with crops on the middle line.")
    print(f"Total number of crops on the middle line: {total_middle_crops}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze crop positions in images.")
    parser.add_argument("folder", help="Path to the data directory with 'images' and 'labels' subfolders.")
    args = parser.parse_args()
    
    analyze_middle_line(args.folder)
