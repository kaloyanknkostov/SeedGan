import os
import random
from PIL import Image

def yolo_to_absolute(yolo_box, image_width, image_height):
    """Converts YOLO format (class, center_x, center_y, width, height) to absolute pixel coordinates (class, x_min, y_min, x_max, y_max)."""
    class_id, center_x, center_y, width, height = yolo_box
    x_min = (center_x - width / 2) * image_width
    y_min = (center_y - height / 2) * image_height
    x_max = (center_x + width / 2) * image_width
    y_max = (center_y + height / 2) * image_height
    return class_id, int(x_min), int(y_min), int(x_max), int(y_max)

def absolute_to_yolo(abs_box, image_width, image_height):
    """Converts absolute pixel coordinates (class, x_min, y_min, x_max, y_max) to YOLO format (class, center_x, center_y, width, height)."""
    class_id, x_min, y_min, x_max, y_max = abs_box
    center_x = (x_min + x_max) / 2 / image_width
    center_y = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return class_id, center_x, center_y, width, height

def check_overlap(box1, box2):
    """Checks if two boxes in absolute pixel coordinates overlap. box = (x_min, y_min, x_max, y_max)"""
    _, x1_min, y1_min, x1_max, y1_max = box1
    _, x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)


def paste_crops():
    """
    Pastes crops onto images based on the plan from count_crops.py.
    """
    plan_dir = "amount_crops_image/"
    images_dir = "rolling images/images/"
    labels_dir = "rolling images/labels/"
    crops_dir = "rolling images/crops/"
    output_images_dir = "rolling images/output/images/"
    output_labels_dir = "rolling images/output/labels/"

    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("--- Starting Crop Pasting Process ---")

    # Get list of crop images
    crop_files = [f for f in os.listdir(crops_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not crop_files:
        print(f"Error: No crop images found in {crops_dir}")
        return

    plan_files = sorted([f for f in os.listdir(plan_dir) if f.endswith(".txt")])

    for plan_file in plan_files:
        image_name_no_ext = os.path.splitext(plan_file)[0]
        plan_path = os.path.join(plan_dir, plan_file)

        with open(plan_path, "r") as f:
            try:
                crops_to_add = int(f.read().strip())
            except ValueError:
                print(f"Warning: Could not read number from {plan_file}. Skipping.")
                continue

        if crops_to_add == 0:
            # If no crops to add, just copy the original image and label
            print(f"No crops to add for {image_name_no_ext}. Copying original files.")
            
            # Copy image
            original_image_path = os.path.join(images_dir, image_name_no_ext + ".jpg")
            output_image_path = os.path.join(output_images_dir, image_name_no_ext + ".jpg")
            if os.path.exists(original_image_path):
                img = Image.open(original_image_path)
                img.save(output_image_path)
            else:
                print(f"Warning: Image not found at {original_image_path}")

            # Copy label
            original_label_path = os.path.join(labels_dir, image_name_no_ext + ".txt")
            output_label_path = os.path.join(output_labels_dir, image_name_no_ext + ".txt")
            if os.path.exists(original_label_path):
                with open(original_label_path, "r") as f_in, open(output_label_path, "w") as f_out:
                    f_out.write(f_in.read())
            else:
                print(f"Warning: Label not found at {original_label_path}")
            
            continue

        # --- Start of pasting logic ---
        print(f"Adding {crops_to_add} crops to {image_name_no_ext}")

        original_image_path = os.path.join(images_dir, image_name_no_ext + ".jpg")
        if not os.path.exists(original_image_path):
            print(f"Warning: Image not found at {original_image_path}")
            continue

        base_image = Image.open(original_image_path)
        image_width, image_height = base_image.size

        # Load existing labels
        existing_labels = []
        original_label_path = os.path.join(labels_dir, image_name_no_ext + ".txt")
        if os.path.exists(original_label_path):
            with open(original_label_path, "r") as f:
                for line in f:
                    try:
                        parts = [float(p) for p in line.strip().replace(',', ' ').split()]
                        class_id = int(parts[0])
                        yolo_box = (class_id, parts[1], parts[2], parts[3], parts[4])
                        existing_labels.append(yolo_to_absolute(yolo_box, image_width, image_height))
                    except (ValueError, IndexError):
                        continue

        # Paste crops
        new_labels = list(existing_labels)
        for _ in range(crops_to_add):
            crop_file = random.choice(crop_files)
            crop_image = Image.open(os.path.join(crops_dir, crop_file))
            crop_width, crop_height = crop_image.size

            pasted = False
            for _ in range(100): # Max 100 attempts to find a non-overlapping spot
                pos_x = random.randint(0, image_width - crop_width)
                pos_y = random.randint(0, image_height - crop_height)

                new_box_abs = (0, pos_x, pos_y, pos_x + crop_width, pos_y + crop_height) # class_id is 0 for crop

                is_overlapping = False
                for existing_box in new_labels:
                    if check_overlap(existing_box, new_box_abs):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    base_image.paste(crop_image, (pos_x, pos_y))
                    new_labels.append(new_box_abs)
                    pasted = True
                    break
            
            if not pasted:
                print(f"Warning: Could not find a non-overlapping spot for a crop in {image_name_no_ext}")

        # Save new image
        output_image_path = os.path.join(output_images_dir, image_name_no_ext + ".jpg")
        base_image.save(output_image_path)

        # Save new labels
        output_label_path = os.path.join(output_labels_dir, image_name_no_ext + ".txt")
        with open(output_label_path, "w") as f:
            for label in new_labels:
                yolo_box = absolute_to_yolo(label, image_width, image_height)
                f.write(f"{yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]} {yolo_box[4]}\n")
        # --- End of pasting logic ---


    print("--- Crop Pasting Process Finished ---")


if __name__ == "__main__":
    paste_crops()
