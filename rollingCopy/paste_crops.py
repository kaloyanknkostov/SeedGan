import os
import sys
import argparse
import random
from PIL import Image, ImageFilter
import numpy as np

def get_existing_boxes(label_path, img_width, img_height):
    """Reads a label file and returns a list of absolute bounding box coordinates."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split(',')
                if len(parts) != 5: continue
                _, x_center, y_center, width, height = map(float, parts)
                
                abs_width = width * img_width
                abs_height = height * img_height
                x1 = (x_center * img_width) - (abs_width / 2)
                y1 = (y_center * img_height) - (abs_height / 2)
                x2 = x1 + abs_width
                y2 = y1 + abs_height
                boxes.append((x1, y1, x2, y2))
            except ValueError:
                continue
    return boxes

def check_overlap(new_box, existing_boxes):
    """Checks if a new box overlaps with any of the existing boxes."""
    nx1, ny1, nx2, ny2 = new_box
    for ex1, ey1, ex2, ey2 in existing_boxes:
        if not (nx2 < ex1 or nx1 > ex2 or ny2 < ey1 or ny1 > ey2):
            return True  # Overlap detected
    return False

def create_blur_mask(size):
    """Creates a feathered/blurred mask for pasting."""
    mask = Image.new('L', size, 0)
    # Draw a white rectangle, slightly smaller than the image, leaving a border
    border = int(min(size) * 0.2)
    inner_rect = (border, border, size[0] - border, size[1] - border)
    mask.paste(255, inner_rect)
    # Blur the mask to create soft edges
    return mask.filter(ImageFilter.GaussianBlur(radius=border))

def get_size_distribution(labels_dir):
    """Analyzes all label files to get a distribution of object areas."""
    areas = []
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    if len(parts) != 5: continue
                    _, _, _, width, height = map(float, parts)
                    # Using area (w*h) as the measure of size. Assuming 1088x1920 resolution for now.
                    # This is a simplification; a better approach might not need a fixed resolution.
                    area = (width * 1920) * (height * 1088)
                    areas.append(area)
                except ValueError:
                    continue
    return areas if areas else [100*100] # Return a default if no labels found

def paste_crops(args):
    # --- 1. Setup Paths ---
    images_dir = os.path.join(args.data_dir, 'images')
    labels_dir = os.path.join(args.data_dir, 'labels')
    crops_dir = os.path.join(args.data_dir, 'crops')
    
    output_images_dir = os.path.join(args.data_dir, 'output', 'images')
    output_labels_dir = os.path.join(args.data_dir, 'output', 'labels')

    # --- 2. Validate Paths and Create Output Dirs ---
    for d in [images_dir, labels_dir, crops_dir]:
        if not os.path.isdir(d):
            print(f"Error: Required directory not found: {d}", file=sys.stderr)
            sys.exit(1)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # --- 3. Load File Lists ---
    field_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    crop_images = [os.path.join(crops_dir, f) for f in os.listdir(crops_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(crop_images)

    if not field_images or not crop_images:
        print("Error: No images found in 'images' or 'crops' directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(field_images)} field images.")
    print(f"Found {len(crop_images)} crop images to paste.")

    # --- 4. Prepare Size Distribution if needed ---
    size_dist = []
    if args.use_distribution:
        print("Analyzing size distribution of original labels...")
        size_dist = get_size_distribution(labels_dir)
        print("Distribution analysis complete.")

    # --- 5. Main Processing Loop ---
    crops_per_field = int(np.ceil(len(crop_images) / len(field_images)))
    crop_idx = 0

    for field_image_name in field_images:
        field_path = os.path.join(images_dir, field_image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(field_image_name)[0] + '.txt')
        
        field_img = Image.open(field_path).convert("RGBA")
        field_width, field_height = field_img.size
        
        existing_boxes = get_existing_boxes(label_path, field_width, field_height)
        original_labels = open(label_path).readlines() if os.path.exists(label_path) else []
        new_labels = []

        # Determine which crops to paste on this image
        crops_to_paste_paths = crop_images[crop_idx : crop_idx + crops_per_field]
        crop_idx += crops_per_field

        # --- Determine Target Sizes ---
        target_sizes = {} # map path to target w,h
        if args.use_distribution and size_dist:
            crops_to_paste_objects = [Image.open(p) for p in crops_to_paste_paths]
            crops_sorted_by_size = sorted(crops_to_paste_objects, key=lambda img: img.size[0] * img.size[1])
            sampled_areas = sorted(random.sample(size_dist, k=len(crops_to_paste_objects)))
            
            for i, crop_obj in enumerate(crops_sorted_by_size):
                target_area = sampled_areas[i]
                original_w, original_h = crop_obj.size

                if original_w <= 0 or original_h <= 0:
                    continue

                aspect_ratio = original_w / original_h
                if aspect_ratio == 0:
                    continue

                target_h = int(np.sqrt(target_area / aspect_ratio))
                target_w = int(target_h * aspect_ratio)

                if target_w <= 0 or target_h <= 0:
                    continue

                target_sizes[crop_obj.filename] = (target_w, target_h)
            [c.close() for c in crops_to_paste_objects]


        for crop_path in crops_to_paste_paths:
            crop_img = Image.open(crop_path).convert("RGBA")
            
            # --- Scaling Logic ---
            if crop_path in target_sizes:
                crop_img = crop_img.resize(target_sizes[crop_path], Image.Resampling.LANCZOS)

            found_spot = False
            for _ in range(10): # Try scaling down up to 10 times
                w, h = crop_img.size
                if w < 10 or h < 10: break # Stop if crop is too small

                # --- Placement Logic ---
                for _ in range(100): # Try to find a spot 100 times
                    pos_x = random.randint(0, field_width - w)
                    pos_y = random.randint(0, field_height - h)
                    new_box = (pos_x, pos_y, pos_x + w, pos_y + h)

                    if not check_overlap(new_box, existing_boxes):
                        # --- Found a valid spot, now paste ---
                        mask = None
                        if crop_img.mode == 'RGBA' and 'A' in crop_img.getbands():
                             mask = crop_img.getchannel('A')
                        elif args.blur_edges:
                            mask = create_blur_mask(crop_img.size)
                        
                        field_img.paste(crop_img, (pos_x, pos_y), mask=mask)
                        
                        # Add to lists for output
                        existing_boxes.append(new_box)
                        
                        # Create new label line
                        yolo_x = (pos_x + w / 2) / field_width
                        yolo_y = (pos_y + h / 2) / field_height
                        yolo_w = w / field_width
                        yolo_h = h / field_height
                        new_labels.append(f"0,{yolo_x},{yolo_y},{yolo_w},{yolo_h}\n")
                        
                        found_spot = True
                        break # Exit placement loop
                
                if found_spot:
                    break # Exit scaling loop
                
                if not args.use_distribution:
                    # If spot not found and not using distribution, scale down and retry
                    new_w = int(w * 0.8)
                    new_h = int(h * 0.8)
                    crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    # If using distribution, we don't rescale, we just fail for this crop
                    break
            
            crop_img.close()

        # --- Save final image and labels ---
        output_image_path = os.path.join(output_images_dir, field_image_name)
        field_img.convert("RGB").save(output_image_path)
        
        output_label_path = os.path.join(output_labels_dir, os.path.splitext(field_image_name)[0] + '.txt')
        with open(output_label_path, 'w') as f:
            f.writelines(original_labels)
            f.writelines(new_labels)
            
        print(f"Processed {field_image_name}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paste crop images onto field images, avoiding existing objects.")
    parser.add_argument("data_dir", help="Directory containing 'images', 'labels', and 'crops' folders.")
    parser.add_argument("--use-distribution", action="store_true", help="Scale pasted crops based on the size distribution of original labels.")
    parser.add_argument("--blur-edges", action="store_true", help="Blur the edges of non-transparent (e.g., JPG) pasted crops for better blending.")
    
    args = parser.parse_args()
    paste_crops(args)
