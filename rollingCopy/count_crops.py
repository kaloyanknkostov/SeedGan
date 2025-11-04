import math
import os


def create_crop_addition_plan():
    """
    Analyzes label files to create a plan for adding crops to balance the dataset.
    """
    labels_dir = "/home/kaloyan/Code/Thesis/SeedGan/rollingCopy/rolling images/labels/"
    output_dir = "/home/kaloyan/Code/Thesis/SeedGan/rollingCopy/amount_crops_image/"  # Output files will be saved here

    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return

    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

    extra_crops_counter = 0
    total_crops_to_add = 0
    total_original_crops = 0
    total_original_weeds = 0

    for label_file in label_files:
        image_name = os.path.splitext(label_file)[0]
        num_crops = 0
        num_weeds = 0

        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                try:
                    class_id = int(line.split(",")[0])
                    if class_id == 0:
                        num_crops += 1
                    elif class_id == 1:
                        num_weeds += 1
                except (ValueError, IndexError):
                    # Handle empty or malformed lines
                    continue

        total_original_crops += num_crops
        total_original_weeds += num_weeds

        crops_to_add_for_this_image = 0
        if num_crops > num_weeds:
            extra_crops_counter += num_crops - num_weeds
            crops_to_add_for_this_image = 0
        else:
            difference = num_weeds - num_crops
            if extra_crops_counter > 0:
                # Use one "extra" crop from the counter
                crops_to_add_for_this_image = max(0, difference - 1)
                extra_crops_counter -= 1
            else:
                crops_to_add_for_this_image = difference
        crops_to_add_for_this_image = int(math.ceil(crops_to_add_for_this_image / 2))
        total_crops_to_add += crops_to_add_for_this_image

        output_file_path = os.path.join(output_dir, f"{image_name}.txt")
        with open(output_file_path, "w") as out_f:
            out_f.write(str(crops_to_add_for_this_image))

    print("--- Crop Addition Plan Summary ---")
    print(f"Total original crops: {total_original_crops}")
    print(f"Total original weeds: {total_original_weeds}")
    print(f"Total crops to be added across all images: {total_crops_to_add}")
    print(f"Final surplus of crops (extra_crops_counter): {extra_crops_counter}")
    print(f"Plan files have been written to: {output_dir}")


if __name__ == "__main__":
    create_crop_addition_plan()
