import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Scans a directory for .txt/image pairs and creates a metadata.jsonl file."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory (containing .txt and image files).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Error: Directory not found at {dataset_dir}")
        sys.exit(1)

    output_file = dataset_dir / "metadata.jsonl"

    print(f"Scanning {dataset_dir} and creating {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        file_count = 0
        # Find all .txt files
        for txt_file in dataset_dir.glob("*.txt"):
            try:
                # Assume the image has the same base name but a .png extension
                image_file = txt_file.with_suffix(".png")

                if not image_file.exists():
                    # Try .jpg as a fallback
                    image_file = txt_file.with_suffix(".jpg")
                    if not image_file.exists():
                        # Try .jpeg as a fallback
                        image_file = txt_file.with_suffix(".jpeg")
                        if not image_file.exists():
                            print(
                                f"Warning: No matching image for {txt_file.name}. Skipping."
                            )
                            continue

                # Read the caption from the .txt file
                caption = txt_file.read_text(encoding="utf-8").strip()

                if not caption:
                    print(f"Warning: Empty caption in {txt_file.name}. Skipping.")
                    continue

                # Create the JSON object
                data_entry = {
                    "file_name": image_file.name,  # e.g., "001.png"
                    "text": caption,
                }

                # Write it as a new line in the .jsonl file
                f.write(json.dumps(data_entry) + "\n")
                file_count += 1

            except Exception as e:
                print(f"Error processing {txt_file.name}: {e}")

    print(f"\nDone. Created metadata for {file_count} image/text pairs.")
    print("You can now run your 'train_lora.sh' script.")


if __name__ == "__main__":
    main()
