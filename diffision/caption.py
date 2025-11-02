import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def main():
    parser = argparse.ArgumentParser(description="Batch caption images with BLIP")
    parser.add_argument(
        "image_dir", type=str, help="Directory containing images to caption"
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a valid directory.")
        return

    print("Loading BLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using fp16 for faster inference and less VRAM
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
    ).to(device)
    print(f"Model loaded on {device} in fp16.")

    image_files = (
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )

    print(f"Found {len(image_files)} images to process.")

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            print(f"Skipping {img_path.name}, caption already exists.")
            continue

        try:
            raw_image = Image.open(img_path).convert("RGB")

            # Generate caption
            inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Write to .txt file
            with open(txt_path, "w") as f:
                f.write(caption)
            print(f"Captioned {img_path.name}: {caption}")

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

    print("Captioning complete.")


if __name__ == "__main__":
    main()
