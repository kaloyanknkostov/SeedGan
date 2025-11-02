import os
from pathlib import Path  # <-- Import pathlib for easy path handling

import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer

# --- 1. Define Input and Output Directories ---
input_dir = Path("/home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background")
output_dir = Path(
    "/home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled4x"
)

# --- 2. Create the output directory if it doesn't exist ---
output_dir.mkdir(parents=True, exist_ok=True)

# Define which file types to process
image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# --- 3. Setup the Model (runs on GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

model_path = "RealESRGAN_x4plus.pth"
state_dict = torch.load(model_path, map_location=torch.device("cpu"))["params_ema"]

model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
)
model.load_state_dict(state_dict, strict=True)
model.eval()  # Set model to evaluation mode (good practice)
model.to(device)

upsampler = RealESRGANer(
    scale=4, model_path=model_path, model=model, tile=0, pre_pad=0, half=True
)

print(f"Processing images from: {input_dir}")
print(f"Saving results to: {output_dir}")

# --- 4. Loop through all files in the input directory ---
for filename in os.listdir(input_dir):
    # Check if the file has one of the allowed extensions
    if any(filename.lower().endswith(ext) for ext in image_extensions):

        # Define the full path for input and output
        input_path = input_dir / filename
        output_path = output_dir / filename

        print(f"Processing: {filename}...")

        try:
            # --- 5. Load, Process, and Save each image ---
            img = Image.open(input_path).convert("RGB")
            img_np = np.array(img)

            output, _ = upsampler.enhance(img_np, outscale=4)

            output_img = Image.fromarray(output)
            output_img.save(output_path)

        except Exception as e:
            # Skip corrupted images and print an error
            print(f"  Could not process {filename}. Error: {e}")

print("All images processed.")
