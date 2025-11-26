import os
import re
import torch
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
import numpy as np

import config
from model import SimpleUnet
from diffusion import Diffusion

def get_latest_filenumber(path):
    """Scans a directory for files named generated_image_N.png and returns the highest N."""
    if not os.path.exists(path):
        return -1
    
    files = os.listdir(path)
    if not files:
        return -1

    numbers = []
    for f in files:
        match = re.match(r"generated_image_(\d+)\.png", f)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) if numbers else -1

def generate(num_images, batch_size):
    # Create directory if it doesn't exist
    os.makedirs(config.GENERATED_IMAGE_DIR, exist_ok=True)

    # Initialize model and diffusion
    model = SimpleUnet().to(config.DEVICE)
    diffusion = Diffusion(timesteps=config.T, device=config.DEVICE)

    # Load the latest checkpoint
    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    if not os.path.exists(latest_checkpoint_path):
        print("No checkpoint found. Please train the model first.")
        return

    print("Loading model from latest checkpoint...")
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Determine starting number for images
    start_num = get_latest_filenumber(config.GENERATED_IMAGE_DIR) + 1
    end_num = start_num + num_images
    print(f"Generating {num_images} images, starting from number {start_num}.")

    # Generate and save images in batches
    with torch.no_grad():
        with tqdm(total=num_images, desc="Generating images") as pbar:
            for i in range(start_num, end_num, batch_size):
                current_batch_size = min(batch_size, end_num - i)
                
                # Create a batch of noise tensors, each with a specific seed
                noise_tensors = []
                for seed in range(i, i + current_batch_size):
                    torch.manual_seed(seed)
                    noise = torch.randn((1, 3, config.IMG_SIZE, config.IMG_SIZE))
                    noise_tensors.append(noise)
                
                batch_noise = torch.cat(noise_tensors, dim=0)

                # Generate images from the noise
                shape = (current_batch_size, 3, config.IMG_SIZE, config.IMG_SIZE)
                generated_images = diffusion.p_sample_loop(model, shape, noise=batch_noise)
                final_images = generated_images[-1]

                # Save the generated images
                for j, img in enumerate(final_images):
                    img_num = i + j
                    save_image(img, os.path.join(config.GENERATED_IMAGE_DIR, f"generated_image_{img_num}.png"), normalize=True)
                
                pbar.update(current_batch_size)

    print(f"{num_images} images generated and saved in {config.GENERATED_IMAGE_DIR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images using a trained DDPM model.")
    parser.add_argument("-n", "--num_images", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for generation.")
    args = parser.parse_args()

    generate(args.num_images, args.batch_size)
