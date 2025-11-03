import os
import torch
from torchvision.utils import save_image
import argparse

import config
from model import SimpleUnet
from diffusion import Diffusion

def generate(num_images=16):
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

    # Generate and save images
    with torch.no_grad():
        shape = (num_images, 3, config.IMG_SIZE, config.IMG_SIZE)
        generated_images = diffusion.p_sample_loop(model, shape)
        # The p_sample_loop returns a list of images at each timestep. We take the last one.
        final_images = generated_images[-1]
        for i, img in enumerate(final_images):
            save_image(img, os.path.join(config.GENERATED_IMAGE_DIR, f"generated_image_{i}.png"), normalize=True)

    print(f"{num_images} images generated and saved in {config.GENERATED_IMAGE_DIR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images using a trained DDPM model.")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate.")
    args = parser.parse_args()

    generate(args.num_images)
