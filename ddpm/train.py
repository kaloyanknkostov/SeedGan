import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

import config
from dataset import get_dataloader
from model import SimpleUnet
from diffusion import Diffusion

def train():
    # Create directories if they don't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.GENERATED_IMAGE_DIR, exist_ok=True)

    # Initialize model, optimizer, and diffusion
    model = SimpleUnet().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    diffusion = Diffusion(timesteps=config.T, device=config.DEVICE)
    dataloader = get_dataloader()

    # Check for the latest checkpoint
    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    start_epoch = 0
    if os.path.exists(latest_checkpoint_path):
        print("Resuming from latest checkpoint...")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, config.EPOCHS):
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(config.DEVICE)

            # Sample a random timestep
            t = torch.randint(0, config.T, (config.BATCH_SIZE,), device=config.DEVICE).long()

            # Add noise to the images
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(x_start=images, t=t, noise=noise)

            # Predict the noise
            predicted_noise = model(x_t, t)

            # Calculate the loss
            loss = F.l1_loss(noise, predicted_noise)

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{config.EPOCHS} | Loss: {loss.item():.4f}")

        # Save a checkpoint after each epoch
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"ddpm_checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        # Save the latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_checkpoint_path)


        # Generate and save a sample image
        if epoch % 10 == 0:
            with torch.no_grad():
                shape = (1, 3, config.IMG_SIZE, config.IMG_SIZE)
                generated_images = diffusion.p_sample_loop(model, shape)
                img = generated_images[-1]
                save_image(img, os.path.join(config.GENERATED_IMAGE_DIR, f"sample_epoch_{epoch}.png"), normalize=True)


if __name__ == '__main__':
    train()
