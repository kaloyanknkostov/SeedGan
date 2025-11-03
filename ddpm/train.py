import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import get_dataloader
from diffusion import Diffusion
from model import SimpleUnet


def train():
    # Create directories if they don't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.GENERATED_IMAGE_DIR, exist_ok=True)

    # Initialize model, optimizer, and diffusion
    model = SimpleUnet().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    diffusion = Diffusion(timesteps=config.T, device=config.DEVICE)
    dataloader = get_dataloader()

    # Check for the latest checkpoint
    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    start_epoch = 0
    global_step = 0
    if os.path.exists(latest_checkpoint_path):
        print("Resuming from latest checkpoint...")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']

    for epoch in range(start_epoch, config.EPOCHS):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        optimizer.zero_grad()
        for i, images in enumerate(pbar):
            images = images.to(config.DEVICE)

            # Learning rate warmup
            if global_step < config.WARMUP_STEPS:
                lr = config.LEARNING_RATE * (global_step + 1) / config.WARMUP_STEPS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Sample a random timestep
            t = torch.randint(0, config.T, (config.BATCH_SIZE,), device=config.DEVICE).long()

            # Add noise to the images
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(x_start=images, t=t, noise=noise)

            # Predict the noise
            predicted_noise = model(x_t, t)

            # Calculate the loss
            loss = F.l1_loss(noise, predicted_noise)
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            epoch_loss += loss.item()

            # Backpropagate
            loss.backward()

            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Epoch {epoch}/{config.EPOCHS} | Loss: {loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f} | LR: {current_lr:.6f}")

        epoch_loss /= len(dataloader)
        if global_step >= config.WARMUP_STEPS:
            scheduler.step(epoch_loss)

        # Save a numbered checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"ddpm_checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
            }, checkpoint_path)

        # Always save the latest checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
        }, latest_checkpoint_path)

        # Generate and save a sample image
        with torch.no_grad():
            shape = (1, 3, config.IMG_SIZE, config.IMG_SIZE)
            generated_images = diffusion.p_sample_loop(model, shape)
            img = generated_images[-1]
            save_image(
                img,
                os.path.join(config.GENERATED_IMAGE_DIR, f"sample_epoch_{epoch}.png"),
                normalize=True,
            )


if __name__ == "__main__":
    train()
