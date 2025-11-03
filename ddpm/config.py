import torch

# Training hyperparameters
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diffusion model hyperparameters
T = 1000  # Number of timesteps

# Dataset and image parameters
IMG_SIZE = 64
DATASET_PATH = "/home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled4x"

# Checkpoint and logging parameters
CHECKPOINT_DIR = "checkpoints"
GENERATED_IMAGE_DIR = "generated_images"
