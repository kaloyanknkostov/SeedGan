import torch

# Training hyperparameters
EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WARMUP_STEPS = 5000

# Gradient accumulation
VIRTUAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = VIRTUAL_BATCH_SIZE // BATCH_SIZE

DEVICE = "cuda"

# Diffusion model hyperparameters
T = 1000  # Number of timesteps

# Dataset and image parameters
IMG_SIZE = 64
DATASET_PATH = "/home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled4x"

# Checkpoint and logging parameters
CHECKPOINT_DIR = "checkpoints"
GENERATED_IMAGE_DIR = "output"
