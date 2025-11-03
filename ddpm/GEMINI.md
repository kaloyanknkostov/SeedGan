# Gemini Workspace Context: DDPM Project

## Project Overview

This project aims to create a Denoising Diffusion Probabilistic Model (DDPM) using PyTorch. The model will be trained on a custom dataset of 14,000 crop images to generate new, similar images.

*   **Objective:** Generate 64x64 pixel images of crops.
*   **Core Technology:** PyTorch
*   **Model Architecture:** U-Net
*   **Dataset:** 14,000 images of crops located at `/home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled4x`.
*   **Training Started:** 03:00, 3rd November 2025.

## Project Structure

The project follows this structure:

```
/home/kaloyan/Code/Thesis/SeedGan/ddpm/
├── config.py         # Hyperparameters and configuration settings.
├── dataset.py        # Custom dataset loader and preprocessor.
├── diffusion.py      # Diffusion and denoising process logic.
├── generate.py       # Script to generate images from a trained model.
├── model.py          # U-Net model architecture.
├── train.py          # Main training script with checkpointing.
└── requirements.txt  # Python dependencies.
```

## Building and Running

### 1. Create Conda Environment

It is recommended to use a Conda environment for this project.

```bash
# Create a new conda environment named 'ddpm'
conda create --name ddpm python=3.9

# Activate the environment
conda activate ddpm
```

### 2. Install Dependencies

First, install PyTorch manually with the correct CUDA version for your system. For example:

```bash
# Example for CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Then, install the rest of the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Train the Model

To start training the model, run:

```bash
python train.py
```

The training script will automatically save checkpoints and can be resumed if interrupted.

### 4. Generate Images

To generate images using the latest trained model, run:

```bash
python generate.py
```

## Development Conventions

*   **Checkpointing:** The training script (`train.py`) saves a `latest_checkpoint.pth` file after every epoch, allowing you to stop and resume training at any time. To save disk space, a numbered checkpoint is only saved every 10 epochs.
*   **Checkpoint Size:** Each checkpoint file is approximately 700MB.
*   **Configuration:** All hyperparameters are managed in `config.py` for easy tuning.
*   **Image Size:** The model is configured to work with 64x64 images.