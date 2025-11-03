# Gemini Workspace Context: DDPM Project

## Project Overview

This project aims to create a Denoising Diffusion Probabilistic Model (DDPM) using PyTorch. The model will be trained on a custom dataset of 14,000 crop images to generate new, similar images.

*   **Objective:** Generate 64x64 pixel images of crops.
*   **Core Technology:** PyTorch
*   **Model Architecture:** U-Net
*   **Dataset:** 14,000 images of crops located at `/home/kaloyan/Code/Thesis/SeedGan/data/crops_with_background_scaled4x`.

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

### 1. Install Dependencies

```bash
# TODO: Create requirements.txt
pip install -r requirements.txt
```

### 2. Train the Model

To start training the model, run:

```bash
python train.py
```

The training script will automatically save checkpoints and can be resumed if interrupted.

### 3. Generate Images

To generate images using the latest trained model, run:

```bash
python generate.py
```

## Development Conventions

*   **Checkpointing:** The training script (`train.py`) will save checkpoints to allow for pausing and resuming training.
*   **Configuration:** All hyperparameters are managed in `config.py` for easy tuning.
*   **Image Size:** The model is configured to work with 64x64 images.
