#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- START CONDA ACTIVATION ---
# This is the recommended way to make 'conda activate' work inside a script
# It initializes the conda shell functions
eval "$(conda shell.bash hook)"
# --- END CONDA ACTIVATION ---

# 1. Activate your conda environment
echo "Activating conda environment: sd_lora_env"
conda activate sd_lora_env

# 2. Set environment variables for your paths
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DIR="/home/kaloyan/Code/Thesis/SeedGan/data/768x768_standardized"
export OUTPUT_DIR="/home/kaloyan/Code/Thesis/SeedGan/diffision/my_crop_lora_output"

echo "---------------------------------"
echo "Starting LoRA Training"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "---------------------------------"

# 3. Launch the training
#    (Assumes you are in the 'diffusers/examples/text_to_image' directory)
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --resolution=768 \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=5 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --rank=128 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_checkpointing \
  --validation_prompt="a high-resolution photo of a healthy wheat crop" \
  --report_to="tensorboard"

echo "---------------------------------"
echo "Training complete. Output saved to $OUTPUT_DIR"
echo "---------------------------------"
