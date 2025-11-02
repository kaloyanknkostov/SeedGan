from pathlib import Path

import torch

from diffusers import StableDiffusionPipeline

# --- 1. Configuration ---
# !! IMPORTANT: Change these paths to your actual files !!

# Path to the base model (the same one you trained on)
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Path to your NEWLY trained LoRA .safetensors file
# This is the file in your output directory
LORA_FILE = "/home/kaloyan/Code/Thesis/SeedGan/diffision/my_crop_lora_output/pytorch_lora_weights.safetensors"

# The directory where you want to save your generated images
OUTPUT_DIR = "/home/kaloyan/Code/Thesis/SeedGan/generated_images"

# --- 2. Generation Parameters ---
# Try prompts based on your captions, e.g., "a photo of a wheat"
PROMPT = "a photo of a  green crop coming out of the snowy ground"
NEGATIVE_PROMPT = "blurry, low quality, cartoon, watermark, text, person, bird"
LORA_WEIGHT = 0.8  # The strength of your LoRA (0.0 to 1.0)
IMAGE_FILENAME = "my_first_crop1.png"


# --- 3. Load Pipeline and Generate ---
def main():
    print(f"Loading base model: {BASE_MODEL_ID}")
    # Load the pipeline in fp16 for fast inference and low VRAM
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")

    print(f"Loading LoRA weights from: {LORA_FILE}")
    # Load the LoRA weights on top of the base model
    pipe.load_lora_weights(LORA_FILE)
    print("LoRA loaded successfully.")

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / IMAGE_FILENAME

    print(f"Generating image with prompt: {PROMPT}")

    # Generate the image
    # We pass the LoRA weight using `cross_attention_kwargs={"scale": ...}`
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": LORA_WEIGHT},
    ).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
