import argparse
from pathlib import Path

import torch

from diffusers import StableDiffusionPipeline

# --- 1. Default Configuration ---
DEFAULT_BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_LORA_FILE = "/home/kaloyan/Code/Thesis/SeedGan/diffision/my_crop_lora_output/pytorch_lora_weights.safetensors"
DEFAULT_OUTPUT_DIR = "/home/kaloyan/Code/Thesis/SeedGan/diffision/output/"
DEFAULT_PROMPT = "a photo of a single green crop average size"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, cartoon, watermark, text, person, bird, many crops, fields, hands, human, animals,"
DEFAULT_LORA_WEIGHT = 0.8
DEFAULT_IMAGE_BASE_NAME = "crop_diffusion"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple images with a LoRA model."
    )
    parser.add_argument(
        "num_images",
        type=int,
        nargs="?",
        default=1,
        help="Number of images to generate. Defaults to 1.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"The prompt to use for image generation. Defaults to: '{DEFAULT_PROMPT}'",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL_ID,
        help="Path to the base model.",
    )
    parser.add_argument(
        "--lora_file",
        type=str,
        default=DEFAULT_LORA_FILE,
        help="Path to the LoRA .safetensors file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="The negative prompt.",
    )
    parser.add_argument(
        "--lora_weight",
        type=float,
        default=DEFAULT_LORA_WEIGHT,
        help="Strength of the LoRA.",
    )
    parser.add_argument(
        "--image_base_name",
        type=str,
        default=DEFAULT_IMAGE_BASE_NAME,
        help="Base name for the output images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to("cuda")

    print(f"Loading LoRA weights from: {args.lora_file}")
    pipe.load_lora_weights(args.lora_file)
    print("LoRA loaded successfully.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_images} image(s) with prompt: '{args.prompt}'")

    for i in range(args.num_images):
        image_number = i + 1
        image_filename = f"{args.image_base_name}{image_number}.png"
        output_path = Path(args.output_dir) / image_filename

        print(f"[{image_number}/{args.num_images}] Generating {output_path}...")

        # By not setting a seed or generator, diffusers will use a random seed for each generation.
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": args.lora_weight},
        ).images[0]

        image.save(output_path)
        print(f"Image saved to: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
