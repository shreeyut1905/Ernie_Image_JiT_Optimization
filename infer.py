from JiT import ErnieImagePipeline_JiT
import torch
import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run FLUX2-Klein JiT inference")
    parser.add_argument(
        "--preset",
        type=str,
        default="default_4x",
        help="Preset name for pipeline.set_params, e.g. default_4x or default_7x",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA device id used for inference",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Device setup
    device = torch.device(f"cuda:{args.gpu_id}")
    seed = 42
    g = torch.Generator(device).manual_seed(seed)
     
    # Load pipeline
    model_path = "Baidu/ERNIE-Image" # Replace this with your local path if you are loading the model offline
    # fp16 can produce NaNs with JiT in this model; prefer bf16 for stability.
    pipeline = ErnieImagePipeline_JiT.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline = pipeline.to(device)

    # Use preset (default_4x or default_7x) or custom parameters
    pipeline.set_params(preset=args.preset)
    """
    # Custom parameters example:
    pipeline.set_params(
        total_steps=11,
        stage_ratios=[0.4, 0.65, 1.0],
        sparsity_ratios=[0.32, 0.6, 1.0],
        use_checkerboard_init=True,
        use_adaptive=True,
        microflow_relax_steps=3,
    )
    """
    
    resolution = 1024
    prompt = "A serene lake surrounded by snow-capped mountains at sunset, digital art, highly detailed, cinematic lighting, golden hour, calm water reflecting sky"

    # Generate image
    print(f"Generating: {prompt}")
    _ = pipeline(
        prompt=prompt,
        height=resolution,  
        width=resolution,   
        num_inference_steps=pipeline.params['total_steps'],
        generator=g,
        guidance_scale=4.0

    ).images[0]
    t0 = time.perf_counter()
    image = pipeline(
        prompt=prompt,
        height=resolution,  
        width=resolution,   
        num_inference_steps=pipeline.params['total_steps'],
        generator=g,
        guidance_scale=4.0

    ).images[0]
    elapsed = time.perf_counter() - t0
    print(f"Cost: {elapsed:.3f} s")
    
    # Save output
    out_dir = f"./outputs_ernie/"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{prompt[:50]}.png")
    image.save(output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
