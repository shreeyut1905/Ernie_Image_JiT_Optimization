#!/usr/bin/env python3
from typing import List, Optional, Union
import argparse
import os
import re
from datetime import datetime

import torch
from diffusers import ErnieImagePipeline, PipelineQuantizationConfig, TorchAoConfig
from diffusers.models.transformers.transformer_ernie_image import (
    ErnieImageTransformer2DModel,
    ErnieImageTransformer2DModelOutput,
)
from torchao.quantization import Float8WeightOnlyConfig

from util_seacache import apply_sea_with_scheduler, rel_l1


def ernie_seacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    text_bth: torch.Tensor,
    text_lens: torch.Tensor,
    return_dict: bool = True,
) -> Union[torch.FloatTensor, ErnieImageTransformer2DModelOutput]:
    """
    Drop-in replacement for ErnieImageTransformer2DModel.forward with SeaCache gating.

    Gating logic follows the Flux script pattern:
    - Track accumulated rescaled relative L1 distance over filtered first-layer modulated inputs.
    - Skip heavy transformer layers when below threshold by reusing the previous residual.
    """
    device, dtype = hidden_states.device, hidden_states.dtype
    batch_size, _, height, width = hidden_states.shape
    patch_size = self.patch_size
    hp, wp = height // patch_size, width // patch_size
    num_img_tokens = hp * wp

    img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
    if self.text_proj is not None and text_bth.numel() > 0:
        text_bth = self.text_proj(text_bth)
    tmax = text_bth.shape[1]
    text_sbh = text_bth.transpose(0, 1).contiguous()

    x = torch.cat([img_sbh, text_sbh], dim=0)
    seq_len = x.shape[0]

    text_ids = (
        torch.cat(
            [
                torch.arange(tmax, device=device, dtype=torch.float32).view(1, tmax, 1).expand(batch_size, -1, -1),
                torch.zeros((batch_size, tmax, 2), device=device),
            ],
            dim=-1,
        )
        if tmax > 0
        else torch.zeros((batch_size, 0, 3), device=device)
    )

    grid_yx = torch.stack(
        torch.meshgrid(
            torch.arange(hp, device=device, dtype=torch.float32),
            torch.arange(wp, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    image_ids = torch.cat(
        [
            text_lens.float().view(batch_size, 1, 1).expand(-1, num_img_tokens, -1),
            grid_yx.view(1, num_img_tokens, 2).expand(batch_size, -1, -1),
        ],
        dim=-1,
    )
    rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

    valid_text = (
        torch.arange(tmax, device=device).view(1, tmax) < text_lens.view(batch_size, 1)
        if tmax > 0
        else torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
    )
    attention_mask = torch.cat([torch.ones((batch_size, num_img_tokens), device=device, dtype=torch.bool), valid_text], dim=1)[
        :, None, None, :
    ]

    sample = self.time_proj(timestep)
    sample = sample.to(dtype=dtype)
    c = self.time_embedding(sample)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
        t.unsqueeze(0).expand(seq_len, -1, -1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
    ]

    # ---- SeaCache gating ----
    should_calc = True
    if getattr(self, "enable_seacache", False):
        # Match the first layer's input modulation to track per-step changes.
        x_norm = self.layers[0].adaLN_sa_ln(x)
        x_mod = (x_norm.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)

        # Only image tokens are spatially structured; use them for SEA distance.
        img_mod = x_mod[:num_img_tokens].transpose(0, 1).contiguous()  # [B, N_img, H]
        img_mod = img_mod.reshape(batch_size, hp, wp, img_mod.shape[-1])
        img_mod = apply_sea_with_scheduler(
            img_mod,
            self.scheduler,
            getattr(self, "cnt", 0),
            power_exp=2.0,
            dims=(-2, -3),
            norm_mode="mean",
        )
        img_mod = img_mod.reshape(batch_size, num_img_tokens, img_mod.shape[-1])

        if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.previous_modulated_input is None:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            self.accumulated_rel_l1_distance += rel_l1(img_mod, self.previous_modulated_input)
            if self.accumulated_rel_l1_distance < float(self.seacache_thresh):
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0

        self.previous_modulated_input = img_mod
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    # ---- Main block compute / skip ----
    if getattr(self, "enable_seacache", False) and (not should_calc) and (self.previous_residual is not None):
        x = x + self.previous_residual
    else:
        ori_x = x
        temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(
                    layer,
                    x,
                    rotary_pos_emb,
                    temb,
                    attention_mask,
                )
            else:
                x = layer(x, rotary_pos_emb, temb, attention_mask)

        if getattr(self, "enable_seacache", False):
            self.previous_residual = x - ori_x

    x = self.final_norm(x, c).type_as(x)
    patches = self.final_linear(x)[:num_img_tokens].transpose(0, 1).contiguous()
    output = (
        patches.view(batch_size, hp, wp, patch_size, patch_size, self.out_channels)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(batch_size, self.out_channels, height, width)
    )

    if not return_dict:
        return (output,)
    return ErnieImageTransformer2DModelOutput(sample=output)


# Replace Diffusers model forward
ErnieImageTransformer2DModel.forward = ernie_seacache_forward


def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_filename(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z._-]", "", name)
    return name or "img"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ernie-Image + SeaCache generation (CPU load + TorchAO FP8 quantization + GPU inference)."
    )
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to text file (one prompt per line).")
    parser.add_argument("--prompt", type=str, default="A serene mountain lake at sunset, cinematic lighting")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--model_id", type=str, default="Baidu/ERNIE-Image")
    parser.add_argument("--seacache_thresh", type=float, default=0.3)
    parser.add_argument("--use_pe", action="store_true", help="Enable prompt enhancer in Ernie pipeline.")
    parser.add_argument("--compile_transformer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prompt_file:
        prompts = read_prompts(args.prompt_file)
        prompt_source = args.prompt_file
    elif args.prompt:
        prompts = [args.prompt]
        prompt_source = "<inline-prompt>"
    else:
        raise ValueError("Provide --prompt_file or --prompt")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load on CPU with TorchAO FP8 quantization mapping (matches main.py intent).
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_mapping={"transformer": TorchAoConfig(Float8WeightOnlyConfig())}
    )
    pipe = ErnieImagePipeline.from_pretrained(
        args.model_id,
        quantization_config=pipeline_quant_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # 2) Move quantized model to GPU for inference.
    pipe = pipe.to("cuda")
    # pipe.transformer.to(memory_format=torch.channels_last)
    # if args.compile_transformer:
        # pipe.transformer.compile(mode="max-autotune", fullgraph=False)

    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.enable_seacache = True
    tr.seacache_thresh = float(args.seacache_thresh)

    base_seed = int(args.seed)
    global_sample_idx = 0

    print(
        f"[{now_str()}] Start | model_id={args.model_id} | steps={args.num_inference_steps} | "
        f"guidance={args.guidance} | seacache_thresh={args.seacache_thresh} | "
        f"prompts={len(prompts)} | seed={base_seed} | quant=torchao-fp8"
    )

    height = (args.height // 16) * 16
    width = (args.width // 16) * 16

    for p_idx, prompt in enumerate(prompts):
        for i in range(int(args.num_images_per_prompt)):
            seed_this = base_seed + global_sample_idx

            tr.cnt = 0
            tr.num_steps = int(args.num_inference_steps)
            tr.accumulated_rel_l1_distance = 0.0
            tr.previous_modulated_input = None
            tr.previous_residual = None

            generator = torch.Generator(device="cuda").manual_seed(42)
            _ = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance),
                num_images_per_prompt=1,
                use_pe=bool(args.use_pe),
                generator=generator,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance),
                num_images_per_prompt=1,
                use_pe=bool(args.use_pe),
                generator=generator,
            )
            end.record()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"Inference time: {start.elapsed_time(end)} ms")
            # image.save(f"output_fp8_{Prompt}.png")
            img = out.images[0]
            base = safe_filename(prompt)[:80]
            fn = f"ErnieSeaCache_{global_sample_idx:05d}-{base}.png"
            img.save(os.path.join(args.output_dir, fn))

            print(f"  - [{p_idx + 1}/{len(prompts)}] #{i + 1}/{args.num_images_per_prompt} saved {fn}")
            global_sample_idx += 1

    print(
        "\n".join(
            [
                "",
                "[Generation Summary]",
                f"  Output dir:                 {args.output_dir}",
                f"  Model id:                   {args.model_id}",
                f"  Steps per image:            {args.num_inference_steps}",
                f"  Guidance:                   {args.guidance}",
                f"  SeaCache threshold:         {args.seacache_thresh}",
                f"  Prompt source:              {prompt_source}",
                f"  Seed (base):                {base_seed}",
                f"  Total images:               {global_sample_idx}",
                f"  Size (WxH):                 {width}x{height}",
            ]
        )
    )


if __name__ == "__main__":
    main()
