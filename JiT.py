"""
Just-in-Time (JiT): Training-Free Spatial Acceleration for Diffusion Transformers
CVPR 2026

Pipeline implementation for Ernie Image model with Spatially Approximated Generative ODE (SAG-ODE)
and Deterministic Micro-Flow (DMF) stage transitions.

Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
# from diffusers.pipelines.flux2.pipeline_output import 
from diffusers.pipelines.ernie_image.pipeline_output import ErnieImagePipelineOutput
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, retrieve_timesteps, compute_empirical_mu, retrieve_latents
from diffusers.pipelines.ernie_image.pipeline_ernie_image import ErnieImagePipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


class ErnieImagePipeline_JiT(ErnieImagePipeline):

    def set_params(
        self,
        preset=None,
        total_steps=None,
        sparsity_ratios=None,
        stage_ratios=[0.4, 0.65, 1.0],
        use_adaptive=True,
        use_checkerboard_init=True,
        microflow_relax_steps=3,
    ):
        """
        Configure JiT pipeline parameters for FLUX2-Klein.

        Args:
            preset: Preset configuration ('default_4x', 'default_7x', or None for custom)
            total_steps: Total ODE integration steps
            stage_ratios: Stage transition boundaries
            sparsity_ratios: Token density per stage
            use_adaptive: Enable variance-driven token activation
            use_checkerboard_init: Use checkerboard pattern for initial anchors
            microflow_relax_steps: DMF interpolation substeps
        """
        if preset == 'default_7x':
            total_steps = 11
            stage_ratios = [0.4, 0.65, 1.0]
            sparsity_ratios = [0.32, 0.6, 1.0]
            use_checkerboard_init = True
            use_adaptive = True
            microflow_relax_steps = 3
        elif preset == 'default_4x':
            total_steps = 18
            stage_ratios = [0.4, 0.65, 1.0]
            sparsity_ratios = [0.35, 0.62, 1.0]
            use_checkerboard_init = True
            use_adaptive = True
            microflow_relax_steps = 3

        if total_steps is None:
            raise ValueError("total_steps must be specified or use a preset")

        self.num_stages = len(stage_ratios)
        self.params = {
            "total_steps": total_steps,
            "stage_ratios": stage_ratios,
            "sparsity_ratios": sparsity_ratios,
            "stage_steps": [int(total_steps * r) for r in stage_ratios],
            "use_adaptive": use_adaptive,
            "use_checkerboard_init": use_checkerboard_init,
            "microflow_relax_steps": microflow_relax_steps,
        }

        self.active_token_indices = {}

        # Precompute coordinate grid for interpolation
        self._coords_full = None
        self._H_packed = None
        self._W_packed = None

        print(f"JiT Configuration:")
        print(f"  Total steps: {total_steps}")
        print(f"  Stage divisions: {self.params['stage_steps']}")
        print(f"  Token densities: {sparsity_ratios}")
        print(f"  Adaptive densification: {use_adaptive}")

    def _ratio_of_stage(self, stage_k: int) -> float:
        ratios = self.params["sparsity_ratios"]
        if ratios[0] <= ratios[-1]:  # Ascending: [few, ..., many]
            return ratios[self.num_stages - 1 - stage_k]
        else:  # Descending: [many, ..., few]
            return ratios[stage_k]

    def _create_sparse_grid(
        self,
        H_packed: int,
        W_packed: int,
        sparsity_ratio: float,
        device: torch.device,
        use_checkerboard: bool = False
    ) -> torch.Tensor:
        """
        Initialize sparse anchor token set (Ω_K) for SAG-ODE.

        Args:
            H_packed, W_packed: Token grid dimensions
            sparsity_ratio: Target token density [0, 1]
            device: torch device
            use_checkerboard: Use fixed checkerboard pattern

        Returns:
            1D tensor of anchor token indices
        """
        N_packed = H_packed * W_packed
        m_k = int(N_packed * sparsity_ratio)

        if use_checkerboard:
            i_coords = torch.arange(H_packed, device=device)
            j_coords = torch.arange(W_packed, device=device)
            ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')

            all_indices = torch.arange(N_packed, device=device)
            mask_core = (ii % 2 == 0) & (jj % 2 == 0)
            mask_boundary = (ii == 0) | (ii == H_packed - 1) | (jj == 0) | (jj == W_packed - 1)
            mask_combined = mask_core | mask_boundary
            indices = all_indices[mask_combined.flatten()]
        else:
            stride = max(1, int(np.sqrt(1.0 / sparsity_ratio)))
            grid_h = torch.arange(0, H_packed, stride, device=device)
            grid_w = torch.arange(0, W_packed, stride, device=device)

            if len(grid_h) == 0 or grid_h[-1] != H_packed - 1:
                grid_h = torch.cat([grid_h, torch.tensor([H_packed-1], device=device)])
            if len(grid_w) == 0 or grid_w[-1] != W_packed - 1:
                grid_w = torch.cat([grid_w, torch.tensor([W_packed-1], device=device)])

            mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
            indices = mesh_h.flatten() * W_packed + mesh_w.flatten()

        if len(indices) < m_k:
            all_indices_set = set(range(N_packed))
            available = list(all_indices_set - set(indices.tolist()))
            if available:
                n_supplement = min(m_k - len(indices), len(available))
                supplement = torch.tensor(
                    np.random.choice(available, n_supplement, replace=False),
                    device=device
                )
                indices = torch.cat([indices, supplement])
        elif len(indices) > m_k:
            perm = torch.randperm(len(indices), device=device)[:m_k]
            indices = indices[perm]

        return indices.long()
    
    def _calculate_blur_params(self, sparsity_ratio, c=0.4):
        if sparsity_ratio <= 0.0:
            return 3, 1.0
        if sparsity_ratio >= 1.0:
            return 3, 1.0

        characteristic_distance_L = 1.0 / np.sqrt(sparsity_ratio)
        sigma = c * characteristic_distance_L
        sigma = np.clip(sigma, a_min=1.0, a_max=10.0) 
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        
        return kernel_size, sigma

    def _precompute_coords(self, H_packed, W_packed, device):
        """Precompute coordinate grid for interpolation."""
        if self._coords_full is None or self._H_packed != H_packed or self._W_packed != W_packed:
            coords_y, coords_x = torch.meshgrid(
                torch.arange(H_packed, device=device),
                torch.arange(W_packed, device=device),
                indexing="ij"
            )
            self._coords_full = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=-1)
            self._H_packed = H_packed
            self._W_packed = W_packed

    def _irregular_interpolation(self, y_active, active_indices, N_full, d, H_packed, W_packed, device, dtype):
        """
        Spatial interpolation operator for SAG-ODE velocity approximation.

        Args:
            y_active: [B, M, d] anchor token values
            active_indices: [M] anchor positions
            N_full: Total token count
            d: Token dimension
            H_packed, W_packed: Grid dimensions
            device, dtype: torch device and dtype

        Returns:
            [B, N_full, d] interpolated full-dimensional tensor
        """
        if active_indices.numel() == 0:
            return torch.zeros(y_active.size(0), N_full, d, device=device, dtype=dtype)

        B, M, _ = y_active.shape

        # Step 1: Nearest neighbor fill using precomputed coordinates
        coords_active = self._coords_full[active_indices]
        dist = torch.cdist(self._coords_full.float(), coords_active.float(), p=2)
        nearest_idx = dist.argmin(dim=-1)
        y_active_expanded = y_active.permute(1, 0, 2)
        gathered = y_active_expanded[nearest_idx]
        y_full_nearest = gathered.permute(1, 0, 2).contiguous()

        # Step 2: Masked Gaussian blur
        y_full_2d_nearest = y_full_nearest.reshape(B, H_packed, W_packed, d).permute(0, 3, 1, 2)
        sparsity_ratio = len(active_indices) / N_full
        kernel_size, sigma = self._calculate_blur_params(sparsity_ratio, c=0.4)
        if kernel_size % 2 == 0:
            kernel_size += 1

        y_full_2d_blur = gaussian_blur(
            y_full_2d_nearest,
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
        )

        active_mask_1d = torch.zeros(N_full, device=device, dtype=dtype)
        active_mask_1d[active_indices] = 1.0
        active_mask_2d = active_mask_1d.reshape(1, 1, H_packed, W_packed).expand(B, -1, -1, -1)
        inactive_mask_2d = 1.0 - active_mask_2d

        y_full_2d_final = y_full_2d_nearest * active_mask_2d + y_full_2d_blur * inactive_mask_2d
        y_full = y_full_2d_final.permute(0, 2, 3, 1).reshape(B, N_full, d)

        return y_full  
 
    def _compute_importance_map(self, y_full, velocity_full, active_indices, H_packed, W_packed):
        """
        Zero-cost importance map for adaptive token activation.

        Computes local variance of cached velocity to identify high-frequency regions.

        Args:
            y_full: [B, N, d] full latent state
            velocity_full: [B, N, d] cached velocity field
            active_indices: Current anchor indices
            H_packed, W_packed: Grid dimensions

        Returns:
            [H_packed, W_packed] variance-based importance map
        """
        B, N, d = y_full.shape
        device = y_full.device

        v_2d = velocity_full.reshape(B, H_packed, W_packed, d).permute(0, 3, 1, 2)

        kernel_size = 3
        mean = F.avg_pool2d(v_2d, kernel_size, stride=1, padding=kernel_size//2)
        var = F.avg_pool2d(v_2d**2, kernel_size, stride=1, padding=kernel_size//2) - mean**2
        importance = var.mean(dim=1).squeeze(0)

        return importance
    
    def _adaptive_densify(self, current_indices, target_count, importance_map, H_packed, W_packed):
        """
        Variance-driven token activation for stage transitions.

        Args:
            current_indices: Current anchor indices
            target_count: Target active token count
            importance_map: [H, W] importance map
            H_packed, W_packed: Grid dimensions

        Returns:
            Updated active token indices
        """
        device = importance_map.device
        N_packed = H_packed * W_packed

        importance_map = (importance_map - importance_map.min()) / (
            importance_map.max() - importance_map.min() + 1e-8
        )

        importance_flat = importance_map.flatten()
        mask = torch.ones(N_packed, dtype=torch.bool, device=device)
        mask[current_indices] = False

        candidate_indices = torch.arange(N_packed, device=device)[mask]
        candidate_importance = importance_flat[candidate_indices]

        num_to_add = target_count - len(current_indices)

        if num_to_add <= 0:
            return current_indices

        if num_to_add >= len(candidate_indices):
            new_indices = torch.cat([current_indices, candidate_indices])
        else:
            probabilities = candidate_importance / (candidate_importance.sum() + 1e-8)
            _, top_k_indices = torch.topk(probabilities, num_to_add)
            selected = candidate_indices[top_k_indices]
            new_indices = torch.cat([current_indices, selected])

        return new_indices.long()

    def _extract_active_tokens(self, y_full, active_indices):
        """Extract active tokens from full state."""
        return y_full[:, active_indices, :], active_indices

    def _timestep_to_sigma(self, timestep):
        """Convert scheduler timestep (often 0..num_train_timesteps) to sigma in [0, 1]."""
        denom = float(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        if torch.is_tensor(timestep):
            sigma = timestep.to(torch.float32) / denom
            return sigma.clamp_(0.0, 1.0)

        sigma = float(timestep) / denom
        return max(0.0, min(1.0, sigma))

    def _compute_variance_schedule(self, timestep):
        sigma = self._timestep_to_sigma(timestep)
        return sigma**2

    # ========== Deterministic Micro-Flow (DMF) ==========

    def _microflow_bridge(self, y_full, new_indices, y_target_new):
        """
        Deterministic Micro-Flow (DMF) for smooth stage transitions.

        Args:
            y_full: [B, N, d] full latent state
            new_indices: Newly activated token indices
            y_target_new: [B, M_new, d] target values for new tokens

        Returns:
            Updated full latent state
        """
        if new_indices.numel() == 0:
            return y_full

        steps = self.params.get("microflow_relax_steps", 3)
        if steps <= 0:
            y_full[:, new_indices, :] = y_target_new
        else:
            current = y_full[:, new_indices, :].clone()
            weight = 1.0 / steps
            y_full[:, new_indices, :] = (1 - weight) * current + weight * y_target_new

        return y_full

    # ========== Auxiliary Methods ==========

    def _prepare_latent_image_ids(self, indices, H_packed, W_packed, device, dtype):
        """
        Generate positional encodings for active tokens (FLUX2 uses 4D coords).

        Args:
            indices: Active token indices
            H_packed, W_packed: Grid dimensions
            device, dtype: torch device and dtype

        Returns:
            [M, 4] positional IDs (t, h, w, l)
        """
        t = torch.zeros(H_packed, W_packed, 1, device=device, dtype=dtype)
        h = torch.arange(H_packed, device=device, dtype=dtype).view(H_packed, 1, 1).expand(-1, W_packed, -1)
        w = torch.arange(W_packed, device=device, dtype=dtype).view(1, W_packed, 1).expand(H_packed, -1, -1)
        l = torch.zeros(H_packed, W_packed, 1, device=device, dtype=dtype)

        latent_image_ids = torch.cat([t, h, w, l], dim=-1)
        latent_image_ids = latent_image_ids.reshape(H_packed * W_packed, 4)
        return latent_image_ids[indices]

    def _predict_x0_latent(self, latents, noise_pred, timestep):
        sigma = self._timestep_to_sigma(timestep)
        latents_dtype = latents.dtype
        latents_x0 = (
            latents.to(torch.float32) -
            noise_pred.to(torch.float32) * sigma
        ).to(latents_dtype)
        return latents_x0

    @torch.no_grad()
    #@replace_example_docstring(Flux2KleinPipeline.EXAMPLE_DOC_STRING)


    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,
    ):
        if not hasattr(self, "params"):
             self.set_params(total_steps=num_inference_steps)
        
        # Check params again if needed
        total_steps = self.params["total_steps"]
        stage_steps = self.params["stage_steps"]
        use_adaptive = self.params["use_adaptive"]
        use_checkerboard_init = self.params["use_checkerboard_init"]
        
        device = self._execution_device
        dtype = self.transformer.dtype
        self._guidance_scale = guidance_scale

        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide `prompt` or `prompt_embeds`.")
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"Height and width must be divisible by {self.vae_scale_factor}")

        if prompt is not None and isinstance(prompt, str): prompt = [prompt]
        if negative_prompt is None: negative_prompt = ""
        batch_size = len(prompt) if prompt is not None else len(prompt_embeds)
        if isinstance(negative_prompt, str): negative_prompt = [negative_prompt] * batch_size
        total_batch_size = batch_size * num_images_per_prompt

        # Enhance Prompt
        revised_prompts = None
        if prompt is not None and use_pe and self.pe is not None and self.pe_tokenizer is not None:
            prompt = [self._enhance_prompt_with_pe(p, device, width=width, height=height) for p in prompt]
            revised_prompts = list(prompt)

        # Text Embeds
        if prompt_embeds is not None:
            text_hiddens = prompt_embeds
        else:
            text_hiddens = self.encode_prompt(prompt, device, num_images_per_prompt)
            
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                uncond_text_hiddens = negative_prompt_embeds
            else:
                uncond_text_hiddens = self.encode_prompt(negative_prompt, device, num_images_per_prompt)
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens
            
        text_bth, text_lens = self._pad_text(
            text_hiddens=cfg_text_hiddens, device=device, dtype=dtype, text_in_dim=self.transformer.config.text_in_dim
        )

        latent_h, latent_w = height // self.vae_scale_factor, width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        
        # In Ernie, patch_size=1 so H_packed=latent_h, W_packed=latent_w
        H_packed, W_packed = latent_h, latent_w
        N_packed = H_packed * W_packed

        if latents is None:
            latents = randn_tensor((total_batch_size, latent_channels, latent_h, latent_w), generator=generator, device=device, dtype=dtype)
        
        # Flatten to (B, N, C) for JiT
        B = total_batch_size
        y_full = latents.reshape(B, latent_channels, N_packed).transpose(1, 2).clone()
        d_sz = latent_channels

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        global_noise = torch.randn(B, N_packed, d_sz, device=device, dtype=dtype)

        # Init stage
        current_stage = self.num_stages - 1
        ratio0 = self._ratio_of_stage(current_stage)
        current_indices = self._create_sparse_grid(H_packed, W_packed, ratio0, device, use_checkerboard_init)
        self.active_token_indices[current_stage] = current_indices

        self._precompute_coords(H_packed, W_packed, device)
        last_velocity_full = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t_curr = t
                t_batch = torch.full((B,), t_curr.item(), device=device, dtype=dtype)

                target_stage = 0
                for s_idx, s_step in enumerate(stage_steps):
                    if i < s_step:
                        target_stage = self.num_stages - 1 - s_idx
                        break

                if target_stage < current_stage:
                    target_count = int(N_packed * self._ratio_of_stage(target_stage))
                    if use_adaptive and i > 0 and last_velocity_full is not None:
                        importance_map = self._compute_importance_map(y_full, last_velocity_full, current_indices, H_packed, W_packed)
                        new_indices = self._adaptive_densify(current_indices, target_count, importance_map, H_packed, W_packed)
                    else:
                        new_indices = self._create_sparse_grid(H_packed, W_packed, self._ratio_of_stage(target_stage), device, use_checkerboard_init)

                    newly_activated = new_indices[~torch.isin(new_indices, current_indices)]
                    sigma_target = torch.sqrt(self._compute_variance_schedule(t_curr))
                    noise = global_noise[:, newly_activated, :] * sigma_target
                    
                    t_before = timesteps[i-1]
                    latents_x0 = self._predict_x0_latent(last_y, last_velocity_full, t_before)
                    x0_interpolated = self._irregular_interpolation(
                        latents_x0[:, current_indices, :], current_indices, N_packed, d_sz, H_packed, W_packed, device, dtype
                    )
                    y_target_new = x0_interpolated[:, newly_activated, :] * (1 - sigma_target) + noise
                    y_full = self._microflow_bridge(y_full, newly_activated, y_target_new)

                    current_indices = new_indices
                    self.active_token_indices[target_stage] = current_indices
                    current_stage = target_stage

                # For ERNIE, we pass the full grid, but we can construct it from y_full
                y_full_reshaped = y_full.transpose(1, 2).reshape(B, latent_channels, H_packed, W_packed)
                
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([y_full_reshaped, y_full_reshaped], dim=0)
                    t_batch_in = torch.cat([t_batch, t_batch], dim=0)
                else:
                    latent_model_input = y_full_reshaped
                    t_batch_in = t_batch

                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch_in,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # pred is [B, C, H, W], we want to extract active tokens
                pred_flat = pred.reshape(B, latent_channels, -1).transpose(1, 2)
                pred_active = pred_flat[:, current_indices, :]

                if current_stage > 0:
                    velocity_full_step = self._irregular_interpolation(
                        pred_active, current_indices, N_packed, latent_channels, H_packed, W_packed, device, dtype
                    )
                else:
                    velocity_full_step = torch.zeros_like(y_full)
                    velocity_full_step[:, current_indices, :] = pred_active

                last_velocity_full = velocity_full_step.clone()
                last_y = y_full.clone()

                velocity_full_step_reshaped = velocity_full_step.transpose(1, 2).reshape(B, latent_channels, H_packed, W_packed)
                
                # Sched step expects (B, C, H, W)
                y_full_reshaped = self.scheduler.step(velocity_full_step_reshaped, t, y_full_reshaped).prev_sample
                y_full = y_full_reshaped.reshape(B, latent_channels, -1).transpose(1, 2)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    y_full_reshaped = callback_outputs.pop("latents", y_full_reshaped)
                    y_full = y_full_reshaped.reshape(B, latent_channels, -1).transpose(1, 2)

                progress_bar.set_postfix(stage=current_stage, ratio=f"{len(current_indices)/N_packed:.1%}")
                progress_bar.update()

        if not torch.isfinite(y_full_reshaped).all():
            raise RuntimeError(
                "Non-finite latents detected in JiT denoising. "
                "This is commonly caused by fp16 numerical instability; load the pipeline with torch_dtype=torch.bfloat16."
            )

        if output_type == "latent":
            return y_full_reshaped
        # Match ERNIE latent decode flow: unnormalize with VAE BN stats, then unpatchify.
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device)
        latents = y_full_reshaped * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)

        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            from PIL import Image
            images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ErnieImagePipelineOutput(images=images, revised_prompts=revised_prompts)
