#!/usr/bin/env python3
# ruff: noqa: T201
"""VFM-SCD one-step conditional video generation.

Implements Algorithm 1 from "Variational Flow Maps: Make Some Noise for
One-Step Conditional Generation" (Mammadov et al., 2026, arXiv:2603.07276)
combined with the SCD encoder-decoder architecture.

Key difference from standard SCD inference:
    Standard SCD: z ~ N(0,I) → 8-30 denoising steps → video
    VFM-SCD:      z ~ qφ(z|y) → 1-4 steps → video

The noise adapter qφ produces observation-dependent noise, so the flow map
(SCD decoder) can produce high-quality conditional samples in far fewer steps.

Usage:
    # One-step image-to-video (requires first frame image)
    python scripts/vfm_inference.py \
        --first-frame /path/to/frame.png \
        --adapter-path /path/to/noise_adapter.safetensors \
        --lora-path /path/to/lora_weights.safetensors \
        --num-steps 1 \
        --output output.mp4

    # 4-step generation with cached embeddings
    python scripts/vfm_inference.py \
        --cached-embedding /path/to/conditions_final/000000.pt \
        --adapter-path /path/to/noise_adapter.safetensors \
        --lora-path /path/to/lora_weights.safetensors \
        --num-steps 4 \
        --num-seconds 5 \
        --output output.mp4
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch import Tensor

# Add parent package to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VFM-SCD one-step conditional video generation")

    # Model paths
    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors",
                        help="Path to base LTX-2 model")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to trained VFM-SCD LoRA weights")
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path to trained noise adapter weights")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Path to VAE decoder (auto-detected from model path if not set)")

    # Conditioning
    parser.add_argument("--cached-embedding", type=str, default=None,
                        help="Path to cached text embedding .pt file")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (requires loading text encoder)")
    parser.add_argument("--first-frame", type=str, default=None,
                        help="Path to first frame image for i2v conditioning")
    parser.add_argument("--task", type=str, default="i2v",
                        choices=["i2v", "inpaint", "sr", "denoise", "t2v"],
                        help="Inverse problem task class")

    # Generation
    parser.add_argument("--num-steps", type=int, default=1,
                        help="Number of sampling steps (1 = one-step VFM)")
    parser.add_argument("--num-seconds", type=float, default=5.0,
                        help="Duration of generated video in seconds")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Noise sampling temperature (lower = more deterministic)")

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])

    # SCD config
    parser.add_argument("--encoder-layers", type=int, default=32)
    parser.add_argument("--decoder-combine", type=str, default="token_concat")

    return parser.parse_args()


@torch.no_grad()
def vfm_sample(
    scd_model: nn.Module,
    noise_adapter: nn.Module,
    video_latents_clean: Tensor,
    video_positions: Tensor,
    video_prompt_embeds: Tensor,
    prompt_attention_mask: Tensor,
    task_class: Tensor,
    tokens_per_frame: int,
    num_frames: int,
    num_steps: int = 1,
    temperature: float = 1.0,
    sigma_head: nn.Module | None = None,
) -> Tensor:
    """VFM multi-step conditional sampling (Algorithm 1 from paper).

    Args:
        scd_model: LTXSCDModel with trained LoRA
        noise_adapter: Trained noise adapter qφ
        video_latents_clean: Clean conditioning latents [B, seq_len, C]
        video_positions: Position embeddings [B, 3, seq_len, 2]
        video_prompt_embeds: Text embeddings
        prompt_attention_mask: Text attention mask
        task_class: [B] task class indices
        tokens_per_frame: Spatial tokens per frame
        num_frames: Number of video frames
        num_steps: Number of sampling steps (K in Algorithm 1)
        temperature: Noise sampling temperature
        sigma_head: Optional per-token sigma predictor. If provided, uses
            SigmaHead output as per-token timesteps instead of uniform t=1.0.
            This matches training when per_token_sigma=True.

    Returns:
        Generated video latents [B, seq_len, C]
    """
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.scd_model import shift_encoder_features

    B = video_latents_clean.shape[0]
    seq_len = video_latents_clean.shape[1]
    device = video_latents_clean.device
    dtype = video_latents_clean.dtype

    # === Step 1: Encoder pass (clean latents, σ=0, causal mask) ===
    encoder_timesteps = torch.zeros(B, seq_len, device=device, dtype=dtype)
    encoder_modality = Modality(
        enabled=True,
        latent=video_latents_clean,
        timesteps=encoder_timesteps,
        positions=video_positions,
        context=video_prompt_embeds,
        context_mask=prompt_attention_mask,
    )

    encoder_video_args, _ = scd_model.forward_encoder(
        video=encoder_modality,
        audio=None,
        perturbations=None,
        tokens_per_frame=tokens_per_frame,
    )

    # Shift encoder features by 1 frame
    encoder_features = encoder_video_args.x
    shifted_features = shift_encoder_features(encoder_features, tokens_per_frame, num_frames)

    # === Step 2: Sample structured noise from adapter ===
    # Algorithm 1, line 3: z ← μφ(y,c) + σφ(y,c) ⊙ ε
    z, adapter_mu = noise_adapter.sample_with_mu(
        encoder_features=shifted_features,
        task_class=task_class,
        temperature=temperature,
    )

    # === Step 2b: Per-token timesteps from SigmaHead (if available) ===
    if sigma_head is not None and adapter_mu is not None:
        per_token_sigmas = sigma_head(adapter_mu.float())  # [B, seq]
        # First frame conditioning tokens get σ=0
        per_token_sigmas[:, :tokens_per_frame] = 0.0
        timesteps = per_token_sigmas.to(dtype)
    else:
        timesteps = torch.ones(B, seq_len, device=device, dtype=dtype)

    # === Step 3: One-step or multi-step decode ===
    if num_steps == 1:
        # One-step: direct flow map evaluation
        decoder_modality = Modality(
            enabled=True,
            latent=z,
            timesteps=timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Decoder predicts velocity v, so x_0 = z - v
        v_pred, _ = scd_model.forward_decoder_per_frame(
            video=decoder_modality,
            encoder_features=shifted_features,
            perturbations=None,
            tokens_per_frame=tokens_per_frame,
            num_frames=num_frames,
        )

        x = z - v_pred  # Flow matching: x_0 = noise - velocity
    else:
        # Multi-step: Algorithm 1, lines 5-7
        # Time partition: 1 = t_0 > t_1 > ... > t_K = 0
        time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        x = z
        for k in range(num_steps):
            t_k = time_steps[k]
            t_km1 = time_steps[k + 1]
            dt = t_km1 - t_k  # Negative (going from 1 to 0)

            # Scale per-token timesteps for this step
            if sigma_head is not None:
                step_timesteps = timesteps * t_k  # Scale down per token
            else:
                step_timesteps = torch.full((B, seq_len), t_k.item(), device=device, dtype=dtype)

            decoder_modality = Modality(
                enabled=True,
                latent=x,
                timesteps=step_timesteps,
                positions=video_positions,
                context=video_prompt_embeds,
                context_mask=prompt_attention_mask,
            )

            v_pred, _ = scd_model.forward_decoder_per_frame(
                video=decoder_modality,
                encoder_features=shifted_features,
                perturbations=None,
                tokens_per_frame=tokens_per_frame,
                num_frames=num_frames,
            )

            # Euler step: x_{k+1} = x_k + dt * u_θ(x_k, t_k, t_{k-1})
            # In our formulation: velocity v = z - x_0, so flow direction is -v
            x = x + dt * (-v_pred)

    return x


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"VFM-SCD Inference")
    print(f"  Steps: {args.num_steps} | Task: {args.task}")
    print(f"  Resolution: {args.width}x{args.height} | Duration: {args.num_seconds}s")
    print(f"  Temperature: {args.temperature}")

    # Load model (delegated to SCD inference utilities)
    # This is a placeholder — in practice, reuse scd_inference.py's model loading
    print("\nTo run inference, integrate with the existing scd_inference.py pipeline.")
    print("The key change is replacing the noise sampling and denoising loop with:")
    print("  1. noise_adapter.sample(encoder_features, task_class)")
    print("  2. vfm_sample() with num_steps=1-4")
    print("\nSee vfm_inference.py:vfm_sample() for the core algorithm.")


if __name__ == "__main__":
    main()
