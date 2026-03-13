#!/usr/bin/env python3
# ruff: noqa: T201
"""Vanilla VFM one-step conditional video generation (NO SCD split).

Uses the full 48-layer transformer as the flow map.
Text embeddings serve as the observation for the noise adapter.

Usage:
    python scripts/vfm_vanilla_inference.py \
        --cached-embedding /media/2TB/omnitransfer/data/ditto_1sample/conditions_final/000000.pt \
        --adapter-path /media/2TB/omnitransfer/output/vfm_vanilla_overfit_1sample/checkpoints/noise_adapter_step_00500.safetensors \
        --lora-path /media/2TB/omnitransfer/output/vfm_vanilla_overfit_1sample/checkpoints/lora_weights_step_00500.safetensors \
        --num-steps 1 \
        --output /media/2TB/omnitransfer/inference/vfm_vanilla_test.mp4
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla VFM one-step video generation")

    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors")
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--text-encoder-path", type=str,
                        default="/media/2TB/ltx-models/gemma")

    cond_group = parser.add_mutually_exclusive_group(required=True)
    cond_group.add_argument("--cached-embedding", type=str)
    cond_group.add_argument("--prompt", type=str)

    parser.add_argument("--task", type=str, default="i2v",
                        choices=["i2v", "inpaint", "sr", "denoise", "t2v"])
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=25,
                        help="Pixel frames (must satisfy frames %% 8 == 1)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae-device", type=str, default="cuda:1")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])

    # Noise adapter config (must match training)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-num-layers", type=int, default=4)
    parser.add_argument("--adapter-variant", type=str, default="mlp",
                        choices=["mlp", "transformer", "v1b"])

    # v1b-specific adapter config
    parser.add_argument("--adapter-num-heads", type=int, default=8)
    parser.add_argument("--adapter-pos-dim", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    assert args.height % 32 == 0, f"Height must be divisible by 32, got {args.height}"
    assert args.width % 32 == 0, f"Width must be divisible by 32, got {args.width}"
    assert args.num_frames % 8 == 1, f"Frames must satisfy frames%%8==1, got {args.num_frames}"

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    tokens_per_frame = latent_h * latent_w
    latent_frames = (args.num_frames - 1) // 8 + 1
    total_tokens = tokens_per_frame * latent_frames

    print()
    print("=" * 65)
    print("  Vanilla VFM Inference (Full 48-layer Transformer)")
    print("=" * 65)
    if args.cached_embedding:
        print(f"  Embedding:   {args.cached_embedding}")
    else:
        print(f"  Prompt:      {args.prompt[:60]}...")
    print(f"  Resolution:  {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"  Frames:      {args.num_frames} pixel / {latent_frames} latent")
    print(f"  Tokens:      {total_tokens} ({tokens_per_frame}/frame x {latent_frames})")
    print(f"  VFM Steps:   {args.num_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Task:        {args.task}")
    print(f"  Quantize:    {args.quantize}")
    print(f"  Output:      {args.output}")
    print("=" * 65)

    t_start = time.time()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Load text embeddings
    # ══════════════════════════════════════════════════════════════════
    if args.cached_embedding:
        print(f"\n[1/5] Loading cached embedding...")
        emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
        prompt_embeds = emb["video_prompt_embeds"].unsqueeze(0).to(dtype)
        prompt_mask = emb["prompt_attention_mask"].unsqueeze(0)
        print(f"  Shape: {prompt_embeds.shape} (dim={prompt_embeds.shape[-1]})")
    else:
        print(f"\n[1/5] Loading text encoder on {args.vae_device}...")
        from ltx_trainer.model_loader import load_text_encoder
        text_encoder = load_text_encoder(
            args.model_path, args.text_encoder_path,
            device=args.vae_device, dtype=dtype,
        )
        text_encoder.eval()
        with torch.inference_mode():
            video_embeds, _audio_embeds, attention_mask = text_encoder(args.prompt)
            prompt_embeds = video_embeds.to(dtype)
            prompt_mask = attention_mask
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)
    text_embed_dim = prompt_embeds.shape[-1]

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Load transformer + quantize + LoRA (NO SCD wrap)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[2/5] Loading transformer...")
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)

    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantize})...")
        transformer = quantize_model(transformer, args.quantize, device=str(device))

    print(f"  Moving to {device}...")
    transformer = transformer.to(device)

    # Apply LoRA
    print(f"  Loading LoRA: {Path(args.lora_path).name}")
    lora_state = load_file(str(args.lora_path))

    # Detect LoRA config
    target_modules = set()
    lora_rank = None
    for key, value in lora_state.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
        # Extract module name: base_model.model.transformer_blocks.0.attn1.to_q.lora_A.weight → to_q
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p in ("to_k", "to_q", "to_v", "to_out"):
                mod = p if i + 1 >= len(parts) or parts[i + 1] != "0" else f"{p}.0"
                # Check if next part is "0" (for to_out.0)
                if i + 1 < len(parts) and parts[i + 1] == "0" and parts[i + 2] in ("lora_A", "lora_B"):
                    mod = f"{p}.0"
                else:
                    mod = p
                target_modules.add(mod)

    print(f"  LoRA rank={lora_rank}, modules={sorted(target_modules)}")

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=sorted(target_modules),
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state)
    transformer = transformer.merge_and_unload()
    transformer.eval()

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  GPU memory: {mem_gb:.1f} GB")

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Load noise adapter
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3/5] Loading noise adapter ({args.adapter_variant})...")

    if args.adapter_variant == "v1b":
        from ltx_core.model.transformer.noise_adapter_v1b import (
            TASK_CLASSES,
            create_noise_adapter_v1b,
        )
        noise_adapter = create_noise_adapter_v1b(
            text_dim=text_embed_dim,
            latent_dim=latent_channels,
            hidden_dim=args.adapter_hidden_dim,
            num_heads=args.adapter_num_heads,
            num_layers=args.adapter_num_layers,
            pos_dim=args.adapter_pos_dim,
        )
    else:
        from ltx_core.model.transformer.noise_adapter import TASK_CLASSES, create_noise_adapter
        noise_adapter = create_noise_adapter(
            input_dim=text_embed_dim,
            latent_dim=latent_channels,
            variant=args.adapter_variant,
            hidden_dim=args.adapter_hidden_dim,
            num_layers=args.adapter_num_layers,
        )

    adapter_state = load_file(args.adapter_path)
    noise_adapter.load_state_dict(adapter_state)
    noise_adapter = noise_adapter.to(device=device)  # Keep float32 (LayerNorm requires it)
    noise_adapter.eval()

    adapter_params = sum(p.numel() for p in noise_adapter.parameters())
    print(f"  Adapter: {adapter_params / 1e6:.1f}M params, variant={args.adapter_variant}")

    task_idx = TASK_CLASSES.get(args.task, 0)
    task_class = torch.tensor([task_idx], device=device)
    print(f"  Task: {args.task} (class {task_idx})")

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Load VAE decoder
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[4/5] Loading VAE decoder on {args.vae_device}...")
    from ltx_trainer.model_loader import load_video_vae_decoder
    vae_decoder = load_video_vae_decoder(args.model_path, device=args.vae_device, dtype=dtype)
    vae_decoder.eval()

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Generate video
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Generating video with {args.num_steps} VFM step(s)...")
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Positions for all frames
    pos_shape = VideoLatentShape(
        frames=latent_frames, height=latent_h, width=latent_w,
        batch=1, channels=latent_channels,
    )
    coords = patchifier.get_patch_grid_bounds(output_shape=pos_shape, device=device)
    positions = get_pixel_coords(
        latent_coords=coords, scale_factors=scale_factors, causal_fix=True
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / args.fps

    gen_start = time.time()

    with torch.inference_mode():
        # ── Noise adapter: structured noise ──
        if args.adapter_variant == "v1b":
            # v1b: pass full text + positions → per-token μ,σ
            mu, log_sigma = noise_adapter.forward(
                text_embeddings=prompt_embeds.float(),
                text_mask=prompt_mask.bool(),
                positions=positions.float(),
                task_class=task_class,
            )
        else:
            # v1a: pool text → tile → same μ,σ for all tokens
            mask = prompt_mask.unsqueeze(-1).float()
            pooled_text = (prompt_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            text_obs = pooled_text.unsqueeze(1).expand(-1, total_tokens, -1)
            mu, log_sigma = noise_adapter.forward(text_obs.float(), task_class)
        sigma = torch.exp(log_sigma)
        eps = torch.randn(mu.shape, device=device, dtype=torch.float32, generator=generator)
        z = (mu + sigma * eps * args.temperature).to(dtype)

        print(f"  Adapter: μ={mu.float().mean().item():.4f}, "
              f"σ={sigma.float().mean().item():.4f}")

        # ── Flow map evaluation ──
        if args.num_steps == 1:
            # One-step: full transformer forward at t=1
            timesteps = torch.ones(1, total_tokens, device=device, dtype=dtype)

            video_mod = Modality(
                enabled=True,
                latent=z,
                timesteps=timesteps,
                positions=positions,
                context=prompt_embeds,
                context_mask=prompt_mask,
            )

            v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)

            # x_0 = z - v
            x = z - v_pred
        else:
            # Multi-step Euler
            time_steps = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device)
            x = z

            for k in range(args.num_steps):
                t_k = time_steps[k]
                t_km1 = time_steps[k + 1]
                dt = t_km1 - t_k

                timesteps = torch.full(
                    (1, total_tokens), t_k.item(), device=device, dtype=dtype,
                )

                video_mod = Modality(
                    enabled=True,
                    latent=x,
                    timesteps=timesteps,
                    positions=positions,
                    context=prompt_embeds,
                    context_mask=prompt_mask,
                )

                v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                x = x + dt * (-v_pred)

                print(f"  Step {k+1}/{args.num_steps}: t={t_k.item():.3f}→{t_km1.item():.3f}")

    gen_time = time.time() - gen_start
    print(f"\n  Generation: {gen_time:.2f}s")

    # ── Unpatchify + VAE decode ──
    print("  Decoding latents to video...")
    x_latent = patchifier.unpatchify(
        x,
        output_shape=VideoLatentShape(
            frames=latent_frames, height=latent_h, width=latent_w,
            batch=1, channels=latent_channels,
        ),
    )

    del transformer, noise_adapter
    gc.collect()
    torch.cuda.empty_cache()

    x_latent = x_latent.to(args.vae_device)
    with torch.inference_mode():
        video_pixels = vae_decoder(x_latent)

    # VAE outputs in [-1, 1] range — rescale to [0, 1] for video encoding
    video_pixels = (video_pixels.float().cpu() * 0.5 + 0.5).clamp(0, 1)

    # ── Save video ──
    print(f"  Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torchvision
    frames = video_pixels[0].permute(1, 2, 3, 0)  # [T, H, W, 3]
    frames = (frames * 255).to(torch.uint8)

    torchvision.io.write_video(
        str(output_path), frames, fps=args.fps,
        video_codec="libx264", options={"crf": "18"},
    )

    total_time = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Done! {output_path}")
    print(f"  Total: {total_time:.1f}s | Generation: {gen_time:.1f}s")
    print(f"{'=' * 65}")

    del vae_decoder
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
