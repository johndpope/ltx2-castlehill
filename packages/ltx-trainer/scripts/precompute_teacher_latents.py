#!/usr/bin/env python3
# ruff: noqa: T201
"""Precompute teacher ODE outputs for DMD2 training.

Runs the base model (no LoRA) through an 8-step Euler ODE for each training
sample, starting from Gaussian noise with a fixed seed. Saves paired
(noise_z, teacher_output) for each sample.

These cached outputs are used as "real" distribution samples in v3a's
discriminator training, eliminating 8 DiT forward passes per training step.

Usage:
    python scripts/precompute_teacher_latents.py \
        --model-path /media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors \
        --data-root /media/12TB/ddit_ditto_data_23_overfit10 \
        --output-dir /media/12TB/ddit_ditto_data_23_overfit10/teacher_latents \
        --num-steps 8 \
        --quantize int8-quanto \
        --device cuda:0
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ltx-core" / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute teacher ODE outputs for DMD2")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to preprocessed dataset (with latents/ and conditions_final/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cached teacher latents")
    parser.add_argument("--num-steps", type=int, default=8,
                        help="Number of Euler ODE steps for teacher")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def run_teacher_ode(
    transformer: torch.nn.Module,
    z: torch.Tensor,
    text_embeds: torch.Tensor,
    positions: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Run teacher 8-step Euler ODE from noise z to clean output."""
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer.modality import Modality
    # timesteps in Modality are raw per-token sigma values, NOT embeddings

    dtype = z.dtype
    B = z.shape[0]
    seq_len = z.shape[1]

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=num_steps).to(device=device)

    x_t = z.clone()
    for i in range(len(sigmas) - 1):
        sc = sigmas[i]
        sn = sigmas[i + 1]
        dt = sn - sc

        sb = sc.unsqueeze(0).expand(B)
        ts = sb.unsqueeze(1).expand(-1, seq_len)  # per-token sigma values

        video_mod = Modality(
            enabled=True, latent=x_t, sigma=sb, timesteps=ts,
            positions=positions, context=text_embeds, context_mask=None,
        )

        result = transformer(video=video_mod, audio=None, perturbations=None)
        # Result is (video_out, audio_out) — video_out may be Modality or Tensor
        vo = result[0] if isinstance(result, tuple) else result
        v = vo.x if hasattr(vo, 'x') else vo
        x_t = x_t + v * dt

    return x_t


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data_root = Path(args.data_root)
    latents_dir = data_root / "latents"
    conds_dir = data_root / "conditions_final"

    # Find all samples
    latent_files = sorted(latents_dir.glob("*.pt"))
    print(f"Found {len(latent_files)} samples in {latents_dir}")

    # ══════════════════════════════════════════════════════════════
    # Load model
    # ══════════════════════════════════════════════════════════════
    print(f"\nLoading transformer from {args.model_path}...")
    t0 = time.time()

    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)

    if args.quantize == "int8-quanto":
        from ltx_trainer.quantization import quantize_model
        print("Quantizing (int8-quanto)...")
        transformer = quantize_model(transformer, precision="int8-quanto", device=args.device)

    transformer = transformer.to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s, VRAM: {torch.cuda.memory_allocated(device) / 1e9:.1f}GB")

    # ══════════════════════════════════════════════════════════════
    # Load embeddings processor (for 3840->4096 projection)
    # ══════════════════════════════════════════════════════════════
    caption_proj = None
    test_cond = torch.load(str(conds_dir / latent_files[0].name), map_location="cpu", weights_only=False)
    if test_cond["video_prompt_embeds"].shape[-1] == 3840:
        print("Loading caption_projection shim (3840 → 4096)...")
        from safetensors import safe_open
        from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
        caption_proj = PixArtAlphaTextProjection(in_features=3840, hidden_size=4096)
        ltx2_path = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors"
        source = ltx2_path if os.path.exists(ltx2_path) else args.model_path
        with safe_open(source, framework="pt") as f:
            prefix = "model.diffusion_model.caption_projection."
            for key in f.keys():
                if key.startswith(prefix):
                    pname = key[len(prefix):]
                    parts = pname.split(".")
                    obj = caption_proj
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    setattr(obj, parts[-1], torch.nn.Parameter(f.get_tensor(key)))
        caption_proj = caption_proj.to(device=device, dtype=dtype).eval()

    # ══════════════════════════════════════════════════════════════
    # Patchifier + position helpers
    # ══════════════════════════════════════════════════════════════
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.types import VideoLatentShape, SpatioTemporalScaleFactors
    from ltx_core.components.patchifiers import get_pixel_coords

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()

    # ══════════════════════════════════════════════════════════════
    # Process each sample
    # ══════════════════════════════════════════════════════════════
    total_time = 0
    for idx, lat_file in enumerate(latent_files):
        out_path = Path(args.output_dir) / lat_file.name
        if out_path.exists():
            print(f"[{idx+1}/{len(latent_files)}] {lat_file.name} — cached, skipping")
            continue

        print(f"[{idx+1}/{len(latent_files)}] {lat_file.name}...", end=" ", flush=True)
        t_start = time.time()

        # Load latent and condition
        lat_data = torch.load(str(lat_file), map_location="cpu", weights_only=False)
        cond_data = torch.load(str(conds_dir / lat_file.name), map_location="cpu", weights_only=False)

        # Get latent shape
        latents = lat_data["latents"]  # [C, F, H, W]
        C, F, H, W = latents.shape

        # Patchify latent and compute positions
        latents_5d = latents.unsqueeze(0).to(device=device, dtype=dtype)  # [1, C, F, H, W]
        patched = patchifier.patchify(latents_5d)  # [1, seq, C]
        seq_len = patched.shape[1]

        latent_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(frames=F, height=H, width=W, batch=1, channels=C),
            device=device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=True,
        ).to(dtype)
        positions[:, 0] = positions[:, 0] / 25.0  # fps normalization

        # Text embeddings
        text_embeds = cond_data["video_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
        if caption_proj is not None and text_embeds.shape[-1] == 3840:
            text_embeds = caption_proj(text_embeds).view(1, -1, 4096)

        # Sample fixed noise (deterministic per sample via seed + idx)
        torch.manual_seed(args.seed + idx)
        z = torch.randn(1, seq_len, C, device=device, dtype=dtype)

        # Run teacher ODE
        x_teacher = run_teacher_ode(
            transformer, z, text_embeds, positions,
            num_steps=args.num_steps, device=device,
        )

        # Save: noise z, teacher output, and metadata
        save_data = {
            "noise_z": z.cpu(),
            "teacher_output": x_teacher.cpu(),
            "latent_shape": torch.tensor([C, F, H, W]),
            "num_steps": torch.tensor([args.num_steps]),
            "seed": torch.tensor([args.seed + idx]),
        }
        torch.save(save_data, str(out_path))

        elapsed = time.time() - t_start
        total_time += elapsed
        print(f"{elapsed:.1f}s ({args.num_steps} steps, {seq_len} tokens)")

    print(f"\nDone! {len(latent_files)} samples, total: {total_time:.1f}s")
    print(f"Saved to: {args.output_dir}")
    print(f"Avg: {total_time / max(len(latent_files), 1):.1f}s/sample")


if __name__ == "__main__":
    main()
