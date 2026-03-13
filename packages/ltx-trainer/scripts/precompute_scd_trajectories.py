#!/usr/bin/env python3
# ruff: noqa: T201
"""Pre-compute SCD decoder trajectories for VFM-SCD distillation.

Unlike precompute_trajectories.py (which runs the full 48-layer base model),
this script operates on the SCD encoder-decoder split:

  1. Encoder (32 layers): Run once on clean latents with causal mask → features
  2. Shift features by 1 frame (causal conditioning)
  3. Decoder (16 layers): Run N denoising steps per-frame → trajectory

The decoder trajectories capture how the 16-layer decoder denoises each frame
given encoder context from the previous frame. This is the correct supervision
signal for VFM-SCD distillation.

What gets saved per sample:
    scd_trajectories/NNNNNN.pt = {
        "sigmas":       [N+1] float — sigma schedule used
        "states":       [N+1, seq, 128] — decoder x_t at each sigma
        "velocities":   [N, seq, 128] — decoder v_pred at each step
        "x0_preds":     [N, seq, 128] — decoder x̂₀ prediction at each step
        "x0_gt":        [seq, 128] — ground truth latent
        "noise_seed":   int — seed used for z
    }

    Note: encoder_features are NOT stored (recomputed during training for
    gradient flow to the noise adapter). This saves ~4x disk per sample
    since encoder features are [seq, 3072] vs latents [seq, 128].

Usage:
    uv run python scripts/precompute_scd_trajectories.py \
        --data-root /media/12TB/ddit_ditto_data \
        --lora-path /path/to/scd_lora_weights.safetensors \
        --num-steps 8 \
        --device cuda:0

    # With BézierFlow schedule:
    uv run python scripts/precompute_scd_trajectories.py \
        --data-root /media/12TB/ddit_ditto_data \
        --lora-path /path/to/scd_lora_weights.safetensors \
        --bezier-schedule /path/to/schedule.pt \
        --num-steps 8 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute SCD decoder trajectories for VFM distillation"
    )

    parser.add_argument(
        "--model-path", type=str,
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors",
    )
    parser.add_argument(
        "--lora-path", type=str, default=None,
        help="Path to SCD LoRA weights (.safetensors). If None, uses base model.",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Root dir with latents_19b/ and conditions_final/",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir (default: data-root/scd_trajectories)",
    )

    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument(
        "--bezier-schedule", type=str, default=None,
        help="Path to learned BézierFlow schedule.pt",
    )
    parser.add_argument("--encoder-layers", type=int, default=32)
    parser.add_argument("--decoder-combine", type=str, default="token_concat")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--fps", type=float, default=24.0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--quantize", type=str, default="int8-quanto",
        choices=["none", "int8-quanto", "fp8-quanto"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    latents_dir = data_root / "latents_19b"
    if not latents_dir.exists():
        latents_dir = data_root / "latents"
    conditions_dir = data_root / "conditions_final"
    traj_dir = Path(args.output_dir) if args.output_dir else data_root / "scd_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    latent_frames = (args.num_frames - 1) // 8 + 1
    tokens_per_frame = latent_h * latent_w
    total_tokens = tokens_per_frame * latent_frames

    # Discover samples
    latent_files = sorted(latents_dir.glob("*.pt"))
    samples = []
    for lf in latent_files:
        cf = conditions_dir / lf.name
        if cf.exists():
            samples.append((lf, cf))

    if args.max_samples:
        samples = samples[: args.max_samples]

    # Filter already-computed
    todo = [(lf, cf) for lf, cf in samples if not (traj_dir / lf.name).exists()]

    print()
    print("=" * 70)
    print("  SCD Decoder Trajectory Pre-computation")
    print("=" * 70)
    print(f"  Model:       {Path(args.model_path).name}")
    print(f"  LoRA:        {args.lora_path or 'None (base model)'}")
    print(f"  Encoder:     {args.encoder_layers} layers")
    print(f"  Decoder:     {48 - args.encoder_layers} layers ({args.decoder_combine})")
    print(f"  Data root:   {data_root}")
    print(f"  Samples:     {len(samples)} total, {len(todo)} to compute, {len(samples) - len(todo)} cached")
    print(f"  Resolution:  {args.width}x{args.height} → latent {latent_w}x{latent_h}")
    print(f"  Frames:      {args.num_frames} pixel / {latent_frames} latent")
    print(f"  Tokens:      {total_tokens} total, {tokens_per_frame}/frame")
    print(f"  Steps:       {args.num_steps}")
    print(f"  Output:      {traj_dir}")
    print("=" * 70)

    if not todo:
        print("\nAll trajectories already computed. Nothing to do.")
        return

    # ── Load base transformer ─────────────────────────────────────
    print("\nLoading base transformer...")
    t0 = time.time()
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)

    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantize})...")
        transformer = quantize_model(transformer, args.quantize, device=str(device))

    transformer = transformer.to(device)
    transformer.eval()

    # ── Apply SCD LoRA if provided ────────────────────────────────
    if args.lora_path:
        print(f"  Loading SCD LoRA from {args.lora_path}...")
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        transformer = get_peft_model(transformer, lora_config)
        lora_state = load_file(args.lora_path, device=str(device))
        transformer.load_state_dict(lora_state, strict=False)
        transformer.eval()
        print(f"  LoRA applied (rank {args.lora_rank})")

    # ── Wrap in SCD model ─────────────────────────────────────────
    from ltx_core.model.transformer.scd_model import LTXSCDModel, shift_encoder_features

    # get_peft_model wraps in PeftModel — unwrap for SCD
    base_for_scd = transformer
    if hasattr(transformer, "base_model") and hasattr(transformer.base_model, "model"):
        base_for_scd = transformer.base_model.model

    scd_model = LTXSCDModel(
        base_model=base_for_scd,
        encoder_layers=args.encoder_layers,
        decoder_input_combine=args.decoder_combine,
    ).to(device)
    scd_model.eval()

    print(f"  Loaded in {time.time() - t0:.1f}s, VRAM: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    # ── Compute sigma schedule ────────────────────────────────────
    if args.bezier_schedule:
        import json
        sched_path = Path(args.bezier_schedule)
        json_path = sched_path.with_suffix(".json")

        sched_type = "bezier"
        if json_path.exists():
            with open(json_path) as f:
                sched_type = json.load(f).get("type", "bezier")

        if sched_type == "bspline":
            from ltx_trainer.bsplineflow import BSplineScheduler
            learned_sched = BSplineScheduler.load(sched_path, device="cpu")
        else:
            from ltx_trainer.bezierflow import BezierScheduler
            learned_sched = BezierScheduler.load(sched_path, device="cpu")

        with torch.no_grad():
            sigmas = learned_sched.get_sigma_schedule(args.num_steps).to(device)

        print(f"\n  Sigma schedule ({args.num_steps} steps) — {sched_type} learned:")
        print(f"    {[f'{s:.4f}' for s in sigmas.tolist()]}")
    else:
        from ltx_core.components.schedulers import LTX2Scheduler
        dummy_latent = torch.zeros(1, latent_channels, latent_frames, latent_h, latent_w)
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=args.num_steps, latent=dummy_latent).to(device)
        del dummy_latent

        print(f"\n  Sigma schedule ({args.num_steps} steps) — linear:")
        print(f"    {[f'{s:.4f}' for s in sigmas.tolist()]}")

    # ── Setup patchifier + positions ──────────────────────────────
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    pos_shape = VideoLatentShape(
        frames=latent_frames, height=latent_h, width=latent_w,
        batch=1, channels=latent_channels,
    )
    coords = patchifier.get_patch_grid_bounds(output_shape=pos_shape, device=device)
    positions = get_pixel_coords(
        latent_coords=coords, scale_factors=scale_factors, causal_fix=True,
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / args.fps

    # ── Process samples ───────────────────────────────────────────
    print(f"\nComputing SCD decoder trajectories for {len(todo)} samples...")
    computed = 0
    failed = 0

    for latent_path, cond_path in tqdm(todo, desc="SCD Trajectories"):
        sample_idx = int(latent_path.stem)
        sample_seed = args.seed + sample_idx

        try:
            # Load ground truth latent + text embeddings
            lat_data = torch.load(latent_path, map_location="cpu", weights_only=True)
            cond_data = torch.load(cond_path, map_location="cpu", weights_only=True)

            # Patchify ground truth latent: [128, F, H, W] → [1, seq, 128]
            x0_raw = lat_data["latents"].unsqueeze(0).to(device, dtype=dtype)
            x0 = patchifier.patchify(x0_raw)  # [1, total_tokens, 128]

            # Text embeddings
            prompt_embeds = cond_data["video_prompt_embeds"].unsqueeze(0).to(device, dtype=dtype)
            prompt_mask = cond_data["prompt_attention_mask"].unsqueeze(0).to(device)

            # ── ENCODER PASS (clean, σ=0, causal mask) ────────────
            encoder_timesteps = torch.zeros(1, total_tokens, device=device, dtype=dtype)
            encoder_modality = Modality(
                enabled=True,
                latent=x0,
                timesteps=encoder_timesteps,
                positions=positions,
                context=prompt_embeds,
                context_mask=prompt_mask,
            )

            with torch.inference_mode():
                encoder_video_args, _ = scd_model.forward_encoder(
                    video=encoder_modality,
                    audio=None,
                    perturbations=None,
                    tokens_per_frame=tokens_per_frame,
                )

            # Shift encoder features by 1 frame
            encoder_features = encoder_video_args.x  # [1, seq, D]
            shifted_features = shift_encoder_features(
                encoder_features, tokens_per_frame, latent_frames,
            )

            # ── DECODER ODE (per-frame, N steps) ──────────────────
            # Sample initial noise
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            z = torch.randn(
                1, total_tokens, latent_channels,
                device=device, dtype=dtype, generator=generator,
            )

            states = [z.cpu()]
            velocities = []
            x0_preds = []

            x_t = z
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                for step_idx in range(len(sigmas) - 1):
                    sigma = sigmas[step_idx]
                    sigma_next = sigmas[step_idx + 1]
                    dt = sigma_next - sigma

                    # Build decoder modality for this step
                    timesteps = torch.full(
                        (1, total_tokens), sigma.item(), device=device, dtype=dtype,
                    )

                    decoder_modality = Modality(
                        enabled=True,
                        latent=x_t,
                        timesteps=timesteps,
                        positions=positions,
                        context=prompt_embeds,
                        context_mask=prompt_mask,
                    )

                    # Run decoder per-frame (matches training/inference)
                    v_pred, _ = scd_model.forward_decoder_per_frame(
                        video=decoder_modality,
                        encoder_features=shifted_features,
                        perturbations=None,
                        tokens_per_frame=tokens_per_frame,
                        num_frames=latent_frames,
                    )

                    # Denoised prediction: x̂₀ = x_t - σ * v
                    x0_pred = (x_t.float() - v_pred.float() * sigma.float()).to(dtype)

                    # Euler step: x_{t+1} = x_t + v * dt
                    x_t = (x_t.float() + v_pred.float() * dt.float()).to(dtype)

                    velocities.append(v_pred.cpu())
                    x0_preds.append(x0_pred.cpu())
                    states.append(x_t.cpu())

            # Save trajectory
            traj_data = {
                "sigmas": sigmas.cpu(),
                "states": torch.stack(states, dim=0).squeeze(1),        # [N+1, seq, 128]
                "velocities": torch.stack(velocities, dim=0).squeeze(1),  # [N, seq, 128]
                "x0_preds": torch.stack(x0_preds, dim=0).squeeze(1),     # [N, seq, 128]
                "x0_gt": x0.cpu().squeeze(0),                            # [seq, 128]
                "noise_seed": torch.tensor([sample_seed]),
            }
            torch.save(traj_data, traj_dir / latent_path.name)
            computed += 1

            if computed % 50 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  Failed sample {sample_idx}: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"  Computed: {computed}, Failed: {failed}, Total cached: {len(samples) - len(todo) + computed}")
    print(f"  Output:   {traj_dir}")

    # Verify a sample
    sample_files = sorted(traj_dir.glob("*.pt"))
    if sample_files:
        t = torch.load(sample_files[0], map_location="cpu", weights_only=True)
        print(f"\n  Sample trajectory ({sample_files[0].name}):")
        print(f"    sigmas:      {t['sigmas'].shape}")
        print(f"    states:      {t['states'].shape} (z → ... → x̂₀)")
        print(f"    velocities:  {t['velocities'].shape}")
        print(f"    x0_preds:    {t['x0_preds'].shape}")
        print(f"    x0_gt:       {t['x0_gt'].shape}")
        print(f"    noise_seed:  {t['noise_seed'].item()}")

        final_state = t["states"][-1]
        x0_gt = t["x0_gt"]
        mse = (final_state.float() - x0_gt.float()).pow(2).mean().item()
        print(f"    Decoder MSE (x̂₀ vs GT): {mse:.6f}")

    print(f"{'=' * 70}")

    del scd_model, transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
