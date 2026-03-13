#!/usr/bin/env python3
# ruff: noqa: T201
"""A/B benchmark: BézierFlow learned schedule vs default linear schedule.

Runs the teacher ODE on a few samples with both schedules and compares:
  - MSE(x̂₀, x₀_gt): How close the denoised output is to ground truth
  - Velocity magnitude profile: How the model's effort distributes across steps
  - Step-by-step convergence: L2 distance to GT at each sigma

Usage:
    uv run python scripts/benchmark_schedule.py \
        --data-root /media/12TB/ddit_ditto_data \
        --bezier-schedule /media/2TB/omnitransfer/output/bezierflow_base/schedule.pt \
        --num-samples 10 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_ode(transformer, z_init, sigmas, positions, prompt_embeds, prompt_mask,
            x0_gt, device, dtype, total_tokens):
    """Run teacher ODE and return per-step metrics."""
    from ltx_core.model.transformer.modality import Modality

    x_t = z_init.clone()
    step_metrics = []

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            dt = sigma_next - sigma

            timesteps = torch.full((1, total_tokens), sigma.item(), device=device, dtype=dtype)
            video_mod = Modality(
                enabled=True, latent=x_t, timesteps=timesteps,
                positions=positions, context=prompt_embeds, context_mask=prompt_mask,
            )
            v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)

            # Denoised prediction at this step
            x0_pred = (x_t.float() - v_pred.float() * sigma.float()).to(dtype)

            # Euler step
            x_t = (x_t.float() + v_pred.float() * dt.float()).to(dtype)

            # Metrics
            mse_to_gt = (x0_pred.float() - x0_gt.float()).pow(2).mean().item()
            v_mag = v_pred.float().norm(dim=-1).mean().item()
            step_metrics.append({
                "sigma": sigma.item(),
                "mse_to_gt": mse_to_gt,
                "v_magnitude": v_mag,
            })

    # Final state MSE
    final_mse = (x_t.float() - x0_gt.float()).pow(2).mean().item()
    return final_mse, step_metrics


def main():
    parser = argparse.ArgumentParser(description="A/B benchmark: BézierFlow vs linear schedule")
    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--bezier-schedule", type=str, required=True,
                        help="Path to learned BézierFlow schedule.pt")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    latent_frames = (args.num_frames - 1) // 8 + 1
    total_tokens = latent_frames * latent_h * latent_w

    # ── Discover samples ──
    latents_dir = data_root / "latents_19b"
    if not latents_dir.exists():
        latents_dir = data_root / "latents"
    conditions_dir = data_root / "conditions_final"

    latent_files = sorted(latents_dir.glob("*.pt"))
    samples = []
    for lf in latent_files:
        cf = conditions_dir / lf.name
        if cf.exists():
            samples.append((lf, cf))
    samples = samples[:args.num_samples]

    print()
    print("=" * 70)
    print("  Schedule A/B Benchmark: BézierFlow vs Linear")
    print("=" * 70)
    print(f"  Samples:    {len(samples)}")
    print(f"  Steps:      {args.num_steps}")
    print(f"  Resolution: {args.width}x{args.height}")
    print("=" * 70)

    # ── Load transformer ──
    print("\nLoading transformer...")
    t0 = time.time()
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)

    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        transformer = quantize_model(transformer, args.quantize, device=str(device))

    transformer = transformer.to(device)
    transformer.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Build schedules ──
    # Linear
    from ltx_core.components.schedulers import LTX2Scheduler
    dummy_latent = torch.zeros(1, latent_channels, latent_frames, latent_h, latent_w)
    linear_sigmas = LTX2Scheduler().execute(steps=args.num_steps, latent=dummy_latent).to(device)
    del dummy_latent

    # BézierFlow
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
        bezier_sigmas = learned_sched.get_sigma_schedule(args.num_steps).to(device)

    print(f"\n  Linear σ:  {[f'{s:.4f}' for s in linear_sigmas.tolist()]}")
    print(f"  Bézier σ:  {[f'{s:.4f}' for s in bezier_sigmas.tolist()]}")

    # ── Positions ──
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
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

    # ── Run A/B comparison ──
    print(f"\nRunning {len(samples)} samples with both schedules...\n")

    linear_mses = []
    bezier_mses = []
    linear_step_metrics = []
    bezier_step_metrics = []

    for i, (latent_path, cond_path) in enumerate(samples):
        sample_idx = int(latent_path.stem)
        sample_seed = args.seed + sample_idx

        lat_data = torch.load(latent_path, map_location="cpu", weights_only=True)
        cond_data = torch.load(cond_path, map_location="cpu", weights_only=True)

        x0_raw = lat_data["latents"].unsqueeze(0).to(device, dtype=dtype)
        x0 = patchifier.patchify(x0_raw)

        prompt_embeds = cond_data["video_prompt_embeds"].unsqueeze(0).to(device, dtype=dtype)
        prompt_mask = cond_data["prompt_attention_mask"].unsqueeze(0).to(device)

        generator = torch.Generator(device=device).manual_seed(sample_seed)
        z = torch.randn(1, total_tokens, latent_channels, device=device, dtype=dtype, generator=generator)

        # Run with linear schedule
        lin_mse, lin_steps = run_ode(
            transformer, z, linear_sigmas, positions, prompt_embeds, prompt_mask,
            x0, device, dtype, total_tokens,
        )
        linear_mses.append(lin_mse)
        linear_step_metrics.append(lin_steps)

        # Run with BézierFlow schedule (same z_init)
        bez_mse, bez_steps = run_ode(
            transformer, z, bezier_sigmas, positions, prompt_embeds, prompt_mask,
            x0, device, dtype, total_tokens,
        )
        bezier_mses.append(bez_mse)
        bezier_step_metrics.append(bez_steps)

        delta = lin_mse - bez_mse
        winner = "BÉZIER" if delta > 0 else "LINEAR"
        print(f"  Sample {sample_idx:06d}: Linear MSE={lin_mse:.6f}  Bézier MSE={bez_mse:.6f}  "
              f"Δ={delta:+.6f} ({winner})")

    # ── Summary ──
    avg_lin = sum(linear_mses) / len(linear_mses)
    avg_bez = sum(bezier_mses) / len(bezier_mses)
    improvement = (avg_lin - avg_bez) / avg_lin * 100

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Avg Linear MSE:  {avg_lin:.6f}")
    print(f"  Avg Bézier MSE:  {avg_bez:.6f}")
    print(f"  Improvement:     {improvement:+.2f}%")
    print(f"  Bézier wins:     {sum(1 for l, b in zip(linear_mses, bezier_mses) if b < l)}/{len(samples)}")
    print()

    # Per-step convergence comparison (averaged)
    n_steps = args.num_steps
    print("  Step-by-step convergence (avg MSE to GT):")
    print(f"  {'Step':>4}  {'σ_lin':>8}  {'MSE_lin':>10}  {'σ_bez':>8}  {'MSE_bez':>10}  {'Winner':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}")
    for step in range(n_steps):
        avg_lin_mse = sum(m[step]["mse_to_gt"] for m in linear_step_metrics) / len(samples)
        avg_bez_mse = sum(m[step]["mse_to_gt"] for m in bezier_step_metrics) / len(samples)
        sig_lin = linear_step_metrics[0][step]["sigma"]
        sig_bez = bezier_step_metrics[0][step]["sigma"]
        w = "BÉZIER" if avg_bez_mse < avg_lin_mse else "LINEAR"
        print(f"  {step:4d}  {sig_lin:8.4f}  {avg_lin_mse:10.6f}  {sig_bez:8.4f}  {avg_bez_mse:10.6f}  {w:>8}")

    # Velocity magnitude comparison
    print()
    print("  Velocity magnitude profile (avg ‖v‖):")
    print(f"  {'Step':>4}  {'‖v‖_lin':>10}  {'‖v‖_bez':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}")
    for step in range(n_steps):
        avg_lin_v = sum(m[step]["v_magnitude"] for m in linear_step_metrics) / len(samples)
        avg_bez_v = sum(m[step]["v_magnitude"] for m in bezier_step_metrics) / len(samples)
        print(f"  {step:4d}  {avg_lin_v:10.4f}  {avg_bez_v:10.4f}")

    print("=" * 70)

    # Cleanup
    del transformer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
