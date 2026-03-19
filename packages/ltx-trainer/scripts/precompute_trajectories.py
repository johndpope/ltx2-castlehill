#!/usr/bin/env python3
# ruff: noqa: T201
"""Pre-compute teacher denoising trajectories for VFM distillation training.

Runs the base LTX-2 model (no LoRA, no adapter) for N steps on each training
sample, saving the full ODE trajectory: z → x_{σ_{N-1}} → ... → x_{σ_1} → x̂_0

This gives VFM training a "correct path" to learn from instead of random
interpolation targets. The adapter learns to produce z that aligns with the
teacher's starting distribution, and the flow map learns the teacher's ODE.

What gets saved per sample:
    trajectories/NNNNNN.pt = {
        "sigmas":       [N+1] float — the sigma schedule used
        "states":       [N+1, seq, 128] — x_t at each sigma (states[0]=z, states[-1]=x̂_0)
        "velocities":   [N, seq, 128] — v_pred at each step
        "x0_pred":      [N, seq, 128] — denoised x̂_0 prediction at each step
        "noise_seed":   int — seed used for z, so adapter can learn to match it
    }

Training with trajectories:
    Instead of: v_target = z - x₀ (random interpolation)
    Use:        v_target = teacher_velocity[step_k] at sigma[k]
    Or:         x0_target = teacher_x0_pred[step_k] (progressive distillation)

Usage:
    # 8-step trajectories (match distilled model's default):
    uv run python scripts/precompute_trajectories.py \
        --data-root /media/2TB/omnitransfer/data/ditto_500sample \
        --num-steps 8 \
        --device cuda:0

    # While VAE encoding runs on cuda:1 (safe to run in parallel)
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
    parser = argparse.ArgumentParser(description="Pre-compute teacher trajectories for VFM distillation")

    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root dir with latents_19b/ and conditions_final/")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for trajectories/ (default: data-root/trajectories)")

    parser.add_argument("--num-steps", type=int, default=8,
                        help="Number of teacher denoising steps")
    parser.add_argument("--bezier-schedule", type=str, default=None,
                        help="Path to a learned BézierFlow/BSplineFlow schedule.pt. "
                        "If provided, uses the learned sigma schedule instead of "
                        "the default LTX2Scheduler linear schedule.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--fps", type=float, default=24.0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (per-sample seed = base_seed + sample_idx)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--skip-samples", type=int, default=0,
                        help="Skip first N samples (for multi-GPU splitting)")

    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    latents_dir = data_root / "latents_19b"
    if not latents_dir.exists():
        latents_dir = data_root / "latents"
    conditions_dir = data_root / "conditions_final"
    traj_dir = Path(args.output_dir) if args.output_dir else data_root / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    latent_frames = (args.num_frames - 1) // 8 + 1
    tokens_per_frame = latent_h * latent_w
    total_tokens = tokens_per_frame * latent_frames

    # Discover samples (need both latent + condition)
    latent_files = sorted(latents_dir.glob("*.pt"))
    samples = []
    for lf in latent_files:
        cf = conditions_dir / lf.name
        if cf.exists():
            samples.append((lf, cf))

    if args.skip_samples:
        samples = samples[args.skip_samples:]
    if args.max_samples:
        samples = samples[:args.max_samples]

    # Filter out already-computed
    todo = [(lf, cf) for lf, cf in samples if not (traj_dir / lf.name).exists()]

    print()
    print("=" * 70)
    print("  Teacher Trajectory Pre-computation")
    print("=" * 70)
    print(f"  Model:       {Path(args.model_path).name}")
    print(f"  Data root:   {data_root}")
    print(f"  Samples:     {len(samples)} total, {len(todo)} to compute, {len(samples) - len(todo)} cached")
    print(f"  Resolution:  {args.width}x{args.height} → latent {latent_w}x{latent_h}")
    print(f"  Frames:      {args.num_frames} pixel / {latent_frames} latent")
    print(f"  Tokens:      {total_tokens}")
    print(f"  Steps:       {args.num_steps}")
    print(f"  Quantize:    {args.quantize}")
    print(f"  Output:      {traj_dir}")
    print("=" * 70)

    if not todo:
        print("\nAll trajectories already computed. Nothing to do.")
        return

    # ── Load transformer ─────────────────────────────────────────────
    print(f"\nLoading teacher transformer...")
    t0 = time.time()
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)

    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantize})...")
        transformer = quantize_model(transformer, args.quantize, device=str(device))

    transformer = transformer.to(device)
    transformer.eval()

    print(f"  Loaded in {time.time() - t0:.1f}s, VRAM: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    # ── Load embeddings processor + caption_projection shim ─────────
    # LTX-2.3 cached embeddings are 3840-dim (from feature extractor precompute).
    # The connector expects 4096-dim (post caption_projection).
    # We load caption_projection from LTX-2 (19B) checkpoint as a shim.
    embeddings_processor = None
    caption_projection_shim = None
    needs_connector = not (hasattr(transformer, 'caption_projection') and transformer.caption_projection is not None)

    if needs_connector:
        print("  LTX-2.3 detected: loading embeddings processor + caption_projection shim...")
        from ltx_trainer.model_loader import load_embeddings_processor
        embeddings_processor = load_embeddings_processor(
            checkpoint_path=args.model_path,
            device=str(device),
            dtype=dtype,
        )
        embeddings_processor.eval()

        # Load caption_projection shim (3840→4096) from LTX-2 checkpoint
        from safetensors import safe_open
        from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
        caption_proj = PixArtAlphaTextProjection(in_features=3840, hidden_size=4096)
        ltx2_path = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors"
        with safe_open(ltx2_path, framework="pt", device="cpu") as f:
            prefix = "model.diffusion_model.caption_projection."
            for key in f.keys():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    param = f.get_tensor(key)
                    parts = param_name.split(".")
                    obj = caption_proj
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], torch.nn.Parameter(param))
        caption_projection_shim = caption_proj.to(device).eval()
        for p in caption_projection_shim.parameters():
            p.requires_grad = False
        print(f"  Embeddings processor + caption_projection shim loaded")

    # ── Compute sigma schedule ───────────────────────────────────────
    if args.bezier_schedule:
        # Load learned BézierFlow/BSplineFlow schedule
        import json
        sched_path = Path(args.bezier_schedule)
        json_path = sched_path.with_suffix(".json")

        sched_type = "bezier"
        if json_path.exists():
            with open(json_path) as f:
                sched_meta = json.load(f)
            sched_type = sched_meta.get("type", "bezier")

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
        # Default: LTX2Scheduler with token-count-dependent shifting
        from ltx_core.components.schedulers import LTX2Scheduler
        dummy_latent = torch.zeros(1, latent_channels, latent_frames, latent_h, latent_w)
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=args.num_steps, latent=dummy_latent).to(device)
        del dummy_latent

        print(f"\n  Sigma schedule ({args.num_steps} steps) — linear:")
        print(f"    {[f'{s:.4f}' for s in sigmas.tolist()]}")

    # ── Setup patchifier + positions ─────────────────────────────────
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

    # ── Process samples ──────────────────────────────────────────────
    print(f"\nComputing trajectories for {len(todo)} samples...")
    computed = 0
    failed = 0

    for latent_path, cond_path in tqdm(todo, desc="Trajectories"):
        sample_idx = int(latent_path.stem)
        sample_seed = args.seed + sample_idx

        try:
            # Load ground truth latent + text embeddings
            lat_data = torch.load(latent_path, map_location="cpu", weights_only=True)
            cond_data = torch.load(cond_path, map_location="cpu", weights_only=True)

            # Patchify ground truth latent: [128, F, H, W] → [1, seq, 128]
            x0_raw = lat_data["latents"].unsqueeze(0).to(device, dtype=dtype)
            x0 = patchifier.patchify(x0_raw)  # [1, total_tokens, 128]

            # Text embeddings — apply connector if needed (3840 → 4096)
            video_features = cond_data["video_prompt_embeds"].unsqueeze(0).to(device, dtype=dtype)
            audio_features = cond_data.get("audio_prompt_embeds", video_features)
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0).to(device, dtype=dtype)
            prompt_mask = cond_data["prompt_attention_mask"].unsqueeze(0).to(device)

            if embeddings_processor is not None:
                from ltx_core.text_encoders.gemma import convert_to_additive_mask
                # Apply caption_projection shim: 3840 → 4096
                if caption_projection_shim is not None:
                    batch_size = video_features.shape[0]
                    video_features = caption_projection_shim(video_features).view(batch_size, -1, 4096)
                # Use video_connector directly (audio connector has different dims in 2.3)
                additive_mask = convert_to_additive_mask(prompt_mask, video_features.dtype)
                prompt_embeds, prompt_mask = embeddings_processor.video_connector(
                    video_features, additive_mask,
                )
                # Convert additive mask back to binary
                prompt_mask = (prompt_mask.squeeze(1).squeeze(1) >= -9000.0).long()
            else:
                prompt_embeds = video_features

            # Sample initial noise z ~ N(0, I) with deterministic seed
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            z = torch.randn(1, total_tokens, latent_channels, device=device, dtype=dtype, generator=generator)

            # ── Run teacher ODE: z → x̂_0 ────────────────────────────
            states = [z.cpu()]       # x_t at each sigma
            velocities = []          # v_pred at each step
            x0_preds = []            # denoised x̂_0 at each step

            x_t = z
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                for step_idx in range(len(sigmas) - 1):
                    sigma = sigmas[step_idx]
                    sigma_next = sigmas[step_idx + 1]
                    dt = sigma_next - sigma

                    # Build modality with current state
                    timesteps = torch.full((1, total_tokens), sigma.item(), device=device, dtype=dtype)
                    video_mod = Modality(
                        enabled=True,
                        sigma=sigma.unsqueeze(0),  # [1,] per-batch sigma for prompt AdaLN (2.3)
                        latent=x_t,
                        timesteps=timesteps,
                        positions=positions,
                        context=prompt_embeds,
                        context_mask=prompt_mask,
                    )

                    # Teacher velocity prediction
                    v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)

                    # Denoised prediction: x̂_0 = x_t - σ * v
                    x0_pred = (x_t.float() - v_pred.float() * sigma.float()).to(dtype)

                    # Euler step: x_{t+1} = x_t + v * dt
                    x_t = (x_t.float() + v_pred.float() * dt.float()).to(dtype)

                    velocities.append(v_pred.cpu())
                    x0_preds.append(x0_pred.cpu())
                    states.append(x_t.cpu())

            # Save trajectory
            traj_data = {
                "sigmas": sigmas.cpu(),                             # [N+1]
                "states": torch.stack(states, dim=0).squeeze(1),    # [N+1, seq, 128]
                "velocities": torch.stack(velocities, dim=0).squeeze(1),  # [N, seq, 128]
                "x0_preds": torch.stack(x0_preds, dim=0).squeeze(1),     # [N, seq, 128]
                "x0_gt": x0.cpu().squeeze(0),                      # [seq, 128] ground truth
                "noise_seed": torch.tensor([sample_seed]),
            }
            torch.save(traj_data, traj_dir / latent_path.name)
            computed += 1

            # Periodic cache clear
            if computed % 50 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  Failed sample {sample_idx}: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"  Computed: {computed}, Failed: {failed}, Total cached: {len(samples) - len(todo) + computed}")
    print(f"  Output:   {traj_dir}")

    # ── Verify a sample ──────────────────────────────────────────────
    sample_files = sorted(traj_dir.glob("*.pt"))
    if sample_files:
        t = torch.load(sample_files[0], map_location="cpu", weights_only=True)
        print(f"\n  Sample trajectory ({sample_files[0].name}):")
        print(f"    sigmas:      {t['sigmas'].shape}")
        print(f"    states:      {t['states'].shape} (z → ... → x̂_0)")
        print(f"    velocities:  {t['velocities'].shape}")
        print(f"    x0_preds:    {t['x0_preds'].shape}")
        print(f"    x0_gt:       {t['x0_gt'].shape}")
        print(f"    noise_seed:  {t['noise_seed'].item()}")

        # Check how close final state is to ground truth
        final_state = t['states'][-1]  # Teacher's denoised output
        x0_gt = t['x0_gt']
        mse = (final_state.float() - x0_gt.float()).pow(2).mean().item()
        print(f"    Teacher MSE (x̂_0 vs GT): {mse:.6f}")

    print(f"{'=' * 70}")

    del transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
