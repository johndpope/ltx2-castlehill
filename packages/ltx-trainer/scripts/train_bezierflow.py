#!/usr/bin/env python3
# ruff: noqa: T201
"""Train BézierFlow sigma schedule for SCD inference.

Distills a high-step teacher schedule into a low-step student via learned
Bézier curve reparameterization. Only 32 learnable parameters, ~10 min training.

Based on BézierFlow (ICLR 2026, arXiv:2512.13255).

Two-pass gradient approach:
  1. Forward: Run decoder steps with detached sigmas, cache velocity tensors.
  2. Replay: Reconstruct z_student = z_init + Σ v_i·dt_i where dt_i carries
     gradient through Bézier params. Velocities are constants.

Usage:
    python scripts/train_bezierflow.py \
        --cached-embedding /media/2TB/omnitransfer/data/ditto_subset/conditions_final/000000.pt \
        --output /media/2TB/omnitransfer/output/bezierflow/schedule.pt

    # Then use in inference:
    python scripts/scd_inference.py \
        --bezier-schedule /media/2TB/omnitransfer/output/bezierflow/schedule.pt \
        --num-inference-steps 4 \
        --cached-embedding ... --output ...
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

# Set CUDA arch before importing quanto
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape


def load_conditions_from_dir(
    conditions_dir: Path,
    max_samples: int,
    device: str,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    """Load precomputed text embeddings from conditions_final/ directory."""
    files = sorted(conditions_dir.glob("*.pt"))[:max_samples]
    samples = []
    for f in files:
        cond = torch.load(f, map_location="cpu", weights_only=True)
        if isinstance(cond, dict):
            embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
            mask = cond.get("prompt_attention_mask", None)
        else:
            embeds = cond
            mask = None
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        samples.append({
            "prompt_embeds": embeds.to(device, dtype),
            "prompt_mask": mask.to(device) if mask is not None else None,
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BézierFlow sigma schedule for SCD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--checkpoint",
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors",
        help="Base model checkpoint (distilled recommended)",
    )
    parser.add_argument(
        "--lora-path",
        default="/media/2TB/omnitransfer/output/scd_token_concat/checkpoints/lora_weights_step_01000.safetensors",
        help="SCD LoRA checkpoint (must match --decoder-combine mode)",
    )

    # Data source — either a directory of embeddings or a single file
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-root",
        type=str,
        help="Dataset root with conditions_final/ directory",
    )
    data_group.add_argument(
        "--cached-embedding",
        type=str,
        help="Single cached embedding .pt file",
    )

    # Scheduler type selection
    parser.add_argument(
        "--scheduler-type", default="bezier", choices=["bezier", "bspline"],
        help="Scheduler parameterization: 'bezier' (global Bernstein basis) or 'bspline' (local B-spline basis)",
    )
    parser.add_argument(
        "--bspline-order", type=int, default=4,
        help="B-spline order (degree=order-1). Only used when --scheduler-type=bspline. 4=cubic.",
    )

    # Training params
    parser.add_argument("--teacher-steps", type=int, default=30, help="Teacher (high-quality) denoising steps")
    parser.add_argument("--student-steps", type=int, default=4, help="Student (target) denoising steps")
    parser.add_argument("--n-control-points", type=int, default=32, help="Control points / coefficients")
    parser.add_argument("--num-iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale for teacher and student")
    parser.add_argument("--max-samples", type=int, default=100, help="Max conditioning samples to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # SCD architecture
    parser.add_argument("--encoder-layers", type=int, default=32)
    parser.add_argument("--decoder-combine", default="token_concat", choices=["add", "token_concat"])
    parser.add_argument("--quantization", default="int8-quanto", choices=["fp8-quanto", "int8-quanto", "none"])
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fps", type=float, default=24.0)

    # Output
    parser.add_argument(
        "--output",
        default="/media/2TB/omnitransfer/output/bezierflow/schedule.pt",
        help="Output path for learned schedule",
    )
    parser.add_argument("--wandb-project", default="bezierflow-scd", help="W&B project name (empty to disable)")

    args = parser.parse_args()

    device = "cuda:0"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    latent_h = args.height // 32
    latent_w = args.width // 32
    tokens_per_frame = latent_h * latent_w
    CHUNK_LATENT = 4  # SCD training window

    use_cfg = args.guidance_scale > 1.0

    _sched_name = "BézierFlow" if args.scheduler_type == "bezier" else f"BSplineFlow (order={args.bspline_order})"
    print()
    print("=" * 65)
    print(f"  {_sched_name} Schedule Training")
    print("=" * 65)
    print(f"  Scheduler:      {args.scheduler_type}")
    print(f"  Teacher steps:  {args.teacher_steps}")
    print(f"  Student steps:  {args.student_steps}")
    print(f"  Control points: {args.n_control_points}")
    print(f"  Iterations:     {args.num_iterations}")
    print(f"  LR:             {args.lr}")
    print(f"  CFG:            {args.guidance_scale}")
    print(f"  Resolution:     {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"  Quantization:   {args.quantization}")
    print("=" * 65)

    # ── W&B ──
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.scheduler_type}_s{args.student_steps}_cp{args.n_control_points}",
            )
        except Exception as e:
            print(f"  W&B init failed: {e}, continuing without logging")

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Load conditioning data
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/3] Loading conditioning data...")
    if args.cached_embedding:
        emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
        prompt_embeds = emb.get("video_prompt_embeds", emb.get("prompt_embeds"))
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_mask = emb.get("prompt_attention_mask", None)
        if prompt_mask is not None and prompt_mask.dim() == 1:
            prompt_mask = prompt_mask.unsqueeze(0)
        samples = [{
            "prompt_embeds": prompt_embeds.to(device, dtype),
            "prompt_mask": prompt_mask.to(device) if prompt_mask is not None else None,
        }]
    else:
        data_root = Path(args.data_root)
        conditions_dir = data_root / "conditions_final"
        if not conditions_dir.exists():
            conditions_dir = data_root / "conditions"
        samples = load_conditions_from_dir(conditions_dir, args.max_samples, device, dtype)
    print(f"  Loaded {len(samples)} conditioning samples")

    # Null embeddings for CFG
    null_embeds = torch.zeros_like(samples[0]["prompt_embeds"])
    null_mask = torch.zeros_like(samples[0]["prompt_mask"]) if samples[0]["prompt_mask"] is not None else None

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Load model (transformer → LoRA on CPU → quantize → SCD wrap)
    # PEFT can't wrap QLinear (quantized), so LoRA MUST be applied before quantization.
    # ══════════════════════════════════════════════════════════════════
    print("\n[2/3] Loading model...")
    from ltx_trainer.model_loader import load_transformer

    transformer = load_transformer(args.checkpoint, device="cpu", dtype=torch.bfloat16)

    # Apply LoRA on CPU BEFORE quantization (PEFT can't wrap QLinear)
    if args.lora_path and Path(args.lora_path).exists():
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        import re as _re
        print(f"  Applying LoRA on CPU: {Path(args.lora_path).name}")
        lora_sd = load_file(args.lora_path)
        # Remove ComfyUI prefix
        lora_sd = {k.replace("diffusion_model.", "", 1): v for k, v in lora_sd.items()}
        # Normalize SCD paths (encoder_blocks/decoder_blocks → transformer_blocks)
        normalized = {}
        for key, value in lora_sd.items():
            if key.startswith("encoder_blocks.") or key.startswith("decoder_blocks."):
                continue
            if key.startswith("base_model."):
                key = key[len("base_model."):]
            normalized[key] = value
        lora_sd = normalized
        # Extract target modules and rank
        target_modules = sorted({m.group(1) for key in lora_sd
                                 if (m := _re.match(r"(.+)\.lora_[AB]\.", key))})
        rank = next(t.shape[0] for k, t in lora_sd.items() if "lora_A" in k and t.ndim == 2)
        print(f"  LoRA rank={rank}, {len(target_modules)} target modules")
        lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules, lora_dropout=0.0)
        transformer = get_peft_model(transformer, lora_config)
        set_peft_model_state_dict(transformer.get_base_model(), lora_sd)
        transformer = transformer.get_base_model()  # Unwrap PeftModel → modified LTXModel

    if args.quantization != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantization})...")
        transformer = quantize_model(transformer, args.quantization, device=device)

    transformer = transformer.to(device)

    # Wrap with SCD
    from ltx_core.model.transformer.scd_model import LTXSCDModel
    scd_model = LTXSCDModel(
        base_model=transformer,
        encoder_layers=args.encoder_layers,
        decoder_input_combine=args.decoder_combine,
    )
    scd_model.eval()

    # Set up patchifier and position helpers
    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    latent_channels = scd_model.base_model.patchify_proj.in_features  # 128

    def get_positions(n_frames: int) -> torch.Tensor:
        coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=n_frames, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
            device=device,
        )
        px = get_pixel_coords(latent_coords=coords, scale_factors=scale_factors, causal_fix=True).to(dtype)
        px[:, 0, ...] = px[:, 0, ...] / args.fps
        return px

    def get_positions_for_frame(frame_idx: int) -> torch.Tensor:
        all_pos = get_positions(frame_idx + 1)
        start = frame_idx * tokens_per_frame
        end = start + tokens_per_frame
        return all_pos[:, :, start:end, :]

    # Teacher sigma schedule
    dummy_latent = torch.empty(1, 1, CHUNK_LATENT, latent_h, latent_w)
    teacher_sigmas = LTX2Scheduler().execute(
        steps=args.teacher_steps, latent=dummy_latent,
    ).to(device=device, dtype=dtype)

    print(f"  Teacher sigma range: [{teacher_sigmas[0]:.4f} → {teacher_sigmas[-1]:.4f}]")
    print(f"  Model ready on {device}")

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Train BézierFlow schedule
    # ══════════════════════════════════════════════════════════════════
    sched_label = "BézierFlow" if args.scheduler_type == "bezier" else "BSplineFlow"
    print(f"\n[3/3] Training {sched_label} ({args.num_iterations} iterations)...")

    if args.scheduler_type == "bezier":
        from ltx_trainer.bezierflow import BezierScheduler
        bezier = BezierScheduler(n_control_points=args.n_control_points).to(device)
    else:
        from ltx_trainer.bsplineflow import BSplineScheduler
        bezier = BSplineScheduler(  # variable named 'bezier' for minimal diff
            n_coefficients=args.n_control_points,
            order=args.bspline_order,
        ).to(device)

    optimizer = torch.optim.RMSprop(bezier.parameters(), lr=args.lr, momentum=0.9)

    # Show initial schedule
    with torch.no_grad():
        init_sched = bezier.get_sigma_schedule(args.student_steps)
        print(f"  Initial σ schedule ({args.student_steps} steps): {[f'{s:.4f}' for s in init_sched.tolist()]}")

    best_loss = float("inf")
    best_state = None
    losses = []

    t_train_start = time.time()

    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        # Sample a random conditioning
        sample = random.choice(samples)
        prompt_embeds = sample["prompt_embeds"]
        prompt_mask = sample["prompt_mask"]

        # Frame 0 of a chunk: encode a zero frame, then decode frame 0
        # For simplicity, we train on a single frame (frame_idx=0) with
        # zero encoder features (first frame has no preceding context).
        frame_idx = 0
        positions = get_positions_for_frame(frame_idx)

        # Sample noise (the initial z)
        z_init = torch.randn(1, latent_channels, 1, latent_h, latent_w, device=device, dtype=dtype)

        # ── Encode frame 0 (zero input, σ=0) ──
        with torch.no_grad():
            from ltx_core.model.transformer.scd_model import KVCache
            enc_input = torch.zeros(1, latent_channels, 1, latent_h, latent_w, device=device, dtype=dtype)
            enc_patch = patchifier.patchify(enc_input)
            enc_modality = Modality(
                enabled=True,
                latent=enc_patch,
                timesteps=torch.zeros(1, tokens_per_frame, device=device, dtype=dtype),
                positions=get_positions_for_frame(0),
                context=prompt_embeds,
                context_mask=prompt_mask,
            )
            kv_cache = KVCache.empty()
            kv_cache.is_cache_step = True
            enc_out, _ = scd_model.forward_encoder(
                video=enc_modality, audio=None, perturbations=None,
                kv_cache=kv_cache, tokens_per_frame=tokens_per_frame,
            )
            enc_features = enc_out.x.detach()

            # Null encoder features for CFG unconditional pass
            if use_cfg:
                enc_null_modality = Modality(
                    enabled=True,
                    latent=enc_patch,
                    timesteps=torch.zeros(1, tokens_per_frame, device=device, dtype=dtype),
                    positions=get_positions_for_frame(0),
                    context=null_embeds,
                    context_mask=null_mask,
                )
                kv_cache_null = KVCache.empty()
                kv_cache_null.is_cache_step = True
                enc_out_null, _ = scd_model.forward_encoder(
                    video=enc_null_modality, audio=None, perturbations=None,
                    kv_cache=kv_cache_null, tokens_per_frame=tokens_per_frame,
                )
                enc_features_null = enc_out_null.x.detach()

        # ── TEACHER: Run high-step denoising (no grad) ──
        with torch.no_grad():
            z_teacher = patchifier.patchify(z_init.clone())
            for step in range(args.teacher_steps):
                sigma = teacher_sigmas[step]
                sigma_next = teacher_sigmas[step + 1]

                dec_modality = Modality(
                    enabled=True,
                    latent=z_teacher,
                    timesteps=torch.full((1, tokens_per_frame), sigma.item(), device=device, dtype=dtype),
                    positions=positions,
                    context=prompt_embeds,
                    context_mask=prompt_mask,
                )
                velocity, _ = scd_model.forward_decoder(
                    video=dec_modality, encoder_features=enc_features,
                    audio=None, perturbations=None,
                )

                if use_cfg:
                    uncond_mod = Modality(
                        enabled=True,
                        latent=z_teacher,
                        timesteps=torch.full((1, tokens_per_frame), sigma.item(), device=device, dtype=dtype),
                        positions=positions,
                        context=null_embeds,
                        context_mask=null_mask,
                    )
                    v_uncond, _ = scd_model.forward_decoder(
                        video=uncond_mod, encoder_features=enc_features_null,
                        audio=None, perturbations=None,
                    )
                    velocity = v_uncond + args.guidance_scale * (velocity - v_uncond)

                dt = sigma_next - sigma
                z_teacher = (z_teacher.float() + velocity.float() * dt.float()).to(dtype)

            z_teacher = z_teacher.detach().clone()  # [1, tokens_per_frame, 128]

        # ── STUDENT: Run low-step denoising (cache velocities, then replay) ──
        # Pass 1: Forward with detached sigmas → cache velocities
        student_sigmas = bezier.get_sigma_schedule(args.student_steps)  # differentiable
        cached_velocities = []

        with torch.no_grad():
            z_student_fwd = patchifier.patchify(z_init.clone())
            for step in range(args.student_steps):
                sigma_val = student_sigmas[step].detach().item()
                sigma_next_val = student_sigmas[step + 1].detach().item()

                dec_modality = Modality(
                    enabled=True,
                    latent=z_student_fwd,
                    timesteps=torch.full((1, tokens_per_frame), sigma_val, device=device, dtype=dtype),
                    positions=positions,
                    context=prompt_embeds,
                    context_mask=prompt_mask,
                )
                velocity, _ = scd_model.forward_decoder(
                    video=dec_modality, encoder_features=enc_features,
                    audio=None, perturbations=None,
                )

                if use_cfg:
                    uncond_mod = Modality(
                        enabled=True,
                        latent=z_student_fwd,
                        timesteps=torch.full((1, tokens_per_frame), sigma_val, device=device, dtype=dtype),
                        positions=positions,
                        context=null_embeds,
                        context_mask=null_mask,
                    )
                    v_uncond, _ = scd_model.forward_decoder(
                        video=uncond_mod, encoder_features=enc_features_null,
                        audio=None, perturbations=None,
                    )
                    velocity = v_uncond + args.guidance_scale * (velocity - v_uncond)

                cached_velocities.append(velocity.detach().float().clone())

                dt = sigma_next_val - sigma_val
                z_student_fwd = (z_student_fwd.float() + velocity.float() * dt).to(dtype)

        # Pass 2: Differentiable replay — gradient flows through dt_i → Bézier params
        z_replay = patchifier.patchify(z_init.clone()).float()
        for step in range(args.student_steps):
            dt = student_sigmas[step + 1] - student_sigmas[step]  # carries gradient!
            z_replay = z_replay + cached_velocities[step] * dt.float()

        # Loss: MSE between student and teacher final states
        loss = torch.nn.functional.mse_loss(z_replay, z_teacher.float())

        # NaN/Inf detection — abort before corrupting weights
        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write(f"  WARN iter {iteration}: loss is {loss.item()}, skipping update")
            nan_count = nan_count + 1 if 'nan_count' in dir() else 1
            if nan_count >= 5:
                tqdm.write("  FATAL: 5 consecutive NaN losses — aborting training")
                break
            continue
        nan_count = 0

        optimizer.zero_grad()
        loss.backward()

        # Check for gradient explosion
        grad_norm = bezier.theta.grad.norm().item() if bezier.theta.grad is not None else 0
        if grad_norm > 100:
            tqdm.write(f"  WARN iter {iteration}: grad_norm={grad_norm:.1f}, clipping hard")

        torch.nn.utils.clip_grad_norm_(bezier.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in bezier.state_dict().items()}

        # ── W&B logging (every iteration for loss, every 10 for schedule) ──
        if wandb_run:
            log_dict = {
                "loss": loss_val,
                "best_loss": best_loss,
                "grad_norm": grad_norm,
            }
            # Log full schedule as individual sigma values + table every 10 iters
            if iteration % 10 == 0 or iteration == args.num_iterations - 1:
                with torch.no_grad():
                    sched = bezier.get_sigma_schedule(args.student_steps)
                    sched_8 = bezier.get_sigma_schedule(8)
                for i, s in enumerate(sched.tolist()):
                    log_dict[f"sigma_{args.student_steps}step/{i}"] = s
                for i, s in enumerate(sched_8.tolist()):
                    log_dict[f"sigma_8step/{i}"] = s
                # Log control points
                cp = bezier.control_points.detach().cpu().tolist()
                log_dict["control_points/min_gap"] = min(cp[i+1] - cp[i] for i in range(len(cp)-1))
                log_dict["control_points/max_gap"] = max(cp[i+1] - cp[i] for i in range(len(cp)-1))
                # Schedule visualization as W&B line plot
                try:
                    import wandb
                    sched_fine = bezier.get_sigma_schedule(30).detach().cpu()
                    table = wandb.Table(
                        data=[[i/30, s.item()] for i, s in enumerate(sched_fine)],
                        columns=["step_fraction", "sigma"],
                    )
                    log_dict["schedule_curve"] = wandb.plot.line(
                        table, "step_fraction", "sigma", title="Learned σ(s) Schedule",
                    )
                except Exception:
                    pass
            wandb_run.log(log_dict, step=iteration)

        if iteration % 20 == 0 or iteration == args.num_iterations - 1:
            with torch.no_grad():
                sched = bezier.get_sigma_schedule(args.student_steps)
            tqdm.write(
                f"  iter {iteration:4d} | loss={loss_val:.6f} | best={best_loss:.6f} | "
                f"grad={grad_norm:.4f} | σ={[f'{s:.3f}' for s in sched.tolist()]}"
            )

    elapsed = time.time() - t_train_start
    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed/args.num_iterations:.2f}s/iter)")

    # ── Save best schedule ──
    if best_state is not None:
        bezier.load_state_dict(best_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bezier.save(output_path)

    with torch.no_grad():
        final_sched = bezier.get_sigma_schedule(args.student_steps)
        final_sched_8 = bezier.get_sigma_schedule(8)
    print(f"\n  Saved to: {output_path}")
    print(f"  Config:   {output_path.with_suffix('.json')}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  {args.student_steps}-step σ: {[f'{s:.4f}' for s in final_sched.tolist()]}")
    print(f"  8-step σ:  {[f'{s:.4f}' for s in final_sched_8.tolist()]}")

    # Monotonicity check
    diffs = final_sched[1:] - final_sched[:-1]
    if (diffs > 0).any():
        print("  WARNING: Schedule is NOT monotonically decreasing!")
    else:
        print("  Monotonicity check: PASSED")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
