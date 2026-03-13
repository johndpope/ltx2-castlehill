#!/usr/bin/env python3
# ruff: noqa: T201
"""Train BézierFlow sigma schedule for the BASE LTX-2 transformer (no SCD).

Optimizes the 8-step ODE sigma schedule so the teacher produces better
denoised outputs x̂₀ — these become distillation targets for VFM.

Unlike train_bezierflow.py (SCD decoder), this operates on the full 48-layer
transformer with the standard Modality interface.

Two-pass gradient approach:
  1. Forward: Run full denoising with detached sigmas, cache velocity tensors.
  2. Replay: z_student = z_init + Σ v_i·dt_i where dt_i carries gradient.
  Loss = MSE(z_student_final, z_teacher_final)  where teacher uses 30 steps.

Usage:
    cd packages/ltx-trainer
    uv run python scripts/train_bezierflow_base.py \
        --data-root /media/2TB/omnitransfer/data/ditto_500sample \
        --student-steps 8 \
        --output /media/2TB/omnitransfer/output/bezierflow_base/schedule.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape


def load_conditions(conditions_dir: Path, max_samples: int, device: str, dtype: torch.dtype):
    """Load precomputed text embeddings."""
    files = sorted(conditions_dir.glob("*.pt"))[:max_samples]
    samples = []
    for f in files:
        cond = torch.load(f, map_location="cpu", weights_only=True)
        embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
        mask = cond.get("prompt_attention_mask", None)
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
        description="Train BézierFlow schedule for base LTX-2 (VFM teacher)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint",
                        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-root", type=str)
    data_group.add_argument("--cached-embedding", type=str)

    parser.add_argument("--scheduler-type", default="bezier", choices=["bezier", "bspline"])
    parser.add_argument("--bspline-order", type=int, default=4)

    parser.add_argument("--teacher-steps", type=int, default=30)
    parser.add_argument("--student-steps", type=int, default=8,
                        help="Target step count (8 for trajectory pre-computation)")
    parser.add_argument("--n-control-points", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--quantize", default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--output",
                        default="/media/2TB/omnitransfer/output/bezierflow_base/schedule.pt")
    parser.add_argument("--wandb-project", default="bezierflow-vfm")

    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    latent_frames = (args.num_frames - 1) // 8 + 1
    total_tokens = latent_frames * latent_h * latent_w

    sched_name = "BézierFlow" if args.scheduler_type == "bezier" else f"BSplineFlow (order={args.bspline_order})"
    print()
    print("=" * 65)
    print(f"  {sched_name} for Base LTX-2 (VFM Teacher)")
    print("=" * 65)
    print(f"  Teacher steps:  {args.teacher_steps}")
    print(f"  Student steps:  {args.student_steps}")
    print(f"  Resolution:     {args.width}x{args.height} → latent {latent_w}x{latent_h}")
    print(f"  Frames:         {args.num_frames} pixel / {latent_frames} latent")
    print(f"  Tokens:         {total_tokens}")
    print(f"  Iterations:     {args.num_iterations}")
    print("=" * 65)

    # ── W&B ──
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.scheduler_type}_base_s{args.student_steps}",
            )
        except Exception as e:
            print(f"  W&B init failed: {e}")

    # ── Load conditioning ──
    print("\n[1/3] Loading conditioning data...")
    if args.cached_embedding:
        emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
        embeds = emb.get("video_prompt_embeds", emb.get("prompt_embeds"))
        mask = emb.get("prompt_attention_mask", None)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        samples = [{"prompt_embeds": embeds.to(device, dtype),
                     "prompt_mask": mask.to(device) if mask is not None else None}]
    else:
        conditions_dir = Path(args.data_root) / "conditions_final"
        if not conditions_dir.exists():
            conditions_dir = Path(args.data_root) / "conditions"
        samples = load_conditions(conditions_dir, args.max_samples, device, dtype)
    print(f"  Loaded {len(samples)} samples")

    # ── Load transformer ──
    print("\n[2/3] Loading base transformer...")
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.checkpoint, device="cpu", dtype=dtype)

    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantize})...")
        transformer = quantize_model(transformer, args.quantize, device=device)

    transformer = transformer.to(device)
    transformer.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    # ── Positions ──
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

    # Teacher sigma schedule (30 steps — high quality reference)
    dummy_latent = torch.zeros(1, latent_channels, latent_frames, latent_h, latent_w)
    teacher_sigmas = LTX2Scheduler().execute(
        steps=args.teacher_steps, latent=dummy_latent
    ).to(device)
    print(f"  Teacher σ range: [{teacher_sigmas[0]:.4f} → {teacher_sigmas[-1]:.4f}]")

    # ── Train BézierFlow ──
    print(f"\n[3/3] Training {sched_name} ({args.num_iterations} iterations)...")

    if args.scheduler_type == "bezier":
        from ltx_trainer.bezierflow import BezierScheduler
        sched = BezierScheduler(n_control_points=args.n_control_points).to(device)
    else:
        from ltx_trainer.bsplineflow import BSplineScheduler
        sched = BSplineScheduler(
            n_coefficients=args.n_control_points,
            order=args.bspline_order,
        ).to(device)

    optimizer = torch.optim.RMSprop(sched.parameters(), lr=args.lr, momentum=0.9)

    with torch.no_grad():
        init_sched = sched.get_sigma_schedule(args.student_steps)
        print(f"  Initial σ: {[f'{s:.4f}' for s in init_sched.tolist()]}")

    best_loss = float("inf")
    best_state = None
    nan_count = 0

    t0 = time.time()

    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        sample = random.choice(samples)
        prompt_embeds = sample["prompt_embeds"]
        prompt_mask = sample["prompt_mask"]

        # Sample noise
        z_init = torch.randn(1, total_tokens, latent_channels, device=device, dtype=dtype)

        # ── TEACHER: 30-step ODE (no grad) ──
        with torch.inference_mode():
            z_teacher = z_init.clone()
            for step in range(args.teacher_steps):
                sigma = teacher_sigmas[step]
                sigma_next = teacher_sigmas[step + 1]
                dt = sigma_next - sigma

                timesteps = torch.full((1, total_tokens), sigma.item(), device=device, dtype=dtype)
                video_mod = Modality(
                    enabled=True, latent=z_teacher, timesteps=timesteps,
                    positions=positions, context=prompt_embeds, context_mask=prompt_mask,
                )
                v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                z_teacher = (z_teacher.float() + v_pred.float() * dt.float()).to(dtype)

            z_teacher = z_teacher.detach().clone()

        # ── STUDENT: N-step with learned sigmas ──
        student_sigmas = sched.get_sigma_schedule(args.student_steps)  # differentiable

        # Pass 1: Forward with detached sigmas → cache velocities
        # NOTE: Use no_grad (not inference_mode) so cloned tensors can be used in autograd replay
        cached_velocities = []
        with torch.no_grad():
            z_fwd = z_init.clone()
            for step in range(args.student_steps):
                sigma_val = student_sigmas[step].detach().item()
                sigma_next_val = student_sigmas[step + 1].detach().item()

                timesteps = torch.full((1, total_tokens), sigma_val, device=device, dtype=dtype)
                video_mod = Modality(
                    enabled=True, latent=z_fwd, timesteps=timesteps,
                    positions=positions, context=prompt_embeds, context_mask=prompt_mask,
                )
                v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                cached_velocities.append(v_pred.detach().float().clone())

                dt = sigma_next_val - sigma_val
                z_fwd = (z_fwd.float() + v_pred.float() * dt).to(dtype)

        # Pass 2: Differentiable replay — gradient → Bézier params via dt
        z_replay = z_init.float()
        for step in range(args.student_steps):
            dt = student_sigmas[step + 1] - student_sigmas[step]  # gradient flows here
            z_replay = z_replay + cached_velocities[step] * dt.float()

        loss = torch.nn.functional.mse_loss(z_replay, z_teacher.float())

        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write(f"  WARN iter {iteration}: loss={loss.item()}, skipping")
            nan_count += 1
            if nan_count >= 5:
                tqdm.write("  FATAL: 5 consecutive NaN — aborting")
                break
            continue
        nan_count = 0

        optimizer.zero_grad()
        loss.backward()

        grad_norm = sched.theta.grad.norm().item() if sched.theta.grad is not None else 0
        torch.nn.utils.clip_grad_norm_(sched.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in sched.state_dict().items()}

        # W&B logging
        if wandb_run:
            log_dict = {"loss": loss_val, "best_loss": best_loss, "grad_norm": grad_norm}
            if iteration % 10 == 0 or iteration == args.num_iterations - 1:
                with torch.no_grad():
                    s_sched = sched.get_sigma_schedule(args.student_steps)
                for i, s in enumerate(s_sched.tolist()):
                    log_dict[f"sigma/{i}"] = s
                cp = sched.control_points.detach().cpu().tolist()
                log_dict["control_points/min_gap"] = min(cp[i+1] - cp[i] for i in range(len(cp)-1))
                log_dict["control_points/max_gap"] = max(cp[i+1] - cp[i] for i in range(len(cp)-1))
            wandb_run.log(log_dict, step=iteration)

        if iteration % 20 == 0 or iteration == args.num_iterations - 1:
            with torch.no_grad():
                cur = sched.get_sigma_schedule(args.student_steps)
            tqdm.write(
                f"  iter {iteration:4d} | loss={loss_val:.6f} | best={best_loss:.6f} | "
                f"σ={[f'{s:.3f}' for s in cur.tolist()]}"
            )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/args.num_iterations:.2f}s/iter)")

    # ── Save ──
    if best_state is not None:
        sched.load_state_dict(best_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sched.save(output_path)

    with torch.no_grad():
        final = sched.get_sigma_schedule(args.student_steps)
    print(f"\n  Saved: {output_path}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  {args.student_steps}-step σ: {[f'{s:.4f}' for s in final.tolist()]}")

    # Compare with linear
    linear_sigmas = LTX2Scheduler().execute(steps=args.student_steps, latent=dummy_latent)
    print(f"  Linear σ:       {[f'{s:.4f}' for s in linear_sigmas.tolist()]}")

    diffs = final[1:] - final[:-1]
    if (diffs > 0).any():
        print("  WARNING: NOT monotonically decreasing!")
    else:
        print("  Monotonicity: PASSED")

    if wandb_run:
        wandb_run.finish()

    # Cleanup
    del transformer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
