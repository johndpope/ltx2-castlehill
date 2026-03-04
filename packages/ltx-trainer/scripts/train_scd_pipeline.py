#!/usr/bin/env python3
# ruff: noqa: T201
"""Unified SCD pipeline training: LoRA + BézierFlow in one shot.

Chains two training phases:
  Phase 1: SCD LoRA (token_concat + per_frame_decoder + Muon)
  Phase 2: BézierFlow schedule optimization (15 min, uses Phase 1 checkpoint)

Usage:
    python scripts/train_scd_pipeline.py

    # Or with custom settings:
    python scripts/train_scd_pipeline.py \
        --scd-steps 1000 \
        --bezier-steps 4 \
        --bezier-iters 200 \
        --skip-scd           # Skip Phase 1, use existing checkpoint
        --skip-bezierflow    # Skip Phase 2

After training, inference:
    python scripts/scd_inference.py \
        --distilled --decoder-combine token_concat \
        --lora-path /media/2TB/omnitransfer/output/scd_token_concat/checkpoints/lora_weights_step_01000.safetensors \
        --bezier-schedule /media/2TB/omnitransfer/output/scd_token_concat/bezierflow/schedule.pt \
        --num-inference-steps 4 \
        --cached-embedding /media/2TB/omnitransfer/data/ditto_subset/conditions_final/000000.pt \
        --num-seconds 5 --output /media/2TB/omnitransfer/inference/scd_realtime.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest SCD LoRA checkpoint in the output directory."""
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))
    return checkpoints[-1] if checkpoints else None


def run_phase(name: str, cmd: list[str]) -> int:
    """Run a training phase with live output."""
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SCD pipeline: LoRA + BézierFlow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Phase control
    parser.add_argument("--skip-scd", action="store_true", help="Skip SCD LoRA training (use existing checkpoint)")
    parser.add_argument("--skip-bezierflow", action="store_true", help="Skip BézierFlow schedule training")

    # SCD LoRA params
    parser.add_argument("--scd-config", default="configs/ltx2_scd_token_concat.yaml", help="SCD training config")
    parser.add_argument("--scd-steps", type=int, default=None, help="Override SCD training steps")

    # BézierFlow params
    parser.add_argument("--bezier-steps", type=int, default=4, help="Target inference steps for BézierFlow")
    parser.add_argument("--bezier-iters", type=int, default=200, help="BézierFlow training iterations")
    parser.add_argument("--bezier-teacher-steps", type=int, default=30, help="Teacher steps for BézierFlow")

    # Shared
    parser.add_argument("--output-dir", default="/media/2TB/omnitransfer/output/scd_token_concat", help="Output root")
    parser.add_argument("--data-root", default="/media/2TB/omnitransfer/data/ditto_subset", help="Training data")
    parser.add_argument("--quantization", default="int8-quanto", choices=["fp8-quanto", "int8-quanto", "none"])

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: SCD LoRA Training
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_scd:
        cmd = [sys.executable, "scripts/train.py", args.scd_config]
        rc = run_phase("Phase 1: SCD LoRA Training (token_concat + per_frame_decoder + Muon)", cmd)
        if rc != 0:
            sys.exit(rc)
    else:
        print("\n  Skipping Phase 1 (SCD LoRA) — using existing checkpoint")

    # Find the latest checkpoint
    lora_path = find_latest_checkpoint(output_dir)
    if lora_path is None:
        print(f"\n  ERROR: No checkpoint found in {output_dir}/checkpoints/")
        print("  Run without --skip-scd to train first.")
        sys.exit(1)
    print(f"\n  Using checkpoint: {lora_path}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: BézierFlow Schedule Optimization
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_bezierflow:
        bezier_output = output_dir / "bezierflow" / "schedule.pt"
        cmd = [
            sys.executable, "scripts/train_bezierflow.py",
            "--lora-path", str(lora_path),
            "--data-root", args.data_root,
            "--student-steps", str(args.bezier_steps),
            "--teacher-steps", str(args.bezier_teacher_steps),
            "--num-iterations", str(args.bezier_iters),
            "--decoder-combine", "token_concat",
            "--quantization", args.quantization,
            "--output", str(bezier_output),
            "--wandb-project", "scd-pipeline",
        ]
        rc = run_phase(f"Phase 2: BézierFlow ({args.bezier_steps}-step schedule, {args.bezier_iters} iters)", cmd)
        if rc != 0:
            print("  WARNING: BézierFlow training failed. You can still use distilled 8-step schedule.")
    else:
        bezier_output = output_dir / "bezierflow" / "schedule.pt"
        print("\n  Skipping Phase 2 (BézierFlow)")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Pipeline Complete ({elapsed / 60:.1f} min)")
    print(f"{'=' * 65}")
    print(f"  SCD LoRA:        {lora_path}")
    if bezier_output.exists():
        print(f"  BézierFlow:      {bezier_output}")
    print()
    print("  Inference command:")
    print(f"    python scripts/scd_inference.py \\")
    print(f"      --distilled --decoder-combine token_concat \\")
    print(f"      --lora-path {lora_path} \\")
    if bezier_output.exists():
        print(f"      --bezier-schedule {bezier_output} \\")
        print(f"      --num-inference-steps {args.bezier_steps} \\")
    print(f"      --cached-embedding {args.data_root}/conditions_final/000000.pt \\")
    print(f"      --num-seconds 5 --output /media/2TB/omnitransfer/inference/scd_pipeline.mp4")
    print()

    # Real-time estimate
    steps = args.bezier_steps if bezier_output.exists() else 8
    est_ms = 7 + (steps * 60) + 20  # encoder + decoder + overhead
    est_fps = 1000 / est_ms * 8  # latent fps * 8 pixel frames per latent
    print(f"  Estimated performance ({steps}-step):")
    print(f"    ~{est_ms}ms/latent frame → {1000/est_ms:.1f} latent fps → ~{est_fps:.0f} pixel fps")
    print(f"    {'REAL-TIME' if est_fps >= 24 else 'NOT real-time'} (target: 24 fps)")
    print()


if __name__ == "__main__":
    main()
