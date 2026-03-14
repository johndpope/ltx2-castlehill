#!/usr/bin/env python3
# ruff: noqa: T201
"""SCD (Separable Causal Diffusion) video generation with trained LoRA.

Implements inference for two papers:

1. **SCD** (arXiv:2602.10095 — "Separable Causal Diffusion for Video Generation")
   - Splits the DiT into Encoder (causal, processes clean x_0) and Decoder (denoises x_sigma)
   - Encoder uses autoregressive KV-cache for O(1) per-frame cost (Section 4.1)
   - Features are shift-by-1 aligned: enc(frame_t) conditions dec(frame_{t+1}) (Section 3.4)
   - Core formulation (Section 3.2):
       Encoder: f_enc = Enc(x_0, sigma=0, causal_mask)  -- processes clean frames
       Decoder: v_hat = Dec(x_sigma, shift(f_enc), sigma) -- denoises with shifted features

2. **DDiT** (arXiv:2602.16968 — "DDiT: Dynamic Diffusion Transformer")
   - Dynamic spatial token merging for decoder speedup (Section 3.1)
   - Merge: z_coarse = MergeLayer(z, scale, H, W) reduces tokens by scale^2
   - Unmerge: z_fine = UnmergeLayer(z_coarse, H, W) restores full resolution
   - Dynamic scheduling via 3rd-order finite differences (Section 3.2, Algorithm 1):
       Delta^3_z = 3rd-order finite diff of denoising trajectory
       if rho-percentile(std(Delta^3_z patches)) < tau -> use coarse scale
   - Trained merge/unmerge adapters preserve quality (Section 3.3)

Generates video autoregressively using the SCD architecture:
- Encoder: processes previously generated (clean) frames with causal mask + KV-cache
- Decoder: denoises new frames conditioned on shifted encoder features

Supports chunked generation for long videos (30+ seconds) by chaining
overlapping chunks, each matching the training window (4 latent frames = 25 pixels).

Two modes for text conditioning:
1. --cached-embedding: Use a precomputed .pt file (skips 28GB text encoder load)
2. --prompt: Encode text live (loads Gemma → encode → unload before inference)

Usage:
    # Quick test with cached embedding (5 seconds)
    python scripts/scd_inference.py \
        --cached-embedding /media/2TB/omnitransfer/data/ditto_subset/conditions_final/000000.pt \
        --num-seconds 5 \
        --output /media/2TB/omnitransfer/inference/scd_test_5s.mp4

    # 30-second clip with live text encoding
    python scripts/scd_inference.py \
        --prompt "A serene mountain landscape with flowing rivers" \
        --num-seconds 30 \
        --output /media/2TB/omnitransfer/inference/scd_30s.mp4

    # Fast 8-step distilled model (~3.75x faster than dev)
    python scripts/scd_inference.py \
        --distilled \
        --cached-embedding /media/2TB/omnitransfer/data/ditto_subset/conditions_final/000000.pt \
        --num-seconds 10 \
        --output /media/2TB/omnitransfer/inference/scd_distilled_10s.mp4
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from tqdm import tqdm

from ltx_core.components.schedulers import LTX2Scheduler

# Set CUDA arch BEFORE importing quanto (needed for fp8-quanto JIT compilation on RTX 5090)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

# Distilled model sigma schedule (from ltx-pipelines constants.py)
# These are the exact sigma values used by the distilled 8-step model.
# The schedule is heavily front-loaded: steps 1-4 are tiny deltas (~0.00625),
# steps 5-8 are large jumps. This preserves the teacher's trajectory quality
# while requiring only 8 denoising steps instead of 30.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# TeaCache (CVPR 2025) — training-free decoder step caching
# ─────────────────────────────────────────────────────────────────────────────
# TeaCache exploits temporal redundancy in the denoising trajectory: consecutive
# steps at similar sigma values produce nearly identical decoder outputs. By
# measuring how much the decoder input changes (relative L1 of patchified latent),
# we cache the velocity and reuse it when the change is below a threshold.
# Expected ~1.5-2x decoder speedup (stacks with DDiT's 1.25x).
# Reference: "TeaCache: Timestep-Aware Cache for Diffusion Transformers" (CVPR 2025)


@dataclass
class TeaCacheState:
    """Per-frame state for TeaCache decoder step-skipping."""

    threshold: float
    coefficients: list[float] = field(default_factory=lambda: [0.0, 1.0])
    prev_signal: torch.Tensor | None = field(default=None, repr=False)
    cached_velocity: torch.Tensor | None = field(default=None, repr=False)
    accumulated_distance: float = 0.0
    hits: int = 0
    misses: int = 0

    def reset(self) -> None:
        """Reset per-frame state (call at the start of each new frame's denoising)."""
        self.prev_signal = None
        self.cached_velocity = None
        self.accumulated_distance = 0.0

    def should_compute(self, signal: torch.Tensor, force: bool = False) -> bool:
        """Decide whether to run the decoder or reuse cached velocity.

        Args:
            signal: Current step's patchified input [1, tpf, C].
            force: If True, always compute (used for first/last step).

        Returns:
            True if decoder should be run, False if cache can be reused.
        """
        if self.prev_signal is None:
            # First step of this frame — no history to compare against
            self.prev_signal = signal.detach()
            self.misses += 1
            return True

        # Relative L1 distance: captures both sigma change and content evolution
        rel_l1 = (signal - self.prev_signal).abs().mean() / (self.prev_signal.abs().mean() + 1e-8)
        d = rel_l1.item()

        # Polynomial rescaling: d_out = c0 + c1*d + c2*d^2 + ...
        # Default [0.0, 1.0] = identity (raw L1). Can be calibrated per-model.
        rescaled = sum(c * (d ** i) for i, c in enumerate(self.coefficients))
        self.accumulated_distance += rescaled
        self.prev_signal = signal.detach()

        if force:
            # Forced compute (first/last step) — reset accumulator
            self.accumulated_distance = 0.0
            self.misses += 1
            return True

        if self.accumulated_distance < self.threshold and self.cached_velocity is not None:
            self.hits += 1
            return False  # Cache hit — skip decoder
        else:
            self.accumulated_distance = 0.0
            self.misses += 1
            return True  # Cache miss — run decoder


# ─────────────────────────────────────────────────────────────────────────────
# LoRA utilities (copied from inference.py for self-containment)
# ─────────────────────────────────────────────────────────────────────────────


def extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Extract target module names from LoRA checkpoint keys."""
    target_modules = set()
    pattern = re.compile(r"(.+)\.lora_[AB]\.")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            target_modules.add(match.group(1))
    return sorted(target_modules)


def normalize_scd_lora_keys(
    state_dict: dict[str, torch.Tensor],
    encoder_layers: int = 32,
) -> dict[str, torch.Tensor]:
    """Normalize SCD-trained LoRA keys to standard transformer_blocks format.

    SCD Paper (arXiv:2602.10095), Section 3.2: The encoder/decoder split creates
    aliased module paths. During SCD training, LoRA is applied to transformer_blocks
    which get aliased as encoder_blocks (blocks 0..N_enc-1) and decoder_blocks
    (blocks N_enc..47). We normalize back to the canonical path so LoRA can be
    applied to the raw transformer BEFORE SCD wrapping.

    SCD wrapping creates three module paths for the same blocks:
    - base_model.transformer_blocks.X  (canonical, covers all 48 blocks)
    - encoder_blocks.X                 (blocks 0..encoder_layers-1)
    - decoder_blocks.X                 (blocks encoder_layers..47)

    We normalize everything to transformer_blocks.X so the LoRA can be
    applied to the raw (unwrapped) transformer.
    """
    normalized: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        # Skip duplicate encoder_blocks / decoder_blocks paths
        if key.startswith("encoder_blocks.") or key.startswith("decoder_blocks."):
            continue

        # base_model.transformer_blocks.X → transformer_blocks.X
        if key.startswith("base_model."):
            key = key[len("base_model."):]

        normalized[key] = value

    return normalized


def load_lora_weights(
    model: torch.nn.Module,
    lora_path: str | Path,
    encoder_layers: int = 32,
) -> torch.nn.Module:
    """Load LoRA weights into a transformer model. Returns the PeftModel wrapper.

    Handles SCD-trained checkpoints where keys have encoder_blocks/decoder_blocks
    prefixes by normalizing to standard transformer_blocks format.
    """
    print(f"  Loading LoRA from {lora_path}...")
    state_dict = load_file(str(lora_path))

    # Remove ComfyUI prefix
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

    # Normalize SCD paths → standard transformer_blocks paths
    state_dict = normalize_scd_lora_keys(state_dict, encoder_layers=encoder_layers)

    target_modules = extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"No LoRA modules found in {lora_path}")

    lora_rank = None
    for key, value in state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
            break
    if lora_rank is None:
        raise ValueError("Could not detect LoRA rank")

    print(f"  {len(target_modules)} target modules, rank {lora_rank}")

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model = get_peft_model(model, config)
    set_peft_model_state_dict(model.get_base_model(), state_dict)
    print("  LoRA weights loaded")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCD Video Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--checkpoint",
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors",
        help="Base model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text-encoder-path",
        default="/media/2TB/ltx-models/gemma",
        help="Gemma text encoder directory",
    )
    parser.add_argument(
        "--lora-path",
        default="/media/2TB/omnitransfer/output/scd_token_concat/checkpoints/lora_weights_step_01000.safetensors",
        help="SCD LoRA checkpoint (must be trained with matching --decoder-combine mode)",
    )

    # Text conditioning (one required)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--prompt",
        type=str,
        help="Text prompt (requires loading text encoder)",
    )
    text_group.add_argument(
        "--cached-embedding",
        type=str,
        help="Path to cached conditions_final .pt file (skips text encoder)",
    )

    # Generation parameters
    # SCD Paper, Section 5.1: Training uses 25-step denoising with Euler ODE solver.
    # We default to 30 steps for slightly higher quality; can reduce for speed.
    parser.add_argument("--height", type=int, default=448, help="Video height (divisible by 32)")
    parser.add_argument("--width", type=int, default=768, help="Video width (divisible by 32)")
    parser.add_argument("--num-seconds", type=float, default=5.0, help="Video duration in seconds")
    parser.add_argument("--fps", type=float, default=24.0, help="Frame rate (default: 24fps)")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Denoising steps per latent frame")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # SCD architecture parameters (should match training)
    # SCD Paper, Section 3.2: The DiT is split into N_enc encoder layers and N_dec decoder layers.
    # The encoder processes clean frames (sigma=0) with causal attention, producing features.
    # The decoder receives noisy frames + shifted encoder features and predicts velocity.
    # Adaptation: LTX-2 has 48 total transformer blocks; default split is 32 enc + 16 dec.
    # The paper uses 2/3 encoder + 1/3 decoder, matching our 32/16 default.
    parser.add_argument("--encoder-layers", type=int, default=32, help="Encoder layers for SCD split")
    # SCD Paper, Section 3.3: Encoder features are combined with decoder input via either
    # additive fusion ("add") or token concatenation ("token_concat"). Additive is simpler
    # and memory-efficient; token_concat preserves more information but doubles sequence length.
    parser.add_argument("--decoder-combine", default="token_concat", choices=["add", "token_concat"])
    parser.add_argument("--quantization", default="fp8-quanto", choices=["fp8-quanto", "int8-quanto", "none"])
    parser.add_argument("--compile", action="store_true", help="torch.compile decoder for ~1.5-2x speedup (warmup takes ~60s)")

    # CFG (Classifier-Free Guidance) — critical for output quality
    parser.add_argument(
        "--guidance-scale", type=float, default=4.0,
        help="Classifier-free guidance scale. 1.0 = no guidance, 4.0 = standard LTX-2 quality. "
             "Higher values produce sharper but potentially oversaturated output.",
    )

    # Diagnostic: bypass SCD and run full 48-block model
    parser.add_argument(
        "--no-scd",
        action="store_true",
        help="[DIAGNOSTIC] Bypass SCD encoder-decoder split. Run the full 48-block model "
             "on a single frame to verify base denoising works. No LoRA, no SCD wrapping.",
    )

    # Distilled model support (8-step, CFG=1.0)
    parser.add_argument(
        "--distilled",
        action="store_true",
        help="Use the distilled 8-step model (ltx-2-19b-distilled.safetensors). "
             "Overrides --num-inference-steps to 8 with a predefined non-uniform sigma schedule. "
             "~3.75x faster than the dev 30-step model.",
    )

    # BézierFlow — learned sigma schedule (ICLR 2026, arXiv:2512.13255)
    parser.add_argument(
        "--bezier-schedule",
        type=str,
        default=None,
        help="Learned Bézier sigma schedule (.pt). Overrides --num-inference-steps "
             "with the step count the schedule was trained for. Train with train_bezierflow.py.",
    )

    # DDiT (Dynamic Patch Scheduling) — reduces decoder tokens for ~2-4x speedup
    # DDiT Paper (arXiv:2602.16968), Section 3: Dynamic Diffusion Transformer reduces
    # computational cost by spatially merging tokens during the decoder's denoising pass.
    # Key insight (Section 3.1): During mid-denoising steps, the model primarily refines
    # coarse structure, not fine details, so a reduced-resolution pass suffices.
    # The merge/unmerge adapters are lightweight learned projections (Section 3.3).
    parser.add_argument(
        "--ddit-adapter",
        type=str,
        default=None,
        # DDiT Paper, Section 3.3: The adapter consists of MergeLayer (spatial pooling +
        # learned projection) and UnmergeLayer (learned upsampling). These are trained
        # while the base DiT is frozen, requiring only ~0.1% additional parameters.
        help="DDiT adapter checkpoint (.safetensors). Enables dynamic patch merging for decoder speedup.",
    )
    parser.add_argument(
        "--ddit-scale",
        type=int,
        default=2,
        choices=[2, 4],
        # DDiT Paper, Section 3.1: scale=s merges s*s spatial tokens into 1, reducing
        # the sequence length by s^2. Since attention is O(N^2), scale=2 gives ~4x fewer
        # tokens and ~16x less attention compute; scale=4 gives ~16x fewer tokens.
        help="DDiT spatial merge scale: 2 = merge 2x2 patches (4x fewer tokens), 4 = merge 4x4 (16x fewer)",
    )
    parser.add_argument(
        "--ddit-native-tail",
        type=int,
        default=3,
        # DDiT Paper, Section 3.2: The last few denoising steps refine fine details
        # and must run at native resolution. The paper finds 2-3 tail steps sufficient
        # to recover fine-grained features without noticeable quality loss.
        help="Number of final denoising steps at native resolution (fine detail refinement)",
    )
    parser.add_argument(
        "--ddit-native-head",
        type=int,
        default=2,
        # DDiT Paper, Section 3.2: The first few steps establish coarse structure from
        # pure noise; running these at native resolution helps set correct global layout.
        # Only relevant for the fixed (non-dynamic) schedule.
        help="Number of initial denoising steps at native resolution (structure establishment). "
             "Only used with --ddit-fixed-schedule.",
    )
    parser.add_argument(
        "--ddit-fixed-schedule",
        action="store_true",
        # DDiT Paper, Section 3.2, Algorithm 1: The dynamic scheduler uses 3rd-order
        # finite differences of the denoising trajectory to decide per-step whether to
        # use coarse or fine resolution:
        #   Delta^3_z_t = z_t - 3*z_{t-1} + 3*z_{t-2} - z_{t-3}
        #   For each spatial patch: compute std(Delta^3_z_patch)
        #   If rho-percentile of patch stds < threshold tau -> use coarse scale
        # The fixed schedule bypasses this analysis with a simple head/tail split.
        help="Use fixed head/tail schedule instead of dynamic DDiTPatchScheduler. "
             "Default is dynamic (per-step scale selection via 3rd-order trajectory analysis).",
    )

    # TeaCache — training-free decoder step caching (CVPR 2025)
    # Caches decoder velocity and reuses it when consecutive denoising steps
    # produce similar outputs. Stacks with DDiT (TeaCache skips steps, DDiT
    # reduces tokens per step). Expected ~1.5-2x decoder speedup.
    parser.add_argument(
        "--teacache-thresh",
        type=float,
        default=None,
        help="Enable TeaCache step-skipping with given threshold. "
             "Lower = more accurate/slower, higher = faster/lower quality. "
             "Suggested: 0.05 (conservative, ~30%% hits), 0.10 (balanced, ~50%% hits), "
             "0.15 (aggressive, ~65%% hits). Default: None (disabled).",
    )

    # Dual-GPU split mode
    # SCD Paper, Section 4.2: The encoder and decoder have asymmetric compute profiles.
    # The encoder processes one clean frame per step (lightweight with KV-cache), while
    # the decoder runs the full denoising loop (num_inference_steps forward passes).
    # Adaptation: We exploit this asymmetry by placing encoder on one GPU and decoder
    # on another, enabling bf16 (no quantization) for higher quality.
    parser.add_argument(
        "--split-gpus",
        action="store_true",
        help="Split encoder→cuda:1 / decoder→cuda:0 in bf16 (no quantization). "
             "Makes decoder compute-bound so DDiT gives full 2-3x speedup.",
    )

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output video path (.mp4)")

    args = parser.parse_args()

    if args.split_gpus:
        if args.quantization != "none":
            print(f"  Note: --split-gpus forces --quantization none (was {args.quantization})")
            args.quantization = "none"

    # ── Handle distilled mode ──
    if args.distilled:
        distilled_path = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors"
        if args.checkpoint == "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors":
            # Auto-switch to distilled checkpoint (only when using default dev path)
            if Path(distilled_path).exists():
                args.checkpoint = distilled_path
                print(f"  Distilled mode: using {Path(distilled_path).name}")
            else:
                print(f"  WARNING: Distilled model not found at {distilled_path}")
                print(f"  Using dev model with distilled sigma schedule (quality may differ)")
        # BézierFlow overrides the distilled step count — user controls steps via CLI
        if not args.bezier_schedule:
            args.num_inference_steps = len(DISTILLED_SIGMA_VALUES) - 1  # 8 steps
        print(f"  Distilled mode: {args.num_inference_steps} steps with {'BézierFlow' if args.bezier_schedule else 'predefined'} sigma schedule")

    # ── Handle learned sigma schedule (BézierFlow or BSplineFlow) ──
    BEZIER_SIGMA_VALUES = None
    if args.bezier_schedule:
        sched_path = Path(args.bezier_schedule)
        json_path = sched_path.with_suffix(".json")
        sched_type = "bezier"
        if json_path.exists():
            import json as _json
            _cfg = _json.loads(json_path.read_text())
            sched_type = _cfg.get("type", "bezier")

        if sched_type == "bspline":
            from ltx_trainer.bsplineflow import BSplineScheduler
            _sched = BSplineScheduler.load(args.bezier_schedule, device="cpu")
            label = "BSplineFlow"
        else:
            from ltx_trainer.bezierflow import BezierScheduler
            _sched = BezierScheduler.load(args.bezier_schedule, device="cpu")
            label = "BézierFlow"

        BEZIER_SIGMA_VALUES = _sched.get_sigma_schedule(args.num_inference_steps).tolist()
        print(f"  {label}: {args.num_inference_steps} steps, σ={[f'{s:.4f}' for s in BEZIER_SIGMA_VALUES]}")
        del _sched

    # ── Validate dimensions ──
    assert args.height % 32 == 0, f"Height {args.height} must be divisible by 32"
    assert args.width % 32 == 0, f"Width {args.width} must be divisible by 32"

    latent_h = args.height // 32
    latent_w = args.width // 32
    tokens_per_frame = latent_h * latent_w

    # ── Calculate frame counts ──
    total_pixel_frames = int(args.num_seconds * args.fps)
    # Round to valid count: (F - 1) % 8 == 0
    total_pixel_frames = max(25, ((total_pixel_frames - 1) // 8) * 8 + 1)
    total_latent_frames = (total_pixel_frames - 1) // 8 + 1

    # SCD Paper (arXiv:2602.10095), Section 4.2 — Chunked Autoregressive Generation:
    # For long videos, we divide generation into overlapping chunks. Each chunk has
    # CHUNK_LATENT frames (matching the training window size). Between chunks, 1 frame
    # overlaps: the last frame of chunk_k is re-encoded as context for chunk_{k+1}.
    # This yields NEW_PER_CHUNK = 3 genuinely new frames per chunk after the first.
    # Chunk parameters: 4 latent frames per chunk (= 25 pixel frames)
    # With 1-frame overlap between chunks for temporal continuity
    CHUNK_LATENT = 4
    NEW_PER_CHUNK = CHUNK_LATENT - 1  # 3 new latent frames per chunk after the first

    if total_latent_frames <= CHUNK_LATENT:
        num_chunks = 1
    else:
        num_chunks = 1 + -(-((total_latent_frames - CHUNK_LATENT)) // NEW_PER_CHUNK)

    actual_latent = CHUNK_LATENT + (num_chunks - 1) * NEW_PER_CHUNK
    actual_pixel = (actual_latent - 1) * 8 + 1
    actual_duration = actual_pixel / args.fps

    use_cfg = args.guidance_scale > 1.0

    print()
    print("=" * 65)
    print("  SCD Video Generation")
    print("=" * 65)
    if args.prompt:
        print(f"  Prompt:      {args.prompt[:60]}{'...' if args.prompt and len(args.prompt) > 60 else ''}")
    else:
        print(f"  Embedding:   {args.cached_embedding}")
    print(f"  Resolution:  {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"  Duration:    {actual_duration:.1f}s ({actual_pixel} frames @ {args.fps} fps)")
    print(f"  Latent:      {actual_latent} frames in {num_chunks} chunk(s)")
    print(f"  Tokens/frame:{tokens_per_frame}")
    if BEZIER_SIGMA_VALUES is not None:
        print(f"  Steps:       {args.num_inference_steps} (BézierFlow learned schedule)")
        print(f"  Sigma range: [{BEZIER_SIGMA_VALUES[0]:.4f} → {BEZIER_SIGMA_VALUES[-2]:.4f} → {BEZIER_SIGMA_VALUES[-1]:.4f}]")
    elif args.distilled:
        print(f"  Steps:       {args.num_inference_steps} (DISTILLED, non-uniform sigma schedule)")
    else:
        # Show the shifted sigma schedule for debugging
        dummy_latent = torch.empty(1, 1, CHUNK_LATENT, latent_h, latent_w)
        _preview_sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps, latent=dummy_latent)
        print(f"  Steps:       {args.num_inference_steps} per latent frame (LTX2Scheduler, shift for {CHUNK_LATENT}×{latent_h}×{latent_w}={CHUNK_LATENT*latent_h*latent_w} tokens)")
        print(f"  Sigma range: [{_preview_sigmas[0]:.4f} → {_preview_sigmas[-2]:.4f} → {_preview_sigmas[-1]:.4f}]")
    print(f"  Model:       {'DISTILLED' if args.distilled else 'DEV'} ({Path(args.checkpoint).name})")
    print(f"  LoRA:        {Path(args.lora_path).name if args.lora_path else 'None'}")
    print(f"  CFG:         {'enabled' if use_cfg else 'disabled'} (guidance_scale={args.guidance_scale})")
    print(f"  Quantization:{args.quantization}")
    print(f"  Compile:     {'yes (first chunk includes JIT warmup)' if args.compile else 'no'}")
    if args.split_gpus:
        print(f"  Split GPUs:  encoder→cuda:0 (32GB), decoder→cuda:1 (24GB), bf16")
    if args.ddit_adapter:
        if args.ddit_fixed_schedule:
            print(f"  DDiT:        scale={args.ddit_scale}x, FIXED head={args.ddit_native_head} tail={args.ddit_native_tail}")
        else:
            print(f"  DDiT:        scale={args.ddit_scale}x, DYNAMIC scheduler (τ=0.001, ρ=0.4)")
        print(f"  DDiT adapter:{Path(args.ddit_adapter).name}")
    else:
        print(f"  DDiT:        disabled (use --ddit-adapter to enable)")
    if args.teacache_thresh is not None:
        print(f"  TeaCache:    enabled (threshold={args.teacache_thresh})")
    else:
        print(f"  TeaCache:    disabled (use --teacache-thresh to enable)")
    print(f"  Output:      {args.output}")
    print("=" * 65)

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Get text embeddings
    # ══════════════════════════════════════════════════════════════════════
    if args.cached_embedding:
        print(f"\n[1/4] Loading cached embedding...")
        emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
        # Format: {video_prompt_embeds: [1024, 3840], prompt_attention_mask: [1024]}
        prompt_embeds = emb["video_prompt_embeds"].unsqueeze(0).to(torch.bfloat16)
        prompt_mask = emb["prompt_attention_mask"].unsqueeze(0)
        print(f"  Shape: {prompt_embeds.shape}")
    else:
        print(f"\n[1/4] Loading text encoder on cuda:1...")
        from ltx_trainer.model_loader import load_text_encoder

        text_encoder = load_text_encoder(
            args.checkpoint,
            args.text_encoder_path,
            device="cuda:1",
            dtype=torch.bfloat16,
        )
        text_encoder.eval()

        print(f"  Encoding: '{args.prompt[:80]}'")
        with torch.inference_mode():
            # text_encoder(str) → AVGemmaEncoderOutput(video_encoding, audio_encoding, attention_mask)
            video_embeds, _audio_embeds, attention_mask = text_encoder(args.prompt)
            prompt_embeds = video_embeds.to(torch.bfloat16)  # [1, 1024, 3840]
            prompt_mask = attention_mask  # [1, 1024]

        print(f"  Shape: {prompt_embeds.shape}")

        # Free text encoder before loading transformer
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("  Text encoder unloaded")

    # Move to cuda:0 (transformer device)
    prompt_embeds = prompt_embeds.to("cuda:0")
    prompt_mask = prompt_mask.to("cuda:0")

    # Create null (unconditional) embeddings for CFG
    # CFG formula: v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    # Null embedding = zeros with full attention mask (all masked out)
    # Create null (unconditional) embeddings for CFG
    if use_cfg:
        null_embeds = torch.zeros_like(prompt_embeds)
        null_mask = torch.zeros_like(prompt_mask)
    else:
        null_embeds = null_mask = None

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Load transformer → quantize → SCD wrap → LoRA
    # ══════════════════════════════════════════════════════════════════════
    # SCD Paper (arXiv:2602.10095), Section 3.2: The key insight is that a standard
    # DiT can be split post-training into an encoder (causal, processes clean x_0)
    # and decoder (bidirectional, denoises x_sigma). The split is along the layer
    # dimension: first N_enc layers become the encoder, remaining N_dec become decoder.
    # This requires NO architectural changes to the base model — just reorganization.
    print(f"\n[2/4] Loading transformer...")
    from ltx_trainer.model_loader import load_transformer

    transformer = load_transformer(args.checkpoint, device="cpu", dtype=torch.bfloat16)

    if args.quantization != "none":
        from ltx_trainer.quantization import quantize_model

        print(f"  Quantizing ({args.quantization})... (first run takes ~20 min for JIT compilation)")
        transformer = quantize_model(transformer, args.quantization, device="cuda:0")

    if not args.split_gpus:
        print("  Moving to cuda:0...")
        transformer = transformer.to("cuda:0")

    # Apply LoRA BEFORE SCD wrapping — LoRA keys use transformer_blocks.X paths
    # which get renamed to encoder_blocks/decoder_blocks by SCD wrapper.
    # SCD Paper, Section 5.1: LoRA (rank 64) is applied to the full DiT during SCD
    # training. The LoRA adapts both encoder and decoder layers jointly. At inference,
    # we load these weights into the base transformer, then wrap with SCD.
    if args.lora_path and not args.no_scd:
        # PEFT can apply LoRA structure on CPU; weight loading via set_peft_model_state_dict
        # also works on CPU. No need to move to GPU first.
        transformer = load_lora_weights(transformer, args.lora_path, encoder_layers=args.encoder_layers)
        # Unwrap PeftModel → get the modified LTXModel with LoRA layers in-place
        transformer = transformer.get_base_model()
    elif args.no_scd:
        print("  [NO-SCD MODE] Skipping LoRA — running raw base model")

    # SCD Paper, Section 3.2: LTXSCDModel wraps the base DiT by creating aliased views:
    #   encoder_blocks = transformer_blocks[0:encoder_layers]      (causal attention)
    #   decoder_blocks = transformer_blocks[encoder_layers:48]     (bidirectional attention)
    # The encoder receives clean frames (sigma=0) and uses causal masking so each frame
    # can only attend to itself and previous frames. The decoder receives noisy frames
    # and attends bidirectionally, but is conditioned on shifted encoder features.
    print(f"  Wrapping with SCD ({args.encoder_layers} encoder, {48 - args.encoder_layers} decoder layers)...")
    from ltx_core.model.transformer.scd_model import LTXSCDModel

    scd_model = LTXSCDModel(
        base_model=transformer,
        encoder_layers=args.encoder_layers,
        decoder_input_combine=args.decoder_combine,
    )

    scd_model.eval()

    # ── Split-GPU: distribute encoder→cuda:0, decoder→cuda:1 ──
    # SCD Paper, Section 4.2: The SCD architecture naturally enables model parallelism
    # because encoder and decoder are independent compute stages with a clear data
    # boundary (encoder features). The encoder processes one frame per autoregressive
    # step (lightweight), while the decoder runs num_inference_steps full forward passes
    # (compute-intensive). This asymmetry makes split-GPU practical even with unequal GPUs.
    #
    # Adaptation: LTX-2 has 48 transformer blocks at ~0.77 GB each in bf16.
    # Each block ≈ 0.77 GB in bf16. 32 encoder blocks = 24.7 GB (needs 32GB GPU).
    # 16 decoder blocks = 12.4 GB (fits on 24GB GPU).
    # cuda:0 = RTX 5090 (32GB) → encoder blocks + preprocessor
    # cuda:1 = RTX PRO 4000 (24GB) → decoder blocks + output projection
    decoder_preprocessor = None  # Only set in split-GPU mode
    if args.split_gpus:
        import copy

        print("  Distributing model across GPUs (bf16, no quantization)...")
        print("    cuda:0 (32GB): encoder blocks + preprocessor")
        print("    cuda:1 (24GB): decoder blocks + output projection")

        # Move encoder blocks to cuda:0 (32GB — fits 32 blocks × 0.77GB = 24.7GB)
        for block in scd_model.encoder_blocks:
            block.to("cuda:0")
        # Move ALL preprocessor nn.Module attributes to cuda:0 (encoder uses it natively)
        # MultiModalTransformerArgsPreprocessor is a plain class holding nn.Modules:
        # simple_preprocessor.{patchify_proj, adaln, caption_projection} + cross_scale_shift_adaln + cross_gate_adaln
        _mmprep = scd_model.base_model.video_args_preprocessor
        for attr_name, attr in vars(_mmprep).items():
            if isinstance(attr, torch.nn.Module):
                attr.to("cuda:0")
        # simple_preprocessor is itself a plain class — move its nn.Module children too
        for attr_name, attr in vars(_mmprep.simple_preprocessor).items():
            if isinstance(attr, torch.nn.Module):
                attr.to("cuda:0")
        scd_model.base_model.patchify_proj.to("cuda:0")
        enc_mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"    Encoder blocks (×{args.encoder_layers}) → cuda:0 ({enc_mem:.1f} GB)")

        # Move decoder blocks to cuda:1 (24GB — fits 16 blocks × 0.77GB = 12.4GB)
        for block in scd_model.decoder_blocks:
            block.to("cuda:1")
        # Output projection params → cuda:1 (decoder output pipeline)
        scd_model.base_model.scale_shift_table.data = scd_model.base_model.scale_shift_table.to("cuda:1")
        scd_model.base_model.norm_out.to("cuda:1")
        scd_model.base_model.proj_out.to("cuda:1")
        # Audio params if present
        for name in ["audio_proj_out", "audio_patchify_proj", "audio_caption_projection"]:
            if hasattr(scd_model.base_model, name):
                getattr(scd_model.base_model, name).to("cuda:1")
        dec_mem = torch.cuda.memory_allocated(1) / 1e9
        print(f"    Decoder blocks (×{48 - args.encoder_layers}) + output → cuda:1 ({dec_mem:.1f} GB)")

        # Adaptation: The LTX-2 transformer preprocessor (patchify_proj, adaln, caption_projection)
        # is shared between encoder and decoder in the base model. In split-GPU mode, we need a
        # copy on each device because these modules process input before the transformer blocks.
        # SCD Paper: The preprocessor converts raw latents + sigma + text embeddings into the
        # transformer's hidden representation — both encoder and decoder need this independently.
        # Clone preprocessor modules for decoder on cuda:1
        # The decoder also needs patchify_proj, adaln, caption_projection for its preprocessing
        orig_prep = scd_model.base_model.video_args_preprocessor.simple_preprocessor
        dec_patchify = copy.deepcopy(scd_model.base_model.patchify_proj).to("cuda:1")
        dec_adaln = copy.deepcopy(orig_prep.adaln).to("cuda:1")
        dec_caption_proj = copy.deepcopy(orig_prep.caption_projection).to("cuda:1")

        from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor
        decoder_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=dec_patchify,
            adaln=dec_adaln,
            caption_projection=dec_caption_proj,
            inner_dim=orig_prep.inner_dim,
            max_pos=orig_prep.max_pos,
            num_attention_heads=orig_prep.num_attention_heads,
            use_middle_indices_grid=orig_prep.use_middle_indices_grid,
            timestep_scale_multiplier=orig_prep.timestep_scale_multiplier,
            double_precision_rope=orig_prep.double_precision_rope,
            positional_embedding_theta=orig_prep.positional_embedding_theta,
            rope_type=orig_prep.rope_type,
        )
        clone_mb = sum(p.numel() * p.element_size() for p in [*dec_patchify.parameters(), *dec_adaln.parameters(), *dec_caption_proj.parameters()]) / 1e6
        print(f"    Decoder preprocessor cloned to cuda:1 ({clone_mb:.1f} MB)")

        # Prompt embeds: cuda:0 for encoder, cuda:1 for decoder
        prompt_embeds_dec = prompt_embeds.to("cuda:1")
        prompt_mask_dec = prompt_mask.to("cuda:1")
        prompt_embeds = prompt_embeds.to("cuda:0")
        prompt_mask = prompt_mask.to("cuda:0")
        if use_cfg:
            null_embeds_dec = null_embeds.to("cuda:1")
            null_mask_dec = null_mask.to("cuda:1")

        gc.collect()
        torch.cuda.empty_cache()
        print(f"    Final: cuda:0 = {torch.cuda.memory_allocated(0) / 1e9:.1f} GB, "
              f"cuda:1 = {torch.cuda.memory_allocated(1) / 1e9:.1f} GB")

    # ── torch.compile for decoder speedup ──
    if args.compile:
        print("  Compiling decoder blocks with torch.compile...")
        t_compile = time.time()
        # Make block.idx dynamic so dynamo doesn't recompile for each block
        torch._dynamo.config.allow_unspec_int_on_nn_module = True  # noqa: SLF001
        torch.set_float32_matmul_precision("high")  # Enable TF32 for matmuls

        # Compile individual blocks — handles graph breaks from dataclass ops.
        # Use mode="default" since reduce-overhead (CUDA graphs) conflicts with fp8-quanto.
        for i, block in enumerate(scd_model.decoder_blocks):
            scd_model.decoder_blocks[i] = torch.compile(block, mode="default", dynamic=True)
        for i, block in enumerate(scd_model.encoder_blocks):
            scd_model.encoder_blocks[i] = torch.compile(block, mode="default", dynamic=True)
        print(f"  Compile setup: {time.time() - t_compile:.1f}s (actual JIT on first forward)")

    # ── Load DDiT adapter (optional, for decoder speedup) ──
    # DDiT Paper (arXiv:2602.16968), Section 3.3: The DDiT adapter is a lightweight
    # module trained to merge/unmerge spatial tokens without quality loss. It consists of:
    #   - MergeLayer: groups s*s spatial patches → 1 token via learned linear projection
    #     Input: [B, H*W, C] → reshape to [B, H/s, s, W/s, s, C] → concat → project
    #     Output: [B, (H/s)*(W/s), D] where D is the transformer's inner_dim
    #   - UnmergeLayer: reverses the merge via learned upsampling projection
    #     Input: [B, (H/s)*(W/s), D] → project to [B, (H/s)*(W/s), C*s*s] → reshape
    #     Output: [B, H*W, C] at original resolution
    #   - Optional residual_block: skip connection for fine detail preservation
    # The adapter is trained with the base DiT frozen (Section 3.3), requiring ~0.1%
    # additional parameters relative to the full model.
    ddit_wrapper = None
    if args.ddit_adapter:
        import json as _json

        from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig

        adapter_path = Path(args.ddit_adapter)
        config_path = adapter_path.parent / "ddit_scd_config.json"
        if not config_path.exists():
            config_path = adapter_path.parent / "ddit_config.json"

        # Load config
        if config_path.exists():
            with open(config_path) as f:
                ddit_cfg = _json.load(f)
            scales = tuple(ddit_cfg.get("scales", [args.ddit_scale]))
        else:
            scales = (args.ddit_scale,)
            ddit_cfg = {}

        # DDiT Paper, Section 3.1: supported_scales includes native (1) plus trained
        # coarse scales. The dynamic scheduler (Algorithm 1) selects between these
        # per denoising step based on trajectory smoothness.
        ddit_config = DDiTConfig(
            enabled=True,
            supported_scales=(1, *scales),
            residual_weight=ddit_cfg.get("residual_weight", 0.0),
        )
        ddit_adapter = DDiTAdapter(
            inner_dim=scd_model.base_model.inner_dim,
            in_channels=128,
            config=ddit_config,
        )

        # DDiT Paper, Section 3.3: Adapter weights are trained on the SCD decoder only.
        # The encoder always runs at native resolution (it processes clean frames and
        # benefits from full spatial detail for KV-cache quality).
        # Load weights
        ddit_state = load_file(str(adapter_path))
        ddit_adapter.load_state_dict(ddit_state)
        _ddit_device = "cuda:1" if args.split_gpus else "cuda:0"
        ddit_adapter = ddit_adapter.to(device=_ddit_device, dtype=torch.bfloat16)
        ddit_adapter.eval()
        param_count = sum(p.numel() for p in ddit_adapter.parameters())
        print(f"  DDiT adapter loaded: {param_count / 1e6:.1f}M params, scales={scales}")

        # DDiT Paper, Section 3.3: When training the DDiT adapter, an optional lightweight
        # LoRA is applied to the decoder blocks to help them adapt to the coarser token
        # resolution. This "DDiT decoder LoRA" is separate from the SCD training LoRA
        # and stacks on top of it via forward hooks (additive residual: out + B(A(inp))).
        # Load DDiT decoder LoRA if present (hook-based, stacks on SCD LoRA)
        decoder_lora_path = adapter_path.parent / "ddit_scd_lora_final.safetensors"
        if decoder_lora_path.exists():
            from ltx_core.model.transformer.ddit import DDiTConfig as _  # noqa: F811

            ddit_lora_rank = ddit_cfg.get("ddit_lora_rank", 16)
            ddit_lora_targets = tuple(ddit_cfg.get("ddit_lora_targets", ["to_q", "to_k", "to_v", "to_out.0"]))
            num_dec_blocks = len(scd_model.decoder_blocks)
            inner_dim = scd_model.base_model.inner_dim

            # Build hook-based LoRA for decoder blocks
            class _DDiTDecoderLoRA(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._hooks: list = []
                    for bi in range(num_dec_blocks):
                        for tgt in ddit_lora_targets:
                            tag = f"b{bi}_{tgt.replace('.', '_')}"
                            setattr(self, f"{tag}_A", torch.nn.Linear(inner_dim, ddit_lora_rank, bias=False))
                            setattr(self, f"{tag}_B", torch.nn.Linear(ddit_lora_rank, inner_dim, bias=False))

                def install_hooks(self, decoder_blocks):
                    for h in self._hooks:
                        h.remove()
                    self._hooks = []
                    for bi, block in enumerate(decoder_blocks):
                        for tgt in ddit_lora_targets:
                            tag = f"b{bi}_{tgt.replace('.', '_')}"
                            lA = getattr(self, f"{tag}_A")
                            lB = getattr(self, f"{tag}_B")
                            parts = ["attn1"] + tgt.split(".")
                            mod = block
                            for p in parts:
                                mod = getattr(mod, p)

                            def _hook(a, b):
                                def fn(m, inp, out):
                                    return out + b(a(inp[0]))
                                return fn

                            self._hooks.append(mod.register_forward_hook(_hook(lA, lB)))

            ddit_lora = _DDiTDecoderLoRA()
            ddit_lora_state = load_file(str(decoder_lora_path))
            ddit_lora.load_state_dict(ddit_lora_state)
            ddit_lora = ddit_lora.to(device=_ddit_device, dtype=torch.bfloat16)
            ddit_lora.eval()
            ddit_lora.install_hooks(list(scd_model.decoder_blocks))
            lora_params = sum(p.numel() for p in ddit_lora.parameters())
            print(f"  DDiT decoder LoRA loaded: rank={ddit_lora_rank}, {lora_params / 1e6:.1f}M params")

        # Store wrapper info (lightweight — no class needed, just the adapter)
        ddit_wrapper = ddit_adapter

    mem_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"  GPU memory (cuda:0): {mem_gb:.1f} GB")
    if args.split_gpus:
        mem_gb_1 = torch.cuda.memory_allocated(1) / 1e9
        print(f"  GPU memory (cuda:1): {mem_gb_1:.1f} GB (decoder)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Load VAE decoder
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[3/4] Loading VAE decoder on cuda:1...")
    from ltx_trainer.model_loader import load_video_vae_decoder

    vae_decoder = load_video_vae_decoder(args.checkpoint, device="cuda:1", dtype=torch.bfloat16)
    vae_decoder.eval()
    mem_gb_vae = torch.cuda.memory_allocated(1) / 1e9
    print(f"  GPU memory (cuda:1): {mem_gb_vae:.1f} GB")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Generate video in chunks (autoregressive SCD with KV-cache)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[4/4] Generating {actual_latent} latent frames in {num_chunks} chunk(s)...")
    from dataclasses import replace as dc_replace

    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.scd_model import KVCache
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    # Adaptation: LTX-2 uses patch_size=1 (no spatial patch merging in the patchifier itself;
    # spatial structure is preserved as individual tokens). DDiT handles spatial merging separately
    # via its learned merge/unmerge layers when enabled.
    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    latent_channels = scd_model.base_model.patchify_proj.in_features  # 128
    # SCD Paper, Section 4.2: In split-GPU mode, encoder and decoder run on different GPUs.
    # The encoder (heavier, 32 blocks) goes on the larger GPU; the decoder (lighter, 16 blocks)
    # plus output projection goes on the smaller GPU.
    # In split-GPU mode: encoder on cuda:0 (32GB), decoder on cuda:1 (24GB)
    enc_device = torch.device("cuda:0")
    dec_device = torch.device("cuda:1") if args.split_gpus else torch.device("cuda:0")
    device = dec_device  # Default device for generation loop (decoder-centric)
    dtype = torch.bfloat16

    def get_positions(n_frames: int, target_device: torch.device | None = None) -> torch.Tensor:
        """Compute pixel-space position embeddings for n_frames latent frames.

        Adaptation: LTX-2 uses 3D RoPE (time, height, width) for positional encoding.
        Coordinates are in pixel space (not latent space), then converted via scale_factors.
        The temporal coordinate is normalized by FPS so the model learns time-invariant
        representations. Shape: [1, 3, n_frames * H * W, 2] (3 axes, each with sin/cos pair).
        """
        d = target_device or device
        coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=n_frames, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
            device=d,
        )
        px = get_pixel_coords(latent_coords=coords, scale_factors=scale_factors, causal_fix=True).to(dtype)
        px[:, 0, ...] = px[:, 0, ...] / args.fps
        return px

    def get_positions_for_frame(frame_idx: int, target_device: torch.device | None = None) -> torch.Tensor:
        """Get position embeddings for a single frame at a specific temporal index."""
        all_pos = get_positions(frame_idx + 1, target_device=target_device)
        start = frame_idx * tokens_per_frame
        end = start + tokens_per_frame
        return all_pos[:, :, start:end, :]

    def encode_single_frame(
        latent: torch.Tensor,
        frame_pos: int,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """Run a single frame through the SCD encoder with KV-cache.

        SCD Paper (arXiv:2602.10095), Section 4.1 — Autoregressive KV-Cache:
        The encoder processes frames autoregressively with causal masking. For each
        new frame, it only computes Q/K/V for that frame's tokens, but attends to
        all previously cached K/V from earlier frames. This gives O(T) total cost
        for T frames instead of O(T^2) if re-encoding the full sequence each time.

        SCD Paper, Section 3.2: The encoder always sees clean frames (sigma=0),
        meaning timesteps are all zeros. This is a key insight — the encoder acts
        as a feature extractor on ground-truth data, while the decoder handles the
        noisy denoising problem.

        In split-GPU mode, encoder runs on cuda:0 (enc_device).
        In single-GPU mode, everything runs on cuda:0.
        Returns encoder features on enc_device (always cuda:0).
        """
        latent = latent.to(enc_device)
        patchified = patchifier.patchify(latent)  # [1, tpf, C]
        # SCD Paper, Section 3.2: sigma=0 for encoder input — processes CLEAN frames.
        # The encoder never sees noise; it extracts features from previously generated
        # (or ground-truth) frames to condition the decoder's denoising process.
        modality = Modality(
            enabled=True,
            latent=patchified,
            timesteps=torch.zeros(1, tokens_per_frame, device=enc_device, dtype=dtype),
            positions=get_positions_for_frame(frame_pos, target_device=enc_device),
            context=prompt_embeds,  # prompt_embeds is on cuda:0 (enc_device)
            context_mask=prompt_mask,
        )
        # SCD Paper, Section 4.1: forward_encoder uses causal attention mask internally.
        # The KV-cache accumulates K/V pairs from all previously encoded frames,
        # enabling autoregressive generation without re-processing history.
        enc_out, _ = scd_model.forward_encoder(
            video=modality,
            audio=None,
            perturbations=None,
            kv_cache=kv_cache,
            tokens_per_frame=tokens_per_frame,
        )
        return enc_out.x  # [1, tpf, D] on enc_device (cuda:0)

    # ── DDiT decode helper (merged-resolution decoder pass) ──
    # DDiT Paper (arXiv:2602.16968), Section 3.1: This function implements the full
    # DDiT decoder pipeline for a single denoising step:
    #   1. merge(z, s, H, W) → z_coarse     [reduce tokens by s^2]
    #   2. project(z_coarse) → hidden        [DDiT's learned patchify]
    #   3. transformer_decoder(hidden) → out [run decoder blocks at coarse resolution]
    #   4. unmerge(out, H, W) → z_fine       [restore to native resolution]
    # This gives O(N^2 / s^4) attention cost vs O(N^2) native, where N = H*W.
    def ddit_decode_one_frame(
        noisy_patch: torch.Tensor,   # [1, tpf, 128] patchified latent
        enc_features: torch.Tensor,  # [1, tpf, D] encoder features
        sigma: float,
        positions: torch.Tensor,     # [1, 3, tpf, 2]
        scale: int,
    ) -> torch.Tensor:
        """Run decoder at merged resolution via DDiT adapter.

        DDiT Paper, Section 3.1: Spatial token merging for efficient denoising.
        Merges s*s spatial tokens → s^2 fewer tokens → O(N^2) attention
        reduces by s^4. Returns velocity at full resolution [1, tpf, 128].

        The key formula (DDiT Eq. 2):
          z_coarse = MergeLayer(z_fine, scale)        -- learned spatial pooling
          v_coarse = Decoder(z_coarse, features, sigma) -- run at reduced resolution
          v_fine = UnmergeLayer(v_coarse, scale)       -- learned spatial upsampling
        """
        from dataclasses import replace as dc_replace

        from ltx_core.guidance.perturbations import BatchedPerturbationConfig

        merge_layer = ddit_wrapper.merge_layers[str(scale)]
        nf, h, w = 1, latent_h, latent_w
        new_h, new_w = h // scale, w // scale
        new_tpf = new_h * new_w

        # DDiT Paper, Section 3.1, Step 1 — Spatial Token Merge:
        # Reshape [1, H*W, C] → [1, H/s, s, W/s, s, C] → concat along s dims → [1, H/s*W/s, C*s*s]
        # This groups s*s neighboring spatial patches into a single coarse token.
        # 1. Merge spatial tokens: [1, tpf, 128] → [1, new_tpf, 128*s*s]
        merged = merge_layer.merge(noisy_patch, nf, h, w)

        # DDiT Paper, Section 3.1, Step 2 — Learned Projection:
        # The merged coarse tokens have dimension C*s*s (e.g., 128*4=512 for scale=2).
        # A learned linear projection maps them to the transformer's inner_dim (4096).
        # patch_id is a learnable bias that helps the model distinguish merged vs native tokens.
        # 2. Project through DDiT's patchify: [1, new_tpf, 128*s*s] → [1, new_tpf, 4096]
        merged_proj = merge_layer.patchify_proj(merged) + merge_layer.patch_id

        # Adaptation: Encoder features must also be downsampled to match the coarse grid.
        # We use adaptive_avg_pool2d since encoder features are semantically smooth.
        # SCD Paper: Encoder features are shift-by-1 aligned, so these are features from
        # the PREVIOUS frame's encoding, not the current noisy frame.
        # 3. Pool encoder features: [1, tpf, D] → [1, new_tpf, D]
        D = enc_features.shape[-1]
        ef = enc_features.view(1, nf, h, w, D).permute(0, 1, 4, 2, 3)  # [1, 1, D, H, W]
        ef = ef.reshape(nf, D, h, w)
        ef = torch.nn.functional.adaptive_avg_pool2d(ef, (new_h, new_w))
        ef = ef.reshape(1, nf, D, new_h, new_w).permute(0, 1, 3, 4, 2)  # [1, 1, h, w, D]
        merged_enc = ef.reshape(1, new_tpf, D)

        # DDiT Paper, Section 3.1: Position embeddings must be adjusted for the coarser
        # spatial grid. RoPE frequencies are recomputed for the merged (H/s, W/s) grid
        # so the transformer's positional encoding remains consistent.
        # 4. Adjust positions for coarser grid
        merged_positions = ddit_wrapper.adjust_positions(positions, scale, nf, h, w)

        # 5. Create dummy modality for preprocessor (timestep/context processing)
        # In split-GPU mode, decoder is on dec_device (cuda:1), use appropriate prompt embeds
        dec_prompt = prompt_embeds_dec if args.split_gpus else prompt_embeds
        dec_prompt_mask = prompt_mask_dec if args.split_gpus else prompt_mask
        dummy_latent = torch.zeros(
            1, new_tpf, scd_model.base_model.patchify_proj.in_features,
            device=dec_device, dtype=dtype,
        )
        merged_ts = torch.full((1, new_tpf), sigma, device=dec_device, dtype=dtype)
        merged_mod = Modality(
            enabled=True,
            latent=dummy_latent,
            timesteps=merged_ts,
            positions=merged_positions,
            context=dec_prompt,
            context_mask=dec_prompt_mask,
        )

        # 6. Run preprocessor to get timestep embeddings, context, positional encodings
        # In split-GPU mode, use the cloned decoder_preprocessor on cuda:1
        prep = decoder_preprocessor if args.split_gpus else scd_model.base_model.video_args_preprocessor.simple_preprocessor
        video_args = prep.prepare(
            scd_model._cast_modality_dtype(merged_mod)
        )

        # DDiT Paper, Section 3.1: Replace the base model's patchified tokens with
        # the DDiT-projected merged tokens. The decoder sees coarse tokens at the
        # transformer's inner_dim, not the original patchified latent channels.
        # 7. Swap in DDiT-projected tokens (bypass base patchify_proj)
        video_args = dc_replace(video_args, x=merged_proj)
        # SCD Paper, Section 3.2: Decoder uses bidirectional attention (no causal mask).
        # Unlike the encoder which is causal, the decoder attends to all tokens freely.
        video_args = dc_replace(video_args, self_attention_mask=None)  # Decoder: bidirectional

        # SCD Paper, Section 3.3: Encoder features are injected into the decoder via
        # the configured combine mode (additive or token concatenation).
        # 8. Combine encoder features
        if merged_enc is not None:
            video_args = scd_model._combine_encoder_decoder(video_args, merged_enc)

        # SCD Paper, Section 3.2: Run the decoder transformer blocks. At merged resolution,
        # there are s^2 fewer tokens, so each block's self-attention is s^4 faster.
        # DDiT Paper, Section 3.1: The decoder blocks are identical to the base model —
        # the only difference is the input/output resolution handled by merge/unmerge.
        # 9. Run decoder blocks
        perturb = BatchedPerturbationConfig.empty(1)
        for block in scd_model.decoder_blocks:
            video_args, _ = block(video=video_args, audio=None, perturbations=perturb)

        # 10. Strip encoder prefix if token_concat
        dec_x = video_args.x
        dec_emb_ts = video_args.embedded_timestep
        if scd_model.decoder_input_combine in ("token_concat", "token_concat_with_proj") and merged_enc is not None:
            enc_seq = merged_enc.shape[1]
            dec_x = dec_x[:, enc_seq:]
            dec_emb_ts = dec_emb_ts[:, enc_seq:]

        # 11. Scale-shift modulation + DDiT proj_out
        scale_shift = scd_model.base_model.scale_shift_table
        shift, scale_val = (
            scale_shift[None, None].to(device=dec_x.device, dtype=dec_x.dtype) + dec_emb_ts[:, :, None]
        ).unbind(dim=2)
        dec_x = scd_model.base_model.norm_out(dec_x)
        dec_x = dec_x * (1 + scale_val) + shift

        # DDiT Paper, Section 3.1, Step 3 — Output Projection at Coarse Resolution:
        # Project from inner_dim back to C*s*s (the merged channel dimension).
        # 12. Project through DDiT merge layer's proj_out → [1, new_tpf, 128*s*s]
        merged_out = merge_layer.proj_out(dec_x)

        # DDiT Paper, Section 3.1, Step 4 — Spatial Token Unmerge:
        # Reverse of the merge operation: [1, H/s*W/s, C*s*s] → reshape → [1, H*W, C]
        # The learned proj_out ensures the unmerged tokens have correct per-pixel values.
        # 13. Unmerge back to full resolution: [1, tpf, 128]
        velocity = merge_layer.unmerge(merged_out, nf, h, w)

        # DDiT Paper, Section 3.3: Optional residual refinement. A lightweight residual
        # block processes the original full-resolution input and adds a weighted skip
        # connection. This helps preserve fine spatial details that may be lost during
        # the merge/unmerge cycle, especially at higher merge scales (4x).
        # 14. Residual refinement
        if ddit_wrapper.config.residual_weight > 0:
            residual = merge_layer.residual_block(noisy_patch)
            velocity = velocity + ddit_wrapper.config.residual_weight * residual

        return velocity

    # ── Main Generation Loop ──
    # SCD Paper (arXiv:2602.10095), Section 4: Autoregressive video generation.
    # The overall algorithm for generating T frames:
    #
    #   KV_cache = empty
    #   for t = 0 to T-1:
    #     f_enc_t = Encoder(x_{t-1}, sigma=0, KV_cache)   # encode PREVIOUS clean frame
    #     x_t = randn()                                     # sample noise for new frame
    #     for s in sigma_schedule:                           # denoise new frame
    #       v = Decoder(x_t, shift(f_enc_t), sigma_s)       # conditioned on shifted features
    #       x_t = x_t + (sigma_{s+1} - sigma_s) * v        # Euler ODE step
    #
    # SCD Paper, Section 4.2: For long videos, we use chunked generation with overlap.
    # Each chunk processes CHUNK_LATENT frames. Between chunks, the last frame of chunk_k
    # becomes the context (first encoded frame) of chunk_{k+1}, providing temporal continuity.

    all_latent_frames: list[torch.Tensor] = []  # Each: [1, 128, 1, H, W]
    prev_context: torch.Tensor | None = None  # [1, 128, 1, latent_h, latent_w]

    # ── TeaCache initialization ──
    tea_cache: TeaCacheState | None = None
    if args.teacache_thresh is not None:
        tea_cache = TeaCacheState(
            threshold=args.teacache_thresh,
            coefficients=[0.0, 1.0],  # Identity polynomial: rescaled = raw_distance
        )

    gen_start = time.time()
    enc_time_total = 0.0
    dec_time_total = 0.0
    ddit_steps_used = 0  # Accumulates across all frames
    teacache_total_hits = 0
    teacache_total_misses = 0

    for chunk_idx in tqdm(range(num_chunks), desc="Chunks", unit="chunk"):
        # SCD Paper, Section 4.2: Chunk overlap for temporal continuity.
        # First chunk: generate all CHUNK_LATENT frames from scratch.
        # Subsequent chunks: use last frame of previous chunk as context (1-frame overlap),
        # then generate NEW_PER_CHUNK = CHUNK_LATENT - 1 new frames.
        # Determine frames for this chunk
        if chunk_idx == 0:
            new_frames = CHUNK_LATENT
            context_frames: list[torch.Tensor] = []
        else:
            new_frames = NEW_PER_CHUNK
            context_frames = [prev_context]  # type: ignore[list-item]

        generator = torch.Generator(device=device).manual_seed(args.seed + chunk_idx)

        # SCD Paper, Section 4.1: Fresh KV-cache per chunk. The encoder's KV-cache
        # accumulates K/V pairs from all frames within this chunk. When starting a new
        # chunk, we reset the cache (the context frame from the previous chunk gets
        # re-encoded into the new cache as the first entry).
        # Fresh KV-cache per chunk (encoder caches K/V across frames within chunk)
        kv_cache = KVCache.empty()
        kv_cache.is_cache_step = True

        chunk_generated: list[torch.Tensor] = []
        prev_enc_features: torch.Tensor | None = None
        frame_pos = 0  # Temporal position within this chunk

        with torch.inference_mode():
            for f_idx in range(new_frames):
                # ══════════════════════════════════════════════════════════════
                # ENCODER PASS: Process one clean frame with KV-cache
                # ══════════════════════════════════════════════════════════════
                # SCD Paper (arXiv:2602.10095), Section 3.2 + 4.1:
                # The encoder processes the PREVIOUS generated frame (clean, sigma=0)
                # through causal self-attention. KV-cache makes this O(1) per new frame
                # (only computes Q for new tokens, reuses K/V from all prior frames).
                # ── ENCODER: process ONE frame with KV-cache ──
                t_enc = time.time()

                if f_idx == 0 and not context_frames:
                    # SCD Paper, Section 4: For the very first frame of the video,
                    # there is no previous frame to condition on. We use a zero
                    # placeholder, which the model interprets as "unconditional start".
                    # First frame of first chunk: zero placeholder
                    enc_latent = torch.zeros(
                        1, latent_channels, 1, latent_h, latent_w,
                        device=device, dtype=dtype,
                    )
                elif f_idx == 0 and context_frames:
                    # SCD Paper, Section 4.2: For subsequent chunks, the last generated
                    # frame from the previous chunk serves as temporal context. This is
                    # re-encoded into the fresh KV-cache to seed the new chunk.
                    # First frame of subsequent chunks: context from prev chunk
                    enc_latent = context_frames[0]
                else:
                    # SCD Paper, Section 3.4: The encoder processes the most recently
                    # generated (denoised) frame — this is a CLEAN frame (x_0), not noisy.
                    # Use the most recently generated frame
                    enc_latent = chunk_generated[-1]

                # SCD Paper, Section 4.1: Encode one frame, appending its K/V to the cache.
                # After encoding T frames, the cache holds K/V for all T frames' tokens.
                current_enc = encode_single_frame(enc_latent, frame_pos, kv_cache)
                current_enc = current_enc.detach()
                frame_pos += 1
                enc_time_total += time.time() - t_enc

                # ══════════════════════════════════════════════════════════════
                # SHIFT-BY-1 FEATURE ALIGNMENT
                # ══════════════════════════════════════════════════════════════
                # SCD Paper (arXiv:2602.10095), Section 3.4 — Feature Shifting:
                # The decoder for frame_t is conditioned on encoder features from
                # frame_{t-1} (NOT frame_t). This "shift-by-1" ensures the encoder
                # features come from a DIFFERENT frame than the one being denoised,
                # preventing information leakage. prev_enc_features holds features
                # from the previous iteration's encoding.
                # ── DECODER context (shift-by-1) ──
                # In split-GPU mode, encoder features are on cuda:0; transfer to cuda:1 for decoder
                if prev_enc_features is not None:
                    dec_enc_ctx = prev_enc_features.to(dec_device)
                else:
                    # First frame has no previous encoder features — use zeros.
                    # The decoder must generate the first frame with text conditioning only.
                    dec_enc_ctx = torch.zeros(
                        1, tokens_per_frame, current_enc.shape[-1],
                        device=dec_device, dtype=dtype,
                    )

                if chunk_idx == 0 and f_idx <= 1:
                    print(f"  [DEBUG] Frame {f_idx}: enc_features mean={current_enc.float().mean().item():.4f} "
                          f"std={current_enc.float().std().item():.4f} | "
                          f"dec_ctx mean={dec_enc_ctx.float().mean().item():.4f} "
                          f"std={dec_enc_ctx.float().std().item():.4f}")

                # ══════════════════════════════════════════════════════════════
                # DECODER: Denoise one new frame (full denoising trajectory)
                # ══════════════════════════════════════════════════════════════
                # SCD Paper (arXiv:2602.10095), Section 3.2:
                # The decoder takes a noisy latent x_sigma and encoder features from
                # the previous frame, then iteratively denoises using the Euler ODE solver.
                # The decoder uses BIDIRECTIONAL attention (no causal mask) since it only
                # processes one frame at a time — there's no temporal ordering within a frame.
                #
                # The denoising follows the flow matching ODE (LTX-2 formulation):
                #   dx/dsigma = v(x, sigma, features, text)
                # where v is the velocity predicted by the decoder, and sigma goes from
                # sigma_max=1.0 (pure noise) to sigma_min=0.0 (clean sample).
                # ── DECODER: denoise one new frame ──
                t_dec = time.time()
                x_t = torch.randn(
                    1, latent_channels, 1, latent_h, latent_w,
                    device=device, dtype=dtype, generator=generator,
                )

                # Sigma schedule depends on model type:
                # - BézierFlow: Learned optimal schedule via Bézier reparameterization
                # - Dev model: LTX2Scheduler with token-count-dependent shift (matches training)
                #   The shift is computed from the TRAINING WINDOW token count (CHUNK_LATENT
                #   frames × H × W), matching the ShiftedLogitNormalTimestepSampler used
                #   during training. A simple linear schedule DOES NOT WORK — the model was
                #   trained with shifted logit-normal sigmas and cannot denoise linear sigmas.
                # - Distilled model: Non-uniform predefined schedule (8 steps, heavily front-loaded)
                if BEZIER_SIGMA_VALUES is not None:
                    sigmas = torch.tensor(BEZIER_SIGMA_VALUES, device=device, dtype=dtype)
                elif args.distilled:
                    sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=dtype)
                else:
                    # Create dummy latent matching training window for correct shift computation
                    dummy_latent = torch.empty(1, 1, CHUNK_LATENT, latent_h, latent_w)
                    scheduler = LTX2Scheduler()
                    sigmas = scheduler.execute(
                        steps=args.num_inference_steps, latent=dummy_latent,
                    ).to(device=device, dtype=dtype)
                # Decoder positions: Use the CORRECT temporal position for each frame.
                # During training, the decoder sees all frames with their actual temporal
                # positions (frame 0 at t~0.0, frame 1 at t~0.33, etc.). Using position 0
                # for every frame creates a train/inference mismatch in RoPE embeddings.
                # frame_pos was already incremented after encoder call, so the frame being
                # decoded is at temporal index (frame_pos - 1).
                dec_frame_idx = frame_pos - 1
                dec_positions = get_positions_for_frame(dec_frame_idx, target_device=dec_device)

                # DDiT Paper (arXiv:2602.16968), Section 3.2, Algorithm 1:
                # Reset the dynamic scheduler's trajectory history for each new frame.
                # The scheduler tracks z_t at each step to compute 3rd-order finite diffs.
                # Reset dynamic scheduler for this frame's denoising
                if ddit_wrapper is not None and not args.ddit_fixed_schedule:
                    ddit_wrapper.scheduler.reset()

                # Reset TeaCache for this frame's denoising
                if tea_cache is not None:
                    tea_cache.reset()

                # ── Denoising loop: Euler ODE solver from sigma_max to sigma_min ──
                _debug_first_frame = (f_idx == 0 and chunk_idx == 0)
                for step in range(args.num_inference_steps):
                    # Sigma schedule: sigma decreases from 1.0 (pure noise) to 0.0 (clean).
                    # Each step moves from sigma to sigma_next along the ODE trajectory.
                    sigma = sigmas[step]
                    sigma_next = sigmas[step + 1]

                    # Patchify the current noisy latent for the transformer
                    # [1, 128, 1, H, W] → [1, H*W, 128] (one token per spatial position)
                    noisy_patch = patchifier.patchify(x_t)
                    dec_seq = noisy_patch.shape[1]

                    # ── TeaCache: check if this decoder step can be skipped ──
                    # TeaCache (CVPR 2025) caches decoder velocity and reuses it when
                    # consecutive denoising steps produce similar outputs (mid-schedule).
                    # On cache hit: skip ALL decoder blocks + CFG, use cached velocity.
                    if tea_cache is not None:
                        # Always record DDiT trajectory, even on cache hits
                        if ddit_wrapper is not None and not args.ddit_fixed_schedule:
                            ddit_wrapper.scheduler.record(noisy_patch)

                        _tc_force = (step == 0) or (step == args.num_inference_steps - 1)
                        if not tea_cache.should_compute(noisy_patch, force=_tc_force):
                            # Cache hit — reuse velocity, skip entire decode
                            velocity = tea_cache.cached_velocity
                            dt = sigma_next - sigma
                            if _debug_first_frame and step in (0, 1, 5, 14, 28, 29):
                                print(f"    [DEBUG step {step:2d}] sigma={sigma.item():.4f}→{sigma_next.item():.4f} "
                                      f"[TC-HIT] reusing cached velocity")
                            noisy_patch = (noisy_patch.float() + velocity.float() * dt.float()).to(dtype)
                            x_t = patchifier.unpatchify(
                                noisy_patch,
                                output_shape=VideoLatentShape(
                                    frames=1, height=latent_h, width=latent_w,
                                    batch=1, channels=latent_channels,
                                ),
                            )
                            continue  # Skip to next denoising step

                    # ── DDiT Dynamic Scale Selection ──
                    # DDiT Paper (arXiv:2602.16968), Section 3.2, Algorithm 1:
                    # At each denoising step, decide whether to run the decoder at native
                    # resolution (scale=1) or coarse resolution (scale=2 or 4).
                    #
                    # Dynamic scheduling uses 3rd-order finite differences of the trajectory:
                    #   Delta^3_z_t = z_t - 3*z_{t-1} + 3*z_{t-2} - z_{t-3}
                    # This measures the "jerk" (rate of change of acceleration) of denoising.
                    # When the trajectory is smooth (low jerk), coarse resolution suffices.
                    # When it's rapidly changing (high jerk), native resolution is needed.
                    #
                    # The decision criterion (DDiT Eq. 5):
                    #   Partition Delta^3_z into spatial patches
                    #   Compute std(patch) for each patch
                    #   If rho-percentile(stds) < threshold tau -> use coarse scale
                    # where tau=0.001 and rho=0.4 are the default hyperparameters.
                    # Decide DDiT scale for this step
                    if ddit_wrapper is not None:
                        if args.ddit_fixed_schedule:
                            # DDiT Paper, Section 5.2: Fixed schedule alternative — run
                            # native resolution for the first few (head) and last few (tail)
                            # steps, coarse resolution for everything in between. Simpler
                            # but less adaptive than the dynamic scheduler.
                            # Legacy fixed schedule
                            ddit_scale = (
                                args.ddit_scale
                                if step >= args.ddit_native_head
                                and step < args.num_inference_steps - args.ddit_native_tail
                                else 1
                            )
                        else:
                            # DDiT Paper, Section 3.2, Algorithm 1: Record current latent
                            # in the trajectory buffer, then compute optimal scale based on
                            # 3rd-order finite differences. Needs at least 4 samples (steps
                            # 0-2 always run native to build up the trajectory history).
                            # Dynamic scheduler (paper's method): record latent, compute optimal scale
                            # Skip recording if TeaCache already recorded for this step
                            if tea_cache is None:
                                ddit_wrapper.scheduler.record(noisy_patch)
                            ddit_scale = ddit_wrapper.scheduler.compute_schedule(
                                noisy_patch, step, 1, latent_h, latent_w,
                            )
                            # Clamp to trained scales (merge_layers keys are strings)
                            if str(ddit_scale) not in ddit_wrapper.merge_layers:
                                ddit_scale = 1
                    else:
                        ddit_scale = 1

                    if ddit_scale > 1:
                        ddit_steps_used += 1
                        # DDiT Paper, Section 3.1: Coarse-resolution decoder path.
                        # merge(z) → decoder(z_coarse) → unmerge(v_coarse) → v_fine
                        # Token count reduced by scale^2, attention cost by scale^4.
                        # DDiT path: merged-resolution decoder
                        velocity = ddit_decode_one_frame(
                            noisy_patch, dec_enc_ctx, sigma.item(),
                            dec_positions, ddit_scale,
                        )
                    else:
                        # SCD Paper, Section 3.2: Native-resolution decoder path.
                        # Standard forward_decoder with full spatial resolution.
                        # Used for head/tail steps (fine detail) or when DDiT is disabled.
                        # Native resolution decoder
                        dec_prompt = prompt_embeds_dec if args.split_gpus else prompt_embeds
                        dec_prompt_mask = prompt_mask_dec if args.split_gpus else prompt_mask
                        dec_modality = Modality(
                            enabled=True,
                            latent=noisy_patch,
                            timesteps=torch.full((1, dec_seq), sigma.item(), device=dec_device, dtype=dtype),
                            positions=dec_positions,
                            context=dec_prompt,
                            context_mask=dec_prompt_mask,
                        )

                        if args.split_gpus:
                            # Adaptation: Split-GPU requires manual decoder execution because
                            # the encoder (cuda:0) and decoder (cuda:1) are on different devices.
                            # We use the cloned preprocessor on cuda:1 to compute timestep
                            # embeddings and positional encodings without cross-device transfers.
                            # Split-GPU: manually run decoder on cuda:1 with cloned preprocessor
                            video_args = decoder_preprocessor.prepare(
                                scd_model._cast_modality_dtype(dec_modality)
                            )
                            # SCD Paper, Section 3.2: Decoder uses bidirectional attention
                            # (self_attention_mask=None removes causal constraint).
                            video_args = dc_replace(video_args, self_attention_mask=None)
                            # SCD Paper, Section 3.3: Inject encoder features into decoder input
                            # using the configured combine mode (add or token_concat).
                            video_args = scd_model._combine_encoder_decoder(video_args, dec_enc_ctx)
                            from ltx_core.guidance.perturbations import BatchedPerturbationConfig
                            perturb = BatchedPerturbationConfig.empty(1)
                            for block in scd_model.decoder_blocks:
                                video_args, _ = block(video=video_args, audio=None, perturbations=perturb)

                            # SCD Paper, Section 3.3: When using token_concat, encoder features
                            # are prepended to the decoder sequence. After the transformer blocks,
                            # we strip the encoder prefix to get only decoder output tokens.
                            # Strip encoder prefix if token_concat, then apply output projection
                            dec_x = video_args.x
                            dec_emb_ts = video_args.embedded_timestep
                            if scd_model.decoder_input_combine in ("token_concat", "token_concat_with_proj") and dec_enc_ctx is not None:
                                enc_seq = dec_enc_ctx.shape[1]
                                dec_x = dec_x[:, enc_seq:]
                                dec_emb_ts = dec_emb_ts[:, enc_seq:]
                            # Adaptation: _process_output applies final layer norm, scale-shift
                            # modulation, and linear projection to produce velocity in latent space.
                            velocity = scd_model.base_model._process_output(
                                scd_model.base_model.scale_shift_table,
                                scd_model.base_model.norm_out,
                                scd_model.base_model.proj_out,
                                dec_x,
                                dec_emb_ts,
                            )
                        else:
                            # SCD Paper, Section 3.2: Standard single-GPU decoder forward pass.
                            # forward_decoder handles preprocessing, encoder feature injection,
                            # decoder blocks, and output projection in one call.
                            velocity, _ = scd_model.forward_decoder(
                                video=dec_modality,
                                encoder_features=dec_enc_ctx,
                                audio=None,
                                perturbations=None,
                            )

                            # CFG: run unconditional pass and apply guidance
                            if use_cfg:
                                uncond_modality = Modality(
                                    enabled=True,
                                    latent=noisy_patch,
                                    timesteps=torch.full((1, dec_seq), sigma.item(), device=dec_device, dtype=dtype),
                                    positions=dec_positions,
                                    context=null_embeds,
                                    context_mask=null_mask,
                                )
                                velocity_uncond, _ = scd_model.forward_decoder(
                                    video=uncond_modality,
                                    encoder_features=dec_enc_ctx,
                                    audio=None,
                                    perturbations=None,
                                )
                                velocity = velocity_uncond + args.guidance_scale * (velocity - velocity_uncond)

                    # ── Update TeaCache with freshly computed velocity ──
                    if tea_cache is not None:
                        tea_cache.cached_velocity = velocity.detach().clone()

                    # ── Euler ODE Step ──
                    # DDiT Paper (arXiv:2602.16968), Eq. 3 / SCD Paper (arXiv:2602.10095), Section 3.1:
                    # Flow matching Euler step along the probability flow ODE:
                    #   z_{t+1} = z_t + (sigma_{t+1} - sigma_t) * v_pred(z_t, sigma_t)
                    #
                    # Here dt = sigma_next - sigma < 0 (sigma decreasing), so this steps
                    # the latent from higher noise (sigma) toward lower noise (sigma_next).
                    # At the final step, sigma_next = 0 and z converges to the clean sample x_0.
                    #
                    # The velocity v_pred is the decoder's output — it predicts the instantaneous
                    # direction of the denoising trajectory in latent space.
                    # Euler step: x_{t+dt} = x_t + v * dt
                    # Use float32 for accumulation to avoid bfloat16 precision loss over 30 steps
                    # (matches EulerDiffusionStep in ltx-core which casts to float32)
                    dt = sigma_next - sigma
                    if _debug_first_frame and step in (0, 1, 5, 14, 28, 29):
                        v_mean = velocity.float().mean().item()
                        v_std = velocity.float().std().item()
                        x_mean = noisy_patch.float().mean().item()
                        x_std = noisy_patch.float().std().item()
                        print(f"    [DEBUG step {step:2d}] sigma={sigma.item():.4f}→{sigma_next.item():.4f} "
                              f"dt={dt.item():.4f} | v: mean={v_mean:.4f} std={v_std:.4f} "
                              f"| x_t: mean={x_mean:.4f} std={x_std:.4f}")
                    noisy_patch = (noisy_patch.float() + velocity.float() * dt.float()).to(dtype)
                    if _debug_first_frame and step in (0, 1, 5, 14, 28, 29):
                        x_after = noisy_patch.float().mean().item()
                        x_after_std = noisy_patch.float().std().item()
                        print(f"              after: x_t: mean={x_after:.4f} std={x_after_std:.4f}")

                    # Unpatchify back to spatial format for the next iteration
                    # [1, H*W, 128] → [1, 128, 1, H, W]
                    x_t = patchifier.unpatchify(
                        noisy_patch,
                        output_shape=VideoLatentShape(
                            frames=1, height=latent_h, width=latent_w,
                            batch=1, channels=latent_channels,
                        ),
                    )

                dec_time_total += time.time() - t_dec

                # SCD Paper, Section 3.4 — Shift-by-1 update:
                # Save current encoder features to be used as decoder context for the
                # NEXT frame. This implements the temporal shift: enc(frame_t) → dec(frame_{t+1}).
                # Store results
                prev_enc_features = current_enc
                chunk_generated.append(x_t.detach())

        # ── Frame Assembly ──
        # SCD Paper, Section 4.2: Collect all newly generated frames from this chunk.
        # Each frame is a fully denoised latent: [1, 128, 1, H, W].
        # Collect new frames
        for frame in chunk_generated:
            all_latent_frames.append(frame.cpu())

        # SCD Paper, Section 4.2 — Chunk Overlap Mechanism:
        # The last generated frame of this chunk becomes the "context frame" for the
        # next chunk. When the next chunk starts, this frame will be:
        #   1. Re-encoded through the encoder (populating the fresh KV-cache)
        #   2. Its encoder features will condition the decoder for the next chunk's first NEW frame
        # This overlap ensures temporal continuity across chunk boundaries.
        # Save last frame as context for next chunk
        prev_context = chunk_generated[-1].detach().clone()

        elapsed = time.time() - gen_start
        frames_done = len(all_latent_frames)
        rate = elapsed / frames_done if frames_done > 0 else 0
        remaining = (actual_latent - frames_done) * rate
        _tc_info = ""
        if tea_cache is not None:
            teacache_total_hits += tea_cache.hits
            teacache_total_misses += tea_cache.misses
            _chunk_total = tea_cache.hits + tea_cache.misses
            _chunk_pct = tea_cache.hits / _chunk_total * 100 if _chunk_total > 0 else 0
            _tc_info = f" | TC: {tea_cache.hits}/{_chunk_total} hits ({_chunk_pct:.0f}%)"
            # Reset per-chunk counters (per-frame reset happens in the loop)
            tea_cache.hits = 0
            tea_cache.misses = 0

        tqdm.write(
            f"  Chunk {chunk_idx + 1}/{num_chunks}: "
            f"{len(chunk_generated)} new frames | "
            f"Total: {frames_done}/{actual_latent} | "
            f"ETA: {remaining / 60:.1f} min{_tc_info}"
        )

        del chunk_generated, kv_cache
        torch.cuda.empty_cache()

    gen_elapsed = time.time() - gen_start
    # SCD Paper, Section 4.2: The encoder/decoder timing split shows the compute asymmetry.
    # With KV-cache, the encoder is typically <10% of total time (O(1) per frame).
    # The decoder dominates because it runs num_inference_steps full forward passes per frame.
    # DDiT Paper: When DDiT is enabled, decoder time should be significantly reduced
    # (2-4x speedup) because coarse-resolution steps have scale^4 fewer attention FLOPs.
    print(f"\n  Generation complete: {gen_elapsed / 60:.1f} min ({gen_elapsed / actual_latent:.1f}s/frame)")
    print(f"  Encoder: {enc_time_total:.1f}s ({enc_time_total / gen_elapsed * 100:.0f}%) | "
          f"Decoder: {dec_time_total:.1f}s ({dec_time_total / gen_elapsed * 100:.0f}%)")
    if ddit_wrapper is not None:
        total_steps_all = args.num_inference_steps * actual_latent
        if args.ddit_fixed_schedule:
            native_steps = args.ddit_native_head + args.ddit_native_tail
            ddit_steps = args.num_inference_steps - native_steps
            print(f"  DDiT: {ddit_steps}/{args.num_inference_steps} steps/frame at {args.ddit_scale}x "
                  f"(FIXED: {args.ddit_native_head} head + {args.ddit_native_tail} tail)")
        else:
            # DDiT Paper, Section 3.2: The dynamic scheduler adaptively selects scale per step.
            # The percentage of merged steps indicates how "smooth" the denoising trajectory was.
            # Typical values: 60-80% merged steps for natural video content.
            pct = ddit_steps_used / total_steps_all * 100 if total_steps_all > 0 else 0
            print(f"  DDiT: DYNAMIC scheduler — merged {ddit_steps_used}/{total_steps_all} total steps "
                  f"({pct:.0f}%)")
    if tea_cache is not None:
        total_steps_all = args.num_inference_steps * actual_latent
        tc_total = teacache_total_hits + teacache_total_misses
        tc_pct = teacache_total_hits / tc_total * 100 if tc_total > 0 else 0
        skipped_calls = teacache_total_hits
        # Each skipped step saves 1 decoder call (or 2 with CFG: cond + uncond)
        calls_per_step = 2 if use_cfg else 1
        saved_decoder_calls = skipped_calls * calls_per_step
        total_decoder_calls = total_steps_all * calls_per_step
        print(f"  TeaCache: {teacache_total_hits}/{tc_total} steps cached ({tc_pct:.0f}% hit rate) | "
              f"{saved_decoder_calls}/{total_decoder_calls} decoder calls saved")

    # ── VAE decode ──
    # Adaptation: LTX-2's temporal VAE compresses 8 pixel frames per latent frame, with the
    # formula: F_pixel = (F_latent - 1) * 8 + 1. This temporal compression means adjacent
    # latent frames share temporal boundary information. Decoding the full sequence at once
    # preserves this cross-frame continuity. Independent small-batch decoding loses cross-batch
    # temporal info and produces fewer frames (e.g., 19 batches * 25 frames = 475 instead of 601).
    # Decode as large a batch as fits in VRAM.
    all_latent = torch.cat(all_latent_frames, dim=2)  # [1, 128, N, H, W]
    n_latent = all_latent.shape[2]
    expected_pixel = (n_latent - 1) * 8 + 1
    print(f"\n  Decoding {n_latent} latent frames → {expected_pixel} pixel frames with VAE...")

    # Try full-sequence decode first; fall back to batched if OOM
    try:
        with torch.inference_mode():
            full_video = vae_decoder(all_latent.to("cuda:1"))  # [1, 3, F, H, W]
        full_video = full_video[0].cpu()  # [3, F, H, W]
        print(f"  Full-sequence decode successful: {full_video.shape[1]} frames")
    except torch.cuda.OutOfMemoryError:
        print("  Full decode OOM — falling back to batched decode with overlap...")
        torch.cuda.empty_cache()

        # Adaptation: When full-sequence VAE decode exceeds VRAM, we fall back to overlapping
        # batched decode. The 1-frame overlap between batches mirrors the SCD chunk overlap
        # strategy (Section 4.2) — it ensures the temporal VAE's causal convolutions have
        # consistent context at batch boundaries. Without overlap, there would be visible
        # seams every DECODE_BATCH * 8 pixel frames.
        # Decode in overlapping batches: 8 latent frames with 1-frame overlap
        DECODE_BATCH = 8
        OVERLAP = 1  # 1 latent frame overlap for temporal continuity
        STRIDE = DECODE_BATCH - OVERLAP
        all_pixel_chunks: list[torch.Tensor] = []

        for i in tqdm(range(0, n_latent, STRIDE), desc="VAE decode", unit="batch"):
            end = min(i + DECODE_BATCH, n_latent)
            batch = all_latent[:, :, i:end].to("cuda:1")

            with torch.inference_mode():
                pixels = vae_decoder(batch)  # [1, 3, F_pixel, H, W]

            # For overlapping batches (not first), skip the overlap region in pixel space
            if i > 0 and OVERLAP > 0:
                skip_pixels = OVERLAP * 8  # Each latent frame ≈ 8 pixel frames
                pixels = pixels[:, :, skip_pixels:]

            all_pixel_chunks.append(pixels.cpu())
            del batch, pixels
            torch.cuda.empty_cache()

        full_video = torch.cat(all_pixel_chunks, dim=2)[0]  # [3, F, H, W]

    del all_latent
    torch.cuda.empty_cache()

    # Convert from VAE output range [-1, 1] to display range [0, 1]
    full_video = (full_video + 1.0) / 2.0
    full_video = full_video.clamp(0, 1)

    # Trim to target frame count
    if full_video.shape[1] > actual_pixel:
        full_video = full_video[:, :actual_pixel]

    print(f"  Final video: {full_video.shape[1]} frames ({full_video.shape[1] / args.fps:.1f}s)")
    print(f"  Pixel range: [{full_video.min():.3f}, {full_video.max():.3f}]")

    # ── Save ──
    from ltx_trainer.video_utils import save_video

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(full_video, output_path, fps=args.fps)

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Saved: {output_path}")
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    print(f"  Video: {full_video.shape[1]} frames, {full_video.shape[1] / args.fps:.1f}s @ {args.fps} fps")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
