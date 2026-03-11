#!/usr/bin/env python
"""
Sanity test: end-to-end forward+backward pass with gFFN-HRR replacing standard FFN.

Loads the real LTX2 transformer, swaps all FFN layers for gFFNGlobalHRR,
runs a forward pass with real precached Ditto data, backward pass, reports results.

Usage:
    cd packages/ltx-trainer
    uv run python scripts/sanity_test_gffn.py              # 4 layers on cuda:1
    uv run python scripts/sanity_test_gffn.py --layers 0   # all 48 layers
    uv run python scripts/sanity_test_gffn.py --cpu         # CPU fallback

This is NOT SCD — just the plain transformer forward pass.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# Add paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-core" / "src"))
sys.path.insert(0, str(ROOT / "ltx-trainer" / "src"))

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
DATA_DIR = Path("/media/2TB/omnitransfer/data/ditto_subset")
LATENT_DIR = DATA_DIR / "latents"
COND_DIR = DATA_DIR / "conditions_final"

# gFFN-HRR config
GFFN_KWARGS = dict(
    num_shifts=8,
    shift_strategy="log_uniform",
    proj_factor=4,
    mode="inner",
)


def parse_args():
    parser = argparse.ArgumentParser(description="gFFN-HRR sanity test")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers (default 4, 0=all 48)")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (slower, no VRAM needed)")
    parser.add_argument("--device", type=str, default="cuda:1", help="CUDA device (default cuda:1 = RTX 5090)")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def load_sample(idx: int = 0):
    """Load one precached Ditto sample (latent + conditions)."""
    latent_path = LATENT_DIR / f"{idx:06d}.pt"
    cond_path = COND_DIR / f"{idx:06d}.pt"
    latent_data = torch.load(latent_path, map_location="cpu", weights_only=True)
    cond_data = torch.load(cond_path, map_location="cpu", weights_only=True)
    return latent_data, cond_data


def patchify_latent(latent_tensor: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """Convert [C, F, H, W] latent to [1, seq_len, C] tokens."""
    C, F, H, W = latent_tensor.shape
    tokens = latent_tensor.permute(1, 2, 3, 0).reshape(1, F * H * W, C)
    return tokens, F, H, W


def make_video_positions(num_frames, height, width, batch_size=1, fps=24.0, device="cpu", dtype=torch.bfloat16):
    """Generate video position embeddings [B, 3, seq_len, 2]."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    shape = VideoLatentShape(frames=num_frames, height=height, width=width, batch=batch_size, channels=128)
    latent_coords = patchifier.get_patch_grid_bounds(output_shape=shape, device=torch.device(device))
    pixel_coords = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=SpatioTemporalScaleFactors.default(),
        causal_fix=True,
    ).to(dtype)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
    return pixel_coords


def replace_ffn_with_gffn(model, gffn_kwargs):
    """Replace all FeedForward modules with gFFNGlobalHRR in-place."""
    from ltx_core.model.transformer.gffn import gFFNGlobalHRR

    count = 0
    for block in model.transformer_blocks:
        if hasattr(block, "ff"):
            old_ff = block.ff
            dim = _get_ffn_dim(old_ff)
            block.ff = gFFNGlobalHRR(dim=dim, dim_out=dim, **gffn_kwargs).to(
                device=next(old_ff.parameters()).device, dtype=next(old_ff.parameters()).dtype
            )
            del old_ff
            count += 1
        if hasattr(block, "audio_ff"):
            old_ff = block.audio_ff
            dim = _get_ffn_dim(old_ff)
            block.audio_ff = gFFNGlobalHRR(dim=dim, dim_out=dim, **gffn_kwargs).to(
                device=next(old_ff.parameters()).device, dtype=next(old_ff.parameters()).dtype
            )
            del old_ff
            count += 1
    return count


def _get_ffn_dim(ffn_module):
    if hasattr(ffn_module, "net"):
        first = ffn_module.net[0]
        if hasattr(first, "proj"):
            return first.proj.in_features
    if hasattr(ffn_module, "dim"):
        return ffn_module.dim
    raise ValueError(f"Cannot determine dim from {type(ffn_module)}")


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    args = parse_args()
    DEVICE = "cpu" if args.cpu else args.device
    DTYPE = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    NUM_LAYERS = args.layers if args.layers > 0 else None
    is_cuda = DEVICE.startswith("cuda")

    print_section("gFFN-HRR Sanity Test: End-to-End Forward+Backward")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Data dir:   {DATA_DIR}")
    print(f"  Device:     {DEVICE}")
    print(f"  Dtype:      {DTYPE}")
    print(f"  gFFN config: {GFFN_KWARGS}")
    print(f"  Num layers: {NUM_LAYERS or 'all 48'}")

    # ── Step 1: Load sample data ────────────────────────────────────────────
    print_section("Step 1: Load precached Ditto sample")
    latent_data, cond_data = load_sample(0)

    if isinstance(latent_data, dict):
        raw_latent = latent_data["latents"]
        fps_meta = latent_data.get("fps", None)
    else:
        raw_latent = latent_data
        fps_meta = None

    print(f"  Raw latent shape: {raw_latent.shape} (C, F, H, W)")
    fps = fps_meta.item() if fps_meta is not None else 24.0
    print(f"  FPS: {fps}")

    tokens, F, H, W = patchify_latent(raw_latent)
    seq_len = tokens.shape[1]
    print(f"  Patchified tokens: {tokens.shape} → seq_len={seq_len}")

    if isinstance(cond_data, dict):
        video_prompt_embeds = cond_data["video_prompt_embeds"]
        prompt_attention_mask = cond_data.get("prompt_attention_mask", None)
    else:
        video_prompt_embeds = cond_data
        prompt_attention_mask = None
    print(f"  Video prompt embeds: {video_prompt_embeds.shape}")

    # ── Step 2: Load transformer ────────────────────────────────────────────
    print_section("Step 2: Load LTX2 transformer")
    t0 = time.time()
    from ltx_trainer.model_loader import load_transformer
    model = load_transformer(CHECKPOINT, device="cpu", dtype=DTYPE)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params / 1e9:.2f}B")

    # ── Step 3: Replace FFN with gFFN-HRR ───────────────────────────────────
    print_section("Step 3: Replace FFN → gFFN-HRR")
    if NUM_LAYERS is not None:
        orig_blocks = len(model.transformer_blocks)
        model.transformer_blocks = model.transformer_blocks[:NUM_LAYERS]
        print(f"  Truncated {orig_blocks} → {NUM_LAYERS} layers")

    n_replaced = replace_ffn_with_gffn(model, GFFN_KWARGS)
    print(f"  Replaced {n_replaced} FFN modules")

    new_total = sum(p.numel() for p in model.parameters())
    gffn_params = sum(p.numel() for n, p in model.named_parameters() if "ff." in n or "audio_ff." in n)
    print(f"  New total params: {new_total / 1e9:.2f}B (was {total_params / 1e9:.2f}B)")
    print(f"  gFFN-HRR params: {gffn_params:,}")

    # ── Step 4: Move to device ──────────────────────────────────────────────
    print_section(f"Step 4: Move model to {DEVICE}")
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    t0 = time.time()
    model = model.to(DEVICE)
    model.train()
    print(f"  Moved in {time.time() - t0:.1f}s")

    if is_cuda:
        print(f"  GPU memory: {torch.cuda.memory_allocated(DEVICE) / 1e9:.1f}GB")

    # Freeze everything except gFFN
    for name, param in model.named_parameters():
        param.requires_grad_("ff." in name or "audio_ff." in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (gFFN only): {trainable:,}")

    # Enable gradient checkpointing
    model.set_gradient_checkpointing(True)
    print(f"  Gradient checkpointing: enabled")

    # ── Step 5: Forward pass ────────────────────────────────────────────────
    print_section("Step 5: Forward pass")
    from ltx_core.model.transformer.modality import Modality

    batch_size = 1
    noisy_latent = tokens.to(device=DEVICE, dtype=DTYPE)
    sigma = 0.5
    noise = torch.randn_like(noisy_latent)
    noisy_latent = (1 - sigma) * noisy_latent + sigma * noise

    video_timesteps = torch.full((batch_size, seq_len), sigma, device=DEVICE, dtype=DTYPE)
    video_positions = make_video_positions(F, H, W, batch_size, fps, DEVICE, DTYPE)

    context = video_prompt_embeds.to(device=DEVICE, dtype=DTYPE)
    if context.dim() == 2:
        context = context.unsqueeze(0)

    ctx_mask = None
    if prompt_attention_mask is not None:
        ctx_mask = prompt_attention_mask.to(device=DEVICE)
        if ctx_mask.dim() == 1:
            ctx_mask = ctx_mask.unsqueeze(0)

    video_modality = Modality(
        enabled=True, latent=noisy_latent, timesteps=video_timesteps,
        positions=video_positions, context=context, context_mask=ctx_mask,
    )

    print(f"  Video latent: {noisy_latent.shape} ({noisy_latent.dtype})")
    print(f"  Video positions: {video_positions.shape}")
    print(f"  Context: {context.shape}")

    if is_cuda:
        torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    try:
        video_out, audio_out = model(video=video_modality, audio=None, perturbations=None)
    except Exception as e:
        print(f"\n  FORWARD PASS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if is_cuda:
        torch.cuda.synchronize(DEVICE)
    fwd_time = time.time() - t0

    print(f"\n  Forward pass OK!")
    print(f"  Video output: {video_out.shape} ({video_out.dtype})")
    print(f"  Forward time: {fwd_time:.2f}s")
    if is_cuda:
        print(f"  GPU memory: {torch.cuda.memory_allocated(DEVICE) / 1e9:.1f}GB "
              f"(peak: {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.1f}GB)")

    # ── Step 6: Backward pass ───────────────────────────────────────────────
    print_section("Step 6: Backward pass")
    target = noise - tokens.to(device=DEVICE, dtype=DTYPE)
    loss = torch.nn.functional.mse_loss(video_out, target)
    print(f"  Loss: {loss.item():.6f}")

    if is_cuda:
        torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    try:
        loss.backward()
    except Exception as e:
        print(f"\n  BACKWARD PASS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if is_cuda:
        torch.cuda.synchronize(DEVICE)
    bwd_time = time.time() - t0

    print(f"  Backward pass OK!")
    print(f"  Backward time: {bwd_time:.2f}s")
    if is_cuda:
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.1f}GB")

    # ── Step 7: Gradient analysis ───────────────────────────────────────────
    print_section("Step 7: Gradient analysis")
    grad_norms = {}
    nan_grad_params = []
    zero_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gn = param.grad.norm().item()
            grad_norms[name] = gn
            if gn == 0.0:
                zero_grad_params.append(name)
            if torch.isnan(param.grad).any():
                nan_grad_params.append(name)

    if grad_norms:
        norms = list(grad_norms.values())
        print(f"  Params with grads: {len(grad_norms)}")
        print(f"  Grad norm: min={min(norms):.2e}, max={max(norms):.2e}, mean={sum(norms)/len(norms):.2e}")
        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top-5 gradient norms:")
        for name, gn in sorted_grads[:5]:
            print(f"    {gn:.2e}  {name}")
    else:
        print("  WARNING: No gradients found!")

    if zero_grad_params:
        print(f"\n  WARNING: {len(zero_grad_params)} params have zero gradients")
    if nan_grad_params:
        print(f"\n  CRITICAL: {len(nan_grad_params)} params have NaN gradients!")
        return 1

    # ── Step 8: Quick optimizer step ────────────────────────────────────────
    print_section("Step 8: Quick optimizer step")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01,
    )
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        video_out2, _ = model(video=video_modality, audio=None, perturbations=None)
    loss2 = torch.nn.functional.mse_loss(video_out2, target)
    print(f"  Loss after 1 step: {loss2.item():.6f} (was {loss.item():.6f})")
    if loss2.item() < loss.item():
        print(f"  Loss decreased — optimization working!")
    else:
        print(f"  Loss did not decrease (may need more steps)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print_section("SUMMARY")
    print(f"  Forward:   OK ({fwd_time:.2f}s)")
    print(f"  Backward:  OK ({bwd_time:.2f}s)")
    print(f"  Gradients: {'OK' if not nan_grad_params else 'NaN!'}")
    print(f"  Optimizer: OK")
    if is_cuda:
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.1f}GB")
    print(f"  Trainable: {trainable:,} params ({trainable/1e6:.1f}M)")
    print(f"\n  gFFN-HRR is ready for training!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
