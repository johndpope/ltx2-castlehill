#!/usr/bin/env python
"""Benchmark: standard FFN vs gFFN-HRR wall-clock speed.

Uses fewer layers to fit both variants on GPU for fair A/B comparison.
Then extrapolates to full 48 layers.
"""

import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-core" / "src"))
sys.path.insert(0, str(ROOT / "ltx-trainer" / "src"))

CHECKPOINT = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
DATA_DIR = Path("/media/2TB/omnitransfer/data/ditto_subset")
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
NUM_LAYERS = 8  # Fit on GPU for fair comparison
WARMUP = 5
RUNS = 20


def load_data():
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    latent_data = torch.load(DATA_DIR / "latents/000000.pt", map_location="cpu", weights_only=True)
    cond_data = torch.load(DATA_DIR / "conditions_final/000000.pt", map_location="cpu", weights_only=True)

    raw = latent_data["latents"]
    C, F, H, W = raw.shape
    patchifier = VideoLatentPatchifier(patch_size=1)
    tokens = patchifier.patchify(raw.unsqueeze(0)).to(DEVICE, DTYPE)
    seq = tokens.shape[1]

    noise = torch.randn_like(tokens)
    noisy = 0.5 * tokens + 0.5 * noise

    shape = VideoLatentShape(frames=F, height=H, width=W, batch=1, channels=C)
    coords = patchifier.get_patch_grid_bounds(output_shape=shape, device=torch.device(DEVICE))
    pos = get_pixel_coords(coords, SpatioTemporalScaleFactors.default(), causal_fix=True).to(DTYPE)
    fps = latent_data.get("fps", torch.tensor(24.0)).item()
    pos[:, 0] /= fps

    ctx = cond_data["video_prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE)
    mask = cond_data.get("prompt_attention_mask", None)
    if mask is not None:
        mask = mask.unsqueeze(0).to(DEVICE)

    ts = torch.full((1, seq), 0.5, device=DEVICE, dtype=DTYPE)
    return Modality(enabled=True, latent=noisy, timesteps=ts, positions=pos, context=ctx, context_mask=mask)


def bench(model, video_mod, label):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    for _ in range(WARMUP):
        with torch.no_grad():
            model(video=video_mod, audio=None, perturbations=None)
        torch.cuda.synchronize(DEVICE)

    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(video=video_mod, audio=None, perturbations=None)
        torch.cuda.synchronize(DEVICE)
        times.append(time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    print(f"  {label}: avg={avg*1000:.1f}ms ± {std*1000:.1f}ms  min={mn*1000:.1f}ms  peak={peak:.1f}GB")
    return avg, mn


def main():
    print("Loading data...")
    video_mod = load_data()

    print("Loading model...")
    from ltx_trainer.model_loader import load_transformer
    model = load_transformer(CHECKPOINT, device="cpu", dtype=DTYPE)

    # Truncate to fit
    orig = len(model.transformer_blocks)
    model.transformer_blocks = model.transformer_blocks[:NUM_LAYERS]
    print(f"  Using {NUM_LAYERS}/{orig} layers for fair A/B comparison")

    model.to(DEVICE)
    mem_ffn = torch.cuda.memory_allocated(DEVICE) / 1e9
    ffn_params = sum(p.numel() for b in model.transformer_blocks for a in ["ff", "audio_ff"]
                     if hasattr(b, a) for p in getattr(b, a).parameters())
    print(f"  Standard FFN: {ffn_params/1e6:.0f}M FFN params, {mem_ffn:.1f}GB VRAM")

    # ── Benchmark standard FFN ──────────────────────────────────────────────
    print(f"\nBenchmark ({WARMUP} warmup + {RUNS} runs):")
    avg_ffn, min_ffn = bench(model, video_mod, "Standard FFN ")

    # ── Replace with gFFN-HRR ──────────────────────────────────────────────
    from ltx_core.model.transformer.gffn import gFFNGlobalHRR

    for block in model.transformer_blocks:
        if hasattr(block, "ff"):
            dim = block.ff.net[0].proj.in_features
            block.ff = gFFNGlobalHRR(dim=dim, dim_out=dim, num_shifts=8, proj_factor=4,
                                      mode="inner", shift_strategy="log_uniform").to(DEVICE, DTYPE)
        if hasattr(block, "audio_ff"):
            dim = block.audio_ff.net[0].proj.in_features
            block.audio_ff = gFFNGlobalHRR(dim=dim, dim_out=dim, num_shifts=8, proj_factor=4,
                                            mode="inner", shift_strategy="log_uniform").to(DEVICE, DTYPE)

    gc.collect()
    torch.cuda.empty_cache()

    gffn_params = sum(p.numel() for b in model.transformer_blocks for a in ["ff", "audio_ff"]
                      if hasattr(b, a) for p in getattr(b, a).parameters())
    mem_gffn = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"\n  gFFN-HRR: {gffn_params/1e6:.0f}M FFN params, {mem_gffn:.1f}GB VRAM")

    avg_gffn, min_gffn = bench(model, video_mod, "gFFN-HRR     ")

    # ── Results ─────────────────────────────────────────────────────────────
    speedup_avg = avg_ffn / avg_gffn
    speedup_min = min_ffn / min_gffn

    print(f"\n{'='*60}")
    print(f"  RESULTS ({NUM_LAYERS} layers, seq_len=1344)")
    print(f"{'='*60}")
    print(f"  Standard FFN:  {avg_ffn*1000:.1f}ms  ({ffn_params/1e6:.0f}M params)")
    print(f"  gFFN-HRR:      {avg_gffn*1000:.1f}ms  ({gffn_params/1e6:.0f}M params)")
    print(f"  Speedup:       {speedup_avg:.2f}x (avg)  {speedup_min:.2f}x (best)")
    print(f"  Param savings: {(1 - gffn_params/ffn_params)*100:.0f}% fewer FFN params")
    print(f"  VRAM savings:  {mem_ffn - mem_gffn:.1f}GB less")
    print(f"\n  Extrapolated to 48 layers:")
    print(f"    FFN:      ~{avg_ffn * 48/NUM_LAYERS * 1000:.0f}ms")
    print(f"    gFFN-HRR: ~{avg_gffn * 48/NUM_LAYERS * 1000:.0f}ms")
    print(f"    Saving:   ~{(avg_ffn - avg_gffn) * 48/NUM_LAYERS * 1000:.0f}ms per forward pass")


if __name__ == "__main__":
    main()
