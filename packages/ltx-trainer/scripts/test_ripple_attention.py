#!/usr/bin/env python3
"""Inspect RippleVideoAttention: shift schedule, single forward pass, NaN check, timing.

Tests at LTX-2.3 dimensions (inner=4096, H=32, D=128) for three resolutions.
Does NOT load the 22B checkpoint — uses randomly-initialised weights.
"""
from __future__ import annotations
import time, math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda:0")
DTYPE  = torch.bfloat16

# LTX-2.3 video self-attention dims
INNER  = 4096
HEADS  = 32
D_HEAD = 128   # INNER / HEADS
N_LAYERS = 48

SEQ_LENGTHS = {
    "25fr @768×448 (current)": 1344,
    "97fr @544×544 (scrya-97)": 3757,
    "145fr@544×544 (scrya-full)": 5491,
}


def make_ripple(layer_idx: int) -> "RippleVideoAttention":
    from ltx_core.model.transformer.ripple_attention import RippleVideoAttention
    return RippleVideoAttention(
        query_dim=INNER, heads=HEADS, dim_head=D_HEAD,
        layer_idx=layer_idx, n_layers=N_LAYERS,
    ).to(DEVICE).to(DTYPE)


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ── 1. Print shift schedule for all 48 layers ─────────────────────────────────
print("=" * 70)
print("RippleVideoAttention — shift schedule (48 layers)")
print("=" * 70)
from ltx_core.model.transformer.ripple_attention import _ripple_shifts_for_layer, _NUM_LEVELS

for i in range(N_LAYERS):
    shifts = _ripple_shifts_for_layer(i, N_LAYERS)
    level  = i % _NUM_LEVELS
    stride = 1 << level
    marker = " ◀" if i % _NUM_LEVELS == 0 else ""
    print(f"  Layer {i:2d}  level={level:2d}  stride=±{stride:>5}  K={len(shifts):2d}  {sorted(shifts)}{marker}")

avg_k = sum(len(_ripple_shifts_for_layer(i, N_LAYERS)) for i in range(N_LAYERS)) / N_LAYERS
print(f"\n  Average K/layer: {avg_k:.1f}")
print()


# ── 2. Single layer forward: inspect output ───────────────────────────────────
print("=" * 70)
print("Single forward pass — shape / NaN / value range / timing")
print("=" * 70)

for label, S in SEQ_LENGTHS.items():
    print(f"\n  {label}  S={S}")
    x = torch.randn(1, S, INNER, device=DEVICE, dtype=DTYPE) * 0.02

    # Standard SDPA baseline
    from ltx_core.model.transformer.attention import Attention
    from ltx_core.model.transformer.rope import LTXRopeType
    attn_std = Attention(query_dim=INNER, heads=HEADS, dim_head=D_HEAD).to(DEVICE).to(DTYPE)

    with torch.no_grad():
        t_std = bench(lambda: attn_std(x))

    # Test every 12th layer (one per level)
    for layer_idx in [0, 6, 11, 23]:
        ripple = make_ripple(layer_idx)
        shifts = ripple.shifts
        stride = 1 << (layer_idx % _NUM_LEVELS)

        with torch.no_grad():
            try:
                out = ripple(x)
                nan_count = torch.isnan(out).sum().item()
                inf_count = torch.isinf(out).sum().item()
                t_ms = bench(lambda: ripple(x))
                triton_tag = "triton" if ripple._triton_available else "python"

                print(f"    layer={layer_idx:2d} stride=±{stride:>5}  K={len(shifts):2d} "
                      f"[{triton_tag}]  "
                      f"out={tuple(out.shape)}  "
                      f"min={out.float().min():.3f}  max={out.float().max():.3f}  "
                      f"NaN={nan_count}  Inf={inf_count}  "
                      f"{t_ms:.2f}ms  (SDPA={t_std:.2f}ms  ratio={t_ms/t_std:.2f}x)")
            except Exception as e:
                print(f"    layer={layer_idx:2d}  FAILED: {e}")

    del x
    torch.cuda.empty_cache()


# ── 3. Full 48-block ripple sweep: one forward per block ──────────────────────
print()
print("=" * 70)
print("48-block sweep — total fwd time at each resolution")
print("=" * 70)

for label, S in SEQ_LENGTHS.items():
    x = torch.randn(1, S, INNER, device=DEVICE, dtype=DTYPE) * 0.02
    blocks = [make_ripple(i) for i in range(N_LAYERS)]

    def fwd_all():
        h = x
        for b in blocks:
            h = b(h)
        return h

    with torch.no_grad():
        out = fwd_all()   # warmup + NaN check
        nan_total = torch.isnan(out).sum().item()
        t_ms = bench(lambda: fwd_all())

    # Compare vs 48 × SDPA
    attn_std = Attention(query_dim=INNER, heads=HEADS, dim_head=D_HEAD).to(DEVICE).to(DTYPE)
    with torch.no_grad():
        t_sdpa = bench(lambda: attn_std(x)) * N_LAYERS

    mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    print(f"  {label:<30}  ripple={t_ms:.1f}ms  sdpa×48≈{t_sdpa:.1f}ms  "
          f"ratio={t_ms/t_sdpa:.2f}x  NaN={nan_total}  peak_mem={mem_mb:.0f}MB")

    del x, blocks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

print()
print("Done.")
