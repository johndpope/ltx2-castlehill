#!/usr/bin/env python3
"""Attention FLOP/latency baseline: full SDPA vs SmallWorld vs Ripple.

LTX-2.3 dimensions (from checkpoint):
  Video self-attn: inner_dim=4096, heads=32, dim_head=128
  Audio self-attn: inner_dim=2048, heads=16, dim_head=128

Sequence lengths tested:
  S=1344  — 768x448, 25 frames  (lat 24x14x4,  our current overfit-10 config)
  S=3757  — 544x544, 97 frames  (lat 17x17x13, reduced scrya)
  S=5491  — 544x544, 145 frames (lat 17x17x19, full scrya)

Variants:
  [1] Full SDPA          — baseline (what LTX-2.3 uses today)
  [2] SmallWorld Python  — ltx-core smallworld_attention.py (32 shifts, Python loop)
  [3] Ripple Python      — hierarchical doubling strides, Python loop
  [4] Ripple Triton      — Triton fused kernel from GRA-hybrid (if available)
"""
from __future__ import annotations
import sys, time, math
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda:0")
DTYPE  = torch.bfloat16

# ── LTX-2.3 actual dims ───────────────────────────────────────────────────────
VIDEO_INNER = 4096
AUDIO_INNER = 2048
N_HEADS     = 32   # video (4096 / 128 = 32)
DIM_HEAD    = 128
N_LAYERS    = 48

SEQ_LENGTHS = {
    "25fr@768x448  (current)": 1344,
    "97fr@544x544  (scrya-97)": 3757,
    "145fr@544x544 (scrya-full)": 5491,
}

# ── FLOP formulas ─────────────────────────────────────────────────────────────

def flops_full_attn(S: int, H: int = N_HEADS, D: int = DIM_HEAD) -> int:
    """4 * B * H * S^2 * D  (QK matmul + AV matmul, B=1)."""
    return 4 * H * S * S * D

def flops_sparse_attn(S: int, K: float, H: int = N_HEADS, D: int = DIM_HEAD) -> int:
    """4 * H * S * K * D  (K attended neighbors per token)."""
    return int(4 * H * S * K * D)

def flops_qkv_proj(S: int, inner: int = VIDEO_INNER) -> int:
    """6 * S * inner^2  (Q + K + V projections, no bias)."""
    return 6 * S * inner * inner

def flops_out_proj(S: int, inner: int = VIDEO_INNER) -> int:
    """2 * S * inner^2  (output projection)."""
    return 2 * S * inner * inner

def fmt_gflops(n: int) -> str:
    return f"{n/1e9:8.1f} GF"

# ── Ripple shift patterns (from model_ripple.py) ──────────────────────────────

def ripple_shifts_for_layer(layer_idx: int, n_layers: int = N_LAYERS,
                             max_len: int = 8192, local_radius: int = 1) -> list[int]:
    shifts = [0]
    for r in range(1, local_radius + 1):
        shifts.extend([r, -r])
    stride = 1 << layer_idx
    if stride <= max_len // 2:
        if stride > local_radius:
            shifts.extend([stride, -stride])
        half_stride = stride // 2
        if half_stride > local_radius and half_stride not in shifts:
            shifts.extend([half_stride, -half_stride])
    if layer_idx >= n_layers - 2:
        for extra in range(layer_idx + 1, min(layer_idx + 4, 16)):
            s = 1 << extra
            if s <= max_len // 2 and s not in shifts and -s not in shifts:
                shifts.extend([s, -s])
    return shifts

# ── Timing helpers ────────────────────────────────────────────────────────────

def bench(fn, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms

# ── Attention implementations ─────────────────────────────────────────────────

def run_full_sdpa(q, k, v):
    # q/k/v: [B, H, S, D]
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)

def run_smallworld_python(q, k, v, shifts: list[int]):
    """Python loop over K shifts — same as ltx-core SmallWorldAttention._smallworld_attention."""
    B, H, S, D = q.shape
    scale = D ** -0.5
    # q/k/v: [B, H, S, D] → reshape to [B, S, H, D] for gather
    q_ = q.permute(0, 2, 1, 3)  # [B, S, H, D]
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)

    positions = torch.arange(S, device=q.device)
    all_scores, all_vals = [], []
    for s in shifts:
        nb_idx = (positions + s) % S
        k_nb = k_[:, nb_idx]  # [B, S, H, D]
        v_nb = v_[:, nb_idx]
        score = (q_ * k_nb).sum(dim=-1) * scale  # [B, S, H]
        all_scores.append(score)
        all_vals.append(v_nb)

    scores = torch.stack(all_scores, dim=-1)  # [B, S, H, K]
    attn   = F.softmax(scores, dim=-1)
    v_stack = torch.stack(all_vals, dim=-1)  # [B, S, H, D, K]
    out = (attn.unsqueeze(-2) * v_stack).sum(dim=-1)  # [B, S, H, D]
    return out.permute(0, 2, 1, 3)  # [B, H, S, D]

def run_ripple_python(q, k, v, layer_idx: int = 6):
    shifts = ripple_shifts_for_layer(layer_idx)
    return run_smallworld_python(q, k, v, shifts)

def try_triton_kernel(q, k, v, shifts: list[int]):
    """Try the fused Triton kernel from GRA-hybrid."""
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/GRA-hybrid")
    try:
        from triton_smallworld import smallworld_attention
        shifts_t = torch.tensor(shifts, dtype=torch.int32, device=q.device)
        edge_bias = torch.zeros(q.shape[1], len(shifts), device=q.device, dtype=torch.float32)
        scale = q.shape[-1] ** -0.5
        result = smallworld_attention(q, k, v, shifts_t, edge_bias, scale)
        return result, None
    except Exception as e:
        return None, str(e)

# ── Main benchmark ─────────────────────────────────────────────────────────────

def main():
    print(f"\nLTX-2.3 Attention FLOP + Latency Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Dtype: {DTYPE}  |  Heads={N_HEADS}  DimHead={DIM_HEAD}  Layers={N_LAYERS}")
    print(f"Note: FLOPs = attention ops only (QKV+out projections are identical across all variants)\n")

    # SmallWorld uses 32 fixed log-spaced shifts (from ltx-core)
    SW_SHIFTS = 32

    # Ripple avg shifts per layer
    ripple_shift_counts = [len(ripple_shifts_for_layer(i)) for i in range(N_LAYERS)]
    ripple_avg = sum(ripple_shift_counts) / N_LAYERS

    print(f"Ripple shift distribution across {N_LAYERS} layers:")
    for i in range(0, N_LAYERS, 6):
        shifts = ripple_shifts_for_layer(i)
        print(f"  Layer {i:2d}: stride=±{1<<i if (1<<i)<=8192 else '∞':>5}  K={len(shifts):2d}  {sorted(shifts)}")
    print(f"  Average K per layer: {ripple_avg:.1f}\n")

    header = f"{'Config':<28} {'Variant':<22} {'Attn FLOPs':>12} {'QKV+Out FLOPs':>14} {'Attn%':>6} {'Time(ms)':>9} {'vs SDPA':>8}"
    print(header)
    print("-" * len(header))

    for label, S in SEQ_LENGTHS.items():
        q = torch.randn(1, N_HEADS, S, DIM_HEAD, device=DEVICE, dtype=DTYPE)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        qkv_flops = flops_qkv_proj(S) + flops_out_proj(S)

        results = {}

        # 1. Full SDPA
        f_full = flops_full_attn(S)
        t_full = bench(lambda: run_full_sdpa(q, k, v))
        attn_pct = f_full / (f_full + qkv_flops) * 100
        results["SDPA (baseline)"] = (f_full, t_full)
        print(f"{label:<28} {'SDPA (baseline)':<22} {fmt_gflops(f_full)} {fmt_gflops(qkv_flops)} {attn_pct:5.1f}% {t_full:8.2f}ms {'1.00x':>8}")

        # 2. SmallWorld Python (32 shifts, flat)
        sw_shifts_list = list(range(-16, 0)) + [0] + list(range(1, 17))  # placeholder uniform; real is log-spaced
        # Use actual log-spaced shifts from clifford pattern
        import math
        log_shifts = [0]
        r = 1
        while r <= S // 2 and len(log_shifts) < SW_SHIFTS // 2:
            log_shifts.extend([r, -r])
            r = max(r + 1, int(r * 1.5))
        sw_shifts_actual = log_shifts[:SW_SHIFTS] if len(log_shifts) >= SW_SHIFTS else log_shifts
        K_sw = len(sw_shifts_actual)

        f_sw = flops_sparse_attn(S, K_sw)
        t_sw = bench(lambda: run_smallworld_python(q, k, v, sw_shifts_actual))
        print(f"{'':28} {'SmallWorld Python (K='+str(K_sw)+')':<22} {fmt_gflops(f_sw)} {'(same)':>14} {f_sw/(f_full)*100:5.1f}% {t_sw:8.2f}ms {t_sw/t_full:7.2f}x")
        results["SmallWorld Python"] = (f_sw, t_sw)

        # 3. Ripple Python (avg K=7.5 shifts, hierarchical)
        # Use mid-layer shifts as representative
        mid_shifts = ripple_shifts_for_layer(N_LAYERS // 2)
        f_ripple = flops_sparse_attn(S, ripple_avg)
        t_ripple = bench(lambda: run_ripple_python(q, k, v, N_LAYERS // 2))
        print(f"{'':28} {'Ripple Python (K≈'+f'{ripple_avg:.1f}'+')':<22} {fmt_gflops(f_ripple)} {'(same)':>14} {f_ripple/(f_full)*100:5.1f}% {t_ripple:8.2f}ms {t_ripple/t_full:7.2f}x")
        results["Ripple Python"] = (f_ripple, t_ripple)

        # 4. Ripple + Triton fused kernel
        triton_result = try_triton_kernel(q, k, v, mid_shifts)
        if isinstance(triton_result, tuple):
            out_t, err = triton_result
        else:
            out_t, err = triton_result, None
        if out_t is not None and err is None:
            t_triton = bench(lambda: try_triton_kernel(q, k, v, mid_shifts))
            print(f"{'':28} {'Ripple Triton (K≈'+f'{ripple_avg:.1f}'+')':<22} {fmt_gflops(f_ripple)} {'(same)':>14} {f_ripple/(f_full)*100:5.1f}% {t_triton:8.2f}ms {t_triton/t_full:7.2f}x")
        else:
            print(f"{'':28} {'Ripple Triton':<22} {'N/A — '+(str(err) if err else str(out_t))[:50]}")

        print()
        del q, k, v
        torch.cuda.empty_cache()

    # ── Per-block breakdown (48 layers × video+audio) ─────────────────────────
    print("\n── Estimated per-step cost (48 layers, video only, fwd+bwd≈3×fwd) ──")
    for label, S in SEQ_LENGTHS.items():
        f_full   = flops_full_attn(S) * N_LAYERS * 3
        f_ripple = flops_sparse_attn(S, ripple_avg) * N_LAYERS * 3
        f_qkv    = (flops_qkv_proj(S) + flops_out_proj(S)) * N_LAYERS * 3
        total_full   = f_full + f_qkv
        total_ripple = f_ripple + f_qkv
        saving_pct   = (1 - total_ripple / total_full) * 100
        print(f"  {label:<30}  full={total_full/1e12:.2f}TF  ripple={total_ripple/1e12:.2f}TF  saving={saving_pct:.1f}%")


if __name__ == "__main__":
    main()
