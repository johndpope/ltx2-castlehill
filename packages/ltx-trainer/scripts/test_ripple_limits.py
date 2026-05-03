#!/usr/bin/env python3
"""
Push-limits validation for RippleVideoAttention with and without PlanarQuant compression.

Tests:
  1. Max sequence length before OOM for SDPA / Ripple-none / Ripple-planar
  2. 48-block sweep timing + peak memory at each length
  3. Output quality: relative error between none and planar modes

LTX-2.3 dims: inner=4096, H=32, D=128, 48 layers
"""
from __future__ import annotations
import sys, time, gc, math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/johndpope/Documents/GitHub/rotorquant")

DEVICE  = torch.device("cuda:0")
DTYPE   = torch.bfloat16
INNER   = 4096
HEADS   = 32
D_HEAD  = 128
N_LAYERS = 48

# Sequence lengths to probe — 25fr → 400fr+ equivalent at scrya resolution
SEQ_PROBE = {
    "25fr @768×448":   1344,
    "97fr @544×544":   3757,
    "145fr@544×544":   5491,
    "215fr@544×544":   8192,
    "323fr@544×544":  12288,
    "430fr@544×544":  16384,
}


def make_ripple(layer_idx: int, compress: str = "none") -> "RippleVideoAttention":
    from ltx_core.model.transformer.ripple_attention import RippleVideoAttention
    return RippleVideoAttention(
        query_dim=INNER, heads=HEADS, dim_head=D_HEAD,
        layer_idx=layer_idx, n_layers=N_LAYERS,
        kv_compress_mode=compress, kv_compress_bits=4,
    ).to(DEVICE).to(DTYPE)


def bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def mem_mb():
    return torch.cuda.max_memory_allocated() / 1e6


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── 1. Single-block max sequence probe ────────────────────────────────────────
print("=" * 72)
print("1. Max sequence probe — single block (layer=0), find OOM boundary")
print("=" * 72)

probe_lengths = [1344, 3757, 5491, 8192, 12288, 16384, 24576, 32768]

def probe_sdpa(S):
    q = torch.randn(1, HEADS, S, D_HEAD, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.sum().item()

def probe_ripple(S, compress="none"):
    m = make_ripple(0, compress)
    x = torch.randn(1, S, INNER, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out = m(x)
    return out.sum().item(), m

sdpa_oom = ripple_none_oom = ripple_planar_oom = False

for S in probe_lengths:
    label = f"S={S:>6}"
    reset_mem()

    # SDPA
    if not sdpa_oom:
        try:
            probe_sdpa(S)
            sdpa_tag = f"ok ({mem_mb():.0f}MB)"
        except torch.cuda.OutOfMemoryError:
            sdpa_oom = True
            sdpa_tag = "OOM"
    else:
        sdpa_tag = "OOM"
    reset_mem()

    # Ripple no-compress
    if not ripple_none_oom:
        try:
            probe_ripple(S, "none")
            rn_tag = f"ok ({mem_mb():.0f}MB)"
        except torch.cuda.OutOfMemoryError:
            ripple_none_oom = True
            rn_tag = "OOM"
    else:
        rn_tag = "OOM"
    reset_mem()

    # Ripple + PlanarQuant
    if not ripple_planar_oom:
        try:
            probe_ripple(S, "planar")
            rp_tag = f"ok ({mem_mb():.0f}MB)"
        except torch.cuda.OutOfMemoryError:
            ripple_planar_oom = True
            rp_tag = "OOM"
    else:
        rp_tag = "OOM"
    reset_mem()

    print(f"  {label}  SDPA={sdpa_tag:<18}  ripple-none={rn_tag:<18}  ripple-planar={rp_tag}")


# ── 2. 48-block sweep: time + peak memory ─────────────────────────────────────
print()
print("=" * 72)
print("2. 48-block sweep — timing + peak memory (no_grad)")
print("=" * 72)
print(f"  {'Label':<26} {'Mode':<14} {'Time(ms)':>9} {'Peak(MB)':>9} {'NaN':>5}")
print("-" * 72)

for label, S in SEQ_PROBE.items():
    reset_mem()
    x = torch.randn(1, S, INNER, device=DEVICE, dtype=DTYPE)

    for compress in ["none", "planar"]:
        try:
            blocks = [make_ripple(i, compress) for i in range(N_LAYERS)]

            def fwd_all():
                h = x
                for b in blocks:
                    h = b(h)
                return h

            with torch.no_grad():
                out = fwd_all()
                nan_n = torch.isnan(out).sum().item()
                t_ms = bench(fwd_all)
                peak = mem_mb()

            print(f"  {label:<26} {compress:<14} {t_ms:9.1f} {peak:9.0f} {nan_n:5}")

        except torch.cuda.OutOfMemoryError:
            print(f"  {label:<26} {compress:<14}      OOM")
        finally:
            reset_mem()


# ── 3. Compression quality: relative error none vs planar ─────────────────────
print()
print("=" * 72)
print("3. Compression quality — none vs planar at different bit-widths (S=5491)")
print("   (tests at layer=6, stride=±64, K=7; weights copied, only quant differs)")
print("=" * 72)
print(f"  {'bits':>5}  {'n_levels':>8}  {'rel_err':>8}  {'cosine_sim':>11}  {'overhead_ms':>12}")
print("-" * 72)

S_quality = 5491
x_q = torch.randn(1, S_quality, INNER, device=DEVICE, dtype=DTYPE)
m_ref = make_ripple(6, "none")

def bench_single(m, x, warmup=2, iters=5):
    with torch.no_grad():
        for _ in range(warmup): m(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters): m(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

t_ref = bench_single(m_ref, x_q)
with torch.no_grad():
    out_ref = m_ref(x_q).float()

for bits in [2, 3, 4]:
    try:
        from ltx_core.model.transformer.ripple_attention import RippleVideoAttention
        m_c = RippleVideoAttention(
            query_dim=INNER, heads=HEADS, dim_head=D_HEAD,
            layer_idx=6, n_layers=N_LAYERS,
            kv_compress_mode="planar", kv_compress_bits=bits,
        ).to(DEVICE).to(DTYPE)
        m_c.load_state_dict(m_ref.state_dict(), strict=False)

        with torch.no_grad():
            out_c = m_c(x_q).float()

        diff = (out_c - out_ref).norm() / (out_ref.norm() + 1e-8)
        cos  = F.cosine_similarity(out_ref.reshape(1,-1), out_c.reshape(1,-1)).item()
        t_c  = bench_single(m_c, x_q)
        n_lv = 2 ** bits
        print(f"  {bits:>5}  {n_lv:>8}  {diff.item():8.4f}  {cos:11.6f}  +{t_c-t_ref:8.2f}ms")
    except Exception as e:
        print(f"  {bits:>5}  error: {e}")
    finally:
        reset_mem()

print()
print("  (none)            ref   0.0000     1.000000          0.00ms")

# ── Absolute quality at S=145fr ───────────────────────────────────────────────
print()
print("=" * 72)
print("4. Output quality at each sequence length — 4-bit (production setting)")
print("=" * 72)
print(f"  {'Label':<26} {'S':>6}  {'rel_err':>8}  {'cosine_sim':>11}  {'NaN':>5}")
print("-" * 72)

for label, S in SEQ_PROBE.items():
    reset_mem()
    if S > 16384:
        print(f"  {label:<26} {S:>6}  (skipped)")
        continue

    x = torch.randn(1, S, INNER, device=DEVICE, dtype=DTYPE)
    try:
        m_n = make_ripple(6, "none")
        m_p = make_ripple(6, "planar")
        m_p.load_state_dict(m_n.state_dict(), strict=False)

        with torch.no_grad():
            out_n = m_n(x).float()
            out_p = m_p(x).float()

        diff = (out_p - out_n).norm() / (out_n.norm() + 1e-8)
        cos  = F.cosine_similarity(out_n.reshape(1,-1), out_p.reshape(1,-1)).item()
        nan_n = torch.isnan(out_p).sum().item()
        print(f"  {label:<26} {S:>6}  {diff.item():8.4f}  {cos:11.6f}  {nan_n:5}")
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:<26} {S:>6}  OOM")
    finally:
        reset_mem()


print()
print("Done.")
