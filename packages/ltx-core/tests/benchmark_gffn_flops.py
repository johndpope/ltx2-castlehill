#!/usr/bin/env python
"""
FLOPs comparison: Standard FFN vs gFFN variants for LTX2.

Computes per-token forward FLOPs analytically and validates with
torch profiler where available.

Assumptions:
  - dim = 4096 (LTX2 video), audio_dim = 2048
  - B=1, L=1 for per-token comparison (scale by B*L for full batch)
  - nn.Linear(in, out) = 2*in*out + out FLOPs (matmul + bias)
  - Elementwise mul/sub/add = 1 FLOP per element
  - SiLU(x) = x * sigmoid(x) ≈ 5 FLOPs per element (1 neg + 1 exp + 1 add + 1 div + 1 mul)
  - GELU(x) ≈ 8 FLOPs per element (tanh approx variant)
  - torch.roll = 0 FLOPs (pure memory permutation)
  - torch.cat = 0 FLOPs (memory concat)
  - torch.mean(dim=1) = (L-1) * dim additions → ~0 for L=1
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FLOPCount:
    """Structured FLOP count with breakdown."""
    name: str
    linear_flops: int = 0       # From nn.Linear matmuls
    elementwise_flops: int = 0  # From mul, sub, add, silu, gelu
    total_flops: int = 0
    params: int = 0

    def __post_init__(self):
        self.total_flops = self.linear_flops + self.elementwise_flops


def count_linear_flops(in_features: int, out_features: int, has_bias: bool = True) -> int:
    """FLOPs for one nn.Linear forward pass on a single token.

    matmul: 2 * in * out (multiply-accumulate counted as 2 ops)
    bias:   out (addition)
    """
    flops = 2 * in_features * out_features
    if has_bias:
        flops += out_features
    return flops


def count_silu_flops(num_elements: int) -> int:
    """SiLU(x) = x * σ(x): neg + exp + add + div + mul = 5 FLOPs/elem."""
    return 5 * num_elements


def count_gelu_tanh_flops(num_elements: int) -> int:
    """GELU(x) with tanh approx: ~8 FLOPs/elem (mul, pow, tanh chain)."""
    return 8 * num_elements


# ============================================================================
# Standard FeedForward: dim -> 4*dim -> dim
# ============================================================================

def flops_standard_ffn(dim: int) -> FLOPCount:
    """
    Architecture:
      GELUApprox(dim, inner_dim=4*dim):
        Linear(dim, 4*dim) + GELU
      Identity()
      Linear(4*dim, dim)
    """
    inner_dim = 4 * dim

    # Up projection: Linear(dim, inner_dim)
    up_linear = count_linear_flops(dim, inner_dim)

    # GELU activation on inner_dim elements
    gelu = count_gelu_tanh_flops(inner_dim)

    # Down projection: Linear(inner_dim, dim)
    down_linear = count_linear_flops(inner_dim, dim)

    linear_total = up_linear + down_linear
    elem_total = gelu

    params = (dim * inner_dim + inner_dim) + (inner_dim * dim + dim)

    return FLOPCount(
        name=f"Standard FFN (dim={dim}, mult=4)",
        linear_flops=linear_total,
        elementwise_flops=elem_total,
        params=params,
    )


# ============================================================================
# gFFN-Global variants
# ============================================================================

def flops_gffn_global(dim: int, num_shifts: int, mode: str) -> FLOPCount:
    """
    Architecture (per token, B=1, L=1):
      1. Global avg pool: mean(x, dim=1) → 0 FLOPs for L=1
      2. SiLU(x) → 5*dim FLOPs
      3. For each shift s:
         - torch.roll(glo, s) → 0 FLOPs
         IF inner:
           - x * rg → dim muls
           - SiLU(result) → 5*dim FLOPs
           Total: 6*dim per shift
         IF wedge:
           - torch.roll(x, s) → 0 FLOPs
           - x * rg → dim muls
           - rx * glo → dim muls
           - subtract → dim subs
           Total: 3*dim per shift
      4. torch.cat → 0 FLOPs
      5. proj_out: Linear(cat_dim, dim)

    cat_dim depends on mode:
      full:  dim * (1 + 2*shifts)
      inner: dim * (1 + shifts)
      wedge: dim * (1 + shifts)
    """
    # Step 2: SiLU on x
    elem_flops = count_silu_flops(dim)

    # Step 3: per-shift operations
    for _ in range(num_shifts):
        if mode in ("full", "inner"):
            elem_flops += dim  # x * rg (mul)
            elem_flops += count_silu_flops(dim)  # SiLU(inner)
        if mode in ("full", "wedge"):
            elem_flops += dim  # x * rg (mul)
            elem_flops += dim  # rx * glo (mul)
            elem_flops += dim  # subtraction

    # Step 5: output projection
    if mode == "full":
        cat_dim = dim * (1 + 2 * num_shifts)
    else:
        cat_dim = dim * (1 + num_shifts)

    proj_linear = count_linear_flops(cat_dim, dim)
    params = cat_dim * dim + dim  # proj_out weights + bias

    return FLOPCount(
        name=f"gFFN-Global (shifts={num_shifts}, mode={mode})",
        linear_flops=proj_linear,
        elementwise_flops=elem_flops,
        params=params,
    )


# ============================================================================
# gFFN-Hybrid (local + global)
# ============================================================================

def flops_gffn_hybrid(
    dim: int, num_shifts: int, mode: str, kernel_size: int = 7, seq_len: int = 1
) -> FLOPCount:
    """
    Additional over gFFN-Global:
      - Two 1D depthwise convolutions (groups=dim): Conv1d(dim, dim, k, groups=dim)
        FLOPs per conv = seq_len * dim * k (depthwise = k muls + k-1 adds per output elem)
      - SiLU between convs
      - Optional subtraction for differential mode
      - Two geometric interaction streams (local + global) instead of one
      - Larger proj_out: Linear(2 * features_per_stream, dim)
    """
    # Local context: 2x DWConv1d
    # DWConv: each output = k multiply-adds = 2*k FLOPs per element
    conv_flops = 2 * (seq_len * dim * 2 * kernel_size)  # 2 convolutions
    silu_between = count_silu_flops(dim * seq_len)
    diff_sub = dim * seq_len  # subtraction for differential mode

    # Two streams of geometric interaction (same as gFFN-Global each)
    per_stream_elem = count_silu_flops(dim)  # SiLU(x)
    for _ in range(num_shifts):
        if mode in ("full", "inner"):
            per_stream_elem += dim + count_silu_flops(dim)
        if mode in ("full", "wedge"):
            per_stream_elem += 3 * dim

    total_elem = conv_flops + silu_between + diff_sub + 2 * per_stream_elem

    # Output projection (2x wider input)
    if mode == "full":
        features_per_stream = dim * (1 + 2 * num_shifts)
    else:
        features_per_stream = dim * (1 + num_shifts)
    cat_dim = 2 * features_per_stream
    proj_linear = count_linear_flops(cat_dim, dim)

    # Params: 2 DWConv + proj_out
    conv_params = 2 * (dim * kernel_size)  # no bias assumed for simplicity
    proj_params = cat_dim * dim + dim
    total_params = conv_params + proj_params

    return FLOPCount(
        name=f"gFFN-Hybrid (shifts={num_shifts}, mode={mode}, k={kernel_size})",
        linear_flops=proj_linear,
        elementwise_flops=total_elem,
        params=total_params,
    )


# ============================================================================
# gFFN-Global HRR: project → superimpose → project
# ============================================================================

def flops_gffn_hrr(
    dim: int, num_shifts: int, mode: str, proj_factor: int = 8
) -> FLOPCount:
    """
    Architecture (per token, B=1, L=1):
      1. Global avg pool → 0 FLOPs for L=1
      2. Compute geometric terms (same as vanilla gFFN-Global):
         SiLU(x) + shifted inner/wedge products
      3. For each of num_terms terms:
         - proj: Linear(dim, dim//proj_factor) → projects to shared subspace
      4. Sum all projected terms (HRR superposition)
      5. Optional: div by num_terms, unit norm
      6. out: Linear(dim//proj_factor, dim) → final projection

    The key insight: vanilla gFFN needs Linear(9*dim, dim) = 9*dim^2 FLOPs.
    HRR needs num_terms * Linear(dim, dim/pf) + Linear(dim/pf, dim)
          = num_terms * 2*dim*(dim/pf) + 2*(dim/pf)*dim
          = (num_terms + 1) * 2*dim^2/pf
    For pf=8, num_terms=5: (5+1) * 2*dim^2/8 = 1.5*dim^2 vs 10*dim^2 → 6.7x cheaper
    """
    proj_dim = dim // proj_factor

    # Count terms
    if mode == "full":
        num_terms = 1 + 2 * num_shifts
    else:  # inner or wedge
        num_terms = 1 + num_shifts

    # Elementwise ops (same as vanilla gFFN-Global)
    elem_flops = count_silu_flops(dim)  # SiLU(x) base term
    for _ in range(num_shifts):
        if mode in ("full", "inner"):
            elem_flops += dim + count_silu_flops(dim)  # mul + silu
        if mode in ("full", "wedge"):
            elem_flops += 3 * dim  # 2 muls + 1 sub

    # Per-term projection: num_terms × Linear(dim, proj_dim)
    proj_flops = num_terms * count_linear_flops(dim, proj_dim)

    # Superposition: (num_terms - 1) additions of [proj_dim] vectors
    elem_flops += (num_terms - 1) * proj_dim

    # Division by num_terms: proj_dim divs
    elem_flops += proj_dim

    # Unit norm: proj_dim muls (square) + proj_dim-1 adds (sum) + 1 sqrt + proj_dim divs
    elem_flops += 3 * proj_dim + 1

    # Post-scale: proj_dim muls (actually dim_out, but same here)
    elem_flops += dim

    # Output projection: Linear(proj_dim, dim)
    out_flops = count_linear_flops(proj_dim, dim)

    linear_total = proj_flops + out_flops

    # Params: proj(dim, proj_dim) + out(proj_dim, dim) + post_scale(dim)
    params = (dim * proj_dim + proj_dim) + (proj_dim * dim + dim) + dim

    return FLOPCount(
        name=f"gFFN-HRR (shifts={num_shifts}, mode={mode}, pf={proj_factor})",
        linear_flops=linear_total,
        elementwise_flops=elem_flops,
        params=params,
    )


# ============================================================================
# Print results
# ============================================================================

def format_flops(flops: int) -> str:
    if flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.1f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.1f}K"
    return str(flops)


def format_params(params: int) -> str:
    if params >= 1e6:
        return f"{params/1e6:.1f}M"
    elif params >= 1e3:
        return f"{params/1e3:.1f}K"
    return str(params)


def run_comparison(dim: int, label: str = ""):
    print(f"\n{'='*80}")
    print(f" FLOPs Comparison — {label} (dim={dim}, per-token, B=1 L=1)")
    print(f"{'='*80}")

    baseline = flops_standard_ffn(dim)

    variants = [
        baseline,
        flops_gffn_global(dim, num_shifts=2, mode="full"),
        flops_gffn_global(dim, num_shifts=2, mode="inner"),
        flops_gffn_global(dim, num_shifts=4, mode="full"),
        flops_gffn_global(dim, num_shifts=4, mode="inner"),
        flops_gffn_global(dim, num_shifts=4, mode="wedge"),
        flops_gffn_hybrid(dim, num_shifts=4, mode="full"),
        flops_gffn_hybrid(dim, num_shifts=4, mode="inner"),
        flops_gffn_hrr(dim, num_shifts=4, mode="inner", proj_factor=4),
        flops_gffn_hrr(dim, num_shifts=4, mode="inner", proj_factor=8),
        flops_gffn_hrr(dim, num_shifts=4, mode="inner", proj_factor=16),
        flops_gffn_hrr(dim, num_shifts=4, mode="full", proj_factor=8),
    ]

    print(f"\n{'Variant':<50} {'Linear':>10} {'Elemwise':>10} {'Total':>10} {'vs FFN':>8} {'Params':>10}")
    print("-" * 100)

    for v in variants:
        ratio = v.total_flops / baseline.total_flops
        marker = " ◀ baseline" if v.name == baseline.name else ""
        print(
            f"{v.name:<50} "
            f"{format_flops(v.linear_flops):>10} "
            f"{format_flops(v.elementwise_flops):>10} "
            f"{format_flops(v.total_flops):>10} "
            f"{ratio:>7.2f}x "
            f"{format_params(v.params):>10}"
            f"{marker}"
        )

    # Per-model totals (48 layers video + 48 layers audio)
    print(f"\n--- Full Model Estimates (48 video layers + 48 audio layers) ---")
    audio_dim = dim // 2  # 2048 for audio when video is 4096

    for name, vfn in [
        ("Standard FFN", lambda d: flops_standard_ffn(d)),
        ("gFFN-G (4 shifts, full)", lambda d: flops_gffn_global(d, 4, "full")),
        ("gFFN-G (4 shifts, inner)", lambda d: flops_gffn_global(d, 4, "inner")),
        ("gFFN-G (2 shifts, full)", lambda d: flops_gffn_global(d, 2, "full")),
        ("gFFN-HRR (4s, inner, pf=8)", lambda d: flops_gffn_hrr(d, 4, "inner", 8)),
        ("gFFN-HRR (4s, inner, pf=4)", lambda d: flops_gffn_hrr(d, 4, "inner", 4)),
    ]:
        v_flops = vfn(dim).total_flops * 48
        a_flops = vfn(audio_dim).total_flops * 48
        total = v_flops + a_flops
        v_params = vfn(dim).params * 48
        a_params = vfn(audio_dim).params * 48
        base_total = flops_standard_ffn(dim).total_flops * 48 + flops_standard_ffn(audio_dim).total_flops * 48
        print(
            f"  {name:<35} "
            f"Video: {format_flops(v_flops):>8}  "
            f"Audio: {format_flops(a_flops):>8}  "
            f"Total: {format_flops(total):>8}  "
            f"vs FFN: {total/base_total:.2f}x  "
            f"Params: {format_params(v_params + a_params):>10}"
        )


def validate_with_torch_profiler():
    """Cross-validate analytical FLOPs with torch profiler."""
    from ltx_core.model.transformer.feed_forward import FeedForward
    from ltx_core.model.transformer.gffn import gFFNGlobal, gFFNGlobalHRR

    dim = 4096
    x = torch.randn(1, 1, dim)

    models = [
        ("Standard FFN", FeedForward(dim, dim_out=dim)),
        ("gFFN-G(4,full)", gFFNGlobal(dim, num_shifts=4, mode="full")),
        ("gFFN-G(4,inner)", gFFNGlobal(dim, num_shifts=4, mode="inner")),
        ("gFFN-HRR(4,inner,pf=8)", gFFNGlobalHRR(dim, num_shifts=4, mode="inner", proj_factor=8)),
        ("gFFN-HRR(4,inner,pf=4)", gFFNGlobalHRR(dim, num_shifts=4, mode="inner", proj_factor=4)),
    ]

    print(f"\n{'='*80}")
    print(f" Torch Profiler Validation (dim={dim})")
    print(f"{'='*80}")

    for name, model in models:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
        ) as prof:
            model(x)

        total_flops = sum(
            e.flops for e in prof.key_averages() if e.flops > 0
        )
        print(f"  {name}: {format_flops(total_flops)} (profiler)")


if __name__ == "__main__":
    run_comparison(4096, label="LTX2 Video")
    run_comparison(2048, label="LTX2 Audio")

    print("\n")
    try:
        validate_with_torch_profiler()
    except Exception as e:
        print(f"  Profiler validation skipped: {e}")
