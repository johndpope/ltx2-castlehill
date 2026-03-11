#!/usr/bin/env python
"""
Analyze shift coverage across channel dimension for different num_shifts.

Key question: CliffordNet used 2-5 shifts on dim=64-128 (CIFAR-100).
At dim=4096 (LTX2), how many shifts do we actually need?

The shifts define which "diagonals" of the D×D channel interaction matrix
we sample. With exponential shifts {1,2,4,...,2^(k-1)}, we get k diagonals
out of D possible. The question is whether k=4 (max shift=8) is enough
when D=4096.
"""

import torch
import torch.nn as nn


def analyze_coverage():
    """Show how much of the channel ring each shift config covers."""
    print("=" * 90)
    print(" Shift Coverage Analysis")
    print("=" * 90)

    for dim in [64, 128, 4096, 2048]:
        print(f"\n  dim={dim}:")
        print(f"  {'Shifts':>8} {'Values':<40} {'Max/Dim':>8} {'Coverage':>10}")
        print(f"  {'-'*70}")

        for n in [2, 4, 6, 8, 10, 12]:
            shifts = [1 << i for i in range(n)]
            max_shift = shifts[-1]
            # Each shift "covers" one diagonal of the D×D matrix
            # But due to periodicity, shift s and shift D-s are equivalent
            # Effective unique coverage = num_shifts unique diagonals out of D/2
            coverage_pct = len(shifts) / (dim // 2) * 100
            max_ratio = max_shift / dim

            if max_shift > dim // 2:
                note = " (wraps!)"
            elif max_ratio < 0.01:
                note = " (tiny)"
            else:
                note = ""

            shifts_str = str(shifts) if len(shifts) <= 8 else str(shifts[:6]) + f"...{shifts[-1]}"
            print(f"  {n:>8} {shifts_str:<40} {max_ratio:>7.3f} {coverage_pct:>8.1f}%{note}")


def analyze_interaction_reach():
    """
    Measure how much of the channel space a token can "see" through
    geometric products at different shift counts.

    With shift s, channel c interacts with channel (c+s)%D.
    After the projection, all channels mix. But the *geometric* mixing
    (inner/wedge products) only sees specific channel pairs.

    Question: what fraction of channel pairs are covered?
    """
    print("\n" + "=" * 90)
    print(" Channel Pair Coverage (geometric interaction reach)")
    print("=" * 90)

    dim = 4096
    print(f"\n  dim={dim}")
    print(f"  Total possible channel pairs: {dim*(dim-1)//2:,}")
    print(f"  {'Shifts':>8} {'Pairs covered':>15} {'Coverage':>10} {'Unique diags':>14}")
    print(f"  {'-'*55}")

    for n in [2, 4, 6, 8, 10, 12]:
        shifts = [1 << i for i in range(n)]
        # Each shift s creates dim unique pairs: (c, (c+s)%D) for all c
        # But pair (c, c+s) = pair (c+s, c) due to symmetry
        # And shift s and shift D-s give same pairs
        unique_pairs = set()
        for s in shifts:
            for c in range(dim):
                pair = (min(c, (c + s) % dim), max(c, (c + s) % dim))
                unique_pairs.add(pair)

        total_pairs = dim * (dim - 1) // 2
        coverage = len(unique_pairs) / total_pairs * 100
        print(f"  {n:>8} {len(unique_pairs):>15,} {coverage:>9.2f}% {len(shifts):>14}")


def flops_impact_of_more_shifts():
    """
    Show FLOPs cost of adding more shifts for each variant.
    Key insight: HRR absorbs more shifts cheaply (shared proj),
    vanilla gFFN pays linear cost per shift (wider cat_dim).
    """
    print("\n" + "=" * 90)
    print(" FLOPs Impact of More Shifts (dim=4096, per-token)")
    print("=" * 90)

    dim = 4096
    ffn_flops = 2 * dim * (4 * dim) + 2 * (4 * dim) * dim  # ~268M

    print(f"\n  Standard FFN baseline: {ffn_flops/1e6:.1f}M FLOPs")
    print()

    # Vanilla gFFN-Global (inner mode)
    print("  Vanilla gFFN-Global (inner mode, concat+project):")
    print(f"  {'Shifts':>8} {'cat_dim':>10} {'proj_out FLOPs':>15} {'vs FFN':>8}")
    print(f"  {'-'*50}")
    for n in [2, 4, 6, 8, 10, 12]:
        cat_dim = dim * (1 + n)  # base + n inner terms
        proj_flops = 2 * cat_dim * dim
        print(f"  {n:>8} {cat_dim:>10,} {proj_flops/1e6:>14.1f}M {proj_flops/ffn_flops:>7.2f}x")

    # HRR gFFN (inner mode)
    for pf in [4, 8]:
        proj_dim = dim // pf
        print(f"\n  HRR gFFN (inner mode, proj_factor={pf}, proj_dim={proj_dim}):")
        print(f"  {'Shifts':>8} {'Num terms':>10} {'All proj FLOPs':>15} {'out FLOPs':>10} {'Total':>10} {'vs FFN':>8}")
        print(f"  {'-'*70}")
        for n in [2, 4, 6, 8, 10, 12]:
            num_terms = 1 + n  # base + n inner terms
            # Shared proj: num_terms × Linear(dim, proj_dim)
            all_proj = num_terms * (2 * dim * proj_dim)
            # Out: Linear(proj_dim, dim)
            out = 2 * proj_dim * dim
            total = all_proj + out
            print(f"  {n:>8} {num_terms:>10} {all_proj/1e6:>14.1f}M {out/1e6:>9.1f}M {total/1e6:>9.1f}M {total/ffn_flops:>7.2f}x")


def test_output_quality_vs_shifts():
    """
    Empirical test: does output variance/rank improve with more shifts?
    Higher rank of output features suggests more expressive mixing.
    """
    print("\n" + "=" * 90)
    print(" Output Expressivity vs Shifts (dim=256, B=4, L=32)")
    print("=" * 90)

    from ltx_core.model.transformer.gffn import gFFNGlobal, gFFNGlobalHRR

    dim = 256  # smaller for speed
    torch.manual_seed(42)
    x = torch.randn(4, 32, dim)

    print(f"\n  {'Variant':<45} {'Out std':>10} {'Eff rank':>10} {'Max rank':>10}")
    print(f"  {'-'*80}")

    for n in [2, 4, 6, 8, 10, 12]:
        for cls, name, kwargs in [
            (gFFNGlobal, f"Vanilla(shifts={n}, inner)", dict(mode="inner")),
            (gFFNGlobalHRR, f"HRR(shifts={n}, inner, pf=4)", dict(mode="inner", proj_factor=4)),
        ]:
            model = cls(dim=dim, num_shifts=n, **kwargs)
            with torch.no_grad():
                out = model(x)  # [4, 32, 256]

            # Flatten batch+seq → [128, 256]
            flat = out.reshape(-1, dim)
            std = flat.std().item()

            # Effective rank via singular values
            svd = torch.linalg.svdvals(flat.float())
            # Effective rank = exp(entropy of normalized singular values)
            p = svd / svd.sum()
            p = p[p > 1e-10]
            eff_rank = torch.exp(-(p * p.log()).sum()).item()

            print(f"  {name:<45} {std:>10.4f} {eff_rank:>10.1f} {min(128, dim):>10}")


if __name__ == "__main__":
    analyze_coverage()
    analyze_interaction_reach()
    flops_impact_of_more_shifts()
    test_output_quality_vs_shifts()
