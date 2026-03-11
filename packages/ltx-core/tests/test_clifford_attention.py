#!/usr/bin/env python
"""Unit tests for CliffordRollingAttention."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltx_core.model.transformer.clifford_attention import (
    CliffordRollingAttention,
    compute_channel_shifts,
    compute_seq_shifts,
)
from ltx_core.model.transformer.attention import Attention, AttentionFunction


def test_seq_shifts():
    """Verify bidirectional log-spaced shifts."""
    shifts = compute_seq_shifts(16, max_len=2048)
    assert shifts[0] == 0, "First shift should be self (0)"
    assert len(shifts) == 16
    # Should have both positive and negative
    assert any(s > 0 for s in shifts)
    assert any(s < 0 for s in shifts)
    print(f"  seq_shifts(16, 2048): {shifts}")


def test_channel_shifts():
    shifts = compute_channel_shifts(4)
    assert shifts == [1, 2, 4, 8]
    print(f"  channel_shifts(4): {shifts}")


def test_self_attention_shape():
    """CliffordRollingAttention produces correct output shape for self-attention."""
    B, L, D = 2, 64, 256
    heads = 4
    dim_head = D // heads

    attn = CliffordRollingAttention(
        query_dim=D, heads=heads, dim_head=dim_head,
        num_seq_shifts=8, num_channel_shifts=2, max_seq_len=128,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    print(f"  Self-attn: {x.shape} → {out.shape} ✓")


def test_cross_attention_shape():
    """CliffordRollingAttention falls back to standard for cross-attention."""
    B, L_q, L_k, D_q, D_k = 2, 64, 32, 256, 128

    attn = CliffordRollingAttention(
        query_dim=D_q, context_dim=D_k, heads=4, dim_head=64,
        num_seq_shifts=8, num_channel_shifts=2,
    )

    x = torch.randn(B, L_q, D_q)
    ctx = torch.randn(B, L_k, D_k)
    out = attn(x, context=ctx)
    assert out.shape == (B, L_q, D_q), f"Expected {(B, L_q, D_q)}, got {out.shape}"
    print(f"  Cross-attn: Q{x.shape} + K{ctx.shape} → {out.shape} ✓")


def test_gradient_flow():
    """Verify gradients flow through CliffordRollingAttention."""
    B, L, D = 1, 32, 128
    attn = CliffordRollingAttention(
        query_dim=D, heads=2, dim_head=64,
        num_seq_shifts=4, num_channel_shifts=2, max_seq_len=64,
    )

    x = torch.randn(B, L, D, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    print(f"  Gradients: OK (norm={x.grad.norm():.4f})")


def test_same_interface_as_attention():
    """CliffordRollingAttention has same forward signature as Attention."""
    B, L, D = 1, 32, 256
    heads, dim_head = 4, 64

    std = Attention(query_dim=D, heads=heads, dim_head=dim_head)
    cliff = CliffordRollingAttention(
        query_dim=D, heads=heads, dim_head=dim_head,
        num_seq_shifts=4, num_channel_shifts=2,
    )

    x = torch.randn(B, L, D)
    # Both should accept same args
    out_std = std(x)
    out_cliff = cliff(x)
    assert out_std.shape == out_cliff.shape
    print(f"  Interface match: both produce {out_std.shape} ✓")


def test_with_gated_attention():
    """CliffordRollingAttention works with per-head gating."""
    B, L, D = 2, 32, 128
    attn = CliffordRollingAttention(
        query_dim=D, heads=2, dim_head=64,
        apply_gated_attention=True,
        num_seq_shifts=4, num_channel_shifts=2,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)
    print(f"  Gated self-attn: {out.shape} ✓")

    # Also test cross-attn with gating
    attn_cross = CliffordRollingAttention(
        query_dim=D, context_dim=64, heads=2, dim_head=64,
        apply_gated_attention=True,
        num_seq_shifts=4, num_channel_shifts=2,
    )
    ctx = torch.randn(B, 16, 64)
    out_cross = attn_cross(x, context=ctx)
    assert out_cross.shape == (B, L, D)
    print(f"  Gated cross-attn: {out_cross.shape} ✓")


def test_no_channel_shifts():
    """Works with num_channel_shifts=0 (pure sparse rolling, no geometric terms)."""
    B, L, D = 2, 32, 128
    attn = CliffordRollingAttention(
        query_dim=D, heads=2, dim_head=64,
        num_seq_shifts=8, num_channel_shifts=0,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)
    print(f"  No channel shifts: {out.shape} ✓")


def test_flops_comparison():
    """Estimate FLOPs for standard vs rolling attention."""
    L = 1344  # Typical LTX2 video sequence length
    H = 32
    D = 128
    num_seq_shifts = 16
    num_channel_shifts = 4
    scores_per_shift = 1 + num_channel_shifts

    standard_flops = L * L * H * D * 2  # Q@K + attn@V
    rolling_flops = L * H * D * num_seq_shifts * scores_per_shift * 2

    ratio = standard_flops / rolling_flops
    print(f"\n  FLOPs comparison (L={L}, H={H}, D={D}):")
    print(f"    Standard:  {standard_flops / 1e9:.1f}G")
    print(f"    Rolling:   {rolling_flops / 1e9:.1f}G")
    print(f"    Ratio:     {ratio:.1f}×")
    assert ratio > 5, f"Expected >5× reduction, got {ratio:.1f}×"


if __name__ == "__main__":
    print("Testing CliffordRollingAttention\n")

    test_seq_shifts()
    test_channel_shifts()
    test_self_attention_shape()
    test_cross_attention_shape()
    test_gradient_flow()
    test_same_interface_as_attention()
    test_with_gated_attention()
    test_no_channel_shifts()
    test_flops_comparison()

    print("\n  All tests passed! ✓")
