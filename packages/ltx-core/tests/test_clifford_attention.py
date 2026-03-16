#!/usr/bin/env python
"""Unit tests for CliffordRollingAttention and CliffordVideoAttention."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltx_core.model.transformer.clifford_attention import (
    CliffordRollingAttention,
    CliffordVideoAttention,
    compute_channel_shifts,
    compute_seq_shifts,
    compute_temporal_shifts,
)
from ltx_core.model.transformer.attention import Attention, AttentionFunction


# ---------------------------------------------------------------------------
# CliffordRollingAttention tests
# ---------------------------------------------------------------------------

def test_seq_shifts():
    """Verify bidirectional log-spaced shifts."""
    shifts = compute_seq_shifts(16, max_len=2048)
    assert shifts[0] == 0, "First shift should be self (0)"
    assert len(shifts) == 16
    assert any(s > 0 for s in shifts)
    assert any(s < 0 for s in shifts)
    print(f"  seq_shifts(16, 2048): {shifts}")


def test_channel_shifts():
    shifts = compute_channel_shifts(4)
    assert shifts == [1, 2, 4, 8]
    print(f"  channel_shifts(4): {shifts}")


def test_temporal_shifts():
    """Verify non-zero bidirectional temporal shifts."""
    shifts = compute_temporal_shifts(4)
    assert len(shifts) == 4
    assert 0 not in shifts, "Temporal shifts should not include 0"
    assert shifts == [1, -1, 2, -2]
    print(f"  temporal_shifts(4): {shifts}")

    shifts_0 = compute_temporal_shifts(0)
    assert shifts_0 == []
    print(f"  temporal_shifts(0): {shifts_0}")


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
    print(f"  Self-attn: {x.shape} -> {out.shape}")


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
    print(f"  Cross-attn: Q{x.shape} + K{ctx.shape} -> {out.shape}")


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
    out_std = std(x)
    out_cliff = cliff(x)
    assert out_std.shape == out_cliff.shape
    print(f"  Interface match: both produce {out_std.shape}")


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
    print(f"  Gated self-attn: {out.shape}")

    attn_cross = CliffordRollingAttention(
        query_dim=D, context_dim=64, heads=2, dim_head=64,
        apply_gated_attention=True,
        num_seq_shifts=4, num_channel_shifts=2,
    )
    ctx = torch.randn(B, 16, 64)
    out_cross = attn_cross(x, context=ctx)
    assert out_cross.shape == (B, L, D)
    print(f"  Gated cross-attn: {out_cross.shape}")


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
    print(f"  No channel shifts: {out.shape}")


def test_perturbation_mask_rolling():
    """CliffordRollingAttention respects perturbation_mask and all_perturbed."""
    B, L, D = 2, 32, 128
    attn = CliffordRollingAttention(
        query_dim=D, heads=2, dim_head=64,
        num_seq_shifts=4, num_channel_shifts=2,
    )

    x = torch.randn(B, L, D)

    # all_perturbed=True should skip Q/K and pass V through
    out_perturbed = attn(x, all_perturbed=True)
    assert out_perturbed.shape == (B, L, D)
    print(f"  all_perturbed=True: {out_perturbed.shape}")

    # perturbation_mask should blend attention output with V
    mask = torch.ones(B, L, 1, device=x.device)
    mask[:, :L // 2, :] = 0.0  # first half bypassed
    out_masked = attn(x, perturbation_mask=mask)
    assert out_masked.shape == (B, L, D)
    print(f"  perturbation_mask: {out_masked.shape}")

    # With all ones mask, output should equal normal attention
    out_normal = attn(x)
    out_ones = attn(x, perturbation_mask=torch.ones(B, L, 1))
    assert torch.allclose(out_normal, out_ones, atol=1e-5), "All-ones mask should match normal"
    print(f"  mask=ones matches normal: OK")


def test_flops_comparison():
    """Estimate FLOPs for standard vs rolling attention."""
    L = 1344
    H = 32
    D = 128
    num_seq_shifts = 16
    num_channel_shifts = 4
    scores_per_shift = 1 + num_channel_shifts

    standard_flops = L * L * H * D * 2
    rolling_flops = L * H * D * num_seq_shifts * scores_per_shift * 2

    ratio = standard_flops / rolling_flops
    print(f"\n  FLOPs comparison (L={L}, H={H}, D={D}):")
    print(f"    Standard:  {standard_flops / 1e9:.1f}G")
    print(f"    Rolling:   {rolling_flops / 1e9:.1f}G")
    print(f"    Ratio:     {ratio:.1f}x")
    assert ratio > 5, f"Expected >5x reduction, got {ratio:.1f}x"


# ---------------------------------------------------------------------------
# CliffordVideoAttention tests
# ---------------------------------------------------------------------------

def test_video_attention_shape():
    """CliffordVideoAttention produces correct output shape."""
    B, T, S = 2, 4, 64
    L = T * S
    D = 256
    heads = 4
    dim_head = D // heads

    attn = CliffordVideoAttention(
        query_dim=D, heads=heads, dim_head=dim_head,
        num_spatial_shifts=8, num_temporal_shifts=4,
        num_channel_shifts=2, max_spatial_len=128,
        num_frames=T,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    print(f"  Video self-attn: {x.shape} -> {out.shape} (T={T}, S={S})")


def test_video_attention_gradient_flow():
    """Verify gradients flow through CliffordVideoAttention."""
    B, T, S = 1, 3, 32
    L = T * S
    D = 128

    attn = CliffordVideoAttention(
        query_dim=D, heads=2, dim_head=64,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T,
    )

    x = torch.randn(B, L, D, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    print(f"  Video gradient flow: OK (norm={x.grad.norm():.4f})")


def test_video_attention_with_gating():
    """CliffordVideoAttention works with per-head gating."""
    B, T, S = 2, 3, 32
    L = T * S
    D = 128

    attn = CliffordVideoAttention(
        query_dim=D, heads=2, dim_head=64,
        apply_gated_attention=True,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)
    print(f"  Video gated attn: {out.shape}")


def test_video_attention_spherical_norm():
    """CliffordVideoAttention with spherical_norm normalizes pre-projection output."""
    B, T, S = 2, 3, 16
    L = T * S
    D = 128

    attn = CliffordVideoAttention(
        query_dim=D, heads=2, dim_head=64,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T,
        spherical_norm=True,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)

    # Gradient should still flow
    x_grad = torch.randn(B, L, D, requires_grad=True)
    out_grad = attn(x_grad)
    out_grad.sum().backward()
    assert x_grad.grad is not None
    assert not torch.isnan(x_grad.grad).any()
    print(f"  Video spherical_norm: {out.shape}, gradients OK")


def test_video_attention_cross_attention_fallback():
    """CliffordVideoAttention falls back to standard for cross-attention."""
    B, T, S = 2, 3, 16
    L = T * S
    D_q, D_k = 128, 64
    L_k = 20

    attn = CliffordVideoAttention(
        query_dim=D_q, context_dim=D_k, heads=2, dim_head=64,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T,
    )

    x = torch.randn(B, L, D_q)
    ctx = torch.randn(B, L_k, D_k)
    out = attn(x, context=ctx)
    assert out.shape == (B, L, D_q)
    print(f"  Video cross-attn fallback: Q{x.shape} + K{ctx.shape} -> {out.shape}")


def test_video_attention_perturbation_mask():
    """CliffordVideoAttention respects perturbation_mask and all_perturbed."""
    B, T, S = 2, 3, 16
    L = T * S
    D = 128

    attn = CliffordVideoAttention(
        query_dim=D, heads=2, dim_head=64,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T,
    )

    x = torch.randn(B, L, D)

    # all_perturbed=True
    out_perturbed = attn(x, all_perturbed=True)
    assert out_perturbed.shape == (B, L, D)
    print(f"  Video all_perturbed=True: {out_perturbed.shape}")

    # perturbation_mask
    mask = torch.ones(B, L, 1)
    mask[:, :L // 2, :] = 0.0
    out_masked = attn(x, perturbation_mask=mask)
    assert out_masked.shape == (B, L, D)
    print(f"  Video perturbation_mask: {out_masked.shape}")

    # All-ones mask should match normal output
    out_normal = attn(x)
    out_ones = attn(x, perturbation_mask=torch.ones(B, L, 1))
    assert torch.allclose(out_normal, out_ones, atol=1e-5), "All-ones mask should match normal"
    print(f"  Video mask=ones matches normal: OK")


def test_video_attention_num_frames_override():
    """CliffordVideoAttention allows num_frames override at forward time."""
    B, D = 2, 128
    T_init = 3
    T_override = 4
    S = 16

    attn = CliffordVideoAttention(
        query_dim=D, heads=2, dim_head=64,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T_init,
    )

    x = torch.randn(B, T_override * S, D)
    out = attn(x, num_frames=T_override)
    assert out.shape == x.shape
    print(f"  num_frames override: init={T_init}, forward={T_override}, L={T_override * S}")


def test_video_attention_no_temporal():
    """CliffordVideoAttention with num_temporal_shifts=0 (spatial-only)."""
    B, T, S = 2, 3, 32
    L = T * S
    D = 128

    attn = CliffordVideoAttention(
        query_dim=D, heads=2, dim_head=64,
        num_spatial_shifts=8, num_temporal_shifts=0,
        num_channel_shifts=2, num_frames=T,
    )

    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)
    assert attn.total_shifts == 8, f"Expected 8 total shifts, got {attn.total_shifts}"
    print(f"  Video spatial-only: {out.shape} (total_shifts={attn.total_shifts})")


def test_video_flops_comparison():
    """Estimate FLOPs for standard vs video rolling attention with temporal savings."""
    T = 9
    S = 150
    L = T * S
    H = 32
    D = 128
    num_spatial_shifts = 12
    num_temporal_shifts = 4
    num_channel_shifts = 4
    total_shifts = num_spatial_shifts + num_temporal_shifts
    scores_per_shift = 1 + num_channel_shifts

    standard_flops = L * L * H * D * 2
    video_rolling_flops = L * H * D * total_shifts * scores_per_shift * 2

    ratio_standard = standard_flops / video_rolling_flops
    print(f"\n  Video FLOPs comparison (T={T}, S={S}, L={L}, H={H}, D={D}):")
    print(f"    Standard:        {standard_flops / 1e9:.1f}G")
    print(f"    Video rolling:   {video_rolling_flops / 1e9:.1f}G")
    print(f"    Ratio vs std:    {ratio_standard:.1f}x")
    print(f"    Total shifts:    {total_shifts} (spatial={num_spatial_shifts}, temporal={num_temporal_shifts})")
    assert ratio_standard > 5, f"Expected >5x reduction, got {ratio_standard:.1f}x"


def test_video_attention_interface_matches_standard():
    """CliffordVideoAttention forward signature is compatible with Attention."""
    B, T, S = 1, 3, 16
    L = T * S
    D = 128
    heads, dim_head = 2, 64

    std = Attention(query_dim=D, heads=heads, dim_head=dim_head)
    video = CliffordVideoAttention(
        query_dim=D, heads=heads, dim_head=dim_head,
        num_spatial_shifts=4, num_temporal_shifts=2,
        num_channel_shifts=2, num_frames=T,
    )

    x = torch.randn(B, L, D)
    out_std = std(x)
    out_video = video(x)
    assert out_std.shape == out_video.shape
    print(f"  Interface match with Attention: both produce {out_std.shape}")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing CliffordRollingAttention\n")

    test_seq_shifts()
    test_channel_shifts()
    test_temporal_shifts()
    test_self_attention_shape()
    test_cross_attention_shape()
    test_gradient_flow()
    test_same_interface_as_attention()
    test_with_gated_attention()
    test_no_channel_shifts()
    test_perturbation_mask_rolling()
    test_flops_comparison()

    print("\n\nTesting CliffordVideoAttention\n")

    test_video_attention_shape()
    test_video_attention_gradient_flow()
    test_video_attention_with_gating()
    test_video_attention_spherical_norm()
    test_video_attention_cross_attention_fallback()
    test_video_attention_perturbation_mask()
    test_video_attention_num_frames_override()
    test_video_attention_no_temporal()
    test_video_flops_comparison()
    test_video_attention_interface_matches_standard()

    print("\n  All tests passed!")
