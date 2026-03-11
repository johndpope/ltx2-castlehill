"""Tests for gFFN (Geometric Feed-Forward Network) modules."""

import pytest
import torch

from ltx_core.model.transformer.gffn import (
    CliffordMode,
    ShiftStrategy,
    compute_shifts,
    gFFNGlobal,
    gFFNGlobalHRR,
    gFFNHybrid,
    create_gffn,
)


@pytest.fixture
def batch():
    """Create a sample batch: [B=2, L=16, D=64]."""
    torch.manual_seed(42)
    return torch.randn(2, 16, 64)


# ============================================================================
# Shift strategy tests
# ============================================================================

class TestComputeShifts:
    def test_exponential(self):
        shifts = compute_shifts(4, dim=4096, strategy="exponential")
        assert shifts == [1, 2, 4, 8]

    def test_exponential_8(self):
        shifts = compute_shifts(8, dim=4096, strategy="exponential")
        assert shifts == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_log_uniform_covers_range(self):
        shifts = compute_shifts(8, dim=4096, strategy="log_uniform")
        assert shifts[0] == 1
        assert shifts[-1] == 2048  # dim//2
        assert len(shifts) >= 6  # might lose some to dedup
        # Should be sorted ascending
        assert shifts == sorted(shifts)

    def test_geometric_covers_range(self):
        shifts = compute_shifts(8, dim=4096, strategy="geometric")
        assert shifts[0] == 1
        assert shifts[-1] == 2048
        assert shifts == sorted(shifts)

    def test_log_uniform_better_coverage_than_exponential(self):
        """Log-uniform should reach further into the channel ring."""
        exp_shifts = compute_shifts(8, dim=4096, strategy="exponential")
        log_shifts = compute_shifts(8, dim=4096, strategy="log_uniform")
        assert max(log_shifts) > max(exp_shifts)

    def test_all_strategies_produce_positive_unique_ints(self):
        for strategy in ShiftStrategy:
            shifts = compute_shifts(8, dim=4096, strategy=strategy)
            assert all(s >= 1 for s in shifts)
            assert len(shifts) == len(set(shifts))

    def test_single_shift(self):
        for strategy in ShiftStrategy:
            shifts = compute_shifts(1, dim=4096, strategy=strategy)
            assert shifts == [1]

    def test_small_dim(self):
        shifts = compute_shifts(4, dim=8, strategy="log_uniform")
        assert all(1 <= s <= 4 for s in shifts)


# ============================================================================
# gFFN-Global tests
# ============================================================================

class TestGFFNGlobal:
    def test_output_shape(self, batch):
        model = gFFNGlobal(dim=64)
        out = model(batch)
        assert out.shape == batch.shape

    def test_output_shape_different_dim_out(self, batch):
        model = gFFNGlobal(dim=64, dim_out=128)
        out = model(batch)
        assert out.shape == (2, 16, 128)

    def test_modes(self, batch):
        for mode in CliffordMode:
            model = gFFNGlobal(dim=64, mode=mode)
            out = model(batch)
            assert out.shape == batch.shape, f"Failed for mode {mode}"

    def test_num_shifts(self, batch):
        for n in [1, 2, 4, 8]:
            model = gFFNGlobal(dim=64, num_shifts=n)
            out = model(batch)
            assert out.shape == batch.shape

    def test_shift_strategies(self, batch):
        for strategy in ShiftStrategy:
            model = gFFNGlobal(dim=64, num_shifts=4, shift_strategy=strategy)
            out = model(batch)
            assert out.shape == batch.shape

    def test_gated(self, batch):
        model = gFFNGlobal(dim=64, gate=True)
        out = model(batch)
        assert out.shape == batch.shape

    def test_gradient_flow(self, batch):
        model = gFFNGlobal(dim=64)
        batch.requires_grad_(True)
        out = model(batch)
        loss = out.sum()
        loss.backward()
        assert batch.grad is not None
        assert batch.grad.shape == batch.shape

    def test_wedge_antisymmetry(self):
        """Wedge product should be antisymmetric: x∧y = -(y∧x)."""
        torch.manual_seed(0)
        x = torch.randn(1, 4, 32)
        g = torch.randn(1, 1, 32)
        s = 1
        rg = torch.roll(g, shifts=s, dims=-1)
        rx = torch.roll(x, shifts=s, dims=-1)
        wedge_xg = x * rg - rx * g
        wedge_gx = g * torch.roll(x, shifts=s, dims=-1) - torch.roll(g, shifts=s, dims=-1) * x
        assert torch.allclose(wedge_xg, -wedge_gx, atol=1e-6)

    def test_param_count_comparison(self):
        """Compare param counts between FFN and gFFN variants."""
        from ltx_core.model.transformer.feed_forward import FeedForward

        dim = 4096
        ffn = FeedForward(dim, dim_out=dim)
        ffn_params = sum(p.numel() for p in ffn.parameters())

        for shifts, mode in [(2, "full"), (4, "full"), (4, "inner")]:
            gffn = gFFNGlobal(dim=dim, num_shifts=shifts, mode=mode)
            gffn_params = sum(p.numel() for p in gffn.parameters())
            ratio = gffn_params / ffn_params
            print(f"  gFFN(shifts={shifts}, mode={mode}): {gffn_params:,} "
                  f"({ratio:.2f}x FFN's {ffn_params:,})")


# ============================================================================
# gFFN-Hybrid tests
# ============================================================================

class TestGFFNHybrid:
    def test_output_shape(self, batch):
        model = gFFNHybrid(dim=64)
        out = model(batch)
        assert out.shape == batch.shape

    def test_differential_mode(self, batch):
        model_diff = gFFNHybrid(dim=64, differential_mode=True)
        model_abs = gFFNHybrid(dim=64, differential_mode=False)
        out_diff = model_diff(batch)
        out_abs = model_abs(batch)
        assert out_diff.shape == batch.shape
        assert out_abs.shape == batch.shape


# ============================================================================
# gFFN-HRR tests
# ============================================================================

class TestGFFNGlobalHRR:
    """Tests for the HRR-augmented gFFN-Global variant."""

    def test_output_shape(self, batch):
        model = gFFNGlobalHRR(dim=64, proj_factor=4)
        out = model(batch)
        assert out.shape == batch.shape

    def test_output_shape_different_dim_out(self, batch):
        model = gFFNGlobalHRR(dim=64, dim_out=128, proj_factor=4)
        out = model(batch)
        assert out.shape == (2, 16, 128)

    def test_modes(self, batch):
        for mode in CliffordMode:
            model = gFFNGlobalHRR(dim=64, proj_factor=4, mode=mode)
            out = model(batch)
            assert out.shape == batch.shape, f"Failed for mode {mode}"

    def test_proj_factors(self, batch):
        for pf in [2, 4, 8, 16]:
            model = gFFNGlobalHRR(dim=64, proj_factor=pf)
            out = model(batch)
            assert out.shape == batch.shape

    def test_shift_strategies(self, batch):
        for strategy in ShiftStrategy:
            model = gFFNGlobalHRR(dim=64, proj_factor=4, shift_strategy=strategy)
            out = model(batch)
            assert out.shape == batch.shape

    def test_unit_norm_off(self, batch):
        model = gFFNGlobalHRR(dim=64, proj_factor=4, unit_norm=False)
        out = model(batch)
        assert out.shape == batch.shape

    def test_div_by_terms_off(self, batch):
        model = gFFNGlobalHRR(dim=64, proj_factor=4, div_by_terms=False)
        out = model(batch)
        assert out.shape == batch.shape

    def test_gradient_flow(self, batch):
        model = gFFNGlobalHRR(dim=64, proj_factor=4)
        batch.requires_grad_(True)
        out = model(batch)
        loss = out.sum()
        loss.backward()
        assert batch.grad is not None
        assert batch.grad.shape == batch.shape

    def test_num_terms_counted_correctly(self):
        # inner mode: 1 base + num_shifts inner terms
        m = gFFNGlobalHRR(dim=64, num_shifts=4, mode="inner", proj_factor=4,
                          shift_strategy="exponential")
        assert m.num_terms == 5  # 1 + 4

        # full mode: 1 base + 2*num_shifts
        m = gFFNGlobalHRR(dim=64, num_shifts=4, mode="full", proj_factor=4,
                          shift_strategy="exponential")
        assert m.num_terms == 9  # 1 + 8

        # wedge mode: 1 base + num_shifts
        m = gFFNGlobalHRR(dim=64, num_shifts=4, mode="wedge", proj_factor=4,
                          shift_strategy="exponential")
        assert m.num_terms == 5  # 1 + 4

    def test_param_count_dramatic_reduction(self):
        """HRR should have dramatically fewer params than vanilla gFFN."""
        from ltx_core.model.transformer.feed_forward import FeedForward

        dim = 4096
        ffn = FeedForward(dim, dim_out=dim)
        ffn_params = sum(p.numel() for p in ffn.parameters())

        for pf in [4, 8, 16]:
            hrr = gFFNGlobalHRR(dim=dim, num_shifts=8, mode="inner", proj_factor=pf)
            hrr_params = sum(p.numel() for p in hrr.parameters())
            ratio = hrr_params / ffn_params
            print(f"  HRR(pf={pf}, 8 shifts): {hrr_params:,} ({ratio:.3f}x FFN's {ffn_params:,})")
            assert hrr_params < ffn_params

    def test_superposition_not_degenerate(self, batch):
        """Superposition shouldn't collapse all terms to zero."""
        model = gFFNGlobalHRR(dim=64, proj_factor=4)
        out = model(batch)
        assert out.abs().mean() > 1e-6, "Output is degenerate (near-zero)"

    def test_log_uniform_default_for_hrr(self):
        """HRR default should use log_uniform shifts (better coverage at high dim)."""
        m = gFFNGlobalHRR(dim=4096, proj_factor=4)
        # Log-uniform should reach much further than exponential
        assert max(m.shifts) > 128, f"Expected wide shifts, got max={max(m.shifts)}"

    def test_bf16_support(self, batch):
        """HRR gFFN should work with bf16."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bf16 test")

        model = gFFNGlobalHRR(dim=64, proj_factor=4).cuda().to(torch.bfloat16)
        x = batch.cuda().to(torch.bfloat16)
        out = model(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape

    def test_many_shifts_affordable(self):
        """With HRR, 12 shifts should still be small params."""
        from ltx_core.model.transformer.feed_forward import FeedForward

        dim = 4096
        ffn_params = sum(p.numel() for p in FeedForward(dim, dim_out=dim).parameters())

        # 12 shifts with HRR should still be tiny
        hrr = gFFNGlobalHRR(dim=dim, num_shifts=12, mode="inner",
                            proj_factor=4, shift_strategy="log_uniform")
        hrr_params = sum(p.numel() for p in hrr.parameters())
        ratio = hrr_params / ffn_params
        print(f"  HRR(12 shifts, pf=4, log_uniform): {hrr_params:,} ({ratio:.3f}x FFN)")
        # Shared proj means params don't grow with shifts
        assert ratio < 0.1


# ============================================================================
# Factory tests
# ============================================================================

class TestFactory:
    def test_create_global(self, batch):
        model = create_gffn(64, variant="global")
        assert isinstance(model, gFFNGlobal)
        assert model(batch).shape == batch.shape

    def test_create_hybrid(self, batch):
        model = create_gffn(64, variant="hybrid")
        assert isinstance(model, gFFNHybrid)
        assert model(batch).shape == batch.shape

    def test_create_hrr(self, batch):
        model = create_gffn(64, variant="hrr", proj_factor=4)
        assert isinstance(model, gFFNGlobalHRR)
        assert model(batch).shape == batch.shape

    def test_create_with_shift_strategy(self, batch):
        model = create_gffn(64, variant="hrr", proj_factor=4,
                           shift_strategy="log_uniform", num_shifts=8)
        assert isinstance(model, gFFNGlobalHRR)
        assert model(batch).shape == batch.shape

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            create_gffn(64, variant="invalid")


# ============================================================================
# Drop-in replacement tests
# ============================================================================

class TestDropInReplacement:
    """Test that gFFN can be a drop-in replacement for FeedForward."""

    def test_same_interface(self, batch):
        """Both FFN and gFFN should accept [B, L, D] → [B, L, D]."""
        from ltx_core.model.transformer.feed_forward import FeedForward

        dim = 64
        ffn = FeedForward(dim, dim_out=dim)
        gffn = gFFNGlobal(dim=dim)
        hrr = gFFNGlobalHRR(dim=dim, proj_factor=4)

        assert ffn(batch).shape == batch.shape
        assert gffn(batch).shape == batch.shape
        assert hrr(batch).shape == batch.shape

    def test_bf16_support(self, batch):
        """gFFN should work with bf16 (LTX2 uses bf16)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bf16 test")

        model = gFFNGlobal(dim=64).cuda().to(torch.bfloat16)
        x = batch.cuda().to(torch.bfloat16)
        out = model(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape
