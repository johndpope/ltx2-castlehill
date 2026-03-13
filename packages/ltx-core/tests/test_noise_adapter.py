"""Tests for VFM noise adapter."""

import torch
import pytest

from ltx_core.model.transformer.noise_adapter import (
    NoiseAdapterMLP,
    NoiseAdapterTransformer,
    TASK_CLASSES,
    create_noise_adapter,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_params():
    return {
        "batch_size": 2,
        "seq_len": 336,  # tokens_per_frame for 24x14
        "input_dim": 3072,  # LTX-2 inner dim
        "latent_dim": 128,  # Video latent channels
    }


class TestNoiseAdapterMLP:
    def test_forward_shape(self, device, batch_params):
        B, S, D, C = (
            batch_params["batch_size"],
            batch_params["seq_len"],
            batch_params["input_dim"],
            batch_params["latent_dim"],
        )
        adapter = NoiseAdapterMLP(input_dim=D, latent_dim=C).to(device)

        features = torch.randn(B, S, D, device=device)
        task_class = torch.zeros(B, dtype=torch.long, device=device)

        mu, log_sigma = adapter(features, task_class)

        assert mu.shape == (B, S, C)
        assert log_sigma.shape == (B, S, C)

    def test_sample_shape(self, device, batch_params):
        B, S, D, C = (
            batch_params["batch_size"],
            batch_params["seq_len"],
            batch_params["input_dim"],
            batch_params["latent_dim"],
        )
        adapter = NoiseAdapterMLP(input_dim=D, latent_dim=C).to(device)

        features = torch.randn(B, S, D, device=device)
        task_class = torch.zeros(B, dtype=torch.long, device=device)

        z = adapter.sample(features, task_class)
        assert z.shape == (B, S, C)

    def test_kl_divergence_nonnegative(self, device, batch_params):
        B, S, D, C = (
            batch_params["batch_size"],
            batch_params["seq_len"],
            batch_params["input_dim"],
            batch_params["latent_dim"],
        )
        adapter = NoiseAdapterMLP(input_dim=D, latent_dim=C).to(device)

        features = torch.randn(B, S, D, device=device)
        task_class = torch.zeros(B, dtype=torch.long, device=device)

        mu, log_sigma = adapter(features, task_class)
        kl = adapter.kl_divergence(mu, log_sigma)

        assert kl.item() >= 0, "KL divergence should be non-negative"

    def test_kl_zero_for_standard_normal(self, device):
        """KL(N(0,1) || N(0,1)) = 0"""
        adapter = NoiseAdapterMLP(input_dim=64, latent_dim=32).to(device)

        mu = torch.zeros(1, 10, 32, device=device)
        log_sigma = torch.zeros(1, 10, 32, device=device)

        kl = adapter.kl_divergence(mu, log_sigma)
        assert kl.item() < 1e-6, f"KL should be ~0 for standard normal, got {kl.item()}"

    def test_different_task_classes(self, device, batch_params):
        """Different task classes should produce different outputs."""
        D, C = batch_params["input_dim"], batch_params["latent_dim"]
        adapter = NoiseAdapterMLP(input_dim=D, latent_dim=C).to(device)

        features = torch.randn(1, 10, D, device=device)

        outputs = {}
        for name, idx in TASK_CLASSES.items():
            task = torch.tensor([idx], device=device)
            mu, _ = adapter(features, task)
            outputs[name] = mu

        # At least some tasks should produce different outputs
        # (after initialization they're similar, but not identical due to task embedding)
        for name1 in list(TASK_CLASSES.keys())[:2]:
            for name2 in list(TASK_CLASSES.keys())[2:]:
                diff = (outputs[name1] - outputs[name2]).abs().mean()
                # They shouldn't be exactly equal
                assert diff > 0, f"Tasks {name1} and {name2} produced identical outputs"

    def test_initialization_near_standard_normal(self, device, batch_params):
        """Initial adapter output should be close to N(0,I)."""
        D, C = batch_params["input_dim"], batch_params["latent_dim"]
        adapter = NoiseAdapterMLP(input_dim=D, latent_dim=C, init_sigma=-2.0).to(device)

        features = torch.randn(1, 100, D, device=device)
        task = torch.zeros(1, dtype=torch.long, device=device)

        mu, log_sigma = adapter(features, task)

        # μ should be near 0 (initialized with zeros)
        assert mu.abs().mean().item() < 0.5, f"Initial mu should be near 0, got mean={mu.abs().mean().item()}"

        # σ should be small (log_sigma initialized to -2.0 → σ ≈ 0.135)
        sigma = torch.exp(log_sigma)
        assert sigma.mean().item() < 0.5, f"Initial sigma should be small, got mean={sigma.mean().item()}"


class TestNoiseAdapterTransformer:
    def test_forward_shape(self, device):
        adapter = NoiseAdapterTransformer(
            input_dim=256, latent_dim=64, hidden_dim=128, num_heads=4, num_layers=2
        ).to(device)

        features = torch.randn(2, 20, 256, device=device)
        task = torch.zeros(2, dtype=torch.long, device=device)

        mu, log_sigma = adapter(features, task)
        assert mu.shape == (2, 20, 64)
        assert log_sigma.shape == (2, 20, 64)

    def test_sample(self, device):
        adapter = NoiseAdapterTransformer(
            input_dim=256, latent_dim=64, hidden_dim=128, num_heads=4, num_layers=2
        ).to(device)

        features = torch.randn(2, 20, 256, device=device)
        task = torch.zeros(2, dtype=torch.long, device=device)

        z = adapter.sample(features, task)
        assert z.shape == (2, 20, 64)


class TestFactory:
    def test_create_mlp(self, device):
        adapter = create_noise_adapter(input_dim=256, latent_dim=64, variant="mlp")
        assert isinstance(adapter, NoiseAdapterMLP)

    def test_create_transformer(self, device):
        adapter = create_noise_adapter(input_dim=256, latent_dim=64, variant="transformer")
        assert isinstance(adapter, NoiseAdapterTransformer)

    def test_invalid_variant(self, device):
        with pytest.raises(ValueError, match="Unknown"):
            create_noise_adapter(input_dim=256, latent_dim=64, variant="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
