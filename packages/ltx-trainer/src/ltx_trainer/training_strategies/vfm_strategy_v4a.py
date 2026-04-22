"""VFM v4a — Gaussian Adapter + Self-E (drops Spherical Cauchy).

Pivot from v3b: same Self-E architecture but uses standard Gaussian noise
instead of Spherical Cauchy. The LTX-2.3 DiT was pretrained on Gaussian noise,
so a rank-32 LoRA can learn to denoise Gaussian adapter noise without the
distribution mismatch that corrupted every Cauchy-based version (v1f→v3b).

Changes from v3b:
- spherical_noise=False: z = mu + exp(log_sigma) * eps (Gaussian reparameterization)
- Gaussian KL: 0.5 * (mu² + exp(2*log_sigma) - 2*log_sigma - 1)
- Dropped Cauchy-specific losses: kappa, magnitude reg, mu_align (those were
  anti-collapse for Cauchy; Gaussian doesn't need them)

Architecture:
    Text → Gemma → NoiseAdapterV1b → mu, log_sigma
                                ↓
                      z = mu + sigma * eps     [Gaussian]
                                ↓
                      DiT 22B (LoRA r=32) → velocity → x̂₀ = z - v
                                ↓
                      data_loss + self_eval(50%) + Gaussian KL
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.vfm_strategy_v3b import (
    SelfEVFMv3bTrainingConfig,
    SelfEVFMv3bTrainingStrategy,
)


class GaussianVFMv4aTrainingConfig(SelfEVFMv3bTrainingConfig):
    """Configuration for VFM v4a (Gaussian adapter + Self-E)."""

    name: Literal["vfm_v4a"] = "vfm_v4a"

    # Force Gaussian — overrides v1f's spherical_noise=True default
    spherical_noise: bool = Field(
        default=False,
        description="Always False for v4a. Uses Gaussian reparameterization.",
    )

    # Cauchy-specific params (kept for config compat, ignored)
    kappa_min: float = Field(default=0.1, description="Unused in v4a.")
    kappa_max: float = Field(default=50.0, description="Unused in v4a.")
    kappa_entropy_weight: float = Field(default=0.0, description="Unused in v4a.")
    kappa_target: float = Field(default=2.0, description="Unused in v4a.")
    kappa_pull_weight: float = Field(default=0.0, description="Unused in v4a.")
    magnitude_reg_weight: float = Field(default=0.0, description="Unused in v4a.")
    target_magnitude: float = Field(default=1.0, description="Unused in v4a.")


class GaussianVFMv4aTrainingStrategy(SelfEVFMv3bTrainingStrategy):
    """VFM v4a — Gaussian adapter noise + Self-E evaluation.

    Identical to v3b except:
    1. Forces spherical_noise=False → Gaussian reparameterization
    2. Computes standard Gaussian KL instead of Spherical Cauchy KL
    3. No Cauchy-specific loss terms (kappa, magnitude, mu_align)
    """

    config: GaussianVFMv4aTrainingConfig

    def __init__(self, config: GaussianVFMv4aTrainingConfig):
        # Force spherical_noise off regardless of what config says
        config_dict = config.model_dump()
        config_dict["spherical_noise"] = False
        super().__init__(config)
        logger.info("VFM v4a: Gaussian adapter noise (Spherical Cauchy disabled)")

    def _compute_gaussian_kl(
        self,
        mu: Tensor,
        log_sigma: Tensor,
        free_bits: float = 0.0,
    ) -> Tensor:
        """Compute standard Gaussian KL: KL(N(mu,σ²) || N(0,1)).

        KL = 0.5 * (mu² + exp(2*log_sigma) - 2*log_sigma - 1)

        Args:
            mu: [B, seq, D] adapter mean
            log_sigma: [B, seq, D] adapter log-std
            free_bits: per-dimension KL floor (prevents posterior collapse)

        Returns:
            Scalar KL loss (mean over batch)
        """
        kl_per_dim = 0.5 * (
            mu.pow(2)
            + torch.exp(2 * log_sigma)
            - 2 * log_sigma
            - 1
        )  # [B, seq, D]

        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        # Mean over seq and D dims, then over batch
        kl_per_sample = kl_per_dim.mean(dim=(1, 2))  # [B]
        return kl_per_sample.mean()

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs,
    ) -> Tensor:
        """Compute v4a loss: data + self_eval + Gaussian KL.

        Overrides v3b to use Gaussian KL instead of reading _vfm_kl_loss.
        """
        cfg = self.config
        device = video_pred.device

        # Use parent's compute_loss for data_loss + self_eval + perceptual + obs
        total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        # Replace the KL component: v3b reads _vfm_kl_loss (always 0).
        # We compute Gaussian KL directly from adapter outputs.
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        if use_adapter and adapter_mu is not None and cfg.kl_weight > 0:
            adapter_log_sigma = inputs._vfm_adapter_log_sigma
            gaussian_kl = self._compute_gaussian_kl(
                adapter_mu, adapter_log_sigma, cfg.kl_free_bits,
            )
            total_loss = total_loss + cfg.kl_weight * gaussian_kl

            # Log Gaussian KL
            if self._current_step % 20 == 0:
                try:
                    import wandb  # noqa: PLC0415
                    if wandb.run is not None:
                        wandb.log({
                            "v4a/gaussian_kl": gaussian_kl.item(),
                            "v4a/mu_norm": adapter_mu.norm(dim=-1).mean().item(),
                            "v4a/sigma_mean": torch.exp(adapter_log_sigma).mean().item(),
                        })
                except Exception:
                    pass

        return total_loss
