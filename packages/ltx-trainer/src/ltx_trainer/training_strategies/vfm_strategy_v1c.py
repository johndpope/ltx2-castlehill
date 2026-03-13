"""VFM v1c — Diversity-regularized Variational Flow Maps for LTX-2.

Upgrades over v1b (vfm_strategy_v1b.py):
  - Token diversity loss: penalizes uniform μ across video tokens
  - Temporal diversity loss: penalizes uniform μ across frames (motion diversity)
  - Spatial diversity loss: penalizes uniform μ across spatial positions within frames

Motivation (from HiAR, arxiv:2603.08703):
  Reverse-KL regularization (mode-seeking) can cause the adapter to collapse
  to a narrow mode where all tokens produce similar μ → uniform noise →
  static/low-diversity video. The diversity regularizer explicitly encourages
  spatial and temporal variation in the adapter's output.

Architecture and adapter are identical to v1b (NoiseAdapterV1b with cross-attn
+ self-attn + positional encoding). Only the loss function changes.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1b import (
    VFMv1bTrainingConfig,
    VFMv1bTrainingStrategy,
)


class VFMv1cTrainingConfig(VFMv1bTrainingConfig):
    """Configuration for VFM v1c (diversity-regularized adapter)."""

    name: Literal["vfm_v1c"] = "vfm_v1c"

    # === Diversity Regularization ===
    diversity_weight: float = Field(
        default=0.1, ge=0.0,
        description="Weight for overall token diversity loss. "
        "Encourages adapter μ to vary across video tokens.",
    )
    temporal_diversity_weight: float = Field(
        default=0.2, ge=0.0,
        description="Extra weight for temporal (cross-frame) diversity. "
        "Prevents low-motion collapse by encouraging μ variation across frames.",
    )
    spatial_diversity_weight: float = Field(
        default=0.05, ge=0.0,
        description="Weight for spatial (within-frame) diversity. "
        "Encourages μ variation across spatial positions within each frame.",
    )
    diversity_warmup_steps: int = Field(
        default=200, ge=0,
        description="Linearly ramp diversity loss from 0 to full weight over this many steps. "
        "Lets the adapter learn basic structure before pushing for diversity.",
    )


class VFMv1cTrainingStrategy(VFMv1bTrainingStrategy):
    """VFM v1c — v1b adapter + diversity regularization in loss.

    Overrides only compute_loss from v1b/v1a to add three diversity terms:
      1. Token diversity: std(μ) across all video tokens
      2. Temporal diversity: std(μ) across frames (averaged over spatial positions)
      3. Spatial diversity: std(μ) across spatial positions (averaged over frames)

    These are maximization objectives (we want HIGH diversity), so they enter
    the loss as negative terms: L_div = -weight * std(μ).
    """

    config: VFMv1cTrainingConfig

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute VFM loss + diversity regularization."""
        # Get base VFM loss from v1a (includes Min-SNR, adaptive scaling, KL, obs)
        total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        cfg = self.config
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        if not use_adapter or adapter_mu is None:
            return total_loss

        # Warmup: linearly ramp diversity weight from 0 → 1
        if cfg.diversity_warmup_steps > 0 and self._current_step < cfg.diversity_warmup_steps:
            warmup_factor = self._current_step / cfg.diversity_warmup_steps
        else:
            warmup_factor = 1.0

        # adapter_mu: [B, video_seq, latent_dim]
        B, seq_len, C = adapter_mu.shape

        # =====================================================
        # 1. Overall token diversity: std(μ) across all tokens
        # =====================================================
        # High std = tokens produce diverse noise = good
        # We maximize by subtracting from loss
        token_std = adapter_mu.std(dim=1).mean()  # [B, seq, C] → std over seq → mean
        loss_token_div = -cfg.diversity_weight * token_std

        # =====================================================
        # 2. Temporal diversity: std(μ) across frames
        # =====================================================
        # Reshape to [B, num_frames, tokens_per_frame, C]
        # Then take mean over spatial → [B, num_frames, C]
        # Then std over frames → [B, C] → scalar
        loss_temporal_div = torch.tensor(0.0, device=adapter_mu.device)
        loss_spatial_div = torch.tensor(0.0, device=adapter_mu.device)

        # Infer num_frames from stored metadata or try to factor seq_len
        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is not None:
            num_frames = raw_latents.shape[2]  # [B, C, F, H, W]
            tokens_per_frame = seq_len // num_frames

            if tokens_per_frame > 0 and num_frames > 1:
                # Reshape: [B, num_frames, tokens_per_frame, C]
                mu_reshaped = adapter_mu[:, :num_frames * tokens_per_frame].reshape(
                    B, num_frames, tokens_per_frame, C
                )

                # Temporal: average over spatial, then std over frames
                mu_per_frame = mu_reshaped.mean(dim=2)  # [B, num_frames, C]
                temporal_std = mu_per_frame.std(dim=1).mean()  # scalar
                loss_temporal_div = -cfg.temporal_diversity_weight * temporal_std

                # Spatial: average over frames, then std over spatial positions
                mu_per_spatial = mu_reshaped.mean(dim=1)  # [B, tokens_per_frame, C]
                spatial_std = mu_per_spatial.std(dim=1).mean()  # scalar
                loss_spatial_div = -cfg.spatial_diversity_weight * spatial_std

        # =====================================================
        # Combine
        # =====================================================
        diversity_loss = warmup_factor * (loss_token_div + loss_temporal_div + loss_spatial_div)
        total_loss = total_loss + diversity_loss

        # Log diversity metrics
        self._last_vfm_metrics.update({
            "vfm/div_token_std": token_std.item(),
            "vfm/div_temporal_std": -loss_temporal_div.item() / max(cfg.temporal_diversity_weight, 1e-8),
            "vfm/div_spatial_std": -loss_spatial_div.item() / max(cfg.spatial_diversity_weight, 1e-8),
            "vfm/div_loss": diversity_loss.item(),
            "vfm/div_warmup": warmup_factor,
        })

        return total_loss
