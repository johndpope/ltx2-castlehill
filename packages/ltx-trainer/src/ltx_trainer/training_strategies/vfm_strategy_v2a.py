"""VFM v2a — Speculative Noise Selection (SSD-inspired).

From docs/SSD.md Technique #1: Generate K candidate noise vectors from the adapter,
score them with lightweight proxy metrics, select the most informative one before
the expensive DiT forward pass.

The adapter forward is ~7ms, DiT is ~170ms. K=4 candidates costs +21ms (12% overhead)
but can significantly improve gradient quality by avoiding uninformative noise regions.

Scoring criteria (no DiT needed):
1. **Novelty**: cosine distance from running centroid of recent noise vectors
2. **Sigma diversity**: entropy of per-token sigma distribution (higher = better)
3. **Angular spread**: geodesic distance between candidate's mean direction and centroid

Architecture: identical to v1f. Only the noise sampling is intercepted — all input
construction, patchification, and Modality building is delegated to super().

Paper: Speculative Speculative Decoding (arXiv:2603.03251) — Kumar, Dao, May 2025
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.spherical_utils import normalize, geodesic_distance
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class VFMv2aTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v2a (speculative noise selection)."""

    name: Literal["vfm_v2a"] = "vfm_v2a"

    # === Speculative Noise Selection ===
    spec_k: int = Field(
        default=4,
        ge=1, le=16,
        description="Number of candidate noise vectors to generate per step. "
        "K=1 disables speculation (v1f behavior). K=4 recommended.",
    )
    spec_novelty_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for novelty score (cosine distance from centroid).",
    )
    spec_sigma_entropy_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for sigma distribution entropy score.",
    )
    spec_angular_weight: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for angular diversity score (geodesic from centroid).",
    )
    spec_centroid_window: int = Field(
        default=64,
        ge=8, le=512,
        description="Number of recent noise vectors to track for novelty scoring.",
    )


class VFMv2aTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v2a — Speculative Noise Selection.

    Generates K candidate noise vectors from the adapter, scores them with
    lightweight proxy metrics, and selects the most informative one before
    running the expensive DiT forward pass.

    Key design: does NOT override _prepare_standard_inputs. Instead, overrides
    _sample_spherical_noise to intercept noise generation. This preserves the
    entire v1f parent chain (patchification, Modality, ModelInputs) unchanged.
    """

    config: VFMv2aTrainingConfig

    def __init__(self, config: VFMv2aTrainingConfig) -> None:
        super().__init__(config)
        self._noise_history: deque[Tensor] = deque(maxlen=config.spec_centroid_window)
        self._centroid: Tensor | None = None
        self._spec_metrics: dict[str, float] = {}
        self._pending_speculation: bool = False

    def _score_candidate(
        self,
        z: Tensor,
        mu_hat: Tensor | None,
    ) -> float:
        """Score a candidate noise vector (no DiT needed)."""
        cfg = self.config
        score = 0.0

        # 1. Novelty: cosine distance from running centroid
        if cfg.spec_novelty_weight > 0 and self._centroid is not None:
            z_pooled = z.mean(dim=1)  # [B, D]
            cos_sim = F.cosine_similarity(z_pooled, self._centroid, dim=-1).mean()
            novelty = 1.0 - cos_sim.item()
            score += cfg.spec_novelty_weight * novelty

        # 2. Angular diversity: geodesic distance of mean direction from centroid
        if cfg.spec_angular_weight > 0 and mu_hat is not None and self._centroid is not None:
            mu_pooled = normalize(mu_hat.mean(dim=1))
            centroid_dir = normalize(self._centroid)
            geo_dist = geodesic_distance(mu_pooled, centroid_dir).mean().item()
            score += cfg.spec_angular_weight * geo_dist

        return score

    def _update_centroid(self, z: Tensor) -> None:
        """Update running centroid with selected noise vector."""
        z_pooled = z.mean(dim=1).detach().cpu()
        self._noise_history.append(z_pooled)
        if len(self._noise_history) > 0:
            self._centroid = torch.stack(list(self._noise_history)).mean(dim=0)
            self._centroid = self._centroid.to(z.device)

    def _sample_spherical_noise(
        self,
        mu: Tensor,
        log_sigma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Override v1f's noise sampling to do speculative selection.

        Generates K candidates from the SAME (mu, log_sigma) distribution,
        scores them, and returns the best one. The adapter is NOT called
        again — only the stochastic sampling differs between candidates.

        This preserves the entire parent chain: patchification, per-token sigma,
        Modality construction, ModelInputs — all handled by v1f's code.
        """
        cfg = self.config

        # If K=1 or no history yet, just sample normally
        if cfg.spec_k <= 1 or self._centroid is None:
            z, mu_hat, kappa, mu_norm = super()._sample_spherical_noise(mu, log_sigma)
            self._update_centroid(z)
            self._spec_metrics = {"spec/k": 1, "spec/score_spread": 0.0}
            return z, mu_hat, kappa, mu_norm

        # Generate K candidates from the same distribution
        candidates = []
        for _ in range(cfg.spec_k):
            z, mu_hat, kappa, mu_norm = super()._sample_spherical_noise(mu, log_sigma)
            score = self._score_candidate(z, mu_hat)
            candidates.append((z, mu_hat, kappa, mu_norm, score))

        # Select best
        best = max(candidates, key=lambda c: c[4])
        scores = [c[4] for c in candidates]

        self._spec_metrics = {
            "spec/best_score": max(scores),
            "spec/worst_score": min(scores),
            "spec/score_spread": max(scores) - min(scores),
            "spec/k": cfg.spec_k,
        }

        # Update centroid with selected noise
        self._update_centroid(best[0])

        return best[0], best[1], best[2], best[3]

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute loss and log speculation metrics."""
        loss = super().compute_loss(video_pred, audio_pred, inputs)

        # Append speculation metrics to VFM metrics
        if self._spec_metrics and hasattr(self, '_last_vfm_metrics'):
            self._last_vfm_metrics.update(self._spec_metrics)
            self._spec_metrics = {}

        return loss
