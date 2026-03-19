"""VFM v1g — HyperSphereDiff-inspired hybrid noise adapter.

Branches from v1f (spherical Cauchy + per-token sigma) and adds insights from
HyperSphereDiff (Dosi et al., ICML 2025, arxiv:2506.10576):

Key changes from v1f:
1. **Hybrid magnitude+direction loss**: Decomposes velocity prediction into
   radial (magnitude) and angular (direction) components. MSE handles magnitude,
   cosine/geodesic loss handles direction. Motivated by trajectory analysis
   showing the teacher ODE is 87% angular in mid-σ range.

2. **Cosine loss (L_cos)**: 1 - cos(v_pred, v_target) — penalizes directional
   misalignment without over-weighting large-magnitude errors. From HyperSphereDiff §4.2.

3. **Geodesic loss (L_geo)**: arccos²(cos(v_pred, v_target)) — penalizes angular
   deviations quadratically, stronger than cosine for large errors.

4. **Adaptive radius head**: Explicit MLP predicting per-token noise magnitude
   r_φ(μ) separate from direction. The hybrid approach (Gaussian magnitude +
   spherical direction) achieves best FID per HyperSphereDiff Table 2.

5. **Phase-aware loss weighting**: Trajectory analysis showed 3 phases:
   - Early (σ≈1): 78% radial → weight MSE higher
   - Mid (σ≈0.5): 87% angular → weight cosine/geodesic higher
   - Late (σ≈0): 58% radial → weight MSE higher
   Per-token sigma from SigmaHead naturally enables this.

Architecture:
    Text → NoiseAdapterV1b → (μ, log_σ) per token
    μ̂ = normalize(μ), κ = exp(mean(log_σ))        [from v1f]
    r = RadiusHead(μ) → learned per-token magnitude  [NEW in v1g]
    z_dir ~ SphericalCauchy(μ̂, κ)
    z = r · z_dir                                     [explicit radius]
    z → SigmaHead → per-token σ_i
    48-layer DiT → velocity v → x̂₀

Loss:
    L = λ_mse·L_mse + λ_cos·L_cos + λ_geo·L_geo + L_kl + L_div + L_sigma + L_mag + L_kappa
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.spherical_utils import (
    geodesic_distance,
    normalize,
)
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class RadiusHead(nn.Module):
    """Predicts per-token noise magnitude from adapter features.

    Separates magnitude from direction — the hybrid approach from
    HyperSphereDiff Table 2 (Gaussian magnitude + spherical direction).

    Architecture: μ → LayerNorm → Linear → SiLU → Linear → softplus → r
    Output r ∈ [r_min, r_max] via softplus + clamping.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        r_min: float = 0.1,
        r_max: float = 20.0,
        init_radius: float = 1.0,
    ):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize to produce target radius via softplus
        # softplus(x) ≈ x for x >> 0, so bias ≈ init_radius
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_radius)

    def forward(self, mu: Tensor) -> Tensor:
        """Predict per-token radius from adapter mean.

        Args:
            mu: [B, seq, D] adapter mean output

        Returns:
            r: [B, seq] positive per-token radius
        """
        r = F.softplus(self.net(mu).squeeze(-1))  # [B, seq]
        return r.clamp(min=self.r_min, max=self.r_max)


class VFMv1gTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v1g (HyperSphereDiff hybrid losses)."""

    name: Literal["vfm_v1g"] = "vfm_v1g"

    # === Hybrid loss weights ===
    cosine_loss_weight: float = Field(
        default=0.5, ge=0.0,
        description="Weight for cosine direction loss: 1 - cos(v_pred, v_target). "
        "Captures angular alignment without over-weighting magnitude errors.",
    )
    geodesic_loss_weight: float = Field(
        default=0.1, ge=0.0,
        description="Weight for geodesic loss: arccos²(cos(v_pred, v_target)). "
        "Stronger than cosine for large angular errors. From HyperSphereDiff §4.2.",
    )
    mse_loss_weight: float = Field(
        default=1.0, ge=0.0,
        description="Weight for standard MSE loss (magnitude-sensitive). "
        "Set to 1.0 to keep backward compatibility with v1f.",
    )

    # === Phase-aware loss weighting ===
    phase_aware_weighting: bool = Field(
        default=True,
        description="Scale cosine/geodesic vs MSE based on per-token sigma. "
        "Higher σ → more angular weight (mid-trajectory is 87%% angular). "
        "Lower σ → more MSE weight (endpoints are magnitude-dominated).",
    )
    angular_phase_scale: float = Field(
        default=2.0, gt=0.0,
        description="Peak scaling factor for angular losses at mid-sigma. "
        "At σ=0.5 (peak angular movement), cosine/geo loss is scaled by this.",
    )

    # === Explicit radius head ===
    use_radius_head: bool = Field(
        default=True,
        description="Use explicit RadiusHead MLP for per-token noise magnitude "
        "instead of using ||μ|| directly. Hybrid approach from HyperSphereDiff.",
    )
    radius_hidden_dim: int = Field(
        default=128, ge=32,
        description="Hidden dimension of RadiusHead MLP.",
    )
    radius_min: float = Field(
        default=0.1, gt=0.0,
        description="Minimum noise radius (prevents magnitude collapse).",
    )
    radius_max: float = Field(
        default=20.0, gt=0.0,
        description="Maximum noise radius.",
    )
    init_radius: float = Field(
        default=1.0, gt=0.0,
        description="Initial target radius for RadiusHead.",
    )
    radius_reg_weight: float = Field(
        default=0.05, ge=0.0,
        description="Weight for radius regularization toward target magnitude. "
        "Replaces v1f's magnitude_reg_weight when radius head is active.",
    )

    # === Hypercone metrics ===
    log_hypercone_metrics: bool = Field(
        default=True,
        description="Log HyperSphereDiff-style hypercone metrics: "
        "angular spread, concentration ratio, cone coverage.",
    )


class VFMv1gTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v1g — HyperSphereDiff hybrid magnitude+direction losses.

    Extends v1f with:
    - Cosine + geodesic directional losses
    - Explicit RadiusHead for magnitude prediction
    - Phase-aware loss weighting based on per-token sigma
    """

    config: VFMv1gTrainingConfig
    _radius_head: RadiusHead | None = None

    def __init__(self, config: VFMv1gTrainingConfig) -> None:
        super().__init__(config)
        self._radius_head = None

    def initialize(
        self,
        transformer: nn.Module,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize strategy + radius head."""
        super().initialize(transformer, device, dtype)

        cfg = self.config
        if cfg.use_radius_head:
            self._radius_head = RadiusHead(
                input_dim=128,  # latent dim
                hidden_dim=cfg.radius_hidden_dim,
                r_min=cfg.radius_min,
                r_max=cfg.radius_max,
                init_radius=cfg.init_radius,
            ).to(device)
            param_count = sum(p.numel() for p in self._radius_head.parameters())
            logger.info(f"  RadiusHead initialized ({param_count:,} params)")
        else:
            self._radius_head = None

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return all trainable params including RadiusHead."""
        params = super().get_trainable_parameters()
        if self._radius_head is not None:
            params.extend(self._radius_head.parameters())
        return params

    def _sample_spherical_noise(
        self,
        mu: Tensor,
        log_sigma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample noise with explicit radius head (v1g override).

        If use_radius_head=True, uses RadiusHead(mu) for magnitude instead of ||mu||.
        Direction and kappa handling inherited from v1f.
        """
        cfg = self.config

        if not cfg.use_radius_head or self._radius_head is None:
            # Fall back to v1f behavior
            return super()._sample_spherical_noise(mu, log_sigma)

        from ltx_trainer.spherical_utils import sample_spherical_cauchy

        B, seq, D = mu.shape

        # Direction: normalize mu to unit sphere
        mu_hat = normalize(mu, dim=-1)

        # Magnitude: explicit RadiusHead instead of ||mu||
        mu_norm = self._radius_head(mu.float()).to(mu.dtype)  # [B, seq]

        # Concentration: scalar kappa per token from log_sigma
        kappa = torch.exp(log_sigma.mean(dim=-1))
        kappa = kappa.clamp(min=cfg.kappa_min, max=cfg.kappa_max)

        # Sample direction from Spherical Cauchy
        mu_hat_flat = mu_hat.reshape(B * seq, D)
        kappa_flat = kappa.reshape(B * seq)
        z_dir_flat = sample_spherical_cauchy(mu_hat_flat, kappa_flat)
        z_dir = z_dir_flat.reshape(B, seq, D)

        # Scale by learned radius
        z = mu_norm.unsqueeze(-1) * z_dir

        return z, mu_hat, kappa, mu_norm

    # ════════════════════════════════════════════════════════════
    # HYBRID LOSSES
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _cosine_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        """Cosine direction loss: 1 - cos(pred, target).

        From HyperSphereDiff Eq. L_c = 1 - E[cos(score, noise)].
        Measures directional misalignment independent of magnitude.
        """
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [B, seq]
        loss = 1.0 - cos_sim  # [B, seq], range [0, 2]

        if mask is not None:
            return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss.mean()

    @staticmethod
    def _geodesic_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        """Geodesic loss: arccos²(cos(pred, target)).

        From HyperSphereDiff Eq. L_g = E[arccos²(cos(score, noise))].
        Penalizes large angular errors more strongly than cosine.
        """
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        cos_sim = cos_sim.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cos_sim)  # [B, seq], radians
        loss = angle.pow(2)

        if mask is not None:
            return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss.mean()

    @staticmethod
    def _phase_weight(sigma: Tensor, peak_scale: float = 2.0) -> Tensor:
        """Compute phase-aware weight for angular losses.

        Trajectory analysis showed:
        - σ≈1.0: 78% radial → low angular weight
        - σ≈0.5: 87% angular → high angular weight (peak)
        - σ≈0.0: 58% radial → low angular weight

        Uses a bell curve centered at σ=0.5:
            w(σ) = 1 + (peak_scale - 1) * exp(-((σ - 0.5) / 0.2)²)
        """
        bell = torch.exp(-((sigma - 0.5) / 0.2).pow(2))
        return 1.0 + (peak_scale - 1.0) * bell

    def _compute_standard_loss_v1f(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """v1g override: adds cosine + geodesic losses alongside MSE."""
        cfg = self.config

        # ═══════════════════════════════════════════
        # MSE LOSS (magnitude-sensitive)
        # ═══════════════════════════════════════════
        video_loss = (video_pred - inputs.video_targets).pow(2)
        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_mf = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_mf = video_loss.mean()

        total_loss = cfg.mse_loss_weight * loss_mf

        # ═══════════════════════════════════════════
        # COSINE LOSS (direction-sensitive)
        # ═══════════════════════════════════════════
        loss_cos = torch.tensor(0.0, device=video_pred.device)
        loss_geo = torch.tensor(0.0, device=video_pred.device)

        if cfg.cosine_loss_weight > 0 or cfg.geodesic_loss_weight > 0:
            seq_mask = inputs.video_loss_mask if inputs.video_loss_mask is not None else None

            # Phase-aware scaling
            cos_scale = 1.0
            geo_scale = 1.0
            per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
            if cfg.phase_aware_weighting and per_token_sigmas is not None:
                phase_w = self._phase_weight(per_token_sigmas, cfg.angular_phase_scale)
                # Average phase weight for loss scaling (per-token would need reshape)
                if seq_mask is not None:
                    active_w = phase_w[seq_mask].mean().item()
                else:
                    active_w = phase_w.mean().item()
                cos_scale = active_w
                geo_scale = active_w

            if cfg.cosine_loss_weight > 0:
                loss_cos = self._cosine_loss(video_pred, inputs.video_targets, seq_mask)
                total_loss = total_loss + cfg.cosine_loss_weight * cos_scale * loss_cos

            if cfg.geodesic_loss_weight > 0:
                loss_geo = self._geodesic_loss(video_pred, inputs.video_targets, seq_mask)
                total_loss = total_loss + cfg.geodesic_loss_weight * geo_scale * loss_geo

        # ═══════════════════════════════════════════
        # KL, OBS, DIVERSITY (inherited from v1f)
        # ═══════════════════════════════════════════
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        mu_hat = getattr(inputs, "_spherical_mu_hat", None)
        kappa = getattr(inputs, "_spherical_kappa", None)
        mu_norm = getattr(inputs, "_spherical_mu_norm", None)

        loss_kl = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and adapter_mu is not None and cfg.kl_weight > 0:
            if cfg.spherical_noise and mu_hat is not None and kappa is not None:
                kl_per_token = self._compute_spherical_kl(mu_hat, kappa)
                kl_per_sample = torch.clamp(kl_per_token.mean(dim=1), min=0.0)
                if cfg.kl_free_bits > 0:
                    kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)
                loss_kl = kl_per_sample.mean()
            else:
                adapter_log_sigma = inputs._vfm_adapter_log_sigma
                kl_raw = 0.5 * (
                    adapter_mu.pow(2)
                    + torch.exp(2 * adapter_log_sigma)
                    - 2 * adapter_log_sigma
                    - 1
                )
                kl_per_sample = kl_raw.mean(dim=(1, 2))
                if cfg.kl_free_bits > 0:
                    kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)
                loss_kl = kl_per_sample.mean()
            total_loss = total_loss + cfg.kl_weight * loss_kl

        # Observation loss
        loss_obs = torch.tensor(0.0, device=video_pred.device)
        if (
            use_adapter
            and cfg.obs_loss_weight > 0
            and getattr(inputs, "_vfm_observation", None) is not None
        ):
            obs = inputs._vfm_observation
            video_noise = inputs._vfm_video_noise
            pred_x0 = video_noise - video_pred
            noise_level = inputs._vfm_noise_level

            if isinstance(noise_level, (int, float)) and noise_level > 0:
                pred_obs = pred_x0 + torch.randn_like(pred_x0) * noise_level
            else:
                pred_obs = pred_x0

            obs_diff = (pred_obs - obs).pow(2)
            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                loss_obs = (obs_diff * mask).sum() / mask.sum().clamp(min=1) / obs_diff.shape[-1]
            else:
                loss_obs = obs_diff.mean()
            total_loss = total_loss + cfg.obs_loss_weight * loss_obs

        # Diversity
        if use_adapter and adapter_mu is not None:
            div_loss = self._compute_diversity_loss(adapter_mu, inputs)
            total_loss = total_loss + div_loss

        # Sigma entropy + complexity-aware pull
        per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
        loss_sigma_entropy = torch.tensor(0.0, device=video_pred.device)
        loss_sigma_pull = torch.tensor(0.0, device=video_pred.device)
        if per_token_sigmas is not None:
            if cfg.sigma_entropy_weight > 0:
                loss_sigma_entropy = self._compute_sigma_entropy_loss(per_token_sigmas, inputs)
                total_loss = total_loss + cfg.sigma_entropy_weight * loss_sigma_entropy
            if cfg.sigma_mean_pull_weight > 0:
                complexity_targets = getattr(inputs, "_sigma_complexity_targets", None)
                if complexity_targets is not None:
                    loss_mask = getattr(inputs, "video_loss_mask", None)
                    if loss_mask is not None:
                        loss_sigma_pull = (per_token_sigmas[loss_mask] - complexity_targets[loss_mask]).pow(2).mean()
                    else:
                        loss_sigma_pull = (per_token_sigmas - complexity_targets).pow(2).mean()
                else:
                    loss_mask = getattr(inputs, "video_loss_mask", None)
                    active_sigmas = per_token_sigmas[loss_mask] if loss_mask is not None else per_token_sigmas.flatten()
                    if active_sigmas.numel() > 0:
                        loss_sigma_pull = (active_sigmas.mean() - cfg.sigma_mean_target).pow(2)
                total_loss = total_loss + cfg.sigma_mean_pull_weight * loss_sigma_pull

        # ═══════════════════════════════════════════
        # RADIUS REGULARIZATION (v1g — replaces v1f magnitude reg)
        # ═══════════════════════════════════════════
        loss_radius = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and mu_norm is not None:
            if cfg.use_radius_head and cfg.radius_reg_weight > 0:
                loss_radius = (mu_norm - cfg.target_magnitude).pow(2).mean()
                total_loss = total_loss + cfg.radius_reg_weight * loss_radius
            elif cfg.magnitude_reg_weight > 0:
                loss_radius = (mu_norm - cfg.target_magnitude).pow(2).mean()
                total_loss = total_loss + cfg.magnitude_reg_weight * loss_radius

        # Kappa regularization (inherited from v1f)
        loss_kappa_pull = torch.tensor(0.0, device=video_pred.device)
        loss_kappa_entropy = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and kappa is not None:
            if cfg.kappa_pull_weight > 0:
                loss_kappa_pull = (kappa.mean() - cfg.kappa_target).pow(2)
                total_loss = total_loss + cfg.kappa_pull_weight * loss_kappa_pull
            if cfg.kappa_entropy_weight > 0 and kappa.numel() > 1:
                loss_kappa_entropy = -kappa.std()
                total_loss = total_loss + cfg.kappa_entropy_weight * loss_kappa_entropy

        # ═══════════════════════════════════════════
        # LOGGING
        # ═══════════════════════════════════════════
        self._last_vfm_metrics = {
            "vfm/loss_mf": loss_mf.item(),
            "vfm/loss_cos": loss_cos.item(),
            "vfm/loss_geo": loss_geo.item(),
            "vfm/loss_kl": loss_kl.item(),
            "vfm/loss_obs": loss_obs.item(),
            "vfm/loss_radius": loss_radius.item(),
            "vfm/loss_total": total_loss.item(),
            "vfm/use_adapter": float(use_adapter),
        }

        if adapter_mu is not None:
            self._last_vfm_metrics["vfm/adapter_mu_mean"] = adapter_mu.mean().item()
            self._last_vfm_metrics["vfm/adapter_sigma_mean"] = torch.exp(
                inputs._vfm_adapter_log_sigma
            ).mean().item()

        if mu_hat is not None and kappa is not None:
            self._last_vfm_metrics.update({
                "vfm/kappa_mean": kappa.mean().item(),
                "vfm/kappa_std": kappa.std().item(),
                "vfm/kappa_min": kappa.min().item(),
                "vfm/kappa_max": kappa.max().item(),
                "vfm/radius_mean": mu_norm.mean().item(),
                "vfm/radius_std": mu_norm.std().item(),
                "vfm/loss_kappa_pull": loss_kappa_pull.item(),
                "vfm/loss_kappa_entropy": loss_kappa_entropy.item(),
            })

            # Hypercone metrics (angular spread analysis)
            if cfg.log_hypercone_metrics and mu_hat.shape[0] > 0:
                self._log_hypercone_metrics(mu_hat, kappa, mu_norm)

            # Geodesic diversity
            n_sample = min(64, mu_hat.shape[1])
            idx = torch.randperm(mu_hat.shape[1])[:n_sample]
            mu_sample = mu_hat[0, idx]
            if n_sample > 1:
                geo_dist = geodesic_distance(mu_sample[:-1], mu_sample[1:]).mean()
                self._last_vfm_metrics["vfm/geodesic_diversity"] = geo_dist.item()

        if per_token_sigmas is not None:
            active = per_token_sigmas[per_token_sigmas > 0]
            if active.numel() > 0:
                self._last_vfm_metrics.update({
                    "vfm/sigma_mean": active.mean().item(),
                    "vfm/sigma_std": active.std().item(),
                    "vfm/sigma_entropy": loss_sigma_entropy.item(),
                    "vfm/loss_sigma_pull": loss_sigma_pull.item(),
                })

        return total_loss

    def _compute_distill_loss_v1f(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Distillation loss with hybrid direction+magnitude losses (v1g override)."""
        cfg = self.config

        # L_distill: MSE between student velocity and teacher target
        video_loss = (video_pred - inputs.video_targets).pow(2)
        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_distill = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_distill = video_loss.mean()

        total_loss = cfg.distill_weight * cfg.mse_loss_weight * loss_distill

        # Cosine + geodesic on distillation targets
        loss_cos = torch.tensor(0.0, device=video_pred.device)
        loss_geo = torch.tensor(0.0, device=video_pred.device)
        seq_mask = inputs.video_loss_mask if inputs.video_loss_mask is not None else None

        if cfg.cosine_loss_weight > 0:
            loss_cos = self._cosine_loss(video_pred, inputs.video_targets, seq_mask)
            total_loss = total_loss + cfg.distill_weight * cfg.cosine_loss_weight * loss_cos

        if cfg.geodesic_loss_weight > 0:
            loss_geo = self._geodesic_loss(video_pred, inputs.video_targets, seq_mask)
            total_loss = total_loss + cfg.distill_weight * cfg.geodesic_loss_weight * loss_geo

        # KL (spherical or Gaussian)
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        mu_hat = getattr(inputs, "_spherical_mu_hat", None)
        kappa = getattr(inputs, "_spherical_kappa", None)
        mu_norm = getattr(inputs, "_spherical_mu_norm", None)
        loss_kl = torch.tensor(0.0, device=video_pred.device)

        if use_adapter and adapter_mu is not None and cfg.kl_weight > 0:
            if cfg.spherical_noise and mu_hat is not None and kappa is not None:
                kl_per_token = self._compute_spherical_kl(mu_hat, kappa)
                kl_per_sample = torch.clamp(kl_per_token.mean(dim=1), min=0.0)
                if cfg.kl_free_bits > 0:
                    kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)
                loss_kl = kl_per_sample.mean()
            else:
                adapter_log_sigma = inputs._vfm_adapter_log_sigma
                kl_raw = 0.5 * (
                    adapter_mu.pow(2)
                    + torch.exp(2 * adapter_log_sigma)
                    - 2 * adapter_log_sigma
                    - 1
                )
                kl_per_sample = kl_raw.mean(dim=(1, 2))
                if cfg.kl_free_bits > 0:
                    kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)
                loss_kl = kl_per_sample.mean()
            total_loss = total_loss + cfg.kl_weight * loss_kl

        # GT loss
        loss_gt = torch.tensor(0.0, device=video_pred.device)
        if cfg.gt_weight > 0 and hasattr(inputs, "_gt_video_targets"):
            gt_loss = (video_pred - inputs._gt_video_targets).pow(2)
            if inputs.video_loss_mask is not None:
                gt_loss = (gt_loss * mask).sum() / mask.sum().clamp(min=1) / gt_loss.shape[-1]
            else:
                gt_loss = gt_loss.mean()
            loss_gt = gt_loss
            total_loss = total_loss + cfg.gt_weight * loss_gt

        # Diversity
        if use_adapter and adapter_mu is not None:
            div_loss = self._compute_diversity_loss(adapter_mu, inputs)
            total_loss = total_loss + div_loss

        # Sigma entropy + complexity-aware pull
        per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
        loss_sigma_entropy = torch.tensor(0.0, device=video_pred.device)
        loss_sigma_pull = torch.tensor(0.0, device=video_pred.device)
        if per_token_sigmas is not None:
            if cfg.sigma_entropy_weight > 0:
                loss_sigma_entropy = self._compute_sigma_entropy_loss(per_token_sigmas, inputs)
                total_loss = total_loss + cfg.sigma_entropy_weight * loss_sigma_entropy
            if cfg.sigma_mean_pull_weight > 0:
                complexity_targets = getattr(inputs, "_sigma_complexity_targets", None)
                if complexity_targets is not None:
                    loss_mask = getattr(inputs, "video_loss_mask", None)
                    if loss_mask is not None:
                        loss_sigma_pull = (per_token_sigmas[loss_mask] - complexity_targets[loss_mask]).pow(2).mean()
                    else:
                        loss_sigma_pull = (per_token_sigmas - complexity_targets).pow(2).mean()
                else:
                    loss_mask = getattr(inputs, "video_loss_mask", None)
                    active_sigmas = per_token_sigmas[loss_mask] if loss_mask is not None else per_token_sigmas.flatten()
                    if active_sigmas.numel() > 0:
                        loss_sigma_pull = (active_sigmas.mean() - cfg.sigma_mean_target).pow(2)
                total_loss = total_loss + cfg.sigma_mean_pull_weight * loss_sigma_pull

        # Radius regularization
        loss_radius = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and mu_norm is not None:
            w = cfg.radius_reg_weight if cfg.use_radius_head else cfg.magnitude_reg_weight
            if w > 0:
                loss_radius = (mu_norm - cfg.target_magnitude).pow(2).mean()
                total_loss = total_loss + w * loss_radius

        # Kappa regularization
        loss_kappa_pull = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and kappa is not None and cfg.kappa_pull_weight > 0:
            loss_kappa_pull = (kappa.mean() - cfg.kappa_target).pow(2)
            total_loss = total_loss + cfg.kappa_pull_weight * loss_kappa_pull

        # Logging
        self._last_vfm_metrics = {
            "vfm/loss_distill": loss_distill.item(),
            "vfm/loss_cos": loss_cos.item(),
            "vfm/loss_geo": loss_geo.item(),
            "vfm/loss_kl": loss_kl.item(),
            "vfm/loss_gt": loss_gt.item(),
            "vfm/loss_radius": loss_radius.item(),
            "vfm/loss_total": total_loss.item(),
            "vfm/distill_mode": cfg.distill_mode,
            "vfm/use_adapter": float(use_adapter),
        }
        if adapter_mu is not None:
            self._last_vfm_metrics["vfm/adapter_mu_mean"] = adapter_mu.mean().item()
            self._last_vfm_metrics["vfm/adapter_sigma_mean"] = torch.exp(
                inputs._vfm_adapter_log_sigma
            ).mean().item()
        if mu_hat is not None and kappa is not None:
            self._last_vfm_metrics.update({
                "vfm/kappa_mean": kappa.mean().item(),
                "vfm/kappa_std": kappa.std().item(),
                "vfm/radius_mean": mu_norm.mean().item(),
            })
        if per_token_sigmas is not None:
            active = per_token_sigmas[per_token_sigmas > 0]
            if active.numel() > 0:
                self._last_vfm_metrics["vfm/sigma_mean"] = active.mean().item()
                self._last_vfm_metrics["vfm/sigma_std"] = active.std().item()

        return total_loss

    # ════════════════════════════════════════════════════════════
    # HYPERCONE METRICS (from HyperSphereDiff §5)
    # ════════════════════════════════════════════════════════════

    def _log_hypercone_metrics(
        self,
        mu_hat: Tensor,
        kappa: Tensor,
        mu_norm: Tensor,
    ) -> None:
        """Log HyperSphereDiff-style hypercone analysis.

        Measures:
        - Angular spread: how spread out adapter directions are (should be diverse)
        - Cone coverage: fraction of S^127 covered by the adapter's output cone
        - Radius-direction correlation: do tokens with high κ also have high r?
        """
        with torch.no_grad():
            # Sample tokens from first batch element
            n = min(256, mu_hat.shape[1])
            idx = torch.randperm(mu_hat.shape[1])[:n]
            dirs = mu_hat[0, idx]  # [n, D]
            k = kappa[0, idx]     # [n]
            r = mu_norm[0, idx]   # [n]

            # Mean direction (centroid on sphere)
            centroid = normalize(dirs.mean(dim=0, keepdim=True))  # [1, D]

            # Angular spread: mean angle from centroid
            cos_to_centroid = (dirs * centroid).sum(dim=-1).clamp(-1, 1)
            angles_to_centroid = torch.acos(cos_to_centroid)
            angular_spread_deg = torch.rad2deg(angles_to_centroid.mean()).item()

            # Cone coverage: using max angle as approximate cone radius
            max_angle_deg = torch.rad2deg(angles_to_centroid.max()).item()

            # Radius-kappa correlation
            if n > 2:
                r_centered = r - r.mean()
                k_centered = k - k.mean()
                corr_num = (r_centered * k_centered).sum()
                corr_den = (r_centered.pow(2).sum() * k_centered.pow(2).sum()).sqrt().clamp(min=1e-8)
                rk_corr = (corr_num / corr_den).item()
            else:
                rk_corr = 0.0

            self._last_vfm_metrics.update({
                "vfm/cone_angular_spread_deg": angular_spread_deg,
                "vfm/cone_max_angle_deg": max_angle_deg,
                "vfm/radius_kappa_corr": rk_corr,
            })
