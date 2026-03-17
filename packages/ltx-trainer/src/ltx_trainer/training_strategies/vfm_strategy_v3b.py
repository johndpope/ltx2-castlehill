"""VFM v3b — Self-Evaluating Model (Self-E) for 1-step Video Generation.

Replaces v3a's GAN discriminator with Self-E's self-evaluation mechanism
(arXiv 2512.22374, Yu et al. 2025). The model evaluates its own generated
samples using its current score estimates — no discriminator, no teacher ODE.

Key insight (Self-E):
- The model runs TWO stop-gradient forward passes on its own output:
  (1) G_θ(x̂_s, s, c) — conditional denoising
  (2) G_θ(x̂_s, s, φ) — unconditional denoising (null text)
- The difference (2)-(1) = classifier score ∝ ∇log q(c|x̂_s)
- This pushes generated samples toward text-aligned, high-density regions
- No extra network needed — the model IS its own teacher

Architecture:
    Text embeddings -> NoiseAdapterV1b -> Spherical Cauchy noise z  (from v1f)
    z -> 48-layer DiT (LoRA) -> velocity v -> x̂₀ = z - v
    Re-noise x̂₀ -> x̂_s at random s
    G_θ(x̂_s, s, φ) - G_θ(x̂_s, s, c) = classifier score feedback
    x_self = sg[x̂₀ - classifier_score]

Loss = ||x̂₀ - x_renorm||² where x_renorm blends x₀ + λ*x_self (energy-preserving)
     + kl_weight * L_KL  (adapter prior)
     + flow_match_weight * L_flow  (temporal consistency, from v3a)

Advantages over v3a:
- No 18M-param discriminator or its optimizer (saves VRAM + compute)
- Self-evaluation improves as model improves (dynamic self-teacher)
- Energy-preserving normalization prevents color bias/saturation
- Naturally supports any-step inference (not just 1-step)
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
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)


class SelfEVFMv3bTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v3b (Self-E self-evaluation + VFM)."""

    name: Literal["vfm_v3b"] = "vfm_v3b"

    # === Self-evaluation settings ===
    self_eval_weight: float = Field(
        default=1.0, ge=0.0,
        description="Base weight for self-evaluation loss. "
        "Actual weight is λ_{s,t} * self_eval_weight where λ = σ_t/α_t - σ_s/α_s.",
    )
    self_eval_cfg_scale: float = Field(
        default=5.0, ge=1.0,
        description="Classifier-free guidance scale for self-evaluation. "
        "Higher = stronger text alignment signal. Paper default: 5.0.",
    )
    self_eval_s_range: tuple[float, float] = Field(
        default=(0.1, 0.5),
        description="Range [s_min, s_max] for random re-noising level s during self-evaluation. "
        "s=0 is clean, s=1 is pure noise. Paper anneals from near-t to 0 over training.",
    )
    energy_preserving_norm: bool = Field(
        default=True,
        description="Apply energy-preserving normalization to combined target (Eq. 19 in Self-E). "
        "Prevents color bias from large λ values.",
    )
    use_ema_for_eval: bool = Field(
        default=False,
        description="Use EMA model for conditional branch in self-evaluation. "
        "Paper uses EMA for conditional, non-EMA for unconditional. "
        "Disabled by default (we don't maintain separate EMA in trainer).",
    )
    self_eval_prob: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Probability of applying self-evaluation (vs pure data loss) per step. "
        "Paper uses 0.5 (s sampled with 50% probability of s=t).",
    )

    # === Reconstruction regularizer (from v3a, optional) ===
    recon_weight: float = Field(
        default=0.0, ge=0.0,
        description="Weight for explicit reconstruction loss |x̂₀ - x₀|². "
        "Self-E's data loss already includes this, so typically 0.",
    )

    # === Flow Distribution Matching (from v3a, kept for temporal consistency) ===
    flow_match_weight: float = Field(
        default=0.0, ge=0.0,
        description="Weight for flow distribution matching loss (DiagDistill). "
        "Matches frame-to-frame latent diffs. 0 = disabled.",
    )

    # Unused v3a fields (kept for config compat)
    fake_score_lr: float = Field(default=1e-4, description="Unused.")
    fake_score_hidden_dim: int = Field(default=256, description="Unused.")
    fake_score_num_heads: int = Field(default=4, description="Unused.")
    fake_score_num_layers: int = Field(default=4, description="Unused.")
    dmd_sigma: float = Field(default=0.5, description="Unused.")
    dmd_weight: float = Field(default=1.0, description="Unused.")
    score_update_ratio: int = Field(default=1, description="Unused.")
    cache_teacher_outputs: bool = Field(default=False, description="Unused.")


class SelfEVFMv3bTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v3b — Self-Evaluating Model for 1-step video generation.

    Uses Self-E's self-evaluation mechanism instead of a GAN discriminator.
    The model evaluates its own generated samples using conditional vs
    unconditional forward passes (classifier score), providing distribution
    matching feedback without any auxiliary network.

    Requires (set by trainer.py):
    - set_transformer(): reference to transformer for self-evaluation passes
    """

    config: SelfEVFMv3bTrainingConfig

    def __init__(self, config: SelfEVFMv3bTrainingConfig):
        super().__init__(config)
        self._transformer_ref: nn.Module | None = None

    def set_transformer(self, transformer: nn.Module, grad_accumulation_steps: int = 1) -> None:
        """Store transformer reference for self-evaluation passes."""
        self._transformer_ref = transformer
        logger.info("VFM v3b (Self-E): Transformer reference set")

    def _self_evaluate(
        self,
        x_hat_0: Tensor,
        text_embeds: Tensor,
        text_mask: Tensor | None,
        positions: Tensor,
        s: float,
    ) -> Tensor:
        """Self-E classifier score: ∇log q(c|x̂_s) via two stop-gradient passes.

        Re-noises x̂₀ to x̂_s, then runs two forward passes:
        - G_θ(x̂_s, s, c) — conditional (what the model thinks clean looks like given text)
        - G_θ(x̂_s, s, φ) — unconditional (what the model thinks clean looks like without text)

        Returns pseudo-target: x_self = sg[x̂₀ - (G_θ(x̂_s, s, φ) - G_θ(x̂_s, s, c))]
        """
        from ltx_core.model.transformer.modality import Modality  # noqa: PLC0415

        B = x_hat_0.shape[0]
        seq_len = x_hat_0.shape[1]
        device = x_hat_0.device
        dtype = x_hat_0.dtype

        # Re-noise x̂₀ at noise level s: x̂_s = (1-s)*x̂₀ + s*ε
        eps = torch.randn_like(x_hat_0)
        x_hat_s = (1 - s) * x_hat_0.detach() + s * eps

        # Per-token timesteps at noise level s
        sigma_batch = torch.full((B,), s, device=device, dtype=dtype)
        timesteps = sigma_batch.unsqueeze(1).expand(-1, seq_len)

        # Conditional pass: G_θ(x̂_s, s, c)
        video_cond = Modality(
            enabled=True, latent=x_hat_s, sigma=sigma_batch,
            timesteps=timesteps, positions=positions,
            context=text_embeds, context_mask=text_mask,
        )

        # Unconditional pass: G_θ(x̂_s, s, φ)  — zeroed text embeddings
        null_embeds = torch.zeros_like(text_embeds)
        video_uncond = Modality(
            enabled=True, latent=x_hat_s, sigma=sigma_batch,
            timesteps=timesteps, positions=positions,
            context=null_embeds, context_mask=text_mask,
        )

        with torch.no_grad():
            # Conditional: predicts velocity → x̂₀ = x̂_s - s * v_cond
            result_cond = self._transformer_ref(video=video_cond, audio=None, perturbations=None)
            # Handle both tuple and Modality returns
            rc = result_cond[0] if isinstance(result_cond, tuple) else result_cond
            v_cond = rc.x if hasattr(rc, 'x') else rc
            g_cond = x_hat_s - s * v_cond  # Denoised estimate (conditional)

            # Unconditional: predicts velocity → x̂₀ = x̂_s - s * v_uncond
            result_uncond = self._transformer_ref(video=video_uncond, audio=None, perturbations=None)
            ru = result_uncond[0] if isinstance(result_uncond, tuple) else result_uncond
            v_uncond = ru.x if hasattr(ru, 'x') else ru
            g_uncond = x_hat_s - s * v_uncond  # Denoised estimate (unconditional)

        # Classifier score direction: uncond - cond
        # (pushes toward high p(c|x) — text-aligned regions)
        classifier_score = g_uncond - g_cond

        # Pseudo-target (Eq. 14 in Self-E paper)
        x_self = (x_hat_0 - classifier_score).detach()

        return x_self

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute Self-E loss: data loss + self-evaluation + optional KL/flow.

        Loss structure:
        1. Data loss: ||x̂₀ - x₀||² (always)
        2. Self-evaluation: ||x̂₀ - x_self||² with energy-preserving norm (probabilistic)
        3. KL: adapter prior loss (from v1f)
        4. Flow matching: temporal consistency (optional, from v3a)
        """
        cfg = self.config
        step = self._current_step
        device = video_pred.device
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        # Fall back to v1f if no adapter or transformer not set
        if not use_adapter or self._transformer_ref is None:
            return super().compute_loss(video_pred, audio_pred, inputs)

        video_noise = inputs._vfm_video_noise  # z (adapter noise)
        video_latents = inputs._vfm_video_latents  # x₀ (ground truth)
        text_embeds = inputs.video.context
        text_mask = inputs.video.context_mask
        positions = inputs.video.positions

        # Student prediction: x̂₀ = z - v_pred (flow matching convention)
        x_hat_0 = video_noise - video_pred  # [B, seq, 128]

        # ════════════════════════════════════════════════════════════
        # DATA LOSS: ||x̂₀ - x₀||² (Eq. 7 in Self-E)
        # ════════════════════════════════════════════════════════════
        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            data_loss = ((x_hat_0 - video_latents).pow(2) * mask).sum() / mask.sum().clamp(min=1) / x_hat_0.shape[-1]
        else:
            data_loss = (x_hat_0 - video_latents).pow(2).mean()

        total_loss = data_loss

        # ════════════════════════════════════════════════════════════
        # SELF-EVALUATION LOSS (Eq. 15-20 in Self-E)
        # ════════════════════════════════════════════════════════════
        apply_self_eval = (
            cfg.self_eval_weight > 0
            and random.random() < cfg.self_eval_prob
            and step > 0  # Skip step 0 (model hasn't learned anything yet)
        )

        loss_self_eval = torch.tensor(0.0, device=device)
        if apply_self_eval:
            # Sample random noise level s for re-noising
            s_min, s_max = cfg.self_eval_s_range
            s = random.uniform(s_min, s_max)

            # Flow matching coefficients: α_t = 1-t, σ_t = t
            # For VFM 1-step: t ≈ 1 (starting from pure noise)
            t = 1.0  # VFM operates at t=1 (full noise → clean)
            alpha_t, sigma_t = 1 - t + 1e-6, t  # avoid div by zero
            alpha_s, sigma_s = 1 - s, s

            # λ_{s,t} = σ_t/α_t - σ_s/α_s (Eq. 17)
            lambda_st = sigma_t / alpha_t - sigma_s / alpha_s
            lambda_st = min(lambda_st, 10.0)  # Clamp to prevent explosion

            # Get self-evaluation pseudo-target
            x_self = self._self_evaluate(
                x_hat_0, text_embeds, text_mask, positions, s,
            )

            if cfg.energy_preserving_norm:
                # Energy-preserving normalization (Eq. 19)
                # x_renorm = (x₀ + λ*x_self) / ||x₀ + λ*x_self|| * ||x₀||
                combined = video_latents + lambda_st * x_self
                x0_norm = video_latents.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                combined_norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                x_target = combined / combined_norm * x0_norm
            else:
                # Simple weighted average (Eq. 18)
                x_target = (video_latents + lambda_st * x_self) / (1 + lambda_st)

            # Self-evaluation loss
            loss_self_eval = (x_hat_0 - x_target).pow(2).mean()
            total_loss = loss_self_eval  # Replace data loss with combined target

        # ════════════════════════════════════════════════════════════
        # ADAPTER KL (from v1f)
        # ════════════════════════════════════════════════════════════
        adapter_kl = getattr(inputs, "_vfm_kl_loss", torch.tensor(0.0, device=device))
        if cfg.kl_weight > 0:
            total_loss = total_loss + cfg.kl_weight * adapter_kl

        # ════════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════════
        if step % 20 == 0:
            try:
                import wandb  # noqa: PLC0415
                if wandb.run is not None:
                    log_data = {
                        "v3b/loss_data": data_loss.item(),
                        "v3b/loss_self_eval": loss_self_eval.item() if isinstance(loss_self_eval, Tensor) else loss_self_eval,
                        "v3b/loss_kl": adapter_kl.item() if isinstance(adapter_kl, Tensor) else adapter_kl,
                        "v3b/loss_total": total_loss.item(),
                        "v3b/self_eval_applied": float(apply_self_eval),
                        "v3b/student_recon_mse": (x_hat_0 - video_latents).pow(2).mean().item(),
                    }
                    # Don't pass step= to avoid conflicting with trainer's wandb.log calls
                    wandb.log(log_data)
            except Exception:
                pass

        return total_loss

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log v1f reconstructions + Self-E specific metrics."""
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )
        return log_dict
