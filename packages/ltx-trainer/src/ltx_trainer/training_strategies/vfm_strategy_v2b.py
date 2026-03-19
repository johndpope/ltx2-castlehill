"""VFM v2b — Multi-Resolution Speculative Training (SSD-inspired).

From docs/SSD.md Technique #3: Evaluate the SAME noise z at K different sigma
levels in a single batched DiT pass. Train for consistency across all ODE paths.

SSD fans out over K possible verification outcomes. We fan out over K possible
sigma levels (noise amounts). Both use a single batched forward pass for efficiency.

Key insight: the pretrained DiT was trained at ALL sigma levels. By batching
K sigma levels together, we get K× training signal for <2× cost (GPU parallelism).
The consistency loss forces the adapter's noise into "basins of attraction" where
all ODE paths converge to the same x₀.

Why this helps VFM:
- Single-sigma training only teaches "denoise from this exact noise level"
- Multi-sigma training teaches "this noise z works well across ALL noise levels"
- The adapter learns noise that's globally easy to denoise, not just locally

Architecture: identical to v1f. Only compute_loss and prepare_training_inputs change.
The prepare step creates K copies at different sigmas. compute_loss adds consistency.

Paper: Speculative Speculative Decoding (arXiv:2603.03251) — Kumar, Dao, May 2025
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
import torch.nn.functional as F
from dataclasses import replace
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class VFMv2bTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v2b (multi-resolution speculative training)."""

    name: Literal["vfm_v2b"] = "vfm_v2b"

    # === Multi-Resolution Speculative ===
    multi_sigma_levels: list[float] = Field(
        default=[1.0, 0.7, 0.4],
        description="Sigma levels for multi-resolution training. "
        "Each level produces a different x_t from the same noise z. "
        "K=3 recommended (1.5-2x cost for 3x signal).",
    )
    consistency_weight: float = Field(
        default=0.3,
        ge=0.0, le=5.0,
        description="Weight for cross-sigma consistency loss. "
        "Penalizes disagreement between x₀ predictions at different sigma levels.",
    )
    multi_sigma_probability: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Probability of using multi-sigma training vs standard single-sigma. "
        "0.5 = half the steps use multi-sigma, half use standard v1f.",
    )


class VFMv2bTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v2b — Multi-Resolution Speculative Training.

    On multi-sigma steps: creates K noisy inputs at different sigma levels
    from the same adapter noise z. The DiT processes all K in a single
    batched forward pass. Loss = standard flow matching + consistency across paths.

    On standard steps: falls back to v1f behavior (unchanged).
    """

    config: VFMv2bTrainingConfig

    def __init__(self, config: VFMv2bTrainingConfig) -> None:
        super().__init__(config)
        self._multi_sigma_active: bool = False
        self._multi_sigma_data: dict[str, Any] = {}

    def _prepare_standard_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs — sometimes multi-sigma, sometimes standard v1f."""
        cfg = self.config

        # Decide whether to do multi-sigma this step
        use_multi = (
            random.random() < cfg.multi_sigma_probability
            and len(cfg.multi_sigma_levels) > 1
        )

        if not use_multi:
            self._multi_sigma_active = False
            return super()._prepare_standard_inputs(batch, timestep_sampler)

        # ════════════════════════════════════════════════
        # MULTI-SIGMA PATH
        # ════════════════════════════════════════════════
        self._multi_sigma_active = True

        latents = batch["latents"]
        video_latents_raw = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents_raw.shape[2]
        height = video_latents_raw.shape[3]
        width = video_latents_raw.shape[4]

        video_latents = self._video_patchifier.patchify(video_latents_raw)  # [B, seq, C]

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}")
        fps = fps[0].item() if fps is not None else 24.0

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype
        tokens_per_frame = video_seq_len // num_frames

        if self._inverse_problem_sampler is None:
            from ltx_trainer.inverse_problems import InverseProblemSampler
            self._inverse_problem_sampler = InverseProblemSampler(
                problems=self._ip_configs,
                tokens_per_frame=tokens_per_frame,
            )

        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height, width=width,
            device=device,
            first_frame_conditioning_p=cfg.first_frame_conditioning_p,
        )

        # ════════════════════════════════════════════════
        # GENERATE ONE NOISE Z (same for all sigma levels)
        # ════════════════════════════════════════════════
        use_adapter_noise = random.random() < cfg.alpha

        if use_adapter_noise and self._noise_adapter is not None:
            ip_sample = self._inverse_problem_sampler.sample(video_latents)

            mu, log_sigma = self._noise_adapter(
                text_embeddings=video_prompt_embeds.detach(),
                text_mask=prompt_attention_mask.bool(),
                positions=video_positions,
                task_class=ip_sample.task_class,
            )

            if cfg.spherical_noise:
                video_noise, mu_hat, kappa, mu_norm = self._sample_spherical_noise(mu, log_sigma)
            else:
                sigma_adapter = torch.exp(log_sigma)
                eps = torch.randn_like(mu)
                video_noise = mu + sigma_adapter * eps
                mu_hat, kappa, mu_norm = None, None, None

            adapter_mu = mu
            adapter_log_sigma = log_sigma
        else:
            video_noise = torch.randn_like(video_latents)
            adapter_mu, adapter_log_sigma = None, None
            mu_hat, kappa, mu_norm = None, None, None
            use_adapter_noise = False

        # ════════════════════════════════════════════════
        # BATCH K SIGMA LEVELS
        # ════════════════════════════════════════════════
        K = len(cfg.multi_sigma_levels)
        sigma_levels = torch.tensor(cfg.multi_sigma_levels, device=device, dtype=dtype)

        # Replicate inputs K times along batch dim: [K*B, seq, C]
        x0_rep = video_latents.repeat(K, 1, 1)
        z_rep = video_noise.repeat(K, 1, 1)
        positions_rep = video_positions.repeat(K, 1, 1, 1)
        embeds_rep = video_prompt_embeds.repeat(K, 1, 1)
        mask_rep = prompt_attention_mask.repeat(K, 1) if prompt_attention_mask is not None else None
        cond_mask_rep = video_conditioning_mask.repeat(K, 1)

        # Create noisy inputs at each sigma level
        # sigma_batch: [K*B] — each group of B gets the same sigma
        sigma_batch = sigma_levels.repeat_interleave(batch_size)
        sigmas_expanded = sigma_batch.view(K * batch_size, 1, 1)

        noisy_video = (1 - sigmas_expanded) * x0_rep + sigmas_expanded * z_rep

        # Ensure conditioning tokens are clean
        cond_mask_expanded = cond_mask_rep.unsqueeze(-1)
        noisy_video = torch.where(cond_mask_expanded, x0_rep, noisy_video)

        # Velocity target: v = z - x₀ (same for all sigma levels)
        video_targets = z_rep - x0_rep

        # Per-token timesteps: uniform sigma per token (no per-token sigma in multi-sigma mode)
        video_timesteps = self._create_per_token_timesteps(
            cond_mask_rep, sigma_batch,
        )

        # ════════════════════════════════════════════════
        # BUILD MODALITY (batched K*B)
        # ════════════════════════════════════════════════
        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            sigma=sigma_batch,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=positions_rep,
            context=embeds_rep,
            context_mask=mask_rep,
        )

        video_loss_mask = ~cond_mask_rep

        model_inputs = ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

        # Store metadata for consistency loss computation
        model_inputs._vfm_adapter_mu = adapter_mu
        model_inputs._vfm_adapter_log_sigma = adapter_log_sigma
        model_inputs._vfm_task_class = None
        model_inputs._vfm_observation = None
        model_inputs._vfm_task_name = "i2v" if use_adapter_noise else "unconditional"
        model_inputs._vfm_noise_level = 0.0
        model_inputs._vfm_video_noise = video_noise  # original (not replicated)
        model_inputs._vfm_video_latents = video_latents
        model_inputs._vfm_use_adapter = use_adapter_noise
        model_inputs._raw_video_latents = video_latents_raw
        model_inputs._per_token_sigmas = None
        model_inputs._sigma_complexity_targets = None
        model_inputs._distill_mode = "none"
        model_inputs._spherical_mu_hat = mu_hat
        model_inputs._spherical_kappa = kappa
        model_inputs._spherical_mu_norm = mu_norm

        # Multi-sigma metadata for consistency loss
        model_inputs._multi_sigma_K = K
        model_inputs._multi_sigma_B = batch_size
        model_inputs._multi_sigma_levels = sigma_levels

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigma_batch[:batch_size].view(-1, 1)

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute loss with optional multi-sigma consistency."""
        if not self._multi_sigma_active:
            return super().compute_loss(video_pred, audio_pred, inputs)

        cfg = self.config
        K = getattr(inputs, '_multi_sigma_K', 1)
        B = getattr(inputs, '_multi_sigma_B', video_pred.shape[0])
        sigma_levels = getattr(inputs, '_multi_sigma_levels', None)

        if K <= 1 or sigma_levels is None:
            return super().compute_loss(video_pred, audio_pred, inputs)

        # ════════════════════════════════════════════════
        # STANDARD FLOW MATCHING LOSS (on all K*B predictions)
        # ════════════════════════════════════════════════
        video_loss = (video_pred - inputs.video_targets).pow(2)
        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_mf = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_mf = video_loss.mean()

        # ════════════════════════════════════════════════
        # CONSISTENCY LOSS: x₀ predictions should agree across sigma levels
        # ════════════════════════════════════════════════
        # Each sigma level's x₀ prediction: x̂₀ = x_t - σ·v
        # But targets are v = z - x₀, and x_t = (1-σ)x₀ + σz
        # So x̂₀ = z - v_pred (same formula regardless of sigma)
        noise = inputs._vfm_video_noise  # [B, seq, C] (original, not replicated)

        # Split predictions by sigma level: [K, B, seq, C]
        v_preds = video_pred.reshape(K, B, *video_pred.shape[1:])

        # Compute x̂₀ for each sigma level
        # x̂₀ = z - v_pred (noise is the same z for all levels)
        x0_preds = []
        for k in range(K):
            x0_hat = noise - v_preds[k]  # [B, seq, C]
            x0_preds.append(x0_hat)

        # Pairwise consistency: all x̂₀ predictions should match
        loss_consistency = torch.tensor(0.0, device=video_pred.device)
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                pair_loss = (x0_preds[i] - x0_preds[j]).pow(2).mean()
                loss_consistency = loss_consistency + pair_loss
                n_pairs += 1
        if n_pairs > 0:
            loss_consistency = loss_consistency / n_pairs

        total_loss = loss_mf + cfg.consistency_weight * loss_consistency

        # ════════════════════════════════════════════════
        # KL (computed on original batch, not replicated)
        # ════════════════════════════════════════════════
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
            total_loss = total_loss + cfg.kl_weight * loss_kl

        # ════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════
        self._last_vfm_metrics = {
            "vfm/loss_mf": loss_mf.item(),
            "vfm/loss_kl": loss_kl.item(),
            "vfm/loss_consistency": loss_consistency.item(),
            "vfm/loss_total": total_loss.item(),
            "vfm/multi_sigma_K": K,
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
                "vfm/mu_norm_mean": mu_norm.mean().item() if mu_norm is not None else 0.0,
            })

        # Per-sigma loss breakdown
        for k, sigma_val in enumerate(cfg.multi_sigma_levels):
            v_k = v_preds[k]
            targets_k = inputs.video_targets.reshape(K, B, *inputs.video_targets.shape[1:])[k]
            loss_k = (v_k - targets_k).pow(2).mean()
            self._last_vfm_metrics[f"vfm/loss_sigma_{sigma_val:.1f}"] = loss_k.item()

        return total_loss
