"""VFM v1e — Content-adaptive VFM with router-guided sigma + detail preservation.

Addresses quality deterioration on detailed content (observed in v1d step ~458):
  - VFM 1-step generation applies uniform effort to all tokens
  - Simple tokens (sky, flat surfaces) need minimal denoising
  - Complex tokens (textures, edges, motion) need much more capacity

Inspired by EVATok (CVPR 2026, arxiv:2603.12267) which allocates more tokens
to complex content. v1e adapts this to the sigma/denoising dimension:

1. **Content Complexity Router**: Small transformer that analyzes GT latent
   features (during training) to predict per-token complexity scores.
   At inference, conditioned on adapter noise structure instead.

2. **Router-guided sigma**: Complex tokens get LOWER sigma (closer to clean,
   easier 1-step denoising). Simple tokens get HIGHER sigma (more noise is OK).
   This replaces v1d's mu-based sigma head with content-aware prediction.

3. **Complexity-weighted loss**: Loss weight per token is proportional to
   complexity score. Model focuses gradient on hard/detailed regions.

4. **Frequency-domain detail loss**: Penalizes loss of high-frequency spatial
   detail via gradient magnitude comparison (GT vs prediction).

Architecture:
    GT latents x₀ → ContentRouter → complexity_scores [B, seq]
    complexity_scores → sigma_schedule: σ_i = σ_max - (σ_max - σ_min) * complexity_i
    (complex tokens get lower σ)

    Adapter → structured noise z
    x_t[i] = (1 - σ_i) · x₀[i] + σ_i · z[i]
    48-layer DiT (per-token σ_i) → velocity v → x̂₀

    Loss = complexity_weight[i] * ||v - target||² + freq_loss(x̂₀, x₀)
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
from ltx_trainer.training_strategies.vfm_strategy_v1d import (
    SigmaHead,
    VFMv1dTrainingConfig,
    VFMv1dTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class ContentRouter(nn.Module):
    """Predicts per-token content complexity from latent features.

    Inspired by EVATok's GlobalViTRouterV2 but adapted for continuous
    complexity scoring instead of discrete token-count assignment.

    During training: analyzes GT latent features to identify complex regions.
    During inference: can use adapter noise structure as a proxy for complexity.

    Architecture:
        latent [B, seq, 128] → projection → transformer blocks → complexity [B, seq]

    Output complexity ∈ [0, 1] where:
        0 = simple content (flat colors, sky, uniform regions)
        1 = complex content (textures, edges, fine detail, motion)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project latent to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Transformer blocks for contextual complexity estimation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Output head: per-token complexity score
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize to output mid-range complexity (~0.5)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, latent: Tensor) -> Tensor:
        """Predict per-token complexity from latent features.

        Args:
            latent: [B, seq, latent_dim] — GT latents (training) or noise (inference)

        Returns:
            complexity: [B, seq] in [0, 1]
        """
        x = self.input_proj(latent)
        x = self.transformer(x)
        raw = self.output_head(x).squeeze(-1)  # [B, seq]
        return torch.sigmoid(raw)


class VFMv1eTrainingConfig(VFMv1dTrainingConfig):
    """Configuration for VFM v1e (content-adaptive router + detail preservation)."""

    name: Literal["vfm_v1e"] = "vfm_v1e"

    # === Content Router ===
    content_router: bool = Field(
        default=True,
        description="Enable content complexity router for adaptive sigma scheduling",
    )
    router_hidden_dim: int = Field(
        default=256,
        description="Hidden dim for content router transformer",
    )
    router_num_layers: int = Field(
        default=2,
        description="Number of transformer layers in router",
    )
    router_num_heads: int = Field(
        default=4,
        description="Attention heads in router transformer",
    )
    router_detach_input: bool = Field(
        default=True,
        description="Detach router input from main computation graph. "
        "True = router learns complexity independently of flow map gradients.",
    )

    # === Complexity-weighted loss ===
    complexity_loss_weight: float = Field(
        default=0.5, ge=0.0,
        description="How much to upweight loss on complex tokens. "
        "Final weight per token = 1 + complexity_loss_weight * complexity_score. "
        "0 = uniform (no reweighting), 1 = 2x weight on max-complexity tokens.",
    )

    # === Detail preservation ===
    frequency_loss_weight: float = Field(
        default=0.05, ge=0.0,
        description="Weight for high-frequency detail preservation loss. "
        "Computes spatial gradient magnitude in latent space.",
    )
    frequency_loss_warmup: int = Field(
        default=100, ge=0,
        description="Warmup steps before applying frequency loss",
    )

    # === Router supervision ===
    router_supervision: bool = Field(
        default=True,
        description="Supervise router with GT complexity (latent gradient magnitude). "
        "Helps router learn meaningful complexity scores faster.",
    )
    router_supervision_weight: float = Field(
        default=0.1, ge=0.0,
        description="Weight for router supervision loss",
    )


class VFMv1eTrainingStrategy(VFMv1dTrainingStrategy):
    """VFM v1e — content-adaptive sigma + detail preservation.

    Extends v1d (trajectory distillation + per-token sigma) with:
    1. ContentRouter: learned complexity estimation per token
    2. Router-guided sigma: complex → low σ, simple → high σ
    3. Complexity-weighted loss: focus on detailed regions
    4. Frequency-domain detail preservation loss
    """

    config: VFMv1eTrainingConfig

    def __init__(self, config: VFMv1eTrainingConfig) -> None:
        super().__init__(config)
        self._content_router: ContentRouter | None = None

        if config.content_router:
            self._content_router = ContentRouter(
                latent_dim=128,
                hidden_dim=config.router_hidden_dim,
                num_heads=config.router_num_heads,
                num_layers=config.router_num_layers,
            )

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return sigma head + router params."""
        params = super().get_trainable_parameters()
        if self._content_router is not None:
            params.extend(list(self._content_router.parameters()))
        return params

    def set_noise_adapter(self, adapter) -> None:
        """Move router to adapter's device too."""
        super().set_noise_adapter(adapter)
        if self._content_router is not None and adapter is not None:
            device = next(adapter.parameters()).device
            self._content_router = self._content_router.to(device)

    def _compute_complexity_scores(
        self,
        video_latents: Tensor,
        video_conditioning_mask: Tensor,
    ) -> Tensor:
        """Compute per-token complexity scores from GT latents.

        Args:
            video_latents: [B, seq, C] ground truth latent tokens
            video_conditioning_mask: [B, seq] True for conditioning tokens

        Returns:
            complexity: [B, seq] in [0, 1], 0 for conditioning tokens
        """
        if self._content_router is None:
            return None

        # Ensure router is on the same device/dtype as input
        router_param = next(self._content_router.parameters())
        if router_param.device != video_latents.device or router_param.dtype != video_latents.dtype:
            self._content_router = self._content_router.to(
                device=video_latents.device, dtype=video_latents.dtype,
            )

        cfg = self.config
        router_input = video_latents.detach() if cfg.router_detach_input else video_latents
        complexity = self._content_router(router_input)  # [B, seq]

        # Zero out conditioning tokens
        complexity = complexity * (~video_conditioning_mask).float().to(complexity.dtype)
        return complexity

    @staticmethod
    def _compute_gt_complexity(
        video_latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tensor:
        """Compute ground-truth complexity from latent spatial gradients.

        Uses gradient magnitude as a proxy for content complexity:
        high gradients = edges, textures, detail = complex.

        Args:
            video_latents: [B, seq, C] patchified latents
            num_frames: number of latent frames
            height: latent height
            width: latent width

        Returns:
            complexity: [B, seq] normalized to [0, 1]
        """
        B, seq, C = video_latents.shape
        tpf = height * width

        # Reshape to spatial: [B*F, C, H, W]
        spatial = video_latents[:, :num_frames * tpf].reshape(
            B, num_frames, height, width, C
        ).permute(0, 1, 4, 2, 3).reshape(B * num_frames, C, height, width)

        # Compute spatial gradients (Sobel-like)
        if height > 1 and width > 1:
            grad_h = (spatial[:, :, 1:, :] - spatial[:, :, :-1, :]).abs()
            grad_w = (spatial[:, :, :, 1:] - spatial[:, :, :, :-1]).abs()

            # Pad to original size
            grad_h = F.pad(grad_h, (0, 0, 0, 1))  # pad height
            grad_w = F.pad(grad_w, (0, 1, 0, 0))  # pad width

            # Gradient magnitude per pixel, averaged over channels
            grad_mag = (grad_h + grad_w).mean(dim=1)  # [B*F, H, W]
        else:
            grad_mag = spatial.abs().mean(dim=1)

        # Reshape back to sequence: [B, seq]
        grad_mag = grad_mag.reshape(B, num_frames * tpf)

        # Normalize to [0, 1] per sample
        g_min = grad_mag.min(dim=1, keepdim=True).values
        g_max = grad_mag.max(dim=1, keepdim=True).values
        complexity = (grad_mag - g_min) / (g_max - g_min + 1e-8)

        # Pad if seq > num_frames * tpf (shouldn't happen but safety)
        if complexity.shape[1] < seq:
            pad = torch.zeros(B, seq - complexity.shape[1], device=complexity.device)
            complexity = torch.cat([complexity, pad], dim=1)

        return complexity

    def _prepare_standard_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Standard VFM with router-guided per-token sigma."""
        cfg = self.config
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents = self._video_patchifier.patchify(video_latents)  # [B, seq, C]

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

        # Initialize inverse problem sampler
        if self._inverse_problem_sampler is None:
            from ltx_trainer.inverse_problems import InverseProblemSampler
            self._inverse_problem_sampler = InverseProblemSampler(
                problems=self._ip_configs,
                tokens_per_frame=tokens_per_frame,
            )

        # Positions
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # First-frame conditioning mask
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height, width=width,
            device=device,
            first_frame_conditioning_p=cfg.first_frame_conditioning_p,
        )

        # ════════════════════════════════════════════════
        # CONTENT COMPLEXITY ROUTER (v1e key feature)
        # ════════════════════════════════════════════════
        complexity_scores = self._compute_complexity_scores(
            video_latents, video_conditioning_mask,
        )

        # GT complexity for router supervision
        gt_complexity = None
        if cfg.router_supervision and self._content_router is not None:
            gt_complexity = self._compute_gt_complexity(
                video_latents, num_frames, height, width,
            )

        # ════════════════════════════════════════════════
        # NOISE ADAPTER
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

            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn_like(mu)
            video_noise = mu + sigma_adapter * eps

            adapter_mu = mu
            adapter_log_sigma = log_sigma
            task_class = ip_sample.task_class
            ip_observation = ip_sample.observation
            ip_task_name = ip_sample.task_name
            ip_noise_level = ip_sample.noise_level
        else:
            video_noise = torch.randn_like(video_latents)
            adapter_mu = None
            adapter_log_sigma = None
            task_class = None
            ip_observation = None
            ip_task_name = "unconditional"
            ip_noise_level = 0.0

        # ════════════════════════════════════════════════
        # ROUTER-GUIDED PER-TOKEN SIGMA
        # ════════════════════════════════════════════════
        if complexity_scores is not None and cfg.per_token_sigma:
            # Complex tokens → LOWER sigma (easier denoising)
            # Simple tokens → HIGHER sigma (can handle more noise)
            # σ_i = σ_max - (σ_max - σ_min) * complexity_i
            per_token_sigmas = (
                cfg.sigma_max - (cfg.sigma_max - cfg.sigma_min) * complexity_scores
            )
            per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()

            sigmas_expanded = per_token_sigmas.unsqueeze(-1)  # [B, seq, 1]
            noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            video_targets = video_noise - video_latents
            video_timesteps = per_token_sigmas

            sigmas_mean = per_token_sigmas[~video_conditioning_mask].mean().detach()
            sigmas_for_logging = sigmas_mean.unsqueeze(0).expand(batch_size)

        elif cfg.per_token_sigma and self._sigma_head is not None and adapter_mu is not None:
            # Fallback to v1d mu-based sigma if no router
            per_token_sigmas = self._sigma_head(adapter_mu.detach())
            per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()

            sigmas_expanded = per_token_sigmas.unsqueeze(-1)
            noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            video_targets = video_noise - video_latents
            video_timesteps = per_token_sigmas

            sigmas_mean = per_token_sigmas[~video_conditioning_mask].mean().detach()
            sigmas_for_logging = sigmas_mean.unsqueeze(0).expand(batch_size)
        else:
            # Uniform sigma
            sigmas = timestep_sampler.sample_for(video_latents)
            sigmas_expanded = sigmas.view(-1, 1, 1)
            noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            video_targets = video_noise - video_latents
            video_timesteps = self._create_per_token_timesteps(
                video_conditioning_mask, sigmas.squeeze()
            )
            per_token_sigmas = None
            complexity_scores = None
            sigmas_for_logging = sigmas.squeeze()

        # Ensure conditioning tokens are clean
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # ════════════════════════════════════════════════
        # BUILD MODALITY
        # ════════════════════════════════════════════════
        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        video_loss_mask = ~video_conditioning_mask

        model_inputs = ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

        # VFM metadata
        model_inputs._vfm_adapter_mu = adapter_mu
        model_inputs._vfm_adapter_log_sigma = adapter_log_sigma
        model_inputs._vfm_task_class = task_class
        model_inputs._vfm_observation = ip_observation
        model_inputs._vfm_task_name = ip_task_name
        model_inputs._vfm_noise_level = ip_noise_level
        model_inputs._vfm_video_noise = video_noise
        model_inputs._vfm_video_latents = video_latents
        model_inputs._vfm_use_adapter = use_adapter_noise
        model_inputs._raw_video_latents = batch["latents"]["latents"]

        # v1e metadata
        model_inputs._per_token_sigmas = per_token_sigmas
        model_inputs._complexity_scores = complexity_scores
        model_inputs._gt_complexity = gt_complexity
        model_inputs._distill_mode = "none"
        model_inputs._num_frames = num_frames
        model_inputs._latent_height = height
        model_inputs._latent_width = width

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas_for_logging.view(-1, 1)

        return model_inputs

    def _prepare_distill_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Trajectory distillation with router-guided sigma."""
        # Get base distill inputs from v1d
        inputs = super()._prepare_distill_inputs(batch, timestep_sampler)

        # Add complexity scores from router
        video_latents = inputs._vfm_video_latents
        video_loss_mask = inputs.video_loss_mask
        video_conditioning_mask = ~video_loss_mask

        complexity_scores = self._compute_complexity_scores(
            video_latents, video_conditioning_mask,
        )
        inputs._complexity_scores = complexity_scores

        # GT complexity for router supervision
        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is not None and self.config.router_supervision:
            num_frames = raw_latents.shape[2]
            height = raw_latents.shape[3]
            width = raw_latents.shape[4]
            inputs._gt_complexity = self._compute_gt_complexity(
                video_latents, num_frames, height, width,
            )
            inputs._num_frames = num_frames
            inputs._latent_height = height
            inputs._latent_width = width
        else:
            inputs._gt_complexity = None

        return inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute v1e loss = v1d loss + complexity weighting + frequency + router supervision.

        Additional loss components over v1d:
        7. Complexity-weighted MSE: per-token loss scaled by complexity score
        8. Frequency detail loss: spatial gradient preservation in latent space
        9. Router supervision: train router to predict GT complexity
        """
        cfg = self.config
        distill_mode = getattr(inputs, "_distill_mode", "none")

        # Get base loss (handles distill vs standard path + diversity + sigma entropy)
        if distill_mode != "none" and distill_mode is not None:
            total_loss = self._compute_distill_loss_v1e(video_pred, audio_pred, inputs)
        else:
            total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        complexity_scores = getattr(inputs, "_complexity_scores", None)

        # ════════════════════════════════════════════════
        # COMPLEXITY-WEIGHTED LOSS ADJUSTMENT
        # ════════════════════════════════════════════════
        if complexity_scores is not None and cfg.complexity_loss_weight > 0:
            # Recompute MSE with complexity weighting
            video_loss = (video_pred - inputs.video_targets).pow(2)  # [B, seq, C]

            # Per-token weight: 1 + w * complexity (complex → higher weight)
            token_weights = 1.0 + cfg.complexity_loss_weight * complexity_scores  # [B, seq]
            token_weights = token_weights.unsqueeze(-1)  # [B, seq, 1]

            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                weighted_loss = (video_loss * token_weights * mask).sum() / (
                    (token_weights * mask).sum().clamp(min=1) * video_loss.shape[-1]
                )
            else:
                weighted_loss = (video_loss * token_weights).mean()

            # Replace the uniform MSE portion with complexity-weighted version
            # Scale: the uniform MSE is ~loss_mf_scaled, replace with weighted version
            complexity_adjustment = cfg.complexity_loss_weight * (weighted_loss - video_loss.mean().detach())
            total_loss = total_loss + complexity_adjustment

        # ════════════════════════════════════════════════
        # FREQUENCY DETAIL PRESERVATION LOSS
        # ════════════════════════════════════════════════
        if cfg.frequency_loss_weight > 0 and self._current_step >= cfg.frequency_loss_warmup:
            freq_loss = self._compute_frequency_loss(video_pred, inputs)
            if freq_loss is not None:
                total_loss = total_loss + cfg.frequency_loss_weight * freq_loss
                self._last_vfm_metrics["vfm/freq_loss"] = freq_loss.item()

        # ════════════════════════════════════════════════
        # ROUTER SUPERVISION LOSS
        # ════════════════════════════════════════════════
        gt_complexity = getattr(inputs, "_gt_complexity", None)
        if (
            cfg.router_supervision
            and gt_complexity is not None
            and complexity_scores is not None
            and cfg.router_supervision_weight > 0
        ):
            # MSE between router's predicted complexity and GT gradient-based complexity
            loss_mask = inputs.video_loss_mask if inputs.video_loss_mask is not None else None
            if loss_mask is not None:
                active_pred = complexity_scores[loss_mask]
                active_gt = gt_complexity[loss_mask]
            else:
                active_pred = complexity_scores.flatten()
                active_gt = gt_complexity.flatten()

            router_loss = F.mse_loss(active_pred, active_gt.detach())
            total_loss = total_loss + cfg.router_supervision_weight * router_loss
            self._last_vfm_metrics["vfm/router_loss"] = router_loss.item()

        # Log complexity metrics
        if complexity_scores is not None:
            active_c = complexity_scores[complexity_scores > 0]
            if active_c.numel() > 0:
                self._last_vfm_metrics.update({
                    "vfm/complexity_mean": active_c.mean().item(),
                    "vfm/complexity_std": active_c.std().item(),
                    "vfm/complexity_max": active_c.max().item(),
                    "vfm/complexity_min": active_c.min().item(),
                })

        return total_loss

    def _compute_distill_loss_v1e(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Distillation loss from v1d, but reusable for v1e's additional terms."""
        return self._compute_distill_loss(video_pred, audio_pred, inputs)

    def _compute_frequency_loss(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
    ) -> Tensor | None:
        """Compute frequency-domain detail preservation loss.

        Compares spatial gradient magnitudes between predicted x̂₀ and GT x₀.
        High-frequency detail (edges, textures) has large gradients.
        If prediction loses these, the frequency loss penalizes it.

        L_freq = ||∇(x̂₀) - ∇(x₀)||² (gradient magnitude matching)
        """
        raw_latents = getattr(inputs, "_raw_video_latents", None)
        video_noise = getattr(inputs, "_vfm_video_noise", None)
        if raw_latents is None or video_noise is None:
            return None

        B, C, F, H, W = raw_latents.shape
        if H < 2 or W < 2:
            return None

        # Reconstruct predicted x̂₀ from velocity: x̂₀ = z - v
        pred_x0 = video_noise - video_pred  # [B, seq, 128]

        # Get GT in sequence form
        gt_x0 = inputs._vfm_video_latents  # [B, seq, 128]

        tpf = H * W
        seq_used = F * tpf

        # Reshape to spatial: [B*F, C, H, W]
        pred_spatial = pred_x0[:, :seq_used].reshape(B, F, H, W, C).permute(0, 1, 4, 2, 3)
        pred_spatial = pred_spatial.reshape(B * F, C, H, W)

        gt_spatial = gt_x0[:, :seq_used].reshape(B, F, H, W, C).permute(0, 1, 4, 2, 3)
        gt_spatial = gt_spatial.reshape(B * F, C, H, W)

        # Compute spatial gradients
        pred_grad_h = pred_spatial[:, :, 1:, :] - pred_spatial[:, :, :-1, :]
        pred_grad_w = pred_spatial[:, :, :, 1:] - pred_spatial[:, :, :, :-1]

        gt_grad_h = gt_spatial[:, :, 1:, :] - gt_spatial[:, :, :-1, :]
        gt_grad_w = gt_spatial[:, :, :, 1:] - gt_spatial[:, :, :, :-1]

        # L2 loss on gradient difference
        freq_loss_h = (pred_grad_h - gt_grad_h).pow(2).mean()
        freq_loss_w = (pred_grad_w - gt_grad_w).pow(2).mean()

        return freq_loss_h + freq_loss_w

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction + complexity heatmap + trajectory plots."""
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        # Add complexity heatmap
        complexity_scores = getattr(inputs, "_complexity_scores", None)
        if complexity_scores is not None:
            try:
                complexity_plots = self._build_complexity_plots(
                    complexity_scores, inputs, step, prefix,
                )
                log_dict.update(complexity_plots)
            except Exception as e:
                logger.warning(f"Failed to build complexity plots: {e}")

        return log_dict

    @staticmethod
    def _build_complexity_plots(
        complexity_scores: Tensor,
        inputs: ModelInputs,
        step: int,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Build complexity score heatmap + comparison with GT complexity."""
        try:
            import wandb
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            scores = complexity_scores[0].float().cpu()  # [seq]
            raw_latents = getattr(inputs, "_raw_video_latents", None)
            if raw_latents is None:
                return {}

            num_frames = raw_latents.shape[2]
            h = raw_latents.shape[3]
            w = raw_latents.shape[4]
            tpf = h * w

            # Router predictions
            score_frames = scores[:num_frames * tpf].reshape(num_frames, h, w)

            # GT complexity
            gt_complexity = getattr(inputs, "_gt_complexity", None)

            num_rows = 2 if gt_complexity is not None else 1
            fig = make_subplots(
                rows=num_rows, cols=num_frames,
                subplot_titles=(
                    [f"Router F{i}" for i in range(num_frames)]
                    + ([f"GT F{i}" for i in range(num_frames)] if gt_complexity is not None else [])
                ),
                vertical_spacing=0.15,
            )

            for f_idx in range(num_frames):
                fig.add_trace(
                    go.Heatmap(
                        z=score_frames[f_idx].numpy(),
                        colorscale="Hot",
                        zmin=0.0, zmax=1.0,
                        showscale=(f_idx == num_frames - 1),
                        colorbar=dict(title="Complexity", y=0.75 if num_rows == 2 else 0.5),
                    ),
                    row=1, col=f_idx + 1,
                )

            if gt_complexity is not None:
                gt = gt_complexity[0].float().cpu()
                gt_frames = gt[:num_frames * tpf].reshape(num_frames, h, w)
                for f_idx in range(num_frames):
                    fig.add_trace(
                        go.Heatmap(
                            z=gt_frames[f_idx].numpy(),
                            colorscale="Hot",
                            zmin=0.0, zmax=1.0,
                            showscale=False,
                        ),
                        row=2, col=f_idx + 1,
                    )

            fig.update_layout(
                title=f"Content Complexity: Router vs GT (step {step})",
                template="plotly_dark",
                height=200 * num_rows + 50,
                width=200 * num_frames,
            )
            return {f"{prefix}/complexity_heatmap": wandb.Plotly(fig)}
        except Exception:
            return {}
