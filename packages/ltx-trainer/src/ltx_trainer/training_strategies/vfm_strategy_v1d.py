"""VFM v1d — Trajectory distillation + per-token timestep scheduling.

Builds on v1c (diversity-regularized adapter) with two key additions:

1. **Trajectory distillation**: Train against pre-computed teacher 8-step ODE
   trajectories instead of random interpolation targets. The adapter learns to
   produce noise z that, when passed through the flow map in 1 step, matches
   the teacher's 8-step denoised output.

2. **Per-token timestep scheduling** (inspired by Self-Flow, arxiv:2603.06507):
   A learned sigma head predicts per-token σ_i ∈ (0, 1), allowing different
   tokens to be at different noise levels. The model can focus computational
   budget on harder tokens (high σ) while spending less on easy ones (low σ).

Architecture:
    Text embeddings → NoiseAdapterV1b → (μ, log_σ) per token → z
    z → SigmaHead → per-token σ_i
    x_t[i] = (1 - σ_i) · x₀[i] + σ_i · z[i]
    48-layer DiT (with per-token timesteps σ_i) → velocity v → x̂₀

The sigma head is a small MLP that takes the adapter's μ output and predicts
per-token noise levels. This ties the timestep schedule to the noise structure:
tokens where the adapter is more/less confident get different σ values.

When trajectories are available, distillation mode is used:
    - output_match: student 1-step output must match teacher 8-step output
    - velocity_match: student velocity matches teacher at random sigma points
    - progressive: halve steps each training round (8→4→2→1)

When trajectories are NOT available, falls back to standard VFM with per-token
timesteps (random interpolation targets, v = z - x₀).
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1c import (
    VFMv1cTrainingConfig,
    VFMv1cTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class SigmaHead(nn.Module):
    """Per-token sigma predictor.

    Takes adapter μ output [B, seq, latent_dim] and predicts per-token
    noise level σ_i ∈ [sigma_min, sigma_max] for each token.

    The output is constrained to [sigma_min, sigma_max] via sigmoid scaling
    to prevent degenerate solutions (all-zero or all-one sigmas).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        sigma_min: float = 0.05,
        sigma_max: float = 0.95,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize to output mid-range sigma (~0.5)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu: Tensor) -> Tensor:
        """Predict per-token sigma from adapter mu.

        Args:
            mu: Adapter mean output [B, seq, latent_dim]

        Returns:
            Per-token sigma [B, seq] in [sigma_min, sigma_max]
        """
        raw = self.net(mu).squeeze(-1)  # [B, seq]
        # Sigmoid → [0, 1] → scale to [sigma_min, sigma_max]
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)


class VFMv1dTrainingConfig(VFMv1cTrainingConfig):
    """Configuration for VFM v1d (trajectory distill + per-token sigma)."""

    name: Literal["vfm_v1d"] = "vfm_v1d"

    # === Per-token Timestep Scheduling ===
    per_token_sigma: bool = Field(
        default=True,
        description="Enable per-token sigma prediction via learned SigmaHead. "
        "Each token gets its own noise level instead of uniform σ.",
    )
    sigma_head_hidden_dim: int = Field(
        default=256,
        description="Hidden dim for the sigma head MLP",
    )
    sigma_min: float = Field(
        default=0.05, ge=0.01, le=0.5,
        description="Minimum per-token sigma (prevents all-clean degenerate solution)",
    )
    sigma_max: float = Field(
        default=0.95, ge=0.5, le=1.0,
        description="Maximum per-token sigma (prevents all-noise degenerate solution)",
    )
    sigma_entropy_weight: float = Field(
        default=0.01, ge=0.0,
        description="Weight for sigma entropy regularization. "
        "Encourages diverse sigma values across tokens (prevents uniform schedule).",
    )

    # === Trajectory Distillation ===
    distill_mode: Literal["output_match", "velocity_match", "progressive", "none"] = Field(
        default="none",
        description="Distillation mode. 'none' = standard VFM (no trajectories needed).",
    )
    student_steps: int = Field(
        default=1,
        description="Number of Euler steps the student takes (1 for output_match)",
    )
    distill_weight: float = Field(
        default=1.0,
        description="Weight for distillation loss",
    )
    gt_weight: float = Field(
        default=0.1,
        description="Weight for ground truth loss when mixing with distillation",
    )
    trajectories_dir: str = Field(
        default="trajectories",
        description="Directory name for pre-computed trajectory files under data root",
    )
    use_teacher_noise: bool = Field(
        default=False,
        description="Use teacher's starting noise z instead of adapter noise",
    )


class VFMv1dTrainingStrategy(VFMv1cTrainingStrategy):
    """VFM v1d — trajectory distillation + per-token timestep scheduling.

    Extends v1c (diversity regularization) with:
    1. SigmaHead: learned per-token noise levels
    2. Trajectory distillation: optional teacher-supervised training
    """

    config: VFMv1dTrainingConfig

    def __init__(self, config: VFMv1dTrainingConfig) -> None:
        super().__init__(config)
        self._sigma_head: SigmaHead | None = None
        self._has_trajectories: bool = False

        # Create sigma head if enabled
        if config.per_token_sigma:
            self._sigma_head = SigmaHead(
                latent_dim=128,
                hidden_dim=config.sigma_head_hidden_dim,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
            )

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return sigma head params so trainer adds them to optimizer."""
        params = []
        if self._sigma_head is not None:
            params.extend(list(self._sigma_head.parameters()))
        return params

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Add trajectories as data source if distillation is enabled."""
        sources = super().get_data_sources()
        if self.config.distill_mode != "none":
            if isinstance(sources, list):
                sources = {s: s for s in sources}
            sources[self.config.trajectories_dir] = "trajectories"
        return sources

    def set_noise_adapter(self, adapter) -> None:
        """Override to also move sigma head to adapter's device."""
        super().set_noise_adapter(adapter)
        if self._sigma_head is not None and adapter is not None:
            device = next(adapter.parameters()).device
            self._sigma_head = self._sigma_head.to(device)

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs with per-token sigma + optional distillation targets."""
        cfg = self.config

        # Check if trajectories are available in this batch
        has_traj = "trajectories" in batch and cfg.distill_mode != "none"

        if has_traj:
            return self._prepare_distill_inputs(batch, timestep_sampler)
        else:
            return self._prepare_standard_inputs(batch, timestep_sampler)

    def _prepare_standard_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Standard VFM with per-token sigma (no trajectories)."""
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
        # PER-TOKEN SIGMA (v1d key feature)
        # ════════════════════════════════════════════════
        if cfg.per_token_sigma and self._sigma_head is not None and adapter_mu is not None:
            # Predict per-token sigma from adapter mu
            per_token_sigmas = self._sigma_head(adapter_mu.detach())  # [B, seq]

            # Zero out sigma for conditioning tokens (first frame)
            per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()

            # Interpolate with per-token sigma
            sigmas_expanded = per_token_sigmas.unsqueeze(-1)  # [B, seq, 1]
            noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            # Velocity target: v = z - x₀ (same as standard, but model sees per-token σ)
            video_targets = video_noise - video_latents

            # Per-token timesteps = per-token sigmas directly
            video_timesteps = per_token_sigmas

            # Store mean sigma for logging
            sigmas_mean = per_token_sigmas[~video_conditioning_mask].mean().detach()
            sigmas_for_logging = sigmas_mean.unsqueeze(0).expand(batch_size)
        else:
            # Standard uniform sigma
            sigmas = timestep_sampler.sample_for(video_latents)
            sigmas_expanded = sigmas.view(-1, 1, 1)
            noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
            noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

            video_targets = video_noise - video_latents
            video_timesteps = self._create_per_token_timesteps(
                video_conditioning_mask, sigmas.squeeze()
            )
            per_token_sigmas = None
            sigmas_for_logging = sigmas.squeeze()

        # Ensure conditioning tokens use clean latents
        if per_token_sigmas is not None:
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

        # v1d metadata
        model_inputs._per_token_sigmas = per_token_sigmas
        model_inputs._distill_mode = "none"

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas_for_logging.view(-1, 1)

        return model_inputs

    def _prepare_distill_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Trajectory-distilled inputs with per-token sigma."""
        cfg = self.config
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents = self._video_patchifier.patchify(video_latents)

        fps = latents.get("fps", None)
        fps = fps[0].item() if fps is not None else 24.0

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Load teacher trajectory
        traj = batch["trajectories"]
        teacher_sigmas = traj["sigmas"].to(device)         # [B, N+1] or [N+1]
        teacher_states = traj["states"].to(device, dtype)   # [B, N+1, seq, C]
        teacher_velocities = traj["velocities"].to(device, dtype)  # [B, N, seq, C]
        teacher_x0_preds = traj["x0_preds"].to(device, dtype)     # [B, N, seq, C]
        teacher_x0_gt = traj.get("x0_gt", video_latents).to(device, dtype)

        if teacher_sigmas.dim() == 1:
            teacher_sigmas = teacher_sigmas.unsqueeze(0).expand(batch_size, -1)

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
        # NOISE: adapter or teacher's starting z
        # ════════════════════════════════════════════════
        use_adapter = random.random() < cfg.alpha

        if use_adapter and self._noise_adapter is not None and not cfg.use_teacher_noise:
            mu, log_sigma = self._noise_adapter(
                text_embeddings=video_prompt_embeds.detach(),
                text_mask=prompt_attention_mask.bool(),
                positions=video_positions,
                task_class=torch.zeros(batch_size, dtype=torch.long, device=device),
            )
            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn_like(mu)
            video_noise = mu + sigma_adapter * eps
            adapter_mu = mu
            adapter_log_sigma = log_sigma
        else:
            video_noise = teacher_states[:, 0]  # Teacher's starting noise
            adapter_mu = None
            adapter_log_sigma = None
            use_adapter = False

        # ════════════════════════════════════════════════
        # DISTILLATION TARGET
        # ════════════════════════════════════════════════
        if cfg.distill_mode == "output_match":
            # Target: teacher's best x̂₀ prediction
            teacher_target = teacher_x0_preds[:, -1]  # [B, seq, C]
            video_targets = video_noise - teacher_target  # v such that z - v = teacher_x̂₀

            if cfg.per_token_sigma and self._sigma_head is not None and adapter_mu is not None:
                # Per-token sigma even in distill mode
                per_token_sigmas = self._sigma_head(adapter_mu.detach())
                per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()
                video_timesteps = per_token_sigmas

                # At distill: use sigma=1 for noisy video (z = pure adapter noise)
                noisy_video = video_noise
            else:
                per_token_sigmas = None
                sigmas = torch.ones(batch_size, 1, device=device, dtype=dtype)
                video_timesteps = self._create_per_token_timesteps(
                    video_conditioning_mask, sigmas.squeeze()
                )
                noisy_video = video_noise

        elif cfg.distill_mode == "velocity_match":
            num_teacher_steps = teacher_velocities.shape[1]
            step_idx = random.randint(0, num_teacher_steps - 1)

            noisy_video = teacher_states[:, step_idx]
            sigma_val = teacher_sigmas[:, step_idx]
            video_targets = teacher_velocities[:, step_idx]

            per_token_sigmas = None
            video_timesteps = self._create_per_token_timesteps(
                video_conditioning_mask, sigma_val
            )

        else:
            raise ValueError(f"Unsupported distill_mode for v1d: {cfg.distill_mode}")

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
        model_inputs._vfm_use_adapter = use_adapter
        model_inputs._vfm_video_noise = video_noise
        model_inputs._vfm_video_latents = video_latents
        model_inputs._vfm_task_class = torch.zeros(batch_size, dtype=torch.long, device=device)
        model_inputs._vfm_task_name = "i2v"
        model_inputs._vfm_noise_level = 0.0
        model_inputs._vfm_observation = None
        model_inputs._raw_video_latents = batch["latents"]["latents"]

        # v1d metadata
        model_inputs._per_token_sigmas = per_token_sigmas
        model_inputs._distill_mode = cfg.distill_mode
        model_inputs._distill_teacher_target = teacher_target if cfg.distill_mode == "output_match" else None
        model_inputs._distill_teacher_x0_gt = teacher_x0_gt
        model_inputs._distill_teacher_states = teacher_states
        model_inputs._distill_teacher_sigmas = teacher_sigmas
        model_inputs._distill_teacher_velocities = teacher_velocities

        # For GT loss mixing
        if cfg.gt_weight > 0:
            model_inputs._gt_video_targets = video_noise - video_latents

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = torch.ones(batch_size, 1, device=device, dtype=dtype)

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute v1d loss = v1c loss + sigma entropy + distillation.

        Loss components:
        1. L_mf (from v1a): flow matching MSE
        2. L_obs (from v1a): observation consistency
        3. L_kl (from v1a): KL regularization
        4. L_div (from v1c): diversity regularization
        5. L_sigma_entropy (v1d): sigma distribution entropy
        6. L_distill (v1d): teacher trajectory matching (when enabled)
        """
        cfg = self.config
        distill_mode = getattr(inputs, "_distill_mode", "none")

        if distill_mode != "none" and distill_mode is not None:
            return self._compute_distill_loss(video_pred, audio_pred, inputs)

        # Standard VFM loss from v1c (includes diversity)
        total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        # Add sigma entropy regularization
        per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
        if per_token_sigmas is not None and cfg.sigma_entropy_weight > 0:
            loss_sigma_entropy = self._compute_sigma_entropy_loss(per_token_sigmas, inputs)
            total_loss = total_loss + cfg.sigma_entropy_weight * loss_sigma_entropy

            self._last_vfm_metrics.update({
                "vfm/sigma_entropy": loss_sigma_entropy.item(),
                "vfm/sigma_mean": per_token_sigmas[per_token_sigmas > 0].mean().item(),
                "vfm/sigma_std": per_token_sigmas[per_token_sigmas > 0].std().item(),
                "vfm/sigma_min_actual": per_token_sigmas[per_token_sigmas > 0].min().item(),
                "vfm/sigma_max_actual": per_token_sigmas[per_token_sigmas > 0].max().item(),
            })

        return total_loss

    def _compute_distill_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Distillation loss + KL + optional GT mixing + diversity + sigma entropy."""
        cfg = self.config

        # L_distill: MSE between student velocity and teacher target
        video_loss = (video_pred - inputs.video_targets).pow(2)

        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_distill = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_distill = video_loss.mean()

        total_loss = cfg.distill_weight * loss_distill

        # L_kl: adapter KL regularization
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        loss_kl = torch.tensor(0.0, device=video_pred.device)

        if use_adapter and adapter_mu is not None and cfg.kl_weight > 0:
            adapter_log_sigma = inputs._vfm_adapter_log_sigma
            kl_raw = 0.5 * (adapter_mu.pow(2) + torch.exp(2 * adapter_log_sigma) - 2 * adapter_log_sigma - 1)
            kl_per_sample = kl_raw.mean(dim=(1, 2))
            if cfg.kl_free_bits > 0:
                kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)
            loss_kl = kl_per_sample.mean()
            total_loss = total_loss + cfg.kl_weight * loss_kl

        # L_gt: optional GT flow matching loss
        loss_gt = torch.tensor(0.0, device=video_pred.device)
        if cfg.gt_weight > 0 and hasattr(inputs, "_gt_video_targets"):
            gt_loss = (video_pred - inputs._gt_video_targets).pow(2)
            if inputs.video_loss_mask is not None:
                gt_loss = (gt_loss * mask).sum() / mask.sum().clamp(min=1) / gt_loss.shape[-1]
            else:
                gt_loss = gt_loss.mean()
            loss_gt = gt_loss
            total_loss = total_loss + cfg.gt_weight * loss_gt

        # L_div: diversity from v1c (computed on adapter mu)
        if use_adapter and adapter_mu is not None:
            div_loss = self._compute_diversity_loss(adapter_mu, inputs)
            total_loss = total_loss + div_loss

        # L_sigma_entropy
        per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
        loss_sigma_entropy = torch.tensor(0.0, device=video_pred.device)
        if per_token_sigmas is not None and cfg.sigma_entropy_weight > 0:
            loss_sigma_entropy = self._compute_sigma_entropy_loss(per_token_sigmas, inputs)
            total_loss = total_loss + cfg.sigma_entropy_weight * loss_sigma_entropy

        # Logging
        self._last_vfm_metrics = {
            "vfm/loss_distill": loss_distill.item(),
            "vfm/loss_kl": loss_kl.item(),
            "vfm/loss_gt": loss_gt.item(),
            "vfm/loss_total": total_loss.item(),
            "vfm/distill_mode": cfg.distill_mode,
            "vfm/use_adapter": float(use_adapter),
        }
        if adapter_mu is not None:
            self._last_vfm_metrics["vfm/adapter_mu_mean"] = adapter_mu.mean().item()
            self._last_vfm_metrics["vfm/adapter_sigma_mean"] = torch.exp(
                inputs._vfm_adapter_log_sigma
            ).mean().item()
        if per_token_sigmas is not None:
            active = per_token_sigmas[per_token_sigmas > 0]
            if active.numel() > 0:
                self._last_vfm_metrics["vfm/sigma_mean"] = active.mean().item()
                self._last_vfm_metrics["vfm/sigma_std"] = active.std().item()
                self._last_vfm_metrics["vfm/sigma_entropy"] = loss_sigma_entropy.item()

        return total_loss

    def _compute_diversity_loss(self, adapter_mu: Tensor, inputs: ModelInputs) -> Tensor:
        """Extract diversity loss computation from v1c for reuse in distill path."""
        cfg = self.config

        if cfg.diversity_warmup_steps > 0 and self._current_step < cfg.diversity_warmup_steps:
            warmup_factor = self._current_step / cfg.diversity_warmup_steps
        else:
            warmup_factor = 1.0

        B, seq_len, C = adapter_mu.shape
        token_std = adapter_mu.std(dim=1).mean()
        loss_token_div = -cfg.diversity_weight * token_std

        loss_temporal_div = torch.tensor(0.0, device=adapter_mu.device)
        loss_spatial_div = torch.tensor(0.0, device=adapter_mu.device)

        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is not None:
            num_frames = raw_latents.shape[2]
            tokens_per_frame = seq_len // num_frames

            if tokens_per_frame > 0 and num_frames > 1:
                mu_reshaped = adapter_mu[:, :num_frames * tokens_per_frame].reshape(
                    B, num_frames, tokens_per_frame, C
                )
                mu_per_frame = mu_reshaped.mean(dim=2)
                temporal_std = mu_per_frame.std(dim=1).mean()
                loss_temporal_div = -cfg.temporal_diversity_weight * temporal_std

                mu_per_spatial = mu_reshaped.mean(dim=1)
                spatial_std = mu_per_spatial.std(dim=1).mean()
                loss_spatial_div = -cfg.spatial_diversity_weight * spatial_std

        return warmup_factor * (loss_token_div + loss_temporal_div + loss_spatial_div)

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction + trajectory plots to W&B."""
        # Parent handles video reconstruction
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        # Add trajectory plots if distillation data is available
        teacher_states = getattr(inputs, "_distill_teacher_states", None)
        if teacher_states is not None:
            try:
                from ltx_trainer.training_strategies.vfm_distill_strategy import VFMDistillStrategy
                traj_plots = VFMDistillStrategy._build_trajectory_plots(
                    video_pred, inputs, step, prefix,
                )
                log_dict.update(traj_plots)
            except Exception as e:
                logger.warning(f"Failed to build trajectory plots: {e}")

        # Log per-token sigma heatmap
        per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
        if per_token_sigmas is not None:
            try:
                sigma_plots = self._build_sigma_plots(per_token_sigmas, inputs, step, prefix)
                log_dict.update(sigma_plots)
            except Exception as e:
                logger.warning(f"Failed to build sigma plots: {e}")

        return log_dict

    @staticmethod
    def _build_sigma_plots(
        per_token_sigmas: Tensor,
        inputs: ModelInputs,
        step: int,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Build per-token sigma visualization for W&B."""
        try:
            import wandb
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            sigmas = per_token_sigmas[0].float().cpu()  # [seq]
            seq_len = sigmas.shape[0]

            raw_latents = getattr(inputs, "_raw_video_latents", None)
            if raw_latents is None:
                return {}

            num_frames = raw_latents.shape[2]
            tpf = seq_len // num_frames
            h = raw_latents.shape[3]
            w = raw_latents.shape[4]

            # Reshape to spatial grid per frame
            sigma_frames = sigmas[:num_frames * tpf].reshape(num_frames, h, w)

            fig = make_subplots(
                rows=1, cols=num_frames,
                subplot_titles=[f"Frame {i}" for i in range(num_frames)],
            )
            for f_idx in range(num_frames):
                fig.add_trace(
                    go.Heatmap(
                        z=sigma_frames[f_idx].numpy(),
                        colorscale="RdYlBu_r",
                        zmin=0.0, zmax=1.0,
                        showscale=(f_idx == num_frames - 1),
                        colorbar=dict(title="σ"),
                    ),
                    row=1, col=f_idx + 1,
                )

            fig.update_layout(
                title=f"Per-Token Sigma Map (step {step})",
                template="plotly_dark",
                height=250, width=200 * num_frames,
            )
            return {f"{prefix}/sigma_heatmap": wandb.Plotly(fig)}
        except Exception:
            return {}

    @staticmethod
    def _compute_sigma_entropy_loss(per_token_sigmas: Tensor, inputs: ModelInputs) -> Tensor:
        """Encourage diverse per-token sigma values.

        Maximizes entropy of the sigma distribution across tokens.
        Low entropy = all tokens have similar sigma = no benefit from per-token scheduling.
        High entropy = diverse sigma values = model allocates compute heterogeneously.

        We use negative std as a simple proxy for negative entropy:
        L = -std(σ) across active tokens.
        """
        loss_mask = getattr(inputs, "video_loss_mask", None)
        if loss_mask is not None:
            # Only consider non-conditioning tokens
            active_sigmas = per_token_sigmas[loss_mask]
        else:
            active_sigmas = per_token_sigmas.flatten()

        if active_sigmas.numel() < 2:
            return torch.tensor(0.0, device=per_token_sigmas.device)

        # Negative std → minimizing this maximizes sigma diversity
        return -active_sigmas.std()
