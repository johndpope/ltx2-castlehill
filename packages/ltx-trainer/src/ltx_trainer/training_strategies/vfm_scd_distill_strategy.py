"""VFM-SCD Distillation Strategy — One-step decoder via teacher trajectories.

Extends VFMSCDTrainingStrategy with teacher trajectory supervision for the
SCD decoder. Instead of random flow matching targets (v = z - x₀), the student
decoder learns to match the teacher's multi-step decoder output in 1 step.

Architecture:
    Encoder (32 layers): Run once on clean latents → features (unchanged)
    Decoder (16 layers): 1 step instead of 8, supervised by teacher trajectory

Training flow:
    1. Encoder forward (clean, σ=0, causal mask) → encoder_features
    2. Shift features by 1 frame
    3. Noise adapter: z ~ q_φ(shifted_features, task_class)
    4. Decoder forward at σ=1: v_pred = decoder(z, features)
    5. Loss = distill_weight * ||v_pred - (z - teacher_x̂₀)||²
           + kl_weight * KL(q_φ || N(0,I))
           + gt_weight * ||v_pred - (z - x₀_gt)||²

The teacher trajectories come from precompute_scd_trajectories.py which runs
the SCD decoder for N steps per-frame with encoder context.

Inference (autoregressive):
    For each chunk:
        Encoder(clean_frame, KV-cache) → features      [1 pass]
        z ~ q_φ(shifted_features)                       [adapter]
        v = Decoder(z, shifted_features, σ=1)           [1 pass]
        x̂₀ = z - v
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import shift_encoder_features
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_scd_strategy import (
    VFMSCDTrainingConfig,
    VFMSCDTrainingStrategy,
)


class VFMSCDDistillConfig(VFMSCDTrainingConfig):
    """Config for VFM-SCD distillation training."""

    name: Literal["vfm_scd_distill"] = "vfm_scd_distill"

    # Distillation mode
    distill_mode: Literal["output_match", "velocity_match", "progressive"] = Field(
        default="output_match",
        description=(
            "output_match: 1-step student matches teacher's final x̂₀. "
            "velocity_match: student matches teacher velocity at random sigma. "
            "progressive: multi-step student (halving schedule)."
        ),
    )

    # Number of decoder steps the student takes (1 for output_match)
    student_steps: int = Field(
        default=1,
        description="Decoder steps for the student (1 for output_match, >1 for progressive)",
    )

    # Loss weights
    distill_weight: float = Field(
        default=1.0,
        description="Weight for distillation loss (teacher matching)",
    )
    gt_weight: float = Field(
        default=0.0,
        description="Weight for ground truth loss. >0 mixes distillation with GT supervision.",
    )

    # Trajectory data
    trajectories_dir: str = Field(
        default="scd_trajectories",
        description="Directory name for pre-computed SCD decoder trajectory files",
    )

    # Use teacher's starting noise vs adapter noise
    use_teacher_noise: bool = Field(
        default=False,
        description="Use teacher's starting noise z instead of adapter. Useful early in training.",
    )

    # Which trajectory steps to train on (velocity_match mode)
    velocity_match_steps: list[int] | None = Field(
        default=None,
        description="Which trajectory step indices for velocity matching. None = all.",
    )


class VFMSCDDistillStrategy(VFMSCDTrainingStrategy):
    """VFM-SCD training with teacher trajectory distillation.

    Inherits the full VFM-SCD machinery (encoder pass, feature shifting,
    noise adapter, KL loss) and replaces the flow matching target with
    teacher-provided supervision from pre-computed SCD decoder trajectories.
    """

    config: VFMSCDDistillConfig

    def __init__(self, config: VFMSCDDistillConfig) -> None:
        super().__init__(config)

    def get_data_sources(self) -> dict[str, str]:
        """Add SCD trajectories as a data source."""
        sources = super().get_data_sources()
        sources[self.config.trajectories_dir] = "trajectories"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs using SCD encoder + teacher trajectory targets.

        Flow:
        1. Run SCD encoder on clean latents (same as parent)
        2. Shift encoder features (same as parent)
        3. Sample noise from adapter or teacher trajectory
        4. Set targets from teacher trajectory (NOT random interpolation)
        """
        cfg = self.config

        # === Extract batch data ===
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents_spatial = video_latents
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
            self.initialize_inverse_problems(tokens_per_frame)

        # === Positions ===
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # === Load teacher trajectory ===
        traj = batch["trajectories"]
        teacher_sigmas = traj["sigmas"].to(device)          # [B, N+1] or [N+1]
        teacher_states = traj["states"].to(device, dtype)    # [B, N+1, seq, C]
        teacher_velocities = traj["velocities"].to(device, dtype)  # [B, N, seq, C]
        teacher_x0_preds = traj["x0_preds"].to(device, dtype)     # [B, N, seq, C]
        teacher_x0_gt = traj["x0_gt"].to(device, dtype)           # [B, seq, C]

        if teacher_sigmas.dim() == 1:
            teacher_sigmas = teacher_sigmas.unsqueeze(0).expand(batch_size, -1)

        num_teacher_steps = teacher_velocities.shape[1]

        # ════════════════════════════════════════
        # ENCODER PASS (clean, σ=0, causal mask)
        # ════════════════════════════════════════
        assert self._scd_model is not None, "SCD model must be set"

        encoder_timesteps = torch.zeros(batch_size, video_seq_len, device=device, dtype=dtype)
        encoder_modality = Modality(
            enabled=True,
            latent=video_latents,
            timesteps=encoder_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        encoder_video_args, encoder_audio_args = self._scd_model.forward_encoder(
            video=encoder_modality,
            audio=None,
            perturbations=None,
            tokens_per_frame=tokens_per_frame,
        )

        # Shift encoder features by 1 frame
        encoder_features = encoder_video_args.x  # [B, seq, D]
        shifted_features = shift_encoder_features(
            encoder_features, tokens_per_frame, num_frames,
        )

        # ════════════════════════════════════════
        # NOISE: adapter or teacher's starting noise
        # ════════════════════════════════════════
        use_adapter = random.random() < cfg.alpha

        if use_adapter and self._noise_adapter is not None and not cfg.use_teacher_noise:
            # Sample inverse problem for task class
            ip_sample = self._inverse_problem_sampler.sample(video_latents)

            mu, log_sigma = self._noise_adapter(
                encoder_features=shifted_features.detach(),
                task_class=ip_sample.task_class,
            )
            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn_like(mu)
            video_noise = mu + sigma_adapter * eps

            adapter_mu = mu
            adapter_log_sigma = log_sigma
            task_class = ip_sample.task_class
            ip_task_name = ip_sample.task_name
            ip_noise_level = ip_sample.noise_level
            ip_observation = ip_sample.observation
        else:
            # Use teacher's starting noise z (states[0])
            video_noise = teacher_states[:, 0]  # [B, seq, C]
            adapter_mu = None
            adapter_log_sigma = None
            task_class = None
            ip_task_name = "unconditional"
            ip_noise_level = 0.0
            ip_observation = None
            use_adapter = False

        # ════════════════════════════════════════
        # DISTILLATION TARGET
        # ════════════════════════════════════════
        if cfg.distill_mode == "output_match":
            # Student at σ=1 must match teacher's final denoised output
            teacher_target = teacher_x0_preds[:, -1]  # [B, seq, C]
            video_targets = video_noise - teacher_target
            sigmas = torch.ones(batch_size, 1, device=device, dtype=dtype)
            noisy_video = video_noise  # At σ=1, x_t = z

        elif cfg.distill_mode == "velocity_match":
            # Train on random teacher trajectory point
            valid_steps = cfg.velocity_match_steps or list(range(num_teacher_steps))
            step_idx = random.choice(valid_steps)

            noisy_video = teacher_states[:, step_idx]
            sigma_val = teacher_sigmas[:, step_idx]
            video_targets = teacher_velocities[:, step_idx]
            sigmas = sigma_val.unsqueeze(-1)

        elif cfg.distill_mode == "progressive":
            teacher_target = teacher_states[:, -1]

            if cfg.student_steps == 1:
                video_targets = video_noise - teacher_target
                sigmas = torch.ones(batch_size, 1, device=device, dtype=dtype)
                noisy_video = video_noise
            else:
                student_sigmas = torch.linspace(
                    teacher_sigmas[0, 0].item(), 0.0, cfg.student_steps + 1, device=device,
                )
                step_idx = random.randint(0, cfg.student_steps - 1)

                s_start = student_sigmas[step_idx]
                s_end = student_sigmas[step_idx + 1]

                teacher_sigma_dists = (teacher_sigmas[0] - s_start).abs()
                closest_idx = teacher_sigma_dists.argmin().item()
                noisy_video = teacher_states[:, closest_idx]

                closest_end_idx = (teacher_sigmas[0] - s_end).abs().argmin().item()
                target_state = teacher_states[:, closest_end_idx]

                dt = s_end - s_start
                video_targets = (target_state - noisy_video) / dt.clamp(min=1e-6)
                sigmas = s_start.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        else:
            raise ValueError(f"Unknown distill_mode: {cfg.distill_mode}")

        # ════════════════════════════════════════
        # FIRST-FRAME CONDITIONING
        # ════════════════════════════════════════
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height, width=width,
            device=device,
            first_frame_conditioning_p=cfg.first_frame_conditioning_p,
        )

        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # ════════════════════════════════════════
        # BUILD DECODER MODALITY
        # ════════════════════════════════════════
        decoder_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze(),
        )

        decoder_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=decoder_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        video_loss_mask = ~video_conditioning_mask

        model_inputs = ModelInputs(
            video=decoder_modality,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

        # SCD-specific
        model_inputs._encoder_features = shifted_features
        model_inputs._scd_model = self._scd_model
        model_inputs._encoder_audio_args = encoder_audio_args
        model_inputs._per_frame_decoder = cfg.per_frame_decoder
        model_inputs._tokens_per_frame = tokens_per_frame
        model_inputs._num_frames = num_frames
        model_inputs._raw_video_latents = batch["latents"]["latents"]

        # VFM adapter
        model_inputs._vfm_adapter_mu = adapter_mu
        model_inputs._vfm_adapter_log_sigma = adapter_log_sigma
        model_inputs._vfm_task_class = task_class
        model_inputs._vfm_observation = ip_observation
        model_inputs._vfm_task_name = ip_task_name
        model_inputs._vfm_noise_level = ip_noise_level
        model_inputs._vfm_video_noise = video_noise
        model_inputs._vfm_video_latents = video_latents
        model_inputs._vfm_use_adapter = use_adapter

        # Distillation-specific (for visualization)
        model_inputs._distill_teacher_target = teacher_target if cfg.distill_mode != "velocity_match" else None
        model_inputs._distill_teacher_x0_gt = teacher_x0_gt
        model_inputs._distill_mode = cfg.distill_mode
        model_inputs._distill_teacher_states = teacher_states
        model_inputs._distill_teacher_sigmas = teacher_sigmas
        model_inputs._distill_teacher_velocities = teacher_velocities

        # For GT loss mixing
        if cfg.gt_weight > 0:
            model_inputs._gt_video_targets = video_noise - video_latents

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute distillation loss + KL + optional GT mixing.

        Losses:
        1. L_distill: MSE between student velocity and teacher target
        2. L_kl: KL(q_φ(z|encoder_features) || N(0,I))
        3. L_gt: Optional ground truth flow matching loss
        """
        cfg = self.config

        # ── L_distill: teacher-supervised loss ──
        video_loss = (video_pred - inputs.video_targets).pow(2)

        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_distill = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_distill = video_loss.mean()

        loss_distill_scaled = cfg.distill_weight * loss_distill

        # ── L_gt: optional ground truth anchor ──
        loss_gt = torch.tensor(0.0, device=video_pred.device)
        if cfg.gt_weight > 0 and hasattr(inputs, "_gt_video_targets"):
            gt_loss = (video_pred - inputs._gt_video_targets).pow(2)
            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                loss_gt = (gt_loss * mask).sum() / mask.sum().clamp(min=1) / gt_loss.shape[-1]
            else:
                loss_gt = gt_loss.mean()

        loss_gt_scaled = cfg.gt_weight * loss_gt

        # ── L_KL: adapter regularization ──
        loss_kl = torch.tensor(0.0, device=video_pred.device)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)

        if use_adapter and adapter_mu is not None:
            adapter_log_sigma = inputs._vfm_adapter_log_sigma
            loss_kl = 0.5 * (
                adapter_mu.pow(2)
                + torch.exp(2 * adapter_log_sigma)
                - 2 * adapter_log_sigma
                - 1
            ).mean()

        loss_kl_scaled = cfg.kl_weight * loss_kl

        # ── Total ──
        total_loss = loss_distill_scaled + loss_gt_scaled + loss_kl_scaled

        # Adaptive loss scaling
        if cfg.adaptive_loss:
            with torch.no_grad():
                weight = 1.0 / (total_loss.detach() + cfg.adaptive_gamma).pow(cfg.adaptive_p)
            total_loss = weight * total_loss

        # Store metrics
        self._last_vfm_metrics = {
            "vfm/loss_distill": loss_distill.item(),
            "vfm/loss_gt": loss_gt.item() if isinstance(loss_gt, Tensor) else loss_gt,
            "vfm/loss_kl": loss_kl.item() if isinstance(loss_kl, Tensor) else loss_kl,
            "vfm/loss_total": total_loss.item(),
            "vfm/distill_mode": cfg.distill_mode,
            "vfm/use_adapter": float(use_adapter),
        }
        if adapter_mu is not None:
            self._last_vfm_metrics["vfm/adapter_mu_mean"] = adapter_mu.mean().item()
            self._last_vfm_metrics["vfm/adapter_mu_std"] = adapter_mu.std().item()
            self._last_vfm_metrics["vfm/adapter_sigma_mean"] = torch.exp(
                inputs._vfm_adapter_log_sigma
            ).mean().item()

        return total_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        meta = super().get_checkpoint_metadata()
        meta["distill_mode"] = self.config.distill_mode
        meta["student_steps"] = self.config.student_steps
        meta["distill_weight"] = self.config.distill_weight
        meta["gt_weight"] = self.config.gt_weight
        return meta
