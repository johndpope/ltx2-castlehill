"""VFM Distillation Strategy — Train against teacher ODE trajectories.

Instead of random interpolation targets (v = z - x₀), this strategy uses
pre-computed teacher trajectories to provide supervision at specific sigma
points along the teacher's denoising path.

Distillation modes:
  1. "output_match" — Student's 1-step output must match teacher's 8-step output.
     Loss = ||f_θ(z, σ=1) - teacher_x̂₀||²
     This is the simplest and most direct approach for VFM.

  2. "velocity_match" — Student's velocity must match teacher's velocity at
     each sigma along the trajectory.
     Loss = Σ_k ||v_student(x_k, σ_k) - v_teacher(x_k, σ_k)||²
     Provides richer supervision across the full sigma range.

  3. "progressive" — Progressive distillation: halve steps each round.
     Round 1: Student does 4 steps matching teacher 8-step output.
     Round 2: Student does 2 steps matching round-1 output.
     Round 3: Student does 1 step matching round-2 output.
     Requires multiple training rounds with decreasing student_steps.

The noise adapter still learns qφ(z|y) with KL regularization.
The key difference is the flow map loss uses teacher-provided targets.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.vfm_strategy_v1b import VFMv1bTrainingConfig, VFMv1bTrainingStrategy
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.timestep_samplers import TimestepSampler


class VFMDistillConfig(VFMv1bTrainingConfig):
    """Config for VFM distillation training."""
    name: str = Field(default="vfm_distill")

    # Distillation mode
    distill_mode: Literal["output_match", "velocity_match", "progressive"] = Field(
        default="output_match",
        description="Distillation mode: output_match (1-step), velocity_match (multi-sigma), progressive (halving)",
    )

    # Progressive distillation: how many steps the student takes
    student_steps: int = Field(
        default=1,
        description="Number of Euler steps the student takes (1 for output_match, >1 for progressive)",
    )

    # Loss weights
    distill_weight: float = Field(
        default=1.0,
        description="Weight for distillation loss (teacher matching)",
    )
    gt_weight: float = Field(
        default=0.0,
        description="Weight for ground truth loss (original VFM flow matching). "
        "Set >0 to mix distillation with GT supervision.",
    )

    # Trajectory data
    trajectories_dir: str = Field(
        default="trajectories",
        description="Directory name for pre-computed trajectory files under data root",
    )

    # Use teacher's starting noise z (from trajectory) vs adapter noise
    use_teacher_noise: bool = Field(
        default=False,
        description="If True, use the teacher's starting noise z from trajectory "
        "instead of sampling from the adapter. Useful early in training.",
    )

    # Velocity match: which trajectory steps to train on
    velocity_match_steps: list[int] | None = Field(
        default=None,
        description="Which trajectory step indices to train velocity matching on. "
        "None = all steps. E.g., [0, 2, 4, 7] for sparse supervision.",
    )


class VFMDistillStrategy(VFMv1bTrainingStrategy):
    """VFM training with teacher trajectory distillation.

    Extends v1b strategy — keeps adapter (cross-attention + positions),
    KL regularization, and obs loss. Replaces the flow matching target
    with teacher-provided supervision.
    """

    def __init__(self, config: VFMDistillConfig) -> None:
        super().__init__(config)
        self._trajectory_cache: dict[int, dict] = {}

    def get_data_sources(self) -> dict[str, str]:
        """Add trajectories as a data source alongside latents + conditions."""
        sources = super().get_data_sources()
        sources[self.config.trajectories_dir] = "trajectories"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs using teacher trajectory for supervision.

        Key differences from base VFM:
        1. Load teacher trajectory (sigmas, states, velocities, x0_preds)
        2. Optionally use teacher's starting noise instead of adapter noise
        3. Set targets from teacher trajectory instead of random interpolation
        """
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

        # Load teacher trajectory
        traj = batch["trajectories"]
        teacher_sigmas = traj["sigmas"].to(device)        # [B, N+1] or [N+1]
        teacher_states = traj["states"].to(device, dtype)  # [B, N+1, seq, C]
        teacher_velocities = traj["velocities"].to(device, dtype)  # [B, N, seq, C]
        teacher_x0_preds = traj["x0_preds"].to(device, dtype)     # [B, N, seq, C]
        teacher_x0_gt = traj["x0_gt"].to(device, dtype)           # [B, seq, C]

        # Ensure sigmas have batch dim
        if teacher_sigmas.dim() == 1:
            teacher_sigmas = teacher_sigmas.unsqueeze(0).expand(batch_size, -1)

        num_teacher_steps = teacher_velocities.shape[1]

        # Positions
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # ════════════════════════════════════════════════════════════
        # NOISE: adapter or teacher's starting noise
        # ════════════════════════════════════════════════════════════
        use_adapter = random.random() < cfg.alpha

        if use_adapter and self._noise_adapter is not None and not cfg.use_teacher_noise:
            # v1b adapter: full text + positions → per-token μ,σ
            mu, log_sigma = self._noise_adapter(
                text_embeddings=video_prompt_embeds.detach(),
                text_mask=prompt_attention_mask,
                positions=video_positions,
                task_class=torch.zeros(batch_size, dtype=torch.long, device=device),  # i2v
            )
            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn_like(mu)
            video_noise = mu + sigma_adapter * eps
            adapter_mu = mu
            adapter_log_sigma = log_sigma
        else:
            # Use teacher's starting noise z (states[0])
            video_noise = teacher_states[:, 0]  # [B, seq, C]
            adapter_mu = None
            adapter_log_sigma = None
            use_adapter = False

        # ════════════════════════════════════════════════════════════
        # DISTILLATION TARGET based on mode
        # ════════════════════════════════════════════════════════════
        if cfg.distill_mode == "output_match":
            # Student starts from z at σ=1, must match teacher's final output in 1 step
            # Target: teacher's denoised x̂₀ (last x0_pred, or final state)
            teacher_target = teacher_x0_preds[:, -1]  # [B, seq, C] — best x̂₀ prediction

            # Velocity target for student at σ=1: v = z - x̂₀_teacher
            # (If student predicts v at σ=1, then x̂₀ = z - v, and we want x̂₀ = teacher_target)
            video_targets = video_noise - teacher_target

            # Use σ=1 for the modality timestep
            sigmas = torch.ones(batch_size, 1, device=device, dtype=dtype)
            noisy_video = video_noise  # At σ=1, x_t = z

        elif cfg.distill_mode == "velocity_match":
            # Train on random teacher trajectory point
            valid_steps = cfg.velocity_match_steps or list(range(num_teacher_steps))
            step_idx = random.choice(valid_steps)

            # Student input: teacher's state at step_idx
            noisy_video = teacher_states[:, step_idx]  # [B, seq, C]
            sigma_val = teacher_sigmas[:, step_idx]     # [B]

            # Target: teacher's velocity at this step
            video_targets = teacher_velocities[:, step_idx]  # [B, seq, C]

            sigmas = sigma_val.unsqueeze(-1)  # [B, 1]

        elif cfg.distill_mode == "progressive":
            # Progressive distillation: student takes student_steps,
            # must match what teacher achieves in num_teacher_steps
            # For now: student starts from z, target is teacher's 8-step output
            teacher_target = teacher_states[:, -1]  # Teacher's final denoised state

            if cfg.student_steps == 1:
                # Same as output_match
                video_targets = video_noise - teacher_target
                sigmas = torch.ones(batch_size, 1, device=device, dtype=dtype)
                noisy_video = video_noise
            else:
                # Multi-step student: pick a random pair of student sigma points
                # Student sigma schedule: evenly spaced in teacher's sigma range
                student_sigmas = torch.linspace(
                    teacher_sigmas[0, 0].item(), 0.0, cfg.student_steps + 1, device=device
                )
                step_idx = random.randint(0, cfg.student_steps - 1)

                s_start = student_sigmas[step_idx]
                s_end = student_sigmas[step_idx + 1]

                # Find teacher's state closest to s_start
                teacher_sigma_dists = (teacher_sigmas[0] - s_start).abs()
                closest_idx = teacher_sigma_dists.argmin().item()
                noisy_video = teacher_states[:, closest_idx]

                # Target: what teacher produces after going from s_start to s_end
                closest_end_idx = (teacher_sigmas[0] - s_end).abs().argmin().item()
                target_state = teacher_states[:, closest_end_idx]

                # Velocity target: v such that x + v * dt = target
                dt = s_end - s_start
                video_targets = (target_state - noisy_video) / dt.clamp(min=1e-6)

                sigmas = s_start.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)

        else:
            raise ValueError(f"Unknown distill_mode: {cfg.distill_mode}")

        # ════════════════════════════════════════════════════════════
        # BUILD MODALITY
        # ════════════════════════════════════════════════════════════
        # First-frame conditioning mask (usually disabled for distill, but keep for compatibility)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height, width=width,
            device=device,
            first_frame_conditioning_p=cfg.first_frame_conditioning_p,
        )

        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        video_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

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

        # VFM-specific data for loss computation
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

        # Distillation-specific — keep full trajectory for visualization
        model_inputs._distill_teacher_target = teacher_target if cfg.distill_mode != "velocity_match" else None
        model_inputs._distill_teacher_x0_gt = teacher_x0_gt
        model_inputs._distill_mode = cfg.distill_mode
        model_inputs._distill_teacher_states = teacher_states   # [B, N+1, seq, C]
        model_inputs._distill_teacher_sigmas = teacher_sigmas   # [B, N+1]
        model_inputs._distill_teacher_velocities = teacher_velocities  # [B, N, seq, C]

        # For GT loss mixing
        if cfg.gt_weight > 0:
            model_inputs._gt_video_targets = video_noise - video_latents  # Original v = z - x₀

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute distillation loss + optional KL + optional GT mixing.

        Losses:
        1. L_distill: MSE between student velocity and teacher target velocity
        2. L_kl: KL(q_φ(z|y) || N(0,I)) — adapter regularization
        3. L_gt: Optional ground truth flow matching loss for stability
        """
        cfg = self.config

        # ── L_distill: teacher-supervised flow matching loss ──
        video_loss = (video_pred - inputs.video_targets).pow(2)

        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_distill = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_distill = video_loss.mean()

        # Apply Min-SNR weighting if configured
        if cfg.min_snr_gamma is not None:
            sigmas = getattr(inputs, "shared_sigmas", None)
            if sigmas is not None:
                sigma_val = sigmas.mean().clamp(min=1e-4, max=1.0 - 1e-4)
                snr = ((1.0 - sigma_val) ** 2) / (sigma_val ** 2 + 1e-8)
                snr_weight = (torch.clamp(snr, max=cfg.min_snr_gamma) / (snr + 1e-8)).detach()
                loss_distill = loss_distill * snr_weight

        total_loss = cfg.distill_weight * loss_distill

        # ── L_kl: adapter KL regularization ──
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        if use_adapter and adapter_mu is not None and cfg.kl_weight > 0:
            adapter_log_sigma = inputs._vfm_adapter_log_sigma
            kl_raw = 0.5 * (adapter_mu.pow(2) + torch.exp(2 * adapter_log_sigma) - 2 * adapter_log_sigma - 1)
            kl_per_sample = kl_raw.mean(dim=(1, 2))

            if cfg.kl_free_bits > 0:
                kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)

            loss_kl = kl_per_sample.mean()
            total_loss = total_loss + cfg.kl_weight * loss_kl

        # ── L_gt: optional ground truth flow matching loss ──
        if cfg.gt_weight > 0 and hasattr(inputs, "_gt_video_targets"):
            gt_loss = (video_pred - inputs._gt_video_targets).pow(2)
            if inputs.video_loss_mask is not None:
                gt_loss = (gt_loss * mask).sum() / mask.sum().clamp(min=1) / gt_loss.shape[-1]
            else:
                gt_loss = gt_loss.mean()
            total_loss = total_loss + cfg.gt_weight * gt_loss

        # ── Logging ──
        if cfg.log_vfm_metrics and hasattr(inputs, "_log_step"):
            step = inputs._log_step
            metrics = {
                "vfm/loss_distill": loss_distill.item(),
                "vfm/distill_mode": cfg.distill_mode,
            }
            if use_adapter and adapter_mu is not None:
                metrics["vfm/adapter_mu_mean"] = adapter_mu.float().mean().item()
                metrics["vfm/adapter_sigma_mean"] = torch.exp(inputs._vfm_adapter_log_sigma).float().mean().item()
                if cfg.kl_weight > 0:
                    metrics["vfm/loss_kl"] = loss_kl.item()
            if cfg.gt_weight > 0 and hasattr(inputs, "_gt_video_targets"):
                metrics["vfm/loss_gt"] = gt_loss.item()

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except ImportError:
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
        """Log video reconstruction + trajectory plots to W&B.

        Extends parent to add plotly trajectory visualizations showing:
        1. Teacher ODE path vs student 1-step jump in PCA latent space
        2. L2 distance to GT at each sigma (teacher converges, student jumps)
        3. Per-step velocity magnitude comparison
        """
        # Parent handles video reconstruction
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        # Add trajectory plots
        try:
            traj_plots = self._build_trajectory_plots(video_pred, inputs, step, prefix)
            log_dict.update(traj_plots)
        except Exception as e:
            logger.warning(f"Failed to build trajectory plots: {e}")

        return log_dict

    @staticmethod
    def _build_trajectory_plots(
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Build plotly trajectory figures for W&B logging."""
        import wandb

        teacher_states = getattr(inputs, "_distill_teacher_states", None)
        teacher_sigmas = getattr(inputs, "_distill_teacher_sigmas", None)
        teacher_velocities = getattr(inputs, "_distill_teacher_velocities", None)
        teacher_x0_gt = getattr(inputs, "_distill_teacher_x0_gt", None)

        if teacher_states is None or teacher_sigmas is None:
            return {}

        # Work on first sample in batch, move to CPU
        states = teacher_states[0].float().cpu()    # [N+1, seq, C]
        sigmas = teacher_sigmas[0].float().cpu()    # [N+1]
        x0_gt = teacher_x0_gt[0].float().cpu()      # [seq, C]

        noise = inputs.shared_noise[0].float().cpu()  # [seq, C] — z
        student_x0 = (noise - video_pred[0].float().cpu())  # x̂₀ = z - v_pred

        num_steps = states.shape[0]  # N+1 points
        seq_len = states.shape[1]

        log_dict = {}

        # ── Plot 1: L2 distance to GT at each sigma ──────────────────
        try:
            import plotly.graph_objects as go

            # Teacher: distance from each state to GT
            teacher_dists = []
            for i in range(num_steps):
                dist = (states[i] - x0_gt).pow(2).mean().sqrt().item()
                teacher_dists.append(dist)

            # Student: start (z) and end (x̂₀)
            student_start_dist = (noise - x0_gt).pow(2).mean().sqrt().item()
            student_end_dist = (student_x0 - x0_gt).pow(2).mean().sqrt().item()

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(
                x=sigmas.tolist(),
                y=teacher_dists,
                mode="lines+markers",
                name="Teacher (8-step ODE)",
                line=dict(color="#2196F3", width=3),
                marker=dict(size=8),
            ))
            fig_dist.add_trace(go.Scatter(
                x=[1.0, 0.0],
                y=[student_start_dist, student_end_dist],
                mode="lines+markers",
                name="Student (1-step)",
                line=dict(color="#FF5722", width=3, dash="dash"),
                marker=dict(size=12, symbol="star"),
            ))
            fig_dist.update_layout(
                title=f"Distance to GT along ODE Path (step {step})",
                xaxis_title="σ (noise level)",
                yaxis_title="L2 distance to x₀ (RMS)",
                xaxis=dict(autorange="reversed"),
                template="plotly_dark",
                legend=dict(x=0.02, y=0.98),
                height=400, width=700,
            )
            log_dict[f"{prefix}/trajectory_distance"] = wandb.Plotly(fig_dist)
        except Exception as e:
            logger.debug(f"Trajectory distance plot failed: {e}")

        # ── Plot 2: PCA projection of trajectory in latent space ─────
        try:
            import plotly.graph_objects as go

            # Average over tokens → [N+1, C] mean latent per step
            state_means = states.mean(dim=1)    # [N+1, C]
            gt_mean = x0_gt.mean(dim=0)          # [C]
            student_x0_mean = student_x0.mean(dim=0)  # [C]

            # Stack all points for PCA: teacher states + GT + student x̂₀
            all_points = torch.cat([
                state_means,          # N+1 teacher points
                gt_mean.unsqueeze(0),  # 1 GT point
                student_x0_mean.unsqueeze(0),  # 1 student point
            ], dim=0)  # [N+3, C]

            # Simple PCA via SVD (center first)
            centered = all_points - all_points.mean(dim=0, keepdim=True)
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            pca_2d = (centered @ Vh[:2].T).numpy()  # [N+3, 2]

            teacher_pca = pca_2d[:num_steps]
            gt_pca = pca_2d[num_steps]
            student_pca = pca_2d[num_steps + 1]

            # Variance explained
            var_explained = (S[:2] ** 2 / (S ** 2).sum() * 100).numpy()

            fig_pca = go.Figure()

            # Teacher trajectory (connected path)
            fig_pca.add_trace(go.Scatter(
                x=teacher_pca[:, 0], y=teacher_pca[:, 1],
                mode="lines+markers+text",
                name="Teacher ODE path",
                text=[f"σ={s:.2f}" for s in sigmas.tolist()],
                textposition="top center",
                textfont=dict(size=9),
                line=dict(color="#2196F3", width=2),
                marker=dict(
                    size=[14] + [8] * (num_steps - 2) + [14],
                    color=sigmas.tolist(),
                    colorscale="Blues_r",
                    showscale=True,
                    colorbar=dict(title="σ", x=1.02),
                ),
            ))

            # Student 1-step jump (z → x̂₀_student)
            fig_pca.add_trace(go.Scatter(
                x=[teacher_pca[0, 0], student_pca[0]],
                y=[teacher_pca[0, 1], student_pca[1]],
                mode="lines+markers",
                name="Student 1-step",
                line=dict(color="#FF5722", width=3, dash="dash"),
                marker=dict(size=14, symbol="star", color="#FF5722"),
            ))

            # Ground truth
            fig_pca.add_trace(go.Scatter(
                x=[gt_pca[0]], y=[gt_pca[1]],
                mode="markers+text",
                name="Ground truth x₀",
                text=["GT"],
                textposition="bottom center",
                marker=dict(size=16, symbol="diamond", color="#4CAF50"),
            ))

            fig_pca.update_layout(
                title=f"Latent Space Trajectory — PCA (step {step})",
                xaxis_title=f"PC1 ({var_explained[0]:.1f}% var)",
                yaxis_title=f"PC2 ({var_explained[1]:.1f}% var)",
                template="plotly_dark",
                legend=dict(x=0.02, y=0.98),
                height=500, width=700,
            )
            log_dict[f"{prefix}/trajectory_pca"] = wandb.Plotly(fig_pca)
        except Exception as e:
            logger.debug(f"Trajectory PCA plot failed: {e}")

        # ── Plot 3: Velocity magnitude comparison ─────────────────────
        try:
            import plotly.graph_objects as go

            if teacher_velocities is not None:
                t_vels = teacher_velocities[0].float().cpu()  # [N, seq, C]
                # RMS velocity magnitude per step
                teacher_vel_mags = t_vels.pow(2).mean(dim=(1, 2)).sqrt().tolist()  # [N]
                teacher_sigma_mids = [
                    (sigmas[i].item() + sigmas[i + 1].item()) / 2
                    for i in range(len(sigmas) - 1)
                ]

                # Student velocity = v_pred
                student_vel_mag = video_pred[0].float().cpu().pow(2).mean().sqrt().item()

                fig_vel = go.Figure()
                fig_vel.add_trace(go.Bar(
                    x=[f"σ={s:.2f}" for s in teacher_sigma_mids],
                    y=teacher_vel_mags,
                    name="Teacher velocity (per step)",
                    marker_color="#2196F3",
                ))
                fig_vel.add_hline(
                    y=student_vel_mag,
                    line_dash="dash", line_color="#FF5722", line_width=2,
                    annotation_text=f"Student 1-step: {student_vel_mag:.3f}",
                    annotation_position="top right",
                )
                fig_vel.update_layout(
                    title=f"Velocity Magnitude: Teacher Steps vs Student (step {step})",
                    xaxis_title="Sigma midpoint",
                    yaxis_title="RMS velocity ||v||",
                    template="plotly_dark",
                    height=400, width=700,
                )
                log_dict[f"{prefix}/trajectory_velocity"] = wandb.Plotly(fig_vel)
        except Exception as e:
            logger.debug(f"Trajectory velocity plot failed: {e}")

        # ── Plot 4: Sigma schedule shape ──────────────────────────────
        try:
            import plotly.graph_objects as go

            step_indices = list(range(num_steps))
            sigma_vals = sigmas.tolist()

            # Compare with uniform linear schedule
            linear_sigmas = torch.linspace(1.0, 0.0, num_steps).tolist()

            fig_sched = go.Figure()
            fig_sched.add_trace(go.Scatter(
                x=step_indices, y=sigma_vals,
                mode="lines+markers",
                name="Teacher schedule (actual)",
                line=dict(color="#2196F3", width=3),
                marker=dict(size=8),
            ))
            fig_sched.add_trace(go.Scatter(
                x=step_indices, y=linear_sigmas,
                mode="lines+markers",
                name="Linear baseline",
                line=dict(color="#9E9E9E", width=2, dash="dot"),
                marker=dict(size=6),
            ))

            # Show step sizes (dt = σ[i+1] - σ[i])
            dt_actual = [sigma_vals[i] - sigma_vals[i + 1] for i in range(num_steps - 1)]
            fig_sched.add_trace(go.Bar(
                x=[f"{i}→{i+1}" for i in range(num_steps - 1)],
                y=dt_actual,
                name="Step size Δσ",
                marker_color="#FF9800",
                opacity=0.5,
                yaxis="y2",
            ))

            fig_sched.update_layout(
                title=f"Sigma Schedule Shape (step {step})",
                xaxis_title="Step index",
                yaxis_title="σ",
                yaxis2=dict(title="Δσ", overlaying="y", side="right", range=[0, max(dt_actual) * 2]),
                template="plotly_dark",
                legend=dict(x=0.02, y=0.98),
                height=400, width=700,
            )
            log_dict[f"{prefix}/trajectory_schedule"] = wandb.Plotly(fig_sched)
        except Exception as e:
            logger.debug(f"Trajectory schedule plot failed: {e}")

        return log_dict
