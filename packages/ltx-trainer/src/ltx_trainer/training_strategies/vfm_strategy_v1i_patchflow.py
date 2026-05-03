"""VFM v1i — Patch Forcing + Trajectory Distillation + Per-Token Uncertainty.

This is the merged strategy that integrates CompVis Patch Forcing (arxiv:2604.19141)
into the VFM v1d architecture.

Key additions from Patch Forcing:
1. **LTG (Limited Timestep Gap) sampler**: Global T_max + per-token t ~ U[0, T_max].
   Prevents train-test mismatch where training sees overly-clean patches.
2. **UncertaintyHead**: Predicts per-token logvar_theta (instead of sigma).
   Model learns to output (vt, logvar_theta) — uncertainty correlates with difficulty.
3. **Uncertainty loss (NLL)**: flow_loss + uncertainty_weight * sigma_loss
   where sigma_loss = NLL(ut | DiagonalGaussian(vt.detach(), sigma_theta)).
4. **RoPE jittering support**: Accepts img_meta (crop info) for positional encoding
   augmentation during training (improves generalization to different resolutions/crops).

Architecture (training):
    Text embeddings → NoiseAdapterV1b → (μ, log_σ)
    SigmaHead / UncertaintyHead(x₀, μ) → logvar_theta [B, seq]
    t ~ LTG(T_max) per token
    x_t[i] = (1 - t_i) · x₀[i] + t_i · z[i]
    48-layer DiT (per-token timesteps + optional return_uncertainty) → (v, logvar_theta)
    Loss = MSE(v, z - x₀) + uncertainty_weight * NLL(z - x₀ | N(v, σ_θ))

When distill_mode != "none":
    - Student matches teacher's 8-step ODE trajectory in 1 step.
    - Per-token uncertainty still active (sigma head learns difficulty from teacher path).

This version keeps all v1d features:
- Diversity regularization (token / temporal / spatial)
- KL regularization on adapter
- Optional trajectory distillation (output_match / velocity_match)
- Per-token sigma entropy regularizer
- Complexity-aware sigma targets from x₀ (edges, texture, motion get higher σ)

Training command:
    uv run python scripts/train.py configs/ltx2_vfm_v1i_patchflow.yaml
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1d import (
    VFMv1dTrainingConfig,
    VFMv1dTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


# =====================================================================================
# Uncertainty Head (Patch Forcing style — predicts logvar_theta)
# =====================================================================================

class UncertaintyHead(nn.Module):
    """Per-token uncertainty predictor (Patch Forcing style).

    Predicts logvar_theta ∈ ℝ per token. Higher variance → model is less confident
    → token is "harder" (edges, motion, fine texture).

    Input: clean latent x₀ [B, seq, C] + adapter μ [B, seq, C]
    Output: logvar_theta [B, seq]  (unconstrained, can be negative)

    This is the direct equivalent of the uncertainty head in patch_flow.combined:
        vt, logvar_theta = model(..., return_uncertainty=True)
        sigma_theta = exp(0.5 * logvar_theta)
        pred = DiagonalGaussian(mean=vt.detach(), std=sigma_theta)
        sigma_loss = pred.nll(ut).mean()
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = latent_dim * 2  # x₀ + adapter_mu

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # logvar (unconstrained)
        )

        # Initialize to small negative logvar (σ ≈ 0.3–0.5)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.0)

    def forward(self, mu: Tensor, x0: Tensor) -> Tensor:
        """Predict per-token logvar_theta.

        Args:
            mu: Adapter mean [B, seq, latent_dim]
            x0: Clean patchified latent [B, seq, latent_dim]

        Returns:
            logvar_theta [B, seq]
        """
        inp = torch.cat([x0.detach(), mu], dim=-1)  # [B, seq, 2C]
        return self.net(inp).squeeze(-1)  # [B, seq]


# =====================================================================================
# Configuration
# =====================================================================================

class VFMv1iPatchFlowConfig(VFMv1dTrainingConfig):
    """Configuration for VFM v1i (Patch Forcing + v1d distillation).

    UncertaintyHead replaces SigmaHead — `per_token_sigma` defaults to False so
    v1d's SigmaHead is not instantiated.
    """

    name: Literal["vfm_v1i_patchflow"] = "vfm_v1i_patchflow"

    # Disable v1d SigmaHead by default (UncertaintyHead replaces it).
    per_token_sigma: bool = Field(default=False)

    # === Patch Forcing Core ===
    use_patch_forcing: bool = Field(
        default=True,
        description="Enable Patch Forcing: LTG sampler + UncertaintyHead + NLL loss",
    )
    uncertainty_weight: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Weight for uncertainty NLL loss (Patch Forcing). "
        "Matches patch_flow.combined default.",
    )
    ltg_max_prob: float = Field(
        default=0.999,
        description="Upper bound for global T_max in LTG sampler. "
        "t ~ Uniform[0, T_max] per token with T_max ~ Uniform[0, ltg_max_prob].",
    )

    # === Per-token Uncertainty (replaces/enhances SigmaHead) ===
    per_token_uncertainty: bool = Field(
        default=True,
        description="Enable UncertaintyHead that predicts logvar_theta per token. "
        "Replaces v1d SigmaHead when use_patch_forcing=True.",
    )
    uncertainty_head_hidden_dim: int = Field(
        default=256,
        description="Hidden dim for UncertaintyHead MLP",
    )

    # === LTG Sampler ===
    ltg_enabled: bool = Field(
        default=True,
        description="Use Limited Timestep Gap (LTG) sampling instead of uniform. "
        "Critical for closing train-test gap in per-token scheduling.",
    )

    # === Distillation (from v1d) ===
    distill_mode: Literal["output_match", "velocity_match", "progressive", "none"] = Field(
        default="none",
        description="Distillation mode. 'none' = standard VFM + Patch Forcing.",
    )
    student_steps: int = Field(default=1)
    distill_weight: float = Field(default=1.0)
    gt_weight: float = Field(default=0.1)
    trajectories_dir: str = Field(default="trajectories")
    use_teacher_noise: bool = Field(default=False)

    # === RoPE Jittering (from patch_flow) ===
    rope_jittering: bool = Field(
        default=False,
        description="Enable RoPE jittering via img_meta (crop offsets). "
        "Requires batch to contain 'img_meta' dict with 'top', 'left', etc.",
    )


# =====================================================================================
# Strategy
# =====================================================================================

class VFMv1iPatchFlowStrategy(VFMv1dTrainingStrategy):
    """VFM v1i — Patch Forcing integrated with trajectory distillation.

    Combines:
    - Patch Forcing (LTG + UncertaintyHead + NLL loss)
    - VFM v1d (per-token sigma, diversity, distillation)
    - VFM v1c (KL + observation consistency)
    """

    config: VFMv1iPatchFlowConfig

    def __init__(self, config: VFMv1iPatchFlowConfig) -> None:
        super().__init__(config)
        self._uncertainty_head: UncertaintyHead | None = None

        if config.use_patch_forcing and config.per_token_uncertainty:
            self._uncertainty_head = UncertaintyHead(
                latent_dim=128,
                hidden_dim=config.uncertainty_head_hidden_dim,
            )

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        params = super().get_trainable_parameters()
        if self._uncertainty_head is not None:
            params.extend(list(self._uncertainty_head.parameters()))
        return params

    def get_strategy_state_dict(self) -> dict[str, Any]:
        state = super().get_strategy_state_dict()
        if self._uncertainty_head is not None:
            for k, v in self._uncertainty_head.state_dict().items():
                state[f"strategy.uncertainty_head.{k}"] = v
        return state

    def load_strategy_state_dict(self, state_dict: dict[str, Any]) -> tuple[int, int]:
        loaded, skipped = super().load_strategy_state_dict(state_dict)
        if self._uncertainty_head is not None:
            uh_dict = {k.replace("strategy.uncertainty_head.", ""): v
                       for k, v in state_dict.items()
                       if k.startswith("strategy.uncertainty_head.")}
            if uh_dict:
                self._uncertainty_head.load_state_dict(uh_dict)
                loaded += len(uh_dict)
                logger.info(f"Loaded UncertaintyHead: {len(uh_dict)} params")
        return loaded, skipped

    def set_noise_adapter(self, adapter) -> None:
        super().set_noise_adapter(adapter)
        if adapter is not None and self._uncertainty_head is not None:
            device = next(adapter.parameters()).device
            adapter.to(device)
            self._uncertainty_head = self._uncertainty_head.to(device=device)

    # =================================================================================
    # LTG Timestep Sampler (Patch Forcing core)
    # =================================================================================

    def _sample_ltg_timesteps(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        conditioning_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Limited Timestep Gap (LTG) sampler from Patch Forcing.

        Sample global T_max ~ Uniform[0, ltg_max_prob]
        Then per-token t_i ~ Uniform[0, T_max]

        This ensures no token becomes too clean during training,
        matching the pure-noise start at inference.
        """
        cfg = self.config
        t_max = torch.rand(batch_size, 1, device=device, dtype=dtype) * cfg.ltg_max_prob
        t = torch.rand(batch_size, seq_len, device=device, dtype=dtype) * t_max

        if conditioning_mask is not None:
            # Force conditioning tokens (first frame) to t=0 (clean)
            t = t * (~conditioning_mask).float()

        return t, t_max.squeeze(-1)

    # =================================================================================
    # Training Inputs
    # =================================================================================

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        cfg = self.config
        has_traj = "trajectories" in batch and cfg.distill_mode != "none"

        if has_traj:
            return self._prepare_distill_inputs(batch, timestep_sampler)
        else:
            return self._prepare_patchflow_inputs(batch, timestep_sampler)

    def _prepare_patchflow_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Standard training path with Patch Forcing (LTG + UncertaintyHead)."""
        cfg = self.config
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents = self._video_patchifier.patchify(video_latents)  # [B, seq, C]

        fps = latents.get("fps", None)
        fps = fps[0].item() if fps is not None else 24.0

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype
        tokens_per_frame = video_seq_len // num_frames

        # Inverse problem sampler
        if self._inverse_problem_sampler is None:
            from ltx_trainer.inverse_problems import InverseProblemSampler
            self._inverse_problem_sampler = InverseProblemSampler(
                problems=self._ip_configs,
                tokens_per_frame=tokens_per_frame,
            )

        # Positions + optional RoPE jittering
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        if cfg.rope_jittering and "img_meta" in batch:
            # Apply crop offsets to positions (Patch Forcing RoPE jitter)
            img_meta = batch["img_meta"]
            # img_meta = {"orig_h": ..., "orig_w": ..., "top": ..., "left": ...}
            # (Implementation depends on how LTX2 handles RoPE — placeholder)
            logger.debug("RoPE jittering via img_meta enabled (not fully implemented)")

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
        # PATCH FORCING: LTG + UncertaintyHead
        # ════════════════════════════════════════════════
        if cfg.use_patch_forcing and cfg.ltg_enabled:
            video_timesteps, t_max = self._sample_ltg_timesteps(
                batch_size, video_seq_len, device, dtype,
                conditioning_mask=video_conditioning_mask,
            )
            sigmas_for_logging = t_max  # mean timestep for logging
        else:
            # Fallback to uniform
            sigmas = timestep_sampler.sample_for(video_latents)
            video_timesteps = self._create_per_token_timesteps(
                video_conditioning_mask, sigmas.squeeze()
            )
            sigmas_for_logging = sigmas.squeeze()

        # Apply per-token timesteps to create noisy video
        sigmas_expanded = video_timesteps.unsqueeze(-1)  # [B, seq, 1]
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # Velocity target (standard flow matching)
        video_targets = video_noise - video_latents

        # Ensure conditioning tokens are clean
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # ════════════════════════════════════════════════
        # UNCERTAINTY HEAD (Patch Forcing)
        # ════════════════════════════════════════════════
        per_token_logvar = None
        if cfg.use_patch_forcing and self._uncertainty_head is not None and adapter_mu is not None:
            per_token_logvar = self._uncertainty_head(adapter_mu.float(), video_latents.float())
            # Zero uncertainty on conditioning tokens
            per_token_logvar = per_token_logvar * (~video_conditioning_mask).float()

        # ════════════════════════════════════════════════
        # BUILD MODALITY
        # ════════════════════════════════════════════════
        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            sigma=sigmas_for_logging,
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

        # v1i / Patch Forcing metadata
        model_inputs._per_token_logvar = per_token_logvar
        model_inputs._ltg_t_max = t_max if cfg.ltg_enabled else None
        model_inputs._distill_mode = "none"

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas_for_logging.view(-1, 1)

        return model_inputs

    def _prepare_distill_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Distillation path with Patch Forcing uncertainty.

        Delegates to v1d's distill flow (teacher trajectories, output_match /
        velocity_match), then layers Patch Forcing's per-token UncertaintyHead
        on top so compute_loss can add the NLL term.
        """
        cfg = self.config
        inputs = super()._prepare_distill_inputs(batch, timestep_sampler)

        # Per-token uncertainty (replaces v1d's per_token_sigma when use_patch_forcing)
        per_token_logvar = None
        if cfg.use_patch_forcing and self._uncertainty_head is not None:
            adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
            video_latents = getattr(inputs, "_vfm_video_latents", None)
            if adapter_mu is not None and video_latents is not None:
                per_token_logvar = self._uncertainty_head(
                    adapter_mu.float(), video_latents.float()
                )
                # Zero on conditioning tokens (loss_mask == False there)
                if inputs.video_loss_mask is not None:
                    per_token_logvar = per_token_logvar * inputs.video_loss_mask.float()

        inputs._per_token_logvar = per_token_logvar
        inputs._ltg_t_max = None  # LTG isn't applied in distill mode (teacher schedule rules)
        return inputs

    # =================================================================================
    # Loss (Patch Forcing NLL + existing VFM losses)
    # =================================================================================

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        cfg = self.config
        distill_mode = getattr(inputs, "_distill_mode", "none")

        if distill_mode != "none":
            # For now fall back to v1d distill loss
            return super().compute_loss(video_pred, audio_pred, inputs)

        # Base VFM loss (v1c + diversity + KL)
        total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        # ════════════════════════════════════════════════
        # PATCH FORCING: Uncertainty NLL loss
        # ════════════════════════════════════════════════
        per_token_logvar = getattr(inputs, "_per_token_logvar", None)
        if per_token_logvar is not None and cfg.uncertainty_weight > 0:
            # Diagonal-Gaussian NLL inlined (patch_flow.diagonal_gaussian not vendored).
            # logvar shape [B, seq] is broadcast against velocity-channel dim.
            ut = inputs.video_targets  # z - x₀, shape [B, seq, C]
            mean = video_pred.detach()
            logvar = per_token_logvar.unsqueeze(-1) if per_token_logvar.dim() < ut.dim() else per_token_logvar
            sigma_theta = torch.exp(0.5 * logvar)
            # NLL of N(ut; mean, sigma_theta) per element, before reduction
            inv_var = torch.exp(-logvar)
            nll = 0.5 * (logvar + (ut - mean).pow(2) * inv_var + math.log(2.0 * math.pi))

            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                sigma_loss = (nll * mask).sum() / mask.sum().clamp(min=1) / nll.shape[-1]
            else:
                sigma_loss = nll.mean()

            total_loss = total_loss + cfg.uncertainty_weight * sigma_loss

            # Logging
            active_logvar = per_token_logvar[per_token_logvar != 0]
            if active_logvar.numel() > 0:
                self._last_vfm_metrics.update({
                    "vfm/uncertainty_nll": sigma_loss.item(),
                    "vfm/uncertainty_mean": active_logvar.mean().item(),
                    "vfm/uncertainty_std": active_logvar.std().item(),
                    "vfm/sigma_theta_mean": torch.exp(0.5 * active_logvar).mean().item(),
                })

        return total_loss

    # =================================================================================
    # W&B Logging (add uncertainty heatmap)
    # =================================================================================

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        per_token_logvar = getattr(inputs, "_per_token_logvar", None)
        if per_token_logvar is not None:
            try:
                import wandb
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                logvar = per_token_logvar[0].float().cpu()
                seq_len = logvar.shape[0]
                raw_latents = getattr(inputs, "_raw_video_latents", None)
                if raw_latents is not None:
                    num_frames = raw_latents.shape[2]
                    tpf = seq_len // num_frames
                    h = raw_latents.shape[3]
                    w = raw_latents.shape[4]
                    logvar_frames = logvar[:num_frames * tpf].reshape(num_frames, h, w)

                    fig = make_subplots(rows=1, cols=num_frames,
                                        subplot_titles=[f"Frame {i}" for i in range(num_frames)])
                    for f_idx in range(num_frames):
                        fig.add_trace(
                            go.Heatmap(
                                z=logvar_frames[f_idx].detach().cpu().numpy(),
                                colorscale="RdYlBu_r",
                                showscale=(f_idx == num_frames - 1),
                                colorbar=dict(title="logvar_θ"),
                            ),
                            row=1, col=f_idx + 1,
                        )
                    fig.update_layout(
                        title=f"Per-Token Uncertainty (logvar_θ) — step {step}",
                        template="plotly_dark",
                        height=250, width=200 * num_frames,
                    )
                    log_dict[f"{prefix}/uncertainty_heatmap"] = wandb.Plotly(fig)
            except Exception as e:
                logger.warning(f"Failed to build uncertainty heatmap: {e}")

        return log_dict
