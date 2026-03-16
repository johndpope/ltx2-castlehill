"""VFM v1f — Spherical Cauchy noise adapter (v1d-spherical).

Branches from v1d (trajectory distillation + per-token sigma) and replaces
the Gaussian noise distribution with Spherical Cauchy on S^127.

Key changes from v1d:
1. **Spherical Cauchy sampling**: Adapter noise z sampled from SphericalCauchy(μ̂, κ)
   instead of N(μ, σ²I). Heavy-tailed distribution enables broader exploration
   early in training and faster convergence (20-30% per spherical-vae benchmarks).

2. **Direction-magnitude decomposition**: Adapter's (μ, log_σ) outputs reinterpreted:
   - μ̂ = normalize(μ) → direction on S^127
   - r = ||μ|| → magnitude (learned noise scale)
   - κ = exp(mean(log_σ, dim=-1)) → concentration (scalar per token)
   - z = r · sample_spherical_cauchy(μ̂, κ)

3. **Spherical KL**: Gaussian KL replaced with closed-form spherical Cauchy KL:
   KL(SpCauchy(μ̂, κ) || Uniform(S^127)) = (D-1)/2 · log(1 + 1/κ)
   Much cheaper to compute, better behaved for high-dim latents.

4. **Optional SLERP interpolation**: Flow matching interpolation can use SLERP
   on the directional component for geodesically smooth trajectories.

Architecture (unchanged adapter, reinterpreted outputs):
    Text embeddings → NoiseAdapterV1b → (μ, log_σ) per token
    μ̂ = normalize(μ), r = ||μ||, κ = exp(mean(log_σ))
    z_dir ~ SphericalCauchy(μ̂, κ), z = r · z_dir
    z → SigmaHead → per-token σ_i  (inherited from v1d)
    x_t[i] = (1 - σ_i) · x₀[i] + σ_i · z[i]
    48-layer DiT → velocity v → x̂₀
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.spherical_utils import (
    geodesic_distance,
    kl_spherical_cauchy_to_uniform,
    normalize,
    sample_spherical_cauchy,
    slerp,
)
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1d import (
    VFMv1dTrainingConfig,
    VFMv1dTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class VFMv1fTrainingConfig(VFMv1dTrainingConfig):
    """Configuration for VFM v1f (spherical Cauchy noise adapter)."""

    name: Literal["vfm_v1f"] = "vfm_v1f"

    # === Spherical Cauchy ===
    spherical_noise: bool = Field(
        default=True,
        description="Use Spherical Cauchy noise distribution instead of Gaussian. "
        "Adapter outputs are reinterpreted as direction + magnitude + concentration.",
    )
    kappa_min: float = Field(
        default=0.1, ge=0.01,
        description="Minimum kappa (concentration). Low κ → broad/uniform, high κ → peaked.",
    )
    kappa_max: float = Field(
        default=50.0, le=200.0,
        description="Maximum kappa to prevent numerical issues.",
    )
    use_slerp_interp: bool = Field(
        default=False,
        description="Use SLERP for flow matching interpolation instead of linear. "
        "Requires normalizing both x₀ and z. Experimental.",
    )
    magnitude_reg_weight: float = Field(
        default=0.01, ge=0.0,
        description="Weight for magnitude regularization. "
        "Keeps ||μ|| close to expected noise magnitude to prevent scale collapse.",
    )
    target_magnitude: float = Field(
        default=1.0, gt=0.0,
        description="Target magnitude for noise vectors. "
        "sqrt(D) ≈ 11.3 for D=128, but 1.0 works well with flow matching normalization.",
    )
    kappa_entropy_weight: float = Field(
        default=0.01, ge=0.0,
        description="Weight for kappa entropy regularization. "
        "Encourages diverse per-token kappa values (prevents uniform concentration).",
    )
    kappa_target: float = Field(
        default=2.0, gt=0.0,
        description="Target mean kappa. Pulls kappa above 1.0 so KL becomes active. "
        "κ > 1 = more concentrated than uniform → positive KL → gradient signal.",
    )
    kappa_pull_weight: float = Field(
        default=0.05, ge=0.0,
        description="Weight for pulling mean kappa toward kappa_target. "
        "Without this, kappa collapses to floor since KL=0 for κ<1.",
    )
    sigma_mean_target: float = Field(
        default=0.3, ge=0.05, le=0.9,
        description="Target mean sigma. Prevents sigma collapse to σ_min. "
        "Without this, sigma head learns σ→0 because less noise = lower MSE.",
    )
    sigma_mean_pull_weight: float = Field(
        default=0.1, ge=0.0,
        description="Weight for pulling mean sigma toward sigma_mean_target. "
        "Must be strong enough to overcome the MSE incentive to minimize sigma.",
    )


class VFMv1fTrainingStrategy(VFMv1dTrainingStrategy):
    """VFM v1f — Spherical Cauchy noise for VFM adapter.

    Extends v1d with spherical noise distribution. No architecture changes
    to the adapter — just reinterprets its (μ, log_σ) outputs.
    """

    config: VFMv1fTrainingConfig

    def _sample_spherical_noise(
        self,
        mu: Tensor,
        log_sigma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample noise from Spherical Cauchy distribution.

        Reinterprets adapter outputs (mu, log_sigma) as:
        - mu_hat: direction on S^127 (normalized mu)
        - mu_norm: magnitude (||mu||)
        - kappa: concentration (clamped exp of mean log_sigma per token)
        - z: noise = mu_norm * sample_from_spherical_cauchy(mu_hat, kappa)

        Args:
            mu: Adapter mean output [B, seq, 128]
            log_sigma: Adapter log-std output [B, seq, 128]

        Returns:
            (z, mu_hat, kappa, mu_norm) where:
                z: [B, seq, 128] sampled noise in R^128
                mu_hat: [B, seq, 128] unit direction vectors
                kappa: [B, seq] concentration per token
                mu_norm: [B, seq] magnitude per token
        """
        cfg = self.config
        B, seq, D = mu.shape

        # Direction: normalize mu to unit sphere
        mu_hat = normalize(mu, dim=-1)  # [B, seq, D]

        # Magnitude: ||mu|| carries scale information
        mu_norm = mu.norm(p=2, dim=-1)  # [B, seq]

        # Concentration: scalar kappa per token from log_sigma
        # Mean across latent dims → single scalar per token
        kappa = torch.exp(log_sigma.mean(dim=-1))  # [B, seq]
        kappa = kappa.clamp(min=cfg.kappa_min, max=cfg.kappa_max)

        # Sample direction from Spherical Cauchy on S^(D-1)
        # Reshape to [B*seq, D] for sampling, then back
        mu_hat_flat = mu_hat.reshape(B * seq, D)
        kappa_flat = kappa.reshape(B * seq)

        z_dir_flat = sample_spherical_cauchy(mu_hat_flat, kappa_flat)
        z_dir = z_dir_flat.reshape(B, seq, D)

        # Scale by learned magnitude
        z = mu_norm.unsqueeze(-1) * z_dir  # [B, seq, D]

        return z, mu_hat, kappa, mu_norm

    def _compute_spherical_kl(
        self,
        mu_hat: Tensor,
        kappa: Tensor,
    ) -> Tensor:
        """Compute KL divergence for spherical Cauchy.

        KL(SpCauchy(mu_hat, kappa) || Uniform(S^127))
        = (D-1)/2 * log(1 + 1/κ)

        Args:
            mu_hat: [B, seq, D] unit direction vectors
            kappa: [B, seq] concentration per token

        Returns:
            Scalar KL loss (mean over batch and tokens)
        """
        D = mu_hat.shape[-1]
        B, seq = kappa.shape

        kl_per_token = kl_spherical_cauchy_to_uniform(
            mu=mu_hat.reshape(B * seq, -1),
            kappa=kappa.reshape(B * seq),
            dim=D,
        )
        kl_per_token = kl_per_token.reshape(B, seq)

        return kl_per_token

    def _prepare_standard_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Standard VFM with spherical Cauchy noise + per-token sigma."""
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
        # NOISE ADAPTER (Spherical Cauchy)
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
                # Spherical Cauchy sampling
                video_noise, mu_hat, kappa, mu_norm = self._sample_spherical_noise(mu, log_sigma)
            else:
                # Fallback to Gaussian (v1d behavior)
                sigma_adapter = torch.exp(log_sigma)
                eps = torch.randn_like(mu)
                video_noise = mu + sigma_adapter * eps
                mu_hat = None
                kappa = None
                mu_norm = None

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
            mu_hat = None
            kappa = None
            mu_norm = None

        # ════════════════════════════════════════════════
        # PER-TOKEN SIGMA (fixed: no interpolation — matches inference)
        # ════════════════════════════════════════════════
        # The old approach interpolated: x_t = (1-σ)·x₀ + σ·z, then used σ as
        # timestep. This caused σ→0 collapse (lower σ = easier MSE) and a 20x
        # train/inference mismatch (training saw σ≈0.05, inference used t=1.0).
        #
        # Fixed: use pure adapter noise z as input (same as inference), and pass
        # per-token σ only as conditioning to AdaLN. No interpolation = no MSE
        # incentive to collapse σ. Both train and inference see identical inputs.
        if cfg.per_token_sigma and self._sigma_head is not None and adapter_mu is not None:
            per_token_sigmas = self._sigma_head(adapter_mu.float(), x0=video_latents.float())  # [B, seq]
            per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()

            # FIXED: input is always pure adapter noise z — matches inference
            noisy_video = video_noise

            video_targets = video_noise - video_latents
            video_timesteps = per_token_sigmas

            sigmas_mean = per_token_sigmas[~video_conditioning_mask].mean().detach()
            sigmas_for_logging = sigmas_mean.unsqueeze(0).expand(batch_size)
            # Per-batch sigma for Modality (used by prompt AdaLN in 2.3)
            batch_sigmas = sigmas_mean.unsqueeze(0).expand(batch_size)
        else:
            sigmas = timestep_sampler.sample_for(video_latents)
            sigmas_expanded = sigmas.view(-1, 1, 1)

            if cfg.use_slerp_interp:
                # Geodesic interpolation: SLERP on direction, lerp on magnitude
                # Produces inputs on the hypersphere manifold instead of cutting
                # through it linearly. Target stays v = z - x₀ for simple reconstruction.
                x0_dir = normalize(video_latents)
                z_dir = normalize(video_noise)
                x0_mag = video_latents.norm(dim=-1, keepdim=True)
                z_mag = video_noise.norm(dim=-1, keepdim=True)
                interp_dir = slerp(x0_dir, z_dir, sigmas_expanded)
                interp_mag = (1 - sigmas_expanded) * x0_mag + sigmas_expanded * z_mag
                noisy_video = interp_dir * interp_mag
            else:
                noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            video_targets = video_noise - video_latents
            video_timesteps = self._create_per_token_timesteps(
                video_conditioning_mask, sigmas.squeeze()
            )
            per_token_sigmas = None
            sigmas_for_logging = sigmas.squeeze()
            batch_sigmas = sigmas.squeeze()

        # Ensure conditioning tokens are clean
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # ════════════════════════════════════════════════
        # BUILD MODALITY
        # ════════════════════════════════════════════════
        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            sigma=batch_sigmas,
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
        model_inputs._sigma_complexity_targets = self._compute_complexity_targets(
            video_latents, cfg.sigma_min, cfg.sigma_max,
        ) if per_token_sigmas is not None else None
        model_inputs._distill_mode = "none"

        # v1f spherical metadata
        model_inputs._spherical_mu_hat = mu_hat
        model_inputs._spherical_kappa = kappa
        model_inputs._spherical_mu_norm = mu_norm

        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas_for_logging.view(-1, 1)

        return model_inputs

    def _prepare_distill_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Trajectory distillation with spherical Cauchy noise."""
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
        teacher_sigmas = traj["sigmas"].to(device)
        teacher_states = traj["states"].to(device, dtype)
        teacher_velocities = traj["velocities"].to(device, dtype)
        teacher_x0_preds = traj["x0_preds"].to(device, dtype)
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
        # NOISE: spherical adapter or teacher's starting z
        # ════════════════════════════════════════════════
        use_adapter = random.random() < cfg.alpha

        if use_adapter and self._noise_adapter is not None and not cfg.use_teacher_noise:
            mu, log_sigma = self._noise_adapter(
                text_embeddings=video_prompt_embeds.detach(),
                text_mask=prompt_attention_mask.bool(),
                positions=video_positions,
                task_class=torch.zeros(batch_size, dtype=torch.long, device=device),
            )

            if cfg.spherical_noise:
                video_noise, mu_hat, kappa, mu_norm = self._sample_spherical_noise(mu, log_sigma)
            else:
                sigma_adapter = torch.exp(log_sigma)
                eps = torch.randn_like(mu)
                video_noise = mu + sigma_adapter * eps
                mu_hat = None
                kappa = None
                mu_norm = None

            adapter_mu = mu
            adapter_log_sigma = log_sigma
        else:
            video_noise = teacher_states[:, 0]
            adapter_mu = None
            adapter_log_sigma = None
            use_adapter = False
            mu_hat = None
            kappa = None
            mu_norm = None

        # ════════════════════════════════════════════════
        # DISTILLATION TARGET
        # ════════════════════════════════════════════════
        if cfg.distill_mode == "output_match":
            teacher_target = teacher_x0_preds[:, -1]
            video_targets = video_noise - teacher_target

            if cfg.per_token_sigma and self._sigma_head is not None and adapter_mu is not None:
                per_token_sigmas = self._sigma_head(adapter_mu.float(), x0=video_latents.float())
                per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()
                video_timesteps = per_token_sigmas
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
            raise ValueError(f"Unsupported distill_mode for v1f: {cfg.distill_mode}")

        # Ensure conditioning tokens are clean
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # ════════════════════════════════════════════════
        # BUILD MODALITY
        # ════════════════════════════════════════════════
        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            sigma=torch.ones(batch_size, device=device, dtype=dtype),
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
        model_inputs._sigma_complexity_targets = self._compute_complexity_targets(
            video_latents, cfg.sigma_min, cfg.sigma_max,
        ) if per_token_sigmas is not None else None
        model_inputs._distill_mode = cfg.distill_mode
        model_inputs._distill_teacher_target = teacher_target if cfg.distill_mode == "output_match" else None
        model_inputs._distill_teacher_x0_gt = teacher_x0_gt
        model_inputs._distill_teacher_states = teacher_states
        model_inputs._distill_teacher_sigmas = teacher_sigmas
        model_inputs._distill_teacher_velocities = teacher_velocities

        # v1f spherical metadata
        model_inputs._spherical_mu_hat = mu_hat
        model_inputs._spherical_kappa = kappa
        model_inputs._spherical_mu_norm = mu_norm

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
        """Compute v1f loss = v1d loss + spherical KL + magnitude regularization.

        Loss components (inherited from v1d):
        1. L_mf: flow matching MSE
        2. L_kl: KL regularization (replaced with spherical KL)
        3. L_div: diversity regularization
        4. L_sigma_entropy: sigma distribution entropy
        5. L_distill: trajectory matching (when enabled)

        New in v1f:
        6. L_mag: magnitude regularization (keeps ||μ|| near target)
        7. Spherical KL replaces Gaussian KL when adapter is active
        """
        cfg = self.config
        distill_mode = getattr(inputs, "_distill_mode", "none")

        if distill_mode != "none" and distill_mode is not None:
            return self._compute_distill_loss_v1f(video_pred, audio_pred, inputs)

        # Standard path: use parent's compute_loss but override KL
        # We need to intercept and replace the Gaussian KL with spherical KL
        total_loss = self._compute_standard_loss_v1f(video_pred, audio_pred, inputs)

        return total_loss

    def _compute_standard_loss_v1f(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Standard VFM loss with spherical KL instead of Gaussian KL."""
        cfg = self.config

        # Flow matching MSE
        video_loss = (video_pred - inputs.video_targets).pow(2)
        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_mf = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_mf = video_loss.mean()

        total_loss = loss_mf

        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        mu_hat = getattr(inputs, "_spherical_mu_hat", None)
        kappa = getattr(inputs, "_spherical_kappa", None)
        mu_norm = getattr(inputs, "_spherical_mu_norm", None)

        loss_kl = torch.tensor(0.0, device=video_pred.device)

        if use_adapter and adapter_mu is not None and cfg.kl_weight > 0:
            if cfg.spherical_noise and mu_hat is not None and kappa is not None:
                # Spherical Cauchy KL
                kl_per_token = self._compute_spherical_kl(mu_hat, kappa)  # [B, seq]
                kl_per_sample = kl_per_token.mean(dim=1)  # [B]

                # Clamp negative KL (κ < 1 gives more entropy than uniform)
                kl_per_sample = torch.clamp(kl_per_sample, min=0.0)

                if cfg.kl_free_bits > 0:
                    kl_per_sample = torch.clamp(kl_per_sample - cfg.kl_free_bits, min=0.0)
                loss_kl = kl_per_sample.mean()
            else:
                # Gaussian KL fallback (v1d behavior)
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

        # Observation loss (inherited from v1a)
        loss_obs = torch.tensor(0.0, device=video_pred.device)
        if (
            use_adapter
            and cfg.obs_loss_weight > 0
            and getattr(inputs, "_vfm_observation", None) is not None
        ):
            obs = inputs._vfm_observation
            # Apply forward operator to prediction: x̂₀ = z - v
            video_noise = inputs._vfm_video_noise
            pred_x0 = video_noise - video_pred
            noise_level = inputs._vfm_noise_level

            if isinstance(noise_level, (int, float)) and noise_level > 0:
                obs_noise = torch.randn_like(pred_x0) * noise_level
                pred_obs = pred_x0 + obs_noise
            else:
                pred_obs = pred_x0

            obs_diff = (pred_obs - obs).pow(2)
            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                loss_obs = (obs_diff * mask).sum() / mask.sum().clamp(min=1) / obs_diff.shape[-1]
            else:
                loss_obs = obs_diff.mean()

            total_loss = total_loss + cfg.obs_loss_weight * loss_obs

        # Diversity loss (inherited from v1c)
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

            # Per-token complexity-aware pull (replaces global mean pull)
            if cfg.sigma_mean_pull_weight > 0:
                complexity_targets = getattr(inputs, "_sigma_complexity_targets", None)
                if complexity_targets is not None:
                    # Per-token MSE toward complexity-derived targets
                    loss_mask = getattr(inputs, "video_loss_mask", None)
                    if loss_mask is not None:
                        loss_sigma_pull = (per_token_sigmas[loss_mask] - complexity_targets[loss_mask]).pow(2).mean()
                    else:
                        loss_sigma_pull = (per_token_sigmas - complexity_targets).pow(2).mean()
                else:
                    # Fallback to global mean pull if no complexity targets
                    loss_mask = getattr(inputs, "video_loss_mask", None)
                    active_sigmas = per_token_sigmas[loss_mask] if loss_mask is not None else per_token_sigmas.flatten()
                    if active_sigmas.numel() > 0:
                        loss_sigma_pull = (active_sigmas.mean() - cfg.sigma_mean_target).pow(2)
                total_loss = total_loss + cfg.sigma_mean_pull_weight * loss_sigma_pull

        # ════════════════════════════════════════════════
        # MAGNITUDE REGULARIZATION (v1f new)
        # ════════════════════════════════════════════════
        loss_mag = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and mu_norm is not None and cfg.magnitude_reg_weight > 0:
            loss_mag = (mu_norm - cfg.target_magnitude).pow(2).mean()
            total_loss = total_loss + cfg.magnitude_reg_weight * loss_mag

        # ════════════════════════════════════════════════
        # KAPPA REGULARIZATION (v1f — prevent κ collapse)
        # ════════════════════════════════════════════════
        loss_kappa_pull = torch.tensor(0.0, device=video_pred.device)
        loss_kappa_entropy = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and kappa is not None:
            # Pull mean kappa toward target (above 1.0 so KL activates)
            if cfg.kappa_pull_weight > 0:
                loss_kappa_pull = (kappa.mean() - cfg.kappa_target).pow(2)
                total_loss = total_loss + cfg.kappa_pull_weight * loss_kappa_pull

            # Encourage diverse kappa across tokens (negative std)
            if cfg.kappa_entropy_weight > 0 and kappa.numel() > 1:
                loss_kappa_entropy = -kappa.std()
                total_loss = total_loss + cfg.kappa_entropy_weight * loss_kappa_entropy

        # ════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════
        self._last_vfm_metrics = {
            "vfm/loss_mf": loss_mf.item(),
            "vfm/loss_kl": loss_kl.item(),
            "vfm/loss_obs": loss_obs.item(),
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
                "vfm/mu_norm_mean": mu_norm.mean().item(),
                "vfm/mu_norm_std": mu_norm.std().item(),
                "vfm/loss_mag": loss_mag.item(),
                "vfm/loss_kappa_pull": loss_kappa_pull.item(),
                "vfm/loss_kappa_entropy": loss_kappa_entropy.item(),
            })

            # Geodesic diversity: mean pairwise geodesic distance between adapter directions
            if mu_hat.shape[0] > 0:
                # Sample a few tokens for efficiency
                n_sample = min(64, mu_hat.shape[1])
                idx = torch.randperm(mu_hat.shape[1])[:n_sample]
                mu_sample = mu_hat[0, idx]  # [n_sample, D]
                if n_sample > 1:
                    geo_dist = geodesic_distance(
                        mu_sample[:-1], mu_sample[1:]
                    ).mean()
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
        """Distillation loss with spherical KL instead of Gaussian KL."""
        cfg = self.config

        # L_distill: MSE between student velocity and teacher target
        video_loss = (video_pred - inputs.video_targets).pow(2)

        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            loss_distill = (video_loss * mask).sum() / mask.sum().clamp(min=1) / video_loss.shape[-1]
        else:
            loss_distill = video_loss.mean()

        total_loss = cfg.distill_weight * loss_distill

        # L_kl: spherical or Gaussian
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

        # L_div: diversity from v1c
        if use_adapter and adapter_mu is not None:
            div_loss = self._compute_diversity_loss(adapter_mu, inputs)
            total_loss = total_loss + div_loss

        # L_sigma_entropy + complexity-aware pull
        per_token_sigmas = getattr(inputs, "_per_token_sigmas", None)
        loss_sigma_entropy = torch.tensor(0.0, device=video_pred.device)
        loss_sigma_pull = torch.tensor(0.0, device=video_pred.device)
        if per_token_sigmas is not None:
            if cfg.sigma_entropy_weight > 0:
                loss_sigma_entropy = self._compute_sigma_entropy_loss(per_token_sigmas, inputs)
                total_loss = total_loss + cfg.sigma_entropy_weight * loss_sigma_entropy

            # Per-token complexity-aware pull (replaces global mean pull)
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

        # L_mag: magnitude regularization (v1f)
        loss_mag = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and mu_norm is not None and cfg.magnitude_reg_weight > 0:
            loss_mag = (mu_norm - cfg.target_magnitude).pow(2).mean()
            total_loss = total_loss + cfg.magnitude_reg_weight * loss_mag

        # L_kappa: kappa regularization (v1f)
        loss_kappa_pull = torch.tensor(0.0, device=video_pred.device)
        loss_kappa_entropy = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and kappa is not None:
            if cfg.kappa_pull_weight > 0:
                loss_kappa_pull = (kappa.mean() - cfg.kappa_target).pow(2)
                total_loss = total_loss + cfg.kappa_pull_weight * loss_kappa_pull
            if cfg.kappa_entropy_weight > 0 and kappa.numel() > 1:
                loss_kappa_entropy = -kappa.std()
                total_loss = total_loss + cfg.kappa_entropy_weight * loss_kappa_entropy

        # Logging
        self._last_vfm_metrics = {
            "vfm/loss_distill": loss_distill.item(),
            "vfm/loss_kl": loss_kl.item(),
            "vfm/loss_gt": loss_gt.item(),
            "vfm/loss_mag": loss_mag.item(),
            "vfm/loss_kappa_pull": loss_kappa_pull.item(),
            "vfm/loss_kappa_entropy": loss_kappa_entropy.item(),
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
                "vfm/mu_norm_mean": mu_norm.mean().item(),
            })
        if per_token_sigmas is not None:
            active = per_token_sigmas[per_token_sigmas > 0]
            if active.numel() > 0:
                self._last_vfm_metrics["vfm/sigma_mean"] = active.mean().item()
                self._last_vfm_metrics["vfm/sigma_std"] = active.std().item()
                self._last_vfm_metrics["vfm/sigma_entropy"] = loss_sigma_entropy.item()
                self._last_vfm_metrics["vfm/loss_sigma_pull"] = loss_sigma_pull.item()

        return total_loss

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction video + trajectory PCA + spherical diagnostics.

        Logs:
        - train/reconstruction_video: GT | Pred side-by-side video (from parent)
        - train/trajectory_pca: PCA of adapter μ → noise z → predicted x̂₀ → GT x₀
        - train/spherical_heatmap: κ and ||μ|| per-token heatmaps
        """
        # Parent logs reconstruction_video + reconstruction image
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        # Remove the static image — we only need the video
        log_dict.pop(f"{prefix}/reconstruction", None)

        # Add trajectory PCA (adapter μ → z → x̂₀ vs GT)
        try:
            traj_plots = self._build_adapter_trajectory_pca(
                video_pred, inputs, step, prefix,
            )
            log_dict.update(traj_plots)
        except Exception as e:
            logger.warning(f"Failed to build trajectory PCA: {e}")

        # Add spherical-specific plots
        kappa = getattr(inputs, "_spherical_kappa", None)
        mu_norm = getattr(inputs, "_spherical_mu_norm", None)
        if kappa is not None and mu_norm is not None:
            try:
                spherical_plots = self._build_spherical_plots(
                    kappa, mu_norm, inputs, step, prefix,
                )
                log_dict.update(spherical_plots)
            except Exception as e:
                logger.warning(f"Failed to build spherical plots: {e}")

        return log_dict

    @staticmethod
    def _build_adapter_trajectory_pca(
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """PCA plot showing adapter's noise generation path in latent space.

        Plots 4 key points:
        - adapter μ: the adapter's mean output (what it "wants" to generate)
        - noise z: the actual sampled noise (μ + stochastic perturbation)
        - predicted x̂₀: what the transformer reconstructs from z (z - v_pred)
        - GT x₀: the ground truth clean video

        When training works, the trajectory should show:
        z → x̂₀ converging toward GT, and μ stabilizing.
        """
        try:
            import wandb
            import plotly.graph_objects as go

            adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
            use_adapter = getattr(inputs, "_vfm_use_adapter", False)
            raw_latents = getattr(inputs, "_raw_video_latents", None)

            if not use_adapter or adapter_mu is None or raw_latents is None:
                return {}

            noise = inputs.shared_noise[0].float().cpu()       # [seq, C]
            mu = adapter_mu[0].float().cpu()                    # [seq, C]
            pred_v = video_pred[0].float().cpu()                # [seq, C]
            pred_x0 = noise - pred_v                            # x̂₀ = z - v
            gt_x0 = raw_latents[0].float().cpu()                # [C, F, H, W]

            # Flatten GT to [seq, C]
            c, f, h, w = gt_x0.shape
            gt_x0_flat = gt_x0.permute(1, 2, 3, 0).reshape(-1, c)  # [F*H*W, C]

            # Average over tokens → [C] per point
            points = torch.stack([
                mu.mean(dim=0),
                noise.mean(dim=0),
                pred_x0.mean(dim=0),
                gt_x0_flat.mean(dim=0),
            ])  # [4, C]

            # PCA via SVD
            centered = points - points.mean(dim=0, keepdim=True)
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            pca = (centered @ Vh[:2].T).detach().cpu().numpy()  # [4, 2]
            var_explained = (S[:2] ** 2 / (S ** 2).sum() * 100).detach().cpu().numpy()

            # Distances
            d_z_to_gt = (noise.mean(0) - gt_x0_flat.mean(0)).pow(2).mean().sqrt().item()
            d_pred_to_gt = (pred_x0.mean(0) - gt_x0_flat.mean(0)).pow(2).mean().sqrt().item()
            d_mu_to_gt = (mu.mean(0) - gt_x0_flat.mean(0)).pow(2).mean().sqrt().item()

            fig = go.Figure()

            # Path: μ → z → x̂₀
            fig.add_trace(go.Scatter(
                x=pca[:3, 0], y=pca[:3, 1],
                mode="lines+markers+text",
                name="Adapter path",
                text=["μ (adapter mean)", "z (sampled noise)", "x̂₀ (predicted)"],
                textposition=["top center", "top center", "bottom center"],
                textfont=dict(size=10),
                line=dict(color="#FF5722", width=3),
                marker=dict(size=[12, 10, 14], symbol=["circle", "circle", "star"],
                            color=["#FF9800", "#FF5722", "#E91E63"]),
            ))

            # GT point
            fig.add_trace(go.Scatter(
                x=[pca[3, 0]], y=[pca[3, 1]],
                mode="markers+text",
                name="GT x₀",
                text=["GT x₀"],
                textposition="bottom center",
                marker=dict(size=16, symbol="diamond", color="#4CAF50"),
            ))

            # Dashed line: x̂₀ → GT (the gap we're trying to close)
            fig.add_trace(go.Scatter(
                x=[pca[2, 0], pca[3, 0]], y=[pca[2, 1], pca[3, 1]],
                mode="lines",
                name=f"Gap (L2={d_pred_to_gt:.3f})",
                line=dict(color="#4CAF50", width=2, dash="dot"),
                showlegend=True,
            ))

            fig.update_layout(
                title=(
                    f"Adapter Trajectory PCA — step {step}<br>"
                    f"<sub>d(z,GT)={d_z_to_gt:.3f}  d(x̂₀,GT)={d_pred_to_gt:.3f}  "
                    f"d(μ,GT)={d_mu_to_gt:.3f}</sub>"
                ),
                xaxis_title=f"PC1 ({var_explained[0]:.1f}%)",
                yaxis_title=f"PC2 ({var_explained[1]:.1f}%)",
                template="plotly_dark",
                legend=dict(x=0.02, y=0.98),
                height=500, width=700,
            )
            return {f"{prefix}/trajectory_pca": wandb.Plotly(fig)}
        except Exception as e:
            logger.debug(f"Adapter trajectory PCA failed: {e}")
            return {}

    @staticmethod
    def _build_spherical_plots(
        kappa: Tensor,
        mu_norm: Tensor,
        inputs: ModelInputs,
        step: int,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Build kappa and magnitude heatmaps for W&B."""
        try:
            import wandb
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            raw_latents = getattr(inputs, "_raw_video_latents", None)
            if raw_latents is None:
                return {}

            num_frames = raw_latents.shape[2]
            h = raw_latents.shape[3]
            w = raw_latents.shape[4]
            tpf = h * w

            kappa_cpu = kappa[0].float().cpu()
            norm_cpu = mu_norm[0].float().cpu()

            kappa_frames = kappa_cpu[:num_frames * tpf].reshape(num_frames, h, w)
            norm_frames = norm_cpu[:num_frames * tpf].reshape(num_frames, h, w)

            fig = make_subplots(
                rows=2, cols=num_frames,
                subplot_titles=(
                    [f"κ F{i}" for i in range(num_frames)]
                    + [f"||μ|| F{i}" for i in range(num_frames)]
                ),
                vertical_spacing=0.15,
            )

            for f_idx in range(num_frames):
                fig.add_trace(
                    go.Heatmap(
                        z=kappa_frames[f_idx].detach().cpu().numpy(),
                        colorscale="Viridis",
                        showscale=(f_idx == num_frames - 1),
                        colorbar=dict(title="κ", y=0.75),
                    ),
                    row=1, col=f_idx + 1,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=norm_frames[f_idx].detach().cpu().numpy(),
                        colorscale="Plasma",
                        showscale=(f_idx == num_frames - 1),
                        colorbar=dict(title="||μ||", y=0.25),
                    ),
                    row=2, col=f_idx + 1,
                )

            fig.update_layout(
                title=f"Spherical Cauchy: κ (concentration) & ||μ|| (magnitude) — step {step}",
                template="plotly_dark",
                height=450, width=200 * num_frames,
            )
            return {f"{prefix}/spherical_heatmap": wandb.Plotly(fig)}
        except Exception:
            return {}
