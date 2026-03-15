"""VFM v1h — Unified adapter with integrated per-token sigma.

Key change from v1f: eliminates the external SigmaHead MLP entirely.
The NoiseAdapterV1b now outputs (μ, log_σ, σ_timestep) in a single forward pass.

Why this is better:
1. The adapter already has text cross-attention + position encoding + self-attention.
   It knows WHAT each token represents (via text) and WHERE it is (via position).
   SigmaHead was trying to rediscover this from x₀ — redundant.

2. One forward pass instead of two (adapter + SigmaHead).

3. The per-token sigma gets gradients from BOTH the flow matching loss AND the KL,
   instead of only through flow matching via a weak path.

4. σ_timestep and noise (μ̂, κ) are jointly optimized in the same model,
   so the adapter can learn to coordinate noise structure with noise level.

Architecture:
    Text embeddings → NoiseAdapterV1b → (μ, log_σ, σ_timestep) per token
    μ̂ = normalize(μ), r = ||μ||, κ = exp(mean(log_σ))
    z_dir ~ SphericalCauchy(μ̂, κ), z = r · z_dir
    x_t[i] = (1 - σ_timestep[i]) · x₀[i] + σ_timestep[i] · z[i]
    48-layer DiT → velocity v → x̂₀
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.spherical_utils import (
    geodesic_distance,
    kl_spherical_cauchy_to_uniform,
    normalize,
    sample_spherical_cauchy,
)
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)
from ltx_trainer.timestep_samplers import TimestepSampler


class VFMv1hTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v1h (unified adapter + per-token sigma)."""

    name: Literal["vfm_v1h"] = "vfm_v1h"

    # Per-token sigma is always on in v1h (built into adapter)
    # These params still control the range and loss weights
    per_token_sigma: bool = Field(
        default=True,
        description="Always True in v1h — sigma is built into the adapter.",
    )


class VFMv1hTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v1h — Unified noise adapter with integrated per-token sigma.

    Eliminates SigmaHead. The adapter's transformer (with text cross-attention,
    position encoding, and self-attention) directly outputs σ_timestep per token.
    """

    config: VFMv1hTrainingConfig

    def initialize(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> None:
        """Initialize strategy — enable sigma head on the adapter, skip SigmaHead."""
        # Call grandparent (v1d) init but skip SigmaHead creation
        # We need to call the base chain but NOT create self._sigma_head

        # Call v1c's initialize (which calls v1b → v1a → base)
        # v1d's initialize creates SigmaHead — we override to skip that
        from ltx_trainer.training_strategies.vfm_strategy_v1c import VFMv1cTrainingStrategy
        VFMv1cTrainingStrategy.initialize(self, model, device)

        # Don't create SigmaHead — adapter handles it
        self._sigma_head = None

        logger.info(
            "v1h: per-token sigma integrated into adapter (no SigmaHead)"
        )

    def _enable_adapter_sigma(self) -> None:
        """Enable the sigma timestep head on the noise adapter.

        Called after the adapter is created and set on the strategy.
        """
        if self._noise_adapter is not None and hasattr(self._noise_adapter, 'enable_sigma_timestep_head'):
            cfg = self.config
            self._noise_adapter.enable_sigma_timestep_head(
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
            )
            # Count new params
            sigma_params = sum(
                p.numel() for p in self._noise_adapter.sigma_timestep_head.parameters()
            )
            logger.info(
                f"v1h: enabled sigma_timestep_head on adapter ({sigma_params} params, "
                f"σ ∈ [{cfg.sigma_min}, {cfg.sigma_max}])"
            )

    def set_noise_adapter(self, adapter: Any) -> None:
        """Override to enable sigma head after adapter is set."""
        super().set_noise_adapter(adapter)
        self._enable_adapter_sigma()

    def get_trainable_parameters(self) -> list:
        """Return trainable params — adapter includes sigma head, no SigmaHead."""
        params = []
        if self._noise_adapter is not None:
            params.extend([p for p in self._noise_adapter.parameters() if p.requires_grad])
        return params

    def get_strategy_params(self) -> dict[str, Any]:
        """Return strategy params for checkpoint — no SigmaHead to save."""
        return {}

    def load_strategy_params(self, checkpoint: dict[str, Any]) -> None:
        """Load strategy params — no SigmaHead to load."""
        pass

    # ════════════════════════════════════════════════════════════
    # PREPARE TRAINING INPUTS — standard path (no distill)
    # ════════════════════════════════════════════════════════════

    def _prepare_standard_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs with adapter-integrated per-token sigma."""
        cfg = self.config
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        # Patchify: [B, C, F, H, W] → [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

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
        # NOISE ADAPTER — outputs (μ, log_σ, σ_timestep) in one pass
        # ════════════════════════════════════════════════
        use_adapter_noise = random.random() < cfg.alpha

        if use_adapter_noise and self._noise_adapter is not None:
            ip_sample = self._inverse_problem_sampler.sample(video_latents)

            result = self._noise_adapter(
                text_embeddings=video_prompt_embeds.detach(),
                text_mask=prompt_attention_mask.bool(),
                positions=video_positions,
                task_class=ip_sample.task_class,
            )

            # v1h: adapter returns 3 outputs (mu, log_sigma, per_token_sigma)
            if len(result) == 3:
                mu, log_sigma, per_token_sigmas = result
            else:
                mu, log_sigma = result
                per_token_sigmas = None

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
            per_token_sigmas = None

        # ════════════════════════════════════════════════
        # PER-TOKEN SIGMA INTERPOLATION (from adapter output)
        # ════════════════════════════════════════════════
        if per_token_sigmas is not None:
            # Zero out sigma for conditioning tokens (first frame)
            per_token_sigmas = per_token_sigmas * (~video_conditioning_mask).float()

            # Interpolate with per-token sigma: x_t[i] = (1-σ_i)·x₀[i] + σ_i·z[i]
            # CRITICAL: detach sigma from interpolation gradient — otherwise σ→σ_min
            # because lower σ = less noise = easier MSE. Sigma gets its gradient only
            # from the complexity-target pull loss and entropy loss.
            sigmas_for_interp = per_token_sigmas.detach().unsqueeze(-1)  # [B, seq, 1]
            noisy_video = (1 - sigmas_for_interp) * video_latents + sigmas_for_interp * video_noise

            video_targets = video_noise - video_latents
            video_timesteps = per_token_sigmas

            sigmas_mean = per_token_sigmas[~video_conditioning_mask].mean().detach()
            sigmas_for_logging = sigmas_mean.unsqueeze(0).expand(batch_size)
            batch_sigmas = sigmas_mean.unsqueeze(0).expand(batch_size)
        else:
            # Standard uniform sigma (non-adapter path)
            sigmas = timestep_sampler.sample_for(video_latents)
            sigmas_expanded = sigmas.view(-1, 1, 1)
            noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

            video_targets = video_noise - video_latents
            video_timesteps = self._create_per_token_timesteps(
                video_conditioning_mask, sigmas.squeeze()
            )
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

        # v1d/v1h metadata
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
