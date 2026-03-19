"""VFM v1b — Temporally-aware Variational Flow Maps for LTX-2.

Upgrades over v1a (vfm_strategy.py):
  1. Self-attention in adapter → tokens coordinate their noise distributions
  2. Cross-attention to FULL text → per-token text grounding (no pooling)
  3. Positional encoding → each token knows its (t, h, w) location

Result: adapter outputs DIFFERENT μ,σ per token, enabling temporally and
spatially structured noise. The flow map (transformer) gets a head start
on coherent video because the noise itself is already structured.

Everything else (loss function, EMA, flow map freezing, logging) is
inherited from VFMTrainingStrategy unchanged.
"""

from typing import Any, Literal

import random
import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.noise_adapter_v1b import (
    TASK_CLASSES,
    NoiseAdapterV1b,
    create_noise_adapter_v1b,
)
from ltx_trainer import logger
from ltx_trainer.inverse_problems import InverseProblemSampler
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
)
from ltx_trainer.training_strategies.vfm_strategy import (
    VFMTrainingConfig,
    VFMTrainingStrategy,
)


class VFMv1bTrainingConfig(VFMTrainingConfig):
    """Configuration for VFM v1b (temporally-aware adapter)."""

    name: Literal["vfm_v1b"] = "vfm_v1b"

    # Override adapter defaults for v1b architecture
    adapter_variant: Literal["mlp", "transformer"] = Field(
        default="transformer",
        description="Ignored in v1b — always uses NoiseAdapterV1b (transformer with cross-attn)",
    )
    adapter_hidden_dim: int = Field(
        default=512,
        description="Hidden dim for v1b adapter transformer blocks",
    )
    adapter_num_heads: int = Field(
        default=8,
        description="Attention heads in v1b adapter blocks",
    )
    adapter_num_layers: int = Field(
        default=4,
        description="Number of (self-attn + cross-attn + FFN) blocks",
    )
    adapter_pos_dim: int = Field(
        default=256,
        description="Sinusoidal positional encoding dimension",
    )


class VFMv1bTrainingStrategy(VFMTrainingStrategy):
    """VFM v1b — temporally-aware noise adapter with cross-attention.

    Overrides only the adapter interface and prepare_training_inputs from v1a.
    Loss computation, EMA, flow map freezing, reconstruction logging are inherited.
    """

    config: VFMv1bTrainingConfig

    def __init__(self, config: VFMv1bTrainingConfig):
        super().__init__(config)
        self._noise_adapter: NoiseAdapterV1b | None = None

    def set_noise_adapter(self, adapter: NoiseAdapterV1b) -> None:
        self._noise_adapter = adapter

    def get_noise_adapter_params(self) -> dict[str, Any]:
        """Return v1b adapter constructor kwargs."""
        text_dim = getattr(self, "_text_embed_dim", 3840)
        return {
            "text_dim": text_dim,
            "latent_dim": 128,
            "hidden_dim": self.config.adapter_hidden_dim,
            "num_heads": self.config.adapter_num_heads,
            "num_layers": self.config.adapter_num_layers,
            "pos_dim": self.config.adapter_pos_dim,
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare VFM v1b training inputs — passes full text + positions to adapter."""
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents = self._video_patchifier.patchify(video_latents)  # [B, seq_len, C]

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]  # [B, text_seq, D]
        prompt_attention_mask = conditions["prompt_attention_mask"]  # [B, text_seq]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype
        tokens_per_frame = video_seq_len // num_frames

        # Initialize inverse problem sampler on first call
        if self._inverse_problem_sampler is None:
            self._inverse_problem_sampler = InverseProblemSampler(
                problems=self._ip_configs,
                tokens_per_frame=tokens_per_frame,
            )

        # First-frame conditioning mask
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Timesteps
        sigmas = timestep_sampler.sample_for(video_latents)

        # Positions: [B, 3, seq_len, 2]
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # ============================
        # VFM v1b: NOISE ADAPTER
        # ============================
        use_adapter_noise = random.random() < self.config.alpha

        if use_adapter_noise and self._noise_adapter is not None:
            # Sample inverse problem
            ip_sample = self._inverse_problem_sampler.sample(video_latents)

            # v1b: pass FULL text embeddings + positions to adapter
            # No pooling — adapter uses cross-attention to attend to text
            mu, log_sigma = self._noise_adapter(
                text_embeddings=video_prompt_embeds.detach(),
                text_mask=prompt_attention_mask.bool(),
                positions=video_positions,
                task_class=ip_sample.task_class,
            )

            # Reparameterization: z = μ + σ ⊙ ε
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

        # ============================
        # NOISY INTERPOLATION
        # ============================
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Velocity target: v = z - x_0
        video_targets = video_noise - video_latents

        # ============================
        # BUILD MODALITY
        # ============================
        from ltx_core.model.transformer.modality import Modality

        video_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

        video_modality = Modality(
            enabled=True,
            sigma=sigmas.squeeze(),
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

        # VFM-specific data for loss computation (inherited compute_loss uses these)
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

        # For reconstruction logging
        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas

        return model_inputs
