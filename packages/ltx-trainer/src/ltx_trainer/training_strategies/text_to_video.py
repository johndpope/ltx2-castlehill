"""Text-to-video training strategy.
This strategy implements standard text-to-video generation training where:
- Only target latents are used (no reference videos)
- Standard noise application and loss computation
- Supports first frame conditioning
- Optionally supports joint audio-video training
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class TextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-video training strategy."""

    name: Literal["text_to_video"] = "text_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    log_reconstructions: bool = Field(
        default=False,
        description="Log target/predict/source reconstruction images to W&B",
    )

    reconstruction_log_interval: int = Field(
        default=50,
        description="Log reconstruction images every N training steps",
    )


class TextToVideoStrategy(TrainingStrategy):
    """Text-to-video training strategy.
    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Supports first frame conditioning
    - Optionally supports joint audio-video training when with_audio=True
    """

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Text-to-video configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        Text-to-video training requires latents and text conditions.
        When with_audio is True, also requires audio latents.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        video_latents = latents["latents"]

        # Get video dimensions (assume same for all batch elements)
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS with backward compatibility
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Generate video positions using ltx_core's native implementation
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Video loss mask: True for tokens we want to compute loss on (non-conditioning tokens)
        video_loss_mask = ~video_conditioning_mask

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        inputs = ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

        # Store raw latents + noise for reconstruction logging
        inputs._raw_video_latents = batch["latents"]["latents"]  # [B, C, F, H, W]
        inputs._noisy_video = noisy_video  # [B, seq_len, C] patchified
        inputs._video_noise = video_noise
        inputs._video_sigmas = sigmas

        return inputs

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint audio-video training.
        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - dataset provides uniform non-patchified format [B, C, T, F]
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

        # Patchify audio latents: [B, C, T, F] -> [B, T, C*F]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio (same sigma as video)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Compute audio targets
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma (no conditioning mask)
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: all tokens contribute to loss (no conditioning)
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and optionally audio."""
        # Video loss
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # If no audio, return video loss only
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        # Audio loss (no conditioning mask)
        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()

        # Combined loss
        return video_loss + audio_loss

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log target/predict/source reconstruction images to W&B.

        Decodes one sample from the batch through the VAE and logs a triplet:
        source (noisy input) | predict (denoised) | target (ground truth).
        """
        try:
            import wandb
        except ImportError:
            return {}
        if wandb.run is None or not self.config.log_reconstructions:
            return {}

        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is None or vae_decoder is None:
            return {}

        import random as _random

        b, c, f, h, w = raw_latents.shape

        # Reconstruct clean prediction from velocity: pred_clean = noisy - sigma * velocity
        # In flow matching: noisy = (1-σ)x₀ + σε, target = ε - x₀, pred ≈ target
        # So x₀_hat = noisy - σ * pred = (1-σ)x₀ + σε - σ(ε - x₀) = x₀
        noisy_video = getattr(inputs, "_noisy_video", None)
        sigmas = getattr(inputs, "_video_sigmas", None)
        if noisy_video is None or sigmas is None:
            return {}

        sigmas_expanded = sigmas.view(-1, 1, 1)
        pred_clean = noisy_video - sigmas_expanded * video_pred  # [B, seq_len, C]

        # Unpatchify to [B, C, F, H, W]
        from ltx_core.types import VideoLatentShape  # noqa: PLC0415
        output_shape = VideoLatentShape(batch=b, channels=c, frames=f, height=h, width=w)
        pred_spatial = self._video_patchifier.unpatchify(pred_clean, output_shape)
        noisy_spatial = self._video_patchifier.unpatchify(noisy_video, output_shape)

        sample_idx = _random.randint(0, b - 1) if b > 1 else 0

        log_dict: dict[str, Any] = {}
        try:
            decoder_device = next(vae_decoder.parameters()).device
            decoder_dtype = next(vae_decoder.parameters()).dtype

            # Decode one at a time, moving results to CPU immediately to save VRAM
            def _decode_one(latent: Tensor) -> Tensor:
                with torch.inference_mode():
                    decoded = vae_decoder(latent.to(device=decoder_device, dtype=decoder_dtype))
                result = decoded.float().clamp(-1, 1) * 0.5 + 0.5
                return result[0].cpu()  # [3, T, H, W]

            # Free training activations before VAE decode
            torch.cuda.empty_cache()

            gt_frames = _decode_one(raw_latents[sample_idx:sample_idx + 1])
            torch.cuda.empty_cache()
            pred_frames = _decode_one(pred_spatial[sample_idx:sample_idx + 1])
            torch.cuda.empty_cache()
            noisy_frames = _decode_one(noisy_spatial[sample_idx:sample_idx + 1])
            torch.cuda.empty_cache()

            # Mid-frame triplet image: source | predict | target
            mid_f = gt_frames.shape[1] // 2
            import torchvision.utils as vutils

            grid = vutils.make_grid(
                [noisy_frames[:, mid_f], pred_frames[:, mid_f], gt_frames[:, mid_f]],
                nrow=3, padding=4, normalize=False,
            )
            log_dict[f"{prefix}/reconstruction"] = wandb.Image(
                grid.permute(1, 2, 0).numpy(),
                caption=f"Step {step} | Source (noisy) | Predict | Target (GT)",
            )

            # Also log as video if multi-frame
            if gt_frames.shape[1] > 1:
                side_by_side = torch.cat([noisy_frames, pred_frames, gt_frames], dim=-1)
                video_np = (side_by_side.permute(1, 0, 2, 3) * 255).clamp(0, 255).to(torch.uint8).numpy()
                log_dict[f"{prefix}/reconstruction_video"] = wandb.Video(
                    video_np, fps=8,
                    caption=f"Step {step} | Source | Predict | Target",
                )

            logger.debug(f"Logged reconstruction triplet at step {step}")
        except Exception as e:
            logger.warning(f"Failed to decode reconstructions: {e}")

        return log_dict
