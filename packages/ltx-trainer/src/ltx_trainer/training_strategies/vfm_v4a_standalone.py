"""VFM v4a Standalone — Gaussian adapter, NO inheritance from v1f/v3b chain.

Architecture:
    text_embeddings → NoiseAdapterV1b → (mu, log_sigma)
    z = mu + exp(log_sigma) * eps        [Gaussian reparameterization]
    sigma = SigmaHead(mu, x0)            [per-token, [0.05, 0.95]]
    DiT(latent=z, timesteps=sigma) → v   [LoRA r=32]
    x_hat = z - v
    loss = MSE(v, z - x0) + kl_weight * GaussianKL(mu, log_sigma)
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.timestep_samplers import TimestepSampler


# ──────────────────────────────────────────────────────────────────
# SigmaHead (copied verbatim from vfm_strategy_v1d.py — no import)
# ──────────────────────────────────────────────────────────────────

class SigmaHead(nn.Module):
    """Per-token sigma predictor: [B, seq, 2*D] → [B, seq] ∈ [sigma_min, sigma_max]."""

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
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu: Tensor, x0: Tensor | None = None) -> Tensor:
        inp = torch.cat([x0.detach() if x0 is not None else torch.zeros_like(mu), mu], dim=-1)
        raw = self.net(inp).squeeze(-1)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

class VFMv4aConfig(TrainingStrategyConfigBase):
    """Standalone VFM v4a config."""

    model_config = ConfigDict(extra="forbid")
    name: Literal["vfm_v4a"] = "vfm_v4a"

    # Adapter
    adapter_variant: Literal["v1b"] = Field(default="v1b")
    adapter_hidden_dim: int = Field(default=512)
    adapter_num_layers: int = Field(default=4)
    adapter_num_heads: int = Field(default=8)
    adapter_pos_dim: int = Field(default=256)

    # SigmaHead
    per_token_sigma: bool = Field(default=True)
    sigma_head_hidden_dim: int = Field(default=256)
    sigma_min: float = Field(default=0.05)
    sigma_max: float = Field(default=0.95)

    # Loss
    kl_weight: float = Field(default=0.001)
    kl_free_bits: float = Field(default=0.0)

    # Task sampling
    alpha: float = Field(default=1.0, description="Fraction of steps using adapter noise")
    first_frame_conditioning_p: float = Field(default=0.1)

    # Audio
    with_audio: bool = Field(default=False)
    audio_latents_dir: str = Field(default="audio_latents")


# ──────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────

class VFMv4aStrategy(TrainingStrategy):
    """Standalone VFM v4a training strategy.

    No inheritance from v1f/v3b. All logic is inline.
    """

    config: VFMv4aConfig

    def __init__(self, config: VFMv4aConfig):
        super().__init__(config)
        self._noise_adapter: Any = None
        self._sigma_head: SigmaHead | None = None
        self._current_step: int = 0

        if config.per_token_sigma:
            self._sigma_head = SigmaHead(
                latent_dim=128,
                hidden_dim=config.sigma_head_hidden_dim,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
            )

    # ── API for trainer ──────────────────────────────────────────

    def set_noise_adapter(self, adapter: Any) -> None:
        self._noise_adapter = adapter
        if adapter is not None and self._sigma_head is not None:
            device = next(adapter.parameters()).device
            self._sigma_head = self._sigma_head.to(device=device)

    def get_noise_adapter_params(self) -> dict[str, Any]:
        return {
            "text_dim": getattr(self, "_text_embed_dim", 4096),
            "latent_dim": 128,
            "hidden_dim": self.config.adapter_hidden_dim,
            "num_heads": self.config.adapter_num_heads,
            "num_layers": self.config.adapter_num_layers,
            "pos_dim": self.config.adapter_pos_dim,
        }

    def set_text_embed_dim(self, dim: int) -> None:
        self._text_embed_dim = dim

    def set_current_step(self, step: int) -> None:
        self._current_step = step

    def get_sigma_head(self) -> SigmaHead | None:
        return self._sigma_head

    def get_sigma_head_params(self) -> dict[str, Any]:
        return {
            "latent_dim": 128,
            "hidden_dim": self.config.sigma_head_hidden_dim,
            "sigma_min": self.config.sigma_min,
            "sigma_max": self.config.sigma_max,
        }

    def set_sigma_head(self, sigma_head: SigmaHead) -> None:
        self._sigma_head = sigma_head

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> dict[str, str]:
        sources = {"latents": "latents", "conditions": "conditions"}
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    # ── Helpers ─────────────────────────────────────────────────

    def _build_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        from ltx_core.components.patchifiers import get_pixel_coords
        from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
        shape = VideoLatentShape(
            frames=num_frames, height=height, width=width,
            batch=batch_size, channels=128,
        )
        coords = self._video_patchifier.get_patch_grid_bounds(output_shape=shape, device=device)
        positions = get_pixel_coords(
            latent_coords=coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype)
        positions[:, 0, ...] = positions[:, 0, ...] / fps
        return positions

    # ── Audio helper ─────────────────────────────────────────────

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        batch_sigma: Tensor,
        audio_embeds: Tensor,
        text_mask: Tensor,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Any, Tensor, Tensor]:
        """Standard flow-matching audio at same sigma as VFM video batch_sigma."""
        from ltx_core.model.transformer.modality import Modality

        audio_data = batch["audio_latents"]
        a0 = self._audio_patchifier.patchify(audio_data["latents"])  # [B, T, C*F]
        audio_seq_len = a0.shape[1]

        eps = torch.randn_like(a0)
        sigma_exp = batch_sigma.view(-1, 1, 1)
        noisy_audio = (1 - sigma_exp) * a0 + sigma_exp * eps
        audio_targets = eps - a0

        audio_timesteps = batch_sigma.view(-1, 1).expand(-1, audio_seq_len)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_data["latents"].shape[2],
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        audio_mod = Modality(
            enabled=True,
            latent=noisy_audio.to(dtype),
            sigma=batch_sigma,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_embeds,
            context_mask=text_mask,
        )
        audio_loss_mask = torch.ones(B, audio_seq_len, dtype=torch.bool, device=device)
        return audio_mod, audio_targets, audio_loss_mask

    # ── Core training step ───────────────────────────────────────

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        from ltx_core.model.transformer.modality import Modality

        cfg = self.config
        latents = batch["latents"]
        x0_raw: Tensor = latents["latents"]  # [B, C, F, H, W]
        B, C, F, H, W = x0_raw.shape

        fps = latents.get("fps", None)
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        text_embeds: Tensor = conditions["video_prompt_embeds"]    # [B, text_seq, D]
        audio_embeds: Tensor = conditions.get(
            "audio_prompt_embeds", text_embeds
        )                                                          # [B, text_seq, D]
        text_mask: Tensor = conditions["prompt_attention_mask"]    # [B, text_seq]

        device = x0_raw.device
        dtype = x0_raw.dtype

        x0 = self._video_patchifier.patchify(x0_raw)  # [B, seq, 128]
        seq_len = x0.shape[1]

        positions = self._build_positions(F, H, W, B, fps, device, dtype)

        # ── First-frame conditioning mask (for i2v) ──────────────
        conditioning_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        tokens_per_frame = H * W
        if cfg.first_frame_conditioning_p > 0 and random.random() < cfg.first_frame_conditioning_p:
            if tokens_per_frame < seq_len:
                conditioning_mask[:, :tokens_per_frame] = True

        # ── Adapter noise ─────────────────────────────────────────
        use_adapter = (
            random.random() < cfg.alpha
            and self._noise_adapter is not None
        )

        if use_adapter:
            from ltx_core.model.transformer.noise_adapter_v1b import TASK_CLASSES
            task_idx = TASK_CLASSES.get("i2v", 0)
            task_class = torch.tensor([task_idx] * B, device=device, dtype=torch.long)

            adapter_out = self._noise_adapter(
                text_embeddings=text_embeds.detach().float(),
                text_mask=text_mask.bool(),
                positions=positions.float(),
                task_class=task_class,
            )
            mu: Tensor = adapter_out[0]              # [B, seq, 128]
            log_sigma: Tensor = adapter_out[1]       # [B, seq, 128]

            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn_like(mu)
            z = (mu + sigma_adapter * eps).to(dtype)
        else:
            z = torch.randn_like(x0)
            mu = None
            log_sigma = None

        # ── Per-token sigma from SigmaHead ───────────────────────
        if use_adapter and cfg.per_token_sigma and self._sigma_head is not None:
            per_token_sigma = self._sigma_head(mu.float(), x0.float())  # [B, seq]
            # Conditioning tokens (first frame for i2v) get sigma=0
            per_token_sigma = per_token_sigma * (~conditioning_mask).float()
            batch_sigma = per_token_sigma[~conditioning_mask].mean().detach()
            batch_sigma = batch_sigma.unsqueeze(0).expand(B)
            video_timesteps = per_token_sigma
        else:
            # Fallback: sample uniform sigma (standard flow matching)
            sigmas = timestep_sampler.sample_for(x0)          # [B, 1, 1] or [B]
            sigma_scalar = sigmas.squeeze()
            batch_sigma = sigma_scalar
            video_timesteps = sigma_scalar.view(-1, 1).expand(-1, seq_len).clone()
            video_timesteps[conditioning_mask] = 0.0

        # ── Build noisy input ────────────────────────────────────
        # For adapter path: pure noise (no interpolation — matches inference)
        # For fallback: standard interpolation x_t = (1-σ)*x0 + σ*z
        if use_adapter:
            noisy_video = z.clone()
        else:
            sigmas_expanded = batch_sigma.view(-1, 1, 1)
            noisy_video = (1 - sigmas_expanded) * x0 + sigmas_expanded * z

        # First-frame tokens are always clean
        noisy_video = torch.where(
            conditioning_mask.unsqueeze(-1),
            x0, noisy_video,
        )

        # Target velocity: v = z - x0
        video_targets = z - x0

        video_mod = Modality(
            enabled=True,
            latent=noisy_video,
            sigma=batch_sigma,
            timesteps=video_timesteps,
            positions=positions,
            context=text_embeds,
            context_mask=text_mask,
        )

        # ── Audio modality (random Gaussian noise, same batch_sigma) ────
        audio_mod = None
        audio_targets_out = None
        audio_loss_mask_out = None
        if cfg.with_audio:
            audio_mod, audio_targets_out, audio_loss_mask_out = self._prepare_audio_inputs(
                batch=batch,
                batch_sigma=batch_sigma,
                audio_embeds=audio_embeds,
                text_mask=text_mask,
                B=B,
                device=device,
                dtype=dtype,
            )

        inputs = ModelInputs(
            video=video_mod,
            audio=audio_mod,
            video_targets=video_targets,
            audio_targets=audio_targets_out,
            video_loss_mask=~conditioning_mask,
            audio_loss_mask=audio_loss_mask_out,
        )

        # Stash for compute_loss
        inputs._vfm_use_adapter = use_adapter
        inputs._vfm_mu = mu
        inputs._vfm_log_sigma = log_sigma
        inputs._vfm_z = z
        inputs._vfm_x0 = x0

        return inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        cfg = self.config
        mask = inputs.video_loss_mask  # [B, seq]

        # Flow matching loss: MSE(v_pred, v_target)
        video_loss = (video_pred - inputs.video_targets).pow(2)
        if mask is not None and mask.any():
            C = video_loss.shape[-1]
            loss_mf = (video_loss * mask.unsqueeze(-1)).sum() / (mask.sum().clamp(min=1) * C)
        else:
            loss_mf = video_loss.mean()

        # Audio loss (standard MSE, all tokens)
        audio_loss = torch.tensor(0.0, device=video_pred.device)
        if cfg.with_audio and audio_pred is not None and inputs.audio_targets is not None:
            audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()

        total_loss = loss_mf + audio_loss

        # Gaussian KL: KL(N(mu, σ²) || N(0,1))
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        mu = getattr(inputs, "_vfm_mu", None)
        log_sigma = getattr(inputs, "_vfm_log_sigma", None)

        if use_adapter and mu is not None and log_sigma is not None and cfg.kl_weight > 0:
            kl_per_dim = 0.5 * (
                mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1
            )
            if cfg.kl_free_bits > 0:
                kl_per_dim = kl_per_dim.clamp(min=cfg.kl_free_bits)
            gaussian_kl = kl_per_dim.mean()
            total_loss = total_loss + cfg.kl_weight * gaussian_kl
        else:
            gaussian_kl = torch.tensor(0.0, device=video_pred.device)

        # W&B logging
        if self._current_step % 20 == 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "vfm/loss_mf": loss_mf.item(),
                        "vfm/loss_audio": audio_loss.item(),
                        "vfm/loss_kl": gaussian_kl.item(),
                        "vfm/loss_total": total_loss.item(),
                        "vfm/use_adapter": float(use_adapter),
                    })
            except Exception:
                pass

        return total_loss
