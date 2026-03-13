"""Vanilla VFM (Variational Flow Maps) training strategy for LTX-2.

Implements VFM (arXiv:2603.07276) directly on the full 48-layer LTX-2 transformer
WITHOUT the SCD encoder-decoder split. Simpler than VFM-SCD: the full transformer
IS the flow map, and text embeddings ARE the observation for the noise adapter.

VFM Paper Eq. 19:
    L(θ,φ) = (1/2τ²) * L_MF(θ;φ) + (1/2σ²) * L_obs(θ⁻,φ) + L_KL(φ)

Architecture:
    Text embeddings (observation y) → Noise Adapter qφ → structured noise z
    → Full 48-layer transformer (flow map fθ) → velocity → x̂₀
"""

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.noise_adapter import (
    TASK_CLASSES,
    NoiseAdapterMLP,
    NoiseAdapterTransformer,
    create_noise_adapter,
)
from ltx_trainer import logger
from ltx_trainer.inverse_problems import (
    InverseProblemConfig,
    InverseProblemSampler,
)
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class VFMTrainingConfig(TrainingStrategyConfigBase):
    """Configuration for vanilla VFM training (no SCD split)."""

    name: Literal["vfm"] = "vfm"

    # === First-frame conditioning ===
    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of first-frame conditioning during training",
        ge=0.0, le=1.0,
    )

    # === VFM Noise Adapter ===
    adapter_variant: Literal["mlp", "transformer"] = Field(
        default="mlp",
        description="Noise adapter architecture",
    )
    adapter_hidden_dim: int = Field(default=1024)
    adapter_num_layers: int = Field(default=4)
    adapter_learning_rate: float = Field(default=1e-4)

    # === VFM Loss Hyperparameters ===
    tau: float = Field(default=1.0, gt=0.0,
        description="Data misfit tolerance τ for L_MF scaling")
    obs_noise_level: float = Field(default=0.2, gt=0.0,
        description="Observation noise σ for L_obs scaling")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0,
        description="P(adapter noise) vs P(standard N(0,I))")
    kl_weight: float = Field(default=3.0, ge=0.0,
        description="Weight for KL divergence loss")
    kl_free_bits: float = Field(default=0.25, ge=0.0,
        description="Per-dim KL floor to prevent sigma collapse")
    obs_loss_weight: float = Field(default=1.0, ge=0.0)

    # === Adaptive Loss ===
    adaptive_loss: bool = Field(default=True)
    adaptive_exclude_kl: bool = Field(default=True,
        description="Exclude L_KL from adaptive scaling to prevent sigma collapse")
    adaptive_gamma: float = Field(default=1.0, gt=0.0)
    adaptive_p: float = Field(default=0.5, gt=0.0)

    # === EMA ===
    ema_decay: float = Field(default=0.999, ge=0.0, le=1.0)

    # === Inverse Problems ===
    inverse_problem_weights: dict[str, float] = Field(
        default_factory=lambda: {"i2v": 0.4, "t2v": 0.6},
    )

    # === Flow Map Freezing ===
    freeze_flow_map_steps: int = Field(default=0, ge=0,
        description="Freeze LoRA for first N steps, training only adapter")

    # === Audio ===
    with_audio: bool = Field(default=False)
    audio_latents_dir: str = Field(default="audio_latents")

    # === Logging ===
    log_reconstructions: bool = Field(default=False)
    reconstruction_log_interval: int = Field(default=50)
    log_vfm_metrics: bool = Field(default=True)


class VFMTrainingStrategy(TrainingStrategy):
    """Vanilla VFM training strategy — full 48-layer transformer as flow map.

    No SCD split. Text embeddings serve as the observation for the noise adapter.
    The adapter learns to map text → structured noise that, when passed through
    the full transformer in 1 step, produces conditional video.
    """

    config: VFMTrainingConfig

    def __init__(self, config: VFMTrainingConfig):
        super().__init__(config)
        self._noise_adapter: NoiseAdapterMLP | NoiseAdapterTransformer | None = None
        self._transformer_ref = None  # Reference to the full transformer (for EMA)
        self._inverse_problem_sampler: InverseProblemSampler | None = None
        self._current_step: int = 0
        self._ema_state: dict[str, torch.Tensor] = {}
        self._last_vfm_metrics: dict[str, Any] = {}

        # Build inverse problem configs
        self._ip_configs = []
        for name, weight in config.inverse_problem_weights.items():
            noise = 0.3 if name == "denoise" else (0.0 if name == "t2v" else config.obs_noise_level)
            self._ip_configs.append(
                InverseProblemConfig(name=name, weight=weight, noise_level=noise)
            )

    def set_noise_adapter(self, adapter: NoiseAdapterMLP | NoiseAdapterTransformer) -> None:
        self._noise_adapter = adapter

    def set_transformer_ref(self, transformer) -> None:
        """Store reference to full transformer for EMA updates."""
        self._transformer_ref = transformer

    def set_current_step(self, step: int) -> None:
        self._current_step = step
        self._update_ema()
        self._update_flow_map_freeze(step)

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self._transformer_ref is None:
            return
        decay = self.config.ema_decay
        for name, param in self._transformer_ref.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._ema_state:
                self._ema_state[name] = param.data.clone()
            else:
                self._ema_state[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _update_flow_map_freeze(self, step: int) -> None:
        freeze_steps = self.config.freeze_flow_map_steps
        if freeze_steps <= 0 or self._transformer_ref is None:
            return
        should_freeze = step < freeze_steps
        for name, param in self._transformer_ref.named_parameters():
            if "noise_adapter" in name:
                continue
            if param.requires_grad != (not should_freeze):
                param.requires_grad = not should_freeze
        if step == freeze_steps:
            logger.info(f"VFM: Unfreezing flow map (LoRA) at step {step}")
        elif step == 0:
            logger.info(f"VFM: Freezing flow map for first {freeze_steps} steps")

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        sources = {"latents": "latents", "conditions": "conditions"}
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def set_text_embed_dim(self, dim: int) -> None:
        """Set actual text embedding dim (called by trainer after loading model)."""
        self._text_embed_dim = dim

    def get_noise_adapter_params(self) -> dict[str, Any]:
        """Return adapter constructor kwargs. Input is text embed dim."""
        input_dim = getattr(self, "_text_embed_dim", 3840)
        return {
            "input_dim": input_dim,
            "latent_dim": 128,  # Video latent channels
            "variant": self.config.adapter_variant,
            "hidden_dim": self.config.adapter_hidden_dim,
            "num_layers": self.config.adapter_num_layers,
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare VFM training inputs using full transformer as flow map."""
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents_spatial = video_latents  # Keep for reconstruction logging
        video_latents = self._video_patchifier.patchify(video_latents)  # [B, seq_len, C]

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

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

        # Positions
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # ============================
        # VFM: NOISE ADAPTER
        # ============================
        use_adapter_noise = random.random() < self.config.alpha

        if use_adapter_noise and self._noise_adapter is not None:
            # Pool text embeddings as observation: [B, text_seq, D] → [B, D]
            # Mask-aware mean pooling
            mask = prompt_attention_mask.unsqueeze(-1)  # [B, text_seq, 1]
            pooled_text = (video_prompt_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # Expand to video sequence length: [B, D] → [B, video_seq_len, D]
            text_obs = pooled_text.unsqueeze(1).expand(-1, video_seq_len, -1)

            # Sample inverse problem
            ip_sample = self._inverse_problem_sampler.sample(video_latents)

            # Get adapter noise distribution
            mu, log_sigma = self._noise_adapter(
                encoder_features=text_obs.detach(),
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
        # BUILD MODALITY (full transformer forward)
        # ============================
        video_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

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

        # NOTE: No _scd_model, no _encoder_features — trainer uses standard
        # self._transformer(video=..., audio=..., perturbations=None) path

        # VFM-specific data for loss computation
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

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute VFM three-part loss (identical math to VFM-SCD)."""
        cfg = self.config

        # L_MF: velocity MSE
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        loss_mf = video_loss.mean()
        loss_mf_scaled = loss_mf / (2.0 * cfg.tau ** 2)

        # L_obs: observation consistency
        loss_obs = torch.tensor(0.0, device=video_pred.device)
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        if use_adapter and adapter_mu is not None:
            observation = inputs._vfm_observation
            noise_level = inputs._vfm_noise_level
            task_name = getattr(inputs, "_vfm_task_name", "t2v")

            if observation is not None and noise_level > 0:
                video_noise = inputs._vfm_video_noise
                x_hat = video_noise - video_pred.detach()
                clean_latents = inputs._vfm_video_latents
                loss_obs = (x_hat - clean_latents).pow(2).mean()

                if self._inverse_problem_sampler is not None and task_name in self._inverse_problem_sampler.operators:
                    operator = self._inverse_problem_sampler.operators[task_name]
                    x_hat_obs = operator(x_hat)
                    if observation.shape == x_hat_obs.shape:
                        loss_obs = loss_obs + (observation - x_hat_obs).pow(2).mean()

            sigma_sq = max(noise_level, cfg.obs_noise_level) ** 2
            loss_obs_scaled = cfg.obs_loss_weight * loss_obs / (2.0 * sigma_sq)
        else:
            loss_obs_scaled = loss_obs

        # L_KL with free bits
        loss_kl = torch.tensor(0.0, device=video_pred.device)
        if use_adapter and adapter_mu is not None:
            adapter_log_sigma = inputs._vfm_adapter_log_sigma
            kl_per_dim = 0.5 * (
                adapter_mu.pow(2)
                + torch.exp(2 * adapter_log_sigma)
                - 2 * adapter_log_sigma
                - 1
            )
            if cfg.kl_free_bits > 0:
                kl_per_dim = torch.clamp(kl_per_dim, min=cfg.kl_free_bits)
            loss_kl = kl_per_dim.mean()

        loss_kl_scaled = cfg.kl_weight * loss_kl

        # Total loss
        flow_frozen = cfg.freeze_flow_map_steps > 0 and self._current_step < cfg.freeze_flow_map_steps
        if flow_frozen:
            adapter_loss = loss_obs_scaled + loss_kl_scaled
            if not adapter_loss.requires_grad and self._noise_adapter is not None:
                dummy = sum(p.sum() * 0.0 for p in self._noise_adapter.parameters())
                adapter_loss = adapter_loss + dummy
            total_loss = loss_mf_scaled.detach() + adapter_loss
        else:
            total_loss = loss_mf_scaled + loss_obs_scaled + loss_kl_scaled

        # Adaptive loss scaling
        if cfg.adaptive_loss:
            with torch.no_grad():
                if cfg.adaptive_exclude_kl:
                    non_kl = loss_mf_scaled + loss_obs_scaled
                    weight = 1.0 / (non_kl.detach() + cfg.adaptive_gamma).pow(cfg.adaptive_p)
                else:
                    weight = 1.0 / (total_loss.detach() + cfg.adaptive_gamma).pow(cfg.adaptive_p)
            if cfg.adaptive_exclude_kl:
                total_loss = weight * (loss_mf_scaled + loss_obs_scaled) + loss_kl_scaled
            else:
                total_loss = weight * total_loss

        # Store metrics
        self._last_vfm_metrics = {
            "vfm/loss_mf": loss_mf.item(),
            "vfm/loss_obs": loss_obs.item() if isinstance(loss_obs, Tensor) else loss_obs,
            "vfm/loss_kl": loss_kl.item() if isinstance(loss_kl, Tensor) else loss_kl,
            "vfm/loss_total": total_loss.item(),
            "vfm/task": getattr(inputs, "_vfm_task_name", "unknown"),
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
        return {
            "vfm_adapter_variant": self.config.adapter_variant,
            "vfm_tau": self.config.tau,
            "vfm_alpha": self.config.alpha,
        }

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction video clips (GT | Pred side-by-side) to W&B.

        Decodes a random sample from the batch through the VAE and logs
        as wandb.Video so you can visually inspect overfitting quality.
        Falls back to a single-frame image if VAE decode fails.
        """
        if not WANDB_AVAILABLE or wandb.run is None or not self.config.log_reconstructions:
            return {}

        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is None:
            return {}

        import numpy as np

        b, c, f, h, w = raw_latents.shape
        noise = inputs.shared_noise
        pred_clean = noise - video_pred
        pred_clean_spatial = pred_clean.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)

        # Pick a random sample from the batch
        sample_idx = random.randint(0, b - 1) if b > 1 else 0

        log_dict = {}
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        task_name = getattr(inputs, "_vfm_task_name", "unknown")
        adapter_tag = f"adapter/{task_name}" if use_adapter else "standard_noise"

        if vae_decoder is not None:
            try:
                decoder_device = next(vae_decoder.parameters()).device
                decoder_dtype = next(vae_decoder.parameters()).dtype
                with torch.inference_mode():
                    gt_decoded = vae_decoder(
                        raw_latents[sample_idx:sample_idx+1].to(
                            device=decoder_device, dtype=decoder_dtype
                        )
                    )
                    pred_decoded = vae_decoder(
                        pred_clean_spatial[sample_idx:sample_idx+1].to(
                            device=decoder_device, dtype=decoder_dtype
                        )
                    )
                # VAE outputs [-1, 1] → [0, 1]
                gt_decoded = gt_decoded.float().clamp(-1, 1) * 0.5 + 0.5
                pred_decoded = pred_decoded.float().clamp(-1, 1) * 0.5 + 0.5

                # gt_decoded shape: [1, 3, T, H, W]
                gt_frames = gt_decoded[0].cpu()    # [3, T, H, W]
                pred_frames = pred_decoded[0].cpu()  # [3, T, H, W]

                # Side-by-side: concatenate along width dimension
                # [3, T, H, W*2]
                side_by_side = torch.cat([gt_frames, pred_frames], dim=-1)

                # Convert to wandb.Video format: [T, C, H, W] uint8
                # wandb.Video expects (T, C, H, W) or (T, H, W, C)
                video_np = (side_by_side.permute(1, 0, 2, 3) * 255).clamp(0, 255).to(torch.uint8).numpy()
                # video_np shape: [T, 3, H, W*2]

                log_dict[f"{prefix}/reconstruction_video"] = wandb.Video(
                    video_np, fps=8,
                    caption=f"Step {step} | {adapter_tag} | Left: GT | Right: Pred (sample {sample_idx})",
                )

                # Also log a mid-frame image for quick glancing
                mid_f = gt_frames.shape[1] // 2
                import torchvision.utils as vutils
                grid = vutils.make_grid(
                    [gt_frames[:, mid_f], pred_frames[:, mid_f]],
                    nrow=2, padding=4,
                )
                log_dict[f"{prefix}/reconstruction"] = wandb.Image(
                    grid.permute(1, 2, 0).numpy(),
                    caption=f"Step {step} | {adapter_tag} | Left: GT | Right: Pred (frame {mid_f})",
                )
                return log_dict
            except Exception as e:
                logger.warning(f"Failed to VAE-decode reconstruction: {e}")

        # Fallback: latent-space pseudo-RGB (single frame)
        mid_f = f // 2
        def normalize(x):
            x = x - x.min()
            return x / (x.max() + 1e-8)

        import torchvision.utils as vutils
        gt_vis = raw_latents[sample_idx, :3, mid_f].cpu().float()
        pred_vis = pred_clean_spatial[sample_idx, :3, mid_f].cpu().float()
        grid = vutils.make_grid([normalize(gt_vis), normalize(pred_vis)], nrow=2, padding=4)
        log_dict[f"{prefix}/reconstruction_latent"] = wandb.Image(
            grid.permute(1, 2, 0).numpy(),
            caption=f"Step {step} | {adapter_tag} | GT | Pred",
        )
        return log_dict
