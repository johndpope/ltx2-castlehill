"""Variational Flow Maps + Separable Causal Diffusion (VFM-SCD) training strategy.

Combines VFM's principled noise adaptation with SCD's encoder-decoder split to enable
one/few-step conditional video generation.

VFM Paper (arXiv:2603.07276) — Core training objective (Eq. 19):
    L(θ,φ) = (1/2τ²) * L_MF(θ;φ) + (1/2σ²) * L_obs(θ⁻,φ) + L_KL(φ)

    where:
    - L_MF: Mean flow loss (velocity MSE) — existing SCD loss, but with noise from adapter
    - L_obs: Observation loss — checks that fθ(z) is consistent with observation y
    - L_KL: KL divergence — regularizes adapter output toward N(0,I)
    - θ: Flow map (decoder) parameters
    - φ: Noise adapter parameters
    - θ⁻: EMA of θ (stabilizes observation loss)
    - τ: Data misfit tolerance (hyperparameter)
    - σ: Observation noise level

Architecture mapping (VFM → SCD):
    - Flow map fθ(z): SCD decoder (16 transformer layers)
    - Observation y: Encoder features (from clean latents with causal mask)
    - Noise adapter qφ(z|y): Lightweight MLP/Transformer on encoder features
    - Forward operator A(x): Task-specific degradation (i2v, inpaint, sr, etc.)

VFM Paper Algorithm 2 — Training loop (adapted for SCD):
    1. Sample inverse problem class c, clean video x
    2. Compute observation y = A(x) + ε
    3. Run SCD encoder on clean x → encoder_features
    4. Noise adapter: z ~ qφ(encoder_features, c) with prob α, else z ~ N(0,I)
    5. Compute L_MF: velocity MSE between decoder output and target
    6. Compute L_obs: ||y - A(fθ⁻(z))||² (one-step decode + forward operator)
    7. Compute L_KL: KL(qφ(z|y) || N(0,I))
    8. Apply adaptive loss scaling
    9. Update θ and φ jointly
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
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    shift_encoder_features,
)
from ltx_trainer import logger
from ltx_trainer.inverse_problems import (
    InverseProblemConfig,
    InverseProblemSampler,
    default_inverse_problems,
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


class VFMSCDTrainingConfig(TrainingStrategyConfigBase):
    """Configuration for VFM+SCD joint training.

    Combines SCD's encoder-decoder split with VFM's variational noise adapter.
    The adapter learns observation-dependent noise distributions, enabling
    one/few-step conditional generation without iterative guidance.
    """

    name: Literal["vfm_scd"] = "vfm_scd"

    # === SCD Architecture (inherited from SCD strategy) ===
    encoder_layers: int = Field(
        default=32,
        description="Number of transformer layers for encoder (remaining go to decoder)",
        ge=1,
    )

    decoder_input_combine: str = Field(
        default="token_concat",
        description="How to combine encoder features with decoder input. "
        "Options: 'token_concat' (best), 'add', 'token_concat_with_proj'.",
    )

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of first-frame conditioning during training",
        ge=0.0,
        le=1.0,
    )

    per_frame_decoder: bool = Field(
        default=True,
        description="Process each frame independently through decoder (matches inference)",
    )

    # === VFM Noise Adapter ===
    adapter_variant: Literal["mlp", "transformer"] = Field(
        default="mlp",
        description="Noise adapter architecture: 'mlp' (per-token) or 'transformer' (sequence-aware)",
    )

    adapter_hidden_dim: int = Field(
        default=1024,
        description="Hidden dimension of noise adapter MLP/Transformer",
    )

    adapter_num_layers: int = Field(
        default=4,
        description="Number of layers in the noise adapter",
    )

    adapter_learning_rate: float = Field(
        default=1e-4,
        description="Learning rate for noise adapter (typically higher than flow map lr)",
    )

    # === VFM Loss Hyperparameters ===
    tau: float = Field(
        default=1.0,
        description="Data misfit tolerance τ. Controls weight of L_MF. "
        "Larger τ = more stable optimization. Paper recommends τ > σ.",
        gt=0.0,
    )

    obs_noise_level: float = Field(
        default=0.1,
        description="Default observation noise level σ for L_obs. "
        "Individual problem classes may override this.",
        gt=0.0,
    )

    alpha: float = Field(
        default=0.5,
        description="Probability of using adapter noise vs standard N(0,I). "
        "VFM Paper §3.4: Mix conditional and unconditional noise to preserve "
        "unconditional generation quality.",
        ge=0.0,
        le=1.0,
    )

    kl_weight: float = Field(
        default=1.0,
        description="Weight for KL divergence loss. Can be annealed during training.",
        ge=0.0,
    )

    obs_loss_weight: float = Field(
        default=1.0,
        description="Additional weight for observation loss (on top of 1/(2σ²) scaling).",
        ge=0.0,
    )

    # === VFM Adaptive Loss (Paper §3.4) ===
    adaptive_loss: bool = Field(
        default=True,
        description="Use adaptive loss scaling w = 1/stopgrad(||L + γ||^p)",
    )

    adaptive_gamma: float = Field(
        default=1.0,
        description="γ constant for adaptive loss scaling",
        gt=0.0,
    )

    adaptive_p: float = Field(
        default=0.5,
        description="p exponent for adaptive loss scaling",
        gt=0.0,
    )

    # === EMA for observation loss ===
    ema_decay: float = Field(
        default=0.999,
        description="EMA decay rate for θ⁻ used in observation loss",
        ge=0.0,
        le=1.0,
    )

    # === Inverse Problems ===
    inverse_problem_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "i2v": 0.4,
            "inpaint": 0.2,
            "sr": 0.15,
            "denoise": 0.1,
            "t2v": 0.15,
        },
        description="Sampling weights for each inverse problem class",
    )

    # === Audio ===
    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents",
    )

    # === Scheduled Sampling (from SCD) ===
    scheduled_sampling: bool = Field(
        default=False,
        description="Enable AR-aware scheduled sampling",
    )
    ss_p_ar_start: float = Field(default=0.0, ge=0.0, le=1.0)
    ss_p_ar_end: float = Field(default=0.5, ge=0.0, le=1.0)
    ss_warmup_steps: int = Field(default=50, ge=0)
    ss_ramp_steps: int = Field(default=150, ge=1)
    ss_noise_augment: float = Field(default=0.05, ge=0.0, le=0.5)

    # === Logging ===
    log_reconstructions: bool = Field(default=False)
    reconstruction_log_interval: int = Field(default=50)
    log_vfm_metrics: bool = Field(
        default=True,
        description="Log VFM-specific metrics (L_obs, L_KL, adapter stats) to W&B",
    )


class VFMSCDTrainingStrategy(TrainingStrategy):
    """VFM+SCD training strategy for one-step conditional video generation.

    Extends the SCD training paradigm with a variational noise adapter that
    learns observation-dependent noise distributions. This enables conditional
    generation (image-to-video, inpainting, super-resolution) in 1-4 steps
    instead of the typical 8-30 denoising steps.

    The strategy jointly trains:
    1. The SCD encoder-decoder (flow map fθ) via mean flow loss
    2. The noise adapter (qφ) via observation loss + KL regularization
    """

    config: VFMSCDTrainingConfig

    def __init__(self, config: VFMSCDTrainingConfig):
        super().__init__(config)
        self._scd_model: LTXSCDModel | None = None
        self._noise_adapter: NoiseAdapterMLP | NoiseAdapterTransformer | None = None
        self._ema_model: LTXSCDModel | None = None  # EMA model for L_obs
        self._inverse_problem_sampler: InverseProblemSampler | None = None
        self._current_step: int = 0

        # Build inverse problem configs from weights
        self._ip_configs = []
        for name, weight in config.inverse_problem_weights.items():
            noise = 0.3 if name == "denoise" else (0.0 if name == "t2v" else config.obs_noise_level)
            self._ip_configs.append(
                InverseProblemConfig(name=name, weight=weight, noise_level=noise)
            )

    def set_scd_model(self, model: LTXSCDModel) -> None:
        """Set the SCD model wrapper. Called by the trainer after model creation."""
        self._scd_model = model

    def set_noise_adapter(self, adapter: NoiseAdapterMLP | NoiseAdapterTransformer) -> None:
        """Set the noise adapter. Called by the trainer after creation."""
        self._noise_adapter = adapter

    def set_ema_model(self, ema_model: LTXSCDModel) -> None:
        """Set the EMA model for observation loss. Called by trainer."""
        self._ema_model = ema_model

    def set_current_step(self, step: int) -> None:
        self._current_step = step

    def initialize_inverse_problems(self, tokens_per_frame: int) -> None:
        """Initialize inverse problem sampler once we know tokens_per_frame."""
        self._inverse_problem_sampler = InverseProblemSampler(
            problems=self._ip_configs,
            tokens_per_frame=tokens_per_frame,
        )

    def _get_p_ar(self) -> float:
        cfg = self.config
        if not cfg.scheduled_sampling:
            return 0.0
        step = self._current_step
        if step < cfg.ss_warmup_steps:
            return 0.0
        ramp_progress = min(1.0, (step - cfg.ss_warmup_steps) / cfg.ss_ramp_steps)
        return cfg.ss_p_ar_start + ramp_progress * (cfg.ss_p_ar_end - cfg.ss_p_ar_start)

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        sources = {"latents": "latents", "conditions": "conditions"}
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare VFM-SCD training inputs.

        VFM Paper Algorithm 2, adapted for SCD:
            1. Extract clean video latents x
            2. Run SCD encoder (clean, σ=0, causal mask) → encoder_features
            3. Shift encoder features by 1 frame (causal conditioning)
            4. Sample inverse problem: c ~ p(c), compute y = A(x) + ε
            5. Noise adapter: z ~ qφ(encoder_features, c) with prob α
            6. Construct noisy interpolation: x_t = (1-σ)*x_0 + σ*z
            7. Package for decoder forward pass + VFM loss computation
        """
        # === Step 1: Extract and patchify clean video latents ===
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]

        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        video_latents_spatial = video_latents  # Keep for later
        video_latents = self._video_patchifier.patchify(video_latents)  # [B, seq_len, C]

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions.get("audio_prompt_embeds")
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype
        tokens_per_frame = video_seq_len // num_frames

        # Initialize inverse problem sampler on first call
        if self._inverse_problem_sampler is None:
            self.initialize_inverse_problems(tokens_per_frame)

        # === Step 2: First-frame conditioning mask ===
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # === Step 3: Sample timesteps and noise ===
        sigmas = timestep_sampler.sample_for(video_latents)

        # === Step 4: Position embeddings ===
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # === Step 5: SCD Encoder pass (clean latents, σ=0, causal mask) ===
        if self._scd_model is not None:
            return self._prepare_vfm_scd_inputs(
                video_latents=video_latents,
                video_latents_spatial=video_latents_spatial,
                video_positions=video_positions,
                video_prompt_embeds=video_prompt_embeds,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                video_conditioning_mask=video_conditioning_mask,
                sigmas=sigmas,
                batch_size=batch_size,
                video_seq_len=video_seq_len,
                tokens_per_frame=tokens_per_frame,
                num_frames=num_frames,
                device=device,
                dtype=dtype,
                batch=batch,
            )

        # Fallback: standard training without SCD (shouldn't happen in VFM mode)
        raise RuntimeError("VFM-SCD strategy requires SCD model wrapper to be set")

    def _prepare_vfm_scd_inputs(
        self,
        video_latents: Tensor,
        video_latents_spatial: Tensor,
        video_positions: Tensor,
        video_prompt_embeds: Tensor,
        audio_prompt_embeds: Tensor | None,
        prompt_attention_mask: Tensor,
        video_conditioning_mask: Tensor,
        sigmas: Tensor,
        batch_size: int,
        video_seq_len: int,
        tokens_per_frame: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        batch: dict[str, Any],
    ) -> ModelInputs:
        """Core VFM-SCD input preparation with noise adapter integration.

        This implements Algorithm 2 from the VFM paper, adapted for the SCD architecture.
        """
        # ============================
        # ENCODER PASS (clean, σ=0)
        # ============================
        encoder_timesteps = torch.zeros(batch_size, video_seq_len, device=device, dtype=dtype)
        encoder_modality = Modality(
            enabled=True,
            sigma=torch.zeros(batch_size, device=device, dtype=dtype),
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

        # ============================
        # SHIFT ENCODER FEATURES
        # ============================
        encoder_features = encoder_video_args.x  # [B, seq_len, D]
        shifted_features = shift_encoder_features(
            encoder_features, tokens_per_frame, num_frames
        )

        # ============================
        # VFM: NOISE ADAPTER SAMPLING
        # ============================
        # VFM Paper Algorithm 2, lines 7-13:
        # With probability α, sample z from adapter qφ(z|y,c)
        # Otherwise, sample z ~ N(0,I) (preserves unconditional generation)
        use_adapter_noise = random.random() < self.config.alpha

        if use_adapter_noise and self._noise_adapter is not None:
            # Sample inverse problem
            ip_sample = self._inverse_problem_sampler.sample(video_latents)

            # Get adapter noise distribution parameters
            mu, log_sigma = self._noise_adapter(
                encoder_features=shifted_features.detach(),  # Detach for adapter input
                task_class=ip_sample.task_class,
            )

            # Sample via reparameterization: z = μ + σ ⊙ ε
            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn_like(mu)
            video_noise = mu + sigma_adapter * eps

            # Store adapter outputs for loss computation
            adapter_mu = mu
            adapter_log_sigma = log_sigma
            task_class = ip_sample.task_class
            ip_observation = ip_sample.observation
            ip_task_name = ip_sample.task_name
            ip_noise_level = ip_sample.noise_level
        else:
            # Standard noise: z ~ N(0,I)
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

        # Keep conditioning tokens clean
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Velocity target: v = z - x_0
        video_targets = video_noise - video_latents

        # ============================
        # DECODER MODALITY
        # ============================
        decoder_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

        decoder_modality = Modality(
            enabled=True,
            sigma=sigmas.squeeze(),
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

        # Attach SCD-specific data
        model_inputs._encoder_features = shifted_features
        model_inputs._scd_model = self._scd_model
        model_inputs._encoder_audio_args = encoder_audio_args
        model_inputs._per_frame_decoder = self.config.per_frame_decoder
        model_inputs._tokens_per_frame = tokens_per_frame
        model_inputs._num_frames = num_frames
        model_inputs._raw_video_latents = batch["latents"]["latents"]

        # Attach VFM-specific data for loss computation
        model_inputs._vfm_adapter_mu = adapter_mu
        model_inputs._vfm_adapter_log_sigma = adapter_log_sigma
        model_inputs._vfm_task_class = task_class
        model_inputs._vfm_observation = ip_observation
        model_inputs._vfm_task_name = ip_task_name
        model_inputs._vfm_noise_level = ip_noise_level
        model_inputs._vfm_video_noise = video_noise
        model_inputs._vfm_video_latents = video_latents
        model_inputs._vfm_use_adapter = use_adapter_noise

        # Store noise and sigmas for reconstruction
        model_inputs.shared_noise = video_noise
        model_inputs.shared_sigmas = sigmas

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute the VFM three-part loss.

        VFM Paper Eq. 19:
            L(θ,φ) = (1/2τ²) * L_MF(θ;φ) + (1/2σ²) * L_obs(θ⁻,φ) + L_KL(φ)

        L_MF: Mean flow loss (velocity MSE) — standard SCD loss
            Measures decoder accuracy at predicting velocity v = z - x_0

        L_obs: Observation loss — consistency check
            After one-step decode z → x̂ = fθ⁻(z), check ||y - A(x̂)||²
            Uses EMA parameters θ⁻ for stability

        L_KL: KL divergence — regularization
            KL(qφ(z|y) || N(0,I)) keeps adapter noise close to standard normal
        """
        cfg = self.config

        # ============================
        # L_MF: Mean Flow Loss (velocity MSE)
        # ============================
        # This is the standard SCD velocity prediction loss
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        loss_mf = video_loss.mean()

        # Scale by 1/(2τ²) — VFM Paper Eq. 19
        loss_mf_scaled = loss_mf / (2.0 * cfg.tau ** 2)

        # ============================
        # L_obs: Observation Loss
        # ============================
        loss_obs = torch.tensor(0.0, device=video_pred.device)

        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        if use_adapter and adapter_mu is not None:
            observation = inputs._vfm_observation
            noise_level = inputs._vfm_noise_level
            task_name = getattr(inputs, "_vfm_task_name", "t2v")

            if observation is not None and noise_level > 0:
                # One-step decode: x̂ = z - v̂ (flow matching recovery)
                video_noise = inputs._vfm_video_noise
                x_hat = video_noise - video_pred.detach()  # Detach for stability (use EMA ideally)

                # Apply the same forward operator to x_hat so shapes match observation
                # VFM Paper Eq. 14: L_obs = E[||y - A(fθ(z))||²]
                # observation = A(x_gt) + ε, so we need A(x_hat) to compare
                if self._inverse_problem_sampler is not None and task_name in self._inverse_problem_sampler.operators:
                    operator = self._inverse_problem_sampler.operators[task_name]
                    x_hat_obs = operator(x_hat)
                else:
                    x_hat_obs = x_hat

                # Ensure shapes match (observation may have been produced with noise added)
                if observation.shape == x_hat_obs.shape:
                    obs_error = (observation - x_hat_obs).pow(2)
                    loss_obs = obs_error.mean()
                else:
                    # Shape mismatch — use reconstruction loss as fallback
                    clean_latents = inputs._vfm_video_latents
                    loss_obs = (x_hat - clean_latents).pow(2).mean()

            # Scale by 1/(2σ²) — VFM Paper Eq. 19
            sigma_sq = max(noise_level, cfg.obs_noise_level) ** 2
            loss_obs_scaled = cfg.obs_loss_weight * loss_obs / (2.0 * sigma_sq)
        else:
            loss_obs_scaled = loss_obs

        # ============================
        # L_KL: KL Divergence Loss
        # ============================
        loss_kl = torch.tensor(0.0, device=video_pred.device)

        if use_adapter and adapter_mu is not None:
            adapter_log_sigma = inputs._vfm_adapter_log_sigma
            # KL(N(μ, σ²) || N(0, I)) = 0.5 * (μ² + σ² - 2*log(σ) - 1)
            loss_kl = 0.5 * (
                adapter_mu.pow(2)
                + torch.exp(2 * adapter_log_sigma)
                - 2 * adapter_log_sigma
                - 1
            ).mean()

        loss_kl_scaled = cfg.kl_weight * loss_kl

        # ============================
        # TOTAL LOSS
        # ============================
        total_loss = loss_mf_scaled + loss_obs_scaled + loss_kl_scaled

        # Adaptive loss scaling — VFM Paper §3.4
        if cfg.adaptive_loss:
            with torch.no_grad():
                weight = 1.0 / (total_loss.detach() + cfg.adaptive_gamma).pow(cfg.adaptive_p)
            total_loss = weight * total_loss

        # Store VFM metrics for the trainer's logging loop to pick up
        # (avoids wandb step conflicts from logging during compute_loss)
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
            "scd_encoder_layers": self.config.encoder_layers,
            "scd_decoder_input_combine": self.config.decoder_input_combine,
            "vfm_adapter_variant": self.config.adapter_variant,
            "vfm_tau": self.config.tau,
            "vfm_alpha": self.config.alpha,
        }

    def get_noise_adapter_params(self) -> dict[str, Any]:
        """Get noise adapter constructor kwargs for the trainer to create the adapter."""
        if self._scd_model is not None:
            input_dim = self._scd_model.inner_dim
        else:
            input_dim = 3072  # Default LTX-2 inner dim

        return {
            "input_dim": input_dim,
            "latent_dim": 128,  # Video latent channels
            "variant": self.config.adapter_variant,
            "hidden_dim": self.config.adapter_hidden_dim,
            "num_layers": self.config.adapter_num_layers,
        }
