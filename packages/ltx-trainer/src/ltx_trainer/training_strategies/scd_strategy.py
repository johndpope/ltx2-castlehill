"""Separable Causal Diffusion (SCD) training strategy for LTX-2.

Implements the training procedure from:
    "Separable Causal Diffusion for Video Generation"
    arXiv:2602.10095

SCD Paper §1 (Introduction):
    The core observation is that in standard video diffusion transformers, the early layers
    produce temporally redundant features across denoising steps — they perform causal temporal
    reasoning that is largely invariant to the noise level. Meanwhile, the deeper layers handle
    per-frame spatial denoising that IS noise-level-dependent.

    This motivates splitting the transformer into:
    - Encoder: Runs ONCE with clean latents (sigma=0) and a frame-level causal mask.
      Captures temporal context (frame t sees frames <= t). Output is shifted by 1 frame
      so that frame t's decoder receives frame t-1's temporal features.
    - Decoder: Runs N times (once per denoising step) with actual noisy latents at
      sampled sigma levels. Conditioned on the shifted encoder features.

    This split enables ~3x inference speedup since the expensive encoder pass is amortized
    across all denoising steps, while the lighter decoder handles iterative refinement.

SCD Paper §5 (Training):
    The model is trained end-to-end. Gradients flow through BOTH encoder and decoder.
    The encoder learns to extract temporally-causal features from clean video, while the
    decoder learns to use those features (from the previous frame) to denoise the current
    frame. The training target is velocity v = epsilon - x_0 under flow matching.

Flow matching formulation used throughout:
    x_t = (1 - sigma) * x_0 + sigma * epsilon     (noisy interpolation)
    v   = epsilon - x_0                            (velocity = noise - clean)
    x_0_hat = epsilon - v_hat                      (recover clean from velocity prediction)
"""

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    build_frame_causal_mask,
    shift_encoder_features,
)
from ltx_trainer import logger
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


class SCDTrainingConfig(TrainingStrategyConfigBase):
    """Configuration for SCD training strategy.

    SCD Paper §5.5 (Encoder-Decoder Ratio):
        The paper recommends a ~2:1 encoder:decoder split. For a 48-layer transformer
        (as in LTX-2), this means 32 encoder layers + 16 decoder layers. The encoder
        handles the expensive causal temporal reasoning (run once), while the decoder
        handles cheaper per-frame denoising (run N times during inference). This ratio
        was found empirically to balance quality vs. speed — too few encoder layers
        degrades temporal coherence, too many wastes compute on the non-amortizable part.
    """

    name: Literal["scd"] = "scd"

    # SCD Paper §5.5: ~2:1 encoder:decoder ratio recommended.
    # For LTX-2's 48 layers: 32 encoder + 16 decoder.
    # The encoder handles causal temporal reasoning (run once at inference),
    # the decoder handles per-frame denoising (run N times at inference).
    encoder_layers: int = Field(
        default=32,
        description="Number of transformer layers for the encoder (remaining go to decoder). "
        "Default 32 follows the SCD paper's ~2:1 encoder:decoder ratio for a 48-layer model.",
        ge=1,
    )

    # SCD Paper §5.5: Token concatenation was found to be the best method for
    # combining encoder features with decoder input. The shifted encoder output
    # (frame t-1's features) is prepended to the decoder's noisy token sequence
    # for frame t, doubling the sequence length within the decoder but allowing
    # full cross-frame attention without additional projection layers.
    decoder_input_combine: str = Field(
        default="token_concat",
        description="How to combine encoder features with decoder input. "
        "Options: 'token_concat' (best from SCD paper), 'add', 'token_concat_with_proj'.",
    )

    # Extension beyond the base SCD paper: allows more than just the first frame
    # to be kept clean as context. With clean_context_ratio=0.0 (default), only
    # the first frame may be conditioned on (controlled by first_frame_conditioning_p).
    clean_context_ratio: float = Field(
        default=0.0,
        description="Fraction of frames (beyond the first) kept clean as additional context. "
        "0.0 means only the first frame is always clean context.",
        ge=0.0,
        le=1.0,
    )

    # SCD Paper §5.6 (Multi-Batch Decoder):
    # During training, multiple decoder passes can be run per single encoder pass.
    # Each decoder pass uses independently sampled noise and sigma, which:
    # (a) Amortizes the encoder's computational cost over multiple gradient updates,
    # (b) Provides more gradient signal per training step,
    # (c) Better simulates the inference regime where the encoder runs once and
    #     the decoder runs N denoising steps.
    # Default 1 for simplicity; paper uses up to 4 in some experiments.
    decoder_multi_batch: int = Field(
        default=1,
        description="Number of decoder passes per encoder pass. Higher values amortize "
        "encoder cost and provide more training signal. Default 1 for simplicity.",
        ge=1,
        le=4,
    )

    # SCD Paper §5.3 (First Frame Conditioning):
    # With probability p (default 0.1), the first frame is kept clean (sigma=0) during
    # the decoder pass. This trains the model for image-to-video (I2V) mode where the
    # first frame is given as input and the model generates subsequent frames.
    # The conditioning mask sets timestep=0 for these first-frame tokens, meaning
    # the decoder sees them as noise-free ground truth.
    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    per_frame_decoder: bool = Field(
        default=True,
        description="Process each frame independently through the decoder during training. "
        "When True (default), the decoder sees one frame (tokens_per_frame tokens) per forward "
        "pass, matching autoregressive inference. When False, all frames pass through the decoder "
        "simultaneously (legacy behavior, causes train/inference mismatch).",
    )

    # AR-aware scheduled sampling (fixes teacher-forcing / inference mismatch)
    # During training, the encoder always sees clean GT latents (σ=0). But at inference,
    # the encoder sees model predictions from previous frames (autoregressive rollout).
    # Scheduled sampling closes this gap by probabilistically replacing GT encoder inputs
    # with the model's own x̂₀ predictions, following a curriculum from 0 → p_ar_end.
    scheduled_sampling: bool = Field(
        default=False,
        description="Enable AR-aware scheduled sampling. Probabilistically replaces "
        "GT encoder inputs with model's own x̂₀ predictions to close the "
        "teacher-forcing / autoregressive inference gap.",
    )
    ss_p_ar_start: float = Field(
        default=0.0,
        description="Initial probability of using model predictions (start of curriculum).",
        ge=0.0,
        le=1.0,
    )
    ss_p_ar_end: float = Field(
        default=0.5,
        description="Final probability of using model predictions (end of curriculum).",
        ge=0.0,
        le=1.0,
    )
    ss_warmup_steps: int = Field(
        default=50,
        description="Steps before scheduled sampling begins (pure teacher forcing).",
        ge=0,
    )
    ss_ramp_steps: int = Field(
        default=150,
        description="Steps over which p_ar ramps from ss_p_ar_start to ss_p_ar_end.",
        ge=1,
    )
    ss_noise_augment: float = Field(
        default=0.05,
        description="Small noise added to x̂₀ to prevent overfitting to clean predictions. "
        "Simulates the slight imperfections of real inference predictions.",
        ge=0.0,
        le=0.5,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    # Reconstruction visualization
    log_reconstructions: bool = Field(
        default=False,
        description="Whether to log reconstruction visualizations to W&B",
    )

    reconstruction_log_interval: int = Field(
        default=50,
        description="Steps between reconstruction visualization logging",
    )


class SCDTrainingStrategy(TrainingStrategy):
    """SCD training strategy for LTX-2.

    Implements the Separable Causal Diffusion training paradigm from arXiv:2602.10095.

    SCD Paper §5.1 (Training Procedure) — Full training flow:
        1. Encoder pass: Feed CLEAN latents (sigma=0) through encoder layers with a
           frame-level causal attention mask. Frame t can only attend to frames <= t.
           This produces temporally-aware features WITHOUT noise contamination.
        2. Shift encoder output by 1 frame: Frame t's decoder receives frame (t-1)'s
           encoder features. Frame 0 gets zero features (no preceding context).
           This enforces causal conditioning — each frame is decoded using only
           past temporal context, never future information.
        3. Decoder pass: Feed NOISY latents (at sampled sigma) through decoder layers,
           conditioned on the shifted encoder features. The decoder predicts the
           velocity v = epsilon - x_0 for flow matching.
        4. Loss: Masked MSE between predicted and target velocity, excluding any
           conditioning tokens (e.g., first frame when I2V conditioning is active).
        5. Gradients flow through BOTH encoder and decoder — the entire pipeline is
           trained end-to-end. The encoder is NOT frozen or pre-trained separately.
    """

    config: SCDTrainingConfig

    def __init__(self, config: SCDTrainingConfig):
        super().__init__(config)
        # SCD Paper §5.1: The SCD model wrapper splits the base transformer into
        # encoder/decoder halves. It is set by the trainer after model creation.
        # When None, we fall back to standard (non-SCD) training as a single pass.
        self._scd_model: LTXSCDModel | None = None
        # Current training step — updated by the trainer each step for scheduled sampling.
        self._current_step: int = 0

    def set_scd_model(self, model: LTXSCDModel) -> None:
        """Set the SCD model wrapper. Called by the trainer after model creation."""
        self._scd_model = model

    def set_current_step(self, step: int) -> None:
        """Set the current training step (called by trainer each step)."""
        self._current_step = step

    def _get_p_ar(self) -> float:
        """Get current probability of using AR predictions based on curriculum.

        The curriculum has three phases:
        1. Warmup (step < ss_warmup_steps): p_ar = 0 (pure teacher forcing)
        2. Ramp (warmup <= step < warmup + ramp): linear interpolation start → end
        3. Plateau (step >= warmup + ramp): p_ar = ss_p_ar_end (max AR exposure)
        """
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
        """Prepare SCD training inputs with split encoder/decoder passes.

        SCD Paper §5.1 (Training Procedure) — Step-by-step:

        Step 1: Patchify clean latents [B, C, F, H, W] -> [B, seq_len, C]
            The VAE-encoded video latents are flattened into a token sequence.
            For LTX-2 with patch_size=1, seq_len = num_frames * height * width.

        Step 2: Build frame-level causal mask for encoder
            SCD Paper §5.1: The encoder uses a frame-level causal mask so that
            frame t can only attend to frames <= t. This is crucial — it forces
            the encoder to build temporally-causal representations that can be
            used autoregressively at inference time.

        Step 3: Encoder pass with timestep=0 (clean signal)
            SCD Paper §5.1: "Encoder sees clean latents (sigma=0) with causal mask."
            The encoder always receives noise-free latents so it learns to extract
            pure temporal features, independent of the noise level. This is what
            makes the encoder's output reusable across denoising steps.

        Step 4: Shift encoder features by 1 frame (frame t-1 -> frame t)
            The causal shift ensures the decoder for frame t only sees encoder
            context from frame t-1 and earlier, never from the current or future
            frames. Frame 0's decoder gets zero context (bootstrap).

        Step 5: Decoder pass with actual sigma + shifted encoder features -> velocity
            SCD Paper §5.1: "Decoder sees noisy latents at sampled sigma."
            The decoder receives the noisy interpolation x_t = (1-sigma)*x_0 + sigma*noise
            along with the shifted encoder features, and predicts the velocity target.

        If the SCD model wrapper is not set, falls back to standard training
        (encoder + decoder as a single pass through the full model).
        """
        # ===================================================================
        # Step 1: Extract and patchify clean video latents
        # ===================================================================
        # Get pre-encoded latents [B, C, F, H, W] from the precomputed dataset.
        # These are VAE-encoded, noise-free ground truth video latents (x_0).
        latents = batch["latents"]
        video_latents = latents["latents"]

        # Use actual latent tensor shape, NOT metadata num_frames which may
        # contain the raw video frame count (e.g. 25) instead of latent frames (e.g. 4).
        # video_latents shape: [B, C, F_lat, H_lat, W_lat]
        num_frames = video_latents.shape[2]
        height = video_latents.shape[3]
        width = video_latents.shape[4]

        # Patchify: [B, C, F, H, W] -> [B, seq_len, C]
        # With patch_size=1, this flattens spatial+temporal dims into a token sequence.
        # seq_len = num_frames * height * width; C = 128 (latent channels).
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS metadata (used for temporal position embeddings)
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}, using first: {fps[0].item()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (precomputed from Gemma text encoder)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # tokens_per_frame = height * width (number of spatial tokens in one frame)
        # This is used to build the frame-level causal mask and for the 1-frame shift.
        tokens_per_frame = video_seq_len // num_frames

        # ===================================================================
        # Step 2: Create first-frame conditioning mask (I2V mode)
        # ===================================================================
        # SCD Paper §5.3 (First Frame Conditioning):
        # With probability first_frame_conditioning_p (default 0.1), the first frame's
        # tokens are marked as "conditioning" — they keep sigma=0 (clean) even in the
        # decoder pass. This trains the model for image-to-video generation where the
        # first frame is given and subsequent frames are generated.
        #
        # The conditioning_mask is a boolean tensor [B, seq_len] where True means
        # "this token is a conditioning token (keep clean, exclude from loss)."
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # ===================================================================
        # Step 3: Sample noise and create noisy latents for the decoder
        # ===================================================================
        # SCD Paper §5.2 (Velocity Prediction) — Flow matching formulation:
        # Sample sigma from the timestep distribution (e.g., logit-normal).
        # Sample Gaussian noise epsilon ~ N(0, I).
        # Construct noisy latent: x_t = (1 - sigma) * x_0 + sigma * epsilon
        #
        # At sigma=0, x_t = x_0 (clean). At sigma=1, x_t = epsilon (pure noise).
        # The interpolation continuously blends between clean and noise.
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * epsilon
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # SCD Paper §5.3: Conditioning tokens (first frame when active) use CLEAN
        # latents even in the decoder — their sigma is effectively 0. This means
        # the decoder sees the ground truth first frame and must generate the rest.
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # SCD Paper §5.2 (Velocity Prediction):
        # The training target is the velocity: v = epsilon - x_0
        # This is preferred over epsilon-prediction for flow matching because:
        # (a) It provides more stable gradients across different noise levels,
        # (b) At sigma=0, the target is well-defined (unlike epsilon-prediction),
        # (c) Clean recovery: x_0_hat = epsilon - v_hat (subtract predicted velocity from noise).
        video_targets = video_noise - video_latents

        # ===================================================================
        # Step 4: Generate position embeddings
        # ===================================================================
        # Position embeddings encode (time, height, width) for RoPE.
        # Temporal positions are scaled by 1/fps to convert frame indices to seconds.
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # ===================================================================
        # Step 5: SCD split encoder/decoder OR standard fallback
        # ===================================================================
        # SCD Paper §5.1: When the SCD model wrapper is available, we split the
        # forward pass into separate encoder and decoder phases. The encoder runs
        # here (inside prepare_training_inputs), and the decoder runs during the
        # trainer's forward pass. This is because the encoder output must be
        # shifted and attached to the ModelInputs before the decoder can run.
        if self._scd_model is not None:
            return self._prepare_scd_inputs(
                video_latents=video_latents,
                noisy_video=noisy_video,
                video_noise=video_noise,
                video_targets=video_targets,
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

        # === Fallback: Standard training (no SCD split) ===
        # When the SCD model is not set, we run the full transformer as a single
        # pass. This is useful for debugging or comparison with non-SCD baselines.
        # Per-token timesteps: conditioning tokens get sigma=0, target tokens get
        # the sampled sigma value.
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

        # SCD Paper §5.4 (Loss Masking): Loss is computed only on non-conditioning
        # tokens. Conditioning frames (first frame when I2V mode is active) are
        # excluded from the loss via this mask.
        video_loss_mask = ~video_conditioning_mask

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            shared_noise=video_noise,
            shared_sigmas=sigmas,
        )

    def _prepare_scd_inputs(
        self,
        video_latents: Tensor,
        noisy_video: Tensor,
        video_noise: Tensor,
        video_targets: Tensor,
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
        """Prepare inputs with SCD encoder/decoder split.

        SCD Paper §5.1 (Training Procedure):
            This method implements the core SCD training logic. It runs the encoder
            pass internally (within the strategy's prepare step) and packages the
            shifted encoder features into ModelInputs so that the trainer's forward
            pass only needs to run the decoder. This two-phase approach ensures:

            1. The encoder sees CLEAN latents with sigma=0 and causal mask
            2. Encoder features are shifted by 1 frame before reaching the decoder
            3. The decoder sees NOISY latents with the actual sampled sigma
            4. Gradients flow end-to-end through both encoder and decoder

        SCD Paper §5.1: "The model is trained end-to-end with both encoder and decoder.
            Gradients flow through BOTH encoder and decoder."
            This is critical — the encoder is NOT frozen. It learns jointly with the
            decoder to produce maximally useful temporal features for denoising.
        """
        # =================================================================
        # ENCODER PASS
        # =================================================================
        # SCD Paper §5.1: "Encoder sees clean latents (sigma=0) with causal mask."
        #
        # The encoder always receives the CLEAN (noise-free) video latents x_0.
        # All tokens get timestep=0 (sigma=0), telling the model there is no noise.
        # This is fundamentally different from the decoder, which gets noisy latents.
        #
        # Why clean? Because the encoder's job is to extract TEMPORAL CONTEXT, not
        # to denoise. By always seeing clean data, the encoder learns a stable
        # representation of the video's temporal structure that can be computed
        # once and reused across all denoising steps at inference time.
        encoder_timesteps = torch.zeros(
            batch_size, video_seq_len, device=device, dtype=dtype
        )

        encoder_modality = Modality(
            enabled=True,
            sigma=torch.zeros(batch_size, device=device, dtype=dtype),
            latent=video_latents,  # Clean latents x_0 for encoder (NOT noisy)
            timesteps=encoder_timesteps,  # All zeros: sigma=0 means "clean signal"
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # SCD Paper §5.1: Run encoder with frame-level causal attention mask.
        # The causal mask (built inside forward_encoder via build_frame_causal_mask)
        # ensures frame t can only attend to frames <= t. This is essential for:
        # (a) Training: Prevents information leakage from future frames
        # (b) Inference: Enables autoregressive frame-by-frame generation
        #     with KV-caching — only the new frame needs to be processed
        #
        # tokens_per_frame tells the model how to construct the frame-level mask:
        # tokens are grouped into frames of size (height * width), and the mask
        # allows bidirectional attention within a frame but only causal (past->future)
        # attention across frames.
        encoder_video_args, encoder_audio_args = self._scd_model.forward_encoder(
            video=encoder_modality,
            audio=None,
            perturbations=None,
            tokens_per_frame=tokens_per_frame,
        )

        # =================================================================
        # SHIFT ENCODER FEATURES BY 1 FRAME (Causal Conditioning)
        # =================================================================
        # SCD Paper §5.1: The encoder output for frame t-1 is used as context for
        # decoding frame t. This 1-frame shift is the mechanism by which the decoder
        # receives temporal context:
        #   - Frame 0's decoder gets ZERO features (no preceding context — bootstrap)
        #   - Frame 1's decoder gets frame 0's encoder features
        #   - Frame t's decoder gets frame (t-1)'s encoder features
        #
        # CRITICAL: Do NOT detach encoder_features here!
        # SCD Paper §5.1: "Gradients flow through BOTH encoder and decoder."
        # The encoder must receive gradients from the decoder's loss to learn
        # what temporal features are most useful for denoising. Detaching would
        # break the end-to-end training and degrade temporal coherence.
        encoder_features = encoder_video_args.x  # [B, seq_len, D]
        shifted_features = shift_encoder_features(
            encoder_features, tokens_per_frame, num_frames
        )

        # =================================================================
        # AR-AWARE SCHEDULED SAMPLING
        # =================================================================
        # When enabled, probabilistically replace GT encoder input for some frames
        # with the model's own x̂₀ predictions, closing the train/inference gap.
        # This runs under no_grad() — the gradient signal comes from the normal
        # decoder pass that follows, which now receives AR-contaminated features.
        p_ar = self._get_p_ar()
        if p_ar > 0.0 and num_frames > 1:
            shifted_features = self._apply_scheduled_sampling(
                video_latents=video_latents,
                shifted_features=shifted_features,
                video_positions=video_positions,
                video_prompt_embeds=video_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigmas=sigmas,
                video_noise=video_noise,
                tokens_per_frame=tokens_per_frame,
                num_frames=num_frames,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                p_ar=p_ar,
            )

        # =================================================================
        # DECODER PASS SETUP
        # =================================================================
        # SCD Paper §5.1: "Decoder sees noisy latents at sampled sigma."
        #
        # The decoder receives:
        # (a) Noisy latents x_t = (1-sigma)*x_0 + sigma*epsilon (the input to denoise)
        # (b) Shifted encoder features from frame t-1 (temporal context)
        # (c) Text embeddings (semantic context from the prompt)
        #
        # Per-token timesteps: conditioning tokens (first frame in I2V mode) get
        # sigma=0; all other tokens get the sampled sigma value.
        # SCD Paper §5.3: The conditioning mask ensures first-frame tokens are
        # treated as clean by the decoder when I2V conditioning is active.
        decoder_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

        # Create decoder modality with noisy latents.
        # The SCD model's forward_decoder will handle combining these noisy tokens
        # with the shifted encoder features using the configured combine mode
        # (token_concat by default — SCD Paper §5.5).
        decoder_modality = Modality(
            enabled=True,
            sigma=sigmas.squeeze(),
            latent=noisy_video,  # Noisy latents x_t for decoder (NOT clean)
            timesteps=decoder_timesteps,  # Per-token: 0 for conditioning, sigma for targets
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # SCD Paper §5.4 (Loss Masking): Loss is only computed on non-conditioning
        # tokens. When first_frame_conditioning is active, the first frame's tokens
        # are excluded from the loss because they were given as clean input —
        # predicting velocity for them would be trivial (target = noise - clean,
        # but the model sees clean, so it would just learn to output noise).
        video_loss_mask = ~video_conditioning_mask

        # Package everything into ModelInputs for the trainer's forward pass.
        # The decoder modality goes into video= (the trainer calls model.forward_decoder
        # with these inputs). The shifted encoder features are attached as private
        # attributes so the trainer can pass them to forward_decoder.
        model_inputs = ModelInputs(
            video=decoder_modality,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            shared_noise=video_noise,
            shared_sigmas=sigmas,
        )

        # Attach SCD-specific data for the trainer's forward pass.
        # These are accessed by the trainer to call scd_model.forward_decoder()
        # with the correct encoder features.
        #
        # SCD Paper §5.1: encoder_features are NOT detached — gradients flow back
        # through the encoder via shifted_features -> encoder_features -> encoder weights.
        # This end-to-end gradient flow is what makes SCD training work.
        model_inputs._encoder_features = shifted_features
        model_inputs._scd_model = self._scd_model
        model_inputs._encoder_audio_args = encoder_audio_args

        # Per-frame decoder metadata: when per_frame_decoder is True, the trainer calls
        # forward_decoder_per_frame() which processes each frame independently through
        # the decoder blocks, matching the autoregressive inference setup (1 frame = 336 tokens).
        # This prevents the train/inference attention scope mismatch that causes grid artifacts.
        model_inputs._per_frame_decoder = self.config.per_frame_decoder
        model_inputs._tokens_per_frame = tokens_per_frame
        model_inputs._num_frames = num_frames

        # Store raw latent shape for reconstruction visualization (unpatchification).
        # This is the pre-patchified [B, C, F, H, W] tensor needed to reshape
        # the patchified velocity prediction back to spatial form for VAE decoding.
        model_inputs._raw_video_latents = batch["latents"]["latents"]  # [B, C, F, H, W]

        return model_inputs

    @torch.no_grad()
    def _apply_scheduled_sampling(
        self,
        video_latents: Tensor,
        shifted_features: Tensor,
        video_positions: Tensor,
        video_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        sigmas: Tensor,
        video_noise: Tensor,
        tokens_per_frame: int,
        num_frames: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        p_ar: float,
    ) -> Tensor:
        """Replace GT encoder features with model-predicted features for random frames.

        Simulates autoregressive inference during training: for each frame (except
        frame 0), flip a coin with probability p_ar. If heads:
          1. Run a single-step decoder pass on the predecessor frame to get v̂
          2. Recover x̂₀ = ε - v̂ (flow matching identity)
          3. Re-encode x̂₀ through the encoder to get "AR-like" features
          4. Replace the target frame's encoder features with these AR features

        The entire operation runs under @torch.no_grad() — we don't backprop through
        the AR simulation. The gradient signal comes from the normal decoder pass that
        follows, which receives AR-contaminated encoder features. This is analogous to
        how scheduled sampling works in seq2seq: the forward pass uses mixed GT/predicted
        inputs, but the loss gradient flows through the final prediction only.

        Args:
            video_latents: Clean patchified latents [B, seq_len, C]
            shifted_features: GT encoder features (already shifted) [B, seq_len, D]
            video_positions: Position embeddings [B, 3, seq_len, 2]
            video_prompt_embeds: Text embeddings
            prompt_attention_mask: Text attention mask
            sigmas: Sampled noise levels [B, 1]
            video_noise: Gaussian noise [B, seq_len, C]
            tokens_per_frame: Spatial tokens per frame (H * W)
            num_frames: Number of video frames
            batch_size: Batch size
            device: Compute device
            dtype: Compute dtype
            p_ar: Current probability of AR replacement

        Returns:
            Modified shifted_features with some frames' features replaced by AR predictions
        """
        tpf = tokens_per_frame
        result = shifted_features.clone()

        # Determine which frames get AR replacement (skip frame 0 — no predecessor)
        ar_mask = torch.rand(num_frames - 1) < p_ar

        if not ar_mask.any():
            return result

        # Build noisy video: x_t = (1 - σ) * x₀ + σ * ε
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For each selected frame, denoise its predecessor then re-encode
        for f_idx in range(num_frames - 1):
            if not ar_mask[f_idx]:
                continue

            pred_f = f_idx      # Predecessor frame to denoise
            tgt_f = f_idx + 1   # Target frame that receives AR features

            # Extract predecessor frame's tokens
            start = pred_f * tpf
            end = start + tpf

            frame_noisy = noisy_video[:, start:end, :]    # [B, tpf, C]
            frame_enc = shifted_features[:, start:end, :]  # [B, tpf, D]
            frame_noise = video_noise[:, start:end, :]     # [B, tpf, C]

            # Build single-frame modality for decoder
            frame_positions = video_positions[:, :, start:end, :]  # [B, 3, tpf, 2]
            frame_timesteps = sigmas.squeeze().expand(batch_size, tpf)  # [B, tpf]

            frame_modality = Modality(
                enabled=True,
                sigma=sigmas.squeeze(),
                latent=frame_noisy,
                timesteps=frame_timesteps,
                positions=frame_positions,
                context=video_prompt_embeds,
                context_mask=prompt_attention_mask,
            )

            # Single-step decoder to get velocity prediction
            v_hat, _ = self._scd_model.forward_decoder_per_frame(
                video=frame_modality,
                encoder_features=frame_enc,
                perturbations=None,
                tokens_per_frame=tpf,
                num_frames=1,
            )

            # Recover x̂₀ from velocity: flow matching says v = ε - x₀, so x₀ = ε - v̂
            x0_hat = frame_noise - v_hat  # [B, tpf, C]

            # Optional noise augmentation — simulates imperfect inference predictions
            if self.config.ss_noise_augment > 0:
                aug_noise = torch.randn_like(x0_hat) * self.config.ss_noise_augment
                x0_hat = x0_hat + aug_noise

            # Re-encode x̂₀ through the encoder (single frame, clean σ=0)
            enc_timesteps = torch.zeros(batch_size, tpf, device=device, dtype=dtype)
            enc_modality = Modality(
                enabled=True,
                sigma=torch.zeros(batch_size, device=device, dtype=dtype),
                latent=x0_hat,
                timesteps=enc_timesteps,
                positions=frame_positions,
                context=video_prompt_embeds,
                context_mask=prompt_attention_mask,
            )

            enc_video_args, _ = self._scd_model.forward_encoder(
                video=enc_modality,
                audio=None,
                perturbations=None,
                tokens_per_frame=tpf,
            )

            # Replace the target frame's encoder features with AR-derived features
            # The shift is already applied — frame tgt_f's features in shifted_features
            # correspond to frame pred_f's encoder output (which we just re-computed)
            tgt_start = tgt_f * tpf
            tgt_end = tgt_start + tpf
            result[:, tgt_start:tgt_end, :] = enc_video_args.x.detach()

        return result

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for SCD training.

        SCD Paper §5.2 (Velocity Prediction):
            Target: v = epsilon - x_0 (velocity = noise - clean)
            Loss: MSE(v_hat, v) where v_hat is the decoder's output.

            The velocity formulation is used because it provides more stable
            gradients than epsilon-prediction for flow matching. At sigma near 0,
            epsilon-prediction has vanishing signal (the clean sample dominates),
            while velocity prediction remains well-conditioned across all sigma values.

        SCD Paper §5.4 (Loss Masking):
            Loss is computed ONLY on non-conditioning tokens. When first-frame
            conditioning is active (I2V mode), the first frame's tokens are
            excluded from the loss via the video_loss_mask. The loss is
            MSE normalized by the number of active (non-masked) tokens:

                L = sum(mask * (v_hat - v)^2) / sum(mask)

            This normalization ensures the effective loss magnitude is independent
            of how many tokens are masked (e.g., whether I2V conditioning is on/off),
            maintaining stable training dynamics.
        """
        # Video loss: masked MSE between predicted and target velocity
        # video_pred: [B, seq_len, C] — decoder's velocity prediction v_hat
        # inputs.video_targets: [B, seq_len, C] — target velocity v = epsilon - x_0
        video_loss = (video_pred - inputs.video_targets).pow(2)

        # SCD Paper §5.4: Apply loss mask — True means "compute loss here."
        # Conditioning tokens (first frame in I2V mode) are False (excluded).
        # Normalize by the fraction of active tokens to keep loss scale consistent.
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # Audio loss if enabled (standard MSE, no masking)
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()
        return video_loss + audio_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include SCD-specific metadata in checkpoints.

        This metadata is needed at inference time to reconstruct the SCD model
        wrapper with the correct encoder/decoder split configuration.
        """
        return {
            "scd_encoder_layers": self.config.encoder_layers,
            "scd_decoder_input_combine": self.config.decoder_input_combine,
        }

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction visualizations to W&B.

        Decodes the velocity prediction back to clean latent, then VAE-decodes
        both ground truth and prediction to pixel space for side-by-side comparison.

        SCD Paper §5.2 (Velocity Prediction) — Recovery formula:
            Given the velocity prediction v_hat and the known noise epsilon:
                x_0_hat = epsilon - v_hat
            This recovers the predicted clean latent from the velocity output.
            We then pass both x_0 (ground truth) and x_0_hat (prediction) through
            the VAE decoder to visualize reconstruction quality in pixel space.

        Args:
            video_pred: Model velocity prediction v_hat [B, seq_len, C]
            inputs: ModelInputs with raw latents and noise info
            step: Current training step
            vae_decoder: VAE decoder for pixel-space visualization
            prefix: W&B metric prefix

        Returns:
            Dictionary of logged metrics
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return {}

        if not self.config.log_reconstructions:
            return {}

        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is None:
            logger.warning("No raw latents stored for reconstruction")
            return {}

        b, c, f, h, w = raw_latents.shape

        # SCD Paper §5.2: Recover predicted clean latent from velocity prediction.
        # Flow matching: v = epsilon - x_0, therefore x_0_hat = epsilon - v_hat
        # noise is the known epsilon used to construct x_t during training.
        noise = inputs.shared_noise  # [B, seq_len, C] (patchified)
        pred_clean = noise - video_pred  # [B, seq_len, C]

        # Unpatchify: [B, seq_len, C] -> [B, C, F, H, W]
        # With patch_size=1, seq_len = F * H * W and C matches the latent channels.
        # We reshape back to the spatial form for VAE decoding.
        pred_clean_spatial = pred_clean.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
        gt_latents = raw_latents  # Already [B, C, F, H, W]

        # Decode to pixel space using the VAE decoder
        if vae_decoder is not None:
            try:
                decoder_device = next(vae_decoder.parameters()).device
                decoder_dtype = next(vae_decoder.parameters()).dtype

                with torch.inference_mode():
                    gt_decoded = vae_decoder(
                        gt_latents[:1].to(device=decoder_device, dtype=decoder_dtype)
                    )
                    pred_decoded = vae_decoder(
                        pred_clean_spatial[:1].to(device=decoder_device, dtype=decoder_dtype)
                    )

                # Clamp to valid range and convert to [0, 1] for visualization
                gt_decoded = gt_decoded.float().clamp(-1, 1) * 0.5 + 0.5
                pred_decoded = pred_decoded.float().clamp(-1, 1) * 0.5 + 0.5

                # Take middle frame for visualization
                mid_f = gt_decoded.shape[2] // 2
                gt_frame = gt_decoded[0, :, mid_f].cpu()   # [C, H, W]
                pred_frame = pred_decoded[0, :, mid_f].cpu()

                # Create side-by-side grid: Ground Truth | Prediction
                import torchvision.utils as vutils
                grid = vutils.make_grid([gt_frame, pred_frame], nrow=2, padding=4)

                log_dict = {
                    f"{prefix}/reconstruction": wandb.Image(
                        grid.permute(1, 2, 0).numpy(),
                        caption=f"Step {step} | Left: Ground Truth | Right: Prediction",
                    ),
                }

                # Also log first and last frames if multiple frames available
                if gt_decoded.shape[2] > 1:
                    gt_first = gt_decoded[0, :, 0].cpu()
                    pred_first = pred_decoded[0, :, 0].cpu()
                    gt_last = gt_decoded[0, :, -1].cpu()
                    pred_last = pred_decoded[0, :, -1].cpu()

                    grid_first = vutils.make_grid([gt_first, pred_first], nrow=2, padding=4)
                    grid_last = vutils.make_grid([gt_last, pred_last], nrow=2, padding=4)

                    log_dict[f"{prefix}/reconstruction_first_frame"] = wandb.Image(
                        grid_first.permute(1, 2, 0).numpy(),
                        caption=f"Step {step} | First frame | GT vs Pred",
                    )
                    log_dict[f"{prefix}/reconstruction_last_frame"] = wandb.Image(
                        grid_last.permute(1, 2, 0).numpy(),
                        caption=f"Step {step} | Last frame | GT vs Pred",
                    )

                wandb.log(log_dict, step=step)
                logger.debug(f"Logged SCD reconstruction images at step {step}")
                return log_dict

            except Exception as e:
                logger.warning(f"Failed to decode reconstruction: {e}")
                # Fall through to latent-space visualization

        # Fallback: latent-space visualization (pseudo-RGB from first 3 channels)
        # When no VAE decoder is available, we visualize the raw latent space
        # by treating the first 3 latent channels as pseudo-RGB values.
        mid_f = f // 2
        gt_vis = raw_latents[0, :3, mid_f].cpu().float()
        pred_vis = pred_clean_spatial[0, :3, mid_f].cpu().float()

        # Normalize to [0, 1] for display
        def normalize(x):
            x = x - x.min()
            return x / (x.max() + 1e-8)

        import torchvision.utils as vutils
        grid = vutils.make_grid([normalize(gt_vis), normalize(pred_vis)], nrow=2, padding=4)

        log_dict = {
            f"{prefix}/reconstruction_latent": wandb.Image(
                grid.permute(1, 2, 0).numpy(),
                caption=f"Step {step} | Latent space (pseudo-RGB) | GT vs Pred",
            ),
        }
        wandb.log(log_dict, step=step)
        return log_dict
