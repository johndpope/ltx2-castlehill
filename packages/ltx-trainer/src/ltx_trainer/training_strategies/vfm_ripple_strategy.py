"""VFM Ripple — VFM v1b noise adapter with joint audio-video support.

Combines:
  1. VFM v1b noise adapter: temporally-aware adapter with self-attention +
     cross-attention to full text embeddings + positional encoding. Each token
     gets an independent (μ_i, σ_i) so noise is spatially/temporally structured.
  2. Joint audio-video generation: audio modality added for talking-head datasets
     (e.g. scrya) where audio and video are generated jointly.
  3. Tiled VAE reconstruction logging: prevents OOM on 145-frame 544×544 videos.
  4. Bootstrap consistency distillation (optional): samples a second noise z₂ from
     the same adapter distribution, denoises it in one step (no_grad), and forces
     x̂₀(z₁) ≈ x̂₀(z₂). This trains the flow map to be variance-reducing — any
     sample from the adapter's distribution should decode to the same video.

Architecture note:
  The ripple sparse attention kernel is transparent to this strategy — it is
  activated by self_attn_type: "ripple" in the model config, which causes
  the trainer's _swap_attention_modules() to replace all Attention blocks with
  RippleVideoAttention before training starts. This strategy doesn't touch
  attention implementation at all.

Training signal:
  L = L_VFM(video) + λ_audio · L_MSE(audio) + λ_distill · L_distill
  where L_VFM = (1/2τ²)·L_MF + (1/2σ²)·L_obs + w_KL·L_KL (inherited from v1b)
  and   L_distill = MSE(x̂₀(z₁), x̂₀(z₂)) — bootstrap consistency
"""

import random
from collections import deque
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
)
from ltx_trainer.training_strategies.vfm_strategy_v1b import (
    VFMv1bTrainingConfig,
    VFMv1bTrainingStrategy,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class NoiseReplayBuffer:
    """Ring buffer of past adapter distributions for KL consistency replay.

    Stores (text_embeds, positions, mu, log_sigma) in fp16 on CPU so the
    adapter can be re-run on stale conditioning and the drift penalised.
    No video latents needed — the adapter takes text+positions, not pixels.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        text_embeds: Tensor,
        text_mask: Tensor | None,
        positions: Tensor,
        task_class: Tensor | None,
        mu: Tensor,
        log_sigma: Tensor,
    ) -> None:
        self._buf.append((
            text_embeds.detach().half().cpu(),
            text_mask.detach().cpu() if text_mask is not None else None,
            positions.detach().half().cpu(),
            task_class.detach().cpu() if task_class is not None else None,
            mu.detach().half().cpu(),
            log_sigma.detach().half().cpu(),
        ))

    def sample(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple | None:
        if not self._buf:
            return None
        te, tm, pos, tc, mu, ls = self._buf[random.randrange(len(self._buf))]
        return (
            te.to(device=device, dtype=dtype),
            tm.to(device=device) if tm is not None else None,
            pos.to(device=device, dtype=dtype),
            tc.to(device=device) if tc is not None else None,
            mu.to(device=device, dtype=dtype),
            ls.to(device=device, dtype=dtype),
        )

    def __len__(self) -> int:
        return len(self._buf)


class VFMRippleConfig(VFMv1bTrainingConfig):
    """Configuration for VFM Ripple (v1b adapter + audio-video)."""

    name: Literal["vfm_ripple"] = "vfm_ripple"

    # Audio support — default True because ripple targets scrya (audio-video dataset)
    with_audio: bool = Field(
        default=True,
        description="Joint audio-video generation (required for scrya dataset)",
    )
    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for pre-encoded audio latents",
    )
    audio_loss_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for audio MSE loss relative to VFM video loss",
    )

    # Reconstruction logging — tiled decode to avoid OOM on 145-frame videos
    log_reconstructions: bool = Field(
        default=False,
        description="Decode and log source|predict|GT triplet to W&B",
    )
    reconstruction_log_interval: int = Field(
        default=50,
        description="Log reconstructions every N optimizer steps",
    )
    recon_log_height: int = Field(
        default=128,
        ge=32,
        description=(
            "Resize decoded frames to this height (px) before logging. "
            "Width is scaled proportionally. 128px keeps W&B uploads under ~1MB. "
            "Use 256 if you need more detail. Full 544px = ~80MB per log — avoid."
        ),
    )
    recon_log_max_frames: int = Field(
        default=24,
        ge=1,
        description=(
            "Maximum number of frames to include in the W&B video strip. "
            "Frames are sampled evenly across the full clip. "
            "24 frames at 8fps = 3s — enough to see motion without large files."
        ),
    )
    recon_log_video: bool = Field(
        default=False,
        description=(
            "Log a full video strip (Noisy | Pred | GT) in addition to the mid-frame image. "
            "False by default — the image grid is sufficient for monitoring and avoids "
            "duplicate media panels when multiple runs are visible in the W&B workspace. "
            "Set true only when you need to inspect temporal motion."
        ),
    )

    # Bootstrap consistency distillation
    self_distill_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight for bootstrap consistency distillation loss. "
            "Samples z₂ ~ q_φ(z|text) independently, denoises in 1 step (no_grad), "
            "and minimises MSE(x̂₀(z₁), x̂₀(z₂)). Trains the flow map to be "
            "variance-reducing — any noise from the adapter should decode to the same "
            "video. 0 = disabled (default). Good starting range: 0.1–0.5."
        ),
    )
    self_distill_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of applying distillation loss per step. "
            "50% avoids doubling compute every step while keeping the signal frequent."
        ),
    )
    self_distill_warmup_steps: int = Field(
        default=100,
        ge=0,
        description=(
            "Linearly ramp self_distill_weight from 0 → full over this many steps. "
            "Prevents a loss spike when distillation is enabled on a partially-trained adapter "
            "whose teacher reconstructions are still noisy. 0 = no ramp (full weight immediately)."
        ),
    )

    # Noise replay buffer
    replay_buffer_size: int = Field(
        default=0,
        ge=0,
        description=(
            "Capacity of the noise replay buffer (0 = disabled). "
            "Stores past adapter distributions (text_embeds, positions, μ, log_σ) in "
            "fp16 on CPU and replays them as a KL consistency penalty. For overfit runs "
            "with 54 clips, 128–256 is plenty. For full training, 2048–8192."
        ),
    )
    replay_prob: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of applying a replay step per optimizer update.",
    )
    replay_kl_weight: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Weight for KL(q_new || q_old) replay loss. "
            "Penalises the adapter drifting away from past distributions. "
            "Keep low (0.05–0.2) so it regularises rather than freezes."
        ),
    )


class VFMRippleStrategy(VFMv1bTrainingStrategy):
    """VFM v1b + joint audio-video for ripple attention training.

    Inherits all VFM v1b logic (temporally-aware adapter, three-part loss,
    EMA, flow-map freezing) and adds:
      - Audio modality preparation (same sigma as video, no conditioning mask)
      - Audio MSE loss term
      - Bootstrap consistency distillation (optional): z₁ vs z₂ from same adapter
      - Tiled VAE decode for reconstruction logging (no OOM on long videos)
    """

    config: VFMRippleConfig

    def __init__(self, config: VFMRippleConfig):
        super().__init__(config)
        self._transformer_ref: nn.Module | None = None
        self._glossary_logged: bool = False
        self._replay_buffer: NoiseReplayBuffer | None = (
            NoiseReplayBuffer(config.replay_buffer_size)
            if config.replay_buffer_size > 0
            else None
        )
        if self._replay_buffer is not None:
            logger.info(
                f"VFM Ripple: noise replay buffer enabled (capacity={config.replay_buffer_size}, "
                f"prob={config.replay_prob}, kl_weight={config.replay_kl_weight})"
            )

    def set_transformer(self, transformer: nn.Module, grad_accumulation_steps: int = 1) -> None:
        """Store transformer ref for bootstrap distillation teacher pass."""
        self._transformer_ref = transformer
        if self.config.self_distill_weight > 0:
            logger.info("VFM Ripple: transformer ref set — bootstrap distillation enabled")

    def _log_metric_glossary(self) -> None:
        """Log a metric reference table + run notes to W&B at training start.

        Creates two artefacts in W&B:
        1. A browsable Table (metric_glossary panel) — every metric explained.
        2. Run notes (overview page) — narrative description of the training setup.
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        # ── Metric glossary table ────────────────────────────────────────────
        # Each row: (metric key, what it measures, healthy trend, diagnostic tip)
        rows = [
            # Core VFM losses (from vfm_strategy.py parent)
            ("vfm/loss_mf",
             "Flow matching MSE: ||v_pred − v_target||². Primary learning signal.",
             "Decreasing",
             "If stuck above 0.5 after 200 steps, adapter is not learning — check lr and mu_std."),
            ("vfm/loss_obs",
             "Observation loss: ||x̂₀ − (x₀+ε)||² with small ε. Forces adapter noise to be near GT.",
             "Decreasing, should be ~10× smaller than loss_mf",
             "If dominant, obs_noise_level is too high or obs_loss_weight too large."),
            ("vfm/loss_kl",
             "KL(q_φ(z|text) ‖ N(0,I)). Keeps adapter distribution from collapsing or exploding.",
             "Stable low plateau (0.1–2.0)",
             "If >10 it dominates training — reduce kl_weight. If 0 the adapter collapsed to prior."),
            ("vfm/loss_total",
             "Sum of all active loss terms for this step.",
             "Decreasing overall",
             "Watch for sudden spikes — usually means a gradient exploded."),
            ("vfm/use_adapter",
             "1 if adapter noise was used this step, 0 if standard N(0,I) (per alpha schedule).",
             "Matches config alpha (~0.8)",
             "If stuck at 0, the alpha schedule never activates — check freeze_flow_map_steps."),
            ("vfm/sigma",
             "Mean noise level σ sampled this step (shifted log-normal distribution).",
             "Random, centred ~0.7",
             "If always near 0 or 1, the timestep sampler is misconfigured."),
            ("vfm/snr",
             "Signal-to-noise ratio = (1−σ)/σ. High SNR = low noise = fine detail steps.",
             "Varies with sigma",
             "Used by min-SNR gamma weighting to balance easy vs hard timesteps."),
            ("vfm/snr_weight",
             "min(min_snr_gamma, SNR) / SNR. Downweights high-SNR (easy) steps.",
             "Varies; never above 1.0",
             "Flat at 1.0 means snr_weight is inactive — raise min_snr_gamma."),
            ("vfm/adapter_mu_mean",
             "Mean value of adapter μ across all tokens and batch.",
             "Near 0, stable",
             "Drifting far from 0 means the adapter is learning a DC bias — check obs loss."),
            ("vfm/adapter_mu_std",
             "Std of adapter μ across tokens. Measures spatial/temporal structure in noise.",
             "Increasing early, then stable (>0.1)",
             "Near 0 means the adapter outputs flat noise — it is not learning content structure."),
            ("vfm/adapter_sigma_mean",
             "Mean exp(log_σ) — the adapter's predicted noise scale per token.",
             "Near 1.0, slight variation",
             "Collapsing to 0 or exploding to 10+ indicates KL or reparameterisation issue."),
            # Audio
            ("vfm/audio_loss",
             "Audio velocity MSE: ||v_audio_pred − v_audio_target||². Joint audio-video signal.",
             "Decreasing alongside loss_mf",
             "If much higher than loss_mf, audio_loss_weight may need reducing."),
            # Bootstrap distillation
            ("vfm/distill_loss",
             "Bootstrap MSE: ||x̂₀(z₁) − x̂₀(z₂)||². Two independent z's from the same adapter.",
             "Decreasing",
             "High = the flow map is still sensitive to which z is drawn. Low = one-step is stable."),
            ("vfm/distill_cosine_sim",
             "Cosine similarity between x̂₀(z₁) and x̂₀(z₂). The clearest consistency signal.",
             "Rising toward 1.0",
             "Jump from ~0 to ~0.8 means the adapter found a text-aligned attractor. "
             "Stuck near 0 = distill_weight too low or adapter not yet structured."),
            ("vfm/distill_loss_norm",
             "distill_loss / ||x̂₀||². Scale-free fraction: what fraction of reconstruction "
             "energy varies across noise samples.",
             "Decreasing toward 0",
             "Useful to compare across resolutions/checkpoints where latent scale differs."),
            # Replay buffer
            ("vfm/replay_kl",
             "KL(q_new ‖ q_old): how far the adapter has drifted from a past distribution.",
             "Stabilising near 0 after initial rise",
             "Rising monotonically = adapter is still changing rapidly (normal early). "
             "Should level off once training converges."),
            ("vfm/replay_mu_drift",
             "||μ_new − μ_old||₂ per token: how far the adapter mean has moved.",
             "Decreasing over training",
             "High mu_drift + low sigma_drift = adapter is still searching for the right mean. "
             "Low both = converged."),
            ("vfm/replay_sigma_drift",
             "MAE of (log_σ_new − log_σ_old): how far the adapter variance has shifted.",
             "Decreasing over training",
             "Sudden spike in sigma_drift = adapter is recalibrating uncertainty — "
             "usually after a hard batch. If persistent, reduce replay_kl_weight."),
            ("vfm/replay_buffer_len",
             "Current number of items in the replay ring buffer.",
             "Rises linearly to capacity, then flat",
             "If stuck near 0, the adapter is not being invoked (use_adapter=0 every step)."),
        ]

        table = wandb.Table(
            columns=["Metric", "What It Measures", "Healthy Trend", "Diagnostic Tip"],
            data=rows,
        )
        wandb.log({"metric_glossary": table}, step=0)

        # ── Run notes (markdown, visible on the W&B run overview page) ──────
        cfg = self.config
        active_features = []
        if cfg.self_distill_weight > 0:
            active_features.append(
                f"**Bootstrap distillation** (weight={cfg.self_distill_weight}, "
                f"prob={cfg.self_distill_prob}): trains the flow map to produce the same "
                f"video regardless of which z is sampled. Watch `distill_cosine_sim` — "
                f"it should rise from ~0 to >0.8 as the adapter converges."
            )
        if cfg.replay_buffer_size > 0:
            active_features.append(
                f"**Noise replay buffer** (size={cfg.replay_buffer_size}, "
                f"prob={cfg.replay_prob}, kl_weight={cfg.replay_kl_weight}): "
                f"re-runs the adapter on past conditioning and penalises KL drift. "
                f"`replay_mu_drift` and `replay_sigma_drift` decompose *why* KL is high "
                f"(mean shift vs variance change)."
            )
        if cfg.with_audio:
            active_features.append(
                f"**Joint audio-video** (loss_weight={cfg.audio_loss_weight}): "
                f"audio and video share the same σ and are generated jointly. "
                f"`audio_loss` should track `loss_mf` in magnitude."
            )

        features_md = "\n\n".join(f"- {f}" for f in active_features) if active_features else "*(none active)*"

        wandb.run.notes = (
            f"## VFM Ripple — {cfg.adapter_hidden_dim}d × {cfg.adapter_num_layers}L adapter\n\n"
            f"Trains a VFM v1b noise adapter on top of a ripple-attention LoRA checkpoint. "
            f"The adapter maps text+position → per-token (μ, σ) structured noise. "
            f"One-step denoising: z ~ q_φ(z|text) → transformer → x̂₀.\n\n"
            f"### Key metric to watch\n"
            f"`vfm/loss_mf` (primary) + `vfm/adapter_mu_std` (adapter health). "
            f"If `adapter_mu_std` stays near 0, the adapter is not learning structure — "
            f"check that `use_adapter` is 1 and `loss_kl` is not dominant.\n\n"
            f"### Active features\n{features_md}\n\n"
            f"### Loss balance targets\n"
            f"- `loss_mf` : `loss_obs` : `loss_kl` ≈ 10 : 1 : 0.1\n"
            f"- `distill_loss_norm` < 0.1 = adapter is producing stable noise\n"
            f"- `replay_kl` < 1.0 = adapter not drifting between steps\n\n"
            f"### Metric reference\n"
            f"See the **metric_glossary** table panel for full descriptions of every metric."
        )

        logger.info("VFM Ripple: logged metric glossary + run notes to W&B")

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {"latents": "latents", "conditions": "conditions"}
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare VFM Ripple training inputs: VFM video + optional audio."""
        # ── 1. VFM video inputs (adapter noise, Modality, targets) ──────────
        inputs = super().prepare_training_inputs(batch, timestep_sampler)

        if not self.config.with_audio:
            return inputs

        # ── 2. Audio inputs (same sigma as video) ───────────────────────────
        conditions = batch["conditions"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        sigmas = inputs.shared_sigmas  # [B] — same noise level for both modalities
        batch_size = inputs.video.latent.shape[0]
        device = inputs.video.latent.device
        dtype = inputs.video.latent.dtype

        audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
            batch=batch,
            sigmas=sigmas,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Patch in-place (ModelInputs is a plain dataclass, not frozen)
        inputs.audio = audio_modality
        inputs.audio_targets = audio_targets
        inputs.audio_loss_mask = audio_loss_mask

        return inputs

    # ── Audio helpers (mirrors TextToVideoStrategy._prepare_audio_inputs) ───

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
        """Prepare audio modality: patchify → noise → targets → positions."""
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]  # [B, C, T, F]

        # Patchify: [B, C, T, F] → [B, T, C*F]
        audio_latents = self._audio_patchifier.patchify(audio_latents)
        audio_seq_len = audio_latents.shape[1]

        audio_noise = torch.randn_like(audio_latents)

        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        audio_targets = audio_noise - audio_latents  # velocity target

        # Per-token timesteps: all audio tokens use the sampled sigma
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """VFM three-part loss on video + audio MSE + bootstrap distillation + replay KL."""
        if not self._glossary_logged:
            self._log_metric_glossary()
            self._glossary_logged = True

        total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        if (
            self.config.with_audio
            and audio_pred is not None
            and inputs.audio_targets is not None
        ):
            audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()
            total_loss = total_loss + self.config.audio_loss_weight * audio_loss
            self._last_vfm_metrics["vfm/audio_loss"] = audio_loss.item()

        # ── Bootstrap consistency distillation ──────────────────────────────
        cfg = self.config
        step = getattr(self, "_current_step", 0)
        if cfg.self_distill_warmup_steps > 0:
            distill_ramp = min(1.0, step / max(cfg.self_distill_warmup_steps, 1))
        else:
            distill_ramp = 1.0
        effective_distill_weight = cfg.self_distill_weight * distill_ramp

        if (
            effective_distill_weight > 0
            and self._transformer_ref is not None
            and getattr(inputs, "_vfm_use_adapter", False)
            and random.random() < cfg.self_distill_prob
        ):
            distill_loss = self._bootstrap_distill_loss(
                video_pred=video_pred,
                inputs=inputs,
            )
            if distill_loss is not None:
                total_loss = total_loss + effective_distill_weight * distill_loss
                self._last_vfm_metrics["vfm/distill_loss"] = distill_loss.item()
                self._last_vfm_metrics["vfm/distill_ramp"] = distill_ramp

        # ── Noise replay buffer ──────────────────────────────────────────────
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        mu = getattr(inputs, "_vfm_adapter_mu", None)
        log_sigma = getattr(inputs, "_vfm_adapter_log_sigma", None)

        if self._replay_buffer is not None and use_adapter and mu is not None and log_sigma is not None:
            # Push current adapter distribution to the buffer (always)
            self._replay_buffer.push(
                text_embeds=inputs.video.context,
                text_mask=inputs.video.context_mask,
                positions=inputs.video.positions,
                task_class=getattr(inputs, "_vfm_task_class", None),
                mu=mu,
                log_sigma=log_sigma,
            )
            # Apply replay KL loss probabilistically
            if (
                self.config.replay_kl_weight > 0
                and len(self._replay_buffer) > 1
                and random.random() < self.config.replay_prob
            ):
                replay_loss = self._replay_kl_loss(video_pred.device, video_pred.dtype)
                if replay_loss is not None:
                    total_loss = total_loss + self.config.replay_kl_weight * replay_loss
                    self._last_vfm_metrics["vfm/replay_kl"] = replay_loss.item()

        return total_loss

    def _bootstrap_distill_loss(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
    ) -> Tensor | None:
        """Bootstrap consistency distillation: MSE(x̂₀(z₁), x̂₀(z₂)).

        The training forward already produced x̂₀ = z₁ - v(z₁) (the student).
        We sample a second noise z₂ from the same adapter (μ, σ) using a fresh
        epsilon, denoise it in one step with no_grad (the teacher), and minimise
        the squared distance between the two reconstructions.

        This trains the flow map to be variance-reducing: regardless of which z is
        drawn from the adapter, the 1-step output should be the same video.
        """
        mu = getattr(inputs, "_vfm_adapter_mu", None)
        log_sigma = getattr(inputs, "_vfm_adapter_log_sigma", None)
        if mu is None or log_sigma is None:
            return None

        device = video_pred.device
        dtype = video_pred.dtype

        # Student reconstruction (already computed, gradient attached via video_pred)
        z1 = inputs._vfm_video_noise  # [B, seq, C] — the noise used in training pass
        x_hat_0 = z1 - video_pred     # [B, seq, C]

        # Teacher: sample z₂ ~ q_φ(z|text) with a fresh epsilon (no_grad)
        # Guard: gradient checkpointing keeps ~3GB of student activations alive;
        # teacher forward needs another ~2.5GB — skip if not enough headroom.
        free_bytes = torch.cuda.mem_get_info(device)[0]
        if free_bytes < 3 * 1024 ** 3:
            logger.debug(
                f"VFM distill skipped (only {free_bytes / 1024**3:.1f} GB free)"
            )
            return None

        torch.cuda.empty_cache()
        with torch.inference_mode():
            sigma = torch.exp(log_sigma.detach())
            eps2 = torch.randn_like(mu)
            z2 = (mu.detach() + sigma * eps2).to(dtype)

            B, seq_len = z2.shape[:2]
            # Sigma=1 means the input IS the noise (pure flow-matching start)
            sigma_1 = torch.ones(B, device=device, dtype=dtype)
            timesteps_1 = torch.ones(B, seq_len, device=device, dtype=dtype)

            video_teacher = Modality(
                enabled=True,
                latent=z2,
                sigma=sigma_1,
                timesteps=timesteps_1,
                positions=inputs.video.positions,
                context=inputs.video.context,
                context_mask=inputs.video.context_mask,
            )
            result = self._transformer_ref(video=video_teacher, audio=None, perturbations=None)
            # Handle both tuple and direct Modality returns
            r = result[0] if isinstance(result, tuple) else result
            v_teacher = r.x if hasattr(r, "x") else r
            recon_teacher = (z2 - v_teacher).detach()  # [B, seq, C]

        distill_loss = (x_hat_0 - recon_teacher).pow(2).mean()

        # ── W&B metrics for distillation ─────────────────────────────────────
        # cosine_sim: most intuitive visual — rises from ~0 toward 1.0 as the
        # flow map becomes consistent (same video regardless of which z is drawn)
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                x_hat_0.float().reshape(-1),
                recon_teacher.float().reshape(-1),
                dim=0,
            )
            # Normalised loss: fraction of reconstruction energy that is variance
            recon_energy = x_hat_0.float().pow(2).mean().clamp(min=1e-8)
            distill_norm = distill_loss.detach() / recon_energy

        self._last_vfm_metrics["vfm/distill_cosine_sim"] = cos_sim.item()
        self._last_vfm_metrics["vfm/distill_loss_norm"] = distill_norm.item()

        return distill_loss

    def _replay_kl_loss(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """KL(q_new || q_old) replay penalty.

        Samples one item from the ring buffer, re-runs the adapter on its stored
        text+position conditioning (with gradients), and penalises the KL divergence
        between the fresh output and the stored reference.

        KL(N(μ₁,σ₁) ‖ N(μ₂,σ₂)) = log(σ₂/σ₁) + (σ₁²+(μ₁-μ₂)²)/(2σ₂²) − ½
        Here: new=distribution 1 (grad flows), old=distribution 2 (detached).
        """
        if self._replay_buffer is None or self._noise_adapter is None:
            return None

        sampled = self._replay_buffer.sample(device, dtype)
        if sampled is None:
            return None

        old_te, old_tm, old_pos, old_tc, old_mu, old_log_sigma = sampled

        # Re-run adapter on stored conditioning — gradients flow through here
        adapter_out = self._noise_adapter(
            text_embeddings=old_te.float(),
            text_mask=old_tm.bool() if old_tm is not None else None,
            positions=old_pos.float(),
            task_class=old_tc,
        )
        new_mu = adapter_out[0].to(dtype)
        new_log_sigma = adapter_out[1].to(dtype)

        # KL(new ‖ old): high when adapter has drifted from past distributions
        old_var = (2 * old_log_sigma).exp() + 1e-8
        kl = (
            (old_log_sigma - new_log_sigma)
            + (new_log_sigma.exp().pow(2) + (new_mu - old_mu).pow(2)) / (2 * old_var)
            - 0.5
        ).mean()

        # ── W&B metrics: decompose KL into mu vs sigma drift ─────────────────
        # mu_drift: how far the adapter's mean has moved (L2, per token)
        # sigma_drift: how far the adapter's log-variance has moved (MAE)
        with torch.no_grad():
            mu_drift = (new_mu - old_mu).norm(dim=-1).mean()
            sigma_drift = (new_log_sigma - old_log_sigma).abs().mean()

        self._last_vfm_metrics["vfm/replay_mu_drift"] = mu_drift.item()
        self._last_vfm_metrics["vfm/replay_sigma_drift"] = sigma_drift.item()
        self._last_vfm_metrics["vfm/replay_buffer_len"] = len(self._replay_buffer)

        return kl.clamp(min=0.0)

    # ── Reconstruction logging with tiled decode ─────────────────────────────

    @staticmethod
    def _resize_and_subsample(frames: Tensor, target_h: int, max_frames: int) -> Tensor:
        """Shrink decoded frames for W&B upload: resize height + subsample temporally.

        Args:
            frames: [C, T, H, W] float in [0, 1] on CPU
            target_h: output pixel height (width scaled proportionally)
            max_frames: maximum number of frames; sampled evenly across the clip
        Returns:
            [C, T', H', W'] where T'≤max_frames and H'=target_h
        """
        C, T, H, W = frames.shape

        # Temporal subsample — evenly spaced indices
        if T > max_frames:
            idx = torch.linspace(0, T - 1, max_frames).long()
            frames = frames[:, idx]
            T = max_frames

        # Spatial resize — only if needed
        if H > target_h:
            target_w = max(1, int(W * target_h / H))
            # F.interpolate expects [N, C, H, W]
            frames = F.interpolate(
                frames.permute(1, 0, 2, 3),  # [T, C, H, W]
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)  # [C, T, H', W']

        return frames

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log GT | Predict | Noisy triplet to W&B using tiled VAE decode.

        Uses tiled decoding (512px tiles, 128fr temporal) to avoid OOM on
        145-frame 544×544 videos on the 5090 while the transformer is resident.
        """
        if not WANDB_AVAILABLE or wandb.run is None or not self.config.log_reconstructions:
            return {}

        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is None or vae_decoder is None:
            return {}

        import random as _random

        b, c, f, h, w = raw_latents.shape

        noise = inputs.shared_noise  # [B, seq_len, C] patchified structured noise
        sigmas = inputs.shared_sigmas  # [B]
        if noise is None or sigmas is None:
            return {}

        # Reconstruct clean prediction: x̂₀ = z - v̂ where z=noise, v̂=video_pred
        pred_clean_seq = noise - video_pred  # [B, seq_len, C] patchified

        # Noisy input at training sigma
        sigmas_exp = sigmas.view(-1, 1, 1)
        noisy_seq = (1 - sigmas_exp) * inputs._vfm_video_latents + sigmas_exp * noise

        from ltx_core.types import VideoLatentShape
        output_shape = VideoLatentShape(batch=b, channels=c, frames=f, height=h, width=w)
        pred_spatial = self._video_patchifier.unpatchify(pred_clean_seq, output_shape)
        noisy_spatial = self._video_patchifier.unpatchify(noisy_seq, output_shape)

        sample_idx = _random.randint(0, b - 1) if b > 1 else 0

        log_dict: dict[str, Any] = {}
        try:
            decoder_device = next(vae_decoder.parameters()).device
            decoder_dtype = next(vae_decoder.parameters()).dtype

            from ltx_core.model.video_vae.tiling import (
                TilingConfig, SpatialTilingConfig, TemporalTilingConfig,
            )
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=512,
                    tile_overlap_in_pixels=128,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=128,
                    tile_overlap_in_frames=24,
                ),
            )

            def _decode_tiled(latent: Tensor) -> Tensor:
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    lat = latent.to(device=decoder_device, dtype=decoder_dtype)
                    chunks = list(vae_decoder.tiled_decode(lat, tiling_config=tiling_config))
                    decoded = torch.cat(chunks, dim=2)
                result = decoded.float().clamp(-1, 1) * 0.5 + 0.5
                return result[0].cpu()  # [3, T, H, W]

            gt_frames = _decode_tiled(raw_latents[sample_idx : sample_idx + 1])
            pred_frames = _decode_tiled(pred_spatial[sample_idx : sample_idx + 1])
            noisy_frames = _decode_tiled(noisy_spatial[sample_idx : sample_idx + 1])

            # ── Shrink before logging — full 544×544×145fr = ~80MB ───────────
            target_h = self.config.recon_log_height
            max_frames = self.config.recon_log_max_frames
            gt_frames = self._resize_and_subsample(gt_frames, target_h, max_frames)
            pred_frames = self._resize_and_subsample(pred_frames, target_h, max_frames)
            noisy_frames = self._resize_and_subsample(noisy_frames, target_h, max_frames)

            mid_f = gt_frames.shape[1] // 2
            import torchvision.utils as vutils

            use_adapter = getattr(inputs, "_vfm_use_adapter", False)
            task = getattr(inputs, "_vfm_task_name", "?")
            caption = f"Step {step} | noise={'adapter/'+task if use_adapter else 'N(0,I)'} | Noisy | Pred | GT"

            grid = vutils.make_grid(
                [noisy_frames[:, mid_f], pred_frames[:, mid_f], gt_frames[:, mid_f]],
                nrow=3, padding=2, normalize=False,
            )
            log_dict[f"{prefix}/reconstruction"] = wandb.Image(
                grid.permute(1, 2, 0).numpy(),
                caption=caption,
            )

            if self.config.recon_log_video and gt_frames.shape[1] > 1:
                side_by_side = torch.cat([noisy_frames, pred_frames, gt_frames], dim=-1)
                video_np = (side_by_side.permute(1, 0, 2, 3) * 255).clamp(0, 255).to(torch.uint8).numpy()
                log_dict[f"{prefix}/reconstruction_video"] = wandb.Video(
                    video_np, fps=8, caption=caption,
                )

            logger.debug(f"Logged VFM Ripple reconstruction at step {step}")
        except Exception as e:
            logger.warning(f"VFM Ripple reconstruction decode failed: {e}")

        return log_dict
