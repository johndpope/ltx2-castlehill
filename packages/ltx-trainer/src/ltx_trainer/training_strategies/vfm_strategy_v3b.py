"""VFM v3b — Self-Evaluating Model (Self-E) for 1-step Video Generation.

Replaces v3a's GAN discriminator with Self-E's self-evaluation mechanism
(arXiv 2512.22374, Yu et al. 2025). The model evaluates its own generated
samples using its current score estimates — no discriminator, no teacher ODE.

Key insight (Self-E):
- The model runs TWO stop-gradient forward passes on its own output:
  (1) G_θ(x̂_s, s, c) — conditional denoising
  (2) G_θ(x̂_s, s, φ) — unconditional denoising (null text)
- The difference (2)-(1) = classifier score ∝ ∇log q(c|x̂_s)
- This pushes generated samples toward text-aligned, high-density regions
- No extra network needed — the model IS its own teacher

Architecture:
    Text embeddings -> NoiseAdapterV1b -> Spherical Cauchy noise z  (from v1f)
    z -> 48-layer DiT (LoRA) -> velocity v -> x̂₀ = z - v
    Re-noise x̂₀ -> x̂_s at random s
    G_θ(x̂_s, s, φ) - G_θ(x̂_s, s, c) = classifier score feedback
    x_self = sg[x̂₀ - classifier_score]

Loss = ||x̂₀ - x_renorm||² where x_renorm blends x₀ + λ*x_self (energy-preserving)
     + kl_weight * L_KL  (adapter prior)
     + flow_match_weight * L_flow  (temporal consistency, from v3a)

Advantages over v3a:
- No 18M-param discriminator or its optimizer (saves VRAM + compute)
- Self-evaluation improves as model improves (dynamic self-teacher)
- Energy-preserving normalization prevents color bias/saturation
- Naturally supports any-step inference (not just 1-step)
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)


class SelfEVFMv3bTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v3b (Self-E self-evaluation + VFM)."""

    name: Literal["vfm_v3b"] = "vfm_v3b"

    # === Self-evaluation settings ===
    self_eval_weight: float = Field(
        default=1.0, ge=0.0,
        description="Base weight for self-evaluation loss. "
        "Actual weight is λ_{s,t} * self_eval_weight where λ = σ_t/α_t - σ_s/α_s.",
    )
    self_eval_cfg_scale: float = Field(
        default=5.0, ge=1.0,
        description="Classifier-free guidance scale for self-evaluation. "
        "Higher = stronger text alignment signal. Paper default: 5.0.",
    )
    self_eval_s_range: tuple[float, float] = Field(
        default=(0.1, 0.5),
        description="Range [s_min, s_max] for random re-noising level s during self-evaluation. "
        "s=0 is clean, s=1 is pure noise. Paper anneals from near-t to 0 over training.",
    )
    energy_preserving_norm: bool = Field(
        default=True,
        description="Apply energy-preserving normalization to combined target (Eq. 19 in Self-E). "
        "Prevents color bias from large λ values.",
    )
    use_ema_for_eval: bool = Field(
        default=False,
        description="Use EMA model for conditional branch in self-evaluation. "
        "Paper uses EMA for conditional, non-EMA for unconditional. "
        "Disabled by default (we don't maintain separate EMA in trainer).",
    )
    self_eval_prob: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Probability of applying self-evaluation (vs pure data loss) per step. "
        "Paper uses 0.5 (s sampled with 50% probability of s=t).",
    )

    # === Reconstruction regularizer (from v3a, optional) ===
    recon_weight: float = Field(
        default=0.0, ge=0.0,
        description="Weight for explicit reconstruction loss |x̂₀ - x₀|². "
        "Self-E's data loss already includes this, so typically 0.",
    )

    # === Latent Perceptual Loss ===
    latent_perceptual_weight: float = Field(
        default=1.0, ge=0.0,
        description="Weight for latent perceptual loss. Uses multi-scale features "
        "from the latent tokens (local patches at 1x, 2x, 4x pooling) to compute "
        "structural similarity. More perceptually meaningful than raw MSE. "
        "0 = disabled.",
    )

    # === Flow Distribution Matching (from v3a, kept for temporal consistency) ===
    flow_match_weight: float = Field(
        default=0.0, ge=0.0,
        description="Weight for flow distribution matching loss (DiagDistill). "
        "Matches frame-to-frame latent diffs. 0 = disabled.",
    )

    # Unused v3a fields (kept for config compat)
    fake_score_lr: float = Field(default=1e-4, description="Unused.")
    fake_score_hidden_dim: int = Field(default=256, description="Unused.")
    fake_score_num_heads: int = Field(default=4, description="Unused.")
    fake_score_num_layers: int = Field(default=4, description="Unused.")
    dmd_sigma: float = Field(default=0.5, description="Unused.")
    dmd_weight: float = Field(default=1.0, description="Unused.")
    score_update_ratio: int = Field(default=1, description="Unused.")
    cache_teacher_outputs: bool = Field(default=False, description="Unused.")


class SelfEVFMv3bTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v3b — Self-Evaluating Model for 1-step video generation.

    Uses Self-E's self-evaluation mechanism instead of a GAN discriminator.
    The model evaluates its own generated samples using conditional vs
    unconditional forward passes (classifier score), providing distribution
    matching feedback without any auxiliary network.

    Requires (set by trainer.py):
    - set_transformer(): reference to transformer for self-evaluation passes
    """

    config: SelfEVFMv3bTrainingConfig

    def __init__(self, config: SelfEVFMv3bTrainingConfig):
        super().__init__(config)
        self._transformer_ref: nn.Module | None = None

    def set_transformer(self, transformer: nn.Module, grad_accumulation_steps: int = 1) -> None:
        """Store transformer reference for self-evaluation passes."""
        self._transformer_ref = transformer
        logger.info("VFM v3b (Self-E): Transformer reference set")

    def _self_evaluate(
        self,
        x_hat_0: Tensor,
        text_embeds: Tensor,
        text_mask: Tensor | None,
        positions: Tensor,
        s: float,
    ) -> Tensor:
        """Self-E classifier score: ∇log q(c|x̂_s) via two stop-gradient passes.

        Re-noises x̂₀ to x̂_s, then runs two forward passes:
        - G_θ(x̂_s, s, c) — conditional (what the model thinks clean looks like given text)
        - G_θ(x̂_s, s, φ) — unconditional (what the model thinks clean looks like without text)

        Returns pseudo-target: x_self = sg[x̂₀ - (G_θ(x̂_s, s, φ) - G_θ(x̂_s, s, c))]
        """
        from ltx_core.model.transformer.modality import Modality  # noqa: PLC0415

        B = x_hat_0.shape[0]
        seq_len = x_hat_0.shape[1]
        device = x_hat_0.device
        dtype = x_hat_0.dtype

        # Re-noise x̂₀ at noise level s: x̂_s = (1-s)*x̂₀ + s*ε
        eps = torch.randn_like(x_hat_0)
        x_hat_s = (1 - s) * x_hat_0.detach() + s * eps

        # Per-token timesteps at noise level s
        sigma_batch = torch.full((B,), s, device=device, dtype=dtype)
        timesteps = sigma_batch.unsqueeze(1).expand(-1, seq_len)

        # Conditional pass: G_θ(x̂_s, s, c)
        video_cond = Modality(
            enabled=True, latent=x_hat_s, sigma=sigma_batch,
            timesteps=timesteps, positions=positions,
            context=text_embeds, context_mask=text_mask,
        )

        # Unconditional pass: G_θ(x̂_s, s, φ)  — zeroed text embeddings
        null_embeds = torch.zeros_like(text_embeds)
        video_uncond = Modality(
            enabled=True, latent=x_hat_s, sigma=sigma_batch,
            timesteps=timesteps, positions=positions,
            context=null_embeds, context_mask=text_mask,
        )

        with torch.no_grad():
            # Conditional: predicts velocity → x̂₀ = x̂_s - s * v_cond
            result_cond = self._transformer_ref(video=video_cond, audio=None, perturbations=None)
            # Handle both tuple and Modality returns
            rc = result_cond[0] if isinstance(result_cond, tuple) else result_cond
            v_cond = rc.x if hasattr(rc, 'x') else rc
            g_cond = x_hat_s - s * v_cond  # Denoised estimate (conditional)

            # Unconditional: predicts velocity → x̂₀ = x̂_s - s * v_uncond
            result_uncond = self._transformer_ref(video=video_uncond, audio=None, perturbations=None)
            ru = result_uncond[0] if isinstance(result_uncond, tuple) else result_uncond
            v_uncond = ru.x if hasattr(ru, 'x') else ru
            g_uncond = x_hat_s - s * v_uncond  # Denoised estimate (unconditional)

        # Classifier score direction: uncond - cond
        # (pushes toward high p(c|x) — text-aligned regions)
        classifier_score = g_uncond - g_cond

        # Pseudo-target (Eq. 14 in Self-E paper)
        x_self = (x_hat_0 - classifier_score).detach()

        return x_self

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute Self-E loss: data loss + self-evaluation + optional KL/flow.

        Loss structure:
        1. Data loss: ||x̂₀ - x₀||² (always)
        2. Self-evaluation: ||x̂₀ - x_self||² with energy-preserving norm (probabilistic)
        3. KL: adapter prior loss (from v1f)
        4. Flow matching: temporal consistency (optional, from v3a)
        """
        cfg = self.config
        step = self._current_step
        device = video_pred.device
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        # Fall back to v1f if no adapter or transformer not set
        if not use_adapter or self._transformer_ref is None:
            return super().compute_loss(video_pred, audio_pred, inputs)

        video_noise = inputs._vfm_video_noise  # z (adapter noise)
        video_latents = inputs._vfm_video_latents  # x₀ (ground truth)
        text_embeds = inputs.video.context
        text_mask = inputs.video.context_mask
        positions = inputs.video.positions

        # Student prediction: x̂₀ = z - v_pred (flow matching convention)
        x_hat_0 = video_noise - video_pred  # [B, seq, 128]

        # ════════════════════════════════════════════════════════════
        # DATA LOSS: ||x̂₀ - x₀||² (Eq. 7 in Self-E)
        # ════════════════════════════════════════════════════════════
        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            data_loss = ((x_hat_0 - video_latents).pow(2) * mask).sum() / mask.sum().clamp(min=1) / x_hat_0.shape[-1]
        else:
            data_loss = (x_hat_0 - video_latents).pow(2).mean()

        total_loss = data_loss

        # ════════════════════════════════════════════════════════════
        # LATENT PERCEPTUAL LOSS (multi-scale structural similarity)
        # ════════════════════════════════════════════════════════════
        loss_perceptual = torch.tensor(0.0, device=device)
        if cfg.latent_perceptual_weight > 0:
            loss_perceptual = self._compute_latent_perceptual_loss(
                x_hat_0, video_latents, positions
            )
            total_loss = total_loss + cfg.latent_perceptual_weight * loss_perceptual

        # ════════════════════════════════════════════════════════════
        # SELF-EVALUATION LOSS (Eq. 15-20 in Self-E)
        # ════════════════════════════════════════════════════════════
        apply_self_eval = (
            cfg.self_eval_weight > 0
            and random.random() < cfg.self_eval_prob
            and step > 0  # Skip step 0 (model hasn't learned anything yet)
        )

        loss_self_eval = torch.tensor(0.0, device=device)
        if apply_self_eval:
            # Sample random noise level s for re-noising
            s_min, s_max = cfg.self_eval_s_range
            s = random.uniform(s_min, s_max)

            # Flow matching coefficients: α_t = 1-t, σ_t = t
            # For VFM 1-step: t ≈ 1 (starting from pure noise)
            t = 1.0  # VFM operates at t=1 (full noise → clean)
            alpha_t, sigma_t = 1 - t + 1e-6, t  # avoid div by zero
            alpha_s, sigma_s = 1 - s, s

            # λ_{s,t} = σ_t/α_t - σ_s/α_s (Eq. 17)
            lambda_st = sigma_t / alpha_t - sigma_s / alpha_s
            lambda_st = min(lambda_st, 10.0)  # Clamp to prevent explosion

            # Get self-evaluation pseudo-target
            x_self = self._self_evaluate(
                x_hat_0, text_embeds, text_mask, positions, s,
            )

            if cfg.energy_preserving_norm:
                # Energy-preserving normalization (Eq. 19)
                # x_renorm = (x₀ + λ*x_self) / ||x₀ + λ*x_self|| * ||x₀||
                combined = video_latents + lambda_st * x_self
                x0_norm = video_latents.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                combined_norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                x_target = combined / combined_norm * x0_norm
            else:
                # Simple weighted average (Eq. 18)
                x_target = (video_latents + lambda_st * x_self) / (1 + lambda_st)

            # Self-evaluation loss
            loss_self_eval = (x_hat_0 - x_target).pow(2).mean()
            total_loss = loss_self_eval  # Replace data loss with combined target

        # ════════════════════════════════════════════════════════════
        # ADAPTER KL (from v1f)
        # ════════════════════════════════════════════════════════════
        adapter_kl = getattr(inputs, "_vfm_kl_loss", torch.tensor(0.0, device=device))
        if cfg.kl_weight > 0:
            total_loss = total_loss + cfg.kl_weight * adapter_kl

        # ════════════════════════════════════════════════════════════
        # OBS LOSS (VFM paper Eq.14 — forces text-conditioned adapter)
        # ════════════════════════════════════════════════════════════
        loss_obs = torch.tensor(0.0, device=device)
        if use_adapter and cfg.obs_loss_weight > 0:
            # Add noise to GT, compare adapter-denoised output
            obs_noise = torch.randn_like(video_latents) * cfg.obs_noise_level
            noisy_gt = video_latents + obs_noise
            obs_diff = (x_hat_0 - noisy_gt).pow(2)
            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                loss_obs = (obs_diff * mask).sum() / mask.sum().clamp(min=1) / obs_diff.shape[-1]
            else:
                loss_obs = obs_diff.mean()
            total_loss = total_loss + cfg.obs_loss_weight * loss_obs

        # ════════════════════════════════════════════════════════════
        # MU ALIGNMENT (anti-collapse: adapter mu must point toward x₀)
        # ════════════════════════════════════════════════════════════
        loss_mu_align = torch.tensor(0.0, device=device)
        adapter_mu = getattr(inputs, "_vfm_adapter_mu", None)
        if adapter_mu is not None and cfg.mu_align_weight > 0:
            # Cosine alignment: mu should point in same direction as x₀
            mu_norm = F.normalize(adapter_mu, dim=-1)
            x0_norm = F.normalize(video_latents, dim=-1)
            cos_align = (mu_norm * x0_norm).sum(dim=-1).mean()
            loss_mu_align = 1 - cos_align
            total_loss = total_loss + cfg.mu_align_weight * loss_mu_align

        # ════════════════════════════════════════════════════════════
        # DIVERSITY (prevent adapter collapse to single noise pattern)
        # ════════════════════════════════════════════════════════════
        loss_diversity = torch.tensor(0.0, device=device)
        if adapter_mu is not None and cfg.diversity_weight > 0 and adapter_mu.shape[0] > 1:
            # Cross-sample diversity: different prompts should give different mu
            mu_flat = adapter_mu.mean(dim=1)  # [B, C]
            mu_flat_norm = F.normalize(mu_flat, dim=-1)
            sim_matrix = torch.mm(mu_flat_norm, mu_flat_norm.T)
            # Penalize high similarity between different samples
            mask_diag = 1 - torch.eye(sim_matrix.shape[0], device=device)
            loss_diversity = (sim_matrix * mask_diag).mean()
            total_loss = total_loss + cfg.diversity_weight * loss_diversity

        # ════════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════════
        if step % 20 == 0:
            try:
                import wandb  # noqa: PLC0415
                if wandb.run is not None:
                    log_data = {
                        # Log under BOTH v3b/ and vfm/ so W&B panels work
                        "v3b/loss_data": data_loss.item(),
                        "v3b/loss_perceptual": loss_perceptual.item() if isinstance(loss_perceptual, Tensor) else loss_perceptual,
                        "v3b/loss_self_eval": loss_self_eval.item() if isinstance(loss_self_eval, Tensor) else loss_self_eval,
                        "v3b/loss_obs": loss_obs.item() if isinstance(loss_obs, Tensor) else loss_obs,
                        "v3b/loss_mu_align": loss_mu_align.item() if isinstance(loss_mu_align, Tensor) else loss_mu_align,
                        "v3b/loss_diversity": loss_diversity.item() if isinstance(loss_diversity, Tensor) else loss_diversity,
                        "v3b/loss_kl": adapter_kl.item() if isinstance(adapter_kl, Tensor) else adapter_kl,
                        "v3b/loss_total": total_loss.item(),
                        "v3b/self_eval_applied": float(apply_self_eval),
                        "v3b/student_recon_mse": (x_hat_0 - video_latents).pow(2).mean().item(),
                        # Mirror to vfm/ prefix for existing W&B panels
                        "vfm/loss_obs": loss_obs.item() if isinstance(loss_obs, Tensor) else loss_obs,
                        "vfm/loss_kl": adapter_kl.item() if isinstance(adapter_kl, Tensor) else adapter_kl,
                        "vfm/loss_mu_align": loss_mu_align.item() if isinstance(loss_mu_align, Tensor) else loss_mu_align,
                        "vfm/loss_mf": data_loss.item(),
                        "vfm/loss_total": total_loss.item(),
                    }
                    # Don't pass step= to avoid conflicting with trainer's wandb.log calls
                    wandb.log(log_data)
            except Exception:
                pass

        return total_loss

    @staticmethod
    def _compute_latent_perceptual_loss(
        pred: Tensor,
        target: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Multi-scale latent perceptual loss on token sequences.

        Computes structural similarity at multiple scales by treating the
        token sequence as a 1D signal and pooling at different window sizes.
        No spatial reshape needed — works regardless of H/W/F dims.

        Three components:
        1. Cosine similarity (structural alignment, scale-invariant)
        2. L1 (magnitude errors)
        3. Multi-scale: pool tokens in groups of 1, 4, 16 for local→global features

        Args:
            pred: [B, seq, C] predicted latent tokens
            target: [B, seq, C] ground truth latent tokens
            positions: unused (kept for API compat)
        """
        B, seq_len, C = pred.shape
        device = pred.device

        total_loss = torch.tensor(0.0, device=device)

        # Scale 1: per-token cosine + L1 (fine detail)
        p_norm = F.normalize(pred, dim=-1)
        t_norm = F.normalize(target, dim=-1)
        cos_sim = (p_norm * t_norm).sum(dim=-1).mean()
        total_loss = total_loss + (1 - cos_sim) + 0.5 * (pred - target).abs().mean()

        # Scale 2: pool groups of 4 tokens (local neighborhood)
        if seq_len >= 4:
            trim = seq_len - (seq_len % 4)
            p4 = pred[:, :trim].reshape(B, trim // 4, 4, C).mean(dim=2)
            t4 = target[:, :trim].reshape(B, trim // 4, 4, C).mean(dim=2)
            cos4 = (F.normalize(p4, dim=-1) * F.normalize(t4, dim=-1)).sum(-1).mean()
            total_loss = total_loss + 0.5 * (1 - cos4) + 0.25 * (p4 - t4).abs().mean()

        # Scale 3: pool groups of 16 tokens (global composition)
        if seq_len >= 16:
            trim = seq_len - (seq_len % 16)
            p16 = pred[:, :trim].reshape(B, trim // 16, 16, C).mean(dim=2)
            t16 = target[:, :trim].reshape(B, trim // 16, 16, C).mean(dim=2)
            cos16 = (F.normalize(p16, dim=-1) * F.normalize(t16, dim=-1)).sum(-1).mean()
            total_loss = total_loss + 0.25 * (1 - cos16) + 0.125 * (p16 - t16).abs().mean()

        return total_loss

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log v1f reconstructions + distribution variation samples.

        Generates K different samples from the adapter for the same prompt
        to visualize distribution diversity (collapsed vs diverse).
        """
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        # Generate variation samples from the distribution
        if self._transformer_ref is not None and vae_decoder is not None:
            try:
                variation_plots = self._log_distribution_variations(
                    inputs, step, vae_decoder, prefix, num_variations=3,
                )
                log_dict.update(variation_plots)
            except Exception as e:
                logger.debug(f"Distribution variation logging failed: {e}")

        return log_dict

    def _log_distribution_variations(
        self,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module,
        prefix: str,
        num_variations: int = 3,
    ) -> dict[str, Any]:
        """Sample K different z's from the adapter, denoise each, show side-by-side.

        Shows: GT | Sample1 | Sample2 | Sample3
        If all samples look the same → distribution collapsed.
        If samples vary meaningfully → distribution is diverse.
        """
        import wandb  # noqa: PLC0415
        from ltx_core.model.transformer.modality import Modality  # noqa: PLC0415
        from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: PLC0415
        from ltx_core.types import VideoLatentShape  # noqa: PLC0415

        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if not use_adapter or raw_latents is None or self._noise_adapter is None:
            return {}

        device = inputs.video.latent.device
        dtype = inputs.video.latent.dtype
        text_embeds = inputs.video.context
        text_mask = inputs.video.context_mask
        positions = inputs.video.positions

        # Get adapter reference
        adapter = self._noise_adapter
        task_class = getattr(inputs, "_vfm_task_class", None)
        if task_class is None:
            task_class = torch.tensor([0], device=device)

        patchifier = VideoLatentPatchifier(patch_size=1)
        gt = raw_latents[0]  # [C, F, H, W]
        C, F, H, W = gt.shape

        variation_frames = []

        with torch.inference_mode():
            # Decode GT first frame for reference
            gt_pixels = vae_decoder(gt.unsqueeze(0).to(vae_decoder.parameters().__next__().device))
            gt_frame = gt_pixels[0, :, 0].clamp(0, 1)  # First frame [C, H, W]
            variation_frames.append(gt_frame.cpu())

            # Generate K variations
            for k in range(num_variations):
                # Sample new z from adapter (different random seed each time)
                torch.manual_seed(step * 1000 + k + 42)
                adapter_out = adapter.forward(
                    text_embeddings=text_embeds[:1].float(),
                    text_mask=text_mask[:1].bool() if text_mask is not None else None,
                    positions=positions[:1].float(),
                    task_class=task_class,
                )
                mu_k, log_sigma_k = adapter_out[0], adapter_out[1]
                sigma_k = torch.exp(log_sigma_k)
                eps_k = torch.randn_like(mu_k)
                z_k = (mu_k + sigma_k * eps_k).to(dtype)

                # 1-step denoise
                seq_len = z_k.shape[1]
                sigma_val = torch.ones(1, device=device, dtype=dtype)
                timesteps = torch.ones(1, seq_len, device=device, dtype=dtype)

                video_mod = Modality(
                    enabled=True, latent=z_k, sigma=sigma_val,
                    timesteps=timesteps, positions=positions[:1],
                    context=text_embeds[:1], context_mask=text_mask[:1] if text_mask is not None else None,
                )
                result = self._transformer_ref(video=video_mod, audio=None, perturbations=None)
                vo = result[0] if isinstance(result, tuple) else result
                v_pred = vo.x if hasattr(vo, 'x') else vo
                x_hat = z_k - v_pred

                # Unpatchify and decode
                x_spatial = patchifier.unpatchify(
                    x_hat,
                    output_shape=VideoLatentShape(frames=F, height=H, width=W, batch=1, channels=C),
                )
                pixels_k = vae_decoder(x_spatial.to(vae_decoder.parameters().__next__().device))
                frame_k = pixels_k[0, :, 0].clamp(0, 1).cpu()  # First frame
                variation_frames.append(frame_k)

        # Stack side-by-side: [GT | Var1 | Var2 | Var3]
        import torchvision  # noqa: PLC0415
        grid = torchvision.utils.make_grid(variation_frames, nrow=len(variation_frames), padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")

        return {
            f"{prefix}/distribution_variations": wandb.Image(
                grid_np,
                caption=f"Step {step} | GT | {num_variations} adapter samples (different z)"
            ),
        }
