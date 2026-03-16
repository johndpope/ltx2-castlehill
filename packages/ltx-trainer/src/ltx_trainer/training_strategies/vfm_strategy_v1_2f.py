"""VFM v1.2f — Iterative Latent Refinement (ILR) on top of v1f.

Extends v1f (Spherical Cauchy + per-token sigma) with a 2-pass training loop:

Pass 1: z -> DiT(z, sigma=1.0) -> v1 -> x_hat_0_1 = z - v1
Pass 2: x_hat_0_1 -> DiT(x_hat_0_1, sigma_refine) -> v2 -> x_hat_0_2 = x_hat_0_1 - v2
Loss = alpha * ||x_hat_0_1 - GT||^2 + (1-alpha) * ||x_hat_0_2 - GT||^2

Key design:
- Stop-gradient on x_hat_0_1 before pass 2 (can't backprop through 2 full 19B DiT passes)
- Warmup: only pass 1 for first N steps (let model learn basics first)
- Adaptive sigma: use sigma_head on x_hat_0_1 to determine per-token refinement sigma
- Error threshold: optionally skip pass 2 when pass 1 is already good enough
- The refinement pass teaches the model self-correction on hard samples

Architecture (unchanged from v1f):
    Text embeddings -> NoiseAdapterV1b -> (mu, log_sigma) per token
    mu_hat = normalize(mu), r = ||mu||, kappa = exp(mean(log_sigma))
    z_dir ~ SphericalCauchy(mu_hat, kappa), z = r * z_dir
    z -> SigmaHead -> per-token sigma_i  (inherited from v1d)
    x_t[i] = (1 - sigma_i) * x_0[i] + sigma_i * z[i]
    48-layer DiT -> velocity v -> x_hat_0

New in v1.2f:
    After pass 1 prediction x_hat_0_1:
    x_hat_0_1 -> DiT(x_hat_0_1, sigma_refine) -> v2 -> x_hat_0_2
    Combined loss trains the model to self-correct.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.vfm_strategy_v1f import (
    VFMv1fTrainingConfig,
    VFMv1fTrainingStrategy,
)


class VFMv12fTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v1.2f (Iterative Latent Refinement)."""

    name: Literal["vfm_v1_2f"] = "vfm_v1_2f"

    # === ILR settings ===
    ilr_enabled: bool = Field(
        default=True,
        description="Enable Iterative Latent Refinement (2-pass training). "
        "When disabled, falls back to standard v1f single-pass.",
    )
    ilr_refine_sigma: float = Field(
        default=0.3, ge=0.01, le=1.0,
        description="Sigma for refinement pass (lower = easier cleanup task). "
        "This represents how 'noisy' the model treats x_hat_0_1 during pass 2.",
    )
    ilr_pass1_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Weight on pass 1 loss in the combined ILR loss.",
    )
    ilr_pass2_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Weight on pass 2 (refinement) loss in the combined ILR loss.",
    )
    ilr_warmup_steps: int = Field(
        default=200, ge=0,
        description="Only do pass 1 for the first N steps. "
        "Lets the model learn basic denoising before adding refinement.",
    )
    ilr_adaptive_sigma: bool = Field(
        default=True,
        description="Use sigma_head to determine per-token refinement sigma. "
        "Higher sigma for tokens where pass 1 was bad (content-adaptive).",
    )
    ilr_error_threshold: float = Field(
        default=0.0, ge=0.0,
        description="If > 0, only do pass 2 when pass 1 MSE exceeds this threshold. "
        "Saves compute on easy samples. 0.0 = always do pass 2.",
    )
    ilr_stop_gradient: bool = Field(
        default=True,
        description="Stop gradient on pass 1 output before pass 2. "
        "Required for memory safety (can't backprop through 2 full 19B passes).",
    )

    # === Hard example replay ===
    hard_example_replay: bool = Field(
        default=False,
        description="Enable hard example replay buffer. "
        "Stores high-loss samples and replays them periodically.",
    )
    replay_buffer_size: int = Field(
        default=100, ge=10,
        description="Maximum number of samples in the replay buffer.",
    )
    replay_every_n_steps: int = Field(
        default=10, ge=1,
        description="Replay a hard example every N steps.",
    )
    replay_loss_percentile: float = Field(
        default=0.9, ge=0.5, le=1.0,
        description="Only replay samples above this loss percentile.",
    )


class VFMv12fTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v1.2f -- Iterative Latent Refinement on top of v1f.

    After the standard v1f forward pass (pass 1), feeds the predicted x_hat_0_1
    back through the transformer at a lower refinement sigma for a second pass.
    The model learns to fix its own mistakes, especially on hard samples.

    Requires a reference to the transformer for the refinement forward pass.
    The trainer calls set_transformer() after model loading.
    """

    config: VFMv12fTrainingConfig

    def __init__(self, config: VFMv12fTrainingConfig):
        super().__init__(config)
        self._transformer_ref_for_ilr: nn.Module | None = None
        self._replay_buffer: deque[dict[str, Tensor]] = deque(maxlen=config.replay_buffer_size)
        self._loss_history: deque[float] = deque(maxlen=1000)

    def set_transformer(self, transformer: nn.Module, grad_accumulation_steps: int = 1) -> None:
        """Store reference to the transformer for ILR refinement passes.

        Called by trainer.py after model loading, similar to set_vae_decoder.
        """
        self._transformer_ref_for_ilr = transformer
        self._grad_accumulation_steps = grad_accumulation_steps
        logger.info(
            f"VFM v1.2f: Stored transformer reference for ILR "
            f"(grad_accum={grad_accumulation_steps})"
        )

    def _run_refinement_pass(
        self,
        x0_hat1: Tensor,
        inputs: ModelInputs,
        refine_sigmas: Tensor | float,
    ) -> Tensor:
        """Run the refinement (pass 2) through the transformer.

        Takes x_hat_0_1 (pass 1 prediction) as the new input and runs
        the transformer at a lower refinement sigma.

        Args:
            x0_hat1: [B, seq, C] pass 1 predicted clean video (detached)
            inputs: Original ModelInputs (for positions, context, masks)
            refine_sigmas: Scalar or [B, seq] refinement sigma values

        Returns:
            video_pred_2: [B, seq, C] velocity prediction from pass 2
        """
        if self._transformer_ref_for_ilr is None:
            raise RuntimeError(
                "VFM v1.2f: transformer reference not set. "
                "Ensure trainer.py calls strategy.set_transformer(transformer) after model loading."
            )

        from ltx_core.model.transformer.modality import Modality

        batch_size = x0_hat1.shape[0]
        seq_len = x0_hat1.shape[1]
        device = x0_hat1.device
        dtype = x0_hat1.dtype

        # Build per-token timesteps for refinement
        # refine_sigmas can be scalar or [B, seq]
        if isinstance(refine_sigmas, (int, float)):
            refine_sigma_scalar = refine_sigmas
            per_token_ts = torch.full(
                (batch_size, seq_len), refine_sigma_scalar,
                device=device, dtype=dtype,
            )
            batch_sigma = torch.full(
                (batch_size,), refine_sigma_scalar,
                device=device, dtype=dtype,
            )
        else:
            # refine_sigmas is [B, seq] tensor
            per_token_ts = refine_sigmas.to(dtype=dtype)
            batch_sigma = per_token_ts.mean(dim=1)

        # Zero out conditioning tokens (same mask as pass 1)
        video_loss_mask = inputs.video_loss_mask
        if video_loss_mask is not None:
            # video_loss_mask is True for non-conditioning tokens
            conditioning_mask = ~video_loss_mask
            per_token_ts = per_token_ts * video_loss_mask.float()
            # Keep conditioning tokens clean in input
            conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)
            # x0_hat1 for conditioning tokens should be GT
            gt_latents = getattr(inputs, "_vfm_video_latents", None)
            if gt_latents is not None:
                x0_hat1_clean = torch.where(conditioning_mask_expanded, gt_latents, x0_hat1)
            else:
                x0_hat1_clean = x0_hat1
        else:
            x0_hat1_clean = x0_hat1

        # Reuse positions and context from original pass
        original_video = inputs.video

        # Build refinement Modality
        refine_video = Modality(
            enabled=True,
            sigma=batch_sigma,
            latent=x0_hat1_clean,
            timesteps=per_token_ts,
            positions=original_video.positions,
            context=original_video.context,
            context_mask=original_video.context_mask,
        )

        # Run transformer forward (no gradient through the model if stop_gradient)
        # Note: we DO want gradients through the transformer weights (they need to learn
        # refinement). The stop_gradient is only on x0_hat1 input.
        video_pred_2, _ = self._transformer_ref_for_ilr(
            video=refine_video,
            audio=None,
            perturbations=None,
        )

        return video_pred_2

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute ILR loss = pass1_weight * L1 + pass2_weight * L2.

        Pass 1 loss is the standard v1f loss.
        Pass 2 loss uses the refinement prediction.

        Falls back to pure v1f loss when:
        - ILR is disabled
        - During warmup steps
        - When transformer reference is not available
        - When error threshold is set and pass 1 error is below it
        """
        cfg = self.config
        step = self._current_step

        # Pass 1 loss (standard v1f)
        pass1_loss = super().compute_loss(video_pred, audio_pred, inputs)

        # Check if we should do ILR
        if not cfg.ilr_enabled:
            return pass1_loss

        if step < cfg.ilr_warmup_steps:
            return pass1_loss

        if self._transformer_ref_for_ilr is None:
            logger.warning_once(
                "VFM v1.2f: transformer reference not set, falling back to v1f loss"
            )
            return pass1_loss

        use_adapter = getattr(inputs, "_vfm_use_adapter", False)
        if not use_adapter:
            # No adapter noise -> no refinement needed
            return pass1_loss

        # ════════════════════════════════════════════════════════════════
        # MEMORY OPTIMIZATION: backward pass 1 first, then forward pass 2
        # This frees pass 1 activation memory before pass 2 allocates.
        # Peak VRAM = model_weights + ONE pass of activations (not two).
        # ════════════════════════════════════════════════════════════════
        # Scale by grad_accum to match what accelerator.backward() does
        grad_accum = getattr(self, "_grad_accumulation_steps", 1)
        scaled_pass1 = cfg.ilr_pass1_weight * pass1_loss / grad_accum
        scaled_pass1.backward(retain_graph=False)
        pass1_loss_value = pass1_loss.item()

        # Free pass 1 computation graph
        del pass1_loss

        # Compute x_hat_0_1 = z - v_pred (pass 1 prediction)
        video_noise = inputs._vfm_video_noise  # [B, seq, C]
        x0_hat1 = (video_noise - video_pred).detach()  # always detach for pass 2 input

        # Optional: only refine if pass 1 error is high
        if cfg.ilr_error_threshold > 0:
            video_latents = inputs._vfm_video_latents  # [B, seq, C]
            pass1_mse = (x0_hat1 - video_latents).pow(2)
            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                pass1_error = (pass1_mse * mask).sum() / mask.sum().clamp(min=1) / pass1_mse.shape[-1]
            else:
                pass1_error = pass1_mse.mean()

            if pass1_error.item() < cfg.ilr_error_threshold:
                # Pass 1 is good enough, skip refinement.
                # Pass 1 grads already accumulated — return zero loss so trainer's
                # backward() is a no-op for pass 2.
                if hasattr(self, "_last_vfm_metrics") and self._last_vfm_metrics is not None:
                    self._last_vfm_metrics["vfm/ilr_skipped"] = 1.0
                    self._last_vfm_metrics["vfm/ilr_pass1_error"] = pass1_error.item()
                return torch.tensor(0.0, device=video_noise.device, requires_grad=True)

        # x0_hat1 is already detached (from the memory optimization above)

        # Determine refinement sigma
        if cfg.ilr_adaptive_sigma and self._sigma_head is not None:
            # Use sigma_head on x_hat_0_1 to determine per-token refinement sigma
            video_latents = inputs._vfm_video_latents  # [B, seq, C]
            with torch.no_grad():
                refine_sigmas = self._sigma_head(
                    x0_hat1.float(),
                    x0=video_latents.float(),
                )  # [B, seq]
            # Scale down: refinement should be lighter than initial denoising
            refine_sigmas = refine_sigmas * cfg.ilr_refine_sigma
            # Clamp to reasonable range
            refine_sigmas = refine_sigmas.clamp(min=0.01, max=cfg.ilr_refine_sigma)
        else:
            refine_sigmas = cfg.ilr_refine_sigma

        # Run refinement pass (x0_hat1 is detached — no grad flows back to pass 1)
        pass2_video_pred = self._run_refinement_pass(
            x0_hat1, inputs, refine_sigmas,
        )

        # x_hat_0_2 = x_hat_0_1 - v2
        x0_hat2 = x0_hat1 - pass2_video_pred

        # Pass 2 loss: ||x_hat_0_2 - GT||^2
        video_latents = inputs._vfm_video_latents  # [B, seq, C]
        pass2_diff = (x0_hat2 - video_latents).pow(2)

        if inputs.video_loss_mask is not None:
            mask = inputs.video_loss_mask.unsqueeze(-1).float()
            pass2_loss = (pass2_diff * mask).sum() / mask.sum().clamp(min=1) / pass2_diff.shape[-1]
        else:
            pass2_loss = pass2_diff.mean()

        # Pass 1 was already backward'd above — return only scaled pass 2 loss.
        # The trainer will call .backward() on this, accumulating gradients
        # from both passes into the same parameters.
        total_loss = cfg.ilr_pass2_weight * pass2_loss

        # Compute improvement metric
        pass1_recon_error = (x0_hat1 - video_latents).pow(2).mean().item()
        pass2_recon_error = (x0_hat2.detach() - video_latents).pow(2).mean().item()
        improvement = (pass1_recon_error - pass2_recon_error) / max(pass1_recon_error, 1e-8)

        # Log ILR metrics
        if hasattr(self, "_last_vfm_metrics") and self._last_vfm_metrics is not None:
            self._last_vfm_metrics["vfm/loss_pass1"] = pass1_loss_value
            self._last_vfm_metrics["vfm/loss_pass2"] = pass2_loss.item()
            self._last_vfm_metrics["vfm/ilr_improvement"] = improvement
            self._last_vfm_metrics["vfm/ilr_skipped"] = 0.0
            self._last_vfm_metrics["vfm/ilr_pass1_recon_error"] = pass1_recon_error
            self._last_vfm_metrics["vfm/ilr_pass2_recon_error"] = pass2_recon_error
            if isinstance(refine_sigmas, Tensor):
                active_mask = inputs.video_loss_mask
                if active_mask is not None:
                    active_sigmas = refine_sigmas[active_mask]
                else:
                    active_sigmas = refine_sigmas.flatten()
                if active_sigmas.numel() > 0:
                    self._last_vfm_metrics["vfm/refine_sigma_mean"] = active_sigmas.mean().item()
                    self._last_vfm_metrics["vfm/refine_sigma_std"] = active_sigmas.std().item()
            else:
                self._last_vfm_metrics["vfm/refine_sigma_mean"] = cfg.ilr_refine_sigma

        # Hard example replay: store high-loss samples
        if cfg.hard_example_replay:
            self._update_replay_buffer(pass1_loss.item(), inputs)

        return total_loss

    def _update_replay_buffer(self, loss: float, inputs: ModelInputs) -> None:
        """Store high-loss samples in replay buffer for later re-training."""
        self._loss_history.append(loss)

        if len(self._loss_history) < 10:
            return

        # Compute loss percentile threshold
        sorted_losses = sorted(self._loss_history)
        threshold_idx = int(len(sorted_losses) * self.config.replay_loss_percentile)
        threshold = sorted_losses[min(threshold_idx, len(sorted_losses) - 1)]

        if loss >= threshold:
            # Store a lightweight copy of the key tensors
            replay_entry = {
                "video_noise": inputs._vfm_video_noise.detach().cpu(),
                "video_latents": inputs._vfm_video_latents.detach().cpu(),
                "loss": loss,
            }
            self._replay_buffer.append(replay_entry)

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction video + ILR-specific diagnostics.

        Extends v1f logging with:
        - Pass 2 reconstruction quality comparison
        - Refinement sigma distribution
        """
        # Parent logs reconstruction_video + trajectory PCA + spherical plots
        log_dict = super().log_reconstructions_to_wandb(
            video_pred=video_pred, inputs=inputs, step=step,
            vae_decoder=vae_decoder, prefix=prefix,
        )

        # Add ILR-specific trajectory extension if we have pass 2 data
        if (
            self.config.ilr_enabled
            and step >= self.config.ilr_warmup_steps
            and self._transformer_ref_for_ilr is not None
            and getattr(inputs, "_vfm_use_adapter", False)
        ):
            try:
                log_dict.update(
                    self._build_ilr_comparison(video_pred, inputs, step, prefix)
                )
            except Exception as e:
                logger.warning(f"Failed to build ILR comparison: {e}")

        return log_dict

    def _build_ilr_comparison(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Build comparison plot of pass 1 vs pass 2 reconstruction errors.

        Shows per-token MSE heatmap for both passes, highlighting where
        refinement helped most.
        """
        try:
            import wandb
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            video_noise = inputs._vfm_video_noise
            video_latents = inputs._vfm_video_latents

            # Pass 1 reconstruction
            x0_hat1 = video_noise - video_pred
            pass1_error = (x0_hat1 - video_latents).pow(2).mean(dim=-1)  # [B, seq]

            # Run pass 2 for visualization (no grad)
            with torch.no_grad():
                x0_hat1_det = x0_hat1.detach()
                pass2_pred = self._run_refinement_pass(
                    x0_hat1_det, inputs, self.config.ilr_refine_sigma,
                )
                x0_hat2 = x0_hat1_det - pass2_pred
                pass2_error = (x0_hat2 - video_latents).pow(2).mean(dim=-1)  # [B, seq]

            # Take first sample
            p1_err = pass1_error[0].detach().cpu().numpy()
            p2_err = pass2_error[0].detach().cpu().numpy()
            improvement_map = p1_err - p2_err  # positive = pass 2 was better

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Pass 1 Error", "Pass 2 Error", "Improvement (P1-P2)"],
            )

            # Reshape to approximate spatial layout
            seq_len = len(p1_err)
            h = int(seq_len ** 0.5)
            w = seq_len // h if h > 0 else seq_len
            if h * w < seq_len:
                w += 1

            import numpy as np
            def pad_reshape(arr, h, w):
                padded = np.zeros(h * w)
                padded[:len(arr)] = arr
                return padded.reshape(h, w)

            fig.add_trace(
                go.Heatmap(z=pad_reshape(p1_err, h, w), colorscale="Reds", name="Pass 1"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Heatmap(z=pad_reshape(p2_err, h, w), colorscale="Reds", name="Pass 2"),
                row=1, col=2,
            )
            fig.add_trace(
                go.Heatmap(z=pad_reshape(improvement_map, h, w), colorscale="RdBu", name="Improvement"),
                row=1, col=3,
            )

            fig.update_layout(
                title=f"ILR Error Comparison (step {step})",
                height=300,
                width=900,
            )

            return {f"{prefix}/ilr_comparison": wandb.Plotly(fig)}

        except ImportError:
            return {}
        except Exception as e:
            logger.warning(f"ILR comparison plot failed: {e}")
            return {}
