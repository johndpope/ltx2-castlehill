"""VFM v3a — Adversarial Distribution Matching for 1-step Video Generation (DMD2/FlashMotion).

Extends v1f (Spherical Cauchy + per-token sigma) with a GAN discriminator that
operates in noisy latent space (DMD2 approach). Instead of per-sample reconstruction
loss alone, the student receives adversarial gradients that push its output
DISTRIBUTION toward the teacher's.

Key insight (DMD2, Yin et al. 2024; FlashMotion, CVPR 2026):
- No separate "fake score network" needed
- A lightweight discriminator (~18M params) in noisy latent space is sufficient
- Teacher provides real samples via multi-step ODE
- Student provides fake samples via 1-step flow map
- Both are noised at random t_critic before discrimination

Architecture:
    Text embeddings -> NoiseAdapterV1b -> Spherical Cauchy noise z  (from v1f)
    z -> 48-layer DiT (student, LoRA rank 32) -> velocity v -> x_student
    z -> 48-layer DiT (teacher, LoRA disabled) -> 8-step ODE -> x_teacher
    D(noisy_real, t_critic) vs D(noisy_fake, t_critic) -> GAN loss

Loss = gan_g_weight * L_G + recon_weight * L_recon + kl_weight * L_KL
"""

from __future__ import annotations

from dataclasses import replace
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


class DMDVFMv3aTrainingConfig(VFMv1fTrainingConfig):
    """Configuration for VFM v3a (adversarial DMD + VFM)."""

    name: Literal["vfm_v3a"] = "vfm_v3a"

    # === GAN / discriminator settings ===
    gan_g_weight: float = Field(
        default=0.01, ge=0.0,
        description="Weight for generator (student) GAN loss.",
    )
    gan_d_weight: float = Field(
        default=1.0, ge=0.0,
        description="Weight for discriminator loss (internal, not added to main loss).",
    )
    disc_update_ratio: int = Field(
        default=1, ge=1,
        description="Update discriminator every N steps. Generator GAN loss every disc_gen_ratio steps.",
    )
    disc_gen_ratio: int = Field(
        default=5, ge=1,
        description="Apply generator GAN loss every N discriminator steps "
        "(FlashMotion default: 5).",
    )
    critic_t_min: float = Field(
        default=0.02, ge=0.0, le=1.0,
        description="Minimum critic timestep for noisy discrimination.",
    )
    critic_t_max: float = Field(
        default=0.98, ge=0.0, le=1.0,
        description="Maximum critic timestep for noisy discrimination.",
    )

    # === Discriminator architecture ===
    disc_lr: float = Field(
        default=2e-6, gt=0.0,
        description="Learning rate for discriminator optimizer.",
    )
    disc_hidden_dim: int = Field(
        default=512, ge=64,
        description="Hidden dimension for discriminator transformer.",
    )
    disc_num_heads: int = Field(
        default=8, ge=1,
        description="Number of attention heads in discriminator.",
    )
    disc_num_layers: int = Field(
        default=4, ge=1,
        description="Number of transformer layers in discriminator.",
    )
    disc_num_registers: int = Field(
        default=4, ge=1,
        description="Number of learnable register tokens in discriminator.",
    )

    # === Reconstruction regularizer ===
    recon_weight: float = Field(
        default=0.1, ge=0.0,
        description="Weight for reconstruction regularizer (decays over recon_decay_steps).",
    )
    recon_decay_steps: int = Field(
        default=1000, ge=0,
        description="Steps to linearly decay recon_weight to 0. 0 = no decay.",
    )

    # === Flow Distribution Matching (temporal consistency) ===
    flow_match_weight: float = Field(
        default=4.0, ge=0.0,
        description="Weight for flow distribution matching loss. "
        "Matches frame-to-frame latent diffs between student and GT. "
        "From DiagDistill (ICLR 2026). 0 = disabled.",
    )

    # === Teacher ODE / precomputed ===
    teacher_num_steps: int = Field(
        default=8, ge=2,
        description="Number of Euler ODE steps for teacher denoising.",
    )
    teacher_latents_dir: str = Field(
        default="teacher_latents",
        description="Directory name (under data_root) with precomputed teacher outputs. "
        "Created by precompute_teacher_latents.py. If not found, uses GT x0.",
    )

    # Keep these for backward compat with config.py but they're unused
    fake_score_lr: float = Field(default=1e-4, description="Unused (kept for config compat).")
    fake_score_hidden_dim: int = Field(default=256, description="Unused.")
    fake_score_num_heads: int = Field(default=4, description="Unused.")
    fake_score_num_layers: int = Field(default=4, description="Unused.")
    dmd_sigma: float = Field(default=0.5, description="Unused.")
    dmd_weight: float = Field(default=1.0, description="Unused.")
    score_update_ratio: int = Field(default=1, description="Unused.")
    cache_teacher_outputs: bool = Field(default=False, description="Unused.")


class DMDVFMv3aTrainingStrategy(VFMv1fTrainingStrategy):
    """VFM v3a — Adversarial DMD for 1-step video generation.

    Uses a lightweight latent discriminator (DMD2/FlashMotion approach) instead of
    a separate fake score network. The discriminator classifies noisy teacher vs
    student outputs, and the generator receives adversarial gradients.

    Requires (set by trainer.py):
    - set_transformer(): reference to transformer for teacher ODE
    - set_fake_score_network(): actually receives the discriminator (reuses same API)
    - set_fake_score_optimizer(): discriminator optimizer
    """

    config: DMDVFMv3aTrainingConfig

    def __init__(self, config: DMDVFMv3aTrainingConfig):
        super().__init__(config)
        self._transformer_ref: nn.Module | None = None
        self._discriminator: nn.Module | None = None
        self._disc_optimizer: torch.optim.Optimizer | None = None
        self._grad_accumulation_steps: int = 1
        self._teacher_cache: dict[str, Tensor] = {}
        self._teacher_cache_built: bool = False
        self._motion_head_student: nn.Module | None = None
        self._motion_head_teacher: nn.Module | None = None

    def set_transformer(self, transformer: nn.Module, grad_accumulation_steps: int = 1) -> None:
        """Store transformer reference for teacher ODE passes."""
        self._transformer_ref = transformer
        self._grad_accumulation_steps = grad_accumulation_steps
        logger.info(f"VFM v3a: Transformer reference set (grad_accum={grad_accumulation_steps})")

    def set_fake_score_network(self, network: nn.Module) -> None:
        """Store discriminator (reuses fake_score API from trainer)."""
        self._discriminator = network
        n = sum(p.numel() for p in network.parameters())
        logger.info(f"VFM v3a: Discriminator set ({n:,} params)")

    def set_fake_score_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Store discriminator optimizer."""
        self._disc_optimizer = optimizer
        logger.info("VFM v3a: Discriminator optimizer set")

    def get_fake_score_params(self) -> dict[str, Any]:
        """Return params for trainer to create the discriminator.

        NOTE: Trainer creates a FakeScoreNetwork with these params, but we
        actually want a LatentDiscriminator. We override set_fake_score_network
        to accept either. The trainer will need to be updated to create the
        correct class for v3a.
        """
        cfg = self.config
        return {
            "latent_dim": 128,
            "hidden_dim": cfg.disc_hidden_dim,
            "num_heads": cfg.disc_num_heads,
            "num_layers": cfg.disc_num_layers,
            "text_dim": 4096,
        }

    def get_discriminator_params(self) -> dict[str, Any]:
        """Return params for the LatentDiscriminator (used by updated trainer)."""
        cfg = self.config
        return {
            "latent_dim": 128,
            "hidden_dim": cfg.disc_hidden_dim,
            "num_heads": cfg.disc_num_heads,
            "num_layers": cfg.disc_num_layers,
            "num_registers": cfg.disc_num_registers,
            "text_dim": 4096,
        }

    def _run_teacher_ode(
        self,
        z: Tensor,
        text_embeds: Tensor,
        text_mask: Tensor | None,
        positions: Tensor,
        num_steps: int,
    ) -> Tensor:
        """Run teacher's multi-step Euler ODE (LoRA disabled)."""
        if self._transformer_ref is None:
            raise RuntimeError("VFM v3a: transformer reference not set.")

        from ltx_core.components.schedulers import LTX2Scheduler  # noqa: PLC0415
        from ltx_core.model.transformer.modality import Modality  # noqa: PLC0415
        from ltx_core.model.transformer.timestep_embedding import get_timestep_embedding  # noqa: PLC0415

        device = z.device
        dtype = z.dtype
        B = z.shape[0]
        seq_len = z.shape[1]

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=num_steps).to(device=device)

        transformer = self._transformer_ref
        self._toggle_lora(transformer, enable=False)

        try:
            x_t = z.clone()
            for i in range(len(sigmas) - 1):
                sc = sigmas[i]
                sn = sigmas[i + 1]
                dt = sn - sc

                sb = sc.unsqueeze(0).expand(B)
                ts = sb.unsqueeze(1).expand(-1, seq_len)  # per-token sigma values

                video_mod = Modality(
                    enabled=True, latent=x_t, sigma=sb, timesteps=ts,
                    positions=positions, context=text_embeds, context_mask=text_mask,
                )

                result = transformer(video=video_mod, audio=None, perturbations=None)
                vo = result[0] if isinstance(result, tuple) else result
                v = vo.x if hasattr(vo, 'x') else vo
                x_t = x_t + v * dt
        finally:
            self._toggle_lora(transformer, enable=True)

        return x_t

    @staticmethod
    def _toggle_lora(model: nn.Module, enable: bool) -> None:
        """Enable/disable LoRA adapter layers."""
        try:
            if enable:
                if hasattr(model, "enable_adapter_layers"):
                    model.enable_adapter_layers()
                else:
                    from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: PLC0415
                    for m in model.modules():
                        if isinstance(m, BaseTunerLayer):
                            m.enable_adapters(True)
            else:
                if hasattr(model, "disable_adapter_layers"):
                    model.disable_adapter_layers()
                else:
                    from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: PLC0415
                    for m in model.modules():
                        if isinstance(m, BaseTunerLayer):
                            m.enable_adapters(False)
        except Exception as e:
            logger.warning(f"VFM v3a: LoRA toggle failed: {e}")

    def _get_teacher_output(
        self,
        z: Tensor,
        x0: Tensor,
        text_embeds: Tensor,
        text_mask: Tensor | None,
        positions: Tensor,
    ) -> Tensor:
        """Get teacher output for discriminator's "real" samples.

        Priority:
        1. Precomputed teacher latents (from precompute_teacher_latents.py)
        2. Ground truth x0 (always available, simplest)
        3. Live teacher ODE (expensive, 8 DiT passes)

        For DMD2, using GT x0 as "real" is valid — the discriminator learns
        data distribution vs student distribution. Precomputed teacher latents
        are an alternative that matches the teacher's output distribution
        specifically (useful if teacher != data, e.g., distilled model).
        """
        # Check for precomputed teacher cache loaded into batch
        teacher_cached = getattr(self, "_batch_teacher_output", None)
        if teacher_cached is not None:
            self._batch_teacher_output = None  # consume
            return teacher_cached.detach()

        # Default: use GT x0 as real (FlashMotion approach)
        return x0.detach()

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute adversarial DMD loss (DMD2/FlashMotion style).

        Phase 1: Discriminator update (real vs fake in noisy latent space)
        Phase 2: Generator loss (GAN + reconstruction + KL)
        """
        cfg = self.config
        step = self._current_step
        device = video_pred.device
        use_adapter = getattr(inputs, "_vfm_use_adapter", False)

        # Fall back to v1f if not ready
        if (
            not use_adapter
            or self._transformer_ref is None
            or self._discriminator is None
            or self._disc_optimizer is None
        ):
            return super().compute_loss(video_pred, audio_pred, inputs)

        video_noise = inputs._vfm_video_noise  # z (adapter noise)
        video_latents = inputs._vfm_video_latents  # x0 (ground truth)
        text_embeds = inputs.video.context
        text_mask = inputs.video.context_mask
        positions = inputs.video.positions

        # Student prediction: x_student = z - v_pred (flow matching convention)
        x_student = video_noise - video_pred  # [B, seq, 128]

        # ════════════════════════════════════════════════════════════
        # PHASE 1: Get teacher output (cached or live ODE)
        # ════════════════════════════════════════════════════════════
        # For small datasets (overfit mode), we can cache teacher outputs
        # keyed by the ground truth x0 hash (adapter noise z varies each step,
        # but we use x0 as the "real" reference for the discriminator instead)
        x_teacher = self._get_teacher_output(
            video_noise, video_latents, text_embeds, text_mask, positions
        )

        # Random critic timestep
        t_critic = torch.rand(1, device=device) * (cfg.critic_t_max - cfg.critic_t_min) + cfg.critic_t_min

        # Noise both at critic timestep (SAME noise for fair comparison)
        eps = torch.randn_like(x_student)
        noisy_real = (1 - t_critic) * x_teacher.detach() + t_critic * eps
        noisy_fake = (1 - t_critic) * x_student.detach() + t_critic * eps

        # Discriminator forward (ensure consistent dtype)
        disc_dtype = next(self._discriminator.parameters()).dtype
        D_real = self._discriminator(noisy_real.to(disc_dtype), t_critic.expand(x_student.shape[0]).to(disc_dtype), text_embeds.detach().to(disc_dtype))
        D_fake = self._discriminator(noisy_fake.to(disc_dtype), t_critic.expand(x_student.shape[0]).to(disc_dtype), text_embeds.detach().to(disc_dtype))

        # Discriminator loss (softplus for stability)
        loss_D = F.softplus(-D_real).mean() + F.softplus(D_fake).mean()

        # Manual backward for discriminator only
        self._disc_optimizer.zero_grad()
        (cfg.gan_d_weight * loss_D).backward(retain_graph=False)
        self._disc_optimizer.step()

        loss_D_val = loss_D.detach()

        # ════════════════════════════════════════════════════════════
        # PHASE 2: Generator (student) loss
        # ════════════════════════════════════════════════════════════
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_G_val = torch.tensor(0.0, device=device)

        # GAN loss for generator (every disc_gen_ratio steps)
        if step % cfg.disc_gen_ratio == 0 and cfg.gan_g_weight > 0:
            # Re-noise student WITH gradient (same t_critic, fresh noise)
            eps_g = torch.randn_like(x_student)
            noisy_fake_grad = (1 - t_critic) * x_student + t_critic * eps_g
            D_fake_grad = self._discriminator(
                noisy_fake_grad.to(disc_dtype), t_critic.expand(x_student.shape[0]).to(disc_dtype), text_embeds.detach().to(disc_dtype)
            )
            loss_G = F.softplus(-D_fake_grad).mean()
            total_loss = total_loss + cfg.gan_g_weight * loss_G
            loss_G_val = loss_G.detach()

        # ════════════════════════════════════════════════════════════
        # RECONSTRUCTION REGULARIZER (decaying)
        # ════════════════════════════════════════════════════════════
        loss_recon = torch.tensor(0.0, device=device)
        effective_recon_weight = cfg.recon_weight
        if cfg.recon_decay_steps > 0 and step > 0:
            effective_recon_weight *= max(0.0, 1.0 - step / cfg.recon_decay_steps)

        if effective_recon_weight > 0:
            if inputs.video_loss_mask is not None:
                mask = inputs.video_loss_mask.unsqueeze(-1).float()
                loss_recon = ((x_student - video_latents).pow(2) * mask).sum() / mask.sum().clamp(min=1) / x_student.shape[-1]
            else:
                loss_recon = (x_student - video_latents).pow(2).mean()
            total_loss = total_loss + effective_recon_weight * loss_recon

        # ════════════════════════════════════════════════════════════
        # FLOW DISTRIBUTION MATCHING (temporal consistency)
        # ════════════════════════════════════════════════════════════
        # DiagDistill (ICLR 2026) + improved SpatialHead:
        # Extract motion features via learned conv head, match distributions.
        # Confidence-weighted loss focuses on regions with meaningful motion.
        loss_flow = torch.tensor(0.0, device=device)
        flow_weight = getattr(cfg, "flow_match_weight", 4.0)

        if flow_weight > 0:
            try:
                # Infer spatial dims from positions [B, 3, seq, 2]
                t_coords = positions[0, 0, :, 0]
                latent_f = int(t_coords.max().item()) + 1

                if latent_f > 1:
                    B_s = x_student.shape[0]
                    C = x_student.shape[-1]
                    tokens_per_frame = x_student.shape[1] // latent_f

                    # Infer H, W from tokens_per_frame
                    h_coords = positions[0, 1, :tokens_per_frame, 0]
                    latent_h = int(h_coords.max().item()) + 1
                    latent_w = tokens_per_frame // latent_h

                    # Reshape to spatial: [B, F, C, H, W]
                    x_s_5d = x_student.reshape(B_s, latent_f, latent_h, latent_w, C)
                    x_s_5d = x_s_5d.permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]
                    x_gt_5d = video_latents.reshape(B_s, latent_f, latent_h, latent_w, C)
                    x_gt_5d = x_gt_5d.permute(0, 1, 4, 2, 3)

                    # Frame diffs = raw motion signal [B, F-1, C, H, W]
                    student_diffs = x_s_5d[:, 1:] - x_s_5d[:, :-1]
                    gt_diffs = x_gt_5d[:, 1:] - x_gt_5d[:, :-1]

                    # Lazy-init SpatialHead on first use
                    if self._motion_head_student is None:
                        from ltx_core.model.transformer.spatial_head import SpatialHead  # noqa: PLC0415
                        self._motion_head_student = SpatialHead(
                            num_channels=C, num_layers=3, hidden_dim=128,
                            predict_confidence=True,
                        ).to(device=device, dtype=student_diffs.dtype)
                        # Teacher head: EMA copy (detached)
                        import copy  # noqa: PLC0415
                        self._motion_head_teacher = copy.deepcopy(self._motion_head_student)
                        self._motion_head_teacher.requires_grad_(False)
                        n = sum(p.numel() for p in self._motion_head_student.parameters())
                        logger.info(f"VFM v3a: SpatialHead created ({n:,} params, confidence=True)")
                        # Add to disc optimizer (reuse existing optimizer for simplicity)
                        if self._disc_optimizer is not None:
                            self._disc_optimizer.add_param_group({
                                "params": list(self._motion_head_student.parameters()),
                                "lr": self._disc_optimizer.param_groups[0]["lr"] * 10,  # higher LR for small head
                            })

                    # Student motion features + confidence
                    student_feats, confidence = self._motion_head_student(student_diffs)
                    # Teacher motion features (detached, EMA-updated)
                    with torch.no_grad():
                        teacher_feats = self._motion_head_teacher(gt_diffs.detach())
                        if isinstance(teacher_feats, tuple):
                            teacher_feats = teacher_feats[0]

                    # Confidence-weighted flow regression loss
                    flow_error = (student_feats - teacher_feats.detach()).pow(2)
                    loss_flow = (flow_error * confidence).mean()

                    # EMA update teacher motion head
                    ema_decay = 0.999
                    with torch.no_grad():
                        for p_s, p_t in zip(
                            self._motion_head_student.parameters(),
                            self._motion_head_teacher.parameters(),
                        ):
                            p_t.data.mul_(ema_decay).add_(p_s.data, alpha=1 - ema_decay)

                    total_loss = total_loss + flow_weight * loss_flow
            except Exception as e:
                if step % 100 == 0:
                    logger.warning(f"VFM v3a: Flow loss skipped: {e}")

        # ════════════════════════════════════════════════════════════
        # ADAPTER KL + other v1f losses (inherited)
        # ════════════════════════════════════════════════════════════
        # Get KL from adapter (Spherical Cauchy prior)
        adapter_kl = getattr(inputs, "_vfm_kl_loss", torch.tensor(0.0, device=device))
        if cfg.kl_weight > 0:
            total_loss = total_loss + cfg.kl_weight * adapter_kl

        # ════════════════════════════════════════════════════════════
        # LOGGING
        # ════════════════════════════════════════════════════════════
        if step % 20 == 0:
            try:
                import wandb  # noqa: PLC0415
                if wandb.run is not None:
                    wandb.log({
                        "v3a/loss_D": loss_D_val.item(),
                        "v3a/loss_G": loss_G_val.item(),
                        "v3a/loss_recon": loss_recon.item() if isinstance(loss_recon, Tensor) else loss_recon,
                        "v3a/loss_flow": loss_flow.item() if isinstance(loss_flow, Tensor) else loss_flow,
                        "v3a/loss_kl": adapter_kl.item() if isinstance(adapter_kl, Tensor) else adapter_kl,
                        "v3a/D_real": D_real.mean().item(),
                        "v3a/D_fake": D_fake.mean().item(),
                        "v3a/t_critic": t_critic.item(),
                        "v3a/recon_weight_effective": effective_recon_weight,
                        "v3a/student_recon_mse": (x_student - video_latents).pow(2).mean().item(),
                    }, step=step)
            except Exception:
                pass

        return total_loss
