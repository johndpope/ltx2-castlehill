"""VFM v5a -- Flow-GRPO Reinforcement Learning for LTX-2.3 SCD.

Applies World-R1 / Flow-GRPO online RL to the SCD decoder LoRA.

Training loop (per step):
  1. SAMPLE: Autoregressively generate K videos using current decoder policy.
     At each denoising step, the SDE transition log-probability is captured so
     we can form the PPO importance ratio later.  The encoder runs once per
     frame (clean, sigma=0) and its features are treated as fixed conditioning.
  2. SCORE: Evaluate each video with reward functions:
       - R_aesthetic: HPSv2 (local or via World-R1 Flask server) on sampled frames
       - R_temporal:  Mean SSIM across consecutive decoded frames [SCD-specific,
                      penalizes AR chunk boundary artifacts]
  3. ADVANTAGE: GRPO normalization per prompt group (subtract mean, divide by std).
  4. UPDATE: Policy gradient loss using stored trajectory data:
       L_pg = -A(tau) * ratio  (PPO clipped)
       L_kl = KL(pi_theta || pi_ref)  (reference regularizer)
       L_sl = ||v_theta - v_target||^2  (supervised anchor)
       L = L_pg + beta * L_kl + sl_weight * L_sl

Key design choices vs World-R1:
  - Log-prob is the *correct SDE transition density* (Gaussian log-prob of
    x_{t-1} given the deterministic mean + injected noise), NOT the velocity
    magnitude proxy that the old code used.
  - R_temporal is novel: SSIM across AR chunk boundaries directly targets
    SCD's main failure mode (temporal seams between generated chunks).
  - Aesthetic reward supports local (in-process hpsv2) AND server mode (HTTP
    POST to World-R1's Flask reward server).
  - Reference model is a snapshot of LoRA weights taken at the start of each
    GRPO outer step.  KL is computed by temporarily swapping weights.

References:
  - World-R1: arXiv:2604.24764
  - Flow-GRPO: arXiv:2405.20673 (Diffusion-DPO successor)
  - GRPO: arXiv:2402.03300 (DeepSeek-R1)
"""

from __future__ import annotations

import gc
import math
import pickle
from io import BytesIO
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
)
from ltx_core.types import VideoLatentShape
from ltx_core.model.transformer.modality import Modality
from ltx_trainer.training_strategies.scd_strategy import (
    SCDTrainingConfig,
    SCDTrainingStrategy,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GRPOv5aConfig(SCDTrainingConfig):
    """Configuration for VFM v5a Flow-GRPO RL training."""

    name: Literal["vfm_v5a"] = "vfm_v5a"

    # --- GRPO rollout ---
    grpo_num_samples: int = Field(
        default=2,
        description="K: number of videos sampled per prompt per GRPO step.",
        ge=2,
    )

    # --- Reward mode ---
    reward_mode: Literal["local", "server"] = Field(
        default="local",
        description="Aesthetic reward backend: 'local' (in-process HPSv2) or "
                    "'server' (HTTP to World-R1's reward server).",
    )
    reward_server_url: str = Field(
        default="http://127.0.0.1:8090",
        description="URL for remote reward server (used when reward_mode='server').",
    )

    # --- Inference params for rollout ---
    num_inference_steps: int = Field(
        default=8,
        description="Denoising steps per frame during rollout.",
        ge=1,
    )
    guidance_scale: float = Field(
        default=1.0,
        description="CFG scale during rollout (1.0 = disabled).",
    )
    encoder_layers: int = Field(
        default=32,
        description="Encoder layers (must match the SCD model split).",
    )
    fps: float = Field(default=24.0, description="Frames per second for rollout videos.")
    resolution_h: int = Field(default=448, description="Rollout video height (must be divisible by 32).")
    resolution_w: int = Field(default=768, description="Rollout video width (must be divisible by 32).")

    # --- PPO ---
    ppo_epsilon: float = Field(
        default=0.2,
        description="PPO clipping parameter epsilon.",
    )
    ppo_adv_clip: float = Field(
        default=5.0,
        description="Hard clip on advantage magnitude before PG loss.",
    )

    # --- KL regularization ---
    kl_beta: float = Field(
        default=0.01,
        description="KL penalty weight. Keeps policy close to reference LoRA.",
    )

    # --- Supervised anchor ---
    sl_weight: float = Field(
        default=0.1,
        description="Supervised learning anchor weight. Prevents reward hacking.",
        ge=0.0,
    )

    # --- Rewards ---
    reward_aesthetic_weight: float = Field(
        default=1.0,
        description="Weight for aesthetic/quality reward.",
        ge=0.0,
    )
    reward_temporal_weight: float = Field(
        default=1.0,
        description="Weight for temporal consistency reward (SSIM across chunk boundary).",
        ge=0.0,
    )
    reward_temporal_frames: int = Field(
        default=4,
        description="Number of frames around each AR chunk boundary for SSIM.",
        ge=2,
    )

    # --- GRPO stat tracker ---
    grpo_global_std: bool = Field(
        default=False,
        description="If True, normalize advantages by global reward std instead of per-prompt.",
    )

    # --- Rollout video params ---
    rollout_num_seconds: float = Field(
        default=5.0,
        description="Duration (seconds) of rollout videos. Longer = more AR chunks.",
    )

    # --- Sparse supervision (SSD, arXiv:2602.02699) ---
    sparse_mask_ratio: float = Field(
        default=0.98,
        description="Fraction of velocity tokens masked for SL anchor loss (0=disabled, 0.98=paper default).",
        ge=0.0,
        lt=1.0,
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _build_temporal_reward(num_boundary_frames: int) -> Callable:
    """SSIM-based temporal consistency reward across AR chunk boundaries.

    Given decoded frames [T, C, H, W] float [0,1], computes mean SSIM between
    frame pairs around each chunk boundary.  High SSIM = smooth transitions =
    good AR continuity.

    This reward directly targets SCD's main failure mode: visible seams where
    one AR chunk ends and the next begins (encoder context switch).
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    except ImportError:
        ssim_fn = None

    def _fn(video_frames: Tensor, chunk_boundaries: list[int]) -> float:
        if ssim_fn is None or not chunk_boundaries:
            return 0.0
        scores: list[float] = []
        for b in chunk_boundaries:
            lo = max(0, b - num_boundary_frames // 2)
            hi = min(video_frames.shape[0] - 1, b + num_boundary_frames // 2)
            for t in range(lo, hi):
                f1 = video_frames[t].unsqueeze(0)   # [1, C, H, W]
                f2 = video_frames[t + 1].unsqueeze(0)
                try:
                    s = ssim_fn(f1, f2, data_range=1.0).item()
                    scores.append(s)
                except Exception:
                    pass
        return float(sum(scores) / len(scores)) if scores else 0.0

    return _fn


# ---------------------------------------------------------------------------
# Per-prompt GRPO advantage tracker (ported from World-R1)
# ---------------------------------------------------------------------------

class PerPromptStatTracker:
    """Tracks per-prompt reward statistics for GRPO advantage normalization."""

    def __init__(self, global_std: bool = False):
        self.global_std = global_std
        self.stats: dict[str, list[float]] = {}

    def update(self, prompts: list[str], rewards: list[float]) -> list[float]:
        import numpy as np
        p = np.array(prompts)
        r = np.array(rewards, dtype=np.float64)
        advantages = np.zeros_like(r)
        for prompt in np.unique(p):
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(r[p == prompt].tolist())
            mean = np.mean(self.stats[prompt])
            std = (np.std(r) if self.global_std else np.std(self.stats[prompt])) + 1e-4
            advantages[p == prompt] = (r[p == prompt] - mean) / std
        return advantages.tolist()

    def clear(self) -> None:
        self.stats.clear()


# ---------------------------------------------------------------------------
# SDE transition log-probability
# ---------------------------------------------------------------------------

def gaussian_log_prob(sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Gaussian log-probability of *sample* under N(mean, std^2).

    Returns a scalar per batch element (mean over seq and channel dims).

    Args:
        sample:  [B, L, C] observed transition outcome (detached).
        mean:    [B, L, C] deterministic part of the SDE step.
        std:     scalar or broadcastable standard deviation.

    Returns:
        [B] log-probability per batch element.
    """
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * std ** 2)
        - torch.log(std)
        - 0.5 * math.log(2 * math.pi)
    )
    # Average over all non-batch dimensions (seq, channel) so that the
    # magnitude is invariant to latent spatial resolution.
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def sde_step_with_logprob(
    velocity: Tensor,
    noisy_patch: Tensor,
    sigma: float | Tensor,
    sigma_next: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """SDE denoising step with log-probability capture.

    The deterministic part is the standard Euler step:
        x_next_mean = x_t + dt * v_theta(x_t, sigma)

    Stochasticity is injected additively:
        x_next = x_next_mean + noise_std * eps,   eps ~ N(0, I)

    where noise_std = sigma * sqrt(|dt|), matching the reverse-SDE
    discretisation used in flow matching.

    Args:
        velocity:    [B, L, C] predicted velocity from the decoder.
        noisy_patch: [B, L, C] current noisy state x_t.
        sigma:       current noise level (scalar or 0-d tensor).
        sigma_next:  next noise level (scalar or 0-d tensor).

    Returns:
        (x_next, log_prob, x_next_mean, noise_std)
    """
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=noisy_patch.device)
    sigma_nt = torch.as_tensor(sigma_next, dtype=torch.float32, device=noisy_patch.device)
    dt = sigma_nt - sigma_t  # negative (denoising)

    x_next_mean = noisy_patch.float() + dt * velocity.float()
    noise_std = (sigma_t * torch.abs(dt).sqrt()).clamp(min=1e-8)
    noise = torch.randn_like(noisy_patch)
    x_next = x_next_mean + noise_std * noise

    log_prob = gaussian_log_prob(x_next.detach(), x_next_mean, noise_std)
    return x_next, log_prob, x_next_mean, noise_std


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class GRPOv5aTrainingStrategy(SCDTrainingStrategy):
    """VFM v5a -- Flow-GRPO RL post-training for LTX-2.3 SCD.

    Inherits the SCD forward pass (encoder + decoder split, per-frame decoder,
    KV-cache).  Adds online rollout -> reward -> GRPO advantage -> policy
    gradient.

    The policy is the SCD decoder LoRA.  The reference policy is a snapshot of
    the same LoRA taken at the start of each GRPO outer step (frozen copy).

    The key difference from the old v5a:
      - Log-prob is the *correct* SDE transition density (Gaussian in x-space),
        NOT the velocity-magnitude approximation.
      - rollout() stores full trajectory data (x_t, sigma, x_next, log_prob, ...)
        on CPU so that compute_log_prob() can re-evaluate under the current
        policy for the PPO ratio.
      - compute_pg_loss() re-runs the decoder on stored trajectory data to get
        new log-probs, forms the clipped PPO ratio, and optionally computes KL
        against the reference LoRA.
    """

    config: GRPOv5aConfig

    def __init__(self, config: GRPOv5aConfig):
        super().__init__(config)
        self._stat_tracker = PerPromptStatTracker(global_std=config.grpo_global_std)
        self._temporal_reward_fn = _build_temporal_reward(config.reward_temporal_frames)
        self._ref_state_dict: dict[str, Tensor] = {}
        logger.info(
            f"VFM v5a: Flow-GRPO | K={config.grpo_num_samples} samples/prompt | "
            f"beta={config.kl_beta} | eps={config.ppo_epsilon} | "
            f"rewards=[{config.reward_mode}, temporal]"
        )

    # ------------------------------------------------------------------
    # Reference model management
    # ------------------------------------------------------------------

    @staticmethod
    def snapshot_lora_state(scd_model: torch.nn.Module) -> dict[str, Tensor]:
        """Snapshot current LoRA weights for reference policy."""
        state: dict[str, Tensor] = {}
        for name, param in scd_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                state[name] = param.data.clone()
        return state

    @staticmethod
    def _load_reference_state(
        scd_model: torch.nn.Module,
        ref_state: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Temporarily load reference LoRA weights.  Returns originals."""
        orig: dict[str, Tensor] = {}
        for name, param in scd_model.named_parameters():
            if name in ref_state:
                orig[name] = param.data.clone()
                param.data.copy_(ref_state[name])
        return orig

    @staticmethod
    def _restore_current_state(
        scd_model: torch.nn.Module,
        orig_state: dict[str, Tensor],
    ) -> None:
        """Restore trainable LoRA weights from a prior snapshot."""
        for name, param in scd_model.named_parameters():
            if name in orig_state:
                param.data.copy_(orig_state[name])

    # ------------------------------------------------------------------
    # Aesthetic reward (dual mode: local or server)
    # ------------------------------------------------------------------

    def _build_aesthetic_reward(
        self,
        video_frames: Tensor,
        prompt: str,
        device: str,
    ) -> float:
        """Score video aesthetics using local HPSv2 or remote server.

        Args:
            video_frames: [T, C, H, W] float [0, 1] pixel frames.
            prompt: text prompt.
            device: device string (used for local model placement).

        Returns:
            Scalar aesthetic score.
        """
        if self.config.reward_mode == "server":
            return self._remote_aesthetic_reward(video_frames, prompt)
        return self._local_aesthetic_reward(video_frames, prompt, device)

    def _local_aesthetic_reward(self, video_frames: Tensor, prompt: str, device: str) -> float:
        """In-process HPSv2 scoring of the middle frame."""
        try:
            import hpsv2
            mid = video_frames[video_frames.shape[0] // 2]
            img = (mid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            from PIL import Image
            pil_img = Image.fromarray(img)
            score = hpsv2.score(pil_img, prompt, hps_version="v2.1")[0]
            return float(score)
        except ImportError:
            logger.warning("hpsv2 not installed. Aesthetic reward returning 0.")
            return 0.0
        except Exception as e:
            logger.warning(f"Local aesthetic reward failed: {e}")
            return 0.0

    def _remote_aesthetic_reward(self, video_frames: Tensor, prompt: str) -> float:
        """HTTP POST to World-R1's reward server (HPSv2 over Flask).

        Pickles the (frames, prompt) pair and POSTs it to the configured URL.
        """
        try:
            import requests
            buf = BytesIO()
            pickle.dump({"frames": video_frames.cpu(), "prompt": prompt}, buf)
            buf.seek(0)
            resp = requests.post(
                f"{self.config.reward_server_url}/score_aesthetic",
                data=buf.getvalue(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=30,
            )
            resp.raise_for_status()
            return float(resp.json()["score"])
        except ImportError:
            logger.warning("requests not installed. Remote aesthetic reward returning 0.")
            return 0.0
        except Exception as e:
            logger.warning(f"Remote aesthetic reward failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        decoded_frames: Tensor,
        prompt: str,
        chunk_boundaries: list[int],
        device: str = "cpu",
    ) -> dict[str, float]:
        """Compute total reward for one generated video.

        Args:
            decoded_frames: [T, C, H, W] float [0,1] pixel frames.
            prompt: text prompt used for generation.
            chunk_boundaries: frame indices where AR chunks start.
            device: device string for aesthetic model.

        Returns:
            dict with 'aesthetic', 'temporal', 'total' reward values.
        """
        r_aesthetic = 0.0
        if self.config.reward_aesthetic_weight > 0:
            r_aesthetic = self._build_aesthetic_reward(decoded_frames, prompt, device)

        r_temporal = 0.0
        if self.config.reward_temporal_weight > 0 and chunk_boundaries:
            r_temporal = self._temporal_reward_fn(decoded_frames, chunk_boundaries)

        total = (
            self.config.reward_aesthetic_weight * r_aesthetic
            + self.config.reward_temporal_weight * r_temporal
        )
        return {"aesthetic": r_aesthetic, "temporal": r_temporal, "total": total}

    # ------------------------------------------------------------------
    # GRPO advantage computation (for standalone script)
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        trajectories: list[dict[str, Any]],
    ) -> Tensor:
        """Compute GRPO-normalized advantages from rollout trajectories.

        Each trajectory dict must have 'reward' (dict with 'total') and
        'prompt' (str) keys.

        Returns:
            [N] tensor of advantages, one per trajectory.
        """
        prompts = [t.get("prompt", "") for t in trajectories]
        rewards = [t.get("reward", {}).get("total", 0.0) for t in trajectories]
        advantages = self._stat_tracker.update(prompts, rewards)
        return torch.tensor(advantages, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Rollout: AR SCD generation with SDE log-prob capture
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def rollout(
        self,
        scd_model: torch.nn.Module,
        prompt_embeds: Tensor,
        prompt_mask: Tensor,
        vae_decoder: torch.nn.Module,
        patchifier,
        scheduler,
        prompts: list[str],
        device: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[list[float], list[list[dict[str, Any]]]]:
        """Generate K videos per prompt, score them, return advantages + trajectories.

        This is the GRPO sampling phase.  Each call:
          1. Runs SCD autoregressive inference K times per prompt, collecting
             full trajectory data (x_t, sigma, x_next, log_prob, enc_features,
             positions) at every denoising step.  Data is moved to CPU immediately.
          2. VAE-decodes to pixel space.
          3. Scores with reward functions.
          4. Normalises per prompt -> advantages.

        Args:
            scd_model: LTXSCDModel instance.
            prompt_embeds: [B, seq_text, D] text embeddings.
            prompt_mask: [B, seq_text] attention mask.
            vae_decoder: VAE decoder module.
            patchifier: VideoLatentPatchifier instance.
            scheduler: LTX2Scheduler instance.
            prompts: list of B prompt strings.
            device: primary device string (e.g. "cuda:0").
            dtype: compute dtype for generation.

        Returns:
            advantages: list[float] of length B*K (one per sample).
            all_trajectories: list of B*K lists, each containing per-step dicts
                with keys: x_t, sigma, x_next, x_next_mean, noise_std,
                log_prob_old, dt, enc_features, positions, frame_idx.
        """
        cfg = self.config
        B = len(prompts)
        K = cfg.grpo_num_samples

        # Snapshot reference LoRA state at the start of this rollout.
        self._ref_state_dict = self.snapshot_lora_state(scd_model)

        all_rewards: list[float] = []
        all_prompts_repeated: list[str] = []
        all_trajectories: list[list[dict[str, Any]]] = []

        for k in range(K):
            for b in range(B):
                try:
                    reward_dict, trajectory = self._single_rollout(
                        scd_model=scd_model,
                        prompt_embeds=prompt_embeds[b : b + 1],
                        prompt_mask=prompt_mask[b : b + 1],
                        vae_decoder=vae_decoder,
                        patchifier=patchifier,
                        scheduler=scheduler,
                        prompt=prompts[b],
                        device=device,
                        dtype=dtype,
                    )
                    all_rewards.append(reward_dict["total"])
                    all_trajectories.append(trajectory)
                except Exception as e:
                    logger.warning(f"Rollout failed (b={b}, k={k}): {e}")
                    all_rewards.append(0.0)
                    all_trajectories.append([])
                all_prompts_repeated.append(prompts[b])

        advantages = self._stat_tracker.update(all_prompts_repeated, all_rewards)

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "grpo/mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
                "grpo/max_reward": max(all_rewards) if all_rewards else 0,
                "grpo/min_reward": min(all_rewards) if all_rewards else 0,
                "grpo/num_trajectories": len(all_trajectories),
            })

        return advantages, all_trajectories

    def _single_rollout(
        self,
        scd_model: torch.nn.Module,
        prompt_embeds: Tensor,
        prompt_mask: Tensor,
        vae_decoder: torch.nn.Module,
        patchifier,
        scheduler,
        prompt: str,
        device: str,
        dtype: torch.dtype,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Run one full AR SCD generation + VAE decode + reward.

        Follows the inference pattern from scd_inference.py:
          - Encoder: clean frame, sigma=0, with KV-cache (incremental).
          - Shift-by-1: decoder for frame t receives encoder features from
            frame t-1.
          - Decoder: full denoising with SDE log-prob capture at each step.

        Returns:
            reward_dict: {aesthetic, temporal, total}
            trajectory: list of per-step dicts with stored data for PG update.
        """
        cfg = self.config

        fps = cfg.fps
        num_frames = int(cfg.rollout_num_seconds * fps)
        # Enforce frames % 8 == 1 rule.
        num_frames = max(1, ((num_frames - 1) // 8) * 8 + 1)
        h_lat = cfg.resolution_h // 32
        w_lat = cfg.resolution_w // 32

        # Tokens per frame (patch_size=1, so tokens = H * W * channels per token).
        # With VideoLatentPatchifier(patch_size=1), patchify rearranges
        # [B, C, F, H, W] -> [B, F*H*W, C] so tokens_per_frame = H * W.
        tpf = h_lat * w_lat

        # Build sigma schedule (same for every frame).
        # Use a dummy latent with 4 frames (typical SCD window) to get the
        # right token-count-dependent shifting.
        dummy_latent = torch.zeros(1, 128, 4, h_lat, w_lat)
        sigmas = scheduler.execute(
            steps=cfg.num_inference_steps,
            latent=dummy_latent,
        ).to(device=device)

        trajectory: list[dict[str, Any]] = []
        chunk_generated: list[Tensor] = []
        chunk_boundaries: list[int] = []

        # ------------------------------------------------------------------
        # AR loop: one latent frame at a time
        # ------------------------------------------------------------------
        from ltx_core.model.transformer.scd_model import KVCache

        kv_cache = KVCache(keys={}, values={}, is_cache_step=True, cached_seq_len=0)
        prev_enc_features: Tensor | None = None

        for frame_idx in range(num_frames):
            # === ENCODER PASS =============================================
            # The encoder receives clean latents (sigma=0) and uses KV-cache
            # so that frame t can attend to all frames <= t.

            if frame_idx == 0:
                # First frame: no prior latent, use zeros as context.
                enc_latent = torch.zeros(
                    1, 128, 1, h_lat, w_lat, device=device, dtype=dtype,
                )
            else:
                # Use previous frame's generated output as encoder input.
                enc_latent = chunk_generated[-1].reshape(1, 128, 1, h_lat, w_lat)

            # Patchify encoder input.
            enc_patchified = patchifier.patchify(enc_latent)  # [1, tpf, 128]
            enc_positions = self._get_frame_positions(
                frame_idx=frame_idx,
                h_lat=h_lat,
                w_lat=w_lat,
                fps=fps,
                device=device,
                dtype=dtype,
            )

            enc_modality = Modality(
                enabled=True,
                latent=enc_patchified,
                sigma=torch.zeros(1, device=device, dtype=dtype),
                timesteps=torch.zeros(1, tpf, device=device, dtype=dtype),
                positions=enc_positions,
                context=prompt_embeds,
                context_mask=prompt_mask,
            )

            enc_video_out, _ = scd_model.forward_encoder(
                video=enc_modality,
                audio=None,
                perturbations=None,
                kv_cache=kv_cache,
                tokens_per_frame=tpf,
            )
            current_enc = enc_video_out.x.detach()

            # === SHIFT-BY-1 ==============================================
            # Frame t's decoder gets frame t-1's encoder features.
            # Frame 0 gets zeros (no prior context).
            if prev_enc_features is None:
                dec_enc_ctx = torch.zeros_like(current_enc)
            else:
                dec_enc_ctx = prev_enc_features
            prev_enc_features = current_enc

            # === DECODER: full denoising with SDE log-prob ================
            x_t = torch.randn(1, 128, 1, h_lat, w_lat, device=device, dtype=dtype)

            for step_idx in range(len(sigmas) - 1):
                sigma_val = sigmas[step_idx]
                sigma_next_val = sigmas[step_idx + 1]

                noisy_patch = patchifier.patchify(x_t)  # [1, tpf, 128]
                dec_positions = self._get_frame_positions(
                    frame_idx=frame_idx,
                    h_lat=h_lat,
                    w_lat=w_lat,
                    fps=fps,
                    device=device,
                    dtype=dtype,
                )

                dec_modality = Modality(
                    enabled=True,
                    latent=noisy_patch,
                    sigma=sigma_val.reshape(1).to(dtype=dtype),
                    timesteps=torch.full(
                        (1, tpf), sigma_val.item(), device=device, dtype=dtype
                    ),
                    positions=dec_positions,
                    context=prompt_embeds,
                    context_mask=prompt_mask,
                )

                velocity, _ = scd_model.forward_decoder(
                    video=dec_modality,
                    encoder_features=dec_enc_ctx,
                    audio=None,
                    perturbations=None,
                )

                # SDE step with log-prob capture.
                x_next, log_prob, x_next_mean, noise_std = sde_step_with_logprob(
                    velocity, noisy_patch, sigma_val, sigma_next_val,
                )

                dt_val = (sigma_next_val - sigma_val).item()

                # Store trajectory data on CPU to save VRAM.
                trajectory.append({
                    "x_t": noisy_patch.detach().cpu(),
                    "sigma": sigma_val.item(),
                    "x_next": x_next.detach().cpu(),
                    "x_next_mean": x_next_mean.detach().cpu(),
                    "noise_std": noise_std.item(),
                    "log_prob_old": log_prob.detach().cpu(),
                    "dt": dt_val,
                    "enc_features": dec_enc_ctx.detach().cpu(),
                    "positions": dec_positions.detach().cpu(),
                    "frame_idx": frame_idx,
                    "prompt_embeds": prompt_embeds.detach().cpu(),
                    "prompt_mask": prompt_mask.detach().cpu(),
                    "velocity_target": velocity.detach().cpu(),
                })

                # Unpatchify: [1, tpf, C] -> [1, C, 1, H, W].
                # patch_size=1 so the rearrange is b (f h w) (c) -> b c f h w.
                x_t = patchifier.unpatchify(
                    x_next,
                    output_shape=VideoLatentShape(
                        batch=1, channels=128, frames=1,
                        height=h_lat, width=w_lat,
                    ),
                ).detach()

            chunk_generated.append(x_t.detach())
            if frame_idx > 0:
                chunk_boundaries.append(frame_idx)

        # ------------------------------------------------------------------
        # VAE decode all frames to pixel space
        # ------------------------------------------------------------------
        try:
            full_latent = torch.stack(chunk_generated, dim=2)  # [1, 128, T, H, W]
            dec_device = next(vae_decoder.parameters()).device
            with torch.inference_mode():
                pixels = vae_decoder(full_latent.to(dec_device))
            # pixels: [1, C, T, H, W] -> [T, C, H, W]
            pixels = pixels.squeeze(0).permute(1, 0, 2, 3).clamp(0, 1).cpu()
        except Exception as e:
            logger.warning(f"VAE decode failed in rollout: {e}")
            pixels = torch.zeros(num_frames, 3, cfg.resolution_h, cfg.resolution_w)

        reward = self.compute_reward(
            decoded_frames=pixels,
            prompt=prompt,
            chunk_boundaries=chunk_boundaries,
            device=device,
        )

        # Free GPU memory eagerly.
        del kv_cache, prev_enc_features
        gc.collect()
        torch.cuda.empty_cache()

        return reward, trajectory

    # ------------------------------------------------------------------
    # Position helper for single-frame rollouts
    # ------------------------------------------------------------------

    def _get_frame_positions(
        self,
        frame_idx: int,
        h_lat: int,
        w_lat: int,
        fps: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build position tensor for a single latent frame.

        Returns [1, 3, tpf, 2] where dim-1 is (time, height, width).
        Temporal coordinate is scaled by 1/fps to get seconds.
        """
        tpf = h_lat * w_lat
        # Temporal: all tokens share the same frame index.
        t_coord = torch.full((1, tpf), frame_idx, device=device, dtype=dtype)
        # Spatial: row/col grid.
        h_coords = torch.arange(h_lat, device=device, dtype=dtype).unsqueeze(1).expand(h_lat, w_lat)
        w_coords = torch.arange(w_lat, device=device, dtype=dtype).unsqueeze(0).expand(h_lat, w_lat)
        h_flat = h_coords.reshape(-1)
        w_flat = w_coords.reshape(-1)
        # Assemble [1, 3, tpf] and scale temporal by 1/fps.
        positions = torch.stack([t_coord[0] / fps, h_flat, w_flat], dim=0).unsqueeze(0)
        # Add a dummy last dim of size 2 for downstream compatibility
        # (the transformer expects [B, 3, seq, 2]).
        positions = positions.unsqueeze(-1).expand(-1, -1, -1, 2)
        return positions

    # ------------------------------------------------------------------
    # Re-evaluate log-prob under current policy (for PPO ratio)
    # ------------------------------------------------------------------

    def compute_log_prob(
        self,
        scd_model: torch.nn.Module,
        step_data: dict[str, Any],
        device: str,
    ) -> tuple[Tensor, Tensor]:
        """Re-run decoder on stored (x_t, sigma) to get current policy log-prob.

        This is called during the PG update phase.  The decoder sees the same
        x_t and encoder features that were recorded during rollout, producing
        a new velocity and hence a new SDE transition log-probability.

        Args:
            scd_model: LTXSCDModel with current (trainable) LoRA weights.
            step_data: dict from rollout trajectory with keys: x_t, sigma,
                enc_features, positions, frame_idx, dt, noise_std, x_next.
            device: target device.

        Returns:
            (log_prob_new, velocity_new)
        """
        x_t = step_data["x_t"].to(device)
        enc_features = step_data["enc_features"].to(device)
        positions = step_data["positions"].to(device)
        sigma = step_data["sigma"]
        prompt_embeds = step_data["prompt_embeds"].to(device)
        prompt_mask = step_data["prompt_mask"].to(device)

        # Build modality for the decoder.
        dec_modality = Modality(
            enabled=True,
            latent=x_t,
            sigma=torch.tensor([sigma], device=device, dtype=x_t.dtype),
            timesteps=torch.full(
                (1, x_t.shape[1]), sigma, device=device, dtype=x_t.dtype
            ),
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_mask,
        )

        velocity, _ = scd_model.forward_decoder(
            video=dec_modality,
            encoder_features=enc_features,
            audio=None,
            perturbations=None,
        )

        # Reconstruct the SDE step to get the new log-prob.
        x_next_mean_new = x_t.float() + step_data["dt"] * velocity.float()
        noise_std = torch.tensor(step_data["noise_std"], device=device)
        log_prob_new = gaussian_log_prob(
            step_data["x_next"].to(device), x_next_mean_new, noise_std
        )
        return log_prob_new, velocity

    def _forward_decoder(
        self,
        scd_model: torch.nn.Module,
        step_data: dict[str, Any],
        device: str,
    ) -> tuple[Tensor, None]:
        """Forward decoder on stored trajectory data (helper for KL)."""
        x_t = step_data["x_t"].to(device)
        enc_features = step_data["enc_features"].to(device)
        positions = step_data["positions"].to(device)
        sigma = step_data["sigma"]
        prompt_embeds = step_data["prompt_embeds"].to(device)
        prompt_mask = step_data["prompt_mask"].to(device)

        dec_modality = Modality(
            enabled=True,
            latent=x_t,
            sigma=torch.tensor([sigma], device=device, dtype=x_t.dtype),
            timesteps=torch.full(
                (1, x_t.shape[1]), sigma, device=device, dtype=x_t.dtype
            ),
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_mask,
        )
        return scd_model.forward_decoder(
            video=dec_modality,
            encoder_features=enc_features,
            audio=None,
            perturbations=None,
        )

    # ------------------------------------------------------------------
    # Policy gradient loss
    # ------------------------------------------------------------------

    def compute_pg_loss(
        self,
        scd_model: torch.nn.Module,
        step_data: dict[str, Any],
        advantage: float,
        ref_state_dict: dict[str, Tensor],
        device: str,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """PPO clipped surrogate + KL penalty for one trajectory step.

        Re-runs the decoder on stored (x_t, sigma, enc_features) to get the
        current policy log-prob, then forms the PPO ratio against the stored
        old log-prob from rollout.

        Optionally computes KL against the reference LoRA by temporarily
        swapping weights, running a forward pass, and restoring.

        Optionally adds an SL anchor if velocity_target is present in
        step_data (for the supervised mixing path).

        Args:
            scd_model: LTXSCDModel with current trainable LoRA.
            step_data: per-step dict from rollout trajectory.
            advantage: GRPO-normalised advantage for this sample.
            ref_state_dict: reference LoRA state dict (from snapshot_lora_state).
            device: target device.

        Returns:
            (total_loss, loss_dict) where loss_dict has keys pg, kl, sl.
        """
        # --- New log-prob under current policy ---
        log_prob_new, velocity_new = self.compute_log_prob(scd_model, step_data, device)

        # --- PPO ratio ---
        log_prob_old = step_data["log_prob_old"].to(device)
        ratio = torch.exp(log_prob_new - log_prob_old)

        # --- Clipped surrogate ---
        adv = torch.tensor(
            advantage, dtype=velocity_new.dtype, device=velocity_new.device
        ).clamp(-self.config.ppo_adv_clip, self.config.ppo_adv_clip)
        unclipped = -adv * ratio
        clipped = -adv * torch.clamp(
            ratio, 1.0 - self.config.ppo_epsilon, 1.0 + self.config.ppo_epsilon
        )
        pg_loss = torch.max(unclipped, clipped).mean()

        # --- KL penalty (reference model) ---
        if self.config.kl_beta > 0 and ref_state_dict:
            with torch.no_grad():
                orig_state = self._load_reference_state(scd_model, ref_state_dict)
                velocity_ref, _ = self._forward_decoder(scd_model, step_data, device)
                self._restore_current_state(scd_model, orig_state)

            x_t_dev = step_data["x_t"].to(device).float()
            dt = step_data["dt"]
            x_next_mean_new = x_t_dev + dt * velocity_new.float()
            x_next_mean_ref = x_t_dev + dt * velocity_ref.float()
            noise_std = torch.tensor(step_data["noise_std"], device=device)
            # Bug C fix: removed .detach() so KL gradient flows through velocity_new
            kl = ((x_next_mean_new - x_next_mean_ref) ** 2) / (2 * noise_std ** 2)
            kl_loss = kl.mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # --- SL anchor with optional sparse supervision (SSD, arXiv:2602.02699) ---
        if self.config.sl_weight > 0 and "velocity_target" in step_data:
            v_target = step_data["velocity_target"].to(device)
            eta = self.config.sparse_mask_ratio
            if eta > 0.0:
                # Mask ~eta fraction of tokens; loss only on visible subset
                sparse_mask = (torch.rand_like(velocity_new) > eta).float()
                num_vis = sparse_mask.sum() + 1e-8
                sl_loss = F.mse_loss(
                    velocity_new * sparse_mask, v_target * sparse_mask, reduction="sum"
                ) / num_vis
            else:
                sl_loss = F.mse_loss(velocity_new, v_target)
        else:
            sl_loss = torch.tensor(0.0, device=device)

        total_loss = pg_loss + self.config.kl_beta * kl_loss + self.config.sl_weight * sl_loss

        return total_loss, {"pg": pg_loss.detach(), "kl": kl_loss.detach(), "sl": sl_loss.detach()}

    # ------------------------------------------------------------------
    # Override compute_loss: SL anchor + optional PG signal
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
        advantage: float = 0.0,
        video_ref: Tensor | None = None,
        sigma: float = 0.5,
    ) -> Tensor:
        """Combined SL anchor + GRPO policy gradient loss.

        When advantage=0 and video_ref=None (standard SCD training step),
        this is identical to SCDTrainingStrategy.compute_loss -- backward compat.

        NOTE: The main GRPO training path does NOT use this method.  Instead,
        the standalone training script calls rollout() -> compute_pg_loss()
        directly.  This override exists for the SL anchor path and for any
        trainer that calls compute_loss with the extra arguments.

        Args:
            video_pred: [B, L, C] decoder velocity prediction.
            audio_pred: audio prediction (passed through).
            inputs: ModelInputs from prepare_training_inputs.
            advantage: GRPO-normalised advantage for this sample (0 = SL-only).
            video_ref: reference policy velocity for KL (None = SL-only).
            sigma: noise level at this denoising step.
        """
        # SL anchor: keep the policy grounded on the flow matching objective.
        sl_loss = super().compute_loss(video_pred, audio_pred, inputs)

        if advantage == 0.0 or video_ref is None:
            return sl_loss

        # Build a minimal step_data so we can reuse compute_pg_loss.
        step_data = {
            "x_t": torch.zeros_like(video_pred),
            "sigma": sigma,
            "x_next": torch.zeros_like(video_pred),
            "log_prob_old": flow_matching_log_prob_fallback(video_ref, sigma).detach(),
            "dt": 0.0,
            "noise_std": 1.0,
            "enc_features": torch.zeros_like(video_pred[:, :1, :]),
            "positions": None,
        }
        pg_loss = -advantage * flow_matching_log_prob_fallback(video_pred, sigma).mean()
        kl_loss = torch.tensor(0.0, device=video_pred.device)

        total_loss = (
            self.config.sl_weight * sl_loss
            + pg_loss
            + self.config.kl_beta * kl_loss
        )

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "loss/sl": sl_loss.item(),
                "loss/pg": pg_loss.item(),
                "loss/kl": kl_loss.item(),
                "loss/total": total_loss.item(),
                "grpo/advantage": advantage,
                "grpo/sigma": sigma,
            })

        return total_loss

    # ------------------------------------------------------------------
    # Data sources / metadata
    # ------------------------------------------------------------------

    def get_data_sources(self) -> list[str] | dict[str, str]:
        return super().get_data_sources()

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        meta = super().get_checkpoint_metadata()
        meta["vfm_version"] = "v5a"
        meta["grpo_num_samples"] = self.config.grpo_num_samples
        meta["kl_beta"] = self.config.kl_beta
        meta["ppo_epsilon"] = self.config.ppo_epsilon
        meta["reward_mode"] = self.config.reward_mode
        return meta


# ---------------------------------------------------------------------------
# Fallback log-prob (velocity-magnitude proxy, used only by the legacy
# compute_loss override path — NOT used by the main GRPO rollout).
# ---------------------------------------------------------------------------

def flow_matching_log_prob_fallback(
    velocity_pred: Tensor,
    sigma: float,
) -> Tensor:
    """Legacy velocity-magnitude log-prob approximation.

    Kept for backward compat with compute_loss(video_ref=...) override.
    The main GRPO path uses gaussian_log_prob / sde_step_with_logprob instead.
    """
    var = sigma ** 2 + 1e-6
    log_prob = -0.5 * (velocity_pred.pow(2) / var).sum(dim=(-1, -2))
    return log_prob
