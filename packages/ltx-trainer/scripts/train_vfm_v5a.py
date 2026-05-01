#!/usr/bin/env python3
# ruff: noqa: T201
"""VFM v5a: Flow-GRPO RL post-training for LTX-2.3 SCD.

Ports World-R1's Flow-GRPO (arXiv:2405.20673) online RL to the SCD pipeline.
The GRPO outer loop:
  1. SAMPLE: Generate K videos per prompt via AR SCD rollout with log-prob capture
  2. REWARD: Score videos (aesthetic + temporal SSIM)
  3. ADVANTAGE: Per-prompt GRPO advantage normalization
  4. TRAIN: PPO clipped surrogate loss on stored trajectories

Unlike standard training, this doesn't fit the `for batch in dataloader` pattern,
so it's a standalone script (like World-R1's train_world_r1.py).

Usage:
    # Minimal smoke test (K=2, 9 frames, 4 steps)
    python scripts/train_vfm_v5a.py \
        --checkpoint /media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-dev.safetensors \
        --lora-path /path/to/scd_lora.safetensors \
        --prompts-file prompts.jsonl \
        --num-epochs 1 --K 2 --num-frames 9 --num-inference-steps 4 \
        --output-dir output/vfm_v5a

    # Full training run
    python scripts/train_vfm_v5a.py \
        --checkpoint /media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-dev.safetensors \
        --lora-path /path/to/scd_lora.safetensors \
        --prompts-file prompts.jsonl \
        --num-epochs 50 --K 4 --num-frames 25 \
        --num-inference-steps 8 \
        --inner-epochs 1 --lr 1e-5 \
        --ppo-epsilon 0.2 --kl-beta 0.01 \
        --output-dir output/vfm_v5a
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

# Allow importing sibling scripts directly (scripts/ is not a package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor

# ── Add project root to path ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── LTX imports ──
from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import KVCache, LTXSCDModel
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

# ── SCD inference utilities (reuse proven AR rollout components) ──
from scd_inference import (
    extract_lora_target_modules,
    load_lora_weights,
    DISTILLED_SIGMA_VALUES,
)

# ── Strategy ──
from ltx_trainer.training_strategies.vfm_strategy_v5a import (
    GRPOv5aConfig,
    GRPOv5aTrainingStrategy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VFM v5a Flow-GRPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──
    p.add_argument("--checkpoint", required=True, help="Base model .safetensors path")
    p.add_argument("--lora-path", required=True, help="SCD LoRA weights .safetensors")
    p.add_argument("--text-encoder-path", default="/media/2TB/ltx-models/gemma", help="Gemma text encoder dir")
    p.add_argument("--encoder-layers", type=int, default=32)
    p.add_argument("--decoder-combine", default="token_concat", choices=["add", "token_concat"])

    # ── Prompts ──
    p.add_argument("--prompts-file", required=True, help="JSONL with 'prompt' field per line")

    # ── Generation ──
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--num-frames", type=int, default=25, help="Total latent frames to generate per rollout")
    p.add_argument("--num-inference-steps", type=int, default=8, help="Denoising steps per frame")
    p.add_argument("--seed", type=int, default=42)

    # ── GRPO ──
    p.add_argument("--K", type=int, default=4, help="Samples per prompt per GRPO step")
    p.add_argument("--num-epochs", type=int, default=50, help="GRPO outer epochs")
    p.add_argument("--inner-epochs", type=int, default=1, help="PPO inner epochs per GRPO step")
    p.add_argument("--timestep-fraction", type=float, default=0.5, help="Fraction of timesteps to sample for training")
    p.add_argument("--ppo-epsilon", type=float, default=0.2, help="PPO clipping parameter")
    p.add_argument("--kl-beta", type=float, default=0.01, help="KL penalty weight")
    p.add_argument("--sl-weight", type=float, default=0.1, help="Supervised learning anchor weight")
    p.add_argument("--ppo-adv-clip", type=float, default=5.0, help="Max absolute advantage")
    p.add_argument("--sparse-mask-ratio", type=float, default=0.98, help="SSD sparse supervision mask ratio (0=disabled, 0.98=paper default)")

    # ── Optimizer ──
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.0)

    # ── Reward ──
    p.add_argument("--reward-mode", default="local", choices=["local", "server", "dummy"])
    p.add_argument("--reward-server-url", default="http://127.0.0.1:8090")
    p.add_argument("--w-aesthetic", type=float, default=0.5, help="Aesthetic reward weight")
    p.add_argument("--w-temporal", type=float, default=0.5, help="Temporal SSIM reward weight")

    # ── Output ──
    p.add_argument("--output-dir", required=True)
    p.add_argument("--wandb-project", default="vfm-v5a")
    p.add_argument("--wandb-run-name", default=None)

    # ── Device ──
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def load_prompts(path: str) -> list[str]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def encode_prompts(
    prompts: list[str],
    checkpoint_path: str,
    text_encoder_path: str,
    device: str = "cuda:1",
) -> list[tuple[Tensor, Tensor]]:
    """Encode all prompts and return (embeds, mask) pairs.

    Loads Gemma → encodes → unloads to free VRAM.
    """
    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor

    print(f"  Encoding {len(prompts)} prompts...")

    # Block 1: Gemma LLM
    text_encoder = load_text_encoder(
        text_encoder_path,
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )
    text_encoder.eval()

    encoded = []
    with torch.inference_mode():
        for prompt in prompts:
            hidden_states, attention_mask = text_encoder.encode(prompt)
            hidden_states = tuple(h.cpu() for h in hidden_states)
            attention_mask = attention_mask.cpu()
            encoded.append((hidden_states, attention_mask))

    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Block 2+3: FeatureExtractor + Connector
    emb_proc = load_embeddings_processor(checkpoint_path, device=device, dtype=torch.bfloat16)
    emb_proc.eval()

    results = []
    with torch.inference_mode():
        for hidden_states, attention_mask in encoded:
            hidden_states = tuple(h.to(device) for h in hidden_states)
            attention_mask = attention_mask.to(device)
            result = emb_proc.process_hidden_states(hidden_states, attention_mask)
            prompt_embeds = result.video_encoding.to(torch.bfloat16).cpu()
            prompt_mask = result.attention_mask.cpu()
            results.append((prompt_embeds, prompt_mask))

    del emb_proc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Encoded {len(results)} prompts, shapes: embeds={results[0][0].shape}, mask={results[0][1].shape}")
    return results


def gaussian_log_prob(sample: Tensor, mean: Tensor, std: float | Tensor) -> Tensor:
    """Gaussian log-probability of sample under N(mean, std^2).

    Returns scalar per batch element (mean over all non-batch dims).
    """
    std_t = torch.as_tensor(std, device=mean.device, dtype=mean.dtype) if not isinstance(std, Tensor) else std
    log_prob = (
        -((sample.detach().float() - mean.float()) ** 2) / (2 * std_t.float() ** 2)
        - torch.log(std_t.float())
        - 0.5 * math.log(2 * math.pi)
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def sde_step_with_logprob(
    velocity: Tensor,
    noisy_patch: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """SDE denoising step with log-probability capture (World-R1 approach).

    Args:
        velocity: [B, seq, C] decoder velocity prediction
        noisy_patch: [B, seq, C] current noisy state x_t (patchified)
        sigma: current noise level
        sigma_next: next noise level

    Returns:
        x_next: [B, seq, C] stochastic next state
        log_prob: [B] transition log-probability
        x_next_mean: [B, seq, C] deterministic prediction (for KL)
        noise_std: scalar transition standard deviation
    """
    dt = sigma_next - sigma  # negative (denoising)

    # Deterministic Euler prediction
    x_next_mean = noisy_patch.float() + dt.float() * velocity.float()

    # Transition noise: proportional to sigma * sqrt(|dt|)
    noise_std = (sigma.float() * torch.abs(dt.float()).sqrt()).clamp(min=1e-8)

    # Stochastic SDE step
    noise = torch.randn_like(noisy_patch)
    x_next = x_next_mean + noise_std * noise

    # Log-prob of the transition under N(x_next_mean, noise_std^2)
    log_prob = gaussian_log_prob(x_next, x_next_mean, noise_std)

    return x_next, log_prob, x_next_mean, noise_std


def main() -> None:
    args = parse_args()

    # ── Validate ──
    assert args.height % 32 == 0, f"Height {args.height} must be divisible by 32"
    assert args.width % 32 == 0, f"Width {args.width} must be divisible by 32"

    latent_h = args.height // 32
    latent_w = args.width // 32
    tokens_per_frame = latent_h * latent_w

    # Frame math: (F_pixel - 1) % 8 == 0, F_latent = (F_pixel - 1) // 8 + 1
    # For --num-frames: treat as LATENT frames
    total_latent_frames = args.num_frames
    total_pixel_frames = (total_latent_frames - 1) * 8 + 1

    # Chunk structure (from scd_inference.py)
    CHUNK_LATENT = 4
    NEW_PER_CHUNK = CHUNK_LATENT - 1  # 3

    if total_latent_frames <= CHUNK_LATENT:
        num_chunks = 1
    else:
        num_chunks = 1 + -(-((total_latent_frames - CHUNK_LATENT)) // NEW_PER_CHUNK)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print()
    print("=" * 65)
    print("  VFM v5a: Flow-GRPO RL Post-Training")
    print("=" * 65)
    print(f"  Resolution:  {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"  Frames:      {total_latent_frames} latent ({total_pixel_frames} pixel)")
    print(f"  Chunks:      {num_chunks}")
    print(f"  Steps/frame: {args.num_inference_steps}")
    print(f"  K samples:   {args.K}")
    print(f"  PPO ε:       {args.ppo_epsilon}")
    print(f"  KL β:        {args.kl_beta}")
    print(f"  SL weight:   {args.sl_weight}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Device:      {device}")
    print("=" * 65)

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ──
    prompts = load_prompts(args.prompts_file)
    print(f"\n  Loaded {len(prompts)} prompts")

    # ── Encode prompts ──
    print("\n[1/4] Encoding prompts...")
    enc_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    prompt_data = encode_prompts(prompts, args.checkpoint, args.text_encoder_path, device=str(enc_device))
    # Move to main device
    prompt_data = [(e.to(device), m.to(device)) for e, m in prompt_data]

    # ── Load transformer → LoRA → SCD wrap ──
    print("\n[2/4] Loading transformer...")
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.quantization import quantize_model

    transformer = load_transformer(args.checkpoint, device="cpu", dtype=dtype)

    # Quantize for inference during rollout
    if device.type == "cuda":
        print("  Quantizing (fp8-quanto)...")
        transformer = quantize_model(transformer, "fp8-quanto", device=str(device))

    # Apply LoRA BEFORE SCD wrapping
    print(f"  Applying LoRA: {args.lora_path}")
    transformer = load_lora_weights(transformer, args.lora_path, encoder_layers=args.encoder_layers)
    transformer = transformer.get_base_model()

    # Wrap with SCD
    print(f"  SCD wrap: {args.encoder_layers} encoder + {48 - args.encoder_layers} decoder")
    scd_model = LTXSCDModel(
        base_model=transformer,
        encoder_layers=args.encoder_layers,
        decoder_input_combine=args.decoder_combine,
    )
    scd_model.to(device)
    scd_model.eval()

    latent_channels = scd_model.base_model.patchify_proj.in_features  # 128

    # ── Patchifier + scheduler ──
    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()

    def get_positions(n_frames: int, target_device: torch.device | None = None) -> Tensor:
        d = target_device or device
        coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(frames=n_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels),
            device=d,
        )
        px = get_pixel_coords(latent_coords=coords, scale_factors=scale_factors, causal_fix=True).to(dtype)
        px[:, 0, ...] = px[:, 0, ...] / args.fps
        return px

    def get_positions_for_frame(frame_idx: int, target_device: torch.device | None = None) -> Tensor:
        all_pos = get_positions(frame_idx + 1, target_device=target_device)
        start = frame_idx * tokens_per_frame
        end = start + tokens_per_frame
        return all_pos[:, :, start:end, :]

    # Sigma schedule (matching training window)
    dummy_latent = torch.empty(1, 1, CHUNK_LATENT, latent_h, latent_w)
    base_sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps, latent=dummy_latent)

    # ── VAE decoder (lazy load, only for reward scoring) ──
    vae_decoder = None  # Loaded on demand

    def get_vae_decoder():
        nonlocal vae_decoder
        if vae_decoder is None:
            print("  Loading VAE decoder for reward scoring...")
            from ltx_trainer.model_loader import load_vae
            vae_decoder = load_vae(args.checkpoint, device=str(device), dtype=dtype)
            vae_decoder.eval()
        return vae_decoder

    # ── Strategy ──
    print("\n[3/4] Initializing strategy...")
    # "dummy" is a script-level shortcut — GRPOv5aConfig only accepts "local"/"server"
    config_reward_mode = "local" if args.reward_mode == "dummy" else args.reward_mode
    config = GRPOv5aConfig(
        name="vfm_v5a",
        num_inference_steps=args.num_inference_steps,
        ppo_epsilon=args.ppo_epsilon,
        kl_beta=args.kl_beta,
        sl_weight=args.sl_weight,
        ppo_adv_clip=args.ppo_adv_clip,
        reward_mode=config_reward_mode,
        reward_server_url=args.reward_server_url,
        reward_aesthetic_weight=args.w_aesthetic,
        reward_temporal_weight=args.w_temporal,
        encoder_layers=args.encoder_layers,
        decoder_input_combine=args.decoder_combine,
        per_frame_decoder=True,
        fps=args.fps,
        resolution_h=args.height,
        resolution_w=args.width,
        sparse_mask_ratio=args.sparse_mask_ratio,
    )
    strategy = GRPOv5aTrainingStrategy(config)

    # ── Optimizer (train LoRA params only) ──
    trainable_params = [p for p in scd_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # ── Wandb ──
    print("\n[4/4] Starting training...")
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # ══════════════════════════════════════════════════════════════════════
    # GRPO Training Loop
    # ══════════════════════════════════════════════════════════════════════

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        print(f"\n{'=' * 65}")
        print(f"  Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'=' * 65}")

        all_trajectories = []

        # ── SAMPLE PHASE: Rollout K videos per prompt ──
        scd_model.eval()
        for prompt_idx, (prompt_embeds, prompt_mask) in enumerate(prompt_data):
            prompt_text = prompts[prompt_idx]
            print(f"\n  Prompt {prompt_idx + 1}/{len(prompts)}: '{prompt_text[:60]}...'")

            for k in range(args.K):
                print(f"    Sample {k + 1}/{args.K}...")
                generator = torch.Generator(device=device).manual_seed(args.seed + epoch * 1000 + prompt_idx * args.K + k)

                trajectory = rollout_one_video(
                    scd_model=scd_model,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    patchifier=patchifier,
                    get_positions_for_frame=get_positions_for_frame,
                    base_sigmas=base_sigmas,
                    device=device,
                    dtype=dtype,
                    latent_channels=latent_channels,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    tokens_per_frame=tokens_per_frame,
                    num_chunks=num_chunks,
                    total_latent_frames=total_latent_frames,
                    CHUNK_LATENT=CHUNK_LATENT,
                    NEW_PER_CHUNK=NEW_PER_CHUNK,
                    generator=generator,
                    timestep_fraction=args.timestep_fraction,
                )

                # ── VAE decode + reward ──
                if args.reward_mode == "dummy":
                    # Skip VAE decode — assign random rewards for smoke test
                    rewards = {
                        "aesthetic": random.gauss(6.0, 1.0),
                        "temporal": random.gauss(0.5, 0.2),
                        "total": random.gauss(0.75, 0.15),
                    }
                else:
                    vae = get_vae_decoder()
                    with torch.inference_mode():
                        # Stack latent frames → [1, C, T, H, W]
                        latent_video = torch.cat(trajectory["latent_frames"], dim=2).to(device)
                        pixels = vae.decode(latent_video)
                        pixels = (pixels + 1) / 2  # [-1,1] → [0,1]
                        # Bug A fix: vae.decode → [1,C,T,H,W], compute_reward expects [T,C,H,W]
                        pixels = pixels.squeeze(0).permute(1, 0, 2, 3).cpu()

                    rewards = strategy.compute_reward(
                        decoded_frames=pixels,
                        prompt=prompt_text,
                        chunk_boundaries=trajectory["chunk_boundaries"],
                        device=device,
                    )
                trajectory["reward"] = rewards
                trajectory["prompt_idx"] = prompt_idx
                trajectory["prompt"] = prompt_text
                all_trajectories.append(trajectory)

                # Free VRAM
                del trajectory["latent_frames"]
                torch.cuda.empty_cache()

        # ── ADVANTAGE PHASE: Per-prompt GRPO normalization ──
        advantages = strategy.compute_advantages(all_trajectories)
        print(f"\n  Advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}")

        # ── SNAPSHOT reference LoRA ──
        ref_state_dict = strategy.snapshot_lora_state(scd_model)
        print(f"  Reference state snapshot: {len(ref_state_dict)} LoRA params")

        # ── TRAIN PHASE: PPO on stored trajectories ──
        scd_model.train()
        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        n_steps = 0

        for inner_epoch in range(args.inner_epochs):
            # Shuffle trajectories
            indices = list(range(len(all_trajectories)))
            random.shuffle(indices)

            for idx in indices:
                traj = all_trajectories[idx]
                adv = advantages[idx]

                # Sample subset of timesteps
                step_data_list = traj["step_data"]
                if args.timestep_fraction < 1.0:
                    n_sample = max(1, int(len(step_data_list) * args.timestep_fraction))
                    sampled = random.sample(step_data_list, min(n_sample, len(step_data_list)))
                else:
                    sampled = step_data_list

                for step_data in sampled:
                    optimizer.zero_grad()

                    loss, loss_dict = strategy.compute_pg_loss(
                        scd_model=scd_model,
                        step_data=step_data,
                        advantage=adv,
                        ref_state_dict=ref_state_dict,
                        device=device,
                        dtype=dtype,
                        patchifier=patchifier,
                        get_positions_for_frame=get_positions_for_frame,
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_pg += loss_dict["pg"].item()
                    total_kl += loss_dict["kl"].item() if isinstance(loss_dict["kl"], Tensor) else loss_dict["kl"]
                    n_steps += 1

        # ── Logging ──
        epoch_time = time.time() - epoch_start
        mean_reward = torch.tensor([t["reward"]["total"] for t in all_trajectories]).mean().item()

        log_data = {
            "epoch": epoch,
            "loss/total": total_loss / max(n_steps, 1),
            "loss/pg": total_pg / max(n_steps, 1),
            "loss/kl": total_kl / max(n_steps, 1),
            "grpo/mean_reward": mean_reward,
            "grpo/advantage_mean": advantages.mean().item(),
            "grpo/advantage_std": advantages.std().item(),
            "grpo/n_trajectories": len(all_trajectories),
            "grpo/n_train_steps": n_steps,
            "time/epoch": epoch_time,
        }

        # Per-reward breakdown
        for t in all_trajectories:
            for key, val in t["reward"].items():
                if key != "total" and isinstance(val, (int, float)):
                    log_data.setdefault(f"reward/{key}", []).append(val)
        for key in list(log_data.keys()):
            if isinstance(log_data[key], list):
                log_data[key] = sum(log_data[key]) / len(log_data[key])

        wandb.log(log_data)

        print(f"\n  Epoch {epoch + 1} summary:")
        print(f"    Loss: {log_data['loss/total']:.4f} (pg={log_data['loss/pg']:.4f}, kl={log_data['loss/kl']:.4f})")
        print(f"    Reward: {mean_reward:.4f}")
        print(f"    Steps: {n_steps}")
        print(f"    Time: {epoch_time:.0f}s")

        # ── Checkpoint ──
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            ckpt_path = output_dir / f"lora_epoch_{epoch + 1:04d}.safetensors"
            save_lora_checkpoint(scd_model, ckpt_path)
            print(f"    Saved: {ckpt_path}")

    wandb.finish()
    print("\n  Training complete!")


def rollout_one_video(
    scd_model: LTXSCDModel,
    prompt_embeds: Tensor,
    prompt_mask: Tensor,
    patchifier: VideoLatentPatchifier,
    get_positions_for_frame,
    base_sigmas: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    latent_channels: int,
    latent_h: int,
    latent_w: int,
    tokens_per_frame: int,
    num_chunks: int,
    total_latent_frames: int,
    CHUNK_LATENT: int,
    NEW_PER_CHUNK: int,
    generator: torch.Generator,
    timestep_fraction: float = 0.5,
) -> dict:
    """Run one AR SCD rollout with SDE log-prob capture.

    Follows scd_inference.py patterns for encoder/decoder split.
    At each denoising step, uses stochastic SDE step + Gaussian log-prob
    instead of deterministic Euler ODE.
    """
    all_step_data = []
    latent_frames = []
    chunk_boundaries = []
    prev_context = None

    for chunk_idx in range(num_chunks):
        if chunk_idx == 0:
            new_frames = CHUNK_LATENT
            context_frames = []
        else:
            new_frames = NEW_PER_CHUNK
            context_frames = [prev_context] if prev_context is not None else []

        kv_cache = KVCache.empty()
        kv_cache.is_cache_step = True
        chunk_generated = []
        prev_enc_features = None
        frame_pos = 0

        sigmas = base_sigmas.to(device=device, dtype=dtype)

        with torch.inference_mode():
            for f_idx in range(new_frames):
                # ══ ENCODER PASS ══
                if f_idx == 0 and not context_frames:
                    enc_latent = torch.zeros(1, latent_channels, 1, latent_h, latent_w, device=device, dtype=dtype)
                elif f_idx == 0 and context_frames:
                    enc_latent = context_frames[0]
                else:
                    enc_latent = chunk_generated[-1]

                enc_latent = enc_latent.to(device)
                patchified_enc = patchifier.patchify(enc_latent)
                enc_modality = Modality(
                    enabled=True,
                    latent=patchified_enc,
                    sigma=torch.zeros(1, device=device, dtype=dtype),
                    timesteps=torch.zeros(1, tokens_per_frame, device=device, dtype=dtype),
                    positions=get_positions_for_frame(frame_pos, target_device=device),
                    context=prompt_embeds,
                    context_mask=prompt_mask,
                )
                enc_out, _ = scd_model.forward_encoder(
                    video=enc_modality,
                    audio=None,
                    perturbations=None,
                    kv_cache=kv_cache,
                    tokens_per_frame=tokens_per_frame,
                )
                current_enc = enc_out.x.detach()
                frame_pos += 1

                # ══ SHIFT-BY-1 ══
                if prev_enc_features is not None:
                    dec_enc_ctx = prev_enc_features
                else:
                    dec_enc_ctx = torch.zeros(
                        1, tokens_per_frame, current_enc.shape[-1],
                        device=device, dtype=dtype,
                    )

                # ══ DECODER: Denoise with SDE log-prob capture ══
                dec_frame_idx = frame_pos - 1
                dec_positions = get_positions_for_frame(dec_frame_idx, target_device=device)

                x_t = torch.randn(
                    1, latent_channels, 1, latent_h, latent_w,
                    device=device, dtype=dtype, generator=generator,
                )

                n_steps = len(sigmas) - 1
                # Which steps to store for training (subsample to save memory)
                store_steps = set()
                if timestep_fraction < 1.0:
                    n_store = max(1, int(n_steps * timestep_fraction))
                    store_steps = set(sorted(random.sample(range(n_steps), min(n_store, n_steps))))
                else:
                    store_steps = set(range(n_steps))

                for step in range(n_steps):
                    sigma = sigmas[step]
                    sigma_next = sigmas[step + 1]

                    noisy_patch = patchifier.patchify(x_t)

                    # Decoder forward
                    dec_modality = Modality(
                        enabled=True,
                        latent=noisy_patch,
                        sigma=sigma.reshape(1).to(device=device, dtype=dtype),
                        timesteps=torch.full((1, tokens_per_frame), sigma.item(), device=device, dtype=dtype),
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

                    # SDE step with log-prob
                    x_next, log_prob, x_next_mean, noise_std = sde_step_with_logprob(
                        velocity, noisy_patch, sigma, sigma_next,
                    )

                    # Store trajectory data (move to CPU to save VRAM)
                    if step in store_steps:
                        all_step_data.append({
                            "x_t": noisy_patch.detach().cpu(),
                            "sigma": sigma.item(),
                            "x_next": x_next.detach().cpu(),
                            "x_next_mean": x_next_mean.detach().cpu(),
                            "noise_std": noise_std.item(),
                            "log_prob_old": log_prob.detach().cpu(),
                            "dt": (sigma_next - sigma).item(),
                            "enc_features": dec_enc_ctx.detach().cpu(),
                            "positions": dec_positions.detach().cpu(),
                            "frame_idx": dec_frame_idx,
                            "prompt_embeds": prompt_embeds.detach().cpu(),
                            "prompt_mask": prompt_mask.detach().cpu(),
                            # SL anchor target: rollout policy velocity (for sparse supervision)
                            "velocity_target": velocity.detach().cpu(),
                        })

                    # Update x_t for next step (use stochastic sample, not deterministic)
                    x_t = patchifier.unpatchify(
                        x_next.to(dtype),
                        output_shape=VideoLatentShape(
                            frames=1, height=latent_h, width=latent_w,
                            batch=1, channels=latent_channels,
                        ),
                    )

                prev_enc_features = current_enc
                chunk_generated.append(x_t.detach())

            # ── Frame assembly ──
            for i, frame in enumerate(chunk_generated):
                latent_frames.append(frame.cpu())
                if chunk_idx > 0 and i == 0:
                    chunk_boundaries.append(len(latent_frames) - 1)

            prev_context = chunk_generated[-1].detach().clone()
            del kv_cache, chunk_generated

    return {
        "step_data": all_step_data,
        "latent_frames": latent_frames,
        "chunk_boundaries": chunk_boundaries,
    }


def save_lora_checkpoint(scd_model: LTXSCDModel, path: Path) -> None:
    """Save LoRA weights from SCD model to safetensors."""
    from safetensors.torch import save_file

    lora_state = {}
    for name, param in scd_model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.cpu()

    save_file(lora_state, str(path))


if __name__ == "__main__":
    main()
