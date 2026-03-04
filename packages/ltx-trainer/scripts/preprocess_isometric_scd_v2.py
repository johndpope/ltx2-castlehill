#!/usr/bin/env python3
"""Preprocess Scrya isometric videos for SCD training on distilled-isometric model.

Filters out lady/woman videos, encodes latents with VAE, computes Gemma embeddings.
Sequential model loading: VAE first (cuda:1, ~8GB), then Gemma 8-bit (cuda:1, ~16GB).

Usage:
    python scripts/preprocess_isometric_scd_v2.py
"""

import gc
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
WATCH_DIR = Path("/home/johndpope/scrya-downloads/Isometric 3D")
OUTPUT_DIR = Path("/media/2TB/omnitransfer/data/isometric_scd_v2")
MODEL_PATH = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled-isometric.safetensors"
TEXT_ENCODER_PATH = "/media/2TB/ltx-models/gemma"
DEVICE = "cuda:1"

TARGET_H = 704       # Nearest ÷32 to dominant 688
TARGET_W = 480       # Nearest ÷32 to dominant 464
TARGET_FRAMES = 25   # 1 second at 24fps, F%8==1 ✓
FPS = 24

# Keywords to filter out
EXCLUDE_KEYWORDS = ["woman", "lady", "girl", "sexy", "female", "breasts"]


def discover_videos() -> list[tuple[Path, str]]:
    """Find MP4s with prompts, excluding lady/woman content.

    Tries plain .txt first, then _combined.txt fallback (from Qwen captioning).
    """
    videos = []
    excluded = 0
    no_prompt = 0
    for mp4 in sorted(WATCH_DIR.glob("*.mp4")):
        txt = mp4.with_suffix(".txt")
        combined_txt = mp4.with_name(mp4.stem + "_combined.txt")
        if txt.exists():
            prompt = txt.read_text().strip()
        elif combined_txt.exists():
            prompt = combined_txt.read_text().strip()
        else:
            no_prompt += 1
            continue
        if any(kw in prompt.lower() for kw in EXCLUDE_KEYWORDS):
            excluded += 1
            continue
        videos.append((mp4, prompt))
    print(f"  Excluded {excluded} videos (keywords), {no_prompt} without prompts")
    return videos


def load_video_frames(path: Path, num_frames: int, target_h: int, target_w: int) -> torch.Tensor:
    """Load first num_frames from video, resize to target resolution.

    Returns: [1, C, F, H, W] tensor normalized to [-1, 1].
    """
    import torchvision.io as tvio

    # Read video
    video, _, info = tvio.read_video(str(path), pts_unit="sec", end_pts=num_frames / FPS + 0.1)
    # video: [T, H, W, C] uint8

    if video.shape[0] < num_frames:
        print(f"  Warning: {path.name} has {video.shape[0]} frames, need {num_frames}")
        # Pad by repeating last frame
        pad = num_frames - video.shape[0]
        video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

    video = video[:num_frames]  # [F, H, W, C]

    # Resize each frame
    frames = []
    for i in range(video.shape[0]):
        img = Image.fromarray(video[i].numpy())
        # Center crop to target aspect ratio
        w, h = img.size
        target_aspect = target_w / target_h
        source_aspect = w / h
        if abs(source_aspect - target_aspect) > 0.01:
            if source_aspect > target_aspect:
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                img = img.crop((start_x, 0, start_x + new_w, h))
            else:
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                img = img.crop((0, start_y, w, start_y + new_h))
        img = img.resize((target_w, target_h), Image.LANCZOS)
        t = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C]
        frames.append(t)

    tensor = torch.stack(frames)  # [F, H, W, C]
    tensor = tensor.permute(3, 0, 1, 2)  # [C, F, H, W]
    tensor = tensor.unsqueeze(0)  # [1, C, F, H, W]
    tensor = tensor * 2.0 - 1.0  # normalize to [-1, 1]
    return tensor


def main():
    # Discover videos
    videos = discover_videos()
    print(f"Found {len(videos)} clean isometric videos (excluded lady/woman content)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latents_dir = OUTPUT_DIR / "latents"
    conditions_dir = OUTPUT_DIR / "conditions_final"
    latents_dir.mkdir(exist_ok=True)
    conditions_dir.mkdir(exist_ok=True)

    # Check existing
    existing = set(int(f.stem) for f in latents_dir.glob("*.pt") if f.stem.isdigit())
    start_idx = max(existing) + 1 if existing else 0
    if existing:
        print(f"  {len(existing)} already processed, starting at index {start_idx}")

    # ── Phase 1: VAE encode all videos ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: VAE encoding {len(videos)} videos → latents")
    print(f"  Resolution: {TARGET_W}x{TARGET_H}, {TARGET_FRAMES} frames")
    print(f"{'='*60}")

    from ltx_trainer.model_loader import load_video_vae_encoder

    vae_encoder = load_video_vae_encoder(MODEL_PATH, dtype=torch.bfloat16)
    vae_encoder = vae_encoder.to(DEVICE)
    vae_encoder.eval()
    mem = torch.cuda.memory_allocated(1) / 1e9
    print(f"  VAE encoder loaded ({mem:.1f} GB)")

    latent_results = {}  # idx -> (video_path, prompt)
    idx = start_idx

    for video_path, prompt in tqdm(videos, desc="VAE encoding"):
        latent_path = latents_dir / f"{idx:06d}.pt"
        if latent_path.exists():
            latent_results[idx] = (video_path, prompt)
            idx += 1
            continue

        try:
            video_tensor = load_video_frames(video_path, TARGET_FRAMES, TARGET_H, TARGET_W)
            video_tensor = video_tensor.to(DEVICE, dtype=torch.bfloat16)

            with torch.inference_mode():
                latent = vae_encoder(video_tensor)

            latent_data = {
                "latents": latent.squeeze(0).cpu(),  # [C, F_lat, H_lat, W_lat]
                "num_frames": torch.tensor([latent.shape[2]]),  # latent frames, not raw video
                "height": torch.tensor([latent.shape[3]]),
                "width": torch.tensor([latent.shape[4]]),
            }

            # Atomic write
            tmp = latent_path.with_suffix(".tmp")
            torch.save(latent_data, tmp)
            os.rename(tmp, latent_path)

            latent_results[idx] = (video_path, prompt)
            idx += 1

        except Exception as e:
            print(f"  Failed {video_path.name}: {e}")

    del vae_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Encoded {len(latent_results)} videos, VAE unloaded")

    # ── Phase 2: Gemma embeddings ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: Computing Gemma embeddings for {len(latent_results)} captions")
    print(f"{'='*60}")

    from ltx_trainer.model_loader import load_text_encoder

    text_encoder = load_text_encoder(
        MODEL_PATH, TEXT_ENCODER_PATH,
        device=DEVICE, dtype=torch.bfloat16,
        load_in_8bit=True,
    )
    text_encoder.eval()
    mem = torch.cuda.memory_allocated(1) / 1e9
    print(f"  Gemma loaded ({mem:.1f} GB)")

    for i, (prompt_idx, (video_path, prompt)) in enumerate(tqdm(latent_results.items(), desc="Embedding")):
        condition_path = conditions_dir / f"{prompt_idx:06d}.pt"
        if condition_path.exists():
            continue

        try:
            with torch.inference_mode():
                video_embeds, audio_embeds, attention_mask = text_encoder(prompt)

            emb_data = {
                "video_prompt_embeds": video_embeds.squeeze(0).cpu().contiguous(),
                "audio_prompt_embeds": (audio_embeds.squeeze(0).cpu().contiguous()
                                        if audio_embeds is not None
                                        else video_embeds.squeeze(0).cpu().contiguous()),
                "prompt_attention_mask": attention_mask.squeeze(0).cpu().contiguous(),
                "is_final_embedding": True,
            }

            tmp = condition_path.with_suffix(".tmp")
            torch.save(emb_data, tmp)
            os.rename(tmp, condition_path)

        except Exception as e:
            print(f"  Failed embedding {video_path.name}: {e}")

    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Embeddings done, Gemma unloaded")

    # ── Save metadata ───────────────────────────────────────────────────────
    metadata = {
        "num_samples": len(latent_results),
        "resolution": f"{TARGET_W}x{TARGET_H}",
        "num_frames": TARGET_FRAMES,
        "fps": FPS,
        "source": "scrya-downloads/Isometric 3D (filtered)",
        "excluded_keywords": EXCLUDE_KEYWORDS,
        "samples": {
            str(k): {"video": str(v[0].name), "prompt": v[1]}
            for k, v in latent_results.items()
        },
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Done! {len(latent_results)} samples at {OUTPUT_DIR}")
    print(f"  Latents: {len(list(latents_dir.glob('*.pt')))}")
    print(f"  Conditions: {len(list(conditions_dir.glob('*.pt')))}")


if __name__ == "__main__":
    main()
