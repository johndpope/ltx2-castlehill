#!/usr/bin/env python3
# ruff: noqa: T201
"""Preprocess Ditto-1M videos for VFM training with 2.3 VAE.

Encodes raw MP4 videos → VAE latents + Gemma text embeddings.
Two-phase processing so VAE and text encoder don't compete for VRAM.

Data flow:
    MP4 + caption → VAE encode → latents_19b/NNNNNN.pt
    caption → Gemma encode → conditions_final/NNNNNN.pt

Output structure:
    /media/12TB/ddit_ditto_data/
    ├── latents_19b/          # Video latents [128, F_lat, H_lat, W_lat]
    ├── conditions_final/     # Text embeddings (post-connector, 3840-dim)
    └── metadata.json

Usage:
    cd packages/ltx-trainer
    # Phase 1: VAE encode on RTX 4000 (cuda:0) — safe to run during training
    uv run python scripts/preprocess_ditto_5k.py --subset-size 5000 --skip-text --vae-device cuda:0

    # Phase 2: Text encode (after training finishes, or on a free GPU)
    uv run python scripts/preprocess_ditto_5k.py --subset-size 5000 --skip-vae --text-device cuda:1

    # Or both at once (needs both GPUs free):
    uv run python scripts/preprocess_ditto_5k.py --subset-size 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder, load_video_vae_encoder
from ltx_trainer.video_utils import read_video

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DITTO_ROOT = Path("/media/12TB/Ditto-1M")
VIDEOS_DIR = DITTO_ROOT / "videos_extracted"
CAPTION_JSON = DITTO_ROOT / "source_video_captions" / "source_video_captions_sorted.json"

OUTPUT_DIR = Path("/media/12TB/ddit_ditto_data")
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors")
GEMMA_PATH = Path("/media/2TB/ltx-models/gemma")

# Video processing — match current training config
TARGET_WIDTH = 768     # 768 ÷ 32 = 24 ✓
TARGET_HEIGHT = 448    # 448 ÷ 32 = 14 ✓
NUM_FRAMES = 25        # 25 % 8 == 1 ✓

# Devices
VAE_DEVICE = "cuda:0"          # RTX PRO 4000 (24GB)
TEXT_ENCODER_DEVICE = "cuda:0"  # Same GPU, phases don't overlap


def load_caption_map(caption_json: Path) -> dict[str, str]:
    """Load source video captions keyed by hash."""
    with open(caption_json) as f:
        entries = json.load(f)

    caption_map = {}
    for entry in entries:
        filename = Path(entry["path"]).stem
        caption_map[filename] = entry["caption"]

    logger.info(f"Loaded {len(caption_map)} source captions")
    return caption_map


def discover_videos(
    videos_dir: Path,
    caption_map: dict[str, str],
    target_count: int,
    seed: int = 42,
) -> list[dict]:
    """Discover MP4 videos that have captions, randomly sample target_count.

    Each Ditto video is named like: hash_N.mp4 (N = variant number).
    We strip _N to find the base hash for caption lookup.
    """
    import re

    all_videos = sorted(videos_dir.rglob("*.mp4"))
    print(f"Found {len(all_videos)} total MP4 files")

    candidates = []
    for video_path in all_videos:
        stem = video_path.stem
        # Strip _N suffix to get base hash
        match = re.match(r"^(.+?)_(\d+)$", stem)
        base_hash = match.group(1) if match else stem

        caption = caption_map.get(base_hash)
        if caption:
            candidates.append({
                "video_path": str(video_path),
                "caption": caption,
                "hash": base_hash,
            })

    print(f"Found {len(candidates)} videos with captions")

    # Randomly sample
    rng = random.Random(seed)
    if len(candidates) > target_count:
        candidates = rng.sample(candidates, target_count)
    else:
        logger.warning(f"Only {len(candidates)} videos with captions, wanted {target_count}")

    print(f"Selected {len(candidates)} videos for processing")
    return candidates


def resize_video(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize video frames. Input: [F, C, H, W] in [0, 1]."""
    return F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)


def encode_video(
    vae_encoder: torch.nn.Module,
    frames: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode video frames to VAE latent. Returns [128, F_lat, H_lat, W_lat]."""
    batch = rearrange(frames, "f c h w -> 1 c f h w") * 2.0 - 1.0
    batch = batch.to(device, dtype=dtype)
    with torch.inference_mode():
        latent = vae_encoder(batch)
    return latent.squeeze(0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Ditto-1M for VFM training (2.3 VAE)")
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--gemma-path", type=Path, default=GEMMA_PATH)
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--vae-device", type=str, default=VAE_DEVICE)
    parser.add_argument("--text-device", type=str, default=TEXT_ENCODER_DEVICE)
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.target_width % 32 != 0 or args.target_height % 32 != 0:
        raise ValueError(f"Dimensions must be ÷32: {args.target_width}x{args.target_height}")
    if args.num_frames % 8 != 1:
        raise ValueError(f"num_frames must satisfy F%%8==1, got {args.num_frames}")

    # ── Step 1: Discover videos ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Discovering videos with captions")
    print("=" * 60)

    caption_map = load_caption_map(CAPTION_JSON)
    videos = discover_videos(VIDEOS_DIR, caption_map, args.subset_size, seed=args.seed)

    lat_f = (args.num_frames - 1) // 8 + 1
    lat_h = args.target_height // 32
    lat_w = args.target_width // 32
    print(f"\n  Resolution: {args.target_width}x{args.target_height} → latent [{lat_w}x{lat_h}]")
    print(f"  Frames: {args.num_frames} → {lat_f} latent frames")
    print(f"  Latent shape: [128, {lat_f}, {lat_h}, {lat_w}]")

    if args.dry_run:
        print(f"\n[DRY RUN] Would produce {len(videos)} samples in {args.output_dir}")
        return

    # Create output dirs
    latents_dir = args.output_dir / "latents_19b"
    conditions_dir = args.output_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: VAE encode ──────────────────────────────────────────────
    if not args.skip_vae:
        print("\n" + "=" * 60)
        print(f"Phase 1: VAE encoding {len(videos)} videos on {args.vae_device}")
        print("=" * 60)

        print(f"Loading VAE encoder from {args.model_path}...")
        vae_encoder = load_video_vae_encoder(args.model_path, dtype=torch.bfloat16)
        vae_encoder = vae_encoder.to(args.vae_device)
        vae_encoder.eval()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(args.vae_device) / 1e9
            print(f"  VAE VRAM: {alloc:.1f}GB")

        encoded = 0
        skipped = 0
        failed = 0

        for idx, entry in enumerate(tqdm(videos, desc="VAE encoding")):
            latent_path = latents_dir / f"{idx:06d}.pt"

            if latent_path.exists():
                skipped += 1
                continue

            try:
                frames, fps = read_video(entry["video_path"], max_frames=args.num_frames)

                if frames.shape[0] < args.num_frames:
                    logger.warning(f"Video {idx} has {frames.shape[0]} frames, need {args.num_frames}. Skipping.")
                    failed += 1
                    continue

                frames = frames[:args.num_frames]
                frames = resize_video(frames, args.target_height, args.target_width)
                latent = encode_video(vae_encoder, frames, args.vae_device, torch.bfloat16)

                torch.save(
                    {
                        "latents": latent,
                        "num_frames": torch.tensor([latent.shape[1]]),
                        "height": torch.tensor([latent.shape[2]]),
                        "width": torch.tensor([latent.shape[3]]),
                        "fps": torch.tensor([fps if fps > 0 else 20.0]),
                    },
                    latent_path,
                )
                encoded += 1

                if idx == 0:
                    print(f"\n  First latent shape: {latent.shape}, fps={fps}")

                # Periodic cache clear
                if encoded % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to encode video {idx} ({entry['video_path']}): {e}")
                failed += 1

        del vae_encoder
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}, Failed: {failed}")
    else:
        print("\n[SKIP] VAE encoding (--skip-vae)")

    # ── Phase 2: Text encode captions ────────────────────────────────────
    if not args.skip_text:
        print("\n" + "=" * 60)
        print(f"Phase 2: Text encoding captions on {args.text_device}")
        print("=" * 60)

        print(f"Loading Gemma text encoder (8-bit) on {args.text_device}...")
        text_encoder = load_text_encoder(
            checkpoint_path=args.model_path,
            gemma_model_path=args.gemma_path,
            device=args.text_device,
            dtype=torch.bfloat16,
            load_in_8bit=True,
        )
        text_encoder.eval()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(args.text_device) / 1e9
            print(f"  Text encoder VRAM: {alloc:.1f}GB")

        encoded = 0
        skipped = 0

        for idx, entry in enumerate(tqdm(videos, desc="Text encoding")):
            cond_path = conditions_dir / f"{idx:06d}.pt"

            if cond_path.exists():
                skipped += 1
                continue

            latent_path = latents_dir / f"{idx:06d}.pt"
            if not latent_path.exists():
                continue

            try:
                with torch.inference_mode():
                    video_embeds, audio_embeds, attention_mask = text_encoder(entry["caption"])

                torch.save(
                    {
                        "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
                        "audio_prompt_embeds": (
                            audio_embeds[0].cpu().contiguous()
                            if audio_embeds is not None
                            else video_embeds[0].cpu().contiguous()
                        ),
                        "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
                        "is_final_embedding": True,
                    },
                    cond_path,
                )
                encoded += 1

                if encoded % 50 == 0:
                    torch.cuda.empty_cache()

                if idx == 0:
                    print(f"\n  First embedding shape: {video_embeds.shape}")

            except Exception as e:
                logger.error(f"Failed to encode text {idx}: {e}")

        del text_encoder
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}")
    else:
        print("\n[SKIP] Text encoding (--skip-text)")

    # ── Write metadata ───────────────────────────────────────────────────
    num_latents = len(list(latents_dir.glob("*.pt")))
    num_conditions = len(list(conditions_dir.glob("*.pt")))

    metadata = {
        "description": (
            f"Ditto-1M subset for VFM training (2.3 VAE). {num_latents} video latents "
            f"with source captions. Resolution: {args.target_width}x{args.target_height}, "
            f"{args.num_frames} frames."
        ),
        "source": "ditto_1m",
        "num_samples": num_latents,
        "num_conditions": num_conditions,
        "resolution": f"{args.target_width}x{args.target_height}",
        "num_frames": args.num_frames,
        "vae_version": "ltx-2.3",
        "videos": [
            {"id": idx, "caption": v["caption"], "video_path": v["video_path"]}
            for idx, v in enumerate(videos)
        ],
    }

    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.output_dir}")
    print(f"  latents_19b/:       {num_latents} files")
    print(f"  conditions_final/:  {num_conditions} files")

    if num_latents != num_conditions:
        print(f"\n  ⚠ Mismatch: {num_latents} latents vs {num_conditions} conditions!")
        print(f"  Re-run with --skip-vae to fill missing text embeddings")


if __name__ == "__main__":
    main()
