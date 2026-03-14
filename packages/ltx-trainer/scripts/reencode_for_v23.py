#!/usr/bin/env python3
# ruff: noqa: T201
"""Re-encode existing Ditto-1M dataset for LTX-2.3.

Reads video paths + captions from the existing metadata.json (encoded with LTX-2),
re-encodes latents with LTX-2.3 VAE and text embeddings with LTX-2.3 text encoder.

LTX-2.3 has a different VAE architecture (different decoder blocks, retrained encoder)
and different text encoder pipeline (FeatureExtractorV2, separate video/audio projections).

Output structure:
    /media/12TB/ddit_ditto_data_23/
    ├── latents/              # Video latents from 2.3 VAE [128, F, H, W]
    ├── conditions_final/     # Text embeddings from 2.3 text encoder
    └── metadata.json

Usage:
    cd packages/ltx-trainer

    # Phase 1: VAE encode (can run during other training on separate GPU)
    uv run python scripts/reencode_for_v23.py --skip-text --vae-device cuda:0

    # Phase 2: Text encode (needs ~14GB VRAM for 8-bit Gemma)
    uv run python scripts/reencode_for_v23.py --skip-vae --text-device cuda:1

    # Both at once (sequentially, same GPU):
    uv run python scripts/reencode_for_v23.py --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder, load_video_vae_encoder
from ltx_trainer.video_utils import read_video

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
OLD_DATA_DIR = Path("/media/12TB/ddit_ditto_data")
NEW_DATA_DIR = Path("/media/12TB/ddit_ditto_data_23")
MODEL_23_PATH = Path("/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors")
GEMMA_PATH = Path("/media/2TB/ltx-models/gemma")

# Video processing — must match training config
TARGET_WIDTH = 768
TARGET_HEIGHT = 448
NUM_FRAMES = 25


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
    parser = argparse.ArgumentParser(description="Re-encode dataset for LTX-2.3")
    parser.add_argument("--old-data", type=Path, default=OLD_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=NEW_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_23_PATH)
    parser.add_argument("--gemma-path", type=Path, default=GEMMA_PATH)
    parser.add_argument("--device", type=str, default="cuda:0", help="Default device for both phases")
    parser.add_argument("--vae-device", type=str, default=None, help="Override device for VAE")
    parser.add_argument("--text-device", type=str, default=None, help="Override device for text encoder")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--batch-start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--batch-end", type=int, default=None, help="End index (for splitting work)")
    args = parser.parse_args()

    vae_device = args.vae_device or args.device
    text_device = args.text_device or args.device

    # Load metadata from old dataset
    metadata_path = args.old_data / "metadata.json"
    with open(metadata_path) as f:
        old_meta = json.load(f)

    videos = old_meta["videos"]
    total = len(videos)

    # Apply range
    start = args.batch_start
    end = args.batch_end or total
    videos_slice = videos[start:end]
    print(f"Processing {len(videos_slice)} videos (indices {start}–{end-1} of {total})")

    # Create output dirs
    latents_dir = args.output_dir / "latents"
    conditions_dir = args.output_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: VAE encode with 2.3 ────────────────────────────────────
    if not args.skip_vae:
        print(f"\n{'='*60}")
        print(f"Phase 1: VAE encoding with LTX-2.3 on {vae_device}")
        print(f"  Model: {args.model_path}")
        print(f"='*60")

        print("Loading LTX-2.3 VAE encoder...")
        vae_encoder = load_video_vae_encoder(args.model_path, dtype=torch.bfloat16)
        vae_encoder = vae_encoder.to(vae_device)
        vae_encoder.eval()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(vae_device) / 1e9
            print(f"  VAE VRAM: {alloc:.1f}GB")

        encoded = 0
        skipped = 0
        failed = 0

        for entry in tqdm(videos_slice, desc="VAE encoding (2.3)"):
            idx = entry["id"]
            latent_path = latents_dir / f"{idx:06d}.pt"

            if latent_path.exists():
                skipped += 1
                continue

            try:
                frames, fps = read_video(entry["video_path"], max_frames=NUM_FRAMES)

                if frames.shape[0] < NUM_FRAMES:
                    logger.warning(f"Video {idx} has {frames.shape[0]} frames, need {NUM_FRAMES}. Skipping.")
                    failed += 1
                    continue

                frames = frames[:NUM_FRAMES]
                frames = resize_video(frames, TARGET_HEIGHT, TARGET_WIDTH)
                latent = encode_video(vae_encoder, frames, vae_device, torch.bfloat16)

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

                if encoded == 1:
                    print(f"\n  First latent shape: {latent.shape}, fps={fps}")

                if encoded % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed video {idx} ({entry['video_path']}): {e}")
                failed += 1

        del vae_encoder
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}, Failed: {failed}")
    else:
        print("\n[SKIP] VAE encoding (--skip-vae)")

    # ── Phase 2: Text encode with 2.3 ──────────────────────────────────
    if not args.skip_text:
        print(f"\n{'='*60}")
        print(f"Phase 2: Text encoding with LTX-2.3 on {text_device}")
        print(f"  Model: {args.model_path}")
        print(f"  Gemma: {args.gemma_path}")
        print(f"='*60")

        print("Loading LTX-2.3 text encoder (8-bit)...")
        text_encoder = load_text_encoder(
            checkpoint_path=args.model_path,
            gemma_model_path=args.gemma_path,
            device=text_device,
            dtype=torch.bfloat16,
            load_in_8bit=True,
        )
        text_encoder.eval()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(text_device) / 1e9
            print(f"  Text encoder VRAM: {alloc:.1f}GB")

        encoded = 0
        skipped = 0

        for entry in tqdm(videos_slice, desc="Text encoding (2.3)"):
            idx = entry["id"]
            cond_path = conditions_dir / f"{idx:06d}.pt"

            if cond_path.exists():
                skipped += 1
                continue

            # Only encode if we have a latent for this video
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

                if encoded == 1:
                    print(f"\n  First embedding shape: {video_embeds.shape}")

                if encoded % 50 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed text {idx}: {e}")

        del text_encoder
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}")
    else:
        print("\n[SKIP] Text encoding (--skip-text)")

    # ── Write metadata ────────────────────────────────────────────────────
    num_latents = len(list(latents_dir.glob("*.pt")))
    num_conditions = len(list(conditions_dir.glob("*.pt")))

    new_meta = {
        "description": (
            f"Ditto-1M subset re-encoded for LTX-2.3 (22B). "
            f"{num_latents} video latents, {num_conditions} text embeddings. "
            f"Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}, {NUM_FRAMES} frames."
        ),
        "source": "ditto_1m",
        "model_version": "ltx-2.3-22b-distilled",
        "num_samples": num_latents,
        "num_conditions": num_conditions,
        "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "num_frames": NUM_FRAMES,
        "vae_version": "ltx-2.3",
        "text_encoder_version": "ltx-2.3",
        "videos": videos,
    }

    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.output_dir}")
    print(f"  latents/:           {num_latents} files")
    print(f"  conditions_final/:  {num_conditions} files")

    if num_latents != num_conditions:
        print(f"\n  ⚠ Mismatch: {num_latents} latents vs {num_conditions} conditions!")
        print("  Re-run with --skip-vae to fill missing text embeddings")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
