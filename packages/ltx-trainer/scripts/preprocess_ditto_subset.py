#!/usr/bin/env python3
# ruff: noqa: T201
"""Preprocess a Ditto-1M subset for SCD training.

Selects N video editing pairs from Ditto-1M, encodes edited videos to VAE latents,
and computes text embeddings for edit instructions. Produces a dataset ready for
vanilla SCD training.

Data flow:
    Ditto pair: (source.mp4, edited.mp4, "Remove the fox")
                                ↓ VAE encode        ↓ Gemma encode
    Training:         latents/000000.pt    conditions_final/000000.pt

Output structure:
    /media/2TB/omnitransfer/data/ditto_subset/
    ├── latents/              # Edited video latents [128, F_lat, H_lat, W_lat]
    ├── conditions_final/     # Edit instruction embeddings (post-connector)
    └── metadata.json         # Full provenance info

Two-phase processing (never load VAE + text encoder simultaneously):
    Phase 1: VAE encode edited videos on cuda:1 (~8GB VRAM)
    Phase 2: Gemma text encode instructions on cuda:0 (~28GB VRAM)

Usage:
    cd packages/ltx-trainer
    uv run python scripts/preprocess_ditto_subset.py --subset-size 500
    uv run python scripts/preprocess_ditto_subset.py --dry-run --subset-size 500
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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
SUBSET_JSON = DITTO_ROOT / "local_50k_subset.json"
CAPTION_JSON = DITTO_ROOT / "source_video_captions" / "source_video_captions_sorted.json"
VIDEOS_DIR = DITTO_ROOT / "videos_extracted"

OUTPUT_DIR = Path("/media/2TB/omnitransfer/data/ditto_subset")
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
GEMMA_PATH = Path("/media/2TB/ltx-models/gemma")

# Video processing
TARGET_WIDTH = 768     # 768 ÷ 32 = 24 ✓
TARGET_HEIGHT = 448    # 448 ÷ 32 = 14 ✓
NUM_FRAMES = 25        # 25 % 8 == 1 ✓

# Devices (dual GPU)
VAE_DEVICE = "cuda:1"          # RTX PRO 4000 (24GB) - VAE encoder
TEXT_ENCODER_DEVICE = "cuda:0"  # RTX 5090 (32GB) - Gemma text encoder


def load_subset_metadata(subset_json: Path, max_scan: int | None = None) -> list[dict]:
    """Load Ditto-1M subset entries."""
    with open(subset_json) as f:
        entries = json.load(f)
    if max_scan is not None:
        entries = entries[:max_scan]
    return entries


def load_caption_map(caption_json: Path) -> dict[str, str]:
    """Load source video captions keyed by hash.

    Caption file has entries like:
        {"path": "0000/hash.mp4", "caption": "..."}

    Ditto subset paths look like:
        "local/0281/hash_N.mp4"

    We key by the hash (without _N suffix) for lookup.
    """
    with open(caption_json) as f:
        entries = json.load(f)

    caption_map = {}
    for entry in entries:
        # Extract hash from path like "0000/hash.mp4"
        filename = Path(entry["path"]).stem  # "hash"
        caption_map[filename] = entry["caption"]

    logger.info(f"Loaded {len(caption_map)} source captions")
    return caption_map


def extract_hash(ditto_path: str) -> str:
    """Extract the base hash from a Ditto video path.

    "local/0281/c4a281d692762e71e7ac7514d4ac1d80_3.mp4"
    → "c4a281d692762e71e7ac7514d4ac1d80"
    """
    stem = Path(ditto_path).stem  # "c4a281d692762e71e7ac7514d4ac1d80_3"
    # Strip trailing _N suffix (variant number)
    match = re.match(r"^(.+?)_(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def validate_pairs(
    entries: list[dict],
    videos_dir: Path,
    caption_map: dict[str, str],
    target_count: int,
) -> list[dict]:
    """Filter entries for valid pairs (both videos exist) up to target_count."""
    valid = []
    skipped_missing = 0
    skipped_caption = 0

    for entry in entries:
        if len(valid) >= target_count:
            break

        source_path = videos_dir / entry["source_path"]
        edited_path = videos_dir / entry["edited_path"]

        if not source_path.exists() or not edited_path.exists():
            skipped_missing += 1
            continue

        # Look up source caption
        source_hash = extract_hash(entry["source_path"])
        source_caption = caption_map.get(source_hash)

        valid.append({
            "instruction": entry["instruction"],
            "source_path": str(source_path),
            "edited_path": str(edited_path),
            "source_caption": source_caption or "A video scene",
            "source_hash": source_hash,
        })

    if skipped_missing:
        logger.warning(f"Skipped {skipped_missing} entries with missing videos")

    logger.info(
        f"Validated {len(valid)} pairs from {len(entries)} entries "
        f"({len(valid) - sum(1 for v in valid if v['source_caption'] == 'A video scene')}/{len(valid)} with captions)"
    )
    return valid


def resize_video(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize video frames to target dimensions.

    Args:
        frames: [F, C, H, W] tensor in [0, 1]
        target_h: Target height (must be ÷32)
        target_w: Target width (must be ÷32)

    Returns:
        [F, C, target_h, target_w] tensor
    """
    return F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)


def encode_video(
    vae_encoder: torch.nn.Module,
    frames: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode video frames to VAE latent.

    Args:
        frames: [F, C, H, W] tensor in [0, 1]

    Returns:
        Latent tensor [128, F_lat, H_lat, W_lat]
    """
    # [F, C, H, W] → [1, C, F, H, W], normalize to [-1, 1]
    batch = rearrange(frames, "f c h w -> 1 c f h w") * 2.0 - 1.0
    batch = batch.to(device, dtype=dtype)

    with torch.inference_mode():
        latent = vae_encoder(batch)

    return latent.squeeze(0).cpu()  # [128, F_lat, H_lat, W_lat]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Ditto-1M subset for SCD training"
    )
    parser.add_argument(
        "--subset-size", type=int, default=500,
        help="Number of valid pairs to select",
    )
    parser.add_argument(
        "--max-scan", type=int, default=None,
        help="Max entries to scan from subset JSON (default: scan until subset-size valid pairs found)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output dataset directory",
    )
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH,
        help="LTX-2 model checkpoint (for VAE + text encoder)",
    )
    parser.add_argument(
        "--gemma-path", type=Path, default=GEMMA_PATH,
        help="Path to Gemma model directory",
    )
    parser.add_argument(
        "--target-width", type=int, default=TARGET_WIDTH,
        help="Target width (must be ÷32)",
    )
    parser.add_argument(
        "--target-height", type=int, default=TARGET_HEIGHT,
        help="Target height (must be ÷32)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=NUM_FRAMES,
        help="Frames per clip (must satisfy F%%8==1)",
    )
    parser.add_argument(
        "--vae-device", type=str, default=VAE_DEVICE,
        help="Device for VAE encoder",
    )
    parser.add_argument(
        "--text-device", type=str, default=TEXT_ENCODER_DEVICE,
        help="Device for text encoder",
    )
    parser.add_argument(
        "--skip-vae", action="store_true",
        help="Skip VAE encoding (latents already exist)",
    )
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Skip text encoding (conditions_final already exist)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stats only, don't encode",
    )
    args = parser.parse_args()

    # Validate dimensions
    if args.target_width % 32 != 0 or args.target_height % 32 != 0:
        raise ValueError(f"Dimensions must be ÷32: {args.target_width}x{args.target_height}")
    if args.num_frames % 8 != 1:
        raise ValueError(f"num_frames must satisfy F%%8==1, got {args.num_frames}")

    # ── Step 1: Load metadata ────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading Ditto-1M metadata")
    print("=" * 60)

    max_scan = args.max_scan or min(args.subset_size * 2, 50000)
    entries = load_subset_metadata(SUBSET_JSON, max_scan=max_scan)
    print(f"Loaded {len(entries)} subset entries (scanning up to {max_scan})")

    caption_map = load_caption_map(CAPTION_JSON)
    print(f"Loaded {len(caption_map)} source video captions")

    # ── Step 2: Validate pairs ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Validating video pairs")
    print("=" * 60)

    valid_pairs = validate_pairs(entries, VIDEOS_DIR, caption_map, args.subset_size)
    if len(valid_pairs) < args.subset_size:
        logger.warning(
            f"Only found {len(valid_pairs)}/{args.subset_size} valid pairs. "
            f"Increase --max-scan to scan more entries."
        )

    print(f"\nSelected {len(valid_pairs)} valid pairs")
    print(f"  Example instruction: {valid_pairs[0]['instruction'][:80]}...")
    print(f"  Example source: {Path(valid_pairs[0]['source_path']).name}")
    print(f"  Example edited: {Path(valid_pairs[0]['edited_path']).name}")

    # Resolution math
    lat_f = (args.num_frames - 1) // 8 + 1
    lat_h = args.target_height // 32
    lat_w = args.target_width // 32
    tokens_per_frame = lat_h * lat_w
    total_tokens = lat_f * tokens_per_frame
    print(f"\n  Resolution: {args.target_width}x{args.target_height} → latent {lat_w}x{lat_h}")
    print(f"  Frames: {args.num_frames} → {lat_f} latent frames")
    print(f"  Tokens: {tokens_per_frame}/frame × {lat_f} frames = {total_tokens} total")

    if args.dry_run:
        print(f"\n[DRY RUN] Would produce:")
        print(f"  {len(valid_pairs)} latents in {args.output_dir / 'latents'}")
        print(f"  {len(valid_pairs)} text embeddings in {args.output_dir / 'conditions_final'}")
        print(f"  Latent shape: [128, {lat_f}, {lat_h}, {lat_w}]")
        return

    # Create output dirs
    latents_dir = args.output_dir / "latents"
    conditions_dir = args.output_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: VAE encode edited videos ────────────────────────────────
    if not args.skip_vae:
        print("\n" + "=" * 60)
        print("Phase 1: VAE encoding edited videos")
        print("=" * 60)

        print(f"Loading VAE encoder on {args.vae_device}...")
        vae_encoder = load_video_vae_encoder(args.model_path, dtype=torch.bfloat16)
        vae_encoder = vae_encoder.to(args.vae_device)
        vae_encoder.eval()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(args.vae_device) / 1e9
            print(f"  VAE VRAM: {alloc:.1f}GB")

        encoded = 0
        skipped = 0
        failed = 0

        for idx, pair in enumerate(tqdm(valid_pairs, desc="VAE encoding")):
            latent_path = latents_dir / f"{idx:06d}.pt"

            if latent_path.exists():
                skipped += 1
                continue

            try:
                # Read edited video
                frames, fps = read_video(pair["edited_path"], max_frames=args.num_frames)

                if frames.shape[0] < args.num_frames:
                    logger.warning(
                        f"Video {idx} has only {frames.shape[0]} frames, need {args.num_frames}. Skipping."
                    )
                    failed += 1
                    continue

                # Trim to exact frame count
                frames = frames[: args.num_frames]

                # Resize to target dimensions
                frames = resize_video(frames, args.target_height, args.target_width)

                # Encode
                latent = encode_video(vae_encoder, frames, args.vae_device, torch.bfloat16)

                # Save in PrecomputedDataset format
                torch.save(
                    {
                        "latents": latent,  # [128, F_lat, H_lat, W_lat]
                        "num_frames": torch.tensor([latent.shape[1]]),
                        "height": torch.tensor([latent.shape[2]]),
                        "width": torch.tensor([latent.shape[3]]),
                        "fps": torch.tensor([20.0]),
                    },
                    latent_path,
                )

                encoded += 1

                if idx == 0:
                    print(f"\n  First latent shape: {latent.shape}")

            except Exception as e:
                logger.error(f"Failed to encode video {idx}: {e}")
                failed += 1
                continue

        # Cleanup VAE
        del vae_encoder
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}, Failed: {failed}")
    else:
        print("\n[SKIP] VAE encoding (--skip-vae)")

    # ── Phase 2: Text encode edit instructions ───────────────────────────
    if not args.skip_text:
        print("\n" + "=" * 60)
        print("Phase 2: Text encoding edit instructions")
        print("=" * 60)

        print(f"Loading text encoder on {args.text_device}...")
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

        for idx, pair in enumerate(tqdm(valid_pairs, desc="Text encoding")):
            cond_path = conditions_dir / f"{idx:06d}.pt"

            if cond_path.exists():
                skipped += 1
                continue

            # Only encode if we have a corresponding latent
            latent_path = latents_dir / f"{idx:06d}.pt"
            if not latent_path.exists():
                continue

            instruction = pair["instruction"]

            try:
                with torch.inference_mode():
                    video_embeds, audio_embeds, attention_mask = text_encoder(instruction)

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
                continue

        # Cleanup
        del text_encoder
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}")
    else:
        print("\n[SKIP] Text encoding (--skip-text)")

    # ── Write metadata ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Writing metadata")
    print("=" * 60)

    # Count actual files produced
    num_latents = len(list(latents_dir.glob("*.pt")))
    num_conditions = len(list(conditions_dir.glob("*.pt")))

    metadata = {
        "task_type": "video_editing",
        "description": (
            f"Ditto-1M subset for SCD training. {num_latents} edited video latents "
            f"with edit instruction text embeddings. "
            f"Resolution: {args.target_width}x{args.target_height}, {args.num_frames} frames."
        ),
        "source": "ditto_1m_local_50k_subset",
        "num_samples": num_latents,
        "num_conditions": num_conditions,
        "resolution": f"{args.target_width}x{args.target_height}",
        "num_frames": args.num_frames,
        "has_final_embeddings": True,
        "conditions_final_dir": "conditions_final",
        "pairs": [
            {
                "id": idx,
                "instruction": pair["instruction"],
                "source_path": pair["source_path"],
                "edited_path": pair["edited_path"],
                "source_caption": pair["source_caption"],
                "caption": pair["instruction"],  # For compute_final_embeddings.py compat
            }
            for idx, pair in enumerate(valid_pairs)
        ],
    }

    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDataset written to {args.output_dir}")
    print(f"  latents/:           {num_latents} files")
    print(f"  conditions_final/:  {num_conditions} files")
    print(f"  metadata.json:      {len(valid_pairs)} entries")

    if num_latents != num_conditions:
        print(f"\n  ⚠ Mismatch: {num_latents} latents vs {num_conditions} conditions!")
        print(f"  Re-run with --skip-vae to fill missing text embeddings")


if __name__ == "__main__":
    main()
