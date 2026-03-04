#!/usr/bin/env python3
# ruff: noqa: T201
"""Preprocess 122 isometric Grok videos for SCD training.

Encodes MP4+TXT pairs into precomputed latents + text embeddings for SCD LoRA
training. Uses combined prompts (_combined.txt from caption_first_frames.py)
when available, falling back to raw .txt prompts with automatic formatting.

Data flow:
    Isometric pair: (uuid.mp4, uuid_combined.txt)
                         ↓ VAE encode          ↓ Gemma encode
    Training:   latents/000000.pt    conditions_final/000000.pt

Output structure:
    /media/2TB/omnitransfer/data/isometric_scd/
    ├── latents/              # Video latents [128, F_lat, H_lat, W_lat]
    ├── conditions_final/     # Text embeddings (post-connector) [1024, 3840]
    └── metadata.json         # Full provenance info

Two-phase processing (never load VAE + text encoder simultaneously):
    Phase 1: VAE encode videos on cuda:1 (~8GB VRAM)
    Phase 2: Gemma text encode prompts on cuda:0 (~28GB VRAM)

Usage:
    cd ltx-trainer
    python scripts/preprocess_isometric_scd.py
    python scripts/preprocess_isometric_scd.py --skip-vae   # text only
    python scripts/preprocess_isometric_scd.py --dry-run     # preview
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
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR = Path("/home/johndpope/scrya-downloads/Isometric 3D")
OUTPUT_DIR = Path("/media/2TB/omnitransfer/data/isometric_scd_v2")
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
GEMMA_PATH = Path("/media/2TB/ltx-models/gemma")

# Video processing — most videos are 464×688 portrait
TARGET_WIDTH = 480     # 480 ÷ 32 = 15 ✓  (native 464, slight upscale)
TARGET_HEIGHT = 704    # 704 ÷ 32 = 22 ✓  (native 688, slight upscale)
NUM_FRAMES = 25        # 25 % 8 == 1 ✓  (~1 second at 24fps)

# Devices (dual GPU)
VAE_DEVICE = "cuda:1"          # RTX PRO 4000 (24GB) — VAE encoder
TEXT_ENCODER_DEVICE = "cuda:0"  # RTX 5090 (32GB) — Gemma text encoder

# Prompt formatting (match isometric_identity training captions)
BASE_PROMPTS = [
    "Static camera, fixed isometric viewpoint",
    "Fixed isometric angle, no camera motion",
    "Isometric 3D view, camera stays completely still",
    "3D isometric scene with static camera",
]
DEFAULT_BASE_PROMPT = BASE_PROMPTS[0]
PROMPT_SUFFIX = "No camera movement."


def is_person_pose_video(caption: str) -> bool:
    """Detect person-posing-on-background videos that aren't isometric scenes."""
    cap = caption.lower()
    is_pose = (
        ('stands confidently' in cap and ('suit' in cap or 'blazer' in cap or 'blouse' in cap or 'outfit' in cap or 'dress' in cap)) or
        ('stands with arms outstretched' in cap and 'city' not in cap and 'robot' not in cap) or
        ('stands confidently with hands on hips' in cap and 'city' not in cap) or
        ('lace lingerie' in cap) or ('lace bra' in cap) or
        ('crop top, black pants' in cap and 'city' not in cap) or
        ('jumpsuit with a' in cap and 'deep v' in cap) or
        ('hands clasped in prayer' in cap) or
        ('red blouse and black pants stands' in cap) or
        ('hands near her face' in cap) or
        ('plaid shirt stands with arms' in cap)
    )
    has_scene = any(w in cap for w in [
        'game r', 'city', 'kitchen', 'desk', 'room', 'grill', 'barbq',
        'downtown', 'landscape', 'building', 'street', 'office',
        'alien', 'robot', 'soldier', 'military', 'invasion', 'attack',
        'explosion', 'ufo', 'burning', 'menacing', 'presidential',
        'climate', 'anxiety', 'laughing', 'excited', 'trump',
        'flames', 'breaking', 'notification',
    ])
    return is_pose and not has_scene


def discover_pairs(input_dir: Path, *, filter_poses: bool = True) -> list[dict]:
    """Discover MP4+TXT pairs, preferring _combined.txt over raw .txt."""
    mp4_files = sorted(input_dir.glob("*.mp4"))
    pairs = []
    filtered_count = 0

    for mp4 in mp4_files:
        uuid = mp4.stem
        combined_txt = mp4.parent / f"{uuid}_combined.txt"
        raw_txt = mp4.with_suffix(".txt")

        if combined_txt.exists():
            prompt = combined_txt.read_text().strip()
            prompt_source = "combined"
        elif raw_txt.exists():
            prompt = raw_txt.read_text().strip()
            prompt_source = "raw"
        else:
            continue

        # Filter out person-posing-on-background videos
        if filter_poses and is_person_pose_video(prompt):
            filtered_count += 1
            continue

        pairs.append({
            "uuid": uuid,
            "mp4_path": str(mp4),
            "prompt": prompt,
            "prompt_source": prompt_source,
        })

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} person-pose videos")

    return pairs


def format_prompt(raw_prompt: str) -> str:
    """Format a raw prompt into training-style caption if needed.

    Combined prompts are already formatted. Raw prompts need wrapping:
        action prompt → "Static camera, fixed isometric viewpoint. {action}. No camera movement."
        scene prompt  → "Static camera, fixed isometric viewpoint. {scene}. No camera movement."
    """
    # If already formatted (starts with camera/isometric prefix), return as-is
    for prefix in BASE_PROMPTS:
        if raw_prompt.lower().startswith(prefix.lower()):
            return raw_prompt

    action = raw_prompt.strip().rstrip(".-")
    return f"{DEFAULT_BASE_PROMPT}. {action}. {PROMPT_SUFFIX}"


def resize_video(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize video frames to target dimensions.

    Args:
        frames: [F, C, H, W] tensor in [0, 1]
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
        description="Preprocess isometric Grok videos for SCD training"
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
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
    args = parser.parse_args()

    # Validate dimensions
    if args.target_width % 32 != 0 or args.target_height % 32 != 0:
        raise ValueError(f"Dimensions must be ÷32: {args.target_width}x{args.target_height}")
    if args.num_frames % 8 != 1:
        raise ValueError(f"num_frames must satisfy F%%8==1, got {args.num_frames}")

    # ── Step 1: Discover pairs ────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Discovering MP4+TXT pairs")
    print("=" * 60)

    pairs = discover_pairs(args.input_dir)
    print(f"Found {len(pairs)} pairs in {args.input_dir}")

    combined_count = sum(1 for p in pairs if p["prompt_source"] == "combined")
    raw_count = sum(1 for p in pairs if p["prompt_source"] == "raw")
    print(f"  {combined_count} with combined prompts (Qwen VL captioned)")
    print(f"  {raw_count} with raw prompts (will be auto-formatted)")

    # Format raw prompts
    for p in pairs:
        if p["prompt_source"] == "raw":
            p["formatted_prompt"] = format_prompt(p["prompt"])
        else:
            p["formatted_prompt"] = p["prompt"]

    # Resolution math
    lat_f = (args.num_frames - 1) // 8 + 1
    lat_h = args.target_height // 32
    lat_w = args.target_width // 32
    tokens_per_frame = lat_h * lat_w
    total_tokens = lat_f * tokens_per_frame

    print(f"\n  Resolution: {args.target_width}x{args.target_height} → latent {lat_w}x{lat_h}")
    print(f"  Frames: {args.num_frames} → {lat_f} latent frames")
    print(f"  Tokens: {tokens_per_frame}/frame × {lat_f} frames = {total_tokens} total")

    # Show sample prompts
    print(f"\n  Sample prompts:")
    for p in pairs[:3]:
        tag = "combined" if p["prompt_source"] == "combined" else "formatted"
        print(f"    [{tag}] {p['formatted_prompt'][:80]}...")

    if args.dry_run:
        print(f"\n[DRY RUN] Would produce:")
        print(f"  {len(pairs)} latents in {args.output_dir / 'latents'}")
        print(f"  {len(pairs)} text embeddings in {args.output_dir / 'conditions_final'}")
        print(f"  Latent shape: [128, {lat_f}, {lat_h}, {lat_w}]")
        return

    # Create output dirs
    latents_dir = args.output_dir / "latents"
    conditions_dir = args.output_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: VAE encode videos ────────────────────────────────────────
    if not args.skip_vae:
        print("\n" + "=" * 60)
        print("Phase 1: VAE encoding videos")
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

        for idx, pair in enumerate(tqdm(pairs, desc="VAE encoding")):
            latent_path = latents_dir / f"{idx:06d}.pt"

            if latent_path.exists():
                skipped += 1
                continue

            try:
                frames, fps = read_video(pair["mp4_path"], max_frames=args.num_frames)

                if frames.shape[0] < args.num_frames:
                    logger.warning(
                        f"Video {idx} ({pair['uuid']}) has only {frames.shape[0]} frames, "
                        f"need {args.num_frames}. Skipping."
                    )
                    failed += 1
                    continue

                # Trim to exact frame count
                frames = frames[:args.num_frames]

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
                        "fps": torch.tensor([24.0]),
                    },
                    latent_path,
                )

                encoded += 1

                if idx == 0:
                    print(f"\n  First latent shape: {latent.shape}")

            except Exception as e:
                logger.error(f"Failed to encode video {idx} ({pair['uuid']}): {e}")
                failed += 1
                continue

        # Cleanup VAE
        del vae_encoder
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n  Encoded: {encoded}, Skipped (cached): {skipped}, Failed: {failed}")
    else:
        print("\n[SKIP] VAE encoding (--skip-vae)")

    # ── Phase 2: Text encode prompts ──────────────────────────────────────
    if not args.skip_text:
        print("\n" + "=" * 60)
        print("Phase 2: Text encoding prompts")
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

        for idx, pair in enumerate(tqdm(pairs, desc="Text encoding")):
            cond_path = conditions_dir / f"{idx:06d}.pt"

            if cond_path.exists():
                skipped += 1
                continue

            # Only encode if we have a corresponding latent
            latent_path = latents_dir / f"{idx:06d}.pt"
            if not latent_path.exists():
                continue

            prompt = pair["formatted_prompt"]

            try:
                with torch.inference_mode():
                    video_embeds, audio_embeds, attention_mask = text_encoder(prompt)

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
                    print(f"  Prompt: {prompt[:80]}...")

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

    # ── Write metadata ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Writing metadata")
    print("=" * 60)

    num_latents = len(list(latents_dir.glob("*.pt")))
    num_conditions = len(list(conditions_dir.glob("*.pt")))

    metadata = {
        "task_type": "isometric_scd",
        "description": (
            f"122 Grok isometric videos for SCD overfitting. "
            f"Resolution: {args.target_width}x{args.target_height}, {args.num_frames} frames."
        ),
        "source": "grok_isometric_scrya_downloads",
        "num_samples": num_latents,
        "num_conditions": num_conditions,
        "resolution": f"{args.target_width}x{args.target_height}",
        "num_frames": args.num_frames,
        "has_final_embeddings": True,
        "conditions_final_dir": "conditions_final",
        "pairs": [
            {
                "id": idx,
                "uuid": pair["uuid"],
                "mp4_path": pair["mp4_path"],
                "prompt": pair["formatted_prompt"],
                "raw_prompt": pair["prompt"],
                "prompt_source": pair["prompt_source"],
            }
            for idx, pair in enumerate(pairs)
        ],
    }

    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nDataset written to {args.output_dir}")
    print(f"  latents/:           {num_latents} files")
    print(f"  conditions_final/:  {num_conditions} files")
    print(f"  metadata.json:      {len(pairs)} entries")

    if num_latents != num_conditions:
        print(f"\n  ⚠ Mismatch: {num_latents} latents vs {num_conditions} conditions!")
        print(f"  Re-run with --skip-vae to fill missing text embeddings")


if __name__ == "__main__":
    main()
