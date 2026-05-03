#!/usr/bin/env python3
"""Preprocess Scrya-downloaded videos for LTX audio-video training.

Scrya download format:
    UUID.mp4         — video clip (~6s, 24fps, AAC audio)
    UUID.txt         — caption (short motion description)
    UUID_thumb.jpg   — thumbnail (ignored)

Output structure:
    /media/12TB/scrya_realistic_photo/
    ├── latents/            [B, 128, F_lat, H_lat, W_lat] video VAE latents
    ├── audio_latents/      audio VAE latents
    ├── conditions_final/   Gemma text embeddings (post-connector)
    └── metadata.json

Resolution buckets:
    560x560 → 544x544x145  (544 = 17×32, 145 % 8 == 1 ✓)
    464x688 → 448x672x145  (448 = 14×32, 672 = 21×32, 145 % 8 == 1 ✓)

Usage:
    cd packages/ltx-trainer

    # Phase 1: encode latents + audio (uses cuda:1 for VAE, leaves cuda:0 for training)
    uv run python scripts/preprocess_scrya.py --phase latents

    # Phase 2: encode text embeddings
    uv run python scripts/preprocess_scrya.py --phase text

    # Both phases (needs both GPUs free):
    uv run python scripts/preprocess_scrya.py --phase all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRYA_DIR = Path("/home/johndpope/scrya-downloads/Realistic Photo")
OUTPUT_DIR = Path("/media/12TB/scrya_realistic_photo")
MODEL_PATH = "/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
GEMMA_PATH = "/media/2TB/ltx-models/gemma"

MANIFEST_PATH = SCRYA_DIR / "dataset.json"   # must be in same dir as videos (data_root)

# Two buckets to handle both aspect ratios
RESOLUTION_BUCKETS = "544x544x145;448x672x145"


def build_manifest() -> list[dict]:
    """Scan scrya dir and build caption+video manifest, skipping missing pairs."""
    entries = []
    mp4_files = sorted(SCRYA_DIR.glob("*.mp4"))

    for mp4 in mp4_files:
        txt = mp4.with_suffix(".txt")
        if not txt.exists():
            print(f"  skip {mp4.name} — no caption")
            continue
        caption = txt.read_text().strip()
        if not caption:
            print(f"  skip {mp4.name} — empty caption")
            continue
        entries.append({"video_path": str(mp4), "caption": caption})

    print(f"Found {len(entries)} video+caption pairs")
    return entries


def run_latents_phase(manifest_path: Path) -> None:
    """Phase 1: encode video + audio latents."""
    from process_videos import compute_latents, parse_resolution_buckets

    resolution_buckets = parse_resolution_buckets(RESOLUTION_BUCKETS)
    latents_dir = OUTPUT_DIR / "latents"
    audio_dir = OUTPUT_DIR / "audio_latents"

    compute_latents(
        dataset_file=str(manifest_path),
        video_column="video_path",
        resolution_buckets=resolution_buckets,
        output_dir=str(latents_dir),
        model_path=MODEL_PATH,
        batch_size=1,
        device="cuda:1",
        reshape_mode="center",
        with_audio=True,
        audio_output_dir=str(audio_dir),
    )
    print(f"Latents → {latents_dir}")
    print(f"Audio latents → {audio_dir}")


def run_text_phase(manifest_path: Path) -> None:
    """Phase 2: encode text embeddings with Gemma + connectors."""
    from process_captions import compute_captions_embeddings

    conditions_dir = OUTPUT_DIR / "conditions_final"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding text embeddings on cuda:1...")
    compute_captions_embeddings(
        dataset_file=str(manifest_path),
        output_dir=str(conditions_dir),
        model_path=MODEL_PATH,
        text_encoder_path=GEMMA_PATH,
        caption_column="caption",
        media_column="video_path",
        lora_trigger=None,
        remove_llm_prefixes=False,
        batch_size=4,
        device="cuda:1",
        load_in_8bit=True,
    )
    print(f"Conditions → {conditions_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["latents", "text", "all"], default="all")
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build manifest once
    if not MANIFEST_PATH.exists() or args.rebuild_manifest:
        entries = build_manifest()
        with open(MANIFEST_PATH, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Manifest → {MANIFEST_PATH}  ({len(entries)} entries)")
    else:
        with open(MANIFEST_PATH) as f:
            entries = json.load(f)
        print(f"Using existing manifest: {len(entries)} entries")

    if args.phase in ("latents", "all"):
        run_latents_phase(MANIFEST_PATH)

    if args.phase in ("text", "all"):
        run_text_phase(MANIFEST_PATH)

    print("\nDone. Dataset ready at:", OUTPUT_DIR)
    print("Next: kill current training run and launch scrya overfit config")


if __name__ == "__main__":
    main()
