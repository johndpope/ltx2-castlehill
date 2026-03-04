#!/usr/bin/env python3
"""Caption first frames of videos using Qwen2.5-VL, then combine with action prompts.

For each MP4+TXT pair in a directory:
1. Extract first frame from the MP4
2. Use Qwen2.5-VL to describe the scene (what's in the image)
3. Combine: "Isometric 3D view. {scene_description}. {action_from_txt}. No camera movement."
4. Save as {uuid}_combined.txt alongside the original

This creates training-format prompts that match the isometric_identity dataset captions.

Usage:
    python scripts/caption_first_frames.py \
        --input-dir "/home/johndpope/scrya-downloads/Isometric 3D" \
        --device cuda:1 \
        --load-in-8bit
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import read_video


def extract_first_frame(video_path: Path) -> Image.Image:
    """Extract first frame from video as PIL Image."""
    video, _, info = read_video(str(video_path), pts_unit="sec", end_pts=0.1)
    # video: [T, H, W, C] uint8
    frame = video[0].numpy()  # [H, W, C]
    return Image.fromarray(frame)


def load_qwen_vl(model_path: str, device: str, load_in_8bit: bool = False):
    """Load Qwen2.5-VL model and processor."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading Qwen2.5-VL from {model_path} on {device}...")

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
    }
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    return model, processor


def caption_image(
    model,
    processor,
    image: Image.Image,
    device: str,
) -> str:
    """Generate a scene description for an image using Qwen2.5-VL.

    The prompt asks for a brief factual description WITHOUT mentioning
    'isometric' or style — we'll add those ourselves.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe what is shown in this image in one sentence. "
                        "Focus on the subject, setting, and notable details. "
                        "Be concise and factual. Do not mention the art style, "
                        "camera angle, or rendering technique."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    # Move inputs to model device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    # Decode only the generated tokens (skip input)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(generated, skip_special_tokens=True).strip()

    # Clean up common prefixes Qwen adds
    for prefix in ["The image shows ", "This image shows ", "In this image, "]:
        if caption.startswith(prefix):
            caption = caption[len(prefix):]
            # Capitalize first letter
            caption = caption[0].upper() + caption[1:] if caption else caption
            break

    return caption


def format_combined_prompt(scene_description: str, action_prompt: str) -> str:
    """Combine scene + action into training-format prompt.

    Training format: "Base. Scene description, action. Suffix."
    """
    scene = scene_description.rstrip(".")
    action = action_prompt.strip().rstrip(".-")

    # If the action prompt IS already a full scene description (starts with "A 3D"),
    # just use the action as-is with camera prefix
    if action.lower().startswith(("a 3d", "a photo", "a pixel", "an isometric")):
        return f"Static camera, fixed isometric viewpoint. {action}. No camera movement."

    return f"Isometric 3D view, static camera. {scene}, {action.lower()}. No camera movement."


def parse_args():
    p = argparse.ArgumentParser(description="Caption first frames with Qwen2.5-VL")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory with MP4+TXT pairs")
    p.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="Load model in INT8 (~14GB VRAM)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing _combined.txt files")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without saving")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Save all prompts to a JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir

    # Find all MP4+TXT pairs
    mp4_files = sorted(input_dir.glob("*.mp4"))
    print(f"Found {len(mp4_files)} MP4 files in {input_dir}")

    pairs = []
    for mp4 in mp4_files:
        txt = mp4.with_suffix(".txt")
        if txt.exists():
            pairs.append((mp4, txt))
    print(f"Found {len(pairs)} MP4+TXT pairs")

    if not pairs:
        print("No pairs found, exiting.")
        return

    # Check how many already have combined prompts
    existing = sum(1 for mp4, _ in pairs if (mp4.parent / f"{mp4.stem}_combined.txt").exists())
    if existing and not args.overwrite:
        print(f"  {existing} already have _combined.txt (use --overwrite to redo)")
        pairs = [(mp4, txt) for mp4, txt in pairs
                 if not (mp4.parent / f"{mp4.stem}_combined.txt").exists()]
        print(f"  Processing {len(pairs)} remaining")

    if not pairs:
        print("All done!")
        return

    # Load model
    model, processor = load_qwen_vl(args.model_path, args.device, args.load_in_8bit)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    results = {}
    start = time.time()

    for idx, (mp4, txt) in enumerate(pairs):
        uuid = mp4.stem
        action = txt.read_text().strip()

        # Extract first frame
        try:
            frame = extract_first_frame(mp4)
        except Exception as e:
            print(f"  [{idx+1}/{len(pairs)}] SKIP {uuid} — frame extraction failed: {e}")
            continue

        # Caption the frame
        scene = caption_image(model, processor, frame, args.device)

        # Combine prompts
        combined = format_combined_prompt(scene, action)

        results[uuid] = {
            "action_prompt": action,
            "scene_caption": scene,
            "combined_prompt": combined,
        }

        if args.dry_run:
            print(f"\n  [{idx+1}/{len(pairs)}] {uuid}")
            print(f"    Action:   {action[:60]}")
            print(f"    Scene:    {scene[:60]}")
            print(f"    Combined: {combined[:80]}")
        else:
            # Save combined prompt
            out_path = mp4.parent / f"{uuid}_combined.txt"
            out_path.write_text(combined)
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start
                rate = (idx + 1) / elapsed
                eta = (len(pairs) - idx - 1) / rate
                print(f"  [{idx+1}/{len(pairs)}] {uuid} — {rate:.1f} it/s, ETA {eta:.0f}s")
                print(f"    → {combined[:70]}")

    # Save JSON summary
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} prompts to {args.output_json}")

    elapsed = time.time() - start
    print(f"\nDone! {len(results)} videos captioned in {elapsed:.0f}s ({elapsed/len(results):.1f}s each)")

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
