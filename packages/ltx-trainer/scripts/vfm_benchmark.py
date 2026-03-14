#!/usr/bin/env python3
# ruff: noqa: T201
"""VFM Benchmark — Match LTX Desktop 2.3 configs and measure timing breakdown.

Compares VFM 1-step and 4-step generation times against LTX Desktop's 8-step times.
Reports per-component timing: model load, adapter forward, transformer forward, VAE decode.

Usage:
    # Run all configs (needs ~19GB VRAM for transformer + adapter):
    uv run python scripts/vfm_benchmark.py \
        --adapter-path /path/to/noise_adapter.safetensors \
        --lora-path /path/to/lora_weights.safetensors \
        --cached-embedding /path/to/conditions_final/000000.pt

    # Just 540p configs:
    uv run python scripts/vfm_benchmark.py --max-resolution 540p ...

    # Generate videos too (slower, includes VAE decode):
    uv run python scripts/vfm_benchmark.py --save-videos ...
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# LTX-2 Desktop configs (Yoav HaCohen, Mar 8 2026)
# All resolutions must be divisible by 32. Frames must satisfy F%8==1.
# Desktop uses 25fps, 8-step diffusion.
# ─────────────────────────────────────────────────────────────────────────────
FPS = 25.0

# Resolution name → (width, height) — nearest div-by-32 to standard
RESOLUTIONS = {
    "540p": (960, 544),    # 960/32=30, 544/32=17
    "720p": (1280, 704),   # 1280/32=40, 704/32=22
    "1080p": (1920, 1088), # 1920/32=60, 1088/32=34
}

# Duration → pixel frames (F%8==1, at 25fps)
DURATIONS = {
    "5s": 121,   # 121/25 = 4.84s
    "6s": 145,   # 145/25 = 5.80s
    "8s": 193,   # 193/25 = 7.72s
    "10s": 241,  # 241/25 = 9.64s
    "20s": 481,  # 481/25 = 19.24s
}

# LTX Desktop 8-step reference times (i2v, RTX 5090, without text encoding)
DESKTOP_TIMES = {
    ("5s", "540p"): 33, ("5s", "720p"): 42, ("5s", "1080p"): 76,
    ("6s", "540p"): 35, ("6s", "720p"): 49,
    ("8s", "540p"): 40, ("8s", "720p"): 57,
    ("10s", "540p"): 44, ("10s", "720p"): 67,
    ("20s", "540p"): 73,
}

# Which configs to benchmark (matching Yoav's tweet)
BENCHMARK_CONFIGS = [
    ("5s", "540p"), ("5s", "720p"), ("5s", "1080p"),
    ("6s", "540p"), ("6s", "720p"),
    ("8s", "540p"), ("8s", "720p"),
    ("10s", "540p"), ("10s", "720p"),
    ("20s", "540p"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VFM Benchmark — compare against LTX Desktop")
    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors")
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--cached-embedding", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae-device", type=str, default="cuda:1")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])
    parser.add_argument("--seed", type=int, default=42)

    # Adapter config (must match training)
    parser.add_argument("--adapter-variant", type=str, default="v1b",
                        choices=["mlp", "transformer", "v1b"])
    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--adapter-num-heads", type=int, default=8)
    parser.add_argument("--adapter-num-layers", type=int, default=4)
    parser.add_argument("--adapter-pos-dim", type=int, default=256)

    parser.add_argument("--max-resolution", type=str, default="1080p",
                        choices=["540p", "720p", "1080p"],
                        help="Skip configs above this resolution")
    parser.add_argument("--vfm-steps", type=int, nargs="+", default=[1, 4],
                        help="VFM step counts to benchmark")
    parser.add_argument("--save-videos", action="store_true",
                        help="Also decode and save videos (slower)")
    parser.add_argument("--output-dir", type=str,
                        default="/media/2TB/omnitransfer/inference/v1b_benchmark")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup pass before timing")
    parser.add_argument("--spherical", action="store_true",
                        help="Use Spherical Cauchy sampling (v1f) instead of Gaussian")
    return parser.parse_args()


def load_models(args):
    """Load transformer + LoRA + adapter. Returns (transformer, noise_adapter, task_class)."""
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Transformer
    print(f"Loading transformer + quantize ({args.quantize})...")
    t0 = time.time()
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)
    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        transformer = quantize_model(transformer, args.quantize, device=str(device))
    transformer = transformer.to(device)

    # LoRA
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    lora_state = load_file(str(args.lora_path))
    target_modules = set()
    lora_rank = None
    for key, value in lora_state.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p in ("to_k", "to_q", "to_v", "to_out"):
                if i + 1 < len(parts) and parts[i + 1] == "0":
                    mod = f"{p}.0"
                else:
                    mod = p
                target_modules.add(mod)
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                             target_modules=sorted(target_modules), lora_dropout=0.0)
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state)
    transformer = transformer.merge_and_unload()
    transformer.eval()
    t_model = time.time() - t0

    # Adapter
    latent_channels = 128
    emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
    text_embed_dim = emb["video_prompt_embeds"].shape[-1]

    if args.adapter_variant == "v1b":
        from ltx_core.model.transformer.noise_adapter_v1b import TASK_CLASSES, create_noise_adapter_v1b
        noise_adapter = create_noise_adapter_v1b(
            text_dim=text_embed_dim, latent_dim=latent_channels,
            hidden_dim=args.adapter_hidden_dim, num_heads=args.adapter_num_heads,
            num_layers=args.adapter_num_layers, pos_dim=args.adapter_pos_dim,
        )
    else:
        from ltx_core.model.transformer.noise_adapter import TASK_CLASSES, create_noise_adapter
        noise_adapter = create_noise_adapter(
            input_dim=text_embed_dim, latent_dim=latent_channels,
            variant=args.adapter_variant, hidden_dim=args.adapter_hidden_dim,
            num_layers=args.adapter_num_layers,
        )

    adapter_state = load_file(args.adapter_path)
    noise_adapter.load_state_dict(adapter_state)
    noise_adapter = noise_adapter.to(device=device)
    noise_adapter.eval()

    from ltx_core.model.transformer.noise_adapter_v1b import TASK_CLASSES
    task_idx = TASK_CLASSES.get("i2v", 0)
    task_class = torch.tensor([task_idx], device=device)

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  Model load: {t_model:.1f}s | GPU: {mem_gb:.1f} GB")
    print(f"  Adapter: {sum(p.numel() for p in noise_adapter.parameters()) / 1e6:.1f}M params")

    return transformer, noise_adapter, task_class, text_embed_dim


def run_single_benchmark(
    args, transformer, noise_adapter, task_class,
    prompt_embeds, prompt_mask, text_embed_dim,
    width, height, num_frames, num_steps,
    vae_decoder=None,
):
    """Run a single benchmark config. Returns dict of timings."""
    device = torch.device(args.device)
    dtype = torch.bfloat16
    latent_channels = 128

    latent_h = height // 32
    latent_w = width // 32
    latent_frames = (num_frames - 1) // 8 + 1
    tokens_per_frame = latent_h * latent_w
    total_tokens = tokens_per_frame * latent_frames

    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Positions
    pos_shape = VideoLatentShape(
        frames=latent_frames, height=latent_h, width=latent_w,
        batch=1, channels=latent_channels,
    )
    coords = patchifier.get_patch_grid_bounds(output_shape=pos_shape, device=device)
    positions = get_pixel_coords(
        latent_coords=coords, scale_factors=scale_factors, causal_fix=True,
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / FPS

    timings = {"tokens": total_tokens, "latent_frames": latent_frames}

    torch.cuda.synchronize(device)

    with torch.inference_mode():
        # ── Adapter forward ──
        t0 = time.time()
        if args.adapter_variant == "v1b":
            mu, log_sigma = noise_adapter.forward(
                text_embeddings=prompt_embeds.float(),
                text_mask=prompt_mask.bool(),
                positions=positions.float(),
                task_class=task_class,
            )
        else:
            mask = prompt_mask.unsqueeze(-1).float()
            pooled_text = (prompt_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            text_obs = pooled_text.unsqueeze(1).expand(-1, total_tokens, -1)
            mu, log_sigma = noise_adapter.forward(text_obs.float(), task_class)

        if args.spherical:
            # Spherical Cauchy sampling (v1f)
            from ltx_trainer.spherical_utils import normalize, sample_spherical_cauchy
            B, seq, D = mu.shape
            mu_hat = normalize(mu.float(), dim=-1)
            mu_norm = mu.float().norm(p=2, dim=-1)
            kappa = torch.exp(log_sigma.float().mean(dim=-1)).clamp(min=0.1, max=50.0)
            mu_hat_flat = mu_hat.reshape(B * seq, D)
            kappa_flat = kappa.reshape(B * seq)
            z_dir = sample_spherical_cauchy(mu_hat_flat, kappa_flat).reshape(B, seq, D)
            z = (mu_norm.unsqueeze(-1) * z_dir).to(dtype)
        else:
            sigma = torch.exp(log_sigma)
            eps = torch.randn(mu.shape, device=device, dtype=torch.float32, generator=generator)
            z = (mu + sigma * eps).to(dtype)
        torch.cuda.synchronize(device)
        timings["adapter_ms"] = (time.time() - t0) * 1000

        # ── Transformer forward (flow map) ──
        t0 = time.time()
        if num_steps == 1:
            timesteps = torch.ones(1, total_tokens, device=device, dtype=dtype)
            video_mod = Modality(
                enabled=True, latent=z, timesteps=timesteps,
                positions=positions, context=prompt_embeds, context_mask=prompt_mask,
            )
            v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
            x = z - v_pred
        else:
            time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            x = z
            for k in range(num_steps):
                t_k = time_steps[k]
                t_km1 = time_steps[k + 1]
                dt = t_km1 - t_k
                timesteps = torch.full((1, total_tokens), t_k.item(), device=device, dtype=dtype)
                video_mod = Modality(
                    enabled=True, latent=x, timesteps=timesteps,
                    positions=positions, context=prompt_embeds, context_mask=prompt_mask,
                )
                v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                x = x + dt * (-v_pred)

        torch.cuda.synchronize(device)
        timings["transformer_ms"] = (time.time() - t0) * 1000

        # ── Unpatchify (always, needed for VAE) ──
        x_latent = patchifier.unpatchify(
            x, output_shape=VideoLatentShape(
                frames=latent_frames, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
        )
        timings["x_latent"] = x_latent.cpu()  # stash on CPU for later VAE decode
        timings["vae_decode_ms"] = None

    timings["total_ms"] = timings["adapter_ms"] + timings["transformer_ms"]
    if timings["vae_decode_ms"] is not None:
        timings["total_ms"] += timings["vae_decode_ms"]

    return timings


def main():
    args = parse_args()

    # Filter configs by max resolution
    res_order = ["540p", "720p", "1080p"]
    max_idx = res_order.index(args.max_resolution)
    allowed_res = set(res_order[: max_idx + 1])
    configs = [(d, r) for d, r in BENCHMARK_CONFIGS if r in allowed_res]

    print()
    print("=" * 80)
    print("  VFM Benchmark — vs LTX Desktop 2.3 (RTX 5090)")
    print("=" * 80)
    print(f"  Configs:    {len(configs)} × {len(args.vfm_steps)} step variants = {len(configs) * len(args.vfm_steps)} runs")
    print(f"  VFM Steps:  {args.vfm_steps}")
    print(f"  Quantize:   {args.quantize}")
    print(f"  Adapter:    {args.adapter_variant}")
    print(f"  Save video: {args.save_videos}")
    print("=" * 80)

    # Load embeddings
    print(f"\nLoading cached embedding...")
    emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
    prompt_embeds = emb["video_prompt_embeds"].unsqueeze(0).to(torch.bfloat16).to(args.device)
    prompt_mask = emb["prompt_attention_mask"].unsqueeze(0).to(args.device)
    text_embed_dim = prompt_embeds.shape[-1]

    # Load models
    transformer, noise_adapter, task_class, _ = load_models(args)

    # Load VAE if saving videos (load on CPU, move to GPU later after transformer offload)
    vae_decoder = None
    if args.save_videos:
        print(f"Loading VAE decoder (will decode after transformer offload)...")
        from ltx_trainer.model_loader import load_video_vae_decoder
        vae_decoder = load_video_vae_decoder(args.model_path, device="cpu", dtype=torch.bfloat16)
        vae_decoder.eval()

    # Warmup
    if args.warmup:
        print("\nWarmup pass (540p 5s 1-step)...")
        w, h = RESOLUTIONS["540p"]
        run_single_benchmark(
            args, transformer, noise_adapter, task_class,
            prompt_embeds, prompt_mask, text_embed_dim,
            width=w, height=h, num_frames=DURATIONS["5s"], num_steps=1,
        )
        print("  Done.")

    # Run benchmarks
    results = []
    print(f"\n{'─' * 80}")
    print(f"{'Config':<18} {'Steps':>5} {'Tokens':>8} {'Adapter':>10} {'Transformer':>13} {'VAE':>10} {'Total':>10} {'Desktop':>10} {'Speedup':>8}")
    print(f"{'─' * 80}")

    for duration, resolution in configs:
        w, h = RESOLUTIONS[resolution]
        num_frames = DURATIONS[duration]
        desktop_time = DESKTOP_TIMES.get((duration, resolution))

        for num_steps in args.vfm_steps:
            config_name = f"i2v {duration} {resolution}"

            try:
                timings = run_single_benchmark(
                    args, transformer, noise_adapter, task_class,
                    prompt_embeds, prompt_mask, text_embed_dim,
                    width=w, height=h, num_frames=num_frames, num_steps=num_steps,
                    vae_decoder=vae_decoder,
                )

                total_s = timings["total_ms"] / 1000
                adapter_s = timings["adapter_ms"] / 1000
                transformer_s = timings["transformer_ms"] / 1000
                vae_s = timings["vae_decode_ms"] / 1000 if timings["vae_decode_ms"] else 0

                speedup = f"{desktop_time / total_s:.1f}x" if desktop_time else "—"

                vae_str = f"{vae_s:.1f}s" if vae_s > 0 else "—"
                print(f"{config_name:<18} {num_steps:>5} {timings['tokens']:>8} "
                      f"{adapter_s:>9.2f}s {transformer_s:>12.2f}s {vae_str:>10} "
                      f"{total_s:>9.1f}s {f'~{desktop_time}s' if desktop_time else '—':>10} "
                      f"{speedup:>8}")

                entry = {
                    "config": config_name,
                    "steps": num_steps,
                    "tokens": timings["tokens"],
                    "adapter_ms": timings["adapter_ms"],
                    "transformer_ms": timings["transformer_ms"],
                    "vae_decode_ms": timings["vae_decode_ms"],
                    "total_ms": timings["total_ms"],
                    "desktop_s": desktop_time,
                }
                if args.save_videos and "x_latent" in timings:
                    entry["x_latent"] = timings["x_latent"]
                results.append(entry)

                # Clear cache between runs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"{config_name:<18} {num_steps:>5} {'OOM':>8}")
                torch.cuda.empty_cache()
                gc.collect()

    print(f"{'─' * 80}")

    # ── VAE decode pass (after freeing transformer VRAM) ──
    if args.save_videos and vae_decoder is not None:
        print(f"\n{'─' * 80}")
        print("  VAE Decode Pass — deleting transformer to free VRAM...")
        del transformer
        del noise_adapter
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        free_mb = (torch.cuda.get_device_properties(args.device).total_memory
                   - torch.cuda.memory_allocated(args.device)) / 1e6
        print(f"  Free VRAM on {args.device}: {free_mb:.0f} MB")

        # Move VAE to the big GPU
        vae_decoder = vae_decoder.to(args.device)

        from ltx_core.model.video_vae.tiling import TilingConfig
        tiling_config = TilingConfig.default()

        for r in results:
            if "x_latent" not in r:
                continue
            x_latent = r["x_latent"].to(args.device)
            try:
                t0 = time.time()
                with torch.inference_mode():
                    # Use tiled decode for memory efficiency on long videos
                    chunks = []
                    for chunk in vae_decoder.tiled_decode(x_latent, tiling_config):
                        chunks.append(chunk.cpu())
                    video_pixels = torch.cat(chunks, dim=2)  # concat along temporal dim
                torch.cuda.synchronize(torch.device(args.device))
                r["vae_decode_ms"] = (time.time() - t0) * 1000
                r["total_ms"] += r["vae_decode_ms"]

                # Save video
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                cfg_parts = r["config"].replace(" ", "_")
                out_path = out_dir / f"bench_{cfg_parts}_{r['steps']}step.mp4"

                import torchvision
                video_pixels = (video_pixels.float() * 0.5 + 0.5).clamp(0, 1)
                frames = video_pixels[0].permute(1, 2, 3, 0)
                frames = (frames * 255).to(torch.uint8)
                torchvision.io.write_video(str(out_path), frames, fps=FPS,
                                           video_codec="libx264", options={"crf": "18"})
                print(f"  {r['config']:<18} {r['steps']}-step: VAE {r['vae_decode_ms']/1000:.1f}s → {out_path.name}")
                del video_pixels, frames, chunks
            except torch.cuda.OutOfMemoryError:
                print(f"  {r['config']:<18} {r['steps']}-step: VAE OOM")
                r["vae_decode_ms"] = None

            del x_latent
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 80}")
    print("  Summary: VFM Speedup over LTX Desktop 8-step")
    print(f"{'=' * 80}")
    for r in results:
        if r["desktop_s"] and r["total_ms"] > 0:
            speedup = r["desktop_s"] / (r["total_ms"] / 1000)
            pct_adapter = r["adapter_ms"] / r["total_ms"] * 100
            pct_transformer = r["transformer_ms"] / r["total_ms"] * 100
            vae_str = ""
            if r.get("vae_decode_ms"):
                pct_vae = r["vae_decode_ms"] / r["total_ms"] * 100
                vae_str = f" | VAE {pct_vae:.0f}%"
            print(f"  {r['config']:<18} {r['steps']}-step: {r['total_ms']/1000:.1f}s "
                  f"(adapter {pct_adapter:.0f}% | transformer {pct_transformer:.0f}%{vae_str}) "
                  f"→ {speedup:.1f}x faster")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
