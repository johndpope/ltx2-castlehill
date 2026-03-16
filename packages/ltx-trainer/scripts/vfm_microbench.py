#!/usr/bin/env python3
# ruff: noqa: T201
"""VFM v1f Microbenchmark — Generate mp4 videos from diverse prompts.

Picks N diverse samples from the dataset, runs 1-step VFM inference
with spherical Cauchy sampling, decodes to video, and saves mp4s.

Usage:
    uv run python scripts/vfm_microbench.py \
        --adapter-path /media/2TB/omnitransfer/output/vfm_v1f_spherical/checkpoints/noise_adapter_step_02500.safetensors \
        --lora-path /media/2TB/omnitransfer/output/vfm_v1f_spherical/checkpoints/lora_weights_step_02500.safetensors \
        --num-samples 10 \
        --output-dir /media/2TB/omnitransfer/inference/v1f_microbench
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VFM v1f Microbenchmark")
    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors")
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--sigma-head-path", type=str, default=None,
                        help="Path to SigmaHead checkpoint. If provided, uses per-token sigma "
                        "for timestep conditioning (matches training). Auto-detected from adapter path.")

    parser.add_argument("--data-root", type=str,
                        default="/media/12TB/ddit_ditto_data")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of diverse samples to generate")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=None,
                        help="Specific sample indices (overrides --num-samples)")

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=25,
                        help="Pixel frames (must satisfy F%%8==1)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds for each generation (multiple = multiple videos per prompt)")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae-device", type=str, default="cuda:1")
    parser.add_argument("--quantize", type=str, default="int8-quanto")

    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--adapter-num-heads", type=int, default=8)
    parser.add_argument("--adapter-num-layers", type=int, default=4)
    parser.add_argument("--adapter-pos-dim", type=int, default=256)

    parser.add_argument("--spherical", action="store_true", default=True,
                        help="Use Spherical Cauchy sampling (v1f)")
    parser.add_argument("--no-spherical", dest="spherical", action="store_false")

    parser.add_argument("--output-dir", type=str,
                        default="/media/2TB/omnitransfer/inference/v1f_microbench")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output mp4 already exists")
    return parser.parse_args()


def pick_diverse_samples(data_root: str, num_samples: int) -> list[dict]:
    """Pick evenly-spaced samples from metadata for diversity."""
    meta_path = os.path.join(data_root, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    videos = meta["videos"]
    total = len(videos)
    step = max(1, total // num_samples)
    indices = [i * step for i in range(num_samples)]
    indices = [i for i in indices if i < total]

    samples = []
    for idx in indices:
        v = videos[idx]
        samples.append({
            "id": v["id"],
            "caption": v.get("caption", f"sample_{v['id']}"),
            "embedding_path": os.path.join(data_root, "conditions_final", f"{v['id']:06d}.pt"),
        })
    return samples


def load_models(args):
    """Load transformer + LoRA + adapter."""
    device = torch.device(args.device)
    dtype = torch.bfloat16

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
                mod = f"{p}.0" if (i + 1 < len(parts) and parts[i + 1] == "0") else p
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
    # Infer text_embed_dim from adapter checkpoint (more reliable than data)
    adapter_state = load_file(args.adapter_path)
    text_embed_dim = adapter_state["text_proj.weight"].shape[1]
    print(f"  Adapter text_embed_dim: {text_embed_dim}")

    from ltx_core.model.transformer.noise_adapter_v1b import TASK_CLASSES, create_noise_adapter_v1b
    noise_adapter = create_noise_adapter_v1b(
        text_dim=text_embed_dim, latent_dim=latent_channels,
        hidden_dim=args.adapter_hidden_dim, num_heads=args.adapter_num_heads,
        num_layers=args.adapter_num_layers, pos_dim=args.adapter_pos_dim,
    )
    noise_adapter.load_state_dict(adapter_state)
    noise_adapter = noise_adapter.to(device=device)
    noise_adapter.eval()

    task_idx = TASK_CLASSES.get("i2v", 0)
    task_class = torch.tensor([task_idx], device=device)

    # Load caption projection shim if needed (LTX-2.3: cached embeds are 3840-dim,
    # but adapter expects 4096-dim after caption projection)
    caption_proj = None
    if text_embed_dim != 3840:
        # Check if cached embeddings need projection
        sample_emb = torch.load(
            os.path.join(args.data_root, "conditions_final", "000000.pt"),
            map_location="cpu", weights_only=True,
        )
        cached_dim = sample_emb["video_prompt_embeds"].shape[-1]
        if cached_dim != text_embed_dim:
            print(f"  Loading caption projection shim ({cached_dim} -> {text_embed_dim})...")
            from safetensors import safe_open
            from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
            caption_proj = PixArtAlphaTextProjection(in_features=cached_dim, hidden_size=text_embed_dim)
            # Load from LTX-2 checkpoint
            ltx2_path = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors"
            source_path = ltx2_path if os.path.exists(ltx2_path) else args.model_path
            with safe_open(source_path, framework="pt") as f:
                prefix = "model.diffusion_model.caption_projection."
                for key in f.keys():
                    if key.startswith(prefix):
                        param_name = key[len(prefix):]
                        param = f.get_tensor(key)
                        parts = param_name.split(".")
                        obj = caption_proj
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], torch.nn.Parameter(param))
            caption_proj = caption_proj.to(device).eval()
            for p in caption_proj.parameters():
                p.requires_grad = False
            print(f"  Caption projection shim loaded")

    # SigmaHead (per-token timestep conditioning — critical for matching training)
    sigma_head = None
    sigma_head_path = args.sigma_head_path
    if sigma_head_path is None:
        # Auto-detect from adapter path
        adapter_dir = Path(args.adapter_path).parent
        adapter_step = Path(args.adapter_path).stem.split("_")[-1]
        auto_path = adapter_dir / f"sigma_head_step_{adapter_step}.safetensors"
        if auto_path.exists():
            sigma_head_path = str(auto_path)

    if sigma_head_path and os.path.exists(sigma_head_path):
        print(f"  Loading SigmaHead from {Path(sigma_head_path).name}...")
        from ltx_trainer.training_strategies.vfm_strategy_v1d import SigmaHead
        sigma_head_state = load_file(sigma_head_path)
        # Infer dimensions from checkpoint
        hidden_dim = sigma_head_state["net.0.weight"].shape[0]
        input_dim = sigma_head_state["net.0.weight"].shape[1]
        # SigmaHead takes (adapter_mu, x0) → input_dim = latent_dim + latent_dim = 256
        # or just latent_dim if x0 not used
        latent_dim = latent_channels  # 128
        sigma_head = SigmaHead(latent_dim=latent_dim, hidden_dim=hidden_dim)
        sigma_head.load_state_dict(sigma_head_state)
        sigma_head = sigma_head.to(device=device).eval()
        print(f"  SigmaHead loaded (hidden={hidden_dim})")
    else:
        print("  No SigmaHead found — using timesteps=1.0 (may not match training!)")

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  Model load: {t_model:.1f}s | GPU: {mem_gb:.1f} GB")
    print(f"  LoRA rank: {lora_rank} | Adapter: {sum(p.numel() for p in noise_adapter.parameters()) / 1e6:.1f}M params")

    return transformer, noise_adapter, task_class, caption_proj, sigma_head


def generate_latent(
    args, transformer, noise_adapter, task_class,
    prompt_embeds, prompt_mask, seed,
    sigma_head=None,
):
    """Generate video latent from a single prompt embedding."""
    device = torch.device(args.device)
    dtype = torch.bfloat16
    latent_channels = 128

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_frames = (args.num_frames - 1) // 8 + 1
    tokens_per_frame = latent_h * latent_w
    total_tokens = tokens_per_frame * latent_frames

    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    generator = torch.Generator(device=device).manual_seed(seed)

    pos_shape = VideoLatentShape(
        frames=latent_frames, height=latent_h, width=latent_w,
        batch=1, channels=latent_channels,
    )
    coords = patchifier.get_patch_grid_bounds(output_shape=pos_shape, device=device)
    positions = get_pixel_coords(
        latent_coords=coords, scale_factors=scale_factors, causal_fix=True,
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / args.fps

    timings = {}

    with torch.inference_mode():
        # Adapter forward
        t0 = time.time()
        mu, log_sigma = noise_adapter.forward(
            text_embeddings=prompt_embeds.float(),
            text_mask=prompt_mask.bool(),
            positions=positions.float(),
            task_class=task_class,
        )

        if args.spherical:
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

        # Transformer forward (flow map)
        t0 = time.time()
        if args.num_steps == 1:
            # Per-token sigma from SigmaHead (matches training conditioning)
            if sigma_head is not None:
                # SigmaHead expects (mu, x0). At inference x0 is unknown,
                # so we pass None (falls back to zero-padding internally)
                per_token_sigmas = sigma_head(mu.float(), x0=None)  # [1, total_tokens]
                timesteps = per_token_sigmas.to(dtype)
                sigma_val = per_token_sigmas.mean(dim=1).to(dtype)  # batch sigma
                timings["sigma_mean"] = per_token_sigmas.mean().item()
                timings["sigma_std"] = per_token_sigmas.std().item()
            else:
                timesteps = torch.ones(1, total_tokens, device=device, dtype=dtype)
                sigma_val = torch.ones(1, device=device, dtype=dtype)

            video_mod = Modality(
                enabled=True, sigma=sigma_val, latent=z, timesteps=timesteps,
                positions=positions, context=prompt_embeds, context_mask=prompt_mask,
            )
            v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
            x = z - v_pred
        else:
            time_steps = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device)
            x = z
            for k in range(args.num_steps):
                t_k = time_steps[k]
                t_km1 = time_steps[k + 1]
                dt = t_km1 - t_k
                timesteps = torch.full((1, total_tokens), t_k.item(), device=device, dtype=dtype)
                sigma_val = torch.full((1,), t_k.item(), device=device, dtype=dtype)
                video_mod = Modality(
                    enabled=True, sigma=sigma_val, latent=x, timesteps=timesteps,
                    positions=positions, context=prompt_embeds, context_mask=prompt_mask,
                )
                v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                x = x + dt * (-v_pred)

        torch.cuda.synchronize(device)
        timings["transformer_ms"] = (time.time() - t0) * 1000

        # Unpatchify
        x_latent = patchifier.unpatchify(
            x, output_shape=VideoLatentShape(
                frames=latent_frames, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
        )

    # Diagnostic info
    timings["kappa_mean"] = kappa.mean().item() if args.spherical else None
    timings["kappa_std"] = kappa.std().item() if args.spherical else None
    timings["mu_norm_mean"] = mu_norm.mean().item() if args.spherical else None
    timings["z_norm"] = z.float().norm(dim=-1).mean().item()

    return x_latent.cpu(), timings


def decode_and_save(x_latent, output_path, args, vae_decoder):
    """VAE decode + save as mp4."""
    from ltx_core.model.video_vae.tiling import TilingConfig

    device = torch.device(args.vae_device)
    tiling_config = TilingConfig.default()

    x_latent = x_latent.to(device)
    t0 = time.time()
    with torch.inference_mode():
        chunks = []
        for chunk in vae_decoder.tiled_decode(x_latent, tiling_config):
            chunks.append(chunk.cpu())
        video_pixels = torch.cat(chunks, dim=2)
    torch.cuda.synchronize(device)
    vae_ms = (time.time() - t0) * 1000

    import torchvision
    video_pixels = (video_pixels.float() * 0.5 + 0.5).clamp(0, 1)
    frames = video_pixels[0].permute(1, 2, 3, 0)  # [F, H, W, C]
    frames = (frames * 255).to(torch.uint8)
    torchvision.io.write_video(str(output_path), frames, fps=args.fps,
                               video_codec="libx264", options={"crf": "18"})

    del video_pixels, frames, chunks, x_latent
    torch.cuda.empty_cache()
    return vae_ms


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick samples
    if args.sample_indices is not None:
        meta_path = os.path.join(args.data_root, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        samples = []
        for idx in args.sample_indices:
            v = meta["videos"][idx]
            samples.append({
                "id": v["id"],
                "caption": v.get("caption", f"sample_{v['id']}"),
                "embedding_path": os.path.join(args.data_root, "conditions_final", f"{v['id']:06d}.pt"),
            })
    else:
        samples = pick_diverse_samples(args.data_root, args.num_samples)

    print()
    print("=" * 90)
    print("  VFM v1f Microbenchmark")
    print("=" * 90)
    print(f"  Samples:     {len(samples)} x {len(args.seeds)} seeds = {len(samples) * len(args.seeds)} videos")
    print(f"  Resolution:  {args.width}x{args.height}, {args.num_frames} frames")
    print(f"  Steps:       {args.num_steps}")
    print(f"  Spherical:   {args.spherical}")
    print(f"  Checkpoint:  {Path(args.adapter_path).stem}")
    print(f"  Output:      {out_dir}")
    print("=" * 90)

    # Print sample captions
    print("\nSamples:")
    for i, s in enumerate(samples):
        caption_short = s["caption"][:100] + "..." if len(s["caption"]) > 100 else s["caption"]
        print(f"  [{s['id']:05d}] {caption_short}")

    # Load models
    print()
    transformer, noise_adapter, task_class, caption_proj, sigma_head = load_models(args)

    # Phase 1: Generate all latents (transformer on GPU)
    print(f"\n{'─' * 90}")
    print("Phase 1: Generate latents (adapter + transformer)")
    print(f"{'─' * 90}")

    latent_results = []
    for s in samples:
        emb = torch.load(s["embedding_path"], map_location="cpu", weights_only=True)
        video_features = emb["video_prompt_embeds"].unsqueeze(0).to(torch.bfloat16).to(args.device)
        audio_features = emb.get("audio_prompt_embeds")
        if audio_features is not None:
            audio_features = audio_features.unsqueeze(0).to(torch.bfloat16).to(args.device)
        prompt_mask = emb["prompt_attention_mask"].unsqueeze(0).to(args.device)

        # Apply caption projection shim if needed (3840 -> 4096)
        if caption_proj is not None:
            with torch.inference_mode():
                video_features = caption_proj(video_features)
                if audio_features is not None:
                    audio_features = caption_proj(audio_features)

        # Note: cached embeddings are already post-connector (from preprocessing).
        # Only the caption projection shim is needed (applied above).
        prompt_embeds = video_features

        for seed in args.seeds:
            fname = f"v1f_{s['id']:05d}_seed{seed}_{args.num_steps}step.mp4"
            out_path = out_dir / fname

            if args.skip_existing and out_path.exists():
                print(f"  [{s['id']:05d}] seed={seed}: SKIP (exists)")
                continue

            torch.manual_seed(seed)
            x_latent, timings = generate_latent(
                args, transformer, noise_adapter, task_class,
                prompt_embeds, prompt_mask, seed,
                sigma_head=sigma_head,
            )

            diag = ""
            if timings.get("kappa_mean") is not None:
                diag = f" | kappa={timings['kappa_mean']:.2f}+/-{timings['kappa_std']:.2f} | mu_norm={timings['mu_norm_mean']:.2f} | z_norm={timings['z_norm']:.2f}"

            print(f"  [{s['id']:05d}] seed={seed}: adapter={timings['adapter_ms']:.0f}ms "
                  f"transformer={timings['transformer_ms']:.0f}ms{diag}")

            latent_results.append({
                "sample": s,
                "seed": seed,
                "x_latent": x_latent,
                "out_path": out_path,
                "timings": timings,
            })

        del prompt_embeds, prompt_mask
        torch.cuda.empty_cache()

    if not latent_results:
        print("\nNo new videos to generate. Done.")
        return

    # Phase 2: Free transformer, decode with VAE
    print(f"\n{'─' * 90}")
    print("Phase 2: VAE decode + save mp4 (freeing transformer VRAM)")
    print(f"{'─' * 90}")

    del transformer, noise_adapter, caption_proj
    gc.collect()
    torch.cuda.empty_cache()

    from ltx_trainer.model_loader import load_video_vae_decoder
    vae_decoder = load_video_vae_decoder(args.model_path, device=args.vae_device, dtype=torch.bfloat16)
    vae_decoder.eval()

    for r in latent_results:
        s = r["sample"]
        try:
            vae_ms = decode_and_save(r["x_latent"], r["out_path"], args, vae_decoder)
            total_ms = r["timings"]["adapter_ms"] + r["timings"]["transformer_ms"] + vae_ms
            print(f"  [{s['id']:05d}] seed={r['seed']}: VAE={vae_ms:.0f}ms | total={total_ms:.0f}ms | {r['out_path'].name}")
        except torch.cuda.OutOfMemoryError:
            print(f"  [{s['id']:05d}] seed={r['seed']}: VAE OOM")
        del r["x_latent"]
        torch.cuda.empty_cache()

    # Phase 3: Write manifest
    manifest = []
    for r in latent_results:
        s = r["sample"]
        entry = {
            "id": s["id"],
            "caption": s["caption"],
            "seed": r["seed"],
            "output": str(r["out_path"]),
            "adapter_ms": r["timings"]["adapter_ms"],
            "transformer_ms": r["timings"]["transformer_ms"],
        }
        if r["timings"].get("kappa_mean") is not None:
            entry["kappa_mean"] = r["timings"]["kappa_mean"]
            entry["kappa_std"] = r["timings"]["kappa_std"]
            entry["mu_norm_mean"] = r["timings"]["mu_norm_mean"]
        manifest.append(entry)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"  Done! {len(latent_results)} videos saved to {out_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
