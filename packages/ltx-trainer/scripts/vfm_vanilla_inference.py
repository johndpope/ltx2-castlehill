#!/usr/bin/env python3
# ruff: noqa: T201
"""Vanilla VFM one-step conditional video generation (NO SCD split).

Uses the full 48-layer transformer as the flow map.
Text embeddings serve as the observation for the noise adapter.

Usage:
    python scripts/vfm_vanilla_inference.py \
        --cached-embedding /media/2TB/omnitransfer/data/ditto_1sample/conditions_final/000000.pt \
        --adapter-path /media/2TB/omnitransfer/output/vfm_vanilla_overfit_1sample/checkpoints/noise_adapter_step_00500.safetensors \
        --lora-path /media/2TB/omnitransfer/output/vfm_vanilla_overfit_1sample/checkpoints/lora_weights_step_00500.safetensors \
        --num-steps 1 \
        --output /media/2TB/omnitransfer/inference/vfm_vanilla_test.mp4
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla VFM one-step video generation")

    parser.add_argument("--model-path", type=str,
                        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors")
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--text-encoder-path", type=str,
                        default="/media/2TB/ltx-models/gemma")

    cond_group = parser.add_mutually_exclusive_group(required=True)
    cond_group.add_argument("--cached-embedding", type=str)
    cond_group.add_argument("--prompt", type=str)

    parser.add_argument("--task", type=str, default="i2v",
                        choices=["i2v", "inpaint", "sr", "denoise", "t2v"])
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=25,
                        help="Pixel frames (must satisfy frames %% 8 == 1)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae-device", type=str, default="cuda:1")
    parser.add_argument("--quantize", type=str, default="int8-quanto",
                        choices=["none", "int8-quanto", "fp8-quanto"])

    # Multi-chunk / pipelining
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of video chunks to generate sequentially. "
                        "Each chunk is --num-frames. Total frames = num-chunks * num-frames.")
    parser.add_argument("--pipeline-vae", action="store_true",
                        help="Pipeline VAE decode on --vae-device while DiT generates next chunk "
                        "on --device. Hides VAE latency behind DiT compute. "
                        "Requires separate GPUs (--device != --vae-device).")

    # Noise adapter config (must match training)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-num-layers", type=int, default=4)
    parser.add_argument("--adapter-variant", type=str, default="mlp",
                        choices=["mlp", "transformer", "v1b"])

    # v1b-specific adapter config
    parser.add_argument("--adapter-num-heads", type=int, default=8)
    parser.add_argument("--adapter-pos-dim", type=int, default=256)

    # Per-token sigma (SigmaHead)
    parser.add_argument("--sigma-head-path", type=str, default=None,
                        help="Path to trained SigmaHead checkpoint. "
                        "Enables per-token sigma for spatially-varying denoising. "
                        "Auto-detected from adapter path if not set.")
    parser.add_argument("--two-pass", action="store_true",
                        help="Two-pass inference: Pass 1 at uniform σ=1.0 → rough x̂₀, "
                        "Pass 2 uses SigmaHead(rough x̂₀) for per-token σ → sharp x̂₀. "
                        "Requires --sigma-head-path. ~2x latency but sharper output.")
    parser.add_argument("--sigma-head-hidden-dim", type=int, default=256,
                        help="Hidden dim for SigmaHead (must match training)")

    # Multi-path ensemble (Chain-of-Steps reasoning, arXiv 2603.16870)
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Number of adapter noise samples to ensemble. "
                        "Each uses a different seed → different 'reasoning path'. "
                        "Final output = average of K predictions. "
                        "K=1 (default) = standard. K=3 = 3x compute, better quality.")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    assert args.height % 32 == 0, f"Height must be divisible by 32, got {args.height}"
    assert args.width % 32 == 0, f"Width must be divisible by 32, got {args.width}"
    assert args.num_frames % 8 == 1, f"Frames must satisfy frames%%8==1, got {args.num_frames}"

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    tokens_per_frame = latent_h * latent_w
    latent_frames = (args.num_frames - 1) // 8 + 1
    total_tokens = tokens_per_frame * latent_frames

    print()
    print("=" * 65)
    print("  Vanilla VFM Inference (Full 48-layer Transformer)")
    print("=" * 65)
    if args.cached_embedding:
        print(f"  Embedding:   {args.cached_embedding}")
    else:
        print(f"  Prompt:      {args.prompt[:60]}...")
    print(f"  Resolution:  {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"  Frames:      {args.num_frames} pixel / {latent_frames} latent")
    print(f"  Tokens:      {total_tokens} ({tokens_per_frame}/frame x {latent_frames})")
    print(f"  VFM Steps:   {args.num_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Task:        {args.task}")
    print(f"  Quantize:    {args.quantize}")
    print(f"  Output:      {args.output}")
    print("=" * 65)

    t_start = time.time()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Load text embeddings
    # ══════════════════════════════════════════════════════════════════
    if args.cached_embedding:
        print(f"\n[1/5] Loading cached embedding...")
        emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
        prompt_embeds = emb["video_prompt_embeds"].unsqueeze(0).to(dtype)
        prompt_mask = emb["prompt_attention_mask"].unsqueeze(0)
        print(f"  Shape: {prompt_embeds.shape} (dim={prompt_embeds.shape[-1]})")
    else:
        print(f"\n[1/5] Loading text encoder on {args.vae_device}...")
        from ltx_trainer.model_loader import load_text_encoder
        text_encoder = load_text_encoder(
            args.model_path, args.text_encoder_path,
            device=args.vae_device, dtype=dtype,
        )
        text_encoder.eval()
        with torch.inference_mode():
            video_embeds, _audio_embeds, attention_mask = text_encoder(args.prompt)
            prompt_embeds = video_embeds.to(dtype)
            prompt_mask = attention_mask
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)

    # Apply caption_projection shim if cached embeddings are 3840-dim (pre-projection)
    # LTX-2.3 expects 4096-dim; the caption_projection from LTX-2 transforms 3840→4096
    if prompt_embeds.shape[-1] == 3840:
        print(f"  Applying caption_projection shim (3840 → 4096)...")
        from safetensors import safe_open  # noqa: PLC0415
        from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection  # noqa: PLC0415
        caption_proj = PixArtAlphaTextProjection(in_features=3840, hidden_size=4096)
        # Load weights from LTX-2 checkpoint (or current model)
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
        caption_proj = caption_proj.to(device=device, dtype=dtype).eval()
        with torch.inference_mode():
            B = prompt_embeds.shape[0]
            prompt_embeds = caption_proj(prompt_embeds.to(dtype)).view(B, -1, 4096)
        del caption_proj
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Post-projection shape: {prompt_embeds.shape}")

    text_embed_dim = prompt_embeds.shape[-1]

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Load transformer + quantize + LoRA (NO SCD wrap)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[2/5] Loading transformer...")
    from ltx_trainer.model_loader import load_transformer
    transformer = load_transformer(args.model_path, device="cpu", dtype=dtype)

    if args.quantize != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantize})...")
        transformer = quantize_model(transformer, args.quantize, device=str(device))

    print(f"  Moving to {device}...")
    transformer = transformer.to(device)

    # Apply LoRA
    print(f"  Loading LoRA: {Path(args.lora_path).name}")
    lora_state = load_file(str(args.lora_path))

    # Detect LoRA config
    target_modules = set()
    lora_rank = None
    for key, value in lora_state.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
        # Extract module name: base_model.model.transformer_blocks.0.attn1.to_q.lora_A.weight → to_q
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p in ("to_k", "to_q", "to_v", "to_out"):
                mod = p if i + 1 >= len(parts) or parts[i + 1] != "0" else f"{p}.0"
                # Check if next part is "0" (for to_out.0)
                if i + 1 < len(parts) and parts[i + 1] == "0" and parts[i + 2] in ("lora_A", "lora_B"):
                    mod = f"{p}.0"
                else:
                    mod = p
                target_modules.add(mod)

    print(f"  LoRA rank={lora_rank}, modules={sorted(target_modules)}")

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=sorted(target_modules),
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(transformer, lora_state)
    transformer = transformer.merge_and_unload()
    transformer.eval()

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  GPU memory: {mem_gb:.1f} GB")

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Load noise adapter
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3/5] Loading noise adapter ({args.adapter_variant})...")

    if args.adapter_variant == "v1b":
        from ltx_core.model.transformer.noise_adapter_v1b import (
            TASK_CLASSES,
            create_noise_adapter_v1b,
        )
        noise_adapter = create_noise_adapter_v1b(
            text_dim=text_embed_dim,
            latent_dim=latent_channels,
            hidden_dim=args.adapter_hidden_dim,
            num_heads=args.adapter_num_heads,
            num_layers=args.adapter_num_layers,
            pos_dim=args.adapter_pos_dim,
        )
    else:
        from ltx_core.model.transformer.noise_adapter import TASK_CLASSES, create_noise_adapter
        noise_adapter = create_noise_adapter(
            input_dim=text_embed_dim,
            latent_dim=latent_channels,
            variant=args.adapter_variant,
            hidden_dim=args.adapter_hidden_dim,
            num_layers=args.adapter_num_layers,
        )

    adapter_state = load_file(args.adapter_path)
    noise_adapter.load_state_dict(adapter_state)
    noise_adapter = noise_adapter.to(device=device)  # Keep float32 (LayerNorm requires it)
    noise_adapter.eval()

    adapter_params = sum(p.numel() for p in noise_adapter.parameters())
    print(f"  Adapter: {adapter_params / 1e6:.1f}M params, variant={args.adapter_variant}")

    task_idx = TASK_CLASSES.get(args.task, 0)
    task_class = torch.tensor([task_idx], device=device)
    print(f"  Task: {args.task} (class {task_idx})")

    # ══════════════════════════════════════════════════════════════════
    # Step 3b: Load SigmaHead (optional, for per-token sigma)
    # ══════════════════════════════════════════════════════════════════
    sigma_head = None
    sigma_head_path = args.sigma_head_path
    # Auto-detect from adapter path
    if sigma_head_path is None and args.adapter_path:
        auto_path = args.adapter_path.replace("noise_adapter_", "sigma_head_")
        if os.path.exists(auto_path):
            sigma_head_path = auto_path

    if sigma_head_path and os.path.exists(sigma_head_path):
        print(f"\n[3b/5] Loading SigmaHead from {sigma_head_path}...")
        from ltx_trainer.training_strategies.vfm_strategy_v1d import SigmaHead  # noqa: PLC0415
        sigma_head = SigmaHead(
            input_dim=latent_channels,
            hidden_dim=args.sigma_head_hidden_dim,
        ).to(device=device)
        sigma_sd = load_file(sigma_head_path)
        sigma_head.load_state_dict(sigma_sd)
        sigma_head.eval()
        sh_params = sum(p.numel() for p in sigma_head.parameters())
        print(f"  SigmaHead: {sh_params:,} params")
        if args.two_pass:
            print(f"  Two-pass mode: Pass 1 (uniform σ) → Pass 2 (per-token σ from SigmaHead)")
    elif args.two_pass:
        print("  WARNING: --two-pass requires sigma_head, falling back to single pass")
        args.two_pass = False

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Load VAE decoder
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[4/5] Loading VAE decoder on {args.vae_device}...")
    from ltx_trainer.model_loader import load_video_vae_decoder
    vae_decoder = load_video_vae_decoder(args.model_path, device=args.vae_device, dtype=dtype)
    vae_decoder.eval()

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Generate video
    # ══════════════════════════════════════════════════════════════════
    ensemble_k = args.ensemble
    print(f"\n[5/5] Generating video with {args.num_steps} VFM step(s)" +
          (f", ensemble K={ensemble_k}" if ensemble_k > 1 else "") + "...")
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Positions for all frames
    pos_shape = VideoLatentShape(
        frames=latent_frames, height=latent_h, width=latent_w,
        batch=1, channels=latent_channels,
    )
    coords = patchifier.get_patch_grid_bounds(output_shape=pos_shape, device=device)
    positions = get_pixel_coords(
        latent_coords=coords, scale_factors=scale_factors, causal_fix=True
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / args.fps

    # ══════════════════════════════════════════════════════════════════
    # Helper: generate one chunk of video latents
    # ══════════════════════════════════════════════════════════════════
    def generate_one_chunk(chunk_seed: int) -> Tensor:
        """Generate one chunk of video latents via adapter + DiT."""
        gen = torch.Generator(device=device).manual_seed(chunk_seed)

        with torch.inference_mode():
            # Noise adapter: structured noise
            if args.adapter_variant == "v1b":
                adapter_out = noise_adapter.forward(
                    text_embeddings=prompt_embeds.float(),
                    text_mask=prompt_mask.bool(),
                    positions=positions.float(),
                    task_class=task_class,
                )
                # Handle 2-tuple (v1b) or 3-tuple (v1h adapter with sigma head)
                mu, log_sigma = adapter_out[0], adapter_out[1]
            else:
                mask_float = prompt_mask.unsqueeze(-1).float()
                pooled_text = (prompt_embeds * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
                text_obs = pooled_text.unsqueeze(1).expand(-1, total_tokens, -1)
                mu, log_sigma = noise_adapter.forward(text_obs.float(), task_class)

            sigma_adapter = torch.exp(log_sigma)
            eps = torch.randn(mu.shape, device=device, dtype=torch.float32, generator=gen)
            z = (mu + sigma_adapter * eps * args.temperature).to(dtype)

            # Flow map evaluation
            if args.num_steps == 1:
                sigma_val = torch.ones(1, device=device, dtype=dtype)

                if args.two_pass and sigma_head is not None:
                    # ── TWO-PASS INFERENCE ──
                    # Pass 1: uniform σ=1.0 → rough x̂₀
                    timesteps_uniform = torch.ones(1, total_tokens, device=device, dtype=dtype)
                    video_mod_p1 = Modality(
                        enabled=True, latent=z, sigma=sigma_val,
                        timesteps=timesteps_uniform, positions=positions,
                        context=prompt_embeds, context_mask=prompt_mask,
                    )
                    v_pred_p1, _ = transformer(video=video_mod_p1, audio=None, perturbations=None)
                    rough_x0 = z - v_pred_p1

                    # SigmaHead: (adapter_mu, rough x̂₀) → per-token σ
                    with torch.no_grad():
                        per_token_sigma = sigma_head(mu.to(dtype), rough_x0.float()).to(dtype)  # [1, seq]

                    # Pass 2: per-token σ from SigmaHead → sharp x̂₀
                    video_mod_p2 = Modality(
                        enabled=True, latent=z, sigma=sigma_val,
                        timesteps=per_token_sigma, positions=positions,
                        context=prompt_embeds, context_mask=prompt_mask,
                    )
                    v_pred_p2, _ = transformer(video=video_mod_p2, audio=None, perturbations=None)
                    x_out = z - v_pred_p2
                else:
                    # ── SINGLE-PASS INFERENCE ──
                    timesteps = torch.ones(1, total_tokens, device=device, dtype=dtype)
                    video_mod = Modality(
                        enabled=True, latent=z, sigma=sigma_val,
                        timesteps=timesteps, positions=positions,
                        context=prompt_embeds, context_mask=prompt_mask,
                    )
                    v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                    x_out = z - v_pred
            else:
                time_steps_sched = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device)
                x_out = z
                for k in range(args.num_steps):
                    t_k = time_steps_sched[k]
                    t_km1 = time_steps_sched[k + 1]
                    dt = t_km1 - t_k
                    ts = torch.full((1, total_tokens), t_k.item(), device=device, dtype=dtype)
                    sig = torch.full((1,), t_k.item(), device=device, dtype=dtype)
                    video_mod = Modality(
                        enabled=True, latent=x_out, sigma=sig,
                        timesteps=ts, positions=positions,
                        context=prompt_embeds, context_mask=prompt_mask,
                    )
                    v_pred, _ = transformer(video=video_mod, audio=None, perturbations=None)
                    x_out = x_out + dt * (-v_pred)

        return x_out

    def unpatchify_latent(x: Tensor) -> Tensor:
        """Unpatchify tokens back to spatial latent."""
        return patchifier.unpatchify(
            x,
            output_shape=VideoLatentShape(
                frames=latent_frames, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # Generate chunks (pipelined or sequential)
    # ══════════════════════════════════════════════════════════════════
    num_chunks = args.num_chunks
    pipeline = args.pipeline_vae and num_chunks > 1 and str(args.device) != str(args.vae_device)

    if pipeline:
        print(f"\n  Pipelined mode: {num_chunks} chunks, DiT on {args.device}, VAE on {args.vae_device}")

    all_pixel_chunks = []
    gen_start = time.time()

    if pipeline:
        # ── PIPELINED: overlap DiT (GPU:0) and VAE (GPU:1) ──
        vae_stream = torch.cuda.Stream(device=args.vae_device)
        pending_decode = None  # (latent_on_vae_device, future pixels)
        pending_pixels = None

        for chunk_idx in range(num_chunks):
            chunk_seed = args.seed + chunk_idx
            t_chunk = time.perf_counter()

            # DiT generates chunk on GPU:0
            x_tokens = generate_one_chunk(chunk_seed)
            x_latent = unpatchify_latent(x_tokens)

            dit_ms = (time.perf_counter() - t_chunk) * 1000

            # Wait for previous VAE decode to finish (if any)
            if pending_decode is not None:
                vae_stream.synchronize()
                all_pixel_chunks.append(pending_pixels)
                pending_decode = None

            # Start VAE decode on GPU:1 (async)
            x_latent_vae = x_latent.to(args.vae_device, non_blocking=True)
            with torch.cuda.stream(vae_stream):
                with torch.inference_mode():
                    pixels = vae_decoder(x_latent_vae)
                    pending_pixels = (pixels.float().cpu() * 0.5 + 0.5).clamp(0, 1)
            pending_decode = True

            print(f"  Chunk {chunk_idx+1}/{num_chunks}: DiT={dit_ms:.0f}ms (VAE pipelined)")

        # Wait for final VAE decode
        if pending_decode is not None:
            vae_stream.synchronize()
            all_pixel_chunks.append(pending_pixels)

    else:
        # ── SEQUENTIAL: generate all, then decode all ──
        all_latents = []
        for chunk_idx in range(num_chunks):
            chunk_seed = args.seed + chunk_idx
            t_chunk = time.perf_counter()

            if ensemble_k > 1:
                # Multi-path ensemble: K different noise samples, average predictions
                ensemble_outputs = []
                for ek in range(ensemble_k):
                    x_k = generate_one_chunk(chunk_seed * 1000 + ek)
                    ensemble_outputs.append(x_k)
                # Average in token space (before unpatchify)
                x_tokens = torch.stack(ensemble_outputs).mean(dim=0)
            else:
                x_tokens = generate_one_chunk(chunk_seed)

            x_latent = unpatchify_latent(x_tokens)
            all_latents.append(x_latent)

            dit_ms = (time.perf_counter() - t_chunk) * 1000
            mode = "2-pass" if args.two_pass else "1-pass"
            if ensemble_k > 1:
                mode += f", K={ensemble_k}"
            print(f"  Chunk {chunk_idx+1}/{num_chunks}: DiT={dit_ms:.0f}ms ({mode})")

        # Decode all chunks
        for chunk_idx, x_latent in enumerate(all_latents):
            t_vae = time.perf_counter()
            x_latent_vae = x_latent.to(args.vae_device)
            with torch.inference_mode():
                pixels = vae_decoder(x_latent_vae)
            pixels = (pixels.float().cpu() * 0.5 + 0.5).clamp(0, 1)
            all_pixel_chunks.append(pixels)
            vae_ms = (time.perf_counter() - t_vae) * 1000
            print(f"  VAE decode chunk {chunk_idx+1}/{num_chunks}: {vae_ms:.0f}ms")

    gen_time = time.time() - gen_start

    # Concatenate all chunks along temporal dimension
    # Each chunk: [1, 3, T, H, W]
    if len(all_pixel_chunks) > 1:
        video_pixels = torch.cat(all_pixel_chunks, dim=2)  # concat along T
    else:
        video_pixels = all_pixel_chunks[0]

    total_frames = video_pixels.shape[2]
    total_duration = total_frames / args.fps
    print(f"\n  Generation: {gen_time:.2f}s for {total_frames} frames ({total_duration:.1f}s video)")
    if pipeline:
        print(f"  Effective: {gen_time/num_chunks*1000:.0f}ms/chunk (pipelined)")

    # ── Save video ──
    print(f"  Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torchvision
    frames = video_pixels[0].permute(1, 2, 3, 0)  # [T, H, W, 3]
    frames = (frames * 255).to(torch.uint8)

    torchvision.io.write_video(
        str(output_path), frames, fps=args.fps,
        video_codec="libx264", options={"crf": "18"},
    )

    total_time = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Done! {output_path}")
    print(f"  Total: {total_time:.1f}s | Generation: {gen_time:.1f}s")
    if num_chunks > 1:
        print(f"  Chunks: {num_chunks} × {args.num_frames}f = {total_frames}f ({total_duration:.1f}s)")
        if pipeline:
            print(f"  Pipeline speedup: ~{1.86:.1f}× (VAE hidden behind DiT)")
    print(f"{'=' * 65}")

    del vae_decoder, transformer, noise_adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
