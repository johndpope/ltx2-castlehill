#!/usr/bin/env python3
"""Diagnostic: Test SCD decoder with MULTI-FRAME input (matching training).

Hypothesis: SCD decoder produces grid artifacts because it's trained on 4-frame chunks
but used on single frames at inference. This test denoises all 4 frames simultaneously
through the decoder (matching training) to see if the output improves.

This is NOT how SCD is supposed to work at inference, but it tests whether the
train/inference mismatch is the cause of the grid artifacts.
"""

import argparse
import time

import torch
from dataclasses import replace as dc_replace
from PIL import Image

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
    parser.add_argument("--lora-path", default="/media/2TB/omnitransfer/output/scd_ditto_subset/checkpoints/lora_weights_step_02000.safetensors")
    parser.add_argument("--cached-embedding", required=True)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--quantization", default="int8-quanto")
    parser.add_argument("--output", default="/media/2TB/omnitransfer/inference/test_scd_multiframe.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=4, help="Latent frames per decoder call (match training)")
    parser.add_argument("--encoder-layers", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    tokens_per_frame = latent_h * latent_w
    nf = args.num_frames

    print(f"Resolution: {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"Tokens per frame: {tokens_per_frame}, frames: {nf}, total tokens: {nf * tokens_per_frame}")

    # 1. Load embedding
    print("\n[1/4] Loading cached embedding...")
    embed_data = torch.load(args.cached_embedding, weights_only=True)
    prompt_embeds = embed_data.get("video_prompt_embeds", embed_data.get("prompt_embeds"))
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    prompt_mask = embed_data.get("prompt_attention_mask")
    if prompt_mask is not None:
        prompt_mask = prompt_mask.to(device=device)
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(0)

    use_cfg = args.guidance_scale > 1.0
    null_embeds = torch.zeros_like(prompt_embeds) if use_cfg else None
    null_mask = torch.zeros_like(prompt_mask) if use_cfg and prompt_mask is not None else None

    # 2. Load transformer + LoRA + SCD
    print("\n[2/4] Loading transformer + LoRA + SCD wrapping...")
    from ltx_trainer.model_loader import load_transformer

    transformer = load_transformer(args.checkpoint)

    if args.quantization != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantization})...")
        quantize_model(transformer, args.quantization)

    transformer = transformer.to(device)

    # Load LoRA
    import sys
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/ltx2-omnitransfer/ltx-trainer/scripts")
    from scd_inference import load_lora_weights
    transformer = load_lora_weights(transformer, args.lora_path, encoder_layers=args.encoder_layers)
    transformer = transformer.get_base_model()

    # SCD wrap
    from ltx_core.model.transformer.scd_model import LTXSCDModel, build_frame_causal_mask, shift_encoder_features
    scd_model = LTXSCDModel(
        base_model=transformer,
        encoder_layers=args.encoder_layers,
        decoder_input_combine="add",
    )
    scd_model.eval()
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  GPU memory: {mem_gb:.1f} GB")

    # 3. Setup
    print("\n[3/4] Setting up...")
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()

    # Positions for nf frames
    coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=nf, height=latent_h, width=latent_w,
            batch=1, channels=latent_channels,
        ),
        device=device,
    )
    positions = get_pixel_coords(latent_coords=coords, scale_factors=scale_factors, causal_fix=True).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / 24.0

    # Sigma schedule for nf-frame window (matching training)
    total_tokens = nf * latent_h * latent_w
    dummy_latent = torch.empty(1, 1, nf, latent_h, latent_w)
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=args.num_inference_steps, latent=dummy_latent).to(device=device, dtype=dtype)
    print(f"  Sigma schedule: [{sigmas[0]:.4f} → {sigmas[-2]:.4f} → {sigmas[-1]:.4f}]")
    print(f"  Token count for shift: {total_tokens}")

    # 4. MULTI-FRAME SCD denoising (matching training)
    print(f"\n[4/4] Multi-frame SCD denoising ({nf} frames, {nf * tokens_per_frame} tokens)...")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Create clean "previous frames" for encoder (use zeros — like training frame 0)
    clean_latents = torch.zeros(1, latent_channels, nf, latent_h, latent_w, device=device, dtype=dtype)

    # Run encoder on clean frames (like training)
    clean_patchified = patchifier.patchify(clean_latents)
    encoder_modality = Modality(
        enabled=True,
        latent=clean_patchified,
        timesteps=torch.zeros(1, nf * tokens_per_frame, device=device, dtype=dtype),
        positions=positions,
        context=prompt_embeds,
        context_mask=prompt_mask,
    )

    print("  Running encoder on clean frames...")
    encoder_video_args, _ = scd_model.forward_encoder(
        video=encoder_modality,
        audio=None,
        perturbations=None,
        tokens_per_frame=tokens_per_frame,
    )
    encoder_features = encoder_video_args.x
    print(f"  Encoder features: mean={encoder_features.float().mean().item():.2f} "
          f"std={encoder_features.float().std().item():.2f}")

    # Shift encoder features by 1 frame (like training)
    shifted_features = shift_encoder_features(encoder_features, tokens_per_frame, nf)
    print(f"  Shifted features: mean={shifted_features.float().mean().item():.2f} "
          f"std={shifted_features.float().std().item():.2f}")

    # Initialize noisy latents for nf frames
    x_t = torch.randn(1, latent_channels, nf, latent_h, latent_w,
                       device=device, dtype=dtype, generator=generator)

    # Denoising loop — ALL frames through decoder at once
    t_start = time.time()
    for step in range(args.num_inference_steps):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]

        noisy_patch = patchifier.patchify(x_t)
        seq_len = noisy_patch.shape[1]

        # Decoder modality for all frames (matching training)
        dec_modality = Modality(
            enabled=True,
            latent=noisy_patch,
            timesteps=torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype),
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_mask,
        )

        velocity, _ = scd_model.forward_decoder(
            video=dec_modality,
            encoder_features=shifted_features,
            audio=None,
            perturbations=None,
        )

        # CFG
        if use_cfg:
            uncond_modality = Modality(
                enabled=True,
                latent=noisy_patch,
                timesteps=torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype),
                positions=positions,
                context=null_embeds,
                context_mask=null_mask,
            )
            velocity_uncond, _ = scd_model.forward_decoder(
                video=uncond_modality,
                encoder_features=shifted_features,
                audio=None,
                perturbations=None,
            )
            velocity = velocity_uncond + args.guidance_scale * (velocity - velocity_uncond)

        # Euler step
        dt = sigma_next - sigma
        noisy_patch = (noisy_patch.float() + velocity.float() * dt.float()).to(dtype)

        if step in (0, 1, 5, 14, 28, 29):
            v_std = velocity.float().std().item()
            x_std = noisy_patch.float().std().item()
            print(f"  Step {step:2d}: sigma={sigma.item():.4f}→{sigma_next.item():.4f} "
                  f"v_std={v_std:.4f} x_std={x_std:.4f}")

        x_t = patchifier.unpatchify(
            noisy_patch,
            output_shape=VideoLatentShape(
                frames=nf, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
        )

    gen_time = time.time() - t_start
    print(f"\n  Denoised in {gen_time:.1f}s")

    # Decode first frame with VAE
    print("\n  Decoding first frame with VAE...")
    from ltx_trainer.model_loader import load_video_vae_decoder
    vae_decoder = load_video_vae_decoder(args.checkpoint)
    vae_decoder = vae_decoder.to("cuda:1").eval()

    with torch.inference_mode():
        first_frame = x_t[:, :, 0:1, :, :].to("cuda:1")
        pixels = vae_decoder(first_frame)
        if hasattr(pixels, "sample"):
            pixels = pixels.sample
        pixels = pixels.float().clamp(0, 1)
        print(f"  Pixel shape: {pixels.shape}, range: [{pixels.min():.3f}, {pixels.max():.3f}]")

    frame = pixels[0, :, 0].permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(frame)
    img.save(args.output)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
