"""Standalone reconstruction script for CliffordVideoAttention experiments.

Loads a trained LoRA checkpoint, runs denoising on precomputed latents,
decodes through VAE, and logs source/predict/target triplet images to W&B.

Usage:
    python scripts/clifford_reconstruct.py \
        --checkpoint /path/to/model.safetensors \
        --lora-path /path/to/lora_weights.safetensors \
        --data-root /path/to/ditto_10sample \
        --output-dir /path/to/output
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ltx-core" / "src"))


def main():
    parser = argparse.ArgumentParser(description="Clifford Attention Reconstruction")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--lora-path", required=True, help="LoRA weights path")
    parser.add_argument("--data-root", required=True, help="Preprocessed data root")
    parser.add_argument("--output-dir", default=".", help="Output directory for images")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to reconstruct")
    parser.add_argument("--num-steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--wandb-project", default="clifford-video-attn", help="W&B project")
    parser.add_argument("--num-spatial-shifts", type=int, default=12)
    parser.add_argument("--num-temporal-shifts", type=int, default=4)
    parser.add_argument("--num-channel-shifts", type=int, default=4)
    args = parser.parse_args()

    device = "cuda:0"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load transformer + Clifford swap + LoRA + quantize
    print("Loading transformer...")
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.quantization import quantize_model
    from ltx_core.model.transformer.clifford_attention import CliffordVideoAttention
    from ltx_core.model.transformer.attention import Attention

    transformer = load_transformer(args.checkpoint, device="cpu", dtype=torch.bfloat16)

    # Swap attention
    swapped = 0
    for block in transformer.transformer_blocks:
        old = block.attn1
        if isinstance(old, Attention):
            new_attn = CliffordVideoAttention(
                query_dim=old.to_q.in_features, heads=old.heads, dim_head=old.dim_head,
                norm_eps=1e-6, rope_type=old.rope_type, attention_function=old.attention_function,
                apply_gated_attention=old.to_gate_logits is not None,
                num_spatial_shifts=args.num_spatial_shifts,
                num_temporal_shifts=args.num_temporal_shifts,
                num_channel_shifts=args.num_channel_shifts,
            )
            new_attn.to_q = old.to_q
            new_attn.to_k = old.to_k
            new_attn.to_v = old.to_v
            new_attn.to_out = old.to_out
            new_attn.q_norm = old.q_norm
            new_attn.k_norm = old.k_norm
            if old.to_gate_logits is not None:
                new_attn.to_gate_logits = old.to_gate_logits
            block.attn1 = new_attn
            swapped += 1
    print(f"Swapped {swapped} blocks")

    # Quantize first, then load LoRA (matches training order)
    print("Quantizing...")
    transformer = quantize_model(transformer, precision="int8-quanto", device=device)
    transformer = transformer.to(device)

    # Load LoRA on top of quantized model
    from scripts.inference import load_lora_weights
    transformer = load_lora_weights(transformer, args.lora_path)
    transformer = transformer.eval()
    print(f"Transformer on GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    # 2. Load VAE decoder to CPU (will move to GPU for decode, then back)
    from ltx_trainer.model_loader import load_video_vae_decoder
    vae = load_video_vae_decoder(args.checkpoint, device="cpu", dtype=torch.bfloat16)

    # 4. Load data samples
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.model.transformer.modality import Modality
    # timestep embedding is handled internally by the model
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.types import VideoLatentShape, SpatioTemporalScaleFactors
    from ltx_core.tools import get_pixel_coords

    VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

    data_root = Path(args.data_root)
    latent_files = sorted((data_root / "latents").glob("*.pt"))[:args.num_samples]
    cond_files = sorted((data_root / "conditions_final").glob("*.pt"))[:args.num_samples]

    patchifier = VideoLatentPatchifier(patch_size=1)
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(args.num_steps).to(device)
    fps = 25.0

    all_triplets = []

    for i, (lf, cf) in enumerate(zip(latent_files, cond_files)):
        print(f"\n--- Sample {i} ---")
        latent_data = torch.load(lf, weights_only=False)
        cond_data = torch.load(cf, weights_only=False)

        # Target latents [C, F, H, W] -> [1, C, F, H, W]
        target_latents = latent_data["latents"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        b, c, f, h, w = target_latents.shape
        print(f"  Latent shape: {target_latents.shape}")

        # Get text embeddings
        video_feats = cond_data["video_prompt_embeds"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        mask = cond_data["prompt_attention_mask"].unsqueeze(0).to(device)

        # Pass raw 3840-dim features — the transformer's caption_projection handles 3840→4096
        video_embeds = video_feats

        # Patchify target
        target_patched = patchifier.patchify(target_latents)
        seq_len = target_patched.shape[1]

        # Create noise
        noise = torch.randn_like(target_patched)

        # Create noisy input at sigma=1.0 (pure noise for generation)
        # For reconstruction test, use sigma=0.5 (half noisy) to see if model can denoise
        test_sigma = 0.5
        noisy_patched = (1 - test_sigma) * target_patched + test_sigma * noise

        # Positions (same as training strategy)
        latent_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(frames=f, height=h, width=w, batch=1, channels=c),
            device=device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        ).to(torch.bfloat16)
        positions[:, 0, ...] = positions[:, 0, ...] / fps

        # Denoise
        print(f"  Denoising from sigma={test_sigma:.2f} ({args.num_steps} steps)...")
        # Find the starting index in sigmas closest to test_sigma
        start_idx = (sigmas - test_sigma).abs().argmin().item()
        active_sigmas = sigmas[start_idx:]

        x = noisy_patched
        with torch.inference_mode():
            for step_i, (sc, sn) in enumerate(zip(active_sigmas[:-1], active_sigmas[1:])):
                sb = sc.unsqueeze(0)
                # Per-token timestep: scalar sigma broadcast to all tokens, then embed
                # Per-token timestep: raw sigma values (model embeds internally)
                ts = sb.unsqueeze(1).expand(-1, seq_len).to(torch.bfloat16)  # [1, seq_len]
                vm = Modality(
                    enabled=True, latent=x, sigma=sb, timesteps=ts,
                    positions=positions, context=video_embeds, context_mask=None,
                )
                result = transformer(video=vm, audio=None, perturbations=None)
                # Model returns (video_out, audio_out) - video_out may be NamedTuple or Tensor
                vo = result[0] if isinstance(result, tuple) else result
                velocity = vo.x if hasattr(vo, 'x') else vo
                x = x + velocity * (sn - sc)

        print("  Denoising done. Decoding via VAE...")

        # Unpatchify
        from ltx_core.types import VideoLatentShape
        output_shape = VideoLatentShape(batch=1, channels=c, frames=f, height=h, width=w)
        pred_spatial = patchifier.unpatchify(x, output_shape)
        noisy_spatial = patchifier.unpatchify(noisy_patched, output_shape)

        # Decode through VAE (move to GPU, decode, move back)
        vae = vae.to(device)
        torch.cuda.empty_cache()

        with torch.inference_mode():
            gt_decoded = vae(target_latents).float().clamp(-1, 1) * 0.5 + 0.5
            torch.cuda.empty_cache()
            pred_decoded = vae(pred_spatial.to(dtype=torch.bfloat16)).float().clamp(-1, 1) * 0.5 + 0.5
            torch.cuda.empty_cache()
            noisy_decoded = vae(noisy_spatial.to(dtype=torch.bfloat16)).float().clamp(-1, 1) * 0.5 + 0.5
            torch.cuda.empty_cache()

        vae = vae.to("cpu")
        torch.cuda.empty_cache()

        # [1, 3, T, H, W] -> mid-frame [3, H, W]
        gt_frame = gt_decoded[0, :, f // 2].cpu()
        pred_frame = pred_decoded[0, :, f // 2].cpu()
        noisy_frame = noisy_decoded[0, :, f // 2].cpu()

        # Save triplet image
        import torchvision.utils as vutils
        grid = vutils.make_grid([noisy_frame, pred_frame, gt_frame], nrow=3, padding=4)
        vutils.save_image(grid, output_dir / f"recon_sample_{i:03d}.png")
        print(f"  Saved: {output_dir / f'recon_sample_{i:03d}.png'}")

        all_triplets.append(grid)

    # 5. Upload to W&B
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, name="clifford-reconstruction", tags=["inference", "reconstruction"])
        for i, grid in enumerate(all_triplets):
            img_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            run.log({
                f"reconstruction/sample_{i}": wandb.Image(
                    img_np, caption=f"Sample {i}: Source (noisy) | Predict | Target (GT)"
                ),
            })
        run.finish()
        print(f"\nUploaded {len(all_triplets)} reconstruction triplets to W&B")
    except Exception as e:
        print(f"W&B upload failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
