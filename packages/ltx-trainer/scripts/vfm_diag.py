#!/usr/bin/env python3
# ruff: noqa: T201
"""Diagnostic: isolate where VFM inference breaks.

Three probes:
  A. Decode z (adapter output, no DiT) → is adapter z reasonable?
  B. Decode z - v_pred at sigma=0.5 uniform → current broken path
  C. Decode z - v_pred at SigmaHead sigma (two-pass) → proposed fix
  D. Decode x0_gt directly → sanity check
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CKPT_DIR = "/media/2TB/omnitransfer/output/vfm_v4a_overfit10/checkpoints"
# Training data (3840-dim LTX-2 style, needs caption_projection_shim)
EMB_PATH = "/media/12TB/ddit_ditto_data_23_overfit10/conditions_final/000000.pt"
GT_LATENT = "/media/12TB/ddit_ditto_data_23_overfit10/latents/000000.pt"
MODEL_PATH = "/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
LTX2_MODEL_PATH = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors"
STEP = 5000
DEVICE = torch.device("cuda:0")
VAE_DEVICE = torch.device("cuda:1")
DTYPE = torch.bfloat16

H, W, FRAMES = 448, 768, 25
latent_h, latent_w = H // 32, W // 32
latent_frames = (FRAMES - 1) // 8 + 1
total_tokens = latent_h * latent_w * latent_frames
latent_channels = 128


def save_video(pixels: torch.Tensor, path: str, fps: float = 24.0):
    import torchvision
    pixels = pixels.clamp(0, 1)
    frames = pixels[0].permute(1, 2, 3, 0)  # [T, H, W, 3]
    frames = (frames * 255).to(torch.uint8)
    torchvision.io.write_video(path, frames, fps=fps, video_codec="libx264", options={"crf": "18"})
    print(f"  → saved {path}")


def decode_latent(latent_spatial, vae_decoder, path: str):
    """Decode spatial latent [1, C, F, H, W] → pixel video."""
    x = latent_spatial.to(VAE_DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        pixels = vae_decoder(x)
    pixels = (pixels.float().cpu() * 0.5 + 0.5)
    save_video(pixels, path)
    return pixels


def main():
    t0 = time.time()
    print("\n=== VFM Inference Diagnostic ===\n")

    # ── Load embeddings ──
    print("[1] Loading embeddings (training data path)...")
    emb = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
    prompt_embeds_raw = emb["video_prompt_embeds"].unsqueeze(0).to(DTYPE)
    prompt_mask = emb["prompt_attention_mask"].unsqueeze(0).to(DEVICE)
    print(f"    raw embed shape: {prompt_embeds_raw.shape}")

    # Apply caption_projection_shim (3840→4096) matching training's use_cached_final_embeddings path
    if prompt_embeds_raw.shape[-1] == 3840:
        print("    Applying caption_projection_shim (3840→4096)...")
        from safetensors import safe_open
        from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
        caption_proj = PixArtAlphaTextProjection(in_features=3840, hidden_size=4096)
        with safe_open(LTX2_MODEL_PATH, framework="pt") as f:
            prefix = "model.diffusion_model.caption_projection."
            for key in f.keys():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    param = f.get_tensor(key)
                    parts = param_name.split(".")
                    obj = caption_proj
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    setattr(obj, parts[-1], torch.nn.Parameter(param))
        caption_proj = caption_proj.to(DEVICE, dtype=DTYPE).eval()
        with torch.inference_mode():
            B = prompt_embeds_raw.shape[0]
            prompt_embeds = caption_proj(prompt_embeds_raw.to(DEVICE)).view(B, -1, 4096)
        del caption_proj
    else:
        prompt_embeds = prompt_embeds_raw.to(DEVICE)

    print(f"    final embed shape: {prompt_embeds.shape}")

    # ── Load GT latent ──
    print("[2] Loading GT latent (training data path)...")
    gt_data = torch.load(GT_LATENT, map_location="cpu", weights_only=True)
    if isinstance(gt_data, dict):
        x0_raw = gt_data.get("latents", gt_data.get("latent", gt_data.get("video_latent", list(gt_data.values())[0])))
    else:
        x0_raw = gt_data
    # Handle patchified format: if [seq, C], unpatchify
    if x0_raw.ndim == 2:
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.types import VideoLatentShape
        patchifier = VideoLatentPatchifier(patch_size=1)
        x0_tokens = x0_raw.unsqueeze(0)  # [1, seq, C]
        x0_spatial = patchifier.unpatchify(
            x0_tokens,
            output_shape=VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels),
        )
    elif x0_raw.ndim == 4:
        x0_spatial = x0_raw.unsqueeze(0)  # [1, C, F, H, W]
    else:
        x0_spatial = x0_raw
    print(f"    GT latent shape: {x0_spatial.shape}")

    # ── Load VAE ──
    print("[3] Loading VAE decoder...")
    from ltx_trainer.model_loader import load_video_vae_decoder
    vae = load_video_vae_decoder(MODEL_PATH, device=VAE_DEVICE, dtype=DTYPE)
    vae.eval()

    # ── PROBE D: GT x0 decode (sanity check) ──
    print("\n[D] Decoding GT x0 (sanity check)...")
    decode_latent(x0_spatial, vae, "/tmp/diag_D_gt.mp4")

    # ── Load transformer + LoRA ──
    print("\n[4] Loading transformer + LoRA...")
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.quantization import quantize_model
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    transformer = load_transformer(MODEL_PATH, device="cpu", dtype=DTYPE)
    print("  Quantizing int8...")
    transformer = quantize_model(transformer, "int8-quanto", device=str(DEVICE))
    transformer = transformer.to(DEVICE)

    lora_path = f"{CKPT_DIR}/lora_weights_step_{STEP:05d}.safetensors"
    lora_state = load_file(lora_path)
    target_modules = set()
    lora_rank = None
    for key, value in lora_state.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p in ("to_k", "to_q", "to_v", "to_out"):
                if i + 1 < len(parts) and parts[i + 1] == "0":
                    target_modules.add(f"{p}.0")
                else:
                    target_modules.add(p)
    print(f"  LoRA rank={lora_rank}, modules={sorted(target_modules)}")
    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=sorted(target_modules), lora_dropout=0.0)
    transformer = get_peft_model(transformer, lora_cfg)
    set_peft_model_state_dict(transformer, lora_state)
    transformer = transformer.merge_and_unload()
    transformer.eval()

    # ── Load adapter ──
    print("[5] Loading adapter...")
    from ltx_core.model.transformer.noise_adapter_v1b import TASK_CLASSES, create_noise_adapter_v1b
    text_embed_dim = prompt_embeds.shape[-1]
    noise_adapter = create_noise_adapter_v1b(
        text_dim=text_embed_dim, latent_dim=latent_channels,
        hidden_dim=512, num_heads=8, num_layers=4, pos_dim=256,
    )
    adapter_state = load_file(f"{CKPT_DIR}/noise_adapter_step_{STEP:05d}.safetensors")
    noise_adapter.load_state_dict(adapter_state)
    noise_adapter = noise_adapter.to(DEVICE).eval()

    task_class = torch.tensor([TASK_CLASSES.get("i2v", 0)], device=DEVICE)

    # ── Load SigmaHead ──
    print("[6] Loading SigmaHead...")
    from ltx_trainer.training_strategies.vfm_strategy_v1d import SigmaHead
    sigma_head = SigmaHead(latent_dim=latent_channels, hidden_dim=256).to(DEVICE)
    sh_state = load_file(f"{CKPT_DIR}/sigma_head_step_{STEP:05d}.safetensors")
    sigma_head.load_state_dict(sh_state)
    sigma_head.eval()

    # ── Build positions ── use fps from training latent (20.0 for this sample)
    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()
    pos_shape = VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels)
    coords = patchifier.get_patch_grid_bounds(output_shape=pos_shape, device=DEVICE)
    positions = get_pixel_coords(latent_coords=coords, scale_factors=scale_factors, causal_fix=True).to(DTYPE)
    gt_fps = 20.0  # from training latent fps metadata
    positions[:, 0, ...] = positions[:, 0, ...] / gt_fps
    print(f"    Positions built with fps={gt_fps} (matches training)")

    # ── Run adapter ──
    print("\n[7] Running adapter...")
    with torch.inference_mode():
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        adapter_out = noise_adapter.forward(
            text_embeddings=prompt_embeds.float(),
            text_mask=prompt_mask.bool(),
            positions=positions.float(),
            task_class=task_class,
        )
        mu, log_sigma = adapter_out[0], adapter_out[1]
        sigma_adapter = torch.exp(log_sigma)
        eps = torch.randn(mu.shape, device=DEVICE, dtype=torch.float32, generator=gen)
        z = (mu + sigma_adapter * eps).to(DTYPE)

    print(f"  mu:          mean={mu.mean():.4f}  std={mu.std():.4f}  norm={mu.norm(dim=-1).mean():.4f}")
    print(f"  sigma_adapt: mean={sigma_adapter.mean():.4f}  std={sigma_adapter.std():.4f}")
    print(f"  z:           mean={z.mean():.4f}  std={z.std():.4f}  norm={z.norm(dim=-1).mean():.4f}")

    # Compare to GT x0 stats
    from ltx_core.components.patchifiers import VideoLatentPatchifier as VLP
    pf2 = VLP(patch_size=1)
    x0_tokens = pf2.patchify(x0_spatial.to(DEVICE, DTYPE))  # [1, seq, 128]
    print(f"  x0_gt:       mean={x0_tokens.mean():.4f}  std={x0_tokens.std():.4f}  norm={x0_tokens.norm(dim=-1).mean():.4f}")

    # ── PROBE A: Decode z directly (bypass DiT) ──
    print("\n[A] Decoding z directly (no DiT)...")
    z_spatial = patchifier.unpatchify(z, output_shape=VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels))
    decode_latent(z_spatial.float(), vae, "/tmp/diag_A_z_raw.mp4")

    # ── PROBE B: sigma=0.5 uniform (current broken path) ──
    print("\n[B] DiT with sigma=0.5 uniform (current path)...")
    with torch.inference_mode():
        boot_sigma = torch.tensor([0.5], device=DEVICE, dtype=DTYPE)
        boot_ts = torch.full((1, total_tokens), 0.5, device=DEVICE, dtype=DTYPE)
        video_mod_b = Modality(enabled=True, latent=z, sigma=boot_sigma, timesteps=boot_ts,
                               positions=positions, context=prompt_embeds, context_mask=prompt_mask)
        v_b, _ = transformer(video=video_mod_b, audio=None, perturbations=None)
        x_b = z - v_b
    print(f"  v_pred: mean={v_b.mean():.4f}  std={v_b.std():.4f}  |v|={v_b.norm(dim=-1).mean():.4f}")
    print(f"  x_out:  mean={x_b.mean():.4f}  std={x_b.std():.4f}  |x|={x_b.norm(dim=-1).mean():.4f}")
    mse_b = (x_b.float() - x0_tokens.float()).pow(2).mean().item()
    print(f"  MSE vs GT: {mse_b:.6f}")
    x_b_spatial = patchifier.unpatchify(x_b, output_shape=VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels))
    decode_latent(x_b_spatial.float(), vae, "/tmp/diag_B_sigma05.mp4")

    # ── PROBE C: SigmaHead sigma (two-pass) ──
    print("\n[C] DiT with SigmaHead sigma (two-pass)...")
    with torch.inference_mode():
        # Use GT x0 as oracle (upper bound)
        per_tok_sigma_oracle = sigma_head(mu.to(DTYPE), x0_tokens.float()).to(DTYPE)
        batch_sigma_oracle = per_tok_sigma_oracle.mean(dim=1)
        print(f"  SigmaHead(mu, GT_x0): mean={per_tok_sigma_oracle.mean():.4f}  std={per_tok_sigma_oracle.std():.4f}")
        print(f"  batch_sigma: {batch_sigma_oracle.item():.4f}")

        video_mod_c = Modality(enabled=True, latent=z, sigma=batch_sigma_oracle, timesteps=per_tok_sigma_oracle,
                               positions=positions, context=prompt_embeds, context_mask=prompt_mask)
        v_c, _ = transformer(video=video_mod_c, audio=None, perturbations=None)
        x_c = z - v_c
    print(f"  v_pred: mean={v_c.mean():.4f}  std={v_c.std():.4f}  |v|={v_c.norm(dim=-1).mean():.4f}")
    print(f"  x_out:  mean={x_c.mean():.4f}  std={x_c.std():.4f}  |x|={x_c.norm(dim=-1).mean():.4f}")
    mse_c = (x_c.float() - x0_tokens.float()).pow(2).mean().item()
    print(f"  MSE vs GT: {mse_c:.6f}")
    x_c_spatial = patchifier.unpatchify(x_c, output_shape=VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels))
    decode_latent(x_c_spatial.float(), vae, "/tmp/diag_C_sigma_head.mp4")

    # ── PROBE C2: two-pass (rough x0 → SigmaHead → re-run) ──
    print("\n[C2] Two-pass: rough x0 → SigmaHead → DiT again...")
    with torch.inference_mode():
        per_tok_sigma_rough = sigma_head(mu.to(DTYPE), x_b.float()).to(DTYPE)
        batch_sigma_rough = per_tok_sigma_rough.mean(dim=1)
        print(f"  SigmaHead(mu, rough_x0): mean={per_tok_sigma_rough.mean():.4f}  std={per_tok_sigma_rough.std():.4f}")

        video_mod_c2 = Modality(enabled=True, latent=z, sigma=batch_sigma_rough, timesteps=per_tok_sigma_rough,
                                positions=positions, context=prompt_embeds, context_mask=prompt_mask)
        v_c2, _ = transformer(video=video_mod_c2, audio=None, perturbations=None)
        x_c2 = z - v_c2
    mse_c2 = (x_c2.float() - x0_tokens.float()).pow(2).mean().item()
    print(f"  MSE vs GT: {mse_c2:.6f}")
    x_c2_spatial = patchifier.unpatchify(x_c2, output_shape=VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=latent_channels))
    decode_latent(x_c2_spatial.float(), vae, "/tmp/diag_C2_twopass.mp4")

    # ── PROBE E: sigma sweep ──
    print("\n[E] Sigma sweep...")
    for s in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        with torch.inference_mode():
            sig = torch.tensor([s], device=DEVICE, dtype=DTYPE)
            ts = torch.full((1, total_tokens), s, device=DEVICE, dtype=DTYPE)
            vm = Modality(enabled=True, latent=z, sigma=sig, timesteps=ts,
                          positions=positions, context=prompt_embeds, context_mask=prompt_mask)
            v_e, _ = transformer(video=vm, audio=None, perturbations=None)
            x_e = z - v_e
        mse_e = (x_e.float() - x0_tokens.float()).pow(2).mean().item()
        vn = v_e.norm(dim=-1).mean().item()
        print(f"  sigma={s:.1f}: v_norm={vn:.4f}  MSE_vs_GT={mse_e:.6f}")

    print(f"\n=== Done in {time.time()-t0:.1f}s ===")
    print("Files: /tmp/diag_A_z_raw.mp4  /tmp/diag_B_sigma05.mp4  /tmp/diag_C_sigma_head.mp4  /tmp/diag_C2_twopass.mp4  /tmp/diag_D_gt.mp4")


if __name__ == "__main__":
    main()
