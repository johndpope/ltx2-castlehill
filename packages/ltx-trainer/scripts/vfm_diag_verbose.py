#!/usr/bin/env python3
# ruff: noqa: T201
"""
Verbose VFM Inference Diagnostic for Grok context.
Logs shapes, dtypes, norms, and key statistics at every step.
Hypothesis being tested: LoRA merge_and_unload fails with int8-quanto quantized models.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from safetensors import safe_open

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ltx-core/src"))

CKPT_DIR = "/media/2TB/omnitransfer/output/vfm_v4a_overfit10/checkpoints"
EMB_PATH = "/media/12TB/ddit_ditto_data_23_overfit10/conditions_final/000000.pt"
GT_LATENT = "/media/12TB/ddit_ditto_data_23_overfit10/latents/000000.pt"
MODEL_PATH = "/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
LTX2_MODEL_PATH = "/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors"
STEP = 5000
DEVICE = torch.device("cuda:0")
VAE_DEVICE = torch.device("cuda:1")
DTYPE = torch.bfloat16


def log(tag: str, t: torch.Tensor | None, extra: str = ""):
    if t is None:
        print(f"  [{tag}] None")
        return
    print(f"  [{tag}] shape={list(t.shape)}  dtype={t.dtype}  "
          f"mean={t.float().mean():.5f}  std={t.float().std():.5f}  "
          f"norm_per_tok={t.float().norm(dim=-1).mean():.4f}  "
          f"min={t.float().min():.4f}  max={t.float().max():.4f}"
          + (f"  {extra}" if extra else ""))


def main():
    t0 = time.time()
    print("=" * 70)
    print("  VFM v4a Verbose Inference Diagnostic")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  checkpoint_step: {STEP}")
    print(f"  model:           {MODEL_PATH}")
    print(f"  embeddings:      {EMB_PATH}")
    print(f"  gt_latent:       {GT_LATENT}")
    print(f"  device:          {DEVICE}")
    print(f"  dtype:           {DTYPE}")

    # ════════════════════════════════════════════════════════════════
    # [1] Embeddings
    # ════════════════════════════════════════════════════════════════
    print("\n[1] Loading embeddings...")
    emb = torch.load(EMB_PATH, map_location="cpu", weights_only=True)
    print(f"  Keys in .pt: {list(emb.keys() if isinstance(emb, dict) else [])}")
    for k, v in emb.items():
        if hasattr(v, "shape"):
            log(f"raw/{k}", v)
    prompt_embeds_raw = emb["video_prompt_embeds"].unsqueeze(0).to(DTYPE)
    prompt_mask = emb["prompt_attention_mask"].unsqueeze(0)
    log("prompt_mask", prompt_mask, f"sum_nonzero={prompt_mask.sum().item()}")

    # caption_projection_shim: 3840 → 4096
    print("\n  Applying caption_projection_shim (3840→4096)...")
    from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection
    cap_proj = PixArtAlphaTextProjection(in_features=3840, hidden_size=4096)
    loaded_keys = []
    with safe_open(LTX2_MODEL_PATH, framework="pt") as f:
        prefix = "model.diffusion_model.caption_projection."
        for key in f.keys():
            if key.startswith(prefix):
                pn = key[len(prefix):]
                param = f.get_tensor(key)
                parts = pn.split(".")
                obj = cap_proj
                for p in parts[:-1]: obj = getattr(obj, p)
                setattr(obj, parts[-1], torch.nn.Parameter(param))
                loaded_keys.append(pn)
    print(f"  caption_proj keys loaded: {loaded_keys}")
    cap_proj = cap_proj.to(DEVICE, dtype=DTYPE).eval()
    with torch.inference_mode():
        prompt_embeds = cap_proj(prompt_embeds_raw.to(DEVICE)).view(1, -1, 4096)
    log("prompt_embeds (post-shim)", prompt_embeds)
    del cap_proj

    # ════════════════════════════════════════════════════════════════
    # [2] GT latent
    # ════════════════════════════════════════════════════════════════
    print("\n[2] Loading GT latent...")
    gt_data = torch.load(GT_LATENT, map_location="cpu", weights_only=True)
    print(f"  Keys: {list(gt_data.keys() if isinstance(gt_data, dict) else ['tensor'])}")
    for k, v in gt_data.items():
        if hasattr(v, "shape"):
            log(f"gt/{k}", v)
    x0_spatial = gt_data["latents"].unsqueeze(0)  # [1, 128, F, H, W]
    C, F, H, W = x0_spatial.shape[1:]
    print(f"  x0_spatial.shape={list(x0_spatial.shape)}  C={C} F={F} H={H} W={W}")
    latent_frames, latent_h, latent_w = F, H, W
    total_tokens = F * H * W

    # ════════════════════════════════════════════════════════════════
    # [3] Patchify x0
    # ════════════════════════════════════════════════════════════════
    print("\n[3] Patchifying GT latent...")
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
    pf = VideoLatentPatchifier(patch_size=1)
    x0_tokens = pf.patchify(x0_spatial.to(DEVICE, DTYPE))  # [1, seq, 128]
    log("x0_tokens", x0_tokens, f"total_tokens={total_tokens}")

    # ════════════════════════════════════════════════════════════════
    # [4] Build positions
    # ════════════════════════════════════════════════════════════════
    print("\n[4] Building positions (fps=20.0 from training metadata)...")
    pos_shape = VideoLatentShape(frames=latent_frames, height=latent_h, width=latent_w, batch=1, channels=128)
    coords = pf.get_patch_grid_bounds(output_shape=pos_shape, device=DEVICE)
    positions = get_pixel_coords(latent_coords=coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True).to(DTYPE)
    positions[:, 0] = positions[:, 0] / 20.0
    log("positions", positions,
        f"t_range=[{positions[0,0,:,0].min().item():.3f},{positions[0,0,:,0].max().item():.3f}]")

    # ════════════════════════════════════════════════════════════════
    # [5] Load adapter
    # ════════════════════════════════════════════════════════════════
    print("\n[5] Loading NoiseAdapterV1b (hidden_dim=512, num_layers=4, num_heads=8, pos_dim=256)...")
    from ltx_core.model.transformer.noise_adapter_v1b import TASK_CLASSES, create_noise_adapter_v1b
    adapter = create_noise_adapter_v1b(
        text_dim=4096, latent_dim=128, hidden_dim=512, num_heads=8, num_layers=4, pos_dim=256
    )
    adapter_state = load_file(f"{CKPT_DIR}/noise_adapter_step_{STEP:05d}.safetensors")
    print(f"  adapter state_dict keys: {list(adapter_state.keys())[:5]}... ({len(adapter_state)} total)")
    for k, v in adapter_state.items():
        if "out_proj" in k or "mu_head" in k or "sigma_head" in k:
            log(f"  adapter_state/{k}", v)
    adapter.load_state_dict(adapter_state)
    adapter = adapter.to(DEVICE).eval()
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  adapter params: {adapter_params/1e6:.1f}M")

    task_class = torch.tensor([TASK_CLASSES.get("i2v", 0)], device=DEVICE, dtype=torch.long)
    print(f"  task_class: {task_class.item()} (i2v)")

    # ════════════════════════════════════════════════════════════════
    # [5b] Run adapter
    # ════════════════════════════════════════════════════════════════
    print("\n[5b] Running adapter forward...")
    with torch.inference_mode():
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        adapter_out = adapter.forward(
            text_embeddings=prompt_embeds.float(),
            text_mask=prompt_mask.bool().to(DEVICE),
            positions=positions.float(),
            task_class=task_class,
        )
        print(f"  adapter output tuple length: {len(adapter_out)}")
        mu, log_sigma = adapter_out[0], adapter_out[1]
        log("mu", mu)
        log("log_sigma", log_sigma)
        sigma_adapt = torch.exp(log_sigma)
        log("sigma_adapt=exp(log_sigma)", sigma_adapt,
            f"std_across_tokens={sigma_adapt.std().item():.6f}  (0=collapsed)")
        eps = torch.randn(mu.shape, device=DEVICE, dtype=torch.float32, generator=gen)
        log("eps", eps)
        z = (mu + sigma_adapt * eps).to(DTYPE)
        log("z = mu + sigma*eps", z)

    # v_target that DiT should have learned
    v_target = (z.float() - x0_tokens.float())
    log("v_target = z - x0", v_target)
    mse_z_x0 = (z.float() - x0_tokens.float()).pow(2).mean().item()
    print(f"\n  BASELINE MSE(z, x0)   = {mse_z_x0:.6f}  (lower bound if DiT predicts v_target perfectly)")
    print(f"  If loss_mf=0.03 then inference MSE should ≈ 0.03")
    print(f"  Correlation(z, x0)   = {(z.float() * x0_tokens.float()).sum() / (z.float().norm() * x0_tokens.float().norm()):.4f}")

    # ════════════════════════════════════════════════════════════════
    # [6] Load SigmaHead
    # ════════════════════════════════════════════════════════════════
    print("\n[6] Loading SigmaHead (latent_dim=128, hidden_dim=256)...")
    from ltx_trainer.training_strategies.vfm_strategy_v1d import SigmaHead
    sigma_head = SigmaHead(latent_dim=128, hidden_dim=256).to(DEVICE)
    sh_state = load_file(f"{CKPT_DIR}/sigma_head_step_{STEP:05d}.safetensors")
    sigma_head.load_state_dict(sh_state)
    sigma_head.eval()
    print(f"  SigmaHead params: {sum(p.numel() for p in sigma_head.parameters()):,}")

    with torch.inference_mode():
        per_token_sigma_oracle = sigma_head(mu.to(DTYPE), x0_tokens.float()).to(DTYPE)
        log("sigma_head(mu, GT_x0)", per_token_sigma_oracle,
            f"[oracle: what training saw]")
        batch_sigma_oracle = per_token_sigma_oracle.mean(dim=1)
        print(f"  batch_sigma (mean of per-token sigma): {batch_sigma_oracle.item():.5f}")

    # ════════════════════════════════════════════════════════════════
    # [7] Load transformer — TEST BOTH: with merge AND without merge
    # ════════════════════════════════════════════════════════════════
    print("\n[7] Loading transformer + LoRA...")
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.quantization import quantize_model
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from ltx_core.model.transformer.modality import Modality

    transformer = load_transformer(MODEL_PATH, device="cpu", dtype=DTYPE)
    print(f"  transformer loaded (bf16, CPU)")
    print("  Quantizing int8-quanto...")
    transformer = quantize_model(transformer, "int8-quanto", device=str(DEVICE))
    transformer = transformer.to(DEVICE)
    print(f"  GPU mem after quant: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")

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
    print(f"  LoRA: rank={lora_rank}, modules={sorted(target_modules)}")
    print(f"  LoRA state_dict keys: {list(lora_state.keys())[:4]}... ({len(lora_state)} total)")
    # Check a few LoRA norms
    for k, v in list(lora_state.items())[:4]:
        log(f"  lora/{k}", v)

    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=sorted(target_modules), lora_dropout=0.0)
    transformer = get_peft_model(transformer, lora_cfg)
    result = set_peft_model_state_dict(transformer, lora_state)
    print(f"  set_peft_model_state_dict result: {result}")

    # ── PROBE: WITHOUT merge_and_unload ──
    print("\n[7a] Testing WITHOUT merge_and_unload (LoRA as PEFT hooks)...")
    transformer.eval()

    def run_dit(sigma_val_scalar: float, ts_tensor: torch.Tensor | None, label: str) -> float:
        """Run DiT forward and return MSE vs GT."""
        s_t = torch.tensor([sigma_val_scalar], device=DEVICE, dtype=DTYPE)
        t_t = ts_tensor if ts_tensor is not None else torch.full((1, total_tokens), sigma_val_scalar, device=DEVICE, dtype=DTYPE)
        with torch.inference_mode():
            vm = Modality(enabled=True, latent=z, sigma=s_t, timesteps=t_t,
                          positions=positions, context=prompt_embeds, context_mask=prompt_mask.to(DEVICE))
            v_out, _ = transformer(video=vm, audio=None, perturbations=None)
            x_hat = z - v_out
        mse = (x_hat.float() - x0_tokens.float()).pow(2).mean().item()
        log(f"  v_pred [{label}]", v_out)
        log(f"  x_hat=z-v [{label}]", x_hat)
        print(f"  MSE(x_hat, x0) [{label}] = {mse:.6f}  (vs baseline {mse_z_x0:.6f})")
        return mse

    mse_no_merge_05 = run_dit(0.5, None, "no_merge σ=0.5")
    mse_no_merge_03 = run_dit(0.3, None, "no_merge σ=0.3")
    mse_no_merge_oracle = run_dit(batch_sigma_oracle.item(), per_token_sigma_oracle, "no_merge σ=oracle")

    # ── PROBE: WITH merge_and_unload ──
    print("\n[7b] Testing WITH merge_and_unload...")
    try:
        transformer_merged = transformer.merge_and_unload()
        transformer_merged.eval()
        print("  merge_and_unload succeeded")

        # Swap transformer reference
        original_transformer = transformer
        transformer = transformer_merged

        mse_merged_05 = run_dit(0.5, None, "merged σ=0.5")
        mse_merged_03 = run_dit(0.3, None, "merged σ=0.3")
        mse_merged_oracle = run_dit(batch_sigma_oracle.item(), per_token_sigma_oracle, "merged σ=oracle")

        transformer = original_transformer  # restore
    except Exception as e:
        print(f"  merge_and_unload FAILED: {e}")
        mse_merged_05 = mse_merged_03 = mse_merged_oracle = float("nan")

    # ── PROBE: BASE MODEL (no LoRA at all) ──
    print("\n[7c] Testing BASE MODEL without any LoRA...")
    # Re-load base transformer
    base_transformer = load_transformer(MODEL_PATH, device="cpu", dtype=DTYPE)
    base_transformer = quantize_model(base_transformer, "int8-quanto", device=str(DEVICE))
    base_transformer = base_transformer.to(DEVICE).eval()

    base_transformer_ref = transformer
    transformer = base_transformer
    mse_base_05 = run_dit(0.5, None, "base (no LoRA) σ=0.5")
    mse_base_03 = run_dit(0.3, None, "base (no LoRA) σ=0.3")
    transformer = base_transformer_ref
    del base_transformer

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Baseline MSE(z, x0):             {mse_z_x0:.6f}  ← lower bound")
    print(f"  Base model (no LoRA) σ=0.5:      {mse_base_05:.6f}")
    print(f"  Base model (no LoRA) σ=0.3:      {mse_base_03:.6f}")
    print(f"  PEFT (no merge) σ=0.5:           {mse_no_merge_05:.6f}")
    print(f"  PEFT (no merge) σ=0.3:           {mse_no_merge_03:.6f}")
    print(f"  PEFT (no merge) oracle sigma:    {mse_no_merge_oracle:.6f}")
    print(f"  Merged LoRA σ=0.5:               {mse_merged_05:.6f}")
    print(f"  Merged LoRA σ=0.3:               {mse_merged_03:.6f}")
    print(f"  Merged LoRA oracle sigma:        {mse_merged_oracle:.6f}")
    print()
    print("  INTERPRETATION:")
    if abs(mse_no_merge_05 - mse_base_05) < 0.1:
        print("  ⚠ PEFT (no merge) ≈ base → LoRA is NOT being applied via PEFT hooks!")
    elif mse_no_merge_oracle < mse_z_x0 + 0.1:
        print("  ✓ PEFT (no merge) + oracle sigma → good reconstruction!")
    else:
        print(f"  ⚠ PEFT MSE ({mse_no_merge_oracle:.4f}) >> baseline ({mse_z_x0:.4f}) → model not denoising correctly")

    if not all(v != v for v in [mse_merged_05, mse_merged_03]):
        if abs(mse_merged_05 - mse_no_merge_05) > 0.3:
            print(f"  ⚠ merge changed MSE significantly: {mse_no_merge_05:.4f} → {mse_merged_05:.4f}")

    print(f"\n  sigma_adapt collapsed? {'YES — std=0 (constant across tokens)' if sigma_adapt.std() < 1e-4 else 'No'}")
    print(f"  sigma_adapt mean: {sigma_adapt.mean().item():.4f}")
    print(f"  SigmaHead oracle mean: {per_token_sigma_oracle.mean().item():.4f}")
    print(f"  Training used: per_token_sigma=True, z=pure_noise_no_interp, v_target=z-x0")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
