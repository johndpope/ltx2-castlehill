# DDiT Implementation Guide

> Paper: [Dynamic Diffusion Transformer (arXiv:2602.16968)](https://arxiv.org/abs/2602.16968)

## Paper Summary

DDiT accelerates DiT inference by **dynamically changing spatial patch resolution** per denoising step. Early steps (noisy, coarse structure) use large patches (fewer tokens), late steps (fine detail) use small patches (more tokens). The "dynamic" part comes from a **3rd-order finite difference** analysis of the denoising trajectory — the scheduler picks the coarsest resolution where spatial variance stays below a threshold.

**Paper targets:** FLUX-1.Dev (T2I) and Wan-2.1 1.3B (T2V).
**Our target:** LTX-2 19B with SCD (T2V / I2V / T2I).

## Architecture: Paper vs Our Implementation

### Core Token Reduction

| | Paper | Ours | Notes |
|---|---|---|---|
| **Mechanism** | Change patchification kernel (p→2p→4p) | Merge 2×2 post-patchified tokens | Mathematically equivalent |
| **Token reduction** | Quadratic: N/(s²) | Same: N/(s²) | s=2 → 4× fewer tokens |
| **Supported scales** | {p, 2p, 4p} → {1, 2, 4} | {1, 2, 4} | Match |
| **Per-scale patchify** | `w^{emb}_{p_new} ∈ ℝ^{p²C × d}` | `Linear(C*s*s, inner_dim)` | Match |
| **Patch-size ID** | Learnable d-dim vector | `nn.Parameter(zeros(1,1,inner_dim))` | Match |
| **Position adjust** | Bilinear interpolation of learned PE | Avg-pool (3D) / min-max-pool (4D RoPE bounds) | Adapted for LTX-2 RoPE |
| **Residual** | Single residual block pre→post | `LayerNorm→Linear→GELU→Linear`, weight=0.1 | Match |

### Scheduling

| | Paper | Ours (Current) | Gap |
|---|---|---|---|
| **Schedule type** | **Dynamic** (3rd-order Δ³) | **Fixed** (head=2, tail=3) | **CRITICAL GAP** |
| **Threshold** | τ=0.001, ρ=0.4 | Implemented in DDiTPatchScheduler but **unused at inference** | Code exists, not wired |
| **Warmup** | 3 steps native | 2 steps native (via head=2) | Close |
| **Tail** | Adaptive (auto) | Fixed 3 steps native | Paper is adaptive |

### Training

| | Paper | Ours | Notes |
|---|---|---|---|
| **Loss** | MSE distillation | Sigma-weighted MSE + cosine | Ours is enhanced |
| **LoRA targets** | FFN layers (`net.0.proj, net.2`) | Attention (`to_q,k,v,out`) | **Different targets** |
| **LoRA rank** | 32 | 16 (SCD decoder) | Lower rank |
| **Optimizer** | AdamW lr=1e-4 (T2V) | AdamW lr=1e-4 | Match |
| **Sigma curriculum** | Not mentioned | Cosine ramp [0.3→0.9] | Our enhancement |
| **Two-phase** | Not mentioned | Phase 1 (recon) → Phase 2 (distill) | Our enhancement |

## Critical Gaps to Fix

### 1. Dynamic Scheduling at Inference (HIGHEST PRIORITY)

The paper's key contribution is the **adaptive per-step scheduling** via `DDiTPatchScheduler`. We have the scheduler implemented in `ddit.py` but `scd_inference.py` uses a hardcoded head/tail schedule instead. This means:

- We waste compute on steps where scale=4 would suffice (early denoising)
- We use merged resolution on steps that need native (late denoising with spatial variance)
- The schedule is **per-prompt** in the paper — different prompts get different schedules

**Fix:** Wire `DDiTPatchScheduler` into `scd_inference.py`'s denoising loop.

### 2. DDiT Must Work for ALL LTX-2 Modalities

Currently DDiT only works via `scd_inference.py` (SCD decoder). It should also work for:

| Mode | Pipeline | DDiT Application Point |
|---|---|---|
| **SCD T2V** | Autoregressive encoder→decoder | Decoder blocks only (current) |
| **Standard T2V** | Full 48-block transformer | All blocks (needs `train_ddit_adapter.py`) |
| **I2V** | Full transformer + first-frame conditioning | All blocks |
| **T2I** | Full transformer (1 frame) | All blocks |
| **I2I** | Full transformer (1 frame) | All blocks |

**Key insight from the paper:** DDiT benefit scales with token count. Standard T2V processes 97 frames × 336 tokens = 32,592 tokens at once. Scale=2 reduces to 8,148 tokens → **16× less attention compute for ALL frames simultaneously**. This is far more impactful than SCD's single-frame decoder (336→84 tokens).

### 3. LoRA Targets Should Match Paper

Paper uses **FFN layers** (`net.0.proj`, `net.2`) which handle spatial mixing. Our SCD LoRA uses **attention** (`to_q,k,v,out`). For the standard (non-SCD) adapter, the paper's targets should be used.

## File Inventory

### Implementation Files

| File | Location | Purpose |
|---|---|---|
| `ddit.py` | `ltx-core/.../transformer/ddit.py` | DDiTAdapter, DDiTMergeLayer, DDiTPatchScheduler, DDiTConfig |
| `scd_inference.py` | `ltx-trainer/scripts/scd_inference.py` | SCD inference with DDiT (fixed schedule) |
| `train_ddit_scd.py` | `sparse-causal-diffusion/scripts/train_ddit_scd.py` | SCD-specific DDiT distillation training |
| `train_ddit_adapter.py` | `sparse-causal-diffusion/scripts/train_ddit_adapter.py` | Full 48-block DDiT distillation training |

### Pre-trained Checkpoints

| File | Location | Description |
|---|---|---|
| `ddit_scd_adapter_final.safetensors` | `sparse-causal-diffusion/outputs/ddit_scd_v2/` | 4.2M params, scale=2, SCD decoder only |
| `ddit_scd_lora_final.safetensors` | `sparse-causal-diffusion/outputs/ddit_scd_v2/` | 8.4M params, rank=16, decoder attention LoRA |
| `ddit_scd_config.json` | `sparse-causal-diffusion/outputs/ddit_scd_v2/` | Config: scales=[2], residual=0.1, cosine_loss=0.5 |

### Training Data Flow

```
SCD DDiT Training (train_ddit_scd.py):
======================================
1. Load full LTX-2 → wrap as SCD (32 enc + 16 dec)
2. Pre-compute encoder features (run encoder ONCE per sample, cache to CPU)
3. Offload encoder blocks to CPU (frees ~24GB)
4. Create DDiT adapter + decoder LoRA
5. Training loop:
   a. Sample sigma from curriculum [0.05, 0.3→0.9]
   b. Add noise to ground truth latent
   c. Teacher: forward_decoder at native res (no_grad)
   d. Student: merge → decoder at coarse res → unmerge
   e. Loss = sigma_weighted_MSE + cosine_similarity
   f. Backprop through adapter + decoder LoRA only

Full DDiT Training (train_ddit_adapter.py):
============================================
1. Load full LTX-2 (all 48 blocks)
2. Apply PEFT LoRA (rank=32, FFN+attention targets)
3. Training loop:
   a. Sample sigma uniform [0.05, 0.95]
   b. Teacher: full model at native res (no_grad)
   c. Student: merge → full model at coarse res → unmerge
   d. Loss = MSE distillation
   e. Backprop through LoRA + adapter
```

## DDiTPatchScheduler Algorithm

```python
# Per-step algorithm (from ddit.py):
def compute_schedule(z, step_idx, num_frames, height, width):
    # 1. Record current latent
    history.append(z.detach())

    # 2. Warmup: first 3 steps always native (need history)
    if step_idx < warmup_steps:
        return 1

    # 3. Third-order finite difference (acceleration of denoising)
    z0, z1, z2 = history[-3], history[-2], history[-1]
    delta3 = (z0 - z1) - (z1 - z2)  # Simplified 3rd-order

    # 4. Per-patch spatial variance
    for scale in sorted(supported_scales, reverse=True):  # Try 4, then 2
        # Reshape delta3 to spatial grid, pool to scale
        patches = reshape_to_patches(delta3, scale)
        per_patch_std = patches.std(dim=-1)

        # ρ-percentile aggregation (40th percentile)
        threshold_val = torch.quantile(per_patch_std, percentile)

        if threshold_val < threshold:  # 0.001
            return scale  # This scale is safe

    return 1  # Default to native
```

## Performance Benchmarks

### Current Results (Fixed Schedule)

| Config | Quant | Decoder Speedup | s/frame | Notes |
|---|---|---|---|---|
| SCD baseline | int8-quanto | 1.0× | 1.7 | Single GPU (RTX 5090) |
| SCD + DDiT 2× | int8-quanto | 1.25× | 1.4 | Memory-bandwidth bound |
| SCD baseline | bf16 (split) | 1.0× | 2.7 | Decoder on PRO 4000 |
| SCD + DDiT 2× | bf16 (split) | 1.48× | 1.9 | Compute-bound ✓ |

### Expected with Dynamic Schedule

With dynamic scheduling, DDiT would use scale=4 on early steps (even fewer tokens) and scale=1 on late steps only when needed. Expected improvement over fixed schedule:
- Early steps (σ > 0.5): Could use scale=4 → 16× fewer tokens
- Middle steps: scale=2 → 4× fewer tokens
- Late steps (σ < 0.1): scale=1 → native resolution

The paper reports **2.1× standalone speedup** on Wan-2.1 1.3B, **3.2× with TeaCache**.

### Theoretical Maximum (Standard T2V, not SCD)

For standard (non-SCD) inference with all 97 frames processed at once:
- Native tokens: 97 frames × 336 = 32,592 tokens
- Scale=2: 97 × 84 = 8,148 tokens → **16× less attention**
- Scale=4: 97 × 21 = 2,037 tokens → **256× less attention**

This is where DDiT's real power lies — standard T2V, not SCD's single-frame decoder.

## Integration Checklist

When implementing DDiT for any LTX-2 inference mode:

- [ ] Use `DDiTPatchScheduler` for dynamic scale selection (NOT fixed head/tail)
- [ ] Call `scheduler.reset()` at start of each denoising sequence
- [ ] Call `scheduler.record(z)` after each step
- [ ] Support multiple scales in a single run (scale may change per step)
- [ ] Handle both SCD (decoder only) and standard (full model) paths
- [ ] Test with at least 30 denoising steps (scheduler needs 3+ for warmup)
- [ ] Verify output quality matches baseline (FID, CLIP-score)
- [ ] Profile memory: adapter adds only ~25MB, should not cause OOM
