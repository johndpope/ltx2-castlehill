# SCD (Separable Causal Diffusion) — Achievement Summary

## What is SCD?

SCD splits LTX-2's monolithic 48-layer DiT transformer into a **32-layer encoder** (runs once per frame, KV-cached) and a **16-layer decoder** (runs N denoising steps per frame). This enables **autoregressive video generation** — each frame is generated conditioned on all previous frames, enabling arbitrarily long videos (30s, 60s, 120s+) on consumer GPUs.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STANDARD LTX-2 (Parallel)                        │
│  All frames denoised simultaneously through 48 layers × N steps     │
│  Memory: O(frames × tokens) — limited to ~6s on 32GB               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    SCD LTX-2 (Autoregressive)                       │
│                                                                     │
│  ENCODER (32 layers) ───▶ KV-cache (runs ONCE per frame, σ=0)     │
│       ↓                                                             │
│  DECODER (16 layers) ───▶ N denoising steps per frame              │
│                                                                     │
│  Memory: O(1 frame) — generates unlimited duration                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Components

### Core Model (`ltx-core`)

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| **LTXSCDModel** | `ltx-core/.../transformer/scd_model.py` | 1,094 | Production |
| **DDiTAdapter** | `ltx-core/.../transformer/ddit.py` | 761 | Production |

**LTXSCDModel** wraps the standard `LTXModel` and provides:
- `forward_encoder()` — 32-layer encoder with KV-cache accumulation
- `forward_decoder()` — 16-layer decoder for multi-frame or per-frame denoising
- `forward_decoder_per_frame()` — Single-frame decoder (matches inference behavior)
- `_combine_encoder_decoder()` — `token_concat` (prepend encoder features as prefix tokens)
- KV-cache management for O(1) encoder cost per new frame

### Training Strategy

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| **SCDTrainingStrategy** | `ltx-trainer/.../scd_strategy.py` | ~800 | Production |

Key features:
- **Per-frame decoder training** — eliminates train/inference mismatch (grid artifacts)
- **Token concat alignment** — decoder self-attends to encoder prefix + noisy tokens
- **Scheduled sampling** — curriculum from teacher forcing → autoregressive exposure
- **Clean context ratio** — paper's 10% clean frames for AR robustness
- **First-frame conditioning** — 50% chance of clean first frame as context

### Inference Pipeline

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| **SCD Inference** | `ltx-trainer/scripts/scd_inference.py` | 1,747 | Production |

Capabilities:
- Autoregressive chunk-based generation (4 latent frames/chunk, 1-frame overlap)
- **DDiT dynamic patch scheduling** — 1.27x decoder speedup
- **Split-GPU mode** — encoder→cuda:0, decoder→cuda:1
- **Distilled model support** — 8 inference steps (vs 30 standard)
- **BezierFlow/BSplineFlow** custom sigma schedules
- **CFG** (classifier-free guidance) with null embeddings
- int8-quanto, fp8-quanto, or bf16 precision modes
- Live text prompts (loads/unloads Gemma) or cached embeddings

### Scheduler Research

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| **BezierScheduler** | `ltx-trainer/.../bezierflow/scheduler.py` | 139 | Production |
| **BSplineScheduler** | `ltx-trainer/.../bsplineflow/scheduler.py` | 259 | Research |
| **Training script** | `ltx-trainer/scripts/train_bezierflow.py` | ~500 | Production |

Both schedulers learn monotonic sigma schedules (32 parameters, ~10 min training). BSplineFlow offers local control (B-spline basis) vs BezierFlow's global control (Bernstein basis).

### Evolution System (Gradient-Free AR Optimization)

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| **Engine** | `ltx-trainer/.../evolution/engine.py` | 1,336 | Production |
| **Fitness** | `ltx-trainer/.../evolution/fitness.py` | 560 | Production |
| **Perturbation** | `ltx-trainer/.../evolution/perturbation.py` | 365 | Production |
| **Evolve script** | `ltx-trainer/scripts/evolve_scd.py` | 331 | Production |

EggRoll-style evolution that optimizes LoRA weights for autoregressive rollout quality using antithetic perturbation pairs. Complements gradient training — gradient descent learns reconstruction, evolution refines AR stability.

---

## Training Runs & Checkpoints

### Completed Training Runs

| Run | Dataset | Steps | Loss | Combine | Key Finding |
|-----|---------|-------|------|---------|-------------|
| `scd_ditto_subset` | Ditto-1M (500 pairs) | 2,000 | 0.386→0.231 | add | First working SCD LoRA |
| `scd_isometric` | Grok videos (122) | 3,000 | — | token_concat | Isometric-domain specialization |
| `scd_isometric_v2` | Grok videos (122) | 3,000 | — | token_concat | Rank-64 LoRA, larger model |
| `scd_distilled_*` | Various | 2,000 | — | add/concat | Distilled 8-step models |
| **`scd_token_concat`** | **Ditto-1M (500)** | **2,000** | **converging** | **token_concat** | **Current best — Muon + warmup** |

### Current Run (In Progress)
- **Config**: `ltx2_scd_token_concat.yaml`
- **Optimizer**: Muon lr=0.005 + 100-step linear warmup
- **Combine mode**: token_concat (paper's best, Table 3)
- **Decoder**: per_frame_decoder=true (672 tokens/frame)
- **Model**: ltx-2-19b-distilled + int8-quanto
- **Data**: 500 Ditto-1M pairs
- **Status**: Step 1839/2000, predictions closely matching targets

### Checkpoint Inventory

```
/media/2TB/omnitransfer/output/
├── scd_ditto_subset/checkpoints/        # 2000 steps, rank-32, 817MB each
├── scd_isometric/checkpoints/           # 3000 steps, rank-64, 1.6GB each
├── scd_isometric_v2/checkpoints/        # 3000 steps, rank-64
├── scd_distilled_*/checkpoints/         # Various distilled runs
├── scd_token_concat/checkpoints/        # Current: 1800 steps saved (ongoing)
│   ├── lora_weights_step_00800.safetensors
│   ├── lora_weights_step_01000.safetensors
│   ├── lora_weights_step_01200.safetensors
│   ├── lora_weights_step_01400.safetensors
│   ├── lora_weights_step_01600.safetensors
│   └── lora_weights_step_01800.safetensors
├── scd_muon/checkpoints/                # Early Muon experiments
├── scd_muon_v2/checkpoints/             # Muon with warmup
├── scd_evolution_distilled/             # 125 generations evolved
└── scd_evolution_cfg/                   # 25 generations with CFG
```

---

## Critical Technical Discoveries

### 1. token_concat > add (Non-Negotiable)

The SCD paper (Table 3) ablation confirmed: `token_concat` significantly outperforms `add` mode.

- **add mode**: Raw element-wise addition of encoder features to decoder tokens. After 32 transformer blocks with AdaLN modulation, encoder features have completely different distribution from freshly patchified decoder tokens. **No learned alignment → produces mush at inference.**
- **token_concat**: Prepends encoder features as prefix tokens. Decoder self-attention learns to attend to both encoder context and noisy tokens. Rich, learned interaction.

With `per_frame_decoder=true`, token_concat only doubles per-frame tokens (336→672) — fits easily on 32GB.

### 2. Per-Frame Decoder Eliminates Grid Artifacts

**Root cause of grid artifacts**: Training decoder sees ALL 4 frames (1344 tokens, cross-frame attention) but inference decoder sees 1 frame (336 tokens). LoRA adapts to multi-frame attention patterns that collapse on single-frame input.

**Diagnostic evidence**:
- Base 48-block (no SCD): v_std converges 1.09→0.81, coherent output
- SCD 4-frame decoder: v_std 1.49→0.83 (improving), poor but better
- **SCD 1-frame decoder: v_std 1.88→1.82 (constant! broken), grid artifacts**

Fix: `per_frame_decoder: true` — trains decoder on 1 frame per forward pass, exactly matching inference.

### 3. Muon Optimizer Requires Careful Tuning for SCD

Muon (Newton-Schulz orthogonalization) converges 1.3-2x faster than AdamW, but:
- **lr=0.02** (standard Muon for LoRA): **Produced pure noise** with token_concat's doubled sequence
- **lr=0.005 + 100-step warmup**: Stable convergence, clear structure by step 58

The doubled sequence length (672 vs 336 tokens) changes the loss landscape — Newton-Schulz estimates need calibration time before aggressive updates.

### 4. Scheduled Sampling Bridges Train-Test Gap

Teacher forcing (clean encoder input) produces good training loss but poor autoregressive inference. Scheduled sampling curriculum:
- Steps 0-50: Pure teacher forcing (warm up)
- Steps 50-350: Linear ramp from 0% → 50% AR exposure
- Steps 350+: 50% AR probability (steady state)
- 5% noise augmentation on AR frames

---

## Performance Benchmarks

### Inference Speed (RTX 5090, 768x448, int8-quanto)

| Duration | Method | Time | s/frame | Decoder Speedup |
|----------|--------|------|---------|-----------------|
| 30s (76 frames) | SCD baseline | 2.9 min | 1.7 | 1.0x |
| 30s (76 frames) | SCD + DDiT 2x (dynamic) | **2.5 min** | **1.3** | **1.27x** |
| 60s (151 frames) | SCD baseline | 5.2 min | 1.7 | 1.0x |
| 60s (151 frames) | SCD + DDiT 2x | 4.4 min | 1.4 | 1.25x |
| 120s (301 frames) | SCD baseline | 9.9 min | 1.7 | 1.0x |
| 120s (301 frames) | SCD + DDiT 2x | **8.3 min** | **1.3** | **1.27x** |

### Split-GPU (bf16, encoder→5090, decoder→PRO 4000)

| Duration | Method | Time | s/frame | DDiT Speedup |
|----------|--------|------|---------|--------------|
| 30s | Split baseline | 4.3 min | 2.7 | 1.0x |
| 30s | Split + DDiT 2x | 3.2 min | 1.9 | **1.48x** |

Key insight: DDiT speedup is **1.48x in bf16** (compute-bound) vs **1.25x in int8-quanto** (memory-bound). This confirms DDiT's token reduction has real impact when not bottlenecked on weight dequantization.

### Scaling Characteristics

- **Encoder O(1) via KV-cache**: ~14.5s per 150 frames (~6% of total time)
- **Decoder scales linearly**: ~1.3-1.7s per frame
- **DDiT speedup improves at scale**: 1.12x total at 30s → 1.19x total at 120s
- **Dynamic scheduler adapts per-prompt**: 90% merged steps vs fixed 83%

---

## Full Pipeline Vision

```
SCD Base LoRA ──► TeaCache ──► Distilled Model ──► BezierFlow Schedule
   (quality)      (cache)      (fewer steps)        (optimal σ curve)

                     Stacking speedups:
  Baseline:          48 layers × 30 steps × 336 tokens = 1.7 s/frame
  SCD:               16 layers × 30 steps × 672 tokens = 1.7 s/frame (same!)
  + DDiT:            16 layers × 30 steps × 168 tokens = 1.3 s/frame (1.27x)
  + Distilled (8s):  16 layers ×  8 steps × 672 tokens = ~0.5 s/frame (est 3.4x)
  + BezierFlow:      16 layers ×  8 steps × optimal σ  = ~0.4 s/frame (est 4.2x)
```

---

## Training Configurations (15 Total)

All configs live in `ltx-trainer/configs/`:

| Config | Strategy | Data | Optimizer | Steps | Key Feature |
|--------|----------|------|-----------|-------|-------------|
| `ltx2_scd_ditto.yaml` | SCD, add | Ditto 500 | AdamW | 2000 | First working run |
| `ltx2_scd_isometric.yaml` | SCD | Isometric 122 | Muon | 3000 | Domain-specific |
| `ltx2_scd_token_concat.yaml` | SCD, token_concat | Ditto 500 | Muon | 2000 | **Current best** |
| `ltx2_scd_token_concat_v2.yaml` | SCD, token_concat | Ditto 500 | Muon | 5000 | Extended (planned) |
| `ltx2_scd_evolution.yaml` | Evolution | Various | ES | 200 gen | Gradient-free AR |
| `ltx2_scd_evolution_v2.yaml` | Evolution | Merged | ES | 200 gen | With CFG |
| `ltx2_scd_evolution_distilled.yaml` | Evolution | Merged | ES | 200 gen | Distilled model |
| Plus 8 more... | | | | | |

---

## Lines of Code

| Category | Files | Lines (approx) |
|----------|-------|----------------|
| SCD Model | 1 | 1,094 |
| SCD Strategy | 1 | 800 |
| SCD Inference | 1 | 1,747 |
| DDiT Adapter | 1 | 761 |
| Evolution System | 4 | 2,592 |
| BezierFlow | 2 | 639 |
| BSplineFlow | 2 | 272 |
| Pipeline Script | 1 | ~500 |
| Configs | 15 | ~2,250 |
| **Total** | **28** | **~10,655** |

---

## What's Next

1. **Validate current 2000-step checkpoint** with autoregressive inference
2. **Extended training (v2)** with paper-aligned settings:
   - `clean_context_ratio: 0.1` (10% clean encoder input for AR robustness)
   - `decoder_lr_ratio: 2.0` (stronger signal to 16-layer decoder)
   - 5000 steps (2.5x current)
3. **TeaCache integration** for decoder feature caching
4. **BezierFlow vs BSplineFlow comparison** — head-to-head scheduler training
5. **Full pipeline benchmark** — stacked speedups end-to-end
