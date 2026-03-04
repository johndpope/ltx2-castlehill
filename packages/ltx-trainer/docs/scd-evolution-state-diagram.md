# SCD + Evolution + PEFT Bypass — State Diagram

## Overview

Two-phase pipeline for long-form autoregressive video quality optimization.

## Phase 1: Gradient-Based SCD LoRA Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: GRADIENT TRAINING                          │
│                    (trainer.py + scd_strategy.py)                      │
│                                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐ │
│  │ Load     │    │ Quantize │    │ Apply    │    │ Wrap with        │ │
│  │ bf16     │───▶│ (quanto) │───▶│ PEFT     │───▶│ LTXSCDModel      │ │
│  │ weights  │    │ nn.Linear│    │ LoRA     │    │ (32 enc + 16 dec)│ │
│  │ (CPU)    │    │ → QLinear│    │ on QLinr │    │                  │ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘ │
│                                       ▲                    │           │
│                                       │                    ▼           │
│                               ┌───────┴───────┐  ┌─────────────────┐ │
│                               │ PEFT creates  │  │ Per-frame decoder│ │
│                               │ LoRA adapters │  │ training loop    │ │
│                               │ on QLinear    │  │ (teacher forcing)│ │
│                               │ modules       │  │                  │ │
│                               │               │  │ Loss → backprop  │ │
│                               │ PEFT v0.14+   │  │ through LoRA     │ │
│                               │ CAN wrap      │  │ params only      │ │
│                               │ QLinear       │  └────────┬────────┘ │
│                               └───────────────┘           │           │
│                                                            ▼           │
│                                                   ┌─────────────────┐ │
│                                                   │ Save LoRA       │ │
│                                                   │ checkpoint      │ │
│                                                   │ (.safetensors)  │ │
│                                                   └────────┬────────┘ │
└────────────────────────────────────────────────────────────┼──────────┘
                                                              │
                              LoRA checkpoint                 │
                              (PEFT format)                   │
                                                              ▼
```

## Phase 2: Evolution (engine.py) — Gradient-Free AR Quality Fine-Tuning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: EVOLUTION (engine.py)                       │
│                    Gradient-free AR quality fine-tuning                 │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MODEL LOADING — PEFT BYPASS via ManualLoRA                     │   │
│  │                                                                  │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │   │
│  │  │ 1. Load  │    │ 2. Quant │    │ 3. Wrap  │    │ 4. Inject│  │   │
│  │  │ bf16     │───▶│ (quanto) │───▶│ SCD      │───▶│ Manual   │  │   │
│  │  │ weights  │    │ int8     │    │ Model    │    │ LoRA     │  │   │
│  │  │ (CPU)    │    │ QLinear  │    │          │    │ wrappers │  │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │   │
│  │                                                       │         │   │
│  │  WHY NOT PEFT?                                        ▼         │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ Circular dependency:                                     │   │   │
│  │  │                                                          │   │   │
│  │  │  Approach A: Quantize first, then PEFT                   │   │   │
│  │  │    nn.Linear → QLinear → PEFT(QLinear) = CRASH           │   │   │
│  │  │    PEFT can't determine in/out features of QLinear       │   │   │
│  │  │                                                          │   │   │
│  │  │  Approach B: PEFT first, then quantize                   │   │   │
│  │  │    nn.Linear → LoraLayer{base_layer, lora_A, lora_B}    │   │   │
│  │  │    quanto quantize(block) → finds lora_A (nn.Linear)    │   │   │
│  │  │    → QUANTIZES LoRA PARAMS! = BROKEN                     │   │   │
│  │  │    (evolution needs bf16 LoRA params for perturbation)   │   │   │
│  │  │                                                          │   │   │
│  │  │  Solution: ManualLoRA (no PEFT)                          │   │   │
│  │  │    nn.Linear → QLinear (quanto, clean)                   │   │   │
│  │  │    QLinear → ManualLoRA{base=QLinear, lora_A, lora_B}    │   │   │
│  │  │    ManualLoRA.forward = base(x) + B(A(x)) * scaling     │   │   │
│  │  │    LoRA params stay bf16, base stays quantized           │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BASELINE EVALUATION + FITNESS NORMALIZATION                    │   │
│  │                                                                  │   │
│  │  1. Run AR rollout on random sample (no perturbation)           │   │
│  │  2. Record raw component values:                                 │   │
│  │     fm=5.18, recon=1.94, tcoh=0.81, lpips=0.79, ssim=0.19      │   │
│  │  3. Compute normalization scales:                                │   │
│  │     fm_scale = 1/5.18 = 0.193                                   │   │
│  │     recon_scale = 1/1.94 = 0.515                                │   │
│  │     tcoh_scale = 1/0.81 = 1.235                                 │   │
│  │     lpips_scale = 1/0.79 = 1.266                                │   │
│  │     ssim_scale = 1/0.19 = 5.263                                 │   │
│  │  4. After normalization: baseline total ≈ -0.90                  │   │
│  │     (each component contributes ~1.0 × its weight)              │   │
│  │  5. Scales saved in checkpoint for resumption                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  EVOLUTION LOOP (ES gradient-free optimization)                 │   │
│  │                                                                  │   │
│  │  for generation in range(300):                                   │   │
│  │                                                                  │   │
│  │    ┌─────────────────────────────────────────────────────────┐   │   │
│  │    │  for seed in antithetic_pairs(population_size=4):       │   │   │
│  │    │                                                         │   │   │
│  │    │  ┌─────────┐   ┌──────────────────┐   ┌────────────┐  │   │   │
│  │    │  │ +e pert │──▶│ AR Rollout       │──▶│ Fitness+   │  │   │   │
│  │    │  │ (seed,  │   │ (2 frames, each  │   │ (normalized│  │   │   │
│  │    │  │  +1)    │   │  denoised 8 steps│   │  latent +  │  │   │   │
│  │    │  │ CACHE   │   │  × 2 eval batch) │   │  pixel     │  │   │   │
│  │    │  │ NOISE   │   └──────────────────┘   │  metrics)  │  │   │   │
│  │    │  └─────────┘                          └────────────┘  │   │   │
│  │    │       │                                      │         │   │   │
│  │    │       ▼                                      │         │   │   │
│  │    │  ┌─────────┐                                 │         │   │   │
│  │    │  │ Revert  │◀────────────────────────────────┘         │   │   │
│  │    │  │ to orig │                                           │   │   │
│  │    │  └─────────┘                                           │   │   │
│  │    │       │                                                │   │   │
│  │    │       ▼                                                │   │   │
│  │    │  ┌─────────┐   ┌──────────────────┐   ┌────────────┐ │   │   │
│  │    │  │ -e pert │──▶│ AR Rollout       │──▶│ Fitness-   │ │   │   │
│  │    │  │ (seed,  │   │ (same samples)   │   │            │ │   │   │
│  │    │  │  -1)    │   │                  │   │            │ │   │   │
│  │    │  │ REUSE   │   └──────────────────┘   └────────────┘ │   │   │
│  │    │  │ CACHED  │                                  │       │   │   │
│  │    │  │ NOISE   │                                  │       │   │   │
│  │    │  └─────────┘                                  │       │   │   │
│  │    │       │                                       │       │   │   │
│  │    │       ▼                                       │       │   │   │
│  │    │  ┌─────────┐                                  │       │   │   │
│  │    │  │ Revert  │◀─────────────────────────────────┘       │   │   │
│  │    │  └─────────┘                                          │   │   │
│  │    │       │                                               │   │   │
│  │    │       ▼                                               │   │   │
│  │    │  diff[seed] = Fitness+ - Fitness-                     │   │   │
│  │    └───────────────────────────────────────────────────────┘   │   │
│  │                                                                │   │
│  │    ┌───────────────────────────────────────────────────────┐   │   │
│  │    │  ES Gradient Update (uses CACHED noise, no regen):   │   │   │
│  │    │  w += (lr / N) * Sum (diff[seed] * e[seed]) / (2s)  │   │   │
│  │    │                                                       │   │   │
│  │    │  Noise annealing: s *= 0.998                          │   │   │
│  │    │  Clear noise cache after update                       │   │   │
│  │    └───────────────────────────────────────────────────────┘   │   │
│  │                                                                │   │
│  │    ┌───────────────────────────────────────────────────────┐   │   │
│  │    │  Every 25 generations:                                │   │   │
│  │    │    - Save evolved params + state checkpoint           │   │   │
│  │    │    - Save full LoRA (PEFT-compatible format)          │   │   │
│  │    │    - Log GT vs Prediction images to W&B               │   │   │
│  │    │      (VAE decode on cuda:1, side-by-side comparison)  │   │   │
│  │    └───────────────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  OUTPUT:                                                               │
│    evolved_lora.safetensors (PEFT-compatible format for inference)     │
└────────────────────────────────────────────────────────────────────────┘
```

## AR Rollout Detail (fitness.py)

```
Frame 0                        Frame 1
┌──────────────────┐           ┌──────────────────┐
│ ENCODER (32 blk) │           │ ENCODER (32 blk) │
│                  │           │                  │
│ input: zeros     │           │ input: gen[0]    │
│ (sigma=0)        │           │ (sigma=0, AR)    │
└────────┬─────────┘           └────────┬─────────┘
         │ enc_feat[0]                  │ enc_feat[1]
         │                              │
         │  shift-by-1                  │
         ▼                              ▼
┌──────────────────┐           ┌──────────────────┐
│ DECODER (16 blk) │           │ DECODER (16 blk) │
│                  │           │                  │
│ context: zeros   │           │ context: enc[0]  │
│                  │           │                  │
│ denoise:         │           │ denoise:         │
│ 8 steps (distil) │           │ 8 steps (distil) │
└────────┬─────────┘           └────────┬─────────┘
         │ gen[0]                       │ gen[1]
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                NORMALIZED FITNESS EVALUATION                        │
│                                                                     │
│  Each component scaled so baseline ≈ 1.0:                          │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ FM Velocity │  │ Latent      │  │ Temporal    │  │ Pixel    │  │
│  │ MSE (0.35)  │  │ Recon MSE   │  │ Coherence  │  │ LPIPS    │  │
│  │ ×fm_scale   │  │ (0.25)      │  │ Gap (0.15) │  │ (0.20)   │  │
│  │             │  │ ×recon_scale│  │ ×tcoh_scale│  │ ×lpips_  │  │
│  │ ~1.0 at     │  │ ~1.0 at     │  │ ~1.0 at    │  │  scale   │  │
│  │ baseline    │  │ baseline    │  │ baseline   │  │ cuda:1   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  └────┬─────┘  │
│         └────────────────┴───────────────┴───────────────┘        │
│                              │                                     │
│                              ▼                                     │
│  total = -Sum(weight_i × metric_i × scale_i) + w_ssim × ssim     │
│                                                                     │
│  Normalized baseline ≈ -0.90  (interpretable, stable ES gradient) │
│  (vs raw baseline ≈ -2.57 where fm dominates everything)          │
└─────────────────────────────────────────────────────────────────────┘
```

## Dual-GPU Memory Layout

```
┌─────────────────────────────────────┐  ┌──────────────────────────────┐
│         cuda:0 (RTX 5090 32GB)      │  │   cuda:1 (PRO 4000 24GB)    │
│                                     │  │                              │
│  ┌───────────────────────────────┐  │  │  ┌────────────────────────┐  │
│  │ LTX-2 Transformer (int8)     │  │  │  │ VAE Decoder (bf16)     │  │
│  │ ~14GB quantized              │  │  │  │ ~8GB                   │  │
│  │                              │  │  │  │                        │  │
│  │ ┌──────────────────────────┐ │  │  │  │ Decode latent→pixel    │  │
│  │ │ Encoder (blocks 0-31)   │ │  │  │  │ for LPIPS + SSIM +     │  │
│  │ │ QLinear base weights    │ │  │  │  │ W&B recon images       │  │
│  │ │ + ManualLoRA (bf16)     │ │  │  │  └────────────────────────┘  │
│  │ │ (loaded but NOT evolved)│ │  │  │                              │
│  │ └──────────────────────────┘ │  │  │  ┌────────────────────────┐  │
│  │ ┌──────────────────────────┐ │  │  │  │ LPIPS (alex) ~0.2GB   │  │
│  │ │ Decoder (blocks 32-47)  │ │  │  │  └────────────────────────┘  │
│  │ │ QLinear base weights    │ │  │  │                              │
│  │ │ + ManualLoRA (bf16)     │ │  │  │  Free: ~16GB                 │
│  │ │ ** EVOLVED by ES **     │ │  │  │                              │
│  │ └──────────────────────────┘ │  │  └──────────────────────────────┘
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Evolution overhead ~2GB      │  │
│  │ (noise cache, samples)       │  │
│  └───────────────────────────────┘  │
│                                     │
│  Free: ~16GB                        │
└─────────────────────────────────────┘
```

## ManualLoRA Module Structure

```
Before ManualLoRA injection:
  transformer_blocks[32].attn.to_q = QLinear(4096, 4096)  # int8 quantized

After ManualLoRA injection:
  transformer_blocks[32].attn.to_q = ManualLoRA(
      base = QLinear(4096, 4096),    # int8 quantized (frozen)
      lora_A = Parameter[32, 4096],  # bf16 (evolved)
      lora_B = Parameter[4096, 32],  # bf16 (evolved)
      scaling = 1.0                  # alpha/rank = 32/32
  )

  forward(x):
      return base(x) + linear(linear(x, lora_A), lora_B) * scaling
      |                 |_______ bf16 LoRA path ________|
      |_ int8 quantized path
```

## V1 vs V2 Performance Comparison

```
┌────────────────────────────────────────────────────────────────────┐
│                    V1 (original)           V2 (optimized)          │
│                                                                    │
│  Population:       8 pairs                 4 pairs                 │
│  Eval batch:       4 samples/pert          2 samples/pert          │
│  AR frames:        4                       2                       │
│  Denoising steps:  8                       8                       │
│                                                                    │
│  Forward passes    8×2×4×4×8 = 2048       4×2×2×2×8 = 256         │
│  per generation:                                                   │
│                                                                    │
│  Time/generation:  ~137s                   ~18s (7.6× faster)      │
│  Total (300 gen):  ~11.5 hours             ~1.5 hours              │
│                                                                    │
│  Fitness scale:    Raw (-2.57 baseline)    Normalized (-0.90)      │
│  Noise caching:    No (regen in update)    Yes (cache +e, reuse)   │
│  W&B images:       No                      Yes (GT vs Pred)        │
│  Checkpoint:       Params + state          + normalization scales   │
└────────────────────────────────────────────────────────────────────┘
```

## Key Insight: Why Evolution Needs Its Own LoRA Strategy

The SCD trainer (Phase 1) uses PEFT + quanto successfully because:
- Modern PEFT (v0.14+) learned to handle QLinear targets
- Gradients flow through the LoRA adapter weights normally
- The trainer controls the full init sequence internally

The evolution engine (Phase 2) **cannot use PEFT** because:
1. It needs inference-only mode (no grad graph overhead)
2. It perturbs parameters in-place with hash-based noise
3. PEFT's adapter management layer adds complexity with no benefit
4. The PEFT+quanto interaction has edge cases when quantizing inside wrappers

ManualLoRA gives us:
- Direct parameter access for perturbation (`.lora_A.data`, `.lora_B.data`)
- Clean separation: quantized base (frozen) + bf16 adapter (evolved)
- PEFT-compatible checkpoint format for inference compatibility
- Zero dependency on PEFT internals

## Noise Cache Flow

```
Generation N:
  seed_0: apply(+e) ──→ CACHE noise[seed_0] ──→ evaluate ──→ revert
          apply(-e) ──→ reuse cached noise   ──→ evaluate ──→ revert
  seed_1: apply(+e) ──→ CACHE noise[seed_1] ──→ evaluate ──→ revert
          apply(-e) ──→ reuse cached noise   ──→ evaluate ──→ revert
  ...
  ES update: use CACHED noise[seed_0..N] for gradient computation
  Clear cache
```

Without caching: 768 params × N_seeds × 2 (apply + update) = 6144+ hash noise generations
With caching: 768 params × N_seeds × 1 (apply only) = 3072 hash noise generations (50% fewer)
