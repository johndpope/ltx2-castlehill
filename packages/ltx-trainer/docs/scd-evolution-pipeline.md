# SCD Training + Evolution Pipeline

## Overview

The SCD (Separable Causal Diffusion) pipeline has two phases:
1. **Phase 1**: Gradient-based SCD LoRA training (teaches encoder/decoder split)
2. **Phase 2**: EggRoll Evolution (optimizes decoder for autoregressive quality)

Phase 2 addresses the **teacher forcing mismatch**: training uses clean GT frames as encoder input, but inference feeds back the model's own (noisy) outputs. Evolution directly optimizes against multi-frame AR rollout, closing this gap.

## State Diagram

```mermaid
stateDiagram-v2
    [*] --> PrecomputeData

    state "Data Preparation" as PrecomputeData {
        [*] --> EncodeLatents: VAE Encoder (~8GB)
        EncodeLatents --> ComputeConditions: Text Encoder (~28GB)
        ComputeConditions --> [*]
        note right of EncodeLatents: latents/ [C,F,H,W]
        note right of ComputeConditions: conditions_final/ [1024,3840]
    }

    PrecomputeData --> Phase1

    state "Phase 1: Gradient Training" as Phase1 {
        [*] --> LoadModel1: Load transformer + quantize
        LoadModel1 --> ApplyLoRA1: Init LoRA (rank=32/64)
        ApplyLoRA1 --> WrapSCD1: LTXSCDModel(32 enc + 16 dec)
        WrapSCD1 --> TrainLoop1

        state "SCD Training Loop" as TrainLoop1 {
            [*] --> SampleBatch
            SampleBatch --> TeacherForcing: Clean GT frames → Encoder
            TeacherForcing --> DecoderDenoise: Noisy target → Decoder
            DecoderDenoise --> ComputeLoss: FM velocity MSE
            ComputeLoss --> BackpropUpdate: LoRA gradient update
            BackpropUpdate --> SampleBatch: Next step
        }

        TrainLoop1 --> SaveLoRA1: lora_weights_step_NNNNN.safetensors
    }

    Phase1 --> Phase2

    state "Phase 2: EggRoll Evolution" as Phase2 {
        [*] --> LoadModel2: Load transformer + quantize
        LoadModel2 --> LoadLoRA2: Load Phase 1 LoRA checkpoint
        LoadLoRA2 --> WrapSCD2: LTXSCDModel(32 enc + 16 dec)
        WrapSCD2 --> InitPerturbation: SelectiveLoRAPerturbation
        InitPerturbation --> EvoLoop

        note right of InitPerturbation
            Targets ONLY decoder LoRA params
            (blocks 32-47 lora_A/lora_B)
        end note

        state "Evolution Loop (per generation)" as EvoLoop {
            [*] --> GenSeeds: generate_perturbation_seeds(pop_size)
            GenSeeds --> PerturbPos

            state "Antithetic Pair Evaluation" as AntitheticEval {
                state "+ε Evaluation" as PerturbPos {
                    [*] --> ApplyPos: apply_perturbation(+1)
                    ApplyPos --> ARRolloutPos: AR rollout (4 frames)
                    ARRolloutPos --> FitnessPos: Compute fitness
                    FitnessPos --> RevertPos: revert_to_original()
                }
                state "-ε Evaluation" as PerturbNeg {
                    [*] --> ApplyNeg: apply_perturbation(-1)
                    ApplyNeg --> ARRolloutNeg: AR rollout (4 frames)
                    ARRolloutNeg --> FitnessNeg: Compute fitness
                    FitnessNeg --> RevertNeg: revert_to_original()
                }
            }

            PerturbPos --> PerturbNeg
            PerturbNeg --> ESGradient: diff = F(+ε) - F(-ε)
            ESGradient --> UpdateWeights: w += lr * Σ diff * ε / (2σ)
            UpdateWeights --> AnnealNoise: σ *= noise_decay
            AnnealNoise --> [*]
        }

        EvoLoop --> SaveEvolved: lora_evolved_final.safetensors
    }

    Phase2 --> Inference

    state "AR Inference" as Inference {
        [*] --> LoadModelInf: Load transformer + quantize
        LoadModelInf --> LoadEvolvedLoRA: Load evolved LoRA
        LoadEvolvedLoRA --> WrapSCDInf: LTXSCDModel
        WrapSCDInf --> ARGenerate

        state "Autoregressive Generation" as ARGenerate {
            [*] --> EncodeFrame: Encoder (1 frame, KV-cache, σ=0)
            EncodeFrame --> ShiftFeatures: Shift-by-1 alignment
            ShiftFeatures --> DenoiseFrame: Decoder (N steps, σ=1→0)
            DenoiseFrame --> NextFrame: Generated frame → Encoder input
            NextFrame --> EncodeFrame: Loop for T frames
            NextFrame --> VAEDecode: All T latent frames
        }

        VAEDecode --> OutputVideo: MP4
    }
```

## AR Rollout (Fitness Evaluation Detail)

```mermaid
flowchart TD
    subgraph "AR Rollout (per perturbation)"
        A[Initialize KV-cache] --> B{Frame f = 0?}
        B -->|Yes| C[enc_input = zeros]
        B -->|No| D[enc_input = generated[f-1]]
        C --> E[Encoder: σ=0, KV-cache]
        D --> E

        E --> F[Shift-by-1: dec_ctx = prev_enc_features]
        F --> G[x_t = randn noise]

        subgraph "Denoising Loop (15 steps)"
            G --> H[Patchify x_t]
            H --> I[Decoder: velocity = forward_decoder]
            I --> J[FM Loss: ||v - v_true||²]
            I --> K[Euler step: x_t += Δσ * v]
            K --> L{More steps?}
            L -->|Yes| H
            L -->|No| M[Frame complete]
        end

        M --> N[Latent Recon: ||x_0 - GT||²]
        M --> O[Temporal Coherence: cos_sim gap]
        M --> P{Use VAE?}
        P -->|Yes cuda:1| Q[VAE Decode → pixels]
        Q --> R[LPIPS + SSIM]
        P -->|No| S[Skip pixel metrics]

        N --> T{More frames?}
        O --> T
        R --> T
        S --> T
        T -->|Yes| B
        T -->|No| U[Aggregate: weighted sum → FitnessResult]
    end
```

## VRAM Layout

```
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│         cuda:0 (RTX 5090)       │  │     cuda:1 (RTX PRO 4000)      │
│            32 GB VRAM           │  │          24 GB VRAM             │
│                                 │  │                                 │
│  ┌──────────────────────────┐   │  │  ┌──────────────────────────┐  │
│  │ LTX-2 Transformer (INT8) │   │  │  │   VAE Decoder (~8 GB)    │  │
│  │     ~12 GB quantized     │   │  │  │                          │  │
│  │                          │   │  │  │  ┌────────────────────┐  │  │
│  │  Encoder (32 blocks)     │   │  │  │  │  Decode latent →   │  │  │
│  │  Decoder (16 blocks)     │   │  │  │  │  RGB pixels        │  │  │
│  │    + LoRA adapters       │   │  │  │  └────────────────────┘  │  │
│  └──────────────────────────┘   │  │  └──────────────────────────┘  │
│                                 │  │                                 │
│  ┌──────────────────────────┐   │  │  ┌──────────────────────────┐  │
│  │  Evolution overhead       │   │  │  │  LPIPS net (~200 MB)    │  │
│  │  - Hash noise generation  │   │  │  │  (AlexNet backbone)     │  │
│  │  - Fitness accumulators   │   │  │  └──────────────────────────┘  │
│  │  ~2 GB                    │   │  │                                 │
│  └──────────────────────────┘   │  │  Free: ~16 GB                   │
│                                 │  │                                 │
│  Free: ~18 GB                   │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

## Performance Estimates

| Config | Time/Gen | Total (200 gen) | VRAM cuda:0 | VRAM cuda:1 |
|--------|----------|-----------------|-------------|-------------|
| Latent-only, batch=1 | ~24s | ~80 min | ~14 GB | — |
| Latent-only, batch=2 | ~48s | ~160 min | ~14 GB | — |
| Latent+Pixel, batch=1 | ~30s | ~100 min | ~14 GB | ~8 GB |
| Latent+Pixel, batch=2 | ~60s | ~200 min | ~14 GB | ~8 GB |

## Checkpoint Compatibility

Evolved LoRA checkpoints are **directly compatible** with `scd_inference.py`:

```bash
# Before evolution:
python scripts/scd_inference.py --lora-path /path/to/phase1_lora.safetensors ...

# After evolution (drop-in replacement):
python scripts/scd_inference.py --lora-path /path/to/lora_evolved_final.safetensors ...
```

The evolution only modifies decoder LoRA weights (blocks 32-47). Encoder LoRA weights (blocks 0-31) are preserved unchanged.
