# VFM v3b — Self-Evaluating Variational Flow Maps

## State Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING STEP                                │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │  Text Embeds  │────▶│ NoiseAdapter │────▶│ Spherical Cauchy │    │
│  │  (cached,     │     │ V1b (38M)    │     │ Sample z ~ qφ   │    │
│  │   4096-dim)   │     │ 4-layer xfmr │     │ μ, κ per-token  │    │
│  └──────────────┘     └──────────────┘     └────────┬─────────┘    │
│                                                      │              │
│                                              z (structured noise)   │
│                                                      │              │
│                                                      ▼              │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │              DiT Forward Pass (22B, LoRA r=32)            │      │
│  │              1 step: z → v_pred (velocity)                │      │
│  │              x̂₀ = z - v_pred                              │      │
│  └─────────────────────────┬─────────────────────────────────┘      │
│                             │                                        │
│                        x̂₀ (student output)                          │
│                             │                                        │
│              ┌──────────────┼──────────────┐                        │
│              │              │              │                        │
│              ▼              ▼              ▼                        │
│     ┌────────────┐  ┌─────────────┐  ┌──────────────────┐          │
│     │ DATA LOSS  │  │  SELF-EVAL  │  │   KL LOSS        │          │
│     │ |x̂₀-x₀|²  │  │  (50% prob) │  │ KL(qφ || prior)  │          │
│     │            │  │             │  │ weight: 0.001    │          │
│     └────────────┘  │  ┌────────┐ │  └──────────────────┘          │
│                     │  │Re-noise│ │                                  │
│                     │  │x̂₀→x̂_s │ │                                  │
│                     │  │s~[0.1, │ │                                  │
│                     │  │  0.5]  │ │                                  │
│                     │  └───┬────┘ │                                  │
│                     │      │      │                                  │
│                     │      ▼      │                                  │
│                     │ ┌─────────────────────────────────┐           │
│                     │ │  TWO DiT Forward Passes (no_grad)│          │
│                     │ │                                   │          │
│                     │ │  G_θ(x̂_s, s, text) → g_cond     │          │
│                     │ │  G_θ(x̂_s, s, ∅)    → g_uncond   │          │
│                     │ │                                   │          │
│                     │ │  classifier_score =               │          │
│                     │ │    g_uncond - g_cond              │          │
│                     │ │                                   │          │
│                     │ │  x_self = sg[x̂₀ - score]        │          │
│                     │ └─────────────────────────────────┘           │
│                     │                                                │
│                     │  Energy-preserving normalization:              │
│                     │  x_target = (x₀ + λ·x_self)                  │
│                     │           / |x₀ + λ·x_self| · |x₀|           │
│                     │  where λ = σ_t/α_t - σ_s/α_s                 │
│                     │                                                │
│                     │  loss_self = |x̂₀ - x_target|²                │
│                     └────────────────────────────────────            │
│                                                                     │
│  Total Loss = loss_self (or loss_data when self_eval not applied)   │
│             + kl_weight · KL                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     1-STEP INFERENCE                                 │
│                                                                     │
│  text_prompt → [Gemma] → [connector] → text_embeds (4096-dim)      │
│                                            │                        │
│                                            ▼                        │
│                                    ┌──────────────┐                 │
│                                    │ NoiseAdapter  │                │
│                                    │ qφ(z|text)    │                │
│                                    └──────┬───────┘                 │
│                                           │                        │
│                                    z (Spherical Cauchy)             │
│                                           │                        │
│                                           ▼                        │
│                                    ┌──────────────┐                 │
│                                    │ DiT (22B+LoRA)│                │
│                                    │ 1 forward pass│                │
│                                    │ z → v → x̂₀   │                │
│                                    └──────┬───────┘                 │
│                                           │                        │
│                                    x̂₀ (clean latent)               │
│                                           │                        │
│                                           ▼                        │
│                                    ┌──────────────┐                 │
│                                    │  VAE Decode   │                │
│                                    │  x̂₀ → pixels  │                │
│                                    └──────┬───────┘                 │
│                                           │                        │
│                                        video.mp4                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Papers & Theoretical Foundation

### Primary: Self-E (Self-Evaluating Diffusion Models)
- **Paper**: Yu et al., "Improving Diffusion Distillation with Self-Evaluation" (arXiv 2512.22374, Dec 2025)
- **Key Idea**: The student model evaluates its own outputs using classifier scores derived from conditional vs unconditional forward passes. No discriminator or separate teacher needed.
- **Core Equation**: `classifier_score = G_θ(x̂_s, s, ∅) - G_θ(x̂_s, s, c)` ∝ ∇log q(c|x̂_s)
- **Why it works**: The classifier score tells the model "this output doesn't match the text prompt well — move it in THIS direction." It's free because the model already knows how to denoise.

### Foundation: VFM (Variational Flow Maps)
- **Paper**: Mammadov et al., "Variational Flow Maps: Make Some Noise for One-Step Conditional Generation" (arXiv 2603.07276, Mar 2026)
- **Key Idea**: Learn a noise adapter qφ(z|y) that produces observation-dependent noise. The flow map (DiT) can then produce high-quality samples from this structured noise in 1 step.
- **Our implementation**: Spherical Cauchy distribution with per-token κ (concentration) and μ (mean direction on S^127).

### Noise Distribution: Spherical Cauchy (v1f)
- **Why not Gaussian?**: Standard Gaussian noise → the flow map must learn to map ALL directions to meaningful content. Spherical Cauchy → the adapter learns which noise directions are useful, concentrating probability mass where the flow map works best.
- **Parameters**: μ (mean direction, 128-dim on unit sphere), κ (concentration, higher = more peaked around μ)
- **KL regularization**: KL(SphericalCauchy(μ,κ) || Uniform(S^127)) prevents collapse to a point mass.

### Per-Token Sigma: SigmaHead (v1d/v1f)
- **Paper**: Inspired by Self-Flow (arXiv 2603.06507) and EVATok (CVPR 2026)
- **Key Idea**: Not all tokens need the same noise level. Complex regions (fine detail, motion boundaries) need higher sigma, simple regions (sky, flat surfaces) need lower sigma.
- **Architecture**: Small MLP that takes x₀ features → predicts per-token σ ∈ [σ_min, σ_max]
- **Critical lesson**: SigmaHead MUST see x₀, not just adapter μ (adapter features can't encode content complexity).

### Energy-Preserving Normalization
- **From Self-E paper, Eq. 19**: `x_target = (x₀ + λ·x_self) / ||x₀ + λ·x_self|| · ||x₀||`
- **Why**: When λ is large (high noise levels), the combined target `x₀ + λ·x_self` has much larger norm than x₀. Without normalization, this causes color bias and saturation in generated videos.
- **Effect**: Keeps the target on the same energy shell as the ground truth.

## Components & Code

| Component | File | Params | Role |
|-----------|------|--------|------|
| **NoiseAdapterV1b** | `ltx-core/.../noise_adapter_v1b.py` | 38M | Spherical Cauchy noise sampling qφ(z\|text) |
| **SigmaHead** | `ltx-trainer/.../vfm_strategy_v1d.py` | ~264K | Per-token σ prediction from x₀ |
| **SelfEVFMv3bStrategy** | `ltx-trainer/.../vfm_strategy_v3b.py` | — | Training loop with Self-E |
| **DiT (LTX-2.3)** | `ltx-core/.../model.py` | 22B (428M LoRA) | Flow map: z → v → x̂₀ |

## Training Cost Per Step

| Pass | Purpose | Grad? | Cost |
|------|---------|-------|------|
| Adapter forward | Sample z from qφ | Yes | ~0.01s |
| **DiT forward** (student) | z → v_pred → x̂₀ | **Yes** (LoRA) | ~1.5s |
| DiT forward (conditional) | Self-eval: G_θ(x̂_s, s, c) | No | ~1.5s |
| DiT forward (unconditional) | Self-eval: G_θ(x̂_s, s, ∅) | No | ~1.5s |
| **Total** | | | **~4.5s** (3 DiT passes) |

Self-eval only applied 50% of steps (`self_eval_prob=0.5`), so average is ~3s/step.

## Inference Pipeline (1-step)

```
Time breakdown (estimated, RTX 5090, int8-quanto, 9 frames 768×448):
  Text encoding:     ~0s (precomputed)
  Adapter forward:   ~0.01s
  DiT forward:       ~0.5s (single pass, no self-eval at inference)
  VAE decode:        ~0.8s
  ─────────────────────
  Total:             ~1.3s for 9 frames (0.36s video at 25fps)
```

## Potential Weaknesses for Inference

1. **Training-inference gap**: Training uses GT x₀ in the data loss term. Inference has no x₀ — the adapter noise must be good enough that 1 DiT pass produces clean output. If the adapter hasn't learned the right noise manifold, inference quality drops.

2. **Self-eval feedback loop**: During training, self-eval pushes x̂₀ toward text-aligned regions. But at inference, there's no self-eval — the model must have internalized this alignment into the LoRA weights. If self-eval was doing most of the heavy lifting (vs the model learning), inference won't match training reconstructions.

3. **Overfit-10 generalization**: Training on 10 samples may produce LoRA weights that perfectly reconstruct those 10 videos but fail on novel prompts. The adapter's noise manifold may be too narrow.

4. **Sigma mismatch**: SigmaHead is trained to predict per-token σ from x₀. At inference, there's no x₀ — the adapter features alone must predict σ. This is the v1d lesson again.

5. **No CFG at inference**: Training uses self-eval (implicit CFG). Inference could benefit from explicit CFG (2 forward passes) but that doubles latency. Need to test 1-step without CFG.

## W&B Runs

| Run | Steps | Description | Link |
|-----|-------|-------------|------|
| hvrpf0ed | 0→2000 | Initial overfit-10, converging | [wandb](https://wandb.ai/snoozie/vfm-v3b/runs/hvrpf0ed) |
| w5jothre | resume→1480 | BAD — LR reset blew up weights | [wandb](https://wandb.ai/snoozie/vfm-v3b/runs/w5jothre) |
| 1pgjl31b | resume→? | Fixed LR, short run | [wandb](https://wandb.ai/snoozie/vfm-v3b/runs/1pgjl31b) |
| spdm6k3n | resume→1480 | With SigmaHead, LR=5e-5 | [wandb](https://wandb.ai/snoozie/vfm-v3b/runs/spdm6k3n) |

## Key Hyperparameters

```yaml
self_eval_weight: 1.0        # Self-E loss weight
self_eval_cfg_scale: 5.0     # CFG scale for classifier score
self_eval_s_range: [0.1, 0.5]  # Re-noising range
self_eval_prob: 0.5          # Apply self-eval 50% of steps
energy_preserving_norm: true # Prevent color bias
kl_weight: 0.001             # Adapter KL regularization
alpha: 0.5                   # VFM interpolation parameter
```

## Relationship to Other Approaches

```
DMD1 (Yin 2024)          → Separate fake score network (19B params!) — impractical
DMD2 (Yin 2024)          → GAN discriminator in noisy space — our v3a
FlashMotion (CVPR 2026)  → Same as DMD2 but for video — our v3a
Self-E (Yu 2025)         → Model evaluates itself — our v3b ← YOU ARE HERE
DiagDistill (ICLR 2026)  → Diagonal denoising + flow matching — future (SCD)
OmniForcing (2026)       → DMD + Self-Forcing for AV streaming — future (SCD+audio)
```
