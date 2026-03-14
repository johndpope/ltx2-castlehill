# Variational Flow Maps (VFM) for LTX-2

> **Goal:** 1-step video generation (8.7x–21.5x faster than LTX Desktop 8-step)
> by learning structured noise z ~ qφ(z|y) that lets the flow map produce clean video in a single forward pass.

## How VFM Works

Standard LTX-2 generates video by denoising random noise over 8 Euler steps. VFM replaces this with:

1. **Noise Adapter qφ(z|y)**: A small network that takes text embeddings and outputs *structured* noise — noise that already encodes information about the target video.
2. **Flow Map fθ(z)**: The full 48-layer LTX-2 transformer, fine-tuned via LoRA. One forward pass at σ=1 converts structured noise directly to video.

```
Standard LTX-2:  z ~ N(0,I) → [8 Euler steps × 48 layers] → video    (slow)
VFM:             z ~ qφ(z|text) → [1 step × 48 layers] → video       (fast)
```

### Loss Function (VFM Paper Eq. 19)

```
L(θ,φ) = (1/2τ²) · L_MF(θ;φ)  +  (1/2σ²) · L_obs(θ⁻,φ)  +  L_KL(φ)
           ↑ flow matching         ↑ observation consistency    ↑ regularization
```

- **L_MF**: MSE between predicted velocity and target velocity `v = z - x₀`
- **L_obs**: Reconstruction quality using observation operator (ensures generated content matches conditioning)
- **L_KL**: KL(qφ(z|y) || N(0,I)) prevents adapter from collapsing to a delta function

---

## Version History

### v1a — Vanilla VFM (`vfm_strategy.py`)

**Architecture:** MLP noise adapter with pooled text input.

| Component | Detail |
|-----------|--------|
| Adapter | `NoiseAdapterMLP`: Linear → SiLU → Linear, 4 layers |
| Input | Pooled text embeddings `[B, D]` expanded to `[B, seq, D]` |
| Output | Same μ,σ for all tokens (no spatial awareness) |
| Loss | L_MF + L_obs + L_KL with adaptive scaling |

**Limitations:** All tokens get identical noise parameters → no spatial/temporal structure in noise → model does all the heavy lifting in one pass.

**Key commit:** `f004107` Add Variational Flow Maps integration

---

### v1b — Temporally-Aware Adapter (`vfm_strategy_v1b.py`)

**Architecture:** Transformer adapter with cross-attention to full text + positional encoding.

| Component | Detail |
|-----------|--------|
| Adapter | `NoiseAdapterV1b`: Self-attn + Cross-attn + FFN, 4 layers, 512 hidden |
| Input | Full text embeddings `[B, text_seq, D]` + sinusoidal positions `(t, h, w)` |
| Output | Per-token μ,σ — each of 1344 tokens gets unique noise params |
| Params | ~19.1M adapter params |
| New | Min-SNR timestep weighting, flow map freezing for warmup |

**Why:** Tokens can coordinate via self-attention. Cross-attention grounds noise in text. Positions give spatial/temporal awareness. Result: structured noise that already has rough motion/layout.

**Key commit:** `7ce0419` Add VFM v1b
**W&B:** Early runs showed adapter μ varying across tokens (success indicator)

---

### v1c — Diversity-Regularized (`vfm_strategy_v1c.py`)

**Architecture:** Same as v1b, adds diversity loss terms.

| Component | Detail |
|-----------|--------|
| Loss addition | Token diversity: `-w · std(μ)` across all tokens |
| Loss addition | Temporal diversity: `-w · std(mean_spatial(μ))` across frames |
| Loss addition | Spatial diversity: `-w · std(mean_temporal(μ))` within frames |
| Warmup | Diversity loss linearly ramped over first 200 steps |

**Why:** Reverse-KL regularization (mode-seeking) can cause the adapter to collapse to a narrow mode where all tokens produce similar μ → uniform noise → static video. The diversity regularizer explicitly pushes for spatial and temporal variation.

**Inspired by:** HiAR (arxiv:2603.08703) forward-KL diversity objectives.

**Key commit:** `6c62df1` VFM variation flow

---

### v1d — Trajectory Distillation + Per-Token Sigma (`vfm_strategy_v1d.py`)

**Architecture:** v1c + teacher trajectory distillation + learned per-token timestep scheduling.

| Component | Detail |
|-----------|--------|
| SigmaHead | MLP (128→256→256→1, sigmoid), 99K params |
| Input | Adapter μ output → predicts per-token σ_i ∈ [0.05, 0.95] |
| Distillation | Train against pre-computed teacher 8-step ODE trajectories |
| Modes | `output_match`: 1-step student matches teacher's 8-step x̂₀ |
| | `velocity_match`: student velocity matches teacher at random σ |
| Entropy | Sigma entropy regularizer prevents uniform schedule collapse |
| Trajectory plots | Plotly PCA, distance-to-GT, velocity magnitude → W&B |

**Per-token sigma scheduling:** Instead of uniform σ for all 1344 tokens, each token gets its own noise level. The model can allocate denoising capacity heterogeneously — more effort on hard tokens, less on easy ones.

**Trajectory distillation:** Pre-compute the teacher's 8-step ODE path (z → x̂₀), then train the student to match the teacher's final output in 1 step. Much stronger supervision than random interpolation.

```bash
# Pre-compute trajectories (5000 samples, ~4 hours on RTX 5090)
uv run python scripts/precompute_trajectories.py \
    --data-root /media/12TB/ddit_ditto_data --device cuda:0

# Train v1d with distillation
uv run python scripts/train.py configs/ltx2_vfm_v1d_distill.yaml
```

**Inspired by:** Self-Flow (BFL, arxiv:2603.06507) dual-timestep scheduling.

**Key commit:** `a06f4b4` Add VFM v1d
**W&B (overfit sanity):** https://wandb.ai/snoozie/vfm-v1d/runs/a7hff56q

**Observation:** Quality deteriorates on detailed/complex content around step 458. Simple regions reconstruct fine but textures/edges lose quality. This motivated v1e.

**Training issues found:**
- `.detach()` on sigma head input cut gradient → sigma head couldn't learn. Fix: remove `.detach()`.
- `kl_free_bits: 0.25` too high → KL floored to 0, no gradient on adapter. Fix: reduce to 0.05.
- `alpha: 0.5` → adapter noise used only 50% of steps. Fix: increase to 0.8.

---

### v1e — Content-Adaptive Router + Detail Preservation (`vfm_strategy_v1e.py`)

**Architecture:** v1d + content complexity router + frequency-domain detail loss.

| Component | Detail |
|-----------|--------|
| ContentRouter | Transformer (2 layers, 4 heads, 256 hidden), 1.6M params |
| Input | GT latent features `[B, seq, 128]` during training |
| Output | Per-token complexity score ∈ [0, 1] |
| Sigma mapping | `σ_i = σ_max - (σ_max - σ_min) × complexity_i` |
| | Complex tokens → LOW σ (easy denoising) |
| | Simple tokens → HIGH σ (can handle noise) |
| Complexity-weighted loss | `weight = 1 + 0.5 × complexity` per token |
| Frequency loss | `||∇(x̂₀) - ∇(x₀)||²` spatial gradient matching |
| Router supervision | MSE vs GT complexity (latent spatial gradient magnitude) |
| W&B logging | Complexity heatmaps, sigma maps, trajectory plots |

**Why:** The core problem is that 1-step generation can't handle ALL tokens equally well. EVATok (CVPR 2026) showed that allocating more capacity to complex content dramatically improves quality. v1e applies this insight to the sigma/denoising dimension:

```
EVATok:  complex regions → more tokens (discrete allocation)
v1e:     complex regions → lower σ (continuous allocation in noise space)
```

**Router supervision:** The router learns to predict complexity from latent features. GT complexity is computed as normalized spatial gradient magnitude — high gradients = edges/textures = complex content. This gives the router a clear learning signal without manual annotation.

**Inspired by:** EVATok (arxiv:2603.12267) adaptive token allocation with router network.

**Key commit:** `97c331b` Add VFM v1e

---

### v1f — Spherical Cauchy Noise Adapter (`vfm_strategy_v1f.py`)

**Architecture:** v1d + Spherical Cauchy noise distribution on S^127 (replaces Gaussian reparameterization).

| Component | Detail |
|-----------|--------|
| Noise distribution | Spherical Cauchy on S^127 (heavy-tailed, geodesic structure) |
| Direction-magnitude decomposition | μ̂ = normalize(μ) → direction, r = ‖μ‖ → magnitude, κ = exp(mean(log_σ)) → concentration |
| Sampling | `z = r · SphericalCauchy(μ̂, κ)` instead of `z = μ + σ·ε` |
| Spherical KL | `KL = (D-1)/2 · [log(κ) - log(1+κ)]` — clean closed form, replaces Gaussian KL |
| Kappa regularization | Pull mean κ toward target (>1.0) + entropy for per-token κ diversity |
| Magnitude regularization | MSE(‖μ‖, target) prevents noise scale collapse |
| Geodesic diversity metric | Mean pairwise angular distance between adapter directions on S^127 |
| W&B logging | κ heatmap + ‖μ‖ heatmap per frame, sigma heatmap, reconstruction videos |

**Why Spherical Cauchy?** The adapter must learn a *structured noise manifold* — not all noise directions are useful for 1-step generation. Spherical Cauchy provides:

1. **Heavy tails** — broader exploration early in training, discovering useful noise directions faster than Gaussian (20-30% faster convergence per spherical-vae benchmarks)
2. **Direction-magnitude separation** — independently learns *what kind of noise* (direction on S^127) vs *how much noise* (magnitude). Gaussian entangles these.
3. **Per-token concentration (κ)** — precise tokens (faces, edges) get high κ (peaked sampling around learned direction), imprecise tokens (sky, flat regions) get low κ (any direction works)
4. **Geodesic structure** — angular distance between noise directions is meaningful and measurable. Diversity can be quantified as angular spread on the hypersphere.

**Key insight:** The NoiseAdapterV1b architecture is unchanged — v1f just reinterprets its (μ, log_σ) outputs geometrically. This means weights transfer from v1d checkpoints.

```
Gaussian (v1d):          z = μ + σ·ε,  ε ~ N(0,I)     (unbounded R^128)
Spherical Cauchy (v1f):  z = ‖μ‖ · SpCauchy(μ̂, κ)     (direction on S^127 × magnitude)
```

**Training pipeline:**
```bash
# Sanity check (no distillation)
uv run python scripts/train.py configs/ltx2_vfm_v1f_spherical.yaml

# Key wandb metrics to watch:
#   vfm/kappa_mean — should rise above 1.0 (kappa_pull drives this)
#   vfm/kappa_std — should increase (kappa_entropy drives this)
#   vfm/mu_norm_mean — should converge near target_magnitude (1.0)
#   vfm/geodesic_diversity — angular spread in radians (higher = more diverse)
#   vfm/loss_kl — activates when κ > 1.0 (below 1.0 → free KL, no gradient)
```

**Training observations (run jwi1hbkp):**
- loss_mf drops 10x in 130 steps (0.21 → 0.02)
- Geodesic diversity reaches 1.35 rad (~77°) — adapter directions well-spread
- κ collapses to floor when no kappa regularization → added kappa_pull and kappa_entropy losses
- Sigma head starts learning after warmup (step 200+): sigma_std 0.000008 → 0.024
- KL stays at 0 until κ > 1.0 — this is by design (free exploration phase)

**Inspired by:** [Spherical Cauchy distribution](https://github.com/johndpope/spherical-vae) for variational sampling on hyperspheres.

**Utility module:** `src/ltx_trainer/spherical_utils.py` — sampling, KL, SLERP, geodesic distance.

---

## Training Data

| Dataset | Location | Samples | Status |
|---------|----------|---------|--------|
| Ditto 5K latents | `/media/12TB/ddit_ditto_data/latents_19b/` | 5,000 | Complete |
| Text embeddings | `/media/12TB/ddit_ditto_data/conditions_final/` | 5,000 | Complete |
| Teacher trajectories | `/media/12TB/ddit_ditto_data/trajectories/` | 5,000 | Complete |

**Resolution:** 768×448 → latent 24w×14h, 4 latent frames, 1344 tokens total (336/frame)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `precompute_trajectories.py` | 8-step ODE trajectories for base 48-layer model |
| `precompute_scd_trajectories.py` | SCD decoder trajectories (16-layer) |
| `vfm_benchmark.py` | VFM speed vs LTX Desktop 8-step (all resolutions) |
| `train_bezierflow_base.py` | Learned sigma schedule (abandoned — linear is better) |
| `benchmark_schedule.py` | A/B comparison of sigma schedules |
| `scd_inference.py` | Autoregressive SCD inference |

---

## Benchmark: VFM vs LTX Desktop

VFM 1-step is benchmarked against LTX Desktop 8-step (RTX 5090, i2v, without text encoding):

```bash
uv run python scripts/vfm_benchmark.py \
    --adapter-path checkpoints/noise_adapter_step_XXXXX.safetensors \
    --lora-path checkpoints/lora_weights_step_XXXXX.safetensors \
    --cached-embedding /media/12TB/ddit_ditto_data/conditions_final/000000.pt
```

**LTX Desktop 8-step reference times** (RTX 5090):

| Config | Desktop 8-step |
|--------|---------------|
| 5s 540p | ~33s |
| 5s 720p | ~42s |
| 5s 1080p | ~76s |
| 10s 540p | ~44s |
| 20s 540p | ~73s |

**VFM theoretical speedup**: 8x fewer transformer forward passes. Actual speedup depends on adapter overhead (~1-5% of total) and VAE decode time (constant). Expected: **6-21x** depending on resolution/duration.

---

## Key Papers

| Paper | Relevance | Used In |
|-------|-----------|---------|
| [VFM](https://arxiv.org/abs/2603.07276) | Core method: noise adapter + flow map for 1-step generation | All versions |
| [Self-Flow](https://arxiv.org/abs/2603.06507) | Per-token timestep scheduling, dual-σ approach | v1d sigma head |
| [HiAR](https://arxiv.org/abs/2603.08703) | Forward-KL diversity regularization for mode collapse prevention | v1c diversity loss |
| [EVATok](https://arxiv.org/abs/2603.12267) | Content-adaptive token allocation via router network | v1e content router |
| [Spherical Cauchy](https://github.com/johndpope/spherical-vae) | Heavy-tailed distribution on hypersphere, direction-magnitude decomposition | v1f noise sampling |
| [Min-SNR](https://arxiv.org/abs/2303.09556) | Timestep-weighted loss for reduced variance | All versions |
| [SCD](https://arxiv.org/abs/2602.10095) | Separable causal diffusion for long-form video | SCD strategy |

---

## Quick Start

```bash
# 1. Overfit sanity check (no trajectories needed, ~15 min)
uv run python scripts/train.py configs/ltx2_vfm_v1d_overfit_1sample.yaml

# 2. Full distillation training (needs trajectories)
uv run python scripts/train.py configs/ltx2_vfm_v1d_distill.yaml

# 3. Spherical Cauchy noise (v1f)
uv run python scripts/train.py configs/ltx2_vfm_v1f_spherical.yaml

# 4. Check wandb for:
#    - vfm/loss_mf should decrease
#    - vfm/adapter_mu_mean should vary (not stuck at 0)
#    - vfm/sigma_mean should be diverse (not all 0.5)
#    - vfm/kappa_mean should rise above 1.0 (v1f)
#    - vfm/geodesic_diversity should increase (v1f)
#    - train/reconstruction_video at log intervals
#    - train/spherical_heatmap (v1f) shows κ and ||μ|| per frame
#    - train/complexity_heatmap (v1e) should show edges highlighted
```

---

## Config Reference

All VFM strategies share common config fields from `VFMTrainingConfig`:

```yaml
training_strategy:
  name: "vfm_v1f"           # v1a="vfm", v1b="vfm_v1b", v1c="vfm_v1c", v1d="vfm_v1d", v1e="vfm_v1e", v1f="vfm_v1f"

  # Noise adapter
  alpha: 0.5                 # P(use adapter noise) vs P(standard N(0,I))
  tau: 1.0                   # Flow matching loss tolerance
  kl_weight: 3.0             # KL regularization strength
  kl_free_bits: 0.25         # Per-dim KL floor (prevents sigma collapse)

  # v1b+ adapter architecture
  adapter_hidden_dim: 512
  adapter_num_heads: 8
  adapter_num_layers: 4
  adapter_pos_dim: 256

  # v1c+ diversity
  diversity_weight: 0.1
  temporal_diversity_weight: 0.2
  spatial_diversity_weight: 0.05

  # v1d+ per-token sigma
  per_token_sigma: true
  sigma_min: 0.05
  sigma_max: 0.95
  sigma_entropy_weight: 0.01

  # v1d+ distillation
  distill_mode: "output_match"  # or "velocity_match", "none"
  trajectories_dir: "trajectories"
  distill_weight: 1.0
  gt_weight: 0.1

  # v1e content router
  content_router: true
  router_hidden_dim: 256
  router_num_layers: 2
  complexity_loss_weight: 0.5
  frequency_loss_weight: 0.05
  router_supervision: true

  # v1f spherical cauchy
  spherical_noise: true
  kappa_min: 0.1
  kappa_max: 50.0
  kappa_target: 2.0              # Pull κ above 1.0 so KL activates
  kappa_pull_weight: 0.05        # Strength of κ → target pull
  kappa_entropy_weight: 0.01     # Encourage per-token κ diversity
  magnitude_reg_weight: 0.01     # Keep ||μ|| near target
  target_magnitude: 1.0
  use_slerp_interp: false        # Experimental: SLERP instead of linear interp
```

---

## Architecture Diagram

```
                    ┌─────────────────────┐
                    │   Text Encoder       │
                    │   (Gemma, frozen)     │
                    └──────────┬──────────┘
                               │ text_embeddings [B, T, D]
                    ┌──────────▼──────────┐
                    │  Noise Adapter qφ    │
                    │  (v1b transformer)   │
                    │  19.1M params        │
                    └──────────┬──────────┘
                               │ μ, log_σ [B, 1344, 128]
                    ┌──────────▼──────────┐
                    │  Reparameterize      │
                    │  v1d: z = μ + σ·ε    │
                    │  v1f: z = ‖μ‖·SC(μ̂,κ)│  ← Spherical Cauchy
                    └──────────┬──────────┘
                               │ z [B, 1344, 128]
                    ┌──────────▼──────────┐
          ┌────────│  SigmaHead (v1d+)     │
          │        │  or Router (v1e)      │
          │        │  99K / 1.6M params    │
          │        └──────────┬───────────┘
          │                   │ per-token σ_i [B, 1344]
          │        ┌──────────▼──────────┐
          │        │  Per-Token Sigma      │
          │        │  σ_i = f(μ) or f(GT)  │
          │        └──────────┬───────────┘
          │                   │ σ [B, 1344]
          │        ┌──────────▼──────────┐
          │        │  Noisy Interpolation  │
     GT x₀────────│  x_t = (1-σ)·x₀+σ·z │
                   └──────────┬───────────┘
                              │ x_t [B, 1344, 128]
                   ┌──────────▼──────────┐
                   │  48-layer LTX-2 DiT  │
                   │  (LoRA, flow map fθ)  │
                   │  ~233M trainable      │
                   └──────────┬───────────┘
                              │ velocity v [B, 1344, 128]
                   ┌──────────▼──────────┐
                   │  x̂₀ = z - v          │
                   │  (1-step denoising)  │
                   └──────────┬───────────┘
                              │ x̂₀ [B, 1344, 128]
                   ┌──────────▼──────────┐
                   │  VAE Decoder          │
                   │  (frozen)             │
                   └──────────┬───────────┘
                              │ video [B, 3, 25, 448, 768]
```

---

## Lessons Learned

1. **BézierFlow abandoned** — learned sigma schedule only improved loss 20%, collapsed to extreme front-loading. Linear schedule works fine.
2. **Min-SNR is critical** — without it, high-σ steps dominate gradients and training is unstable.
3. **KL free bits prevent collapse** — without floor, adapter σ → 0 (delta function), no noise variety.
4. **Diversity loss warmup matters** — applying too early fights adapter learning; 200 steps warmup is good.
5. **Detailed content needs special handling** — uniform 1-step can't serve all tokens equally (→ v1e).
6. **5000 trajectories is enough** — full teacher ODE paths give much stronger signal than random interpolation.
7. **`torch.no_grad()` not `inference_mode()`** for caching — inference_mode tensors can't be used in autograd replay.
8. **Never `.detach()` sigma head input** — cuts gradient from flow matching loss, sigma head can't learn which tokens need different noise levels.
9. **Spherical Cauchy κ needs explicit pull above 1.0** — for D=128, any κ < 1 gives negative KL (more entropy than uniform on S^127), so KL=0 and no gradient flows. Add `kappa_pull` loss to push κ into the regime where KL is positive.
10. **Direction-magnitude decomposition is free** — reinterpreting existing adapter outputs geometrically requires no architecture changes, and weights transfer from Gaussian checkpoints.
