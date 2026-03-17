# CastleHill: SCD + VFM for LTX-2

## Project Overview

**CastleHill** extracts Separable Causal Diffusion (SCD) from `ltx2-omnitransfer` into a clean, standalone repo based on upstream [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2).

**Focus:** Long-form video generation via SCD + one-step generation via VFM (Variational Flow Maps). SCD turns LTX-2's 48-layer DiT into an encoder (32 layers) + decoder (16 layers) with KV-cache. VFM trains a noise adapter qφ(z|y) so the flow map produces clean video in 1 step instead of 8.

**Key features:**
- SCD training strategy (LoRA + per-frame decoder)
- SCD autoregressive inference (30s+ videos on consumer GPUs)
- VFM noise adapter (v1a→v1b→v1c→v1d→v1e) — see **[docs/VFM.md](docs/VFM.md)** for full breakdown
- VFM 1-step inference: 8.7x–21.5x faster than LTX Desktop 8-step
- DDiT dynamic patch scheduling (optional decoder speedup)
- BézierFlow / BSplineFlow learned sigma schedulers
- Muon optimizer support

> **IMPORTANT:** When creating or modifying any VFM version (v1a–v1e+), always update **[docs/VFM.md](docs/VFM.md)** with: version details, architecture changes, W&B run links, key commits, and lessons learned.

## Proactive Optimization Guidance

When working on this project, **eagerly anticipate** the user's goals and suggest concrete next steps. Always consider:

### Architecture Iterations — Completed
- ✅ **Trajectory distillation** (v1d): Pre-computed teacher 8-step ODE paths. Implemented but creates σ mismatch with per-token sigma — use `distill_mode: "none"` when per_token_sigma is enabled.
- ✅ **Per-token timestep scheduling** (v1d→v1f): SigmaHead MLP predicts per-token σ. **Must take x₀ as input** (not just adapter_mu) — adapter features can't encode content complexity. Complexity-aware targets from x₀ replace global mean pull.
- ✅ **Spherical Cauchy noise** (v1f): Direction-magnitude decomposition of adapter noise on S^127. Working baseline with sharp reconstructions.
- ✅ **SLERP geodesic interpolation** (v1f option): `use_slerp_interp: true` — geodesic path through latent space. Code ready, needs A/B testing.
- ❌ **Integrated adapter sigma** (v1h): Tried adding σ output head to adapter — collapses because adapter features (text+position) can't encode x₀ content complexity. Keep SigmaHead separate.
- ❌ **Speculative noise selection** (v2a): K candidates from same distribution are nearly identical (unimodal Cauchy, κ≈4). Score spread ~0.01. +16% cost for no quality gain.
- ❌ **Content-adaptive routing** (v1e): Added complexity but not validated before stacking.
- ❌ **v1g HyperSphereDiff losses**: Over-engineered — too many loss terms fighting each other.

### Architecture Iterations — In Progress
- ✅ **v3a DMD2 Adversarial Distribution Matching** (active): GAN discriminator in noisy latent space (DMD2/FlashMotion) + Flow Distribution Matching with SpatialHead (DiagDistill). LoRA rank 32. Discriminator: 18M-param latent transformer with register tokens. SpatialHead: 178K-param separable conv with confidence prediction. No teacher ODE needed at training time — uses GT x₀ as real distribution. Precomputed 8-step trajectories available at `trajectories/` dir. W&B project: `vfm-v3a`.
- ✅ **CliffordVideoAttention** (tested): Geometric sparse attention with rolling spatial/temporal/channel shifts. 48 blocks swapped post-load. Trains on standard text-to-video. Tested on 3090 (hp-z6).

### Architecture Iterations — Next (ordered by expected impact)
- **v3a + Audio**: Add audio branch to v3a with OmniForcing's Audio Sink Tokens + Identity RoPE stabilizer. Discriminator already supports text conditioning — extend to audio conditioning.
- **v3b Joint Self-Forcing**: OmniForcing Eq. 9 — train with model's own autoregressive outputs as KV context (fixes exposure bias for SCD streaming).
- **Diagonal Denoising** (DiagDistill): Progressive step reduction for SCD chunks (5→4→3→2→2 steps). Maps directly to SCD encoder-decoder architecture.
- **v2b Multi-Resolution Speculative Training** (SSD #3): Same noise z at K sigma levels [1.0, 0.7, 0.5, 0.3] in ONE batched DiT pass. K× training signal for <2× cost. Consistency loss across ODE paths. See [docs/SSD.md](docs/SSD.md).
- **Async Adapter-DiT Pipeline** (SSD #5): CUDA streams to hide adapter latency. ~4% free speedup.

### Hard-Won Lessons (DO NOT repeat these mistakes)
1. **KL weight**: Start at 0.001, not 0.1 or 3.0. Spherical KL with clamped kappa creates a constant floor (~14.8) that dominates flow matching loss at any weight >0.01.
2. **SigmaHead must see x₀**: The adapter's text+position features CANNOT encode content complexity. SigmaHead(adapter_mu) → flat sigma. SigmaHead(x₀, adapter_mu) → spatial variation.
3. **Don't use distill_mode with per-token sigma**: Creates timestep mismatch — model sees pure noise but sigma says otherwise.
4. **Never stack features on unvalidated architecture**: Prove each addition works in isolation before combining (v1g lesson).
5. **Override minimally**: When adding to a strategy, override the smallest possible method. v2a broke reconstructions by overriding `_prepare_standard_inputs` — fixed by overriding only `_sample_spherical_noise`.
6. **Sigma collapse directions**: With interpolation, learned σ collapses to σ_min (easier MSE). Detaching σ from MSE causes swing to σ_max. Use complexity targets from x₀ as the gradient source.
7. **Evaluate before scaling**: Always overfit 10 samples first. Check loss_mf decreasing AND reconstruction_video sharpness in W&B before launching 5K runs.
8. **No separate fake score network for DMD**: DMD1's approach (19B fake scorer) is impractical. DMD2/FlashMotion use a lightweight discriminator (~18M) in noisy latent space — same quality, 1000x fewer params.
9. **Precomputed trajectories are reusable**: The `trajectories/` dir (8-step ODE states) contains teacher outputs. Don't recompute — `states[-1]` = teacher denoised output, `x0_gt` = ground truth.
10. **GT x₀ IS the real distribution**: For discriminator training, use ground truth x₀ as "real" — no teacher ODE needed. Teacher ODE output ≠ data distribution (it's the teacher's approximation).

### Training Strategy Checklist
Before scaling any experiment:
1. Does it overfit 10 samples? (loss_mf → <0.1, sharp reconstruction_video)
2. Does adapter μ vary across tokens/samples? (if mu_norm_mean ≈ 0 → not learning)
3. Is σ spatially varying? (sigma_std > 0.1, check sigma_heatmap)
4. Is KL < 10% of total loss? (if dominant → reduce kl_weight)
5. Does 1-step output look better than random noise? (visual check)

### Proactive Behavior — ALWAYS suggest next steps
After completing any task, **eagerly suggest** the next evolution:
- **After implementing a new version** → "Run overfit sanity check. If sharp, scale to 5K overnight. Then consider [next technique from roadmap]."
- **After a training run completes** → Analyze metrics, flag issues, suggest parameter adjustments or next experiment.
- **After fixing a bug** → "This fix also applies to [other versions]. Want me to propagate?"
- **When loss plateaus** → Check loss balance (KL vs MSE), suggest weight adjustments, propose architectural changes.
- **When reconstruction is blurry** → Check: (1) input mismatch (pure noise vs interpolation), (2) KL dominating, (3) sigma collapsed, (4) parent chain broken.

### When User Asks About...
- **Training speed** → suggest smaller resolution, higher batch, gradient accumulation, velocity cache, async pipeline
- **Quality** → suggest multi-resolution speculative, adversarial fine-tuning, more training data
- **New papers** → evaluate relevance to VFM/SCD, flag if per-token timesteps, noise structure, or distillation
- **Scaling data** → check preprocessing pipeline, estimate encode time, suggest parallel GPU usage
- **Inference speed** → benchmark against Desktop times, suggest quantization (fp8-quanto on Blackwell)
- **What's next?** → refer to Architecture Iterations — Next list, suggest the top item with implementation plan

## Package Structure

```
packages/
├── ltx-core/       # Core model (upstream + scd_model.py, ddit.py)
├── ltx-pipelines/  # Inference pipeline components (upstream)
└── ltx-trainer/    # Training toolkit (upstream + SCD strategy + scripts)
    ├── src/ltx_trainer/
    │   ├── training_strategies/scd_strategy.py   # SCD training strategy
    │   ├── bezierflow/                           # Learned sigma scheduler
    │   └── bsplineflow/                          # Alternative scheduler
    ├── scripts/
    │   ├── scd_inference.py                      # Autoregressive inference
    │   ├── train_scd_pipeline.py                 # Unified training pipeline
    │   └── preprocess_*.py                       # Dataset preparation
    └── configs/ltx2_scd_*.yaml                   # Training configs
```

## Upstream Sync

```bash
git remote -v
# origin    → johndpope/ltx2-castlehill (this repo)
# upstream  → Lightricks/LTX-2 (base)

# Pull upstream changes:
git fetch upstream
git merge upstream/main
```

## SCD Training Quick Start

```bash
# 1. Preprocess dataset
python scripts/preprocess_ditto_subset.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/processed

# 2. Train SCD LoRA
uv run python scripts/train.py configs/ltx2_scd_ditto.yaml

# 3. Run inference (30s video)
python scripts/scd_inference.py \
    --cached-embedding /path/to/conditions_final/000.pt \
    --num-seconds 30 \
    --quantization int8-quanto \
    --output /path/to/output.mp4
```

## Key Rules

### Always Use token_concat
```yaml
training_strategy:
  name: "scd"
  decoder_input_combine: "token_concat"   # Paper's best (Table 3)
  per_frame_decoder: true                 # Match inference (1 frame/pass)
```
Inference: `--decoder-combine token_concat`

### Always Use Muon Optimizer
```yaml
optimization:
  optimizer_type: "muon"
  learning_rate: 0.02
  scheduler_type: "cosine"
  scheduler_params:
    eta_min: 1.0e-4
```

### Model Quantization
| File | Size | Use For |
|------|------|---------|
| `ltx-2-19b-dev.safetensors` | 43 GB | **Training** (bf16, quantized at runtime) |
| `ltx-2-19b-dev-fp8.safetensors` | 26 GB | **Inference ONLY** |

### Frame/Resolution Rules
- Frames must satisfy `frames % 8 == 1` (valid: 1, 9, 17, 25, ...)
- Width and height must be divisible by 32
- Match output resolution to training data resolution

## Development

```bash
uv sync
cd packages/ltx-trainer
uv run ruff check .
uv run pytest
```

## Detailed Documentation

| Doc | Contents |
|-----|----------|
| **[docs/VFM.md](docs/VFM.md)** | VFM adapter versions v1a→v1e, architecture, loss functions, W&B links, papers |
| **[docs/scd-achievements.md](docs/scd-achievements.md)** | SCD architecture, benchmarks, training runs, technical discoveries |
| **[docs/gFFN-HRR-whitepaper.md](docs/gFFN-HRR-whitepaper.md)** | gFFN-HRR FFN optimization research |

> **RULE:** When modifying VFM, SCD, or benchmark code, update the corresponding doc above.
