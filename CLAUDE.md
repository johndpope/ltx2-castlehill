# CastleHill: SCD for LTX-2

## Project Overview

**CastleHill** extracts Separable Causal Diffusion (SCD) from `ltx2-omnitransfer` into a clean, standalone repo based on upstream [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2).

**Focus:** Long-form video generation via SCD — autoregressive, chunked denoising that turns LTX-2's 48-layer DiT into an encoder (32 layers, run once per frame) + decoder (16 layers, run N denoising steps per frame). KV-cache on the encoder enables O(1) per-frame context accumulation.

**Key features:**
- SCD training strategy (LoRA + per-frame decoder)
- SCD autoregressive inference (30s+ videos on consumer GPUs)
- DDiT dynamic patch scheduling (optional decoder speedup)
- BézierFlow / BSplineFlow learned sigma schedulers
- Muon optimizer support

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

## SCD Architecture

```
Frame N generation:

ENCODER (32 layers, once per frame, KV-cached):
  Frame N-1 (clean, σ=0) → [32 blocks] → encoder_features
                              KV-cache grows across frames

DECODER (16 layers, N_steps per frame):
  token_concat(encoder_features, noisy_frame_N) → [16 blocks] → velocity
```

- **Encoder**: O(1) per frame via KV-cache (~6% of total time)
- **Decoder**: Linear scaling with frame count (~1.3-1.7s/frame)
- **DDiT** (optional): 4× fewer tokens in decoder → 1.25-1.48× speedup
