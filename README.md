# dllm-castlehill

**Diffusion research across two modalities: video DiT and discrete language diffusion.**

Forked from [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) (video) and [Open-dLLM](https://github.com/scrya-com/Open-dLLM) (language). Both converge on the same core idea: replace autoregressive next-token prediction with iterative denoising.

---

## Research Network

```
                    DIFFUSION RESEARCH
                          |
          ┌───────────────┴───────────────┐
          │                               │
   VIDEO DiT (continuous)        LANGUAGE (discrete)
   LTX-2.3 22B DiT                Qwen3/3.5/3.6 AR → DLM
          │                               │
    ┌─────┴─────┐              ┌──────────┴──────────┐
    │           │              │                     │
   SCD         VFM         Repr-Align             LDLM
(streaming) (1-step)    (bidirectional        (Perceiver +
    │           │          finetune)            DiT head)
    │           │              │                     │
RotorQuant  v1a→v3b        Cola DLM              LDLM-35B
(KV cache)  (archived)    (latent head)          A3B MoE
                               │
                    ┌──────────┼──────────┐
                    │          │          │
                 CARD      FastBlock  Block-Causal
               (soft-tail) (AR-spec)   (original)
                                          │
                                       MTP Head
                                    (Qwen3.5 MoE)
```

---

## Track 1 — Video DiT (LTX-2.3 22B)

### SCD — Separable Causal Diffusion
*Autoregressive streaming video: 30s+ on consumer GPU.*

Split the 48-layer DiT into encoder(32) + decoder(16). KV-cache the encoder output across temporal chunks. Enables streaming video without recomputing the full 22B model per chunk.

**Results:** 30s video on single RTX 5090 via SCD inference.

```
LTX-2.3 DiT (48 layers)
    ├── Encoder: layers 0–31 → KV-cache (RotorQuant compressed)
    └── Decoder: layers 32–47 → per-chunk autoregressive
```

| Doc | |
|-----|-|
| [docs/scd-achievements.md](docs/scd-achievements.md) | Architecture, benchmarks, KV layout |

### VFM — Variational Flow Maps
*1-step video generation (archived; pivoted to consistency distillation).*

| Version | What changed | Status |
|---------|-------------|--------|
| v1a | MLP adapter + Gaussian noise | baseline |
| v1b | Transformer adapter (38M, cross-attn to text) | |
| v1d | Per-token sigma + trajectory distillation | |
| v1f | Spherical Cauchy noise + anti-collapse losses | W&B: [kxks35j8](https://wandb.ai/snoozie/vfm-v1f/runs/kxks35j8) |
| v3a | DMD2 adversarial (LatentDiscriminator 18M) | W&B: [ev21ymgl](https://wandb.ai/snoozie/vfm-v3a/runs/ev21ymgl) |
| v3b | Self-E distillation (no discriminator) | W&B: [hvrpf0ed](https://wandb.ai/snoozie/vfm-v3b/runs/hvrpf0ed) |

**Post-mortem:** Spherical Cauchy noise + LoRA = distribution mismatch. LoRA (~1% of 22B params) cannot remap a Gaussian-pretrained DiT to Cauchy noise. Pivoting to consistency distillation (Gaussian noise, MSE, full fine-tune or large LoRA rank).

### RotorQuant KV Cache
*Clifford rotor rotation + Lloyd-Max quantization for SCD KV cache.*

| Metric | 3-bit | 4-bit |
|--------|-------|-------|
| K/V cosine similarity | 0.971 | **0.993** |
| Top-1 attention match | 60% | **79%** |
| Cache size (20f × 8L) | 332 MB | 416 MB vs 1.25 GB FP16 |
| Peak VRAM | **1.09 GB** | 1.09 GB |
| Compress speed | 36ms/frame | 36ms/frame |

v0a: overhead for vanilla attention (no VRAM win). v0b: correct integration into SCD KV cache.

---

## Track 2 — Discrete Language Diffusion (Open-dLLM)

Converting autoregressive LMs (Qwen3/3.5/3.6) to masked diffusion LMs via three paths:

### Path 1 — Repr-Align
*Flip attention mask bidirectional + cosine alignment to frozen causal teacher.*

No new parameters. Reuses existing weights. 3–4× faster convergence than MDM training from scratch. Cosine-sim alignment loss against precomputed frozen teacher hidden states (layers 7, 14, 21, 28).

```
Frozen causal teacher (precomputed anchors on disk)
           ↕ cosine-sim alignment loss
Bidirectional student (same weights, mask flipped)
           ↕ MDM masked diffusion loss
```

**Key insight:** Teacher is a frozen anchor, not a live distillation source. Precompute hidden states once (`scripts/precompute_anchor.py`), cache to disk, reuse forever.

Supported models: Qwen3-1.7B, Qwen3-8B, Qwen3.6-27B, Qwen3.6-35B-A3B.

### Path 2 — LDLM (Latent Diffusion LM)
*Perceiver encoder/decoder + DiT head on top of frozen AR encoder.*

1.39B–6.75B trainable params on top of frozen Qwen3.5/3.6 encoder. Latent-space diffusion over Perceiver compressed representations. Encoder deleted at inference; only DiT head runs.

```
Frozen AR encoder (Qwen3.6)  →  Perceiver encoder  →  z_latent
                                    DiT denoiser  ↔  z_noisy
                                 Perceiver decoder  →  token logits
```

Configs: `configs/pretrain/qwen3_6_27b_ldlm.yaml`, `configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml`.

### Path 3 — Cola DLM (opt-in auxiliary head on Repr-Align)
*Hierarchical Text VAE + block-causal DiT denoiser as auxiliary head.*

Adds a TextVAE encoder (Perceiver → `z_global` + `z_local`) and a block-causal DiT denoiser on top of Repr-Align hidden states. Three head variants:

```
Repr-Aligned LM hidden states  (cola_source_layer = -3)
             │
    TextVAEEncoder
    ┌──────────────────────┐
    │  Global Perceiver    │  → z_global (G tokens)
    │  Local Perceiver     │  → z_local  (L tokens)
    └──────────────────────┘
             │ z = [z_global ; z_local]
             │
    ┌────────┴────────────────────────────┐
    │                                     │
  CARD                   FastBlock              Block-Causal
  (soft-tail)            (AR-friendly)          (original)
  causal attn            complementary          bidirectional
  tail = λ·N·t          masks + token-shift    within blocks
  loss on tail only      speculative decode     causal across
  [arXiv:2605.06548]
```

**CARD** (`cola_variant: card`): Strictly causal attention + soft-tailed masking. Only the tail of the sequence is noisy at timestep t; prefix provides clean context anchors. `cola_lambda_tail` (0.5–1.0) controls aggressiveness.

**FastBlock** (`cola_variant: fast_block`): Block-causal + complementary masks + token-shift. Most AR-compatible variant — latents shift one position to mimic autoregressive prefix conditioning. Best candidate for speculative decoding.

**Block-Causal** (`cola_variant: block_causal`): Original variant. Bidirectional within blocks, causal across. Balanced stability/quality.

### MTP Head (Qwen3.5/3.6 MoE)
*Multi-token prediction à la Qwen3.6: auxiliary next-N head for speculative decode speedup.*

Lightweight (1 decoder layer + RMSNorm + lm_head). Enabled via `mtp_num_layers > 0` in `Qwen3_5MoeConfig`. Adds auxiliary loss to `Qwen3_5MoeForCausalLM.forward()`. ~300–800 MB extra VRAM; enables 1.5–3× speculative decode speedup.

---

## Papers Implemented

### Video DiT
| Paper | Method | Used in |
|-------|--------|---------|
| [VFM](https://arxiv.org/abs/2603.07276) — Mammadov 2026 | Noise adapter + flow map for 1-step | v1a–v1f |
| [Self-Flow](https://arxiv.org/abs/2603.06507) | Per-token timestep scheduling | v1d+ |
| [Self-E](https://arxiv.org/abs/2512.22374) — Yu 2025 | Self-evaluating distillation | v3b |
| [DMD2](https://arxiv.org/abs/2405.14867) — Yin 2024 | GAN in noisy latent space | v3a |
| [FlashMotion](https://arxiv.org/abs/2603.12146) — CVPR 2026 | Adversarial post-training for video | v3a |
| [DiagDistill](https://arxiv.org/abs/2603.09488) — ICLR 2026 | Flow Distribution Matching + SpatialHead | v3a flow loss |
| [OmniForcing](https://arxiv.org/abs/2603.11647) | Audio Sink Tokens, Joint Self-Forcing | Planned (SCD+audio) |
| [Chain-of-Steps](https://arxiv.org/abs/2603.16870) | Multi-path ensemble at inference | `--ensemble K` flag |

### Discrete LLM Diffusion
| Paper | Method | Used in |
|-------|--------|---------|
| [MDM](https://arxiv.org/abs/2406.07524) — Shi 2024 | Masked diffusion LM | core MDM loss |
| [Cola DLM](https://arxiv.org/abs/2605.06548) | Hierarchical Text VAE + block-causal DiT | Cola heads |
| [Repr-Align](https://arxiv.org/abs/2502.09992) | Bidirectional finetune + alignment loss | Repr-Align path |
| [MDLM](https://arxiv.org/abs/2406.11473) — Sahoo 2024 | Masked diffusion with absorbing state | loss baselines |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Hybrid linear/full attention | Qwen3.5/3.6 arch |

---

## Measured Results

### Video Inference Speed (RTX 5090 / Blackwell)
| Method | DiT passes | Wall clock | Speedup |
|--------|-----------|------------|---------|
| LTX-2 8-step baseline | 8 | ~4.0s | 1× |
| Consistency 1-step | 1 | ~0.5s | **8×** |
| VFM 1-step (v1a–v3b, archived) | 1 | 0.45s | 8.9× (low quality) |

### Cola DLM Smoke Test (Qwen3-1.7B + CARD, step 1)
| Metric | Value |
|--------|-------|
| MDM loss | 16.88 |
| Repr-align loss | 0.70 |
| Cola diff loss | 837.0 (random init, expected) |
| GPU VRAM (peak) | 9.41 GB |
| Step time | ~8.3s |
| Platform | RTX 5090, ZeRO-2 + CPU optimizer offload |

---

## Active Configs

| Config | Description |
|--------|-------------|
| `configs/pretrain/qwen3_1_7b_cola_card_smoke.yaml` | 1.7B CARD smoke test, 100 steps |
| `configs/pretrain/qwen3_6_27b_cola_card.yaml` | 27B CARD, cloud (2× RTX PRO 6000 Blackwell) |
| `configs/pretrain/qwen3_6_35b_a3b_cola_card.yaml` | 35B-A3B CARD |
| `configs/pretrain/qwen3_6_27b_cola_fast_block.yaml` | 27B FastBlock |
| `configs/pretrain/qwen3_6_27b_cola_mtp.yaml` | 27B + MTP head |
| `configs/pretrain/cloud_27b.yaml` | 27B Repr-Align, Vast.ai cloud |

---

## Known Bugs Fixed

### Qwen3.6-27B (qwen3_5 architecture)
- **Gated DeltaNet NaN backward** — 75% linear attention layers fall back to NaN-prone torch impl without `flash-linear-attention` + `causal-conv1d`. Install: `uv pip install causal-conv1d flash-linear-attention` (build from source for SM 12.0).
- **Weight key prefix mismatch** — released weights use `model.language_model.*`; veomni uses `model.*`. Fixed via `_remap_key()` in `load_hf_weights_zero3`.
- **`head_dim` override** — `Qwen3Config.__init__` recomputes `head_dim = hidden_size // num_heads` (128), overriding the correct 256. Fixed by restoring `_head_dim` after `super().__init__()`.
- **RoPE partial rotation shape** — `apply_rotary_pos_emb_partial` did `transpose(1,2)` expecting `(B, rotary_dim, S)` input; actual shape is `(B, S, rotary_dim)`. Fixed to `unsqueeze(1)` for head broadcast.

### Cola DLM
- **`cola_head` CPU/GPU device mismatch** — `build_cola_head()` creates on CPU; must call `.to(lm_device)` before `ColaReprAlignWrapper`.
- **`PerceiverResampler` latents on CPU** — `latents.expand(...).to(context.dtype)` missing `.to(context.device)`.
- **`zero.Init()` used for ZeRO-2** — ZeRO-3 only; ZeRO-2 replicates weights like DDP, must skip meta init and `zero.Init()`.
- **`get_global_grad_norm()` returns None** — ZeRO-2 + CPU optimizer doesn't expose grad norm; default to `0.0`.

---

## Lessons Learned

1. **Spherical Cauchy + LoRA = distribution mismatch** — rank-32 LoRA (1–2% of 22B params) cannot remap a Gaussian-pretrained DiT to Cauchy noise; corrupts base model at all step counts.
2. **Training loss convergence ≠ inference quality** — v3b overfit to 10 samples (loss 0.04) but produced garbage at inference. Test inference early.
3. **Alpha mixing creates train/test mismatch** — training with alpha=0.5 (50% Gaussian / 50% Cauchy) then inferring with pure Cauchy guarantees failure.
4. **Repr-Align teacher is a frozen anchor** — not a live distillation source. Precompute hidden states once, cache to disk, reuse forever. No RPC infrastructure needed.
5. **RotorQuant works** — 4-bit KV cache at 0.993 cosine fidelity is production-ready for SCD.
6. **Vast.ai spot bids don't cap cost** — without `--price` flag, Vast.ai auto-upgrades to on-demand when host raises price. Always pass `VAST_MAX_BID=x.xx` to provisioning scripts.

---

## Quick Start

```bash
# Discrete LLM diffusion — 1.7B CARD smoke test
cd Open-dLLM
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc_per_node=1 \
    tasks/train_torch.py configs/pretrain/qwen3_1_7b_cola_card_smoke.yaml

# Repr-Align anchor precompute (run once before training)
python scripts/precompute_anchor.py \
    --model_path Qwen/Qwen3-1.7B \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/anchors/qwen3-1.7b \
    --layers 7,14,21,28 --max_seq_len 2048

# SCD inference — 30s+ video (LTX-2.3)
python packages/ltx-trainer/scripts/scd_inference.py \
    --cached-embedding data/conditions_final/000000.pt \
    --num-seconds 30 --quantization int8-quanto --output output.mp4

# Provision Vast.ai instance (capped at $1/hr)
VAST_MAX_BID=1.00 bash Open-dLLM/scripts/cloud/launch_vast.sh
```

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/scd-achievements.md](docs/scd-achievements.md) | SCD architecture, benchmarks, training runs |
| [docs/VFM.md](docs/VFM.md) | VFM adapter v1a–v3b, post-mortem |
| [docs/v3b-architecture.md](docs/v3b-architecture.md) | v3b Self-E state diagram |
| [Open-dLLM/CLAUDE.md](https://github.com/scrya-com/Open-dLLM/blob/main/CLAUDE.md) | Full discrete diffusion architecture guide |
| [Open-dLLM/docs/cola_ldm.md](https://github.com/scrya-com/Open-dLLM/blob/main/docs/cola_ldm.md) | Cola DLM head variants |
| [Open-dLLM/docs/cloud_training.md](https://github.com/scrya-com/Open-dLLM/blob/main/docs/cloud_training.md) | Vast.ai provisioning guide |
