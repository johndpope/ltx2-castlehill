# Grok Context — CastleHill VFM + Audio, 2026-04-23

## What We're Building

**CastleHill** trains VFM (Variational Flow Maps) on top of LTX-2.3 (Lightricks, 22B param joint audio-video DiT).

Goal: replace LTX-2.3's 8-step Euler ODE with **1-step generation** by training a noise adapter
`q_φ(z|y)` that produces structured noise instead of sampling `z ~ N(0,I)`.

Secondary goal now in scope: **Audio + Image-to-Video talking head** — given a face image + text,
generate a synchronized audio-video clip in 1 step.

---

## Hardware

| nvidia-smi | PyTorch | GPU | VRAM |
|-----------|---------|-----|------|
| GPU 0 | cuda:1 | RTX PRO 4000 Blackwell | 24 GB |
| GPU 1 | cuda:0 | RTX 5090 | 32 GB |

Training: transformer → cuda:0 (5090), VAE → cuda:1 (4000).

---

## LTX-2.3 Architecture (base model)

- 48-layer DiT with joint video + audio token streams
- Video tokens: patchified latents [B, seq, 128], from 32× spatially / 8× temporally compressed VAE
- Audio tokens: patchified latents [B, T, 128], from separate audio VAE (mel spectrogram → latents)
- **Cross-modal attention every block**: video attends to audio and vice versa
- Text conditioning: Gemma 3840-dim → 4096-dim projection → cross-attention
- Trained on {video only, audio+video} pairs via flow matching (8-step Euler at inference)
- Model path: `/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors`

---

## VFM v4a Architecture (our addition)

```
text_embeddings [B, seq_text, 4096]
    └─→ NoiseAdapterV1b (19.2M params, 4-layer transformer)
            ├─→ mu      [B, seq_video, 128]
            └─→ log_sigma [B, seq_video, 128]

z = mu + exp(log_sigma) * eps    # Gaussian reparameterization
sigma = SigmaHead(mu, x0)        # [B, seq_video] ∈ [0.05, 0.95]
                                 # zero-init, takes x0 for content complexity

noisy_video = z                  # pure adapter noise (NOT interpolated)
video_target = z - x0            # flow velocity

DiT-22B [LoRA r=32 on to_q, to_k, to_v, to_out.0]
    input: noisy_video at timestep=sigma
    output: v_pred

x_hat = z - v_pred               # 1-step reconstruction

Loss = MSE(v_pred, z - x0)  +  kl_weight * KL(N(mu, exp(log_sigma)^2) || N(0,1))
```

**NoiseAdapterV1b** (in `ltx_core/model/transformer/noise_adapter_v1b.py`):
- 4-layer transformer, hidden_dim=512, 8 heads
- Input: text_embeddings (4096-dim, pooled) + spatial positions (256-dim PE)
- Output: (mu, log_sigma) for each video token
- 19.2M parameters, trained at 5e-4 LR alongside DiT LoRA at 2e-4

**SigmaHead** (in `vfm_v4a_standalone.py`, 131K params):
- 3-layer MLP: [256+128 → 256 → 256 → 1]
- Input: concat(x0.detach(), mu) — x0 provides content complexity signal
- Output: per-token sigma ∈ [sigma_min=0.05, sigma_max=0.95] via sigmoid
- Zero-initialized (starts predicting sigma=0.5 uniformly)
- Critical: must take x0 as input. Adapter features (text+position only) cannot encode content
  complexity → flat sigma collapse if using mu alone

---

## Audio Extension (just implemented, 2026-04-23)

We added `with_audio: bool` to `VFMv4aConfig` and `_prepare_audio_inputs` to `VFMv4aStrategy`.

**Design choice — Option A (implemented):**
- Video: structured noise from adapter `z ~ q_φ(z|y)`, pure noise input
- Audio: standard flow matching `noisy_audio = (1-σ)*a0 + σ*eps`, same `batch_sigma` as video
- Audio targets: `eps - a0` (standard velocity)
- The DiT's cross-modal attention couples video ↔ audio denoising

```python
# Audio gets standard interpolation at same sigma as adapter video
sigma_exp = batch_sigma.view(-1, 1, 1)
noisy_audio = (1 - sigma_exp) * a0 + sigma_exp * eps
audio_targets = eps - a0
audio_timesteps = batch_sigma.view(-1, 1).expand(-1, audio_seq_len)
```

**Rationale**: Audio stays in the same distribution it was pretrained on (standard Gaussian flow
matching). Video gets the adapter speedup. Cross-modal attention already present in LTX-2.3
handles audio-video synchronization.

**Open question for Grok**: Is 1-step audio generation from Gaussian noise feasible given the
base model was trained with 8 steps? Or do we need an audio adapter (`q_φ_audio(z_audio|y)`)
analogous to the video adapter? Would the audio quality at 1-step be acceptable for talking
head use cases?

---

## Training Results

### Overfit-10 (completed, validation run)
- Config: `ltx2_vfm_v4a_overfit10.yaml`, 5000 steps, 10 samples
- Final loss_mf: **0.032**
- Result: all 10 training samples reconstruct cleanly (1.2s total: 348ms DiT + 940ms VAE)
- W&B: `wandb.ai/snoozie/vfm-v4a`
- Key finding: proves VFM v4a standalone architecture works

### Full dataset run (running now, step ~8000/15000)
- Config: `ltx2_vfm_v4a_full.yaml`, 15000 steps, 5000 samples (Ditto-1M subset)
- Dataset: `/media/12TB/ddit_ditto_data_23` — **all talking head videos, no audio**
- W&B: `wandb.ai/snoozie/vfm-v4a/runs/k9uhh7dj`
- Expected issue: domain-locked to talking heads, won't generalize to OOD prompts

### Scrya audio-video overfit (pending, preprocessing in progress)
- 54 clips, ~6s each, 24fps, AAC 44100Hz
- Source: `/home/johndpope/scrya-downloads/Realistic Photo/`
- Resolutions: 560×560 (50 clips) → 544×544, 464×688 (4 clips) → 448×672
- Pre-processing: `preprocess_scrya.py --phase latents` (running on cuda:1)
- OOM hit: VAE tried to allocate 17.27 GiB for 544×544×145 frames on 24GB GPU
- Next: fix with `vae_tiling=True` or reduce to 544×544×97 frames

---

## Three Hard-Won Implementation Bugs (from overfit-10 cycle)

### Bug 1: LoRA key format mismatch
Trainer saves: `diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight`
PEFT expects:  `base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight`

`set_peft_model_state_dict` silently loaded 0/2304 keys. Fix: manual remapping + `load_state_dict(strict=False)`.

### Bug 2: merge_and_unload() + quanto = silent discard
`quanto`'s `QLinear` (int8) cannot absorb bf16 LoRA delta in-place. `merge_and_unload()` silently
discards LoRA weights. Fix: remove `merge_and_unload()` call, keep PEFT hooks active at inference.

### Bug 3: isinstance ordering
`VFMv4aStrategy` was not in the trainer's `isinstance` checks for adapter/sigma head routing.
Model trained as pure flow matching for 5000 steps. Fix: add to all relevant isinstance checks.

---

## Key Code Files

**Strategy**: `packages/ltx-trainer/src/ltx_trainer/training_strategies/vfm_v4a_standalone.py`
```python
# Full file content below (446 lines):
"""VFM v4a Standalone — Gaussian adapter, NO inheritance from v1f/v3b chain.

Architecture:
    text_embeddings → NoiseAdapterV1b → (mu, log_sigma)
    z = mu + exp(log_sigma) * eps        [Gaussian reparameterization]
    sigma = SigmaHead(mu, x0)            [per-token, [0.05, 0.95]]
    DiT(latent=z, timesteps=sigma) → v   [LoRA r=32]
    x_hat = z - v
    loss = MSE(v, z - x0) + kl_weight * GaussianKL(mu, log_sigma)
"""
# ... [full code as above in VFMv4aStrategy section]
```

**Inference script**: `packages/ltx-trainer/scripts/vfm_vanilla_inference.py`
- Loads NoiseAdapterV1b + SigmaHead + LoRA (with manual key remapping)
- Runs 1-step: z=adapter(text) → DiT(z, sigma) → v → x_hat=z-v → VAE decode
- Wall-clock: 1.2s total (348ms DiT + 940ms VAE decode)

**Audio preprocessing**: `packages/ltx-trainer/scripts/preprocess_scrya.py`
- Scans scrya directory for UUID.mp4 + UUID.txt pairs
- Calls `compute_latents(with_audio=True)` → video + audio VAE latents
- Then `compute_captions_embeddings` → conditions_final/

---

## Current Open Questions for Grok

### Q1: Audio quality at 1-step
The base LTX-2.3 generates audio in 8 steps from Gaussian noise. Our VFM does 1 step.
Video gets structured adapter noise (close to answer). Audio gets random Gaussian (far from answer).
**Will the audio tokens denoised in 1 step produce intelligible/usable audio?**
Evidence either way from other 1-step models (SDXL-Turbo, LCM) suggests 1-step from structured
noise is fine, but 1-step from pure Gaussian is much harder for complex signals like speech.

### Q2: Audio adapter design
If Option A (random audio noise) gives poor quality, what's the minimal architecture for
`q_φ_audio(z_audio|y)`? The video adapter (NoiseAdapterV1b) uses spatial positions as input.
Audio positions are 1D (time steps). Could we reuse the same adapter with audio positions?
Or should audio noise be conditioned on the generated video tokens (face→audio sync)?

### Q3: Convergence with 5K talking head dataset
The full run has 5K Ditto-1M talking head clips. At step 8000/15000, we don't have loss values
(they go to W&B only). The fundamental question: can VFM converge to a useful talking head
specialist with 5K samples, or is 50K+ needed? What's the minimum dataset size for VFM to
generalize within a domain (not OOD)?

### Q4: 1-step talking head vs multi-step quality gap
At what training step count does 1-step VFM output quality become acceptable for talking head
generation? The overfit-10 at 5K steps gives loss=0.032 (10 samples memorized). For 5K samples,
what loss threshold indicates useful generalization?

### Q5: Cross-modal synchronization without audio adapter
Given the DiT's cross-modal attention, if video is generated well in 1 step, will audio
automatically synchronize to lip movements? Or does audio need its own structured noise
to achieve synchronization? Any papers on joint 1-step A+V generation?

---

## Roadmap (ordered by expected impact)

1. **Fix preprocessing OOM** — use `vae_tiling=True` or reduce to 97 frames (3.8s clips)
2. **Run scrya overfit** — prove audio pipeline trains without errors
3. **Inference with audio** — extend `vfm_vanilla_inference.py` to decode + save audio
4. **Evaluate 1-step audio quality** — is it intelligible?
5. **Audio adapter (if needed)** — Option B: extend NoiseAdapterV1b for audio tokens
6. **Scale to full Ditto-1M** — 1M talking head clips with audio (if available)

---

## Repo

`github.com/johndpope/ltx2-castlehill`, branch `feature/vfm-v1b`

Whitepaper: `paper/vfm_scd.pdf` (7 pages, covers architecture + 3 implementation bugs + results)
