# CastleHill: SCD + VFM for LTX-2

**CastleHill** extends [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) with **1-step video generation** (VFM) and **long-form streaming** (SCD).

**Key results:**
- **0.5s generation** for 9 frames (vs 4s baseline 8-step) on RTX 5090
- **8x faster** than standard LTX-2 inference
- **Distribution-level training** via Self-E + adversarial DMD

## Experiment Network Graph

```
LTX-2 (22B DiT, Lightricks)
    |
    +-- SCD (Separable Causal Diffusion)
    |   |   Split 48-layer DiT into encoder(32) + decoder(16)
    |   |   KV-cache for autoregressive streaming (30s+ video)
    |   +-- docs/scd-achievements.md
    |
    +-- VFM (Variational Flow Maps) ---- 1-step video generation
        |
        +-- v1a  Baseline MLP adapter + Gaussian noise
        |
        +-- v1b  Transformer adapter (cross-attn to text, 38M params)
        |
        +-- v1d  + Per-token sigma (SigmaHead predicts per-token noise level)
        |         + Trajectory distillation (8-step ODE precomputed)
        |
        +-- v1f  + Spherical Cauchy noise (direction-magnitude on S^127)  <-- VALIDATED
        |   |      + Anti-collapse: obs_loss=25, mu_align=5, diversity
        |   |      + W&B: https://wandb.ai/snoozie/vfm-v1f
        |   |
        |   +-- v1f anticollapse (LTX-2.3 22B, 5K dataset)
        |       W&B: https://wandb.ai/snoozie/vfm-v1f/runs/kxks35j8
        |
        +-- v3a  DMD2 Adversarial (GAN discriminator in noisy latent space)
        |   |    + LatentDiscriminator (18M params, register tokens)
        |   |    + Flow Distribution Matching (SpatialHead, DiagDistill)
        |   |    + W&B: https://wandb.ai/snoozie/vfm-v3a
        |   |
        |   +-- v3a overfit-10 (proof of concept)
        |       W&B: https://wandb.ai/snoozie/vfm-v3a/runs/ev21ymgl
        |
        +-- v3b  Self-E (Self-Evaluating Model) ---- MOST PROMISING  <--
            |    No discriminator! Model evaluates own outputs via
            |    conditional vs unconditional classifier score.
            |    + Latent perceptual loss (multi-scale cosine + L1)
            |    + obs_loss + mu_align + diversity (from v1f)
            |    + Energy-preserving normalization
            |    + W&B: https://wandb.ai/snoozie/vfm-v3b
            |
            +-- v3b overfit-10 (converged, loss_data=0.04)
            |   W&B: https://wandb.ai/snoozie/vfm-v3b/runs/hvrpf0ed
            |   W&B: https://wandb.ai/snoozie/vfm-v3b/runs/ib67nuyf
            |
            +-- v3b 5K dataset (scaling from overfit)
            |   W&B: https://wandb.ai/snoozie/vfm-v3b/runs/kyubxb40
            |
            +-- v3b 5K + obs_loss (CURRENT, all losses at 100%)
                W&B: https://wandb.ai/snoozie/vfm-v3b/runs/cgxdj1g6

    Explored & Rejected:
        x-- v1e  Content-adaptive routing (unvalidated complexity)
        x-- v1g  HyperSphereDiff (too many fighting loss terms)
        x-- v1h  Integrated adapter sigma (adapter can't encode x0 complexity)
        x-- v2a  Speculative noise selection (unimodal, no diversity)
```

## Papers Implemented

| Paper | What we took | Version |
|-------|-------------|---------|
| [VFM](https://arxiv.org/abs/2603.07276) (Mammadov 2026) | Noise adapter + flow map for 1-step | v1a-v1f |
| [Self-Flow](https://arxiv.org/abs/2603.06507) | Per-token timestep scheduling | v1d+ |
| [Self-E](https://arxiv.org/abs/2512.22374) (Yu 2025) | Self-evaluating distillation (no discriminator) | **v3b** |
| [DMD2](https://arxiv.org/abs/2405.14867) (Yin 2024) | GAN in noisy latent space | v3a |
| [FlashMotion](https://arxiv.org/abs/2603.12146) (CVPR 2026) | Adversarial post-training for video | v3a |
| [DiagDistill](https://arxiv.org/abs/2603.09488) (ICLR 2026) | Flow Distribution Matching + SpatialHead | v3a flow loss |
| [OmniForcing](https://arxiv.org/abs/2603.11647) | Audio Sink Tokens, Joint Self-Forcing | Planned (SCD+audio) |
| [Chain-of-Steps](https://arxiv.org/abs/2603.16870) | Multi-path ensemble at inference | `--ensemble K` flag |

## Measured Inference Speed (RTX 5090, int8-quanto)

| Method | DiT passes | Wall clock | Speedup |
|--------|-----------|------------|---------|
| LTX-2 8-step baseline | 8 | ~4.0s | 1x |
| **VFM 1-step** | **1** | **0.5s** | **8x** |
| VFM 2-pass (SigmaHead) | 2 | ~1.0s | 4x |
| VFM ensemble K=3 | 3 | ~1.5s | 2.7x |

## Architecture (v3b — Current Best)

```
Text prompt -> Gemma -> Connector (3840->4096)
                           |
                    NoiseAdapterV1b (38M)
                    Spherical Cauchy: mu, kappa
                           |
                    z ~ q_phi(z|text)
                           |
                    DiT 22B (LoRA r=32) -- 1 forward pass
                           |
                    x_hat_0 = z - v_pred
                           |
            +-------+------+------+--------+
            |       |      |      |        |
         data_loss  obs  self-E  percept  mu_align
         |x-x0|^2  noisy  cond   multi   cos(mu,x0)
                    recon  vs     scale
                           uncond
```

## Quick Start

```bash
# Train VFM v3b (Self-E, 1-step generation)
uv run python packages/ltx-trainer/scripts/train.py \
    packages/ltx-trainer/configs/ltx2_vfm_v3b_self_e_5k_obs.yaml

# Inference (1-step, 0.5s per clip)
python packages/ltx-trainer/scripts/vfm_vanilla_inference.py \
    --adapter-path checkpoints/noise_adapter_step_10000.safetensors \
    --lora-path checkpoints/lora_weights_step_10000.safetensors \
    --cached-embedding data/conditions_final/000000.pt \
    --adapter-variant v1b --output output.mp4

# Two-pass inference (sharper, 1.0s)
python scripts/vfm_vanilla_inference.py ... --two-pass

# Ensemble inference (3 reasoning paths, 1.5s)
python scripts/vfm_vanilla_inference.py ... --ensemble 3
```

## Roadmap

- **v3b + 5K dataset** (in progress) - Scale from overfit to diverse prompts
- **v3b + Audio** - OmniForcing Audio Sink Tokens + Identity RoPE
- **SCD + VFM** - 1-step decoder per chunk for real-time streaming at 25 FPS
- **Diagonal Denoising** - Progressive step reduction (5->4->3->2 per chunk)
- **CliffordVideoAttention** - Geometric sparse attention (17x fewer self-attn FLOPs)

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/VFM.md](docs/VFM.md) | VFM adapter versions v1a-v3b, architecture, losses |
| [docs/v3b-architecture.md](docs/v3b-architecture.md) | v3b Self-E state diagram, papers, weaknesses |
| [docs/scd-achievements.md](docs/scd-achievements.md) | SCD architecture, benchmarks, training runs |

See [docs/scd-achievements.md](docs/scd-achievements.md) for SCD documentation.

---

# LTX-2 (Upstream)

[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-2-playground/i2v)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/ltxplatform)

**LTX-2** is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one model: synchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, API access, and open access.

<div align="center">
  <video src="https://github.com/user-attachments/assets/4414adc0-086c-43de-b367-9362eeb20228" width="70%" poster=""> </video>
</div>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# Set up the environment
uv sync --frozen
source .venv/bin/activate
```

### Required Models

Download the following models from the [LTX-2.3 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2.3):

**LTX-2.3 Model Checkpoint** (choose and download one of the following)
  * [`ltx-2.3-22b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors)
  * [`ltx-2.3-22b-distilled.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors)

**Spatial Upscaler** - Required for current two-stage pipeline implementations in this repository
  * [`ltx-2.3-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors)
  * [`ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors)

**Temporal Upscaler** - Supported by the model and will be required for future pipeline implementations
  * [`ltx-2.3-temporal-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors)

**Distilled LoRA** - Required for current two-stage pipeline implementations in this repository (except DistilledPipeline and ICLoraPipeline)
  * [`ltx-2.3-22b-distilled-lora-384.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-lora-384.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors)

**Gemma Text Encoder** (download all assets from the repository)
  * [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)

**LoRAs**
  * [`LTX-2.3-22b-IC-LoRA-Union-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors)
  * [`LTX-2.3-22b-IC-LoRA-Motion-Track-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/resolve/main/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors)
  * [`LTX-2-19b-IC-LoRA-Detailer`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors)
  * [`LTX-2-19b-IC-LoRA-Pose-Control`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-In`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Left`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Out`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Right`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Down`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Up`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Static`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors)

### Available Pipelines

* **[TI2VidTwoStagesPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)** - Production-quality text/image-to-video with 2x upsampling (recommended)
* **[TI2VidTwoStagesHQPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py)** - Same two-stage flow as above but uses the res_2s second-order sampler (fewer steps, better quality)
* **[TI2VidOneStagePipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)** - Single-stage generation for quick prototyping
* **[DistilledPipeline](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)** - Fastest inference with 8 predefined sigmas
* **[ICLoraPipeline](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)** - Video-to-video and image-to-video transformations (uses distilled model.)
* **[KeyframeInterpolationPipeline](packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)** - Interpolate between keyframe images
* **[A2VidPipelineTwoStage](packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py)** - Audio-to-video generation conditioned on an input audio file
* **[RetakePipeline](packages/ltx-pipelines/src/ltx_pipelines/retake.py)** - Regenerate a specific time region of an existing video

### ⚡ Optimization Tips

* **Use DistilledPipeline** - Fastest inference with only 8 predefined sigmas (8 steps stage 1, 4 steps stage 2)
* **Enable FP8 quantization** - Enables lower memory footprint: `--quantization fp8-cast` (CLI) or `quantization=QuantizationPolicy.fp8_cast()` (Python). For Hopper GPUs with TensorRT-LLM, use `--quantization fp8-scaled-mm` for FP8 scaled matrix multiplication.
* **Install attention optimizations** - Use xFormers (`uv sync --extra xformers`) or [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) for Hopper GPUs
* **Use gradient estimation** - Reduce inference steps from 40 to 20-30 while maintaining quality (see [pipeline documentation](packages/ltx-pipelines/README.md#denoising-loop-optimization))
* **Skip memory cleanup** - If you have sufficient VRAM, disable automatic memory cleanup between stages for faster processing
* **Choose single-stage pipeline** - Use `TI2VidOneStagePipeline` for faster generation when high resolution isn't required

## ✍️ Prompting for LTX-2

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

- Start with main action in a single sentence
- Add specific details about movements and gestures
- Describe character/object appearances precisely
- Include background and environment details
- Specify camera angles and movements
- Describe lighting and colors
- Note any changes or sudden events

For additional guidance on writing a prompt please refer to <https://ltx.video/blog/how-to-prompt-for-ltx-2>

### Automatic Prompt Enhancement

LTX-2 pipelines support automatic prompt enhancement via an `enhance_prompt` parameter.

## 🔌 ComfyUI Integration

To use our model with ComfyUI, please follow the instructions at <https://github.com/Lightricks/ComfyUI-LTXVideo/>.

## 📦 Packages

This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** - Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and other generation modes
* **[ltx-trainer](packages/ltx-trainer/)** - Training and fine-tuning tools for LoRA, full fine-tuning, and IC-LoRA

Each package has its own README and documentation. See the [Documentation](#-documentation) section below.

## 📚 Documentation

Each package includes comprehensive documentation:

* **[LTX-Core README](packages/ltx-core/README.md)** - Core model implementation, inference stack, and utilities
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** - High-level pipeline implementations and usage guides
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** - Training and fine-tuning documentation with detailed guides
