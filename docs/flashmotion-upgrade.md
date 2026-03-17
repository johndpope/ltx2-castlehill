# FlashMotion Adversarial Post-Training — Upgrade Plan (v3a)

> **Status:** Planned. Waiting for v1f to converge first.
> **Source:** FlashMotion (CVPR 2026, arXiv:2603.12146), code at `/home/johndpope/Documents/GitHub/FlashMotion`
> **Target:** VFM v3a strategy for LTX-2/2.3

## Overview

FlashMotion demonstrates that **adversarial post-training** (hybrid diffusion + GAN) produces sharper, more accurate outputs for few-step video generation. Their Stage 3 adds a discriminator that classifies real vs fake in noisy latent space, with alternating generator/discriminator updates.

This upgrade ports FlashMotion's Stage 3 to our VFM pipeline as v3a, applied AFTER v1f has converged.

## Key Concepts from FlashMotion

### Noisy Latent Discrimination
The discriminator doesn't see clean video — it sees **noisy latent space** at a random "critic timestep." This is critical for stability:
```
t_critic ~ Uniform[0.02, 0.98]
x̂₀_noisy = (1 - t) · x̂₀ + t · ε    # fake
x₀_noisy  = (1 - t) · x₀  + t · ε    # real (same noise, same t)
D(x_noisy | t_critic) → logit
```

### Loss Functions
```
Generator:   L_G = softplus(-D(G(z)_noisy))     = log(1 + exp(-logit))
Discriminator: L_D = softplus(-D(x_real_noisy))  + softplus(D(G(z)_noisy))
R1 penalty:  L_R1 = r1_weight · ((D(x+ε) - D(x)) / σ)²
```

### Training Loop
- Discriminator updates every step
- Generator GAN loss added every Nth step (`dfake_gen_update_ratio=5`)
- Separate optimizers (gen LR: 2e-6, disc LR: 2e-6)
- GAN loss mixed with diffusion loss (x0 prediction)

## Proposed Architecture for v3a

### Discriminator: Lightweight Latent Transformer (~15M params)

NOT reusing 22B DiT (would double memory). Standalone 4-layer transformer:
```
x [B, 1344, 128] → Linear(128→512) + TimestepEmbed(t_critic)
→ prepend 4 register tokens
→ 4× TransformerEncoderLayer(dim=512, heads=8)
→ extract registers → flatten → MLP → logit [B, 1]
```

### Integration with VFM Pipeline

```python
class VFMv3aTrainingStrategy(VFMv1fTrainingStrategy):
    def compute_loss(self, video_pred, audio_pred, inputs):
        # 1. Standard v1f loss (flow matching + KL + obs + mu_align)
        base_loss = super()._compute_standard_loss_v1f(...)

        # 2. Reconstruct x̂₀ = z - v_pred
        pred_x0 = inputs._vfm_video_noise - video_pred
        real_x0 = inputs._vfm_video_latents

        # 3. Add noise at random critic timestep
        t_critic = rand(0.02, 0.98)
        noisy_fake = (1-t) * pred_x0.detach() + t * noise
        noisy_real = (1-t) * real_x0 + t * noise

        # 4. Discriminator step (every step)
        D_real = disc(noisy_real, t_critic)
        D_fake = disc(noisy_fake, t_critic)
        loss_D = softplus(-D_real) + softplus(D_fake)
        loss_D.backward()  # internal, disjoint params
        disc_optimizer.step()

        # 5. Generator GAN loss (every Nth step)
        if step % 5 == 0:
            noisy_fake_grad = (1-t) * pred_x0 + t * noise  # WITH gradient
            D_fake_grad = disc(noisy_fake_grad, t_critic)
            loss_G = softplus(-D_fake_grad)
            base_loss += 0.01 * loss_G

        return base_loss
```

### Memory Budget (32GB 5090, int8-quanto)

| Component | Memory |
|-----------|--------|
| DiT (generator) | ~22GB |
| Discriminator (15M) | ~0.5GB |
| Disc backward | ~0.5GB |
| **Total peak** | **~23GB** (fits batch_size=1) |

## FlashMotion Hyperparameters (Stage 3 Config)

```yaml
gan_g_weight: 0.01          # Generator GAN loss weight
gan_d_weight: 0.01          # Discriminator loss weight
r1_weight: 0.0              # R1 gradient penalty (disabled)
dfake_gen_update_ratio: 5   # Gen update every 5 disc steps
gen_lr: 2e-6                # Generator optimizer LR
critic_lr: 2e-6             # Discriminator LR
beta1: 0.0                  # Adam beta1 (no momentum)
beta2: 0.999
denoising_loss_type: x0     # Predict clean video
```

## Files to Create

| File | Purpose |
|------|---------|
| `ltx-trainer/src/ltx_trainer/training_strategies/vfm_discriminator.py` | Discriminator module |
| `ltx-trainer/src/ltx_trainer/training_strategies/vfm_strategy_v3a.py` | Strategy (extends v1f) |
| `ltx-trainer/configs/ltx2_vfm_v3a_overfit_23.yaml` | Config |

## Files to Modify

| File | Change |
|------|--------|
| `config.py` | Register VFMv3aTrainingConfig |
| `__init__.py` | Import + factory case |
| `trainer.py` | Add `prepare_with_accelerator` hook (~3 lines) |

## Prerequisites

1. **v1f must converge first** — adapter must produce text-conditioned, non-collapsed outputs
2. Load v1f checkpoint (adapter + LoRA + SigmaHead) as starting point
3. Training mode: `frozen` (only train adapter + discriminator, freeze DiT)

## FlashMotion Reference Files

| File | Content |
|------|---------|
| `/home/johndpope/Documents/GitHub/FlashMotion/model/flashmotion.py` | GAN losses (critic_loss, generator_loss) |
| `/home/johndpope/Documents/GitHub/FlashMotion/trainer/gan.py` | Training loop (alternating updates) |
| `/home/johndpope/Documents/GitHub/FlashMotion/utils/wan_wrapper.py` | Discriminator branch (adding_cls_branch) |
| `/home/johndpope/Documents/GitHub/FlashMotion/wan/modules/model.py` | GanAttentionBlock |
| `/home/johndpope/Documents/GitHub/FlashMotion/configs/train/stage3.yaml` | All hyperparameters |

## W&B Metrics to Add

- `vfm/loss_D` — discriminator loss (~0.69 when balanced)
- `vfm/loss_G_gan` — generator adversarial loss (decreasing)
- `vfm/D_real_logit` — score on real (positive = correct)
- `vfm/D_fake_logit` — score on fake (negative = correct)
- `vfm/gan_warmup` — warmup ramp factor
