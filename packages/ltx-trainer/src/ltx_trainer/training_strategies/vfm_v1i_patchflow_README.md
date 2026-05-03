# VFM v1i — Patch Forcing Integration

**File:** `vfm_strategy_v1i_patchflow.py`

This strategy merges **CompVis Patch Forcing** (arxiv:2604.19141, CVPR 2026) into the VFM v1d architecture.

## Core Patch Forcing Additions

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| **LTG Sampler** | `_sample_ltg_timesteps()` — global `T_max` + per-token `t ~ U[0, T_max]` | Closes train-test gap for per-token scheduling |
| **UncertaintyHead** | MLP(x₀ + adapter_μ) → `logvar_theta` per token | Predicts per-token difficulty (edges/motion get higher variance) |
| **NLL Loss** | `flow_loss + uncertainty_weight * NLL(ut \| N(vt, σ_θ))` | Model learns to output calibrated uncertainty (like SRM) |
| **RoPE Jittering** | Accepts `img_meta` (top/left crop offsets) | Better generalization across resolutions/crops |

## Architecture (Training Forward)

```
Text embeddings
      │
      ▼
NoiseAdapterV1b ──► (μ, log_σ) ──► z = μ + σ·ε
      │
      ▼
UncertaintyHead(x₀, μ) ──► logvar_θ [B, seq]
      │
      ▼
LTG Timestep Sampler ──► t_i ∈ [0, T_max] per token
      │
      ▼
x_t[i] = (1 - t_i)·x₀[i] + t_i·z[i]
      │
      ▼
48-layer DiT (per-token timesteps)
      │
      ▼
(vt, logvar_theta)   ←── if return_uncertainty=True (model change needed)
      │
      ▼
Loss = MSE(vt, z - x₀) + uncertainty_weight * NLL(z - x₀ | N(vt, σ_θ))
```

## Usage

### 1. Config (`configs/ltx2_vfm_v1i_patchflow.yaml`)

```yaml
training_strategy:
  _target_: ltx_trainer.training_strategies.vfm_strategy_v1i_patchflow.VFMv1iPatchFlowStrategy
  config:
    name: vfm_v1i_patchflow
    use_patch_forcing: true
    uncertainty_weight: 0.01          # Patch Forcing NLL weight
    ltg_enabled: true
    per_token_uncertainty: true
    uncertainty_head_hidden_dim: 256
    alpha: 0.8                        # Adapter noise probability (from v1d fix)
    kl_free_bits: 0.05                # Lower than v1d default
    distill_mode: none                # Use "output_match" only if trajectories exist
    rope_jittering: false             # Set true if you have img_meta in batch
```

### 2. Model Change Required (LTX2 DiT)

In your transformer forward (e.g. `ltx_core/model/transformer/dit.py`):

```python
def forward(self, x, t, y=None, txt_emb=None, return_uncertainty=False, **kwargs):
    # ... existing blocks ...
    vt = self.velocity_head(...)                    # [B, seq, C]

    if return_uncertainty:
        # Small head on top of features + vt
        logvar = self.uncertainty_head(             # NEW
            torch.cat([features, vt.detach()], dim=-1)
        ).squeeze(-1)
        return vt, logvar
    return vt
```

Add `UncertaintyHead` (identical to the one in `vfm_strategy_v1i_patchflow.py`) as a submodule.

### 3. Training

```bash
# Standard (no trajectories)
uv run python scripts/train.py configs/ltx2_vfm_v1i_patchflow.yaml

# With trajectory distillation (if you have pre-computed trajectories/)
uv run python scripts/train.py configs/ltx2_vfm_v1i_patchflow_distill.yaml
```

### 4. W&B Logging

The strategy automatically logs:
- `vfm/uncertainty_nll`
- `vfm/uncertainty_mean` / `std`
- `vfm/sigma_theta_mean`
- Per-token uncertainty heatmap (Plotly)

## Differences from Original Patch Forcing

| Aspect | Original PF | VFM v1i |
|--------|-------------|---------|
| Backbone | DiT / SiT (image) | LTX2 48-layer video DiT (latent tokens) |
| Noise | Standard diffusion | VFM adapter (μ, log_σ) + Spherical Cauchy option |
| Timestep | LTG per-patch | LTG per-latent-token + complexity targets from x₀ |
| Uncertainty | Separate head on model | UncertaintyHead(x₀ + adapter_μ) — content-aware |
| Distillation | None | Optional teacher 8-step ODE matching |
| Extra losses | Only NLL | + KL + Diversity (token/temporal/spatial) |

## Known Issues / TODO

1. **Distill + Patch Forcing path** is stubbed (`_prepare_distill_inputs` raises NotImplemented). Use `distill_mode: none` for now.
2. **RoPE jittering** placeholder — needs actual position offset logic from your RoPE implementation.
3. **Model `return_uncertainty`** must be wired in LTX2 core (small change, ~20 lines).
4. **Sigma entropy** from v1d is kept but now secondary to the NLL term.

## References

- Patch Forcing: https://compvis.github.io/patch-forcing/ (arxiv:2604.19141)
- VFM v1d: `vfm_strategy_v1d.py` (trajectory distillation + SigmaHead)
- LTX2: `ltx-core.combined` / `ltx-trainer.combined`

---

**Next steps after this merge:**
- Implement model-side `UncertaintyHead` + `return_uncertainty` flag.
- A/B test `uncertainty_weight` ∈ {0.005, 0.01, 0.02}.
- Enable `distill_mode: output_match` once trajectories are generated with per-token uncertainty.
- Consider adding **difficulty-aware adaptive sampling** at inference (dual-loop / look-ahead from original PF paper).
