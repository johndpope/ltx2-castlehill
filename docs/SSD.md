# Speculative Flow Matching: SSD Concepts Applied to VFM

> **Paper:** [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251) — Kumar, Dao, May (Stanford/Princeton/Together AI, March 2025)
>
> **Core idea:** While an expensive model computes, pre-compute multiple possible outcomes in parallel using a cheap model. Verify which speculation matches. Cache hits skip expensive work entirely.

## SSD Paper Summary

### The Problem (LLM Inference)
Standard speculative decoding (SD) uses a fast "draft" model to predict tokens, then a slow "target" model verifies them in parallel. But SD is still sequential: the draft must **wait** for verification before speculating again.

### The Solution: Speculative Speculative Decoding (SSD)
- While the target **verifies** round T, the draft **predicts** likely verification outcomes
- Pre-computes speculations for each predicted outcome → stores in a **speculation cache**
- When verification finishes, checks cache → **cache hit** = return immediately (zero draft overhead)
- **Cache miss** = fall back to standard SD

### Key Innovations

| Component | What It Does |
|-----------|-------------|
| **Speculation Cache** (Def. 4) | Dictionary mapping predicted verification outcomes → pre-computed next speculations |
| **Geometric Fan-Out** (Thm. 12) | Allocate cache budget via capped geometric series: `F_k = F_0 · a^(k/(1+r))` — more speculation early, less late |
| **Saguaro Sampling** (Def. 14) | Bias draft distribution to increase cache hit rate: downweight top-F draft tokens → residual concentrates on cached tokens |
| **Fallback Strategy** (§4.3) | Use primary speculator at low batch sizes, switch to fast backup speculator at large batch sizes |

### Results
- **2x** faster than optimized speculative decoding
- **5x** faster than autoregressive decoding
- Llama-3.1-70B on 4×H100, batch size 1, greedy decoding

---

## Mapping SSD → VFM Video Training

The VFM training pipeline has the same asymmetry as SSD:

| SSD (LLM Inference) | VFM (Video Training) |
|-----|------|
| Target model (70B LLM, slow) | 48-layer DiT forward pass (~170ms) |
| Draft model (1B LLM, fast) | Noise adapter (4 transformer blocks, ~7ms) |
| Token sequence | Noise vector z per sample |
| Speculation: pre-compute next tokens | Pre-compute candidate noise vectors |
| Verification: target evaluates draft tokens | DiT evaluates adapter noise quality |
| Acceptance rate α | Gradient quality / loss improvement |
| Speculation cache | Velocity cache (noise fingerprint → DiT output) |
| Geometric fan-out | K candidate noise vectors, geometrically distributed |
| Saguaro sampling | Noise shaping toward smooth velocity regions |

---

## Proposed Techniques

### 1. Speculative Noise Selection

**Analogy:** SSD's speculation cache (Def. 4) + geometric fan-out (Thm. 12)

**Problem:** Each training step, the adapter generates ONE noise z. If that z falls in an uninformative region (low gradient magnitude, already-learned), the expensive DiT forward pass is wasted.

**Solution:** Generate K candidate noise vectors (fan-out), score them with lightweight proxy metrics, select the most informative one before the DiT runs.

```python
def speculative_noise_selection(self, adapter, x0, text_embs, positions, task_class, K=4):
    """Generate K candidate noise vectors, select best for DiT evaluation.

    Adapter forward is ~7ms, DiT is ~170ms.
    K=4 candidates costs +21ms (12% overhead) but can 2x gradient quality.
    """
    candidates = []
    for k in range(K):
        mu, log_sigma, sigma_t = adapter(text_embs, positions, task_class)
        z, mu_hat, kappa, r = self._sample_spherical_noise(mu, log_sigma)
        candidates.append((z, mu, log_sigma, sigma_t, mu_hat, kappa, r))

    # Lightweight scoring (no DiT needed)
    scores = []
    for z, mu, log_sigma, sigma_t, mu_hat, kappa, r in candidates:
        # Novelty: LSH distance to recently-used noise vectors
        novelty = self._cache_novelty(z)

        # Sigma entropy: prefer diverse per-token noise levels
        sigma_ent = -(sigma_t * torch.log(sigma_t.clamp(min=1e-8))).mean()

        # Angular diversity: prefer directions far from recent centroid
        angular_nov = 1.0 - self._centroid_similarity(mu_hat) if mu_hat is not None else 0.0

        scores.append(novelty + 0.5 * sigma_ent + 0.3 * angular_nov)

    best_idx = torch.argmax(torch.tensor(scores))
    return candidates[best_idx]
```

**Expected impact:** Better gradient quality per DiT evaluation. Analogous to how SSD's fan-out increases expected tokens per round.

---

### 2. Velocity Speculation Cache

**Analogy:** SSD's speculation cache S^T (Def. 4) with power-law cache hit rate (Def. 11)

**Problem:** The adapter updates slowly (small LR). Consecutive noise vectors are similar. The DiT computes nearly-identical velocities each step — redundant work.

**Solution:** Cache (noise_fingerprint → DiT_velocity) pairs. Check cache before running DiT. Early training: low hit rate (~10%). Late training: high hit rate (~60-80%) — exactly when you need efficiency most.

```python
class VelocitySpeculationCache:
    """Cache DiT velocity predictions for similar noise inputs.

    SSD shows cache miss rates follow a power law: 1 - p_hit(F) = 1/F^r
    In VFM training, the "fan-out" F corresponds to cache capacity,
    and r depends on how fast the adapter distribution changes.
    """

    def __init__(self, capacity=1024, hit_threshold=0.02):
        self.capacity = capacity
        self.threshold = hit_threshold
        self.keys = []      # noise fingerprints (pooled z + sigma)
        self.values = []    # (velocity, step_number)
        self.hit_rate_ema = 0.0

    def _fingerprint(self, z: Tensor, sigma: Tensor) -> Tensor:
        """Locality-sensitive hash of (noise, sigma)."""
        z_pool = z.mean(dim=1)  # [B, D]
        return torch.cat([z_pool, sigma.unsqueeze(-1)], dim=-1)

    def lookup(self, z, sigma):
        """O(1) amortized lookup via LSH bucketing."""
        fp = self._fingerprint(z, sigma)
        for key, (vel, step) in zip(self.keys, self.values):
            if (fp - key).norm() < self.threshold:
                staleness = self._current_step - step
                correction = 1.0 - 0.01 * staleness
                self.hit_rate_ema = 0.9 * self.hit_rate_ema + 0.1
                return vel * correction, True
        self.hit_rate_ema = 0.9 * self.hit_rate_ema + 0.0
        return None, False

    def store(self, z, sigma, velocity):
        """FIFO eviction when capacity reached."""
        fp = self._fingerprint(z, sigma)
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(fp.detach())
        self.values.append((velocity.detach(), self._current_step))
```

**Expected impact:** Skip 40-60% of DiT forward passes in later training stages. Monitor `cache_hit_rate` in W&B — if it's >0.5, the adapter has converged and you can increase the threshold or reduce LR.

---

### 3. Multi-Resolution Speculative Training

**Analogy:** SSD's fan-out across K verification outcomes (§4.1)

**Problem:** VFM trains at one σ per sample, but inference needs the noise to work at ALL σ levels (1-step at σ=1.0, 2-step at σ∈{0.5,1.0}, etc.). No guarantee that good σ=1.0 noise produces good σ=0.5 denoising.

**Solution:** Like SSD pre-computes speculations for K possible verification outcomes, evaluate the SAME noise z at K different sigma levels in a single batched DiT pass. Train for consistency across all paths.

```python
def speculative_multi_sigma_step(self, adapter, dit, x0, text_embs, positions):
    """Train on multiple ODE paths simultaneously.

    SSD fans out over K possible verification outcomes.
    We fan out over K possible sigma levels (noise amounts).
    Both use a single batched forward pass for efficiency.
    """
    B = x0.shape[0]

    # Adapter generates one noise distribution
    mu, log_sigma, per_token_sigma = adapter(text_embs, positions, task_class)
    z, mu_hat, kappa, r = self._sample_spherical_noise(mu, log_sigma)

    # Fan-out: K sigma levels (geometric spacing like Saguaro Thm. 12)
    sigma_levels = [1.0, 0.7, 0.5, 0.3]  # K=4

    # Create noisy inputs at each level
    x_batch = torch.cat([
        (1 - s) * x0 + s * z for s in sigma_levels
    ], dim=0)  # [K*B, seq, D]

    sigma_batch = torch.cat([
        torch.full((B,), s, device=x0.device) for s in sigma_levels
    ])

    # Single batched DiT call — K levels for ~1.5-2x cost of 1 level
    v_batch = dit(x_batch, sigma_batch, text_embs.repeat(len(sigma_levels), 1, 1))
    velocities = v_batch.chunk(len(sigma_levels), dim=0)

    # Each velocity predicts x₀: x̂₀ = x_σ - σ·v
    x0_preds = []
    for s, v in zip(sigma_levels, velocities):
        x0_hat = ((1 - s) * x0 + s * z) - s * v
        x0_preds.append(x0_hat)

    # Consistency loss: all paths should reconstruct the same x₀
    consistency = 0
    n_pairs = 0
    for i in range(len(sigma_levels)):
        for j in range(i + 1, len(sigma_levels)):
            consistency += F.mse_loss(x0_preds[i], x0_preds[j])
            n_pairs += 1
    consistency /= n_pairs

    # Standard flow matching loss at adapter's chosen per-token sigma
    fm_loss = standard_flow_matching_loss(velocities[0], z - x0)

    return fm_loss + 0.3 * consistency
```

**Why this works:** The batching trick — `[K*B, seq, D]` through the DiT — costs ~1.5-2x a single pass (not K×) due to GPU parallelism. You get K times the training signal for <2x the cost. The consistency loss forces the adapter's noise into "basins of attraction" where all ODE paths converge.

---

### 4. Saguaro Noise Shaping Loss

**Analogy:** Saguaro sampling scheme (Def. 14) — reshape the distribution for higher "acceptance rate"

**Problem:** The adapter doesn't know which regions of noise space the DiT can handle well. Some noise vectors land in regions where the velocity field is chaotic (high curvature) → bad 1-step output.

**Solution:** Penalize noise that produces high-curvature velocity fields. Train the adapter to concentrate its noise distribution on smooth velocity regions — exactly where 1-step generation succeeds.

This is the direct translation of Saguaro's insight: **don't just train for output quality — reshape the noise distribution so that the denoising process itself is more predictable.**

```python
class SaguaroNoiseShapingLoss(nn.Module):
    """Shape adapter noise distribution toward smooth velocity regions.

    Saguaro Def. 14: σ_{F,C}(z) ∝ {C·exp(z_t) if t ∈ top_F(z), exp(z_t) otherwise}

    VFM translation: penalize noise z where the velocity field has high curvature.
    High curvature = small perturbation to z causes large change in v = DiT(x_σ, σ).
    Low curvature = the ODE path is nearly straight → 1-step works well.
    """

    def __init__(self, perturbation_scale=0.01, weight=0.1):
        super().__init__()
        self.eps_scale = perturbation_scale
        self.weight = weight

    def forward(self, z, x0, sigma_t, v_pred, dit):
        """
        Args:
            z: adapter noise [B, seq, D]
            x0: clean video latents [B, seq, D]
            sigma_t: per-token sigma [B, seq]
            v_pred: DiT velocity at (x_sigma, sigma) — already computed
            dit: frozen DiT for curvature estimation
        """
        sigma_exp = sigma_t.unsqueeze(-1)

        # Perturb noise slightly
        eps = self.eps_scale * torch.randn_like(z)
        z_perturbed = z + eps
        x_sigma_perturbed = (1 - sigma_exp) * x0 + sigma_exp * z_perturbed

        # Measure velocity change (curvature proxy)
        with torch.no_grad():
            v_perturbed = dit(x_sigma_perturbed, sigma_t)

        # Per-token curvature: how sensitive is the velocity to noise perturbation?
        curvature = (v_pred - v_perturbed).pow(2).mean(dim=-1)  # [B, seq]

        # Weight by sigma — high σ (early denoising) matters most for 1-step
        weighted_curvature = (curvature * sigma_t).mean()

        return self.weight * weighted_curvature
```

**Expected impact:** The adapter learns to avoid "cliffs" in the velocity landscape. Noise naturally concentrates in regions where the ODE is approximately linear → better 1-step quality without changing the DiT or the number of training steps.

**Monitoring:** Log `vfm/velocity_curvature` to W&B. Should decrease over training as the adapter learns smooth regions. If it plateaus high, increase the shaping weight.

---

### 5. Async Adapter-DiT Pipeline

**Analogy:** SSD's core architecture — speculator and verifier run on separate hardware in parallel

**Problem:** VFM training is sequential: adapter forward → DiT forward → loss → backward. The adapter (GPU:1, 7ms) is idle during DiT forward (GPU:0, 170ms) — 96% idle time.

**Solution:** Pipeline the adapter and DiT across two CUDA streams. While DiT processes batch N, the adapter pre-computes noise for batch N+1 on a separate device.

```
Stream A (GPU:0 — DiT):     ──[forward N]──[backward N]──[forward N+1]──[backward N+1]──
Stream B (GPU:1 — Adapter):    [noise N+1]──[noise N+2]─────[noise N+2]──[noise N+3]────
                                    ↑                            ↑
                              pre-computed                 pre-computed
                              while DiT busy               while DiT busy
```

**Implementation:** Use `torch.cuda.Stream` and events for synchronization. The adapter's output (noise z) is transferred to GPU:0 via `z.to(device='cuda:0', non_blocking=True)` which overlaps with DiT compute.

**Expected impact:** Hides 100% of adapter + data preparation latency. Training throughput improves by ~15-20% (the adapter + data prep fraction of each step).

---

## Integration with Existing VFM Versions

| Technique | Fits v1f | Fits v1g | Fits v1h | Notes |
|-----------|----------|----------|----------|-------|
| Speculative Noise Selection | Yes | Yes | Yes | Add to `_prepare_standard_inputs` |
| Velocity Cache | Yes | Yes | Yes | New class, optional integration |
| Multi-Resolution Speculative | Yes | Yes | **Best** | v1h's per-token sigma enables per-level weighting |
| Saguaro Noise Shaping | Yes | **Best** | Yes | v1g's phase-aware weighting aligns with curvature |
| Async Pipeline | Yes | Yes | Yes | Config-level change, no strategy modification |

### Recommended Implementation Order

1. **Saguaro Noise Shaping Loss** (Idea 4) — one new loss term, lowest risk, directly improves 1-step quality
2. **Multi-Resolution Speculative Training** (Idea 3) — most creative, addresses multi-step consistency
3. **Speculative Noise Selection** (Idea 1) — improves gradient quality per step
4. **Velocity Cache** (Idea 2) — speeds up late-stage training
5. **Async Pipeline** (Idea 5) — infra optimization, independent of loss changes

---

## Key Metrics to Monitor

| Metric | What It Tells You | Target |
|--------|-------------------|--------|
| `vfm/velocity_curvature` | Smoothness of velocity field at adapter noise | Decreasing over training |
| `vfm/cache_hit_rate` | Fraction of steps using cached velocity | >0.5 in late training |
| `vfm/multi_sigma_consistency` | Agreement of x₀ predictions across σ levels | Decreasing (lower = more consistent) |
| `vfm/noise_selection_score` | Quality of selected vs rejected candidates | Gap should increase |
| `vfm/adapter_idle_fraction` | Time adapter spends waiting for DiT | <0.05 with async pipeline |

---

## References

- Kumar, Dao, May. *Speculative Speculative Decoding*. arXiv:2603.03251, 2025.
- Dosi et al. *HyperSphereDiff*. ICML 2025, arXiv:2506.10576. (v1g directional losses)
- Zheng & He. *Self-Flow*. arXiv:2603.06507, 2025. (v1d per-token sigma inspiration)
- Chen et al. *EVATok*. arXiv:2603.12267, 2025. (v1e content-adaptive routing)
- Pooladian et al. *Variational Flow Maps*. NeurIPS 2024. (original VFM paper)
