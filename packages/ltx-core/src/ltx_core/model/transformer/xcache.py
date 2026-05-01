"""X-Cache: Cross-chunk residual caching for training-free inference acceleration.

Exploits cross-chunk redundancy in autoregressive video generation: consecutive chunks
share most visual content, so at matching denoising step t and block b, the residual
r = block(x) - x is nearly identical between chunk N and N-1.

Key properties:
- Training-free: no retraining or fine-tuning needed
- Cross-chunk: caches from previous chunk's trajectory (unlike TeaCache which is cross-step)
- Survives few-step distillation: cross-chunk similarity is independent of step count
- Complementary to TeaCache, CalibAtt, gFFN-HRR, quantization

Based on: X-Cache (cross-chunk residual reuse for autoregressive DiT inference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class XCacheConfig:
    """Configuration for X-Cache cross-chunk residual caching.

    All thresholds control the skip/recompute tradeoff:
    - Higher tau_floor → fewer skips → higher quality
    - Lower tau_floor → more skips → faster inference
    """

    enabled: bool = True
    warmup_chunks: int = 1  # First W chunks always fully computed (no cache yet)
    front_anchor_blocks: int = 1  # First Fn blocks always computed (critical early features)
    back_anchor_blocks: int = 0  # Last Bn blocks always computed (0 = disabled)
    tau_floor: float = 0.97  # Minimum cosine similarity threshold
    margin: float = 0.02  # Adaptive margin below EMA similarity
    ema_alpha: float = 0.3  # EMA smoothing for adaptive threshold
    tau_dev: float = 2.0  # Max token deviation threshold
    max_staleness: int = 0  # Max consecutive skips before forced recompute (0 = disabled)
    step0_protection: bool = False  # Force full compute on first denoising step
    kv_update_protection: bool = True  # Force full compute on KV-update frames
    fingerprint_k: int = 32  # Number of tokens in 3D subsample fingerprint


@dataclass
class _PerKeyState:
    """Internal per-(step, block) tracking state."""

    residual: Optional[Tensor] = None
    fingerprint: Optional[Tensor] = None
    ema_sim: float = 0.95
    staleness: int = 0


class XCacheManager:
    """Manages cross-chunk residual caching for DiT blocks during AR inference.

    For each (denoising_step, block_index) pair, caches the residual from the
    previous chunk. On subsequent chunks, computes a lightweight fingerprint of
    the block input and compares against the cached version to decide whether
    to skip the block and reuse the cached residual.

    Usage in the decoder loop:
        xcache = XCacheManager(num_blocks=16, num_steps=8, config=cfg, device=device)

        for chunk in chunks:
            xcache.reset_for_new_chunk(is_kv_update=...)
            for t in range(num_steps):
                for b, block in enumerate(decoder_blocks):
                    fp = xcache.compute_fingerprint(x, latent_shape=(F, H, W))
                    if xcache.should_skip(t, b, fp):
                        x = x + xcache.get_cached_residual(t, b)
                    else:
                        out = block(x, ...)
                        xcache.update_cache(t, b, out - x, fp)
                        x = out
    """

    def __init__(
        self,
        num_blocks: int,
        num_steps: int,
        config: XCacheConfig,
        device: torch.device,
    ):
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.cfg = config
        self.device = device

        # Per-(step, block) state
        self._state: dict[tuple[int, int], _PerKeyState] = {}

        self.current_chunk: int = 0
        self._is_kv_update: bool = False

        # Statistics
        self.total_decisions: int = 0
        self.total_skips: int = 0

    def reset_for_new_chunk(self, is_kv_update: bool = False) -> None:
        """Advance to next chunk. Call at the start of each AR chunk."""
        self.current_chunk += 1
        self._is_kv_update = is_kv_update

    @property
    def skip_rate(self) -> float:
        """Fraction of block evaluations that were skipped (0.0 to 1.0)."""
        if self.total_decisions == 0:
            return 0.0
        return self.total_skips / self.total_decisions

    def compute_fingerprint(
        self,
        x: Tensor,
        latent_shape: Optional[tuple[int, int, int]] = None,
    ) -> Tensor:
        """Compute a lightweight fingerprint of the block input for similarity comparison.

        The fingerprint has two components:
        1. 3D spatial subsample: Cartesian product of uniformly spaced indices along
           F, H, W axes, proportional to the aspect ratio so coverage is balanced.
        2. Global mean: channel-averaged representation of all tokens

        Args:
            x: Block input hidden state [B, L, D]
            latent_shape: Optional (F, H, W) latent dimensions for 3D subsampling.
                If None, takes the first `fingerprint_k` tokens.

        Returns:
            Flattened fingerprint tensor [B, fingerprint_dim]
        """
        B, L, D = x.shape
        k = self.cfg.fingerprint_k

        # 3D grid subsample with Cartesian product indexing
        if latent_shape is not None:
            Fg, Hg, Wg = latent_shape
            if L == Fg * Hg * Wg and Fg > 0 and Hg > 0 and Wg > 0:
                phi = self._fingerprint_3d(x, Fg, Hg, Wg, k)
            else:
                phi = x[:, : min(k, L)].reshape(B, -1)
        else:
            phi = x[:, : min(k, L)].reshape(B, -1)

        # Append global mean channel for robustness
        global_mean = x.mean(dim=1)  # [B, D]
        phi = torch.cat([phi, global_mean], dim=-1)

        return phi

    @staticmethod
    def _fingerprint_3d(x: Tensor, F: int, H: int, W: int, k: int) -> Tensor:
        """3D grid subsample using Cartesian product of uniformly spaced indices.

        Allocates kF : kH : kW proportional to F : H : W, then adjusts to hit
        the target token count k. Uses meshgrid for proper Cartesian indexing.
        """
        B, L, C = x.shape
        total = F * H * W

        # Proportional allocation
        kF = max(1, round(k * F / total))
        kH = max(1, round(k * H / total))
        kW = max(1, round(k * W / total))

        # Grow smallest dimension that can still grow until we hit target
        for _ in range(k):  # bounded iterations (worst case: grow by 1 each time)
            if kF * kH * kW >= k:
                break
            # Find the smallest dim that hasn't hit its ceiling
            candidates = []
            if kF < F:
                candidates.append((kF, "F"))
            if kH < H:
                candidates.append((kH, "H"))
            if kW < W:
                candidates.append((kW, "W"))
            if not candidates:
                break  # all dims maxed
            _, dim = min(candidates)
            if dim == "F":
                kF += 1
            elif dim == "H":
                kH += 1
            else:
                kW += 1

        # Uniformly spaced indices along each axis
        f_idx = torch.linspace(0, F - 1, kF, dtype=torch.long, device=x.device)
        h_idx = torch.linspace(0, H - 1, kH, dtype=torch.long, device=x.device)
        w_idx = torch.linspace(0, W - 1, kW, dtype=torch.long, device=x.device)

        # Cartesian product → flat token indices
        f_grid, h_grid, w_grid = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
        flat_indices = (f_grid * H * W + h_grid * W + w_grid).reshape(-1)

        # Index into flattened token sequence
        selected = x[:, flat_indices]  # [B, kF*kH*kW, C]
        return selected.reshape(B, -1)

    def should_skip(
        self,
        step: int,
        block: int,
        current_fp: Tensor,
    ) -> bool:
        """Decide whether to skip this block and reuse the cached residual.

        Uses dual-metric gating:
        1. Cosine similarity >= adaptive threshold (EMA-based)
        2. Max absolute deviation < tau_dev

        Plus structural protections:
        - Warmup: first W chunks always fully computed
        - Anchor blocks: first Fn / last Bn blocks always computed
        - KV-update protection: force full compute on context-update frames
        - Step-0 protection: optionally protect first denoising step
        - Staleness: force recompute after max_staleness consecutive skips

        Args:
            step: Current denoising step index (0..num_steps-1)
            block: Current block index (0..num_blocks-1)
            current_fp: Fingerprint of current block input

        Returns:
            True if the block should be skipped (reuse cached residual)
        """
        self.total_decisions += 1

        # --- Structural protections ---
        if self.current_chunk <= self.cfg.warmup_chunks:
            return False

        if self._is_kv_update and self.cfg.kv_update_protection:
            return False

        if step == 0 and self.cfg.step0_protection:
            return False

        if block < self.cfg.front_anchor_blocks:
            return False

        if block >= self.num_blocks - self.cfg.back_anchor_blocks:
            return False

        # Staleness check
        state = self._state.get((step, block))
        if state is None or state.fingerprint is None:
            return False

        if (
            self.cfg.max_staleness > 0
            and state.staleness >= self.cfg.max_staleness
        ):
            return False

        # --- Dual-metric similarity gate ---
        prev_fp = state.fingerprint

        # Min cosine similarity across batch
        cos_sim = F.cosine_similarity(current_fp, prev_fp, dim=-1).min().item()

        # Max absolute deviation (normalized by mean magnitude)
        abs_diff = (current_fp - prev_fp).abs()
        max_dev = (abs_diff.max() / (prev_fp.abs().mean() + 1e-8)).item()

        # Adaptive threshold from EMA
        tau = max(self.cfg.tau_floor, state.ema_sim - self.cfg.margin)

        skip = (cos_sim >= tau) and (max_dev < self.cfg.tau_dev)

        if skip:
            self.total_skips += 1

        return skip

    def get_cached_residual(self, step: int, block: int) -> Optional[Tensor]:
        """Get the cached residual for (step, block), or None if not available."""
        state = self._state.get((step, block))
        if state is not None and state.residual is not None:
            state.staleness += 1
            return state.residual
        return None

    def update_cache(
        self,
        step: int,
        block: int,
        residual: Tensor,
        fingerprint: Tensor,
    ) -> None:
        """Cache the residual and fingerprint for (step, block).

        Updates the EMA similarity tracking for adaptive thresholding.

        Args:
            step: Denoising step index
            block: Block index
            residual: Block output - block input [B, L, D]
            fingerprint: Fingerprint of block input [B, fp_dim]
        """
        key = (step, block)

        if key not in self._state:
            self._state[key] = _PerKeyState()

        state = self._state[key]

        # Update EMA similarity
        if state.fingerprint is not None:
            sim = F.cosine_similarity(fingerprint, state.fingerprint, dim=-1).mean().item()
            state.ema_sim = self.cfg.ema_alpha * state.ema_sim + (1 - self.cfg.ema_alpha) * sim

        state.residual = residual.detach().clone()
        state.fingerprint = fingerprint.detach().clone()
        state.staleness = 0

    def clear(self) -> None:
        """Clear all cached state. Call between independent generation sequences."""
        self._state.clear()
        self.current_chunk = 0
        self.total_decisions = 0
        self.total_skips = 0
