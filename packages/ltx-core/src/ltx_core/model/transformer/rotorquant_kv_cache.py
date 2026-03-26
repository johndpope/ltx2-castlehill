"""RotorQuant v0b: KV cache compression for SCD autoregressive inference.

Wraps the SCD KVCache to compress K/V tensors before storing and
decompress on read. This reduces KV cache memory by ~5x, enabling
much longer videos on the same VRAM.

For a 30s video at 24fps on LTX-2 SCD (32 encoder layers):
- FP16 KV cache: ~67.5 GB
- 3-bit RotorQuant: ~12.7 GB (5.3x savings)

Usage:
    from ltx_core.model.transformer.rotorquant_kv_cache import RotorQuantKVCache

    # Replace the standard KVCache
    kv_cache = RotorQuantKVCache.empty(bits=3, dim_head=128, num_heads=32)
    kv_cache.is_cache_step = True

    # Use exactly like KVCache — compress/decompress is transparent
    for block in encoder_blocks:
        layer_cache = kv_cache.get_layer_cache(block.idx)
        video_args, audio_args = block(..., kv_cache=layer_cache)
        kv_cache.update_from_layer_cache(layer_cache)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import torch
from torch import Tensor

from ltx_core.model.transformer.rotorquant_attention import RotorQuantCompressor


@dataclass
class RotorQuantKVCache:
    """KV cache with RotorQuant compression for SCD encoder.

    Stores K/V as quantized indices + norms instead of full-precision tensors.
    Decompresses on read for attention computation.
    """
    # Compressed storage: layer_idx -> (indices, norms, orig_dim)
    compressed_keys: dict[int, tuple[Tensor, Tensor, int]] = field(default_factory=dict)
    compressed_values: dict[int, tuple[Tensor, Tensor, int]] = field(default_factory=dict)

    # Full-precision fallback for layers not yet compressed
    keys: dict[int, Tensor] = field(default_factory=dict)
    values: dict[int, Tensor] = field(default_factory=dict)

    is_cache_step: bool = False
    cached_seq_len: int = 0

    # Compression settings
    bits: int = 3
    dim_head: int = 128
    num_heads: int = 32

    # Per-layer compressors (created on first use)
    _k_compressors: dict[int, RotorQuantCompressor] = field(default_factory=dict)
    _v_compressors: dict[int, RotorQuantCompressor] = field(default_factory=dict)

    @staticmethod
    def empty(bits: int = 3, dim_head: int = 128, num_heads: int = 32) -> RotorQuantKVCache:
        return RotorQuantKVCache(
            bits=bits, dim_head=dim_head, num_heads=num_heads
        )

    @property
    def has_cache(self) -> bool:
        return len(self.compressed_keys) > 0 or len(self.keys) > 0

    def _get_compressor(self, layer_idx: int, for_keys: bool, device: torch.device) -> RotorQuantCompressor:
        """Get or create a compressor for a layer."""
        cache = self._k_compressors if for_keys else self._v_compressors
        if layer_idx not in cache:
            seed = layer_idx * 1000 + (0 if for_keys else 500)
            comp = RotorQuantCompressor(self.dim_head, bits=self.bits, seed=seed)
            comp = comp.to(device)
            cache[layer_idx] = comp
        return cache[layer_idx]

    def _compress_tensor(self, x: Tensor, layer_idx: int, for_keys: bool) -> tuple[Tensor, Tensor, int]:
        """Compress a K or V tensor.

        x: [B, seq_len, H*D] -> stored as compressed per-head
        """
        B, S, HD = x.shape
        D = self.dim_head
        H = HD // D

        comp = self._get_compressor(layer_idx, for_keys, x.device)

        # Reshape to per-head: (B*H, S, D)
        x_heads = x.view(B, S, H, D).permute(0, 2, 1, 3).reshape(B * H, S, D)

        # Compress
        indices, norms, orig_dim = comp.compress(x_heads)

        return indices, norms, orig_dim

    def _decompress_tensor(self, compressed: tuple[Tensor, Tensor, int],
                           layer_idx: int, for_keys: bool) -> Tensor:
        """Decompress a K or V tensor back to full precision."""
        indices, norms, orig_dim = compressed
        BH, S, _ = indices.shape
        H = self.num_heads
        B = BH // H

        comp = self._get_compressor(layer_idx, for_keys, indices.device)
        x_heads = comp.decompress(indices, norms, orig_dim)

        # Reshape back: (B*H, S, D) -> (B, S, H*D)
        x = x_heads.reshape(B, H, S, self.dim_head).permute(0, 2, 1, 3).reshape(B, S, H * self.dim_head)
        return x

    def get_layer_cache(self, layer_idx: int) -> dict:
        """Get per-layer cache dict compatible with Attention.forward().

        Decompresses stored K/V on read.
        """
        k = None
        v = None

        # Try compressed first, then fall back to uncompressed
        if layer_idx in self.compressed_keys:
            k = self._decompress_tensor(self.compressed_keys[layer_idx], layer_idx, for_keys=True)
        elif layer_idx in self.keys:
            k = self.keys[layer_idx]

        if layer_idx in self.compressed_values:
            v = self._decompress_tensor(self.compressed_values[layer_idx], layer_idx, for_keys=False)
        elif layer_idx in self.values:
            v = self.values[layer_idx]

        return {
            "keys": k,
            "values": v,
            "is_cache_step": self.is_cache_step,
            "_layer_idx": layer_idx,
        }

    def update_from_layer_cache(self, layer_cache: dict) -> None:
        """Compress and store K/V from a per-layer cache dict."""
        layer_idx = layer_cache["_layer_idx"]

        if layer_cache.get("keys") is not None:
            k = layer_cache["keys"]
            # Compress and store
            self.compressed_keys[layer_idx] = self._compress_tensor(k, layer_idx, for_keys=True)
            # Remove any uncompressed version
            self.keys.pop(layer_idx, None)

        if layer_cache.get("values") is not None:
            v = layer_cache["values"]
            self.compressed_values[layer_idx] = self._compress_tensor(v, layer_idx, for_keys=False)
            self.values.pop(layer_idx, None)

    def memory_usage_bytes(self) -> dict:
        """Report memory usage of compressed vs uncompressed cache."""
        compressed_bytes = 0
        for layer_idx in self.compressed_keys:
            idx, norms, _ = self.compressed_keys[layer_idx]
            compressed_bytes += idx.nbytes + norms.nbytes
        for layer_idx in self.compressed_values:
            idx, norms, _ = self.compressed_values[layer_idx]
            compressed_bytes += idx.nbytes + norms.nbytes

        uncompressed_bytes = 0
        for t in self.keys.values():
            uncompressed_bytes += t.nbytes
        for t in self.values.values():
            uncompressed_bytes += t.nbytes

        # Estimate what full FP16 would cost
        total_elements = 0
        for layer_idx in self.compressed_keys:
            idx, norms, orig_dim = self.compressed_keys[layer_idx]
            BH, S, _ = idx.shape
            total_elements += BH * S * orig_dim
        fp16_equivalent = total_elements * 2 * 2  # K + V, 2 bytes each

        return {
            "compressed_bytes": compressed_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "total_bytes": compressed_bytes + uncompressed_bytes,
            "fp16_equivalent_bytes": fp16_equivalent,
            "compression_ratio": fp16_equivalent / max(compressed_bytes + uncompressed_bytes, 1),
            "n_layers_compressed": len(self.compressed_keys),
        }
