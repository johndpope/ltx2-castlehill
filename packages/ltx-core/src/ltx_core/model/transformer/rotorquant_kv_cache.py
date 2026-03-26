"""RotorQuant v0b: KV cache compression with fused CUDA kernel.

Compress on write via fused kernel (embed→rotate→quant→inverse→extract in 12μs).
Store compressed uint8 indices + fp16 norms.
Decompress on read via einsum (no kernel needed — reads are less frequent).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
from torch import Tensor

from ltx_core.model.transformer.rotorquant_attention import RotorQuantCompressor


@dataclass
class RotorQuantKVCache:
    """KV cache with RotorQuant compression. Drop-in for KVCache."""

    compressed_keys: dict[int, tuple[Tensor, Tensor, int]] = field(default_factory=dict)
    compressed_values: dict[int, tuple[Tensor, Tensor, int]] = field(default_factory=dict)
    keys: dict[int, Tensor] = field(default_factory=dict)
    values: dict[int, Tensor] = field(default_factory=dict)

    is_cache_step: bool = False
    cached_seq_len: int = 0

    bits: int = 3
    hidden_dim: int = 4096

    _k_comp: RotorQuantCompressor | None = None
    _v_comp: RotorQuantCompressor | None = None

    @staticmethod
    def empty(bits: int = 3, hidden_dim: int = 4096) -> RotorQuantKVCache:
        return RotorQuantKVCache(bits=bits, hidden_dim=hidden_dim)

    @property
    def has_cache(self) -> bool:
        return len(self.compressed_keys) > 0 or len(self.keys) > 0

    def _ensure_compressors(self, device: torch.device) -> None:
        if self._k_comp is None:
            self._k_comp = RotorQuantCompressor(self.hidden_dim, bits=self.bits, seed=42).to(device)
            self._v_comp = RotorQuantCompressor(self.hidden_dim, bits=self.bits, seed=4200).to(device)

    def get_layer_cache(self, layer_idx: int) -> dict:
        k = v = None
        if layer_idx in self.compressed_keys:
            idx, norms, orig_dim = self.compressed_keys[layer_idx]
            self._ensure_compressors(idx.device)
            k = self._k_comp.decompress(idx, norms, orig_dim)
        elif layer_idx in self.keys:
            k = self.keys[layer_idx]

        if layer_idx in self.compressed_values:
            idx, norms, orig_dim = self.compressed_values[layer_idx]
            self._ensure_compressors(idx.device)
            v = self._v_comp.decompress(idx, norms, orig_dim)
        elif layer_idx in self.values:
            v = self.values[layer_idx]

        return {"keys": k, "values": v, "is_cache_step": self.is_cache_step, "_layer_idx": layer_idx}

    def update_from_layer_cache(self, layer_cache: dict) -> None:
        layer_idx = layer_cache["_layer_idx"]
        if layer_cache.get("keys") is not None:
            k = layer_cache["keys"]
            self._ensure_compressors(k.device)
            self.compressed_keys[layer_idx] = self._k_comp.compress(k)
            self.keys.pop(layer_idx, None)
        if layer_cache.get("values") is not None:
            v = layer_cache["values"]
            self._ensure_compressors(v.device)
            self.compressed_values[layer_idx] = self._v_comp.compress(v)
            self.values.pop(layer_idx, None)

    def memory_usage_bytes(self) -> dict:
        comp = sum(idx.nbytes + norms.nbytes for idx, norms, _ in self.compressed_keys.values())
        comp += sum(idx.nbytes + norms.nbytes for idx, norms, _ in self.compressed_values.values())
        uncomp = sum(t.nbytes for t in self.keys.values()) + sum(t.nbytes for t in self.values.values())
        fp16 = sum(B * S * d * 2 for (idx, _, d) in self.compressed_keys.values() for B, S, _ in [idx.shape])
        fp16 += sum(B * S * d * 2 for (idx, _, d) in self.compressed_values.values() for B, S, _ in [idx.shape])
        total = comp + uncomp
        return {
            "compressed_bytes": comp, "uncompressed_bytes": uncomp, "total_bytes": total,
            "fp16_equivalent_bytes": fp16, "compression_ratio": fp16 / max(total, 1),
            "n_layers_compressed": len(self.compressed_keys),
        }
