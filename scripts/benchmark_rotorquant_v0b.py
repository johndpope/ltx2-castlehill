#!/usr/bin/env python3
"""
Benchmark RotorQuant v0b: SCD KV cache compression.

Simulates SCD autoregressive inference — encoding frames one at a time
with growing KV cache — and measures:
1. Memory savings (compressed vs FP16 cache)
2. Compression/decompression overhead per frame
3. Attention output fidelity after compress→decompress round-trip

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rotorquant_v0b.py
"""

import torch
import torch.nn.functional as F
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'ltx-core', 'src'))

from ltx_core.model.transformer.scd_model import KVCache
from ltx_core.model.transformer.rotorquant_kv_cache import RotorQuantKVCache


def benchmark_cache_memory():
    """Compare memory: FP16 KVCache vs RotorQuantKVCache."""
    print("=" * 70)
    print("TEST 1: KV Cache Memory Comparison")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # LTX-2 dimensions
    heads = 32
    dim_head = 128
    hidden_dim = heads * dim_head  # 4096
    encoder_layers = 32
    tokens_per_frame = 1024  # typical for ~480p
    batch = 1

    print(f"  heads={heads}, dim_head={dim_head}, encoder_layers={encoder_layers}")
    print(f"  tokens_per_frame={tokens_per_frame}\n")

    print(f"  {'frames':>6s}  {'FP16 cache':>12s}  {'RQ 3-bit':>12s}  {'ratio':>8s}  {'savings':>10s}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*10}")

    for n_frames in [1, 5, 10, 30, 60]:
        seq_len = n_frames * tokens_per_frame

        # FP16 KV cache size
        # K + V per layer: 2 × B × seq_len × H*D × 2 bytes
        fp16_bytes = encoder_layers * 2 * batch * seq_len * hidden_dim * 2

        # RotorQuant 3-bit: indices (uint8) + norms (float32)
        # Per head per layer: indices = seq_len × n_groups*3 bytes, norms = seq_len × 4 bytes
        n_groups = (dim_head + 2) // 3  # 43
        # K + V per layer per head: indices + norms
        idx_bytes = seq_len * n_groups * 3  # uint8
        norm_bytes = seq_len * 4  # float32 norms
        rq_bytes = encoder_layers * 2 * batch * heads * (idx_bytes + norm_bytes)

        ratio = fp16_bytes / rq_bytes

        def fmt(b):
            if b < 1024**2: return f"{b/1024:.1f} KB"
            if b < 1024**3: return f"{b/1024**2:.1f} MB"
            return f"{b/1024**3:.2f} GB"

        savings = fp16_bytes - rq_bytes
        print(f"  {n_frames:>6d}  {fmt(fp16_bytes):>12s}  {fmt(rq_bytes):>12s}  {ratio:>7.1f}x  {fmt(savings):>10s}")

    print()


def benchmark_roundtrip_fidelity():
    """Test compression→decompression fidelity on realistic K/V tensors."""
    print("=" * 70)
    print("TEST 2: Compress→Decompress Fidelity")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = 32
    dim_head = 128
    hidden_dim = heads * dim_head
    batch = 1

    print(f"  Simulating K/V tensors from encoder attention\n")

    print(f"  {'bits':>4s}  {'seq_len':>8s}  {'K cosine':>10s}  {'V cosine':>10s}  {'attn cos':>10s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")

    for bits in [2, 3, 4]:
        for seq_len in [1024, 4096]:
            cache = RotorQuantKVCache.empty(bits=bits, hidden_dim=heads * dim_head)
            cache.is_cache_step = True

            # Simulate K/V from attention projection (not random — has structure)
            torch.manual_seed(42)
            # Simulate projected K/V with realistic magnitude
            k_orig = torch.randn(batch, seq_len, hidden_dim, device=device) * 0.1
            v_orig = torch.randn(batch, seq_len, hidden_dim, device=device) * 0.1

            # Store via layer_cache interface (triggers compression)
            layer_cache = {"keys": k_orig, "values": v_orig,
                           "is_cache_step": True, "_layer_idx": 0}
            cache.update_from_layer_cache(layer_cache)

            # Read back (triggers decompression)
            read_cache = cache.get_layer_cache(0)
            k_hat = read_cache["keys"]
            v_hat = read_cache["values"]

            # Fidelity metrics
            k_cos = F.cosine_similarity(
                k_orig.reshape(-1, dim_head), k_hat.reshape(-1, dim_head)
            ).mean().item()
            v_cos = F.cosine_similarity(
                v_orig.reshape(-1, dim_head), v_hat.reshape(-1, dim_head)
            ).mean().item()

            # Attention score fidelity
            # Use last token as query
            q = torch.randn(batch, 1, hidden_dim, device=device) * 0.1
            q_h = q.view(batch, 1, heads, dim_head).transpose(1, 2)

            k_h_orig = k_orig.view(batch, seq_len, heads, dim_head).transpose(1, 2)
            k_h_hat = k_hat.view(batch, seq_len, heads, dim_head).transpose(1, 2)

            scores_orig = (q_h @ k_h_orig.transpose(-2, -1)) / math.sqrt(dim_head)
            scores_hat = (q_h @ k_h_hat.transpose(-2, -1)) / math.sqrt(dim_head)

            attn_cos = F.cosine_similarity(
                scores_orig.reshape(-1, seq_len), scores_hat.reshape(-1, seq_len)
            ).mean().item()

            print(f"  {bits:>4d}  {seq_len:>8d}  {k_cos:>10.6f}  {v_cos:>10.6f}  {attn_cos:>10.6f}")

            del cache, k_orig, v_orig, k_hat, v_hat
            torch.cuda.empty_cache() if device == "cuda" else None

    print()


def benchmark_autoregressive_simulation():
    """Simulate SCD autoregressive inference with growing cache."""
    print("=" * 70)
    print("TEST 3: Autoregressive Inference Simulation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = 32
    dim_head = 128
    hidden_dim = heads * dim_head
    encoder_layers = 8  # use 8 for speed (real model has 32)
    tokens_per_frame = 256  # smaller for benchmark
    batch = 1
    n_frames = 20

    print(f"  Simulating {n_frames} frames, {encoder_layers} encoder layers")
    print(f"  tokens_per_frame={tokens_per_frame}\n")

    for label, CacheClass, kwargs in [
        ("FP16", KVCache, {}),
        ("RQ-3bit", RotorQuantKVCache, {"bits": 3, "hidden_dim": heads * dim_head}),
    ]:
        if CacheClass == KVCache:
            cache = KVCache.empty()
        else:
            cache = RotorQuantKVCache.empty(**kwargs)
        cache.is_cache_step = True

        torch.cuda.empty_cache() if device == "cuda" else None
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        baseline_mem = torch.cuda.memory_allocated() if device == "cuda" else 0

        total_compress_us = 0
        total_decompress_us = 0

        torch.manual_seed(42)
        for frame in range(n_frames):
            seq_len = tokens_per_frame  # new frame tokens

            for layer_idx in range(encoder_layers):
                # Simulate K/V for this frame
                k = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=torch.float16) * 0.1
                v = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=torch.float16) * 0.1

                # Read existing cache (decompress)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                existing = cache.get_layer_cache(layer_idx)
                if device == "cuda":
                    torch.cuda.synchronize()
                total_decompress_us += (time.perf_counter() - t0) * 1e6

                # Concatenate new K/V with cached
                if existing["keys"] is not None:
                    new_k = torch.cat([existing["keys"].to(k.dtype), k], dim=1)
                    new_v = torch.cat([existing["values"].to(v.dtype), v], dim=1)
                else:
                    new_k = k
                    new_v = v

                # Store back (compress)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                layer_cache = {
                    "keys": new_k.float(),  # compressor needs float32
                    "values": new_v.float(),
                    "is_cache_step": True,
                    "_layer_idx": layer_idx,
                }
                cache.update_from_layer_cache(layer_cache)
                if device == "cuda":
                    torch.cuda.synchronize()
                total_compress_us += (time.perf_counter() - t0) * 1e6

                del k, v, new_k, new_v, existing

        peak_mem = torch.cuda.max_memory_allocated() - baseline_mem if device == "cuda" else 0

        # Get cache stats
        if isinstance(cache, RotorQuantKVCache):
            stats = cache.memory_usage_bytes()
            cache_bytes = stats["total_bytes"]
            fp16_equiv = stats["fp16_equivalent_bytes"]
        else:
            cache_bytes = sum(t.nbytes for t in cache.keys.values()) + sum(t.nbytes for t in cache.values.values())
            fp16_equiv = cache_bytes

        def fmt(b):
            if b < 1024**2: return f"{b/1024:.1f} KB"
            if b < 1024**3: return f"{b/1024**2:.1f} MB"
            return f"{b/1024**3:.2f} GB"

        comp_per_frame = total_compress_us / n_frames / 1000
        decomp_per_frame = total_decompress_us / n_frames / 1000

        print(f"  {label}:")
        print(f"    Cache size:       {fmt(cache_bytes):>10s}  (FP16 equiv: {fmt(fp16_equiv)})")
        if fp16_equiv > 0 and cache_bytes > 0:
            print(f"    Compression:      {fp16_equiv / cache_bytes:.1f}x")
        print(f"    Peak VRAM:        {fmt(peak_mem):>10s}")
        print(f"    Compress/frame:   {comp_per_frame:.2f} ms")
        print(f"    Decompress/frame: {decomp_per_frame:.2f} ms")
        print()

        del cache
        torch.cuda.empty_cache() if device == "cuda" else None


def benchmark_attention_quality():
    """Compare attention output quality using cached K/V."""
    print("=" * 70)
    print("TEST 4: Attention Quality with Cached K/V")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = 32
    dim_head = 128
    hidden_dim = heads * dim_head
    batch = 1

    print(f"  Comparing attention scores: FP16 cache vs RotorQuant cache\n")

    print(f"  {'bits':>4s}  {'cached':>8s}  {'new':>6s}  {'cosine_sim':>12s}  {'top1_match':>12s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*12}")

    for bits in [3, 4]:
        for cached_len in [1024, 4096]:
            new_len = 256  # new frame tokens

            torch.manual_seed(42)

            # Full-precision cached K/V
            k_cached = torch.randn(batch, cached_len, hidden_dim, device=device) * 0.1
            v_cached = torch.randn(batch, cached_len, hidden_dim, device=device) * 0.1

            # New frame query
            q = torch.randn(batch, new_len, hidden_dim, device=device) * 0.1

            # Compress cached K/V through RotorQuant
            rq_cache = RotorQuantKVCache.empty(bits=bits, hidden_dim=heads * dim_head)
            rq_cache.is_cache_step = True
            rq_cache.update_from_layer_cache({
                "keys": k_cached, "values": v_cached,
                "is_cache_step": True, "_layer_idx": 0,
            })
            read = rq_cache.get_layer_cache(0)
            k_hat = read["keys"]
            v_hat = read["values"]

            # Compute attention scores
            q_h = q.view(batch, new_len, heads, dim_head).transpose(1, 2)
            scale = 1.0 / math.sqrt(dim_head)

            k_orig_h = k_cached.view(batch, cached_len, heads, dim_head).transpose(1, 2)
            k_hat_h = k_hat.view(batch, cached_len, heads, dim_head).transpose(1, 2)

            scores_orig = (q_h @ k_orig_h.transpose(-2, -1)) * scale  # (B, H, new, cached)
            scores_hat = (q_h @ k_hat_h.transpose(-2, -1)) * scale

            # Cosine similarity of attention score vectors (per query position)
            cos_sims = F.cosine_similarity(
                scores_orig.reshape(-1, cached_len),
                scores_hat.reshape(-1, cached_len)
            ).mean().item()

            # Top-1 match rate
            top1_orig = scores_orig.argmax(dim=-1)  # (B, H, new)
            top1_hat = scores_hat.argmax(dim=-1)
            top1_match = (top1_orig == top1_hat).float().mean().item() * 100

            print(f"  {bits:>4d}  {cached_len:>8d}  {new_len:>6d}  {cos_sims:>12.6f}  {top1_match:>10.1f}%")

            del rq_cache, k_cached, v_cached, k_hat, v_hat
            torch.cuda.empty_cache() if device == "cuda" else None

    print()


if __name__ == "__main__":
    print()
    print("RotorQuant v0b Benchmark — SCD KV Cache Compression")
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    benchmark_cache_memory()
    benchmark_roundtrip_fidelity()
    benchmark_autoregressive_simulation()
    benchmark_attention_quality()

    print("=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
