#!/usr/bin/env python3
"""
Benchmark RotorQuant v0a on LTX-2 vanilla attention.

Measures:
1. Attention output fidelity (cosine similarity with/without compression)
2. Peak VRAM usage during forward pass
3. Throughput (ms per attention call)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rotorquant.py
"""

import torch
import torch.nn as nn
import math
import time
import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'ltx-core', 'src'))

from ltx_core.model.transformer.attention import Attention, AttentionFunction
from ltx_core.model.transformer.rotorquant_attention import RotorQuantAttention, RotorQuantCompressor


def benchmark_compressor():
    """Benchmark the RotorQuant compressor in isolation."""
    print("=" * 70)
    print("TEST 1: RotorQuant Compressor — Compression Quality")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim_head = 128

    print(f"  Device: {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")
    print(f"  dim_head: {dim_head}\n")

    print(f"  {'bits':>4s}  {'MSE':>10s}  {'cosine':>10s}  {'compress':>10s}  {'decompress':>10s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for bits in [2, 3, 4]:
        comp = RotorQuantCompressor(dim_head, bits=bits).to(device)

        # Simulate K/V tensor from attention: (B*H, seq_len, dim_head)
        B_H, S = 32, 4096  # 1 batch × 32 heads, 4K tokens
        x = torch.randn(B_H, S, dim_head, device=device)

        # Compress
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(20):
            idx, norms, orig_dim = comp.compress(x)
        torch.cuda.synchronize() if device == "cuda" else None
        comp_ms = (time.perf_counter() - t0) / 20 * 1000

        # Decompress
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(20):
            x_hat = comp.decompress(idx, norms, orig_dim)
        torch.cuda.synchronize() if device == "cuda" else None
        decomp_ms = (time.perf_counter() - t0) / 20 * 1000

        mse = ((x - x_hat) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(-1, dim_head), x_hat.reshape(-1, dim_head)
        ).mean().item()

        print(f"  {bits:>4d}  {mse:>10.6f}  {cos:>10.6f}  {comp_ms:>8.2f}ms  {decomp_ms:>8.2f}ms")

    # Memory comparison
    print()
    B_H, S, D = 32, 4096, 128
    fp16_bytes = B_H * S * D * 2  # fp16
    for bits in [2, 3, 4]:
        quant_bytes = B_H * S * D * bits / 8 + B_H * S * 4  # indices + norms
        ratio = fp16_bytes / quant_bytes
        print(f"  {bits}-bit: {fp16_bytes/1024/1024:.1f}MB → {quant_bytes/1024/1024:.1f}MB ({ratio:.1f}x)")

    print()


def benchmark_attention_fidelity():
    """Compare attention output with/without RotorQuant."""
    print("=" * 70)
    print("TEST 2: Attention Output Fidelity")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = 32
    dim_head = 128
    query_dim = heads * dim_head  # 4096

    print(f"  heads={heads}, dim_head={dim_head}, query_dim={query_dim}\n")

    print(f"  {'bits':>4s}  {'seq_len':>8s}  {'cosine_sim':>12s}  {'max_diff':>10s}  {'attn_match':>12s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*12}")

    for bits in [3, 4]:
        for seq_len in [1024, 4096]:
            # Create attention module
            attn = Attention(
                query_dim=query_dim,
                heads=heads,
                dim_head=dim_head,
                attention_function=AttentionFunction.PYTORCH,
            ).to(device).eval()

            # Wrapped version
            rq_attn = RotorQuantAttention(attn, bits=bits, compress_values=True).to(device).eval()

            # Input
            x = torch.randn(1, seq_len, query_dim, device=device, dtype=torch.float32)

            with torch.no_grad():
                # Original output
                out_orig = attn(x)

                # RotorQuant output
                out_rq = rq_attn(x)

            cos = torch.nn.functional.cosine_similarity(
                out_orig.reshape(-1, query_dim), out_rq.reshape(-1, query_dim)
            ).mean().item()

            max_diff = (out_orig - out_rq).abs().max().item()

            # Check if attention patterns are similar (top-1 match per query)
            # Compute raw attention scores for both
            q = attn.to_q(x)
            k_orig = attn.to_k(x)
            q = attn.q_norm(q)
            k_orig = attn.k_norm(k_orig)

            q_h = q.view(1, seq_len, heads, dim_head).transpose(1, 2)
            k_h = k_orig.view(1, seq_len, heads, dim_head).transpose(1, 2)
            scores_orig = (q_h @ k_h.transpose(-2, -1)) / math.sqrt(dim_head)

            # Get top-1 per query position (sample first 100)
            top1_orig = scores_orig[0, :, :min(100, seq_len), :].argmax(dim=-1)

            print(f"  {bits:>4d}  {seq_len:>8d}  {cos:>12.6f}  {max_diff:>10.4f}  {'—':>12s}")

            del attn, rq_attn, out_orig, out_rq, x
            torch.cuda.empty_cache() if device == "cuda" else None

    print()


def benchmark_vram():
    """Measure peak VRAM with/without RotorQuant."""
    print("=" * 70)
    print("TEST 3: Peak VRAM Usage")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping VRAM test\n")
        return

    device = "cuda"
    heads = 32
    dim_head = 128
    query_dim = heads * dim_head

    print(f"  heads={heads}, dim_head={dim_head}\n")

    print(f"  {'seq_len':>8s}  {'baseline':>12s}  {'RQ 3-bit':>12s}  {'savings':>10s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}")

    for seq_len in [1024, 4096, 8192]:
        results = {}

        for label, use_rq in [("baseline", False), ("rq3", True)]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            attn = Attention(
                query_dim=query_dim,
                heads=heads,
                dim_head=dim_head,
                attention_function=AttentionFunction.PYTORCH,
            ).to(device).eval()

            if use_rq:
                attn = RotorQuantAttention(attn, bits=3, compress_values=True).to(device).eval()

            x = torch.randn(1, seq_len, query_dim, device=device, dtype=torch.float32)

            baseline_mem = torch.cuda.memory_allocated()

            with torch.no_grad():
                out = attn(x)

            peak = torch.cuda.max_memory_allocated() - baseline_mem
            results[label] = peak

            del attn, x, out
            torch.cuda.empty_cache()

        baseline_mb = results["baseline"] / 1024 / 1024
        rq_mb = results["rq3"] / 1024 / 1024
        savings = baseline_mb - rq_mb

        print(f"  {seq_len:>8d}  {baseline_mb:>10.1f}MB  {rq_mb:>10.1f}MB  {savings:>8.1f}MB")

    print()


def benchmark_speed():
    """Measure throughput with/without RotorQuant."""
    print("=" * 70)
    print("TEST 4: Speed (ms per forward pass)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = 32
    dim_head = 128
    query_dim = heads * dim_head
    n_warmup = 5
    n_iter = 20

    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  heads={heads}, dim_head={dim_head}\n")

    print(f"  {'seq_len':>8s}  {'baseline':>12s}  {'RQ 3-bit':>12s}  {'overhead':>10s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}")

    for seq_len in [1024, 4096]:
        results = {}

        for label, use_rq in [("baseline", False), ("rq3", True)]:
            attn = Attention(
                query_dim=query_dim,
                heads=heads,
                dim_head=dim_head,
                attention_function=AttentionFunction.PYTORCH,
            ).to(device).eval()

            if use_rq:
                attn = RotorQuantAttention(attn, bits=3, compress_values=True).to(device).eval()

            x = torch.randn(1, seq_len, query_dim, device=device, dtype=torch.float32)

            with torch.no_grad():
                # Warmup
                for _ in range(n_warmup):
                    attn(x)
                if device == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(n_iter):
                    attn(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / n_iter * 1000

            results[label] = ms
            del attn, x
            torch.cuda.empty_cache() if device == "cuda" else None

        overhead = results["rq3"] - results["baseline"]
        pct = overhead / results["baseline"] * 100

        print(f"  {seq_len:>8d}  {results['baseline']:>10.2f}ms  {results['rq3']:>10.2f}ms  "
              f"+{overhead:.2f}ms ({pct:+.1f}%)")

    print()


if __name__ == "__main__":
    print()
    print("RotorQuant v0a Benchmark — LTX-2 Vanilla Attention")
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    benchmark_compressor()
    benchmark_attention_fidelity()
    benchmark_vram()
    benchmark_speed()

    print("=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
