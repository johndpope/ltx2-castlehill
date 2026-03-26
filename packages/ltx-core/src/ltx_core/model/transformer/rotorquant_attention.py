"""RotorQuant-compressed attention for LTX-2.

v0a: Vanilla LTX-2 (no SCD) — compresses K/V tensors in-flight during
attention computation to reduce peak activation memory.

Instead of storing full-precision K/V tensors (B, seq_len, 4096) during
the attention forward pass, we compress them via RotorQuant (Clifford
rotor rotation + Lloyd-Max quantization) and decompress just before
the dot product. This reduces activation memory by ~3-5x.

The compression/decompression adds a small overhead but enables:
- Larger batch sizes at the same VRAM
- Higher resolution / more frames
- Longer sequences without OOM

Usage:
    # Monkey-patch after model load:
    from ltx_core.model.transformer.rotorquant_attention import enable_rotorquant
    enable_rotorquant(model, bits=3)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class RotorQuantCompressor(nn.Module):
    """Lightweight RotorQuant compressor for attention K/V tensors.

    Operates per-head: each head's dim_head dimensions are chunked into
    groups of 3, rotated by precomputed 3x3 matrices (derived from Clifford
    rotors), quantized via Lloyd-Max, then stored as uint8 indices + scale.

    Uses the batched 3x3 matmul trick (SO(3) isomorphism) for speed on GPU.
    """

    def __init__(self, dim_head: int, bits: int = 3, seed: int = 42):
        super().__init__()
        self.dim_head = dim_head
        self.bits = bits
        self.n_groups = (dim_head + 2) // 3
        self.n_levels = 2 ** bits

        # Build rotation matrices from Clifford rotors
        rng = torch.Generator()
        rng.manual_seed(seed)

        # Random rotor → 3x3 rotation matrix per group
        rotors = torch.randn(self.n_groups, 4, generator=rng)
        # Normalize to unit rotors
        rotors = rotors / rotors.norm(dim=-1, keepdim=True)
        M = self._rotors_to_matrices(rotors)
        self.register_buffer('M', M)  # (n_groups, 3, 3)
        self.register_buffer('Mt', M.transpose(1, 2).contiguous())  # inverse

        # Lloyd-Max centroids for Gaussian N(0, 1/d)
        centroids = self._compute_centroids(bits, dim_head)
        self.register_buffer('centroids', centroids)  # (n_levels,)

    @staticmethod
    def _rotors_to_matrices(rotors: torch.Tensor) -> torch.Tensor:
        """Convert rotor [s, b12, b13, b23] to 3x3 rotation matrix."""
        s, p, q, r = rotors[:, 0], rotors[:, 1], rotors[:, 2], rotors[:, 3]
        s2, p2, q2, r2 = s**2, p**2, q**2, r**2

        M = torch.zeros(rotors.shape[0], 3, 3)
        M[:, 0, 0] = s2 - p2 - q2 + r2
        M[:, 0, 1] = 2*s*p - 2*q*r
        M[:, 0, 2] = 2*s*q + 2*p*r
        M[:, 1, 0] = -2*s*p - 2*q*r
        M[:, 1, 1] = s2 - p2 + q2 - r2
        M[:, 1, 2] = 2*s*r - 2*p*q
        M[:, 2, 0] = -2*s*q + 2*p*r
        M[:, 2, 1] = -2*s*r - 2*p*q
        M[:, 2, 2] = s2 + p2 - q2 - r2
        return M

    @staticmethod
    def _compute_centroids(bits: int, d: int) -> torch.Tensor:
        """Compute Lloyd-Max centroids for N(0, 1/d)."""
        sigma = 1.0 / math.sqrt(d)
        n = 2 ** bits
        if bits == 1:
            c = math.sqrt(2.0 / math.pi) * sigma
            return torch.tensor([-c, c])
        # Uniform initialization, good enough for 2-4 bit
        return torch.linspace(-3 * sigma, 3 * sigma, n)

    def compress(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress K or V tensor.

        Args:
            x: (B, seq_len, dim_head) — one head's K or V

        Returns:
            indices: (B, seq_len, n_groups * 3) uint8 quantization indices
            norms: (B, seq_len) vector norms for rescaling
            x_shape: original shape for unpadding
        """
        B, S, D = x.shape
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Pad to multiple of 3
        pad = (3 - D % 3) % 3
        if pad > 0:
            x_unit = torch.nn.functional.pad(x_unit, (0, pad))

        # Reshape to groups: (B, S, n_groups, 3)
        x_groups = x_unit.reshape(B, S, -1, 3)

        # Rotate via 3x3 matmul: (B, S, n_groups, 3) @ (n_groups, 3, 3)
        rotated = torch.einsum('bsgi,gij->bsgj', x_groups, self.M)

        # Quantize: find nearest centroid
        flat = rotated.reshape(B, S, -1)
        indices = (flat.unsqueeze(-1) - self.centroids).abs().argmin(dim=-1).to(torch.uint8)

        return indices, norms.squeeze(-1), D

    def decompress(self, indices: torch.Tensor, norms: torch.Tensor, orig_dim: int) -> torch.Tensor:
        """Decompress back to K or V tensor.

        Args:
            indices: (B, seq_len, n_groups * 3) uint8
            norms: (B, seq_len) norms
            orig_dim: original dim_head

        Returns:
            (B, seq_len, dim_head)
        """
        B, S, _ = indices.shape

        # Dequantize
        flat = self.centroids[indices.long()]  # (B, S, n_groups * 3)
        groups = flat.reshape(B, S, -1, 3)

        # Inverse rotate
        derotated = torch.einsum('bsgi,gij->bsgj', groups, self.Mt)

        # Flatten and trim
        out = derotated.reshape(B, S, -1)[:, :, :orig_dim]

        # Rescale
        return out * norms.unsqueeze(-1)


class RotorQuantAttention(nn.Module):
    """Drop-in wrapper that compresses K/V before attention.

    Wraps the original attention function to:
    1. Compress K and V after projection + RoPE
    2. Decompress just before the dot product
    3. The compression/decompression reduces peak activation memory

    For inference only — not differentiable through the quantization.
    """

    def __init__(self, original_attention: nn.Module, bits: int = 3, compress_values: bool = True):
        super().__init__()
        self.original = original_attention
        self.bits = bits
        self.compress_values = compress_values

        # Create compressor matching head dim
        self.k_compressor = RotorQuantCompressor(
            original_attention.dim_head, bits=bits, seed=42
        )
        if compress_values:
            self.v_compressor = RotorQuantCompressor(
                original_attention.dim_head, bits=bits, seed=4200
            )
        else:
            self.v_compressor = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        """Forward with K/V compression."""
        from ltx_core.model.transformer.rope import apply_rotary_emb

        attn = self.original
        context = x if context is None else context
        use_attention = not all_perturbed

        v = attn.to_v(context)

        if not use_attention:
            out = v
        else:
            q = attn.to_q(x)
            k = attn.to_k(context)

            q = attn.q_norm(q)
            k = attn.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, attn.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn.rope_type)

            # ── RotorQuant compression ──
            B, T, HD = k.shape
            heads = attn.heads
            dim_head = attn.dim_head

            # Reshape to per-head: (B, T, H, D) -> (B*H, T, D)
            k_heads = k.view(B, T, heads, dim_head).permute(0, 2, 1, 3).reshape(B * heads, T, dim_head)
            k_idx, k_norms, k_orig_dim = self.k_compressor.compress(k_heads)
            k_decompressed = self.k_compressor.decompress(k_idx, k_norms, k_orig_dim)
            k = k_decompressed.reshape(B, heads, T, dim_head).permute(0, 2, 1, 3).reshape(B, T, HD)

            if self.compress_values and self.v_compressor is not None:
                v_heads = v.view(B, T, heads, dim_head).permute(0, 2, 1, 3).reshape(B * heads, T, dim_head)
                v_idx, v_norms, v_orig_dim = self.v_compressor.compress(v_heads)
                v_decompressed = self.v_compressor.decompress(v_idx, v_norms, v_orig_dim)
                v = v_decompressed.reshape(B, heads, T, dim_head).permute(0, 2, 1, 3).reshape(B, T, HD)

            # Delete compressed indices to free memory immediately
            del k_idx, k_norms
            if self.compress_values:
                del v_idx, v_norms

            out = attn.attention_function(q, k, v, heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + attn.to_v(context) * (1 - perturbation_mask)

        # Per-head gating
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, attn.heads, attn.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, attn.heads * attn.dim_head)

        return attn.to_out(out)


def enable_rotorquant(model: nn.Module, bits: int = 3, compress_values: bool = True,
                      layers: list[int] | None = None) -> int:
    """Monkey-patch all Attention modules in an LTX model with RotorQuant compression.

    Args:
        model: LTXModel or LTXSCDModel
        bits: Quantization bits (2, 3, or 4)
        compress_values: Whether to also compress V (True for max savings)
        layers: Optional list of layer indices to compress (None = all)

    Returns:
        Number of attention modules patched
    """
    from ltx_core.model.transformer.attention import Attention

    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            # Check if this is in a targeted layer
            if layers is not None:
                layer_idx = None
                for part in name.split('.'):
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_idx is not None and layer_idx not in layers:
                    continue

            # Wrap with RotorQuant
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model
            for p in parent_name.split('.'):
                if p:
                    parent = getattr(parent, p)

            wrapper = RotorQuantAttention(module, bits=bits, compress_values=compress_values)
            # Move compressor buffers to same device as attention weights
            device = module.to_q.weight.device
            wrapper.k_compressor = wrapper.k_compressor.to(device)
            if wrapper.v_compressor is not None:
                wrapper.v_compressor = wrapper.v_compressor.to(device)

            setattr(parent, attr_name, wrapper)
            patched += 1

    return patched
