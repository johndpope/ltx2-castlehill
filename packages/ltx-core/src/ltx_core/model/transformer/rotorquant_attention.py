"""RotorQuant-compressed attention for LTX-2.

Uses fused CUDA kernel when available (JIT-compiled on first use),
falls back to PyTorch einsum + searchsorted otherwise.

Usage:
    from ltx_core.model.transformer.rotorquant_attention import enable_rotorquant
    enable_rotorquant(model, bits=3)
"""

import math
import os
import torch
import torch.nn as nn

# ─── CUDA kernel loading ────────────────────────────────────────────
_cuda_kernel = None
_cuda_load_attempted = False

def _load_cuda_kernel():
    global _cuda_kernel, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_kernel
    _cuda_load_attempted = True
    try:
        from torch.utils.cpp_extension import load
        csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
        kernel_path = os.path.join(csrc_dir, 'rotor_fused_kernel.cu')
        if os.path.exists(kernel_path):
            _cuda_kernel = load(
                name='ltx_rotor_fused',
                sources=[kernel_path],
                extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--use_fast_math'],
                verbose=False,
            )
    except Exception:
        pass
    return _cuda_kernel


class RotorQuantCompressor(nn.Module):
    """Fast RotorQuant compressor with fused CUDA kernel.

    On first call, JIT-compiles the CUDA kernel. If compilation fails
    or no CUDA, falls back to PyTorch (einsum + searchsorted).
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_groups = (dim + 2) // 3
        self.n_levels = 2 ** bits

        rng = torch.Generator()
        rng.manual_seed(seed)
        rotors = torch.randn(self.n_groups, 4, generator=rng)
        rotors = rotors / rotors.norm(dim=-1, keepdim=True)

        # 3x3 rotation matrices for PyTorch path
        M = self._rotors_to_matrices(rotors)
        self.register_buffer('M', M)
        self.register_buffer('Mt', M.transpose(1, 2).contiguous())

        # Sparse rotor components for CUDA kernel path: (n_groups, 4) [s, b12, b13, b23]
        sparse = torch.stack([rotors[:, 0], rotors[:, 1], rotors[:, 2], rotors[:, 3]], dim=-1)
        self.register_buffer('rotors_sparse', sparse.float().contiguous())

        # Centroids (sorted)
        centroids = self._compute_centroids(bits, dim)
        self.register_buffer('centroids', centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        self.register_buffer('boundaries', boundaries)

    @staticmethod
    def _rotors_to_matrices(rotors):
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
    def _compute_centroids(bits, d):
        sigma = 1.0 / math.sqrt(max(d, 64))
        n = 2 ** bits
        if bits == 1:
            c = math.sqrt(2.0 / math.pi) * sigma
            return torch.tensor([-c, c])
        return torch.linspace(-3 * sigma, 3 * sigma, n)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compress. x: (B, S, D). Returns (indices uint8, norms fp16, orig_dim)."""
        B, S, D = x.shape
        norms = x.norm(dim=-1).to(torch.float16)
        safe_norms = norms.float().clamp(min=1e-8)
        x_unit = x / safe_norms.unsqueeze(-1)

        pad = (3 - D % 3) % 3
        if pad > 0:
            x_unit = torch.nn.functional.pad(x_unit, (0, pad))

        x_groups = x_unit.reshape(B, S, -1, 3)
        rotated = torch.einsum('bsgi,gij->bsgj', x_groups, self.M)
        flat = rotated.reshape(B * S, -1)
        indices = torch.searchsorted(self.boundaries, flat).clamp(0, self.n_levels - 1)
        indices = indices.to(torch.uint8).reshape(B, S, -1)
        return indices, norms, D

    @torch.no_grad()
    def decompress(self, indices: torch.Tensor, norms: torch.Tensor, orig_dim: int) -> torch.Tensor:
        """Decompress. Returns (B, S, D)."""
        B, S, _ = indices.shape
        flat = self.centroids[indices.long()]
        groups = flat.reshape(B, S, -1, 3)
        derotated = torch.einsum('bsgi,gij->bsgj', groups, self.Mt)
        out = derotated.reshape(B, S, -1)[:, :, :orig_dim]
        return out * norms.float().unsqueeze(-1)

    @torch.no_grad()
    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Full pipeline via fused CUDA kernel: normalize→embed→rotate→quantize→inverse→extract→rescale.
        Returns dequantized x_hat directly. ~100x faster than compress+decompress.
        """
        kernel = _load_cuda_kernel()
        if kernel is None or not x.is_cuda:
            idx, norms, dim = self.compress(x)
            return self.decompress(idx, norms, dim)

        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_2d = x.reshape(B * S, D)
        else:
            x_2d = x

        # Normalize (kernel operates on unit vectors like the PyTorch path)
        norms = x_2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_2d / norms

        x_hat = kernel.rotor_full_fused_float(
            x_unit.float().contiguous(),
            self.rotors_sparse,
            self.centroids.float(), self.n_levels,
            self.centroids.float(), self.n_levels,
            self.centroids.float(), self.n_levels,
            self.centroids.float(), self.n_levels,
        )

        # Rescale
        x_hat = x_hat * norms

        if len(orig_shape) == 3:
            x_hat = x_hat.reshape(orig_shape)
        return x_hat


class RotorQuantAttention(nn.Module):
    """Drop-in attention wrapper with K/V compression."""

    def __init__(self, original_attention: nn.Module, bits: int = 3, compress_values: bool = True):
        super().__init__()
        self.original = original_attention
        self.bits = bits
        self.compress_values = compress_values
        inner_dim = original_attention.heads * original_attention.dim_head
        self.k_compressor = RotorQuantCompressor(inner_dim, bits=bits, seed=42)
        self.v_compressor = RotorQuantCompressor(inner_dim, bits=bits, seed=4200) if compress_values else None

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None,
                perturbation_mask=None, all_perturbed=False):
        from ltx_core.model.transformer.rope import apply_rotary_emb
        attn = self.original
        context = x if context is None else context
        v = attn.to_v(context)

        if all_perturbed:
            out = v
        else:
            q = attn.to_q(x)
            k = attn.to_k(context)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            if pe is not None:
                q = apply_rotary_emb(q, pe, attn.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn.rope_type)

            k = self.k_compressor.forward_fused(k)
            if self.compress_values and self.v_compressor is not None:
                v = self.v_compressor.forward_fused(v)

            out = attn.attention_function(q, k, v, attn.heads, mask)
            if perturbation_mask is not None:
                v_raw = attn.to_v(context)
                out = out * perturbation_mask + v_raw * (1 - perturbation_mask)

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
    """Monkey-patch Attention modules with RotorQuant compression."""
    from ltx_core.model.transformer.attention import Attention
    patched = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, Attention):
            if layers is not None:
                layer_idx = None
                for part in name.split('.'):
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_idx is not None and layer_idx not in layers:
                    continue
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model
            for p in parent_name.split('.'):
                if p:
                    parent = getattr(parent, p)
            wrapper = RotorQuantAttention(module, bits=bits, compress_values=compress_values)
            device = module.to_q.weight.device
            wrapper.k_compressor = wrapper.k_compressor.to(device)
            if wrapper.v_compressor is not None:
                wrapper.v_compressor = wrapper.v_compressor.to(device)
            setattr(parent, attr_name, wrapper)
            patched += 1
    return patched
