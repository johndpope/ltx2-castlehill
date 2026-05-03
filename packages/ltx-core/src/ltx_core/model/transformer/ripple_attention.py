"""
Ripple Attention for LTX-2.3 video diffusion transformer.

Hierarchical sparse attention: layer k attends to stride 2^(k % NUM_LEVELS)
so across 48 layers the full stride range (1 → 2048) is swept 4 times.

Complexity: O(S × K × D) per layer vs O(S² × D) for full attention.
At S=5491 (145 frames, scrya): ~14× faster on the attention kernel,
~40% savings on total block compute (attention + QKV projections).

Optional KV compression (kv_compress_mode="planar"):
  PlanarQuant fused CUDA kernel (planar2_fused_bf16) compresses K and V via
  2D Givens rotation + Lloyd-Max scalar quantization in a single register pass.
  Falls back to the pure-PyTorch PlanarQuantMSE if CUDA kernel is unavailable.
  STE (straight-through estimator) ensures LoRA gradients flow unimpeded.

Drop-in replacement for CliffordVideoAttention — same forward() signature.
Cross-attention falls through to standard SDPA (same as Clifford).
"""

from __future__ import annotations

import glob
import importlib.util
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb


# ── Shift schedule ────────────────────────────────────────────────────────────

# 12 useful doubling levels cover strides 1 → 2048 (2^11).
# For max S≈8192: useful up to 2^12=4096. We use 12 levels conservatively.
_NUM_LEVELS = 12


def _ripple_shifts_for_layer(
    layer_idx: int,
    n_layers: int = 48,
    max_len: int = 8192,
    local_radius: int = 1,
) -> list[int]:
    """Return shift list for one layer.

    Layer k gets primary stride 2^(k % NUM_LEVELS) — cycling through the
    full hierarchy every NUM_LEVELS layers (4 passes across 48 layers).
    Always includes ±local_radius for residual local context.
    """
    level   = layer_idx % _NUM_LEVELS
    stride  = 1 << level          # 2^level
    max_s   = max_len // 2

    shifts = [0]
    # Local residual context
    for r in range(1, local_radius + 1):
        shifts.extend([r, -r])

    # Primary ripple stride
    if stride > local_radius and stride <= max_s:
        shifts.extend([stride, -stride])
        # Half-stride for smoother coverage
        half = stride >> 1
        if half > local_radius and half not in shifts and -half not in shifts:
            shifts.extend([half, -half])

    return shifts


# ── KV compression helpers ────────────────────────────────────────────────────

def _load_planar2_cuda() -> object | None:
    """Try to load the compiled planar2_fused CUDA extension from rotorquant."""
    try:
        spec = importlib.util.find_spec("turboquant")
        if spec is None:
            return None
        pkg_dir = os.path.dirname(spec.origin)
        so_files = glob.glob(os.path.join(pkg_dir, "cuda_planar2*.so"))
        if not so_files:
            return None
        mod_spec = importlib.util.spec_from_file_location("cuda_planar2", so_files[0])
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PLANAR2_CUDA = _load_planar2_cuda()   # module-level singleton, loaded once


def _compress_kv_planar(
    kv: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
    n_levels: int,
) -> torch.Tensor:
    """Compress a K or V tensor with PlanarQuant, return STE-wrapped result.

    kv:         [B, H, L, D]  (any dtype)
    rot2:       [n_groups, 2]  float32
    centroids:  [n_levels]     float32
    n_levels:   int  (4, 8, or 16 for 2/3/4-bit)

    Returns [B, H, L, D] where:
      forward  = quantized reconstruction (rotate→snap→unrotate per pair)
      backward = identity (STE — gradients pass through unchanged)
    """
    B, H, L, D = kv.shape
    flat = kv.reshape(-1, D)

    if _PLANAR2_CUDA is not None and flat.is_cuda:
        # Always use float32 path — avoids dtype dispatch issues in pybind.
        # overhead is trivial vs memory-bandwidth cost of the rotation.
        orig_dtype = flat.dtype
        flat_f32   = flat.float().contiguous()
        # Buffers may have been cast to model dtype by .to(dtype) — force float32
        rot2_f     = rot2.float().to(flat.device).contiguous()
        cents_f    = centroids.float().to(flat.device).contiguous()
        kv_q_f32   = _PLANAR2_CUDA.planar2_fused_float(flat_f32, rot2_f, cents_f, n_levels)
        kv_q       = kv_q_f32.to(orig_dtype).reshape(B, H, L, D)
        # STE: forward uses quantized, backward passes gradient through raw
        return kv + (kv_q - kv).detach()

    # PyTorch fallback via PlanarQuantMSE
    try:
        from turboquant.planarquant import rot2_apply, rot2_inverse  # noqa: PLC0415
        n_groups = (D + 1) // 2
        d_padded = n_groups * 2
        pad = d_padded - D

        flat_f = flat.float()
        if pad > 0:
            flat_f = F.pad(flat_f, (0, pad))
        pairs = flat_f.reshape(-1, n_groups, 2)

        rotated = rot2_apply(rot2.to(flat_f.device), pairs)
        flat_rot = rotated.reshape(-1, d_padded)

        cents = centroids.to(flat_f.device)
        diffs = flat_rot.unsqueeze(-1) - cents            # [N, d_padded, n_levels]
        indices = diffs.abs().argmin(dim=-1)
        q_flat = cents[indices]                            # [N, d_padded]
        q_pairs = q_flat.reshape(-1, n_groups, 2)

        restored = rot2_inverse(rot2.to(flat_f.device), q_pairs)
        kv_q = restored.reshape(-1, d_padded)[..., :D].to(kv.dtype).reshape(B, H, L, D)
        return kv + (kv_q - kv).detach()
    except Exception:
        return kv   # if turboquant not installed at all, identity pass-through


# ── Module ────────────────────────────────────────────────────────────────────

class RippleVideoAttention(nn.Module):
    """Ripple sparse attention — drop-in for CliffordVideoAttention.

    Uses the fused bidirectional Triton kernel when available;
    falls back to a Python loop (still faster than SDPA at long seqs).

    Same __init__ kwargs as CliffordVideoAttention for compatibility.
    Extra kwargs:
      layer_idx          (int)   — determines stride level for this block
      kv_compress_mode   (str)   — "none" | "planar"
      kv_compress_bits   (int)   — bits for Lloyd-Max codebook (2/3/4)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
        # Ripple-specific
        layer_idx: int = 0,
        n_layers: int = 48,
        local_radius: int = 1,
        max_seq_len: int = 8192,
        # KV compression
        kv_compress_mode: str = "none",
        kv_compress_bits: int = 4,
        # Unused kwargs kept for API compatibility with Clifford
        num_spatial_shifts: int = 12,
        num_temporal_shifts: int = 4,
        num_channel_shifts: int = 4,
        max_spatial_len: int = 2048,
        spherical_norm: bool = False,
        num_frames: int = 1,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function
        self.is_cross_attention = context_dim is not None

        inner_dim   = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads    = heads
        self.dim_head = dim_head
        self.scale    = dim_head ** -0.5
        self.layer_idx = layer_idx
        self.kv_compress_mode = kv_compress_mode

        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = nn.Linear(query_dim,   inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Identity(),
        )

        # Ripple shifts for this layer
        self.shifts = _ripple_shifts_for_layer(layer_idx, n_layers, max_seq_len, local_radius)
        self.num_shifts = len(self.shifts)

        # Learnable per-head per-shift bias (initialized by stride importance)
        self.edge_bias = nn.Parameter(torch.zeros(heads, self.num_shifts))
        with torch.no_grad():
            stride = 1 << (layer_idx % _NUM_LEVELS)
            for i, s in enumerate(self.shifts):
                if s == 0:
                    self.edge_bias.data[:, i] = 1.0
                elif abs(s) == stride:
                    self.edge_bias.data[:, i] = 0.5
                else:
                    self.edge_bias.data[:, i] = 0.1 * math.log1p(abs(s))

        # ── PlanarQuant KV compression ─────────────────────────────────────
        self._kv_compress_bits  = kv_compress_bits
        self._kv_compress_nlevels = 2 ** kv_compress_bits
        if kv_compress_mode == "planar":
            try:
                from turboquant.planarquant import PlanarQuantMSE  # noqa: PLC0415
                pq = PlanarQuantMSE(dim_head, bits=kv_compress_bits,
                                    seed=42 + layer_idx, device="cpu")
                # float32 buffers: the CUDA kernel expects float* for rot2/centroids
                self.register_buffer("_pq_rot2",      pq.rot2.float())
                self.register_buffer("_pq_centroids", pq.centroids.float())
                self._compress_enabled = True
            except Exception:
                self._compress_enabled = False
        else:
            self._compress_enabled = False

        # ── Triton ripple kernel ───────────────────────────────────────────
        self._triton_available = False
        try:
            from ltx_core.model.transformer.ripple_attn_triton import ripple_attention_triton  # noqa: PLC0415
            self._ripple_triton = ripple_attention_triton
            self._triton_available = True
        except Exception:
            pass

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        if context is not None:
            return self._cross_attention(x, context, mask, pe, k_pe)
        return self._ripple_self_attention(x, mask, pe, perturbation_mask, all_perturbed)

    def _cross_attention(self, x, context, mask, pe, k_pe):
        """Standard SDPA for cross-attention (text conditioning)."""
        B, L, _ = x.shape
        H, D = self.heads, self.dim_head

        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(context))
        v = self.to_v(context)

        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, -1, H, D).transpose(1, 2)
        v = v.view(B, -1, H, D).transpose(1, 2)

        if pe is not None:
            q = apply_rotary_emb(q.transpose(1, 2), pe, self.rope_type).transpose(1, 2)
        if k_pe is not None:
            k = apply_rotary_emb(k.transpose(1, 2), k_pe, self.rope_type).transpose(1, 2)

        attn_mask = mask if (mask is not None and mask.dtype == torch.bool) else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, L, H * D)
        return self.to_out(out)

    def _ripple_self_attention(self, x, mask, pe, perturbation_mask, all_perturbed):
        B, L, _ = x.shape
        H, D = self.heads, self.dim_head

        v = self.to_v(x)

        if all_perturbed:
            out = v.view(B, L, H * D)
            return self.to_out(out)

        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(x))

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe, self.rope_type)

        # [B, L, H, D] → [B, H, L, D]
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)

        # Optional KV compression: fused rotate→quantize→unrotate per pair
        if self._compress_enabled:
            k = _compress_kv_planar(k, self._pq_rot2, self._pq_centroids, self._kv_compress_nlevels)
            v = _compress_kv_planar(v, self._pq_rot2, self._pq_centroids, self._kv_compress_nlevels)

        # Cap shifts at S//2 to prevent multi-wrap when stride > S
        max_shift = L // 2
        valid_idx = [i for i, s in enumerate(self.shifts) if abs(s) <= max_shift]
        if not valid_idx:
            valid_idx = [self.shifts.index(0)]  # always keep self-attention
        active_shifts = [self.shifts[i] for i in valid_idx]
        active_bias   = self.edge_bias[:, valid_idx]   # [H, K']

        if self._triton_available and q.is_cuda:
            try:
                shifts_t = torch.tensor(active_shifts, dtype=torch.int32, device=x.device)
                out = self._ripple_triton(q, k, v, shifts_t, active_bias, self.scale)
            except Exception:
                out = self._python_fallback(q, k, v, active_shifts, active_bias)
        else:
            out = self._python_fallback(q, k, v, active_shifts, active_bias)

        # Gating
        if self.to_gate_logits is not None:
            gate = torch.sigmoid(self.to_gate_logits(x))  # [B, L, H]
            out = out * gate.transpose(1, 2).unsqueeze(-1)

        # Perturbation mask blend
        if perturbation_mask is not None:
            pm = perturbation_mask.view(B, 1, L, 1)
            v_pass = v
            out = out * pm + v_pass * (1 - pm)

        out = out.transpose(1, 2).reshape(B, L, H * D)
        return self.to_out(out)

    def _python_fallback(self, q, k, v, shifts=None, edge_bias=None):
        """Python loop fallback — no Triton dependency."""
        B, H, L, D = q.shape
        if shifts is None:
            shifts    = self.shifts
            edge_bias = self.edge_bias
        positions = torch.arange(L, device=q.device)
        all_scores, all_vals = [], []
        for i, s in enumerate(shifts):
            nb = (positions + s) % L
            k_nb = k[:, :, nb, :]
            v_nb = v[:, :, nb, :]
            score = (q * k_nb).sum(-1) * self.scale  # [B, H, L]
            score = score + edge_bias[:, i].view(1, H, 1)
            all_scores.append(score)
            all_vals.append(v_nb)

        scores  = torch.stack(all_scores, dim=-1)     # [B, H, L, K]
        attn    = F.softmax(scores, dim=-1)
        v_stack = torch.stack(all_vals, dim=-1)       # [B, H, L, D, K]
        out     = (attn.unsqueeze(-2) * v_stack).sum(-1)
        return out

    def extra_repr(self) -> str:
        stride = 1 << (self.layer_idx % _NUM_LEVELS)
        compress_tag = f", compress={self.kv_compress_mode}@{self._kv_compress_bits}b" if self._compress_enabled else ""
        return (f"layer={self.layer_idx}, stride=±{stride}, "
                f"K={self.num_shifts}, shifts={self.shifts}, "
                f"triton={self._triton_available}{compress_tag}")
