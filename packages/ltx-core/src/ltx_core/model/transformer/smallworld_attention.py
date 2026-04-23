"""
SmallWorld Attention: multi-hop message passing on sparse sequence/video graphs.

Drop-in replacement for CliffordRollingAttention and CliffordVideoAttention.
Same interface (Q/K/V projections, RoPE, gating, cross-attention fallback)
but replaces single-hop sparse scoring with multi-hop message passing.

Key difference from Clifford:
  Clifford: each layer independently scores 16 sparse positions (1-hop)
  SmallWorld: each layer propagates messages along graph edges (multi-hop)
  After K layers, information reaches every position through relay nodes.

This fixes generation collapse in sparse attention — single-hop creates blind
spots at unattended positions that compound during autoregressive generation.
Multi-hop eliminates blind spots via relay.

Complexity: same O(L × S × d) per layer as Clifford — just different semantics.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb
from ltx_core.model.transformer.clifford_attention import (
    compute_seq_shifts, compute_temporal_shifts,
)


class SmallWorldAttention(nn.Module):
    """SmallWorld multi-hop message passing — drop-in for CliffordRollingAttention.

    Same interface: forward(x, context, mask, pe, k_pe, perturbation_mask, all_perturbed)
    Same projections: to_q, to_k, to_v, q_norm, k_norm, to_out, to_gate_logits
    Same cross-attention fallback for context != None.

    Difference: self-attention uses GAT-style neighbor gathering + scoring
    instead of independent per-layer sparse scoring. The graph structure
    (log-spaced shifts) is identical — the aggregation is what changes.
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
        # SmallWorld-specific
        num_seq_shifts: int = 32,
        max_seq_len: int = 2048,
        edge_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function
        self.is_cross_attention = context_dim is not None

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Message passing projections (replace Q/K/V with msg/key/val)
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)    # doubles as W_msg
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)  # doubles as W_key
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)  # doubles as W_val

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim, bias=True),
            torch.nn.Identity(),
        )

        # Graph structure: log-spaced shifts
        self.seq_shifts = compute_seq_shifts(num_seq_shifts, max_seq_len)
        self.num_seq_shifts = len(self.seq_shifts)

        # Learnable edge bias per shift per head
        self.edge_bias = nn.Parameter(torch.zeros(heads, self.num_seq_shifts))
        with torch.no_grad():
            for i, s in enumerate(self.seq_shifts):
                self.edge_bias.data[:, i] = 0.1 * math.log(1 + abs(s))

        self.edge_dropout = edge_dropout

    def _standard_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard attention for cross-attention fallback."""
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        out = self.attention_function(q, k, v, self.heads, mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)

    def _smallworld_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
        all_perturbed: bool,
    ) -> torch.Tensor:
        """SmallWorld multi-hop message passing (self-attention only).

        Each token gathers messages from its graph neighbors (log-spaced shifts),
        scores them via dot product + edge bias, softmax, and aggregates.
        Multi-hop relay happens across transformer layers (not within this call).
        """
        B, L, _ = x.shape
        H = self.heads
        D = self.dim_head

        v = self.to_v(x)

        if all_perturbed:
            out = v
        else:
            # Project to message/key/value spaces
            msg = self.to_q(x)
            key = self.to_k(x)

            msg = self.q_norm(msg)
            key = self.k_norm(key)

            if pe is not None:
                msg = apply_rotary_emb(msg, pe, self.rope_type)
                key = apply_rotary_emb(key, pe, self.rope_type)

            # Reshape to [B, L, H, D]
            msg = msg.view(B, L, H, D)
            key = key.view(B, L, H, D)
            v_4d = v.view(B, L, H, D)

            positions = torch.arange(L, device=x.device)
            all_scores = []
            all_neighbor_vals = []

            for i, s in enumerate(self.seq_shifts):
                neighbor_idx = (positions + s) % L

                k_neighbor = key[:, neighbor_idx, :, :]
                v_neighbor = v_4d[:, neighbor_idx, :, :]

                score = (msg * k_neighbor).sum(dim=-1) * self.scale
                score = score + self.edge_bias[:, i].view(1, 1, H)

                all_scores.append(score)
                all_neighbor_vals.append(v_neighbor)

            scores = torch.stack(all_scores, dim=-1)  # [B, L, H, num_shifts]

            # Apply mask if provided
            if mask is not None:
                for idx, s in enumerate(self.seq_shifts):
                    target_indices = (positions + s) % L
                    if mask.dim() == 4:
                        shift_mask = mask[:, :, positions, target_indices]
                        shift_mask = shift_mask.squeeze(1).unsqueeze(2)
                        scores[:, :, :, idx] = scores[:, :, :, idx].masked_fill(
                            shift_mask.expand_as(scores[:, :, :, idx]) == 0, -1e9
                        )

            # Edge dropout during training
            if self.training and self.edge_dropout > 0:
                edge_mask = torch.rand(1, 1, H, self.num_seq_shifts,
                                       device=x.device) > self.edge_dropout
                edge_mask[:, :, :, 0] = True  # always keep self-connection
                scores = scores.masked_fill(~edge_mask, float('-inf'))

            # Guard all-inf rows
            all_inf = (scores == float('-inf')).all(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_inf.expand_as(scores), 0.0)

            attn = F.softmax(scores, dim=-1)
            attn = attn.masked_fill(all_inf, 0.0)

            v_stack = torch.stack(all_neighbor_vals, dim=-1)
            out = (attn.unsqueeze(-2) * v_stack).sum(dim=-1)

            out = out.reshape(B, L, H * D)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Per-head gating
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            out = out.view(B, L, H, D)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.reshape(B, L, H * D)

        return self.to_out(out)

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
        if context is not None:
            return self._standard_attention(x, context, mask, pe, k_pe)
        return self._smallworld_attention(x, mask, pe, perturbation_mask, all_perturbed)


class SmallWorldVideoAttention(nn.Module):
    """SmallWorld multi-hop message passing for video — drop-in for CliffordVideoAttention.

    Same interface and behavior as SmallWorldAttention but with separate
    spatial (within-frame) and temporal (across-frame) graph edges.
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
        # Video-specific
        num_spatial_shifts: int = 12,
        num_temporal_shifts: int = 4,
        max_spatial_len: int = 2048,
        spherical_norm: bool = False,
        num_frames: int = 1,
        edge_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function
        self.is_cross_attention = context_dim is not None

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Projections
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim, bias=True),
            torch.nn.Identity(),
        )

        # Spatial shifts: within each frame
        self.spatial_shifts = compute_seq_shifts(num_spatial_shifts, max_spatial_len)
        self.num_spatial_shifts = len(self.spatial_shifts)

        # Temporal shifts: across frames
        self.temporal_shifts = compute_temporal_shifts(num_temporal_shifts)
        self.num_temporal_shifts = len(self.temporal_shifts)

        self.total_shifts = self.num_spatial_shifts + self.num_temporal_shifts

        # Learnable edge bias per shift per head
        self.edge_bias = nn.Parameter(torch.zeros(heads, self.total_shifts))
        with torch.no_grad():
            for i, s in enumerate(self.spatial_shifts):
                self.edge_bias.data[:, i] = 0.1 * math.log(1 + abs(s))
            for i, s in enumerate(self.temporal_shifts):
                self.edge_bias.data[:, self.num_spatial_shifts + i] = 0.2 * math.log(1 + abs(s))

        self.spherical_norm = spherical_norm
        self.num_frames = num_frames
        self.edge_dropout = edge_dropout

    def _standard_attention(self, x, context, mask, pe, k_pe):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
        out = self.attention_function(q, k, v, self.heads, mask)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)
        return self.to_out(out)

    def _video_smallworld_attention(
        self, x, mask, pe, perturbation_mask, all_perturbed, num_frames,
    ):
        B, L, _ = x.shape
        H = self.heads
        D = self.dim_head
        T = num_frames if num_frames is not None else self.num_frames
        S = L // T

        if L % T != 0:
            raise ValueError(f"Sequence length {L} not divisible by num_frames {T}.")

        v = self.to_v(x)

        if all_perturbed:
            out = v
        else:
            msg = self.to_q(x)
            key = self.to_k(x)
            msg = self.q_norm(msg)
            key = self.k_norm(key)

            if pe is not None:
                msg = apply_rotary_emb(msg, pe, self.rope_type)
                key = apply_rotary_emb(key, pe, self.rope_type)

            # Reshape to [B, T, S, H, D]
            msg = msg.view(B, T, S, H, D)
            key = key.view(B, T, S, H, D)
            v_5d = v.view(B, T, S, H, D)

            all_scores = []
            all_neighbor_vals = []

            # Spatial shifts: gather within each frame (dim=2)
            for s in self.spatial_shifts:
                nb_idx = (torch.arange(S, device=x.device) + s) % S
                k_nb = key[:, :, nb_idx, :, :]
                v_nb = v_5d[:, :, nb_idx, :, :]
                score = (msg * k_nb).sum(dim=-1) * self.scale  # [B, T, S, H]
                all_scores.append(score)
                all_neighbor_vals.append(v_nb)

            # Temporal shifts: gather across frames (dim=1)
            for t_shift in self.temporal_shifts:
                nb_idx = (torch.arange(T, device=x.device) + t_shift) % T
                k_nb = key[:, nb_idx, :, :, :]
                v_nb = v_5d[:, nb_idx, :, :, :]
                score = (msg * k_nb).sum(dim=-1) * self.scale
                all_scores.append(score)
                all_neighbor_vals.append(v_nb)

            # Stack: [B, T, S, H, total_shifts]
            scores = torch.stack(all_scores, dim=-1)

            # Add edge bias
            scores = scores + self.edge_bias.view(1, 1, 1, H, self.total_shifts)

            # Apply mask if provided
            if mask is not None and mask.dim() == 4:
                flat_indices = torch.arange(L, device=x.device)
                for idx in range(self.total_shifts):
                    if idx < self.num_spatial_shifts:
                        s = self.spatial_shifts[idx]
                        frame_pos = flat_indices % S
                        frame_start = flat_indices - frame_pos
                        target_indices = frame_start + (frame_pos + s) % S
                    else:
                        t_shift = self.temporal_shifts[idx - self.num_spatial_shifts]
                        frame_idx = flat_indices // S
                        spatial_pos = flat_indices % S
                        target_indices = ((frame_idx + t_shift) % T) * S + spatial_pos

                    shift_mask = mask[:, :, flat_indices, target_indices]
                    shift_mask = shift_mask.squeeze(1).view(B, T, S, 1)
                    scores[:, :, :, :, idx] = scores[:, :, :, :, idx].masked_fill(
                        shift_mask.expand_as(scores[:, :, :, :, idx]) == 0, -1e9
                    )

            # Edge dropout
            if self.training and self.edge_dropout > 0:
                edge_mask = torch.rand(1, 1, 1, H, self.total_shifts,
                                       device=x.device) > self.edge_dropout
                edge_mask[:, :, :, :, 0] = True
                scores = scores.masked_fill(~edge_mask, float('-inf'))

            # Guard all-inf
            all_inf = (scores == float('-inf')).all(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_inf.expand_as(scores), 0.0)

            attn = F.softmax(scores, dim=-1)
            attn = attn.masked_fill(all_inf, 0.0)

            v_stack = torch.stack(all_neighbor_vals, dim=-1)  # [B, T, S, H, D, total_shifts]
            # Transpose for broadcast: attn [B,T,S,H,shifts] × v [B,T,S,H,D,shifts]
            out = (attn.unsqueeze(-2) * v_stack).sum(dim=-1)  # [B, T, S, H, D]

            out = out.reshape(B, L, H * D)

            if self.spherical_norm:
                out = F.normalize(out, p=2, dim=-1)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            out = out.view(B, L, H, D)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.reshape(B, L, H * D)

        return self.to_out(out)

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
            return self._standard_attention(x, context, mask, pe, k_pe)
        return self._video_smallworld_attention(
            x, mask, pe, perturbation_mask, all_perturbed, num_frames
        )
