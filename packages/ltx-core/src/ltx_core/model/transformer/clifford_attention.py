"""
Clifford Rolling Attention: sparse geometric attention with sub-quadratic complexity.

Replaces the O(L² × d) score computation in standard attention with
O(L × d × num_seq_shifts × num_score_terms) sparse rolling scores.

Each token attends to a fixed set of positions determined by bidirectional
log-spaced sequence shifts, with additional scoring diversity from
channel-shifted geometric products (inner/wedge terms from CliffordNet).

For self-attention only — cross-attention falls back to standard attention
since Q and K come from different sequences of different lengths.

Complexity comparison (video self-attn, L=1344, H=32, D=128):
  Standard:  O(L² × H × D) = ~7.4G multiply-adds
  Rolling:   O(L × H × D × seq_shifts × score_terms) = ~440M  (~17× fewer)

For longer sequences the savings grow quadratically:
  L=5000: Standard ~102G, Rolling ~1.6G  (~64× fewer)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb


def compute_seq_shifts(num_shifts: int, max_len: int = 2048) -> list[int]:
    """Bidirectional log-spaced sequence shifts.

    Generates shifts like [0, 1, -1, 3, -3, 11, -11, 38, -38, ...]
    covering local, medium-range, and long-range positions.

    Args:
        num_shifts: Total number of shifts (including self=0)
        max_len: Maximum sequence length to scale shifts to

    Returns:
        List of integer shifts, starting with 0 (self-attention)
    """
    if num_shifts <= 1:
        return [0]

    # We need (num_shifts - 1) non-zero shifts → ceil-half positive, floor-half negative
    # For num_shifts=16: 0 + 8 positive + 7 negative (extra positive for long-range)
    half = (num_shifts - 1 + 1) // 2  # ceil
    max_s = max(max_len // 2, 2)

    if half <= 1:
        positive = [1]
    else:
        # Log-uniform across [1, max_s] — generate exactly `half` unique values
        # Use log-linear interpolation: exp(linspace(0, log(max_s), half))
        positive = sorted(set(
            max(1, round(math.exp(math.log(max_s) * i / (half - 1))))
            for i in range(half)
        ))
        # If dedup removed values, fill gaps with linear interpolation
        while len(positive) < half:
            # Insert midpoints between consecutive values
            new_vals = []
            for i in range(len(positive) - 1):
                mid = (positive[i] + positive[i + 1]) // 2
                if mid not in positive and mid not in new_vals:
                    new_vals.append(mid)
                if len(positive) + len(new_vals) >= half:
                    break
            if not new_vals:
                # Last resort: add next integers
                v = positive[-1] + 1
                while len(positive) + len(new_vals) < half and v <= max_s:
                    new_vals.append(v)
                    v += 1
            positive = sorted(set(positive + new_vals))[:half]

    shifts = [0]
    for s in positive:
        shifts.append(s)
        shifts.append(-s)

    return shifts[:num_shifts]


def compute_channel_shifts(num_shifts: int) -> list[int]:
    """Exponential channel shifts for geometric product diversity.

    Args:
        num_shifts: Number of channel shifts (e.g., 4 → [1, 2, 4, 8])
    """
    return [1 << i for i in range(num_shifts)]


class CliffordRollingAttention(nn.Module):
    """Sparse rolling geometric attention — drop-in replacement for Attention.

    For self-attention: uses sparse rolling scores with geometric product terms.
    For cross-attention: falls back to standard scaled dot-product attention.

    The module keeps the same Q/K/V projections, QK norms, RoPE, gated attention,
    and output projection as the standard Attention class.

    Args:
        query_dim: Input query dimension
        context_dim: Context dimension (None = self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
        norm_eps: Epsilon for RMSNorm
        rope_type: RoPE embedding type
        attention_function: Fallback attention function for cross-attention
        apply_gated_attention: Whether to use per-head sigmoid gating
        num_seq_shifts: Number of sequence position shifts (bidirectional log-spaced)
        num_channel_shifts: Number of channel shifts for geometric scoring (0=disable)
        max_seq_len: Maximum expected sequence length (for shift computation)
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
        # Clifford-specific
        num_seq_shifts: int = 16,
        num_channel_shifts: int = 4,
        max_seq_len: int = 2048,
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

        # Q/K/V projections (same as standard Attention)
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim, bias=True),
            torch.nn.Identity(),
        )

        # Clifford rolling config (only used for self-attention)
        self.seq_shifts = compute_seq_shifts(num_seq_shifts, max_seq_len)
        self.channel_shifts = compute_channel_shifts(num_channel_shifts) if num_channel_shifts > 0 else []

        # Number of score terms per sequence shift:
        # 1 (standard dot product) + num_channel_shifts (geometric terms)
        self.scores_per_seq_shift = 1 + len(self.channel_shifts)

        # Total attention targets = num_seq_shifts
        # Each target has scores_per_seq_shift score contributions
        self.num_seq_shifts = len(self.seq_shifts)

        # Learnable mixing weights for combining score terms within each shift
        # Maps scores_per_seq_shift scores → 1 scalar per seq_shift per head
        if self.scores_per_seq_shift > 1:
            self.score_mix = nn.Linear(self.scores_per_seq_shift, 1, bias=False)
        else:
            self.score_mix = None

    def _standard_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard attention (used for cross-attention fallback)."""
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

    def _rolling_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sparse rolling geometric attention (self-attention only).

        Complexity: O(L × H × D × num_seq_shifts × scores_per_shift)
        vs standard: O(L × L × H × D)
        """
        B, L, _ = x.shape
        H = self.heads
        D = self.dim_head

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe, self.rope_type)

        # Reshape to [B, L, H, D]
        q = q.view(B, L, H, D)
        k = k.view(B, L, H, D)
        v = v.view(B, L, H, D)

        num_shifts = len(self.seq_shifts)

        # Compute sparse rolling scores: [B, L, H, num_shifts]
        # For each sequence shift s:
        #   base_score[i] = dot(Q[i], K[(i+s)%L]) / sqrt(D)
        #   geo_score_c[i] = dot(roll_channel(Q[i], c), K[(i+s)%L]) / sqrt(D)
        # Then mix all score terms into a single score per shift

        all_scores = []  # Will be [B, L, H, num_shifts, scores_per_shift]
        shifted_v = []   # [num_shifts, B, L, H, D]

        for s in self.seq_shifts:
            k_shifted = torch.roll(k, shifts=s, dims=1)  # [B, L, H, D]
            v_shifted = torch.roll(v, shifts=s, dims=1)
            shifted_v.append(v_shifted)

            # Base dot product score: O(B * L * H * D)
            base_score = (q * k_shifted).sum(dim=-1) * self.scale  # [B, L, H]
            shift_scores = [base_score]

            # Channel-shifted geometric scores
            for c in self.channel_shifts:
                q_rolled = torch.roll(q, shifts=c, dims=-1)  # roll along D (channel dim)
                geo_score = (q_rolled * k_shifted).sum(dim=-1) * self.scale  # [B, L, H]
                shift_scores.append(geo_score)

            # Stack score terms: [B, L, H, scores_per_shift]
            all_scores.append(torch.stack(shift_scores, dim=-1))

        # [B, L, H, num_shifts, scores_per_shift]
        all_scores = torch.stack(all_scores, dim=3)

        # Mix score terms per shift → [B, L, H, num_shifts]
        if self.score_mix is not None:
            scores = self.score_mix(all_scores).squeeze(-1)  # [B, L, H, num_shifts]
        else:
            scores = all_scores.squeeze(-1)  # [B, L, H, num_shifts]

        # Apply mask if provided
        if mask is not None:
            # mask is typically [B, 1, L, L] or [1, 1, L, L]
            # We need to extract the relevant positions for each shift
            # For simplicity, we create a sparse mask from the dense mask
            for idx, s in enumerate(self.seq_shifts):
                # For shift s, token i attends to token (i+s)%L
                target_indices = (torch.arange(L, device=x.device) + s) % L
                source_indices = torch.arange(L, device=x.device)
                # Extract mask values for these specific (source, target) pairs
                if mask.dim() == 4:
                    shift_mask = mask[:, :, source_indices, target_indices]  # [B, 1, L]
                    shift_mask = shift_mask.squeeze(1).unsqueeze(2)  # [B, L, 1]
                    scores[:, :, :, idx] = scores[:, :, :, idx].masked_fill(
                        shift_mask.expand_as(scores[:, :, :, idx]) == 0, -1e9
                    )

        # Softmax over the num_shifts dimension
        attn_weights = F.softmax(scores, dim=-1)  # [B, L, H, num_shifts]

        # Weighted sum of shifted values
        # shifted_v: list of [B, L, H, D], length = num_shifts
        # Stack to [B, L, H, num_shifts, D]
        v_stack = torch.stack(shifted_v, dim=3)

        # attn_weights: [B, L, H, num_shifts] → [B, L, H, num_shifts, 1]
        out = (attn_weights.unsqueeze(-1) * v_stack).sum(dim=3)  # [B, L, H, D]

        # Reshape back to [B, L, H*D]
        out = out.reshape(B, L, H * D)

        # Apply per-head gating
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
    ) -> torch.Tensor:
        """Forward pass — routes to rolling (self-attn) or standard (cross-attn).

        Args:
            x: Query input [B, L, query_dim]
            context: Key/Value input [B, C, context_dim] (None for self-attention)
            mask: Attention mask
            pe: Positional embeddings (RoPE cos/sin tuple)
            k_pe: Key positional embeddings (for cross-modal attention)
        """
        if context is not None:
            # Cross-attention: different sequence lengths, rolling doesn't apply
            return self._standard_attention(x, context, mask, pe, k_pe)

        return self._rolling_attention(x, mask, pe)
