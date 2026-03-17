"""
Clifford Rolling Attention: sparse geometric attention with sub-quadratic complexity.

Replaces the O(L^2 x d) score computation in standard attention with
O(L x d x num_seq_shifts x num_score_terms) sparse rolling scores.

Each token attends to a fixed set of positions determined by bidirectional
log-spaced sequence shifts, with additional scoring diversity from
channel-shifted geometric products (inner/wedge terms from CliffordNet).

For self-attention only -- cross-attention falls back to standard attention
since Q and K come from different sequences of different lengths.

Complexity comparison (video self-attn, L=1344, H=32, D=128):
  Standard:  O(L^2 x H x D) = ~7.4G multiply-adds
  Rolling:   O(L x H x D x seq_shifts x score_terms) = ~440M  (~17x fewer)

For longer sequences the savings grow quadratically:
  L=5000: Standard ~102G, Rolling ~1.6G  (~64x fewer)

CliffordVideoAttention adds video-aware temporal/spatial rolling on top.
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

    # We need (num_shifts - 1) non-zero shifts -> ceil-half positive, floor-half negative
    # For num_shifts=16: 0 + 8 positive + 7 negative (extra positive for long-range)
    half = (num_shifts - 1 + 1) // 2  # ceil
    max_s = max(max_len // 2, 2)

    if half <= 1:
        positive = [1]
    else:
        # Log-uniform across [1, max_s] -- generate exactly `half` unique values
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
        num_shifts: Number of channel shifts (e.g., 4 -> [1, 2, 4, 8])
    """
    return [1 << i for i in range(num_shifts)]


def compute_temporal_shifts(num_shifts: int) -> list[int]:
    """Non-zero bidirectional integer shifts for temporal (cross-frame) rolling.

    Generates shifts like [1, -1, 2, -2, ...] (excludes 0 since spatial shifts
    already cover same-frame self-attention).

    Args:
        num_shifts: Total number of temporal shifts (should be even for symmetry)

    Returns:
        List of non-zero integer shifts
    """
    if num_shifts <= 0:
        return []
    shifts = []
    t = 1
    while len(shifts) < num_shifts:
        shifts.append(t)
        if len(shifts) < num_shifts:
            shifts.append(-t)
        t += 1
    return shifts


class CliffordRollingAttention(nn.Module):
    """Sparse rolling geometric attention -- drop-in replacement for Attention.

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
        # Maps scores_per_seq_shift scores -> 1 scalar per seq_shift per head
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
        perturbation_mask: torch.Tensor | None,
        all_perturbed: bool,
    ) -> torch.Tensor:
        """Sparse rolling geometric attention (self-attention only).

        Complexity: O(L x H x D x num_seq_shifts x scores_per_shift)
        vs standard: O(L x L x H x D)

        Args:
            x: Input tensor [B, L, query_dim]
            mask: Attention mask
            pe: Positional embeddings (RoPE)
            perturbation_mask: Optional [0,1] mask blending attn output with raw V.
                1 keeps full attention, 0 bypasses to value pass-through.
            all_perturbed: If True, skip Q/K computation entirely; pass V through.
        """
        B, L, _ = x.shape
        H = self.heads
        D = self.dim_head

        v = self.to_v(x)

        if all_perturbed:
            # Skip Q/K path entirely, pass V through
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(x)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe, self.rope_type)

            # Reshape to [B, L, H, D]
            q = q.view(B, L, H, D)
            k = k.view(B, L, H, D)
            v_4d = v.view(B, L, H, D)

            all_scores = []  # Will be [B, L, H, num_shifts, scores_per_shift]
            shifted_v = []   # [num_shifts, B, L, H, D]

            for s in self.seq_shifts:
                k_shifted = torch.roll(k, shifts=s, dims=1)  # [B, L, H, D]
                v_shifted = torch.roll(v_4d, shifts=s, dims=1)
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

            # Mix score terms per shift -> [B, L, H, num_shifts]
            if self.score_mix is not None:
                scores = self.score_mix(all_scores).squeeze(-1)  # [B, L, H, num_shifts]
            else:
                scores = all_scores.squeeze(-1)  # [B, L, H, num_shifts]

            # Apply mask if provided
            if mask is not None:
                for idx, s in enumerate(self.seq_shifts):
                    target_indices = (torch.arange(L, device=x.device) + s) % L
                    source_indices = torch.arange(L, device=x.device)
                    if mask.dim() == 4:
                        shift_mask = mask[:, :, source_indices, target_indices]  # [B, 1, L]
                        shift_mask = shift_mask.squeeze(1).unsqueeze(2)  # [B, L, 1]
                        scores[:, :, :, idx] = scores[:, :, :, idx].masked_fill(
                            shift_mask.expand_as(scores[:, :, :, idx]) == 0, -1e9
                        )

            # Softmax over the num_shifts dimension
            attn_weights = F.softmax(scores, dim=-1)  # [B, L, H, num_shifts]

            # Weighted sum of shifted values
            v_stack = torch.stack(shifted_v, dim=3)  # [B, L, H, num_shifts, D]
            out = (attn_weights.unsqueeze(-1) * v_stack).sum(dim=3)  # [B, L, H, D]

            # Reshape back to [B, L, H*D]
            out = out.reshape(B, L, H * D)

            # Blend with raw V using perturbation mask
            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

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
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        """Forward pass -- routes to rolling (self-attn) or standard (cross-attn).

        Args:
            x: Query input [B, L, query_dim]
            context: Key/Value input [B, C, context_dim] (None for self-attention)
            mask: Attention mask
            pe: Positional embeddings (RoPE cos/sin tuple)
            k_pe: Key positional embeddings (for cross-modal attention)
            perturbation_mask: Optional [0,1] mask blending attn output with raw V.
                1 keeps full attention, 0 bypasses to value pass-through.
            all_perturbed: If True, skip Q/K computation entirely; pass V through.
        """
        if context is not None:
            # Cross-attention: different sequence lengths, rolling doesn't apply
            return self._standard_attention(x, context, mask, pe, k_pe)

        return self._rolling_attention(x, mask, pe, perturbation_mask, all_perturbed)


class CliffordVideoAttention(nn.Module):
    """Video-aware sparse geometric attention with separate temporal and spatial rolling.

    Key differences from CliffordRollingAttention:
    - Reshapes [B, T*H*W, D] -> [B, T, H*W, D] for structured rolling
    - Temporal shifts: rolls along frame dimension (dim=1) for motion coherence
    - Spatial shifts: rolls along spatial tokens within each frame (dim=2)
    - Channel shifts for geometric product diversity (same as CliffordRollingAttention)
    - Optional spherical norm for VFM S^127 compatibility

    Complexity: O(B * T * S * H * D * (num_spatial_shifts + num_temporal_shifts) * scores_per_shift)
    vs standard: O(B * L^2 * H * D) where L = T*S

    For self-attention only -- cross-attention falls back to standard attention.

    Args:
        query_dim: Input query dimension
        context_dim: Context dimension (None = self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
        norm_eps: Epsilon for RMSNorm
        rope_type: RoPE embedding type
        attention_function: Fallback attention function for cross-attention
        apply_gated_attention: Whether to use per-head sigmoid gating
        num_spatial_shifts: Bidirectional log-spaced shifts within each frame
        num_temporal_shifts: Shifts across frames (non-zero: +/-1, +/-2, ...)
        num_channel_shifts: Channel shifts for geometric product diversity
        max_spatial_len: Max tokens per frame for spatial shift computation
        spherical_norm: Apply L2 normalization to output (for VFM)
        num_frames: Number of video frames (can be overridden at forward time)
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
        num_channel_shifts: int = 4,
        max_spatial_len: int = 2048,
        spherical_norm: bool = False,
        num_frames: int = 1,
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

        # Spatial shifts: bidirectional log-spaced within each frame
        self.spatial_shifts = compute_seq_shifts(num_spatial_shifts, max_spatial_len)
        self.num_spatial_shifts = len(self.spatial_shifts)

        # Temporal shifts: non-zero integer shifts across frames
        self.temporal_shifts = compute_temporal_shifts(num_temporal_shifts)
        self.num_temporal_shifts = len(self.temporal_shifts)

        # Channel shifts for geometric product diversity
        self.channel_shifts = compute_channel_shifts(num_channel_shifts) if num_channel_shifts > 0 else []

        # Number of score terms per shift position:
        # 1 (standard dot product) + num_channel_shifts (geometric terms)
        self.scores_per_shift = 1 + len(self.channel_shifts)

        # Total number of attention targets = spatial + temporal
        self.total_shifts = self.num_spatial_shifts + self.num_temporal_shifts

        # Learnable mixing weights for combining score terms within each shift
        if self.scores_per_shift > 1:
            self.score_mix = nn.Linear(self.scores_per_shift, 1, bias=False)
        else:
            self.score_mix = None

        # Learned per-head sigmoid gates (replaces softmax over shifts)
        # Each position projects to per-head, per-shift gate values [0,1]
        # This ensures ALL shifts contribute (no winner-take-all suppression)
        inner_dim = dim_head * heads
        self.shift_gate_proj = nn.Linear(inner_dim, self.total_shifts * heads, bias=True)

        self.spherical_norm = spherical_norm
        self.num_frames = num_frames

    def _compute_shift_scores(
        self,
        q: torch.Tensor,
        k_shifted: torch.Tensor,
    ) -> torch.Tensor:
        """Compute base + geometric scores for a shifted key tensor.

        Args:
            q: Query tensor (any shape with last dim = D)
            k_shifted: Shifted key tensor (same shape as q)

        Returns:
            Score tensor with shape [..., scores_per_shift]
        """
        # Base dot product score
        base_score = (q * k_shifted).sum(dim=-1) * self.scale
        shift_scores = [base_score]

        # Channel-shifted geometric scores
        for c in self.channel_shifts:
            q_rolled = torch.roll(q, shifts=c, dims=-1)
            geo_score = (q_rolled * k_shifted).sum(dim=-1) * self.scale
            shift_scores.append(geo_score)

        return torch.stack(shift_scores, dim=-1)

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

    def _video_rolling_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        perturbation_mask: torch.Tensor | None,
        all_perturbed: bool,
        num_frames: int | None,
    ) -> torch.Tensor:
        """Video-aware sparse rolling geometric attention (self-attention only).

        Reshapes the flat sequence [B, T*S, D] into [B, T, S, H, D] and applies
        spatial rolling within frames and temporal rolling across frames.

        Args:
            x: Input tensor [B, L, query_dim] where L = T * S
            mask: Attention mask
            pe: Positional embeddings (RoPE)
            perturbation_mask: Optional [0,1] mask blending attn output with raw V.
            all_perturbed: If True, skip Q/K computation entirely; pass V through.
            num_frames: Number of frames (overrides self.num_frames if provided)
        """
        B, L, _ = x.shape
        H = self.heads
        D = self.dim_head
        T = num_frames if num_frames is not None else self.num_frames
        S = L // T  # tokens per frame

        if L % T != 0:
            raise ValueError(
                f"Sequence length {L} is not evenly divisible by num_frames {T}. "
                f"Ensure L = T * S (tokens_per_frame)."
            )

        v = self.to_v(x)

        if all_perturbed:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(x)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe, self.rope_type)

            # Reshape to [B, T, S, H, D] for structured rolling
            q = q.view(B, T, S, H, D)
            k = k.view(B, T, S, H, D)
            v_5d = v.view(B, T, S, H, D)

            all_scores = []   # Will hold [B, T, S, H, scores_per_shift] per shift
            all_shifted_v = []  # Corresponding shifted values

            # --- Spatial shifts: roll within each frame (dim=2) ---
            for s in self.spatial_shifts:
                k_shifted = torch.roll(k, shifts=s, dims=2)  # [B, T, S, H, D]
                v_shifted = torch.roll(v_5d, shifts=s, dims=2)
                all_shifted_v.append(v_shifted)
                all_scores.append(self._compute_shift_scores(q, k_shifted))

            # --- Temporal shifts: roll across frames (dim=1) ---
            for t_shift in self.temporal_shifts:
                k_shifted = torch.roll(k, shifts=t_shift, dims=1)  # [B, T, S, H, D]
                v_shifted = torch.roll(v_5d, shifts=t_shift, dims=1)
                all_shifted_v.append(v_shifted)
                all_scores.append(self._compute_shift_scores(q, k_shifted))

            # Stack all scores: [B, T, S, H, total_shifts, scores_per_shift]
            all_scores = torch.stack(all_scores, dim=4)

            # Mix score terms per shift -> [B, T, S, H, total_shifts]
            if self.score_mix is not None:
                scores = self.score_mix(all_scores).squeeze(-1)
            else:
                scores = all_scores.squeeze(-1)

            # Apply mask if provided (sparse extraction from dense mask)
            if mask is not None and mask.dim() == 4:
                flat_indices = torch.arange(L, device=x.device)
                for idx in range(self.total_shifts):
                    if idx < self.num_spatial_shifts:
                        s = self.spatial_shifts[idx]
                        # Spatial shift: within-frame roll
                        frame_pos = flat_indices % S
                        frame_start = flat_indices - frame_pos
                        target_indices = frame_start + (frame_pos + s) % S
                    else:
                        t_shift = self.temporal_shifts[idx - self.num_spatial_shifts]
                        # Temporal shift: across-frame roll
                        frame_idx = flat_indices // S
                        spatial_pos = flat_indices % S
                        target_indices = ((frame_idx + t_shift) % T) * S + spatial_pos

                    shift_mask = mask[:, :, flat_indices, target_indices]  # [B, 1, L]
                    # Reshape to [B, T, S, 1]
                    shift_mask = shift_mask.squeeze(1).view(B, T, S, 1)
                    scores[:, :, :, :, idx] = scores[:, :, :, :, idx].masked_fill(
                        shift_mask.expand_as(scores[:, :, :, :, idx]) == 0, -1e9
                    )

            # Sigmoid-gated weighted average (replaces softmax over shifts)
            # Softmax is winner-take-all, suppressing most of the 16 shifts.
            # Sigmoid gates let ALL shifts contribute proportionally.
            # Gate logits are projected from input x (content-dependent gating).
            gate_logits = self.shift_gate_proj(x.view(B, L, -1))  # [B, L, total_shifts * H]
            gates = torch.sigmoid(gate_logits.view(B, T, S, H, self.total_shifts))  # [0,1] per shift

            # Also modulate by score relevance (score-weighted gates)
            gates = gates * torch.sigmoid(scores)  # combine content gates with score gates

            v_stack = torch.stack(all_shifted_v, dim=4)  # [B, T, S, H, total_shifts, D]
            weighted_v = gates.unsqueeze(-1) * v_stack  # [B, T, S, H, total_shifts, D]
            gate_sum = gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, T, S, H, 1]
            out = weighted_v.sum(dim=4) / gate_sum  # Normalized weighted average

            # Reshape back to [B, L, H*D]
            out = out.reshape(B, L, H * D)

            # Optional spherical normalization for VFM compatibility
            if self.spherical_norm:
                out = F.normalize(out, p=2, dim=-1)

            # Blend with raw V using perturbation mask
            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

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
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        """Forward pass -- routes to video rolling (self-attn) or standard (cross-attn).

        Args:
            x: Query input [B, L, query_dim] where L = T * S for video self-attention
            context: Key/Value input [B, C, context_dim] (None for self-attention)
            mask: Attention mask
            pe: Positional embeddings (RoPE cos/sin tuple)
            k_pe: Key positional embeddings (for cross-modal attention)
            perturbation_mask: Optional [0,1] mask blending attn output with raw V.
                1 keeps full attention, 0 bypasses to value pass-through.
            all_perturbed: If True, skip Q/K computation entirely; pass V through.
            num_frames: Override for number of video frames. If None, uses self.num_frames.
        """
        if context is not None:
            # Cross-attention: different sequence lengths, rolling doesn't apply
            return self._standard_attention(x, context, mask, pe, k_pe)

        return self._video_rolling_attention(
            x, mask, pe, perturbation_mask, all_perturbed, num_frames
        )
