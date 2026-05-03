"""
Triton kernel for Ripple/SmallWorld sparse attention — bidirectional (non-causal).

Adapted from GRA-hybrid/triton_smallworld.py.
Key change from original: IS_CAUSAL=False so diffusion transformers attend
bidirectionally to all shifted neighbors (not just past positions).

Forward:  O_i = Σ_s softmax(Q_i · K_neighbor(i,s) + edge_bias_s) · V_neighbor(i,s)
Backward: dQ, dK, dV via recomputed attention weights (no S×S matrix stored)
"""

import math

import torch
import triton
import triton.language as tl


# ── Forward kernel (bidirectional) ────────────────────────────────────────────

@triton.jit
def _ripple_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    LSE_ptr,
    Shifts_ptr,
    EdgeBias_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lse_bh, stride_lse_m,
    stride_eb_h, stride_eb_s,
    seqlen,
    head_dim: tl.constexpr,
    num_shifts: tl.constexpr,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_start = pid_m * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + pid_bh * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
    q_mask = (q_offs[:, None] < seqlen) & (d_offs[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    m_i  = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for s_idx in tl.static_range(0, num_shifts):
        shift = tl.load(Shifts_ptr + s_idx)
        bias  = tl.load(EdgeBias_ptr + pid_bh * stride_eb_h + s_idx * stride_eb_s)

        neighbor_pos = (q_offs + shift + seqlen) % seqlen
        valid = q_offs < seqlen   # bidirectional: all in-bounds neighbors valid

        k_ptrs = K_ptr + pid_bh * stride_kh + neighbor_pos[:, None] * stride_kn + d_offs[None, :] * stride_kd
        k_mask = valid[:, None] & (d_offs[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        v_ptrs = V_ptr + pid_bh * stride_vh + neighbor_pos[:, None] * stride_vn + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        score = tl.sum(q * k, axis=1) * sm_scale + bias
        score = tl.where(valid, score, -1e9)

        m_i_new = tl.maximum(m_i, score)
        alpha   = tl.exp(m_i - m_i_new)
        p       = tl.exp(score - m_i_new)

        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        acc += p[:, None] * v
        l_i += p
        m_i = m_i_new

    acc = acc / tl.maximum(l_i[:, None], 1e-8)

    out_ptrs = Out_ptr + pid_bh * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
    out_mask = (q_offs[:, None] < seqlen) & (d_offs[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)

    lse = tl.log(tl.maximum(l_i, 1e-8)) + m_i
    lse_ptrs = LSE_ptr + pid_bh * stride_lse_bh + q_offs * stride_lse_m
    tl.store(lse_ptrs, lse, mask=q_offs < seqlen)


# ── Backward kernel (bidirectional) ──────────────────────────────────────────

@triton.jit
def _ripple_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    dOut_ptr, dQ_ptr, dK_ptr, dV_ptr,
    LSE_ptr,
    Shifts_ptr,
    EdgeBias_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lse_bh, stride_lse_m,
    stride_eb_h, stride_eb_s,
    seqlen,
    head_dim: tl.constexpr,
    num_shifts: tl.constexpr,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_start = pid_m * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)

    q_mask = (q_offs[:, None] < seqlen) & (d_offs[None, :] < head_dim)

    q_ptrs  = Q_ptr  + pid_bh * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
    o_ptrs  = Out_ptr + pid_bh * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
    do_ptrs = dOut_ptr + pid_bh * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_od

    q  = tl.load(q_ptrs,  mask=q_mask, other=0.0).to(tl.float32)
    o  = tl.load(o_ptrs,  mask=q_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    lse_ptrs = LSE_ptr + pid_bh * stride_lse_bh + q_offs * stride_lse_m
    lse = tl.load(lse_ptrs, mask=q_offs < seqlen, other=0.0)

    Di   = tl.sum(do * o, axis=1)
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for s_idx in tl.static_range(0, num_shifts):
        shift = tl.load(Shifts_ptr + s_idx)
        bias  = tl.load(EdgeBias_ptr + pid_bh * stride_eb_h + s_idx * stride_eb_s)

        neighbor_pos = (q_offs + shift + seqlen) % seqlen
        valid = q_offs < seqlen  # bidirectional

        k_ptrs = K_ptr + pid_bh * stride_kh + neighbor_pos[:, None] * stride_kn + d_offs[None, :] * stride_kd
        k_mask = valid[:, None] & (d_offs[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        v_ptrs = V_ptr + pid_bh * stride_vh + neighbor_pos[:, None] * stride_vn + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        score  = tl.sum(q * k, axis=1) * sm_scale + bias
        score  = tl.where(valid, score, -1e9)
        alpha  = tl.exp(score - lse)
        alpha  = tl.where(valid, alpha, 0.0)

        dalpha = tl.sum(do * v, axis=1)
        dscore = alpha * (dalpha - Di) * sm_scale

        dq_acc += dscore[:, None] * k

        dk_contrib = dscore[:, None] * q
        dk_ptrs = dK_ptr + pid_bh * stride_kh + neighbor_pos[:, None] * stride_kn + d_offs[None, :] * stride_kd
        tl.atomic_add(dk_ptrs, dk_contrib.to(dK_ptr.dtype.element_ty), mask=k_mask)

        dv_contrib = alpha[:, None] * do
        dv_ptrs = dV_ptr + pid_bh * stride_vh + neighbor_pos[:, None] * stride_vn + d_offs[None, :] * stride_vd
        tl.atomic_add(dv_ptrs, dv_contrib.to(dV_ptr.dtype.element_ty), mask=k_mask)

    dq_ptrs = dQ_ptr + pid_bh * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=q_mask)


# ── Autograd wrapper ──────────────────────────────────────────────────────────

class _RippleAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, shifts, edge_bias, scale):
        B, H, T, D = q.shape
        S = shifts.shape[0]

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        eb = edge_bias.unsqueeze(0).expand(B, H, S).reshape(B * H, S).contiguous().float()

        out = torch.empty_like(q)
        lse = torch.empty(B * H, T, device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid    = (triton.cdiv(T, BLOCK_M), B * H)

        _ripple_fwd_kernel[grid](
            q, k, v, out, lse, shifts, eb,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1),
            eb.stride(0), eb.stride(1),
            T, D, S, scale,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        ctx.save_for_backward(q, k, v, out, lse, shifts, eb)
        ctx.scale = scale
        ctx.shape = (B, H, T, D, S)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, shifts, eb = ctx.saved_tensors
        B, H, T, D, S = ctx.shape
        scale = ctx.scale

        dout = dout.contiguous()
        dq = torch.zeros_like(q)
        # Accumulate dK/dV in fp32: bf16 has no hardware atomic support,
        # so bf16 atomic_add serializes to CAS loops causing 100x slowdown.
        dk = torch.zeros(B, H, T, D, device=q.device, dtype=torch.float32)
        dv = torch.zeros(B, H, T, D, device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid    = (triton.cdiv(T, BLOCK_M), B * H)

        _ripple_bwd_kernel[grid](
            q, k, v, out, dout, dq, dk, dv, lse, shifts, eb,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1),
            eb.stride(0), eb.stride(1),
            T, D, S, scale,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        )
        return dq, dk.to(q.dtype), dv.to(q.dtype), None, None, None


def ripple_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shifts: torch.Tensor,
    edge_bias: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused bidirectional sparse attention with Triton forward+backward.

    Args:
        q, k, v:    [B, H, S, D]  bf16 or fp16
        shifts:     [K]           int32, shift offsets (positive and negative)
        edge_bias:  [H, K]        float32, learnable per-head per-shift bias
        scale:      softmax scale (default 1/sqrt(D))
    Returns:
        out: [B, H, S, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    shifts = shifts.to(torch.int32).contiguous()
    return _RippleAttnFunc.apply(q, k, v, shifts, edge_bias, scale)
