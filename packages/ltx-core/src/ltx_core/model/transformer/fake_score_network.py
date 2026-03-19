"""
Fake Score Network for Distribution Matching Distillation (DMD).

A lightweight (~50M param) score estimation network that approximates
nabla log p(x) of the data distribution at a given noise level sigma.

In DMD-VFM training, this network plays the role of the "fake" score:
it is trained on samples from the one-step generator (the VFM adapter)
to estimate the score of the generated distribution. The real score
comes from the frozen pretrained DiT. The KL divergence between the
two distributions is minimized by matching these scores, which provides
a gradient signal to the generator without needing adversarial training.

Reference: Yin et al., "One-step Diffusion with Distribution Matching
Distillation" (arXiv:2311.18828).

Architecture:
    4-layer transformer with self-attention, cross-attention to text,
    SwiGLU FFN, and AdaLN-Zero conditioning on the noise level sigma.
    Input/output dimension is 128 (patchified video latent channels).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_core.model.transformer.timestep_embedding import (
    get_timestep_embedding,
    TimestepEmbedding,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm modulation (scale + shift + gate) from a conditioning vector.

    Produces 6 modulation parameters per block: scale1, shift1, gate1 (self-attn),
    scale2, shift2, gate2 (ffn). Cross-attention gets an additional 3 params.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, num_params: int = 9):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, num_params * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> list[torch.Tensor]:
        # c: [B, cond_dim] -> [B, num_params * hidden_dim]
        return self.linear(self.silu(c)).chunk(9, dim=-1)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Uses gated linear units with SiLU activation. The intermediate
    dimension is 8/3 * hidden_dim (rounded to nearest multiple of 64)
    to keep parameter count comparable to a standard 4x FFN.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # 8/3 ratio keeps param count ~= 4x FFN with standard GELU
        intermediate = int(hidden_dim * 8 / 3)
        # Round to nearest multiple of 64 for hardware efficiency
        intermediate = ((intermediate + 63) // 64) * 64
        self.w1 = nn.Linear(hidden_dim, intermediate, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate, bias=False)
        self.w3 = nn.Linear(intermediate, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class FakeScoreBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention, and FFN.

    Uses AdaLN-Zero conditioning: each sub-layer output is modulated by
    learned scale/shift/gate derived from the timestep embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        text_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Self-attention
        self.norm1 = RMSNorm(hidden_dim)
        self.self_attn_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.self_attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cross-attention to text
        self.norm2 = RMSNorm(hidden_dim)
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attn_kv = nn.Linear(text_dim, 2 * hidden_dim, bias=False)
        self.cross_attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # FFN
        self.norm3 = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, dropout=dropout)

        # AdaLN modulation: 9 params (3 sub-layers x 3 params each: scale, shift, gate)
        self.adaln = AdaLNModulation(hidden_dim, hidden_dim, num_params=9)

        self.attn_dropout = nn.Dropout(dropout)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scaled dot-product attention with optional mask."""
        B, H, S, D = q.shape
        scale = 1.0 / math.sqrt(D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim] token features
            c: [B, hidden_dim] timestep conditioning
            text_kv: [B, text_seq, text_dim] text embeddings (already projected)
            text_mask: [B, text_seq] boolean mask (True = attend)
        """
        B, S, D = x.shape

        # Get AdaLN modulation params
        (
            scale1, shift1, gate1,  # self-attention
            scale2, shift2, gate2,  # cross-attention
            scale3, shift3, gate3,  # ffn
        ) = self.adaln(c)

        # --- Self-attention ---
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        qkv = self.self_attn_qkv(h)
        q, k, v = qkv.reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        h = self._attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.self_attn_out(h)
        x = x + gate1.unsqueeze(1) * h

        # --- Cross-attention ---
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        q = self.cross_attn_q(h).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        text_seq = text_kv.shape[1]
        kv = self.cross_attn_kv(text_kv)
        k, v = kv.reshape(B, text_seq, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # Reshape text_mask for attention: [B, text_seq] -> [B, 1, 1, text_seq]
        attn_mask = None
        if text_mask is not None:
            attn_mask = text_mask[:, None, None, :]  # [B, 1, 1, text_seq]

        h = self._attention(q, k, v, mask=attn_mask)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.cross_attn_out(h)
        x = x + gate2.unsqueeze(1) * h

        # --- FFN ---
        h = self.norm3(x)
        h = h * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        h = self.ffn(h)
        x = x + gate3.unsqueeze(1) * h

        return x


class FakeScoreNetwork(nn.Module):
    """Lightweight score estimation network for DMD training.

    Estimates nabla log p(x) for the generated (fake) distribution.
    During DMD training:
      - The frozen pretrained DiT provides the "real" score.
      - This network is trained to estimate the score of samples
        produced by the one-step VFM generator.
      - The generator receives gradients from the score difference
        (real - fake), which minimizes the KL divergence between
        the generated and data distributions.

    This is much cheaper than running the full 19B DiT for the fake
    score, making DMD practical for video diffusion.

    Args:
        latent_dim: Dimension of patchified video latent tokens (default: 128).
        hidden_dim: Internal transformer dimension (default: 576).
        num_heads: Number of attention heads (default: 8).
        num_layers: Number of transformer blocks (default: 4).
        text_dim: Dimension of text conditioning embeddings (default: 4096).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 576,
        num_heads: int = 8,
        num_layers: int = 4,
        text_dim: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection: latent tokens -> hidden dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Timestep embedding: sigma -> sinusoidal -> MLP -> hidden_dim
        # Uses the same pattern as the main DiT
        self.time_embed = TimestepEmbedding(
            in_channels=256,  # sinusoidal embedding dim
            time_embed_dim=hidden_dim,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FakeScoreBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                text_dim=text_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final norm + output projection
        self.final_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Zero-initialize output projection for stable training start
        # (network starts as identity-like, outputting near-zero scores)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate the score of the data distribution at noise level sigma.

        Args:
            x: [B, seq_len, 128] patchified video latent tokens.
            sigma: [B] noise level for each sample in the batch.
            text_embeddings: [B, text_seq, text_dim] text conditioning.
            text_mask: [B, text_seq] boolean mask (True = valid token).

        Returns:
            score: [B, seq_len, 128] estimated score vector nabla log p(x).
        """
        # Project input tokens
        h = self.input_proj(x)

        # Compute timestep conditioning from sigma
        # Sinusoidal embedding (256-dim) -> MLP -> hidden_dim
        t_emb = get_timestep_embedding(sigma, 256)
        c = self.time_embed(t_emb.to(h.dtype))  # [B, hidden_dim]

        # Run through transformer blocks
        for block in self.blocks:
            h = block(h, c, text_embeddings, text_mask)

        # Final norm + output projection
        h = self.final_norm(h)
        score = self.output_proj(h)

        return score

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def from_config(
        latent_dim: int = 128,
        hidden_dim: int = 576,
        num_heads: int = 8,
        num_layers: int = 4,
        text_dim: int = 4096,
        dropout: float = 0.0,
    ) -> "FakeScoreNetwork":
        """Create a FakeScoreNetwork from configuration values."""
        net = FakeScoreNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            text_dim=text_dim,
            dropout=dropout,
        )
        return net
