"""Lightweight latent-space discriminator for DMD2/FlashMotion-style adversarial training.

Operates on noisy patchified video latent tokens. Classifies real (teacher ODE output)
vs fake (student 1-step output) at a random critic timestep.

Architecture: 4-layer transformer with register tokens -> scalar logit.
~15M params at default settings. Much cheaper than the 19B DiT.

Used in VFM v3a for distribution-level training (DMD2 approach):
- Discriminator sees noisy latents at random t_critic
- Generator (student DiT + adapter) receives GAN gradient
- No separate "fake score network" needed (DMD2 insight)

References:
- DMD2 (Yin et al., 2024): Improved Distribution Matching Distillation
- FlashMotion (CVPR 2026): Adversarial post-training for few-step video
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDiscriminator(nn.Module):
    """Discriminator operating on noisy patchified video latent tokens.

    Takes noisy latent tokens + critic timestep -> scalar logit.
    Uses learnable register tokens that aggregate global information
    before being projected to a single real/fake logit.

    Args:
        latent_dim: Dimension of patchified latent tokens (default: 128).
        hidden_dim: Internal transformer dimension (default: 512).
        num_heads: Number of attention heads (default: 8).
        num_layers: Number of transformer blocks (default: 4).
        num_registers: Number of learnable register tokens (default: 4).
        text_dim: Dimension of text conditioning (default: 4096).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_registers: int = 4,
        text_dim: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_registers = num_registers

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Timestep embedding (sinusoidal -> MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Text projection (optional conditioning)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Learnable register tokens
        self.registers = nn.Parameter(torch.randn(1, num_registers, hidden_dim) * 0.02)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # Output head: registers -> flatten -> logit
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * num_registers, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights. Output head zero-init for stable start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Zero-init final projection for stable training
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1).to(t.dtype)

    def forward(
        self,
        x: torch.Tensor,
        t_critic: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, seq_len, latent_dim] noisy patchified latent tokens
            t_critic: [B] critic timestep in [0, 1]
            text_embeddings: [B, text_seq, text_dim] optional text conditioning

        Returns:
            logit: [B, 1] real/fake logit (positive = real)
        """
        B = x.shape[0]

        # Project latent tokens
        h = self.input_proj(x)  # [B, seq, hidden]

        # Add timestep conditioning
        t_emb = self._sinusoidal_embedding(t_critic, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [B, hidden]
        h = h + t_emb.unsqueeze(1)  # broadcast to all tokens

        # Optionally add text conditioning (mean-pool -> add)
        if text_embeddings is not None:
            text_h = self.text_proj(text_embeddings)  # [B, T, hidden]
            text_pool = text_h.mean(dim=1, keepdim=True)  # [B, 1, hidden]
            h = h + text_pool

        # Prepend register tokens
        regs = self.registers.expand(B, -1, -1)  # [B, num_reg, hidden]
        h = torch.cat([regs, h], dim=1)  # [B, num_reg + seq, hidden]

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        # Extract register tokens and classify
        reg_out = h[:, :self.num_registers]  # [B, num_reg, hidden]
        reg_out = self.output_norm(reg_out)
        reg_flat = reg_out.reshape(B, -1)  # [B, num_reg * hidden]
        logit = self.output_head(reg_flat)  # [B, 1]

        return logit
