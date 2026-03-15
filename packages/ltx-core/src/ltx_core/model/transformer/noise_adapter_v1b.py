"""VFM Noise Adapter v1b — Temporally-aware conditional noise generation.

Improvements over v1a (noise_adapter.py):
  1. Self-attention across tokens → inter-token noise coordination
  2. Cross-attention to FULL text sequence (not pooled) → per-token text grounding
  3. Sinusoidal positional encoding → each token knows its (t, h, w) location

The v1a MLP processes every token identically (same pooled text input → same μ,σ).
v1b produces DIFFERENT μ,σ per token because each token knows where it sits in
the spatiotemporal grid and can attend to relevant parts of the text.

Architecture:
    Input: text_embeddings [B, text_seq, D_text] + positions [B, 3, video_seq, 2]
    → Sinusoidal pos encoding → [B, video_seq, D_pos]
    → Input projection: Linear(D_pos + task_embed_dim → hidden_dim)
    → N blocks of:
        - Self-attention (video tokens attend to each other)
        - Cross-attention (video tokens attend to text tokens)
        - FFN
    → μ head: Linear(hidden_dim → latent_dim)
    → log_σ head: Linear(hidden_dim → latent_dim)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionEncoding(nn.Module):
    """Encode (t, h, w) coordinates into a fixed-dim vector via sinusoids.

    Takes the spatiotemporal bounds [B, 3, seq, 2] and produces [B, seq, pos_dim].
    Uses the midpoint of each (start, end) pair as the coordinate value.
    """

    def __init__(self, pos_dim: int = 256, num_axes: int = 3):
        super().__init__()
        self.num_axes = num_axes
        # Each axis gets pos_dim // num_axes dims, rounded to even
        self.dim_per_axis = (pos_dim // num_axes) // 2 * 2  # must be even for sin/cos pairs
        # Actual output dim (may differ slightly from requested pos_dim)
        self.pos_dim = self.dim_per_axis * num_axes

        # Precompute frequency bands (log-spaced like standard transformer PE)
        freqs = torch.exp(
            torch.arange(0, self.dim_per_axis, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_per_axis)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, positions: Tensor) -> Tensor:
        """Encode positions to sinusoidal embeddings.

        Args:
            positions: [B, 3, seq_len, 2] — (time, height, width) × (start, end)

        Returns:
            [B, seq_len, pos_dim] sinusoidal encoding
        """
        # Use midpoint of each (start, end) bound
        coords = positions.mean(dim=-1)  # [B, 3, seq_len] — midpoint per axis

        encodings = []
        for axis in range(self.num_axes):
            c = coords[:, axis, :]  # [B, seq_len]
            c = c.unsqueeze(-1)  # [B, seq_len, 1]
            freqs = self.freqs.to(c.device, c.dtype)  # [dim_per_axis // 2]

            angles = c * freqs  # [B, seq_len, dim_per_axis // 2]
            enc = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, seq_len, dim_per_axis]
            encodings.append(enc)

        return torch.cat(encodings, dim=-1)  # [B, seq_len, pos_dim]


class AdapterBlock(nn.Module):
    """Single transformer block: self-attn → cross-attn → FFN.

    Pre-norm (LayerNorm before each sub-layer), matching modern transformer conventions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to text
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        text_kv: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: [B, video_seq, hidden_dim] video token features
            text_kv: [B, text_seq, hidden_dim] projected text features
            text_mask: [B, text_seq] bool mask (True = valid, False = padding)

        Returns:
            [B, video_seq, hidden_dim]
        """
        # Self-attention: video tokens attend to each other
        residual = x
        x_norm = self.self_attn_norm(x)
        x = residual + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention: video tokens attend to text tokens
        # MHA expects key_padding_mask where True = IGNORE, so invert
        key_padding_mask = ~text_mask if text_mask is not None else None
        residual = x
        x_norm = self.cross_attn_norm(x)
        text_kv_norm = self.text_norm(text_kv)
        x = residual + self.cross_attn(
            x_norm, text_kv_norm, text_kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        # FFN
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x


class NoiseAdapterV1b(nn.Module):
    """Enhanced noise adapter with spatiotemporal awareness and text cross-attention.

    Key differences from v1a NoiseAdapterMLP:
      - Sinusoidal positional encoding: each token knows its (t, h, w) location
      - Cross-attention to full text: no mean-pooling, per-token text grounding
      - Self-attention: tokens coordinate their noise distributions

    This means different spatial locations and different frames get DIFFERENT
    μ and σ values, enabling temporally structured noise.
    """

    def __init__(
        self,
        text_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_task_classes: int = 5,
        task_embed_dim: int = 128,
        pos_dim: int = 256,
        init_sigma: float = 0.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            text_dim: Dimension of text embeddings (3840 for LTX-2)
            latent_dim: Output noise dimension (128 = video latent channels)
            hidden_dim: Internal transformer hidden dim
            num_heads: Attention heads per block
            num_layers: Number of (self-attn + cross-attn + FFN) blocks
            num_task_classes: Number of inverse problem classes
            task_embed_dim: Task class embedding dimension
            pos_dim: Positional encoding dimension
            init_sigma: Initial log_σ bias (0.0 → σ starts at 1.0)
            dropout: Attention dropout
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.init_sigma = init_sigma
        self.hidden_dim = hidden_dim

        # Positional encoding
        self.pos_encoder = SinusoidalPositionEncoding(pos_dim=pos_dim)
        actual_pos_dim = self.pos_encoder.pos_dim  # may differ from requested

        # Task embedding
        self.task_embedding = nn.Embedding(num_task_classes, task_embed_dim)

        # Project (pos_encoding + task_embedding) → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(actual_pos_dim + task_embed_dim),
            nn.Linear(actual_pos_dim + task_embed_dim, hidden_dim),
        )

        # Project text embeddings → hidden_dim (for cross-attention KV)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdapterBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

        # Optional per-token timestep sigma head (v1h+)
        # Outputs a single scalar σ_timestep ∈ [sigma_min, sigma_max] per token,
        # replacing the external SigmaHead. The adapter already has text cross-attention
        # + position encoding + self-attention, so it knows what each token needs.
        self.sigma_timestep_head: nn.Linear | None = None

        # Initialize near identity (start at N(0, I) prior)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, init_sigma)

    def enable_sigma_timestep_head(self, sigma_min: float = 0.3, sigma_max: float = 1.0) -> None:
        """Add per-token timestep sigma output head (v1h).

        Once enabled, forward() returns (mu, log_sigma, per_token_sigma) instead of
        (mu, log_sigma). The sigma head shares the same transformer features —
        no separate SigmaHead MLP needed.
        """
        # Get device from existing parameters
        device = next(self.parameters()).device
        self.sigma_timestep_head = nn.Linear(self.hidden_dim, 1).to(device)
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        # Initialize to output mid-range sigma (~0.5 after sigmoid)
        nn.init.zeros_(self.sigma_timestep_head.weight)
        nn.init.zeros_(self.sigma_timestep_head.bias)

    def forward(
        self,
        text_embeddings: Tensor,
        text_mask: Tensor,
        positions: Tensor,
        task_class: Tensor,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Compute per-token noise distribution parameters.

        Args:
            text_embeddings: [B, text_seq, text_dim] FULL text embeddings (not pooled)
            text_mask: [B, text_seq] attention mask (True = valid token)
            positions: [B, 3, video_seq, 2] spatiotemporal coordinates
            task_class: [B] integer task class indices

        Returns:
            If sigma_timestep_head is None (v1a-v1g behavior):
                mu: [B, video_seq, latent_dim] per-token mean
                log_sigma: [B, video_seq, latent_dim] per-token log std
            If sigma_timestep_head is enabled (v1h+):
                mu: [B, video_seq, latent_dim] per-token mean
                log_sigma: [B, video_seq, latent_dim] per-token log std
                per_token_sigma: [B, video_seq] timestep sigma in [sigma_min, sigma_max]
        """
        B, video_seq = positions.shape[0], positions.shape[2]

        # 1. Encode positions: [B, 3, video_seq, 2] → [B, video_seq, pos_dim]
        pos_enc = self.pos_encoder(positions.float())

        # 2. Task embedding: [B] → [B, video_seq, task_embed_dim]
        task_emb = self.task_embedding(task_class)
        task_emb = task_emb.unsqueeze(1).expand(-1, video_seq, -1)

        # 3. Input projection: concat pos + task → hidden_dim
        x = torch.cat([pos_enc, task_emb], dim=-1)  # [B, video_seq, pos_dim + task_embed_dim]
        x = self.input_proj(x)  # [B, video_seq, hidden_dim]

        # 4. Project text for cross-attention KV
        text_kv = self.text_proj(text_embeddings.float())  # [B, text_seq, hidden_dim]

        # 5. Transformer blocks: self-attn + cross-attn + FFN
        for block in self.blocks:
            x = block(x, text_kv, text_mask)

        # 6. Output heads
        x = self.output_norm(x)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)

        # Clamp log_sigma to prevent collapse/explosion
        log_sigma = log_sigma.clamp(min=-1.0, max=2.0)

        # 7. Optional per-token timestep sigma (v1h+)
        if self.sigma_timestep_head is not None:
            raw_sigma = self.sigma_timestep_head(x).squeeze(-1)  # [B, video_seq]
            per_token_sigma = self._sigma_min + (self._sigma_max - self._sigma_min) * torch.sigmoid(raw_sigma)
            return mu, log_sigma, per_token_sigma

        return mu, log_sigma

    def sample(
        self,
        text_embeddings: Tensor,
        text_mask: Tensor,
        positions: Tensor,
        task_class: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample structured noise z ~ qφ(z|y).

        Returns:
            z: [B, video_seq, latent_dim]
        """
        mu, log_sigma = self.forward(text_embeddings, text_mask, positions, task_class)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps * temperature

    def sample_with_mu(
        self,
        text_embeddings: Tensor,
        text_mask: Tensor,
        positions: Tensor,
        task_class: Tensor,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Sample structured noise z ~ qφ(z|y) and return adapter mu.

        Returns:
            (z, mu) where z: [B, video_seq, latent_dim], mu: [B, video_seq, latent_dim]
        """
        mu, log_sigma = self.forward(text_embeddings, text_mask, positions, task_class)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps * temperature
        return z, mu

    def kl_divergence(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        """KL(qφ(z|y) || N(0,I))."""
        kl = 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1)
        return kl.mean()


# Re-export task classes for consistency with v1a
TASK_CLASSES = {
    "i2v": 0,
    "inpaint": 1,
    "sr": 2,
    "denoise": 3,
    "t2v": 4,
}


def create_noise_adapter_v1b(
    text_dim: int,
    latent_dim: int = 128,
    **kwargs,
) -> NoiseAdapterV1b:
    """Factory function for v1b noise adapter.

    Args:
        text_dim: Text embedding dimension (3840 for LTX-2)
        latent_dim: Video latent channels (128)
        **kwargs: Passed to NoiseAdapterV1b constructor
    """
    return NoiseAdapterV1b(text_dim=text_dim, latent_dim=latent_dim, **kwargs)
