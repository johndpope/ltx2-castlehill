"""Multi-Head VQ-VAE for MotionBricks-style structured isometric video tokenization.

Implements MotionBricks Section 4: Structured Multi-Head Tokenizer with K separate
codebooks along the feature dimension, plus root-pose disentanglement.

The standard LTX-VAE produces a 128-channel latent. This module splits it into
K=4 heads (32 channels each) and quantizes each with a separate codebook.

Head structure for isometric:
  Head 0 (0-31):  Structure  — parallel lines, grid alignment, architectural edges
  Head 1 (32-63): Depth      — occlusion ordering, isometric z-ordering
  Head 2 (64-95): Motion     — temporal dynamics, object movement
  Head 3 (96-127): Texture   — appearance, surface detail

Root-pose disentanglement: The encoder can optionally output 4 additional root
channels (x, y, z, isometric_angle) that bypass VQ and go directly to the decoder
via skip connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiHeadVQConfig:
    """Configuration for the multi-head VQ-VAE tokenizer."""
    num_heads: int = 4
    head_dims: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    codebook_sizes: list[int] = field(default_factory=lambda: [4096, 4096, 4096, 4096])
    codebook_dim: int = 32
    commitment_cost: float = 0.25
    use_ema: bool = True                # EMA codebook updates (more stable)
    ema_decay: float = 0.99
    epsilon: float = 1e-5
    use_root_disentanglement: bool = True
    root_dim: int = 4                   # (x, y, z, isometric_angle)

    def __post_init__(self):
        assert len(self.head_dims) == self.num_heads, \
            f"head_dims ({len(self.head_dims)}) must match num_heads ({self.num_heads})"
        assert len(self.codebook_sizes) == self.num_heads, \
            f"codebook_sizes ({len(self.codebook_sizes)}) must match num_heads ({self.num_heads})"
        assert sum(self.head_dims) == self.codebook_dim * self.num_heads, \
            f"head_dims sum ({sum(self.head_dims)}) must equal codebook_dim * num_heads ({self.codebook_dim * self.num_heads})"
        assert self.num_heads > 0, "num_heads must be > 0"
        assert all(d > 0 for d in self.head_dims), "all head_dims must be > 0"
        assert all(s > 0 for s in self.codebook_sizes), "all codebook_sizes must be > 0"


# ---------------------------------------------------------------------------
# Single-Head Vector Quantizer
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Single codebook vector quantizer with EMA update."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        w = self.embedding.weight.data
        w.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        w = F.normalize(w, p=2, dim=1)

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w", w.clone())
            self.register_buffer("_initted", torch.tensor(False))

    def forward(self, z: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        original_shape = z.shape
        B, D = z.shape[0], z.shape[1]
        assert D == self.embedding_dim

        flat_z = z.permute(0, *range(2, z.ndim), 1).reshape(-1, D)

        z_sq = (flat_z ** 2).sum(dim=1, keepdim=True)
        emb_sq = (self.embedding.weight ** 2).sum(dim=1).unsqueeze(0)
        distances = z_sq + emb_sq - 2.0 * torch.mm(flat_z, self.embedding.weight.t())

        encoding_indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(encoding_indices)

        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
            usage = (encodings.sum(dim=0) > 0).float().mean()
            min_distances = distances.min(dim=1).values
            avg_distance = min_distances.mean()

        if self.use_ema and self.training:
            self._ema_update(flat_z.detach(), encoding_indices)

        commitment_loss = self.commitment_cost * F.mse_loss(z_q_flat.detach(), flat_z)
        z_q = z + (z_q_flat.view_as(z) - z).detach()
        z_q = z_q.contiguous()

        metrics = {
            "perplexity": perplexity.item(),
            "codebook_usage": usage.item(),
            "avg_distance": avg_distance.item(),
            "commitment_loss": commitment_loss.item(),
        }
        return z_q, metrics

    @torch.no_grad()
    def _ema_update(self, flat_z: Tensor, encoding_indices: Tensor) -> None:
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        cluster_size = encodings.sum(dim=0)
        encoded_sum = encodings.t() @ flat_z

        if not self._initted:
            self.ema_cluster_size.data = cluster_size
            self.ema_w.data = encoded_sum
            self._initted.data = torch.tensor(True)
        else:
            self.ema_cluster_size.data = self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
            self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * encoded_sum

        n = self.ema_cluster_size.sum()
        smoothed_size = self.ema_cluster_size + self.epsilon
        normalized_weights = (smoothed_size / smoothed_size.sum()) * n
        ema_weights = self.ema_w / normalized_weights.unsqueeze(1)
        ema_weights = F.normalize(ema_weights, p=2, dim=1)
        self.embedding.weight.data = ema_weights


# ---------------------------------------------------------------------------
# Multi-Head Vector Quantizer
# ---------------------------------------------------------------------------

class MultiHeadVectorQuantizer(nn.Module):
    """Wraps K VectorQuantizer instances, one per head."""

    def __init__(self, config: MultiHeadVQConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dims = config.head_dims

        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                num_embeddings=config.codebook_sizes[i],
                embedding_dim=config.head_dims[i],
                commitment_cost=config.commitment_cost,
                use_ema=config.use_ema,
                decay=config.ema_decay,
                epsilon=config.epsilon,
            )
            for i in range(config.num_heads)
        ])

        if config.use_root_disentanglement:
            self.root_predictor = nn.Sequential(
                nn.Linear(config.codebook_dim * config.num_heads, 64),
                nn.GELU(),
                nn.Linear(64, config.root_dim),
            )
        else:
            self.root_predictor = None

    def forward(self, z: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        total_dim = z.shape[1]
        assert total_dim == sum(self.head_dims)

        z_heads = z.split(self.head_dims, dim=1)

        z_q_list = []
        all_metrics = {}
        for i, (z_head, quantizer) in enumerate(zip(z_heads, self.quantizers)):
            z_q_head, head_metrics = quantizer(z_head)
            z_q_list.append(z_q_head)
            for k, v in head_metrics.items():
                all_metrics[f"head_{i}/{k}"] = v

        z_q = torch.cat(z_q_list, dim=1)

        with torch.no_grad():
            perplexities = [all_metrics[f"head_{i}/perplexity"] for i in range(self.num_heads)]
            usages = [all_metrics[f"head_{i}/codebook_usage"] for i in range(self.num_heads)]
            commitment_losses = [all_metrics[f"head_{i}/commitment_loss"] for i in range(self.num_heads)]

        all_metrics["perplexity"] = float(np.mean(perplexities))
        all_metrics["codebook_usage"] = float(np.mean(usages))
        all_metrics["commitment_loss"] = float(np.mean(commitment_losses))
        all_metrics["num_heads"] = self.num_heads

        if self.root_predictor is not None and self.training:
            if z_q.ndim == 5:
                pooled = z_q.mean(dim=[2, 3, 4])
            else:
                pooled = z_q.mean(dim=[2, 3])
            root_pred = self.root_predictor(pooled)
            all_metrics["root_pred"] = root_pred
            all_metrics["root_norm"] = root_pred.norm(dim=-1).mean().item()

        return z_q, all_metrics

    @torch.no_grad()
    def encode_to_indices(self, z: Tensor) -> list[Tensor]:
        z_heads = z.split(self.head_dims, dim=1)
        all_indices = []
        flat_z = z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)
        z_head_flats = flat_z.split(self.head_dims, dim=-1)

        for z_head_flat, quantizer in zip(z_head_flats, self.quantizers):
            B, N, D = z_head_flat.shape
            flat = z_head_flat.reshape(-1, D)
            z_sq = (flat ** 2).sum(dim=1, keepdim=True)
            emb_sq = (quantizer.embedding.weight ** 2).sum(dim=1).unsqueeze(0)
            distances = z_sq + emb_sq - 2.0 * torch.mm(flat, quantizer.embedding.weight.t())
            indices = torch.argmin(distances, dim=1)
            indices = indices.reshape(B, N)
            all_indices.append(indices)
        return all_indices

    @torch.no_grad()
    def decode_from_indices(self, indices: list[Tensor]) -> Tensor:
        z_q_parts = []
        for idx, quantizer in zip(indices, self.quantizers):
            z_q = quantizer.embedding(idx)
            z_q_parts.append(z_q)
        z_q = torch.cat(z_q_parts, dim=-1)
        return z_q.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Full Multi-Head VQ-VAE Wrapper
# ---------------------------------------------------------------------------

class MultiHeadVQVideoVAE(nn.Module):
    """Full VQ-VAE wrapping existing encoder/decoder with multi-head quantization."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: MultiHeadVQConfig,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.vq = MultiHeadVectorQuantizer(config)
        self.use_root_disentanglement = config.use_root_disentanglement
        self.root_adapter = None
        if self.use_root_disentanglement:
            self._check_and_prepare_decoder()

    def _check_and_prepare_decoder(self) -> None:
        decoder_conv_in = getattr(self.decoder, "conv_in", None)
        if decoder_conv_in is None:
            logger.warning("Decoder has no conv_in — root injection disabled")
            self.root_adapter = None
            return
        in_channels = decoder_conv_in.in_channels
        expected_in = in_channels + self.config.root_dim
        self.root_adapter = nn.Linear(expected_in, in_channels, bias=False)
        with torch.no_grad():
            self.root_adapter.weight.data.zero_()
            self.root_adapter.weight.data[:, :in_channels] = torch.eye(in_channels)

    def encode(self, video: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        z = self.encoder(video)
        z_q, metrics = self.vq(z)
        return z_q, metrics

    def decode(self, z_q: Tensor, root_traj: Optional[Tensor] = None) -> Tensor:
        if self.use_root_disentanglement and root_traj is not None and self.root_adapter is not None:
            B, C, F_lat, H_lat, W_lat = z_q.shape
            root_expanded = root_traj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            root_expanded = root_expanded.expand(B, -1, F_lat, H_lat, W_lat)
            z_with_root = torch.cat([z_q, root_expanded], dim=1)
            z_flat = z_with_root.flatten(2).permute(0, 2, 1)
            z_proj = self.root_adapter(z_flat)
            z_proj = z_proj.permute(0, 2, 1).view_as(z_q)
            video = self.decoder(z_proj)
        else:
            video = self.decoder(z_q)
        return video

    def forward(self, video: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        z_q, metrics = self.encode(video)
        root_pred = metrics.get("root_pred")
        recon = self.decode(z_q, root_traj=root_pred)
        recon_loss = F.mse_loss(recon, video)
        metrics["recon_loss"] = recon_loss.item()
        return recon, metrics

    @torch.no_grad()
    def encode_to_indices(self, video: Tensor) -> list[Tensor]:
        z = self.encoder(video)
        return self.vq.encode_to_indices(z)

    def get_token_count(self) -> int:
        return sum(self.config.codebook_sizes)
