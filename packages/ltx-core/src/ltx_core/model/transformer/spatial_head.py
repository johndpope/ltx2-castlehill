"""Learnable motion feature extractor for Flow Distribution Matching.

Processes frame-to-frame latent diffs to extract motion features.
Used in DMD2/DiagDistill-style training to match temporal distributions.

Improvements over DiagDistill's original SpatialHead:
- Separable convolutions (depthwise + pointwise) for efficiency
- Residual connections inside the head for better gradient flow
- Confidence prediction branch to reweight flow loss per-spatial-location
- Bottleneck output projection for richer high-frequency features

Based on recommendations from video diffusion distillation literature (2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class SpatialHead(nn.Module):
    """Motion feature extractor operating on latent frame diffs.

    Takes [B, T, C, H, W] latent diffs and produces motion features
    of the same shape, plus optional per-pixel confidence weights.

    Args:
        num_channels: Number of input/output channels (latent dim per spatial position).
        num_layers: Number of conv layers (default: 3).
        kernel_size: Spatial kernel size (default: 3).
        hidden_dim: Internal channel dimension (default: 128).
        use_separable: Use depthwise-separable convs (default: True).
        use_residual: Add residual connections inside layers (default: True).
        predict_confidence: Output per-pixel confidence for loss reweighting (default: True).
        norm_num_groups: GroupNorm groups (default: 32, set to 1 for LayerNorm-like).
        norm_eps: Norm epsilon (default: 1e-5).
    """

    def __init__(
        self,
        num_channels: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        hidden_dim: int = 128,
        use_separable: bool = True,
        use_residual: bool = True,
        predict_confidence: bool = True,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"

        self.num_channels = num_channels
        self.use_residual = use_residual
        self.predict_confidence = predict_confidence
        padding = (kernel_size - 1) // 2

        # Learnable residual scale
        if use_residual:
            self.res_weight = nn.Parameter(torch.tensor(0.1))

        self.in_act = nn.SiLU()

        # Build conv layers
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            in_ch = num_channels if i == 0 else hidden_dim
            if use_separable and i > 0:
                # Depthwise-separable: depthwise conv + pointwise conv
                layer = nn.Sequential(
                    # Depthwise
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                              padding=padding, groups=in_ch),
                    # Pointwise
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                    nn.GroupNorm(
                        num_groups=min(norm_num_groups, hidden_dim),
                        num_channels=hidden_dim, eps=norm_eps,
                    ),
                    nn.SiLU(),
                )
            else:
                # Standard conv for first layer (channel expansion)
                layer = nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, padding=padding),
                    nn.GroupNorm(
                        num_groups=min(norm_num_groups, hidden_dim),
                        num_channels=hidden_dim, eps=norm_eps,
                    ),
                    nn.SiLU(),
                )
            self.layers.append(layer)

        # Bottleneck output projection
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, num_channels, kernel_size=1),
        )
        # Zero-init final conv for stable training start
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

        # Confidence prediction branch (sigmoid → [0, 1] per pixel)
        if predict_confidence:
            self.confidence_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            )
            # Init to output ~0.5 confidence everywhere (bias = 0 → sigmoid(0) = 0.5)
            nn.init.zeros_(self.confidence_head[-1].weight)
            nn.init.zeros_(self.confidence_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, T, C, H, W] latent frame diffs (T = num_frames - 1)

        Returns:
            If predict_confidence=False:
                motion_features: [B, T, C, H, W] (same shape as input)
            If predict_confidence=True:
                (motion_features, confidence): features + [B, T, 1, H, W] confidence map
        """
        b, t, c, h, w = x.shape
        x_in = x

        # Flatten batch and time for 2D conv processing
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.in_act(x)

        # Conv layers with optional residual
        for layer in self.layers:
            if self.use_residual and x.shape[1] == layer[0].out_channels if hasattr(layer[0], 'out_channels') else False:
                res = x
                x = layer(x)
                x = x + res * self.res_weight
            else:
                x = layer(x)

        # Extract features before output projection (for confidence branch)
        features_hidden = x  # [B*T, hidden_dim, H, W]

        # Motion features
        x = self.conv_out(features_hidden)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        # Residual connection to input
        x = x + x_in

        if self.predict_confidence:
            conf = self.confidence_head(features_hidden)
            conf = torch.sigmoid(conf)
            conf = rearrange(conf, "(b t) c h w -> b t c h w", b=b, t=t)
            return x, conf

        return x


class IdentitySpatialHead(nn.Module):
    """Pass-through spatial head (no learnable motion extraction)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
