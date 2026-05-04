"""Unified Keyframe Interface for MotionBricks-style conditioning.

Implements the constraint hierarchy from MotionBricks Section 4.3 and Section 6:
  - T1: Local root/view (camera trajectory: x, y, z, isometric_angle)
  - T2: Global root/view (scene layout, background)
  - T3: Pose/object (joint positions, object interactions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class KeyframeConstraint:
    """A single keyframe constraint at a specific hierarchy level."""
    frame_idx: int
    level: Literal["T1", "T2", "T3"]
    data: Tensor
    hardness: float = 1.0  # tau: 0=hard constraint, >0=soft guidance
    object_id: Optional[int] = None


class KeyframeConditioningEncoder(nn.Module):
    """Encodes keyframe constraints into cross-attention tokens."""

    def __init__(
        self,
        cross_attention_dim: int = 3840,
        max_keyframes: int = 10,
        hidden_dim: int = 1024,
        t1_dim: int = 4,
        t2_dim: int = 128,
        t3_pose_dim: int = 10,
        t3_action_dim: int = 16,
        num_transformer_layers: int = 2,
        num_transformer_heads: int = 4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.max_keyframes = max_keyframes
        self.hidden_dim = hidden_dim

        self.t1_proj = nn.Sequential(
            nn.Linear(t1_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.t2_proj = nn.Sequential(
            nn.Linear(t2_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.t3_pose_proj = nn.Sequential(
            nn.Linear(t3_pose_dim, hidden_dim // 2), nn.GELU(),
        )
        self.t3_action_embed = nn.Embedding(32, t3_action_dim) if t3_action_dim > 0 else None
        self.t3_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2 + t3_action_dim, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_transformer_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, cross_attention_dim), nn.GELU(), nn.LayerNorm(cross_attention_dim),
        )
        self.mask_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_keyframes, hidden_dim) * 0.02)
        self.level_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self._level_to_id = {"T1": 0, "T2": 1, "T3": 2}

    def forward(
        self,
        keyframes: list[KeyframeConstraint] | list[list[KeyframeConstraint]],
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if not keyframes:
            B, device, dtype = 1, self.mask_embed.device, self.mask_embed.dtype
            embeddings = self.mask_embed.expand(B, self.max_keyframes, -1).clone() + self.pos_embed
            kf_tokens = self.output_proj(self.transformer(embeddings))
            return kf_tokens

        if isinstance(keyframes[0], KeyframeConstraint):
            keyframes = [keyframes]

        B = len(keyframes)
        device = self.mask_embed.device
        dtype = self.mask_embed.dtype

        embeddings = self.mask_embed.expand(B, self.max_keyframes, -1).clone()
        embeddings = embeddings + self.pos_embed

        for b in range(B):
            kfs = keyframes[b]
            kfs = sorted(kfs, key=lambda kf: kf.frame_idx)
            if len(kfs) > self.max_keyframes:
                indices = torch.linspace(0, len(kfs) - 1, self.max_keyframes).long()
                kfs = [kfs[i] for i in indices]

            for i, kf in enumerate(kfs):
                if i >= self.max_keyframes:
                    break
                data = kf.data.to(device=device, dtype=dtype).unsqueeze(0)
                if kf.level == "T1":
                    feat = self.t1_proj(data)
                elif kf.level == "T2":
                    feat = self.t2_proj(data)
                elif kf.level == "T3":
                    pose_feat = self.t3_pose_proj(data[..., :10])
                    if self.t3_action_embed is not None:
                        if data.shape[-1] > 10:
                            action_idx = data[..., -1].long().clamp(0, 31)
                        else:
                            action_idx = torch.zeros(1, dtype=torch.long, device=device)
                        action_feat = self.t3_action_embed(action_idx)
                        feat = self.t3_proj(torch.cat([pose_feat, action_feat], dim=-1))
                    else:
                        feat = self.t3_proj(pose_feat)
                else:
                    raise ValueError(f"Unknown level: {kf.level}")

                level_id = self._level_to_id[kf.level]
                feat = feat + self.level_embed[level_id].unsqueeze(0)
                hardness = torch.tensor([kf.hardness], device=device, dtype=dtype)
                signal_strength = torch.exp(-hardness)
                feat = feat * signal_strength.unsqueeze(-1)
                embeddings[b, i:i+1] = feat

        embeddings = self.transformer(embeddings)
        kf_tokens = self.output_proj(embeddings)
        return kf_tokens

    def get_keyframe_mask(self, kf_tokens: Tensor, keyframe_list: list) -> Tensor:
        if not keyframe_list:
            return torch.zeros(1, self.max_keyframes, device=kf_tokens.device, dtype=torch.bool)
        if isinstance(keyframe_list[0], KeyframeConstraint):
            keyframe_list = [keyframe_list]
        B = len(keyframe_list)
        device = kf_tokens.device
        mask = torch.zeros(B, self.max_keyframes, device=device, dtype=torch.bool)
        for b in range(B):
            kfs = keyframe_list[b]
            n_valid = min(len(kfs), self.max_keyframes)
            mask[b, :n_valid] = True
        return mask


class KeyframeMaskGenerator(nn.Module):
    """Generates training-time keyframe masks with cosine scheduling."""

    def __init__(self, mask_min: int = 0, mask_max: int = 10, cosine_schedule: bool = True, mask_dropout_p: float = 0.1):
        super().__init__()
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.cosine_schedule = cosine_schedule
        self.mask_dropout_p = mask_dropout_p

    def sample_mask(self, num_frames: int, noise_level: Optional[Tensor] = None, device: torch.device = torch.device("cpu")) -> Tensor:
        if self.training and torch.rand(1).item() < self.mask_dropout_p:
            if self.cosine_schedule and noise_level is not None:
                sigma = noise_level.mean().item() if noise_level.numel() > 1 else noise_level.item()
                cos_factor = (1 - torch.cos(torch.tensor(sigma * torch.pi))) / 2
                mask_count = int((self.mask_max - self.mask_min) * cos_factor.item() + self.mask_min)
            else:
                mask_count = torch.randint(self.mask_min, self.mask_max + 1, (1,)).item()
            keep_count = max(1, num_frames - mask_count)
            keep_indices = torch.randperm(num_frames)[:keep_count].sort().values
            mask = torch.zeros(num_frames, device=device, dtype=torch.bool)
            mask[keep_indices] = True
        else:
            mask = torch.ones(num_frames, device=device, dtype=torch.bool)
        return mask
