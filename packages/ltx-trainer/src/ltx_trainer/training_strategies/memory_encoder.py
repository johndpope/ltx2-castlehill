"""Memory condition encoder for IsoGen game state → cross-attention tokens.

Maps per-frame game state (geo_conditioning, poses, actions) into tokens
that can be concatenated with text embeddings as additional cross-attention
context for the LTX-2 transformer.

Input dimensions (per clip):
    geo_cond:  [num_frames, 4, 17, 17]  — isometric tile features
    poses:     [num_frames, 10]          — agent pose (position, rotation, etc.)
    actions:   [num_frames]              — discrete action indices

Output: [num_memory_tokens, cross_attention_dim] tokens ready for concat
with video_prompt_embeds.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MemoryConditionEncoder(nn.Module):
    """Encodes IsoGen game state into cross-attention tokens.

    Produces one token per frame in the clip. Each token encodes the full
    game state for that timestep: flattened geo_conditioning + pose + action.
    """

    def __init__(
        self,
        cross_attention_dim: int = 3840,
        geo_channels: int = 4,
        geo_grid_size: int = 17,
        pose_dim: int = 10,
        num_actions: int = 32,
        action_embed_dim: int = 16,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim

        # Action embedding (discrete → continuous)
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)

        # Input: flattened geo_cond + pose + action_embed
        geo_flat_dim = geo_channels * geo_grid_size * geo_grid_size  # 4*17*17 = 1156
        input_dim = geo_flat_dim + pose_dim + action_embed_dim  # 1156 + 10 + 16 = 1182

        # Two-layer MLP: input → hidden → cross_attention_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(
        self,
        geo_cond: Tensor,   # [B, F, 4, 17, 17]
        poses: Tensor,      # [B, F, 10]
        actions: Tensor,    # [B, F] int64
    ) -> Tensor:
        """Encode memory state into cross-attention tokens.

        Returns:
            tokens: [B, F, cross_attention_dim] — one token per frame
        """
        B, F = geo_cond.shape[:2]
        device = geo_cond.device
        dtype = geo_cond.dtype

        # Flatten geo_cond: [B, F, 4, 17, 17] → [B, F, 1156]
        geo_flat = geo_cond.reshape(B, F, -1)

        # Embed actions: [B, F] → [B, F, action_embed_dim]
        act_emb = self.action_embed(actions).to(dtype)

        # Concat all features: [B, F, 1182]
        features = torch.cat([geo_flat, poses, act_emb], dim=-1)

        # Project to cross-attention dim: [B, F, 3840]
        tokens = self.encoder(features)

        return tokens
