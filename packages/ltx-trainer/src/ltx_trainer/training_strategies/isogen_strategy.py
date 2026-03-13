"""IsoGen training strategy — VFM + memory conditioning for isometric game worlds.

Extends VFMTrainingStrategy with game state memory conditioning:
- Loads per-clip memory (geo_cond, poses, actions) alongside latents + conditions
- Encodes memory into cross-attention tokens via MemoryConditionEncoder
- Concatenates memory tokens with text embeddings as additional context

The transformer sees [text_tokens | memory_tokens] in cross-attention,
giving it temporal game-state awareness with zero backbone changes.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.memory_encoder import MemoryConditionEncoder
from ltx_trainer.training_strategies.vfm_strategy import (
    VFMTrainingConfig,
    VFMTrainingStrategy,
)


class IsoGenTrainingConfig(VFMTrainingConfig):
    """Configuration for IsoGen VFM training with memory conditioning."""

    name: Literal["isogen"] = "isogen"

    # === Memory Encoder ===
    memory_hidden_dim: int = Field(default=1024, description="Hidden dim for memory encoder MLP")
    memory_num_actions: int = Field(default=32, description="Number of discrete actions")
    memory_action_embed_dim: int = Field(default=16, description="Action embedding dimension")
    memory_learning_rate: float = Field(default=1e-4, description="Learning rate for memory encoder")
    memory_dropout_p: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Probability of dropping memory tokens (classifier-free guidance)",
    )

    # === Memory data ===
    memory_dir: str = Field(default="memory", description="Directory name for memory .pt files")


class IsoGenTrainingStrategy(VFMTrainingStrategy):
    """IsoGen training: VFM + memory conditioning for isometric game worlds.

    Inherits all VFM logic (noise adapter, three-part loss, EMA, adaptive loss)
    and adds memory conditioning as extra cross-attention tokens.
    """

    config: IsoGenTrainingConfig

    def __init__(self, config: IsoGenTrainingConfig):
        super().__init__(config)
        self._memory_encoder: MemoryConditionEncoder | None = None

    def create_memory_encoder(
        self,
        cross_attention_dim: int = 3840,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> MemoryConditionEncoder:
        """Create and store the memory encoder. Called by trainer after model init."""
        self._memory_encoder = MemoryConditionEncoder(
            cross_attention_dim=cross_attention_dim,
            hidden_dim=self.config.memory_hidden_dim,
            num_actions=self.config.memory_num_actions,
            action_embed_dim=self.config.memory_action_embed_dim,
        ).to(device=device, dtype=dtype)
        logger.info(
            f"IsoGen: Created MemoryConditionEncoder "
            f"(hidden={self.config.memory_hidden_dim}, "
            f"actions={self.config.memory_num_actions}, "
            f"output_dim={cross_attention_dim}) on {device}"
        )
        param_count = sum(p.numel() for p in self._memory_encoder.parameters())
        logger.info(f"IsoGen: Memory encoder params: {param_count:,}")
        return self._memory_encoder

    @property
    def memory_encoder(self) -> MemoryConditionEncoder | None:
        return self._memory_encoder

    def get_data_sources(self) -> dict[str, str]:
        """Add memory directory to VFM data sources."""
        sources = super().get_data_sources()
        sources[self.config.memory_dir] = "memory"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare VFM inputs with memory tokens concatenated to text context."""

        # Encode memory → cross-attention tokens BEFORE calling super
        memory_data = batch.get("memory")
        memory_tokens = None
        memory_mask = None

        if memory_data is not None and self._memory_encoder is not None:
            geo_cond = memory_data["geo_cond"]    # [B, F, 4, 17, 17]
            poses = memory_data["poses"]          # [B, F, 10]
            actions = memory_data["actions"]      # [B, F]

            device = batch["latents"]["latents"].device
            dtype = batch["latents"]["latents"].dtype

            geo_cond = geo_cond.to(device=device, dtype=dtype)
            poses = poses.to(device=device, dtype=dtype)
            actions = actions.to(device=device)

            # Encode: [B, F, cross_attention_dim]
            memory_tokens = self._memory_encoder(geo_cond, poses, actions)
            memory_tokens = memory_tokens.to(dtype=dtype)

            B, num_mem_tokens = memory_tokens.shape[:2]

            # Classifier-free guidance: randomly drop memory tokens
            if self.training and self.config.memory_dropout_p > 0:
                import random
                if random.random() < self.config.memory_dropout_p:
                    memory_tokens = torch.zeros_like(memory_tokens)

            # Memory mask: all ones (all tokens attend)
            memory_mask = torch.ones(B, num_mem_tokens, dtype=torch.long, device=device)

        # Temporarily inject memory tokens into conditions before super() call
        if memory_tokens is not None:
            conditions = batch["conditions"]
            orig_embeds = conditions["video_prompt_embeds"]    # [B, 1024, 3840]
            orig_mask = conditions["prompt_attention_mask"]     # [B, 1024]

            # Concat: [B, 1024 + F, 3840]
            conditions["video_prompt_embeds"] = torch.cat(
                [orig_embeds, memory_tokens], dim=1
            )
            conditions["prompt_attention_mask"] = torch.cat(
                [orig_mask, memory_mask], dim=1
            )

        # Call parent VFM logic (noise adapter, flow matching, modality building)
        model_inputs = super().prepare_training_inputs(batch, timestep_sampler)

        # Store memory info for logging
        if memory_tokens is not None:
            model_inputs._isogen_memory_tokens = memory_tokens
            model_inputs._isogen_num_memory_tokens = memory_tokens.shape[1]

        # Restore original conditions (avoid leaking modified tensors across batches)
        if memory_tokens is not None:
            conditions["video_prompt_embeds"] = orig_embeds
            conditions["prompt_attention_mask"] = orig_mask

        return model_inputs

    @property
    def training(self) -> bool:
        """Check if memory encoder is in training mode."""
        if self._memory_encoder is not None:
            return self._memory_encoder.training
        return True

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute VFM loss (inherited) + log memory metrics."""
        total_loss = super().compute_loss(video_pred, audio_pred, inputs)

        # Add memory metrics to VFM metrics
        num_mem = getattr(inputs, "_isogen_num_memory_tokens", 0)
        self._last_vfm_metrics["isogen/memory_tokens"] = num_mem

        if hasattr(inputs, "_isogen_memory_tokens") and inputs._isogen_memory_tokens is not None:
            mem_tokens = inputs._isogen_memory_tokens
            self._last_vfm_metrics["isogen/memory_norm"] = mem_tokens.norm(dim=-1).mean().item()

        return total_loss

    def get_trainable_parameters(self) -> list:
        """Return memory encoder parameters for the optimizer."""
        if self._memory_encoder is not None:
            return list(self._memory_encoder.parameters())
        return []

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        meta = super().get_checkpoint_metadata()
        meta["isogen_memory_hidden_dim"] = self.config.memory_hidden_dim
        meta["isogen_memory_num_actions"] = self.config.memory_num_actions
        return meta
