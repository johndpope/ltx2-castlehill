"""Training strategies for different conditioning modes.
This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio)
- Video-to-video training (IC-LoRA mode with reference videos)
- SCD training (Separable Causal Diffusion for long-form video)
- VFM-SCD training (Variational Flow Maps + SCD for one-step conditional generation)
Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy
from ltx_trainer.training_strategies.scd_strategy import SCDTrainingConfig, SCDTrainingStrategy
from ltx_trainer.training_strategies.vfm_scd_strategy import VFMSCDTrainingConfig, VFMSCDTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy import VFMTrainingConfig, VFMTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1b import VFMv1bTrainingConfig, VFMv1bTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1c import VFMv1cTrainingConfig, VFMv1cTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1d import VFMv1dTrainingConfig, VFMv1dTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1e import VFMv1eTrainingConfig, VFMv1eTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1f import VFMv1fTrainingConfig, VFMv1fTrainingStrategy
# from ltx_trainer.training_strategies.vfm_strategy_v1_1f import VFMv11fTrainingConfig, VFMv11fTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1_2f import VFMv12fTrainingConfig, VFMv12fTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1g import VFMv1gTrainingConfig, VFMv1gTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v1h import VFMv1hTrainingConfig, VFMv1hTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v2a import VFMv2aTrainingConfig, VFMv2aTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v2b import VFMv2bTrainingConfig, VFMv2bTrainingStrategy
from ltx_trainer.training_strategies.vfm_strategy_v3a import DMDVFMv3aTrainingConfig, DMDVFMv3aTrainingStrategy
from ltx_trainer.training_strategies.vfm_distill_strategy import VFMDistillConfig, VFMDistillStrategy
from ltx_trainer.training_strategies.vfm_scd_distill_strategy import VFMSCDDistillConfig, VFMSCDDistillStrategy
from ltx_trainer.training_strategies.isogen_strategy import IsoGenTrainingConfig, IsoGenTrainingStrategy

# Type alias for all strategy config types
TrainingStrategyConfig = TextToVideoConfig | VideoToVideoConfig | SCDTrainingConfig | VFMSCDDistillConfig | VFMSCDTrainingConfig | VFMTrainingConfig | VFMv1bTrainingConfig | VFMv1cTrainingConfig | VFMv1dTrainingConfig | VFMv1eTrainingConfig | VFMv1fTrainingConfig | VFMv12fTrainingConfig | VFMv1gTrainingConfig | VFMv1hTrainingConfig | VFMv2aTrainingConfig | VFMv2bTrainingConfig | DMDVFMv3aTrainingConfig | VFMDistillConfig | IsoGenTrainingConfig

__all__ = [
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "IsoGenTrainingConfig",
    "IsoGenTrainingStrategy",
    "ModelInputs",
    "SCDTrainingConfig",
    "SCDTrainingStrategy",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "VFMSCDTrainingConfig",
    "VFMSCDTrainingStrategy",
    "VFMTrainingConfig",
    "VFMTrainingStrategy",
    "VFMDistillConfig",
    "VFMDistillStrategy",
    "VFMSCDDistillConfig",
    "VFMSCDDistillStrategy",
    "VFMv1bTrainingConfig",
    "VFMv1bTrainingStrategy",
    "VFMv1cTrainingConfig",
    "VFMv1cTrainingStrategy",
    "VFMv1dTrainingConfig",
    "VFMv1dTrainingStrategy",
    "VFMv1eTrainingConfig",
    "VFMv1eTrainingStrategy",
    "VFMv1fTrainingConfig",
    "VFMv1fTrainingStrategy",
    "VFMv12fTrainingConfig",
    "VFMv12fTrainingStrategy",
    "VFMv1gTrainingConfig",
    "VFMv1gTrainingStrategy",
    "VFMv1hTrainingConfig",
    "VFMv1hTrainingStrategy",
    "VFMv2aTrainingConfig",
    "VFMv2aTrainingStrategy",
    "DMDVFMv3aTrainingConfig",
    "DMDVFMv3aTrainingStrategy",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.
    The strategy is determined by the `name` field in the configuration.
    Args:
        config: Strategy-specific configuration with a `name` field
    Returns:
        The appropriate training strategy instance
    Raises:
        ValueError: If strategy name is not supported
    """

    match config:
        case TextToVideoConfig():
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            strategy = VideoToVideoStrategy(config)
        case SCDTrainingConfig():
            strategy = SCDTrainingStrategy(config)
        case VFMSCDDistillConfig():
            strategy = VFMSCDDistillStrategy(config)
        case VFMSCDTrainingConfig():
            strategy = VFMSCDTrainingStrategy(config)
        case IsoGenTrainingConfig():
            strategy = IsoGenTrainingStrategy(config)
        case VFMDistillConfig():
            strategy = VFMDistillStrategy(config)
        case DMDVFMv3aTrainingConfig():
            strategy = DMDVFMv3aTrainingStrategy(config)
        case VFMv2bTrainingConfig():
            strategy = VFMv2bTrainingStrategy(config)
        case VFMv2aTrainingConfig():
            strategy = VFMv2aTrainingStrategy(config)
        case VFMv1hTrainingConfig():
            strategy = VFMv1hTrainingStrategy(config)
        case VFMv1gTrainingConfig():
            strategy = VFMv1gTrainingStrategy(config)
        case VFMv12fTrainingConfig():
            strategy = VFMv12fTrainingStrategy(config)
        case VFMv1fTrainingConfig():
            strategy = VFMv1fTrainingStrategy(config)
        case VFMv1eTrainingConfig():
            strategy = VFMv1eTrainingStrategy(config)
        case VFMv1dTrainingConfig():
            strategy = VFMv1dTrainingStrategy(config)
        case VFMv1cTrainingConfig():
            strategy = VFMv1cTrainingStrategy(config)
        case VFMv1bTrainingConfig():
            strategy = VFMv1bTrainingStrategy(config)
        case VFMTrainingConfig():
            strategy = VFMTrainingStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    audio_mode = "(audio enabled 🔈)" if getattr(config, "with_audio", False) else "(audio disabled 🔇)"
    logger.debug(f"🎯 Using {strategy.__class__.__name__} training strategy {audio_mode}")
    return strategy
