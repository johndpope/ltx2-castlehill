import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F  # noqa: N812
from torchvision.utils import save_image

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.model_loader import load_text_encoder
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)

        # Check if using cached final embeddings (skip loading text encoder entirely)
        self._use_cached_final_embeddings = self._config.data.use_cached_final_embeddings
        if self._use_cached_final_embeddings:
            logger.info("🚀 Using cached final embeddings - skipping text encoder load (~28GB VRAM saved)")
            self._text_encoder = None
            self._cached_validation_embeddings = None
            # Warn if validation is configured - it won't work without text encoder
            if self._config.validation.prompts and self._config.validation.interval:
                logger.warning(
                    "⚠️ Validation is configured but text encoder is not loaded. "
                    "Validation will be SKIPPED. To enable validation with cached embeddings, "
                    "either set validation.interval to null, or disable use_cached_final_embeddings."
                )
        else:
            self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()

        self._load_models()

        # Pass VAE decoder to strategy for pixel-space losses (if strategy supports it)
        if hasattr(self._training_strategy, "set_vae_decoder"):
            self._training_strategy.set_vae_decoder(self._vae_decoder)
            logger.info("Passed VAE decoder to training strategy for pixel-space loss computation")

        self._setup_accelerator()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._init_wandb()

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        # Use the same seed for all processes and ensure deterministic operations
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()
        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=cfg.optimization.steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()
        self._global_step = 0

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            # Initial validation before training starts
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(progress)
                if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

            self._accelerator.wait_for_everyone()

            for step in range(cfg.optimization.steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Epoch boundary — check for new samples from live ingest
                    if self._config.data.live_ingest_enabled:
                        new_count = self._dataset.rescan()
                        if new_count > 0:
                            self._rebuild_dataloader()
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    # Check if we should log reconstructions this step (W&B)
                    should_log_recon = (
                        is_optimization_step
                        and IS_MAIN_PROCESS
                        and hasattr(self._training_strategy, "log_reconstructions_to_wandb")
                        and hasattr(self._training_strategy, "config")
                        and getattr(self._training_strategy.config, "log_reconstructions", False)
                        and self._global_step > 0
                        and self._global_step % getattr(
                            self._training_strategy.config, "reconstruction_log_interval", 500
                        ) == 0
                    )

                    # Always get predictions on optimization steps for debug image
                    should_save_debug = is_optimization_step and IS_MAIN_PROCESS

                    if should_log_recon or should_save_debug:
                        loss, video_pred, model_inputs = self._training_step(batch, return_for_logging=True)
                    else:
                        loss = self._training_step(batch)
                        video_pred, model_inputs = None, None

                    self._accelerator.backward(loss)

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        # Cast any FP8 gradients to bf16 before clipping — PyTorch's
                        # foreach_norm doesn't support FP8 dtypes (pre-quantized models)
                        for p in self._trainable_params:
                            if p.grad is not None and p.grad.dtype in (
                                torch.float8_e4m3fn, torch.float8_e5m2,
                            ):
                                p.grad = p.grad.to(torch.bfloat16)

                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        if self._accelerator.distributed_type == DistributedType.FSDP:
                            # FSDP: All processes must participate in validation
                            sampled_videos_paths = self._sample_videos(progress)
                            if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)
                        # DDP: Only main process runs validation
                        elif IS_MAIN_PROCESS:
                            sampled_videos_paths = self._sample_videos(progress)
                            if sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps

                    progress.update_training(
                        loss=loss.item(),
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        metrics = {
                            "train/loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": self._global_step,
                        }
                        # Log scheduled sampling probability if active
                        if hasattr(self._training_strategy, "_get_p_ar"):
                            p_ar = self._training_strategy._get_p_ar()
                            if p_ar > 0.0 or hasattr(self._training_strategy, "_current_step"):
                                metrics["train/p_ar"] = p_ar
                        self._log_metrics(metrics)

                        # Save debug image every optimization step (overwrites)
                        if should_save_debug and video_pred is not None and model_inputs is not None:
                            self._save_debug_image(video_pred.detach(), model_inputs, self._global_step)

                        # Log reconstructions to W&B if this is a logging step
                        if should_log_recon and video_pred is not None and model_inputs is not None:
                            try:
                                # Move VAE decoder to GPU temporarily for fast decoding
                                # (single-GPU mode keeps it on CPU to save VRAM)
                                hw_devices = self._config.hardware.devices
                                is_multi_gpu = hw_devices.transformer != hw_devices.vae_decoder
                                if not is_multi_gpu and self._vae_decoder is not None:
                                    self._vae_decoder = self._vae_decoder.to(hw_devices.vae_decoder)

                                recon_metrics = self._training_strategy.log_reconstructions_to_wandb(
                                    video_pred=video_pred.detach(),
                                    inputs=model_inputs,
                                    step=self._global_step,
                                    vae_decoder=self._vae_decoder,
                                )
                                if recon_metrics:
                                    self._log_metrics(recon_metrics)

                                # Move VAE decoder back to CPU (single-GPU only)
                                if not is_multi_gpu and self._vae_decoder is not None:
                                    self._vae_decoder = self._vae_decoder.to("cpu")
                                    torch.cuda.empty_cache()
                            except Exception as e:
                                logger.warning(f"Failed to log reconstructions: {e}")
                                # Ensure VAE back on CPU even on error
                                if not is_multi_gpu and self._vae_decoder is not None:
                                    self._vae_decoder = self._vae_decoder.to("cpu")

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = cfg.optimization.steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _move_batch_to_device(self, batch: dict[str, dict[str, Tensor]], device: str) -> dict[str, dict[str, Tensor]]:
        """Move all tensors in a batch to the specified device.

        Args:
            batch: Nested dict of tensors from the dataloader
            device: Target device (e.g., "cuda:0", "cuda:1")

        Returns:
            Batch with all tensors moved to the device
        """
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                moved_batch[key] = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in value.items()
                }
            elif isinstance(value, Tensor):
                moved_batch[key] = value.to(device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _training_step(
        self, batch: dict[str, dict[str, Tensor]], return_for_logging: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Any]:
        """Perform a single training step using the configured strategy.

        Args:
            batch: Training batch
            return_for_logging: If True, also return predictions and inputs for logging

        Returns:
            Loss tensor, or (loss, video_pred, model_inputs) if return_for_logging=True
        """
        # Accelerate's dataloader places data on accelerator.device (cuda:0)
        # but connectors and transformer may be on cuda:1
        transformer_device = self._config.hardware.devices.transformer
        batch = self._move_batch_to_device(batch, transformer_device)

        conditions = batch["conditions"]

        if self._use_cached_final_embeddings:
            # Use pre-computed final embeddings directly (no connector call needed)
            # Format from conditions_final/*.pt: video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask
            # These are already in final form [seq_len, 4096]
            pass  # conditions already has video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask
        else:
            # Apply embedding connectors to transform raw pre-computed text embeddings
            # Format from conditions/*.pt: prompt_embeds [1024, 3840], prompt_attention_mask [1024]
            video_embeds, audio_embeds, attention_mask = self._text_encoder._run_connectors(
                conditions["prompt_embeds"], conditions["prompt_attention_mask"]
            )
            conditions["video_prompt_embeds"] = video_embeds
            conditions["audio_prompt_embeds"] = audio_embeds
            conditions["prompt_attention_mask"] = attention_mask

        # Pass current step to strategy (for scheduled sampling curriculum)
        if hasattr(self._training_strategy, "set_current_step"):
            self._training_strategy.set_current_step(self._global_step)

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        # For SCD strategy: encoder already ran in prepare_training_inputs;
        # here we only run the decoder with encoder features
        if hasattr(model_inputs, "_scd_model") and model_inputs._scd_model is not None:
            # Per-frame decoder: process each frame independently through the decoder,
            # matching the autoregressive inference setup (1 frame per forward pass).
            # This prevents the train/inference attention scope mismatch that causes
            # grid artifacts when the decoder LoRA is trained on multi-frame input
            # but used on single-frame input at inference.
            per_frame = getattr(model_inputs, "_per_frame_decoder", False)
            if per_frame:
                video_pred, audio_pred = model_inputs._scd_model.forward_decoder_per_frame(
                    video=model_inputs.video,
                    encoder_features=model_inputs._encoder_features,
                    perturbations=None,
                    tokens_per_frame=model_inputs._tokens_per_frame,
                    num_frames=model_inputs._num_frames,
                )
            else:
                # Pass EditCtrl control signals if available
                local_control = getattr(model_inputs, "_local_control", None)
                global_context = getattr(model_inputs, "_global_context", None)

                video_pred, audio_pred = model_inputs._scd_model.forward_decoder(
                    video=model_inputs.video,
                    encoder_features=model_inputs._encoder_features,
                    audio=model_inputs.audio,
                    perturbations=None,
                    encoder_audio_args=model_inputs._encoder_audio_args,
                    local_control=local_control,
                    global_context=global_context,
                )
        else:
            video_pred, audio_pred = self._transformer(
                video=model_inputs.video,
                audio=model_inputs.audio,
                perturbations=None,
            )

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        if return_for_logging:
            return loss, video_pred, model_inputs
        return loss

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder, computes and returns validation embeddings."""

        # This method:
        #   1. Loads the text encoder on GPU
        #   2. If validation prompts are configured, computes and caches their embeddings
        #   3. Unloads the heavy Gemma model while keeping the lightweight embedding connectors
        #   The text encoder is kept (as self._text_encoder) but with model/tokenizer/feature_extractor
        #   set to None. Only the embedding connectors remain for use during training.

        # Load text encoder on configured device (supports multi-GPU)
        text_encoder_device = self._config.hardware.devices.text_encoder
        logger.debug(f"Loading text encoder on {text_encoder_device}...")

        self._text_encoder = load_text_encoder(
            checkpoint_path=self._config.model.model_path,
            gemma_model_path=self._config.model.text_encoder_path,
            device=text_encoder_device,
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        if self._config.validation.prompts:
            logger.info(f"Pre-computing embeddings for {len(self._config.validation.prompts)} validation prompts...")
            cached_embeddings = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    v_ctx_pos, a_ctx_pos, _ = self._text_encoder(prompt)
                    v_ctx_neg, a_ctx_neg, _ = self._text_encoder(self._config.validation.negative_prompt)

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=v_ctx_pos.cpu(),
                            audio_context_positive=a_ctx_pos.cpu(),
                            video_context_negative=v_ctx_neg.cpu() if v_ctx_neg is not None else None,
                            audio_context_negative=a_ctx_neg.cpu() if a_ctx_neg is not None else None,
                        )
                    )

        # Unload heavy components to free VRAM, keeping only the embedding connectors
        self._text_encoder.model = None
        self._text_encoder.tokenizer = None
        self._text_encoder.feature_extractor_linear = None

        logger.debug("Validation prompt embeddings cached. Gemma model unloaded")
        return cached_embeddings

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        # Get hardware device config for multi-GPU support
        hw_devices = self._config.hardware.devices
        is_multi_gpu = hw_devices.transformer != hw_devices.vae_decoder

        # Always load transformer to CPU first - the bf16 model is ~38GB which
        # won't fit on most GPUs. We'll quantize it on CPU, then move to GPU.
        # VAE and other components will be moved to their target devices after loading.
        logger.debug("Loading transformer to CPU (will quantize then move to GPU)")

        # Load all model components (except text encoder - already handled)
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
        )

        # Extract components and move to configured devices (supports multi-GPU)
        # Note: hw_devices was already set above for transformer loading
        self._transformer = components.transformer
        self._scheduler = components.scheduler

        # VAE decoder on configured device
        self._vae_decoder = components.video_vae_decoder.to(device=hw_devices.vae_decoder, dtype=torch.bfloat16)

        # VAE encoder on configured device
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(device=hw_devices.vae_encoder, dtype=torch.bfloat16)

        # Audio components on configured device
        self._audio_vae = components.audio_vae_decoder
        if self._audio_vae is not None:
            self._audio_vae = self._audio_vae.to(device=hw_devices.audio_vae)
        self._vocoder = components.vocoder
        if self._vocoder is not None:
            self._vocoder = self._vocoder.to(device=hw_devices.vocoder)

        # Note: self._text_encoder was set in _load_text_encoder_and_cache_embeddings
        # Note: transformer device is handled by Accelerate, but we track the target device
        self._transformer_device = hw_devices.transformer

        # Log multi-GPU configuration if not default (devices are different)
        if hw_devices.transformer != hw_devices.vae_decoder:
            logger.info(f"Multi-GPU config: transformer→{hw_devices.transformer}, VAE→{hw_devices.vae_decoder}")

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32

        self._transformer = self._transformer.to(dtype=transformer_dtype)

        # Quantize on CPU first (the full bf16 model is ~38GB, won't fit on most GPUs)
        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
                device=hw_devices.transformer,
            )

        # Free any GPU memory cached during blockwise quantization
        if self._config.acceleration.quantization is not None:
            torch.cuda.empty_cache()

        # After quantization (or if no quantization), move transformer to target device
        # INT8 quantized model is ~12GB, FP8 pre-quantized ~25GB, bf16 LoRA model is ~38GB
        if is_multi_gpu:
            model_size = "quantized" if self._config.acceleration.quantization else "bf16"
            logger.info(f"Moving {model_size} transformer to {hw_devices.transformer}...")
            self._transformer = self._transformer.to(hw_devices.transformer)

        # Wrap transformer with SCD model if using SCD or VFM-SCD strategy
        from ltx_trainer.training_strategies.scd_strategy import SCDTrainingStrategy  # noqa: PLC0415
        from ltx_trainer.training_strategies.vfm_scd_strategy import VFMSCDTrainingStrategy  # noqa: PLC0415
        if isinstance(self._training_strategy, (SCDTrainingStrategy, VFMSCDTrainingStrategy)):
            from ltx_core.model.transformer.scd_model import LTXSCDModel  # noqa: PLC0415
            scd_config = self._training_strategy.config
            # Pass EditCtrl local control injection config if available
            lc_injection = getattr(scd_config, 'local_control_injection', 'pre_decoder')
            lc_layers = getattr(scd_config, 'local_control_layers', None)
            self._transformer = LTXSCDModel(
                base_model=self._transformer,
                encoder_layers=scd_config.encoder_layers,
                decoder_input_combine=scd_config.decoder_input_combine,
                local_control_injection=lc_injection,
                local_control_layers=lc_layers,
            )
            self._training_strategy.set_scd_model(self._transformer)
            logger.info(
                f"SCD wrapper: {scd_config.encoder_layers} encoder layers, "
                f"{len(self._transformer.decoder_blocks)} decoder layers, "
                f"combine={scd_config.decoder_input_combine}"
            )

        # VFM: Create noise adapter and attach to strategy
        self._noise_adapter = None
        if isinstance(self._training_strategy, VFMSCDTrainingStrategy):
            from ltx_core.model.transformer.noise_adapter import create_noise_adapter  # noqa: PLC0415
            adapter_kwargs = self._training_strategy.get_noise_adapter_params()
            self._noise_adapter = create_noise_adapter(**adapter_kwargs)
            # Move adapter to same device as transformer
            adapter_device = next(self._transformer.parameters()).device
            self._noise_adapter = self._noise_adapter.to(adapter_device)
            self._training_strategy.set_noise_adapter(self._noise_adapter)

            adapter_params = sum(p.numel() for p in self._noise_adapter.parameters())
            logger.info(
                f"VFM noise adapter: {adapter_kwargs['variant']}, "
                f"{adapter_params:,} params, "
                f"hidden_dim={adapter_kwargs['hidden_dim']}, "
                f"layers={adapter_kwargs['num_layers']}"
            )

        # Placeholder for optional modules (EditCtrl, TMA — not included in CastleHill)
        self._local_context_module = None
        self._global_embedder = None
        self._tma = None

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode and stage.

        For multi-stage training:
        - Stage 1: Train DiT (LoRA) + strategy components
        - Stage 2: Freeze DiT, train connector only
        - Stage 3: Joint fine-tuning of all components
        """
        # Check if strategy uses stage-based training
        training_stage = self._get_training_stage()

        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()

            # Stage 2: Freeze LoRA parameters (only train TMA)
            if training_stage == 2:
                logger.info("Stage 2: Freezing DiT/LoRA parameters (training TMA only)")
                for param in self._transformer.parameters():
                    param.requires_grad = False


        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            if training_stage == 2:
                # Stage 2: Freeze transformer (only train TMA)
                logger.info("Stage 2: Freezing transformer (training TMA only)")
                self._transformer.requires_grad_(False)
            else:
                self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # Collect transformer parameters (if not frozen)
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]

        # Add strategy-specific trainable parameters (TPB, ConceptEmbedding, TMA)
        strategy_params = self._get_strategy_trainable_params()
        if strategy_params:
            self._trainable_params.extend(strategy_params)
            logger.info(
                f"Added {len(strategy_params)} strategy parameters "
                f"({sum(p.numel() for p in strategy_params):,} total)"
            )

        # VFM: Add noise adapter parameters to trainable params
        if self._noise_adapter is not None:
            adapter_params = list(self._noise_adapter.parameters())
            self._trainable_params.extend(adapter_params)
            logger.info(
                f"Added VFM noise adapter: {sum(p.numel() for p in adapter_params):,} params"
            )

        total_params = sum(p.numel() for p in self._trainable_params)
        logger.info(f"Total trainable params: {total_params:,} (stage {training_stage or 'N/A'})")

    def _get_training_stage(self) -> int | None:
        """Get the training stage from strategy config, if applicable."""
        strategy_config = self._config.training_strategy
        if hasattr(strategy_config, "training_stage"):
            return strategy_config.training_stage
        return None

    def _get_strategy_trainable_params(self) -> list:
        """Get trainable parameters from the training strategy.

        This allows strategies to add their own trainable parameters to the optimizer.
        """
        if hasattr(self._training_strategy, "get_trainable_parameters"):
            return self._training_strategy.get_trainable_parameters()
        return []

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Split strategy params (strategy.*) from LoRA params (diffusion_model.*)
        strategy_dict = {k: v for k, v in state_dict.items() if k.startswith("strategy.")}
        lora_dict = {k: v for k, v in state_dict.items() if not k.startswith("strategy.")}

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        lora_dict = {k.replace("diffusion_model.", "", 1): v for k, v in lora_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, lora_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")

        # Load strategy parameters (ConceptEmbedding, TMA, etc.) if present
        if strategy_dict and hasattr(self._training_strategy, "load_strategy_state_dict"):
            loaded, skipped = self._training_strategy.load_strategy_state_dict(strategy_dict)
            if loaded:
                logger.info(f"✅ Loaded {len(loaded)} strategy params from checkpoint")
            if skipped:
                logger.warning(f"⚠️ Skipped {len(skipped)} strategy params (modules not initialised)")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # Handle device placement based on hardware config
        hw_devices = self._config.hardware.devices
        is_multi_gpu = hw_devices.transformer != hw_devices.vae_decoder

        if is_multi_gpu:
            # Multi-GPU: Keep VAE on its assigned device (e.g., cuda:0)
            # VAE was already placed on correct device in _load_models
            logger.debug(f"Multi-GPU: VAE staying on {hw_devices.vae_decoder}")
        else:
            # Single-GPU: Keep frozen models on CPU for memory efficiency
            self._vae_decoder = self._vae_decoder.to("cpu")
            if self._vae_encoder is not None:
                self._vae_encoder = self._vae_encoder.to("cpu")

        # Move transformer to target device before Accelerate prepares it
        # (In multi-GPU mode, transformer was already moved in _load_models before quantization)
        if is_multi_gpu:
            # Verify transformer is on correct device (should already be there)
            logger.debug(f"Transformer should already be on {hw_devices.transformer} (moved before quantization)")

            # Move text encoder connectors to transformer device for training
            # (they need to be on the same device as the transformer for the forward pass)
            if hasattr(self, '_text_encoder') and self._text_encoder is not None:
                if hasattr(self._text_encoder, "embeddings_connector"):
                    self._text_encoder.embeddings_connector = self._text_encoder.embeddings_connector.to(
                        hw_devices.transformer
                    )
                if hasattr(self._text_encoder, "audio_embeddings_connector"):
                    self._text_encoder.audio_embeddings_connector = self._text_encoder.audio_embeddings_connector.to(
                        hw_devices.transformer
                    )
                logger.debug(f"Text encoder connectors moved to {hw_devices.transformer}")

        # Embedding connectors are already on GPU from _load_text_encoder_and_cache_embeddings

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()

            # Override conditions directory if using cached final embeddings
            if self._use_cached_final_embeddings:
                final_dir = self._config.data.final_embeddings_dir
                # Replace 'conditions' with final embeddings directory in data_sources
                if isinstance(data_sources, dict):
                    data_sources = {
                        (final_dir if k == "conditions" else k): v
                        for k, v in data_sources.items()
                    }
                elif isinstance(data_sources, list):
                    data_sources = [
                        final_dir if s == "conditions" else s
                        for s in data_sources
                    ]
                logger.info(f"Using cached final embeddings from: {final_dir}/")

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _rebuild_dataloader(self) -> None:
        """Rebuild DataLoader after dataset rescan to pick up new samples.

        persistent_workers=True forks the dataset at DataLoader creation time,
        so workers hold stale copies. The only way to refresh is to create a
        brand-new DataLoader from the updated dataset.
        """
        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(dataloader)
        logger.info(f"Rebuilt DataLoader with {len(self._dataset)} samples")

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        wd = opt_cfg.weight_decay
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr, weight_decay=wd)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(self._trainable_params, lr=lr, weight_decay=wd)
        elif opt_cfg.optimizer_type == "muon":
            from torch.optim import Muon  # noqa: PLC0415

            # Muon requires 2D+ params; 1D params (biases, norms) fall back to AdamW.
            muon_params = [p for p in self._trainable_params if p.ndim >= 2]
            adamw_params = [p for p in self._trainable_params if p.ndim < 2]

            if adamw_params:
                logger.info(
                    f"Muon optimizer: {len(muon_params)} params (2D+) via Muon, "
                    f"{len(adamw_params)} params (1D) via AdamW fallback"
                )
                optimizer = Muon(
                    [
                        {"params": muon_params, "lr": lr, "weight_decay": wd},
                        {"params": adamw_params, "lr": lr * 0.1, "weight_decay": wd, "muon": False},
                    ],
                )
            else:
                logger.info(f"Muon optimizer: {len(muon_params)} params (all 2D+)")
                optimizer = Muon(self._trainable_params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        # Add scheduler initialization
        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        warmup_steps = self._config.optimization.warmup_steps

        if scheduler_type is None:
            return None

        # Subtract warmup from main scheduler duration so total = warmup + main = steps
        main_steps = max(steps - warmup_steps, 1)

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=main_steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),  # First restart cycle length
                T_mult=params.pop("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # Wrap with linear warmup if warmup_steps > 0
        if warmup_steps > 0 and scheduler is not None:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-3,  # Start at 0.1% of peak lr
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps],
            )
            logger.info(f"LR warmup: {warmup_steps} steps (linear 0.001× → 1×), then {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        # Skip validation if using cached final embeddings without cached validation embeddings
        if self._use_cached_final_embeddings and self._cached_validation_embeddings is None:
            logger.debug("Skipping validation - no cached validation embeddings available")
            return None

        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        # Zero gradients and free GPU memory to reclaim memory before validation sampling
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        # Start sampling progress tracking
        sampling_ctx = progress.start_sampling(
            num_prompts=len(self._config.validation.prompts),
            num_steps=inference_steps,
        )

        # Create validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, num_frames = self._config.validation.video_dims

        for prompt_idx, prompt in enumerate(self._config.validation.prompts):
            # Update progress to show current video
            sampling_ctx.start_video(prompt_idx)

            # Load conditioning image if provided
            condition_image = None
            if use_images:
                image_path = self._config.validation.images[prompt_idx]
                image = open_image_as_srgb(image_path)
                # Convert PIL image to tensor [C, H, W] in [0, 1]
                condition_image = F.to_tensor(image)

            # Load reference video if provided (for IC-LoRA)
            reference_video = None
            if use_reference_videos:
                ref_video_path = self._config.validation.reference_videos[prompt_idx]
                # read_video returns [F, C, H, W] in [0, 1]
                reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

            # Get cached embeddings for this prompt if available
            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )

            # Create generation config
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
            )

            # Generate sample
            video, audio = sampler.generate(
                config=gen_config,
                device=self._accelerator.device,
            )

            # Save output (image for single frame, video otherwise)
            if IS_MAIN_PROCESS:
                ext = "png" if num_frames == 1 else "mp4"
                output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1}.{ext}"
                if num_frames == 1:
                    save_image(video, output_path)
                else:
                    save_video(
                        video_tensor=video,
                        output_path=output_path,
                        fps=self._config.validation.frame_rate,
                        audio=audio,
                        audio_sample_rate=self._vocoder.output_sample_rate if audio is not None else None,
                    )
                video_paths.append(output_path)

        # Clean up progress tasks
        sampling_ctx.cleanup()

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Include strategy-owned parameters (ConceptEmbedding, TMA, etc.)
            if hasattr(self._training_strategy, "get_strategy_state_dict"):
                strategy_sd = self._training_strategy.get_strategy_state_dict()
                if strategy_sd:
                    state_dict.update(strategy_sd)
                    logger.debug(f"Included {len(strategy_sd)} strategy params in checkpoint")

            # Cast to configured precision
            state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

            # Save to disk
            save_file(state_dict, saved_weights_path)
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}

            # Save to disk
            self._accelerator.save(full_state_dict, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        # Keep track of checkpoint paths, and cleanup old checkpoints if needed
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoints: {old_checkpoint}")
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".webp")
        media_cls = wandb.Image if is_image else wandb.Video

        samples = [media_cls(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)

    def _save_scd_debug_image(
        self,
        video_pred: Tensor,
        model_inputs: "ModelInputs",
        step: int,
    ) -> None:
        """Save SCD debug reconstruction image.

        For EditCtrl: 3-column grid (GT | Masked Input | Composited Prediction)
        For vanilla SCD: 2-column grid (GT | Prediction)

        Saves to outputs/debug_recon.png (latent pseudo-RGB) and
        outputs/debug_decoded.png (VAE decoded pixel space).
        """
        try:
            from torchvision.utils import save_image

            raw_latents = model_inputs._raw_video_latents  # [B, C, F, H, W]
            noise = model_inputs.shared_noise  # [B, seq_len, C] (patchified)
            sigmas = model_inputs.shared_sigmas

            b, c, f, h, w = raw_latents.shape

            # Check for EditCtrl mask
            edit_mask = getattr(model_inputs, "_edit_mask", None)  # [B, seq_len] bool

            # Recover predicted clean latent: clean = noise - v
            pred_clean = noise - video_pred  # [B, seq_len, C]

            # For EditCtrl: composite clean background + predicted edit region
            if edit_mask is not None:
                mask_expanded = edit_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                # Flatten raw_latents to patchified form for compositing
                gt_patchified = raw_latents.permute(0, 2, 3, 4, 1).reshape(b, -1, c)  # [B, seq_len, C]
                # Composite: use GT for unmasked, prediction for masked
                composited = gt_patchified * (1 - mask_expanded) + pred_clean * mask_expanded
                composited_spatial = composited.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
                # Also build the noisy input for visualization
                noisy_input = gt_patchified * (1 - mask_expanded) + noise * mask_expanded
                noisy_spatial = noisy_input.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
            else:
                composited_spatial = pred_clean.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
                noisy_spatial = None

            # Latent pseudo-RGB (first 3 channels, middle frame)
            mid_f = f // 2
            gt_vis = raw_latents[0, :3, mid_f].cpu().float()
            comp_vis = composited_spatial[0, :3, mid_f].cpu().float()

            def norm(x):
                x = x - x.min()
                x = x / (x.max() + 1e-8)
                return x.clamp(0, 1)

            if edit_mask is not None and noisy_spatial is not None:
                noisy_vis = noisy_spatial[0, :3, mid_f].cpu().float()
                # 3-image grid: GT | Masked Input | Composited Prediction
                grid = torch.cat([norm(gt_vis), norm(noisy_vis), norm(comp_vis)], dim=2)
            else:
                # 2-image grid: GT | Prediction
                grid = torch.cat([norm(gt_vis), norm(comp_vis)], dim=2)

            out_dir = Path(self._config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_image(grid, out_dir / "debug_recon.png")

            # Save step info
            sigma = sigmas.flatten()[0].item() if sigmas is not None else 0.0
            mask_pct = edit_mask[0].float().mean().item() * 100 if edit_mask is not None else 0
            with open(out_dir / "debug_info.txt", "w") as info_file:
                info_file.write(f"Step: {step}\nSigma: {sigma:.4f}\nMode: SCD\n")
                if edit_mask is not None:
                    info_file.write(f"Mask: {mask_pct:.1f}% tokens\n")

            # VAE-decoded frames
            if self._vae_decoder is not None:
                hw_devices = self._config.hardware.devices
                is_multi_gpu = hw_devices.transformer != hw_devices.vae_decoder
                vae_device = hw_devices.vae_decoder

                if not is_multi_gpu:
                    self._vae_decoder = self._vae_decoder.to(vae_device)

                vae_dtype = next(self._vae_decoder.parameters()).dtype

                def norm_decoded(x):
                    return ((x.float() + 1) / 2).clamp(0, 1)

                # Select up to 4 evenly spaced frames
                num_display = min(4, f)
                if num_display > 1:
                    frame_indices = [int(i * (f - 1) / (num_display - 1)) for i in range(num_display)]
                else:
                    frame_indices = [mid_f]

                gt_frames = []
                masked_frames = []
                comp_frames = []

                with torch.inference_mode():
                    for fidx in frame_indices:
                        gt_lat = raw_latents[0:1, :, fidx:fidx+1, :, :].to(vae_device, dtype=vae_dtype)
                        gt_dec = self._vae_decoder(gt_lat)[0, :, 0].cpu()
                        gt_frames.append(norm_decoded(gt_dec))

                        comp_lat = composited_spatial[0:1, :, fidx:fidx+1, :, :].to(vae_device, dtype=vae_dtype)
                        comp_dec = self._vae_decoder(comp_lat)[0, :, 0].cpu()
                        comp_frames.append(norm_decoded(comp_dec))

                        if noisy_spatial is not None:
                            noisy_lat = noisy_spatial[0:1, :, fidx:fidx+1, :, :].to(vae_device, dtype=vae_dtype)
                            noisy_dec = self._vae_decoder(noisy_lat)[0, :, 0].cpu()
                            masked_frames.append(norm_decoded(noisy_dec))

                if not is_multi_gpu:
                    self._vae_decoder = self._vae_decoder.to("cpu")
                torch.cuda.empty_cache()

                # Build grid columns
                gt_col = torch.cat(gt_frames, dim=1)
                comp_col = torch.cat(comp_frames, dim=1)

                if masked_frames:
                    # EditCtrl: 4 columns — GT | Mask Overlay | Masked Input | Composited
                    mask_col = torch.cat(masked_frames, dim=1)

                    # Build mask overlay: GT with red tint on masked region
                    if edit_mask is not None:
                        mask_overlay_frames = []
                        mask_3d = edit_mask[0].cpu().reshape(f, h, w)  # [F, H, W]
                        for fidx in frame_indices:
                            gt_frame = gt_frames[frame_indices.index(fidx)]  # [3, H_pix, W_pix]
                            frame_mask = mask_3d[fidx]  # [H_lat, W_lat]
                            # Upsample mask to pixel resolution
                            mask_up = torch.nn.functional.interpolate(
                                frame_mask.float().unsqueeze(0).unsqueeze(0),
                                size=gt_frame.shape[1:],
                                mode="nearest",
                            ).squeeze()  # [H_pix, W_pix]
                            # Red tint: boost red channel, dim green+blue on masked region
                            overlay = gt_frame.clone()
                            overlay[0] = torch.where(mask_up > 0.5, torch.clamp(overlay[0] * 0.5 + 0.5, 0, 1), overlay[0])
                            overlay[1] = torch.where(mask_up > 0.5, overlay[1] * 0.3, overlay[1])
                            overlay[2] = torch.where(mask_up > 0.5, overlay[2] * 0.3, overlay[2])
                            mask_overlay_frames.append(overlay)
                        overlay_col = torch.cat(mask_overlay_frames, dim=1)
                        decoded_grid = torch.cat([gt_col, overlay_col, mask_col, comp_col], dim=2)
                    else:
                        decoded_grid = torch.cat([gt_col, mask_col, comp_col], dim=2)
                else:
                    # Vanilla SCD: 2 columns — GT | Prediction
                    decoded_grid = torch.cat([gt_col, comp_col], dim=2)

                save_image(decoded_grid, out_dir / "debug_decoded.png")

        except Exception as e:
            logger.debug(f"SCD debug image save failed: {e}")

    def _save_debug_image(
        self,
        video_pred: Tensor,
        model_inputs: "ModelInputs",
        step: int,
    ) -> None:
        """Save a debug reconstruction image (overwrites each step).

        For I2V mode creates 4-image grid: Ref Image | Ref Video | Ground Truth | Prediction
        For non-I2V creates 3-image grid: Reference | Target | Prediction
        Saves to outputs/debug_decoded.png
        """
        try:
            # Check for SCD strategy debug image
            raw_video_latents = getattr(model_inputs, "_raw_video_latents", None)
            if raw_video_latents is not None:
                self._save_scd_debug_image(video_pred, model_inputs, step)
                return

            # Check for strategy-specific model inputs with raw latents
            if not hasattr(model_inputs, "ref_latent_raw"):
                return

            # Get raw latents
            ref_lat = model_inputs.ref_latent_raw  # [B, C, F, H, W]
            tgt_lat = model_inputs.tgt_latent_raw
            tgt_noisy = model_inputs.tgt_latent_noisy
            noise = model_inputs.noise
            sigmas = model_inputs.sigmas  # [B] or [B, 1, ...]

            # Check for I2V mode (first_frame_latent exists)
            first_frame_lat = getattr(model_inputs, 'first_frame_latent_raw', None)
            is_i2v_mode = first_frame_lat is not None

            # Get sigma value
            if sigmas.dim() > 1:
                sigma = sigmas.flatten()[0].item()
            else:
                sigma = sigmas[0].item()

            b, c, f, h, w = tgt_lat.shape

            # Take middle frame, first 3 channels as pseudo-RGB
            mid_f = f // 2
            ref_mid_f = min(mid_f, ref_lat.shape[2] - 1)  # Clamp for single-frame refs
            ref_vis = ref_lat[0, :3, ref_mid_f, :, :].cpu().float()  # [3, H, W]
            tgt_vis = tgt_lat[0, :3, mid_f, :, :].cpu().float()

            # For I2V mode, get first frame latent visualization
            if is_i2v_mode:
                # first_frame_lat shape: [B, C, 1, H, W]
                first_frame_vis = first_frame_lat[0, :3, 0, :, :].cpu().float()  # [3, H, W]

            # Reconstruct prediction from velocity prediction
            # v = noise - clean, so clean = noise - v
            # But we have noisy = clean + sigma * noise
            # So pred_clean = noisy - sigma * pred_v (approximately)
            try:
                from ltx_core.types import VideoLatentShape

                # Get prediction for target only (skip reference tokens)
                ref_seq_len = model_inputs.ref_seq_len
                tgt_pred_patched = video_pred[:, ref_seq_len:, :]  # [B, tgt_seq_len, C]

                # Create proper VideoLatentShape for unpatchify
                output_shape = VideoLatentShape(
                    batch=b,
                    channels=c,
                    frames=f,
                    height=h,
                    width=w
                )

                # Unpatchify to get back latent shape
                tgt_pred_latent = self._training_strategy._video_patchifier.unpatchify(
                    tgt_pred_patched,
                    output_shape=output_shape
                )  # [B, C, F, H, W]

                # Reconstruct clean from noisy using velocity
                # pred_clean = tgt_noisy - sigma * tgt_pred_latent
                sigma_view = sigmas.view(b, 1, 1, 1, 1)
                pred_clean = tgt_noisy - sigma_view * tgt_pred_latent

                pred_vis = pred_clean[0, :3, mid_f, :, :].cpu().float()
            except Exception as e:
                logger.debug(f"Prediction reconstruction failed: {e}")
                # Fallback: use noisy target
                pred_vis = tgt_noisy[0, :3, mid_f, :, :].cpu().float()

            # Normalize to [0, 1]
            def norm(x):
                x = x - x.min()
                x = x / (x.max() + 1e-8)
                return x.clamp(0, 1)

            ref_norm = norm(ref_vis)
            tgt_norm = norm(tgt_vis)
            pred_norm = norm(pred_vis)

            # Create grid based on mode
            if is_i2v_mode:
                first_frame_norm = norm(first_frame_vis)
                # 4-image grid: Ref Image | Ref Video | Ground Truth | Prediction
                grid = torch.cat([first_frame_norm, ref_norm, tgt_norm, pred_norm], dim=2)
            else:
                # 3-image grid: ref | target | prediction
                grid = torch.cat([ref_norm, tgt_norm, pred_norm], dim=2)

            # Save latent visualization
            out_path = Path(self._config.output_dir) / "debug_recon.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(grid, out_path)

            # Also save a text file with current step/sigma
            info_path = Path(self._config.output_dir) / "debug_info.txt"
            with open(info_path, "w") as info_file:
                info_file.write(f"Step: {step}\nSigma: {sigma:.4f}\nI2V Mode: {is_i2v_mode}\n")

            # Decode actual frames every step (move VAE to configured device temporarily)
            if self._vae_decoder is not None:
                try:
                    # Get VAE device from config (multi-GPU support)
                    hw_devices = self._config.hardware.devices
                    is_multi_gpu = hw_devices.transformer != hw_devices.vae_decoder
                    vae_device = hw_devices.vae_decoder

                    # Only move VAE if single-GPU (it stays on its device in multi-GPU)
                    if not is_multi_gpu:
                        self._vae_decoder = self._vae_decoder.to(vae_device)

                    # Get VAE dtype for consistency
                    vae_dtype = next(self._vae_decoder.parameters()).dtype

                    # Normalize: VAE outputs [-1, 1] -> [0, 1]
                    def norm_decoded(x):
                        return ((x.float() + 1) / 2).clamp(0, 1)

                    # Select 4 evenly spaced frame indices for multi-frame comparison
                    num_display_frames = min(4, f)
                    if num_display_frames > 1:
                        frame_indices = [int(i * (f - 1) / (num_display_frames - 1)) for i in range(num_display_frames)]
                    else:
                        frame_indices = [mid_f]

                    # Decode reference video middle frame (clamp for single-frame refs)
                    ref_frame_lat = ref_lat[0:1, :, ref_mid_f:ref_mid_f+1, :, :].to(vae_device, dtype=vae_dtype)

                    # For I2V mode, decode first frame (reference image)
                    if is_i2v_mode:
                        first_frame_decode_lat = first_frame_lat[0:1, :, 0:1, :, :].to(vae_device, dtype=vae_dtype)

                    # Decode multiple GT and Pred frames
                    gt_frames_decoded = []
                    pred_frames_decoded = []

                    with torch.inference_mode():
                        # Decode ref video frame
                        ref_decoded = self._vae_decoder(ref_frame_lat)[0, :, 0].cpu()

                        # Decode first frame if I2V
                        if is_i2v_mode:
                            first_frame_decoded = self._vae_decoder(first_frame_decode_lat)[0, :, 0].cpu()

                        # Decode multiple frames from GT and Prediction
                        for fidx in frame_indices:
                            # GT frame
                            tgt_frame_lat = tgt_lat[0:1, :, fidx:fidx+1, :, :].to(vae_device, dtype=vae_dtype)
                            gt_dec = self._vae_decoder(tgt_frame_lat)[0, :, 0].cpu()
                            gt_frames_decoded.append(norm_decoded(gt_dec))

                            # Prediction frame
                            try:
                                pred_frame_lat = pred_clean[0:1, :, fidx:fidx+1, :, :].to(vae_device, dtype=vae_dtype)
                            except Exception:
                                pred_frame_lat = tgt_noisy[0:1, :, fidx:fidx+1, :, :].to(vae_device, dtype=vae_dtype)
                            pred_dec = self._vae_decoder(pred_frame_lat)[0, :, 0].cpu()
                            pred_frames_decoded.append(norm_decoded(pred_dec))

                    # Move VAE back to CPU only in single-GPU mode
                    if not is_multi_gpu:
                        self._vae_decoder = self._vae_decoder.to("cpu")
                    torch.cuda.empty_cache()

                    # Create PORTRAIT grid (taller than wide) with 2 columns:
                    # Column 1 (GT): Ref Image, then GT frames stacked vertically
                    # Column 2 (Pred): Ref Video, then Pred frames stacked vertically
                    # This gives aspect ratio ~1.35 portrait

                    if is_i2v_mode:
                        ref_img_norm = norm_decoded(first_frame_decoded)
                        ref_vid_norm = norm_decoded(ref_decoded)

                        # Left column: Ref Image + GT frames
                        left_col = torch.cat([ref_img_norm] + gt_frames_decoded, dim=1)

                        # Right column: Ref Video + Pred frames
                        right_col = torch.cat([ref_vid_norm] + pred_frames_decoded, dim=1)

                        # Stack columns horizontally -> portrait image
                        decoded_grid = torch.cat([left_col, right_col], dim=2)  # [3, H*(1+num_frames), W*2]
                    else:
                        # Non-I2V: GT column | Pred column
                        ref_vid_norm = norm_decoded(ref_decoded)

                        # Left: Ref + GT frames
                        left_col = torch.cat([ref_vid_norm] + gt_frames_decoded, dim=1)

                        # Right: Padding + Pred frames
                        padding = torch.zeros_like(ref_vid_norm)
                        right_col = torch.cat([padding] + pred_frames_decoded, dim=1)

                        decoded_grid = torch.cat([left_col, right_col], dim=2)

                    save_image(decoded_grid, Path(self._config.output_dir) / "debug_decoded.png")
                except Exception as e:
                    logger.debug(f"Decoded frame save failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # Ensure VAE back on CPU even on error (only in single-GPU mode)
                    if hasattr(self, '_vae_decoder') and self._vae_decoder is not None:
                        hw_devices = self._config.hardware.devices
                        if hw_devices.transformer == hw_devices.vae_decoder:
                            self._vae_decoder = self._vae_decoder.to("cpu")

        except Exception as e:
            # Silent fail - debug only
            logger.debug(f"Debug image save failed: {e}")
