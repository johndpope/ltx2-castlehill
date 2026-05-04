"""IsoGen-MotionBricks training strategy — isometric video with MotionBricks modules.

Extends SCDTrainingStrategy with:
  1. Multi-Head VQ-VAE tokenizer for structured latent quantization
  2. Unified keyframe interface (T1/camera, T2/scene, T3/object)
  3. Smart Camera (spring model + neural refinement)
  4. Smart Scene Object (keyframe-based interactions)

Two-stage pretraining:
  Stage 1 (0-500 steps): Train VQ-VAE only, freeze everything else
  Stage 2 (500+ steps): Full training, LoRA for DiT
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_core.model.video_vae.multi_head_vq import MultiHeadVQConfig, MultiHeadVQVideoVAE
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.keyframe_interface import (
    KeyframeConditioningEncoder,
    KeyframeConstraint,
    KeyframeMaskGenerator,
)
from ltx_trainer.training_strategies.scd_strategy import (
    SCDTrainingConfig,
    SCDTrainingStrategy,
)
from ltx_trainer.training_strategies.smart_primitives import SmartCameraModule, SmartSceneObjectModule

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MotionBricksTrainingConfig(SCDTrainingConfig):
    """Configuration for MotionBricks-integrated isometric video training."""

    name: Literal["motionbricks"] = "motionbricks"

    # === Multi-Head VQ-VAE ===
    vq_num_heads: int = Field(default=4, description="Number of VQ codebooks")
    vq_head_dims: list[int] = Field(default_factory=lambda: [32, 32, 32, 32])
    vq_codebook_sizes: list[int] = Field(default_factory=lambda: [4096, 4096, 4096, 4096])
    vq_commitment_cost: float = Field(default=0.25, ge=0.0)
    vq_loss_weight: float = Field(default=1.0, ge=0.0)
    vq_use_ema: bool = Field(default=True)
    vq_use_root_disentanglement: bool = Field(default=True)
    vq_root_dim: int = Field(default=4)

    # === Root/View Module ===
    root_module_hidden_dim: int = Field(default=512)
    root_module_num_layers: int = Field(default=2)
    root_learning_rate: float = Field(default=1e-4)

    # === Smart Camera ===
    camera_spring_omega: float = Field(default=2.0, ge=0.1, le=10.0)
    camera_neural_refiner: bool = Field(default=True)
    camera_refiner_hidden_dim: int = Field(default=256)
    camera_loss_weight: float = Field(default=0.1, ge=0.0)

    # === Smart Scene Object ===
    scene_object_num: int = Field(default=16)
    scene_object_hidden_dim: int = Field(default=128)
    scene_loss_weight: float = Field(default=0.05, ge=0.0)

    # === Keyframe Interface ===
    max_keyframes: int = Field(default=10, ge=1, le=10)
    keyframe_hidden_dim: int = Field(default=1024)
    keyframe_dropout_p: float = Field(default=0.1, ge=0.0, le=1.0)
    keyframe_mask_min: int = Field(default=0)
    keyframe_mask_max: int = Field(default=10)
    keyframe_loss_weight: float = Field(default=0.1, ge=0.0)

    # === Data Sources ===
    memory_dir: str = Field(default="memory")
    keyframe_dir: str = Field(default="keyframes")
    camera_traj_dir: str = Field(default="camera_trajectories")

    # === Pretraining ===
    vq_pretrain_steps: int = Field(default=500)
    root_loss_weight: float = Field(default=0.1, ge=0.0)
    t3_pose_dim: int = Field(default=10)
    t3_action_dim: int = Field(default=16)


class MotionBricksTrainingStrategy(SCDTrainingStrategy):
    """MotionBricks training strategy extending SCD."""

    config: MotionBricksTrainingConfig

    def __init__(self, config: MotionBricksTrainingConfig):
        super().__init__(config)
        self._vq_vae: Optional[MultiHeadVQVideoVAE] = None
        self._keyframe_encoder: Optional[KeyframeConditioningEncoder] = None
        self._keyframe_mask_gen: Optional[KeyframeMaskGenerator] = None
        self._camera_module: Optional[SmartCameraModule] = None
        self._scene_module: Optional[SmartSceneObjectModule] = None
        self._root_adapter: Optional[torch.nn.Module] = None
        self._vq_config: Optional[MultiHeadVQConfig] = None

    # ── Module Factory Methods ──────────────────────────────────────────────

    def create_vq_vae(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> MultiHeadVQVideoVAE:
        cfg = self.config
        self._vq_config = MultiHeadVQConfig(
            num_heads=cfg.vq_num_heads, head_dims=cfg.vq_head_dims,
            codebook_sizes=cfg.vq_codebook_sizes, codebook_dim=cfg.vq_head_dims[0],
            commitment_cost=cfg.vq_commitment_cost, use_ema=cfg.vq_use_ema,
            use_root_disentanglement=cfg.vq_use_root_disentanglement, root_dim=cfg.vq_root_dim,
        )
        self._vq_vae = MultiHeadVQVideoVAE(encoder=encoder, decoder=decoder, config=self._vq_config)
        logger.info(f"Created VQ-VAE: {cfg.vq_num_heads} heads, codebooks={cfg.vq_codebook_sizes}")
        return self._vq_vae

    def create_keyframe_encoder(self, cross_attention_dim: int = 4096) -> KeyframeConditioningEncoder:
        cfg = self.config
        self._keyframe_encoder = KeyframeConditioningEncoder(
            cross_attention_dim=cross_attention_dim, max_keyframes=cfg.max_keyframes,
            hidden_dim=cfg.keyframe_hidden_dim, t1_dim=4, t2_dim=128,
            t3_pose_dim=cfg.t3_pose_dim, t3_action_dim=cfg.t3_action_dim,
        )
        self._keyframe_mask_gen = KeyframeMaskGenerator(
            mask_min=cfg.keyframe_mask_min, mask_max=cfg.keyframe_mask_max, mask_dropout_p=cfg.keyframe_dropout_p,
        )
        logger.info(f"Created KeyframeEncoder: {cfg.max_keyframes} max keyframes")
        return self._keyframe_encoder

    def create_camera_module(self) -> SmartCameraModule:
        cfg = self.config
        self._camera_module = SmartCameraModule(
            omega=cfg.camera_spring_omega, neural_refiner=cfg.camera_neural_refiner,
            refiner_hidden_dim=cfg.camera_refiner_hidden_dim,
        )
        logger.info(f"Created SmartCamera: omega={cfg.camera_spring_omega}")
        return self._camera_module

    def create_scene_module(self) -> SmartSceneObjectModule:
        cfg = self.config
        self._scene_module = SmartSceneObjectModule(num_objects=cfg.scene_object_num, hidden_dim=cfg.scene_object_hidden_dim)
        logger.info(f"Created SmartScene: {cfg.scene_object_num} objects")
        return self._scene_module

    # ── Data Sources ────────────────────────────────────────────────────────

    def get_data_sources(self) -> dict[str, str]:
        sources = {"latents": "latents", "conditions": "conditions"}
        sources[self.config.memory_dir] = self.config.memory_dir
        sources[self.config.keyframe_dir] = self.config.keyframe_dir
        sources[self.config.camera_traj_dir] = self.config.camera_traj_dir
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    # ── Core Training Logic ─────────────────────────────────────────────────

    def prepare_training_inputs(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> ModelInputs:
        latents = batch["latents"]
        video_latents = latents["latents"]
        conditions = batch["conditions"]
        is_vq_pretrain = self.config.vq_pretrain_steps > 0 and self._current_step < self.config.vq_pretrain_steps

        if self._vq_vae is not None and self._vq_vae.encoder is not None:
            z_q, vq_metrics = self._vq_vae.encode(video_latents)
            self._current_vq_metrics = vq_metrics
            video_latents_data = z_q
        else:
            video_latents_data = video_latents
            self._current_vq_metrics = {}
            self._current_vq_z = None

        kf_tokens = None
        if self._keyframe_encoder is not None:
            kf_tokens = self._process_keyframes(batch, video_latents.shape[2])

        camera_traj = None
        if self._camera_module is not None:
            camera_traj = self._process_camera(batch, video_latents.shape[2])
            self._current_camera_traj = camera_traj
        else:
            self._current_camera_traj = None

        if kf_tokens is not None:
            video_prompt_embeds = conditions["video_prompt_embeds"]
            prompt_attention_mask = conditions["prompt_attention_mask"]
            B = video_prompt_embeds.shape[0]
            kf_padding = torch.ones(B, kf_tokens.shape[1], device=kf_tokens.device, dtype=prompt_attention_mask.dtype)
            conditions["video_prompt_embeds"] = torch.cat([video_prompt_embeds, kf_tokens], dim=1)
            conditions["prompt_attention_mask"] = torch.cat([prompt_attention_mask, kf_padding], dim=1)

        if is_vq_pretrain and self._vq_vae is not None:
            return self._prepare_vq_pretrain_inputs(video_latents, video_latents_data, batch, timestep_sampler)

        original_latents = batch["latents"]["latents"]
        batch["latents"]["latents"] = video_latents_data
        self._current_vq_z = original_latents
        try:
            model_inputs = super().prepare_training_inputs(batch, timestep_sampler)
        finally:
            batch["latents"]["latents"] = original_latents
        return model_inputs

    def _process_keyframes(self, batch: dict[str, Any], num_frames: int) -> Optional[Tensor]:
        if self._keyframe_encoder is None or self._keyframe_mask_gen is None:
            return None
        device = self._keyframe_encoder.mask_embed.device
        B = batch["latents"]["latents"].shape[0]
        keyframe_dir = batch.get(self.config.keyframe_dir, {})
        has_precomputed = "keyframes" in keyframe_dir

        if has_precomputed:
            kf_data = keyframe_dir["keyframes"]
            all_kf_batches = []
            for b in range(B):
                kf_list = []
                for k in range(kf_data.get("frame_indices", torch.zeros(B, 0)).shape[1]):
                    fi = kf_data["frame_indices"][b, k].item()
                    if "camera_traj" in kf_data:
                        kf_list.append(KeyframeConstraint(frame_idx=fi, level="T1", data=kf_data["camera_traj"][b, k].to(device), hardness=0.0))
                    if "scene_latents" in kf_data:
                        kf_list.append(KeyframeConstraint(frame_idx=fi, level="T2", data=kf_data["scene_latents"][b, k].to(device), hardness=0.0))
                    if "object_poses" in kf_data and "object_actions" in kf_data:
                        obj_data = torch.cat([kf_data["object_poses"][b, k].float(), kf_data["object_actions"][b, k].float().unsqueeze(-1)], dim=-1).to(device)
                        kf_list.append(KeyframeConstraint(frame_idx=fi, level="T3", data=obj_data, hardness=0.0))
                all_kf_batches.append(kf_list if kf_list else [])
            if any(kfs for kfs in all_kf_batches):
                return self._keyframe_encoder(all_kf_batches)

        memory_dir = batch.get(self.config.memory_dir, {})
        if "geo_cond" in memory_dir:
            geo_cond = memory_dir["geo_cond"]
            poses = memory_dir.get("poses", None)
            actions = memory_dir.get("actions", None)
            stride = max(1, num_frames // (self.config.max_keyframes + 1))
            all_kf_batches = []
            for b in range(B):
                kf_list = []
                for k in range(0, num_frames, stride):
                    if len(kf_list) >= self.config.max_keyframes: break
                    fi = min(k, num_frames - 1)
                    scene_data = geo_cond[b, fi].reshape(-1).to(device)
                    if scene_data.shape[0] > 128: scene_data = scene_data[:128]
                    elif scene_data.shape[0] < 128: scene_data = F.pad(scene_data, (0, 128 - scene_data.shape[0]))
                    kf_list.append(KeyframeConstraint(frame_idx=fi, level="T2", data=scene_data, hardness=0.0))
                    if poses is not None:
                        pose_data = poses[b, fi].to(device)
                        if actions is not None:
                            pose_data = torch.cat([pose_data, actions[b, fi].float().to(device).unsqueeze(0)])
                        else:
                            pose_data = F.pad(pose_data, (0, 1))
                        kf_list.append(KeyframeConstraint(frame_idx=fi, level="T3", data=pose_data, hardness=0.0))
                all_kf_batches.append(kf_list)
            return self._keyframe_encoder(all_kf_batches)
        return None

    def _process_camera(self, batch: dict[str, Any], num_frames: int) -> Optional[Tensor]:
        if self._camera_module is None:
            return None
        device = next(self._camera_module.parameters()).device
        B = batch["latents"]["latents"].shape[0]
        cam_dir = batch.get(self.config.camera_traj_dir, {})
        if "camera_trajectories" in cam_dir:
            return cam_dir["camera_trajectories"].to(device=device, dtype=torch.float32)
        camera_trajs = []
        for b in range(B):
            kp = torch.tensor([[0.0, 0.0, 0.0, 0.785]], device=device).unsqueeze(0)
            traj = self._camera_module(keypoints=kp, num_frames=num_frames)
            camera_trajs.append(traj)
        return torch.cat(camera_trajs, dim=0)

    def _prepare_vq_pretrain_inputs(self, video_latents, video_latents_data, batch, timestep_sampler) -> ModelInputs:
        conditions = batch["conditions"]
        B = video_latents.shape[0]
        device = video_latents.device
        dtype = video_latents.dtype
        video_noise = torch.randn_like(video_latents)
        video_targets = video_noise - video_latents_data
        video_loss_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)
        video_timesteps = torch.zeros(B, 1, device=device, dtype=dtype)
        video_modality = type('Obj', (), {
            'enabled': True, 'sigma': torch.zeros(B, device=device, dtype=dtype),
            'latent': video_latents_data, 'timesteps': video_timesteps,
            'positions': None, 'context': conditions['video_prompt_embeds'],
            'context_mask': conditions['prompt_attention_mask'],
        })
        model_inputs = ModelInputs(
            video=video_modality, audio=None,
            video_targets=video_targets, audio_targets=None,
            video_loss_mask=video_loss_mask, audio_loss_mask=None,
            shared_noise=video_noise,
            shared_sigmas=torch.zeros(B, device=device, dtype=dtype),
        )
        model_inputs._vq_pretrain = True
        model_inputs._raw_video_latents = video_latents
        return model_inputs

    def compute_loss(self, video_pred, audio_pred, inputs) -> Tensor:
        total_loss = torch.tensor(0.0, device=video_pred.device)
        if not getattr(inputs, "_vq_pretrain", False):
            scd_loss = super().compute_loss(video_pred, audio_pred, inputs)
            total_loss = total_loss + scd_loss
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"loss/scd": scd_loss.item(), "step": self._current_step})

        vq_loss_val = torch.tensor(0.0, device=video_pred.device)
        if hasattr(self, '_current_vq_metrics') and self._current_vq_metrics:
            commit_loss = self._current_vq_metrics.get("commitment_loss", 0.0)
            vq_loss_val = torch.tensor(commit_loss, device=video_pred.device)
            total_loss = total_loss + self.config.vq_loss_weight * vq_loss_val
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"loss/vq": vq_loss_val.item(), "vq/perplexity": self._current_vq_metrics.get("perplexity", 0.0), "vq/codebook_usage": self._current_vq_metrics.get("codebook_usage", 0.0), "step": self._current_step})

        root_loss_val = torch.tensor(0.0, device=video_pred.device)
        if hasattr(self, '_current_vq_metrics') and self._current_vq_metrics.get("root_pred") is not None:
            root_pred = self._current_vq_metrics["root_pred"]
            root_loss_val = root_pred.norm(dim=-1).mean() * 0.01
            total_loss = total_loss + self.config.root_loss_weight * root_loss_val
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"loss/root": root_loss_val.item(), "step": self._current_step})

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({"loss/total": total_loss.item(), "step": self._current_step})
        return total_loss

    def get_trainable_parameters(self) -> list[dict]:
        params = []
        if hasattr(self.config, 'optimization') and hasattr(self.config.optimization, 'learning_rate'):
            base_lr = self.config.optimization.learning_rate
        else:
            base_lr = 1e-4
        if self._vq_vae is not None:
            params.append({"params": [p for p in self._vq_vae.vq.parameters()], "lr": base_lr * 2.0, "name": "vq_vae"})
        if self._keyframe_encoder is not None:
            params.append({"params": self._keyframe_encoder.parameters(), "lr": base_lr * 2.0, "name": "keyframe_encoder"})
        if self._camera_module is not None and self._camera_module.neural_refiner is not None:
            params.append({"params": self._camera_module.neural_refiner.parameters(), "lr": base_lr * 2.0, "name": "camera_refiner"})
        if hasattr(self, '_root_adapter') and self._root_adapter is not None:
            params.append({"params": self._root_adapter.parameters(), "lr": base_lr, "name": "root_adapter"})
        return params

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        meta = super().get_checkpoint_metadata()
        meta.update({"motionbricks_vq_heads": self.config.vq_num_heads, "motionbricks_vq_codebook_sizes": self.config.vq_codebook_sizes})
        return meta
