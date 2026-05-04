"""Smart Primitives for MotionBricks-style isometric video control.

Implements MotionBricks Section 6:
  1. Smart Camera (§6.1): Critically damped spring + neural refinement
  2. Smart Scene Object (§6.2): Keyframe-based interactions
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 1. SMART CAMERA
# ============================================================================

class CriticallyDampedSpring(nn.Module):
    """Critically damped spring model for smooth camera trajectories."""

    def __init__(self, omega: float = 2.0, dt: float = 1.0 / 24.0):
        super().__init__()
        self.omega = omega
        self.dt = dt
        self.gamma = omega

    def forward(self, keypoints: Tensor, num_frames: int, initial_velocity: Optional[Tensor] = None) -> Tensor:
        B, K, D = keypoints.shape
        device = keypoints.device
        dtype = keypoints.dtype
        if initial_velocity is None:
            initial_velocity = torch.zeros(B, D, device=device, dtype=dtype)

        dense_target = torch.zeros(B, num_frames, D, device=device, dtype=dtype)
        for b in range(B):
            for d in range(D):
                dense_target[b, :, d] = F.interpolate(
                    keypoints[b:b+1, :, d:d+1].permute(0, 2, 1),
                    size=num_frames, mode='linear', align_corners=True,
                ).squeeze(0).squeeze(0)

        gamma = self.gamma
        dt = self.dt
        trajectory = torch.zeros(B, num_frames, D, device=device, dtype=dtype)
        r_curr = torch.zeros(B, D, device=device, dtype=dtype)
        v_curr = initial_velocity.clone()

        for t in range(num_frames):
            r_target = dense_target[:, t, :]
            t_sec = t * dt
            exp_factor = math.exp(-gamma * t_sec)
            r_t = (exp_factor * (r_curr - r_target) + (v_curr + gamma * (r_curr - r_target)) * t_sec + r_target)
            trajectory[:, t, :] = r_t
            r_curr = r_t

        if num_frames >= 6:
            for b in range(B):
                for d in range(D):
                    for i in range(3):
                        alpha = (i + 1) / 3.0
                        trajectory[b, i, d] = trajectory[b, i, d] * alpha + keypoints[b, 0, d].item() * (1 - alpha)
        return trajectory


class NeuralCameraRefiner(nn.Module):
    """Lightweight MLP that refines the spring-smoothed camera trajectory."""

    def __init__(self, hidden_dim: int = 256, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 4), nn.Tanh(),
        )
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.mul_(0.01)

    def forward(self, trajectory: Tensor) -> Tensor:
        B, F, D = trajectory.shape
        velocity = torch.zeros_like(trajectory)
        velocity[:, 1:, :] = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        inp = torch.cat([trajectory, velocity], dim=-1)
        inp_flat = inp.reshape(-1, 8)
        residual = self.net(inp_flat).reshape(B, F, 4) * 0.05
        return trajectory + residual


class SmartCameraModule(nn.Module):
    """Complete Smart Camera: spring smooth + neural refinement."""

    def __init__(self, omega: float = 2.0, dt: float = 1.0/24.0, neural_refiner: bool = True, refiner_hidden_dim: int = 256):
        super().__init__()
        self.spring = CriticallyDampedSpring(omega=omega, dt=dt)
        self.neural_refiner = NeuralCameraRefiner(hidden_dim=refiner_hidden_dim) if neural_refiner else None
        self.omega = omega
        self.dt = dt

    def forward(self, keypoints: Tensor, num_frames: int, apply_refinement: bool = True) -> Tensor:
        trajectory = self.spring(keypoints, num_frames)
        if apply_refinement and self.neural_refiner is not None:
            trajectory = self.neural_refiner(trajectory)
        return trajectory


# ============================================================================
# 2. SMART SCENE OBJECT
# ============================================================================

class InteractionBinding(nn.Module):
    """Maps object keyframes to scene embeddings with hardness modulation."""

    def __init__(self, num_objects: int = 16, pose_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.num_objects = num_objects
        self.hidden_dim = hidden_dim
        self.object_embed = nn.Embedding(num_objects, hidden_dim)
        self.pose_encoder = nn.Sequential(nn.Linear(pose_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.hardness_encoder = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, object_poses: Tensor, object_ids: Tensor, hardness: Optional[Tensor] = None) -> Tensor:
        pose_feats = self.pose_encoder(object_poses)
        obj_feats = self.object_embed(object_ids)
        combined = self.fusion(torch.cat([pose_feats, obj_feats], dim=-1))
        if hardness is not None:
            h_scale = self.hardness_encoder(hardness.unsqueeze(-1))
            combined = combined * h_scale
        return combined


class KeyframeAnchor(nn.Module):
    """Anchors object keyframes to specific frames with interpolation."""

    def __init__(self, num_objects: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.num_objects = num_objects
        self.hidden_dim = hidden_dim
        self.interp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, object_tokens: Tensor, anchor_indices: Tensor, num_frames: int) -> Tensor:
        B, K, N_obj, H = object_tokens.shape
        device = object_tokens.device
        output = torch.zeros(B, num_frames, N_obj, H, device=device)
        for b in range(B):
            for k in range(K):
                idx = anchor_indices[b, k].item()
                if 0 <= idx < num_frames:
                    output[b, idx] = object_tokens[b, k]
            valid_anchors = [(anchor_indices[b, k].item(), k) for k in range(K) if 0 <= anchor_indices[b, k].item() < num_frames]
            valid_anchors.sort()
            if len(valid_anchors) >= 2:
                for n in range(num_frames):
                    if not any(a[0] == n for a in valid_anchors):
                        prev_idx = next_idx = -1
                        for a_idx, a_k in valid_anchors:
                            if a_idx <= n:
                                prev_idx, prev_k = a_idx, a_k
                            if a_idx >= n and next_idx < 0:
                                next_idx, next_k = a_idx, a_k
                        if prev_idx >= 0 and next_idx >= 0 and prev_idx != next_idx:
                            alpha = (n - prev_idx) / (next_idx - prev_idx)
                            prev_tok = object_tokens[b, prev_k]
                            next_tok = object_tokens[b, next_k]
                            weight = torch.tensor([alpha], device=device)
                            interp_input = torch.cat([(1-alpha)*prev_tok + alpha*next_tok, weight.expand(N_obj, 1)], dim=-1)
                            output[b, n] = self.interp(interp_input)
        return output


class SmartSceneObjectModule(nn.Module):
    """Combines InteractionBinding + KeyframeAnchor for object-level conditioning."""

    def __init__(self, num_objects: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.binding = InteractionBinding(num_objects=num_objects, hidden_dim=hidden_dim)
        self.anchor = KeyframeAnchor(num_objects=num_objects, hidden_dim=hidden_dim)

    def forward(self, object_poses: Tensor, object_ids: Tensor, anchor_indices: Tensor, num_frames: int, hardness: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, K, N_obj, pose_dim = object_poses.shape
        all_tokens = []
        for k in range(K):
            bound = self.binding(object_poses[:, k], object_ids[:, k], hardness[:, k] if hardness is not None else None)
            all_tokens.append(bound.unsqueeze(1))
        bound_tokens = torch.cat(all_tokens, dim=1)
        per_frame = self.anchor(bound_tokens, anchor_indices, num_frames)
        scene_embedding = per_frame.mean(dim=[1, 2], keepdim=True)
        return per_frame, scene_embedding
