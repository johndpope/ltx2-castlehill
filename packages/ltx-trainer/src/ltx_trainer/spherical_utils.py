"""Spherical geometry utilities for VFM noise adapters.

Provides Spherical Cauchy sampling, KL divergence, SLERP, and geodesic operations
for use in VFM v1e (spherical noise adapter). Adapted from spherical-vae.

Key concepts:
- Spherical Cauchy: heavy-tailed distribution on S^(D-1), faster convergence than vMF
- SLERP: geodesically smooth interpolation on hypersphere
- Exp/Log maps: tangent space operations for reparameterization
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def normalize(v: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Project vectors to unit sphere."""
    return F.normalize(v, p=2, dim=dim, eps=eps)


def sample_spherical_cauchy(
    mu: Tensor,
    kappa: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Sample from Spherical Cauchy distribution on S^(D-1).

    Heavy-tailed distribution that enables broader exploration early in training
    and faster convergence to high-density regions vs Gaussian/vMF.

    Args:
        mu: Mean direction on sphere [..., D] (will be normalized)
        kappa: Concentration parameter [...] or [..., 1] (positive)
        eps: Numerical stability constant

    Returns:
        Samples on unit sphere [..., D]
    """
    mu = normalize(mu)
    dim = mu.shape[-1]
    device = mu.device
    leading_shape = mu.shape[:-1]

    # Ensure kappa has shape [..., 1]
    if kappa.shape[-1] != 1:
        kappa = kappa.unsqueeze(-1)

    # Inverse CDF for Cauchy: t = tan(π(u - 0.5))
    u = torch.rand(*leading_shape, 1, device=device)
    t = torch.tan(math.pi * (u - 0.5))

    # Scale by concentration: r = sqrt(kappa) * t
    r = torch.sqrt(kappa + eps) * t
    r = torch.clamp(r, -10.0, 10.0)  # Prevent numerical explosions

    # Random tangent vector orthogonal to mu via Gram-Schmidt
    noise = torch.randn_like(mu)
    dot_product = (noise * mu).sum(dim=-1, keepdim=True)
    tangent = noise - dot_product * mu
    tangent = normalize(tangent)

    # Exponential map: sample = mu * cos(r) + tangent * sin(r)
    sample = torch.cos(r) * mu + torch.sin(r) * tangent
    return normalize(sample)


def kl_spherical_cauchy_to_uniform(
    mu: Tensor,
    kappa: Tensor,
    dim: int,
    eps: float = 1e-8,
) -> Tensor:
    """KL divergence: KL(SphericalCauchy(mu, kappa) || Uniform(S^(D-1))).

    In our parametrization, higher κ = MORE spread (r = sqrt(κ) * Cauchy):
        κ → 0: concentrated at μ (delta-like) → KL large
        κ → ∞: spread out (uniform-like) → KL → 0

    Closed-form: KL ≈ (D-1)/2 * log(1 + 1/κ)

    Args:
        mu: Mean direction [..., D] (unused in formula but kept for API consistency)
        kappa: Concentration [...] or [..., 1]
        dim: Dimension D of the embedding space

    Returns:
        KL divergence [...], always ≥ 0
    """
    if kappa.dim() > 0 and kappa.shape[-1] == 1:
        kappa = kappa.squeeze(-1)

    # KL = (D-1)/2 * log((1+κ)/κ) = (D-1)/2 * log(1 + 1/κ)
    # Always ≥ 0: concentrated (small κ) → large KL, spread (large κ) → small KL
    kl = ((dim - 1) / 2.0) * torch.log(1.0 + 1.0 / (kappa + eps))
    return kl


def slerp(a: Tensor, b: Tensor, t: float | Tensor) -> Tensor:
    """Spherical Linear Interpolation on unit hypersphere.

    SLERP follows the geodesic (great circle) between a and b,
    producing geometrically smooth interpolation vs linear blend.

    Args:
        a: Start point on sphere [..., D]
        b: End point on sphere [..., D]
        t: Interpolation parameter (0 = a, 1 = b)

    Returns:
        Interpolated point on sphere [..., D]
    """
    a = normalize(a)
    b = normalize(b)

    dot = (a * b).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Handle parallel vectors (sin_theta ≈ 0) → use linear interp
    parallel_mask = sin_theta.abs() < 1e-6

    if isinstance(t, Tensor):
        t_expanded = t
        if t.dim() < a.dim():
            t_expanded = t.unsqueeze(-1)
    else:
        t_expanded = t

    w1 = torch.sin((1 - t_expanded) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(t_expanded * theta) / (sin_theta + 1e-8)

    result = w1 * a + w2 * b

    # Fallback to linear for near-parallel vectors
    if parallel_mask.any():
        linear_interp = (1 - t_expanded) * a + t_expanded * b
        result = torch.where(parallel_mask, linear_interp, result)

    return normalize(result)


def geodesic_distance(a: Tensor, b: Tensor) -> Tensor:
    """Geodesic (angular) distance on sphere in radians.

    Args:
        a, b: Points on sphere [..., D]

    Returns:
        Angular distance [...] in [0, π]
    """
    a = normalize(a)
    b = normalize(b)
    dot = (a * b).sum(dim=-1)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot)
