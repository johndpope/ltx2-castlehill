"""Monotonic Bézier curve sigma scheduler for flow-matching models.

Based on BézierFlow (ICLR 2026, arXiv:2512.13255): learns optimal denoising
trajectory via Bézier reparameterization. Only 32 learnable parameters.

LTX-2 uses x(t) = (1-σ)·x_clean + σ·noise, so we learn a monotonically
INCREASING function B(s): [0,1] → [0,1] and define σ(s) = 1 - B(s),
yielding a monotonically DECREASING sigma schedule from 1.0 to 0.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class BezierScheduler(nn.Module):
    """Learnable monotonic sigma schedule via cumulative-softmax Bézier curve.

    Args:
        n_control_points: Number of interior control points (total degree = n+1).
            More points = more expressive schedule. Paper uses 32.
    """

    def __init__(self, n_control_points: int = 32) -> None:
        super().__init__()
        self.n_control_points = n_control_points
        # Unconstrained parameters — monotonicity enforced by cumulative softmax
        self.theta = nn.Parameter(torch.zeros(n_control_points))
        # Precompute log-binomial coefficients for Bernstein basis (degree n+1)
        degree = n_control_points + 1
        self.register_buffer(
            "_log_binom",
            self._compute_log_binom(degree),
            persistent=False,
        )

    @staticmethod
    def _compute_log_binom(n: int) -> Tensor:
        """Precompute log(binom(n, k)) for k=0..n using lgamma."""
        k = torch.arange(n + 1, dtype=torch.float64)
        log_binom = (
            torch.lgamma(torch.tensor(n + 1, dtype=torch.float64))
            - torch.lgamma(k + 1)
            - torch.lgamma(torch.tensor(n + 1, dtype=torch.float64) - k)
        )
        return log_binom.float()

    @property
    def control_points(self) -> Tensor:
        """Sorted control points in [0, 1] via cumulative softmax (Eq. 12).

        Returns shape [n_control_points + 2] with endpoints pinned at 0 and 1.
        """
        # softmax → probabilities summing to 1, cumsum → sorted values in (0, 1)
        interior = torch.cumsum(torch.softmax(self.theta, dim=0), dim=0)
        zeros = torch.zeros(1, device=self.theta.device, dtype=self.theta.dtype)
        ones = torch.ones(1, device=self.theta.device, dtype=self.theta.dtype)
        return torch.cat([zeros, interior, ones])  # [n+2]

    def evaluate(self, s: Tensor) -> Tensor:
        """Evaluate Bézier curve B(s) using Bernstein basis in log-space.

        Args:
            s: Evaluation points in [0, 1], any shape.

        Returns:
            B(s) in [0, 1], same shape as s. Monotonically increasing.
        """
        cp = self.control_points  # [n+2]
        degree = cp.shape[0] - 1  # n+1

        # Clamp to avoid log(0)
        s_clamped = s.clamp(1e-7, 1.0 - 1e-7)

        # Bernstein basis in log-space: log(binom(n,k)) + k*log(s) + (n-k)*log(1-s)
        k = torch.arange(degree + 1, device=s.device, dtype=s.dtype)  # [degree+1]
        log_binom = self._log_binom.to(device=s.device, dtype=s.dtype)

        # Expand for broadcasting: s is [...], k is [degree+1]
        log_s = torch.log(s_clamped).unsqueeze(-1)  # [..., 1]
        log_1ms = torch.log(1.0 - s_clamped).unsqueeze(-1)  # [..., 1]

        log_basis = log_binom + k * log_s + (degree - k) * log_1ms  # [..., degree+1]
        basis = torch.exp(log_basis)  # Bernstein basis polynomials

        # Weighted sum: B(s) = Σ basis_k * C_k
        result = (basis * cp).sum(dim=-1)

        # Pin endpoints exactly
        result = torch.where(s <= 1e-7, torch.zeros_like(result), result)
        result = torch.where(s >= 1.0 - 1e-7, torch.ones_like(result), result)
        return result

    def get_sigma_schedule(self, num_steps: int) -> Tensor:
        """Get sigma values for a given number of denoising steps.

        Args:
            num_steps: Number of denoising steps (N). Returns N+1 sigma values
                       from σ=1.0 (pure noise) to σ=0.0 (clean).

        Returns:
            Tensor of shape [num_steps + 1], monotonically decreasing.
        """
        s = torch.linspace(0.0, 1.0, num_steps + 1, device=self.theta.device)
        return 1.0 - self.evaluate(s)  # σ = 1 - B(s)

    def save(self, path: str | Path) -> None:
        """Save scheduler to .pt (weights) + .json (config)."""
        path = Path(path)
        torch.save(self.state_dict(), path)
        config_path = path.with_suffix(".json")
        config_path.write_text(json.dumps({
            "n_control_points": self.n_control_points,
            "sigma_schedule_4": self.get_sigma_schedule(4).detach().cpu().tolist(),
            "sigma_schedule_8": self.get_sigma_schedule(8).detach().cpu().tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> BezierScheduler:
        """Load scheduler from .pt file, reading config from companion .json."""
        path = Path(path)
        config_path = path.with_suffix(".json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            n_cp = config["n_control_points"]
        else:
            # Infer from state dict
            sd = torch.load(path, map_location=device, weights_only=True)
            n_cp = sd["theta"].shape[0]
        scheduler = cls(n_control_points=n_cp)
        scheduler.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        return scheduler.to(device)
