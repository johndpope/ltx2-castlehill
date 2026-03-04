"""Monotonic B-spline sigma scheduler for flow-matching models.

Replaces BézierFlow's global Bernstein basis with local B-spline basis functions.
Each coefficient only affects a window of k+1 knot spans (k=3 for cubic), enabling
independent tuning of early-step (structure) vs late-step (detail) phases.

Monotonicity is enforced by construction: unconstrained θ → softmax → cumsum gives
monotonically increasing coefficients, and a monotone B-spline curve requires
monotone coefficients (Schoenberg's theorem).

LTX-2 convention: x(t) = (1-σ)·x_clean + σ·noise, so σ(s) = 1 - B(s) where
B(s) is our monotonically increasing spline from 0 to 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class BSplineScheduler(nn.Module):
    """Learnable monotonic sigma schedule via B-spline with cumulative-softmax coefficients.

    Uses a uniform clamped B-spline of order k (degree k-1) with n_coefficients
    interior control values. Boundary coefficients are pinned at 0 and 1.

    Compared to BézierFlow:
    - LOCAL support: each coefficient affects only k+1 neighbouring knot spans
    - SAME monotonicity guarantee via cumulative softmax
    - SAME interface: get_sigma_schedule(), save(), load()
    - Similar parameter count (~32 floats)

    Args:
        n_coefficients: Number of interior B-spline coefficients (total = n+2 with endpoints).
        order: B-spline order (degree = order-1). Default 4 = cubic B-spline.
    """

    def __init__(self, n_coefficients: int = 32, order: int = 4) -> None:
        super().__init__()
        self.n_coefficients = n_coefficients
        self.order = order  # k (degree = k-1)
        # Unconstrained parameters — monotonicity enforced by cumulative softmax
        self.theta = nn.Parameter(torch.zeros(n_coefficients))

        # Total coefficients including pinned endpoints
        n_total = n_coefficients + 2  # c_0=0, c_1..c_n, c_{n+1}=1

        # Build clamped uniform knot vector
        # Clamped: k copies of 0 at start, k copies of 1 at end
        # Interior: n_total - k + 1 uniformly spaced values (includes endpoints)
        n_interior = n_total - order + 2  # number of distinct knot values
        interior_knots = torch.linspace(0.0, 1.0, n_interior)
        knots = torch.cat([
            torch.zeros(order - 1),  # clamped start
            interior_knots,
            torch.ones(order - 1),   # clamped end
        ])
        self.register_buffer("knots", knots, persistent=True)

    @property
    def coefficients(self) -> Tensor:
        """Sorted coefficients in [0, 1] via softplus increments.

        Unlike BézierFlow's cumulative softmax (which couples ALL params through
        softmax normalization), we use softplus for independent positive increments:
          Δ_i = softplus(θ_i)   — each Δ depends on only ONE θ
          c_i = cumsum(Δ)_i / sum(Δ)   — normalized to [0, 1]

        This preserves B-spline locality: changing θ_i primarily affects c_i and
        its neighbours (through basis overlap), not distant coefficients.

        Returns shape [n_coefficients + 2] with endpoints pinned at 0 and 1.
        """
        deltas = torch.nn.functional.softplus(self.theta)  # [n], each > 0
        raw_cumsum = torch.cumsum(deltas, dim=0)            # monotonically increasing
        total = raw_cumsum[-1]                               # normalization constant
        interior = raw_cumsum / total                        # [n], in (0, 1)
        zeros = torch.zeros(1, device=self.theta.device, dtype=self.theta.dtype)
        ones = torch.ones(1, device=self.theta.device, dtype=self.theta.dtype)
        return torch.cat([zeros, interior, ones])  # [n+2]

    def _bspline_basis(self, s: Tensor, i: int, k: int) -> Tensor:
        """Evaluate B-spline basis function N_{i,k}(s) via Cox-de Boor recursion.

        Uses the numerically stable form with safe division (0/0 = 0 convention).

        Args:
            s: Evaluation points, any shape.
            i: Basis function index.
            k: Order (1 = piecewise constant, 4 = cubic).

        Returns:
            N_{i,k}(s), same shape as s.
        """
        knots = self.knots

        if k == 1:
            # Base case: indicator function for [t_i, t_{i+1})
            # Special case for last span: include right endpoint
            left = knots[i]
            right = knots[i + 1]
            if i + 1 == len(knots) - 1:
                return ((s >= left) & (s <= right)).float()
            return ((s >= left) & (s < right)).float()

        # Recursive case
        left = self._bspline_basis(s, i, k - 1)
        right = self._bspline_basis(s, i + 1, k - 1)

        # Safe division: d1 = (s - t_i) / (t_{i+k-1} - t_i), 0 if denominator is 0
        denom1 = knots[i + k - 1] - knots[i]
        w1 = torch.where(
            denom1.abs() > 1e-10,
            (s - knots[i]) / denom1,
            torch.zeros_like(s),
        )

        denom2 = knots[i + k] - knots[i + 1]
        w2 = torch.where(
            denom2.abs() > 1e-10,
            (knots[i + k] - s) / denom2,
            torch.zeros_like(s),
        )

        return w1 * left + w2 * right

    def _evaluate_basis_matrix(self, s: Tensor) -> Tensor:
        """Evaluate all B-spline basis functions at once.

        More efficient than calling _bspline_basis per coefficient —
        uses vectorized de Boor's algorithm.

        Args:
            s: Evaluation points [M].

        Returns:
            Basis matrix [M, n_coefficients + 2].
        """
        n_total = self.n_coefficients + 2
        k = self.order
        knots = self.knots
        M = s.shape[0]

        # Initialize order-1 (piecewise constant) basis
        n_spans = len(knots) - 1
        # N_{i,1}(s) for each span
        basis = torch.zeros(M, n_spans, device=s.device, dtype=s.dtype)
        for i in range(n_spans):
            left = knots[i]
            right = knots[i + 1]
            if i == n_spans - 1:
                basis[:, i] = ((s >= left) & (s <= right)).float()
            else:
                basis[:, i] = ((s >= left) & (s < right)).float()

        # Build up through orders 2..k via recurrence
        for p in range(2, k + 1):
            new_basis = torch.zeros(M, n_spans - p + 1, device=s.device, dtype=s.dtype)
            for i in range(n_spans - p + 1):
                # Left term
                denom1 = knots[i + p - 1] - knots[i]
                if denom1.abs() > 1e-10:
                    w1 = (s - knots[i]) / denom1
                else:
                    w1 = torch.zeros_like(s)

                # Right term
                denom2 = knots[i + p] - knots[i + 1]
                if denom2.abs() > 1e-10:
                    w2 = (knots[i + p] - s) / denom2
                else:
                    w2 = torch.zeros_like(s)

                left_val = basis[:, i] if i < basis.shape[1] else torch.zeros(M, device=s.device)
                right_val = basis[:, i + 1] if (i + 1) < basis.shape[1] else torch.zeros(M, device=s.device)

                new_basis[:, i] = w1 * left_val + w2 * right_val

            basis = new_basis

        return basis[:, :n_total]  # [M, n_total]

    def evaluate(self, s: Tensor) -> Tensor:
        """Evaluate B-spline curve B(s) = Σ c_i · N_{i,k}(s).

        Args:
            s: Evaluation points in [0, 1], any shape.

        Returns:
            B(s) in [0, 1], same shape as s. Monotonically increasing.
        """
        original_shape = s.shape
        s_flat = s.reshape(-1)

        # Clamp to valid range
        s_clamped = s_flat.clamp(0.0, 1.0)

        # Get coefficients and basis
        coeffs = self.coefficients  # [n+2]
        basis = self._evaluate_basis_matrix(s_clamped)  # [M, n+2]

        # Weighted sum
        result = (basis * coeffs.unsqueeze(0)).sum(dim=-1)

        # Pin endpoints exactly
        result = torch.where(s_clamped <= 1e-7, torch.zeros_like(result), result)
        result = torch.where(s_clamped >= 1.0 - 1e-7, torch.ones_like(result), result)

        return result.reshape(original_shape)

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
            "type": "bspline",
            "n_coefficients": self.n_coefficients,
            "order": self.order,
            "sigma_schedule_4": self.get_sigma_schedule(4).detach().cpu().tolist(),
            "sigma_schedule_8": self.get_sigma_schedule(8).detach().cpu().tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> BSplineScheduler:
        """Load scheduler from .pt file, reading config from companion .json."""
        path = Path(path)
        config_path = path.with_suffix(".json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            n_coeff = config["n_coefficients"]
            order = config.get("order", 4)
        else:
            sd = torch.load(path, map_location=device, weights_only=True)
            n_coeff = sd["theta"].shape[0]
            order = 4
        scheduler = cls(n_coefficients=n_coeff, order=order)
        scheduler.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        return scheduler.to(device)
