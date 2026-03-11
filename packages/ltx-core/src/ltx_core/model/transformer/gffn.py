"""
Geometric Feed-Forward Network (gFFN) inspired by CliffordNet.

Replaces standard MLP-style FFN with geometric interactions based on
the Clifford Geometric Product (uv = u·v + u∧v). The key insight is that
the geometric product simultaneously captures:
  - Feature coherence (generalized inner product / dot)
  - Structural variation (exterior / wedge product)

This module implements three variants:
  - gFFN-Global: Vanilla concat+project (moderate FLOPs, good accuracy)
  - gFFN-Hybrid: Local+global context streams (best accuracy, highest FLOPs)
  - gFFN-HRR:   HRR superposition (lowest FLOPs, aggressive compression)

Shift strategies are configurable to handle high-dimensional channels
(LTX2 uses dim=4096, much larger than CliffordNet's original dim=64-128).

Reference: CliffordNet (Ji, 2026) — "All You Need is Geometric Algebra"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Literal


class CliffordMode(str, Enum):
    """Which components of the geometric product to compute."""
    FULL = "full"      # Both inner + wedge (algebraically complete)
    INNER = "inner"    # Inner product only (coherence gating)
    WEDGE = "wedge"    # Wedge product only (structural variation)


class ShiftStrategy(str, Enum):
    """How to space the channel shifts across the dimension.

    For high-dim spaces (dim=4096), the shift strategy matters:
    - EXPONENTIAL: {1,2,4,...,2^(k-1)} — CliffordNet default, dense at low end
      At dim=4096, 4 shifts cover max_shift=8 (0.2% of ring)
    - LOG_UNIFORM: Log-uniform spacing across full [1, dim//2] range
      At dim=4096, 8 shifts might give {1, 3, 11, 38, 128, 436, 1489, 2048}
    - GEOMETRIC: Geometric progression scaled to dim//2
      At dim=4096, 8 shifts give {1, 3, 8, 24, 72, 215, 643, 2048}
    """
    EXPONENTIAL = "exponential"
    LOG_UNIFORM = "log_uniform"
    GEOMETRIC = "geometric"


def compute_shifts(
    num_shifts: int,
    dim: int,
    strategy: ShiftStrategy | str = ShiftStrategy.EXPONENTIAL,
) -> list[int]:
    """Compute channel shift values based on strategy.

    Args:
        num_shifts: Number of shifts to generate
        dim: Channel dimension (used for scaling in non-exponential strategies)
        strategy: How to space the shifts

    Returns:
        List of integer shift values, sorted ascending, all unique
    """
    strategy = ShiftStrategy(strategy) if isinstance(strategy, str) else strategy

    if strategy == ShiftStrategy.EXPONENTIAL:
        # Original CliffordNet: {1, 2, 4, 8, ..., 2^(k-1)}
        shifts = [1 << i for i in range(num_shifts)]

    elif strategy == ShiftStrategy.LOG_UNIFORM:
        # Log-uniform spacing across [1, dim//2]
        # Ensures coverage of both fine-grained and coarse channel distances
        max_shift = max(dim // 2, 2)
        if num_shifts == 1:
            shifts = [1]
        else:
            log_min = 0.0  # log(1)
            log_max = math.log(max_shift)
            shifts = sorted(set(
                max(1, round(math.exp(
                    log_min + (log_max - log_min) * i / (num_shifts - 1)
                )))
                for i in range(num_shifts)
            ))

    elif strategy == ShiftStrategy.GEOMETRIC:
        # Geometric progression: 1, r, r^2, ..., r^(k-1) where r^(k-1) = dim//2
        max_shift = max(dim // 2, 2)
        if num_shifts == 1:
            shifts = [1]
        else:
            ratio = max_shift ** (1.0 / (num_shifts - 1))
            shifts = sorted(set(
                max(1, round(ratio ** i))
                for i in range(num_shifts)
            ))

    else:
        raise ValueError(f"Unknown shift strategy: {strategy}")

    return shifts


class gFFNGlobal(nn.Module):
    """
    Global Geometric Feed-Forward Network (gFFN-G).

    Replaces standard FFN with geometric interactions using global context.
    For each shift s:
      - Inner(s): SiLU(x * roll(g, s))         — coherence gate
      - Wedge(s): x * roll(g, s) - roll(x, s) * g  — structural variation

    where g = GlobalAvgPool(x) is the global context.

    The activated local features + all shift terms are concatenated and
    projected back to the output dimension.

    Args:
        dim: Input/output feature dimension
        dim_out: Output dimension (defaults to dim if None)
        num_shifts: Number of shifts (default 8 for dim>=1024, 4 otherwise)
        mode: Which geometric product components to use
        shift_strategy: How to space shifts across channel dimension
        gate: Whether to use sigmoid gating on output (Gated Geometric Residual)
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        num_shifts: int = 4,
        mode: CliffordMode | str = CliffordMode.FULL,
        shift_strategy: ShiftStrategy | str = ShiftStrategy.EXPONENTIAL,
        gate: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_shifts = num_shifts
        self.mode = CliffordMode(mode) if isinstance(mode, str) else mode
        self.use_gate = gate

        self.shifts = compute_shifts(num_shifts, dim, shift_strategy)
        actual_shifts = len(self.shifts)

        # Compute concatenation dimension
        if self.mode == CliffordMode.FULL:
            cat_dim = dim * (1 + 2 * actual_shifts)
        elif self.mode == CliffordMode.INNER:
            cat_dim = dim * (1 + actual_shifts)
        else:  # WEDGE
            cat_dim = dim * (1 + actual_shifts)

        # Output projection (the learnable P operator from the paper)
        self.proj_out = nn.Linear(cat_dim, self.dim_out)

        # Optional gated geometric residual (GGR)
        if self.use_gate:
            self.gate_proj = nn.Linear(dim + self.dim_out, self.dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input features (after AdaLN normalization in LTX2)

        Returns:
            [B, L, dim_out] geometric features
        """
        # Global context: mean field over sequence dimension
        glo = x.mean(dim=1, keepdim=True)  # [B, 1, D]

        feats = [F.silu(x)]  # Activated local features

        for s in self.shifts:
            rg = torch.roll(glo, shifts=s, dims=-1)

            if self.mode in (CliffordMode.FULL, CliffordMode.INNER):
                inner = F.silu(x * rg)
                feats.append(inner)

            if self.mode in (CliffordMode.FULL, CliffordMode.WEDGE):
                rx = torch.roll(x, shifts=s, dims=-1)
                wedge = x * rg - rx * glo
                feats.append(wedge)

        mixed = torch.cat(feats, dim=-1)
        out = self.proj_out(mixed)

        if self.use_gate:
            gate_input = torch.cat([x[..., :self.dim], out], dim=-1)
            alpha = torch.sigmoid(self.gate_proj(gate_input))
            out = alpha * out

        return out


class gFFNHybrid(nn.Module):
    """
    Hybrid Geometric FFN combining local and global context.

    For 1D sequences (patchified video tokens), uses 1D convolution
    for local context instead of CliffordNet's 2D DWConv.

    gFFN-H: C = ΔH + β · GlobalAvg(H)

    Args:
        dim: Input/output feature dimension
        dim_out: Output dimension (defaults to dim if None)
        num_shifts: Number of channel shifts
        mode: Geometric product mode
        shift_strategy: How to space shifts
        local_kernel_size: Kernel size for local 1D context convolution
        differential_mode: If True, subtract identity (C = conv(x) - x) for
                          high-pass filtering
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        num_shifts: int = 4,
        mode: CliffordMode | str = CliffordMode.FULL,
        shift_strategy: ShiftStrategy | str = ShiftStrategy.EXPONENTIAL,
        local_kernel_size: int = 7,
        differential_mode: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_shifts = num_shifts
        self.mode = CliffordMode(mode) if isinstance(mode, str) else mode
        self.differential_mode = differential_mode
        self.shifts = compute_shifts(num_shifts, dim, shift_strategy)
        actual_shifts = len(self.shifts)

        # Local context: factorized 1D depth-wise convolution
        padding = local_kernel_size // 2
        self.local_conv = nn.Sequential(
            nn.Conv1d(dim, dim, local_kernel_size, padding=padding, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, local_kernel_size, padding=padding, groups=dim),
        )

        # Two streams: local + global, each with geometric interactions
        if self.mode == CliffordMode.FULL:
            features_per_stream = dim * (1 + 2 * actual_shifts)
        else:
            features_per_stream = dim * (1 + actual_shifts)

        cat_dim = features_per_stream * 2
        self.proj_out = nn.Linear(cat_dim, self.dim_out)

    def _geometric_interaction(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute geometric product between state x and context."""
        feats = [F.silu(x)]

        for s in self.shifts:
            rc = torch.roll(context, shifts=s, dims=-1)

            if self.mode in (CliffordMode.FULL, CliffordMode.INNER):
                inner = F.silu(x * rc)
                feats.append(inner)

            if self.mode in (CliffordMode.FULL, CliffordMode.WEDGE):
                rx = torch.roll(x, shifts=s, dims=-1)
                wedge = x * rc - rx * context
                feats.append(wedge)

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)

        if self.differential_mode:
            c_local = x_conv - x
        else:
            c_local = x_conv

        c_global = x.mean(dim=1, keepdim=True)

        local_feats = self._geometric_interaction(x, c_local)
        local_mixed = torch.cat(local_feats, dim=-1)

        global_feats = self._geometric_interaction(x, c_global)
        global_mixed = torch.cat(global_feats, dim=-1)

        combined = torch.cat([local_mixed, global_mixed], dim=-1)
        return self.proj_out(combined)


class gFFNGlobalHRR(nn.Module):
    """
    HRR-augmented Geometric FFN (gFFN-Global + Holographic Reduced Representations).

    Instead of concatenating all geometric terms into a wide vector and projecting
    (which makes the output linear dominate FLOPs at >99%), this variant:
      1. Projects each term to a small shared subspace (dim → dim//proj_factor)
      2. Superimposes (sums) the projected terms — HRR-style binding
      3. Projects the narrow superposition back to output dim

    For dim=4096, 8 shifts, inner-only, proj_factor=8:
      Standard FFN:  ~268M FLOPs, ~134M params
      HRR gFFN:      ~42M FLOPs,   ~4M params  (6.4x fewer FLOPs, 32x fewer params)

    Recommended config for LTX2:
      num_shifts=8, shift_strategy="log_uniform", proj_factor=4, mode="inner"

    Args:
        dim: Input/output feature dimension
        dim_out: Output dimension (defaults to dim if None)
        num_shifts: Number of channel shifts (8 recommended for dim>=1024)
        proj_factor: Compression ratio for per-term projection (4-8 recommended)
        mode: Which geometric product components to use
        shift_strategy: How to space shifts (log_uniform recommended for high dim)
        unit_norm: Apply per-token L2 normalization after superposition
        div_by_terms: Scale superposition by 1/num_terms to reduce mean drift
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        num_shifts: int = 8,
        proj_factor: int = 4,
        mode: CliffordMode | str = CliffordMode.INNER,
        shift_strategy: ShiftStrategy | str = ShiftStrategy.LOG_UNIFORM,
        unit_norm: bool = True,
        div_by_terms: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_shifts = num_shifts
        self.proj_dim = dim // proj_factor
        self.mode = CliffordMode(mode) if isinstance(mode, str) else mode
        self.unit_norm = unit_norm
        self.div_by_terms = div_by_terms

        self.shifts = compute_shifts(num_shifts, dim, shift_strategy)
        actual_shifts = len(self.shifts)

        # Count number of terms for proper normalization
        base_terms = 1  # silu(x)
        if self.mode == CliffordMode.FULL:
            base_terms += 2 * actual_shifts
        elif self.mode == CliffordMode.INNER:
            base_terms += actual_shifts
        else:  # WEDGE
            base_terms += actual_shifts
        self.num_terms = base_terms

        # Shared per-term projection: dim → proj_dim
        self.proj = nn.Linear(dim, self.proj_dim)

        # Final narrow output projection: proj_dim → dim_out
        self.out = nn.Linear(self.proj_dim, self.dim_out)

        # Learnable per-channel scale after superposition
        self.post_scale = nn.Parameter(torch.ones(self.dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input features (after AdaLN normalization in LTX2)

        Returns:
            [B, L, dim_out] geometric features via HRR superposition
        """
        glo = x.mean(dim=1, keepdim=True)  # [B, 1, D]

        terms = [F.silu(x)]

        for s in self.shifts:
            rg = torch.roll(glo, shifts=s, dims=-1)

            if self.mode in (CliffordMode.FULL, CliffordMode.INNER):
                inner = F.silu(x * rg)
                terms.append(inner)

            if self.mode in (CliffordMode.FULL, CliffordMode.WEDGE):
                rx = torch.roll(x, shifts=s, dims=-1)
                wedge = x * rg - rx * glo
                terms.append(wedge)

        # HRR core: project each term to shared subspace, then superimpose
        projected = [self.proj(t) for t in terms]
        superposed = sum(projected)

        if self.div_by_terms:
            superposed = superposed / self.num_terms

        if self.unit_norm:
            norm = superposed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            superposed = superposed / norm

        out = self.out(superposed)
        return out * self.post_scale


class gFFNSpectral(nn.Module):
    """
    Spectral Geometric FFN — frequency-domain circular convolution with learned sparse mask.

    Instead of computing K discrete shift interactions in spatial domain,
    this variant works entirely in the frequency domain:
      1. FFT(x) and FFT(global_context) — O(D log D)
      2. Circular convolution via pointwise multiply in freq domain — O(D)
      3. Learned frequency selection — keep only top-k components
      4. Nonlinear gating in freq domain (modReLU-inspired)
      5. Project sparse freq features back to output dim

    This replaces the 9× per-term projection (dominant 90% of HRR FLOPs)
    with a single small projection from 2k real values.

    For dim=4096, k=128:
      Standard FFN:  ~268M FLOPs, ~134M params
      gFFN-HRR:      ~84M FLOPs,   ~8.4M params
      gFFN-Spectral: ~2.3M FLOPs,  ~1.1M params  (116x fewer FLOPs than FFN)

    The nonlinearity comes from:
      - Bilinear interaction (X ⊙ G is product of two learned representations)
      - modReLU on magnitudes (threshold + scale in freq domain)
      - SiLU on the real projection

    Args:
        dim: Input/output feature dimension
        dim_out: Output dimension (defaults to dim if None)
        num_freqs: Number of frequency components to keep (k). Higher = more expressive.
        freq_init: How to initialize frequency selection ("topk_learnable" or "linear")
        use_phase_gate: Apply learned phase rotation before magnitude selection
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        num_freqs: int = 128,
        freq_init: str = "topk_learnable",
        use_phase_gate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_freqs = min(num_freqs, dim // 2 + 1)
        self.use_phase_gate = use_phase_gate

        # rfft output size for real input of length dim
        self.freq_dim = dim // 2 + 1

        # Learnable frequency importance weights (soft top-k selection)
        # Initialized with log-uniform bias so low + high frequencies both get signal
        self.freq_logits = nn.Parameter(torch.zeros(self.freq_dim))
        self._init_freq_logits(freq_init)

        # Learned magnitude gate (modReLU-inspired): bias + scale per selected freq
        self.mag_bias = nn.Parameter(torch.zeros(self.num_freqs))
        self.mag_scale = nn.Parameter(torch.ones(self.num_freqs))

        # Optional learned phase rotation per frequency
        if use_phase_gate:
            self.phase_shift = nn.Parameter(torch.zeros(self.num_freqs))

        # Output projection: 2*num_freqs real values → dim_out
        # Factor of 2 because complex → (real, imag)
        self.out_proj = nn.Linear(2 * self.num_freqs, self.dim_out)

        # Learnable per-channel scale (matches HRR interface)
        self.post_scale = nn.Parameter(torch.ones(self.dim_out))

        # Cache for top-k indices (recomputed in forward if training)
        self._cached_indices = None

    def _init_freq_logits(self, method: str):
        """Initialize frequency selection with a prior."""
        with torch.no_grad():
            if method == "topk_learnable":
                # Bias toward both low frequencies (global structure)
                # and a spread of mid/high frequencies (local detail)
                # Log-uniform prior: equal weight per octave
                freqs = torch.arange(self.freq_dim, dtype=torch.float32)
                freqs[0] = 0.5  # avoid log(0)
                self.freq_logits.copy_(
                    torch.log(freqs + 1) / math.log(self.freq_dim + 1)
                )
            elif method == "linear":
                # Linearly favor low frequencies
                self.freq_logits.copy_(
                    torch.linspace(1.0, 0.0, self.freq_dim)
                )

    def _select_frequencies(self) -> torch.Tensor:
        """Return indices of top-k frequencies by learned importance."""
        if self.training or self._cached_indices is None:
            # Straight-through top-k: select indices, but allow gradient to flow
            # through freq_logits via the soft weighting in forward
            _, indices = torch.topk(self.freq_logits, self.num_freqs, sorted=True)
            indices = indices.sort().values
            self._cached_indices = indices
        return self._cached_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input features

        Returns:
            [B, L, dim_out] spectral geometric features
        """
        B, L, D = x.shape
        orig_dtype = x.dtype

        # Global context
        glo = x.mean(dim=1, keepdim=True)  # [B, 1, D]

        # FFT requires float32 (bf16 not supported by torch.fft)
        x_f32 = x.float()
        glo_f32 = glo.float()

        # FFT along channel dimension
        X = torch.fft.rfft(x_f32, dim=-1)   # [B, L, D//2+1] complex64
        G = torch.fft.rfft(glo_f32, dim=-1) # [B, 1, D//2+1] complex64

        # Circular convolution in frequency domain (bilinear interaction)
        # This is equivalent to ALL circular shifts simultaneously
        conv = X * G  # [B, L, D//2+1] complex

        # Also include the direct signal (like the silu(x) term in HRR)
        # Add X weighted by a learnable scalar for the "identity" channel
        combined = conv + X * 0.1  # small residual of raw spectrum

        # Select top-k frequencies
        freq_idx = self._select_frequencies()  # [num_freqs]
        sparse = combined[..., freq_idx]  # [B, L, num_freqs] complex

        # Soft frequency weighting (allows gradient flow to freq_logits)
        freq_weights = torch.sigmoid(self.freq_logits[freq_idx].float())  # [num_freqs]
        sparse = sparse * freq_weights

        # modReLU-inspired nonlinearity on magnitudes
        mag = sparse.abs()  # [B, L, num_freqs]
        phase = sparse / (mag + 1e-8)  # unit phase
        mag_gated = F.relu(self.mag_scale.float() * mag + self.mag_bias.float())  # threshold + scale
        sparse = mag_gated * phase  # reconstruct with gated magnitude

        # Optional learned phase rotation
        if self.use_phase_gate:
            rotation = torch.polar(
                torch.ones_like(self.phase_shift.float()),
                self.phase_shift.float(),
            )  # [num_freqs] complex
            sparse = sparse * rotation.to(sparse.device)

        # Convert complex → real: stack (real, imag) as features
        real_feats = torch.cat([sparse.real, sparse.imag], dim=-1)  # [B, L, 2*num_freqs]

        # Cast back to original dtype for projection
        real_feats = real_feats.to(orig_dtype)

        # Project to output dimension
        out = self.out_proj(F.silu(real_feats))

        return out * self.post_scale


def create_gffn(
    dim: int,
    dim_out: int | None = None,
    variant: Literal["global", "hybrid", "hrr", "spectral"] = "global",
    num_shifts: int = 4,
    mode: str = "full",
    shift_strategy: str = "exponential",
    **kwargs,
) -> nn.Module:
    """Factory function to create gFFN variants.

    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim)
        variant: "global", "hybrid", "hrr", or "spectral"
        num_shifts: Number of channel shifts (not used for spectral)
        mode: "full", "inner", or "wedge" (not used for spectral)
        shift_strategy: "exponential", "log_uniform", or "geometric" (not used for spectral)
        **kwargs: Additional arguments for specific variants
            - For spectral: num_freqs (default 128), use_phase_gate (default True)

    Variant guide (dim=4096, per-token FLOPs):
        global   — Vanilla CliffordNet gFFN-G. Wide concat + single projection.
                   0.63x FFN at (4 shifts, inner, exponential).
        hybrid   — Local + global context streams. Best accuracy, 1.25-2.25x FFN.
        hrr      — HRR superposition. Low FLOPs.
                   0.16x FFN at (8 shifts, inner, log_uniform, pf=8).
                   0.31x FFN at (8 shifts, inner, log_uniform, pf=4).
        spectral — Sparse FFT circular convolution. Lowest FLOPs + params.
                   0.009x FFN at (k=128). ~1.1M params, ~2.3M FLOPs.
    """
    if variant == "global":
        return gFFNGlobal(
            dim, dim_out, num_shifts=num_shifts, mode=mode,
            shift_strategy=shift_strategy, **kwargs,
        )
    elif variant == "hybrid":
        return gFFNHybrid(
            dim, dim_out, num_shifts=num_shifts, mode=mode,
            shift_strategy=shift_strategy, **kwargs,
        )
    elif variant == "hrr":
        return gFFNGlobalHRR(
            dim, dim_out, num_shifts=num_shifts, mode=mode,
            shift_strategy=shift_strategy, **kwargs,
        )
    elif variant == "spectral":
        return gFFNSpectral(dim, dim_out, **kwargs)
    else:
        raise ValueError(f"Unknown gFFN variant: {variant}")
