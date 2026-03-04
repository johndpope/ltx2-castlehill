"""DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers.

Paper: arXiv:2602.16968 — "DDiT: Dynamic Diffusion Transformer"
Paper Section 3: The core idea is that different denoising timesteps exhibit
different levels of spatial complexity. Early steps (high noise, coarse global
structure) can use fewer, larger patches without quality loss. Later steps
(low noise, fine details) need the full resolution. DDiT dynamically selects
the patch scale at each timestep, achieving up to 2.1x speedup on Wan-2.1 T2V
(Paper Table 2) and 3.2x when combined with TeaCache.

Paper Figure 2: Shows the overall merge → transformer → unmerge pipeline with
a residual bypass from input to output at coarse resolutions.

Adaptation: LTX-2 uses patch_size=1 (each VAE voxel = 1 token), so DDiT
merges 2x2 or 4x4 spatial neighbors POST-patchification rather than changing
the patchification kernel size as in the paper. The paper (Section 3.1) uses
patch sizes p, 2p, 4p applied at the VAE-to-token boundary with per-scale
projection weights w^{emb}_{p_new}. Our approach achieves the same token
reduction (s² factor) but operates on already-patchified tokens.

Adaptation: LTX-2's SCD (Split Causal DiT) mode means DDiT operates on only
the decoder portion (16 blocks out of 48 total), not all transformer blocks
as in the paper. This limits the computational savings to the decoder pass
but avoids interfering with the encoder's full-resolution feature extraction.

Usage at inference:
    ddit = DDiTAdapter(base_model, inner_dim=4096, in_channels=128)
    for t in timesteps:
        patch_scale = ddit.schedule(z_t, z_prev, z_prev2, t)
        z_merged = ddit.merge(z_t, patch_scale, spatial_shape)
        positions = ddit.adjust_positions(positions, patch_scale, spatial_shape)
        # ... run transformer with z_merged ...
        z_out = ddit.unmerge(output, patch_scale, spatial_shape)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DDiTConfig:
    """Configuration for DDiT dynamic patch scheduling.

    Paper Section 3: These parameters control the dynamic scheduling behavior.
    Paper Section 3.1: supported_scales corresponds to the paper's multi-resolution
    patchification with scales {p, 2p, 4p}. For scale s, the spatial token count
    reduces by s² (quadratic reduction).
    """

    enabled: bool = False

    # Paper Section 3.1: Supported patch scales. The paper uses {p, 2p, 4p} where p
    # is the base patch size. For LTX-2 with patch_size=1, this becomes {1, 2, 4}.
    # Adaptation: Since LTX-2's base patch_size=1, scale=2 means merging 2x2=4 tokens,
    # and scale=4 means merging 4x4=16 tokens, giving 4x and 16x sequence reduction.
    supported_scales: tuple[int, ...] = (1, 2, 4)

    # Paper Section 3.2, Algorithm 1: Scheduling parameters.
    # threshold (τ): If the ρ-th percentile of per-patch std of Δ³ is below τ,
    # the current scale is sufficient (spatial content is smooth enough for coarse patches).
    # Paper default: τ=0.001
    threshold: float = 0.001

    # Paper Section 3.2, Algorithm 1: ρ-th percentile used to summarize per-patch
    # spatial variance. Lower ρ = more aggressive coarsening (fewer patches survive
    # the threshold test). Paper default: ρ=0.4
    percentile: float = 0.4

    # Paper Section 3.2: First N steps always use fine patches (scale=1).
    # This is needed because the scheduler requires at least 3 latent history
    # states to compute 3rd-order finite differences (Δ³). Paper default: 3 steps.
    warmup_steps: int = 3

    # Paper Section 3.1.2: Residual connection strength (0 = no residual, 1 = full residual).
    # The paper initializes the residual block to zero output for stable training start,
    # then lets it learn refinements. This weight controls the residual bypass strength.
    residual_weight: float = 0.0

    # Paper Section 3.3: LoRA rank for adaptation layers. The paper trains LoRA on FFN
    # layers (net.0.proj and net.2) with rank 32. Setting to 0 disables LoRA and uses
    # raw projection matrices instead.
    lora_rank: int = 32


class DDiTMergeLayer(nn.Module):
    """Spatial token merge + projection for a specific patch scale.

    Paper Section 3.1 (Multi-resolution Patchification): Each scale s has its own
    patchify projection w^{emb}_{p_new} ∈ ℝ^{p²C × d} (Paper Eq. 1) that maps
    the merged channel dimension back to the model's hidden dimension d.

    For scale=2: merges 2x2 spatial neighbors → 4x input dim → project to inner_dim.
    For scale=4: merges 4x4 spatial neighbors → 16x input dim → project to inner_dim.

    Adaptation: The paper changes the patchification kernel size at the VAE boundary.
    Since LTX-2 already patchifies with patch_size=1, we instead merge s×s spatial
    neighbors in token space. The math is equivalent: both reduce spatial tokens by s²
    and concatenate the channel dimensions before projection.

    The merge operation reshapes [B, F, H, W, C] → [B, F, H/s, W/s, C*s*s]
    then projects back to inner_dim.
    """

    def __init__(self, scale: int, in_channels: int, inner_dim: int):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.merged_dim = in_channels * scale * scale

        # Paper Eq. (1): Per-scale patchify projection w^{emb}_{p_new} ∈ ℝ^{p²C × d}.
        # Maps the merged channel dimension (C * s²) to the model's inner dimension d.
        # Adaptation: The paper learns separate projection weights per scale. We do the
        # same but operate post-patchification rather than at the VAE boundary.
        self.patchify_proj = nn.Linear(self.merged_dim, inner_dim, bias=True)

        # Paper Figure 2: De-patchify projection (unmerge layer) maps transformer
        # output back from inner_dim to the merged channel dim for spatial unmerging.
        # This is the inverse of patchify_proj, appearing on the right side of Figure 2.
        self.proj_out = nn.Linear(inner_dim, self.merged_dim)

        # Paper Section 3.1.1 (Patch-size Identifier): A learnable d-dimensional
        # embedding e_s added to ALL tokens when operating at scale s. This lets the
        # transformer distinguish which resolution it's currently processing, since the
        # same transformer weights are shared across all scales. Initialized to zero
        # so the model starts with no scale-dependent bias.
        self.patch_id = nn.Parameter(torch.zeros(1, 1, inner_dim))

        # Paper Section 3.1.2 (Residual Connection): A lightweight residual block
        # (LayerNorm → Linear → GELU → Linear) connecting the input directly to the
        # output at coarse resolutions. This provides a "shortcut" for high-frequency
        # details that may be lost during spatial merging. The paper emphasizes that
        # both linear layers are initialized to zero for stable training start —
        # the residual block outputs zero initially and gradually learns refinements.
        # Paper Figure 2: This is the bypass arrow from input to output in the diagram.
        self.residual_block = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def init_from_base(self, base_patchify: nn.Linear, base_proj_out: nn.Linear) -> None:
        """Initialize from base model weights for near-identity behavior.

        Paper Section 3.3 (Training via Knowledge Distillation): The student DDiT
        model is initialized to behave as close to the frozen teacher (base model)
        as possible. For patchify projections, we tile the base model's weights
        across the merged dimension with 1/s² scaling, so that averaging s² tokens
        produces the same output as the base model processing a single token.

        This initialization ensures that at the start of distillation training,
        the DDiT model produces outputs identical to the teacher, and LoRA + the
        residual block gradually learn the corrections needed for each scale.

        Uses a tiling strategy: the base patchify_proj maps [C] → [D].
        For scale s, we map [C*s*s] → [D] by tiling the base weights s*s times
        with 1/(s*s) scaling, so averaging s*s tokens gives same result as base.
        """
        s2 = self.scale * self.scale
        with torch.no_grad():
            # Patchify: tile base weights across merged dimension.
            # Paper Section 3.3: Initialize student to match teacher output.
            # base_patchify.weight: [inner_dim, in_channels]
            # new weight: [inner_dim, in_channels * s * s]
            # The 1/s² scaling ensures that if all s² merged tokens are identical
            # (uniform patch), the output equals the base model's single-token output.
            base_w = base_patchify.weight.data  # [D, C]
            tiled_w = base_w.repeat(1, s2) / s2  # [D, C*s*s] — averaging
            self.patchify_proj.weight.data.copy_(tiled_w)
            if base_patchify.bias is not None:
                self.patchify_proj.bias.data.copy_(base_patchify.bias.data)

            # Proj_out: tile base weights to scatter back.
            # Each of the s² output positions gets a copy of the base projection weights.
            # base_proj_out.weight: [out_channels, inner_dim]
            # new weight: [out_channels * s * s, inner_dim]
            base_out_w = base_proj_out.weight.data  # [C, D]
            tiled_out_w = base_out_w.repeat(s2, 1)  # [C*s*s, D]
            self.proj_out.weight.data.copy_(tiled_out_w)
            if base_proj_out.bias is not None:
                tiled_out_b = base_proj_out.bias.data.repeat(s2)
                self.proj_out.bias.data.copy_(tiled_out_b)

            # Paper Section 3.1.2: Residual block initialized as zero output for
            # stable training start. Both linear layers have zero weights and biases
            # so the residual path contributes nothing initially. The block gradually
            # learns to refine high-frequency details lost during merging.
            nn.init.zeros_(self.residual_block[1].weight)
            nn.init.zeros_(self.residual_block[1].bias)
            nn.init.zeros_(self.residual_block[3].weight)
            nn.init.zeros_(self.residual_block[3].bias)

            # Paper Section 3.1.1: Patch-size identifier initialized to zero so the
            # model starts scale-agnostic and learns scale-specific adjustments.
            nn.init.zeros_(self.patch_id)

    def merge(self, latent: Tensor, num_frames: int, height: int, width: int) -> Tensor:
        """Merge spatial tokens into larger patches.

        Paper Section 3.1: For scale s, spatial tokens reduce by s² (quadratic).
        This implements the spatial merging step that precedes the per-scale
        projection w^{emb}_{p_new} from Paper Eq. (1).

        Adaptation: The paper merges at the patchification stage (changing kernel
        size). We merge already-patchified tokens by grouping s×s spatial neighbors
        and concatenating their channels. The token count reduction is identical:
        seq_len / s² tokens with s²·C channels each.

        Args:
            latent: [B, F*H*W, C] token sequence
            num_frames, height, width: spatial dimensions

        Returns:
            [B, F*(H/s)*(W/s), C*s*s] merged tokens
        """
        B, S, C = latent.shape
        s = self.scale

        # Reshape to spatial grid
        x = latent.view(B, num_frames, height, width, C)

        # Merge s×s spatial patches by reshaping and permuting.
        # [B, F, H, W, C] → [B, F, H/s, s, W/s, s, C] → [B, F, H/s, W/s, C*s*s]
        # This groups each s×s block of spatial tokens together, then concatenates
        # their channel dimensions, producing s²·C channels per merged token.
        x = x.view(B, num_frames, height // s, s, width // s, s, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # [B, F, H/s, W/s, s, s, C]
        x = x.view(B, num_frames, height // s, width // s, C * s * s)

        # Flatten back to sequence: F * (H/s) * (W/s) tokens
        new_seq_len = num_frames * (height // s) * (width // s)
        return x.view(B, new_seq_len, C * s * s)

    def unmerge(self, x: Tensor, num_frames: int, height: int, width: int) -> Tensor:
        """Scatter merged patches back to original spatial resolution.

        Paper Figure 2: This is the unmerge step on the right side of the diagram,
        after the transformer output has been projected back to C*s*s channels via
        proj_out. The merged tokens are spatially scattered to recover the original
        F*H*W token grid.

        Args:
            x: [B, F*(H/s)*(W/s), C*s*s] merged tokens (after proj_out)
            num_frames, height, width: ORIGINAL spatial dimensions

        Returns:
            [B, F*H*W, C] tokens at original resolution
        """
        B = x.shape[0]
        s = self.scale
        C = self.in_channels

        # Reshape to coarse spatial grid with merged channels.
        # Inverse of the merge operation: split the C*s*s channels back into
        # s×s spatial positions with C channels each.
        x = x.view(B, num_frames, height // s, width // s, s, s, C)

        # Scatter back: [B, F, H/s, W/s, s, s, C] → [B, F, H/s, s, W/s, s, C] → [B, F, H, W, C]
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, num_frames, height, width, C)

        # Flatten to sequence
        return x.view(B, num_frames * height * width, C)

    def forward_patchify(self, latent: Tensor, num_frames: int, height: int, width: int) -> Tensor:
        """Merge + project to inner_dim.

        Paper Section 3.1, Eq. (1): This combines the spatial merging with the
        per-scale projection w^{emb}_{p_new} and the patch-size identifier e_s.
        The full operation is: merge(x) → project(merged) → add patch_id(e_s).

        Paper Section 3.1.1: The patch-size identifier is added after projection
        so the transformer can distinguish which scale it's currently processing.

        Args:
            latent: [B, F*H*W, C] at original resolution

        Returns:
            [B, F*(H/s)*(W/s), inner_dim] ready for transformer
        """
        merged = self.merge(latent, num_frames, height, width)  # [B, seq_new, C*s*s]
        x = self.patchify_proj(merged)  # Paper Eq. (1): w^{emb}_{p_new} projection
        x = x + self.patch_id  # Paper Section 3.1.1: Add scale-specific identifier e_s
        return x

    def forward_unpatchify(
        self,
        x: Tensor,
        input_latent: Tensor,
        num_frames: int,
        height: int,
        width: int,
        residual_weight: float = 0.0,
    ) -> Tensor:
        """Project back + unmerge to original resolution.

        Paper Figure 2: This implements the right side of the diagram —
        project transformer output back to merged dim, spatially unmerge,
        and optionally add the residual bypass.

        Paper Section 3.1.2: The residual connection provides a direct path
        for high-frequency information from input to output, bypassing the
        lossy merge/unmerge cycle. The residual block (LayerNorm → Linear →
        GELU → Linear) learns to extract and refine details lost during merging.

        Args:
            x: [B, seq_new, inner_dim] transformer output
            input_latent: [B, F*H*W, C] original input (for residual bypass)
            num_frames, height, width: original spatial dims
            residual_weight: strength of residual connection (Paper Section 3.1.2)

        Returns:
            [B, F*H*W, C] at original resolution
        """
        merged_out = self.proj_out(x)  # [B, seq_new, C*s*s] — inverse of patchify_proj
        output = self.unmerge(merged_out, num_frames, height, width)  # [B, F*H*W, C]

        # Paper Section 3.1.2: Residual refinement. The residual block operates at
        # the ORIGINAL resolution (input_latent), preserving spatial detail that
        # was lost during the merge/transformer/unmerge cycle. Controlled by
        # residual_weight which acts as a mixing coefficient.
        if residual_weight > 0:
            residual = self.residual_block(input_latent)
            output = output + residual_weight * residual

        return output


class DDiTPatchScheduler(nn.Module):
    """Determines optimal patch scale per denoising timestep.

    Paper Section 3.2 (Dynamic Scheduling): Uses 3rd-order finite differences Δ³
    of the denoising trajectory z_t to measure spatial variance. The key insight
    is that Δ³ captures the "acceleration" of the denoising process — when the
    trajectory is smooth (low Δ³), spatial content is changing slowly and coarse
    patches suffice. When Δ³ is large, fine details are emerging and full
    resolution is needed.

    Paper Algorithm 1: For each scale from coarsest to finest:
        1. Compute per-patch std of Δ³ at scale s
        2. Take the ρ-th percentile of those std values
        3. If percentile < threshold τ → this scale is sufficient, use it
        4. Otherwise try the next finer scale
    Paper defaults: τ=0.001, ρ=0.4, warmup=3 steps.
    """

    def __init__(self, config: DDiTConfig):
        super().__init__()
        self.config = config
        self.supported_scales = config.supported_scales
        # Paper Section 3.2: Need at least 3 latent states (z_{t-2}, z_{t-1}, z_t)
        # to compute 3rd-order finite differences Δ³. The history buffer stores these.
        self._z_history: list[Tensor] = []  # Last 3 latents for finite differences

    def reset(self) -> None:
        """Reset latent history (call at start of each generation).

        Paper Section 3.2: The latent history must be cleared between different
        generation runs since the denoising trajectories are independent.
        """
        self._z_history = []

    def record(self, z: Tensor) -> None:
        """Record a latent state for trajectory analysis.

        Paper Section 3.2: Each denoising step's latent z_t is recorded to build
        the trajectory history needed for computing Δ³. Only the last 3 states
        are kept since 3rd-order differences require exactly 3 consecutive samples.
        """
        self._z_history.append(z.detach())
        if len(self._z_history) > 3:
            self._z_history.pop(0)

    @torch.no_grad()
    def compute_schedule(
        self,
        z: Tensor,
        step_idx: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> int:
        """Determine the optimal patch scale for the current timestep.

        Paper Section 3.2, Algorithm 1: Dynamic scheduling based on Δ³ spatial
        variance. The algorithm iterates from coarsest to finest scale, testing
        whether the spatial content at each granularity is "smooth enough" for
        that scale by comparing per-patch standard deviations against threshold τ.

        Args:
            z: Current latent [B, F*H*W, C]
            step_idx: Current denoising step index (0 = noisiest)
            num_frames, height, width: spatial dimensions

        Returns:
            Patch scale (1, 2, or 4)
        """
        # Paper Section 3.2: Warmup phase — use fine patches (scale=1) while
        # building latent history. Need at least 3 recorded states to compute Δ³.
        # Paper default warmup = 3 steps.
        if step_idx < self.config.warmup_steps or len(self._z_history) < 3:
            return 1

        # Paper Section 3.2: Compute third-order finite difference Δ³.
        # Given three consecutive latent states z_{t-2}, z_{t-1}, z_t:
        #   Δ¹ = z_t - z_{t-1}           (first-order: velocity)
        #   Δ² = Δ¹_t - Δ¹_{t-1}         (second-order: acceleration)
        #   Δ³ = Δ²_t - Δ²_{t-1}         (third-order: jerk)
        # Δ³ being small means the denoising trajectory is smooth → coarse patches OK.
        # Δ³ being large means rapid spatial changes → need fine patches.
        z2, z1, z0 = self._z_history[-3], self._z_history[-2], self._z_history[-1]
        delta_0 = z0 - z1  # Δz_{t}
        delta_1 = z1 - z2  # Δz_{t+1}

        # Third-order finite difference approximation.
        # Δ³ = (Δz_{t-1} + Δz_{t+1}) - Δz_t (simplified from paper)
        third_order = delta_0 + delta_1 - 2 * (z0 - z2) / 2

        # Reshape to spatial grid for per-patch analysis
        B, S, C = third_order.shape
        spatial = third_order.view(B, num_frames, height, width, C)

        # Paper Algorithm 1: Try each scale from coarsest to finest.
        # The first scale whose ρ-th percentile of per-patch std falls below τ
        # is selected. This greedy approach ensures maximum token reduction.
        for scale in sorted(self.supported_scales, reverse=True):
            if scale == 1:
                return 1  # Fallback to finest — always valid

            if height % scale != 0 or width % scale != 0:
                continue  # Skip incompatible scales (dimensions must be divisible)

            # Paper Algorithm 1, line 4-6: Compute per-patch variance at this scale.
            # Reshape Δ³ into s×s patches and compute std within each patch.
            h_p, w_p = height // scale, width // scale
            patches = spatial.view(B, num_frames, h_p, scale, w_p, scale, C)

            # Per-patch std across the s×s spatial extent (dims 3 and 5).
            # High std within a patch → that patch contains fine spatial detail
            # that would be lost if merged.
            patch_std = patches.std(dim=(3, 5))  # [B, F, h_p, w_p, C]

            # Aggregate: mean over channels, then flatten spatial dims.
            # This gives one score per patch location.
            patch_scores = patch_std.mean(dim=-1).view(B, -1)  # [B, F*h_p*w_p]

            # Paper Algorithm 1, line 7-8: Take ρ-th percentile of patch scores.
            # Using percentile rather than max or mean makes the scheduler robust
            # to outlier patches — a few complex patches won't force the entire
            # frame to use fine resolution. Paper default ρ=0.4.
            k = max(1, int(patch_scores.shape[1] * self.config.percentile))
            percentile_val = patch_scores.kthvalue(k, dim=1).values.mean().item()

            # Paper Algorithm 1, line 9: Compare against threshold τ.
            # If the ρ-th percentile < τ, spatial content is smooth enough
            # for this scale. Paper default τ=0.001.
            if percentile_val < self.config.threshold:
                return scale

        return 1  # Default to finest resolution


class DDiTAdapter(nn.Module):
    """DDiT adapter for LTX-2 / SCD inference.

    Paper Section 3: This is the main DDiT module that wraps the base DiT model.
    Paper Figure 2: Implements the merge → transformer → unmerge pipeline with
    residual bypass, where the transformer blocks themselves are unchanged —
    they just see fewer tokens when a larger patch scale is selected.

    Paper Section 3.3 (Training via Knowledge Distillation): The teacher is the
    frozen base model running at native resolution. The student uses the DDiT
    merge/unmerge layers. Loss = MSE(student_output, teacher_output). The paper
    trains LoRA on FFN layers (net.0.proj, net.2) with rank 32.

    Adaptation: LTX-2's SCD mode splits the 48-block transformer into 32 encoder
    blocks + 16 decoder blocks. DDiT only wraps the decoder blocks, leaving the
    encoder at full resolution. This differs from the paper which applies DDiT to
    all transformer blocks.

    The base patchify_proj (scale=1) is kept frozen. Additional merge layers
    for scale 2 and 4 are trainable (initialized from base weights via the
    tiling strategy described in init_from_base).
    """

    def __init__(
        self,
        inner_dim: int,
        in_channels: int,
        config: DDiTConfig | None = None,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.in_channels = in_channels
        self.config = config or DDiTConfig()

        # Paper Section 3.1: Create per-scale merge layers. Each scale s > 1 gets
        # its own patchify projection w^{emb}_{p_new} (Eq. 1), de-patchify projection,
        # patch-size identifier (Section 3.1.1), and residual block (Section 3.1.2).
        # Scale=1 uses the base model's existing patchify_proj and is not duplicated here.
        self.merge_layers = nn.ModuleDict()
        for scale in self.config.supported_scales:
            if scale > 1:
                self.merge_layers[str(scale)] = DDiTMergeLayer(
                    scale=scale,
                    in_channels=in_channels,
                    inner_dim=inner_dim,
                )

        # Paper Section 3.2: Dynamic patch scheduler. Determines the optimal scale
        # at each denoising step using 3rd-order finite differences (Algorithm 1).
        self.scheduler = DDiTPatchScheduler(self.config)

        # Output scale-shift table for non-base scales.
        # Paper: The base model uses a scale_shift_table for adaptive LayerNorm
        # de-embedding at the output. Each DDiT scale needs its own copy since
        # the output normalization may need scale-specific adjustments.
        # Initialized from base model weights in init_from_base_model().
        self.output_scale_shift = nn.ParameterDict()
        for scale in self.config.supported_scales:
            if scale > 1:
                self.output_scale_shift[str(scale)] = nn.Parameter(
                    torch.zeros(2, inner_dim)
                )

    def init_from_base_model(self, base_model: nn.Module) -> None:
        """Initialize merge layer weights from the base model's projections.

        Paper Section 3.3: Knowledge distillation requires the student to start
        near the teacher's behavior. This method initializes all per-scale
        projections from the base model's weights using the tiling strategy
        (see DDiTMergeLayer.init_from_base), and copies the output scale_shift
        table so de-embedding starts identical to the base model.
        """
        for scale_str, merge_layer in self.merge_layers.items():
            merge_layer.init_from_base(
                base_patchify=base_model.patchify_proj,
                base_proj_out=base_model.proj_out,
            )

        # Copy output scale_shift from base model for each non-base scale.
        # This ensures the adaptive LayerNorm de-embedding starts identical.
        for scale_str in self.output_scale_shift:
            self.output_scale_shift[scale_str].data.copy_(
                base_model.scale_shift_table.data
            )

    def get_current_scale(
        self,
        z: Tensor,
        step_idx: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> int:
        """Determine patch scale for current timestep.

        Paper Section 3.2, Algorithm 1: Delegates to DDiTPatchScheduler which
        computes Δ³ and selects the coarsest scale whose spatial variance is
        below threshold τ.
        """
        return self.scheduler.compute_schedule(z, step_idx, num_frames, height, width)

    def patchify(
        self,
        latent: Tensor,
        scale: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tensor:
        """Apply patchification at the given scale.

        Paper Section 3.1, Eq. (1): For scale > 1, merges s×s spatial tokens
        and projects through per-scale weights w^{emb}_{p_new}, then adds the
        patch-size identifier e_s (Section 3.1.1).

        For scale=1: returns None to signal the caller should use the base
        model's native patchify_proj (which remains frozen).

        Args:
            latent: [B, F*H*W, C] input tokens
            scale: Patch scale (1, 2, or 4)
            num_frames, height, width: spatial dims

        Returns:
            [B, new_seq_len, inner_dim] projected tokens, or None for scale=1
        """
        if scale == 1:
            return None  # Let base patchify_proj handle it (frozen, native resolution)

        layer = self.merge_layers[str(scale)]
        return layer.forward_patchify(latent, num_frames, height, width)

    def unpatchify(
        self,
        x: Tensor,
        input_latent: Tensor,
        scale: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tensor:
        """Reverse patchification at the given scale.

        Paper Figure 2: The unmerge path (right side of diagram). Projects
        transformer output back to merged channel dim, spatially unmerges to
        original resolution, and applies the residual bypass (Section 3.1.2).

        Args:
            x: [B, new_seq_len, inner_dim] transformer output
            input_latent: [B, F*H*W, C] original input (for residual bypass,
                          Paper Section 3.1.2)
            scale: Patch scale used
            num_frames, height, width: ORIGINAL spatial dims

        Returns:
            [B, F*H*W, C] at original resolution, or None for scale=1
        """
        if scale == 1:
            return None  # Let base proj_out handle it (frozen, native resolution)

        layer = self.merge_layers[str(scale)]
        return layer.forward_unpatchify(
            x, input_latent, num_frames, height, width,
            residual_weight=self.config.residual_weight,
        )

    def adjust_positions(
        self,
        positions: Tensor,
        scale: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tensor:
        """Adjust position coordinates for coarser patch grid.

        Paper Section 3.1: When operating at coarser scales, the positional
        encodings must be adjusted to reflect the reduced spatial resolution.
        The paper uses bilinear interpolation of learned positional embeddings.

        Adaptation: LTX-2 uses 4D RoPE positions [B, 3, seq_len, 2] with
        (start, end) bounds per patch, NOT learned positional embeddings.
        Since RoPE positions represent continuous coordinate ranges, we use:
        - Min pooling for start bounds (take the smallest coordinate in each
          merged s×s patch to get the patch's lower-left corner)
        - Max pooling for end bounds (take the largest coordinate to get the
          patch's upper-right corner)
        This preserves the correct spatial extent of each merged patch in RoPE
        space, whereas the paper's bilinear interpolation of learned PE would
        produce a centroid-like position.

        Handles both 3D [B, 3, seq_len] and 4D [B, 3, seq_len, 2] formats.
        LTX-2 uses 4D positions with [start, end) bounds per patch.
        For scale>1, we compute pooled positions of each merged patch.

        Args:
            positions: [B, 3, F*H*W] or [B, 3, F*H*W, 2] position coordinates
            scale: Patch scale
            num_frames, height, width: original spatial dims

        Returns:
            Same shape format as input, but with reduced seq_len
        """
        if scale == 1:
            return positions

        B = positions.shape[0]
        s = scale
        has_bounds = positions.ndim == 4  # [B, 3, seq, 2]

        if has_bounds:
            # Adaptation: Handle LTX-2's 4D positions [B, 3, seq_len, 2] with
            # (start, end) bounds. The paper doesn't address this format since
            # it uses learned positional embeddings with simple bilinear interpolation.
            # We use min/max pooling to correctly compute the spatial extent of
            # each merged s×s patch.
            results = []
            for bound_idx in range(2):
                pos = positions[:, :, :, bound_idx]  # [B, 3, seq_len]
                pos = pos.view(B, 3, num_frames, height, width)
                pos_flat = pos.reshape(B * 3 * num_frames, 1, height, width)
                pooled = F.avg_pool2d(pos_flat, kernel_size=s, stride=s)
                new_h, new_w = pooled.shape[-2], pooled.shape[-1]
                pos_coarse = pooled.view(B, 3, num_frames, new_h, new_w)
                new_seq = num_frames * new_h * new_w
                results.append(pos_coarse.reshape(B, 3, new_seq))

            # Adaptation: For end bounds, use MAX pooling to capture the outermost
            # coordinate of the merged patch. This ensures the RoPE encoding spans
            # the full spatial extent of all s² original tokens.
            pos_end = positions[:, :, :, 1].view(B, 3, num_frames, height, width)
            pos_end_flat = pos_end.reshape(B * 3 * num_frames, 1, height, width)
            # Only max-pool spatial dims (H, W), not temporal (F is preserved)
            pooled_end = F.max_pool2d(pos_end_flat, kernel_size=s, stride=s)
            new_h, new_w = pooled_end.shape[-2], pooled_end.shape[-1]
            results[1] = pooled_end.view(B, 3, num_frames, new_h, new_w).reshape(B, 3, num_frames * new_h * new_w)

            # Adaptation: For start bounds, use MIN pooling (implemented as
            # negate → max_pool → negate) to capture the innermost coordinate.
            # This gives each merged patch a start position at its lower-left corner.
            pos_start = positions[:, :, :, 0].view(B, 3, num_frames, height, width)
            pos_start_flat = pos_start.reshape(B * 3 * num_frames, 1, height, width)
            pooled_start = -F.max_pool2d(-pos_start_flat, kernel_size=s, stride=s)
            results[0] = pooled_start.view(B, 3, num_frames, new_h, new_w).reshape(B, 3, num_frames * new_h * new_w)
            return torch.stack(results, dim=-1)  # [B, 3, new_seq, 2]
        else:
            # Handle 3D positions [B, 3, seq_len] — simpler case, avg_pool suffices.
            # This path is for models that don't use (start, end) bound pairs.
            pos = positions.view(B, 3, num_frames, height, width)
            pos_flat = pos.reshape(B * 3 * num_frames, 1, height, width)
            pooled = F.avg_pool2d(pos_flat, kernel_size=s, stride=s)
            new_h, new_w = pooled.shape[-2], pooled.shape[-1]
            pos_coarse = pooled.view(B, 3, num_frames, new_h, new_w)
            new_seq_len = num_frames * new_h * new_w
            return pos_coarse.view(B, 3, new_seq_len)

    def adjust_mask(
        self,
        mask: Tensor | None,
        scale: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tensor | None:
        """Adjust attention mask for coarser patch grid.

        Adaptation: For LTX-2's SCD (Split Causal DiT) mode, the attention mask
        enforces frame-causal ordering where each frame can only attend to itself
        and previous frames. When the spatial resolution changes due to DDiT merging,
        the mask must be rebuilt at the new sequence length with the new tokens-per-frame
        count. The paper does not discuss causal masking since it targets non-causal
        T2V/I2V models (Wan-2.1).

        For SCD causal masks, rebuild at the new sequence length.
        """
        if mask is None or scale == 1:
            return mask

        from ltx_core.model.transformer.scd_model import build_frame_causal_mask

        new_h = height // scale
        new_w = width // scale
        new_tpf = new_h * new_w  # Tokens per frame at the merged resolution
        new_seq_len = num_frames * new_tpf

        return build_frame_causal_mask(
            seq_len=new_seq_len,
            tokens_per_frame=new_tpf,
            device=mask.device,
            dtype=mask.dtype,
        )
