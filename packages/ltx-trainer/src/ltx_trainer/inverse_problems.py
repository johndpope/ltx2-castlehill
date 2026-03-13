"""Forward operators for VFM inverse problems on video latents.

VFM Paper §2.2 (Inverse Problems):
    An inverse problem seeks to recover signal x from noisy observations:
        y = A(x) + ε,  ε ~ N(0, σ²I)
    where A is a known forward operator.

VFM Paper §3.2 (Amortizing Over Multiple Inverse Problems):
    We define a family of forward operators {A_c} indexed by class c.
    The noise adapter qφ(z|y,c) is conditioned on both the observation y
    and the class label c, allowing a single model to handle multiple tasks.

For video generation, our inverse problems operate in VAE latent space:
    - Latents are [B, seq_len, C] where C=128 (video latent channels)
    - Each "token" represents a spatial patch within a video frame
    - Forward operators degrade the latent in task-specific ways
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class InverseProblemSample:
    """Container for an inverse problem instance.

    Attributes:
        observation: Degraded observation y = A(x) + ε  [B, seq_len, C]
        task_class: Integer task class index [B]
        task_name: String name of the task (for logging)
        forward_operator_fn: Callable that applies A(x) given clean latents
        noise_level: σ used for observation noise
    """
    observation: Tensor
    task_class: Tensor
    task_name: str
    noise_level: float


class ForwardOperator:
    """Base class for forward operators A(x) in inverse problems."""

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        """Apply forward operator to clean latents.

        Args:
            x: Clean video latents [B, seq_len, C]

        Returns:
            Degraded observation [B, obs_len, C] (obs_len may differ from seq_len)
        """
        raise NotImplementedError


class IdentityOperator(ForwardOperator):
    """Identity forward operator: A(x) = x.

    Used for text-to-video (t2v) where no visual observation is given.
    The observation is just zeros (placeholder), and the adapter must
    rely entirely on the task class embedding.
    """

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        return x


class FirstFrameOperator(ForwardOperator):
    """Extract first frame: A(x) = x[:, :tpf, :].

    Used for image-to-video (i2v) conditioning. The observation is the
    clean first frame latent, and the model must generate the rest.

    Args:
        tokens_per_frame: Number of tokens per video frame (H * W)
    """

    def __init__(self, tokens_per_frame: int):
        self.tokens_per_frame = tokens_per_frame

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        return x[:, :self.tokens_per_frame, :]


class MaskOperator(ForwardOperator):
    """Random masking: A(x) = x ⊙ M where M is a binary mask.

    Used for video inpainting. Randomly masks contiguous spatial/temporal
    regions. Unmasked tokens are passed through; masked tokens become zero.

    VFM Paper §3.2: "c now defines a collection of inverse problems A_c = {A^ω_c}
    where ω defines random masks."

    Args:
        mask_ratio_range: (min_ratio, max_ratio) fraction of tokens to mask
    """

    def __init__(self, mask_ratio_range: tuple[float, float] = (0.2, 0.8)):
        self.mask_ratio_range = mask_ratio_range

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        B, S, C = x.shape
        ratio = random.uniform(*self.mask_ratio_range)
        num_mask = int(S * ratio)

        # Random mask per batch element
        mask = torch.ones(B, S, 1, device=x.device, dtype=x.dtype)
        for b in range(B):
            # Contiguous block masking (more realistic than random scatter)
            start = random.randint(0, S - num_mask)
            mask[b, start:start + num_mask, :] = 0.0

        return x * mask


class DownsampleOperator(ForwardOperator):
    """Spatial downsampling: keeps every k-th token within each frame.

    Used for super-resolution. Simulates observing a low-resolution version
    of the video in latent space.

    Args:
        factor: Downsampling factor (keep 1/factor of spatial tokens)
        tokens_per_frame: Tokens per frame (H * W)
    """

    def __init__(self, factor: int = 4, tokens_per_frame: int = 336):
        self.factor = factor
        self.tokens_per_frame = tokens_per_frame

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        B, S, C = x.shape
        num_frames = S // self.tokens_per_frame
        tpf = self.tokens_per_frame

        # For each frame, keep every factor-th token (stride sampling)
        result = torch.zeros_like(x)
        for f in range(num_frames):
            start = f * tpf
            end = start + tpf
            frame = x[:, start:end, :]
            # Zero out non-sampled positions (simulates low-res observation)
            keep_mask = torch.zeros(tpf, device=x.device, dtype=torch.bool)
            keep_mask[::self.factor] = True
            result[:, start:end, :] = frame * keep_mask.unsqueeze(0).unsqueeze(-1).float()

        return result


class GaussianNoiseOperator(ForwardOperator):
    """Add Gaussian noise: A(x) = x (observation is x + ε).

    Used for denoising tasks. The forward operator is identity, but the
    observation noise σ is larger than for other tasks.

    Args:
        noise_level: Standard deviation of observation noise
    """

    def __init__(self, noise_level: float = 0.3):
        self.noise_level = noise_level

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        return x  # Noise is added externally when constructing y = A(x) + ε


# Registry of forward operators
OPERATOR_REGISTRY = {
    "i2v": FirstFrameOperator,
    "inpaint": MaskOperator,
    "sr": DownsampleOperator,
    "denoise": GaussianNoiseOperator,
    "t2v": IdentityOperator,
}


@dataclass
class InverseProblemConfig:
    """Configuration for a single inverse problem class.

    Attributes:
        name: Task name (must be in TASK_CLASSES)
        weight: Sampling probability during training
        noise_level: σ for observation noise y = A(x) + ε
        operator_kwargs: Additional kwargs for the forward operator constructor
    """
    name: str
    weight: float = 1.0
    noise_level: float = 0.1
    operator_kwargs: dict | None = None


class InverseProblemSampler:
    """Samples inverse problems during VFM training.

    VFM Paper Algorithm 2, line 4-6:
        Sample c ~ p(c), x ~ p(x)
        Sample forward operator A^ω_c ∈ A_c
        y ← A^ω_c(x) + ε, ε ~ N(0, σ²I)

    This class handles:
    1. Randomly selecting an inverse problem class based on configured weights
    2. Applying the corresponding forward operator to clean latents
    3. Adding observation noise
    """

    def __init__(
        self,
        problems: list[InverseProblemConfig],
        tokens_per_frame: int = 336,
    ):
        self.problems = problems
        self.tokens_per_frame = tokens_per_frame

        # Normalize weights to probabilities
        total_weight = sum(p.weight for p in problems)
        self.probs = [p.weight / total_weight for p in problems]

        # Pre-build operators
        self.operators: dict[str, ForwardOperator] = {}
        for problem in problems:
            op_class = OPERATOR_REGISTRY[problem.name]
            kwargs = problem.operator_kwargs or {}
            if problem.name == "i2v":
                kwargs["tokens_per_frame"] = tokens_per_frame
            elif problem.name == "sr":
                kwargs.setdefault("tokens_per_frame", tokens_per_frame)
            self.operators[problem.name] = op_class(**kwargs)

        # Task class indices (must match TASK_CLASSES in noise_adapter.py)
        from ltx_core.model.transformer.noise_adapter import TASK_CLASSES
        self.task_class_map = TASK_CLASSES

    def sample(
        self,
        clean_latents: Tensor,
    ) -> InverseProblemSample:
        """Sample an inverse problem and apply it to clean latents.

        Args:
            clean_latents: [B, seq_len, C] clean video latents x

        Returns:
            InverseProblemSample with observation y, task class, etc.
        """
        B = clean_latents.shape[0]

        # Sample problem class
        idx = random.choices(range(len(self.problems)), weights=self.probs, k=1)[0]
        problem = self.problems[idx]

        # Apply forward operator
        operator = self.operators[problem.name]
        y_clean = operator(clean_latents)

        # Add observation noise: y = A(x) + ε
        noise = torch.randn_like(y_clean) * problem.noise_level
        observation = y_clean + noise

        # Task class tensor
        task_idx = self.task_class_map[problem.name]
        task_class = torch.full((B,), task_idx, dtype=torch.long, device=clean_latents.device)

        return InverseProblemSample(
            observation=observation,
            task_class=task_class,
            task_name=problem.name,
            noise_level=problem.noise_level,
        )


def default_inverse_problems() -> list[InverseProblemConfig]:
    """Default inverse problem configuration for video generation.

    Weights reflect relative importance:
    - i2v (40%): Primary use case for video generation
    - inpaint (20%): Important for editing workflows
    - sr (15%): Super-resolution for upscaling
    - denoise (10%): Denoising for quality improvement
    - t2v (15%): Text-only generation (maintains unconditional capability)
    """
    return [
        InverseProblemConfig(name="i2v", weight=0.4, noise_level=0.05),
        InverseProblemConfig(name="inpaint", weight=0.2, noise_level=0.1),
        InverseProblemConfig(name="sr", weight=0.15, noise_level=0.1),
        InverseProblemConfig(name="denoise", weight=0.1, noise_level=0.3),
        InverseProblemConfig(name="t2v", weight=0.15, noise_level=0.0),
    ]
