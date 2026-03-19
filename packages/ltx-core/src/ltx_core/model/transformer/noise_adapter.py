"""Variational Flow Maps (VFM) Noise Adapter for conditional generation.

Implements the noise adapter network qφ(z|y) from:
    "Variational Flow Maps: Make Some Noise for One-Step Conditional Generation"
    Mammadov et al., 2026 (arXiv:2603.07276)

VFM Paper §3 (Core Idea):
    Rather than guide a sampling path (as in diffusion guidance), VFM learns the
    "proper initial noise" for conditional generation. Given an observation y
    (e.g., first frame, text, degraded video), the noise adapter outputs a
    Gaussian distribution qφ(z|y) = N(z | μφ(y), diag(σ²φ(y))).

    Sampling z ~ qφ(z|y) and mapping through the flow map x = fθ(z) produces
    conditional samples in a single step — no iterative guidance needed.

VFM Paper §3.1 (Joint Training):
    The adapter and flow map are trained jointly via the variational objective:
        L = (1/2τ²) * L_MF + (1/2σ²) * L_obs + L_KL
    Joint training allows fθ to compensate for the Gaussian simplicity of qφ
    by reshaping the noise-to-data coupling (Proposition 3.1).

Integration with SCD:
    In our SCD architecture, the noise adapter sits between the encoder and decoder:
    - Encoder output (shifted features from frame t-1) serves as the observation y
    - The adapter transforms these features into structured noise parameters (μ, σ)
    - The decoder (flow map) maps z ~ qφ(z|y) to the denoised frame

    This replaces the standard N(0,I) noise sampling with observation-dependent
    noise, enabling one/few-step conditional generation.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class NoiseAdapterMLP(nn.Module):
    """MLP-based noise adapter: encoder_features → (μ, log_σ).

    Takes encoder features (the "observation" y in VFM notation) and an optional
    task class embedding, and outputs Gaussian parameters for structured noise.

    VFM Paper §3.2 (Amortization):
        The adapter is amortized over multiple inverse problem classes c ∈ {1,...,C}.
        A task class embedding conditions the adapter on which degradation was applied:
        - i2v: image-to-video (first frame given)
        - inpaint: masked region completion
        - sr: super-resolution (low-res input)
        - denoise: noisy input denoising
        - t2v: text-only (no visual observation)

    Architecture:
        Input: encoder_features [B, seq_len, D] + task_class_emb [B, D_class]
        → LayerNorm → Linear(D → H) → SiLU → Linear(H → H) → SiLU
        → Linear(H → latent_dim) for μ
        → Linear(H → latent_dim) for log_σ
        Output: μ [B, seq_len, latent_dim], log_σ [B, seq_len, latent_dim]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_task_classes: int = 5,
        task_embed_dim: int = 128,
        init_sigma: float = -2.0,
    ):
        """Initialize the noise adapter MLP.

        Args:
            input_dim: Dimension of encoder features (D from transformer)
            latent_dim: Dimension of output noise (matches video latent channels, typically 128)
            hidden_dim: Hidden dimension of the MLP
            num_layers: Number of hidden layers
            num_task_classes: Number of inverse problem classes
            task_embed_dim: Dimension of task class embedding
            init_sigma: Initial value for log_σ output bias (negative = start near standard normal)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.init_sigma = init_sigma

        # Task class embedding: c → embedding vector
        self.task_embedding = nn.Embedding(num_task_classes, task_embed_dim)

        # Input projection: concat encoder features + task embedding
        total_input_dim = input_dim + task_embed_dim
        self.input_norm = nn.LayerNorm(total_input_dim)

        # Build MLP layers
        layers = []
        in_dim = total_input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Separate heads for μ and log_σ
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

        # Initialize: μ near zero (start near standard normal)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

        # Initialize: log_σ biased negative so σ starts small
        # VFM Paper §3: We want qφ(z|y) ≈ N(0, I) initially so the flow map
        # doesn't need to immediately adapt to a wildly different noise distribution.
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, init_sigma)

    def forward(
        self,
        encoder_features: Tensor,
        task_class: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute noise distribution parameters from encoder features.

        Args:
            encoder_features: [B, seq_len, D] encoder hidden features (observation y)
            task_class: [B] integer task class indices

        Returns:
            mu: [B, seq_len, latent_dim] mean of noise distribution
            log_sigma: [B, seq_len, latent_dim] log std of noise distribution
        """
        B, S, D = encoder_features.shape

        # Expand task embedding to match sequence length: [B, task_embed_dim] -> [B, S, task_embed_dim]
        task_emb = self.task_embedding(task_class)  # [B, task_embed_dim]
        task_emb = task_emb.unsqueeze(1).expand(-1, S, -1)  # [B, S, task_embed_dim]

        # Concatenate encoder features with task embedding
        x = torch.cat([encoder_features, task_emb], dim=-1)  # [B, S, D + task_embed_dim]
        x = self.input_norm(x)

        # MLP forward
        h = self.mlp(x)  # [B, S, hidden_dim]

        # Output heads
        mu = self.mu_head(h)  # [B, S, latent_dim]
        log_sigma = self.log_sigma_head(h)  # [B, S, latent_dim]

        # Clamp log_sigma for numerical stability
        log_sigma = log_sigma.clamp(min=-10.0, max=2.0)

        return mu, log_sigma

    def sample(
        self,
        encoder_features: Tensor,
        task_class: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample structured noise z ~ qφ(z|y) via reparameterization trick.

        VFM Paper §3 (Reparameterization):
            z = μφ(y,c) + σφ(y,c) ⊙ ε, where ε ~ N(0,I)

        Args:
            encoder_features: [B, seq_len, D] encoder hidden features
            task_class: [B] integer task class indices
            temperature: Scaling factor for the noise (1.0 = standard)

        Returns:
            z: [B, seq_len, latent_dim] sampled noise
        """
        mu, log_sigma = self.forward(encoder_features, task_class)
        sigma = torch.exp(log_sigma)

        # Reparameterization trick
        eps = torch.randn_like(mu)
        z = mu + sigma * eps * temperature

        return z

    def sample_with_mu(
        self,
        encoder_features: Tensor,
        task_class: Tensor,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Sample structured noise z ~ qφ(z|y) and return adapter mu.

        Returns:
            (z, mu) where z: [B, seq_len, latent_dim], mu: [B, seq_len, latent_dim]
        """
        mu, log_sigma = self.forward(encoder_features, task_class)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps * temperature
        return z, mu

    def kl_divergence(
        self,
        mu: Tensor,
        log_sigma: Tensor,
    ) -> Tensor:
        """Compute KL divergence KL(qφ(z|y) || N(0,I)).

        VFM Paper Eq. 15:
            L_KL = E_p(y) [KL(qφ(z|y) || p(z))]

        For diagonal Gaussian qφ vs standard normal p(z):
            KL = 0.5 * Σ(μ² + σ² - log(σ²) - 1)

        Args:
            mu: [B, seq_len, latent_dim] mean
            log_sigma: [B, seq_len, latent_dim] log std

        Returns:
            Scalar KL divergence (averaged over batch and dimensions)
        """
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - 2*log(σ) - 1)
        kl = 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1)
        return kl.mean()


class NoiseAdapterTransformer(nn.Module):
    """Lightweight transformer-based noise adapter for better sequence modeling.

    When encoder features have strong spatial/temporal structure, an MLP processes
    each token independently. This variant adds self-attention to capture
    inter-token dependencies in the noise distribution.

    Architecture:
        - 4-6 transformer layers with self-attention
        - Much smaller than the main model (~50M vs 19B params)
        - Shares the same μ/log_σ output head structure as the MLP variant
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_task_classes: int = 5,
        task_embed_dim: int = 128,
        init_sigma: float = -2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_sigma = init_sigma

        self.task_embedding = nn.Embedding(num_task_classes, task_embed_dim)

        # Project input to hidden dim
        total_input_dim = input_dim + task_embed_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.Linear(total_input_dim, hidden_dim),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output heads
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

        # Same initialization strategy as MLP variant
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, init_sigma)

    def forward(
        self,
        encoder_features: Tensor,
        task_class: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, S, D = encoder_features.shape

        task_emb = self.task_embedding(task_class).unsqueeze(1).expand(-1, S, -1)
        x = torch.cat([encoder_features, task_emb], dim=-1)
        x = self.input_proj(x)

        x = self.transformer(x)

        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x).clamp(min=-10.0, max=2.0)

        return mu, log_sigma

    def sample(
        self,
        encoder_features: Tensor,
        task_class: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        mu, log_sigma = self.forward(encoder_features, task_class)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps * temperature

    def kl_divergence(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        kl = 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1)
        return kl.mean()


# Task class indices for inverse problem types
TASK_CLASSES = {
    "i2v": 0,        # Image-to-video: first frame given
    "inpaint": 1,    # Inpainting: masked region completion
    "sr": 2,         # Super-resolution: low-res input
    "denoise": 3,    # Denoising: noisy input
    "t2v": 4,        # Text-to-video: no visual observation (unconditional)
}


def create_noise_adapter(
    input_dim: int,
    latent_dim: int = 128,
    variant: Literal["mlp", "transformer"] = "mlp",
    **kwargs,
) -> NoiseAdapterMLP | NoiseAdapterTransformer:
    """Factory function to create a noise adapter.

    Args:
        input_dim: Dimension of encoder features
        latent_dim: Dimension of output noise (video latent channels)
        variant: "mlp" for independent per-token processing, "transformer" for sequence-aware
        **kwargs: Additional arguments passed to the adapter constructor

    Returns:
        Noise adapter module
    """
    if variant == "mlp":
        return NoiseAdapterMLP(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
    elif variant == "transformer":
        return NoiseAdapterTransformer(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown noise adapter variant: {variant}")
