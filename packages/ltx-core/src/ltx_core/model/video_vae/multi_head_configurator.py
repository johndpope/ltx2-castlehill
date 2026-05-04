"""Configurators for the multi-head VQ-VAE.

Follows the same pattern as VideoEncoderConfigurator / VideoDecoderConfigurator.
"""

from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.video_vae.multi_head_vq import MultiHeadVQConfig, MultiHeadVQVideoVAE


class MultiHeadVQVAEConfigurator(ModelConfigurator[MultiHeadVQVideoVAE]):
    """Configurator for creating a MultiHeadVQVideoVAE from a config dict."""

    @classmethod
    def from_config(cls, config: dict) -> MultiHeadVQVideoVAE:
        vae_config = config.get("vae", {})
        vq_config_dict = config.get("multi_head_vq", {})

        latent_channels = vae_config.get("latent_channels", 128)
        num_heads = vq_config_dict.get("num_heads", 4)
        head_dims = vq_config_dict.get("head_dims", [latent_channels // num_heads] * num_heads)
        codebook_sizes = vq_config_dict.get("codebook_sizes", [4096] * num_heads)

        vq_config = MultiHeadVQConfig(
            num_heads=num_heads,
            head_dims=head_dims,
            codebook_sizes=codebook_sizes,
            codebook_dim=vq_config_dict.get("codebook_dim", head_dims[0]),
            commitment_cost=vq_config_dict.get("commitment_cost", 0.25),
            use_ema=vq_config_dict.get("use_ema", True),
            ema_decay=vq_config_dict.get("ema_decay", 0.99),
            use_root_disentanglement=vq_config_dict.get("use_root_disentanglement", True),
            root_dim=vq_config_dict.get("root_dim", 4),
        )

        return MultiHeadVQVideoVAE(
            encoder=None,
            decoder=None,
            config=vq_config,
        )
