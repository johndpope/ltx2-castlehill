"""Separable Causal Diffusion (SCD) wrapper for LTX-2.

Splits the LTX-2 48-layer transformer into an encoder (causal temporal, runs once per frame)
and a decoder (per-frame denoising, runs N times per step). Based on the SCD paper
(arxiv 2602.10095) which shows early transformer layers produce redundant features across
denoising steps, while deeper layers handle intra-frame spatial rendering.

The key insight: causal temporal reasoning is separable from multi-step denoising,
enabling ~3x inference speedup with minimal quality loss.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer_args import TransformerArgs


@dataclass
class KVCache:
    """Key-value cache for encoder attention during autoregressive inference.

    During autoregressive generation, the encoder processes one new frame at a time.
    This cache stores K/V from previous frames so that each new frame can attend to
    the full history without recomputing.

    Usage in the encoder loop:
        for block in encoder_blocks:
            layer_cache = kv_cache.get_layer_cache(block.idx)
            video_args, audio_args = block(..., kv_cache=layer_cache)
    The Attention layer reads/writes to the per-layer dict in-place.
    """
    keys: dict[int, Tensor]      # layer_idx -> cached keys [B, prev_seq, H*D]
    values: dict[int, Tensor]    # layer_idx -> cached values [B, prev_seq, H*D]
    is_cache_step: bool = False  # Whether we should update the cache this step
    cached_seq_len: int = 0      # Total cached sequence length

    @staticmethod
    def empty() -> KVCache:
        return KVCache(keys={}, values={}, is_cache_step=False, cached_seq_len=0)

    @property
    def has_cache(self) -> bool:
        return len(self.keys) > 0

    def get_layer_cache(self, layer_idx: int) -> dict:
        """Get a mutable per-layer dict compatible with Attention.forward(kv_cache=...).

        The returned dict is updated in-place by the attention layer when is_cache_step=True,
        and we sync those updates back into our main dicts via update_from_layer_cache().
        """
        return {
            "keys": self.keys.get(layer_idx),
            "values": self.values.get(layer_idx),
            "is_cache_step": self.is_cache_step,
            "_layer_idx": layer_idx,
        }

    def update_from_layer_cache(self, layer_cache: dict) -> None:
        """Sync updates from a per-layer dict back into the main cache."""
        layer_idx = layer_cache["_layer_idx"]
        if layer_cache.get("keys") is not None:
            self.keys[layer_idx] = layer_cache["keys"]
        if layer_cache.get("values") is not None:
            self.values[layer_idx] = layer_cache["values"]


def build_frame_causal_mask(
    seq_len: int,
    tokens_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build frame-level causal attention mask for the encoder.

    Frame t can attend to frames <= t. Within a frame, all tokens see each other
    (bidirectional). Across frames, only past-to-future direction is allowed.

    Args:
        seq_len: Total number of tokens across all frames
        tokens_per_frame: Number of tokens per frame (height * width)
        device: Target device
        dtype: Target dtype (float for additive mask)

    Returns:
        Additive attention mask [1, seq_len, seq_len] where blocked positions have -inf
    """
    # Compute which frame each token belongs to
    frame_idx = torch.arange(seq_len, device=device) // tokens_per_frame

    # mask[row, col]: row=query, col=key. Query frame can attend to key frame if query >= key.
    allowed = frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)  # [seq, seq]

    # Create additive mask: 0 for allowed, -inf for blocked
    mask = torch.zeros(1, seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class LTXSCDModel(nn.Module):
    """Separable Causal Diffusion wrapper for LTXModel.

    Splits the existing 48-layer transformer into encoder (layers 0..N-1) and
    decoder (layers N..47). Shares weights with the base model — no duplication.

    The encoder runs with a causal mask (frame t sees frames <= t) and outputs
    hidden features. The decoder takes these features (shifted by 1 frame) along
    with noisy tokens and produces velocity predictions.

    Args:
        base_model: The LTXModel to wrap
        encoder_layers: Number of layers for the encoder (default 32, ~2:1 ratio)
        decoder_input_combine: How to combine encoder features with decoder input.
            "token_concat" (default, best from SCD paper): concatenate along sequence dim
            "add": element-wise addition
            "concat": concatenate along feature dimension with learned projection (2D→D)
            "token_concat_with_proj": token_concat with learned alignment projection
    """

    def __init__(
        self,
        base_model: nn.Module,
        encoder_layers: int = 32,
        decoder_input_combine: str = "token_concat",
        local_control_injection: str = "pre_decoder",
        local_control_layers: list[int] | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.encoder_layers = encoder_layers
        self.decoder_input_combine = decoder_input_combine
        self.local_control_injection = local_control_injection
        self.local_control_layers = local_control_layers

        total_layers = len(base_model.transformer_blocks)
        if encoder_layers >= total_layers:
            raise ValueError(
                f"encoder_layers ({encoder_layers}) must be < total layers ({total_layers})"
            )

        # Views into the same ModuleList — shared weights, no duplication
        self.encoder_blocks = base_model.transformer_blocks[:encoder_layers]
        self.decoder_blocks = base_model.transformer_blocks[encoder_layers:]

        # Optional alignment projection for token_concat_with_proj
        if decoder_input_combine == "token_concat_with_proj":
            self.decoder_alignment = nn.Sequential(
                nn.LayerNorm(base_model.inner_dim),
                nn.Linear(base_model.inner_dim, base_model.inner_dim),
            )
            # Initialize as near-identity
            nn.init.zeros_(self.decoder_alignment[1].weight)
            nn.init.zeros_(self.decoder_alignment[1].bias)
        else:
            self.decoder_alignment = None

        # Feature-dim concat: cat along last dim then project 2D→D
        if decoder_input_combine == "concat":
            D = base_model.inner_dim
            self.feature_concat_proj = nn.Sequential(
                nn.LayerNorm(2 * D),
                nn.Linear(2 * D, D, bias=False),
            )
            # Initialize projection close to averaging (enc + dec) / 2
            with torch.no_grad():
                W = self.feature_concat_proj[1].weight  # [D, 2D]
                W.zero_()
                W[:, :D] = 0.5 * torch.eye(D)   # decoder half
                W[:, D:] = 0.5 * torch.eye(D)   # encoder half
        else:
            self.feature_concat_proj = None

    def _cast_modality_dtype(self, modality: Modality) -> Modality:
        """Cast modality tensors to match patchify_proj weight dtype.

        When the model is quantized (e.g. int8-quanto), patchify_proj may stay
        in a different dtype than the input latents. This ensures compatibility.
        """
        target_dtype = self.base_model.patchify_proj.weight.dtype
        if modality.latent.dtype == target_dtype:
            return modality
        return replace(modality, latent=modality.latent.to(target_dtype))

    @property
    def inner_dim(self) -> int:
        return self.base_model.inner_dim

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Proxy to base model's gradient checkpointing setting."""
        self.base_model.set_gradient_checkpointing(enable)

    def forward_encoder(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
        kv_cache: KVCache | None = None,
        tokens_per_frame: int | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Run encoder blocks with causal mask. Returns hidden features (no proj_out).

        The encoder processes all frames with a frame-level causal mask: frame t
        can attend to frames <= t. With KV-cache during inference, only new frame
        tokens are computed while attending to cached K/V from previous frames.

        Args:
            video: Video modality input
            audio: Audio modality input (passed through unchanged)
            perturbations: Perturbation config for STG
            kv_cache: Optional KV cache for autoregressive inference
            tokens_per_frame: Tokens per frame (height * width), needed for causal mask.
                If None, the causal mask is not applied (for backward compatibility).

        Returns:
            Tuple of (video_args, audio_args) with encoder hidden features in video_args.x
        """
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(
                (video or audio).latent.shape[0]
            )

        # Preprocess inputs using base model's preprocessors
        video_args = (
            self.base_model.video_args_preprocessor.prepare(
                self._cast_modality_dtype(video)
            )
            if video is not None
            else None
        )
        audio_args = (
            self.base_model.audio_args_preprocessor.prepare(audio)
            if audio is not None and hasattr(self.base_model, "audio_args_preprocessor")
            else None
        )

        # Build and inject frame-level causal mask for encoder self-attention
        if video_args is not None and tokens_per_frame is not None:
            seq_len = video_args.x.shape[1]
            causal_mask = build_frame_causal_mask(
                seq_len=seq_len,
                tokens_per_frame=tokens_per_frame,
                device=video_args.x.device,
                dtype=video_args.x.dtype,
            )
            video_args = replace(video_args, self_attn_mask=causal_mask)

        # Run through encoder blocks only
        for block in self.encoder_blocks:
            # Get per-layer KV-cache dict for inference (not used during training)
            layer_cache = kv_cache.get_layer_cache(block.idx) if kv_cache is not None else None

            if self.base_model._enable_gradient_checkpointing and self.base_model.training:
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block,
                    video_args,
                    audio_args,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video_args, audio_args = block(
                    video=video_args,
                    audio=audio_args,
                    perturbations=perturbations,
                )

            # Sync KV-cache updates back from per-layer dict
            if kv_cache is not None and layer_cache is not None:
                kv_cache.update_from_layer_cache(layer_cache)

        # Return hidden features — NOT passed through proj_out
        return video_args, audio_args

    def forward_decoder(
        self,
        video: Modality | None,
        encoder_features: Tensor | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
        encoder_audio_args: TransformerArgs | None = None,
        local_control: Tensor | None = None,
        global_context: Tensor | None = None,
        capture_attention_layers: set[int] | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run decoder blocks with encoder features. Returns velocity prediction.

        The decoder runs per-frame with the actual noise timestep. Encoder features
        (from the previous frame) are coupled with noisy tokens using the configured
        combine mode.

        Args:
            video: Video modality input (noisy tokens with actual sigma timestep)
            encoder_features: Hidden features from encoder [B, enc_seq, D]
            audio: Audio modality input
            perturbations: Perturbation config for STG
            encoder_audio_args: Audio args from encoder pass (for cross-attention)
            local_control: EditCtrl local control signal [B, seq_len, D]
            global_context: EditCtrl global background tokens [B, num_global, D]
            capture_attention_layers: Set of decoder layer indices to capture
                attention weights from.

        Returns:
            Tuple of (video_prediction, audio_prediction) velocity tensors
        """
        if perturbations is None:
            bs = (video or audio).latent.shape[0]
            perturbations = BatchedPerturbationConfig.empty(bs)

        # Preprocess decoder inputs
        video_args = (
            self.base_model.video_args_preprocessor.prepare(
                self._cast_modality_dtype(video)
            )
            if video is not None
            else None
        )
        audio_args = (
            self.base_model.audio_args_preprocessor.prepare(audio)
            if audio is not None and hasattr(self.base_model, "audio_args_preprocessor")
            else None
        )

        # Combine encoder features with decoder tokens
        if video_args is not None and encoder_features is not None:
            video_args = self._combine_encoder_decoder(video_args, encoder_features)

        # --- EditCtrl: prepare local control for injection ---
        local_control_padded = None
        if video_args is not None and local_control is not None:
            if self.decoder_input_combine in ("token_concat", "token_concat_with_proj"):
                if encoder_features is not None:
                    enc_seq = encoder_features.shape[1]
                    pad = local_control.new_zeros(local_control.shape[0], enc_seq, local_control.shape[2])
                    local_control_padded = torch.cat([pad, local_control], dim=1)
                else:
                    local_control_padded = local_control
            else:
                local_control_padded = local_control

            if self.local_control_injection == "pre_decoder":
                video_args = replace(video_args, x=video_args.x + local_control_padded)

        # --- EditCtrl: inject global context into cross-attention ---
        if video_args is not None and global_context is not None:
            new_context = torch.cat([global_context, video_args.context], dim=1)
            global_mask = torch.ones(
                global_context.shape[0], global_context.shape[1],
                device=video_args.context_mask.device,
                dtype=video_args.context_mask.dtype,
            )
            new_context_mask = torch.cat([global_mask, video_args.context_mask], dim=1)
            video_args = replace(video_args, context=new_context, context_mask=new_context_mask)

        # Use audio args from encoder if available and decoder doesn't have its own
        if audio_args is None and encoder_audio_args is not None:
            audio_args = encoder_audio_args

        # Determine which layers get local control injection for "per_layer" mode
        if self.local_control_injection == "per_layer" and local_control_padded is not None:
            if self.local_control_layers is not None:
                inject_layers = set(self.local_control_layers)
            else:
                inject_layers = set(range(len(self.decoder_blocks)))
        else:
            inject_layers = set()

        # Run through decoder blocks only (no causal mask)
        for i, block in enumerate(self.decoder_blocks):
            if self.base_model._enable_gradient_checkpointing and self.base_model.training:
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block,
                    video_args,
                    audio_args,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video_args, audio_args = block(
                    video=video_args,
                    audio=audio_args,
                    perturbations=perturbations,
                )

            # Per-layer local control injection after FFN
            if i in inject_layers:
                video_args = replace(video_args, x=video_args.x + local_control_padded)

        # Process output through proj_out (only for decoder, not encoder)
        vx = None
        if video_args is not None:
            decoder_x = video_args.x
            decoder_emb_ts = video_args.embedded_timestep
            if self.decoder_input_combine in ("token_concat", "token_concat_with_proj"):
                if encoder_features is not None:
                    enc_seq_len = encoder_features.shape[1]
                    decoder_x = decoder_x[:, enc_seq_len:]
                    decoder_emb_ts = decoder_emb_ts[:, enc_seq_len:]

            vx = self.base_model._process_output(
                self.base_model.scale_shift_table,
                self.base_model.norm_out,
                self.base_model.proj_out,
                decoder_x,
                decoder_emb_ts,
            )

        ax = None
        if audio_args is not None and hasattr(self.base_model, "audio_scale_shift_table"):
            ax = self.base_model._process_output(
                self.base_model.audio_scale_shift_table,
                self.base_model.audio_norm_out,
                self.base_model.audio_proj_out,
                audio_args.x,
                audio_args.embedded_timestep,
            )

        return vx, ax

    @staticmethod
    def _duplicate_pe(pe: tuple[Tensor, Tensor] | Tensor | None) -> tuple[Tensor, Tensor] | Tensor | None:
        """Duplicate positional embeddings along the sequence dimension for token_concat.

        RoPE PE in LTX-2 can be:
        - tuple (cos, sin), each [B, num_heads, seq_len, dim_head] (4D)
        - tuple (cos, sin), each [B, seq_len, inner_dim] (3D)
        - single Tensor [B, seq_len, dim] (3D)
        We cat along the sequence dimension in all cases.
        """
        if pe is None:
            return None
        if isinstance(pe, tuple):
            sample = pe[0]
            seq_dim = 2 if sample.ndim == 4 else 1
            return tuple(torch.cat([t, t], dim=seq_dim) for t in pe)
        return torch.cat([pe, pe], dim=1)

    def _combine_encoder_decoder(
        self, video_args: TransformerArgs, encoder_features: Tensor
    ) -> TransformerArgs:
        """Combine encoder features with decoder tokens using the configured mode."""
        if self.decoder_input_combine == "add":
            return replace(video_args, x=video_args.x + encoder_features)

        elif self.decoder_input_combine == "concat":
            combined = torch.cat([video_args.x, encoder_features], dim=-1)
            projected = self.feature_concat_proj(combined)
            return replace(video_args, x=projected)

        elif self.decoder_input_combine == "token_concat":
            combined_x = torch.cat([encoder_features, video_args.x], dim=1)
            combined_pe = self._duplicate_pe(video_args.positional_embeddings)

            combined_ts = torch.cat(
                [video_args.timesteps, video_args.timesteps], dim=-2
            )
            combined_emb_ts = torch.cat(
                [video_args.embedded_timestep, video_args.embedded_timestep], dim=1
            )

            return replace(
                video_args,
                x=combined_x,
                positional_embeddings=combined_pe,
                timesteps=combined_ts,
                embedded_timestep=combined_emb_ts,
            )

        elif self.decoder_input_combine == "token_concat_with_proj":
            aligned = self.decoder_alignment(encoder_features)
            combined_x = torch.cat([aligned, video_args.x], dim=1)
            combined_pe = self._duplicate_pe(video_args.positional_embeddings)

            combined_ts = torch.cat(
                [video_args.timesteps, video_args.timesteps], dim=-2
            )
            combined_emb_ts = torch.cat(
                [video_args.embedded_timestep, video_args.embedded_timestep], dim=1
            )

            return replace(
                video_args,
                x=combined_x,
                positional_embeddings=combined_pe,
                timesteps=combined_ts,
                embedded_timestep=combined_emb_ts,
            )
        else:
            raise ValueError(
                f"Unknown decoder_input_combine: {self.decoder_input_combine}"
            )

    def forward_decoder_per_frame(
        self,
        video: Modality | None,
        encoder_features: Tensor | None,
        perturbations: BatchedPerturbationConfig | None,
        tokens_per_frame: int = 336,
        num_frames: int = 1,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run decoder one frame at a time, matching autoregressive inference.

        This prevents the train/inference attention scope mismatch that causes
        grid artifacts when the decoder LoRA is trained on multi-frame input
        but used on single-frame input at inference.

        Each frame is decoded independently: the decoder sees only that frame's
        noisy tokens and the corresponding shifted encoder features.

        Args:
            video: Full video modality with all frames' noisy tokens
            encoder_features: Shifted encoder features [B, total_seq, D]
            perturbations: Optional perturbation config
            tokens_per_frame: Spatial tokens per frame (H * W)
            num_frames: Number of frames in the video

        Returns:
            Tuple of (video_pred, audio_pred) concatenated across all frames
        """
        if video is None:
            return None, None

        all_video_preds = []
        tpf = tokens_per_frame

        for f in range(num_frames):
            start = f * tpf
            end = start + tpf

            # Extract single frame's data from full video modality
            frame_latent = video.latent[:, start:end, :]
            frame_timesteps = video.timesteps[:, start:end] if video.timesteps.ndim > 1 else video.timesteps

            # Extract frame's positional embeddings
            if video.positions is not None:
                frame_positions = video.positions[:, :, start:end, :]
            else:
                frame_positions = None

            frame_modality = Modality(
                enabled=video.enabled,
                latent=frame_latent,
                timesteps=frame_timesteps,
                positions=frame_positions,
                context=video.context,
                context_mask=video.context_mask,
            )

            # Extract frame's encoder features
            frame_enc = encoder_features[:, start:end, :] if encoder_features is not None else None

            # Decode single frame
            vx, ax = self.forward_decoder(
                video=frame_modality,
                encoder_features=frame_enc,
                audio=None,
                perturbations=perturbations,
            )

            if vx is not None:
                all_video_preds.append(vx)

        # Concatenate all frames
        video_pred = torch.cat(all_video_preds, dim=1) if all_video_preds else None
        return video_pred, None

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Standard forward pass — delegates to base model for backward compat."""
        return self.base_model(video, audio, perturbations)


def shift_encoder_features(
    encoder_features: Tensor,
    tokens_per_frame: int,
    num_frames: int,
) -> Tensor:
    """Shift encoder features by 1 frame for decoder input.

    In SCD, the encoder output for frame t-1 is used as context for decoding
    frame t. This is the causal conditioning: each frame is decoded using
    the temporal context from the preceding frame.

    Frame 0 gets zero features (no preceding context).

    Args:
        encoder_features: [B, total_seq, D] encoder output
        tokens_per_frame: Number of tokens per frame
        num_frames: Number of frames

    Returns:
        Shifted features [B, total_seq, D] where frame t has frame t-1's features
    """
    B, S, D = encoder_features.shape
    features = encoder_features.view(B, num_frames, tokens_per_frame, D)

    # Shift: frame t gets frame t-1's features; frame 0 gets zeros
    zeros = torch.zeros(B, 1, tokens_per_frame, D, device=features.device, dtype=features.dtype)
    shifted = torch.cat([zeros, features[:, :-1]], dim=1)

    return shifted.view(B, S, D)
