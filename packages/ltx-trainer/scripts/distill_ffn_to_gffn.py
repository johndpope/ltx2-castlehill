#!/usr/bin/env python
"""
Distill standard FFN → gFFN with wandb logging and VAE reconstruction.

Supports multiple gFFN variants:
  - hrr:      gFFN-HRR (Holographic Reduced Representations) — default
  - spectral: gFFN-Spectral (sparse FFT circular convolution) — smallest

Uses independent layer distillation: each gFFN layer learns to mimic its
teacher FFN independently, given the same pre-FFN input (captured via hooks).

Device layout (dual GPU):
  - cuda:0 (RTX 5090, 32GB): Student model (non-FFN shared weights + gFFN)
  - cuda:1 (RTX PRO 4000, 24GB): Teacher FFNs + VAE decoder

Loss = task_loss + kd_weight * kd_loss
  - task_loss: MSE(student_velocity, target_velocity) — flow matching objective
  - kd_loss: MSE(student_ffn_out, teacher_ffn_out) per layer, averaged

Usage:
    cd packages/ltx-trainer
    uv run python scripts/distill_ffn_to_gffn.py
    uv run python scripts/distill_ffn_to_gffn.py --variant spectral --num-freqs 512 --kd-layers-per-step 96
    uv run python scripts/distill_ffn_to_gffn.py --steps 5000 --lr 2e-4
    uv run python scripts/distill_ffn_to_gffn.py --no-wandb --recon-every 0
"""

import argparse
import gc
import logging
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-core" / "src"))
sys.path.insert(0, str(ROOT / "ltx-trainer" / "src"))

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
CHECKPOINT = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
DATA_DIR = "/media/2TB/omnitransfer/data/ditto_subset"

GFFN_DEFAULTS = dict(
    num_shifts=8,
    shift_strategy="log_uniform",
    proj_factor=4,
    mode="inner",
)


def parse_args():
    p = argparse.ArgumentParser(description="Distill FFN → gFFN (HRR or Spectral)")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--output-dir", default="outputs/gffn_distill")
    # Variant selection
    p.add_argument("--variant", choices=["hrr", "spectral"], default="hrr",
                    help="gFFN variant: hrr (default) or spectral")
    # Training
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    p.add_argument("--kd-weight", type=float, default=0.5, help="Weight for KD loss vs task loss")
    p.add_argument("--kd-layers-per-step", type=int, default=8,
                    help="Number of FFN layers to distill per step (0=disable KD, 96=all layers)")
    p.add_argument("--kd-loss-type", choices=["mse", "swd", "cosine", "swd+cosine"],
                    default="mse", help="KD loss: mse (pointwise), swd (distribution matching), "
                    "cosine (direction only), swd+cosine (combined)")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--keep-checkpoints", type=int, default=2, help="Max checkpoints to keep (0=all)")
    p.add_argument("--resume", type=str, default=None, help="Path to gFFN checkpoint to resume from")
    # Devices
    p.add_argument("--train-device", default="cuda:0", help="GPU for transformer (RTX 5090)")
    p.add_argument("--teacher-device", default="cuda:1", help="GPU for teacher FFNs + VAE (PRO 4000)")
    # Reconstruction
    p.add_argument("--recon-every", type=int, default=50, help="Reconstruct every N steps (0=disable)")
    p.add_argument("--checkpoint-every", type=int, default=500)
    # wandb
    p.add_argument("--wandb-project", default="gffn-distill")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    # Data subdirectories (within --data-dir)
    p.add_argument("--latents-subdir", default="latents_19b", help="Subdir for precomputed latents")
    p.add_argument("--conditions-subdir", default="conditions_final", help="Subdir for precomputed conditions")
    # gFFN-HRR config
    p.add_argument("--num-shifts", type=int, default=GFFN_DEFAULTS["num_shifts"])
    p.add_argument("--proj-factor", type=int, default=GFFN_DEFAULTS["proj_factor"])
    p.add_argument("--mode", default=GFFN_DEFAULTS["mode"])
    p.add_argument("--shift-strategy", default=GFFN_DEFAULTS["shift_strategy"])
    # gFFN-Spectral config
    p.add_argument("--num-freqs", type=int, default=128,
                    help="Number of frequency components for spectral variant (128, 256, 512)")
    p.add_argument("--use-phase-gate", action="store_true", default=True,
                    help="Use learned phase rotation in spectral variant")
    return p.parse_args()


# ── Hook-based FFN I/O collector ─────────────────────────────────────────────

class FFNIOCollector:
    """Forward hooks to capture per-layer FFN inputs AND outputs.

    We capture the input to compute teacher FFN output on a different device,
    and the output to compute KD loss against the teacher.
    """

    def __init__(self, model: nn.Module):
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self._hooks = []
        for idx, block in enumerate(model.transformer_blocks):
            if hasattr(block, "ff"):
                h = block.ff.register_forward_hook(self._make_hook(f"video.{idx}"))
                self._hooks.append(h)
            if hasattr(block, "audio_ff"):
                h = block.audio_ff.register_forward_hook(self._make_hook(f"audio.{idx}"))
                self._hooks.append(h)

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            # inp is a tuple; first element is the tensor input
            self.inputs[name] = inp[0].detach()
            self.outputs[name] = out
        return hook_fn

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def sliced_wasserstein_distance(x: torch.Tensor, y: torch.Tensor, n_projections: int = 128) -> torch.Tensor:
    """Sliced Wasserstein Distance between two batches of features.

    Projects high-dim distributions onto random 1D directions, computes
    1D Wasserstein (= sorted L1) on each, and averages. Differentiable
    via the sorting operation.

    Args:
        x: [N, D] student features (flattened over batch & sequence)
        y: [N, D] teacher features
        n_projections: number of random 1D projections

    Returns:
        Scalar SWD loss
    """
    D = x.shape[-1]
    device = x.device

    # Random projection directions (unit vectors)
    projections = torch.randn(n_projections, D, device=device, dtype=x.dtype)
    projections = projections / projections.norm(dim=-1, keepdim=True)

    # Project both distributions: [N, D] @ [D, n_proj] → [N, n_proj]
    x_proj = x @ projections.T  # [N, n_proj]
    y_proj = y @ projections.T  # [N, n_proj]

    # Sort along sample dimension and compute L1 (= 1D Wasserstein)
    x_sorted = x_proj.sort(dim=0).values
    y_sorted = y_proj.sort(dim=0).values

    return (x_sorted - y_sorted).abs().mean()


def cosine_distance_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cosine distance loss — ignores magnitude, matches direction only.

    Useful for frequency-domain outputs where magnitudes may differ
    but directional structure should match.

    Args:
        x: [N, D] student features
        y: [N, D] teacher features

    Returns:
        Scalar cosine distance (1 - cosine_similarity), averaged
    """
    cos_sim = nn.functional.cosine_similarity(x, y, dim=-1)
    return (1 - cos_sim).mean()


def compute_layer_kd_loss(student_out, teacher_out, kd_loss_type="mse"):
    """Compute KD loss between student and teacher FFN outputs.

    Args:
        student_out: [B, L, D] student FFN output (in computation graph)
        teacher_out: [B, L, D] teacher FFN output (detached, on student device)
        kd_loss_type: "mse", "swd", "cosine", or "swd+cosine"
    """
    if kd_loss_type == "mse":
        return nn.functional.mse_loss(student_out, teacher_out)

    # Flatten [B, L, D] → [B*L, D] for distribution matching
    s_flat = student_out.reshape(-1, student_out.shape[-1])
    t_flat = teacher_out.reshape(-1, teacher_out.shape[-1])

    if kd_loss_type == "swd":
        return sliced_wasserstein_distance(s_flat, t_flat)
    elif kd_loss_type == "cosine":
        return cosine_distance_loss(s_flat, t_flat)
    elif kd_loss_type == "swd+cosine":
        swd = sliced_wasserstein_distance(s_flat, t_flat)
        cos = cosine_distance_loss(s_flat, t_flat)
        return swd + cos
    else:
        raise ValueError(f"Unknown kd_loss_type: {kd_loss_type}")


class RotatingKDCollector:
    """Hooks a random subset of FFN layers each step for memory-efficient KD.

    Only `k` layers are hooked at a time, keeping GPU memory ~k/96 of full KD.
    Outputs stay in the computation graph (not detached) so gradients flow.
    Inputs are detached (only needed for teacher forward on another device).
    """

    def __init__(self, model: nn.Module, teacher_ffns: dict, k: int = 8,
                 kd_loss_type: str = "mse"):
        self.model = model
        self.teacher_ffns = teacher_ffns
        self.k = k
        self.kd_loss_type = kd_loss_type
        self._hooks = []
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}

        # Build list of all hookable (name, module) pairs
        self._all_layers = []
        for idx, block in enumerate(model.transformer_blocks):
            if hasattr(block, "ff") and f"video.{idx}" in teacher_ffns:
                self._all_layers.append((f"video.{idx}", block.ff))
            if hasattr(block, "audio_ff") and f"audio.{idx}" in teacher_ffns:
                self._all_layers.append((f"audio.{idx}", block.audio_ff))

    def register_random_subset(self):
        """Hook k random layers. Call before each forward pass."""
        self.remove()
        self.clear()
        subset = random.sample(self._all_layers, min(self.k, len(self._all_layers)))
        for name, module in subset:
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            self.inputs[name] = inp[0].detach()  # detach input (for teacher)
            self.outputs[name] = out              # keep in graph (for student grads)
        return hook_fn

    def compute_kd_loss(self, teacher_device):
        """Compute KD loss for hooked layers. Returns scalar on student device."""
        if not self.outputs:
            return torch.tensor(0.0)

        losses = []
        student_device = None
        for key in self.outputs:
            student_out = self.outputs[key]
            student_device = student_out.device

            # Run teacher on teacher_device
            ffn_input = self.inputs[key].to(teacher_device)
            with torch.no_grad():
                teacher_out = self.teacher_ffns[key](ffn_input)

            # Compute loss on student device, student_out stays in graph
            teacher_out_on_student = teacher_out.to(student_device)
            layer_loss = compute_layer_kd_loss(
                student_out, teacher_out_on_student, self.kd_loss_type
            )
            losses.append(layer_loss)

            del ffn_input, teacher_out, teacher_out_on_student

        return sum(losses) / len(losses)

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── FFN replacement ────────────────────────────────────────────────────────

def extract_teacher_ffns(model, teacher_device, dtype):
    """Extract teacher FFN modules from model BEFORE replacement. Move to teacher_device."""
    teacher_ffns = {}
    for idx, block in enumerate(model.transformer_blocks):
        if hasattr(block, "ff"):
            teacher_ffns[f"video.{idx}"] = block.ff.to(teacher_device, dtype)
            for p in teacher_ffns[f"video.{idx}"].parameters():
                p.requires_grad_(False)
        if hasattr(block, "audio_ff"):
            teacher_ffns[f"audio.{idx}"] = block.audio_ff.to(teacher_device, dtype)
            for p in teacher_ffns[f"audio.{idx}"].parameters():
                p.requires_grad_(False)
    return teacher_ffns


def replace_ffn_with_gffn(model, variant, gffn_kwargs, dtype=torch.bfloat16):
    """Replace all FeedForward modules with gFFN variant. Returns count."""
    from ltx_core.model.transformer.gffn import create_gffn

    count = 0
    for block in model.transformer_blocks:
        if hasattr(block, "ff"):
            old = block.ff
            dim = old.net[0].proj.in_features
            block.ff = create_gffn(dim=dim, dim_out=dim, variant=variant, **gffn_kwargs).to(dtype=dtype)
            del old
            count += 1
        if hasattr(block, "audio_ff"):
            old = block.audio_ff
            dim = old.net[0].proj.in_features
            block.audio_ff = create_gffn(dim=dim, dim_out=dim, variant=variant, **gffn_kwargs).to(dtype=dtype)
            del old
            count += 1
    return count


# ── Data loading ───────────────────────────────────────────────────────────

def build_dataloader(data_dir, batch_size, latents_subdir="latents_19b", conditions_subdir="conditions_final"):
    """Build a simple dataloader over precached latents + conditions."""
    from ltx_trainer.datasets import PrecomputedDataset

    dataset = PrecomputedDataset(
        data_root=data_dir,
        data_sources={latents_subdir: "latents", conditions_subdir: "conditions"},
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )


def batch_to_modality(batch, device, dtype):
    """Convert a PrecomputedDataset batch to Modality objects for the model."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)

    latents_dict = batch["latents"]
    video_latents_5d = latents_dict["latents"].to(device=device, dtype=dtype)  # [B, C, F, H, W]
    B, C, F, H, W = video_latents_5d.shape

    # Patchify: [B, C, F, H, W] → [B, seq_len, C]
    video_latents = patchifier.patchify(video_latents_5d)
    seq_len = video_latents.shape[1]

    # Add noise (flow matching)
    sigma = torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.99 + 0.01  # [0.01, 1.0]
    noise = torch.randn_like(video_latents)
    noisy = (1 - sigma) * video_latents + sigma * noise
    target = noise - video_latents  # velocity target

    # Per-token timesteps
    timesteps = sigma.squeeze(-1).expand(B, seq_len)

    # Positions
    fps = latents_dict.get("fps", None)
    fps_val = fps[0].item() if fps is not None else 24.0
    num_frames = latents_dict["num_frames"][0].item()
    height = latents_dict["height"][0].item()
    width = latents_dict["width"][0].item()

    shape = VideoLatentShape(frames=num_frames, height=height, width=width, batch=B, channels=C)
    latent_coords = patchifier.get_patch_grid_bounds(output_shape=shape, device=device)
    positions = get_pixel_coords(
        latent_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / fps_val

    # Context
    conditions = batch["conditions"]
    context = conditions["video_prompt_embeds"].to(device=device, dtype=dtype)
    mask = conditions.get("prompt_attention_mask", None)
    if mask is not None:
        mask = mask.to(device=device)

    video_mod = Modality(
        enabled=True, latent=noisy, timesteps=timesteps,
        positions=positions, context=context, context_mask=mask,
    )

    return video_mod, target, video_latents_5d, sigma.view(B)


# ── VAE Reconstruction ─────────────────────────────────────────────────────

class ReconstructionLogger:
    """Decodes latents → video frames on a separate GPU and logs to wandb."""

    def __init__(self, checkpoint_path, vae_device, wandb_run=None):
        self.vae_device = torch.device(vae_device)
        self.wandb_run = wandb_run
        self.vae = None
        self._load_vae(checkpoint_path)

    def _load_vae(self, checkpoint_path):
        from ltx_trainer.model_loader import load_video_vae_decoder
        print(f"  Loading VAE decoder on {self.vae_device}...")
        self.vae = load_video_vae_decoder(checkpoint_path, device=self.vae_device, dtype=torch.bfloat16)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        mem = torch.cuda.memory_allocated(self.vae_device) / 1e9
        print(f"  VAE loaded. GPU memory: {mem:.1f}GB")

    @torch.inference_mode()
    def log_reconstruction(self, pred_latent_5d, clean_latent_5d, step, sigma=None):
        """Decode predicted and clean latents, log comparison to wandb."""
        import wandb

        pred = pred_latent_5d[:1].to(device=self.vae_device, dtype=torch.bfloat16)
        clean = clean_latent_5d[:1].to(device=self.vae_device, dtype=torch.bfloat16)

        try:
            pred_video = self.vae(pred)
            clean_video = self.vae(clean)
        except Exception as e:
            print(f"  [recon] VAE decode failed: {e}")
            return

        pred_frames = ((pred_video[0] + 1) / 2).clamp(0, 1).cpu().float()
        clean_frames = ((clean_video[0] + 1) / 2).clamp(0, 1).cpu().float()

        F = pred_frames.shape[1]
        mid = F // 2

        pred_img = pred_frames[:, mid]
        clean_img = clean_frames[:, mid]
        comparison = torch.cat([clean_img, pred_img], dim=2)

        # Log image first (always works), then try video separately
        caption = f"Left: GT | Right: Pred (step {step}, σ={sigma:.3f})" if sigma is not None else f"Left: GT | Right: Pred (step {step})"
        if self.wandb_run is not None:
            self.wandb_run.log({
                "reconstruction/comparison": wandb.Image(comparison.permute(1, 2, 0).numpy(), caption=caption),
                "reconstruction/step": step,
            }, step=step)

        if F > 1 and self.wandb_run is not None:
            try:
                pred_video_uint8 = (pred_frames.permute(1, 0, 2, 3) * 255).to(torch.uint8)
                clean_video_uint8 = (clean_frames.permute(1, 0, 2, 3) * 255).to(torch.uint8)
                self.wandb_run.log({
                    "reconstruction/pred_video": wandb.Video(pred_video_uint8.numpy(), fps=4),
                    "reconstruction/clean_video": wandb.Video(clean_video_uint8.numpy(), fps=4),
                }, step=step)
            except Exception as e:
                print(f"  [recon] Video logging failed (install wandb[media]): {e}")

        print(f"  [recon] Logged reconstruction at step {step}")


# ── KD loss computation on teacher device ──────────────────────────────────

def compute_kd_loss(student_ffn_inputs, student_ffn_outputs, teacher_ffns, teacher_device):
    """Compute per-layer KD loss by running teacher FFNs on teacher_device.

    1. Send student FFN inputs to teacher_device
    2. Run through teacher FFN to get reference outputs
    3. Compare with student FFN outputs (also sent to teacher_device)
    4. Return scalar loss on train_device

    This keeps teacher FFNs permanently on teacher_device — no swapping needed.
    """
    losses = []
    for key in student_ffn_inputs:
        if key not in teacher_ffns:
            continue

        # Move student's FFN input to teacher device, run through teacher FFN
        ffn_input = student_ffn_inputs[key].to(teacher_device)
        with torch.no_grad():
            teacher_out = teacher_ffns[key](ffn_input)

        # Move student output to teacher device for comparison
        student_out = student_ffn_outputs[key].to(teacher_device)

        # MSE loss (computed on teacher device, returned as scalar)
        layer_loss = nn.functional.mse_loss(student_out, teacher_out)
        losses.append(layer_loss)

        del ffn_input, teacher_out, student_out

    if not losses:
        return torch.tensor(0.0)

    # Average across layers, move back to student's device
    return sum(losses) / len(losses)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    train_device = torch.device(args.train_device)
    teacher_device = torch.device(args.teacher_device)
    dtype = torch.bfloat16

    variant = args.variant
    if variant == "hrr":
        gffn_kwargs = dict(
            num_shifts=args.num_shifts,
            proj_factor=args.proj_factor,
            mode=args.mode,
            shift_strategy=args.shift_strategy,
        )
    elif variant == "spectral":
        gffn_kwargs = dict(
            num_freqs=args.num_freqs,
            use_phase_gate=args.use_phase_gate,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    variant_label = f"gFFN-{variant.upper()}"
    print("=" * 70)
    print(f"  {variant_label} Distillation Training")
    print("=" * 70)
    print(f"  Variant: {variant_label}")
    print(f"  Train device:   {train_device}")
    print(f"  Teacher device: {teacher_device}")
    print(f"  Steps: {args.steps}, LR: {args.lr}, KD weight: {args.kd_weight}, KD layers/step: {args.kd_layers_per_step}")
    print(f"  gFFN config: {gffn_kwargs}")

    # ── wandb ───────────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_tags = ["gffn-distill", f"variant-{variant}"]
        if variant == "hrr":
            wandb_tags += [f"shifts-{args.num_shifts}", f"pf-{args.proj_factor}"]
        elif variant == "spectral":
            wandb_tags += [f"freqs-{args.num_freqs}"]
        if args.kd_layers_per_step >= 96:
            wandb_tags.append("full-kd")
        if args.kd_loss_type != "mse":
            wandb_tags.append(f"kd-{args.kd_loss_type}")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "variant": variant,
                "gffn": gffn_kwargs,
                "lr": args.lr,
                "kd_weight": args.kd_weight,
                "kd_layers_per_step": args.kd_layers_per_step,
                "kd_loss_type": args.kd_loss_type,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "checkpoint": args.checkpoint,
                "train_device": str(train_device),
                "teacher_device": str(teacher_device),
            },
            tags=wandb_tags,
        )
        print(f"  wandb run: {wandb_run.url}")

    # ── Load model ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading model...")
    t0 = time.time()
    from ltx_trainer.model_loader import load_transformer
    model = load_transformer(args.checkpoint, device="cpu", dtype=dtype)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    teacher_ffn_params = sum(
        p.numel() for block in model.transformer_blocks
        for attr in ["ff", "audio_ff"] if hasattr(block, attr)
        for p in getattr(block, attr).parameters()
    )
    print(f"  Total: {total_params / 1e9:.2f}B, FFN: {teacher_ffn_params / 1e6:.0f}M params")

    # ── Extract teacher FFNs → teacher_device ──────────────────────────────
    print(f"\n[2/6] Extracting teacher FFNs to {teacher_device}...")
    t0 = time.time()
    teacher_ffns = extract_teacher_ffns(model, teacher_device, dtype)
    teacher_ffn_mem = torch.cuda.memory_allocated(teacher_device) / 1e9
    print(f"  {len(teacher_ffns)} teacher FFNs on {teacher_device}: {teacher_ffn_mem:.1f}GB")
    print(f"  Extracted in {time.time() - t0:.1f}s")

    # ── Replace FFN → gFFN (student) ────────────────────────────────────
    print(f"\n[3/6] Creating student (replace FFN → {variant_label})...")
    n_replaced = replace_ffn_with_gffn(model, variant, gffn_kwargs)

    student_ffn_params = sum(
        p.numel() for block in model.transformer_blocks
        for attr in ["ff", "audio_ff"] if hasattr(block, attr)
        for p in getattr(block, attr).parameters()
    )
    print(f"  Replaced {n_replaced} FFN modules")
    compression = teacher_ffn_params / max(student_ffn_params, 1)
    print(f"  Student {variant_label}: {student_ffn_params / 1e6:.1f}M params (was {teacher_ffn_params / 1e6:.0f}M, {compression:.0f}x reduction)")

    # ── Resume from checkpoint ─────────────────────────────────────────
    resume_step = 0
    if args.resume:
        print(f"\n  Loading gFFN weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["gffn_state_dict"], strict=False)
        resume_step = ckpt.get("step", 0)
        prev_loss = ckpt.get("loss", "?")
        print(f"  Resumed from step {resume_step}, prev loss={prev_loss}")
        print(f"  Loaded {len(ckpt['gffn_state_dict'])} keys, {len(missing)} missing, {len(unexpected)} unexpected")
        del ckpt

    # Move student to train device
    print(f"\n  Moving student to {train_device}...")
    gc.collect()
    model = model.to(train_device)
    model.train()

    # Freeze non-FFN, enable gFFN gradients
    for name, param in model.named_parameters():
        param.requires_grad_("ff." in name or "audio_ff." in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} params ({trainable / 1e6:.1f}M)")

    # Gradient checkpointing
    model.set_gradient_checkpointing(True)
    print(f"  Gradient checkpointing: enabled")

    if train_device.type == "cuda":
        student_mem = torch.cuda.memory_allocated(train_device) / 1e9
        print(f"  Student GPU mem: {student_mem:.1f}GB on {train_device}")

    # ── VAE reconstruction logger ───────────────────────────────────────────
    recon_logger = None
    if args.recon_every > 0:
        print(f"\n[4/6] Loading VAE decoder on {teacher_device}...")
        try:
            recon_logger = ReconstructionLogger(args.checkpoint, teacher_device, wandb_run)
            vae_mem = torch.cuda.memory_allocated(teacher_device) / 1e9
            print(f"  Total on {teacher_device}: {vae_mem:.1f}GB (teacher FFNs + VAE)")
        except Exception as e:
            print(f"  WARNING: Could not load VAE: {e}")
            print(f"  Continuing without reconstruction logging")
    else:
        print(f"\n[4/6] Skipping VAE (recon_every=0)")

    # ── Data ────────────────────────────────────────────────────────────────
    print(f"\n[5/6] Loading dataset from {args.data_dir}...")
    dataloader = build_dataloader(args.data_dir, args.batch_size, args.latents_subdir, args.conditions_subdir)
    print(f"  Samples: {len(dataloader.dataset)}")

    # ── KD collector + optimizer ────────────────────────────────────────────
    kd_collector = None
    if args.kd_layers_per_step > 0:
        kd_collector = RotatingKDCollector(
            model, teacher_ffns, k=args.kd_layers_per_step,
            kd_loss_type=args.kd_loss_type,
        )
        print(f"  Rotating KD: {args.kd_layers_per_step}/{len(kd_collector._all_layers)} layers per step, weight={args.kd_weight}, loss={args.kd_loss_type}")
    else:
        print(f"  KD disabled (--kd-layers-per-step 0)")

    kd_eval_every = 50  # Full KD eval every N steps (no_grad, monitoring only)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "muon":
        from torch.optim import Muon
        muon_params = [p for p in trainable if p.ndim >= 2]
        adamw_params = [p for p in trainable if p.ndim < 2]
        print(f"  Muon optimizer: {len(muon_params)} params (2D+), {len(adamw_params)} params (1D→AdamW)")
        # Muon only accepts 2D+ params; use separate AdamW for 1D (biases, norms)
        optimizer = Muon(muon_params, lr=args.lr, weight_decay=0.01)
        optimizer_1d = torch.optim.AdamW(adamw_params, lr=args.lr * 0.1, weight_decay=0.01) if adamw_params else None
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        optimizer_1d = None
        print(f"  AdamW optimizer: {len(trainable)} params")

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"\n[6/6] Starting distillation for {args.steps} steps...")
    kd_mode = f"rotating {args.kd_layers_per_step}-layer KD + task" if kd_collector else "task_loss only"
    print(f"  Strategy: {kd_mode}, full KD monitored every {kd_eval_every} steps")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = resume_step
    epoch = 0
    best_loss = float("inf")
    last_kd_loss = float("nan")

    while step < args.steps:
        epoch += 1
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            if step >= args.steps:
                break

            optimizer.zero_grad()
            if optimizer_1d is not None:
                optimizer_1d.zero_grad()

            # Convert batch to Modality
            try:
                video_mod, target, clean_latent_5d, sigmas = batch_to_modality(
                    batch, train_device, dtype
                )
            except Exception as e:
                print(f"  [step {step}] Batch conversion failed: {e}")
                continue

            # ── Register rotating KD hooks before forward ─────────────
            if kd_collector is not None:
                kd_collector.register_random_subset()

            # ── Student forward pass ─────────────────────────────────
            student_vid_out, _ = model(video=video_mod, audio=None, perturbations=None)

            # ── Combined loss ─────────────────────────────────────────
            task_loss = nn.functional.mse_loss(student_vid_out, target)

            if kd_collector is not None and kd_collector.outputs:
                kd_loss_val = kd_collector.compute_kd_loss(teacher_device)
                total_loss = task_loss + args.kd_weight * kd_loss_val
                last_kd_loss = kd_loss_val.item()
            else:
                total_loss = task_loss

            # Save detached copy before backward frees the graph
            if recon_logger and args.recon_every > 0 and (step + 1) % args.recon_every == 0:
                student_vid_out_detached = student_vid_out.detach().clone()
            else:
                student_vid_out_detached = None

            total_loss.backward()

            # Clean up hooks immediately after backward
            if kd_collector is not None:
                kd_collector.remove()
                kd_collector.clear()

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )

            optimizer.step()
            if optimizer_1d is not None:
                optimizer_1d.step()
            step += 1

            # ── Periodic full KD eval (no_grad, all layers) ───────────
            if step % kd_eval_every == 0:
                try:
                    full_collector = FFNIOCollector(model)
                    with torch.no_grad():
                        model(video=video_mod, audio=None, perturbations=None)
                    full_kd = compute_kd_loss(
                        full_collector.inputs, full_collector.outputs,
                        teacher_ffns, teacher_device,
                    )
                    last_kd_loss = full_kd.item()
                    full_collector.remove()
                    full_collector.clear()
                    del full_collector
                except Exception as e:
                    print(f"  [kd_eval] Failed at step {step}: {e}")

            # ── Logging ─────────────────────────────────────────────
            metrics = {
                "loss/task": task_loss.item(),
                "loss/total": total_loss.item(),
                "loss/kd": last_kd_loss,
                "train/step": step,
                "train/epoch": epoch,
                "train/sigma_mean": sigmas.mean().item(),
            }

            if step <= 10 or step % 10 == 0:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.requires_grad and p.grad is not None
                ) ** 0.5
                metrics["train/grad_norm"] = grad_norm
                if train_device.type == "cuda":
                    metrics["train/gpu_mem_gb"] = torch.cuda.memory_allocated(train_device) / 1e9

            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

            if task_loss.item() < best_loss:
                best_loss = task_loss.item()

            pbar.set_postfix(
                task=f"{task_loss.item():.4f}",
                kd=f"{last_kd_loss:.4f}",
                best=f"{best_loss:.4f}",
            )

            # ── Reconstruction ──────────────────────────────────────
            if student_vid_out_detached is not None:
                try:
                    from ltx_core.components.patchifiers import VideoLatentPatchifier
                    from ltx_core.types import VideoLatentShape

                    patchifier = VideoLatentPatchifier(patch_size=1)
                    B, C, F, H, W = clean_latent_5d.shape

                    sigma_view = sigmas.view(B, 1, 1)
                    pred_clean_patched = video_mod.latent.detach() - sigma_view * student_vid_out_detached

                    pred_clean_5d = patchifier.unpatchify(
                        pred_clean_patched,
                        output_shape=VideoLatentShape(batch=B, channels=C, frames=F, height=H, width=W),
                    )

                    recon_logger.log_reconstruction(
                        pred_clean_5d, clean_latent_5d,
                        step=step, sigma=sigmas.mean().item(),
                    )
                    del student_vid_out_detached
                except Exception as e:
                    import traceback
                    print(f"  [recon] Failed at step {step}: {e}")
                    traceback.print_exc()

            # ── Checkpoint ──────────────────────────────────────────
            if step % args.checkpoint_every == 0:
                ckpt_path = output_dir / f"gffn_step_{step:06d}.pt"
                gffn_state = {
                    k: v.cpu() for k, v in model.state_dict().items()
                    if "ff." in k or "audio_ff." in k
                }
                torch.save({
                    "gffn_state_dict": gffn_state,
                    "step": step,
                    "loss": task_loss.item(),
                    "variant": variant,
                    "config": gffn_kwargs,
                }, ckpt_path)
                print(f"  Checkpoint: {ckpt_path}")
                # Rotate old checkpoints
                if args.keep_checkpoints > 0:
                    existing = sorted(output_dir.glob("gffn_step_*.pt"))
                    while len(existing) > args.keep_checkpoints:
                        old = existing.pop(0)
                        old.unlink()
                        print(f"  Removed old checkpoint: {old.name}")

            if step % 50 == 0:
                gc.collect()

    # ── Final save ──────────────────────────────────────────────────────────
    final_path = output_dir / "gffn_final.pt"
    gffn_state = {
        k: v.cpu() for k, v in model.state_dict().items()
        if "ff." in k or "audio_ff." in k
    }
    torch.save({
        "gffn_state_dict": gffn_state,
        "step": step,
        "loss": best_loss,
        "variant": variant,
        "config": gffn_kwargs,
    }, final_path)

    print(f"\n{'='*70}")
    print(f"  Distillation complete!")
    print(f"  Steps: {step}, Best task loss: {best_loss:.6f}")
    print(f"  Last KD loss: {last_kd_loss:.6f}")
    print(f"  Final weights: {final_path}")
    print(f"{'='*70}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
