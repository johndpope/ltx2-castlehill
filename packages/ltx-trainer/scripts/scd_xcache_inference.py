#!/usr/bin/env python3
# ruff: noqa: T201
"""Standalone SCD inference with X-Cache cross-chunk residual reuse.

Wraps `scd_inference.py` without modifying it. All extra args are stripped from
sys.argv before delegating, so scd_inference's own argparser is unaffected.

Usage:
    # Baseline (xcache off)
    python scripts/scd_xcache_inference.py \
        --cached-embedding /path/to/conditions.pt \
        --num-seconds 30 \
        --output /path/to/baseline_30s.mp4

    # X-Cache on with default thresholds
    python scripts/scd_xcache_inference.py \
        --xcache \
        --cached-embedding /path/to/conditions.pt \
        --num-seconds 30 \
        --output /path/to/xcache_30s.mp4

    # Aggressive (more skips, lower quality floor)
    python scripts/scd_xcache_inference.py \
        --xcache --xcache-tau-floor 0.93 --xcache-warmup 1 \
        --xcache-front-anchor 1 --xcache-max-staleness 4 \
        --num-seconds 60 \
        --output /path/to/xcache_60s_aggressive.mp4
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Strip x-cache args before scd_inference's argparser sees them.
_xcache_parser = argparse.ArgumentParser(add_help=False)
_xcache_parser.add_argument("--xcache", action="store_true",
    help="Enable cross-chunk residual cache during decoder inference.")
_xcache_parser.add_argument("--xcache-tau-floor", type=float, default=0.97)
_xcache_parser.add_argument("--xcache-margin", type=float, default=0.02)
_xcache_parser.add_argument("--xcache-ema-alpha", type=float, default=0.3)
_xcache_parser.add_argument("--xcache-tau-dev", type=float, default=2.0)
_xcache_parser.add_argument("--xcache-warmup", type=int, default=1)
_xcache_parser.add_argument("--xcache-front-anchor", type=int, default=1)
_xcache_parser.add_argument("--xcache-back-anchor", type=int, default=0)
_xcache_parser.add_argument("--xcache-max-staleness", type=int, default=0)
_xcache_parser.add_argument("--xcache-fingerprint-k", type=int, default=32)
_xcache_parser.add_argument("--xcache-step0-protection", action="store_true")
_xcache_parser.add_argument("--xcache-no-kv-protection", action="store_true",
    help="Disable kv_update_protection (default on in XCacheConfig).")
_xcache_parser.add_argument("--xcache-min-step", type=int, default=0,
    help="Only allow skips when step_in_frame >= this value (protects early "
         "noisy denoising steps where frame-to-frame similarity is structurally low).")
_xcache_parser.add_argument("--xcache-stats-json", type=str, default=None,
    help="If set, write final skip-rate stats as JSON to this path.")
_xcache_parser.add_argument("--xcache-diag", action="store_true",
    help="Collect cosine-similarity histogram and print per-(step,block) stats "
         "at the end. Useful to see what threshold WOULD fire.")
xcache_args, remaining = _xcache_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining


def _install_xcache_patch(args: argparse.Namespace) -> dict:
    """Monkey-patch LTXSCDModel.forward_decoder to interpose XCacheManager
    around the decoder block loop. Returns a stats dict updated in place.
    """
    import torch  # noqa: F401  (ensure torch is imported in this scope)
    from ltx_core.model.transformer import scd_model as scd_mod
    from ltx_core.model.transformer.xcache import XCacheConfig, XCacheManager

    cfg = XCacheConfig(
        enabled=True,
        warmup_chunks=args.xcache_warmup,
        front_anchor_blocks=args.xcache_front_anchor,
        back_anchor_blocks=args.xcache_back_anchor,
        tau_floor=args.xcache_tau_floor,
        margin=args.xcache_margin,
        ema_alpha=args.xcache_ema_alpha,
        tau_dev=args.xcache_tau_dev,
        max_staleness=args.xcache_max_staleness,
        step0_protection=args.xcache_step0_protection,
        kv_update_protection=not args.xcache_no_kv_protection,
        fingerprint_k=args.xcache_fingerprint_k,
    )

    orig_forward_decoder = scd_mod.LTXSCDModel.forward_decoder
    state = {
        "manager": None,
        "cfg": cfg,
        "last_sigma": None,    # detect frame boundary (sigma resets upward)
        "step_in_frame": -1,   # step index within current denoising trajectory
        "frame_count": 0,      # incremented on each new frame (treated as a "chunk")
        "min_step": args.xcache_min_step,
        "diag": args.xcache_diag,
        # Diagnostic: list of (frame_count, step_in_frame, block_idx, cos_sim, max_dev)
        "diag_log": [],
        "current_latent_shape": None,  # (F, H, W) inferred from Modality.positions
    }

    def _current_sigma(video_args) -> float | None:
        if video_args is None:
            return None
        try:
            return float(video_args.timesteps.flatten()[0].item())
        except Exception:
            return None

    def _infer_latent_shape(video) -> tuple[int, int, int] | None:
        """Infer (F, H, W) from a Modality's positions tensor.
        Modality.positions has shape (B, 3, T) with rows = (frame, h, w)."""
        if video is None or not hasattr(video, "positions") or video.positions is None:
            return None
        try:
            pos = video.positions
            if pos.ndim != 3 or pos.shape[1] < 3:
                return None
            # take batch 0; coords are integer-valued
            f = int(pos[0, 0].max().item()) + 1
            h = int(pos[0, 1].max().item()) + 1
            w = int(pos[0, 2].max().item()) + 1
            return (f, h, w)
        except Exception:
            return None

    def _patched_forward_decoder(
        self,
        video,
        encoder_features,
        audio,
        perturbations,
        encoder_audio_args=None,
        local_control=None,
        global_context=None,
        capture_attention_layers=None,
    ):
        # We need to interpose around `self.decoder_blocks` iteration. Simplest:
        # wrap each block's __call__ for this single forward, then unwrap.
        mgr: XCacheManager = state["manager"]

        # Detect step / frame boundaries from sigma trajectory
        sigma_now = _current_sigma(video)
        if sigma_now is not None:
            last = state["last_sigma"]
            # New frame when sigma jumps up (or on first call)
            if last is None or sigma_now > last + 1e-6:
                state["frame_count"] += 1
                state["step_in_frame"] = 0
                if mgr is not None:
                    # Each frame = one "chunk" for x-cache purposes (option a).
                    # KV-update protection is conservatively False here since we
                    # don't observe a true KV-cache reset from this call site.
                    mgr.reset_for_new_chunk(is_kv_update=False)
            else:
                state["step_in_frame"] += 1
            state["last_sigma"] = sigma_now

        if mgr is None or not state["cfg"].enabled:
            return orig_forward_decoder(
                self, video, encoder_features, audio, perturbations,
                encoder_audio_args=encoder_audio_args,
                local_control=local_control,
                global_context=global_context,
                capture_attention_layers=capture_attention_layers,
            )

        step_idx = state["step_in_frame"]
        # Refresh inferred latent shape for this forward (cheap, avoids stale (F,H,W))
        state["current_latent_shape"] = _infer_latent_shape(video)
        # Pre-import torch.nn.functional locally for diagnostic similarity
        import torch.nn.functional as _F

        def make_wrapper(block_idx, orig_call):
            def wrapper(video=None, audio=None, perturbations=None, **kw):
                if video is None:
                    return orig_call(video=video, audio=audio,
                                     perturbations=perturbations, **kw)
                x_in = video.x
                latent_shape = state["current_latent_shape"]
                # If x_in token-count doesn't match F*H*W (e.g. token_concat with
                # encoder features prepended), don't pass latent_shape — fingerprint
                # falls back to first-K tokens.
                effective_shape = latent_shape
                if latent_shape is not None:
                    F_, H_, W_ = latent_shape
                    if x_in.shape[1] != F_ * H_ * W_:
                        effective_shape = None
                fp = mgr.compute_fingerprint(x_in, latent_shape=effective_shape)

                # Diagnostic: record cos_sim against cached fingerprint when present
                if state["diag"]:
                    pkey = (step_idx, block_idx)
                    pstate = mgr._state.get(pkey)
                    if pstate is not None and pstate.fingerprint is not None:
                        try:
                            sim = _F.cosine_similarity(
                                fp.float(), pstate.fingerprint.float(), dim=-1
                            ).min().item()
                            abs_diff = (fp.float() - pstate.fingerprint.float()).abs()
                            mdev = (abs_diff.max() / (pstate.fingerprint.float().abs().mean() + 1e-8)).item()
                            state["diag_log"].append(
                                (state["frame_count"], step_idx, block_idx, sim, mdev)
                            )
                        except Exception:
                            pass

                # Step-aware gating: protect early denoising steps where the
                # noise dominates and frame-to-frame similarity is structurally low.
                allow = step_idx >= state["min_step"]
                if allow and mgr.should_skip(step_idx, block_idx, fp):
                    cached = mgr.get_cached_residual(step_idx, block_idx)
                    if cached is not None and cached.shape == x_in.shape:
                        new_video = replace(video, x=x_in + cached)
                        return new_video, audio
                    # Shape mismatch (e.g. token_concat changed seq_len) -> recompute
                elif not allow:
                    # Account: the call wasn't a real "decision"; don't poison
                    # the manager's denominator. We still update_cache below so
                    # later steps can compare.
                    pass
                out_video, out_audio = orig_call(
                    video=video, audio=audio, perturbations=perturbations, **kw)
                if out_video is not None and out_video.x.shape == x_in.shape:
                    residual = out_video.x - x_in
                    mgr.update_cache(step_idx, block_idx, residual, fp)
                return out_video, out_audio
            return wrapper

        # Install wrappers via __dict__ (per-instance override of __call__ is tricky;
        # we patch the bound `forward` attribute by replacing the block in the list
        # is not safe either. Instead, intercept by overriding the block's forward.)
        original_forwards = []
        for i, block in enumerate(self.decoder_blocks):
            original_forwards.append(block.forward)
            block.forward = make_wrapper(i, block.forward)  # type: ignore[assignment]

        try:
            return orig_forward_decoder(
                self, video, encoder_features, audio, perturbations,
                encoder_audio_args=encoder_audio_args,
                local_control=local_control,
                global_context=global_context,
                capture_attention_layers=capture_attention_layers,
            )
        finally:
            for block, orig_fwd in zip(self.decoder_blocks, original_forwards):
                block.forward = orig_fwd  # type: ignore[assignment]

    # Lazy manager construction: we need num_blocks + device, only known at first call.
    orig_forward_decoder_first = orig_forward_decoder

    def _bootstrapping_forward_decoder(self, video, *a, **kw):
        if state["manager"] is None and args.xcache:
            num_blocks = len(self.decoder_blocks)
            # Pull a sample param to get device
            device = next(self.parameters()).device
            # num_steps is unknown a priori; XCacheManager only uses it as a hint.
            # We pass a generous upper bound (50) — the per-key dict is keyed by
            # (step, block) and grows lazily, so over-provisioning is harmless.
            state["manager"] = XCacheManager(
                num_blocks=num_blocks,
                num_steps=50,
                config=state["cfg"],
                device=device,
            )
            print(f"[xcache] enabled — {num_blocks} blocks, "
                  f"tau_floor={state['cfg'].tau_floor}, "
                  f"warmup={state['cfg'].warmup_chunks}, "
                  f"front_anchor={state['cfg'].front_anchor_blocks}, "
                  f"max_staleness={state['cfg'].max_staleness}")
        return _patched_forward_decoder(self, video, *a, **kw)

    scd_mod.LTXSCDModel.forward_decoder = _bootstrapping_forward_decoder
    return state


def main() -> None:
    if not xcache_args.xcache:
        print("[xcache] flag not set — running vanilla scd_inference")
    else:
        state = _install_xcache_patch(xcache_args)

    # Delegate to the unmodified scd_inference.main()
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import scd_inference

    try:
        scd_inference.main()
    finally:
        if xcache_args.xcache:
            mgr = state["manager"]  # type: ignore[possibly-undefined]
            if mgr is not None:
                stats = {
                    "total_decisions": mgr.total_decisions,
                    "total_skips": mgr.total_skips,
                    "skip_rate": mgr.skip_rate,
                    "frames_seen": state["frame_count"],
                    "tau_floor": state["cfg"].tau_floor,
                    "warmup_chunks": state["cfg"].warmup_chunks,
                    "front_anchor_blocks": state["cfg"].front_anchor_blocks,
                    "max_staleness": state["cfg"].max_staleness,
                }
                if state["diag"] and state["diag_log"]:
                    import statistics
                    log = state["diag_log"]
                    sims = [r[3] for r in log]
                    devs = [r[4] for r in log]
                    print(f"\n[xcache-diag] {len(log)} comparisons recorded")
                    sims_sorted = sorted(sims)
                    pct = lambda p: sims_sorted[max(0, min(len(sims_sorted)-1, int(p*len(sims_sorted))))]
                    print(f"[xcache-diag] cos_sim  min={min(sims):.3f}  p10={pct(0.10):.3f}  "
                          f"p50={pct(0.50):.3f}  p90={pct(0.90):.3f}  max={max(sims):.3f}  "
                          f"mean={statistics.mean(sims):.3f}")
                    print(f"[xcache-diag] max_dev  mean={statistics.mean(devs):.2f}  "
                          f"max={max(devs):.2f}")
                    # what skip rate would each threshold give?
                    for tau in (0.99, 0.97, 0.95, 0.90, 0.80, 0.50):
                        hit = sum(1 for s in sims if s >= tau) / len(sims)
                        print(f"[xcache-diag] tau={tau:.2f} -> would-skip rate {hit:.1%}")
                    # Per-step breakdown (which steps are most cache-friendly?)
                    by_step: dict[int, list[float]] = {}
                    for _frame, st, _b, sim, _d in log:
                        by_step.setdefault(st, []).append(sim)
                    print("[xcache-diag] per-step mean cos_sim:")
                    for st in sorted(by_step):
                        ms = statistics.mean(by_step[st])
                        print(f"             step {st:2d}  n={len(by_step[st]):4d}  mean_sim={ms:.3f}")
                    stats["diag"] = {
                        "n": len(log),
                        "sim_mean": statistics.mean(sims),
                        "sim_min": min(sims),
                        "sim_max": max(sims),
                        "per_step_mean": {str(s): statistics.mean(by_step[s]) for s in by_step},
                    }
                print(f"\n[xcache] frames={stats['frames_seen']}  "
                      f"decisions={stats['total_decisions']}  "
                      f"skips={stats['total_skips']}  "
                      f"skip_rate={stats['skip_rate']:.3%}")
                if xcache_args.xcache_stats_json:
                    import json
                    Path(xcache_args.xcache_stats_json).write_text(json.dumps(stats, indent=2))
                    print(f"[xcache] stats -> {xcache_args.xcache_stats_json}")


if __name__ == "__main__":
    main()
