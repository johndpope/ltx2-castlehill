#!/usr/bin/env python
"""
VFM Noise Distribution Illustrations
=====================================
1. Spherical Cauchy vs Gaussian noise on S^127 (projected to 2D/3D)
2. Per-token sigma effect: what broke and how to fix it

Usage:
    python scripts/plot_vfm_illustrations.py
    # Saves to /tmp/vfm_spherical_vs_gaussian.png and /tmp/vfm_per_token_sigma.png
"""

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0e27",
    "axes.facecolor": "#0a0e27",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "axes.edgecolor": "#333333",
    "font.family": "monospace",
    "font.size": 10,
})

CYAN = "#00d4ff"
MAGENTA = "#ff3399"
GOLD = "#ffaa00"
GREEN = "#44ff88"
RED = "#ff4444"
DIM = "#555555"


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Spherical Cauchy vs Gaussian noise distributions
# ═══════════════════════════════════════════════════════════════════════

def sample_gaussian_2d(mu, sigma, n=500):
    """Standard diagonal Gaussian in R^2."""
    eps = np.random.randn(n, 2)
    return mu + sigma * eps


def sample_spherical_cauchy_2d(mu_dir, kappa, n=500):
    """Approximate Spherical Cauchy on S^1 projected to 2D."""
    mu_dir = mu_dir / np.linalg.norm(mu_dir)
    # Angle of mu on unit circle
    mu_angle = np.arctan2(mu_dir[1], mu_dir[0])
    # Sample angles from wrapped Cauchy (1D analog of spherical Cauchy)
    u = np.random.uniform(0, 1, n)
    # Wrapped Cauchy: concentration rho = exp(-1/kappa)
    rho = np.exp(-1.0 / max(kappa, 0.01))
    # Inverse CDF of wrapped Cauchy
    angles = mu_angle + 2 * np.arctan(
        ((1 - rho) / (1 + rho)) * np.tan(np.pi * (u - 0.5))
    )
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)


def plot_spherical_vs_gaussian():
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    np.random.seed(42)
    mu_dir = np.array([0.7, 0.7])
    mu_dir = mu_dir / np.linalg.norm(mu_dir)

    # ── Panel A: Gaussian in R^2 (VFM paper) ──
    ax1 = fig.add_subplot(gs[0])
    sigma = 0.4
    samples_g = sample_gaussian_2d(mu_dir * 0.8, sigma, n=800)

    # Draw unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), color=DIM, linewidth=1, linestyle="--", alpha=0.5)

    ax1.scatter(samples_g[:, 0], samples_g[:, 1], s=3, c=CYAN, alpha=0.4, zorder=2)
    ax1.scatter(*mu_dir * 0.8, s=80, c=GOLD, marker="*", zorder=5, label="μ_φ(y)")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect("equal")
    ax1.set_title("VFM Paper: Gaussian in R^D", color=CYAN, fontsize=13, fontweight="bold")
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")

    # Annotations
    ax1.annotate("z ~ N(μ_φ, σ²I)", xy=(0.5, 0.02), xycoords="axes fraction",
                 fontsize=11, color=CYAN, ha="center", style="italic")
    ax1.annotate("Samples leak off sphere\n(unconstrained R^D)",
                 xy=(-1.5, 1.5), fontsize=8, color="#888888")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Panel B: Spherical Cauchy on S^1 (our v1f) ──
    ax2 = fig.add_subplot(gs[1])

    # Low kappa (broad)
    samples_low = sample_spherical_cauchy_2d(mu_dir, kappa=1.0, n=400)
    # High kappa (concentrated)
    samples_high = sample_spherical_cauchy_2d(mu_dir, kappa=10.0, n=400)

    ax2.plot(np.cos(theta), np.sin(theta), color=DIM, linewidth=1.5, linestyle="-", alpha=0.7)

    ax2.scatter(samples_low[:, 0], samples_low[:, 1], s=5, c=MAGENTA, alpha=0.3,
                zorder=2, label="κ=1 (broad)")
    ax2.scatter(samples_high[:, 0], samples_high[:, 1], s=5, c=GREEN, alpha=0.5,
                zorder=3, label="κ=10 (peaked)")
    ax2.scatter(*mu_dir, s=80, c=GOLD, marker="*", zorder=5, label="μ̂ (direction)")

    # Draw arrow for mu direction
    ax2.annotate("", xy=mu_dir * 0.9, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title("Our v1f: Spherical Cauchy on S^(D-1)", color=MAGENTA, fontsize=13, fontweight="bold")
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.annotate("z_dir ~ SpCauchy(μ̂, κ)\nz = ‖μ‖ · z_dir",
                 xy=(0.5, 0.02), xycoords="axes fraction",
                 fontsize=11, color=MAGENTA, ha="center", style="italic")
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.3)

    # ── Panel C: Comparison table ──
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    rows = [
        ["", "VFM Paper", "Our v1f"],
        ["Geometry", "Flat R^D", "Sphere S^(D-1)"],
        ["Distribution", "N(μ, σ²I)", "SpCauchy(μ̂, κ)"],
        ["KL target", "N(0, I)", "Uniform(S^(D-1))"],
        ["Tails", "Light (Gaussian)", "Heavy (Cauchy)"],
        ["Parameters", "μ ∈ R^D, σ ∈ R^D", "μ̂ ∈ S^(D-1), κ ∈ R₊, r ∈ R₊"],
        ["Decomposition", "Mean + scale", "Direction + concentration\n+ magnitude"],
        ["Early training", "Concentrated", "Broad exploration\n(heavy tails)"],
        ["KL formula", "Standard Gaussian KL", "(D-1)/2 · log(1 + 1/κ)"],
    ]

    table = ax3.table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_text_props(color="#e0e0e0")
        if r == 0:
            cell.set_facecolor("#1a2040")
            cell.set_text_props(fontweight="bold", color=GOLD)
        elif c == 0:
            cell.set_facecolor("#0f1530")
            cell.set_text_props(color="#aaaaaa", fontweight="bold")
        else:
            cell.set_facecolor("#0a0e27")
        cell.set_height(0.11)

    ax3.set_title("Comparison", color=GOLD, fontsize=13, fontweight="bold", pad=10)

    fig.suptitle("Noise Distribution: Gaussian (VFM paper) vs Spherical Cauchy (our v1f)",
                 fontsize=15, color="#ffffff", fontweight="bold", y=0.98)

    plt.savefig("/tmp/vfm_spherical_vs_gaussian.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved: /tmp/vfm_spherical_vs_gaussian.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Per-token sigma — what broke and how to fix
# ═══════════════════════════════════════════════════════════════════════

def plot_per_token_sigma():
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, hspace=0.4, wspace=0.35)

    np.random.seed(42)

    # ── Panel A: What happened (sigma collapse) ──
    ax_a = fig.add_subplot(gs[0, 0])
    steps = np.arange(0, 2500)
    # Simulate sigma collapse: starts ~0.5, collapses to sigma_min=0.05
    sigma_mean = 0.5 * np.exp(-steps / 300) + 0.05
    sigma_mean += np.random.randn(len(steps)) * 0.005

    ax_a.plot(steps, sigma_mean, color=RED, linewidth=1.5, alpha=0.8)
    ax_a.axhline(y=0.05, color=DIM, linestyle="--", linewidth=1, label="σ_min = 0.05")
    ax_a.axhline(y=1.0, color=CYAN, linestyle="--", linewidth=1, alpha=0.5, label="Inference t = 1.0")
    ax_a.fill_between(steps, 0.05, sigma_mean, alpha=0.1, color=RED)

    # Annotate the gap
    ax_a.annotate("", xy=(2200, 0.05), xytext=(2200, 1.0),
                  arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2))
    ax_a.annotate("20× gap!", xy=(2250, 0.5), fontsize=12, color=GOLD, fontweight="bold")

    ax_a.set_xlabel("Training Step")
    ax_a.set_ylabel("σ (noise level)")
    ax_a.set_title("A) What Happened: σ Collapsed", color=RED, fontsize=12, fontweight="bold")
    ax_a.set_ylim(0, 1.1)
    ax_a.legend(fontsize=8)

    # ── Panel B: Train vs Inference mismatch ──
    ax_b = fig.add_subplot(gs[0, 1])

    # Training: model sees σ ≈ 0.05 (almost clean)
    t_train = np.linspace(0, 1, 100)
    # Noise level distribution seen during training
    train_dist = np.exp(-((t_train - 0.05) ** 2) / (2 * 0.02**2))
    train_dist /= train_dist.max()
    # Inference: single point at t=1.0
    infer_x = [1.0]
    infer_y = [1.0]

    ax_b.fill_between(t_train, 0, train_dist, alpha=0.3, color=CYAN, label="Training σ distribution")
    ax_b.plot(t_train, train_dist, color=CYAN, linewidth=2)
    ax_b.scatter(infer_x, infer_y, s=200, c=RED, marker="v", zorder=5, label="Inference t=1.0")

    # Add "never seen!" annotation
    ax_b.annotate("Model never\nsaw this region!", xy=(0.7, 0.5),
                  fontsize=10, color=RED, fontweight="bold", ha="center",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0000", edgecolor=RED, alpha=0.8))

    ax_b.fill_between(t_train[t_train > 0.15], 0, 0.05, alpha=0.1, color=RED)
    ax_b.set_xlabel("Noise level σ / timestep t")
    ax_b.set_ylabel("Density / Frequency")
    ax_b.set_title("B) Train ↔ Inference Mismatch", color=GOLD, fontsize=12, fontweight="bold")
    ax_b.legend(fontsize=8)

    # ── Panel C: Token grid showing per-token sigma ──
    ax_c = fig.add_subplot(gs[0, 2])

    # Simulate a 8x8 grid of tokens with collapsed sigmas
    token_sigmas_collapsed = np.full((8, 8), 0.05) + np.random.randn(8, 8) * 0.005
    token_sigmas_collapsed = np.clip(token_sigmas_collapsed, 0.04, 0.07)

    im = ax_c.imshow(token_sigmas_collapsed, cmap="RdYlGn_r", vmin=0, vmax=1.0,
                     interpolation="nearest", aspect="equal")
    cbar = plt.colorbar(im, ax=ax_c, shrink=0.8)
    cbar.set_label("σ_i (per token)", color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#888888")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#888888")

    ax_c.set_title("C) Collapsed: All σ ≈ 0.05", color=RED, fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Token column")
    ax_c.set_ylabel("Token row")

    # ── Panel D: Fix Option 1 — Use SigmaHead at inference ──
    ax_d = fig.add_subplot(gs[1, 0])

    # Ideal: sigma varies per token AND is used at inference
    token_sigmas_ideal = np.random.beta(2, 3, (8, 8))  # Varied distribution
    token_sigmas_ideal = 0.2 + 0.7 * token_sigmas_ideal  # Range [0.2, 0.9]
    # Make it spatially correlated (smoother)
    from scipy.ndimage import gaussian_filter
    token_sigmas_ideal = gaussian_filter(token_sigmas_ideal, sigma=1.0)
    token_sigmas_ideal = np.clip(token_sigmas_ideal, 0.2, 0.9)

    im_d = ax_d.imshow(token_sigmas_ideal, cmap="viridis", vmin=0, vmax=1.0,
                       interpolation="nearest", aspect="equal")
    cbar_d = plt.colorbar(im_d, ax=ax_d, shrink=0.8)
    cbar_d.set_label("σ_i", color="#e0e0e0")
    cbar_d.ax.yaxis.set_tick_params(color="#888888")
    plt.setp(plt.getp(cbar_d.ax.axes, 'yticklabels'), color="#888888")

    ax_d.set_title("D) Fix: σ_i Used at BOTH\nTrain + Inference", color=GREEN, fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Token column")
    ax_d.set_ylabel("Token row")

    # ── Panel E: Fix Option 2 — Standard timestep (baseline) ──
    ax_e = fig.add_subplot(gs[1, 1])

    # Standard shifted_logit_normal covers full range
    t_vals = np.linspace(0.001, 0.999, 1000)
    # shifted_logit_normal: logit(t) ~ N(0, 1) then shift
    from scipy.stats import norm
    logit_t = np.log(t_vals / (1 - t_vals))
    std_dist = norm.pdf(logit_t, loc=0, scale=1) / (t_vals * (1 - t_vals))
    std_dist /= std_dist.max()

    ax_e.fill_between(t_vals, 0, std_dist, alpha=0.3, color=GREEN)
    ax_e.plot(t_vals, std_dist, color=GREEN, linewidth=2, label="shifted_logit_normal")
    ax_e.scatter([1.0], [0.3], s=200, c=GREEN, marker="v", zorder=5, label="Inference t=1.0 ✓")

    ax_e.annotate("Training covers t=1.0\n→ No mismatch!", xy=(0.75, 0.7),
                  fontsize=10, color=GREEN, fontweight="bold", ha="center",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#001a00", edgecolor=GREEN, alpha=0.8))

    ax_e.set_xlabel("Noise level σ / timestep t")
    ax_e.set_ylabel("Density")
    ax_e.set_title("E) Baseline Fix: Standard Timesteps\n(current training run)", color=GREEN, fontsize=12, fontweight="bold")
    ax_e.legend(fontsize=8)

    # ── Panel F: Architecture diagram ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    # Draw flow diagram
    boxes = [
        # (x, y, w, h, text, color)
        (0.05, 0.75, 0.25, 0.15, "Noise Adapter\nq_φ(z|y)", CYAN),
        (0.40, 0.82, 0.22, 0.10, "μ, log_σ", DIM),
        (0.40, 0.68, 0.22, 0.10, "SigmaHead\n(per-token σ_i)", RED),
        (0.72, 0.75, 0.25, 0.15, "DiT\nf_θ(z, σ, y)", MAGENTA),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            facecolor="#0f1530", edgecolor=color, linewidth=2
        )
        ax_f.add_patch(rect)
        ax_f.text(x + w/2, y + h/2, text, ha="center", va="center",
                  fontsize=8, color=color, fontweight="bold")

    # Arrows
    ax_f.annotate("", xy=(0.38, 0.85), xytext=(0.30, 0.83),
                  arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))
    ax_f.annotate("", xy=(0.38, 0.73), xytext=(0.30, 0.78),
                  arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))
    ax_f.annotate("", xy=(0.70, 0.83), xytext=(0.62, 0.85),
                  arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    # "BROKEN" label on SigmaHead
    ax_f.text(0.51, 0.60, "⚠ IGNORED at inference!", fontsize=9, color=RED,
              ha="center", fontweight="bold",
              bbox=dict(boxstyle="round", facecolor="#1a0000", edgecolor=RED, alpha=0.8))

    # Fix options text
    fix_text = """FIXES (from Grok research):

1. Use sigma_i at inference too
   -> Condition f_theta on per-token sigma_i
   -> Train & infer see same distribution

2. Drop per-token sigma entirely (baseline)
   -> Standard timestep sampling
   -> Same sigma for all tokens
   -> [OK] Currently training

3. Self-correcting loop (Self-Flow)
   -> Multi-step adaptive denoising
   -> Penalty on overconfident low-sigma
   -> sigma predictions drive sampler steps"""

    ax_f.text(0.02, 0.45, fix_text, fontsize=8, color="#cccccc",
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.05", facecolor="#0f1530",
                       edgecolor="#333333", alpha=0.9))

    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 1)
    ax_f.set_title("F) Per-Token σ Architecture + Fixes", color=GOLD, fontsize=12, fontweight="bold")

    fig.suptitle("Per-Token Sigma: What Broke and How to Fix It",
                 fontsize=16, color="#ffffff", fontweight="bold", y=0.98)

    plt.savefig("/tmp/vfm_per_token_sigma.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved: /tmp/vfm_per_token_sigma.png")


if __name__ == "__main__":
    plot_spherical_vs_gaussian()
    plot_per_token_sigma()
    print("\nDone! Check /tmp/vfm_spherical_vs_gaussian.png and /tmp/vfm_per_token_sigma.png")
