#!/usr/bin/env python3
"""
Generate side-by-side 2D contour plots comparing MF-DL prediction vs HF simulation
for von Mises stress (sigma_VM) in the SOFC anode-electrolyte cross-section after 5,000 h.

- Left: (a) MF-DL Prediction
- Right: (b) HF Simulation (Ground Truth)
- Shared vertical colorbar to the right
- Color range: 0..100 MPa with emphasis >80 MPa
- Visual elements: anode (top) with Ni clusters, electrolyte (bottom), high-stress band at interface
- Annotations: Spatial Correlation = 0.98, arrows for CTE Mismatch and Ni Cluster Stress Concentration
- Outputs: figures/stress_accuracy_mf_vs_hf.png and .svg

This script synthesizes plausible stress fields shaped to the description and aims to emulate
an ABAQUS-like look via contour styling and edge smoothing.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl

# --- Configuration ---
np.random.seed(42)
width_px = 640
height_px = 400
anode_fraction = 0.72  # top portion height fraction for anode
interface_row = int(anode_fraction * height_px)

sigma_min = 0.0
sigma_max = 100.0

# Choose scientific colormap similar to ABAQUS but modern
# We'll use 'plasma' which provides strong separation in higher range
cmap_name = "plasma"

# --- Synthetic geometry helpers ---
def generate_ni_clusters_mask(h: int, w: int, n_clusters: int = 35, r_mean: float = 10.0, r_std: float = 4.0) -> np.ndarray:
    """Return a boolean mask of Ni clusters within the anode region (top portion).
    We place random circular/elliptical inclusions.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=bool)

    # constrain to anode region only
    anode_h = interface_row

    for _ in range(n_clusters):
        cy = np.random.uniform(0.12 * anode_h, 0.95 * anode_h)
        cx = np.random.uniform(0.08 * w, 0.92 * w)
        ry = max(4.0, np.random.normal(r_mean, r_std))
        rx = max(4.0, np.random.normal(r_mean * 0.8, r_std * 0.6))
        theta = np.random.uniform(0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # elliptical distance
        dy = yy - cy
        dx = xx - cx
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        inside = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0
        mask |= inside

    # zero out electrolyte portion
    mask[interface_row:, :] = False
    return mask


def smooth_field(field: np.ndarray, ksize: int = 9, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-like smoothing using separable convolution."""
    # construct 1D gaussian kernel
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel /= kernel.sum()
    # convolve
    tmp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis=1, arr=field)
    smoothed = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis=0, arr=tmp)
    return smoothed


# --- Base stress field construction ---
H, W = height_px, width_px
Y, X = np.mgrid[0:H, 0:W]

# Base gradient: lower stress in anode bulk, higher near interface, then drop in electrolyte
base = np.zeros((H, W), dtype=float)

# Distance to interface (anode side positive towards interface)
# Normalize to [0, 1] across a band around the interface
interface_band = 18.0
anode_dist = np.clip((interface_row - Y) / interface_band, 0.0, 1.0)
electrolyte_dist = np.clip((Y - interface_row) / interface_band, 0.0, 1.0)

# High-stress band centered at interface
band_peak = 92.0
band_width = 1.0  # used as multiplier inside exp
band_profile = band_peak * np.exp(-((Y - interface_row) ** 2) / (2.0 * (band_width * interface_band) ** 2))

# Anode bulk slight gradient with horizontal undulations
anode_trend = 20.0 + 6.0 * np.sin(2 * np.pi * X / (W / 2.5)) * (anode_dist ** 0.6)
# Electrolyte more uniform but nonzero
electrolyte_trend = 18.0 - 6.0 * (electrolyte_dist ** 0.7)

base += band_profile
base[:interface_row, :] += anode_trend[:interface_row, :]
base[interface_row:, :] += electrolyte_trend[interface_row:, :]

# Add microstructural Ni clusters increasing local stress via inclusion effect
ni_mask = generate_ni_clusters_mask(H, W, n_clusters=42, r_mean=8.0, r_std=3.0)
clusters_field = np.zeros_like(base)
clusters_field[ni_mask] = 30.0
clusters_field = smooth_field(clusters_field, ksize=13, sigma=3.0)

# Combine fields
hf_field = base + clusters_field
hf_field = smooth_field(hf_field, ksize=7, sigma=1.6)

# Normalize and clip to [sigma_min, sigma_max]
hf_field = np.clip(hf_field, sigma_min, sigma_max)

# --- MF-DL field: perturb HF slightly but keep spatial correlation ~0.98 ---
noise = smooth_field(np.random.normal(scale=3.0, size=(H, W)), ksize=11, sigma=3.0)
# small bias drift to imitate learning bias
bias = 0.8 * np.tanh((X - W * 0.5) / (0.3 * W))

mf_field = hf_field + noise + bias
mf_field = np.clip(mf_field, sigma_min, sigma_max)

# Compute spatial correlation (Pearson r) across entire field
hf_flat = hf_field.ravel()
mf_flat = mf_field.ravel()

# subtract means
hf_mean = hf_flat.mean()
mf_mean = mf_flat.mean()
num = np.sum((hf_flat - hf_mean) * (mf_flat - mf_mean))
den = np.sqrt(np.sum((hf_flat - hf_mean) ** 2) * np.sum((mf_flat - mf_mean) ** 2))
spatial_corr = float(num / den)

# Adjust to target 0.98 by blending if necessary
TARGET_R = 0.98
if spatial_corr < TARGET_R:
    # Blend towards HF to raise correlation
    alpha = 0.25  # blend weight
    for _ in range(4):
        mf_field = (1 - alpha) * mf_field + alpha * hf_field
        mf_flat = mf_field.ravel()
        mf_mean = mf_flat.mean()
        num = np.sum((hf_flat - hf_mean) * (mf_flat - mf_mean))
        den = np.sqrt(np.sum((hf_flat - hf_mean) ** 2) * np.sum((mf_flat - mf_mean) ** 2))
        spatial_corr = float(num / den)
        if spatial_corr >= TARGET_R:
            break

# --- Figure ---
plt.rcParams.update({
    "figure.figsize": (10, 4.5),
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
})

fig, axes = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)

# Common contour levels
levels = np.linspace(sigma_min, sigma_max, 101)

# Choose colormap
cmap = plt.get_cmap(cmap_name)

# Emphasize high end by using power-law normalization
norm = mpl.colors.PowerNorm(gamma=0.9, vmin=sigma_min, vmax=sigma_max)

# Plot MF-DL
im0 = axes[0].contourf(mf_field, levels=levels, cmap=cmap, norm=norm, antialiased=True)
axes[0].axhline(interface_row, color="k", linewidth=1.2)
axes[0].set_title("(a) MF-DL Prediction")
axes[0].set_xlabel("X (arb. units)")
axes[0].set_ylabel("Y (arb. units)")

# Overlay subtle Ni cluster outlines for visual cue (not solid fill)
contours0 = axes[0].contour(ni_mask.astype(float), levels=[0.5], colors="white", linewidths=0.6, linestyles="--", alpha=0.6)

# Plot HF
im1 = axes[1].contourf(hf_field, levels=levels, cmap=cmap, norm=norm, antialiased=True)
axes[1].axhline(interface_row, color="k", linewidth=1.2)
axes[1].set_title("(b) HF Simulation (Ground Truth)")
axes[1].set_xlabel("X (arb. units)")

# Shared colorbar on the right
cbar = fig.colorbar(im1, ax=axes, location="right", shrink=0.92, pad=0.02)
cbar.set_label("von Mises Stress, $\\sigma_{VM}$ (MPa)")
cbar.set_ticks(np.linspace(sigma_min, sigma_max, 6))

# Add >80 MPa threshold highlight via contour overlay
for ax in axes:
    ax.contour(hf_field, levels=[80.0], colors="yellow", linewidths=1.0, alpha=0.8)

# Spatial correlation text centrally positioned below panels
fig.text(0.5, -0.02, f"Spatial Correlation = {spatial_corr:.2f}", ha="center", va="top", fontsize=12)

# Optional callouts
# Arrow 1: CTE Mismatch at interface mid-span
x_iface = W * 0.52
axes[1].annotate(
    "CTE Mismatch",
    xy=(x_iface, interface_row), xycoords="data",
    xytext=(W * 0.78, interface_row - 45), textcoords="data",
    arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", alpha=0.9),
    fontsize=9,
)

# Arrow 2: Ni cluster stress concentration in anode region
# Find a cluster location to point at
cluster_points = np.argwhere(ni_mask)
if cluster_points.size > 0:
    cy, cx = cluster_points[len(cluster_points) // 2]
    axes[0].annotate(
        "Ni Cluster Stress Concentration",
        xy=(cx, cy), xycoords="data",
        xytext=(cx + 80, cy - 60), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", alpha=0.9),
        fontsize=9,
    )

# Style refinements for ABAQUS-like feel
for ax in axes:
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)  # invert to have anode on top visually
    ax.grid(False)

fig.patch.set_facecolor("white")

# Export
png_path = "/workspace/figures/stress_accuracy_mf_vs_hf.png"
svg_path = "/workspace/figures/stress_accuracy_mf_vs_hf.svg"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(svg_path, dpi=300, bbox_inches="tight")

print(f"Saved figure to: {png_path}\n{svg_path}")
