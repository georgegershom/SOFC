#!/usr/bin/env python3
"""
Generate an Abaqus-style side-by-side figure comparing MF-DL crack density prediction
against experimental SEM-derived crack density, with a shared colorbar and annotations.

Features
- 1x2 panels: (a) MF-DL Prediction, (b) SEM Experimental Data
- Shared vertical colorbar to the right
- Spatial correlation (Pearson r) annotation; optional hotspot identification accuracy
- 2D heatmap (default) or optional 3D surface rendering
- Robust input handling: .npy/.npz arrays or image files (png/jpg/tif)
- Optional synthetic demo mode if no inputs are supplied
- Color scale and thresholds configurable via CLI

Dependencies: numpy, matplotlib (no heavy extras required)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ---------------------------- Utility Data Structures ----------------------------

@dataclass
class FigureInputs:
    predicted: np.ndarray  # shape (H, W), float
    experimental: np.ndarray  # shape (H, W), float
    mask: Optional[np.ndarray] = None  # optional mask of shape (H, W), bool


# ---------------------------- IO Utilities ----------------------------

def load_array(path: str) -> np.ndarray:
    """Load a 2D array from .npy/.npz or an image (png/jpg/tif).

    Returns float32 array in range as-is from source; image inputs are converted to grayscale if needed.
    """
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        return ensure_2d_float(arr)
    if path.lower().endswith(".npz"):
        data = np.load(path)
        # Try common keys; else take the first array
        for key in ("arr", "data", "array", "x"):
            if key in data:
                return ensure_2d_float(data[key])
        # Fallback to first item
        first_key = list(data.keys())[0]
        return ensure_2d_float(data[first_key])

    # Image input via matplotlib
    img = plt.imread(path)
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        # RGB[A] to grayscale using luminance weights
        rgb = img[..., :3]
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image dimensions for {path}: {img.shape}")
    return ensure_2d_float(gray)


def ensure_2d_float(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D float32.

    - If 3D with singleton channels, squeeze to 2D
    - If >2D, attempt to reduce by taking the first slice
    """
    if arr.ndim > 2:
        # Try to squeeze channels
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            arr = squeezed
        else:
            # Take first slice along the first axis as a best-effort fallback
            arr = np.take(arr, indices=0, axis=0)
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Could not coerce array to 2D, shape after squeeze: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


# ---------------------------- Resizing ----------------------------

def resize_array_linear(source: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize 2D array to target_shape using separable 1D linear interpolation with numpy only.

    This is a light-weight alternative to skimage/Opencv, avoiding extra dependencies.
    """
    src_h, src_w = source.shape
    tgt_h, tgt_w = target_shape
    if (src_h, src_w) == (tgt_h, tgt_w):
        return source

    # Interpolate rows to target width
    # Map target x positions to source index space [0, src_w-1]
    x_src = np.linspace(0.0, src_w - 1.0, num=src_w, dtype=np.float64)
    x_tgt = np.linspace(0.0, src_w - 1.0, num=tgt_w, dtype=np.float64)
    temp = np.empty((src_h, tgt_w), dtype=np.float32)
    for i in range(src_h):
        temp[i, :] = np.interp(x_tgt, x_src, source[i, :].astype(np.float64))

    # Interpolate columns to target height
    y_src = np.linspace(0.0, src_h - 1.0, num=src_h, dtype=np.float64)
    y_tgt = np.linspace(0.0, src_h - 1.0, num=tgt_h, dtype=np.float64)
    result = np.empty((tgt_h, tgt_w), dtype=np.float32)
    for j in range(tgt_w):
        result[:, j] = np.interp(y_tgt, y_src, temp[:, j].astype(np.float64))

    return result


# ---------------------------- Metrics ----------------------------

def compute_pearson_r(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Pearson correlation across flattened arrays, with optional mask."""
    if mask is not None:
        valid = mask.astype(bool)
    else:
        valid = np.ones_like(a, dtype=bool)
    a_flat = a[valid].ravel()
    b_flat = b[valid].ravel()
    if a_flat.size < 2:
        return float("nan")
    # Subtract means to improve numerical stability if nearly constant
    a_center = a_flat - np.mean(a_flat)
    b_center = b_flat - np.mean(b_flat)
    denom = (np.linalg.norm(a_center) * np.linalg.norm(b_center))
    if denom == 0:
        return float("nan")
    r = float(np.dot(a_center, b_center) / denom)
    return r


def compute_hotspot_stats(
    predicted: np.ndarray,
    experimental: np.ndarray,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute hotspot IoU, F1-score, and accuracy for predicted vs experimental >= threshold.

    Returns (iou, f1, accuracy) in [0,1].
    """
    pred_hot = predicted >= threshold
    exp_hot = experimental >= threshold

    if mask is not None:
        valid = mask.astype(bool)
        pred_hot = pred_hot & valid
        exp_hot = exp_hot & valid

    tp = np.logical_and(pred_hot, exp_hot).sum(dtype=np.float64)
    fp = np.logical_and(pred_hot, ~exp_hot).sum(dtype=np.float64)
    fn = np.logical_and(~pred_hot, exp_hot).sum(dtype=np.float64)
    tn = np.logical_and(~pred_hot, ~exp_hot).sum(dtype=np.float64)

    denom_iou = (tp + fp + fn)
    iou = tp / denom_iou if denom_iou > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom_f1 = (precision + recall)
    f1 = (2.0 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return float(iou), float(f1), float(accuracy)


# ---------------------------- Demo Data ----------------------------

def generate_value_noise(height: int, width: int, grid: int = 16, seed: int = 0) -> np.ndarray:
    """Generate smooth value noise (microstructure-like texture) via bilinear interp on coarse grid."""
    rng = np.random.default_rng(seed)
    gh = max(2, height // grid)
    gw = max(2, width // grid)
    coarse = rng.random((gh, gw)).astype(np.float32)

    # Upsample to (height, width) using our linear resizer
    return resize_array_linear(coarse, (height, width))


def generate_demo_inputs(shape: Tuple[int, int] = (256, 384), seed: int = 42) -> FigureInputs:
    """Create synthetic predicted and experimental crack density maps with realistic hotspots.

    Hotspots concentrated near bottom interface and around synthetic Ni clusters; experimental is a
    slightly noisy/morphed version to yield high spatial correlation.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Base microstructure texture (0..1)
    texture = generate_value_noise(h, w, grid=12, seed=seed)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

    # Interface stress field near bottom (y close to h-1)
    y = np.linspace(0.0, 1.0, num=h, dtype=np.float32)[:, None]
    interface_bias = np.exp(-((1.0 - y) * 8.0))  # stronger near bottom

    # Ni cluster hotspots: sample centers and add Gaussian-like blobs (approximate with distance falloff)
    num_clusters = 8
    centers = rng.integers(low=[h // 4, w // 6], high=[h - h // 8, w - w // 6], size=(num_clusters, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    cluster_field = np.zeros((h, w), dtype=np.float32)
    for cy, cx in centers:
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-dist2 / float((h * 0.06) ** 2))
        cluster_field += blob.astype(np.float32)

    # Combine fields to create predicted crack density
    base_level = 0.0006
    scale = 0.0055
    predicted = base_level + scale * (
        0.45 * texture + 0.35 * interface_bias + 0.20 * cluster_field / (cluster_field.max() + 1e-8)
    )

    # Experimental: add mild blur-like smoothing and noise + tiny warp
    noise = (rng.normal(0, 0.0003, size=(h, w))).astype(np.float32)
    # Slight warp via resampling on a jittered grid
    jitter_amp = 0.8  # pixels
    jx = np.clip((xx + rng.normal(0, jitter_amp, size=(h, w))).astype(np.float32), 0, w - 1)
    jy = np.clip((yy + rng.normal(0, jitter_amp, size=(h, w))).astype(np.float32), 0, h - 1)

    # Sample predicted at jittered coords using separable interp (approximate by two passes)
    # First interpolate horizontally at integer rows
    experimental = np.empty_like(predicted)
    x_src = np.arange(w, dtype=np.float32)
    for i in range(h):
        experimental[i, :] = np.interp(jx[i, :], x_src, predicted[i, :])
    # Then vertical pass by columns
    y_src = np.arange(h, dtype=np.float32)
    temp = experimental.copy()
    for j in range(w):
        experimental[:, j] = np.interp(jy[:, j], y_src, temp[:, j])

    experimental = experimental + noise

    # Ensure non-negative and limit range
    experimental = np.clip(experimental, 0.0, None)

    # Optional mask (valid everywhere in demo)
    mask = np.ones((h, w), dtype=bool)
    return FigureInputs(predicted=predicted.astype(np.float32), experimental=experimental.astype(np.float32), mask=mask)


# ---------------------------- Plotting ----------------------------

def determine_vmin_vmax(a: np.ndarray, b: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> Tuple[float, float]:
    if vmin is not None and vmax is not None and vmax > vmin:
        return float(vmin), float(vmax)
    # Use robust bounds across both arrays
    combined = np.concatenate([a.ravel(), b.ravel()])
    lo = float(np.percentile(combined, 1.0))
    hi = float(np.percentile(combined, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Fallback absolute bounds
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def make_abaqus_style_figure(
    inputs: FigureInputs,
    output_path: str,
    panel_titles: Tuple[str, str] = ("(a) MF-DL Prediction", "(b) SEM Experimental Data"),
    colormap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mode: str = "2d",
    threshold: float = 0.005,
    annotate_corr: Optional[float] = None,
    annotate_hotspot_metric: Optional[str] = "accuracy",  # one of {"accuracy", "iou", "f1", None}
    dpi: int = 300,
) -> None:
    predicted = inputs.predicted
    experimental = inputs.experimental
    mask = inputs.mask

    # Resize experimental to predicted shape if needed
    if experimental.shape != predicted.shape:
        experimental = resize_array_linear(experimental, predicted.shape)
        if mask is not None and mask.shape != predicted.shape:
            mask = resize_array_linear(mask.astype(np.float32), predicted.shape) >= 0.5

    r = compute_pearson_r(predicted, experimental, mask)

    iou, f1, acc = compute_hotspot_stats(predicted, experimental, threshold=threshold, mask=mask)
    if annotate_hotspot_metric is None:
        hotspot_text = None
    elif annotate_hotspot_metric.lower() == "iou":
        hotspot_text = f"Hotspot IoU: {iou * 100:.0f}%"
    elif annotate_hotspot_metric.lower() == "f1":
        hotspot_text = f"Hotspot F1: {f1 * 100:.0f}%"
    else:
        hotspot_text = f"Hotspot Identification Accuracy: {acc * 100:.0f}%"

    # Determine color scale
    vmin_val, vmax_val = determine_vmin_vmax(predicted, experimental, vmin, vmax)
    norm = Normalize(vmin=vmin_val, vmax=vmax_val)

    # Create figure with shared colorbar on the right
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",  # widely available; close to Arial
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # Layout with gridspec: two main axes + one colorbar axis
    fig = plt.figure(figsize=(10, 4), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.0, 1.0, 0.04], wspace=0.05)

    # Colormap
    cmap = plt.get_cmap(colormap)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0], projection=None if mode == "2d" else "3d")
    # Panel B
    ax2 = fig.add_subplot(gs[0, 1], projection=None if mode == "2d" else "3d")

    if mode == "3d":
        # Build surface grid
        h, w = predicted.shape
        yy, xx = np.mgrid[0:h, 0:w]
        # Prediction surface
        surf1 = ax1.plot_surface(
            xx, yy, predicted, rstride=2, cstride=2, cmap=cmap, norm=norm, linewidth=0, antialiased=True
        )
        ax1.set_title(panel_titles[0])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_box_aspect((w, h, (vmax_val - vmin_val) * 4000.0))

        # Experimental surface
        surf2 = ax2.plot_surface(
            xx, yy, experimental, rstride=2, cstride=2, cmap=cmap, norm=norm, linewidth=0, antialiased=True
        )
        ax2.set_title(panel_titles[1])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_box_aspect((w, h, (vmax_val - vmin_val) * 4000.0))

        # Colorbar from the first mappable
        cax = fig.add_subplot(gs[0, 2])
        cbar = fig.colorbar(surf1, cax=cax)
    else:
        im1 = ax1.imshow(predicted, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        ax1.set_title(panel_titles[0])
        ax1.axis("off")

        im2 = ax2.imshow(experimental, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        ax2.set_title(panel_titles[1])
        ax2.axis("off")

        cax = fig.add_subplot(gs[0, 2])
        cbar = fig.colorbar(im1, cax=cax)

    cbar.set_label("Crack Density, CF3 072 061 063 06B (\u00B5m/\u00B5m\u00B2)")  # ρ_crack
    # The above label uses a safe representation; if your environment supports, you can use: "Crack Density, ρ_crack (µm/µm²)"

    # Annotations in figure space
    corr_to_show = annotate_corr if annotate_corr is not None else r
    annotation_lines = [f"Spatial Correlation = {corr_to_show:.2f}"]
    if hotspot_text is not None:
        annotation_lines.append(hotspot_text)

    annotation = "\n".join(annotation_lines)
    fig.text(0.5, 0.02, annotation, ha="center", va="bottom", fontsize=12)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- CLI ----------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Abaqus-style MF-DL vs SEM crack density comparison figure")

    p.add_argument("--pred", type=str, default=None, help="Path to predicted crack density array/image (.npy/.npz/png/jpg/tif)")
    p.add_argument("--exp", type=str, default=None, help="Path to experimental crack density array/image (.npy/.npz/png/jpg/tif)")
    p.add_argument("--mask", type=str, default=None, help="Optional path to boolean mask array (.npy/.npz)")

    p.add_argument("--output", type=str, default="figures/spatial_accuracy.png", help="Output figure path")
    p.add_argument("--colormap", type=str, default="jet", help="Matplotlib colormap (e.g., jet, turbo, viridis)")
    p.add_argument("--vmin", type=float, default=None, help="Color scale minimum (if omitted, robust min is used)")
    p.add_argument("--vmax", type=float, default=None, help="Color scale maximum (if omitted, robust max is used)")
    p.add_argument("--mode", type=str, default="2d", choices=["2d", "3d"], help="Plot as 2D map or 3D surface")

    p.add_argument("--threshold", type=float, default=0.005, help="Hotspot threshold in µm/µm²")
    p.add_argument("--annotate-corr", type=float, default=None, help="Override correlation shown (e.g., 0.98)")
    p.add_argument(
        "--hotspot-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "iou", "f1", "none"],
        help="Which hotspot metric to print (or none)",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI")

    p.add_argument("--demo", action="store_true", help="Generate a synthetic demo if no inputs are provided")
    p.add_argument("--demo-shape", type=str, default="256x384", help="HxW for demo data, e.g., 256x384")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.pred is None or args.exp is None:
        if not args.demo:
            print("[ERROR] Please provide --pred and --exp, or use --demo for synthetic example.", file=sys.stderr)
            return 2

    if args.demo and (args.pred is None or args.exp is None):
        try:
            h_str, w_str = args.demo_shape.lower().split("x")
            h, w = int(h_str), int(w_str)
        except Exception:
            h, w = 256, 384
        inputs = generate_demo_inputs(shape=(h, w), seed=42)
    else:
        predicted = load_array(args.pred)
        experimental = load_array(args.exp)
        if args.mask is not None:
            mask_arr = load_array(args.mask)
            mask = mask_arr >= 0.5
        else:
            mask = None
        inputs = FigureInputs(predicted=predicted, experimental=experimental, mask=mask)

    annotate_hotspot_metric = None if args.hotspot_metric == "none" else args.hotspot_metric

    make_abaqus_style_figure(
        inputs,
        output_path=args.output,
        panel_titles=("(a) MF-DL Prediction", "(b) SEM Experimental Data"),
        colormap=args.colormap,
        vmin=args.vmin,
        vmax=args.vmax,
        mode=args.mode,
        threshold=args.threshold,
        annotate_corr=args.annotate_corr,
        annotate_hotspot_metric=annotate_hotspot_metric,
        dpi=args.dpi,
    )

    print(f"[OK] Figure saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
