from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from skimage.segmentation import slic, find_boundaries


@dataclass
class Microstructure:
    labels: np.ndarray  # int32 labels, starting at 1
    boundary_mask: np.ndarray  # bool
    initial_porosity: np.ndarray  # bool
    voxel_size_um: float


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_grain_labels(
    shape: Tuple[int, int, int], approx_num_grains: int, seed: int | None = None
) -> np.ndarray:
    """Generate 3D polycrystalline-like labels using SLIC supervoxels.

    This approximates grains for synthetic data without heavy Voronoi tessellation.
    """
    z, y, x = shape
    gen = _rng(seed)

    base_noise = gen.normal(0.0, 1.0, size=shape).astype(np.float32)
    # Smooth to create low-frequency texture for SLIC to respect
    smooth_noise = gaussian_filter(base_noise, sigma=3.0)

    # skimage.slic supports nD; treat volume as grayscale (channel_axis=None)
    labels = slic(
        smooth_noise,
        n_segments=int(max(8, approx_num_grains)),
        compactness=0.2,
        max_num_iter=10,
        start_label=1,
        channel_axis=None,
        slic_zero=True,
    ).astype(np.int32)

    return labels


def compute_boundary_mask(labels: np.ndarray) -> np.ndarray:
    return find_boundaries(labels, mode="outer")


def generate_initial_porosity(
    shape: Tuple[int, int, int],
    target_porosity_fraction: float,
    boundary_mask: np.ndarray,
    voxel_size_um: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate initial pores biased to grain boundaries.

    Uses low-frequency noise thresholding plus a boundary-weighted bias function.
    """
    gen = _rng(seed)
    base = gen.normal(0.0, 1.0, size=shape).astype(np.float32)
    low_freq = gaussian_filter(base, sigma=2.5)

    # Boundary bias: increase pore probability near boundaries
    boundary_blur = gaussian_filter(boundary_mask.astype(np.float32), sigma=1.2)
    score = 0.8 * low_freq + 1.5 * boundary_blur

    # Determine threshold achieving approximate target fraction
    flat = score.ravel()
    kth = int((1.0 - float(np.clip(target_porosity_fraction, 0.0, 0.5))) * flat.size)
    kth = max(1, min(kth, flat.size - 1))
    thr = np.partition(flat, kth)[kth]
    pores = score > thr

    # Shape pores into more rounded cavities roughly scaled by voxel size
    radius_vox = max(1, int(round(2.0 / max(1e-6, voxel_size_um))))
    if radius_vox > 1:
        structure = _ellipsoid_structure(radius_vox, radius_vox, radius_vox)
        pores = binary_erosion(pores, structure)
        pores = binary_dilation(pores, structure)

    return pores


def _ellipsoid_structure(rx: int, ry: int, rz: int) -> np.ndarray:
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return (xx * xx) / (rx * rx + 1e-6) + (yy * yy) / (ry * ry + 1e-6) + (zz * zz) / (rz * rz + 1e-6) <= 1.0
