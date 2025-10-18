from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import MICRO_CT_VOLUME_SHAPE
from .utils import (
    bfs_connected_fraction,
    distance_to_spheres,
    draw_random_cracks_3d,
    ensure_dir,
    fractal_noise_3d,
    write_pgm_grayscale,
)


def _seed(specimen_id: str) -> int:
    return abs(hash(("CT", specimen_id))) % (2**32)


def generate_micro_ct_for_specimen(
    out_dir: Path, specimen_id: str, params: Dict[str, float]
) -> Dict[str, float]:
    """
    Generate a synthetic 3D micro-CT-like labeled volume and summary preview slices.
    Labels: 0=void/pore, 1=paste, 2=aggregate, 3=rubber/residue, 4=crack
    Returns specimen-level porosity/crack metrics.
    """
    shape = MICRO_CT_VOLUME_SHAPE
    z, y, x = shape
    rng = np.random.default_rng(_seed(specimen_id))

    volume = np.ones(shape, dtype=np.uint8)  # start as paste (1)

    # Random aggregates (spheres)
    num_agg = 4
    agg_spheres: List[Tuple[float, float, float, float]] = []
    for _ in range(num_agg):
        r = rng.uniform(8, 16)
        cx = rng.uniform(r + 2, x - r - 2)
        cy = rng.uniform(r + 2, y - r - 2)
        cz = rng.uniform(r + 2, z - r - 2)
        agg_spheres.append((cx, cy, cz, r))

    # Random rubber inclusions (smaller spheres)
    num_rub = 6
    rub_spheres: List[Tuple[float, float, float, float]] = []
    for _ in range(num_rub):
        r = rng.uniform(5, 10)
        cx = rng.uniform(r + 2, x - r - 2)
        cy = rng.uniform(r + 2, y - r - 2)
        cz = rng.uniform(r + 2, z - r - 2)
        rub_spheres.append((cx, cy, cz, r))

    # Paint aggregates and rubber
    dist_agg = distance_to_spheres(z, y, x, agg_spheres)
    dist_rub = distance_to_spheres(z, y, x, rub_spheres)
    volume[dist_agg <= 0] = 2
    volume[dist_rub <= 0] = 3

    # ITZ shell width grows with degradation
    itz_w = int(2 + 4 * params["itz_deg"])  # voxels

    # Porosity field (fractal noise) with threshold controlled by micro_porosity
    noise = fractal_noise_3d(shape, octaves=4, seed=_seed(specimen_id) + 1)
    pore_thresh = 0.92 - 0.6 * params["micro_porosity"]
    pores = noise > pore_thresh

    # Higher porosity in ITZ shells
    itz_shell = ((np.abs(dist_agg) <= itz_w) | (np.abs(dist_rub) <= itz_w))
    pores |= (noise > (pore_thresh - 0.08)) & itz_shell

    # Cracks as planar segments
    cracks = draw_random_cracks_3d(shape, density=params["crack_density"], seed=_seed(specimen_id) + 2)

    # Rubber degradation -> convert portion of rubber to voids as function of itz_deg
    rub_mask = (volume == 3)
    rub_to_void = (rng.random(size=shape) < (0.3 + 0.6 * params["itz_deg"])) & rub_mask

    # Apply pores and cracks respecting aggregates (aggregates stay solid)
    volume[(pores | rub_to_void) & (volume != 2)] = 0
    volume[cracks & (volume != 2)] = 4

    # Save central slices as previews
    ensure_dir(out_dir)
    zc, yc, xc = z // 2, y // 2, x // 2
    def label_to_gray(lbl: np.ndarray) -> np.ndarray:
        # Map labels to grayscale for visualization
        mapping = {0: 30, 1: 180, 2: 230, 3: 120, 4: 10}
        flat = np.vectorize(mapping.get, otypes=[np.uint8])(lbl)
        return flat.astype(np.uint8)

    write_pgm_grayscale(label_to_gray(volume[zc]), out_dir / f"{specimen_id}_slice_Z.pgm")
    write_pgm_grayscale(label_to_gray(volume[:, yc, :]), out_dir / f"{specimen_id}_slice_Y.pgm")
    write_pgm_grayscale(label_to_gray(volume[:, :, xc]), out_dir / f"{specimen_id}_slice_X.pgm")

    # Save raw labeled volume as .npy
    np.save(out_dir / f"{specimen_id}_volume.npy", volume)

    # Metrics
    total_vox = float(volume.size)
    void_frac = float((volume == 0).sum()) / total_vox
    crack_frac = float((volume == 4).sum()) / total_vox
    total_void_like = void_frac + crack_frac

    # Connectivity of pores (downsampled for efficiency)
    ds = max(1, int(round(max(shape) / 64)))
    vol_ds = volume[::ds, ::ds, ::ds]
    conn_frac = bfs_connected_fraction((vol_ds == 0))

    return {
        "ct_porosity": float(void_frac),
        "ct_crack_fraction": float(crack_frac),
        "ct_void_plus_crack": float(total_void_like),
        "ct_connected_porosity_frac": float(conn_frac),
        "ct_itz_width_vox": int(itz_w),
    }
