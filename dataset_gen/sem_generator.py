from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import SEM_IMAGE_SIZE, SEM_IMAGES_PER_SPECIMEN, SEM_PIXEL_SIZE_UM
from .utils import (
    draw_disks_mask,
    distance_to_disks,
    draw_random_cracks_2d,
    ensure_dir,
    fractal_noise_2d,
    write_pgm_grayscale,
)


def _seed(specimen_id: str) -> int:
    return abs(hash(("SEM", specimen_id))) % (2**32)


def generate_sem_for_specimen(
    out_dir: Path, specimen_id: str, params: Dict[str, float]
) -> Dict[str, float]:
    """
    Generate SEM-like grayscale images focusing on ITZ. Compute metrics.
    Returns dict of specimen-level metrics.
    """
    h, w = SEM_IMAGE_SIZE
    rng = np.random.default_rng(_seed(specimen_id))

    # Define synthetic inclusions: aggregates (larger, brighter), rubber (smaller, darker/voided)
    num_agg = 6
    num_rubber = 10
    agg = []
    rub = []
    for _ in range(num_agg):
        r = rng.uniform(25, 60)
        cx = rng.uniform(r + 2, w - r - 2)
        cy = rng.uniform(r + 2, h - r - 2)
        agg.append((cx, cy, r))
    for _ in range(num_rubber):
        r = rng.uniform(8, 18)
        cx = rng.uniform(r + 2, w - r - 2)
        cy = rng.uniform(r + 2, h - r - 2)
        rub.append((cx, cy, r))

    itz_width = 6 + int(10 * params["itz_deg"])  # pixels

    # ITZ mask around aggregates and rubber inclusions
    itz_mask = (
        (np.abs(distance_to_disks(h, w, agg)) <= itz_width)
        |
        (np.abs(distance_to_disks(h, w, rub)) <= itz_width)
    )

    # Base paste texture
    base = 0.65 + 0.35 * fractal_noise_2d((h, w), octaves=4, seed=_seed(specimen_id) + 1)

    # Initialize with paste base intensity (uint8 0-255)
    img = (180 * base).astype(np.float32)

    # Place aggregates (brighter)
    m_agg = draw_disks_mask(h, w, agg)
    img[m_agg] = 200 + 40 * fractal_noise_2d((h, w), seed=_seed(specimen_id) + 2)[m_agg]

    # Place rubber (darker / void-like with heating)
    m_rub = draw_disks_mask(h, w, rub)
    rubber_intensity = 120 - 100 * params["itz_deg"]
    img[m_rub] = rubber_intensity

    # Introduce pores and cracks
    cracks = draw_random_cracks_2d(h, w, density=params["crack_density"], seed=_seed(specimen_id) + 3)
    noise = fractal_noise_2d((h, w), octaves=3, seed=_seed(specimen_id) + 4)
    pore_thresh = 0.85 - 0.5 * params["micro_porosity"]
    pores = noise > pore_thresh

    # Emphasize porosity in ITZ
    pores |= (noise > (pore_thresh - 0.1)) & itz_mask

    # Darken pores and cracks
    img[pores] = np.minimum(img[pores], 60.0)
    img[cracks] = 15.0

    # ITZ ring contrast
    itz_ring = itz_mask & (~m_agg) & (~m_rub)
    img[itz_ring] = np.minimum(img[itz_ring], 140.0)

    # Clip + cast
    img = np.clip(img + rng.normal(0, 2.0, size=img.shape), 0, 255).astype(np.uint8)

    # Save images and compute metrics across multiple views
    ensure_dir(out_dir)
    crack_px_total = 0
    itz_area_px_total = 0
    itz_pore_px_total = 0

    for i in range(SEM_IMAGES_PER_SPECIMEN):
        # For additional views, jitter texture and crack mask slightly
        jitter = rng.integers(-3, 4)
        roll_img = np.roll(np.roll(img, shift=jitter, axis=0), shift=-jitter, axis=1)
        view_path = out_dir / f"{specimen_id}_ITZ_{i+1:02d}.pgm"
        write_pgm_grayscale(roll_img, view_path)

        # Recompute masks on rolled grid
        if jitter != 0:
            jit_cracks = np.roll(np.roll(cracks, shift=jitter, axis=0), shift=-jitter, axis=1)
            jit_pores = np.roll(np.roll(pores, shift=jitter, axis=0), shift=-jitter, axis=1)
            jit_itz = np.roll(np.roll(itz_mask, shift=jitter, axis=0), shift=-jitter, axis=1)
        else:
            jit_cracks, jit_pores, jit_itz = cracks, pores, itz_mask

        crack_px_total += int(jit_cracks.sum())
        itz_area_px_total += int(jit_itz.sum())
        itz_pore_px_total += int((jit_pores & jit_itz).sum())

    pixel_len_um = SEM_PIXEL_SIZE_UM
    crack_length_mm = (crack_px_total * pixel_len_um) / 1000.0
    itz_porosity_fraction = (itz_pore_px_total / max(1, itz_area_px_total))

    return {
        "sem_crack_length_mm": float(crack_length_mm),
        "sem_itz_porosity": float(itz_porosity_fraction),
        "sem_itz_width_px": int(itz_width),
    }
