from typing import Dict, List
import numpy as np

from .config import VOXEL_SIZE_UM
from .utils import connected_components_2d, equivalent_diameter_from_pixel_count_2d


def generate_sem_stats(volume: np.ndarray, itz_damage_index: float, crack_factor: float, rng: np.random.Generator, num_fovs: int = 5) -> List[Dict[str, float]]:
    """Generate SEM-like 2D field-of-view metrics from 3D volume slices.
    Returns list of per-FOV dicts with pore size stats and crack density estimate.
    """
    zdim, ydim, xdim = volume.shape
    fovs: List[Dict[str, float]] = []

    # Choose random planes along z for simplicity
    zs = rng.integers(low=0, high=zdim, size=num_fovs)
    for zi in zs:
        slice2d = (volume[zi] > 0).astype(np.uint8)
        labels, sizes, bboxes = connected_components_2d(slice2d)
        if len(sizes) == 0:
            pore_diams = np.array([0.0])
        else:
            pore_diams = np.array([equivalent_diameter_from_pixel_count_2d(s, VOXEL_SIZE_UM) for s in sizes], dtype=float)

        # Crack density proxy: sum of major axis lengths of elongated components per mm^2 area
        crack_mm = 0.0
        area_mm2 = (slice2d.shape[0] * VOXEL_SIZE_UM / 1000.0) * (slice2d.shape[1] * VOXEL_SIZE_UM / 1000.0)
        for (min_y, min_x, max_y, max_x), s in zip(bboxes, sizes):
            dy = (max_y - min_y + 1) * VOXEL_SIZE_UM / 1000.0
            dx = (max_x - min_x + 1) * VOXEL_SIZE_UM / 1000.0
            major = max(dx, dy)
            minor = min(dx, dy)
            if minor > 0 and major / minor >= 3.0:  # elongated -> crack-like
                crack_mm += major

        crack_density = (crack_mm / area_mm2) if area_mm2 > 0 else 0.0

        fovs.append({
            "pore_diam_um_p10": float(np.percentile(pore_diams, 10)),
            "pore_diam_um_p50": float(np.percentile(pore_diams, 50)),
            "pore_diam_um_p90": float(np.percentile(pore_diams, 90)),
            "crack_density_mm_per_mm2": float(crack_density * (1.0 + 0.5 * crack_factor)),
            "itz_damage_index": float(min(1.0, max(0.0, itz_damage_index + rng.normal(0, 0.03)))),
        })

    return fovs
