from typing import Dict, List
import numpy as np

from .config import VOXEL_SIZE_UM
from .utils import connected_components_3d, component_spans_axes, equivalent_diameter_from_voxel_count_3d


def compute_micro_ct_metrics(volume: np.ndarray) -> Dict[str, float]:
    # volume: uint8, 0=solid, 1=void
    pore = volume.astype(bool)
    porosity = float(pore.mean())
    labels, sizes = connected_components_3d(pore)
    num_components = int(len(sizes))

    if sizes:
        eq_diams = [equivalent_diameter_from_voxel_count_3d(s, VOXEL_SIZE_UM) for s in sizes]
        p50 = float(np.percentile(eq_diams, 50))
        p90 = float(np.percentile(eq_diams, 90))
        # Spanning check across components
        spans_x = False
        spans_y = False
        spans_z = False
        for label_id in range(1, num_components + 1):
            sx, sy, sz = component_spans_axes(labels, label_id)
            spans_x = spans_x or sx
            spans_y = spans_y or sy
            spans_z = spans_z or sz
    else:
        p50 = 0.0
        p90 = 0.0
        spans_x = spans_y = spans_z = False

    metrics = {
        "porosity": porosity,
        "pore_components": float(num_components),
        "pore_eq_diam_p50_um": p50,
        "pore_eq_diam_p90_um": p90,
        "percolates_x": float(1.0 if spans_x else 0.0),
        "percolates_y": float(1.0 if spans_y else 0.0),
        "percolates_z": float(1.0 if spans_z else 0.0),
    }
    return metrics
