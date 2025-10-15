from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class StressField:
    # Voxelized stress tensor components on a uniform grid
    # Stored as dict of arrays with shape (nz, ny, nx)
    tensors: Dict[str, np.ndarray]


def voxelize_layer_stresses(layer_stress_xy: Dict[str, np.ndarray], layers: Tuple[str, ...], grid: Tuple[int, int, int]) -> StressField:
    nx, ny, nz = grid
    # Allocate tensors (sigma_xx, sigma_yy, tau_xy), other terms set to zero for simplicity
    tensors = {
        "sigma_xx": np.zeros((nz, ny, nx), dtype=np.float32),
        "sigma_yy": np.zeros((nz, ny, nx), dtype=np.float32),
        "sigma_xy": np.zeros((nz, ny, nx), dtype=np.float32),
        "sigma_zz": np.zeros((nz, ny, nx), dtype=np.float32),
        "tau_xz": np.zeros((nz, ny, nx), dtype=np.float32),
        "tau_yz": np.zeros((nz, ny, nx), dtype=np.float32),
    }
    # Split z equally per layer
    layers_n = len(layers)
    z_slices_per_layer = max(1, nz // layers_n)
    z = 0
    for layer_name in layers:
        sx, sy, txy = layer_stress_xy[layer_name]
        z_end = min(nz, z + z_slices_per_layer)
        for zi in range(z, z_end):
            tensors["sigma_xx"][zi, :, :] = sx
            tensors["sigma_yy"][zi, :, :] = sy
            tensors["sigma_xy"][zi, :, :] = txy
        z = z_end
    return StressField(tensors=tensors)


def warp_surfaces_to_heightmap(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Provide a 2.5D height map for the top surface (bottom is symmetric offset)
    return W.astype(np.float32)
