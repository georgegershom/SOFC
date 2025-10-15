from __future__ import annotations
import numpy as np
from typing import Tuple


def downsample_average(vol: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """Downsample a 3D volume by simple block averaging to the target shape."""
    in_shape = vol.shape
    sx = max(1, in_shape[0] // out_shape[0])
    sy = max(1, in_shape[1] // out_shape[1])
    sz = max(1, in_shape[2] // out_shape[2])

    nx = (in_shape[0] // sx) * sx
    ny = (in_shape[1] // sy) * sy
    nz = (in_shape[2] // sz) * sz

    v = vol[:nx, :ny, :nz]
    v = v.reshape(nx // sx, sx, ny // sy, sy, nz // sz, sz).mean(axis=(1, 3, 5))

    # If shapes don't match exactly, pad/crop
    out = v
    if out.shape != out_shape:
        out = np.zeros(out_shape, dtype=vol.dtype)
        x = min(out_shape[0], v.shape[0])
        y = min(out_shape[1], v.shape[1])
        z = min(out_shape[2], v.shape[2])
        out[:x, :y, :z] = v[:x, :y, :z]
    return out
