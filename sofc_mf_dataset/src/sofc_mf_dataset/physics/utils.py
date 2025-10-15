from __future__ import annotations
import numpy as np


def smooth_nd(field: np.ndarray, iters: int = 6, alpha: float = 0.5) -> np.ndarray:
    """Diffuse-smooth an N-D field by neighbor averaging.
    alpha in [0,1]: 0 -> no smooth, 1 -> full neighbor mean.
    """
    f = field
    for _ in range(max(0, iters)):
        neighbor_sum = np.zeros_like(f)
        for axis in range(f.ndim):
            neighbor_sum += np.roll(f, 1, axis=axis)
            neighbor_sum += np.roll(f, -1, axis=axis)
        neighbor_mean = neighbor_sum / (2 * f.ndim)
        f = (1.0 - alpha) * f + alpha * neighbor_mean
    return f


def random_field(shape: tuple[int, ...], rng: np.random.Generator, corr_iters: int = 10, alpha: float = 0.5) -> np.ndarray:
    f = rng.normal(0.0, 1.0, size=shape)
    f = smooth_nd(f, iters=corr_iters, alpha=alpha)
    f -= f.mean()
    std = f.std()
    if std > 1e-12:
        f /= std
    return f


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)
