from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from .constants import DomainConfig


def sample_interior(domain: DomainConfig, n_points: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, domain.length_x, size=(n_points, 1))
    y = rng.uniform(0.0, domain.length_y, size=(n_points, 1))
    if domain.has_time:
        t = rng.uniform(0.0, domain.time_max, size=(n_points, 1))
        pts = np.concatenate([x, y, t], axis=1)
    else:
        pts = np.concatenate([x, y], axis=1)
    return pts


def sample_boundary_edges(domain: DomainConfig, n_per_edge: int, seed: int | None = None) -> Dict[str, np.ndarray]:
    """Return dict of points on rectangle edges with outward normals.
    Keys: left, right, bottom, top. Each value is dict with fields: coords (N, D), normal (N, 2)
    """
    rng = np.random.default_rng(seed)
    # y in [0, Ly], x in [0, Lx]
    y_left = rng.uniform(0.0, domain.length_y, size=(n_per_edge, 1))
    x_left = np.zeros_like(y_left)
    y_right = rng.uniform(0.0, domain.length_y, size=(n_per_edge, 1))
    x_right = np.full_like(y_right, fill_value=domain.length_x)

    x_bottom = rng.uniform(0.0, domain.length_x, size=(n_per_edge, 1))
    y_bottom = np.zeros_like(x_bottom)
    x_top = rng.uniform(0.0, domain.length_x, size=(n_per_edge, 1))
    y_top = np.full_like(x_top, fill_value=domain.length_y)

    def pack(xv, yv):
        if domain.has_time:
            t = rng.uniform(0.0, domain.time_max, size=(xv.shape[0], 1))
            return np.concatenate([xv, yv, t], axis=1)
        return np.concatenate([xv, yv], axis=1)

    edges = {
        "left": {
            "coords": pack(x_left, y_left),
            "normal": np.tile(np.array([[-1.0, 0.0]]), (n_per_edge, 1)),
        },
        "right": {
            "coords": pack(x_right, y_right),
            "normal": np.tile(np.array([[+1.0, 0.0]]), (n_per_edge, 1)),
        },
        "bottom": {
            "coords": pack(x_bottom, y_bottom),
            "normal": np.tile(np.array([[0.0, -1.0]]), (n_per_edge, 1)),
        },
        "top": {
            "coords": pack(x_top, y_top),
            "normal": np.tile(np.array([[0.0, +1.0]]), (n_per_edge, 1)),
        },
    }
    return edges


def sample_internal_interface(domain: DomainConfig, x_location: float, n_points: int, seed: int | None = None) -> Dict[str, np.ndarray]:
    """Sample a vertical internal interface at x = x_location with normal +x (from left to right)."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, domain.length_y, size=(n_points, 1))
    x = np.full_like(y, fill_value=x_location)

    if domain.has_time:
        t = rng.uniform(0.0, domain.time_max, size=(n_points, 1))
        coords = np.concatenate([x, y, t], axis=1)
    else:
        coords = np.concatenate([x, y], axis=1)

    normal = np.tile(np.array([[+1.0, 0.0]]), (n_points, 1))
    return {"coords": coords, "normal": normal}
