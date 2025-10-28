from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class StressField:
    equivalent_stress_mpa: np.ndarray  # shape (Z,Y,X)
    load_direction_unit: np.ndarray  # shape (3,)
    elastic_strain: np.ndarray  # same shape as stress, dimensionless


def _unit_vector(direction: Union[str, Tuple[float, float, float]]) -> np.ndarray:
    if isinstance(direction, str):
        mapping = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        vec = np.array(mapping.get(direction.lower(), (0.0, 0.0, 1.0)), dtype=np.float32)
    else:
        vec = np.array(direction, dtype=np.float32)
    n = np.linalg.norm(vec) + 1e-12
    return vec / n


def generate_residual_stress_field(
    shape: Tuple[int, int, int],
    base_stress_mpa: float = 50.0,
    heterogeneity: float = 0.35,
    load_direction: Union[str, Tuple[float, float, float]] = "z",
    seed: int | None = None,
) -> StressField:
    """Create a synthetic residual stress and elastic strain field.

    Equivalent stress varies smoothly and increases along load direction with noise.
    """
    z, y, x = shape
    dir_vec = _unit_vector(load_direction)

    # Normalized coordinates in [-1, 1]
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, z, dtype=np.float32),
        np.linspace(-1.0, 1.0, y, dtype=np.float32),
        np.linspace(-1.0, 1.0, x, dtype=np.float32),
        indexing="ij",
    )

    # Project position onto load direction to create macro gradient
    pos = np.stack([xx, yy, zz], axis=0)
    proj = dir_vec[0] * pos[0] + dir_vec[1] * pos[1] + dir_vec[2] * pos[2]
    proj = (proj - proj.min()) / (proj.ptp() + 1e-6)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=3.0)
    noise = (noise - noise.min()) / (noise.ptp() + 1e-6)

    eq_stress = base_stress_mpa * (0.6 + 0.4 * proj) * (1.0 - 0.3 * heterogeneity + 0.6 * heterogeneity * noise)

    # Elastic strain via simple Hooke's law with an effective modulus
    effective_E_gpa = 200.0  # ruby rough steel-like modulus
    elastic_strain = (eq_stress / 1000.0) / (effective_E_gpa + 1e-6)

    return StressField(equivalent_stress_mpa=eq_stress.astype(np.float32), load_direction_unit=dir_vec.astype(np.float32), elastic_strain=elastic_strain.astype(np.float32))
