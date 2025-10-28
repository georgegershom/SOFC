from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.measure import label as cc_label


@dataclass
class TimeStepResult:
    ct_volume_uint16: np.ndarray  # (Z,Y,X)
    porosity_mask: np.ndarray  # bool
    porosity_fraction: float


class EvolutionConfig:
    def __init__(
        self,
        num_time_steps: int = 10,
        dt_seconds: float = 300.0,
        nucleation_rate: float = 0.002,
        growth_rate: float = 0.9,
        crack_bias_strength: float = 0.6,
        load_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        random_seed: int | None = None,
    ) -> None:
        self.num_time_steps = int(num_time_steps)
        self.dt_seconds = float(dt_seconds)
        self.nucleation_rate = float(np.clip(nucleation_rate, 0.0, 0.2))
        self.growth_rate = float(np.clip(growth_rate, 0.0, 2.0))
        self.crack_bias_strength = float(np.clip(crack_bias_strength, 0.0, 1.0))
        vec = np.array(load_direction, dtype=np.float32)
        self.load_direction = vec / (np.linalg.norm(vec) + 1e-12)
        self.random_seed = random_seed


def _anisotropic_structure(load_dir: np.ndarray, base_radius: int = 1, elongation: float = 2.0) -> np.ndarray:
    # Create a small 3D ellipsoidal structuring element elongated along load_dir
    rx = ry = rz = max(1, int(base_radius))
    # Direction to axis lengths mapping
    ad = np.abs(load_dir) + 1e-6
    scale = 1.0 + (elongation - 1.0) * (ad / ad.max())
    sx = int(round(rx * scale[0])) or 1
    sy = int(round(ry * scale[1])) or 1
    sz = int(round(rz * scale[2])) or 1
    zz, yy, xx = np.mgrid[-sz:sz + 1, -sy:sy + 1, -sx:sx + 1]
    se = (xx * xx) / (sx * sx + 1e-6) + (yy * yy) / (sy * sy + 1e-6) + (zz * zz) / (sz * sz + 1e-6) <= 1.0
    return se


def simulate_creep(
    labels: np.ndarray,
    boundary_mask: np.ndarray,
    initial_porosity: np.ndarray,
    stress_equivalent: np.ndarray,
    evo_cfg: EvolutionConfig,
) -> List[TimeStepResult]:
    """Simulate pore nucleation, growth, and crack coalescence over time.

    Returns a list of per-time-step CT volumes and porosity masks.
    """
    rng = np.random.default_rng(evo_cfg.random_seed)
    shape = labels.shape

    # Per-grain attenuation variation for microstructure contrast
    num_grains = int(labels.max())
    grain_mu = rng.normal(1.0, 0.02, size=num_grains + 1).astype(np.float32)  # index 0 unused
    # Normalize stress to [0, 1]
    stress_norm = (stress_equivalent - stress_equivalent.min()) / (stress_equivalent.ptp() + 1e-6)

    por = initial_porosity.copy()

    # Precompute anisotropic struct element
    se = _anisotropic_structure(evo_cfg.load_direction, base_radius=1, elongation=1.0 + 3.0 * evo_cfg.crack_bias_strength)

    results: List[TimeStepResult] = []

    for t in range(evo_cfg.num_time_steps):
        # Nucleation at boundaries under high stress
        nuc_prob = evo_cfg.nucleation_rate * (0.3 + 0.7 * stress_norm)
        nuc_candidates = boundary_mask & (rng.random(size=shape) < nuc_prob)
        por |= nuc_candidates

        # Growth biased by stress and load direction
        # Adaptive dilation iterations proportional to growth_rate and stress
        growth_mask = por
        if evo_cfg.growth_rate > 0.0:
            # Apply more dilation in high-stress areas probabilistically
            grow_prob = np.clip(evo_cfg.growth_rate * (0.2 + 0.8 * stress_norm), 0.0, 1.0)
            stochastic_grow = rng.random(size=shape) < grow_prob
            growth_seed = growth_mask & stochastic_grow
            if growth_seed.any():
                grown = binary_dilation(growth_seed, se)
                por |= grown

        # Coalescence happens naturally as dilation merges pores.

        # Convert to CT attenuation image
        ct = _render_ct(labels, por, grain_mu, rng)

        results.append(
            TimeStepResult(
                ct_volume_uint16=ct,
                porosity_mask=por.copy(),
                porosity_fraction=float(por.mean()),
            )
        )

    return results


def _render_ct(labels: np.ndarray, pores: np.ndarray, grain_mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Base attenuation and per-grain contrast
    base_mu = 0.9
    mu = base_mu * np.ones_like(labels, dtype=np.float32)
    mu *= grain_mu[np.clip(labels, 0, grain_mu.size - 1)]

    # Pores and cracks have near-zero attenuation
    mu[pores] = 0.03

    # Simulate acquisition noise and smoothness
    mu = gaussian_filter(mu, sigma=0.6)
    noisy = mu + rng.normal(0.0, 0.01, size=mu.shape).astype(np.float32)

    # Map to 16-bit intensity range
    noisy = np.clip(noisy, 0.0, 1.2)
    img16 = (noisy / 1.2 * 65535.0).astype(np.uint16)
    return img16
