from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class XRDConfig:
    energy_keV: float = 60.0
    detector_distance_mm: float = 800.0
    pixel_size_um: float = 100.0
    image_size: Tuple[int, int] = (1024, 1024)
    center: Tuple[float, float] | None = None
    texture_anisotropy: float = 0.25  # 0..1, azimuthal intensity modulation
    microstrain: float = 0.001  # broadening and slight azimuthal shift
    random_seed: int | None = None


_PHASE_DB: Dict[str, List[Tuple[float, float]]] = {
    # (d-spacing in Å, relative intensity)
    "Ni": [(2.034, 1.0), (1.765, 0.55), (1.246, 0.35), (1.063, 0.28)],
    "FeCr": [(2.026, 1.0), (1.433, 0.45), (1.170, 0.35)],
    "YSZ": [(2.953, 0.6), (2.563, 0.45), (1.812, 0.25)],
}


def _bragg_radius_px(d_angstrom: float, energy_keV: float, det_dist_mm: float, pixel_um: float) -> float:
    lam = 12.39842 / energy_keV  # Å
    # Guard against invalid domain for arcsin
    s = np.clip(lam / (2.0 * d_angstrom), -1.0, 1.0)
    theta = np.arcsin(s)
    two_theta = 2.0 * theta
    r_mm = det_dist_mm * np.tan(two_theta)
    r_px = r_mm * 1000.0 / pixel_um
    return float(r_px)


def simulate_xrd_image(phases: Dict[str, float], cfg: XRDConfig) -> np.ndarray:
    """Simulate a 2D diffraction image with Debye-Scherrer rings.

    phases: mapping of phase name -> scale factor (relative abundance)
    """
    h, w = cfg.image_size
    cx, cy = (w / 2.0, h / 2.0) if cfg.center is None else cfg.center

    rng = np.random.default_rng(cfg.random_seed)

    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    rr = np.hypot(dx, dy)
    az = np.arctan2(dy, dx)

    img = np.zeros((h, w), dtype=np.float32)

    for phase, scale in phases.items():
        if phase not in _PHASE_DB or scale <= 0.0:
            continue
        for d_space, rel_int in _PHASE_DB[phase]:
            r0 = _bragg_radius_px(d_space, cfg.energy_keV, cfg.detector_distance_mm, cfg.pixel_size_um)
            if not (0.0 < r0 < 1.5 * max(h, w)):
                continue
            # Ring width (px) increases with microstrain
            sigma = 1.5 + 200.0 * cfg.microstrain

            # Azimuthal intensity modulation (texture) and slight radius shift (strain anisotropy)
            az_mod = 1.0 + cfg.texture_anisotropy * np.cos(2.0 * (az - 0.0))
            r_shift = r0 * (1.0 + 0.5 * cfg.microstrain * np.cos(2.0 * (az - 0.0)))

            prof = np.exp(-0.5 * ((rr - r_shift) / sigma) ** 2)
            img += float(scale * rel_int) * az_mod * prof

    # Background and smoothing
    background = 0.02 + 0.000015 * rr
    img = img + background + rng.normal(0.0, 0.002, size=img.shape).astype(np.float32)
    img = gaussian_filter(img, sigma=0.5)
    img = np.clip(img, 0.0, None)

    # Normalize to 16-bit
    img = img / (img.max() + 1e-6)
    img16 = (img * 65535.0).astype(np.uint16)
    return img16
