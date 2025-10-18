from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import XRD_RANGE_DEG, XRD_STEP_DEG
from .utils import ensure_dir


def _seed(specimen_id: str) -> int:
    return abs(hash(("XRD", specimen_id))) % (2**32)


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_xrd_for_specimen(out_dir: Path, specimen_id: str, params: Dict[str, float]) -> Dict[str, float]:
    """
    Simulate XRD pattern with temperature-dependent phase changes.
    Returns specimen-level semi-quantitative metrics.
    """
    two_theta = np.arange(XRD_RANGE_DEG[0], XRD_RANGE_DEG[1] + 1e-9, XRD_STEP_DEG)
    baseline = 50.0

    # C-S-H amorphous hump (broad background between 20-35°)
    hump_center = 27.0
    hump_sigma = 5.0
    hump_amp = 300.0 * params["csh_hump"]
    csh_hump = hump_amp * _gaussian(two_theta, hump_center, hump_sigma)

    # Portlandite (CH) peaks
    ch_amp = 600.0 * params["ch_rel"]
    ch_peaks = [18.0, 34.1]  # main CH peaks
    ch_sigma = 0.12
    ch = sum(ch_amp * _gaussian(two_theta, p, ch_sigma) for p in ch_peaks)

    # Calcite peaks
    caco3_amp = 350.0 * params["caco3_rel"]
    caco3_peaks = [29.4, 39.4, 43.1]
    caco3_sigma = 0.10
    caco3 = sum(caco3_amp * _gaussian(two_theta, p, caco3_sigma) for p in caco3_peaks)

    # Quartz from aggregates (stable)
    qtz_amp = 220.0
    qtz_peaks = [20.8, 26.6, 36.5]
    qtz_sigma = 0.10
    quartz = sum(qtz_amp * _gaussian(two_theta, p, qtz_sigma) for p in qtz_peaks)

    # CaO peaks (increase with temperature)
    cao_amp = 300.0 * params["cao_rel"]
    cao_peaks = [32.2, 37.3]
    cao_sigma = 0.10
    cao = sum(cao_amp * _gaussian(two_theta, p, cao_sigma) for p in cao_peaks)

    pattern = baseline + csh_hump + ch + caco3 + quartz + cao

    # Add mild noise and rubber-related baseline offset
    rng = np.random.default_rng(_seed(specimen_id))
    noise = rng.normal(0.0, 5.0, size=two_theta.shape)
    pattern = np.clip(pattern + noise, 0, None)

    # Save CSV
    ensure_dir(out_dir)
    out_csv = out_dir / f"{specimen_id}_xrd.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["two_theta_deg", "intensity"])
        for t, i in zip(two_theta, pattern):
            w.writerow([f"{t:.4f}", f"{i:.2f}"])

    # Semi-quant metrics: integrate under regions around canonical peaks
    def integrate_region(center: float, width: float) -> float:
        mask = (two_theta >= center - width) & (two_theta <= center + width)
        return float(np.trapz(pattern[mask], two_theta[mask]))

    ch_metric = sum(integrate_region(p, 0.25) for p in ch_peaks)
    caco3_metric = sum(integrate_region(p, 0.25) for p in caco3_peaks)
    cao_metric = sum(integrate_region(p, 0.25) for p in cao_peaks)
    hump_metric = float(np.trapz(csh_hump, two_theta))

    return {
        "xrd_ch_rel_area": ch_metric,
        "xrd_caco3_rel_area": caco3_metric,
        "xrd_cao_rel_area": cao_metric,
        "xrd_csh_hump_area": hump_metric,
    }
