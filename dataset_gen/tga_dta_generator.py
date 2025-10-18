from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np

from .config import TGA_MAX_TEMP_C, TGA_STEP_C
from .utils import ensure_dir


def _seed(specimen_id: str) -> int:
    return abs(hash(("TGA", specimen_id))) % (2**32)


def _sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) / (width + 1e-6)))


def generate_tga_dta_for_specimen(
    out_dir: Path, specimen_id: str, params: Dict[str, float]
) -> Dict[str, float]:
    """
    Simulate TGA mass loss and DTG derivative across temperature.
    Returns specimen-level mass loss metrics for key events.
    """
    T = np.arange(25, TGA_MAX_TEMP_C + 1e-9, TGA_STEP_C)

    # Event fractions (approximate, normalized-like)
    free_water = 0.03
    bound_water = 0.07
    csh_high_temp = 0.03
    ch = 0.18 * params["ch_rel"] + 0.02
    caco3 = 0.05 * params["caco3_rel"]
    rubber = params["rubber_pyrolysis_fraction"]

    # Construct cumulative loss using sigmoids
    loss_fw = free_water * _sigmoid(T, 90, 8)
    loss_bw = bound_water * _sigmoid(T, 160, 15)
    loss_csh = csh_high_temp * _sigmoid(T, 320, 25)
    loss_ch = ch * _sigmoid(T, 460, 25)
    loss_rub = rubber * _sigmoid(T, 420, 40)
    loss_caco3 = caco3 * _sigmoid(T, 800, 60)

    total_loss = loss_fw + loss_bw + loss_csh + loss_ch + loss_rub + loss_caco3

    mass = 1.0 - total_loss
    mass = np.clip(mass, 0.0, 1.0)

    dt = np.gradient(T)
    dtg = -np.gradient(mass, dt)

    # Save CSV
    ensure_dir(out_dir)
    out_csv = out_dir / f"{specimen_id}_tga_dta.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature_C", "mass", "dtg"])
        for t, m, d in zip(T, mass, dtg):
            w.writerow([f"{t:.2f}", f"{m:.6f}", f"{d:.6f}"])

    # Key metrics
    def mass_at(temp: float) -> float:
        idx = int((temp - 25) / TGA_STEP_C)
        idx = max(0, min(idx, len(T) - 1))
        return float(mass[idx])

    metrics = {
        "tga_mass_200C": mass_at(200),
        "tga_mass_400C": mass_at(400),
        "tga_mass_600C": mass_at(600),
        "tga_mass_800C": mass_at(800),
        "tga_total_loss": float(1.0 - mass[-1]),
    }
    return metrics
