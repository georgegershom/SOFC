import os
from typing import Dict, List, Tuple

import numpy as np

from .utils import gaussian, normalize01, ensure_dir


# Characteristic peaks (2θ in degrees) for key phases (Cu Kα)
CH_PEAKS = [18.0, 34.1, 47.1]  # Portlandite (Ca(OH)2)
CALCITE_PEAKS = [29.4, 39.4, 43.1]
QUARTZ_PEAKS = [20.9, 26.6, 36.5]


def _simulate_xrd_pattern(specimen: str, temp_c: int, two_theta: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    # Amorphous hump for C-S-H centered ~ 29°-32°
    hump_center = 30.0
    hump_width = 6.0
    hump_amp = 0.6
    baseline = 0.1
    intensity = baseline + hump_amp * np.exp(-0.5 * ((two_theta - hump_center) / hump_width) ** 2)

    # Portlandite amount decreases with temperature
    ch_scale = float(np.clip(np.interp(temp_c, [20, 400, 600, 800], [1.0, 0.6, 0.15, 0.05]), 0.0, 1.0))
    # Calcite: carbonation may be present at ambient; decarbonates above ~650°C
    calcite_scale = float(np.clip(np.interp(temp_c, [20, 600, 800], [0.5, 0.5, 0.1]), 0.0, 1.0))
    # Quartz constant (aggregate)
    quartz_scale = 0.3

    # Rubber mixes show more disorder and peak broadening
    broadening = 0.25 if specimen == "rubber" else 0.18

    # Add CH peaks
    for mu in CH_PEAKS:
        intensity += gaussian(two_theta, mu, sigma=broadening, amplitude=0.9 * ch_scale)
    # Calcite peaks
    for mu in CALCITE_PEAKS:
        intensity += gaussian(two_theta, mu, sigma=0.22, amplitude=0.6 * calcite_scale)
    # Quartz peaks
    for mu in QUARTZ_PEAKS:
        intensity += gaussian(two_theta, mu, sigma=0.18, amplitude=0.5 * quartz_scale)

    # Noise and normalization
    intensity = np.maximum(intensity, 0.0)
    # Add shot noise
    intensity = intensity + np.random.normal(0.0, 0.03, size=two_theta.shape)
    intensity = normalize01(intensity)

    # Quantify CH: integrate around main CH peaks
    ch_windows = [(mu - 0.6, mu + 0.6) for mu in CH_PEAKS]
    dx = two_theta[1] - two_theta[0]
    ch_int = 0.0
    for lo, hi in ch_windows:
        mask = (two_theta >= lo) & (two_theta <= hi)
        ch_int += float(np.trapz(np.maximum(intensity[mask] - baseline, 0.0), dx=dx))

    metrics = {
        "ch_relative_intensity": float(ch_int),
        "calcite_relative": float(calcite_scale),
        "pattern_broadening_sigma": float(broadening),
    }
    return intensity, metrics


def generate_xrd_batch(output_dir: str, specimen: str, temp_c: int, replicates: int) -> List[Dict]:
    ensure_dir(output_dir)
    items: List[Dict] = []

    two_theta = np.linspace(5.0, 60.0, 4000)
    for i in range(replicates):
        intensity, metrics = _simulate_xrd_pattern(specimen, temp_c, two_theta)

        base = f"xrd_{specimen}_{temp_c}C_rep{i+1}"
        csv_path = os.path.join(output_dir, base + ".csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("two_theta,intensity\n")
            for tt, ii in zip(two_theta, intensity):
                f.write(f"{tt:.4f},{ii:.6f}\n")

        item = {
            "modality": "XRD",
            "specimen": specimen,
            "temperature_c": temp_c,
            "replicate": i + 1,
            "csv_path": csv_path,
            "metrics": metrics,
            "notes": "Synthetic XRD pattern with CH, calcite, quartz peaks; CH decreases with temperature; rubber mixes show broader peaks (more disorder).",
        }
        items.append(item)

    return items
