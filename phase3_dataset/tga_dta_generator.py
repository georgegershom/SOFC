import os
from typing import Dict, List, Tuple

import numpy as np

from .utils import ensure_dir


# Temperature ranges (°C) for processes
FREE_WATER = (30, 150)
BOUND_WATER = (150, 450)
CH_DEHYDROX = (400, 560)
CARBONATE_DEC = (650, 800)
RUBBER_PYRO = (250, 450)


def _sigmoid_step(x: np.ndarray, start: float, end: float, magnitude: float) -> np.ndarray:
    # Smooth step between start and end with total drop = magnitude
    center = 0.5 * (start + end)
    width = (end - start) / 8.0
    return magnitude / (1.0 + np.exp(-(x - center) / width))


def _simulate_tga_dta(specimen: str, temp_c: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    t = np.linspace(25, 900, 2000)

    # Baseline mass (normalized to 1.0 at 25°C)
    mass = np.ones_like(t)

    # Free water content varies with mix & preheating; scaled to temp window explored
    free_water_loss = 0.04 + 0.02 * np.random.rand()
    bound_water_loss = 0.08 + 0.03 * np.random.rand()
    ch_loss = 0.10 + 0.03 * np.random.rand()
    carb_loss = 0.05 + 0.02 * np.random.rand()
    rubber_loss = 0.0

    if specimen == "rubber":
        # Rubber polymers add additional mass loss during pyrolysis
        rubber_loss = 0.06 + 0.03 * np.random.rand()
        # Rubberized mixes tend to trap more moisture; slight uptick
        free_water_loss *= 1.15

    # Apply smooth drops across characteristic ranges
    mass -= _sigmoid_step(t, FREE_WATER[0], FREE_WATER[1], free_water_loss)
    mass -= _sigmoid_step(t, BOUND_WATER[0], BOUND_WATER[1], bound_water_loss)
    mass -= _sigmoid_step(t, CH_DEHYDROX[0], CH_DEHYDROX[1], ch_loss)
    mass -= _sigmoid_step(t, CARBONATE_DEC[0], CARBONATE_DEC[1], carb_loss)
    if rubber_loss > 0:
        mass -= _sigmoid_step(t, RUBBER_PYRO[0], RUBBER_PYRO[1], rubber_loss)

    # Ensure monotonic non-increasing
    mass = np.maximum.accumulate(mass[::-1])[::-1]
    mass += 0.003 * np.random.normal(0.0, 1.0, size=mass.shape)
    mass = np.clip(mass, 0.2, 1.02)

    # DTG (derivative) approximates DTA endotherms for dehydration; rubber pyrolysis can be exothermic but approximate here
    dt = t[1] - t[0]
    dtg = -np.gradient(mass, dt)

    # Quantify losses per window by integrating DTG in those windows
    def integrate_loss(window: Tuple[float, float]) -> float:
        lo, hi = window
        mask = (t >= lo) & (t <= hi)
        return float(np.trapz(dtg[mask], dx=dt))

    metrics = {
        "loss_free_water": integrate_loss(FREE_WATER),
        "loss_bound_water": integrate_loss(BOUND_WATER),
        "loss_ch_dehydrox": integrate_loss(CH_DEHYDROX),
        "loss_caco3_decarb": integrate_loss(CARBONATE_DEC),
        "loss_rubber_pyro": integrate_loss(RUBBER_PYRO) if specimen == "rubber" else 0.0,
        "total_mass_loss": float(1.0 - mass[-1]),
    }

    # Clip ranges to given residual temp context for emphasis in downstream analysis (not applied here to curves)
    return t, mass, dtg, metrics


def generate_tga_dta_batch(output_dir: str, specimen: str, temp_c: int, replicates: int) -> List[Dict]:
    ensure_dir(output_dir)

    items: List[Dict] = []
    for i in range(replicates):
        t, mass, dtg, metrics = _simulate_tga_dta(specimen, temp_c)
        base = f"tga_{specimen}_{temp_c}C_rep{i+1}"
        csv_path = os.path.join(output_dir, base + ".csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("temperature_c,mass_fraction,dtg\n")
            for tt, m, d in zip(t, mass, dtg):
                f.write(f"{tt:.2f},{m:.6f},{d:.6f}\n")

        item = {
            "modality": "TGA_DTA",
            "specimen": specimen,
            "temperature_c": temp_c,
            "replicate": i + 1,
            "csv_path": csv_path,
            "metrics": metrics,
            "notes": "Synthetic TGA/DTG: stepwise mass losses for free/bound water, CH dehydroxylation, CaCO3 decarbonation; rubber mixes include pyrolysis mass loss.",
        }
        items.append(item)

    return items
