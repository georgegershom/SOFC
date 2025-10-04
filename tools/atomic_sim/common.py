from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

# Boltzmann constant in eV/K
K_B_EV_PER_K: float = 8.617333262145e-5

# Basic material database with rough, plausible anchors (not authoritative)
MATERIAL_DATABASE: Dict[str, Dict[str, float]] = {
    # values are intentionally approximate
    "Al": {
        "vacancy_formation_e0_eV": 0.67,
        "vacancy_migration_e0_eV": 0.55,
        "surface_energy_base_J_m2": 1.10,
        "gb_energy_base_J_m2": 0.45,
        "dislocation_climb_barrier_e0_eV": 1.60,
        "shear_modulus_GPa": 26.0,
    },
    "Cu": {
        "vacancy_formation_e0_eV": 1.28,
        "vacancy_migration_e0_eV": 0.76,
        "surface_energy_base_J_m2": 1.80,
        "gb_energy_base_J_m2": 0.65,
        "dislocation_climb_barrier_e0_eV": 1.90,
        "shear_modulus_GPa": 48.0,
    },
    "Ni": {
        "vacancy_formation_e0_eV": 1.63,
        "vacancy_migration_e0_eV": 1.00,
        "surface_energy_base_J_m2": 2.45,
        "gb_energy_base_J_m2": 0.75,
        "dislocation_climb_barrier_e0_eV": 2.30,
        "shear_modulus_GPa": 76.0,
    },
    "Fe": {
        "vacancy_formation_e0_eV": 2.00,
        "vacancy_migration_e0_eV": 0.67,
        "surface_energy_base_J_m2": 2.45,
        "gb_energy_base_J_m2": 0.90,
        "dislocation_climb_barrier_e0_eV": 2.50,
        "shear_modulus_GPa": 82.0,
    },
    "Ti": {
        "vacancy_formation_e0_eV": 1.80,
        "vacancy_migration_e0_eV": 1.10,
        "surface_energy_base_J_m2": 2.00,
        "gb_energy_base_J_m2": 0.85,
        "dislocation_climb_barrier_e0_eV": 2.40,
        "shear_modulus_GPa": 44.0,
    },
}

SURFACE_MILLER_INDICES: List[str] = ["111", "100", "110"]
# Common CSL Sigma values (small to moderate)
GB_SIGMAS: List[int] = [3, 5, 7, 9, 11, 13, 17, 19, 21, 27]


def set_global_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)


def choose_surface() -> str:
    return random.choice(SURFACE_MILLER_INDICES)


def choose_gb_sigma() -> int:
    return random.choice(GB_SIGMAS)


def plausible_misorientation_deg(sigma: int) -> float:
    # Favor moderate misorientations; special boundaries have narrower spread
    base = 30.0 if sigma not in (3, 5) else 20.0
    spread = 20.0 if sigma not in (3, 5) else 10.0
    return max(1.0, min(89.0, random.gauss(base, spread)))


def temperature_grid(min_k: int, max_k: int, count: int) -> List[int]:
    if count <= 1:
        return [int(min_k)]
    step = (max_k - min_k) / (count - 1)
    return [int(round(min_k + i * step)) for i in range(count)]


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def arrhenius(prefactor: float, activation_energy_eV: float, temperature_K: float) -> float:
    return prefactor * math.exp(-activation_energy_eV / (K_B_EV_PER_K * temperature_K))
