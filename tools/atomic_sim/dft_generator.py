from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, Iterable, List

from .common import (
    MATERIAL_DATABASE,
    SURFACE_MILLER_INDICES,
    GB_SIGMAS,
    clamp,
)


def _temp_softening_factor(temperature_k: float, strength: float = 0.0005) -> float:
    # Linear softening around 300K reference
    return max(0.2, 1.0 - strength * max(0.0, temperature_k - 300.0))


def generate_defect_formation_energies(
    materials: Iterable[str],
    temperatures_k: Iterable[int],
    output_csv_path: str,
    rng_noise_eV: float = 0.05,
) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material", "temperature_K", "defect_type", "energy_eV",
        ])
        for material in materials:
            props = MATERIAL_DATABASE[material]
            base_vac_e = props["vacancy_formation_e0_eV"]
            for T in temperatures_k:
                # Vacancy
                temp_term = -0.06 * (T - 300.0) / 1000.0  # mild decrease
                e_vac = base_vac_e + temp_term + random.gauss(0.0, rng_noise_eV)
                e_vac = clamp(e_vac, 0.2, 4.0)
                writer.writerow([material, T, "vacancy", f"{e_vac:.5f}"])
                # Interstitial, larger than vacancy typically
                e_int = 1.6 * base_vac_e + 0.15 + temp_term + random.gauss(0.0, rng_noise_eV * 1.5)
                e_int = clamp(e_int, 0.4, 6.0)
                writer.writerow([material, T, "interstitial", f"{e_int:.5f}"])


def generate_activation_barriers(
    materials: Iterable[str],
    temperatures_k: Iterable[int],
    output_csv_path: str,
    rng_noise_eV: float = 0.03,
) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material", "temperature_K", "process", "solute_at_pct", "barrier_eV",
        ])
        for material in materials:
            props = MATERIAL_DATABASE[material]
            base_mig = props["vacancy_migration_e0_eV"]
            climb = props["dislocation_climb_barrier_e0_eV"]
            for T in temperatures_k:
                # Vacancy migration: weak T dependence due to entropy; small scatter
                e_mig = base_mig + random.gauss(0.0, rng_noise_eV)
                e_mig = clamp(e_mig, 0.1, 2.0)
                writer.writerow([material, T, "vacancy_migration", "", f"{e_mig:.5f}"])

                # Solute drag barrier depends on solute content; sample a few compositions
                for solute_at_pct in (0.0, 0.5, 1.0, 2.0, 5.0):
                    # Base ~0.2-0.6 + composition effect (sqrt law) + small T softening
                    base = 0.25 + 0.10 * math.sqrt(max(0.0, solute_at_pct))
                    e_drag = base * _temp_softening_factor(T, strength=0.0002) + random.gauss(0.0, rng_noise_eV)
                    e_drag = clamp(e_drag, 0.05, 1.2)
                    writer.writerow([material, T, "solute_drag", f"{solute_at_pct:.2f}", f"{e_drag:.5f}"])

                # Dislocation climb: higher barrier, weak variation
                e_climb = climb + random.gauss(0.0, rng_noise_eV * 2.0)
                e_climb = clamp(e_climb, 0.8, 4.0)
                writer.writerow([material, T, "dislocation_climb", "", f"{e_climb:.5f}"])


def generate_surface_energies(
    materials: Iterable[str],
    temperatures_k: Iterable[int],
    output_csv_path: str,
    rng_noise: float = 0.02,
) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material", "temperature_K", "surface_hkl", "surface_energy_J_m2",
        ])
        for material in materials:
            base = MATERIAL_DATABASE[material]["surface_energy_base_J_m2"]
            # Ordering commonly: gamma(111) < gamma(100) < gamma(110)
            orientation_factor = {"111": 0.88, "100": 1.00, "110": 1.08}
            for T in temperatures_k:
                temp_factor = 0.98 + 0.02 * _temp_softening_factor(T, strength=0.0006)
                for hkl in ("111", "100", "110"):
                    gamma = base * orientation_factor[hkl] * temp_factor + random.gauss(0.0, rng_noise)
                    gamma = clamp(gamma, 0.2, 4.0)
                    writer.writerow([material, T, hkl, f"{gamma:.6f}"])


def generate_grain_boundary_energies(
    materials: Iterable[str],
    temperatures_k: Iterable[int],
    output_csv_path: str,
    rng_noise: float = 0.01,
) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material", "temperature_K", "gb_sigma", "misorientation_deg", "gb_energy_J_m2",
        ])
        for material in materials:
            base = MATERIAL_DATABASE[material]["gb_energy_base_J_m2"]
            for T in temperatures_k:
                temp_factor = 0.9 + 0.1 * _temp_softening_factor(T, strength=0.0007)
                for sigma in GB_SIGMAS:
                    # Special boundaries (low sigma) have lower energies
                    special_cusp = 0.85 if sigma in (3, 5) else 1.0
                    mis = max(1.0, min(89.0, random.gauss(30.0, 12.0)))
                    # Sigma dependence: mild increase for higher sigma
                    sigma_factor = 1.0 + 0.18 * (1.0 / math.sqrt(float(sigma)))
                    gamma = base * temp_factor * sigma_factor * special_cusp + random.gauss(0.0, rng_noise)
                    gamma = clamp(gamma, 0.1, 2.5)
                    writer.writerow([material, T, sigma, f"{mis:.2f}", f"{gamma:.6f}"])
