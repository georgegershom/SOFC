from __future__ import annotations

import csv
import math
import os
import random
import uuid
from typing import Iterable

from .common import MATERIAL_DATABASE, K_B_EV_PER_K, clamp


def _temp_softening_factor(temperature_k: float, strength: float = 0.0005) -> float:
    return max(0.2, 1.0 - strength * max(0.0, temperature_k - 300.0))


def generate_gb_sliding_curves(
    materials: Iterable[str],
    temperatures_k: Iterable[int],
    output_csv_path: str,
    n_points_per_curve: int = 60,
    rng_noise_MPa: float = 5.0,
) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "curve_id", "material", "gb_sigma", "temperature_K", "misorientation_deg",
            "strain", "shear_stress_MPa", "sliding_velocity_m_s",
        ])
        for material in materials:
            shear_modulus_gpa = MATERIAL_DATABASE[material]["shear_modulus_GPa"]
            for T in temperatures_k:
                temp_factor = _temp_softening_factor(T, strength=0.0008)
                # Sample a few boundary types per material-temperature
                for gb_sigma in (3, 5, 11, 13, 19):
                    curve_id = str(uuid.uuid4())
                    misorientation_deg = max(1.0, min(89.0, random.gauss(30.0, 10.0)))

                    # Estimate a peak shear stress proportional to shear modulus
                    tau_peak_mpa = shear_modulus_gpa * 20.0 * temp_factor  # MPa
                    tau_peak_mpa = clamp(tau_peak_mpa, 50.0, 1200.0)

                    # Yield strain and softening rate
                    yield_strain = 0.02
                    softening_k = random.uniform(8.0, 14.0)

                    for i in range(n_points_per_curve):
                        strain = (i / (n_points_per_curve - 1)) * 0.10  # up to 10% shear
                        if strain <= yield_strain:
                            shear_stress = (tau_peak_mpa / yield_strain) * strain
                        else:
                            shear_stress = tau_peak_mpa * math.exp(-softening_k * (strain - yield_strain))
                        shear_stress += random.gauss(0.0, rng_noise_MPa)
                        shear_stress = clamp(shear_stress, 0.0, 1.5 * tau_peak_mpa)

                        # Sliding velocity: stress-driven and thermally activated
                        # v = v_ref * (tau/tau_peak)^m * exp(-Q/(k_B T))
                        v_ref = 0.3  # m/s
                        m = 1.4
                        Q_eV = 0.45
                        tau_ratio = 0.0 if tau_peak_mpa <= 0 else clamp(shear_stress / tau_peak_mpa, 0.0, 2.0)
                        velocity = v_ref * (tau_ratio ** m) * math.exp(-Q_eV / (K_B_EV_PER_K * float(T)))
                        velocity = clamp(velocity, 0.0, 5.0)

                        writer.writerow([
                            curve_id, material, gb_sigma, T, f"{misorientation_deg:.2f}",
                            f"{strain:.5f}", f"{shear_stress:.3f}", f"{velocity:.6f}",
                        ])


def generate_dislocation_mobility(
    materials: Iterable[str],
    temperatures_k: Iterable[int],
    output_csv_path: str,
    n_stress_points: int = 12,
    rng_noise_m_s: float = 0.02,
) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material", "dislocation_type", "temperature_K", "applied_shear_MPa", "velocity_m_s",
        ])
        for material in materials:
            shear_modulus_gpa = MATERIAL_DATABASE[material]["shear_modulus_GPa"]
            for T in temperatures_k:
                temp_factor = _temp_softening_factor(T, strength=0.0010)
                for disl_type in ("edge", "screw"):
                    tau_min = 0.05 * shear_modulus_gpa * 1000.0  # MPa
                    tau_max = 0.35 * shear_modulus_gpa * 1000.0  # MPa
                    for i in range(n_stress_points):
                        tau = tau_min + (tau_max - tau_min) * (i / (n_stress_points - 1))
                        # Mobility law: v = v0 * (tau/tau_ref)^m * exp(-Q/(k_B T))
                        v0 = 800.0  # m/s
                        tau_ref = 0.25 * shear_modulus_gpa * 1000.0
                        m_exp = 1.0 if disl_type == "edge" else 1.2
                        Q_eV = 0.65 if disl_type == "edge" else 0.80
                        tau_ratio = clamp(tau / tau_ref, 0.05, 3.0)
                        v = v0 * (tau_ratio ** m_exp) * math.exp(-Q_eV / (K_B_EV_PER_K * float(T)))
                        v += random.gauss(0.0, rng_noise_m_s)
                        v = clamp(v, 0.0, 2000.0)
                        writer.writerow([material, disl_type, T, f"{tau:.2f}", f"{v:.6f}"])
