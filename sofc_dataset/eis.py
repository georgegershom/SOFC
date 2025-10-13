from __future__ import annotations
import numpy as np
from typing import Dict


def synthesize_eis(
    T_K: float,
    current_density_A_per_cm2: float,
    age_hours: float,
    seed: int = 0,
    n_points: int = 60,
    f_min: float = 0.1,
    f_max: float = 1e5,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Log-spaced frequencies
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    # Simplified equivalent circuit: Rs - (R1||CPE1) - (R2||CPE2)
    # Temperature and age affect resistances and CPE
    Rs = 0.1 + 0.2 * np.exp(-(T_K - 600.0) / 300.0) + 0.02 * (age_hours / 5000.0)
    R1 = 0.2 + 0.1 * (current_density_A_per_cm2)
    R2 = 0.1 + 0.15 * np.exp(-(T_K - 700.0) / 400.0) + 0.05 * (age_hours / 5000.0)
    Q1 = 0.5e-2 * (700.0 / T_K)
    n1 = 0.85 - 0.1 * (age_hours / 5000.0)
    Q2 = 1.5e-2 * (800.0 / T_K)
    n2 = 0.75 - 0.05 * (age_hours / 5000.0)

    w = 2 * np.pi * freqs

    # Constant Phase Element admittance: Y = Q (j w)^n
    j = 1j
    Y_cpe1 = Q1 * (j * w) ** n1
    Y_cpe2 = Q2 * (j * w) ** n2

    Z_R1_cpe1 = 1.0 / (1.0 / R1 + Y_cpe1)
    Z_R2_cpe2 = 1.0 / (1.0 / R2 + Y_cpe2)

    Z_total = Rs + Z_R1_cpe1 + Z_R2_cpe2

    # Add small noise
    noise = (rng.normal(scale=0.002, size=Z_total.shape) + 1j * rng.normal(scale=0.002, size=Z_total.shape))
    Z_noisy = Z_total + noise

    return {
        "frequency_Hz": freqs.astype(np.float64),
        "Zreal_ohm": np.real(Z_noisy).astype(np.float64),
        "Zimag_ohm": np.imag(Z_noisy).astype(np.float64),
    }
