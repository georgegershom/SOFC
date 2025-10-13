from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class ParamRange:
    min_value: float
    max_value: float
    log_scale: bool = False


# Centralized parameter ranges and basic units
PARAM_RANGES: Dict[str, ParamRange] = {
    # Operating conditions
    "current_density_A_per_cm2": ParamRange(0.1, 2.0, False),
    "fuel_utilization_frac": ParamRange(0.3, 0.9, False),
    "air_utilization_frac": ParamRange(0.1, 0.5, False),
    "inlet_fuel_temp_C": ParamRange(600.0, 850.0, False),
    "inlet_air_temp_C": ParamRange(600.0, 850.0, False),

    # Fuel composition (percentages that will be normalized)
    "fuel_H2_frac_raw": ParamRange(0.1, 0.9, False),
    "fuel_H2O_frac_raw": ParamRange(0.0, 0.6, False),
    "fuel_CO_frac_raw": ParamRange(0.0, 0.3, False),
    "fuel_CH4_frac_raw": ParamRange(0.0, 0.3, False),

    # Material & geometric
    "anode_porosity": ParamRange(0.2, 0.5, False),
    "anode_tortuosity": ParamRange(1.5, 3.5, False),
    "cathode_porosity": ParamRange(0.2, 0.5, False),
    "cathode_tortuosity": ParamRange(1.5, 3.5, False),
    "ionic_conductivity_S_per_m": ParamRange(1.0, 20.0, False),
    "electronic_conductivity_S_per_m": ParamRange(1e2, 1e5, True),
    "anode_thickness_um": ParamRange(10.0, 150.0, False),
    "electrolyte_thickness_um": ParamRange(5.0, 60.0, False),
    "cathode_thickness_um": ParamRange(10.0, 150.0, False),
    "interconnect_CTE_per_K": ParamRange(12e-6, 16e-6, False),
    "interconnect_YoungsModulus_GPa": ParamRange(150.0, 220.0, False),

    # Degradation/initial state
    "initial_crack_length_mm": ParamRange(0.0, 3.0, False),
    "initial_crack_angle_deg": ParamRange(0.0, 180.0, False),
    "initial_crack_x_frac": ParamRange(0.1, 0.9, False),
    "initial_crack_y_frac": ParamRange(0.1, 0.9, False),
    "initial_porosity_amp": ParamRange(0.0, 0.2, False),
    "initial_porosity_corrlen_frac": ParamRange(0.05, 0.3, False),
    "operating_hours": ParamRange(0.0, 5000.0, False),
}


def _lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    # Latin Hypercube in [0,1]^d
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.uniform(size=(n_samples, n_dims))
    a = cut[:-1]
    b = cut[1:]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    # permute for each dimension
    result = np.zeros_like(rdpoints)
    for j in range(n_dims):
        order = rng.permutation(n_samples)
        result[:, j] = rdpoints[order, 0]
    return result


def _transform(value01: float, pr: ParamRange) -> float:
    if pr.log_scale:
        log_min = math.log(pr.min_value)
        log_max = math.log(pr.max_value)
        return math.exp(log_min + value01 * (log_max - log_min))
    return pr.min_value + value01 * (pr.max_value - pr.min_value)


def latin_hypercube_samples(n_samples: int, seed: int = 42) -> List[Dict[str, float]]:
    names = list(PARAM_RANGES.keys())
    rng = np.random.default_rng(seed)
    lhs = _lhs(n_samples, len(names), rng)
    samples: List[Dict[str, float]] = []

    for i in range(n_samples):
        s: Dict[str, float] = {}
        for j, name in enumerate(names):
            s[name] = _transform(lhs[i, j], PARAM_RANGES[name])

        # normalize fuel composition to sum to 1 on H2/H2O/CO/CH4
        raw = np.array([
            s["fuel_H2_frac_raw"],
            s["fuel_H2O_frac_raw"],
            s["fuel_CO_frac_raw"],
            s["fuel_CH4_frac_raw"],
        ])
        raw = np.clip(raw, 1e-6, None)
        raw = raw / raw.sum()
        s["fuel_H2_frac"] = float(raw[0])
        s["fuel_H2O_frac"] = float(raw[1])
        s["fuel_CO_frac"] = float(raw[2])
        s["fuel_CH4_frac"] = float(raw[3])
        # drop raw keys for cleanliness
        for k in ("fuel_H2_frac_raw", "fuel_H2O_frac_raw", "fuel_CO_frac_raw", "fuel_CH4_frac_raw"):
            del s[k]

        samples.append(s)

    return samples
