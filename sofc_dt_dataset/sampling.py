from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InputSample:
    sample_id: int
    Uf: float  # fuel utilization (0-1)
    T_in: float  # inlet temperature [K]
    i_avg: float  # average current density [A/m^2]
    pressure: float  # system pressure [Pa]
    y_H2_in: float  # anode inlet H2 mole fraction
    y_H2O_in: float  # anode inlet H2O mole fraction
    y_O2_in: float  # cathode inlet O2 mole fraction
    anode_thickness: float  # [m]
    electrolyte_thickness: float  # [m]
    cathode_thickness: float  # [m]
    E: float  # Young's modulus [Pa]
    nu: float  # Poisson's ratio [-]
    alpha_CTE: float  # thermal expansion [1/K]
    operating_hours: float  # representative hours under load [h]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _scale(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * u


def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    """Lightweight Latin Hypercube Sampling in [0,1]^d."""
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    a = cut[:n]
    b = cut[1:n + 1]
    points = a[:, None] + (b - a)[:, None] * u
    # permute each dimension
    for j in range(d):
        rng.shuffle(points[:, j])
    return points


def sample_inputs(num_samples: int, base_seed: int) -> InputSample:
    # Define parameter ranges (domain knowledge-inspired)
    bounds = np.array([
        [0.5, 0.9],          # Uf
        [973.0, 1173.0],     # T_in [K]
        [2000.0, 10000.0],   # i_avg [A/m^2]
        [1.0e5, 3.0e5],      # pressure [Pa]
        [0.4, 0.85],         # y_H2_in
        [0.05, 0.5],         # y_H2O_in
        [0.19, 0.30],        # y_O2_in
        [200e-6, 1000e-6],   # anode thickness [m]
        [5e-6, 50e-6],       # electrolyte thickness [m]
        [20e-6, 100e-6],     # cathode thickness [m]
        [150e9, 250e9],      # E [Pa]
        [0.20, 0.35],        # nu [-]
        [10e-6, 13e-6],      # alpha_CTE [1/K]
        [200.0, 40000.0],    # operating_hours [h]
    ])

    dim = bounds.shape[0]
    u = _lhs(num_samples, dim, seed=base_seed)

    samples = []
    for idx, ui in enumerate(u):
        vals = bounds[:, 0] + ui * (bounds[:, 1] - bounds[:, 0])
        sample = InputSample(
            sample_id=idx,
            Uf=float(vals[0]),
            T_in=float(vals[1]),
            i_avg=float(vals[2]),
            pressure=float(vals[3]),
            y_H2_in=float(vals[4]),
            y_H2O_in=float(vals[5]),
            y_O2_in=float(vals[6]),
            anode_thickness=float(vals[7]),
            electrolyte_thickness=float(vals[8]),
            cathode_thickness=float(vals[9]),
            E=float(vals[10]),
            nu=float(vals[11]),
            alpha_CTE=float(vals[12]),
            operating_hours=float(vals[13]),
        )
        samples.append(sample)

    return samples
