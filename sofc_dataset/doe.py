from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class ParamRanges:
    # Plate dimensions (meters)
    Lx: Tuple[float, float] = (0.05, 0.12)  # 50-120 mm
    Ly: Tuple[float, float] = (0.05, 0.12)
    # Layer thicknesses (meters)
    t_anode: Tuple[float, float] = (0.3e-3, 1.2e-3)
    t_electrolyte: Tuple[float, float] = (40e-6, 180e-6)
    t_cathode: Tuple[float, float] = (30e-6, 200e-6)
    # Young's moduli (Pa)
    E_anode: Tuple[float, float] = (80e9, 160e9)
    E_electrolyte: Tuple[float, float] = (170e9, 230e9)
    E_cathode: Tuple[float, float] = (120e9, 190e9)
    # Poisson's ratio
    nu_anode: Tuple[float, float] = (0.20, 0.30)
    nu_electrolyte: Tuple[float, float] = (0.20, 0.27)
    nu_cathode: Tuple[float, float] = (0.20, 0.30)
    # Coefficients of thermal expansion (1/K)
    alpha_anode: Tuple[float, float] = (11e-6, 13.5e-6)
    alpha_electrolyte: Tuple[float, float] = (9.5e-6, 11.5e-6)
    alpha_cathode: Tuple[float, float] = (11e-6, 13.0e-6)
    # Sinter eigenstrain magnitude (unitless compressive magnitude)
    es_anode: Tuple[float, float] = (0.002, 0.015)
    es_electrolyte: Tuple[float, float] = (0.000, 0.005)
    es_cathode: Tuple[float, float] = (0.000, 0.010)
    # Cooling delta T (K)
    delta_T: Tuple[float, float] = (600.0, 1200.0)
    # Creep/relaxation factor [0,1]
    relax_anode: Tuple[float, float] = (0.7, 1.0)
    relax_electrolyte: Tuple[float, float] = (0.9, 1.0)
    relax_cathode: Tuple[float, float] = (0.8, 1.0)


def latin_hypercube(n_samples: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(size=(n_samples, dim))
    a = cut[:-1]
    b = cut[1:]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    # Permute for each dimension
    H = np.zeros_like(rdpoints)
    for j in range(dim):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, 0]
    return H


def sample_params(n: int, seed: int, ranges: ParamRanges | None = None) -> List[Dict]:
    if ranges is None:
        ranges = ParamRanges()
    # List of parameter bounds in fixed order
    keys = [
        "Lx",
        "Ly",
        "t_anode",
        "t_electrolyte",
        "t_cathode",
        "E_anode",
        "E_electrolyte",
        "E_cathode",
        "nu_anode",
        "nu_electrolyte",
        "nu_cathode",
        "alpha_anode",
        "alpha_electrolyte",
        "alpha_cathode",
        "es_anode",
        "es_electrolyte",
        "es_cathode",
        "delta_T",
        "relax_anode",
        "relax_electrolyte",
        "relax_cathode",
    ]
    bounds = [getattr(ranges, k) for k in keys]
    H = latin_hypercube(n, dim=len(bounds), seed=seed)

    samples: List[Dict] = []
    for i in range(n):
        s: Dict[str, float] = {}
        for j, (low, high) in enumerate(bounds):
            s[keys[j]] = low + (high - low) * H[i, j]
        # Enforce a reasonable total thickness: keep as sampled but ensure > min
        h_total = s["t_anode"] + s["t_electrolyte"] + s["t_cathode"]
        if h_total < 0.3e-3:
            scale = 0.3e-3 / h_total
            s["t_anode"] *= scale
            s["t_electrolyte"] *= scale
            s["t_cathode"] *= scale
        samples.append(s)
    return samples
