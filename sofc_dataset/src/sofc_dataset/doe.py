from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class DOESample:
    # Geometric and process parameters for one scenario
    plate_Lx: float
    plate_Ly: float
    # Layer thicknesses (m)
    t_anode: float
    t_electrolyte: float
    t_cathode: float
    # Peak sintering temperature (K) and cool to ambient (K)
    T_peak: float
    T_ambient: float
    # Residual eigenstrains at end-of-process (fraction of free shrink)
    eigen_relax_anode: float
    eigen_relax_electrolyte: float
    eigen_relax_cathode: float


@dataclass
class DOESpace:
    # Ranges for uniform sampling
    Lx_range: Tuple[float, float]
    Ly_range: Tuple[float, float]
    t_ranges: Dict[str, Tuple[float, float]]
    T_peak_range: Tuple[float, float]
    eigen_relax_range: Tuple[float, float]


def sample_doe(space: DOESpace, n: int, rng: np.random.Generator) -> List[DOESample]:
    samples: List[DOESample] = []
    for _ in range(n):
        Lx = rng.uniform(*space.Lx_range)
        Ly = rng.uniform(*space.Ly_range)
        t_an = rng.uniform(*space.t_ranges["anode"]) 
        t_el = rng.uniform(*space.t_ranges["electrolyte"]) 
        t_ca = rng.uniform(*space.t_ranges["cathode"]) 
        T_peak = rng.uniform(*space.T_peak_range)
        T_ambient = 293.15
        er = space.eigen_relax_range
        s = DOESample(
            plate_Lx=Lx,
            plate_Ly=Ly,
            t_anode=t_an,
            t_electrolyte=t_el,
            t_cathode=t_ca,
            T_peak=T_peak,
            T_ambient=T_ambient,
            eigen_relax_anode=rng.uniform(*er),
            eigen_relax_electrolyte=rng.uniform(*er),
            eigen_relax_cathode=rng.uniform(*er),
        )
        samples.append(s)
    return samples
