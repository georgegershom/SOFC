from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import numpy as np


@dataclass(frozen=True)
class Parameter:
    name: str
    bounds: Tuple[float, float]
    # Optional discrete values; if provided, we map LHS quantiles to nearest
    choices: Sequence[float] | None = None


class LatinHypercubeSampler:
    def __init__(self, parameters: List[Parameter], random_seed: int | None = 42):
        self.parameters = parameters
        self.rng = np.random.default_rng(random_seed)

    def sample(self, num_samples: int) -> Dict[str, np.ndarray]:
        # Classic LHS: stratify [0,1] into N bins, jitter within each, then permute per dimension
        u = (np.arange(num_samples) + self.rng.random(num_samples)) / num_samples
        samples = {}
        for p in self.parameters:
            perm = self.rng.permutation(num_samples)
            uu = u[perm]
            if p.choices is not None and len(p.choices) > 0:
                # Map quantiles to discrete choices
                q = np.clip(np.floor(uu * len(p.choices)).astype(int), 0, len(p.choices) - 1)
                vals = np.array(p.choices, dtype=float)[q]
            else:
                low, high = p.bounds
                vals = low + uu * (high - low)
            samples[p.name] = vals
        return samples


def default_parameter_space() -> List[Parameter]:
    # Operating conditions
    params = [
        Parameter("current_density_A_per_cm2", (0.1, 1.5)),
        Parameter("fuel_utilization_pct", (40.0, 90.0)),
        Parameter("air_utilization_pct", (10.0, 60.0)),
        Parameter("inlet_fuel_temp_C", (500.0, 800.0)),
        Parameter("inlet_air_temp_C", (500.0, 800.0)),
        Parameter("fuel_H2_pct", (40.0, 100.0)),
        Parameter("fuel_H2O_pct", (0.0, 40.0)),
        Parameter("fuel_CO_pct", (0.0, 20.0)),
        Parameter("fuel_CH4_pct", (0.0, 30.0)),
        # Material & geometric
        Parameter("anode_porosity", (0.2, 0.5)),
        Parameter("cathode_porosity", (0.2, 0.5)),
        Parameter("anode_tortuosity", (2.0, 6.0)),
        Parameter("cathode_tortuosity", (2.0, 6.0)),
        Parameter("anode_ionic_cond_S_per_m", (1000.0, 10000.0)),
        Parameter("cathode_ionic_cond_S_per_m", (1000.0, 10000.0)),
        Parameter("electrolyte_thickness_um", (5.0, 30.0)),
        Parameter("anode_thickness_um", (200.0, 800.0)),
        Parameter("cathode_thickness_um", (20.0, 100.0)),
        Parameter("interconnect_CTE_ppmK", (10.0, 16.0)),
        Parameter("interconnect_E_GPa", (150.0, 220.0)),
        # Degradation / initial conditions
        Parameter("init_crack_length_mm", (0.0, 5.0)),
        Parameter("init_delamination_mm", (0.0, 5.0)),
        Parameter("init_porosity_variance", (0.0, 0.1)),
    ]
    return params
