from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class InputSample:
    # Operating
    fuel_utilization: float  # Uf (0.6-0.9)
    inlet_temperature_K: float  # 973-1123 K
    average_current_Apcm2: float  # 0.2-1.5 A/cm^2
    anode_inlet_h2_fraction: float  # 0.4-0.8
    cathode_inlet_o2_fraction: float  # 0.15-0.25 (air ~0.21)
    anode_flow_sccm: float  # 200-1200
    cathode_flow_sccm: float  # 500-3000

    # Geometry / materials
    anode_thickness_um: float  # 300-1000
    electrolyte_thickness_um: float  # 5-20
    cathode_thickness_um: float  # 30-80
    anode_porosity: float  # 0.25-0.45
    cathode_porosity: float  # 0.25-0.45

    cycles: int  # 100-3000

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.fuel_utilization,
            self.inlet_temperature_K,
            self.average_current_Apcm2,
            self.anode_inlet_h2_fraction,
            self.cathode_inlet_o2_fraction,
            self.anode_flow_sccm,
            self.cathode_flow_sccm,
            self.anode_thickness_um,
            self.electrolyte_thickness_um,
            self.cathode_thickness_um,
            self.anode_porosity,
            self.cathode_porosity,
            self.cycles,
        ], dtype=float)

    @staticmethod
    def names() -> list[str]:
        return [
            "fuel_utilization",
            "inlet_temperature_K",
            "average_current_Apcm2",
            "anode_inlet_h2_fraction",
            "cathode_inlet_o2_fraction",
            "anode_flow_sccm",
            "cathode_flow_sccm",
            "anode_thickness_um",
            "electrolyte_thickness_um",
            "cathode_thickness_um",
            "anode_porosity",
            "cathode_porosity",
            "cycles",
        ]


def _rand_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def sample_inputs(n: int, seed: int | None = None) -> list[InputSample]:
    rng = np.random.default_rng(seed)
    samples: list[InputSample] = []
    for _ in range(n):
        samples.append(
            InputSample(
                fuel_utilization=_rand_range(rng, 0.6, 0.9),
                inlet_temperature_K=_rand_range(rng, 973.0, 1123.0),
                average_current_Apcm2=_rand_range(rng, 0.2, 1.5),
                anode_inlet_h2_fraction=_rand_range(rng, 0.4, 0.8),
                cathode_inlet_o2_fraction=_rand_range(rng, 0.15, 0.25),
                anode_flow_sccm=_rand_range(rng, 200.0, 1200.0),
                cathode_flow_sccm=_rand_range(rng, 500.0, 3000.0),
                anode_thickness_um=_rand_range(rng, 300.0, 1000.0),
                electrolyte_thickness_um=_rand_range(rng, 5.0, 20.0),
                cathode_thickness_um=_rand_range(rng, 30.0, 80.0),
                anode_porosity=_rand_range(rng, 0.25, 0.45),
                cathode_porosity=_rand_range(rng, 0.25, 0.45),
                cycles=int(rng.integers(100, 3001)),
            )
        )
    return samples
