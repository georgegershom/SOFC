from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
import math


@dataclass
class MaterialModel:
    name: str
    # Elastic properties at reference temperature (small-strain, plane stress)
    youngs_modulus_ref: float  # Pa
    poisson_ratio: float
    # Thermal expansion (linear) at reference temp
    cte_ref: float  # 1/K
    # Sintering shrinkage strain at full densification
    sintering_linear_shrinkage: float  # e.g., 0.12 means 12% linear shrinkage
    # Temperature-dependent scaling functions
    modulus_vs_T: Callable[[float], float]
    cte_vs_T: Callable[[float], float]
    # Viscous/creep softening factor (0-1) vs T for stress relaxation
    creep_relaxation_vs_T: Callable[[float], float]


def make_default_materials() -> Dict[str, MaterialModel]:
    """Provide simplified temperature-dependent properties for common SOFC layers.

    These are not authoritative; they are shaped to produce qualitatively realistic
    mismatch and shrinkage during sintering and cooling.
    """

    def steel_like_modulus(T: float) -> float:
        # Roughly decreases with T, clamp to min
        E0 = 200e9
        dE = 0.4e9 * (T - 293.15)
        return max(20e9, E0 - dE)

    def ceramic_modulus(T: float) -> float:
        E0 = 150e9
        dE = 0.2e9 * (T - 293.15)
        return max(30e9, E0 - dE)

    def cte_linear(base: float) -> Callable[[float], float]:
        def fn(T: float) -> float:
            return base * (1.0 + 2.0e-4 * (T - 293.15))
        return fn

    def creep_relaxer(T: float) -> float:
        # 0 at low T, 1 at high T (sintering window ~ 1200-1400 C)
        Tc = 1273.15
        w = 200.0
        # logistic-like
        return 1.0 / (1.0 + math.exp(-(T - Tc) / w))

    anode = MaterialModel(
        name="anode",
        youngs_modulus_ref=120e9,
        poisson_ratio=0.28,
        cte_ref=11.0e-6,
        sintering_linear_shrinkage=0.10,
        modulus_vs_T=ceramic_modulus,
        cte_vs_T=cte_linear(11.0e-6),
        creep_relaxation_vs_T=creep_relaxer,
    )

    electrolyte = MaterialModel(
        name="electrolyte",
        youngs_modulus_ref=180e9,
        poisson_ratio=0.25,
        cte_ref=10.5e-6,
        sintering_linear_shrinkage=0.02,
        modulus_vs_T=ceramic_modulus,
        cte_vs_T=cte_linear(10.5e-6),
        creep_relaxation_vs_T=creep_relaxer,
    )

    cathode = MaterialModel(
        name="cathode",
        youngs_modulus_ref=140e9,
        poisson_ratio=0.27,
        cte_ref=12.0e-6,
        sintering_linear_shrinkage=0.05,
        modulus_vs_T=ceramic_modulus,
        cte_vs_T=cte_linear(12.0e-6),
        creep_relaxation_vs_T=creep_relaxer,
    )

    return {m.name: m for m in [anode, electrolyte, cathode]}
