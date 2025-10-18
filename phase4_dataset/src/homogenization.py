from __future__ import annotations
from typing import Tuple
from microstructure import Microstructure

# Simple mixture/homogenization rules at 20°C (baseline)

def compute_room_temperature_effective_properties(micro: Microstructure) -> Tuple[float, float, float, float]:
    """
    Returns: (E0 [Pa], k0 [W/m/K], rho0 [kg/m3], alpha0 [1/K]) at 20°C based on microstructure.
    """
    vf_r = micro.rubber_volume_fraction
    vf_a = micro.aggregate_volume_fraction
    vf_b = micro.binder_volume_fraction
    vf_void = max(0.0, 1.0 - (vf_r + vf_a + vf_b))

    # Hashin-Shtrikman upper-bound inspired simple mixing for stiffness
    E_m = micro.matrix_elastic_modulus_pa
    E_a = micro.aggregate_elastic_modulus_pa
    E_r = micro.rubber_elastic_modulus_pa

    # Sequential two-step homogenization: paste+rubber -> composite matrix, then add aggregates
    E_paste_rubber = (vf_b * E_m + vf_r * E_r) / max(1e-9, (vf_b + vf_r)) if (vf_b + vf_r) > 0 else 0.0
    # Mori-Tanaka style uplift with stiff aggregates
    E0 = E_paste_rubber * (1.0 + 4.0 * vf_a * (E_a / (E_paste_rubber + 1e-9)) / (1.0 + 3.0 * vf_a))
    E0 *= (1.0 - 1.5 * vf_void)  # reduce for voids
    E0 = max(5e9, min(60e9, E0))

    # Maxwell-Eucken for thermal conductivity
    k_m = micro.matrix_thermal_conductivity_wmk
    k_a = micro.aggregate_thermal_conductivity_wmk
    k_r = micro.rubber_thermal_conductivity_wmk
    k_void = 0.03

    def maxwell_eucken(k_matrix: float, k_incl: float, vf_incl: float) -> float:
        if vf_incl <= 0.0:
            return k_matrix
        num = k_incl + 2 * k_matrix - 2 * vf_incl * (k_matrix - k_incl)
        den = k_incl + 2 * k_matrix + vf_incl * (k_matrix - k_incl)
        return k_matrix * (num / max(1e-12, den))

    k0 = maxwell_eucken(k_m, k_r, vf_r)
    k0 = maxwell_eucken(k0, k_a, vf_a)
    # include voids
    k0 = maxwell_eucken(k0, k_void, vf_void)
    k0 = max(0.2, min(3.0, k0))

    # Density: volumetric mixture
    rho0 = (
        vf_b * micro.matrix_density_kgm3
        + vf_a * micro.aggregate_density_kgm3
        + vf_r * micro.rubber_density_kgm3
        + vf_void * 1.2
    )
    rho0 = max(1600.0, min(2500.0, rho0))

    # Coefficient of thermal expansion: volumetric average weighted by stiffness fractions
    weight_b = vf_b * E_m
    weight_a = vf_a * E_a
    weight_r = max(1e3, vf_r * E_r)
    alpha0 = (
        weight_b * micro.matrix_cte_1k
        + weight_a * micro.aggregate_cte_1k
        + weight_r * micro.rubber_cte_1k
    ) / max(1e-9, (weight_b + weight_a + weight_r))
    alpha0 = max(6e-6, min(40e-6, alpha0))

    return E0, k0, rho0, alpha0
