from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from microstructure import Microstructure
from homogenization import compute_room_temperature_effective_properties


@dataclass
class BaselineEffective:
    elastic_modulus_pa: float
    thermal_conductivity_wmk: float
    density_kgm3: float
    cte_1k: float


def compute_baseline_effective(micro: Microstructure) -> BaselineEffective:
    e0, k0, rho0, alpha0 = compute_room_temperature_effective_properties(micro)
    return BaselineEffective(
        elastic_modulus_pa=e0,
        thermal_conductivity_wmk=k0,
        density_kgm3=rho0,
        cte_1k=alpha0,
    )


def temperature_grid_c(start_c: float = 20.0, end_c: float = 800.0, step_c: float = 10.0) -> List[float]:
    n = int(round((end_c - start_c) / step_c))
    return [start_c + i * step_c for i in range(n + 1)]


def logistic(x: float, x0: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def bounded(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def compute_degradation_and_state(micro: Microstructure, base: BaselineEffective, t_c: float) -> Dict[str, float]:
    vf_r = micro.rubber_volume_fraction

    # Rubber phase integrity: transitions rapidly between 150-300°C due to softening/pyrolysis
    r_soften = logistic(t_c, 160.0, 0.05)
    r_pyro = logistic(t_c, 280.0, 0.06)
    rubber_integrity = 1.0 - 0.75 * r_soften - 0.10 * r_pyro
    rubber_integrity = bounded(rubber_integrity, 0.1, 1.0)

    # Free-water dehydration
    dehyd_free = logistic(t_c, 100.0, 0.10)
    # Bound-water and portlandite dehydration
    dehyd_bound = 0.7 * logistic(t_c, 450.0, 0.03)
    dehydration_fraction = bounded(0.7 * dehyd_free + 0.3 * dehyd_bound, 0.0, 1.0)

    # Microcracking/damage proxy increasing with temperature and rubber softening
    base_damage = 0.85 * logistic(t_c, 450.0, 0.010)
    rubber_damage = 0.6 * vf_r * logistic(t_c, 220.0, 0.030)
    damage = bounded(base_damage + rubber_damage, 0.0, 0.98)

    # Porosity growth from dehydration and microcracking
    phi0 = micro.initial_porosity
    dphi_dehyd = 0.12 * dehydration_fraction * (0.04 + 0.5 * micro.water_mass_fraction)
    dphi_damage = 0.08 * damage * (1.0 + 1.5 * vf_r)
    phi = bounded(phi0 + dphi_dehyd + dphi_damage, 0.05, 0.45)

    # Effective void fraction for thermal transport
    effective_void_vf = bounded(phi + 0.02 * vf_r * (1.0 - rubber_integrity), 0.0, 0.60)

    return {
        "rubber_integrity": rubber_integrity,
        "dehydration_fraction": dehydration_fraction,
        "damage": damage,
        "porosity": phi,
        "effective_void_vf": effective_void_vf,
    }


def compute_properties_vs_temperature(micro: Microstructure, t_c: float) -> Dict[str, float]:
    base = compute_baseline_effective(micro)
    state = compute_degradation_and_state(micro, base, t_c)

    vf_r = micro.rubber_volume_fraction

    # Density
    rho0 = base.density_kgm3
    water_loss = micro.water_mass_fraction * state["dehydration_fraction"]
    rubber_mass_loss = 0.9 * vf_r * (1.0 - state["rubber_integrity"]) * (micro.rubber_density_kgm3 / rho0)
    rho = rho0 * (1.0 - 0.85 * water_loss - 0.8 * rubber_mass_loss)
    rho = bounded(rho, 1200.0, 2600.0)

    # Specific heat (effective), baseline plus peaks near 100°C and 500°C
    cp_base = 850.0 + 0.08 * (t_c - 20.0)
    peak100 = 1600.0 * math.exp(-((t_c - 105.0) ** 2) / (2 * 22.0 ** 2))
    peak500 = 500.0 * math.exp(-((t_c - 520.0) ** 2) / (2 * 45.0 ** 2))
    cp = bounded(cp_base + peak100 + peak500, 500.0, 3500.0)

    # Thermal conductivity: Maxwell-Eucken with increased voids + temperature effect
    k0 = base.thermal_conductivity_wmk
    kv = 0.03
    vf_void_eff = state["effective_void_vf"]
    def maxwell(km: float, ki: float, vf: float) -> float:
        if vf <= 0.0:
            return km
        num = ki + 2 * km - 2 * vf * (km - ki)
        den = ki + 2 * km + vf * (km - ki)
        return km * (num / max(1e-12, den))

    # First, base k reduced by rubber state
    rubber_k_eff = micro.rubber_thermal_conductivity_wmk * (0.5 + 0.5 * state["rubber_integrity"])
    paste_k_eff = micro.matrix_thermal_conductivity_wmk * (1.0 - 0.2 * state["dehydration_fraction"]) \
        * (1.0 - 0.25 * state["damage"]) 
    # Combine phases roughly via two-step
    k_mix = maxwell(paste_k_eff, rubber_k_eff, vf_r)
    k_mix = maxwell(k_mix, micro.aggregate_thermal_conductivity_wmk, micro.aggregate_volume_fraction)
    k = maxwell(k_mix, kv, vf_void_eff)
    k *= 1.0 - 0.00025 * max(0.0, t_c - 20.0)
    k = bounded(k, 0.15, 3.0)

    # Elastic modulus degradation
    e0 = base.elastic_modulus_pa
    d1 = 0.62 * logistic(t_c, 420.0, 0.012)
    d2 = 0.22 * logistic(t_c, 600.0, 0.020)
    d_rubber = 0.45 * vf_r * logistic(t_c, 220.0, 0.025)
    damage_e = bounded(d1 + d2 + d_rubber, 0.0, 0.97)
    e = e0 * (1.0 - damage_e)
    e = bounded(e, 1.0e9, e0)

    # Poisson's ratio
    nu0 = micro.base_poisson_ratio
    nu = bounded(nu0 + 0.04 * state["damage"] + 0.015 * vf_r * (1.0 - state["rubber_integrity"]), 0.18, 0.36)

    # Thermal expansion coefficient
    alpha0 = base.cte_1k
    alpha = alpha0 * (1.0 + 0.6 * state["damage"] + 0.2 * state["dehydration_fraction"]) \
        * (1.0 + 0.35 * vf_r * (1.0 - state["rubber_integrity"]))
    alpha = bounded(alpha, 6e-6, 80e-6)

    # Free thermal strain relative to 20°C
    # Approximate by alpha(T) * (T - 20°C) with mild nonlinearity due to state changes
    free_thermal_strain = alpha * (t_c - 20.0) * (1.0 + 0.15 * state["damage"]) 

    # Compressive strength: control slight rise up to 100°C then decay
    fc0 = 90e6 * (1.0 - 0.8 * vf_r)
    strength_gain = 0.05 * math.exp(-((t_c - 100.0) ** 2) / (2 * 30.0 ** 2))
    strength_loss = logistic(t_c, 350.0, 0.012) * 0.65 + logistic(t_c, 600.0, 0.020) * 0.20
    fc = fc0 * (1.0 + strength_gain) * (1.0 - bounded(strength_loss + 0.3 * vf_r * (1.0 - state["rubber_integrity"]), 0.0, 0.9))
    fc = bounded(fc, 5e6, 120e6)

    # Permeability (Darcy): exponential increase with porosity and damage
    k_perm0 = micro.base_permeability_m2
    k_perm = k_perm0 * math.exp(18.0 * (state["porosity"] - micro.initial_porosity)) * math.exp(8.0 * state["damage"]) \
        * (1.0 + 6.0 * vf_r * (1.0 - state["rubber_integrity"]))
    k_perm = bounded(k_perm, 1e-20, 1e-14)

    # Biot coefficient
    alpha_biot = bounded(
        micro.base_biot_coefficient
        + 0.25 * state["damage"]
        + 0.15 * (state["porosity"] - micro.initial_porosity),
        0.55,
        0.98,
    )

    # Moisture diffusivity proxy (m^2/s), increases with T and porosity
    d_moisture = 1e-11 * (1.0 + 0.02 * (t_c - 20.0)) * (1.0 + 30.0 * (state["porosity"] - micro.initial_porosity))
    d_moisture = bounded(d_moisture, 1e-12, 1e-8)

    return {
        "density_kgm3": rho,
        "specific_heat_jkgk": cp,
        "thermal_conductivity_wmk": k,
        "elastic_modulus_pa": e,
        "poisson_ratio": nu,
        "cte_1k": alpha,
        "free_thermal_strain": free_thermal_strain,
        "compressive_strength_pa": fc,
        "permeability_m2": k_perm,
        "biot_coefficient": alpha_biot,
        "moisture_diffusivity_m2s": d_moisture,
        "porosity": state["porosity"],
        "rubber_integrity": state["rubber_integrity"],
        "dehydration_fraction": state["dehydration_fraction"],
        "damage_index": state["damage"],
    }
