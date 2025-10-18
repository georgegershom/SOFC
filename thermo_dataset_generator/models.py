from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class MicrostructureParams:
    rubber_volume_fraction: float  # 0..0.3 typical
    rubber_particle_size_factor: float  # 0.7 small, 1.0 medium, 1.3 large
    water_content_mass_fraction: float  # 0..0.12 typical
    initial_porosity: float  # 0.08..0.20
    initial_density_rho20: float  # kg/m^3
    base_thermal_conductivity_k20: float  # W/m-K
    base_specific_heat_cp20: float  # J/kg-K
    base_elastic_modulus_E20: float  # Pa
    base_compressive_strength_fc20: float  # Pa
    base_poisson_ratio_nu20: float  # dimensionless
    base_permeability_k0: float  # m^2
    base_biot_alpha: float  # 0.5..0.95


def get_microstructure_for_mix(mix_id: str) -> MicrostructureParams:
    # Baseline for control high-performance concrete (HPC)
    # Values are representative and physically plausible
    control = MicrostructureParams(
        rubber_volume_fraction=0.0,
        rubber_particle_size_factor=1.0,
        water_content_mass_fraction=0.06,
        initial_porosity=0.10,
        initial_density_rho20=2400.0,
        base_thermal_conductivity_k20=1.9,
        base_specific_heat_cp20=900.0,
        base_elastic_modulus_E20=40e9,
        base_compressive_strength_fc20=80e6,
        base_poisson_ratio_nu20=0.20,
        base_permeability_k0=1e-18,
        base_biot_alpha=0.65,
    )

    def rubberized(vr: float, size: str) -> MicrostructureParams:
        size_factor = {"S": 0.85, "M": 1.0, "L": 1.15}[size]
        # Rubber reduces density, conductivity, E, strength; increases porosity and cp slightly
        rho = control.initial_density_rho20 * (1 - 0.9 * vr)  # rubber ~ density 1100 vs 2400, but composite effect
        k0 = control.base_thermal_conductivity_k20 * (1 - 1.5 * vr) * (0.9 if size == "L" else 1.0)
        cp0 = control.base_specific_heat_cp20 * (1 + 0.2 * vr)
        E0 = control.base_elastic_modulus_E20 * (1 - 2.0 * vr) * (0.95 if size == "L" else 1.0)
        fc0 = control.base_compressive_strength_fc20 * (1 - 1.2 * vr) * (0.97 if size == "L" else 1.0)
        phi0 = control.initial_porosity + 0.03 * vr * (1.2 if size == "L" else 1.0)
        perm0 = control.base_permeability_k0 * (1 + 50 * vr * (1.3 if size == "L" else 1.0))
        biot0 = min(0.95, control.base_biot_alpha + 0.15 * vr)
        nu0 = min(0.25, control.base_poisson_ratio_nu20 + 0.02 * vr)
        return MicrostructureParams(
            rubber_volume_fraction=vr,
            rubber_particle_size_factor=size_factor,
            water_content_mass_fraction=control.water_content_mass_fraction * (1 + 0.2 * vr),
            initial_porosity=phi0,
            initial_density_rho20=rho,
            base_thermal_conductivity_k20=k0,
            base_specific_heat_cp20=cp0,
            base_elastic_modulus_E20=E0,
            base_compressive_strength_fc20=fc0,
            base_poisson_ratio_nu20=nu0,
            base_permeability_k0=perm0,
            base_biot_alpha=biot0,
        )

    mix_id = mix_id.upper()
    if mix_id == "C":
        return control
    if mix_id == "R5S":
        return rubberized(0.05, "S")
    if mix_id == "R10S":
        return rubberized(0.10, "S")
    if mix_id == "R15S":
        return rubberized(0.15, "S")
    if mix_id == "R20S":
        return rubberized(0.20, "S")
    if mix_id == "R10L":
        return rubberized(0.10, "L")
    raise ValueError(f"Unknown Mix_ID: {mix_id}")


def _sigmoid(x: float, x0: float, width: float) -> float:
    return 1.0 / (1.0 + math.exp(-(x - x0) / max(1e-9, width)))


def _gaussian(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / max(1e-9, sigma)
    return math.exp(-0.5 * z * z)


def compute_property_curves(
    micro: MicrostructureParams,
    temperature_c: List[float],
    rng: random.Random,
) -> Dict[str, List[float]]:
    # Microstructure-driven base parameters
    rho0 = micro.initial_density_rho20
    k0 = micro.base_thermal_conductivity_k20
    cp0 = micro.base_specific_heat_cp20
    E0 = micro.base_elastic_modulus_E20
    fc0 = micro.base_compressive_strength_fc20
    nu0 = micro.base_poisson_ratio_nu20
    phi0 = micro.initial_porosity
    kperm0 = micro.base_permeability_k0
    biot0 = micro.base_biot_alpha

    vr = micro.rubber_volume_fraction
    size_factor = micro.rubber_particle_size_factor
    water_mass_fraction = micro.water_content_mass_fraction

    # Sample stochastic perturbations (lognormal-ish for positive params)
    def lognormal_factor(std_frac: float) -> float:
        # Convert desired fractional std to lognormal sigma approximately
        sigma = math.sqrt(math.log(1 + std_frac * std_frac))
        return math.exp(rng.gauss(0.0, sigma))

    # Global multiplicative variability for each property
    rho_factor = lognormal_factor(0.01 + 0.10 * vr)
    k_factor = lognormal_factor(0.05 + 0.20 * vr)
    cp_factor = lognormal_factor(0.03 + 0.05 * vr)
    E_factor = lognormal_factor(0.08 + 0.25 * vr)
    fc_factor = lognormal_factor(0.08 + 0.25 * vr)
    nu_add = rng.uniform(-0.01, 0.01) * (1 + 0.5 * vr)
    phi_factor = lognormal_factor(0.05 + 0.20 * vr)
    kperm_factor = lognormal_factor(0.30 + 0.40 * vr)
    biot_add = rng.uniform(-0.02, 0.02)

    # Prepare arrays
    rho_list: List[float] = []
    k_list: List[float] = []
    cp_list: List[float] = []
    E_list: List[float] = []
    nu_list: List[float] = []
    alpha_th_list: List[float] = []
    kperm_list: List[float] = []
    phi_list: List[float] = []
    biot_list: List[float] = []
    fc_list: List[float] = []

    # Physically-based temperature functions
    for T in temperature_c:
        # Mass loss: free water ~100C, rubber pyrolysis 300-450C, CSH dehydration 400-700C
        water_loss = water_mass_fraction * _sigmoid(T, 100.0, 18.0)
        rubber_loss = 0.0
        if vr > 0.0:
            rubber_loss = 0.12 * vr * _sigmoid(T, 380.0, 40.0)  # fraction of composite mass
        csh_loss = 0.04 * _sigmoid(T, 600.0, 80.0)
        total_mass_loss = min(0.25, water_loss + rubber_loss + csh_loss)

        rho_T = max(1200.0, rho0 * rho_factor * (1.0 - total_mass_loss))
        rho_list.append(rho_T)

        # Specific heat with endothermic peak near 100C (evaporation) and broad variation
        cp_peak = 1300.0 * _gaussian(T, 105.0, 22.0) * (1 + 0.5 * water_mass_fraction)
        cp_T = max(600.0, min(2000.0, (cp0 * cp_factor) * (0.95 + 0.15 * (T / 800.0)) + cp_peak))
        cp_list.append(cp_T)

        # Porosity increases with temperature due to dehydration and microcracking
        delta_phi_water = 0.02 * _sigmoid(T, 120.0, 25.0) * (1 + 0.5 * water_mass_fraction)
        delta_phi_damage = 0.05 * _sigmoid(T, 500.0, 90.0) * (1 + 0.5 * vr * size_factor)
        phi_T = min(0.45, (phi0 * phi_factor) + delta_phi_water + delta_phi_damage)
        phi_list.append(phi_T)

        # Thermal conductivity scales with density and porosity; decreases with T due to radiation/contacts loss
        k_T = max(
            0.25,
            (k0 * k_factor)
            * (rho_T / rho0) ** 1.5
            * (1.0 - 1.8 * (phi_T - phi0))
            * (1.0 - 0.25 * (T / 800.0)),
        )
        k_list.append(k_T)

        # Elastic modulus decreases with damage, density reduction, and rubber fraction
        damage_factor = (1.0 - 0.15 * _sigmoid(T, 150.0, 30.0)) * (1.0 - 0.55 * _sigmoid(T, 500.0, 80.0))
        E_T = max(
            2.0e9,
            (E0 * E_factor)
            * (rho_T / rho0) ** 2.2
            * damage_factor
            * (1.0 - 0.8 * vr),
        )
        E_list.append(E_T)

        # Compressive strength reduction stronger than E
        fc_T = max(
            5e6,
            (fc0 * fc_factor)
            * (rho_T / rho0) ** 2.8
            * (1.0 - 0.25 * _sigmoid(T, 150.0, 30.0))
            * (1.0 - 0.70 * _sigmoid(T, 500.0, 80.0))
            * (1.0 - 0.6 * vr),
        )
        fc_list.append(fc_T)

        # Poisson's ratio typically rises slightly then may drop; keep bounded
        nu_T = max(0.15, min(0.28, (nu0 + nu_add + 0.01 * _sigmoid(T, 120.0, 25.0) - 0.01 * _sigmoid(T, 650.0, 70.0))))
        nu_list.append(nu_T)

        # Thermal expansion coefficient increases with T; rubber changes mismatch; ensure bounds
        alpha_base = 8.5e-6 + 5.5e-6 * _sigmoid(T, 550.0, 120.0)
        alpha_rubber_adjust = -2.0e-6 * vr * (1.1 if size_factor > 1.0 else 0.9)
        alpha_T = max(4e-6, min(18e-6, alpha_base + alpha_rubber_adjust))
        alpha_th_list.append(alpha_T)

        # Permeability increases with porosity and temperature-induced microcracking
        kperm_T = max(1e-20, kperm0 * kperm_factor * (phi_T / max(1e-6, phi0)) ** 4.0 * math.exp(3.0 * (T / 800.0)))
        kperm_list.append(kperm_T)

        # Biot coefficient approaches 1 as skeleton weakens
        biot_T = max(0.50, min(1.0, biot0 + biot_add + 0.25 * _sigmoid(T, 550.0, 120.0)))
        biot_list.append(biot_T)

    return {
        "Temperature_C": temperature_c,
        "rho_kgm3": rho_list,
        "k_WmK": k_list,
        "cp_JkgK": cp_list,
        "E_Pa": E_list,
        "nu": nu_list,
        "alpha_th_1K": alpha_th_list,
        "k_perm_m2": kperm_list,
        "phi": phi_list,
        "alpha_biot": biot_list,
        "fc_Pa": fc_list,
    }


def monte_carlo_summary(
    micro: MicrostructureParams,
    temperature_c: List[float],
    num_samples: int,
    seed: int,
) -> Dict[str, Tuple[List[float], List[float]]]:
    rng = random.Random(seed)
    # Collect samples per property
    property_names = [
        "rho_kgm3",
        "k_WmK",
        "cp_JkgK",
        "E_Pa",
        "nu",
        "alpha_th_1K",
        "k_perm_m2",
        "phi",
        "alpha_biot",
        "fc_Pa",
    ]
    accum: Dict[str, List[List[float]]] = {name: [] for name in property_names}

    for _ in range(num_samples):
        # Derive per-sample RNG by splitting seed deterministically
        sub_seed = rng.randrange(0, 2**31 - 1)
        curves = compute_property_curves(micro, temperature_c, random.Random(sub_seed))
        for name in property_names:
            accum[name].append(curves[name])

    # Compute mean and std at each temperature
    summary: Dict[str, Tuple[List[float], List[float]]] = {}
    for name, samples in accum.items():
        n = len(samples)
        m = len(temperature_c)
        means: List[float] = []
        stds: List[float] = []
        for j in range(m):
            vals = [samples[i][j] for i in range(n)]
            mean_v = sum(vals) / n
            var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, n - 1)
            std_v = math.sqrt(var_v)
            means.append(mean_v)
            stds.append(std_v)
        summary[name] = (means, stds)

    return summary
