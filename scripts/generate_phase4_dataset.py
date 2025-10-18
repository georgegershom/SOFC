#!/usr/bin/env python3

"""
Phase 4: Numerical Modeling Dataset Generator

Topic: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant
Structural Elements Utilizing High-Performance Rubberized Concrete (HPRC).

This script synthesizes:
- Model Input Data: temperature-dependent material properties for multiple mixtures
  (HPC reference, HPRC low-rubber, HPRC high-rubber).
- Model Validation Data: transient thermocouple temperatures in a 1D heated slab,
  deformation/strain history under sustained load during heating, and spalling events
  plus time-to-failure estimate under combined thermal and mechanical effects.

No external dependencies beyond the Python standard library.

All units are SI unless otherwise stated.
"""

from __future__ import annotations

import os
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable


# ----------------------------- Paths and Utilities -----------------------------

def get_base_dirs() -> Dict[str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "datasets", "phase4_numerical_modeling")
    paths = {
        "project_root": project_root,
        "base": base_dir,
        "input": os.path.join(base_dir, "model_input"),
        "validation": os.path.join(base_dir, "model_validation"),
        "meta": os.path.join(base_dir, "meta"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def write_csv(path: str, header: List[str], rows: Iterable[Iterable[float | int | str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------- Mixtures and Params -----------------------------

@dataclass
class MaterialMixture:
    name: str
    description: str
    rubber_volume_fraction: float  # 0.0 - 0.30 typical
    baseline_thermal_conductivity_W_mK: float  # at 20 C
    baseline_specific_heat_J_kgK: float  # at 20 C
    baseline_density_kg_m3: float  # at 20 C
    baseline_compressive_strength_MPa: float  # fc at 28 days, 20 C
    baseline_tensile_strength_MPa: float  # splitting or direct tensile
    baseline_elastic_modulus_GPa: float  # at 20 C
    poisson_ratio_20C: float  # nominal at 20 C
    nominal_porosity: float  # 0-1 (effective porosity)


# Reasonable baseline values for high-performance concrete and rubberized variants
HPC_REF = MaterialMixture(
    name="HPC_ref",
    description="High-performance concrete reference (no rubber).",
    rubber_volume_fraction=0.0,
    baseline_thermal_conductivity_W_mK=2.2,
    baseline_specific_heat_J_kgK=900.0,
    baseline_density_kg_m3=2350.0,
    baseline_compressive_strength_MPa=70.0,
    baseline_tensile_strength_MPa=5.5,
    baseline_elastic_modulus_GPa=38.0,
    poisson_ratio_20C=0.20,
    nominal_porosity=0.12,
)

HPRC_LOW = MaterialMixture(
    name="HPRC_lowRubber",
    description="HPRC with ~10% rubber by volume.",
    rubber_volume_fraction=0.10,
    baseline_thermal_conductivity_W_mK=1.6,
    baseline_specific_heat_J_kgK=950.0,
    baseline_density_kg_m3=2100.0,
    baseline_compressive_strength_MPa=58.0,
    baseline_tensile_strength_MPa=4.6,
    baseline_elastic_modulus_GPa=32.0,
    poisson_ratio_20C=0.22,
    nominal_porosity=0.17,
)

HPRC_HIGH = MaterialMixture(
    name="HPRC_highRubber",
    description="HPRC with ~20% rubber by volume.",
    rubber_volume_fraction=0.20,
    baseline_thermal_conductivity_W_mK=1.3,
    baseline_specific_heat_J_kgK=980.0,
    baseline_density_kg_m3=1950.0,
    baseline_compressive_strength_MPa=50.0,
    baseline_tensile_strength_MPa=4.0,
    baseline_elastic_modulus_GPa=28.0,
    poisson_ratio_20C=0.24,
    nominal_porosity=0.21,
)

MIXTURES = [HPC_REF, HPRC_LOW, HPRC_HIGH]


# ----------------------------- Property Models vs T -----------------------------

# Temperature domain for tabulated properties
T_MIN_C = 20.0
T_MAX_C = 1200.0
T_STEP_C = 10.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def thermal_conductivity_W_mK(T_C: float, mix: MaterialMixture) -> float:
    # Rubber reduces connectivity of mineral skeleton; conductivity drops with T.
    # Piecewise smooth decrease with mild minimum near 150 C due to evap.
    k20 = mix.baseline_thermal_conductivity_W_mK
    r = mix.rubber_volume_fraction
    k = (
        k20
        * (1.0 - 0.15 * (T_C / 200.0))
        * (1.0 - 0.35 * (T_C / 1200.0))
    )
    k *= (1.0 - 0.4 * r)
    # moisture effect dip between 80-180 C
    if 80.0 <= T_C <= 180.0:
        k *= 0.85
    return clamp(k, 0.25, k20)


def specific_heat_J_kgK(T_C: float, mix: MaterialMixture) -> float:
    # cp rises with T with a dehydration peak near 100-150 C.
    cp0 = mix.baseline_specific_heat_J_kgK
    peak = 2800.0 * math.exp(-((T_C - 120.0) ** 2) / (2 * 35.0 ** 2))
    tail = 200.0 * (1.0 - math.exp(-T_C / 400.0))
    cp = cp0 + peak + tail
    return clamp(cp, 700.0, 2800.0 + cp0 + 250.0)


def density_kg_m3(T_C: float, mix: MaterialMixture) -> float:
    # Density reduces with T due to moisture loss and decomposition.
    rho0 = mix.baseline_density_kg_m3
    # staged mass loss: gentle to 200 C, stronger to 800 C, then pronounced
    loss = 0.02 * min(T_C / 200.0, 1.0) + 0.08 * clamp((T_C - 200.0) / 600.0, 0.0, 1.0) + 0.08 * clamp((T_C - 800.0) / 400.0, 0.0, 1.0)
    rho = rho0 * (1.0 - loss)
    # rubber content further reduces effective density at high T
    rho *= (1.0 - 0.05 * mix.rubber_volume_fraction * clamp((T_C - 200.0) / 400.0, 0.0, 1.0))
    return clamp(rho, 1200.0, rho0)


def residual_ratio_monotone(T_C: float, T_knots: List[float], r_knots: List[float]) -> float:
    # Linear interpolation of residual ratios vs T; knots monotone decreasing.
    for i in range(len(T_knots) - 1):
        if T_knots[i] <= T_C <= T_knots[i + 1]:
            t0, t1 = T_knots[i], T_knots[i + 1]
            r0, r1 = r_knots[i], r_knots[i + 1]
            if t1 == t0:
                return r0
            w = (T_C - t0) / (t1 - t0)
            return r0 * (1 - w) + r1 * w
    return r_knots[-1] if T_C > T_knots[-1] else r_knots[0]


def compressive_strength_MPa(T_C: float, mix: MaterialMixture) -> float:
    # Inspired by Eurocode-type residuals, adjusted for HPRC
    T_knots = [20, 200, 400, 600, 800, 1000, 1200]
    base = [1.00, 0.95, 0.80, 0.55, 0.32, 0.18, 0.10]
    penalty = 0.08 * mix.rubber_volume_fraction  # HPRC degrades a bit faster in compression
    r = [max(0.05, b - penalty) for b in base]
    ratio = residual_ratio_monotone(T_C, T_knots, r)
    return mix.baseline_compressive_strength_MPa * ratio


def tensile_strength_MPa(T_C: float, mix: MaterialMixture) -> float:
    # Tensile degrades faster than compressive
    T_knots = [20, 200, 400, 600, 800, 1000, 1200]
    base = [1.00, 0.85, 0.55, 0.28, 0.12, 0.06, 0.03]
    improvement = 0.05 * mix.rubber_volume_fraction  # rubber adds ductility, slower early loss
    r = [clamp(b + improvement, 0.02, 1.05) for b in base]
    ratio = residual_ratio_monotone(T_C, T_knots, r)
    return mix.baseline_tensile_strength_MPa * ratio


def elastic_modulus_GPa(T_C: float, mix: MaterialMixture) -> float:
    T_knots = [20, 200, 400, 600, 800, 1000, 1200]
    base = [1.00, 0.85, 0.60, 0.35, 0.18, 0.10, 0.06]
    softening = 0.10 * mix.rubber_volume_fraction
    r = [max(0.04, b - softening) for b in base]
    ratio = residual_ratio_monotone(T_C, T_knots, r)
    return mix.baseline_elastic_modulus_GPa * ratio


def poisson_ratio(T_C: float, mix: MaterialMixture) -> float:
    # Slight increase with T due to microcracking until 600 C, then decrease
    nu0 = mix.poisson_ratio_20C
    if T_C <= 600.0:
        return clamp(nu0 + 0.00015 * T_C / 1.0, 0.15, 0.34)
    else:
        return clamp(nu0 + 0.09 - 0.00010 * (T_C - 600.0), 0.12, 0.34)


def cte_alpha_per_K(T_C: float, mix: MaterialMixture) -> float:
    # Coefficient of thermal expansion; non-linear with T
    # Baseline ~10-12e-6/K; increases then stabilizes
    alpha0 = 10.5e-6 * (1.0 - 0.2 * mix.rubber_volume_fraction)
    alpha = alpha0 * (1.0 + 0.8 * (1.0 - math.exp(-T_C / 400.0)))
    return clamp(alpha, 6.5e-6, 20e-6)


def transient_thermal_strain(T_C: float, mix: MaterialMixture) -> float:
    # Negative strain due to drying and microstructural relaxation during heating
    # Increases in magnitude up to ~600 C then saturates
    cap = -0.0022 * (1.0 - 0.4 * mix.rubber_volume_fraction)
    mag = cap * (1.0 - math.exp(-T_C / 350.0))
    return mag


def porosity(T_C: float, mix: MaterialMixture) -> float:
    # Porosity increases with T; HPRC starts higher and increases more
    phi0 = mix.nominal_porosity
    dphi = 0.10 * (1.0 - math.exp(-T_C / 500.0))
    extra = 0.05 * mix.rubber_volume_fraction * (1.0 - math.exp(-T_C / 300.0))
    phi = phi0 + dphi + extra
    return clamp(phi, 0.06, 0.40)


def intrinsic_permeability_m2(T_C: float, damage: float, mix: MaterialMixture) -> float:
    # Base permeability very low; exponential growth with T and damage
    base_perm = 1e-19 * (1.0 + 3.0 * mix.rubber_volume_fraction)
    temp_factor = math.exp(3.5 * clamp((T_C - 150.0) / 500.0, 0.0, 1.0))
    damage_factor = math.exp(4.0 * clamp(damage, 0.0, 1.0))
    perm = base_perm * temp_factor * damage_factor
    return clamp(perm, 1e-21, 5e-15)


# ----------------------------- ISO 834 Fire Curve ------------------------------

def iso834_gas_temperature_C(t_seconds: float) -> float:
    # ISO 834: T_gas [C] = 20 + 345 * log10(8*t_min + 1)
    t_min = t_seconds / 60.0
    return 20.0 + 345.0 * math.log10(8.0 * t_min + 1.0)


# ----------------------------- 1D Heat Simulation ------------------------------

@dataclass
class SlabConfig:
    thickness_m: float
    num_nodes: int
    t_end_s: float
    dt_s: float


def simulate_1d_heating(mix: MaterialMixture, cfg: SlabConfig, record_dt_s: float,
                         thermocouple_depths_m: List[float]) -> Tuple[List[float], Dict[float, List[float]], List[float]]:
    """
    Returns time_s, temperatures_by_depth_C, avg_temperature_C.
    Boundary conditions:
    - Exposed face x=0: Dirichlet to ISO 834 gas temperature.
    - Unexposed face x=L: Adiabatic (zero gradient).
    """
    nx = cfg.num_nodes
    L = cfg.thickness_m
    dx = L / (nx - 1)
    # Initialize temperature profile
    T = [20.0 for _ in range(nx)]
    time_s: List[float] = []
    avg_T: List[float] = []
    tc_positions_idx: Dict[float, int] = {}
    for d in thermocouple_depths_m:
        idx = int(round(d / dx))
        idx = clamp(idx, 0, nx - 1)  # type: ignore[arg-type]
        tc_positions_idx[d] = int(idx)

    temperatures_by_depth: Dict[float, List[float]] = {d: [] for d in thermocouple_depths_m}

    next_record_time = 0.0
    t = 0.0
    steps = 0
    max_steps = int(math.ceil(cfg.t_end_s / cfg.dt_s)) + 2

    while t <= cfg.t_end_s + 1e-9 and steps <= max_steps:
        # Apply boundary at exposed face
        T[0] = iso834_gas_temperature_C(t)
        # Compute alpha per node from last T
        alpha = []
        for i in range(nx):
            k = thermal_conductivity_W_mK(T[i], mix)
            cp = specific_heat_J_kgK(T[i], mix)
            rho = density_kg_m3(T[i], mix)
            alpha.append(k / (rho * cp))
        # Update interior nodes using explicit scheme with variable alpha
        Tnew = T[:]
        for i in range(1, nx - 1):
            r = alpha[i] * cfg.dt_s / (dx * dx)
            Tnew[i] = T[i] + r * (T[i + 1] - 2.0 * T[i] + T[i - 1])
        # Adiabatic at far face: dT/dx = 0 => mirror last gradient
        Tnew[nx - 1] = Tnew[nx - 2]
        T = Tnew

        # Record at intervals
        if t >= next_record_time - 1e-9:
            time_s.append(t)
            mean_T = sum(T) / nx
            avg_T.append(mean_T)
            for d, idx in tc_positions_idx.items():
                temperatures_by_depth[d].append(T[idx])
            next_record_time += record_dt_s
        
        t += cfg.dt_s
        steps += 1

    return time_s, temperatures_by_depth, avg_T


# ----------------------------- Strain and Damage -------------------------------

def compute_strain_history(time_s: List[float], avg_temp_C: List[float], mix: MaterialMixture,
                           sustained_stress_ratio: float) -> Dict[str, List[float]]:
    """
    Returns dict with time_s, eps_thermal, eps_tts, eps_mech, eps_total arrays.
    """
    assert len(time_s) == len(avg_temp_C)
    n = len(time_s)
    eps_th = [0.0] * n
    eps_tts = [0.0] * n
    eps_mech = [0.0] * n
    eps_total = [0.0] * n

    fc0 = mix.baseline_compressive_strength_MPa
    E0 = mix.baseline_elastic_modulus_GPa * 1e9
    sigma = sustained_stress_ratio * fc0 * 1e6  # Pa

    # Integrate thermal strain by cumulative sum of alpha(T) dT
    last_T = avg_temp_C[0]
    cum_eps_th = 0.0

    for i in range(n):
        T = avg_temp_C[i]
        dT = T - last_T
        alpha_mid = cte_alpha_per_K(0.5 * (T + last_T), mix)
        cum_eps_th += alpha_mid * dT
        eps_th[i] = cum_eps_th
        eps_tts[i] = transient_thermal_strain(T, mix)
        E_T = max(0.05 * E0, elastic_modulus_GPa(T, mix) * 1e9)
        eps_mech[i] = sigma / E_T
        eps_total[i] = eps_th[i] + eps_tts[i] + eps_mech[i]
        last_T = T

    # Add small measurement noise to mimic experiments
    rng = random.Random(42)
    noise = lambda s: s + rng.gauss(0.0, 1e-5)
    eps_th = [noise(x) for x in eps_th]
    eps_tts = [noise(x) for x in eps_tts]
    eps_mech = [noise(x) for x in eps_mech]
    eps_total = [noise(x) for x in eps_total]

    return {
        "time_s": time_s,
        "eps_thermal": eps_th,
        "eps_tts": eps_tts,
        "eps_mech": eps_mech,
        "eps_total": eps_total,
    }


def spalling_and_failure(time_s: List[float], surface_temp_C: List[float], avg_temp_C: List[float],
                         mix: MaterialMixture, sustained_stress_ratio: float,
                         nominal_width_m: float, nominal_thickness_m: float) -> Dict:
    """
    Heuristic spalling events and failure time based on capacity vs load and surface heating.
    Returns dict with events and time_to_failure_s.
    """
    fc0 = mix.baseline_compressive_strength_MPa * 1e6
    A0 = nominal_width_m * nominal_thickness_m
    F_load = sustained_stress_ratio * fc0 * A0

    spall_events = []  # list of {time_s, depth_m, reason}
    effective_thickness_m = nominal_thickness_m

    last_surface_T = surface_temp_C[0]
    last_time = time_s[0]

    for i in range(1, len(time_s)):
        t = time_s[i]
        Ts = surface_temp_C[i]
        dTdt = (Ts - last_surface_T) / max(1e-6, (t - last_time))
        ft = tensile_strength_MPa(avg_temp_C[i], mix) * 1e6
        phi = porosity(avg_temp_C[i], mix)
        risk = 0.0
        risk += clamp((Ts - 350.0) / 200.0, 0.0, 1.0)  # temperature level risk
        risk += clamp(dTdt / 1.0, 0.0, 1.0) * 0.3       # heating rate risk (~1 C/s threshold)
        risk += clamp((0.20 + 0.6 * mix.rubber_volume_fraction - (ft / 5e6)), 0.0, 1.0) * 0.5
        risk += clamp((phi - 0.14) / 0.12, 0.0, 1.0) * 0.4

        if risk >= 1.2 and Ts > 380.0:
            # Generate a spall event of 5-18 mm depending on porosity
            depth = clamp(0.005 + 0.020 * (phi - 0.10) / 0.20 + random.uniform(0.0, 0.005), 0.004, 0.020)
            if effective_thickness_m - depth >= 0.05:  # ensure section remains meaningful
                spall_events.append({
                    "time_s": t,
                    "depth_m": round(depth, 5),
                    "surface_temp_C": round(Ts, 1),
                    "risk_index": round(risk, 2),
                    "reason": "Combined thermal, rate, tensile and porosity risk",
                })
                effective_thickness_m -= depth
                # Reduce instantaneous surface temperature a touch due to removal
                last_surface_T = max(20.0, Ts - 5.0)
                last_time = t
                continue  # skip capacity check this cycle to apply spall first

        last_surface_T = Ts
        last_time = t

    # Capacity vs load over time to find failure
    time_to_failure_s = None
    for i in range(len(time_s)):
        fc_T = compressive_strength_MPa(avg_temp_C[i], mix) * 1e6
        # reduction factor due to spalling (reduced thickness)
        A_eff = nominal_width_m * effective_thickness_m
        F_cap = 0.85 * fc_T * A_eff
        if F_cap < F_load:
            time_to_failure_s = time_s[i]
            break

    return {
        "events": spall_events,
        "effective_thickness_m": round(effective_thickness_m, 4),
        "time_to_failure_s": time_to_failure_s,
        "failure_mode": "capacity_drop_below_load" if time_to_failure_s is not None else None,
    }


# ----------------------------- Permeability Grid -------------------------------

def damage_index(T_C: float, stress_ratio: float) -> float:
    # Simple additive damage index combining thermal and mechanical contributions
    dT = clamp((T_C - 200.0) / 900.0, 0.0, 1.0)
    dM = clamp(1.2 * stress_ratio, 0.0, 1.0)
    return clamp(0.7 * dT + 0.3 * dM, 0.0, 1.0)


def generate_permeability_rows(mix: MaterialMixture, stress_ratios: List[float]) -> List[List[float]]:
    rows: List[List[float]] = []
    for T_C in frange(T_MIN_C, T_MAX_C, T_STEP_C):
        for sr in stress_ratios:
            D = damage_index(T_C, sr)
            perm = intrinsic_permeability_m2(T_C, D, mix)
            rows.append([round(T_C, 1), round(sr, 2), round(D, 4), f"{perm:.4e}"])
    return rows


# ----------------------------- Helpers -----------------------------

def frange(start: float, stop: float, step: float) -> Iterable[float]:
    x = start
    while x <= stop + 1e-9:
        yield x
        x += step


# ----------------------------- Main Generation -----------------------------

def generate_material_properties(paths: Dict[str, str], mix: MaterialMixture) -> Dict[str, str]:
    rows = []
    for T_C in frange(T_MIN_C, T_MAX_C, T_STEP_C):
        k = thermal_conductivity_W_mK(T_C, mix)
        cp = specific_heat_J_kgK(T_C, mix)
        rho = density_kg_m3(T_C, mix)
        fc = compressive_strength_MPa(T_C, mix)
        ft = tensile_strength_MPa(T_C, mix)
        E = elastic_modulus_GPa(T_C, mix)
        nu = poisson_ratio(T_C, mix)
        alpha = cte_alpha_per_K(T_C, mix)
        eps_tts = transient_thermal_strain(T_C, mix)
        phi = porosity(T_C, mix)
        rows.append([
            round(T_C, 1), f"{k:.4f}", f"{cp:.1f}", f"{rho:.1f}", f"{fc:.3f}", f"{ft:.3f}",
            f"{E:.3f}", f"{nu:.4f}", f"{alpha:.6e}", f"{eps_tts:.6e}", f"{phi:.4f}"
        ])

    header = [
        "T_C", "k_W_mK", "cp_J_kgK", "rho_kg_m3", "fc_MPa", "ft_MPa",
        "E_GPa", "nu", "alpha_1_K", "eps_tts", "porosity"
    ]
    out_path = os.path.join(paths["input"], f"material_properties_{mix.name}.csv")
    write_csv(out_path, header, rows)

    # Permeability grid
    perm_header = ["T_C", "stress_ratio", "damage_index", "intrinsic_permeability_m2"]
    perm_rows = generate_permeability_rows(mix, [0.0, 0.2, 0.4, 0.6])
    perm_path = os.path.join(paths["input"], f"permeability_{mix.name}.csv")
    write_csv(perm_path, perm_header, perm_rows)

    return {
        "properties_csv": out_path,
        "permeability_csv": perm_path,
    }


def generate_validation(paths: Dict[str, str], mix: MaterialMixture) -> Dict[str, str]:
    # Heat simulation config
    slab = SlabConfig(thickness_m=0.20, num_nodes=41, t_end_s=7200.0, dt_s=0.5)
    record_dt_s = 5.0
    depths_m = [0.010, 0.050, 0.100, 0.150]

    time_s, temp_by_depth, avg_T = simulate_1d_heating(
        mix, slab, record_dt_s, depths_m
    )

    # Add measurement-like noise to thermocouples (±1.5 C std dev)
    rng = random.Random(7)
    for d in depths_m:
        noisy = []
        for v in temp_by_depth[d]:
            noisy.append(v + rng.gauss(0.0, 1.5))
        temp_by_depth[d] = noisy

    # Thermocouple CSV
    header = ["time_s", "T_surface_C"] + [f"T_{int(d*1000)}mm_C" for d in depths_m]
    rows = []
    for i, t in enumerate(time_s):
        surface_T = iso834_gas_temperature_C(t)
        row = [int(round(t)), f"{surface_T:.2f}"]
        for d in depths_m:
            row.append(f"{temp_by_depth[d][i]:.2f}")
        rows.append(row)
    tc_path = os.path.join(paths["validation"], f"thermocouples_{mix.name}.csv")
    write_csv(tc_path, header, rows)

    # Strain history under sustained load ratio (e.g., 0.30)
    stress_ratio = 0.30
    strains = compute_strain_history(time_s, avg_T, mix, sustained_stress_ratio=stress_ratio)
    strain_header = ["time_s", "eps_thermal", "eps_tts", "eps_mech", "eps_total"]
    strain_rows = []
    for i in range(len(time_s)):
        strain_rows.append([
            int(round(strains["time_s"][i])),
            f"{strains["eps_thermal"][i]:.6e}",
            f"{strains["eps_tts"][i]:.6e}",
            f"{strains["eps_mech"][i]:.6e}",
            f"{strains["eps_total"][i]:.6e}",
        ])
    strain_path = os.path.join(paths["validation"], f"strain_history_{mix.name}.csv")
    write_csv(strain_path, strain_header, strain_rows)

    # Spalling and failure
    surface_T_series = [iso834_gas_temperature_C(t) for t in time_s]
    spall = spalling_and_failure(time_s, surface_T_series, avg_T, mix, sustained_stress_ratio=stress_ratio,
                                 nominal_width_m=0.10, nominal_thickness_m=0.20)
    spall_path = os.path.join(paths["validation"], f"spalling_{mix.name}.json")
    write_json(spall_path, spall)

    return {
        "thermocouples_csv": tc_path,
        "strain_csv": strain_path,
        "spalling_json": spall_path,
    }


def write_metadata(paths: Dict[str, str], outputs_index: Dict[str, Dict[str, str]]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "dataset": "Phase 4: Numerical Modeling Dataset",
        "topic": "Thermo-Mechanical Model for Fire-Resistant Structural Elements with HPRC",
        "created_at_utc": now,
        "mixtures": {m.name: {
            "description": m.description,
            "rubber_volume_fraction": m.rubber_volume_fraction,
            "baseline_props": {
                "k_W_mK@20C": m.baseline_thermal_conductivity_W_mK,
                "cp_J_kgK@20C": m.baseline_specific_heat_J_kgK,
                "rho_kg_m3@20C": m.baseline_density_kg_m3,
                "fc_MPa@20C": m.baseline_compressive_strength_MPa,
                "ft_MPa@20C": m.baseline_tensile_strength_MPa,
                "E_GPa@20C": m.baseline_elastic_modulus_GPa,
                "nu@20C": m.poisson_ratio_20C,
                "porosity": m.nominal_porosity,
            },
        } for m in MIXTURES},
        "files": outputs_index,
        "notes": [
            "Synthesized properties guided by Eurocode-style residuals and typical HPRC trends.",
            "Thermal solver uses 1D explicit scheme with variable properties vs temperature.",
            "Validation includes thermocouples (noisy), strain history under 0.30 load ratio, and spalling/failure heuristics.",
            "Units: SI. Temperatures in Celsius, stresses in MPa, modulus in GPa.",
        ],
        "license": "CC-BY-4.0",
        "citation": "Synthetic dataset generated for benchmarking and model development purposes.",
        "generator_script": os.path.relpath(__file__, start=os.path.dirname(paths["base"]))
    }
    out_path = os.path.join(paths["meta"], "metadata.json")
    write_json(out_path, meta)
    return out_path


def main() -> None:
    paths = get_base_dirs()

    outputs_index: Dict[str, Dict[str, str]] = {}

    for mix in MIXTURES:
        props_paths = generate_material_properties(paths, mix)
        val_paths = generate_validation(paths, mix)
        outputs_index[mix.name] = {**props_paths, **val_paths}

    meta_path = write_metadata(paths, outputs_index)

    print("Generated dataset at:", paths["base"])  # for visibility
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
