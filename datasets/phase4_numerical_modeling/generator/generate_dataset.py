#!/usr/bin/env python3
"""
Phase 4: Numerical Modeling Dataset Generator
Topic: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant
       Structural Elements Utilizing High-Performance Rubberized Concrete (HPRC)

Outputs under ../properties and ../validation using seeded randomness and
physics-informed trends. Pure-Python (no external deps).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# ---------------------------- Configuration ---------------------------------
SEED = 42
random.seed(SEED)

# Dataset roots (relative to this file)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PROPS_DIR = os.path.join(ROOT_DIR, "properties")
VALID_DIR = os.path.join(ROOT_DIR, "validation")

os.makedirs(PROPS_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)

# Temperature grid for properties (deg C)
T_MIN = 20
T_MAX = 1000
T_STEP = 20
TEMP_GRID = list(range(T_MIN, T_MAX + 1, T_STEP))

# Validation simulation settings
TOTAL_TIME_S = 2 * 60 * 60  # 2 hours
DT_S = 5                     # 5-second step
N_STEPS = int(TOTAL_TIME_S // DT_S) + 1

# 1D slab for thermocouples (explicit scheme)
SLAB_THICKNESS_M = 0.20  # 200 mm
DX_M = 0.01              # 10 mm nodes
NX = int(SLAB_THICKNESS_M / DX_M) + 1
ALPHA_M2_S = 1.0e-6      # thermal diffusivity ~ concrete order of magnitude
H_W_M2K = 30.0           # convective coefficient
K_W_MK = 1.2             # effective thermal conductivity for BC

# Thermocouple positions from exposed face
TC_DEPTHS_M = [0.01, 0.05, 0.10]  # 10, 50, 100 mm

# Mechanical loading (for strain validation)
FC20_MPA = 70.0         # 20C compressive strength reference for HPRC
E20_GPA = 35.0          # 20C Elastic Modulus reference
APPLIED_STRESS_MPA = 0.30 * FC20_MPA  # 30% sustained service load

# Metadata for the mix
MIX_METADATA = {
    "material": "High-Performance Rubberized Concrete (HPRC)",
    "rubber_content_vol_frac": 0.10,  # 10% rubber particles by volume
    "fiber_type": "polypropylene",
    "fiber_vol_frac": 0.002,  # 0.2%
    "moisture_state": "sealed_cured_28d",
    "aggregate_type": "siliceous",
    "admixtures": ["silica_fume", "superplasticizer"],
    "notes": "Synthetic dataset; magnitudes and trends are literature-informed, not exact.",
}


# ------------------------- Utility / Helper Functions ------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return (1 - t) * y0 + t * y1


def piecewise_linear(points: List[Tuple[float, float]], x: float) -> float:
    """points as sorted (x_i, y_i)."""
    if x <= points[0][0]:
        return points[0][1]
    for i in range(1, len(points)):
        if x <= points[i][0]:
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            return lerp(x0, y0, x1, y1, x)
    return points[-1][1]


# ------------------- Temperature-Dependent Property Models -------------------
# Rubberized concrete tends to have reduced k, slightly lower density, and
# heightened permeability growth with temperature.

def thermal_conductivity_w_mk(T: float) -> float:
    # Base ~1.5 at 20C, drops with T; rubber lowers k.
    k0 = 1.5 - 0.3 * MIX_METADATA["rubber_content_vol_frac"]  # rubber lowers
    k = k0 * (1.0 - 0.35 * (T - 20.0) / 980.0)
    k = clamp(k, 0.45, 2.0)
    # slight random perturbation (~±2%)
    k *= (1.0 + random.uniform(-0.02, 0.02))
    return k


def specific_heat_j_kgk(T: float) -> float:
    # Base ~900 J/kgK at 20C; peak around 120C (moisture effects), rising with T.
    base = 900.0 + 0.20 * (T - 20.0)
    peak = 300.0 * math.exp(-((T - 120.0) ** 2) / (2 * 30.0 ** 2))
    cp = base + peak
    cp *= (1.0 + random.uniform(-0.02, 0.02))
    return clamp(cp, 650.0, 1700.0)


def density_kg_m3(T: float) -> float:
    # Reduced by rubber; small drop with T due to moisture depletion and microcracking.
    rho0 = 2350.0 * (1.0 - 0.08 * MIX_METADATA["rubber_content_vol_frac"])  # ~2160
    rho = rho0 - 0.09 * (T - 20.0)
    rho *= (1.0 + random.uniform(-0.005, 0.005))
    return clamp(rho, 1500.0, 2400.0)


# Mechanical
_MECH_POINTS = [
    (20.0, 1.00),
    (200.0, 0.90),
    (400.0, 0.70),
    (600.0, 0.50),
    (800.0, 0.30),
    (1000.0, 0.15),
]


def compressive_strength_mpa(T: float) -> float:
    ratio = piecewise_linear(_MECH_POINTS, T)
    fc = FC20_MPA * ratio
    fc *= (1.0 + random.uniform(-0.02, 0.02))
    return max(fc, 1.0)


def tensile_strength_mpa(T: float) -> float:
    # Roughly ~10% of fc at 20C and degrades faster
    fc = compressive_strength_mpa(T)
    # degrade factor beyond fc to reflect brittleness at high T
    extra = clamp(1.0 - 0.0009 * (T - 20.0), 0.4, 1.0)
    ft = 0.10 * fc * extra
    return max(ft, 0.1)


def elastic_modulus_gpa(T: float) -> float:
    # Degrades faster than strength
    points = [
        (20.0, 1.00), (200.0, 0.85), (400.0, 0.60), (600.0, 0.40), (800.0, 0.20), (1000.0, 0.10)
    ]
    ratio = piecewise_linear(points, T)
    E = E20_GPA * ratio
    E *= (1.0 + random.uniform(-0.02, 0.02))
    return max(E, 1.0)


def poisson_ratio(T: float) -> float:
    # Slight increase with temperature and damage
    nu = 0.20 + 0.08 * (T - 20.0) / 980.0
    return clamp(nu, 0.18, 0.30)


# Deformation

def cte_per_c(T: float) -> float:
    # microstrain per C baseline ~8e-6 rising to ~16e-6
    base = 8.0e-6
    peak = 16.0e-6
    s = (T - 20.0) / (1000.0 - 20.0)
    s = clamp(s, 0.0, 1.0)
    alpha = base + (peak - base) * (1.0 / (1.0 + math.exp(-6.0 * (s - 0.5))))
    alpha *= (1.0 + random.uniform(-0.05, 0.05))
    return clamp(alpha, 6.0e-6, 20.0e-6)


def transient_thermal_strain(T: float, load_ratio: float = 0.30) -> float:
    # Negative peak around 250C from moisture migration, scales with load ratio
    peak_mag = -400e-6 * (0.8 + 0.6 * load_ratio)  # -320 to -640 microstrain
    width = 100.0
    center = 250.0
    eps = peak_mag * math.exp(-((T - center) ** 2) / (2 * width ** 2))
    return eps


# Poro-mechanical

def porosity(T: float) -> float:
    # Base porosity increases with T due to microcracking and dehydration
    base = 0.12 + 0.08 * (T - 20.0) / 980.0
    base *= (1.0 + 0.10 * MIX_METADATA["rubber_content_vol_frac"])  # slightly higher with rubber
    return clamp(base, 0.10, 0.25)


def permeability_m2(T: float) -> float:
    # Grows rapidly with temperature and damage (related to E degradation)
    E_ratio = elastic_modulus_gpa(T) / E20_GPA
    damage = clamp(1.0 - E_ratio, 0.0, 1.0)
    k0 = 1e-18 * (1.0 + 2.5 * MIX_METADATA["rubber_content_vol_frac"])  # higher baseline with rubber
    kT = k0 * (1.0 + 2e3 * damage ** 2) * (1.0 + 1.5 * (T - 20.0) / 980.0)
    # Bound to plausible range for heated concrete
    return clamp(kT, 1e-19, 5e-14)


# ---------------------- Fire Curve and Heat Conduction -----------------------

def iso_834_gas_temperature_c(time_s: float) -> float:
    # ISO 834: T = 20 + 345 * log10(8t + 1), t in minutes
    t_min = time_s / 60.0
    return 20.0 + 345.0 * math.log10(8.0 * t_min + 1.0)


def simulate_slab_temperatures() -> Tuple[List[float], Dict[str, List[float]]]:
    """Explicit 1D conduction with convective boundary at x=0 and adiabatic at x=L.
    Returns times [s] and dict of TC temperatures in C.
    """
    Fo = ALPHA_M2_S * DT_S / (DX_M ** 2)
    Bi = H_W_M2K * DX_M / K_W_MK
    # Stability check (not enforced strictly, but we choose parameters safely)
    if Fo > 0.5:
        # reduce dt if needed (should not happen with given constants)
        raise RuntimeError(f"Unstable scheme: Fo={Fo:.3f} > 0.5")

    # Initialize
    T = [20.0 for _ in range(NX)]
    times: List[float] = []
    tc_records: Dict[str, List[float]] = {f"TC{i+1}_C": [] for i in range(len(TC_DEPTHS_M))}

    node_indices = [int(round(depth / DX_M)) for depth in TC_DEPTHS_M]
    node_indices = [clamp(idx, 0, NX - 1) for idx in node_indices]  # type: ignore

    t = 0.0
    for step in range(N_STEPS):
        times.append(t)
        # Record thermocouples
        for i, idx in enumerate(node_indices):
            tc_records[f"TC{i+1}_C"].append(T[int(idx)])

        # Advance one step
        T_new = T.copy()
        T_gas = iso_834_gas_temperature_c(t)

        # Interior nodes
        for i in range(1, NX - 1):
            T_new[i] = T[i] + Fo * (T[i + 1] - 2 * T[i] + T[i - 1])

        # Exposed face (x=0): convective boundary
        # T_new[0] = T[0] + 2*Fo*(T[1] - T[0] + Bi*(T_gas - T[0]))
        T_new[0] = T[0] + 2.0 * Fo * (T[1] - T[0] + Bi * (T_gas - T[0]))

        # Insulated back face (x=L): zero gradient
        # Mirror method: T[-1] close to interior
        T_new[-1] = T[-1] + 2.0 * Fo * (T[-2] - T[-1])

        T = T_new
        t += DT_S

    return times, tc_records


# --------------------------- Strain History Model ----------------------------

def compute_strain_history(times: List[float], tc_dict: Dict[str, List[float]]) -> Tuple[List[float], List[float], List[float]]:
    """Compute axial and transverse strain histories at mid-depth.
    Returns (axial_strain, transverse_strain, temperature_mid).
    """
    mid_idx = int(round((SLAB_THICKNESS_M / 2.0) / DX_M))

    # Build a temperature series at mid-depth using the simulated field.
    # Derive mid-depth from the field by reconstructing T field is heavy; we don't store it.
    # Approximate mid-depth temperature by averaging the deepest TC and a nearby virtual node.
    # Alternatively, we recompute TC-like approximation using semi-empirical lag.
    # Here, use TC3 (100 mm) as proxy and apply small lag correction.
    T_mid_series = []
    tc3 = tc_dict["TC3_C"]
    for k, T3 in enumerate(tc3):
        # Apply a small lag/offset to mimic deeper point
        correction = -5.0 * (1.0 - math.exp(-0.0005 * times[k]))
        T_mid_series.append(max(20.0, T3 + correction))

    axial = []
    transverse = []

    # Mechanical stress is constant (sustained load)
    sigma_mpa = APPLIED_STRESS_MPA

    # Integrate thermal strain incrementally and add mechanical and transient components
    eps_th = 0.0
    last_T = T_mid_series[0]
    for i, t in enumerate(times):
        Tm = T_mid_series[i]
        # Thermal strain increment using instantaneous CTE at current T
        alpha = cte_per_c(Tm)
        dT = Tm - last_T
        eps_th += alpha * dT
        last_T = Tm

        # Mechanical elastic strain under constant stress and degrading E(T)
        E_gpa = elastic_modulus_gpa(Tm)
        eps_mech = (sigma_mpa / (E_gpa * 1000.0))  # MPa/GPa => 1e-3

        # Transient thermal strain at this temperature
        eps_tts = transient_thermal_strain(Tm, load_ratio=APPLIED_STRESS_MPA / FC20_MPA)

        # Total axial strain: thermal + mechanical + transient
        eps_ax = eps_th + eps_mech + eps_tts

        # Transverse strain assuming Poisson effect (compression axial)
        nu = poisson_ratio(Tm)
        eps_tr = -nu * eps_mech + 0.2 * eps_th  # thermal transverse component scaled

        axial.append(eps_ax)
        transverse.append(eps_tr)

    return axial, transverse, T_mid_series


# --------------------------- Spalling Event Model ----------------------------

def fabricate_spalling_events(times: List[float], tc_dict: Dict[str, List[float]]) -> Tuple[List[Dict], float]:
    """Create plausible spalling events based on near-surface temperatures and timing.
    Returns (events, time_to_failure_s).
    """
    surface_temp = tc_dict["TC1_C"]

    def time_when_crosses(temp_c: float) -> float | None:
        for t, Ts in zip(times, surface_temp):
            if Ts >= temp_c:
                return t
        return None

    events: List[Dict] = []

    t1 = time_when_crosses(300.0)
    if t1 is not None:
        events.append({
            "event_id": 1,
            "time_s": t1 + random.uniform(-60, 60),
            "location_label": "surface_corner_band",
            "depth_mm": round(10 + random.uniform(-3, 5), 1),
            "area_cm2": round(80 + random.uniform(-20, 30), 1),
            "description": "Initial surface scaling and corner spalls due to pore-pressure."
        })

    t2 = time_when_crosses(550.0)
    if t2 is not None:
        events.append({
            "event_id": 2,
            "time_s": t2 + random.uniform(-90, 120),
            "location_label": "near_surface_face",
            "depth_mm": round(25 + random.uniform(-5, 8), 1),
            "area_cm2": round(200 + random.uniform(-50, 80), 1),
            "description": "Moderate spalling with shallow depth across exposed face."
        })

    t3 = time_when_crosses(750.0)
    if t3 is not None:
        events.append({
            "event_id": 3,
            "time_s": t3 + random.uniform(-120, 180),
            "location_label": "localized_patches",
            "depth_mm": round(40 + random.uniform(-10, 12), 1),
            "area_cm2": round(350 + random.uniform(-100, 120), 1),
            "description": "Localized deeper spalling patches, partial aggregate exposure."
        })

    # Define a notional failure time as when mid-depth E drops below 15% and
    # cumulative spall depth exceeds ~60 mm or surface reaches ~900C.
    # Approximate from times array
    time_to_failure_s = None  # type: ignore
    # Check surface temperature 900C time
    t_fail_temp = time_when_crosses(900.0)

    # Approximate mid-depth E criterion using TC3 temperature as proxy
    tc3 = tc_dict["TC3_C"]
    t_fail_E = None
    for t, T3 in zip(times, tc3):
        if elastic_modulus_gpa(T3) <= 0.15 * E20_GPA:
            t_fail_E = t
            break

    # Spall depth cumulative
    cum_depth = sum(e.get("depth_mm", 0.0) for e in events)
    if cum_depth >= 60.0 and events:
        time_to_failure_s = max(e["time_s"] for e in events) + 300.0

    if time_to_failure_s is None:
        candidates = [v for v in [t_fail_temp, t_fail_E] if v is not None]
        if candidates:
            time_to_failure_s = min(candidates) + 300.0
        else:
            time_to_failure_s = times[-1]

    return events, float(time_to_failure_s)


# ------------------------------- I/O Helpers --------------------------------

def write_csv(path: str, header: List[str], rows: List[List[float | int | str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------- Main Run ----------------------------------

def generate_properties() -> None:
    # Thermal properties
    rows = []
    for T in TEMP_GRID:
        rows.append([
            T,
            round(thermal_conductivity_w_mk(T), 4),
            round(specific_heat_j_kgk(T), 2),
            round(density_kg_m3(T), 1),
        ])
    write_csv(
        os.path.join(PROPS_DIR, "thermal_properties.csv"),
        ["temperature_C", "k_W_mK", "cp_J_kgK", "density_kg_m3"],
        rows,
    )

    # Mechanical properties
    rows = []
    for T in TEMP_GRID:
        rows.append([
            T,
            round(compressive_strength_mpa(T), 3),
            round(tensile_strength_mpa(T), 3),
            round(elastic_modulus_gpa(T), 3),
            round(poisson_ratio(T), 4),
        ])
    write_csv(
        os.path.join(PROPS_DIR, "mechanical_properties.csv"),
        ["temperature_C", "fc_MPa", "ft_MPa", "E_GPa", "nu"],
        rows,
    )

    # Deformation properties
    rows = []
    for T in TEMP_GRID:
        rows.append([
            T,
            round(cte_per_c(T) * 1e6, 3),  # microstrain per C
            round(transient_thermal_strain(T) * 1e6, 3),  # microstrain
        ])
    write_csv(
        os.path.join(PROPS_DIR, "deformation_properties.csv"),
        ["temperature_C", "cte_microstrain_per_C", "transient_thermal_strain_microstrain"],
        rows,
    )

    # Poro-mechanical properties
    rows = []
    for T in TEMP_GRID:
        rows.append([
            T,
            f"{permeability_m2(T):.3e}",
            round(porosity(T), 4),
        ])
    write_csv(
        os.path.join(PROPS_DIR, "poro_mechanical_properties.csv"),
        ["temperature_C", "permeability_m2", "porosity"],
        rows,
    )

    # Metadata JSON
    write_json(
        os.path.join(PROPS_DIR, "metadata.json"),
        {
            "mix": MIX_METADATA,
            "seed": SEED,
            "temperature_grid_C": [T for T in TEMP_GRID],
        },
    )


def generate_validation() -> None:
    # Fire curve
    times = [i * DT_S for i in range(N_STEPS)]
    gas_T = [round(iso_834_gas_temperature_c(t), 2) for t in times]
    write_csv(
        os.path.join(VALID_DIR, "iso_834_fire_curve.csv"),
        ["time_s", "gas_temperature_C"],
        [[t, Tg] for t, Tg in zip(times, gas_T)],
    )

    # Thermocouple temps from conduction model
    times_sim, tc_dict = simulate_slab_temperatures()
    header = ["time_s"] + list(tc_dict.keys())
    rows = []
    for i, t in enumerate(times_sim):
        rows.append([t] + [round(tc_dict[k][i], 2) for k in tc_dict.keys()])
    write_csv(
        os.path.join(VALID_DIR, "thermocouples.csv"),
        header,
        rows,
    )

    # Strain history
    axial, transverse, Tmid = compute_strain_history(times_sim, tc_dict)
    rows = []
    for i, t in enumerate(times_sim):
        rows.append([
            t,
            round(axial[i] * 1e6, 2),      # microstrain
            round(transverse[i] * 1e6, 2), # microstrain
            round(Tmid[i], 2),
        ])
    write_csv(
        os.path.join(VALID_DIR, "strain_history.csv"),
        ["time_s", "axial_strain_microstrain", "transverse_strain_microstrain", "mid_depth_temperature_C"],
        rows,
    )

    # Spalling events
    events, ttf = fabricate_spalling_events(times_sim, tc_dict)
    write_csv(
        os.path.join(VALID_DIR, "spalling_events.csv"),
        ["event_id", "time_s", "location_label", "depth_mm", "area_cm2", "description"],
        [[e.get("event_id"), round(e.get("time_s", 0.0), 1), e.get("location_label"), e.get("depth_mm"), e.get("area_cm2"), e.get("description")] for e in events],
    )

    write_json(
        os.path.join(VALID_DIR, "spalling_map.json"),
        {
            "events": events,
            "time_to_failure_s": ttf,
            "notes": "Schematic spalling description, not from image analysis."
        },
    )


def main() -> None:
    print("Generating properties...")
    generate_properties()
    print("Properties generated in:", PROPS_DIR)

    print("Generating validation data...")
    generate_validation()
    print("Validation generated in:", VALID_DIR)

    # Save a root-level manifest
    write_json(
        os.path.join(ROOT_DIR, "manifest.json"),
        {
            "topic": "Thermo-Mechanical Model for Fire-Resistant HPRC",
            "phase": 4,
            "seed": SEED,
            "outputs": {
                "properties": [
                    "thermal_properties.csv",
                    "mechanical_properties.csv",
                    "deformation_properties.csv",
                    "poro_mechanical_properties.csv",
                    "metadata.json"
                ],
                "validation": [
                    "iso_834_fire_curve.csv",
                    "thermocouples.csv",
                    "strain_history.csv",
                    "spalling_events.csv",
                    "spalling_map.json"
                ]
            }
        },
    )
    print("Manifest written.")


if __name__ == "__main__":
    main()
