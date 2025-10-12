#!/usr/bin/env python3
"""
Generate a synthetic, physics-informed dataset for acoustic attenuation in stratified two-phase flows.

This script fabricates plausible attenuation values by combining:
- Thermo-viscous boundary layer (Kirchhoff-style) losses
- Interface scattering due to impedance mismatch and roughness (wavy interfaces)
- Shear-induced mode conversion at the gas-liquid interface
- Radiation/leakage losses from small perforations (beyond single-phase leakage acoustics)

It also computes relevant dimensionless groups and per-mechanism contributions, and writes
both a CSV dataset and a JSON metadata file.

Notes:
- The model is simplified for dataset generation; parameters are tunable via constants below.
- Where external data are requested, the script will optionally attempt to discover and download
  related open datasets via public APIs (best-effort). If unavailable, synthetic data are still produced.

License for generated synthetic dataset: CC BY 4.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Constants and configuration
# ----------------------------

GENERATOR_VERSION = "1.0.0"
RNG = random.Random()

G_ACCEL = 9.80665  # m/s^2
PI = math.pi
DB_PER_NP = 8.685889638  # dB = Np * 20/log(10); per unit length use same factor

# Interface/leakage model constants (tunable to shape dataset)
C_INTERFACE = 0.50
C_SHEAR = 0.10
C_LEAK = 2.00
MIN_ALPHA_DB_PER_M = 1e-5  # small floor to avoid zeros

# Surface tension approximations (N/m)
SIGMA_WATER = 0.072
SIGMA_LIGHT_OIL = 0.030

# Gas constants for air
R_AIR = 287.058  # J/(kg*K)
GAMMA_AIR = 1.4
CP_AIR = 1006.0  # J/(kg*K)
K_AIR = 0.026  # W/(m*K) approximate at room temp

# Water properties (approx)
CP_WATER = 4182.0  # J/(kg*K)
K_WATER = 0.6     # W/(m*K)
C_WATER = 1482.0  # m/s speed of sound (approx)
RHO_WATER_293K = 998.2  # kg/m^3 at ~293K

# Light oil approximations (broad-brush for variety)
C_LIGHT_OIL = 1300.0
RHO_LIGHT_OIL = 800.0
CP_LIGHT_OIL = 2000.0
K_LIGHT_OIL = 0.15


@dataclass
class PhaseProperties:
    name: str
    density: float  # kg/m^3
    viscosity: float  # Pa*s
    speed_of_sound: float  # m/s
    prandtl: float
    gamma: Optional[float] = None  # heat capacity ratio (for gases)


def sutherland_dynamic_viscosity_air(temperature_k: float) -> float:
    """Dynamic viscosity of air via Sutherland's law (Pa*s)."""
    # Sutherland constants for air
    mu0 = 1.716e-5  # Pa*s
    T0 = 273.15     # K
    C = 110.4       # K
    mu = mu0 * ((temperature_k / T0) ** 1.5) * (T0 + C) / (temperature_k + C)
    return mu


def air_properties(temperature_k: float, pressure_pa: float) -> PhaseProperties:
    rho = pressure_pa / (R_AIR * temperature_k)
    mu = sutherland_dynamic_viscosity_air(temperature_k)
    pr = (mu * CP_AIR) / K_AIR
    # Speed of sound in ideal gas ~ sqrt(gamma * R * T)
    c = math.sqrt(GAMMA_AIR * R_AIR * temperature_k)
    return PhaseProperties(
        name="air",
        density=rho,
        viscosity=mu,
        speed_of_sound=c,
        prandtl=pr,
        gamma=GAMMA_AIR,
    )


def water_dynamic_viscosity(temperature_k: float) -> float:
    """Approximate dynamic viscosity for water (Pa*s) using empirical correlation.
    mu = 2.414e-5 * 10^(247.8 / (T - 140)), T in K
    """
    t = max(temperature_k, 274.0)
    mu = 2.414e-5 * 10.0 ** (247.8 / (t - 140.0))
    return mu


def water_density(temperature_k: float) -> float:
    """Rough approximation of water density vs temperature (kg/m^3)."""
    # Simple linearized approximation around room temp; bounded
    delta = temperature_k - 293.15
    rho = RHO_WATER_293K - 0.3 * delta
    return max(950.0, min(1000.0, rho))


def water_properties(temperature_k: float, pressure_pa: float) -> PhaseProperties:
    rho = water_density(temperature_k)
    mu = water_dynamic_viscosity(temperature_k)
    pr = (mu * CP_WATER) / K_WATER
    return PhaseProperties(
        name="water",
        density=rho,
        viscosity=mu,
        speed_of_sound=C_WATER,
        prandtl=pr,
        gamma=None,
    )


def light_oil_properties(temperature_k: float, pressure_pa: float) -> PhaseProperties:
    # Light oil: add temperature dependence to viscosity loosely
    base_mu = 0.02  # Pa*s baseline
    mu = max(0.006, min(0.12, base_mu * (293.15 / max(temperature_k, 250.0))))
    pr = (mu * CP_LIGHT_OIL) / K_LIGHT_OIL
    return PhaseProperties(
        name="light_oil",
        density=RHO_LIGHT_OIL,
        viscosity=mu,
        speed_of_sound=C_LIGHT_OIL,
        prandtl=pr,
        gamma=None,
    )


@dataclass
class Sample:
    # Primary inputs
    frequency_hz: float
    pipe_diameter_m: float
    temperature_k: float
    pressure_pa: float
    gas_volume_fraction: float  # alpha_g
    superficial_velocity_gas: float  # Usg (m/s)
    superficial_velocity_liquid: float  # Usl (m/s)
    regime: str  # 'stratified_smooth' | 'stratified_wavy' | 'intermittent'
    interface_rms_m: float
    leak_diameter_m: float
    leaks_count: int
    path_length_m: float
    liquid_name: str

    # Computed properties and outputs will be filled post-init


# ----------------------------
# Physics helper functions
# ----------------------------


def wood_mixture_speed_of_sound(alpha_g: float, rho_g: float, c_g: float, rho_l: float, c_l: float) -> float:
    """Wood's equation for two-phase mixture speed of sound."""
    alpha_l = 1.0 - alpha_g
    denom = (alpha_g / (rho_g * c_g * c_g)) + (alpha_l / (rho_l * c_l * c_l))
    rho_m = alpha_g * rho_g + alpha_l * rho_l
    if denom <= 0.0 or rho_m <= 0.0:
        return min(c_g, c_l)
    c_mix = math.sqrt(1.0 / (rho_m * denom))
    return max(50.0, min(2000.0, c_mix))


def wavenumber(frequency_hz: float, c_m: float) -> float:
    return (2.0 * PI * frequency_hz) / max(c_m, 1e-6)


def interface_reflection_coeff(rho_g: float, c_g: float, rho_l: float, c_l: float) -> float:
    zg = rho_g * c_g
    zl = rho_l * c_l
    if (zg + zl) == 0:
        return 0.0
    R = abs(zl - zg) / (zl + zg)
    return max(0.0, min(1.0, R))


def kirchhoff_thermoviscous_alpha_np_per_m(
    frequency_hz: float,
    radius_m: float,
    c_mix: float,
    alpha_g: float,
    gas: PhaseProperties,
    liquid: PhaseProperties,
) -> float:
    """Approximate Kirchhoff-type thermo-viscous attenuation (Np/m) for a two-phase mixture.

    We blend gas and liquid kinematic viscosities weighted by volume fraction. The
    prefactor reflects gas thermal losses with a (1 + (gamma-1)/sqrt(Pr)) term; liquids
    are approximated with a factor of 1.
    """
    omega = 2.0 * PI * frequency_hz
    alpha_l = 1.0 - alpha_g
    nu_g = gas.viscosity / max(gas.density, 1e-9)
    nu_l = liquid.viscosity / max(liquid.density, 1e-9)
    nu_eff = alpha_g * nu_g + alpha_l * nu_l

    gas_factor = 1.0 + ((gas.gamma - 1.0) / math.sqrt(max(gas.prandtl, 1e-6))) if gas.gamma else 1.0
    liquid_factor = 1.0  # thermal effects small compared to gas in this simplification
    factor = alpha_g * gas_factor + alpha_l * liquid_factor

    alpha_np = factor * math.sqrt(max(nu_eff, 0.0) * max(omega, 0.0)) / (max(radius_m, 1e-9) * max(c_mix * c_mix, 1e-9))
    return max(0.0, alpha_np)


def interface_scattering_alpha_np_per_m(
    k_1pm: float,
    radius_m: float,
    R_int: float,
    alpha_g: float,
    interface_rms_m: float,
) -> float:
    """Interface scattering loss (Np/m) scaling with roughness amplitude and impedance mismatch."""
    alpha_l = 1.0 - alpha_g
    roughness_ratio = min(1.0, max(0.0, interface_rms_m / max(radius_m, 1e-9)))
    alpha_np = C_INTERFACE * (R_int ** 2) * (roughness_ratio ** 2) * k_1pm * alpha_g * alpha_l
    return max(0.0, alpha_np)


def shear_mode_conversion_alpha_np_per_m(
    k_1pm: float,
    delta_u: float,
    c_mix: float,
    alpha_g: float,
) -> float:
    """Shear-induced mode conversion (Np/m) scaling with interfacial velocity jump."""
    alpha_l = 1.0 - alpha_g
    rel = delta_u / max(c_mix, 1e-9)
    alpha_np = C_SHEAR * (rel ** 2) * k_1pm * alpha_g * alpha_l
    return max(0.0, alpha_np)


def leak_radiation_alpha_np_per_m(
    k_1pm: float,
    leak_radius_m: float,
    leaks_count: int,
    radius_m: float,
) -> float:
    """Small-hole radiation/leakage loss (Np/m) with subwavelength scaling (k a)^4.
    Treated as distributed loss per unit length normalized by pipe radius.
    """
    if leaks_count <= 0 or leak_radius_m <= 0.0:
        return 0.0
    ka = k_1pm * leak_radius_m
    alpha_np = C_LEAK * leaks_count * (ka ** 4) / max(radius_m, 1e-9)
    return max(0.0, alpha_np)


# ----------------------------
# Sampling utilities
# ----------------------------


def choose_liquid_name(rng: random.Random) -> str:
    return rng.choices(["water", "light_oil"], weights=[0.75, 0.25])[0]


def sample_regime(rng: random.Random, u_sg: float, u_sl: float) -> str:
    # Coarse selection based on superficial velocities
    if u_sl < 0.2 and u_sg < 5.0:
        return "stratified_smooth"
    if u_sg >= 5.0 and u_sl < 1.0:
        return "stratified_wavy"
    return "intermittent"


def sample_interface_rms(rng: random.Random, regime: str, diameter_m: float, u_sg: float, u_sl: float) -> float:
    if regime == "stratified_smooth":
        return rng.uniform(0.001, 0.005) * diameter_m
    if regime == "stratified_wavy":
        base = 0.02 + 0.01 * math.tanh(0.15 * u_sg + 0.5 * u_sl)
        return min(0.20 * diameter_m, base * diameter_m)
    # intermittent
    base = 0.01 + 0.008 * math.tanh(0.1 * u_sg + 0.3 * u_sl)
    return min(0.12 * diameter_m, base * diameter_m)


def compute_dimensionless(
    rho_g: float,
    mu_g: float,
    rho_l: float,
    mu_l: float,
    u_g: float,
    u_l: float,
    u_mix: float,
    diameter_m: float,
    sigma: float,
    c_mix: float,
) -> Dict[str, float]:
    re_g = rho_g * u_g * diameter_m / max(mu_g, 1e-12)
    re_l = rho_l * u_l * diameter_m / max(mu_l, 1e-12)
    fr = u_mix / math.sqrt(max(G_ACCEL * diameter_m, 1e-12))
    we_l = rho_l * (u_l ** 2) * diameter_m / max(sigma, 1e-12)
    bo = (rho_l - rho_g) * G_ACCEL * (diameter_m ** 2) / max(sigma, 1e-12)
    ma = u_mix / max(c_mix, 1e-12)
    return {
        "reynolds_gas": re_g,
        "reynolds_liquid": re_l,
        "froude_number": fr,
        "weber_liquid": we_l,
        "bond_number": bo,
        "mach_mixture": ma,
    }


# ----------------------------
# External data attempt (best-effort)
# ----------------------------


def attempt_fetch_external_data(output_dir: str, rng: random.Random) -> List[Dict[str, str]]:
    """Best-effort attempt to discover and download related open datasets.
    Returns a list of fetched file records. Non-fatal on failure or no network.
    """
    try:
        import requests  # type: ignore
    except Exception:
        return []

    base = "https://zenodo.org/api/records"
    queries = [
        "stratified flow acoustics",
        "two-phase flow acoustics",
        "acoustic attenuation pipe",
    ]

    os.makedirs(os.path.join(output_dir, "external"), exist_ok=True)
    fetched: List[Dict[str, str]] = []

    for q in queries:
        try:
            resp = requests.get(base, params={"q": q, "size": 5, "type": "dataset"}, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits:
                files = hit.get("files", []) or []
                for f in files:
                    fname = f.get("key")
                    link = f.get("links", {}).get("self")
                    size = f.get("size", 0)
                    if not fname or not link:
                        continue
                    # Only download small CSV/TSV files
                    if not (fname.lower().endswith(".csv") or fname.lower().endswith(".tsv")):
                        continue
                    if size and size > 50_000_000:
                        continue
                    out_path = os.path.join(output_dir, "external", fname)
                    try:
                        r2 = requests.get(link, timeout=30)
                        if r2.status_code == 200 and r2.content:
                            with open(out_path, "wb") as fh:
                                fh.write(r2.content)
                            fetched.append({"source": "zenodo", "query": q, "file": out_path})
                    except Exception:
                        continue
        except Exception:
            continue

    return fetched


# ----------------------------
# Dataset generation
# ----------------------------


def generate_sample(rng: random.Random) -> Tuple[Sample, Dict[str, float]]:
    # Primary sampling ranges
    frequency_hz = 10 ** rng.uniform(1.7, 4.3)  # ~50 Hz to 20 kHz
    pipe_diameter_m = 10 ** rng.uniform(-2.0, -0.3)  # 0.01 m to ~0.5 m
    temperature_k = rng.uniform(280.0, 360.0)
    pressure_pa = 10 ** rng.uniform(5.0, 6.5)  # 1e5 to ~3e6 Pa
    alpha_g = rng.uniform(0.03, 0.97)  # gas volume fraction

    # Superficial velocities (m/s)
    u_sg = 10 ** rng.uniform(-2.0, 1.2)  # 0.01 to ~16 m/s
    u_sl = 10 ** rng.uniform(-3.0, 0.7)  # 0.001 to ~5 m/s

    regime = sample_regime(rng, u_sg, u_sl)
    interface_rms_m = sample_interface_rms(rng, regime, pipe_diameter_m, u_sg, u_sl)

    # Leak configuration
    leak_diameter_m = 10 ** rng.uniform(-5.0, -2.0)  # 10 microns to 1 mm
    leaks_count = rng.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]

    path_length_m = 10 ** rng.uniform(-0.5, 2.0)  # ~0.3 m to 100 m

    liquid_name = choose_liquid_name(rng)

    sample = Sample(
        frequency_hz=frequency_hz,
        pipe_diameter_m=pipe_diameter_m,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        gas_volume_fraction=alpha_g,
        superficial_velocity_gas=u_sg,
        superficial_velocity_liquid=u_sl,
        regime=regime,
        interface_rms_m=interface_rms_m,
        leak_diameter_m=leak_diameter_m,
        leaks_count=leaks_count,
        path_length_m=path_length_m,
        liquid_name=liquid_name,
    )

    # Compute phase properties
    gas = air_properties(temperature_k, pressure_pa)
    if liquid_name == "water":
        liquid = water_properties(temperature_k, pressure_pa)
        sigma = SIGMA_WATER
    else:
        liquid = light_oil_properties(temperature_k, pressure_pa)
        sigma = SIGMA_LIGHT_OIL

    # Mixture acoustics
    c_mix = wood_mixture_speed_of_sound(alpha_g, gas.density, gas.speed_of_sound, liquid.density, liquid.speed_of_sound)
    rho_mix = alpha_g * gas.density + (1.0 - alpha_g) * liquid.density
    k_1pm = wavenumber(frequency_hz, c_mix)
    radius_m = 0.5 * pipe_diameter_m

    # Actual phase velocities (approximate):
    u_g = u_sg / max(alpha_g, 1e-6)
    u_l = u_sl / max(1.0 - alpha_g, 1e-6)
    u_mix = u_sg + u_sl
    delta_u = abs(u_g - u_l)

    # Dimensionless groups
    dims = compute_dimensionless(
        rho_g=gas.density,
        mu_g=gas.viscosity,
        rho_l=liquid.density,
        mu_l=liquid.viscosity,
        u_g=u_g,
        u_l=u_l,
        u_mix=u_mix,
        diameter_m=pipe_diameter_m,
        sigma=sigma,
        c_mix=c_mix,
    )

    # Mechanism contributions (Np/m)
    r_int = interface_reflection_coeff(gas.density, gas.speed_of_sound, liquid.density, liquid.speed_of_sound)

    alpha_vt_np = kirchhoff_thermoviscous_alpha_np_per_m(
        frequency_hz=frequency_hz,
        radius_m=radius_m,
        c_mix=c_mix,
        alpha_g=alpha_g,
        gas=gas,
        liquid=liquid,
    )

    alpha_int_np = interface_scattering_alpha_np_per_m(
        k_1pm=k_1pm,
        radius_m=radius_m,
        R_int=r_int,
        alpha_g=alpha_g,
        interface_rms_m=interface_rms_m,
    )

    alpha_shear_np = shear_mode_conversion_alpha_np_per_m(
        k_1pm=k_1pm,
        delta_u=delta_u,
        c_mix=c_mix,
        alpha_g=alpha_g,
    )

    alpha_leak_np = leak_radiation_alpha_np_per_m(
        k_1pm=k_1pm,
        leak_radius_m=0.5 * leak_diameter_m,
        leaks_count=leaks_count,
        radius_m=radius_m,
    )

    # Convert to dB/m
    alpha_vt_dbpm = DB_PER_NP * alpha_vt_np
    alpha_int_dbpm = DB_PER_NP * alpha_int_np
    alpha_shear_dbpm = DB_PER_NP * alpha_shear_np
    alpha_leak_dbpm = DB_PER_NP * alpha_leak_np

    alpha_total_dbpm = max(
        MIN_ALPHA_DB_PER_M,
        alpha_vt_dbpm + alpha_int_dbpm + alpha_shear_dbpm + alpha_leak_dbpm,
    )

    # Measurement noise and uncertainty estimate
    noise = RNG.gauss(0.0, 0.15)
    alpha_measured_dbpm = max(MIN_ALPHA_DB_PER_M, alpha_total_dbpm + noise)
    uncertainty_dbpm = 3.0 * 0.15

    dominant = "vt"
    contributions = {
        "vt": alpha_vt_dbpm,
        "interface": alpha_int_dbpm,
        "shear": alpha_shear_dbpm,
        "leak": alpha_leak_dbpm,
    }
    dominant = max(contributions.items(), key=lambda kv: kv[1])[0]

    outputs = {
        # Phase props
        "density_gas_kgm3": gas.density,
        "viscosity_gas_PaS": gas.viscosity,
        "speed_sound_gas_mps": gas.speed_of_sound,
        "prandtl_gas": gas.prandtl,
        "density_liquid_kgm3": liquid.density,
        "viscosity_liquid_PaS": liquid.viscosity,
        "speed_sound_liquid_mps": liquid.speed_of_sound,
        "prandtl_liquid": liquid.prandtl,
        # Mixture and wave
        "density_mixture_kgm3": rho_mix,
        "speed_sound_mixture_mps": c_mix,
        "wavenumber_1pm": k_1pm,
        "reflection_coeff_interface": r_int,
        # Dimensionless
        **dims,
        # Contributions (dB/m)
        "alpha_vt_dbpm": alpha_vt_dbpm,
        "alpha_interface_dbpm": alpha_int_dbpm,
        "alpha_shear_dbpm": alpha_shear_dbpm,
        "alpha_leak_dbpm": alpha_leak_dbpm,
        # Totals
        "attenuation_total_dbpm": alpha_total_dbpm,
        "attenuation_measured_dbpm": alpha_measured_dbpm,
        "attenuation_over_path_db": alpha_measured_dbpm * sample.path_length_m,
        "uncertainty_dbpm": uncertainty_dbpm,
        "dominant_mechanism": dominant,
        # Kinematics
        "actual_velocity_gas_ms": u_g,
        "actual_velocity_liquid_ms": u_l,
        "delta_u_ms": delta_u,
    }

    return sample, outputs


def write_dataset(
    rows: List[Tuple[Sample, Dict[str, float]]],
    output_csv_path: str,
) -> None:
    # Assemble header
    base_fields = [
        "sample_id",
        "frequency_hz",
        "pipe_diameter_m",
        "temperature_k",
        "pressure_pa",
        "gas_volume_fraction",
        "superficial_velocity_gas_ms",
        "superficial_velocity_liquid_ms",
        "regime",
        "interface_rms_m",
        "leak_diameter_m",
        "leaks_count",
        "path_length_m",
        "liquid_name",
    ]

    # Gather dynamic keys from outputs
    output_keys = set()
    for _, out in rows:
        output_keys.update(out.keys())
    output_fields = sorted(output_keys)

    fieldnames = base_fields + output_fields

    with open(output_csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (s, out) in enumerate(rows):
            row = {
                "sample_id": idx,
                "frequency_hz": s.frequency_hz,
                "pipe_diameter_m": s.pipe_diameter_m,
                "temperature_k": s.temperature_k,
                "pressure_pa": s.pressure_pa,
                "gas_volume_fraction": s.gas_volume_fraction,
                "superficial_velocity_gas_ms": s.superficial_velocity_gas,
                "superficial_velocity_liquid_ms": s.superficial_velocity_liquid,
                "regime": s.regime,
                "interface_rms_m": s.interface_rms_m,
                "leak_diameter_m": s.leak_diameter_m,
                "leaks_count": s.leaks_count,
                "path_length_m": s.path_length_m,
                "liquid_name": s.liquid_name,
            }
            row.update(out)
            writer.writerow(row)


def write_metadata(
    output_json_path: str,
    n_rows: int,
    seed: int,
    fetched_external: List[Dict[str, str]],
) -> None:
    record = {
        "title": "Synthetic dataset: Attenuation mechanisms in stratified flows (beyond single-phase leakage acoustics)",
        "description": (
            "Physics-informed synthetic dataset combining thermo-viscous, interface scattering, "
            "shear mode-conversion, and leak radiation losses for stratified two-phase flows in pipes."
        ),
        "generator_version": GENERATOR_VERSION,
        "created_unix": int(time.time()),
        "license": "CC BY 4.0",
        "provenance": {
            "external_fetch_attempted": bool(fetched_external),
            "external_files": fetched_external,
        },
        "schema": {
            "primary_keys": ["sample_id"],
            "units": {
                "frequency_hz": "Hz",
                "pipe_diameter_m": "m",
                "temperature_k": "K",
                "pressure_pa": "Pa",
                "gas_volume_fraction": "-",
                "superficial_velocity_gas_ms": "m/s",
                "superficial_velocity_liquid_ms": "m/s",
                "interface_rms_m": "m",
                "leak_diameter_m": "m",
                "path_length_m": "m",
                "attenuation_total_dbpm": "dB/m",
                "attenuation_measured_dbpm": "dB/m",
                "attenuation_over_path_db": "dB",
            },
        },
        "notes": [
            "Thermo-viscous losses approximate Kirchhoff-type dependence sqrt(nu*omega)/(a*c^2).",
            "Interface scattering scales with impedance mismatch R^2, roughness^2, and wavenumber.",
            "Shear conversion scales with (DeltaU/c)^2 and wavenumber.",
            "Leak radiation approximated by (k*r_leak)^4 scaling for subwavelength perforations.",
            "Mixture speed of sound via Wood's equation.",
        ],
        "random_seed": seed,
        "rows": n_rows,
        "references": [
            "Kirchhoff, G. (1868). On the influence of heat conduction in a gas on sound propagation.",
            "Sutherland, W. (1893). The viscosity of gases and molecular force.",
            "Wood, A. B. (1930). A Textbook of Sound (mixture compressibility).",
        ],
    }
    with open(output_json_path, "w") as fh:
        json.dump(record, fh, indent=2)


# ----------------------------
# Main
# ----------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stratified flow attenuation dataset (synthetic)")
    p.add_argument("--n", type=int, default=10000, help="Number of samples to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out_dir", type=str, default="/workspace/data/stratified_flow_attenuation", help="Output directory")
    p.add_argument("--try_download", action="store_true", help="Attempt to fetch related external datasets (best-effort)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    RNG.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    fetched_external: List[Dict[str, str]] = []
    if args.try_download:
        fetched_external = attempt_fetch_external_data(args.out_dir, RNG)

    rows: List[Tuple[Sample, Dict[str, float]]] = []
    for _ in range(args.n):
        s, out = generate_sample(RNG)
        rows.append((s, out))

    csv_path = os.path.join(args.out_dir, "stratified_flow_attenuation_dataset.csv")
    json_path = os.path.join(args.out_dir, "metadata.json")

    write_dataset(rows, csv_path)
    write_metadata(json_path, n_rows=len(rows), seed=args.seed, fetched_external=fetched_external)

    print(f"Wrote dataset: {csv_path}")
    print(f"Wrote metadata: {json_path}")
    if fetched_external:
        print(f"Fetched {len(fetched_external)} external file(s) to {os.path.join(args.out_dir, 'external')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
