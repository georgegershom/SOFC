#!/usr/bin/env python3

"""
Synthetic dataset generator: Stratified flow acoustic attenuation

This script fabricates a physically plausible dataset related to attenuation
mechanisms in stratified gas–liquid flows, with terms that go beyond
single-phase leakage acoustics by including interfacial scattering and
thermoviscous effects for two-phase mixtures.

Outputs a CSV with per-case operating conditions and computed metrics.

No third-party dependencies required.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, Tuple

# Physical constants (approximate)
G_ACCEL = 9.81  # m/s^2
R_AIR = 287.05  # J/(kg·K)
GAMMA_AIR = 1.4


def sutherland_mu_air(T_K: float) -> float:
    """Return dynamic viscosity of air (Pa·s) using Sutherland's law.

    mu(T) = mu0 * (T/T0)^(3/2) * (T0 + S) / (T + S)
    Reference values (approx): mu0=1.716e-5 Pa·s at T0=273.15 K, S=111 K.
    """
    T0 = 273.15
    mu0 = 1.716e-5
    S = 111.0
    return mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)


def andrade_mu_water(T_K: float) -> float:
    """Return dynamic viscosity of liquid water (Pa·s) via Andrade eqn (approx).

    mu = A * 10^(B / (T_C + C)) where T_C is Celsius.
    Common parameters: A=2.414e-5 Pa·s, B=247.8 K, C=133.15 K.
    Valid roughly for 0–100°C.
    """
    T_C = T_K - 273.15
    A = 2.414e-5
    B = 247.8
    C = 133.15
    return A * (10.0 ** (B / (T_C + C)))


def speed_of_sound_air(T_K: float) -> float:
    return math.sqrt(GAMMA_AIR * R_AIR * T_K)


def speed_of_sound_water(T_K: float) -> float:
    """Approximate speed of sound in water (m/s) as weak function of T.
    Base ~1480 m/s near 20°C. We'll vary ~1400–1550 m/s.
    """
    T_C = T_K - 273.15
    c = 1480.0 + 3.5 * (T_C - 20.0)
    return max(1400.0, min(1550.0, c))


def water_density(T_K: float) -> float:
    """Approximate water density (kg/m^3) vs temperature.
    Simple linearized model near ambient conditions.
    """
    T_C = T_K - 273.15
    rho = 1000.0 - 0.3 * (T_C - 20.0)
    return max(950.0, min(1000.0, rho))


def air_density(P_Pa: float, T_K: float) -> float:
    return P_Pa / (R_AIR * T_K)


def water_surface_tension(T_K: float) -> float:
    """Approximate water surface tension (N/m) near ambient.
    ~0.072 N/m at 20°C, decreasing with temperature.
    """
    T_C = T_K - 273.15
    sigma = 0.072 - 0.00015 * (T_C - 20.0)
    return max(0.050, min(0.075, sigma))


def compute_gas_properties(P_Pa: float, T_K: float) -> Tuple[float, float, float]:
    rho_g = air_density(P_Pa, T_K)
    mu_g = sutherland_mu_air(T_K)
    c_g = speed_of_sound_air(T_K)
    return rho_g, mu_g, c_g


def compute_liquid_properties(T_K: float) -> Tuple[float, float, float, float]:
    rho_l = water_density(T_K)
    mu_l = andrade_mu_water(T_K)
    c_l = speed_of_sound_water(T_K)
    sigma = water_surface_tension(T_K)
    return rho_l, mu_l, c_l, sigma


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_void_fraction(
    U_g: float,
    U_l: float,
    rho_g: float,
    rho_l: float,
    regime: str,
    inclination_deg: float,
) -> float:
    """Estimate gas volume fraction alpha for stratified flow.

    Uses a simple slip-ratio style estimate tuned for stratified conditions,
    and adjusts slightly with inclination (upflow retains more gas).
    """
    # Base slip ratio; keep within plausible range for stratified
    slip = (rho_l / max(1e-9, rho_g)) ** 0.25 * (1.0 + 0.5 * (U_g / (U_l + 1e-6)))
    slip = clamp(slip, 1.0, 10.0)

    # Inclination effect (upward increases alpha slightly; downward decreases)
    inc_factor = 1.0 + 0.002 * inclination_deg

    # Regime influence: wavy flows exhibit more holdup
    regime_factor = 1.1 if regime == "stratified_wavy" else 1.0

    effective_slip = slip / (inc_factor * regime_factor)
    alpha = U_g / (U_g + effective_slip * U_l + 1e-9)
    return clamp(alpha, 0.001, 0.60)


def wood_mixture_speed_of_sound(
    alpha: float, rho_g: float, rho_l: float, c_g: float, c_l: float
) -> float:
    rho_mix = alpha * rho_g + (1.0 - alpha) * rho_l
    denom = alpha / (rho_g * c_g * c_g) + (1.0 - alpha) / (rho_l * c_l * c_l)
    if denom <= 0.0:
        return c_g
    c_mix_sq = 1.0 / (rho_mix * denom)
    if c_mix_sq <= 0.0:
        return c_g
    return math.sqrt(c_mix_sq)


def effective_mixture_viscosity(mu_g: float, mu_l: float, alpha: float) -> float:
    # Log mixing to keep between phases and ensure monotonicity
    return math.exp(alpha * math.log(mu_g) + (1.0 - alpha) * math.log(mu_l))


def compute_attenuation(
    frequency_hz: float,
    regime: str,
    alpha: float,
    rho_mix: float,
    c_mix: float,
    mu_mix: float,
    interfacial_roughness_m: float,
    corr_length_m: float,
    leak_present: bool,
    leak_diameter_m: float,
    leak_density_per_m: float,
) -> Dict[str, float]:
    """Compute linear and dB/m attenuation terms and total.

    Simple phenomenology:
    - Thermoviscous: proportional to nu * w^2 / c^3
    - Interfacial scattering: ~ a * k^2 * eta^2 / L_c
    - Leakage radiation (beyond single-phase): ~ b * (k a)^2 * n_leaks
    """
    omega = 2.0 * math.pi * frequency_hz
    k = omega / max(1e-9, c_mix)

    nu_mix = mu_mix / max(1e-12, rho_mix)

    # Regime-dependent multiplier for enhanced dissipation in wavy flows
    regime_mult = 1.7 if regime == "stratified_wavy" else 1.0

    # Thermoviscous (Np/m)
    a_tv = 0.03 * regime_mult
    alpha_tv = a_tv * nu_mix * (omega * omega) / max(1e-9, c_mix ** 3)

    # Interfacial scattering (Np/m)
    a_sc = 0.20 * regime_mult
    alpha_sc = a_sc * (k * k) * (interfacial_roughness_m ** 2) / max(1e-6, corr_length_m)

    # Leakage radiation term (Np/m)
    alpha_leak = 0.0
    if leak_present and leak_diameter_m > 0.0 and leak_density_per_m > 0.0:
        a = 0.5 * leak_diameter_m
        ka = k * a
        b_leak = 0.08  # calibrated to keep within plausible ranges
        alpha_leak = b_leak * (ka * ka) * leak_density_per_m

    alpha_total_np_per_m = max(1e-8, alpha_tv + alpha_sc + alpha_leak)
    attenuation_db_per_m = 8.686 * alpha_total_np_per_m

    return {
        "attenuation_np_per_m": alpha_total_np_per_m,
        "attenuation_db_per_m": attenuation_db_per_m,
        "alpha_tv_np_per_m": alpha_tv,
        "alpha_sc_np_per_m": alpha_sc,
        "alpha_leak_np_per_m": alpha_leak,
    }


def sample_case(rng: random.Random, case_id: int) -> Dict[str, float | int | str | bool]:
    # Operating conditions
    pressure_Pa = rng.uniform(1.0e5, 1.0e6)  # 1–10 bar
    temperature_K = rng.uniform(283.0, 353.0)  # 10–80 C

    # Geometry and roughness
    pipe_diameter_m = rng.uniform(0.02, 0.20)
    pipe_roughness_m = rng.uniform(1e-6, 2e-4)
    inclination_deg = rng.uniform(-5.0, 5.0)  # mild up/down

    # Superficial velocities
    U_g = rng.uniform(0.1, 15.0)  # m/s
    U_l = rng.uniform(0.05, 2.0)  # m/s

    # Regime selection heuristic (wavy more likely with larger We, Re)
    rho_l, mu_l, c_l, sigma = compute_liquid_properties(temperature_K)
    rho_g, mu_g, c_g = compute_gas_properties(pressure_Pa, temperature_K)

    Re_g = rho_g * U_g * pipe_diameter_m / max(1e-12, mu_g)
    Re_l = rho_l * U_l * pipe_diameter_m / max(1e-12, mu_l)

    We_l = rho_l * (U_l ** 2) * pipe_diameter_m / max(1e-12, sigma)

    wavy_score = 0.0
    wavy_score += clamp((Re_g - 2e4) / 3e4, 0.0, 1.0)
    wavy_score += clamp((Re_l - 2e4) / 2e4, 0.0, 1.0)
    wavy_score += clamp((We_l - 2.0) / 10.0, 0.0, 1.0)

    regime = "stratified_wavy" if wavy_score > 0.8 else "stratified_smooth"

    alpha = compute_void_fraction(U_g, U_l, rho_g, rho_l, regime, inclination_deg)

    rho_mix = alpha * rho_g + (1.0 - alpha) * rho_l
    mu_mix = effective_mixture_viscosity(mu_g, mu_l, alpha)
    c_mix = wood_mixture_speed_of_sound(alpha, rho_g, rho_l, c_g, c_l)

    # Acoustic settings
    frequency_hz = rng.uniform(50.0, 5000.0)
    propagation_length_m = rng.uniform(1.0, 100.0)
    source_level_dB = rng.uniform(85.0, 150.0)

    # Interfacial roughness and correlation length
    if regime == "stratified_wavy":
        interfacial_roughness_m = rng.uniform(0.0005, 0.005)  # 0.5–5 mm
        corr_length_m = rng.uniform(0.03, 0.20)  # 3–20 cm
    else:
        interfacial_roughness_m = rng.uniform(0.00005, 0.0005)  # 0.05–0.5 mm
        corr_length_m = rng.uniform(0.05, 0.30)  # 5–30 cm

    # Leakage features (beyond single-phase): random sparse occurrence
    leak_present = rng.random() < 0.25
    leak_density_per_m = rng.uniform(0.0, 2.0) if leak_present else 0.0
    leak_diameter_m = rng.uniform(0.001, 0.01) if leak_present else 0.0

    att = compute_attenuation(
        frequency_hz=frequency_hz,
        regime=regime,
        alpha=alpha,
        rho_mix=rho_mix,
        c_mix=c_mix,
        mu_mix=mu_mix,
        interfacial_roughness_m=interfacial_roughness_m,
        corr_length_m=corr_length_m,
        leak_present=leak_present,
        leak_diameter_m=leak_diameter_m,
        leak_density_per_m=leak_density_per_m,
    )

    total_att_db_per_m = att["attenuation_db_per_m"]
    transmitted_level_dB = max(0.0, source_level_dB - total_att_db_per_m * propagation_length_m)

    # Mixture kinematics
    U_mix = alpha * U_g + (1.0 - alpha) * U_l
    Ma_mix = U_mix / max(1e-9, c_mix)

    Fr_l = U_l / max(1e-9, math.sqrt(G_ACCEL * pipe_diameter_m))
    Bo = (rho_l - rho_g) * G_ACCEL * (pipe_diameter_m ** 2) / max(1e-12, sigma)

    # Output row
    return {
        "case_id": case_id,
        "regime": regime,
        "pipe_diameter_m": pipe_diameter_m,
        "pipe_roughness_m": pipe_roughness_m,
        "inclination_deg": inclination_deg,
        "pressure_Pa": pressure_Pa,
        "temperature_K": temperature_K,
        "rho_g_kg_m3": rho_g,
        "mu_g_Pa_s": mu_g,
        "c_g_m_s": c_g,
        "rho_l_kg_m3": rho_l,
        "mu_l_Pa_s": mu_l,
        "c_l_m_s": c_l,
        "sigma_N_m": sigma,
        "U_g_m_s": U_g,
        "U_l_m_s": U_l,
        "alpha_gas": alpha,
        "rho_mix_kg_m3": rho_mix,
        "mu_mix_Pa_s": mu_mix,
        "c_mix_m_s": c_mix,
        "frequency_Hz": frequency_hz,
        "propagation_length_m": propagation_length_m,
        "source_level_dB": source_level_dB,
        "transmitted_level_dB": transmitted_level_dB,
        "attenuation_db_per_m": total_att_db_per_m,
        "attenuation_np_per_m": att["attenuation_np_per_m"],
        "alpha_tv_np_per_m": att["alpha_tv_np_per_m"],
        "alpha_sc_np_per_m": att["alpha_sc_np_per_m"],
        "alpha_leak_np_per_m": att["alpha_leak_np_per_m"],
        "Re_g": Re_g,
        "Re_l": Re_l,
        "We_l": We_l,
        "Fr_l": Fr_l,
        "Bo": Bo,
        "leak_present": leak_present,
        "leak_density_per_m": leak_density_per_m,
        "leak_diameter_m": leak_diameter_m,
        "Mach_mix": Ma_mix,
    }


def generate_rows(n_rows: int, seed: int) -> Dict[str, Dict[str, float | int | str | bool]]:
    rng = random.Random(seed)
    for i in range(n_rows):
        yield sample_case(rng, i + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stratified flow attenuation dataset (synthetic)")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--out",
        type=str,
        default="/workspace/data/stratified_flow_attenuation/stratified_flow_attenuation.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Prepare writer
    first_row = next(generate_rows(1, args.seed))
    fieldnames = list(first_row.keys())

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first_row)
        for row in generate_rows(args.rows - 1, args.seed + 1):
            writer.writerow(row)

    print(f"Wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
