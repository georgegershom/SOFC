#!/usr/bin/env python3
"""
Welding inverse-design dataset generator.

Generates physics-informed synthetic datasets for laser welding:
- Experimental-like dataset with uncertainties
- Simulation-like dataset calibrated to experimental distribution
- Extreme-temperature performance for a subset of experimental samples
- Master dataset combining both, with a JSON schema
- Zipped archive for download

Usage:
  python scripts/generate_welding_inverse_design_dataset.py \
    --exp-n 300 --sim-n 5000 --out-dir data --seed 42 --zip true
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------- Constants and helpers ----------------------
MATERIAL_PROPS = {
    # Properties are dimensionless factors for simplified models
    # absorption_factor: higher -> more energy absorbed
    # conductivity_factor: higher -> more heat conducted away (reduces melt size)
    # cte_mismatch_factor: higher -> more thermal cycling damage
    # strength_factor: relative base strength scaling
    "Cu-Al": {
        "absorption_factor": 0.75,
        "conductivity_factor": 1.15,
        "cte_mismatch_factor": 1.20,
        "strength_factor": 1.00,
        "imc_enabled": True,
    },
    "Al-Al": {
        "absorption_factor": 0.70,
        "conductivity_factor": 1.00,
        "cte_mismatch_factor": 0.85,
        "strength_factor": 0.95,
        "imc_enabled": False,
    },
    "Cu-Cu": {
        "absorption_factor": 0.65,
        "conductivity_factor": 1.30,
        "cte_mismatch_factor": 0.90,
        "strength_factor": 1.05,
        "imc_enabled": False,
    },
    "Al-Steel": {
        "absorption_factor": 0.80,
        "conductivity_factor": 0.95,
        "cte_mismatch_factor": 1.10,
        "strength_factor": 0.90,
        "imc_enabled": True,
    },
}

SHIELD_GAS_ADJUSTMENTS = {
    "Argon": {"absorption_multiplier": 1.00, "porosity_reduction": 0.70},
    "Nitrogen": {"absorption_multiplier": 0.98, "porosity_reduction": 0.80},
    "Helium": {"absorption_multiplier": 1.03, "porosity_reduction": 0.65},
    "None": {"absorption_multiplier": 0.90, "porosity_reduction": 1.00},
}

JOINT_TYPE_ADJUSTMENTS = {
    "Lap": {"penetration_requirement": 0.50},  # fraction of total thickness
    "Butt": {"penetration_requirement": 1.00},
}

# Temperature constants for simple Arrhenius-like growth
R_GAS_CONSTANT = 8.314  # J/(mol*K)
# Synthetic activation energy (not physical), in J/mol
IMC_ACTIVATION_ENERGY = 9.5e4


@dataclass
class ParameterRanges:
    power_w: Tuple[float, float] = (500.0, 3000.0)
    speed_mm_s: Tuple[float, float] = (10.0, 200.0)
    pulse_frequency_hz: Tuple[float, float] = (0.0, 500.0)
    pulse_duration_ms: Tuple[float, float] = (0.1, 10.0)
    focus_position_mm: Tuple[float, float] = (-2.0, 2.0)
    spot_size_um: Tuple[float, float] = (50.0, 400.0)
    clamping_pressure_mpa: Tuple[float, float] = (0.5, 10.0)
    gas_flow_l_min: Tuple[float, float] = (0.0, 20.0)
    thickness_mm: Tuple[float, float] = (0.1, 2.0)
    overlap_distance_mm: Tuple[float, float] = (0.0, 5.0)


RANGES = ParameterRanges()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sample_choice(choices: List[str], size: int) -> List[str]:
    return random.choices(choices, k=size)


def sample_uniform(low: float, high: float, size: int) -> np.ndarray:
    return np.random.uniform(low, high, size)


def sample_positive_normal(mean: float, cv: float, size: int) -> np.ndarray:
    sigma = abs(mean) * cv
    values = np.random.normal(mean, sigma, size)
    return np.clip(values, a_min=0.0, a_max=None)


# ---------------------- Physics-inspired models ----------------------

def compute_duty_cycle(pulse_frequency_hz: np.ndarray, pulse_duration_ms: np.ndarray) -> np.ndarray:
    raw = pulse_frequency_hz * (pulse_duration_ms * 1e-3)
    # When frequency is ~0 (CW), interpret as duty cycle 1.0
    duty = np.where(pulse_frequency_hz < 1e-6, 1.0, np.clip(raw, 0.01, 1.0))
    return duty


def compute_effective_absorption(material: str, shield_gas: str, focus_position_mm: np.ndarray) -> np.ndarray:
    base = MATERIAL_PROPS[material]["absorption_factor"]
    gas_mult = SHIELD_GAS_ADJUSTMENTS[shield_gas]["absorption_multiplier"]
    # Slight reduction when defocused; focus_position near 0 is best
    focus_penalty = 1.0 - 0.05 * np.clip(np.abs(focus_position_mm), 0.0, 2.0)
    return base * gas_mult * focus_penalty


def compute_heat_input_and_density(
    power_w: np.ndarray,
    speed_mm_s: np.ndarray,
    duty_cycle: np.ndarray,
    spot_size_um: np.ndarray,
    eff_absorption: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Heat input per unit length (J/mm): (W * duty) / (mm/s)
    heat_input_j_per_mm = np.clip(power_w * duty_cycle / np.clip(speed_mm_s, 1e-6, None), 1e-6, None)
    heat_input_j_per_mm *= eff_absorption
    # Energy density: distribute heat over spot width (approx)
    spot_diameter_mm = np.clip(spot_size_um, 1e-3, None) / 1000.0
    energy_density_j_per_mm2 = heat_input_j_per_mm / np.clip(spot_diameter_mm, 1e-6, None)
    return heat_input_j_per_mm, energy_density_j_per_mm2


def compute_morphology(
    material: str,
    joint_type: str,
    thickness_mm: np.ndarray,
    heat_input_j_per_mm: np.ndarray,
    energy_density_j_per_mm2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    props = MATERIAL_PROPS[material]
    conductivity_factor = props["conductivity_factor"]

    # Nugget width grows sublinearly with energy density, penalized by conduction and thickness
    nugget_width_mm = 0.15 * np.power(energy_density_j_per_mm2, 0.45)
    nugget_width_mm *= 1.0 / np.power(conductivity_factor, 0.35)
    nugget_width_mm *= 1.0 / np.power(np.clip(thickness_mm, 0.05, None), 0.10)
    nugget_width_mm = np.clip(nugget_width_mm, 0.1, 5.0)

    # Penetration depth scales with heat input and joint requirement
    base_penetration_mm = 0.12 * np.power(heat_input_j_per_mm, 0.55)
    base_penetration_mm *= 1.0 / np.power(conductivity_factor, 0.45)
    required_fraction = JOINT_TYPE_ADJUSTMENTS[joint_type]["penetration_requirement"]
    penetration_depth_mm = base_penetration_mm
    # For butt joints, deeper penetration is required; lap joints allow shallower penetration
    penetration_depth_mm *= 0.8 + 0.4 * required_fraction
    penetration_depth_mm = np.clip(penetration_depth_mm, 0.02, 3.0)

    # HAZ width increases with heat input and decreases with conductivity and thickness
    haz_width_mm = 0.20 * np.power(heat_input_j_per_mm, 0.40)
    haz_width_mm *= 1.0 / np.power(conductivity_factor, 0.60)
    haz_width_mm *= 1.0 / np.power(np.clip(thickness_mm, 0.05, None), 0.15)
    haz_width_mm = np.clip(haz_width_mm, 0.05, 6.0)

    return nugget_width_mm, penetration_depth_mm, haz_width_mm


def compute_defect_probabilities(
    material: str,
    shield_gas: str,
    gas_flow_l_min: np.ndarray,
    speed_mm_s: np.ndarray,
    heat_input_j_per_mm: np.ndarray,
    pulse_frequency_hz: np.ndarray,
    clamping_pressure_mpa: np.ndarray,
) -> Dict[str, np.ndarray]:
    porosity_base = 0.25
    porosity_gas_factor = SHIELD_GAS_ADJUSTMENTS[shield_gas]["porosity_reduction"]
    porosity_flow_effect = np.exp(-np.clip(gas_flow_l_min, 0.0, 30.0) / 20.0)  # more flow -> less porosity
    porosity_speed_effect = np.clip(speed_mm_s / 150.0, 0.0, 1.5)  # higher speed -> more porosity
    porosity = porosity_base * porosity_gas_factor * porosity_flow_effect * (0.7 + 0.6 * porosity_speed_effect)

    expulsion = np.clip((heat_input_j_per_mm / 30.0) ** 2, 0.0, 1.0)
    expulsion *= np.clip(pulse_frequency_hz / 400.0, 0.6, 1.6)  # pulsed instability

    undercut = np.clip((speed_mm_s / 180.0) * (15.0 / np.clip(heat_input_j_per_mm, 1.0, None)), 0.0, 1.0)

    cte = MATERIAL_PROPS[material]["cte_mismatch_factor"]
    cracks = np.clip(0.10 * cte * (0.5 + 0.5 * expulsion) * (0.8 + 0.4 * porosity), 0.0, 1.0)
    cracks *= 1.0 / (0.8 + 0.2 * np.clip(clamping_pressure_mpa / 6.0, 0.6, 1.4))

    return {
        "porosity_probability": np.clip(porosity, 0.0, 1.0),
        "expulsion_probability": np.clip(expulsion, 0.0, 1.0),
        "undercut_probability": np.clip(undercut, 0.0, 1.0),
        "crack_probability": np.clip(cracks, 0.0, 1.0),
    }


def sample_defects(probabilities: np.ndarray) -> np.ndarray:
    return (np.random.uniform(0.0, 1.0, size=probabilities.shape[0]) < probabilities).astype(int)


def compute_mechanical_electrical(
    material: str,
    nugget_width_mm: np.ndarray,
    penetration_depth_mm: np.ndarray,
    haz_width_mm: np.ndarray,
    defects: Dict[str, np.ndarray],
    clamping_pressure_mpa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    props = MATERIAL_PROPS[material]
    fused_area_mm2 = np.pi * np.power(np.clip(nugget_width_mm, 0.05, None) / 2.0, 2)
    effective_fused_volume_mm3 = fused_area_mm2 * np.clip(penetration_depth_mm, 0.02, None)

    defect_penalty = (
        0.30 * defects["porosity_probability"]
        + 0.25 * defects["expulsion_probability"]
        + 0.20 * defects["undercut_probability"]
        + 0.25 * defects["crack_probability"]
    )

    tensile_shear_strength_n = 900.0 * np.power(effective_fused_volume_mm3, 0.35)
    tensile_shear_strength_n *= props["strength_factor"]
    tensile_shear_strength_n *= (1.0 - 0.60 * defect_penalty)
    tensile_shear_strength_n = np.clip(tensile_shear_strength_n, 200.0, 7000.0)

    peel_strength_n = 0.55 * tensile_shear_strength_n

    # Contact resistance decreases with area and clamping pressure; increases with defects
    base_resistance_uohm = 130.0 / np.power(np.clip(fused_area_mm2, 0.05, None), 0.65)
    base_resistance_uohm *= 1.0 / np.power(0.8 + 0.2 * np.clip(clamping_pressure_mpa, 0.5, 12.0), 0.4)
    base_resistance_uohm *= (1.0 + 1.1 * defect_penalty)
    contact_resistance_uohm = np.clip(base_resistance_uohm, 5.0, 200.0)

    return tensile_shear_strength_n, peel_strength_n, contact_resistance_uohm


def compute_peak_temperature_c(energy_density_j_per_mm2: np.ndarray, material: str) -> np.ndarray:
    base = 350.0 * np.power(energy_density_j_per_mm2, 0.25)
    base /= MATERIAL_PROPS[material]["conductivity_factor"]
    return np.clip(base, 100.0, 1600.0)


def compute_imc_thickness_um(
    material: str,
    peak_temperature_c: np.ndarray,
    aging_hours_at_c: float,
    start_thickness_um: np.ndarray,
) -> np.ndarray:
    if not MATERIAL_PROPS[material]["imc_enabled"]:
        return np.full_like(peak_temperature_c, fill_value=np.nan, dtype=float)

    # Parabolic growth: x^2 = x0^2 + k * t * exp(-Q/(R*T))
    t_seconds = aging_hours_at_c * 3600.0
    temperature_k = np.clip(peak_temperature_c + 273.15, 250.0, 2000.0)
    # Synthetic kinetics; k chosen to produce microns over hundreds of hours
    k_growth = 2.0e-12  # (um^2)/s at reference
    arrhenius = np.exp(-IMC_ACTIVATION_ENERGY / (R_GAS_CONSTANT * temperature_k))
    growth_um2 = k_growth * t_seconds / np.maximum(arrhenius, 1e-20)
    result = np.sqrt(np.power(start_thickness_um, 2.0) + growth_um2)
    return np.clip(result, 0.0, 50.0)


def compute_thermal_cycling_effects(
    material: str,
    cycles: np.ndarray,
    imc_thickness_um: np.ndarray,
    baseline_strength_n: np.ndarray,
    baseline_resistance_uohm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cte = MATERIAL_PROPS[material]["cte_mismatch_factor"]
    # Strength degradation increases with cycles and IMC thickness
    strength_degradation_frac = 0.03 * np.power(cycles / 500.0, 0.7) * (0.7 + 0.6 * cte)
    strength_degradation_frac *= (1.0 + 0.015 * np.nan_to_num(imc_thickness_um))
    strength_degradation_frac = np.clip(strength_degradation_frac, 0.0, 0.80)

    # Resistance increase scales similarly but weaker dependence on IMC
    resistance_increase_frac = 0.04 * np.power(cycles / 500.0, 0.6) * (0.8 + 0.4 * cte)
    resistance_increase_frac *= (1.0 + 0.008 * np.nan_to_num(imc_thickness_um))
    resistance_increase_frac = np.clip(resistance_increase_frac, 0.0, 1.50)

    post_strength = baseline_strength_n * (1.0 - strength_degradation_frac)
    post_resistance = baseline_resistance_uohm * (1.0 + resistance_increase_frac)

    # Define a loose failure criterion: 40% strength loss OR 120% resistance increase
    cycles_to_failure = 200.0 * (1.0 + 1.5 * (1.2 - cte))
    cycles_to_failure *= (1.0 + 0.05 * np.nan_to_num(imc_thickness_um))

    return (
        100.0 * strength_degradation_frac,
        100.0 * resistance_increase_frac,
        np.clip(cycles_to_failure, 100.0, 5000.0),
    )


def compute_creep_time_to_failure_hours(
    material: str,
    baseline_strength_n: np.ndarray,
    test_temperature_c: float = 100.0,
    load_fraction_of_uts: float = 0.50,
) -> np.ndarray:
    props = MATERIAL_PROPS[material]
    # Norton-like relation: time ~ A * (strength/load)^m * exp(Q/RT)
    temperature_k = test_temperature_c + 273.15
    m = 2.8
    Q = 6.5e4  # synthetic activation energy J/mol
    A = 1.0e-6  # scale to get hours
    load_n = np.clip(load_fraction_of_uts * baseline_strength_n, 1.0, None)
    ratio = np.clip(baseline_strength_n / load_n, 1.0, 20.0)
    base_time_h = A * np.power(ratio, m) * np.exp(Q / (R_GAS_CONSTANT * temperature_k))
    base_time_h *= 1.0 / props["cte_mismatch_factor"]
    return np.clip(base_time_h, 1.0, 1.0e6)


# ---------------------- Data generation ----------------------

def generate_inputs(n: int, rng: ParameterRanges, is_experimental: bool) -> Dict[str, np.ndarray | List[str]]:
    material = sample_choice(list(MATERIAL_PROPS.keys()), n)
    joint_type = sample_choice(list(JOINT_TYPE_ADJUSTMENTS.keys()), n)
    shield_gas = sample_choice(list(SHIELD_GAS_ADJUSTMENTS.keys()), n)

    power = sample_uniform(rng.power_w[0], rng.power_w[1], n)
    speed = sample_uniform(rng.speed_mm_s[0], rng.speed_mm_s[1], n)
    pulse_freq = sample_uniform(rng.pulse_frequency_hz[0], rng.pulse_frequency_hz[1], n)
    pulse_dur = sample_uniform(rng.pulse_duration_ms[0], rng.pulse_duration_ms[1], n)
    focus_pos = sample_uniform(rng.focus_position_mm[0], rng.focus_position_mm[1], n)
    spot_um = sample_uniform(rng.spot_size_um[0], rng.spot_size_um[1], n)
    clamp_mpa = sample_uniform(rng.clamping_pressure_mpa[0], rng.clamping_pressure_mpa[1], n)
    gas_flow = sample_uniform(rng.gas_flow_l_min[0], rng.gas_flow_l_min[1], n)
    thickness = sample_uniform(rng.thickness_mm[0], rng.thickness_mm[1], n)

    overlap = sample_uniform(rng.overlap_distance_mm[0], rng.overlap_distance_mm[1], n)
    # Enforce geometry constraints for butt joints
    for i in range(n):
        if joint_type[i] == "Butt":
            overlap[i] = 0.0
    # Enforce gas flow 0 when no gas
    for i in range(n):
        if shield_gas[i] == "None":
            gas_flow[i] = 0.0

    return {
        "Material_Combination": material,
        "Joint_Type": joint_type,
        "Shield_Gas": shield_gas,
        "Laser_Power_W": power,
        "Welding_Speed_mm_s": speed,
        "Pulse_Frequency_Hz": pulse_freq,
        "Pulse_Duration_ms": pulse_dur,
        "Beam_Focus_Position_mm": focus_pos,
        "Beam_Spot_Size_um": spot_um,
        "Clamping_Pressure_MPa": clamp_mpa,
        "Shield_Gas_Flow_L_min": gas_flow,
        "Sheet_Thickness_mm": thickness,
        "Overlap_Distance_mm": overlap,
    }


def generate_outputs(inputs: Dict[str, np.ndarray | List[str]], source: str) -> Dict[str, np.ndarray | List[int] | List[float]]:
    n = len(inputs["Laser_Power_W"])  # type: ignore[index]

    material = inputs["Material_Combination"]  # type: ignore[assignment]
    joint_type = inputs["Joint_Type"]  # type: ignore[assignment]
    shield_gas = inputs["Shield_Gas"]  # type: ignore[assignment]

    power = inputs["Laser_Power_W"]  # type: ignore[assignment]
    speed = inputs["Welding_Speed_mm_s"]  # type: ignore[assignment]
    pulse_freq = inputs["Pulse_Frequency_Hz"]  # type: ignore[assignment]
    pulse_dur = inputs["Pulse_Duration_ms"]  # type: ignore[assignment]
    focus_pos = inputs["Beam_Focus_Position_mm"]  # type: ignore[assignment]
    spot_um = inputs["Beam_Spot_Size_um"]  # type: ignore[assignment]
    clamp_mpa = inputs["Clamping_Pressure_MPa"]  # type: ignore[assignment]
    gas_flow = inputs["Shield_Gas_Flow_L_min"]  # type: ignore[assignment]
    thickness = inputs["Sheet_Thickness_mm"]  # type: ignore[assignment]

    duty = compute_duty_cycle(pulse_freq, pulse_dur)

    eff_absorption = np.empty(n)
    for i in range(n):
        eff_absorption[i] = compute_effective_absorption(material[i], shield_gas[i], np.array([focus_pos[i]]))[0]

    heat_input_j_per_mm, energy_density_j_per_mm2 = compute_heat_input_and_density(
        power, speed, duty, spot_um, eff_absorption
    )

    nugget_w = np.empty(n)
    penetration = np.empty(n)
    haz_w = np.empty(n)
    for i in range(n):
        w, p, h = compute_morphology(material[i], joint_type[i], np.array([thickness[i]]),
                                     np.array([heat_input_j_per_mm[i]]), np.array([energy_density_j_per_mm2[i]]))
        nugget_w[i], penetration[i], haz_w[i] = w[0], p[0], h[0]

    defect_probs = {k: np.empty(n) for k in ["porosity_probability", "expulsion_probability", "undercut_probability", "crack_probability"]}
    for i in range(n):
        probs = compute_defect_probabilities(
            material[i], shield_gas[i], np.array([gas_flow[i]]), np.array([speed[i]]),
            np.array([heat_input_j_per_mm[i]]), np.array([pulse_freq[i]]), np.array([clamp_mpa[i]])
        )
        for k in defect_probs:
            defect_probs[k][i] = probs[k][0]

    has_porosity = sample_defects(defect_probs["porosity_probability"])  # 0/1
    has_expulsion = sample_defects(defect_probs["expulsion_probability"])  # 0/1
    has_undercut = sample_defects(defect_probs["undercut_probability"])  # 0/1
    has_cracks = sample_defects(defect_probs["crack_probability"])  # 0/1

    tensile_strength = np.empty(n)
    peel_strength = np.empty(n)
    contact_res = np.empty(n)
    for i in range(n):
        ts, ps, cr = compute_mechanical_electrical(
            material[i], np.array([nugget_w[i]]), np.array([penetration[i]]), np.array([haz_w[i]]),
            {k: np.array([defect_probs[k][i]]) for k in defect_probs}, np.array([clamp_mpa[i]])
        )
        tensile_strength[i], peel_strength[i], contact_res[i] = ts[0], ps[0], cr[0]

    peak_temp_c = np.empty(n)
    for i in range(n):
        peak_temp_c[i] = compute_peak_temperature_c(np.array([energy_density_j_per_mm2[i]]), material[i])[0]

    # IMC thickness at baseline (post-weld, pre-aging)
    imc_initial = np.full(n, np.nan)
    for i in range(n):
        if MATERIAL_PROPS[material[i]]["imc_enabled"]:
            imc_initial[i] = np.clip(np.random.lognormal(mean=0.0, sigma=0.4), 0.05, 6.0)

    # Surface condition levels (0-3)
    spatter_level = np.clip((has_expulsion * 2 + (defect_probs["expulsion_probability"] > 0.6)), 0, 3)
    discoloration_level = np.clip((peak_temp_c > 600.0).astype(int) + (peak_temp_c > 900.0).astype(int), 0, 3)

    outputs: Dict[str, np.ndarray | List[int] | List[float]] = {
        "Nugget_Width_mm": nugget_w,
        "Penetration_Depth_mm": penetration,
        "HAZ_Width_mm": haz_w,
        "Has_Porosity": has_porosity,
        "Has_Expulsion": has_expulsion,
        "Has_Undercut": has_undercut,
        "Has_Cracks": has_cracks,
        "Tensile_Shear_Strength_N": tensile_strength,
        "Peel_Strength_N": peel_strength,
        "Contact_Resistance_uOhm": contact_res,
        "Peak_Temperature_C": peak_temp_c,
        "IMC_Thickness_um": imc_initial,
        "Spatter_Level_0_3": spatter_level,
        "Discoloration_Level_0_3": discoloration_level,
    }

    # Add uncertainties for experimental-like data
    if source == "Experimental":
        # Replicates for mechanical and electrical
        n_rep = np.random.randint(3, 6, size=n)
        ts_std = np.clip(0.05 * tensile_strength * np.random.uniform(0.6, 1.2, size=n), 5.0, None)
        cr_std = np.clip(0.06 * contact_res * np.random.uniform(0.6, 1.4, size=n), 0.5, None)
        outputs.update({
            "Replicates": n_rep,
            "Tensile_Shear_Strength_Std_N": ts_std,
            "Contact_Resistance_Std_uOhm": cr_std,
        })
    else:
        outputs.update({
            "Replicates": np.zeros(n, dtype=int),
            "Tensile_Shear_Strength_Std_N": np.full(n, np.nan),
            "Contact_Resistance_Std_uOhm": np.full(n, np.nan),
        })

    return outputs


def calibrate_simulation_to_experimental(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> pd.DataFrame:
    calibrated = sim_df.copy()
    # Align distributions for key outputs using mean-std scaling
    for col in [
        "Nugget_Width_mm",
        "Penetration_Depth_mm",
        "HAZ_Width_mm",
        "Tensile_Shear_Strength_N",
        "Peel_Strength_N",
        "Contact_Resistance_uOhm",
        "Peak_Temperature_C",
    ]:
        if col not in sim_df or col not in exp_df:
            continue
        sim_mean, sim_std = sim_df[col].mean(), sim_df[col].std(ddof=0)
        exp_mean, exp_std = exp_df[col].mean(), exp_df[col].std(ddof=0)
        if sim_std <= 1e-9:
            continue
        z = (calibrated[col] - sim_mean) / sim_std
        calibrated[col] = exp_mean + z * (exp_std if exp_std > 1e-9 else sim_std)
        # Ensure positivity where relevant
        if "Resistance" in col or "Strength" in col or "Width" in col or "Depth" in col or "Temperature" in col:
            calibrated[col] = np.clip(calibrated[col], 1e-6, None)
    return calibrated


def extreme_temperature_for_subset(exp_df: pd.DataFrame, subset_fraction: float = 0.4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = exp_df.copy()
    n = len(df)
    k = max(1, int(subset_fraction * n))
    idx = rng.choice(df.index.values, size=k, replace=False)

    # Choose cycles per sample
    cycles = rng.integers(200, 1201, size=k)

    imc_thickness_um = df.loc[idx, "IMC_Thickness_um"].to_numpy()
    imc_thickness_um = np.nan_to_num(imc_thickness_um, nan=0.2)

    # Compute effects per-sample using row-wise material
    strength_deg_pct = np.zeros(k)
    resistance_inc_pct = np.zeros(k)
    cycles_to_failure = np.zeros(k)
    post_strength = np.zeros(k)
    post_resistance = np.zeros(k)
    creep_time_h = np.zeros(k)
    imc_post_aging_um = np.zeros(k)

    for j, row_idx in enumerate(idx):
        material: str = df.at[row_idx, "Material_Combination"]
        baseline_strength = df.at[row_idx, "Tensile_Shear_Strength_N"]
        baseline_res = df.at[row_idx, "Contact_Resistance_uOhm"]
        imc0 = imc_thickness_um[j]
        c = cycles[j]

        sd, ri, ctf = compute_thermal_cycling_effects(material, np.array([c]), np.array([imc0]),
                                                      np.array([baseline_strength]), np.array([baseline_res]))
        strength_deg_pct[j] = sd[0]
        resistance_inc_pct[j] = ri[0]
        cycles_to_failure[j] = ctf[0]
        post_strength[j] = baseline_strength * (1.0 - 0.01 * strength_deg_pct[j])
        post_resistance[j] = baseline_res * (1.0 + 0.01 * resistance_inc_pct[j])

        creep_time_h[j] = compute_creep_time_to_failure_hours(material, np.array([baseline_strength]))[0]

        peak_t = df.at[row_idx, "Peak_Temperature_C"]
        imc_post_aging_um[j] = compute_imc_thickness_um(material, np.array([peak_t]), 500.0, np.array([imc0]))[0]

    df.loc[idx, "Thermal_Cycles"] = cycles
    df.loc[idx, "Strength_Degradation_Percent"] = strength_deg_pct
    df.loc[idx, "Resistance_Increase_Percent"] = resistance_inc_pct
    df.loc[idx, "Cycles_to_Failure_Estimate"] = cycles_to_failure
    df.loc[idx, "Post_ThermalCycle_Tensile_Shear_Strength_N"] = post_strength
    df.loc[idx, "Post_ThermalCycle_Contact_Resistance_uOhm"] = post_resistance
    df.loc[idx, "Creep_Time_to_Failure_h_100C_50pctUTS"] = creep_time_h
    df.loc[idx, "IMC_Thickness_PostAging_500h_120C_um"] = imc_post_aging_um

    # Fill NaNs for non-tested samples
    cols = [
        "Thermal_Cycles",
        "Strength_Degradation_Percent",
        "Resistance_Increase_Percent",
        "Cycles_to_Failure_Estimate",
        "Post_ThermalCycle_Tensile_Shear_Strength_N",
        "Post_ThermalCycle_Contact_Resistance_uOhm",
        "Creep_Time_to_Failure_h_100C_50pctUTS",
        "IMC_Thickness_PostAging_500h_120C_um",
    ]
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df


def build_schema() -> Dict[str, Dict[str, str]]:
    # Basic JSON schema-like mapping for columns
    return {
        # Inputs
        "Weld_ID": {"type": "string", "description": "Unique ID (W- for exp, S- for sim)"},
        "Data_Source": {"type": "string", "enum": ["Experimental", "Simulation"]},
        "Material_Combination": {"type": "string", "enum": list(MATERIAL_PROPS.keys())},
        "Joint_Type": {"type": "string", "enum": list(JOINT_TYPE_ADJUSTMENTS.keys())},
        "Shield_Gas": {"type": "string", "enum": list(SHIELD_GAS_ADJUSTMENTS.keys())},
        "Laser_Power_W": {"type": "number", "unit": "W"},
        "Welding_Speed_mm_s": {"type": "number", "unit": "mm/s"},
        "Pulse_Frequency_Hz": {"type": "number", "unit": "Hz"},
        "Pulse_Duration_ms": {"type": "number", "unit": "ms"},
        "Beam_Focus_Position_mm": {"type": "number", "unit": "mm"},
        "Beam_Spot_Size_um": {"type": "number", "unit": "um"},
        "Clamping_Pressure_MPa": {"type": "number", "unit": "MPa"},
        "Shield_Gas_Flow_L_min": {"type": "number", "unit": "L/min"},
        "Sheet_Thickness_mm": {"type": "number", "unit": "mm"},
        "Overlap_Distance_mm": {"type": "number", "unit": "mm"},
        # Outputs
        "Nugget_Width_mm": {"type": "number", "unit": "mm"},
        "Penetration_Depth_mm": {"type": "number", "unit": "mm"},
        "HAZ_Width_mm": {"type": "number", "unit": "mm"},
        "Has_Porosity": {"type": "integer", "enum": [0, 1]},
        "Has_Expulsion": {"type": "integer", "enum": [0, 1]},
        "Has_Undercut": {"type": "integer", "enum": [0, 1]},
        "Has_Cracks": {"type": "integer", "enum": [0, 1]},
        "Spatter_Level_0_3": {"type": "integer", "unit": "level"},
        "Discoloration_Level_0_3": {"type": "integer", "unit": "level"},
        "Tensile_Shear_Strength_N": {"type": "number", "unit": "N"},
        "Peel_Strength_N": {"type": "number", "unit": "N"},
        "Contact_Resistance_uOhm": {"type": "number", "unit": "uOhm"},
        "Peak_Temperature_C": {"type": "number", "unit": "C"},
        "IMC_Thickness_um": {"type": "number", "unit": "um"},
        # Uncertainties
        "Replicates": {"type": "integer", "unit": "count"},
        "Tensile_Shear_Strength_Std_N": {"type": "number", "unit": "N"},
        "Contact_Resistance_Std_uOhm": {"type": "number", "unit": "uOhm"},
        # Extreme-temperature
        "Thermal_Cycles": {"type": "number", "unit": "cycles"},
        "Strength_Degradation_Percent": {"type": "number", "unit": "%"},
        "Resistance_Increase_Percent": {"type": "number", "unit": "%"},
        "Cycles_to_Failure_Estimate": {"type": "number", "unit": "cycles"},
        "Post_ThermalCycle_Tensile_Shear_Strength_N": {"type": "number", "unit": "N"},
        "Post_ThermalCycle_Contact_Resistance_uOhm": {"type": "number", "unit": "uOhm"},
        "Creep_Time_to_Failure_h_100C_50pctUTS": {"type": "number", "unit": "h"},
        "IMC_Thickness_PostAging_500h_120C_um": {"type": "number", "unit": "um"},
        # Meta
        "Fidelity_Weight": {"type": "number"},
    }


def assemble_dataframe(
    inputs: Dict[str, np.ndarray | List[str]],
    outputs: Dict[str, np.ndarray | List[int] | List[float]],
    source: str,
    start_index: int,
) -> pd.DataFrame:
    n = len(inputs["Laser_Power_W"])  # type: ignore[index]
    df = pd.DataFrame({**inputs, **outputs})
    if source == "Experimental":
        ids = [f"W-{start_index + i:04d}" for i in range(n)]
        weight = np.full(n, 1.0)
    else:
        ids = [f"S-{start_index + i:05d}" for i in range(n)]
        weight = np.full(n, 0.35)

    df.insert(0, "Weld_ID", ids)
    df.insert(1, "Data_Source", source)
    df["Fidelity_Weight"] = weight
    return df


def generate_datasets(exp_n: int, sim_n: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_everything(seed)

    exp_inputs = generate_inputs(exp_n, RANGES, is_experimental=True)
    exp_outputs = generate_outputs(exp_inputs, source="Experimental")
    exp_df = assemble_dataframe(exp_inputs, exp_outputs, source="Experimental", start_index=1)

    # Extreme-temperature for a subset of experimental samples
    exp_df = extreme_temperature_for_subset(exp_df, subset_fraction=0.4, seed=seed)

    sim_inputs = generate_inputs(sim_n, RANGES, is_experimental=False)
    sim_outputs = generate_outputs(sim_inputs, source="Simulation")
    sim_df_raw = assemble_dataframe(sim_inputs, sim_outputs, source="Simulation", start_index=1)

    # Calibrate to experimental distributions
    sim_df = calibrate_simulation_to_experimental(sim_df_raw, exp_df)

    common_cols = sorted(set(exp_df.columns).union(set(sim_df.columns)))
    exp_df = exp_df.reindex(columns=common_cols)
    sim_df = sim_df.reindex(columns=common_cols)

    master_df = pd.concat([exp_df, sim_df], ignore_index=True)
    return exp_df, sim_df, master_df


def write_outputs(exp_df: pd.DataFrame, sim_df: pd.DataFrame, master_df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "experimental_csv": os.path.join(out_dir, "experimental.csv"),
        "simulation_csv": os.path.join(out_dir, "simulation.csv"),
        "master_csv": os.path.join(out_dir, "master.csv"),
        "schema_json": os.path.join(out_dir, "schema.json"),
        "zip_path": os.path.join(os.path.dirname(out_dir), "welding_inverse_design_dataset.zip"),
    }

    exp_df.to_csv(paths["experimental_csv"], index=False)
    sim_df.to_csv(paths["simulation_csv"], index=False)
    master_df.to_csv(paths["master_csv"], index=False)

    with open(paths["schema_json"], "w", encoding="utf-8") as f:
        json.dump(build_schema(), f, indent=2)

    # Create a zip with the data directory contents
    with zipfile.ZipFile(paths["zip_path"], mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in ["experimental.csv", "simulation.csv", "master.csv", "schema.json"]:
            zf.write(os.path.join(out_dir, filename), arcname=os.path.join("data", filename))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate welding inverse-design datasets")
    parser.add_argument("--exp-n", type=int, default=300, help="Number of experimental-like samples")
    parser.add_argument("--sim-n", type=int, default=5000, help="Number of simulation-like samples")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for CSVs and schema")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--zip", dest="make_zip", type=str, default="true", help="Create zip archive (true/false)")
    args = parser.parse_args()

    exp_df, sim_df, master_df = generate_datasets(args.exp_n, args.sim_n, args.seed)

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)

    paths = write_outputs(exp_df, sim_df, master_df, out_dir)

    print(f"Experimental dataset: {paths['experimental_csv']}")
    print(f"Simulation dataset:   {paths['simulation_csv']}")
    print(f"Master dataset:       {paths['master_csv']}")
    print(f"Schema:               {paths['schema_json']}")
    print(f"Zip archive:          {paths['zip_path']}")


if __name__ == "__main__":
    main()
