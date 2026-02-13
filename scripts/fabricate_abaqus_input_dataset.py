#!/usr/bin/env python3
"""
Fabricate a reproducible YSZ/GDC/LSCF dataset for:
"Calibration and Mesh Objectivity in Mixed-Mode Fracture of YSZ/GDC/LSCF Interfaces:
Bridging Implicit UEL and UMAT Frameworks"

Outputs:
- CSV tables grouped by data section
- Publication-style figures (PNG)
- QA bounds report
- Single zip archive containing all CSV files
"""

from __future__ import annotations

import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RNG_SEED = 790
TOPIC = (
    "Calibration and Mesh Objectivity in Mixed-Mode Fracture of YSZ/GDC/LSCF Interfaces: "
    "Bridging Implicit UEL and UMAT Frameworks"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "fabricated_ysz_gdc_lscf_dataset"
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
META_DIR = OUT_DIR / "meta"


def ensure_directories() -> None:
    for path in (OUT_DIR, CSV_DIR, FIG_DIR, META_DIR):
        path.mkdir(parents=True, exist_ok=True)


def linear_interp(temp_c: float, t0: float, t1: float, v0: float, v1: float) -> float:
    ratio = (temp_c - t0) / (t1 - t0)
    return v0 + ratio * (v1 - v0)


def build_geometric_microstructural() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "case_id": "BASELINE",
            "t_YSZ_um": 12.0,
            "t_GDC_um": 1.5,
            "t_LSCF_um": 35.0,
            "Ra_YSZ_GDC_um": 0.08,
            "lambda_YSZ_GDC_um": 2.2,
            "Ra_GDC_LSCF_um": 0.14,
            "lambda_GDC_LSCF_um": 3.1,
            "phi_pore_LSCF": 0.35,
            "rve_size_um": 60.0,
            "mesh_target_h_um": 0.40,
            "phase_distribution_strategy": "Synthetic RVE + FIB-SEM morphology inspired",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_method": "SEM/AFM-informed synthetic calibration",
            "notes": "Nominal architecture used for UMAT-UEL coupling baseline.",
        },
        {
            "case_id": "THIN_YSZ",
            "t_YSZ_um": 8.0,
            "t_GDC_um": 1.5,
            "t_LSCF_um": 35.0,
            "Ra_YSZ_GDC_um": 0.08,
            "lambda_YSZ_GDC_um": 2.2,
            "Ra_GDC_LSCF_um": 0.14,
            "lambda_GDC_LSCF_um": 3.1,
            "phi_pore_LSCF": 0.35,
            "rve_size_um": 60.0,
            "mesh_target_h_um": 0.30,
            "phase_distribution_strategy": "Synthetic RVE + FIB-SEM morphology inspired",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_method": "Parametric sweep bound",
            "notes": "Thickness down-sweep for curvature and mode-mix sensitivity.",
        },
        {
            "case_id": "THICK_GDC",
            "t_YSZ_um": 12.0,
            "t_GDC_um": 3.0,
            "t_LSCF_um": 35.0,
            "Ra_YSZ_GDC_um": 0.10,
            "lambda_YSZ_GDC_um": 2.8,
            "Ra_GDC_LSCF_um": 0.15,
            "lambda_GDC_LSCF_um": 3.2,
            "phi_pore_LSCF": 0.35,
            "rve_size_um": 60.0,
            "mesh_target_h_um": 0.40,
            "phase_distribution_strategy": "Synthetic RVE + FIB-SEM morphology inspired",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_method": "Parametric sweep bound",
            "notes": "Barrier-layer up-sweep to test mixed-mode delamination shift.",
        },
        {
            "case_id": "HIGH_ROUGHNESS",
            "t_YSZ_um": 12.0,
            "t_GDC_um": 1.5,
            "t_LSCF_um": 35.0,
            "Ra_YSZ_GDC_um": 0.20,
            "lambda_YSZ_GDC_um": 4.0,
            "Ra_GDC_LSCF_um": 0.30,
            "lambda_GDC_LSCF_um": 5.0,
            "phi_pore_LSCF": 0.35,
            "rve_size_um": 70.0,
            "mesh_target_h_um": 0.25,
            "phase_distribution_strategy": "Synthetic RVE + FIB-SEM morphology inspired",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_method": "AFM/profilometry-informed high roughness envelope",
            "notes": "Aggressive interface roughness for mesh-objectivity stress test.",
        },
        {
            "case_id": "LOW_POROSITY",
            "t_YSZ_um": 12.0,
            "t_GDC_um": 1.5,
            "t_LSCF_um": 35.0,
            "Ra_YSZ_GDC_um": 0.08,
            "lambda_YSZ_GDC_um": 2.2,
            "Ra_GDC_LSCF_um": 0.12,
            "lambda_GDC_LSCF_um": 2.9,
            "phi_pore_LSCF": 0.25,
            "rve_size_um": 55.0,
            "mesh_target_h_um": 0.40,
            "phase_distribution_strategy": "Synthetic RVE + FIB-SEM morphology inspired",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_method": "Image-analysis lower bound",
            "notes": "Higher effective cathode stiffness scenario.",
        },
        {
            "case_id": "HIGH_POROSITY",
            "t_YSZ_um": 12.0,
            "t_GDC_um": 1.5,
            "t_LSCF_um": 35.0,
            "Ra_YSZ_GDC_um": 0.08,
            "lambda_YSZ_GDC_um": 2.2,
            "Ra_GDC_LSCF_um": 0.18,
            "lambda_GDC_LSCF_um": 3.4,
            "phi_pore_LSCF": 0.45,
            "rve_size_um": 75.0,
            "mesh_target_h_um": 0.50,
            "phase_distribution_strategy": "Synthetic RVE + FIB-SEM morphology inspired",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_method": "Image-analysis upper bound",
            "notes": "Lower effective cathode stiffness scenario.",
        },
    ]
    geom_df = pd.DataFrame(rows)

    phase_stats_rows = [
        {
            "rve_id": "RVE_A",
            "phase": "YSZ",
            "volume_fraction": 0.29,
            "mean_feature_um": 1.9,
            "std_feature_um": 0.8,
            "correlation_length_um": 2.6,
            "connectivity_index": 0.88,
            "synthetic_seed": RNG_SEED,
        },
        {
            "rve_id": "RVE_A",
            "phase": "GDC",
            "volume_fraction": 0.13,
            "mean_feature_um": 0.6,
            "std_feature_um": 0.2,
            "correlation_length_um": 0.9,
            "connectivity_index": 0.82,
            "synthetic_seed": RNG_SEED,
        },
        {
            "rve_id": "RVE_A",
            "phase": "LSCF",
            "volume_fraction": 0.37,
            "mean_feature_um": 3.8,
            "std_feature_um": 1.3,
            "correlation_length_um": 4.9,
            "connectivity_index": 0.79,
            "synthetic_seed": RNG_SEED,
        },
        {
            "rve_id": "RVE_A",
            "phase": "Pore",
            "volume_fraction": 0.21,
            "mean_feature_um": 2.4,
            "std_feature_um": 1.1,
            "correlation_length_um": 3.2,
            "connectivity_index": 0.71,
            "synthetic_seed": RNG_SEED,
        },
        {
            "rve_id": "RVE_B",
            "phase": "YSZ",
            "volume_fraction": 0.27,
            "mean_feature_um": 2.1,
            "std_feature_um": 0.9,
            "correlation_length_um": 2.8,
            "connectivity_index": 0.86,
            "synthetic_seed": RNG_SEED + 1,
        },
        {
            "rve_id": "RVE_B",
            "phase": "GDC",
            "volume_fraction": 0.14,
            "mean_feature_um": 0.7,
            "std_feature_um": 0.2,
            "correlation_length_um": 1.0,
            "connectivity_index": 0.83,
            "synthetic_seed": RNG_SEED + 1,
        },
        {
            "rve_id": "RVE_B",
            "phase": "LSCF",
            "volume_fraction": 0.36,
            "mean_feature_um": 4.0,
            "std_feature_um": 1.4,
            "correlation_length_um": 5.1,
            "connectivity_index": 0.80,
            "synthetic_seed": RNG_SEED + 1,
        },
        {
            "rve_id": "RVE_B",
            "phase": "Pore",
            "volume_fraction": 0.23,
            "mean_feature_um": 2.5,
            "std_feature_um": 1.2,
            "correlation_length_um": 3.3,
            "connectivity_index": 0.70,
            "synthetic_seed": RNG_SEED + 1,
        },
        {
            "rve_id": "RVE_C",
            "phase": "YSZ",
            "volume_fraction": 0.30,
            "mean_feature_um": 2.0,
            "std_feature_um": 0.8,
            "correlation_length_um": 2.7,
            "connectivity_index": 0.87,
            "synthetic_seed": RNG_SEED + 2,
        },
        {
            "rve_id": "RVE_C",
            "phase": "GDC",
            "volume_fraction": 0.12,
            "mean_feature_um": 0.6,
            "std_feature_um": 0.2,
            "correlation_length_um": 0.9,
            "connectivity_index": 0.82,
            "synthetic_seed": RNG_SEED + 2,
        },
        {
            "rve_id": "RVE_C",
            "phase": "LSCF",
            "volume_fraction": 0.39,
            "mean_feature_um": 3.6,
            "std_feature_um": 1.2,
            "correlation_length_um": 4.7,
            "connectivity_index": 0.78,
            "synthetic_seed": RNG_SEED + 2,
        },
        {
            "rve_id": "RVE_C",
            "phase": "Pore",
            "volume_fraction": 0.19,
            "mean_feature_um": 2.2,
            "std_feature_um": 1.0,
            "correlation_length_um": 3.1,
            "connectivity_index": 0.73,
            "synthetic_seed": RNG_SEED + 2,
        },
    ]
    phase_df = pd.DataFrame(phase_stats_rows)
    return geom_df, phase_df


def build_thermoelastic() -> pd.DataFrame:
    temperatures = [25, 200, 400, 600, 800]
    properties = {
        "YSZ": {
            "E_25": 205.0,
            "E_800": 170.0,
            "nu_25": 0.23,
            "nu_800": 0.23,
            "alpha_25": 10.0,
            "alpha_800": 10.7,
            "density": 5900.0,
        },
        "GDC": {
            "E_25": 190.0,
            "E_800": 152.0,
            "nu_25": 0.24,
            "nu_800": 0.25,
            "alpha_25": 11.5,
            "alpha_800": 12.8,
            "density": 7200.0,
        },
        "LSCF": {
            "E_25": 98.0,
            "E_800": 72.0,
            "nu_25": 0.27,
            "nu_800": 0.29,
            "alpha_25": 14.8,
            "alpha_800": 16.4,
            "density": 6400.0,
        },
    }

    rows = []
    for material, prop in properties.items():
        for temp in temperatures:
            e_gpa = linear_interp(temp, 25, 800, prop["E_25"], prop["E_800"])
            nu = linear_interp(temp, 25, 800, prop["nu_25"], prop["nu_800"])
            alpha = linear_interp(temp, 25, 800, prop["alpha_25"], prop["alpha_800"])
            rows.append(
                {
                    "material": material,
                    "temperature_C": temp,
                    "E_GPa": round(e_gpa, 4),
                    "E_Pa": round(e_gpa * 1.0e9, 2),
                    "nu": round(nu, 5),
                    "alpha_1e-6_per_K": round(alpha, 4),
                    "alpha_per_K": alpha * 1.0e-6,
                    "density_kg_m3": prop["density"],
                    "status": "FABRICATED_LIT_ANCHORED",
                    "source_strategy": "Nanoindentation/small-punch bounded by literature",
                }
            )
    return pd.DataFrame(rows)


def build_defect_chemical() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_rows = [
        {
            "material": "GDC",
            "temperature_C": 700,
            "beta_iso_per_delta": 0.0105,
            "beta_11_per_delta": np.nan,
            "beta_33_per_delta": np.nan,
            "delta_profile_model": "uniform_or_gradient",
            "delta_min": 0.008,
            "delta_max": 0.025,
            "pO2_atm": "1e-5_to_0.21",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "High-T XRD + controlled pO2 dilatometry surrogate",
            "notes": "Isotropic coefficient for barrier-layer chemical strain.",
        },
        {
            "material": "GDC",
            "temperature_C": 800,
            "beta_iso_per_delta": 0.0118,
            "beta_11_per_delta": np.nan,
            "beta_33_per_delta": np.nan,
            "delta_profile_model": "uniform_or_gradient",
            "delta_min": 0.010,
            "delta_max": 0.030,
            "pO2_atm": "1e-5_to_0.21",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "High-T XRD + controlled pO2 dilatometry surrogate",
            "notes": "Baseline operation temperature value for UMAT eigenstrain.",
        },
        {
            "material": "LSCF",
            "temperature_C": 700,
            "beta_iso_per_delta": np.nan,
            "beta_11_per_delta": 0.0069,
            "beta_33_per_delta": 0.0128,
            "delta_profile_model": "uniform_or_gradient",
            "delta_min": 0.030,
            "delta_max": 0.110,
            "pO2_atm": "1e-5_to_0.21",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Textured-sample XRD anisotropic fitting surrogate",
            "notes": "Anisotropy retained for cathode mode-mix sensitivity.",
        },
        {
            "material": "LSCF",
            "temperature_C": 800,
            "beta_iso_per_delta": np.nan,
            "beta_11_per_delta": 0.0075,
            "beta_33_per_delta": 0.0139,
            "delta_profile_model": "uniform_or_gradient",
            "delta_min": 0.035,
            "delta_max": 0.120,
            "pO2_atm": "1e-5_to_0.21",
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Textured-sample XRD anisotropic fitting surrogate",
            "notes": "Used in coupled UMAT chemical-eigenstrain decomposition.",
        },
    ]
    defect_df = pd.DataFrame(data_rows)

    x_norm = np.linspace(0.0, 1.0, 31)
    delta_uniform = np.full_like(x_norm, 0.07)
    delta_gradient = 0.03 + 0.09 * np.power(x_norm, 1.3)
    delta_gdc_gradient = 0.008 + 0.017 * x_norm
    beta11 = 0.0075
    beta33 = 0.0139
    beta_iso_gdc = 0.0118

    prof_df = pd.DataFrame(
        {
            "x_norm_from_YSZ_interface": x_norm,
            "x_in_LSCF_um_at_35um_thickness": x_norm * 35.0,
            "delta_LSCF_uniform": delta_uniform,
            "delta_LSCF_gradient": delta_gradient,
            "delta_GDC_gradient": delta_gdc_gradient,
            "eps_ch_LSCF_11_uniform": beta11 * delta_uniform,
            "eps_ch_LSCF_33_uniform": beta33 * delta_uniform,
            "eps_ch_LSCF_11_gradient": beta11 * delta_gradient,
            "eps_ch_LSCF_33_gradient": beta33 * delta_gradient,
            "eps_ch_GDC_iso_gradient": beta_iso_gdc * delta_gdc_gradient,
        }
    )
    return defect_df, prof_df


def build_fracture_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fracture_rows = [
        {
            "entity_class": "bulk",
            "entity": "YSZ",
            "Gc_bulk_J_m2": 14.2,
            "Gc_I_J_m2": np.nan,
            "Gc_II_J_m2": np.nan,
            "phi_n_J_m2": np.nan,
            "phi_t_J_m2": np.nan,
            "Tmax_n_MPa": np.nan,
            "Tmax_t_MPa": np.nan,
            "char_length_um": np.nan,
            "eta_BK": np.nan,
            "weibull_m": 12.0,
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Indentation fracture + Weibull fit surrogate",
            "notes": "Bulk phase-field fracture energy for YSZ.",
        },
        {
            "entity_class": "bulk",
            "entity": "GDC",
            "Gc_bulk_J_m2": 10.8,
            "Gc_I_J_m2": np.nan,
            "Gc_II_J_m2": np.nan,
            "phi_n_J_m2": np.nan,
            "phi_t_J_m2": np.nan,
            "Tmax_n_MPa": np.nan,
            "Tmax_t_MPa": np.nan,
            "char_length_um": np.nan,
            "eta_BK": np.nan,
            "weibull_m": 10.0,
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Indentation fracture + Weibull fit surrogate",
            "notes": "Bulk phase-field fracture energy for GDC.",
        },
        {
            "entity_class": "bulk",
            "entity": "LSCF",
            "Gc_bulk_J_m2": 7.0,
            "Gc_I_J_m2": np.nan,
            "Gc_II_J_m2": np.nan,
            "phi_n_J_m2": np.nan,
            "phi_t_J_m2": np.nan,
            "Tmax_n_MPa": np.nan,
            "Tmax_t_MPa": np.nan,
            "char_length_um": np.nan,
            "eta_BK": np.nan,
            "weibull_m": 8.0,
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Small-punch statistical strength surrogate",
            "notes": "Porous cathode bulk fracture energy.",
        },
        {
            "entity_class": "interface",
            "entity": "YSZ|GDC",
            "Gc_bulk_J_m2": np.nan,
            "Gc_I_J_m2": 4.2,
            "Gc_II_J_m2": 10.1,
            "phi_n_J_m2": 4.2,
            "phi_t_J_m2": 10.1,
            "Tmax_n_MPa": 180.0,
            "Tmax_t_MPa": 245.0,
            "char_length_um": 0.75,
            "eta_BK": 2.1,
            "weibull_m": np.nan,
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Four-point/micro-cantilever mixed-mode surrogate",
            "notes": "BK exponent retained at assumed default value 2.1.",
        },
        {
            "entity_class": "interface",
            "entity": "GDC|LSCF",
            "Gc_bulk_J_m2": np.nan,
            "Gc_I_J_m2": 2.9,
            "Gc_II_J_m2": 7.2,
            "phi_n_J_m2": 2.9,
            "phi_t_J_m2": 7.2,
            "Tmax_n_MPa": 135.0,
            "Tmax_t_MPa": 190.0,
            "char_length_um": 1.10,
            "eta_BK": 2.1,
            "weibull_m": np.nan,
            "status": "FABRICATED_LIT_ANCHORED",
            "source_strategy": "Four-point/micro-cantilever mixed-mode surrogate",
            "notes": "More compliant interface with lower normal traction capacity.",
        },
    ]
    fracture_df = pd.DataFrame(fracture_rows)

    coh_rows = []
    for interface, gc_i, tmax in [
        ("YSZ|GDC", 4.2, 180.0),
        ("GDC|LSCF", 2.9, 135.0),
    ]:
        delta0_m = max(1e-12, 2.0 * gc_i / (tmax * 1.0e6))
        delta_values_m = np.linspace(0.0, 6.0 * delta0_m, 200)
        x = delta_values_m / delta0_m
        traction_mpa = tmax * x * np.exp(1.0 - x)
        for d_val, t_val in zip(delta_values_m, traction_mpa):
            coh_rows.append(
                {
                    "interface": interface,
                    "delta_n_um": d_val * 1.0e6,
                    "traction_n_MPa": t_val,
                    "delta0_um": delta0_m * 1.0e6,
                    "law_type": "Xu-Needleman-like exponential envelope",
                }
            )
    cohesive_curves_df = pd.DataFrame(coh_rows)
    return fracture_df, cohesive_curves_df


def build_validation_targets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temperature = np.arange(25, 801, 25)
    kappa_model = 0.00024 * (temperature - 25) + 1.05e-6 * np.square(temperature - 25)
    kappa_target = kappa_model * (1.0 + 0.03 * np.sin(temperature / 120.0))
    tolerance = np.maximum(0.015, 0.06 * kappa_target)
    curv_df = pd.DataFrame(
        {
            "temperature_C": temperature,
            "kappa_model_1_per_m": kappa_model,
            "kappa_target_1_per_m": kappa_target,
            "kappa_tolerance_1_per_m": tolerance,
            "validation_method": "DIC / optical dilatometry surrogate",
            "usage": "UMAT thermo-chemical strain field validation target",
        }
    )

    p_o2 = np.array([2.1e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5])
    onset_rows = []
    ysz_gdc_temp = [865, 850, 835, 820, 805]
    gdc_lscf_temp = [810, 790, 770, 745, 720]
    for p_val, t1, t2 in zip(p_o2, ysz_gdc_temp, gdc_lscf_temp):
        onset_rows.append(
            {
                "interface": "YSZ|GDC",
                "pO2_atm": p_val,
                "critical_temperature_C": t1,
                "critical_delta_n_um": 0.095 - 0.010 * math.log10(1.0 / p_val),
                "validation_method": "In-situ mixed-mode loading surrogate",
                "usage": "UEL mixed-mode delamination onset calibration",
            }
        )
        onset_rows.append(
            {
                "interface": "GDC|LSCF",
                "pO2_atm": p_val,
                "critical_temperature_C": t2,
                "critical_delta_n_um": 0.078 - 0.009 * math.log10(1.0 / p_val),
                "validation_method": "In-situ mixed-mode loading surrogate",
                "usage": "UEL mixed-mode delamination onset calibration",
            }
        )
    onset_df = pd.DataFrame(onset_rows)

    crack_rows = [
        {
            "specimen_id": "CPM_01",
            "loading_mode": "thermal_cooldown",
            "dominant_failure": "interface_delamination",
            "primary_interface": "GDC|LSCF",
            "branching_angle_deg": 18.0,
            "path_tortuosity": 1.23,
            "interface_damage_fraction": 0.79,
            "bulk_kink_events": 1,
            "validation_method": "Post-mortem FIB-SEM surrogate descriptors",
        },
        {
            "specimen_id": "CPM_02",
            "loading_mode": "high_pO2_operation",
            "dominant_failure": "mixed_interface_bulk",
            "primary_interface": "YSZ|GDC",
            "branching_angle_deg": 24.0,
            "path_tortuosity": 1.31,
            "interface_damage_fraction": 0.68,
            "bulk_kink_events": 2,
            "validation_method": "In-situ SEM surrogate descriptors",
        },
        {
            "specimen_id": "CPM_03",
            "loading_mode": "low_pO2_operation",
            "dominant_failure": "interface_delamination",
            "primary_interface": "GDC|LSCF",
            "branching_angle_deg": 14.0,
            "path_tortuosity": 1.18,
            "interface_damage_fraction": 0.84,
            "bulk_kink_events": 1,
            "validation_method": "In-situ SEM surrogate descriptors",
        },
        {
            "specimen_id": "CPM_04",
            "loading_mode": "thermal_cycle",
            "dominant_failure": "mixed_interface_bulk",
            "primary_interface": "YSZ|GDC",
            "branching_angle_deg": 27.0,
            "path_tortuosity": 1.35,
            "interface_damage_fraction": 0.63,
            "bulk_kink_events": 3,
            "validation_method": "DIC + SEM overlay surrogate descriptors",
        },
    ]
    crack_df = pd.DataFrame(crack_rows)

    return curv_df, onset_df, crack_df


def build_parametric_sweep_matrix() -> pd.DataFrame:
    rows = [
        ["t_YSZ", "Geometric", "um", 12.0, 8.0, 16.0, "uniform", 5, "mesh+UMAT"],
        ["t_GDC", "Geometric", "um", 1.5, 0.8, 3.0, "uniform", 5, "mesh+UMAT"],
        ["t_LSCF", "Geometric", "um", 35.0, 20.0, 45.0, "uniform", 5, "mesh+UMAT"],
        ["Ra_YSZ_GDC", "Microstructure", "um", 0.08, 0.03, 0.20, "uniform", 5, "mesh"],
        ["lambda_YSZ_GDC", "Microstructure", "um", 2.2, 1.2, 4.5, "uniform", 5, "mesh"],
        ["phi_pore_LSCF", "Microstructure", "-", 0.35, 0.20, 0.50, "uniform", 5, "UMAT"],
        ["E_YSZ_800C", "Thermoelastic", "GPa", 170.0, 155.0, 190.0, "normal", 5, "UMAT"],
        ["E_GDC_800C", "Thermoelastic", "GPa", 152.0, 135.0, 170.0, "normal", 5, "UMAT"],
        ["E_LSCF_800C", "Thermoelastic", "GPa", 72.0, 60.0, 90.0, "normal", 5, "UMAT"],
        ["alpha_YSZ", "Thermoelastic", "1e-6/K", 10.7, 10.2, 11.2, "normal", 5, "UMAT"],
        ["alpha_GDC", "Thermoelastic", "1e-6/K", 12.8, 12.0, 13.6, "normal", 5, "UMAT"],
        ["alpha_LSCF", "Thermoelastic", "1e-6/K", 16.4, 15.2, 17.6, "normal", 5, "UMAT"],
        ["beta_iso_GDC", "Chemical", "per_delta", 0.0118, 0.0090, 0.0140, "uniform", 5, "UMAT"],
        ["beta11_LSCF", "Chemical", "per_delta", 0.0075, 0.0058, 0.0090, "uniform", 5, "UMAT"],
        ["beta33_LSCF", "Chemical", "per_delta", 0.0139, 0.0105, 0.0165, "uniform", 5, "UMAT"],
        ["Gc_bulk_YSZ", "Fracture", "J/m2", 14.2, 10.0, 18.0, "normal", 5, "phase_field"],
        ["GcI_YSZ_GDC", "Fracture", "J/m2", 4.2, 2.5, 6.0, "normal", 5, "UEL"],
        ["GcII_YSZ_GDC", "Fracture", "J/m2", 10.1, 6.0, 14.0, "normal", 5, "UEL"],
        ["Tmax_n_YSZ_GDC", "Fracture", "MPa", 180.0, 140.0, 220.0, "uniform", 5, "UEL"],
        ["Tmax_t_YSZ_GDC", "Fracture", "MPa", 245.0, 190.0, 290.0, "uniform", 5, "UEL"],
        ["eta_BK", "Fracture", "-", 2.1, 1.6, 2.6, "uniform", 5, "UEL"],
        ["weibull_m_YSZ", "Fracture", "-", 12.0, 8.0, 16.0, "uniform", 5, "probabilistic"],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "parameter_symbol",
            "section",
            "unit",
            "baseline",
            "low",
            "high",
            "distribution",
            "levels",
            "target_block",
        ],
    )


def build_abaqus_blocks(thermo_df: pd.DataFrame, fracture_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    umat_rows = []
    line_idx = 1
    for material in ["YSZ", "GDC", "LSCF"]:
        subset = thermo_df[thermo_df["material"] == material].sort_values("temperature_C")
        umat_rows.append({"line_index": line_idx, "block_type": "MATERIAL", "inp_line": f"*MATERIAL, NAME={material}"})
        line_idx += 1
        umat_rows.append(
            {"line_index": line_idx, "block_type": "ELASTIC", "inp_line": "*ELASTIC, TYPE=ISO, DEPENDENCIES=1"}
        )
        line_idx += 1
        for _, r in subset.iterrows():
            umat_rows.append(
                {
                    "line_index": line_idx,
                    "block_type": "ELASTIC",
                    "inp_line": f"{r['E_Pa']:.3e}, {r['nu']:.5f}, {int(r['temperature_C'])}",
                }
            )
            line_idx += 1
        umat_rows.append({"line_index": line_idx, "block_type": "EXPANSION", "inp_line": "*EXPANSION, ZERO=25."})
        line_idx += 1
        for _, r in subset.iterrows():
            umat_rows.append(
                {
                    "line_index": line_idx,
                    "block_type": "EXPANSION",
                    "inp_line": f"{r['alpha_per_K']:.6e}, {int(r['temperature_C'])}",
                }
            )
            line_idx += 1
        umat_rows.append({"line_index": line_idx, "block_type": "DEPVAR", "inp_line": "*DEPVAR"})
        line_idx += 1
        umat_rows.append({"line_index": line_idx, "block_type": "DEPVAR", "inp_line": "6"})
        line_idx += 1

    umat_df = pd.DataFrame(umat_rows)

    uel_rows = []
    line_idx = 1
    for _, row in fracture_df[fracture_df["entity_class"] == "interface"].iterrows():
        iface = row["entity"].replace("|", "_")
        uel_rows.append(
            {
                "line_index": line_idx,
                "block_type": "UEL_PROPERTY",
                "inp_line": f"*UEL PROPERTY, ELSET=COH_{iface}",
            }
        )
        line_idx += 1
        uel_rows.append(
            {
                "line_index": line_idx,
                "block_type": "UEL_PROPERTY",
                "inp_line": (
                    f"{row['Gc_I_J_m2']:.4f}, {row['Gc_II_J_m2']:.4f}, {row['Tmax_n_MPa']:.3f}, "
                    f"{row['Tmax_t_MPa']:.3f}, {row['eta_BK']:.3f}, {row['char_length_um']:.4f}"
                ),
            }
        )
        line_idx += 1
    uel_df = pd.DataFrame(uel_rows)
    return umat_df, uel_df


def run_qa_checks(thermo_df: pd.DataFrame, fracture_df: pd.DataFrame, geom_df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    nu_ok = thermo_df["nu"].between(-1.0, 0.5).all()
    checks.append(
        {
            "check_id": "QA001",
            "rule": "-1 < nu < 0.5",
            "result": "PASS" if nu_ok else "FAIL",
            "detail": f"nu range = [{thermo_df['nu'].min():.4f}, {thermo_df['nu'].max():.4f}]",
        }
    )

    e_ok = (thermo_df["E_GPa"] > 0.0).all()
    checks.append(
        {
            "check_id": "QA002",
            "rule": "E > 0",
            "result": "PASS" if e_ok else "FAIL",
            "detail": f"minimum E_GPa = {thermo_df['E_GPa'].min():.4f}",
        }
    )

    alpha_ok = (thermo_df["alpha_per_K"] > 0.0).all()
    checks.append(
        {
            "check_id": "QA003",
            "rule": "alpha > 0",
            "result": "PASS" if alpha_ok else "FAIL",
            "detail": f"minimum alpha_per_K = {thermo_df['alpha_per_K'].min():.6e}",
        }
    )

    iface = fracture_df[fracture_df["entity_class"] == "interface"].copy()
    gc_ok = (iface["Gc_II_J_m2"] >= iface["Gc_I_J_m2"]).all()
    checks.append(
        {
            "check_id": "QA004",
            "rule": "Gc_II >= Gc_I for interfaces",
            "result": "PASS" if gc_ok else "FAIL",
            "detail": (
                "min(Gc_II - Gc_I) = "
                f"{(iface['Gc_II_J_m2'] - iface['Gc_I_J_m2']).min():.6f} J/m2"
            ),
        }
    )

    traction_ok = (iface["Tmax_n_MPa"] > 0.0).all() and (iface["Tmax_t_MPa"] > 0.0).all()
    checks.append(
        {
            "check_id": "QA005",
            "rule": "Tmax_n > 0 and Tmax_t > 0",
            "result": "PASS" if traction_ok else "FAIL",
            "detail": (
                f"min Tmax_n = {iface['Tmax_n_MPa'].min():.3f} MPa, "
                f"min Tmax_t = {iface['Tmax_t_MPa'].min():.3f} MPa"
            ),
        }
    )

    porosity_ok = geom_df["phi_pore_LSCF"].between(0.0, 1.0).all()
    checks.append(
        {
            "check_id": "QA006",
            "rule": "0 <= phi_pore_LSCF <= 1",
            "result": "PASS" if porosity_ok else "FAIL",
            "detail": (
                f"phi_pore range = [{geom_df['phi_pore_LSCF'].min():.4f}, "
                f"{geom_df['phi_pore_LSCF'].max():.4f}]"
            ),
        }
    )

    return pd.DataFrame(checks)


def create_figures(
    geom_df: pd.DataFrame,
    thermo_df: pd.DataFrame,
    defect_profile_df: pd.DataFrame,
    fracture_df: pd.DataFrame,
    cohesive_df: pd.DataFrame,
    curv_df: pd.DataFrame,
    onset_df: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline = geom_df[geom_df["case_id"] == "BASELINE"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    layers = ["YSZ", "GDC", "LSCF"]
    thickness = [baseline["t_YSZ_um"], baseline["t_GDC_um"], baseline["t_LSCF_um"]]
    axes[0].bar(layers, thickness, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_ylabel("Thickness (um)")
    axes[0].set_title("Baseline layer thicknesses")

    x = np.linspace(0.0, 20.0, 800)
    y1 = baseline["Ra_YSZ_GDC_um"] * np.sin(2.0 * np.pi * x / baseline["lambda_YSZ_GDC_um"])
    y2 = baseline["Ra_GDC_LSCF_um"] * np.sin(2.0 * np.pi * x / baseline["lambda_GDC_LSCF_um"] + 0.7)
    axes[1].plot(x, y1, label="YSZ|GDC profile")
    axes[1].plot(x, y2, label="GDC|LSCF profile")
    axes[1].set_xlabel("Along-interface distance (um)")
    axes[1].set_ylabel("Height (um)")
    axes[1].set_title("Interface roughness envelopes")
    axes[1].legend(loc="upper right")
    fig.suptitle("Figure 1: Geometric and microstructural baseline", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_01_geometry_microstructure.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    for mat in ["YSZ", "GDC", "LSCF"]:
        sub = thermo_df[thermo_df["material"] == mat].sort_values("temperature_C")
        axes[0].plot(sub["temperature_C"], sub["E_GPa"], marker="o", label=mat)
        axes[1].plot(sub["temperature_C"], sub["alpha_1e-6_per_K"], marker="o", label=mat)
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Young's modulus (GPa)")
    axes[0].set_title("Temperature-dependent modulus")
    axes[1].set_xlabel("Temperature (C)")
    axes[1].set_ylabel("Secant CTE (1e-6/K)")
    axes[1].set_title("Temperature-dependent secant CTE")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Figure 2: Thermo-elastic continuum inputs for UMAT", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_02_thermoelastic_inputs.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    axes[0].plot(
        defect_profile_df["x_norm_from_YSZ_interface"],
        defect_profile_df["delta_LSCF_uniform"],
        label="LSCF delta uniform",
    )
    axes[0].plot(
        defect_profile_df["x_norm_from_YSZ_interface"],
        defect_profile_df["delta_LSCF_gradient"],
        label="LSCF delta gradient",
    )
    axes[0].plot(
        defect_profile_df["x_norm_from_YSZ_interface"],
        defect_profile_df["delta_GDC_gradient"],
        label="GDC delta gradient",
    )
    axes[0].set_xlabel("Normalized distance from YSZ interface")
    axes[0].set_ylabel("Oxygen non-stoichiometry delta")
    axes[0].set_title("Delta profiles across active layers")
    axes[0].legend()

    axes[1].plot(
        defect_profile_df["x_norm_from_YSZ_interface"],
        defect_profile_df["eps_ch_LSCF_11_gradient"] * 1e3,
        label="LSCF eps_ch_11 gradient",
    )
    axes[1].plot(
        defect_profile_df["x_norm_from_YSZ_interface"],
        defect_profile_df["eps_ch_LSCF_33_gradient"] * 1e3,
        label="LSCF eps_ch_33 gradient",
    )
    axes[1].plot(
        defect_profile_df["x_norm_from_YSZ_interface"],
        defect_profile_df["eps_ch_GDC_iso_gradient"] * 1e3,
        label="GDC eps_ch_iso gradient",
    )
    axes[1].set_xlabel("Normalized distance from YSZ interface")
    axes[1].set_ylabel("Chemical strain (x1e-3)")
    axes[1].set_title("Chemical eigenstrain components")
    axes[1].legend()
    fig.suptitle("Figure 3: Defect-chemical expansion driver", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_03_chemical_expansion.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    eta = 2.1
    mode_mix = np.linspace(0.0, 1.0, 200)
    for _, row in fracture_df[fracture_df["entity_class"] == "interface"].iterrows():
        gc_eff = row["Gc_I_J_m2"] + (row["Gc_II_J_m2"] - row["Gc_I_J_m2"]) * np.power(mode_mix, eta)
        axes[0].plot(mode_mix, gc_eff, label=row["entity"])
    axes[0].set_xlabel("Mode mix ratio GII/(GI+GII)")
    axes[0].set_ylabel("Effective toughness Gc,eff (J/m2)")
    axes[0].set_title("BK mixed-mode envelope")
    axes[0].legend()

    for interface in cohesive_df["interface"].unique():
        sub = cohesive_df[cohesive_df["interface"] == interface]
        axes[1].plot(sub["delta_n_um"], sub["traction_n_MPa"], label=interface)
    axes[1].set_xlabel("Normal separation delta_n (um)")
    axes[1].set_ylabel("Normal traction Tn (MPa)")
    axes[1].set_title("Xu-Needleman style traction-separation")
    axes[1].legend()
    fig.suptitle("Figure 4: Cohesive and mixed-mode fracture laws", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_04_cohesive_laws.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    axes[0].plot(curv_df["temperature_C"], curv_df["kappa_model_1_per_m"], label="Model")
    axes[0].plot(curv_df["temperature_C"], curv_df["kappa_target_1_per_m"], label="Target")
    axes[0].fill_between(
        curv_df["temperature_C"],
        curv_df["kappa_target_1_per_m"] - curv_df["kappa_tolerance_1_per_m"],
        curv_df["kappa_target_1_per_m"] + curv_df["kappa_tolerance_1_per_m"],
        alpha=0.2,
        label="Tolerance band",
    )
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Curvature kappa (1/m)")
    axes[0].set_title("Global curvature evolution")
    axes[0].legend()

    for interface in onset_df["interface"].unique():
        sub = onset_df[onset_df["interface"] == interface].sort_values("pO2_atm", ascending=False)
        axes[1].plot(np.log10(sub["pO2_atm"]), sub["critical_temperature_C"], marker="o", label=interface)
    axes[1].set_xlabel("log10(pO2 [atm])")
    axes[1].set_ylabel("Critical onset temperature (C)")
    axes[1].set_title("Delamination onset map")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].legend()
    fig.suptitle("Figure 5: Experimental target functions", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_05_validation_targets.png")
    plt.close(fig)

    rng = np.random.default_rng(RNG_SEED)
    field = rng.normal(size=(160, 240))
    quant = np.quantile(field, [0.32, 0.52, 0.78])
    phase_map = np.zeros_like(field, dtype=int)
    phase_map[field > quant[0]] = 1
    phase_map[field > quant[1]] = 2
    phase_map[field > quant[2]] = 3
    cmap = plt.matplotlib.colors.ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    labels = {0: "YSZ", 1: "GDC", 2: "LSCF", 3: "Pore"}

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=170)
    im = ax.imshow(phase_map, cmap=cmap, origin="lower", interpolation="nearest")
    ax.set_title("Figure 6: Synthetic phase map for RVE mesh-objectivity studies")
    ax.set_xlabel("Voxel i")
    ax.set_ylabel("Voxel j")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([labels[i] for i in [0, 1, 2, 3]])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_06_synthetic_rve_phase_map.png")
    plt.close(fig)


def write_manifest(file_map: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in sorted(file_map.items()):
        rows.append(
            {
                "artifact_name": name,
                "relative_path": str(path.relative_to(OUT_DIR)),
                "bytes": path.stat().st_size if path.exists() else np.nan,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(CSV_DIR / "00_dataset_manifest.csv", index=False)
    return manifest


def zip_csvs() -> Path:
    zip_path = OUT_DIR / "ysz_gdc_lscf_csv_bundle.zip"
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_file in csv_files:
            zf.write(csv_file, arcname=csv_file.name)
    return zip_path


def write_metadata(zip_path: Path, qa_df: pd.DataFrame) -> None:
    metadata = {
        "topic": TOPIC,
        "seed": RNG_SEED,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "csv_directory": str(CSV_DIR.relative_to(ROOT)),
        "figures_directory": str(FIG_DIR.relative_to(ROOT)),
        "csv_zip": str(zip_path.relative_to(ROOT)),
        "qa_status": "PASS" if (qa_df["result"] == "PASS").all() else "FAIL",
        "disclaimer": (
            "This is a fabricated, literature-anchored calibration dataset for workflow development. "
            "Replace fabricated values with experimentally measured values before publication."
        ),
    }
    with open(META_DIR / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme_text = f"""# Fabricated YSZ/GDC/LSCF Abaqus Input Dataset

Topic: {TOPIC}

## What this package contains
- CSV tables for geometry, thermo-elastic UMAT inputs, chemical expansion, fracture/cohesive UEL inputs, and validation targets.
- A parametric sweep matrix and QA report.
- PNG figures for rapid reporting and model sanity checks.
- A zip archive (`ysz_gdc_lscf_csv_bundle.zip`) that contains all CSV files.

## Important note
This package is intentionally fabricated but literature-anchored for reproducible pipeline setup.
Use it for model plumbing, calibration workflow testing, and mesh-objectivity studies.
Before external publication, replace fabricated values with experimentally validated values.
"""
    with open(OUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_text)


def main() -> None:
    np.random.seed(RNG_SEED)
    ensure_directories()

    geom_df, phase_df = build_geometric_microstructural()
    thermo_df = build_thermoelastic()
    defect_df, defect_profile_df = build_defect_chemical()
    fracture_df, cohesive_df = build_fracture_data()
    curv_df, onset_df, crack_df = build_validation_targets()
    sweep_df = build_parametric_sweep_matrix()
    umat_df, uel_df = build_abaqus_blocks(thermo_df, fracture_df)
    qa_df = run_qa_checks(thermo_df, fracture_df, geom_df)

    file_map: dict[str, Path] = {}

    out_map = {
        "01_geometric_microstructural_data.csv": geom_df,
        "01b_phase_distribution_statistics.csv": phase_df,
        "02_thermoelastic_continuum_data.csv": thermo_df,
        "03_defect_chemical_expansion_data.csv": defect_df,
        "03b_oxygen_nonstoichiometry_profiles.csv": defect_profile_df,
        "04_fracture_cohesive_zone_data.csv": fracture_df,
        "04b_cohesive_traction_separation_samples.csv": cohesive_df,
        "05a_validation_global_curvature.csv": curv_df,
        "05b_validation_delamination_onset.csv": onset_df,
        "05c_validation_crack_path_morphology.csv": crack_df,
        "06_parametric_sweep_matrix.csv": sweep_df,
        "07_abaqus_umat_material_blocks.csv": umat_df,
        "08_abaqus_uel_property_blocks.csv": uel_df,
        "09_pre_simulation_qa_report.csv": qa_df,
    }
    for file_name, df in out_map.items():
        path = CSV_DIR / file_name
        df.to_csv(path, index=False)
        file_map[file_name] = path

    create_figures(geom_df, thermo_df, defect_profile_df, fracture_df, cohesive_df, curv_df, onset_df)
    for fig_path in sorted(FIG_DIR.glob("*.png")):
        file_map[fig_path.name] = fig_path

    manifest = write_manifest(file_map)
    file_map["00_dataset_manifest.csv"] = CSV_DIR / "00_dataset_manifest.csv"
    zip_path = zip_csvs()
    file_map["ysz_gdc_lscf_csv_bundle.zip"] = zip_path

    write_metadata(zip_path, qa_df)

    print("Dataset generation complete.")
    print(f"Output directory: {OUT_DIR}")
    print(f"CSV files: {len(list(CSV_DIR.glob('*.csv')))}")
    print(f"Figures: {len(list(FIG_DIR.glob('*.png')))}")
    print(f"CSV zip: {zip_path}")
    print(f"QA overall: {'PASS' if (qa_df['result'] == 'PASS').all() else 'FAIL'}")
    print(f"Manifest rows: {len(manifest)}")


if __name__ == "__main__":
    main()
