#!/usr/bin/env python3
"""
Master Dataset Generator for:
"Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in SOCs"

Generates all synthetic CSV files, the analysis script, and publication-quality figures.

Author: Auto-generated
Date: 2026-02-11
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm, lognorm
from numpy.random import default_rng

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
N_SAMPLES = 10000  # Monte Carlo samples for stochastic inputs
N_MICRO = 50       # Number of micro-cantilever test specimens
CSV_DIR = "csv"
FIG_DIR = "figures"

rng = default_rng(SEED)

# Physical constants / baseline values (room temperature)
# Interfacial fracture toughness (J/m^2)
GC_YSZ_GDC_MEAN_RT = 3.8    # YSZ|GDC interface
GC_YSZ_GDC_STD_RT = 0.65    # Standard deviation

GC_GDC_LSCF_MEAN_RT = 2.5   # GDC|LSCF interface
GC_GDC_LSCF_STD_RT = 0.50

# High-temperature scaling factor (800°C)
HT_SCALE_FACTOR = 0.72       # Mean shifts down (softening)
# Note: real sigma_HT is UNKNOWN - we keep it identical (the dangerous assumption)

# Elastic properties
E_YSZ = 210.0   # GPa
E_GDC = 190.0   # GPa
E_LSCF = 120.0  # GPa
E_NiYSZ = 95.0  # GPa (anode cermet)

NU_YSZ = 0.30
NU_GDC = 0.28
NU_LSCF = 0.25
NU_NiYSZ = 0.32

# CTE (ppm/K)
CTE_YSZ = 10.5
CTE_GDC = 12.5
CTE_LSCF = 15.5
CTE_NiYSZ = 12.2

# Layer thicknesses (µm)
T_YSZ = 10.0
T_GDC = 5.0
T_LSCF = 30.0
T_NiYSZ = 500.0

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
# DATASET 01: 04_uncertainty_material_properties.csv
# ============================================================================
def generate_material_properties():
    """Generate material property table with uncertainty bounds."""
    data = {
        'Property': [
            'E_YSZ', 'E_GDC', 'E_LSCF', 'E_NiYSZ',
            'nu_YSZ', 'nu_GDC', 'nu_LSCF', 'nu_NiYSZ',
            'CTE_YSZ', 'CTE_GDC', 'CTE_LSCF', 'CTE_NiYSZ',
            'Gc_YSZ_GDC', 'Gc_GDC_LSCF',
            't_YSZ', 't_GDC', 't_LSCF', 't_NiYSZ',
            'Sintering_Temp', 'Operating_Temp'
        ],
        'Symbol': [
            'E_YSZ', 'E_GDC', 'E_LSCF', 'E_NiYSZ',
            'ν_YSZ', 'ν_GDC', 'ν_LSCF', 'ν_NiYSZ',
            'α_YSZ', 'α_GDC', 'α_LSCF', 'α_NiYSZ',
            'Gc_YSZ|GDC', 'Gc_GDC|LSCF',
            't_YSZ', 't_GDC', 't_LSCF', 't_NiYSZ',
            'T_sinter', 'T_op'
        ],
        'Mean_Value': [
            E_YSZ, E_GDC, E_LSCF, E_NiYSZ,
            NU_YSZ, NU_GDC, NU_LSCF, NU_NiYSZ,
            CTE_YSZ, CTE_GDC, CTE_LSCF, CTE_NiYSZ,
            GC_YSZ_GDC_MEAN_RT, GC_GDC_LSCF_MEAN_RT,
            T_YSZ, T_GDC, T_LSCF, T_NiYSZ,
            1350, 800
        ],
        'Std_Dev': [
            12.0, 10.0, 8.5, 7.0,
            0.02, 0.02, 0.02, 0.02,
            0.3, 0.4, 0.6, 0.5,
            GC_YSZ_GDC_STD_RT, GC_GDC_LSCF_STD_RT,
            1.0, 0.5, 3.0, 25.0,
            10, 5
        ],
        'Unit': [
            'GPa', 'GPa', 'GPa', 'GPa',
            '-', '-', '-', '-',
            'ppm/K', 'ppm/K', 'ppm/K', 'ppm/K',
            'J/m²', 'J/m²',
            'µm', 'µm', 'µm', 'µm',
            '°C', '°C'
        ],
        'Distribution': [
            'Normal', 'Normal', 'Normal', 'Normal',
            'Normal', 'Normal', 'Normal', 'Normal',
            'Normal', 'Normal', 'Normal', 'Normal',
            'Lognormal', 'Lognormal',
            'Normal', 'Normal', 'Normal', 'Normal',
            'Deterministic', 'Deterministic'
        ],
        'Source': [
            'Literature + Nanoindentation', 'Literature + Nanoindentation',
            'Literature + Nanoindentation', 'Literature + Nanoindentation',
            'Literature', 'Literature', 'Literature', 'Literature',
            'Dilatometry', 'Dilatometry', 'Dilatometry', 'Dilatometry',
            'Micro-cantilever (RT)', 'Micro-cantilever (RT)',
            'SEM cross-section', 'SEM cross-section', 'SEM cross-section', 'SEM cross-section',
            'Process spec', 'Operating spec'
        ],
        'CoV_percent': [
            5.7, 5.3, 7.1, 7.4,
            6.7, 7.1, 8.0, 6.3,
            2.9, 3.2, 3.9, 4.1,
            17.1, 20.0,
            10.0, 10.0, 10.0, 5.0,
            0.7, 0.6
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(CSV_DIR, '04_uncertainty_material_properties.csv'), index=False)
    print("[OK] 04_uncertainty_material_properties.csv")
    return df


# ============================================================================
# DATASET 02: 16_micro_cantilever_fracture_data.csv
# ============================================================================
def generate_micro_cantilever_data():
    """Generate synthetic micro-cantilever fracture toughness test data."""
    specimen_ids = [f"MC-{i+1:03d}" for i in range(N_MICRO)]
    interfaces = rng.choice(['YSZ_GDC', 'GDC_LSCF'], size=N_MICRO)

    gc_values = []
    beam_widths = []
    beam_lengths = []
    notch_depths = []
    peak_loads = []
    test_temps = []

    for iface in interfaces:
        if iface == 'YSZ_GDC':
            gc = rng.lognormal(
                np.log(GC_YSZ_GDC_MEAN_RT) - 0.5 * (GC_YSZ_GDC_STD_RT / GC_YSZ_GDC_MEAN_RT)**2,
                GC_YSZ_GDC_STD_RT / GC_YSZ_GDC_MEAN_RT
            )
        else:
            gc = rng.lognormal(
                np.log(GC_GDC_LSCF_MEAN_RT) - 0.5 * (GC_GDC_LSCF_STD_RT / GC_GDC_LSCF_MEAN_RT)**2,
                GC_GDC_LSCF_STD_RT / GC_GDC_LSCF_MEAN_RT
            )

        bw = rng.normal(3.0, 0.2)   # µm
        bl = rng.normal(15.0, 0.5)  # µm
        nd = rng.normal(1.5, 0.1)   # µm
        # Approximate peak load from Gc using beam theory
        pload = gc * bw * 1e-6 * 1e3 + rng.normal(0, 0.05)  # mN (with noise)

        gc_values.append(round(gc, 4))
        beam_widths.append(round(bw, 2))
        beam_lengths.append(round(bl, 2))
        notch_depths.append(round(nd, 2))
        peak_loads.append(round(max(pload, 0.01), 4))
        test_temps.append(25)  # Room temperature only

    df = pd.DataFrame({
        'Specimen_ID': specimen_ids,
        'Interface': interfaces,
        'Gc_Jm2': gc_values,
        'Beam_Width_um': beam_widths,
        'Beam_Length_um': beam_lengths,
        'Notch_Depth_um': notch_depths,
        'Peak_Load_mN': peak_loads,
        'Test_Temperature_C': test_temps,
        'Test_Method': ['FIB-notched cantilever'] * N_MICRO,
        'Valid_Test': rng.choice([True, True, True, True, False], size=N_MICRO)
    })
    df.to_csv(os.path.join(CSV_DIR, '16_micro_cantilever_fracture_data.csv'), index=False)
    print("[OK] 16_micro_cantilever_fracture_data.csv")
    return df


# ============================================================================
# DATASET 03: 13_LSCF_ferroelastic_stress_strain.csv
# ============================================================================
def generate_lscf_ferroelastic():
    """Generate LSCF ferroelastic stress-strain data at multiple temperatures."""
    temps = [25, 200, 400, 600, 800]
    rows = []
    for T in temps:
        # LSCF exhibits ferroelastic switching: nonlinear stress-strain
        sigma_switch = 80 - 0.05 * T  # Switching stress decreases with T
        strain_max = 0.008 + 0.001 * (T / 800)

        n_points = 200
        strain = np.linspace(0, strain_max, n_points)
        # Piecewise model: linear + ferroelastic plateau + hardening
        E_eff = E_LSCF * (1 - 0.15 * T / 800)  # Softening with temperature (GPa)
        stress = np.zeros(n_points)
        for i, eps in enumerate(strain):
            sigma_lin = E_eff * 1e3 * eps  # MPa
            if sigma_lin < sigma_switch:
                stress[i] = sigma_lin
            else:
                # Ferroelastic switching: reduced tangent modulus
                excess = eps - sigma_switch / (E_eff * 1e3)
                stress[i] = sigma_switch + 0.3 * E_eff * 1e3 * excess

        for i in range(n_points):
            rows.append({
                'Temperature_C': T,
                'Strain': round(strain[i], 6),
                'Stress_MPa': round(stress[i], 2),
                'E_tangent_GPa': round(E_eff if strain[i] < sigma_switch / (E_eff * 1e3) else 0.3 * E_eff, 2),
                'Phase': 'linear' if stress[i] < sigma_switch else 'ferroelastic'
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, '13_LSCF_ferroelastic_stress_strain.csv'), index=False)
    print("[OK] 13_LSCF_ferroelastic_stress_strain.csv")
    return df


# ============================================================================
# DATASET 04: 21_missing_data_assumption_flags.csv
# ============================================================================
def generate_missing_data_flags():
    """Generate the missing data assumption flags file."""
    data = {
        'Flag_ID': [
            'MISSING_DATASET_01',
            'MISSING_DATASET_02',
            'MISSING_DATASET_03',
            'MISSING_DATASET_04',
            'MISSING_DATASET_05'
        ],
        'Missing_Quantity': [
            'Interfacial Failure Correlation Coefficient (rho_anode_cathode)',
            'High-Temperature (800C) Fracture Toughness PDF',
            'Cyclic Fatigue Crack Growth Rate (da/dN) at Interface',
            'Residual Stress Spatial Map After Sintering',
            'Porosity-Dependent Fracture Toughness Relationship'
        ],
        'Symbol': [
            'rho(Gc_YSZ_GDC, Gc_GDC_LSCF)',
            'f_Gc(Gc | T=800C)',
            'da/dN = C*(DeltaK)^m',
            'sigma_res(x,y,z)',
            'Gc(phi) where phi=porosity'
        ],
        'True_Value': [
            'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'
        ],
        'Surrogate_Value': [
            'rho=0.0 (baseline), rho=0.5 (sensitivity)',
            'Gc_HT = 0.72 * Gc_RT (mean shift only, sigma unchanged)',
            'Not modeled - static fracture only assumed',
            'Uniform residual stress from analytical cool-down model',
            'Gc assumed independent of porosity (phi=0.30 for anode)'
        ],
        'Surrogate_Source_File': [
            'stochastic_inputs_rho0.00.csv / stochastic_inputs_rho0.50.csv',
            'stochastic_inputs_HT_uncorrelated.csv',
            'N/A - Not included in current framework',
            '04_uncertainty_material_properties.csv (CTE mismatch)',
            '04_uncertainty_material_properties.csv'
        ],
        'Impact_If_Wrong': [
            'Can underpredict joint interface failure probability by up to 2x for rho>0',
            'Misestimated spread can distort failure probability map tails',
            'Ignores fatigue degradation under thermal cycling - non-conservative for long-term',
            'Local stress concentrations near edges/defects are smoothed out',
            'Porous anode may have 30-50% lower toughness than assumed'
        ],
        'Risk_Level': [
            'HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW'
        ],
        'Mitigation_Strategy': [
            'Run sensitivity analysis at rho=0.0, 0.25, 0.50, 0.75, 1.0',
            'Perform in-situ 800C micro-cantilever tests (5+ samples)',
            'Plan future thermal cycling test campaign',
            'Perform XRD residual stress measurements on cross-sections',
            'Measure Gc on samples with controlled porosity variation'
        ],
        'Date_Flagged': [
            '2026-01-15', '2026-01-15', '2026-01-20', '2026-02-01', '2026-02-05'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(CSV_DIR, '21_missing_data_assumption_flags.csv'), index=False)
    print("[OK] 21_missing_data_assumption_flags.csv")
    return df


# ============================================================================
# HELPER: Generate correlated lognormal samples
# ============================================================================
def generate_correlated_lognormal(mu1, sig1, mu2, sig2, rho, n, rng_inst):
    """
    Generate correlated lognormal samples using Gaussian copula.

    Parameters:
        mu1, sig1: Mean and std of first lognormal (in real space)
        mu2, sig2: Mean and std of second lognormal (in real space)
        rho: Desired Pearson correlation in Gaussian (copula) space
        n: Number of samples
    """
    # Lognormal parameters
    sigma1_ln = np.sqrt(np.log(1 + (sig1 / mu1)**2))
    mu1_ln = np.log(mu1) - 0.5 * sigma1_ln**2

    sigma2_ln = np.sqrt(np.log(1 + (sig2 / mu2)**2))
    mu2_ln = np.log(mu2) - 0.5 * sigma2_ln**2

    # Correlated normal samples
    cov_matrix = [[1.0, rho], [rho, 1.0]]
    z = rng_inst.multivariate_normal([0, 0], cov_matrix, size=n)

    # Transform to lognormal
    x1 = np.exp(mu1_ln + sigma1_ln * z[:, 0])
    x2 = np.exp(mu2_ln + sigma2_ln * z[:, 1])

    return x1, x2


# ============================================================================
# DATASET 05: stochastic_inputs_rho0.00.csv (Independent)
# ============================================================================
def generate_stochastic_rho000():
    """Generate stochastic inputs with rho=0.0 (independent interfaces)."""
    gc1, gc2 = generate_correlated_lognormal(
        GC_YSZ_GDC_MEAN_RT, GC_YSZ_GDC_STD_RT,
        GC_GDC_LSCF_MEAN_RT, GC_GDC_LSCF_STD_RT,
        rho=0.0, n=N_SAMPLES, rng_inst=rng
    )

    # Also add uncertainty on elastic properties and CTE
    E_ysz_s = rng.normal(E_YSZ, 12.0, N_SAMPLES)
    E_gdc_s = rng.normal(E_GDC, 10.0, N_SAMPLES)
    E_lscf_s = rng.normal(E_LSCF, 8.5, N_SAMPLES)
    CTE_ysz_s = rng.normal(CTE_YSZ, 0.3, N_SAMPLES)
    CTE_gdc_s = rng.normal(CTE_GDC, 0.4, N_SAMPLES)
    CTE_lscf_s = rng.normal(CTE_LSCF, 0.6, N_SAMPLES)
    delta_T = rng.normal(1350 - 800, 10, N_SAMPLES)  # Cool-down ΔT

    # Analytical thermal mismatch stress (simplified Stoney eq.)
    sigma_mismatch_12 = E_ysz_s * 1e3 * (CTE_gdc_s - CTE_ysz_s) * 1e-6 * delta_T  # MPa
    sigma_mismatch_23 = E_gdc_s * 1e3 * (CTE_lscf_s - CTE_gdc_s) * 1e-6 * delta_T  # MPa

    # Approximate ERR from thermal mismatch (Hutchinson & Suo)
    G_thermal_12 = (sigma_mismatch_12**2 * T_GDC * 1e-6) / (2 * E_ysz_s * 1e3)  # J/m²
    G_thermal_23 = (sigma_mismatch_23**2 * T_LSCF * 1e-6) / (2 * E_gdc_s * 1e3)  # J/m²

    df = pd.DataFrame({
        'Sample_ID': np.arange(1, N_SAMPLES + 1),
        'Gc_YSZ_GDC': np.round(gc1, 4),
        'Gc_GDC_LSCF': np.round(gc2, 4),
        'E_YSZ_GPa': np.round(E_ysz_s, 2),
        'E_GDC_GPa': np.round(E_gdc_s, 2),
        'E_LSCF_GPa': np.round(E_lscf_s, 2),
        'CTE_YSZ_ppmK': np.round(CTE_ysz_s, 3),
        'CTE_GDC_ppmK': np.round(CTE_gdc_s, 3),
        'CTE_LSCF_ppmK': np.round(CTE_lscf_s, 3),
        'Delta_T_C': np.round(delta_T, 1),
        'Sigma_mismatch_YSZ_GDC_MPa': np.round(sigma_mismatch_12, 2),
        'Sigma_mismatch_GDC_LSCF_MPa': np.round(sigma_mismatch_23, 2),
        'G_thermal_YSZ_GDC_Jm2': np.round(G_thermal_12, 4),
        'G_thermal_GDC_LSCF_Jm2': np.round(G_thermal_23, 4),
        'Rho_assumed': 0.0,
        'Temperature_regime': 'RT'
    })
    df.to_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'), index=False)
    print("[OK] stochastic_inputs_rho0.00.csv")
    return df


# ============================================================================
# DATASET 06: stochastic_inputs_rho0.50.csv (Correlated)
# ============================================================================
def generate_stochastic_rho050():
    """Generate stochastic inputs with rho=0.5 (correlated interfaces)."""
    rng2 = default_rng(SEED + 1)  # Different seed for different realization

    gc1, gc2 = generate_correlated_lognormal(
        GC_YSZ_GDC_MEAN_RT, GC_YSZ_GDC_STD_RT,
        GC_GDC_LSCF_MEAN_RT, GC_GDC_LSCF_STD_RT,
        rho=0.5, n=N_SAMPLES, rng_inst=rng2
    )

    E_ysz_s = rng2.normal(E_YSZ, 12.0, N_SAMPLES)
    E_gdc_s = rng2.normal(E_GDC, 10.0, N_SAMPLES)
    E_lscf_s = rng2.normal(E_LSCF, 8.5, N_SAMPLES)
    CTE_ysz_s = rng2.normal(CTE_YSZ, 0.3, N_SAMPLES)
    CTE_gdc_s = rng2.normal(CTE_GDC, 0.4, N_SAMPLES)
    CTE_lscf_s = rng2.normal(CTE_LSCF, 0.6, N_SAMPLES)
    delta_T = rng2.normal(1350 - 800, 10, N_SAMPLES)

    sigma_mismatch_12 = E_ysz_s * 1e3 * (CTE_gdc_s - CTE_ysz_s) * 1e-6 * delta_T
    sigma_mismatch_23 = E_gdc_s * 1e3 * (CTE_lscf_s - CTE_gdc_s) * 1e-6 * delta_T

    G_thermal_12 = (sigma_mismatch_12**2 * T_GDC * 1e-6) / (2 * E_ysz_s * 1e3)
    G_thermal_23 = (sigma_mismatch_23**2 * T_LSCF * 1e-6) / (2 * E_gdc_s * 1e3)

    df = pd.DataFrame({
        'Sample_ID': np.arange(1, N_SAMPLES + 1),
        'Gc_YSZ_GDC': np.round(gc1, 4),
        'Gc_GDC_LSCF': np.round(gc2, 4),
        'E_YSZ_GPa': np.round(E_ysz_s, 2),
        'E_GDC_GPa': np.round(E_gdc_s, 2),
        'E_LSCF_GPa': np.round(E_lscf_s, 2),
        'CTE_YSZ_ppmK': np.round(CTE_ysz_s, 3),
        'CTE_GDC_ppmK': np.round(CTE_gdc_s, 3),
        'CTE_LSCF_ppmK': np.round(CTE_lscf_s, 3),
        'Delta_T_C': np.round(delta_T, 1),
        'Sigma_mismatch_YSZ_GDC_MPa': np.round(sigma_mismatch_12, 2),
        'Sigma_mismatch_GDC_LSCF_MPa': np.round(sigma_mismatch_23, 2),
        'G_thermal_YSZ_GDC_Jm2': np.round(G_thermal_12, 4),
        'G_thermal_GDC_LSCF_Jm2': np.round(G_thermal_23, 4),
        'Rho_assumed': 0.5,
        'Temperature_regime': 'RT'
    })
    df.to_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.50.csv'), index=False)
    print("[OK] stochastic_inputs_rho0.50.csv")
    return df


# ============================================================================
# DATASET 07: stochastic_inputs_HT_uncorrelated.csv (High-T, rho=0)
# ============================================================================
def generate_stochastic_HT():
    """Generate high-temperature stochastic inputs with deterministic mean scaling."""
    rng3 = default_rng(SEED + 2)

    # HIGH-TEMPERATURE: Mean scaled by HT_SCALE_FACTOR, StdDev kept the same (the dangerous assumption!)
    gc1_ht_mean = GC_YSZ_GDC_MEAN_RT * HT_SCALE_FACTOR
    gc2_ht_mean = GC_GDC_LSCF_MEAN_RT * HT_SCALE_FACTOR
    gc1_ht_std = GC_YSZ_GDC_STD_RT   # UNCHANGED - this is the missing data gap
    gc2_ht_std = GC_GDC_LSCF_STD_RT   # UNCHANGED

    gc1, gc2 = generate_correlated_lognormal(
        gc1_ht_mean, gc1_ht_std,
        gc2_ht_mean, gc2_ht_std,
        rho=0.0, n=N_SAMPLES, rng_inst=rng3
    )

    # Temperature-dependent elastic moduli (soften at 800°C)
    E_ysz_s = rng3.normal(E_YSZ * 0.90, 12.0, N_SAMPLES)
    E_gdc_s = rng3.normal(E_GDC * 0.88, 10.0, N_SAMPLES)
    E_lscf_s = rng3.normal(E_LSCF * 0.80, 8.5, N_SAMPLES)
    CTE_ysz_s = rng3.normal(CTE_YSZ * 1.05, 0.3, N_SAMPLES)
    CTE_gdc_s = rng3.normal(CTE_GDC * 1.05, 0.4, N_SAMPLES)
    CTE_lscf_s = rng3.normal(CTE_LSCF * 1.08, 0.6, N_SAMPLES)
    delta_T = rng3.normal(1350 - 800, 10, N_SAMPLES)

    sigma_mismatch_12 = E_ysz_s * 1e3 * (CTE_gdc_s - CTE_ysz_s) * 1e-6 * delta_T
    sigma_mismatch_23 = E_gdc_s * 1e3 * (CTE_lscf_s - CTE_gdc_s) * 1e-6 * delta_T

    G_thermal_12 = (sigma_mismatch_12**2 * T_GDC * 1e-6) / (2 * E_ysz_s * 1e3)
    G_thermal_23 = (sigma_mismatch_23**2 * T_LSCF * 1e-6) / (2 * E_gdc_s * 1e3)

    df = pd.DataFrame({
        'Sample_ID': np.arange(1, N_SAMPLES + 1),
        'Gc_YSZ_GDC': np.round(gc1, 4),
        'Gc_GDC_LSCF': np.round(gc2, 4),
        'Gc_YSZ_GDC_800C': np.round(gc1, 4),  # explicit 800C column
        'Gc_GDC_LSCF_800C': np.round(gc2, 4),
        'E_YSZ_GPa': np.round(E_ysz_s, 2),
        'E_GDC_GPa': np.round(E_gdc_s, 2),
        'E_LSCF_GPa': np.round(E_lscf_s, 2),
        'CTE_YSZ_ppmK': np.round(CTE_ysz_s, 3),
        'CTE_GDC_ppmK': np.round(CTE_gdc_s, 3),
        'CTE_LSCF_ppmK': np.round(CTE_lscf_s, 3),
        'Delta_T_C': np.round(delta_T, 1),
        'Sigma_mismatch_YSZ_GDC_MPa': np.round(sigma_mismatch_12, 2),
        'Sigma_mismatch_GDC_LSCF_MPa': np.round(sigma_mismatch_23, 2),
        'G_thermal_YSZ_GDC_Jm2': np.round(G_thermal_12, 4),
        'G_thermal_GDC_LSCF_Jm2': np.round(G_thermal_23, 4),
        'Rho_assumed': 0.0,
        'Temperature_regime': '800C',
        'HT_scaling_factor': HT_SCALE_FACTOR,
        'sigma_HT_assumed_equal_RT': True
    })
    df.to_csv(os.path.join(CSV_DIR, 'stochastic_inputs_HT_uncorrelated.csv'), index=False)
    print("[OK] stochastic_inputs_HT_uncorrelated.csv")
    return df


# ============================================================================
# DATASET 08: Additional sensitivity rho values
# ============================================================================
def generate_stochastic_rho_sweep():
    """Generate stochastic inputs for a sweep of rho values for comprehensive sensitivity."""
    rho_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    all_rows = []
    for rho_val in rho_values:
        rng_sw = default_rng(SEED + int(rho_val * 100))
        gc1, gc2 = generate_correlated_lognormal(
            GC_YSZ_GDC_MEAN_RT, GC_YSZ_GDC_STD_RT,
            GC_GDC_LSCF_MEAN_RT, GC_GDC_LSCF_STD_RT,
            rho=rho_val, n=N_SAMPLES, rng_inst=rng_sw
        )
        for i in range(N_SAMPLES):
            all_rows.append({
                'Sample_ID': i + 1,
                'Rho': rho_val,
                'Gc_YSZ_GDC': round(gc1[i], 4),
                'Gc_GDC_LSCF': round(gc2[i], 4),
                'R_system': round(gc1[i] + gc2[i], 4)
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho_sweep.csv'), index=False)
    print("[OK] stochastic_inputs_rho_sweep.csv")
    return df


# ============================================================================
# DATASET 09: Failure probability summary table
# ============================================================================
def generate_failure_probability_summary():
    """Generate a summary table of failure probabilities across scenarios."""
    J_app_values = np.arange(1.0, 8.5, 0.25)
    scenarios = {
        'rho0.00_RT': ('stochastic_inputs_rho0.00.csv', 0.0, 'RT'),
        'rho0.50_RT': ('stochastic_inputs_rho0.50.csv', 0.5, 'RT'),
        'rho0.00_HT': ('stochastic_inputs_HT_uncorrelated.csv', 0.0, '800C'),
    }

    rows = []
    for scenario_name, (fname, rho, temp) in scenarios.items():
        df_s = pd.read_csv(os.path.join(CSV_DIR, fname))
        R_system = df_s['Gc_YSZ_GDC'] + df_s['Gc_GDC_LSCF']

        for J in J_app_values:
            pf = np.mean(R_system < J)
            pf_lower = max(0, pf - 1.96 * np.sqrt(pf * (1 - pf) / len(R_system)))
            pf_upper = min(1, pf + 1.96 * np.sqrt(pf * (1 - pf) / len(R_system)))
            rows.append({
                'Scenario': scenario_name,
                'Rho': rho,
                'Temperature': temp,
                'J_applied_Jm2': round(J, 2),
                'P_failure': round(pf, 6),
                'P_failure_95CI_lower': round(pf_lower, 6),
                'P_failure_95CI_upper': round(pf_upper, 6),
                'N_samples': len(R_system)
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CSV_DIR, 'failure_probability_summary.csv'), index=False)
    print("[OK] failure_probability_summary.csv")
    return df


# ============================================================================
# FIGURE 1: The "Cone of Ignorance" - Fragility Curves
# ============================================================================
def plot_cone_of_ignorance():
    """Plot the main fragility curves showing epistemic uncertainty gap."""
    df_indep = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'))
    df_corr = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.50.csv'))
    df_ht = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_HT_uncorrelated.csv'))

    R_indep = df_indep['Gc_YSZ_GDC'] + df_indep['Gc_GDC_LSCF']
    R_corr = df_corr['Gc_YSZ_GDC'] + df_corr['Gc_GDC_LSCF']
    R_ht = df_ht['Gc_YSZ_GDC'] + df_ht['Gc_GDC_LSCF']

    J_range = np.linspace(0.5, 12, 500)

    Pf_indep = [np.mean(R_indep < J) for J in J_range]
    Pf_corr = [np.mean(R_corr < J) for J in J_range]
    Pf_ht = [np.mean(R_ht < J) for J in J_range]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Fill between curves 1 and 2 (epistemic gap from correlation)
    ax.fill_between(J_range, Pf_indep, Pf_corr, alpha=0.25, color='#2196F3',
                     label='Unquantified Risk (Missing Dataset 01: ρ unknown)')
    # Fill between curve 1 and 3 (epistemic gap from temperature)
    ax.fill_between(J_range, Pf_indep, Pf_ht, alpha=0.15, color='#FF5722',
                     label='Unquantified Risk (Missing Dataset 02: HT data)')

    ax.plot(J_range, Pf_indep, 'b-', linewidth=2.5,
            label='Scenario A: ρ = 0.0, RT (Independent)')
    ax.plot(J_range, Pf_corr, 'b--', linewidth=2.5,
            label='Scenario B: ρ = 0.5, RT (Correlated)')
    ax.plot(J_range, Pf_ht, 'r-', linewidth=2.5,
            label='Scenario C: ρ = 0.0, 800°C (High-T)')

    # Reference line at Pf = 0.01 (1% failure)
    ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(0.8, 0.013, 'P_fail = 1% threshold', fontsize=9, color='gray')

    # Reference line at typical J_applied
    ax.axvline(x=3.0, color='green', linestyle='-.', linewidth=1.5, alpha=0.5)
    ax.text(3.1, 0.85, 'J_app = 3.0 J/m²\n(typical thermal load)',
            fontsize=9, color='green', rotation=0)

    ax.set_xlabel('Applied Energy Release Rate, $J_{applied}$ (J/m²)', fontsize=14)
    ax.set_ylabel('Probability of System Failure, $P_f$', fontsize=14)
    ax.set_title('The "Cone of Ignorance": Fragility Curves for SOC Interface Failure\n'
                 'Epistemic vs. Aleatory Uncertainty Decomposition', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xlim(0.5, 12)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig01_cone_of_ignorance_fragility_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig01_cone_of_ignorance_fragility_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig01_cone_of_ignorance_fragility_curves.png/pdf")


# ============================================================================
# FIGURE 2: Joint PDF scatter plots showing correlation effect
# ============================================================================
def plot_joint_scatter():
    """Plot joint distribution of Gc at both interfaces for different rho."""
    df_indep = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'))
    df_corr = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.50.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    n_plot = 2000  # Subsample for clarity

    # Panel A: Independent
    ax = axes[0]
    ax.scatter(df_indep['Gc_YSZ_GDC'][:n_plot], df_indep['Gc_GDC_LSCF'][:n_plot],
               alpha=0.15, s=8, c='#2196F3', edgecolors='none')
    # Danger zone: both below threshold
    threshold = 3.0
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    rect = plt.Rectangle((0, 0), threshold, threshold, color='red', alpha=0.08)
    ax.add_patch(rect)
    ax.text(1.0, 0.8, 'Both Fail\n(Catastrophic)', fontsize=10, color='red', fontweight='bold')
    ax.set_xlabel('$G_c$ (YSZ|GDC) [J/m²]', fontsize=12)
    ax.set_ylabel('$G_c$ (GDC|LSCF) [J/m²]', fontsize=12)
    ax.set_title(f'(a) Independent: ρ = 0.0\nPearson r = {np.corrcoef(df_indep["Gc_YSZ_GDC"][:n_plot], df_indep["Gc_GDC_LSCF"][:n_plot])[0,1]:.3f}',
                 fontsize=13)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.2)

    # Panel B: Correlated
    ax = axes[1]
    ax.scatter(df_corr['Gc_YSZ_GDC'][:n_plot], df_corr['Gc_GDC_LSCF'][:n_plot],
               alpha=0.15, s=8, c='#FF9800', edgecolors='none')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    rect2 = plt.Rectangle((0, 0), threshold, threshold, color='red', alpha=0.08)
    ax.add_patch(rect2)
    ax.text(1.0, 0.8, 'Both Fail\n(Catastrophic)', fontsize=10, color='red', fontweight='bold')
    ax.set_xlabel('$G_c$ (YSZ|GDC) [J/m²]', fontsize=12)
    ax.set_ylabel('$G_c$ (GDC|LSCF) [J/m²]', fontsize=12)
    ax.set_title(f'(b) Correlated: ρ = 0.5\nPearson r = {np.corrcoef(df_corr["Gc_YSZ_GDC"][:n_plot], df_corr["Gc_GDC_LSCF"][:n_plot])[0,1]:.3f}',
                 fontsize=13)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Joint Distribution of Interfacial Fracture Toughness\n'
                 'Impact of Unknown Correlation on Catastrophic Failure Region',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig02_joint_scatter_correlation_effect.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig02_joint_scatter_correlation_effect.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig02_joint_scatter_correlation_effect.png/pdf")


# ============================================================================
# FIGURE 3: CDF comparison of system resistance
# ============================================================================
def plot_cdf_comparison():
    """Plot CDFs of system resistance R = Gc1 + Gc2 for all three scenarios."""
    df_indep = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'))
    df_corr = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.50.csv'))
    df_ht = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_HT_uncorrelated.csv'))

    R_indep = np.sort(df_indep['Gc_YSZ_GDC'] + df_indep['Gc_GDC_LSCF'])
    R_corr = np.sort(df_corr['Gc_YSZ_GDC'] + df_corr['Gc_GDC_LSCF'])
    R_ht = np.sort(df_ht['Gc_YSZ_GDC'] + df_ht['Gc_GDC_LSCF'])

    cdf = np.linspace(0, 1, N_SAMPLES)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    ax.plot(R_indep, cdf, 'b-', linewidth=2.5, label='Scenario A: ρ = 0.0, RT')
    ax.plot(R_corr, cdf, 'b--', linewidth=2.5, label='Scenario B: ρ = 0.5, RT')
    ax.plot(R_ht, cdf, 'r-', linewidth=2.5, label='Scenario C: ρ = 0.0, 800°C')

    # Annotate the epistemic gap at CDF = 0.05
    idx_05 = int(0.05 * N_SAMPLES)
    ax.annotate('', xy=(R_indep[idx_05], 0.05), xytext=(R_corr[idx_05], 0.05),
                arrowprops=dict(arrowstyle='<->', color='#2196F3', lw=2))
    mid_x = (R_indep[idx_05] + R_corr[idx_05]) / 2
    ax.text(mid_x, 0.07, f'Δ = {abs(R_indep[idx_05] - R_corr[idx_05]):.2f} J/m²\n(Epistemic: ρ)',
            fontsize=9, ha='center', color='#2196F3', fontweight='bold')

    ax.annotate('', xy=(R_indep[idx_05], 0.05), xytext=(R_ht[idx_05], 0.05),
                arrowprops=dict(arrowstyle='<->', color='#FF5722', lw=2))
    mid_x2 = (R_indep[idx_05] + R_ht[idx_05]) / 2
    ax.text(mid_x2, 0.02, f'Δ = {abs(R_indep[idx_05] - R_ht[idx_05]):.2f} J/m²\n(Epistemic: T)',
            fontsize=9, ha='center', color='#FF5722', fontweight='bold')

    ax.axvline(x=3.0, color='green', linestyle='-.', linewidth=1.5, alpha=0.5)
    ax.text(3.1, 0.9, '$J_{app}$ = 3.0', fontsize=10, color='green')

    ax.set_xlabel('System Resistance, $R = G_c^{(1)} + G_c^{(2)}$ (J/m²)', fontsize=14)
    ax.set_ylabel('Cumulative Distribution Function (CDF)', fontsize=14)
    ax.set_title('CDF of System Resistance: Epistemic Uncertainty Decomposition\n'
                 'Horizontal Distance = Knowledge Gap', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim(1, 14)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig03_cdf_system_resistance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig03_cdf_system_resistance_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig03_cdf_system_resistance_comparison.png/pdf")


# ============================================================================
# FIGURE 4: PDF histograms of individual interface toughness
# ============================================================================
def plot_pdf_histograms():
    """Plot PDF histograms for each interface at RT and HT."""
    df_rt = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'))
    df_ht = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_HT_uncorrelated.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: YSZ|GDC interface
    ax = axes[0]
    ax.hist(df_rt['Gc_YSZ_GDC'], bins=60, density=True, alpha=0.5, color='#2196F3',
            label='RT (measured PDF)', edgecolor='white', linewidth=0.5)
    ax.hist(df_ht['Gc_YSZ_GDC'], bins=60, density=True, alpha=0.5, color='#FF5722',
            label='800°C (surrogate PDF)', edgecolor='white', linewidth=0.5)

    # Overlay fitted lognormal
    x = np.linspace(0.5, 8, 300)
    shape_rt, loc_rt, scale_rt = lognorm.fit(df_rt['Gc_YSZ_GDC'], floc=0)
    ax.plot(x, lognorm.pdf(x, shape_rt, loc_rt, scale_rt), 'b-', linewidth=2)
    shape_ht, loc_ht, scale_ht = lognorm.fit(df_ht['Gc_YSZ_GDC'], floc=0)
    ax.plot(x, lognorm.pdf(x, shape_ht, loc_ht, scale_ht), 'r-', linewidth=2)

    ax.set_xlabel('$G_c$ (YSZ|GDC) [J/m²]', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('(a) YSZ|GDC Interface', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 8)

    # Panel B: GDC|LSCF interface
    ax = axes[1]
    ax.hist(df_rt['Gc_GDC_LSCF'], bins=60, density=True, alpha=0.5, color='#2196F3',
            label='RT (measured PDF)', edgecolor='white', linewidth=0.5)
    ax.hist(df_ht['Gc_GDC_LSCF'], bins=60, density=True, alpha=0.5, color='#FF5722',
            label='800°C (surrogate PDF)', edgecolor='white', linewidth=0.5)

    shape_rt2, loc_rt2, scale_rt2 = lognorm.fit(df_rt['Gc_GDC_LSCF'], floc=0)
    ax.plot(x, lognorm.pdf(x, shape_rt2, loc_rt2, scale_rt2), 'b-', linewidth=2)
    shape_ht2, loc_ht2, scale_ht2 = lognorm.fit(df_ht['Gc_GDC_LSCF'], floc=0)
    ax.plot(x, lognorm.pdf(x, shape_ht2, loc_ht2, scale_ht2), 'r-', linewidth=2)

    ax.set_xlabel('$G_c$ (GDC|LSCF) [J/m²]', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('(b) GDC|LSCF Interface', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 6)

    plt.suptitle('PDF of Interfacial Fracture Toughness: Room Temperature vs. 800°C\n'
                 'Note: σ_HT assumed equal to σ_RT (Missing Dataset 02)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig04_pdf_histograms_RT_vs_HT.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig04_pdf_histograms_RT_vs_HT.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig04_pdf_histograms_RT_vs_HT.png/pdf")


# ============================================================================
# FIGURE 5: Rho sensitivity sweep
# ============================================================================
def plot_rho_sensitivity():
    """Plot failure probability as function of J_app for different rho values."""
    df_sweep = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho_sweep.csv'))

    rho_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    colors = ['#1565C0', '#1E88E5', '#42A5F5', '#90CAF9', '#BBDEFB']
    J_range = np.linspace(0.5, 12, 300)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for rho_val, color in zip(rho_values, colors):
        subset = df_sweep[df_sweep['Rho'] == rho_val]
        R = subset['R_system'].values
        Pf = [np.mean(R < J) for J in J_range]
        ax.plot(J_range, Pf, linewidth=2.5, color=color, label=f'ρ = {rho_val:.2f}')

    ax.axvline(x=3.0, color='green', linestyle='-.', linewidth=1.5, alpha=0.5)
    ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xlabel('Applied Energy Release Rate, $J_{applied}$ (J/m²)', fontsize=14)
    ax.set_ylabel('Probability of System Failure, $P_f$', fontsize=14)
    ax.set_title('Sensitivity of Failure Probability to Unknown Correlation ρ\n'
                 'Full Parametric Sweep (Missing Dataset 01)', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, title='Correlation ρ')
    ax.set_xlim(0.5, 12)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig05_rho_sensitivity_sweep.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig05_rho_sensitivity_sweep.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig05_rho_sensitivity_sweep.png/pdf")


# ============================================================================
# FIGURE 6: LSCF Ferroelastic Stress-Strain
# ============================================================================
def plot_lscf_stress_strain():
    """Plot LSCF ferroelastic stress-strain curves at multiple temperatures."""
    df_lscf = pd.read_csv(os.path.join(CSV_DIR, '13_LSCF_ferroelastic_stress_strain.csv'))

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    temps = [25, 200, 400, 600, 800]
    cmap = plt.cm.coolwarm
    norm_color = plt.Normalize(vmin=25, vmax=800)

    for T in temps:
        subset = df_lscf[df_lscf['Temperature_C'] == T]
        color = cmap(norm_color(T))
        ax.plot(subset['Strain'] * 100, subset['Stress_MPa'], linewidth=2.5,
                color=color, label=f'{T}°C')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Temperature (°C)')

    ax.set_xlabel('Strain (%)', fontsize=14)
    ax.set_ylabel('Stress (MPa)', fontsize=14)
    ax.set_title('LSCF Ferroelastic Stress-Strain Response\n'
                 'Temperature-Dependent Nonlinear Behavior', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig06_LSCF_ferroelastic_stress_strain.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig06_LSCF_ferroelastic_stress_strain.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig06_LSCF_ferroelastic_stress_strain.png/pdf")


# ============================================================================
# FIGURE 7: Micro-cantilever data summary
# ============================================================================
def plot_micro_cantilever_summary():
    """Plot micro-cantilever fracture data with box plots and scatter."""
    df_mc = pd.read_csv(os.path.join(CSV_DIR, '16_micro_cantilever_fracture_data.csv'))
    df_valid = df_mc[df_mc['Valid_Test'] == True]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Box plot by interface
    ax = axes[0]
    interfaces = ['YSZ_GDC', 'GDC_LSCF']
    data_by_iface = [df_valid[df_valid['Interface'] == iface]['Gc_Jm2'].values for iface in interfaces]

    bp = ax.boxplot(data_by_iface, labels=['YSZ|GDC', 'GDC|LSCF'], widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    colors_box = ['#2196F3', '#FF9800']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay individual data points
    for i, iface in enumerate(interfaces):
        y = df_valid[df_valid['Interface'] == iface]['Gc_Jm2'].values
        x = rng.normal(i + 1, 0.05, len(y))
        ax.scatter(x, y, alpha=0.4, s=20, color=colors_box[i], edgecolors='none')

    ax.set_ylabel('Fracture Toughness, $G_c$ (J/m²)', fontsize=13)
    ax.set_title('(a) Distribution by Interface', fontsize=13)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel B: Gc vs Beam Width (quality check)
    ax = axes[1]
    for iface, color, marker in zip(interfaces, colors_box, ['o', 's']):
        subset = df_valid[df_valid['Interface'] == iface]
        ax.scatter(subset['Beam_Width_um'], subset['Gc_Jm2'], alpha=0.6, s=40,
                   color=color, marker=marker, edgecolors='gray', linewidth=0.5,
                   label=iface.replace('_', '|'))

    ax.set_xlabel('Beam Width (µm)', fontsize=13)
    ax.set_ylabel('$G_c$ (J/m²)', fontsize=13)
    ax.set_title('(b) Gc vs. Beam Geometry (Size Effect Check)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Micro-Cantilever Fracture Toughness Data Summary\n'
                 'Room Temperature Only (Missing Dataset 02: No HT data)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig07_micro_cantilever_fracture_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig07_micro_cantilever_fracture_summary.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig07_micro_cantilever_fracture_summary.png/pdf")


# ============================================================================
# FIGURE 8: Probabilistic Failure Map (2D contour)
# ============================================================================
def plot_failure_map_2d():
    """Plot 2D probabilistic failure map: Gc1 vs Gc2 with failure regions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create a grid
    gc1_range = np.linspace(0.5, 7, 200)
    gc2_range = np.linspace(0.5, 5, 200)
    GC1, GC2 = np.meshgrid(gc1_range, gc2_range)

    # System failure when J_applied > R_system = Gc1 + Gc2
    J_applied_values = [2.0, 3.0, 4.0, 5.0, 6.0]
    colors_contour = ['#1B5E20', '#4CAF50', '#FFC107', '#FF5722', '#B71C1C']

    for J_app, color in zip(J_applied_values, colors_contour):
        failure_boundary = GC1 + GC2 - J_app
        ax.contour(GC1, GC2, failure_boundary, levels=[0], colors=[color], linewidths=2)
        # Label
        idx = np.argmin(np.abs(gc1_range - J_app / 2))
        ax.text(J_app / 2 - 0.2, J_app / 2 + 0.3, f'$J$ = {J_app}',
                fontsize=10, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Overlay the data cloud (rho=0 case)
    df_indep = pd.read_csv(os.path.join(CSV_DIR, 'stochastic_inputs_rho0.00.csv'))
    ax.scatter(df_indep['Gc_YSZ_GDC'][:1000], df_indep['Gc_GDC_LSCF'][:1000],
               alpha=0.08, s=5, color='blue', edgecolors='none')

    # Mark means
    ax.plot(GC_YSZ_GDC_MEAN_RT, GC_GDC_LSCF_MEAN_RT, 'k*', markersize=15,
            label=f'RT Mean ({GC_YSZ_GDC_MEAN_RT}, {GC_GDC_LSCF_MEAN_RT})')
    ax.plot(GC_YSZ_GDC_MEAN_RT * HT_SCALE_FACTOR, GC_GDC_LSCF_MEAN_RT * HT_SCALE_FACTOR,
            'r*', markersize=15,
            label=f'800°C Mean ({GC_YSZ_GDC_MEAN_RT*HT_SCALE_FACTOR:.2f}, {GC_GDC_LSCF_MEAN_RT*HT_SCALE_FACTOR:.2f})')

    ax.set_xlabel('$G_c$ (YSZ|GDC) [J/m²]', fontsize=14)
    ax.set_ylabel('$G_c$ (GDC|LSCF) [J/m²]', fontsize=14)
    ax.set_title('Probabilistic Failure Map: Interface Toughness Space\n'
                 'Failure Boundaries for Different Applied Loads',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, 7)
    ax.set_ylim(0.5, 5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig08_probabilistic_failure_map_2D.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig08_probabilistic_failure_map_2D.pdf'), bbox_inches='tight')
    plt.close()
    print("[OK] fig08_probabilistic_failure_map_2D.png/pdf")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GENERATING DATASETS: Probabilistic Failure Maps for SOC Interfaces")
    print("=" * 70)

    # Step 1: Generate all CSV datasets
    print("\n--- Generating CSV Datasets ---")
    generate_material_properties()
    generate_micro_cantilever_data()
    generate_lscf_ferroelastic()
    generate_missing_data_flags()
    generate_stochastic_rho000()
    generate_stochastic_rho050()
    generate_stochastic_HT()
    generate_stochastic_rho_sweep()

    # Step 2: Generate derived summary (needs the CSVs above)
    generate_failure_probability_summary()

    # Step 3: Generate all figures
    print("\n--- Generating Figures ---")
    plot_cone_of_ignorance()
    plot_joint_scatter()
    plot_cdf_comparison()
    plot_pdf_histograms()
    plot_rho_sensitivity()
    plot_lscf_stress_strain()
    plot_micro_cantilever_summary()
    plot_failure_map_2d()

    print("\n" + "=" * 70)
    print("ALL DATASETS AND FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nCSV files in: ./{CSV_DIR}/")
    print(f"Figures in:   ./{FIG_DIR}/")
