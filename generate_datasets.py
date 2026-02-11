#!/usr/bin/env python3
"""
Generate fabricated datasets for:
  "Probabilistic Failure Maps: Uncertainty Quantification of
   Interfacial Toughness in Solid Oxide Cells"

Outputs
-------
CSV files:
  04_uncertainty_material_properties.csv
  16_micro_cantilever_fracture_data.csv

Figures:
  fig_scatter_independent_vs_correlated.png
  fig_fracture_energy_histograms.png
  fig_weibull_probability_plot.png
  fig_material_property_distributions.png

Archive:
  datasets.zip  (contains both CSV files)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import zipfile
import os

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# 1.  04_uncertainty_material_properties.csv
# ═══════════════════════════════════════════════════════════════════════

def generate_material_properties():
    """
    Fabricate a material-properties table for YSZ, GDC, LSCF, Ni-YSZ,
    and Crofer 22 APU with uncertainty bounds.  Values are based on
    published ranges in the SOFC/SOC literature.
    """
    rows = []

    # ---------- YSZ (8 mol% Yttria-Stabilized Zirconia) Electrolyte ----------
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Young's Modulus",
                     Symbol="E", Temperature_C=25, Mean=205.0, Std=8.5,
                     CV_pct=4.15, Distribution="Normal",
                     Lower_95=188.3, Upper_95=221.7, Unit="GPa",
                     Source="Selcuk & Atkinson 1997"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Young's Modulus",
                     Symbol="E", Temperature_C=800, Mean=170.0, Std=7.2,
                     CV_pct=4.24, Distribution="Normal",
                     Lower_95=155.9, Upper_95=184.1, Unit="GPa",
                     Source="Radovic & Lara-Curzio 2004"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Poisson's Ratio",
                     Symbol="nu", Temperature_C=25, Mean=0.31, Std=0.01,
                     CV_pct=3.23, Distribution="Normal",
                     Lower_95=0.29, Upper_95=0.33, Unit="-",
                     Source="Adams et al. 1997"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="CTE",
                     Symbol="alpha", Temperature_C=800, Mean=10.5e-6, Std=0.3e-6,
                     CV_pct=2.86, Distribution="Normal",
                     Lower_95=9.9e-6, Upper_95=11.1e-6, Unit="1/K",
                     Source="Hayashi et al. 2005"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Fracture Toughness",
                     Symbol="K_Ic", Temperature_C=25, Mean=2.0, Std=0.3,
                     CV_pct=15.0, Distribution="Weibull",
                     Lower_95=1.41, Upper_95=2.59, Unit="MPa·m^0.5",
                     Source="Cutler et al. 1987"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Fracture Energy",
                     Symbol="Gc", Temperature_C=25, Mean=18.8, Std=5.5,
                     CV_pct=29.3, Distribution="Weibull",
                     Lower_95=8.0, Upper_95=29.6, Unit="J/m²",
                     Source="Derived from K_Ic"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Flexural Strength",
                     Symbol="sigma_f", Temperature_C=25, Mean=270.0, Std=35.0,
                     CV_pct=12.96, Distribution="Weibull",
                     Lower_95=201.4, Upper_95=338.6, Unit="MPa",
                     Source="Masaki 1986"))
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Flexural Strength",
                     Symbol="sigma_f", Temperature_C=800, Mean=220.0, Std=30.0,
                     CV_pct=13.64, Distribution="Weibull",
                     Lower_95=161.2, Upper_95=278.8, Unit="MPa",
                     Source="Proportional scaling"))

    # ---------- YSZ|GDC Interface ----------
    rows.append(dict(Material="YSZ|GDC", Layer="Interface", Property="Interfacial Fracture Energy",
                     Symbol="Gc_int", Temperature_C=25, Mean=2.15, Std=0.40,
                     CV_pct=18.6, Distribution="Weibull",
                     Lower_95=1.37, Upper_95=2.93, Unit="J/m²",
                     Source="Micro-cantilever tests"))
    rows.append(dict(Material="YSZ|GDC", Layer="Interface", Property="Interfacial Fracture Energy",
                     Symbol="Gc_int", Temperature_C=800, Mean=1.78, Std=0.33,
                     CV_pct=18.6, Distribution="Weibull",
                     Lower_95=1.13, Upper_95=2.43, Unit="J/m²",
                     Source="CV-constant scaling"))
    rows.append(dict(Material="YSZ|GDC", Layer="Interface", Property="Mode Mixity Angle",
                     Symbol="psi", Temperature_C=25, Mean=42.0, Std=3.5,
                     CV_pct=8.33, Distribution="Normal",
                     Lower_95=35.1, Upper_95=48.9, Unit="deg",
                     Source="FEA of cantilever geometry"))

    # ---------- GDC|LSCF Interface ----------
    rows.append(dict(Material="GDC|LSCF", Layer="Interface", Property="Interfacial Fracture Energy",
                     Symbol="Gc_int", Temperature_C=25, Mean=1.00, Std=0.20,
                     CV_pct=20.0, Distribution="Weibull",
                     Lower_95=0.61, Upper_95=1.39, Unit="J/m²",
                     Source="Micro-cantilever tests"))
    rows.append(dict(Material="GDC|LSCF", Layer="Interface", Property="Interfacial Fracture Energy",
                     Symbol="Gc_int", Temperature_C=800, Mean=0.83, Std=0.17,
                     CV_pct=20.0, Distribution="Weibull",
                     Lower_95=0.50, Upper_95=1.16, Unit="J/m²",
                     Source="CV-constant scaling"))
    rows.append(dict(Material="GDC|LSCF", Layer="Interface", Property="Mode Mixity Angle",
                     Symbol="psi", Temperature_C=25, Mean=38.0, Std=4.0,
                     CV_pct=10.53, Distribution="Normal",
                     Lower_95=30.2, Upper_95=45.8, Unit="deg",
                     Source="FEA of cantilever geometry"))

    # ---------- GDC (Gadolinium-Doped Ceria) interlayer ----------
    rows.append(dict(Material="GDC", Layer="Interlayer", Property="Young's Modulus",
                     Symbol="E", Temperature_C=25, Mean=215.0, Std=10.0,
                     CV_pct=4.65, Distribution="Normal",
                     Lower_95=195.4, Upper_95=234.6, Unit="GPa",
                     Source="Morales et al. 2010"))
    rows.append(dict(Material="GDC", Layer="Interlayer", Property="Young's Modulus",
                     Symbol="E", Temperature_C=800, Mean=180.0, Std=9.0,
                     CV_pct=5.0, Distribution="Normal",
                     Lower_95=162.4, Upper_95=197.6, Unit="GPa",
                     Source="Estimated from E(T) trend"))
    rows.append(dict(Material="GDC", Layer="Interlayer", Property="CTE",
                     Symbol="alpha", Temperature_C=800, Mean=12.5e-6, Std=0.4e-6,
                     CV_pct=3.2, Distribution="Normal",
                     Lower_95=11.7e-6, Upper_95=13.3e-6, Unit="1/K",
                     Source="Sameshima et al. 1999"))
    rows.append(dict(Material="GDC", Layer="Interlayer", Property="Poisson's Ratio",
                     Symbol="nu", Temperature_C=25, Mean=0.33, Std=0.015,
                     CV_pct=4.55, Distribution="Normal",
                     Lower_95=0.30, Upper_95=0.36, Unit="-",
                     Source="Morales et al. 2010"))

    # ---------- LSCF (Lanthanum Strontium Cobalt Ferrite) Cathode ----------
    rows.append(dict(Material="LSCF", Layer="Cathode", Property="Young's Modulus",
                     Symbol="E", Temperature_C=25, Mean=115.0, Std=12.0,
                     CV_pct=10.43, Distribution="Normal",
                     Lower_95=91.5, Upper_95=138.5, Unit="GPa",
                     Source="Huang et al. 2009"))
    rows.append(dict(Material="LSCF", Layer="Cathode", Property="Young's Modulus",
                     Symbol="E", Temperature_C=800, Mean=88.0, Std=10.0,
                     CV_pct=11.36, Distribution="Normal",
                     Lower_95=68.4, Upper_95=107.6, Unit="GPa",
                     Source="Estimated"))
    rows.append(dict(Material="LSCF", Layer="Cathode", Property="CTE",
                     Symbol="alpha", Temperature_C=800, Mean=15.3e-6, Std=0.6e-6,
                     CV_pct=3.92, Distribution="Normal",
                     Lower_95=14.1e-6, Upper_95=16.5e-6, Unit="1/K",
                     Source="Tai et al. 1995"))
    rows.append(dict(Material="LSCF", Layer="Cathode", Property="Poisson's Ratio",
                     Symbol="nu", Temperature_C=25, Mean=0.32, Std=0.02,
                     CV_pct=6.25, Distribution="Normal",
                     Lower_95=0.28, Upper_95=0.36, Unit="-",
                     Source="Huang et al. 2009"))

    # ---------- Ni-YSZ Anode ----------
    rows.append(dict(Material="Ni-YSZ", Layer="Anode", Property="Young's Modulus",
                     Symbol="E", Temperature_C=25, Mean=96.0, Std=8.0,
                     CV_pct=8.33, Distribution="Normal",
                     Lower_95=80.3, Upper_95=111.7, Unit="GPa",
                     Source="Pihlatie et al. 2009"))
    rows.append(dict(Material="Ni-YSZ", Layer="Anode", Property="Young's Modulus",
                     Symbol="E", Temperature_C=800, Mean=75.0, Std=7.0,
                     CV_pct=9.33, Distribution="Normal",
                     Lower_95=61.3, Upper_95=88.7, Unit="GPa",
                     Source="Pihlatie et al. 2009"))
    rows.append(dict(Material="Ni-YSZ", Layer="Anode", Property="CTE",
                     Symbol="alpha", Temperature_C=800, Mean=13.2e-6, Std=0.5e-6,
                     CV_pct=3.79, Distribution="Normal",
                     Lower_95=12.2e-6, Upper_95=14.2e-6, Unit="1/K",
                     Source="Mori et al. 1998"))
    rows.append(dict(Material="Ni-YSZ", Layer="Anode", Property="Poisson's Ratio",
                     Symbol="nu", Temperature_C=25, Mean=0.36, Std=0.02,
                     CV_pct=5.56, Distribution="Normal",
                     Lower_95=0.32, Upper_95=0.40, Unit="-",
                     Source="Pihlatie et al. 2009"))
    rows.append(dict(Material="Ni-YSZ", Layer="Anode", Property="Porosity",
                     Symbol="phi", Temperature_C=25, Mean=0.30, Std=0.04,
                     CV_pct=13.33, Distribution="Normal",
                     Lower_95=0.22, Upper_95=0.38, Unit="-",
                     Source="Typical reduction conditions"))

    # ---------- Crofer 22 APU Interconnect ----------
    rows.append(dict(Material="Crofer22APU", Layer="Interconnect", Property="Young's Modulus",
                     Symbol="E", Temperature_C=25, Mean=220.0, Std=5.0,
                     CV_pct=2.27, Distribution="Normal",
                     Lower_95=210.2, Upper_95=229.8, Unit="GPa",
                     Source="ThyssenKrupp datasheet"))
    rows.append(dict(Material="Crofer22APU", Layer="Interconnect", Property="CTE",
                     Symbol="alpha", Temperature_C=800, Mean=11.9e-6, Std=0.3e-6,
                     CV_pct=2.52, Distribution="Normal",
                     Lower_95=11.3e-6, Upper_95=12.5e-6, Unit="1/K",
                     Source="ThyssenKrupp datasheet"))
    rows.append(dict(Material="Crofer22APU", Layer="Interconnect", Property="Poisson's Ratio",
                     Symbol="nu", Temperature_C=25, Mean=0.29, Std=0.01,
                     CV_pct=3.45, Distribution="Normal",
                     Lower_95=0.27, Upper_95=0.31, Unit="-",
                     Source="ThyssenKrupp datasheet"))

    # ---------- Glass-Ceramic Sealant ----------
    rows.append(dict(Material="Glass-Ceramic", Layer="Sealant", Property="Young's Modulus",
                     Symbol="E", Temperature_C=25, Mean=85.0, Std=6.0,
                     CV_pct=7.06, Distribution="Normal",
                     Lower_95=73.2, Upper_95=96.8, Unit="GPa",
                     Source="Chou et al. 2007"))
    rows.append(dict(Material="Glass-Ceramic", Layer="Sealant", Property="CTE",
                     Symbol="alpha", Temperature_C=800, Mean=10.2e-6, Std=0.5e-6,
                     CV_pct=4.9, Distribution="Normal",
                     Lower_95=9.2e-6, Upper_95=11.2e-6, Unit="1/K",
                     Source="Chou et al. 2007"))

    # ---------- Weibull parameters for critical fracture properties ----------
    rows.append(dict(Material="8YSZ", Layer="Electrolyte", Property="Weibull Modulus (Strength)",
                     Symbol="m", Temperature_C=25, Mean=10.5, Std=1.5,
                     CV_pct=14.29, Distribution="Point estimate range",
                     Lower_95=7.6, Upper_95=13.4, Unit="-",
                     Source="Selcuk & Atkinson 2000"))
    rows.append(dict(Material="YSZ|GDC", Layer="Interface", Property="Weibull Modulus (Gc)",
                     Symbol="m", Temperature_C=25, Mean=6.2, Std=1.0,
                     CV_pct=16.13, Distribution="Point estimate range",
                     Lower_95=4.2, Upper_95=8.2, Unit="-",
                     Source="Fitted to cantilever data"))
    rows.append(dict(Material="GDC|LSCF", Layer="Interface", Property="Weibull Modulus (Gc)",
                     Symbol="m", Temperature_C=25, Mean=5.5, Std=1.2,
                     CV_pct=21.82, Distribution="Point estimate range",
                     Lower_95=3.1, Upper_95=7.9, Unit="-",
                     Source="Fitted to cantilever data"))

    df = pd.DataFrame(rows)
    col_order = ["Material", "Layer", "Property", "Symbol", "Temperature_C",
                 "Mean", "Std", "CV_pct", "Distribution",
                 "Lower_95", "Upper_95", "Unit", "Source"]
    return df[col_order]


# ═══════════════════════════════════════════════════════════════════════
# 2.  16_micro_cantilever_fracture_data.csv
# ═══════════════════════════════════════════════════════════════════════

def generate_micro_cantilever_data():
    """
    Fabricate micro-cantilever fracture test results for YSZ|GDC
    and GDC|LSCF interfaces, with realistic scatter and dimensions.
    """
    rows = []
    sample_id = 0

    # ---------- YSZ|GDC interface specimens ----------
    n_ysz_gdc = 45
    for i in range(n_ysz_gdc):
        sample_id += 1
        # Cantilever geometry with small fabrication scatter
        length = np.random.normal(15.0, 0.4)       # µm
        width  = np.random.normal(3.0, 0.15)        # µm
        depth  = np.random.normal(3.0, 0.15)        # µm
        notch_depth = np.random.normal(1.0, 0.08)   # µm
        notch_width = np.random.normal(0.10, 0.015)  # µm

        # Effective beam dimensions
        b_eff = width
        d_eff = depth - notch_depth

        # Fracture load from Weibull-like distribution (shape ~6, scale tuned)
        # Using a Weibull with shape=6.2 and scale giving mean ~ 0.35 mN
        F_frac = stats.weibull_min.rvs(c=6.2, scale=0.38, size=1)[0]
        F_frac = max(F_frac, 0.08)  # floor

        # Compute stress intensity factor K = F*L / (b * d^1.5) * Y(a/d)
        a_over_d = notch_depth / depth
        # Geometry factor (approximate Tada formula for edge-notch bend)
        Y_factor = 1.12 - 0.23 * a_over_d + 10.6 * a_over_d**2 \
                   - 21.7 * a_over_d**3 + 30.4 * a_over_d**4
        K_Ic = (F_frac * 1e-3 * length * 1e-6) / (b_eff * 1e-6 * (d_eff * 1e-6)**1.5) \
               * Y_factor * 1e-6  # MPa√m
        # Adjust to realistic range (1.0 – 3.5 MPa√m)
        K_Ic = np.clip(np.abs(K_Ic) * 0.00008 + np.random.normal(1.85, 0.35), 0.8, 3.8)

        # Fracture energy: Gc = K^2 / E' where E' ~ 195 GPa (plane strain)
        E_prime = np.random.normal(195.0, 8.0)  # GPa
        Gc = (K_Ic**2) / E_prime * 1e3  # J/m²

        rows.append(dict(
            Sample_ID=f"YG-{sample_id:03d}",
            Interface="YSZ|GDC",
            Temperature_C=25,
            Cantilever_Length_um=round(length, 2),
            Cantilever_Width_um=round(width, 2),
            Cantilever_Depth_um=round(depth, 2),
            Notch_Depth_um=round(notch_depth, 2),
            Notch_Width_um=round(notch_width, 3),
            a_over_d=round(a_over_d, 4),
            Fracture_Load_mN=round(F_frac, 4),
            Geometry_Factor_Y=round(Y_factor, 4),
            K_Ic_MPa_sqrt_m=round(K_Ic, 4),
            E_prime_GPa=round(E_prime, 2),
            Gc_J_per_m2=round(Gc, 4),
            Failure_Mode="Interfacial",
            FIB_Quality="Good" if np.random.rand() > 0.15 else "Acceptable",
            Notes=""
        ))

    # ---------- GDC|LSCF interface specimens ----------
    n_gdc_lscf = 38
    for i in range(n_gdc_lscf):
        sample_id += 1
        length = np.random.normal(14.5, 0.5)
        width  = np.random.normal(2.8, 0.2)
        depth  = np.random.normal(2.8, 0.2)
        notch_depth = np.random.normal(0.95, 0.1)
        notch_width = np.random.normal(0.10, 0.02)

        b_eff = width
        d_eff = depth - notch_depth

        F_frac = stats.weibull_min.rvs(c=5.5, scale=0.28, size=1)[0]
        F_frac = max(F_frac, 0.05)

        a_over_d = notch_depth / depth
        Y_factor = 1.12 - 0.23 * a_over_d + 10.6 * a_over_d**2 \
                   - 21.7 * a_over_d**3 + 30.4 * a_over_d**4
        K_Ic = np.clip(np.abs(F_frac * 0.5) + np.random.normal(0.85, 0.22), 0.4, 2.5)

        E_prime = np.random.normal(185.0, 10.0)
        Gc = (K_Ic**2) / E_prime * 1e3

        failure_mode = np.random.choice(
            ["Interfacial", "Interfacial", "Interfacial", "Mixed", "Cohesive (LSCF)"],
            p=[0.55, 0.20, 0.10, 0.10, 0.05]
        )

        rows.append(dict(
            Sample_ID=f"GL-{sample_id:03d}",
            Interface="GDC|LSCF",
            Temperature_C=25,
            Cantilever_Length_um=round(length, 2),
            Cantilever_Width_um=round(width, 2),
            Cantilever_Depth_um=round(depth, 2),
            Notch_Depth_um=round(notch_depth, 2),
            Notch_Width_um=round(notch_width, 3),
            a_over_d=round(a_over_d, 4),
            Fracture_Load_mN=round(F_frac, 4),
            Geometry_Factor_Y=round(Y_factor, 4),
            K_Ic_MPa_sqrt_m=round(K_Ic, 4),
            E_prime_GPa=round(E_prime, 2),
            Gc_J_per_m2=round(Gc, 4),
            Failure_Mode=failure_mode,
            FIB_Quality="Good" if np.random.rand() > 0.2 else "Acceptable",
            Notes=""
        ))

    # ---------- Add a handful of 800 °C high-temperature tests ----------
    # (Limited dataset – only 12 specimens total at high T)
    for i in range(7):
        sample_id += 1
        length = np.random.normal(15.0, 0.4)
        width  = np.random.normal(3.0, 0.15)
        depth  = np.random.normal(3.0, 0.15)
        notch_depth = np.random.normal(1.0, 0.08)
        notch_width = np.random.normal(0.10, 0.015)
        a_over_d = notch_depth / depth
        Y_factor = 1.12 - 0.23*a_over_d + 10.6*a_over_d**2 \
                   - 21.7*a_over_d**3 + 30.4*a_over_d**4

        F_frac = stats.weibull_min.rvs(c=5.8, scale=0.33, size=1)[0]
        F_frac = max(F_frac, 0.06)
        K_Ic = np.clip(np.random.normal(1.55, 0.35), 0.6, 3.0)
        E_prime = np.random.normal(165.0, 8.0)
        Gc = (K_Ic**2) / E_prime * 1e3

        rows.append(dict(
            Sample_ID=f"YG-HT-{sample_id:03d}",
            Interface="YSZ|GDC",
            Temperature_C=800,
            Cantilever_Length_um=round(length, 2),
            Cantilever_Width_um=round(width, 2),
            Cantilever_Depth_um=round(depth, 2),
            Notch_Depth_um=round(notch_depth, 2),
            Notch_Width_um=round(notch_width, 3),
            a_over_d=round(a_over_d, 4),
            Fracture_Load_mN=round(F_frac, 4),
            Geometry_Factor_Y=round(Y_factor, 4),
            K_Ic_MPa_sqrt_m=round(K_Ic, 4),
            E_prime_GPa=round(E_prime, 2),
            Gc_J_per_m2=round(Gc, 4),
            Failure_Mode="Interfacial",
            FIB_Quality="Good" if np.random.rand() > 0.25 else "Acceptable",
            Notes="High-temperature in-situ test"
        ))

    for i in range(5):
        sample_id += 1
        length = np.random.normal(14.5, 0.5)
        width  = np.random.normal(2.8, 0.2)
        depth  = np.random.normal(2.8, 0.2)
        notch_depth = np.random.normal(0.95, 0.1)
        notch_width = np.random.normal(0.10, 0.02)
        a_over_d = notch_depth / depth
        Y_factor = 1.12 - 0.23*a_over_d + 10.6*a_over_d**2 \
                   - 21.7*a_over_d**3 + 30.4*a_over_d**4

        F_frac = stats.weibull_min.rvs(c=5.0, scale=0.25, size=1)[0]
        F_frac = max(F_frac, 0.04)
        K_Ic = np.clip(np.random.normal(0.72, 0.18), 0.3, 1.8)
        E_prime = np.random.normal(160.0, 10.0)
        Gc = (K_Ic**2) / E_prime * 1e3

        rows.append(dict(
            Sample_ID=f"GL-HT-{sample_id:03d}",
            Interface="GDC|LSCF",
            Temperature_C=800,
            Cantilever_Length_um=round(length, 2),
            Cantilever_Width_um=round(width, 2),
            Cantilever_Depth_um=round(depth, 2),
            Notch_Depth_um=round(notch_depth, 2),
            Notch_Width_um=round(notch_width, 3),
            a_over_d=round(a_over_d, 4),
            Fracture_Load_mN=round(F_frac, 4),
            Geometry_Factor_Y=round(Y_factor, 4),
            K_Ic_MPa_sqrt_m=round(K_Ic, 4),
            E_prime_GPa=round(E_prime, 2),
            Gc_J_per_m2=round(Gc, 4),
            Failure_Mode="Interfacial",
            FIB_Quality="Good" if np.random.rand() > 0.3 else "Acceptable",
            Notes="High-temperature in-situ test"
        ))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 3.  FIGURES
# ═══════════════════════════════════════════════════════════════════════

def plot_scatter_independent_vs_correlated(outpath):
    """
    Figure: Scatter Plot of Fracture Energies — Independent vs Correlated
    """
    n = 500
    mu1, sigma1 = 2.15, 0.40   # YSZ|GDC
    mu2, sigma2 = 1.00, 0.20   # GDC|LSCF

    # Independent (rho = 0)
    cov_indep = [[sigma1**2, 0], [0, sigma2**2]]
    samples_indep = np.random.multivariate_normal([mu1, mu2], cov_indep, n)

    # Correlated (rho = 0.7)
    rho = 0.7
    cov_corr = [[sigma1**2, rho*sigma1*sigma2],
                [rho*sigma1*sigma2, sigma2**2]]
    samples_corr = np.random.multivariate_normal([mu1, mu2], cov_corr, n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Style
    colors = ["#2196F3", "#E91E63"]
    
    ax = axes[0]
    ax.scatter(samples_indep[:, 0], samples_indep[:, 1],
               alpha=0.35, s=18, c=colors[0], edgecolors="none")
    ax.set_title(r"Series A: Independent ($\rho = 0$)" + "\n(Current Assumption)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$G_c^{YSZ|GDC}$ [J/m²]", fontsize=11)
    ax.set_ylabel(r"$G_c^{GDC|LSCF}$ [J/m²]", fontsize=11)
    ax.axhline(mu2, ls="--", lw=0.7, color="gray")
    ax.axvline(mu1, ls="--", lw=0.7, color="gray")
    ax.set_xlim(0.5, 3.8)
    ax.set_ylim(0.0, 2.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(samples_corr[:, 0], samples_corr[:, 1],
               alpha=0.35, s=18, c=colors[1], edgecolors="none")
    ax.set_title(r"Series B: Correlated ($\rho = 0.7$)" + "\n(Hypothetical Reality)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$G_c^{YSZ|GDC}$ [J/m²]", fontsize=11)
    ax.axhline(mu2, ls="--", lw=0.7, color="gray")
    ax.axvline(mu1, ls="--", lw=0.7, color="gray")
    ax.set_xlim(0.5, 3.8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Impact of Missing Correlation Data on Input Parameter Space",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Simulation currently uses Series A (Independent), potentially "
             "underestimating joint failure probabilities compared to Series B.",
             ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {outpath}")


def plot_fracture_energy_histograms(df_cant, outpath):
    """
    Histograms of Gc for each interface, RT vs 800 °C.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    combos = [
        ("YSZ|GDC", 25),
        ("YSZ|GDC", 800),
        ("GDC|LSCF", 25),
        ("GDC|LSCF", 800),
    ]
    colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00"]

    for idx, (iface, temp) in enumerate(combos):
        ax = axes[idx // 2][idx % 2]
        sub = df_cant[(df_cant["Interface"] == iface) &
                      (df_cant["Temperature_C"] == temp)]
        gc = sub["Gc_J_per_m2"].values
        if len(gc) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", fontsize=14)
            continue

        n_bins = max(5, int(np.sqrt(len(gc))))
        ax.hist(gc, bins=n_bins, color=colors[idx], alpha=0.7,
                edgecolor="white", linewidth=0.5, density=True)

        # Overlay a fitted normal
        mu, std = gc.mean(), gc.std()
        x = np.linspace(gc.min() - 0.3, gc.max() + 0.3, 200)
        ax.plot(x, stats.norm.pdf(x, mu, std), "k-", lw=1.5,
                label=f"Normal fit\n$\\mu$={mu:.2f}, $\\sigma$={std:.2f}")

        ax.set_title(f"{iface}  @  {temp} °C   (n={len(gc)})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$G_c$ [J/m²]", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Fracture Energy Distributions from Micro-Cantilever Tests",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {outpath}")


def plot_weibull_probability(df_cant, outpath):
    """
    Weibull probability plot for RT interfacial fracture energies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    interfaces = ["YSZ|GDC", "GDC|LSCF"]
    colors = ["#1565C0", "#C62828"]

    for idx, iface in enumerate(interfaces):
        ax = axes[idx]
        sub = df_cant[(df_cant["Interface"] == iface) &
                      (df_cant["Temperature_C"] == 25)]
        gc = np.sort(sub["Gc_J_per_m2"].values)
        n = len(gc)
        if n < 3:
            continue
        # Median-rank estimator
        F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        ln_gc = np.log(gc)
        ln_ln = np.log(-np.log(1 - F))

        ax.scatter(ln_gc, ln_ln, s=30, c=colors[idx], zorder=5,
                   edgecolors="white", linewidth=0.5)

        # Linear fit
        slope, intercept, r, _, _ = stats.linregress(ln_gc, ln_ln)
        x_fit = np.linspace(ln_gc.min() - 0.2, ln_gc.max() + 0.2, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray", lw=1.2,
                label=f"Weibull fit: m = {slope:.2f}, R² = {r**2:.3f}")

        ax.set_title(f"{iface} Interface (RT)", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$\ln(G_c)$  [ln(J/m²)]", fontsize=11)
        ax.set_ylabel(r"$\ln(-\ln(1 - F))$", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Weibull Probability Plots — Interfacial Fracture Energy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {outpath}")


def plot_material_property_distributions(df_mat, outpath):
    """
    Bar chart showing material property means with ±2σ error bars
    for key properties across SOC layers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # --- Subplot 1: Young's Modulus at RT ---
    ax = axes[0]
    sub = df_mat[(df_mat["Property"] == "Young's Modulus") &
                 (df_mat["Temperature_C"] == 25)]
    materials = sub["Material"].values
    means = sub["Mean"].values
    stds = sub["Std"].values
    colors_bar = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F", "#00796B"]
    bars = ax.barh(materials, means, xerr=2*stds, height=0.6,
                   color=colors_bar[:len(materials)], alpha=0.85,
                   capsize=4, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Young's Modulus [GPa]", fontsize=11)
    ax.set_title("Young's Modulus @ RT", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # --- Subplot 2: CTE at 800 °C ---
    ax = axes[1]
    sub = df_mat[(df_mat["Property"] == "CTE") & (df_mat["Temperature_C"] == 800)]
    materials = sub["Material"].values
    means = sub["Mean"].values * 1e6  # convert to 10^-6 /K
    stds = sub["Std"].values * 1e6
    bars = ax.barh(materials, means, xerr=2*stds, height=0.6,
                   color=colors_bar[:len(materials)], alpha=0.85,
                   capsize=4, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(r"CTE [$\times 10^{-6}$ K$^{-1}$]", fontsize=11)
    ax.set_title("CTE @ 800 °C", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # --- Subplot 3: Interfacial Fracture Energy RT ---
    ax = axes[2]
    sub = df_mat[(df_mat["Property"] == "Interfacial Fracture Energy") &
                 (df_mat["Temperature_C"] == 25)]
    materials = sub["Material"].values
    means = sub["Mean"].values
    stds = sub["Std"].values
    bars = ax.barh(materials, means, xerr=2*stds, height=0.5,
                   color=["#1565C0", "#C62828"][:len(materials)], alpha=0.85,
                   capsize=4, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(r"$G_c$ [J/m²]", fontsize=11)
    ax.set_title("Interfacial Fracture Energy @ RT", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Material Property Distributions — Uncertainty Overview",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {outpath}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Generating fabricated datasets for Probabilistic Failure Maps")
    print("=" * 65)

    # 1. Material properties
    print("\n[1/5] Generating 04_uncertainty_material_properties.csv ...")
    df_mat = generate_material_properties()
    df_mat.to_csv("04_uncertainty_material_properties.csv", index=False)
    print(f"  -> {len(df_mat)} rows written.")

    # 2. Micro-cantilever data
    print("\n[2/5] Generating 16_micro_cantilever_fracture_data.csv ...")
    df_cant = generate_micro_cantilever_data()
    df_cant.to_csv("16_micro_cantilever_fracture_data.csv", index=False)
    print(f"  -> {len(df_cant)} specimens written.")

    # 3. Figures
    print("\n[3/5] Generating figures ...")
    plot_scatter_independent_vs_correlated("fig_scatter_independent_vs_correlated.png")
    plot_fracture_energy_histograms(df_cant, "fig_fracture_energy_histograms.png")
    plot_weibull_probability(df_cant, "fig_weibull_probability_plot.png")
    plot_material_property_distributions(df_mat, "fig_material_property_distributions.png")

    # 4. Quick sanity print
    print("\n[4/5] Quick preview of generated data ...")
    print("\n--- 04_uncertainty_material_properties.csv (first 500 chars) ---")
    with open("04_uncertainty_material_properties.csv") as f:
        print(f.read()[:500])
    print("\n--- 16_micro_cantilever_fracture_data.csv (first 500 chars) ---")
    with open("16_micro_cantilever_fracture_data.csv") as f:
        print(f.read()[:500])

    # 5. Zip
    print("\n[5/5] Packaging CSV files into datasets.zip ...")
    with zipfile.ZipFile("datasets.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("04_uncertainty_material_properties.csv")
        zf.write("16_micro_cantilever_fracture_data.csv")
    print("  -> datasets.zip created.")

    print("\n" + "=" * 65)
    print("DONE. Files generated:")
    for f in ["04_uncertainty_material_properties.csv",
              "16_micro_cantilever_fracture_data.csv",
              "fig_scatter_independent_vs_correlated.png",
              "fig_fracture_energy_histograms.png",
              "fig_weibull_probability_plot.png",
              "fig_material_property_distributions.png",
              "datasets.zip"]:
        size_kb = os.path.getsize(f) / 1024
        print(f"  {f:50s} {size_kb:6.1f} KB")
    print("=" * 65)
