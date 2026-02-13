#!/usr/bin/env python3
"""
Publication-Quality Figure Generator
======================================
Generates all figures for the Abaqus simulation dataset.
Formatted for Nature-family submission standards:
  - 300+ DPI
  - Clear axis labels with LaTeX notation
  - Consistent color scheme
  - Error bars where applicable
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy.stats import weibull_min
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Nature-style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color palette
COLORS = {
    'YSZ': '#2166AC',    # Blue
    'GDC': '#B2182B',    # Red
    'LSCF': '#1B7837',   # Green
    'LSCF_porous': '#762A83',  # Purple
    'interface1': '#E08214',   # Orange (YSZ|GDC)
    'interface2': '#542788',   # Deep purple (GDC|LSCF)
    'model': '#000000',        # Black
    'experiment': '#D6604D',   # Salmon
    'uncertainty': '#BDBDBD',  # Gray
}

MARKERS = {'YSZ': 'o', 'GDC': 's', 'LSCF': '^', 'LSCF_porous': 'D'}


def save_fig(fig, name):
    """Save figure in both PNG and PDF formats."""
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


# ============================================================================
# FIGURE 1: Layer Geometry Schematic
# ============================================================================
def fig01_layer_schematic():
    df = pd.read_csv(os.path.join(CSV_DIR, "01_layer_thicknesses.csv"))
    df_rough = pd.read_csv(os.path.join(CSV_DIR, "01_interface_roughness.csv"))

    fig, ax = plt.subplots(figsize=(7, 4))

    # Draw layers
    layers = [
        ("YSZ Electrolyte\n(Dense)", 0, 10, COLORS['YSZ'], 0.3),
        ("GDC Interlayer\n(Dense, nanoscale)", 10, 2, COLORS['GDC'], 0.3),
        ("LSCF Cathode\n(Porous, φ=0.35)", 12, 20, COLORS['LSCF'], 0.25),
    ]

    for label, y0, height, color, alpha in layers:
        rect = Rectangle((0, y0), 30, height, linewidth=1.2, edgecolor='black',
                         facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(15, y0 + height/2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

    # Mark interfaces
    for y, label, color in [(10, 'YSZ|GDC Interface', COLORS['interface1']),
                             (12, 'GDC|LSCF Interface', COLORS['interface2'])]:
        ax.axhline(y=y, color=color, linewidth=2, linestyle='--')
        ax.annotate(label, xy=(30.5, y), fontsize=8, color=color, va='center')

    # Dimension annotations
    ax.annotate('', xy=(-2, 0), xytext=(-2, 10),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(-3.5, 5, r'$t_{YSZ}$=10 µm', fontsize=9, rotation=90, va='center', ha='center')

    ax.annotate('', xy=(-2, 10), xytext=(-2, 12),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(-3.5, 11, r'$t_{GDC}$=2 µm', fontsize=9, rotation=90, va='center', ha='center')

    ax.annotate('', xy=(-2, 12), xytext=(-2, 32),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(-3.5, 22, r'$t_{LSCF}$=20 µm', fontsize=9, rotation=90, va='center', ha='center')

    ax.set_xlim(-6, 42)
    ax.set_ylim(-2, 35)
    ax.set_xlabel('In-plane direction (µm)', fontsize=10)
    ax.set_ylabel('Through-thickness direction (µm)', fontsize=10)
    ax.set_title('(a) YSZ/GDC/LSCF Tri-layer Half-Cell Geometry', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(False)

    save_fig(fig, "fig01_layer_geometry_schematic")


# ============================================================================
# FIGURE 2: Mesh Sensitivity Study
# ============================================================================
def fig02_mesh_sensitivity():
    df = pd.read_csv(os.path.join(CSV_DIR, "01_mesh_sensitivity.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Energy release rate vs mesh size
    ax1.plot(df['Mesh_Size_um'], df['Normalized_Energy_Release'],
             'o-', color=COLORS['YSZ'], markersize=7, linewidth=2)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(df['Mesh_Size_um'], 0.99, 1.01, alpha=0.15, color='green',
                     label='±1% convergence band')
    ax1.set_xlabel('Element size, $h$ (µm)')
    ax1.set_ylabel('Normalized energy release rate, $G/G_{converged}$')
    ax1.set_title('(a) Mesh Objectivity: Energy Release Rate')
    ax1.set_xscale('log')
    ax1.legend(loc='lower left')
    ax1.set_ylim(0.82, 1.02)

    # (b) DOF vs mesh size
    ax2.plot(df['Mesh_Size_um'], df['DOF_Total'],
             's-', color=COLORS['GDC'], markersize=7, linewidth=2)
    ax2.set_xlabel('Element size, $h$ (µm)')
    ax2.set_ylabel('Total DOF')
    ax2.set_title('(b) Computational Cost: DOF vs Mesh Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Annotate convergence threshold
    ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Convergence threshold')
    ax2.legend()

    plt.tight_layout()
    save_fig(fig, "fig02_mesh_sensitivity")


# ============================================================================
# FIGURE 3: RVE Convergence
# ============================================================================
def fig03_rve_convergence():
    df = pd.read_csv(os.path.join(CSV_DIR, "01_rve_convergence.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    dims = df['Dimension_X_um']

    # (a) Effective modulus convergence
    ax1.plot(dims, df['Eff_Modulus_GPa'], 'o-', color=COLORS['LSCF'], markersize=7, linewidth=2)
    converged_val = df['Eff_Modulus_GPa'].iloc[-1]
    ax1.axhline(y=converged_val, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(dims, converged_val * 0.99, converged_val * 1.01,
                     alpha=0.15, color='green', label='±1% band')
    ax1.set_xlabel('RVE edge length (µm)')
    ax1.set_ylabel(r'Effective modulus, $E_{eff}$ (GPa)')
    ax1.set_title(r'(a) RVE Convergence: $E_{eff}$(LSCF, $\phi$=0.35)')
    ax1.legend()

    # (b) Effective CTE convergence
    ax2.plot(dims, df['Eff_CTE_1e6_perK'], 's-', color=COLORS['LSCF_porous'],
             markersize=7, linewidth=2)
    converged_cte = df['Eff_CTE_1e6_perK'].iloc[-1]
    ax2.axhline(y=converged_cte, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(dims, converged_cte * 0.99, converged_cte * 1.01,
                     alpha=0.15, color='green', label='±1% band')
    ax2.set_xlabel('RVE edge length (µm)')
    ax2.set_ylabel(r'Effective CTE, $\alpha_{eff}$ ($\times 10^{-6}$/K)')
    ax2.set_title(r'(b) RVE Convergence: $\alpha_{eff}$(LSCF, $\phi$=0.35)')
    ax2.legend()

    plt.tight_layout()
    save_fig(fig, "fig03_rve_convergence")


# ============================================================================
# FIGURE 4: Young's Modulus vs Temperature
# ============================================================================
def fig04_youngs_modulus():
    df = pd.read_csv(os.path.join(CSV_DIR, "02_youngs_modulus.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))

    T = df['Temperature_C']
    for mat, col, marker, label in [
        ('E_YSZ_GPa', COLORS['YSZ'], 'o', r'8YSZ ($E_{YSZ}$)'),
        ('E_GDC_GPa', COLORS['GDC'], 's', r'GDC ($E_{GDC}$)'),
        ('E_LSCF_dense_GPa', COLORS['LSCF'], '^', r'LSCF dense ($E_{LSCF}$)'),
        ('E_LSCF_porous_GPa', COLORS['LSCF_porous'], 'D', r'LSCF porous ($\phi$=0.35)')
    ]:
        unc_col = mat.replace('_GPa', '_uncertainty_GPa').replace('_dense', '').replace('_porous', '')
        if unc_col in df.columns:
            ax.errorbar(T, df[mat], yerr=df[unc_col], fmt=f'{marker}-', color=col,
                       label=label, capsize=3, markersize=6)
        else:
            ax.plot(T, df[mat], f'{marker}-', color=col, label=label, markersize=6)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel("Young's Modulus, $E$ (GPa)")
    ax.set_title("Temperature-Dependent Young's Modulus for UMAT Input")
    ax.legend(loc='upper right')
    ax.set_xlim(-25, 850)

    # Add SOFC operating region
    ax.axvspan(600, 800, alpha=0.08, color='red', label='Operating range')
    ax.text(700, ax.get_ylim()[1] * 0.95, 'SOFC\nOperating\nRange',
            ha='center', va='top', fontsize=8, color='red', alpha=0.7)

    save_fig(fig, "fig04_youngs_modulus_vs_T")


# ============================================================================
# FIGURE 5: Poisson's Ratio vs Temperature
# ============================================================================
def fig05_poissons_ratio():
    df = pd.read_csv(os.path.join(CSV_DIR, "02_poissons_ratio.csv"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    T = df['Temperature_C']

    for mat, col, marker in [('nu_YSZ', COLORS['YSZ'], 'o'),
                              ('nu_GDC', COLORS['GDC'], 's'),
                              ('nu_LSCF', COLORS['LSCF'], '^')]:
        label = mat.replace('nu_', r'$\nu_{') + '}$'
        ax.plot(T, df[mat], f'{marker}-', color=col, label=label, markersize=6)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(r"Poisson's Ratio, $\nu$")
    ax.set_title(r"Temperature-Dependent Poisson's Ratio")
    ax.legend()
    ax.set_xlim(-25, 850)
    ax.axvspan(600, 800, alpha=0.08, color='red')

    save_fig(fig, "fig05_poissons_ratio_vs_T")


# ============================================================================
# FIGURE 6: CTE with Mismatch
# ============================================================================
def fig06_thermal_expansion():
    df = pd.read_csv(os.path.join(CSV_DIR, "02_thermal_expansion.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    T = df['Temperature_C']

    # (a) CTE vs T
    for mat, col, marker, unc_col in [
        ('alpha_YSZ_1e6_perK', COLORS['YSZ'], 'o', 'alpha_YSZ_uncertainty'),
        ('alpha_GDC_1e6_perK', COLORS['GDC'], 's', 'alpha_GDC_uncertainty'),
        ('alpha_LSCF_1e6_perK', COLORS['LSCF'], '^', 'alpha_LSCF_uncertainty')
    ]:
        label = mat.split('_')[1]
        ax1.errorbar(T, df[mat], yerr=df[unc_col], fmt=f'{marker}-', color=col,
                    label=r'$\alpha_{' + label + '}$', capsize=3, markersize=6)

    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel(r'Secant CTE, $\alpha$ ($\times 10^{-6}$/K)')
    ax1.set_title('(a) Coefficient of Thermal Expansion')
    ax1.legend()
    ax1.axvspan(600, 800, alpha=0.08, color='red')

    # (b) CTE mismatch
    ax2.fill_between(T, 0, df['CTE_Mismatch_YSZ_GDC'],
                     alpha=0.3, color=COLORS['interface1'], label=r'$\Delta\alpha$ (GDC-YSZ)')
    ax2.fill_between(T, 0, df['CTE_Mismatch_GDC_LSCF'],
                     alpha=0.3, color=COLORS['interface2'], label=r'$\Delta\alpha$ (LSCF-GDC)')
    ax2.plot(T, df['CTE_Mismatch_YSZ_GDC'], '-', color=COLORS['interface1'], linewidth=2)
    ax2.plot(T, df['CTE_Mismatch_GDC_LSCF'], '-', color=COLORS['interface2'], linewidth=2)

    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel(r'CTE Mismatch, $\Delta\alpha$ ($\times 10^{-6}$/K)')
    ax2.set_title('(b) Interface CTE Mismatch (Driving Force)')
    ax2.legend()
    ax2.axvspan(600, 800, alpha=0.08, color='red')

    plt.tight_layout()
    save_fig(fig, "fig06_thermal_expansion_and_mismatch")


# ============================================================================
# FIGURE 7: GDC Chemical Expansion
# ============================================================================
def fig07_gdc_chemical_expansion():
    df = pd.read_csv(os.path.join(CSV_DIR, "03_gdc_chemical_expansion.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    temps = df['Temperature_C'].unique()
    cmap = plt.cm.hot_r

    # (a) Nonstoichiometry vs pO2
    for i, T in enumerate(temps):
        mask = df['Temperature_C'] == T
        color = cmap(0.3 + 0.3 * i)
        ax1.plot(df[mask]['log10_pO2'], df[mask]['Delta_delta'],
                'o-', color=color, label=f'{T}°C', markersize=5)

    ax1.set_xlabel(r'$\log_{10}(p_{O_2}$ / atm)')
    ax1.set_ylabel(r'Oxygen nonstoichiometry, $\Delta\delta$')
    ax1.set_title(r'(a) GDC: $\Delta\delta$ vs $p_{O_2}$')
    ax1.legend()

    # (b) Chemical strain vs pO2
    for i, T in enumerate(temps):
        mask = df['Temperature_C'] == T
        color = cmap(0.3 + 0.3 * i)
        ax2.plot(df[mask]['log10_pO2'], df[mask]['Epsilon_chem_isotropic'] * 100,
                's-', color=color, label=f'{T}°C', markersize=5)

    ax2.set_xlabel(r'$\log_{10}(p_{O_2}$ / atm)')
    ax2.set_ylabel(r'Chemical strain, $\varepsilon^{ch}$ (%)')
    ax2.set_title(r'(b) GDC: $\varepsilon^{ch}_{iso}$ vs $p_{O_2}$')
    ax2.legend()

    plt.tight_layout()
    save_fig(fig, "fig07_gdc_chemical_expansion")


# ============================================================================
# FIGURE 8: LSCF Anisotropic Chemical Expansion
# ============================================================================
def fig08_lscf_chemical_expansion():
    df = pd.read_csv(os.path.join(CSV_DIR, "03_lscf_chemical_expansion.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df['Delta_delta'], df['Epsilon_chem_11'] * 100, 'o-',
            color=COLORS['interface1'], label=r'$\varepsilon^{ch}_{11}$ (in-plane)', markersize=6)
    ax.plot(df['Delta_delta'], df['Epsilon_chem_33'] * 100, 's-',
            color=COLORS['interface2'], label=r'$\varepsilon^{ch}_{33}$ (out-of-plane)', markersize=6)
    ax.plot(df['Delta_delta'], df['Epsilon_chem_volumetric'] * 100, '^-',
            color=COLORS['LSCF'], label=r'$\varepsilon^{ch}_{vol}$ (volumetric)', markersize=6)

    # Annotate anisotropy
    ax.annotate(r'$\beta_{33}/\beta_{11}$ = ' + f"{df['Anisotropy_ratio_beta33_beta11'].iloc[0]:.2f}",
                xy=(0.10, df['Epsilon_chem_33'].iloc[10] * 100),
                xytext=(0.12, 0.5),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel(r'Oxygen nonstoichiometry, $\Delta\delta$')
    ax.set_ylabel(r'Chemical strain, $\varepsilon^{ch}$ (%)')
    ax.set_title(r'LSCF Anisotropic Chemical Expansion ($\beta_{11}$=0.032, $\beta_{33}$=0.045)')
    ax.legend()

    save_fig(fig, "fig08_lscf_anisotropic_chemical_expansion")


# ============================================================================
# FIGURE 9: Nonstoichiometry Profile
# ============================================================================
def fig09_nonstoichiometry_profile():
    df = pd.read_csv(os.path.join(CSV_DIR, "03_nonstoichiometry_profile.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    temps = df['Temperature_C'].unique()
    cmap_temps = {600: '#2166AC', 700: '#B2182B', 800: '#1B7837'}

    for T in temps:
        mask = df['Temperature_C'] == T
        color = cmap_temps[T]

        # (a) delta profile
        ax1.plot(df[mask]['z_um_from_interface'], df[mask]['Delta_delta_local'],
                '-', color=color, linewidth=2, label=f'{T}°C')

        # (b) Chemical strain profile
        ax2.plot(df[mask]['z_um_from_interface'], df[mask]['Epsilon_chem_33_local'] * 100,
                '-', color=color, linewidth=2, label=rf'$\varepsilon^{{ch}}_{{33}}$ at {T}°C')
        ax2.plot(df[mask]['z_um_from_interface'], df[mask]['Epsilon_chem_11_local'] * 100,
                '--', color=color, linewidth=1.5, label=rf'$\varepsilon^{{ch}}_{{11}}$ at {T}°C')

    ax1.set_xlabel('Distance from GDC|LSCF interface (µm)')
    ax1.set_ylabel(r'Local nonstoichiometry, $\Delta\delta(z)$')
    ax1.set_title(r'(a) $\Delta\delta$ Profile across LSCF Cathode')
    ax1.legend()
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.annotate('GDC|LSCF\ninterface', xy=(0, 0.005), fontsize=8, color='gray')

    ax2.set_xlabel('Distance from GDC|LSCF interface (µm)')
    ax2.set_ylabel(r'Chemical strain (%)')
    ax2.set_title(r'(b) Chemical Strain Profile across LSCF')
    ax2.legend(fontsize=7)

    plt.tight_layout()
    save_fig(fig, "fig09_nonstoichiometry_profile")


# ============================================================================
# FIGURE 10: Bulk Fracture Toughness
# ============================================================================
def fig10_bulk_fracture_toughness():
    df = pd.read_csv(os.path.join(CSV_DIR, "04_bulk_fracture_toughness.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    T = df['Temperature_C']

    # (a) Gc vs T
    for mat, col, marker, unc in [
        ('Gc_YSZ_Jm2', COLORS['YSZ'], 'o', 'Gc_YSZ_uncertainty_Jm2'),
        ('Gc_GDC_Jm2', COLORS['GDC'], 's', 'Gc_GDC_uncertainty_Jm2'),
        ('Gc_LSCF_Jm2', COLORS['LSCF'], '^', 'Gc_LSCF_uncertainty_Jm2')
    ]:
        label = mat.split('_')[1]
        ax1.errorbar(T, df[mat], yerr=df[unc], fmt=f'{marker}-', color=col,
                    label=r'$G_{c,' + label + '}$', capsize=3, markersize=6)

    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel(r'Fracture toughness, $G_c$ (J/m²)')
    ax1.set_title(r'(a) Bulk Fracture Toughness $G_{c,b}(T)$')
    ax1.legend()
    ax1.axvspan(600, 800, alpha=0.08, color='red')

    # (b) KIc vs T
    for mat, col, marker in [
        ('KIc_YSZ_MPam05', COLORS['YSZ'], 'o'),
        ('KIc_GDC_MPam05', COLORS['GDC'], 's'),
        ('KIc_LSCF_MPam05', COLORS['LSCF'], '^')
    ]:
        label = mat.split('_')[1]
        ax2.plot(T, df[mat], f'{marker}-', color=col,
                label=r'$K_{Ic,' + label + '}$', markersize=6)

    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel(r'Fracture toughness, $K_{Ic}$ (MPa$\cdot$m$^{0.5}$)')
    ax2.set_title(r'(b) Stress Intensity Factor $K_{Ic}(T)$')
    ax2.legend()
    ax2.axvspan(600, 800, alpha=0.08, color='red')

    plt.tight_layout()
    save_fig(fig, "fig10_bulk_fracture_toughness")


# ============================================================================
# FIGURE 11: Interface Toughness vs Temperature
# ============================================================================
def fig11_interface_toughness():
    df = pd.read_csv(os.path.join(CSV_DIR, "04_interface_toughness_vs_T.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for intf, color, marker_I, marker_II in [
        ('YSZ_GDC', COLORS['interface1'], 'o', 's'),
        ('GDC_LSCF', COLORS['interface2'], '^', 'D')
    ]:
        mask = df['Interface'] == intf
        label_clean = intf.replace('_', '|')
        ax1.plot(df[mask]['Temperature_C'], df[mask]['Gc_I_Jm2'],
                f'{marker_I}-', color=color, label=f'{label_clean} Mode I', markersize=6)
        ax1.plot(df[mask]['Temperature_C'], df[mask]['Gc_II_Jm2'],
                f'{marker_II}--', color=color, label=f'{label_clean} Mode II', markersize=6)

    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel(r'Interface toughness (J/m²)')
    ax1.set_title(r'(a) Interface Toughness: $G_{c,I}$ and $G_{c,II}$ vs $T$')
    ax1.legend(fontsize=8)
    ax1.axvspan(600, 800, alpha=0.08, color='red')

    # (b) Mode mixity angle
    for intf, color, marker in [
        ('YSZ_GDC', COLORS['interface1'], 'o'),
        ('GDC_LSCF', COLORS['interface2'], 's')
    ]:
        mask = df['Interface'] == intf
        label_clean = intf.replace('_', '|')
        ax2.plot(df[mask]['Temperature_C'], df[mask]['Gc_II_over_Gc_I'],
                f'{marker}-', color=color, label=f'{label_clean}', markersize=6)

    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel(r'$G_{c,II} / G_{c,I}$ ratio')
    ax2.set_title(r'(b) Mode II/Mode I Toughness Ratio')
    ax2.legend()
    ax2.axvspan(600, 800, alpha=0.08, color='red')

    plt.tight_layout()
    save_fig(fig, "fig11_interface_toughness_vs_T")


# ============================================================================
# FIGURE 12: BK Mixed-Mode Failure Envelope
# ============================================================================
def fig12_mixed_mode_envelope():
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # BK criterion: Gc = GcI + (GcII - GcI) * (GII/(GI+GII))^eta
    eta_values = [1.0, 1.5, 2.0, 2.1, 2.5, 3.0]
    GcI, GcII = 3.5, 5.5  # GDC|LSCF RT values

    mode_mix = np.linspace(0, 1, 100)  # GII/(GI+GII)

    colors_eta = plt.cm.viridis(np.linspace(0.1, 0.9, len(eta_values)))

    for eta, color in zip(eta_values, colors_eta):
        Gc_mixed = GcI + (GcII - GcI) * mode_mix ** eta
        # Convert to G_I and G_II components
        GI_comp = Gc_mixed * (1 - mode_mix)
        GII_comp = Gc_mixed * mode_mix
        style = '-' if eta != 2.1 else '-'
        lw = 1.5 if eta != 2.1 else 3.0
        ax.plot(GI_comp, GII_comp, style, color=color, linewidth=lw,
                label=rf'$\eta$ = {eta}')

    # Mark pure mode points
    ax.plot(GcI, 0, 'ko', markersize=10, zorder=5)
    ax.annotate(f'$G_{{c,I}}$ = {GcI} J/m²', xy=(GcI, 0), xytext=(GcI + 0.3, 0.3),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
    ax.plot(0, GcII, 'ks', markersize=10, zorder=5)
    ax.annotate(f'$G_{{c,II}}$ = {GcII} J/m²', xy=(0, GcII), xytext=(0.3, GcII + 0.3),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

    # Safe/fail regions
    ax.fill_between(np.linspace(0, GcI, 50), 0,
                    GcI + (GcII - GcI) * (np.linspace(0, 1, 50)) ** 2.1 * np.linspace(0, 1, 50),
                    alpha=0.05, color='green')

    ax.set_xlabel(r'Mode I component, $G_I$ (J/m²)')
    ax.set_ylabel(r'Mode II component, $G_{II}$ (J/m²)')
    ax.set_title(r'BK Mixed-Mode Failure Envelope (GDC|LSCF, RT)')
    ax.legend(title=r'BK exponent $\eta$')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7)

    save_fig(fig, "fig12_bk_mixed_mode_envelope")


# ============================================================================
# FIGURE 13: Weibull Distribution
# ============================================================================
def fig13_weibull_distribution():
    df = pd.read_csv(os.path.join(CSV_DIR, "04_weibull_statistics.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # (a) Weibull PDF for each material at RT
    rt_data = df[df['Temperature_C'] == 25]
    sigma_range = np.linspace(0, 500, 500)

    for _, row in rt_data.iterrows():
        mat = row['Material']
        m = row['Weibull_Modulus_m']
        sigma_0 = row['Characteristic_Strength_MPa']
        sigma_th = row['Threshold_Stress_MPa']

        # 3-parameter Weibull PDF
        x = sigma_range - sigma_th
        x = np.maximum(x, 0)
        pdf = (m / sigma_0) * (x / sigma_0) ** (m - 1) * np.exp(-(x / sigma_0) ** m)

        color = COLORS.get(mat.split('_')[0] if '_' not in mat else mat, 'gray')
        if mat == 'LSCF_dense':
            color = COLORS['LSCF']
        elif mat == 'LSCF_porous':
            color = COLORS['LSCF_porous']

        ax1.plot(sigma_range, pdf, '-', color=color, label=f'{mat} (m={m})', linewidth=2)

    ax1.set_xlabel(r'Stress, $\sigma$ (MPa)')
    ax1.set_ylabel(r'Probability density, $f(\sigma)$')
    ax1.set_title('(a) Weibull Strength Distribution (RT)')
    ax1.legend(fontsize=8)

    # (b) Weibull modulus comparison RT vs 800°C
    materials = df['Material'].unique()
    x_pos = np.arange(len(materials))
    width = 0.35

    m_RT = df[df['Temperature_C'] == 25]['Weibull_Modulus_m'].values
    m_800 = df[df['Temperature_C'] == 800]['Weibull_Modulus_m'].values

    bars1 = ax2.bar(x_pos - width/2, m_RT, width, label='RT (25°C)',
                    color=[COLORS.get(m.split('_')[0], 'gray') for m in materials], alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, m_800, width, label='800°C',
                    color=[COLORS.get(m.split('_')[0], 'gray') for m in materials], alpha=0.4)

    ax2.set_xlabel('Material')
    ax2.set_ylabel('Weibull modulus, $m$')
    ax2.set_title('(b) Weibull Modulus: RT vs Operating Temperature')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(materials, rotation=30, ha='right')
    ax2.legend()

    plt.tight_layout()
    save_fig(fig, "fig13_weibull_distribution")


# ============================================================================
# FIGURE 14: Curvature Evolution
# ============================================================================
def fig14_curvature_evolution():
    df = pd.read_csv(os.path.join(CSV_DIR, "05_curvature_evolution.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))

    T = df['Temperature_C']

    # Model predictions
    ax.plot(T, df['Kappa_thermal_only_1_per_m'] * 1000, '--',
            color=COLORS['YSZ'], linewidth=2, label='Model: thermal only')
    ax.plot(T, df['Kappa_thermal_plus_chemical_1_per_m'] * 1000, '-',
            color=COLORS['model'], linewidth=2.5, label='Model: thermal + chemical')

    # DIC "measurements"
    ax.errorbar(T, df['DIC_measured_kappa_1_per_m'] * 1000,
               yerr=df['DIC_uncertainty_1_per_m'] * 1000,
               fmt='o', color=COLORS['experiment'], markersize=4,
               capsize=2, label='DIC measurement', alpha=0.7, zorder=3)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(r'Curvature, $\kappa$ ($\times 10^{-3}$ m$^{-1}$)')
    ax.set_title(r'Global Curvature Evolution $\kappa(T)$ During Cooling')
    ax.legend()
    ax.invert_xaxis()

    # Add arrow for cooling direction
    ax.annotate('Cooling', xy=(200, 0.05), xytext=(500, 0.05),
                fontsize=10, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    save_fig(fig, "fig14_curvature_evolution")


# ============================================================================
# FIGURE 15: Crack Path Morphology
# ============================================================================
def fig15_crack_path():
    df = pd.read_csv(os.path.join(CSV_DIR, "05_crack_path_morphology.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # (a) YSZ|GDC interface
    ax1 = axes[0]
    sc1 = ax1.scatter(df['X_um'], df['Y_YSZ_GDC_interface_um'],
                      c=df['Phase_field_d_YSZ_GDC'], cmap='hot_r',
                      s=15, vmin=0, vmax=1, zorder=3)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Nominal interface')
    ax1.fill_between(df['X_um'], -0.5, 0, alpha=0.08, color=COLORS['YSZ'])
    ax1.fill_between(df['X_um'], 0, 0.5, alpha=0.08, color=COLORS['GDC'])
    ax1.text(28, -0.35, 'YSZ', fontsize=9, color=COLORS['YSZ'], fontweight='bold')
    ax1.text(28, 0.30, 'GDC', fontsize=9, color=COLORS['GDC'], fontweight='bold')
    ax1.set_ylabel(r'$y$ - position (µm)')
    ax1.set_title(r'(a) YSZ|GDC Interface: Crack Path & Phase-Field Damage $d$')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(-0.6, 0.6)
    plt.colorbar(sc1, ax=ax1, label='Phase-field damage $d$', shrink=0.8)

    # (b) GDC|LSCF interface
    ax2 = axes[1]
    sc2 = ax2.scatter(df['X_um'], df['Y_GDC_LSCF_interface_um'],
                      c=df['Phase_field_d_GDC_LSCF'], cmap='hot_r',
                      s=15, vmin=0, vmax=1, zorder=3)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Nominal interface')
    ax2.fill_between(df['X_um'], -0.8, 0, alpha=0.08, color=COLORS['GDC'])
    ax2.fill_between(df['X_um'], 0, 0.8, alpha=0.08, color=COLORS['LSCF'])
    ax2.text(28, -0.55, 'GDC', fontsize=9, color=COLORS['GDC'], fontweight='bold')
    ax2.text(28, 0.50, 'LSCF', fontsize=9, color=COLORS['LSCF'], fontweight='bold')
    ax2.set_xlabel(r'Crack path, $x$ (µm)')
    ax2.set_ylabel(r'$y$ - position (µm)')
    ax2.set_title(r'(b) GDC|LSCF Interface: Crack Path & Phase-Field Damage $d$')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_ylim(-0.8, 0.8)
    plt.colorbar(sc2, ax=ax2, label='Phase-field damage $d$', shrink=0.8)

    plt.tight_layout()
    save_fig(fig, "fig15_crack_path_morphology")


# ============================================================================
# FIGURE 16: Comprehensive UMAT Input Summary
# ============================================================================
def fig16_umat_summary_heatmap():
    df = pd.read_csv(os.path.join(CSV_DIR, "02_umat_full_input.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    materials = ['YSZ', 'GDC', 'LSCF_dense', 'LSCF_porous']
    mat_labels = ['8YSZ', 'GDC', 'LSCF\n(dense)', 'LSCF\n(porous)']

    # (a) E heatmap
    E_matrix = []
    for mat in materials:
        mask = df['Material'] == mat
        E_matrix.append(df[mask]['E_GPa'].values)
    E_matrix = np.array(E_matrix)

    im1 = axes[0].imshow(E_matrix, aspect='auto', cmap='YlOrRd_r')
    axes[0].set_xticks(range(len(T_grid)))
    axes[0].set_xticklabels(T_grid, fontsize=7, rotation=45)
    axes[0].set_yticks(range(len(materials)))
    axes[0].set_yticklabels(mat_labels, fontsize=8)
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_title(r"(a) Young's Modulus $E$ (GPa)")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    # Add text annotations
    for i in range(len(materials)):
        for j in range(len(T_grid)):
            axes[0].text(j, i, f'{E_matrix[i,j]:.0f}', ha='center', va='center', fontsize=6)

    # (b) Lame Lambda heatmap
    L_matrix = []
    for mat in materials:
        mask = df['Material'] == mat
        L_matrix.append(df[mask]['Lambda_GPa'].values)
    L_matrix = np.array(L_matrix)

    im2 = axes[1].imshow(L_matrix, aspect='auto', cmap='YlGnBu')
    axes[1].set_xticks(range(len(T_grid)))
    axes[1].set_xticklabels(T_grid, fontsize=7, rotation=45)
    axes[1].set_yticks(range(len(materials)))
    axes[1].set_yticklabels(mat_labels, fontsize=8)
    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_title(r'(b) Lamé $\lambda$ (GPa)')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    for i in range(len(materials)):
        for j in range(len(T_grid)):
            axes[1].text(j, i, f'{L_matrix[i,j]:.0f}', ha='center', va='center', fontsize=6)

    # (c) Shear modulus heatmap
    M_matrix = []
    for mat in materials:
        mask = df['Material'] == mat
        M_matrix.append(df[mask]['Mu_GPa'].values)
    M_matrix = np.array(M_matrix)

    im3 = axes[2].imshow(M_matrix, aspect='auto', cmap='PuBuGn')
    axes[2].set_xticks(range(len(T_grid)))
    axes[2].set_xticklabels(T_grid, fontsize=7, rotation=45)
    axes[2].set_yticks(range(len(materials)))
    axes[2].set_yticklabels(mat_labels, fontsize=8)
    axes[2].set_xlabel('Temperature (°C)')
    axes[2].set_title(r'(c) Shear Modulus $\mu$ (GPa)')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    for i in range(len(materials)):
        for j in range(len(T_grid)):
            axes[2].text(j, i, f'{M_matrix[i,j]:.0f}', ha='center', va='center', fontsize=6)

    plt.tight_layout()
    save_fig(fig, "fig16_umat_input_heatmaps")


# ============================================================================
# FIGURE 17: Parametric Sweep Overview
# ============================================================================
def fig17_parametric_sweep():
    df = pd.read_csv(os.path.join(CSV_DIR, "06_parametric_sweep_config.csv"))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sweep_names = df['Sweep_Name'].unique()
    colors_sweep = plt.cm.Set2(np.linspace(0, 1, len(sweep_names)))

    for i, name in enumerate(sweep_names):
        mask = df['Sweep_Name'] == name
        subset = df[mask]
        ax.barh([f"S{j+1}: {name}\n({row['Parameter_Varied']}={row['Value']})"
                 for j, (_, row) in enumerate(subset.iterrows())],
                subset['Expected_CPU_hours'], color=colors_sweep[i], alpha=0.8,
                label=name)

    ax.set_xlabel('Expected CPU hours')
    ax.set_title('Parametric Sweep Configuration: Estimated Computational Cost')
    ax.legend(loc='lower right', fontsize=7, ncol=2)

    plt.tight_layout()
    save_fig(fig, "fig17_parametric_sweep_overview")


# ============================================================================
# FIGURE 18: Energy Functional Decomposition Schematic
# ============================================================================
def fig18_energy_functional():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title - simplified LaTeX for matplotlib compatibility
    ax.text(5, 7.5,
            r'$\Pi(\mathbf{u}, d) = \int_\Omega [g(d)\Psi_0^+ + \Psi_0^-] d\Omega'
            r' + \int_\Omega G_{c,b}\gamma(d,\nabla d) d\Omega'
            r' + \int_{\Gamma_c} \phi(\Delta_n, \Delta_t) d\Gamma$',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', alpha=0.9))

    # Connection boxes
    boxes = [
        (1.5, 4.5, 'Elastic Energy\n(UMAT)', '#2166AC', [
            r'$E(T), \nu(T)$ → Sec. 2',
            r'$\alpha(T)\Delta T$ → Sec. 2',
            r'$\beta\Delta\delta$ → Sec. 3'
        ]),
        (5.0, 4.5, 'Phase-Field\nFracture', '#B2182B', [
            r'$G_{c,b}(T)$ → Sec. 4',
            r'$l_0$ → Sec. 1 (mesh)',
            r'$g(d) = (1-d)^2$'
        ]),
        (8.5, 4.5, 'Cohesive Interface\n(UEL)', '#1B7837', [
            r'$G_{c,I}, G_{c,II}$ → Sec. 4',
            r'$T_{max,n}, T_{max,t}$ → Sec. 4',
            r'$\eta_{BK}$ → Sec. 4'
        ]),
    ]

    for x, y, title, color, items in boxes:
        rect = Rectangle((x - 1.3, y - 1.8), 2.6, 3.0,
                         linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.1)
        ax.add_patch(rect)
        ax.text(x, y + 0.8, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
        for j, item in enumerate(items):
            ax.text(x, y - 0.1 - j * 0.5, item, ha='center', va='center', fontsize=8)

    # Validation box at bottom
    rect_val = Rectangle((2.0, 0.5), 6.0, 1.5,
                         linewidth=1.5, edgecolor='#D6604D', facecolor='#D6604D', alpha=0.1)
    ax.add_patch(rect_val)
    ax.text(5, 1.6, 'Experimental Validation (Sec. 5)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#D6604D')
    ax.text(5, 1.0, r'$\kappa(T)$ curvature  |  Crack path overlay  |  Delamination onset ($T_c$, $p_{O_2,c}$)',
            ha='center', va='center', fontsize=8.5)

    # Arrows from functional terms to validation
    for x in [1.5, 5.0, 8.5]:
        ax.annotate('', xy=(x, 2.0), xytext=(x, 2.7),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.set_title('Dataset-to-Simulation Mapping: Governing Energy Functional',
                fontsize=12, fontweight='bold', pad=20)

    save_fig(fig, "fig18_energy_functional_schematic")


# ============================================================================
# FIGURE 19: Cohesive Traction-Separation Law
# ============================================================================
def fig19_traction_separation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Xu-Needleman potential
    # T_n = (phi_n / delta_n) * (Delta_n / delta_n) * exp(-Delta_n/delta_n) * exp(-Delta_t^2/delta_t^2)
    delta_n = np.linspace(0, 0.2, 200)  # µm
    delta_t = np.linspace(-0.3, 0.3, 200)

    # Parameters for YSZ|GDC
    phi_n = 5.0  # J/m²
    delta_n_cr = 0.04  # µm
    T_max_n = phi_n / (delta_n_cr * np.exp(1)) * 1e-6  # Convert properly

    # (a) Normal traction-separation
    for intf, GcI, d_cr, color, label in [
        ('YSZ|GDC', 5.0, 0.04, COLORS['interface1'], 'YSZ|GDC'),
        ('GDC|LSCF', 3.5, 0.04, COLORS['interface2'], 'GDC|LSCF')
    ]:
        T_n_max = GcI / (d_cr * np.exp(1))  # peak traction
        tn = T_n_max * (delta_n / d_cr) * np.exp(1 - delta_n / d_cr)
        ax1.plot(delta_n, tn, '-', color=color, linewidth=2, label=f'{label} (RT)')
        ax1.axhline(y=T_n_max, color=color, linestyle=':', alpha=0.3)

    ax1.set_xlabel(r'Normal separation, $\Delta_n$ (µm)')
    ax1.set_ylabel(r'Normal traction, $T_n$ (J/m²/µm)')
    ax1.set_title(r'(a) Mode I: $T_n(\Delta_n)$')
    ax1.legend()

    # (b) Shear traction-separation
    for intf, GcII, d_cr, color, label in [
        ('YSZ|GDC', 8.0, 0.08, COLORS['interface1'], 'YSZ|GDC'),
        ('GDC|LSCF', 5.5, 0.08, COLORS['interface2'], 'GDC|LSCF')
    ]:
        T_t_max = GcII / (d_cr * np.sqrt(np.exp(1) / 2))
        tt = T_t_max * (delta_t / d_cr) * np.exp(0.5 - 0.5 * (delta_t / d_cr) ** 2)
        ax2.plot(delta_t, tt, '-', color=color, linewidth=2, label=f'{label} (RT)')

    ax2.set_xlabel(r'Tangential separation, $\Delta_t$ (µm)')
    ax2.set_ylabel(r'Shear traction, $T_t$ (J/m²/µm)')
    ax2.set_title(r'(b) Mode II: $T_t(\Delta_t)$')
    ax2.legend()

    plt.tight_layout()
    save_fig(fig, "fig19_traction_separation_law")


# ============================================================================
# FIGURE 20: Complete Dataset Summary Dashboard
# ============================================================================
def fig20_summary_dashboard():
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # Panel 1: E(T)
    ax1 = fig.add_subplot(gs[0, 0])
    df_E = pd.read_csv(os.path.join(CSV_DIR, "02_youngs_modulus.csv"))
    for col, color, label in [('E_YSZ_GPa', COLORS['YSZ'], 'YSZ'),
                               ('E_GDC_GPa', COLORS['GDC'], 'GDC'),
                               ('E_LSCF_dense_GPa', COLORS['LSCF'], 'LSCF')]:
        ax1.plot(df_E['Temperature_C'], df_E[col], 'o-', color=color, label=label, markersize=4)
    ax1.set_xlabel('T (°C)')
    ax1.set_ylabel('E (GPa)')
    ax1.set_title(r"$E(T)$", fontsize=10)
    ax1.legend(fontsize=7)

    # Panel 2: alpha(T)
    ax2 = fig.add_subplot(gs[0, 1])
    df_a = pd.read_csv(os.path.join(CSV_DIR, "02_thermal_expansion.csv"))
    for col, color, label in [('alpha_YSZ_1e6_perK', COLORS['YSZ'], 'YSZ'),
                               ('alpha_GDC_1e6_perK', COLORS['GDC'], 'GDC'),
                               ('alpha_LSCF_1e6_perK', COLORS['LSCF'], 'LSCF')]:
        ax2.plot(df_a['Temperature_C'], df_a[col], 'o-', color=color, label=label, markersize=4)
    ax2.set_xlabel('T (°C)')
    ax2.set_ylabel(r'$\alpha$ ($10^{-6}$/K)')
    ax2.set_title(r'$\alpha(T)$', fontsize=10)
    ax2.legend(fontsize=7)

    # Panel 3: Gc(T)
    ax3 = fig.add_subplot(gs[0, 2])
    df_Gc = pd.read_csv(os.path.join(CSV_DIR, "04_bulk_fracture_toughness.csv"))
    for col, color, label in [('Gc_YSZ_Jm2', COLORS['YSZ'], 'YSZ'),
                               ('Gc_GDC_Jm2', COLORS['GDC'], 'GDC'),
                               ('Gc_LSCF_Jm2', COLORS['LSCF'], 'LSCF')]:
        ax3.plot(df_Gc['Temperature_C'], df_Gc[col], 'o-', color=color, label=label, markersize=4)
    ax3.set_xlabel('T (°C)')
    ax3.set_ylabel(r'$G_c$ (J/m²)')
    ax3.set_title(r'$G_{c,b}(T)$', fontsize=10)
    ax3.legend(fontsize=7)

    # Panel 4: Interface toughness
    ax4 = fig.add_subplot(gs[1, 0])
    df_intf = pd.read_csv(os.path.join(CSV_DIR, "04_interface_toughness_vs_T.csv"))
    for intf, color in [('YSZ_GDC', COLORS['interface1']), ('GDC_LSCF', COLORS['interface2'])]:
        mask = df_intf['Interface'] == intf
        ax4.plot(df_intf[mask]['Temperature_C'], df_intf[mask]['Gc_I_Jm2'],
                'o-', color=color, label=f'{intf.replace("_","|")} I', markersize=4)
        ax4.plot(df_intf[mask]['Temperature_C'], df_intf[mask]['Gc_II_Jm2'],
                's--', color=color, label=f'{intf.replace("_","|")} II', markersize=4)
    ax4.set_xlabel('T (°C)')
    ax4.set_ylabel(r'$G_c$ (J/m²)')
    ax4.set_title('Interface Toughness', fontsize=10)
    ax4.legend(fontsize=6)

    # Panel 5: Chemical expansion
    ax5 = fig.add_subplot(gs[1, 1])
    df_lscf = pd.read_csv(os.path.join(CSV_DIR, "03_lscf_chemical_expansion.csv"))
    ax5.plot(df_lscf['Delta_delta'], df_lscf['Epsilon_chem_11'] * 100, 'o-',
            color=COLORS['interface1'], label=r'$\varepsilon^{ch}_{11}$', markersize=4)
    ax5.plot(df_lscf['Delta_delta'], df_lscf['Epsilon_chem_33'] * 100, 's-',
            color=COLORS['interface2'], label=r'$\varepsilon^{ch}_{33}$', markersize=4)
    ax5.set_xlabel(r'$\Delta\delta$')
    ax5.set_ylabel(r'$\varepsilon^{ch}$ (%)')
    ax5.set_title('LSCF Chemical Expansion', fontsize=10)
    ax5.legend(fontsize=7)

    # Panel 6: CTE mismatch
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(df_a['Temperature_C'], df_a['CTE_Mismatch_YSZ_GDC'],
            'o-', color=COLORS['interface1'], label='GDC-YSZ', markersize=4)
    ax6.plot(df_a['Temperature_C'], df_a['CTE_Mismatch_GDC_LSCF'],
            's-', color=COLORS['interface2'], label='LSCF-GDC', markersize=4)
    ax6.set_xlabel('T (°C)')
    ax6.set_ylabel(r'$\Delta\alpha$ ($10^{-6}$/K)')
    ax6.set_title('CTE Mismatch', fontsize=10)
    ax6.legend(fontsize=7)

    # Panel 7: Mesh sensitivity
    ax7 = fig.add_subplot(gs[2, 0])
    df_mesh = pd.read_csv(os.path.join(CSV_DIR, "01_mesh_sensitivity.csv"))
    ax7.semilogx(df_mesh['Mesh_Size_um'], df_mesh['Normalized_Energy_Release'],
                'o-', color=COLORS['YSZ'], markersize=5)
    ax7.set_xlabel('h (µm)')
    ax7.set_ylabel(r'$G/G_{conv}$')
    ax7.set_title('Mesh Objectivity', fontsize=10)
    ax7.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    # Panel 8: Curvature
    ax8 = fig.add_subplot(gs[2, 1])
    df_curv = pd.read_csv(os.path.join(CSV_DIR, "05_curvature_evolution.csv"))
    ax8.plot(df_curv['Temperature_C'], df_curv['Kappa_thermal_plus_chemical_1_per_m'] * 1000,
            '-', color=COLORS['model'], linewidth=2, label='Model')
    ax8.plot(df_curv['Temperature_C'], df_curv['DIC_measured_kappa_1_per_m'] * 1000,
            'o', color=COLORS['experiment'], markersize=3, label='DIC', alpha=0.7)
    ax8.set_xlabel('T (°C)')
    ax8.set_ylabel(r'$\kappa$ ($\times 10^{-3}$ m$^{-1}$)')
    ax8.set_title(r'Curvature $\kappa(T)$', fontsize=10)
    ax8.legend(fontsize=7)
    ax8.invert_xaxis()

    # Panel 9: Nonstoichiometry profile
    ax9 = fig.add_subplot(gs[2, 2])
    df_prof = pd.read_csv(os.path.join(CSV_DIR, "03_nonstoichiometry_profile.csv"))
    cmap_t = {600: '#2166AC', 700: '#B2182B', 800: '#1B7837'}
    for T in [600, 700, 800]:
        mask = df_prof['Temperature_C'] == T
        ax9.plot(df_prof[mask]['z_um_from_interface'], df_prof[mask]['Delta_delta_local'],
                '-', color=cmap_t[T], label=f'{T}°C', linewidth=1.5)
    ax9.set_xlabel('z from interface (µm)')
    ax9.set_ylabel(r'$\Delta\delta(z)$')
    ax9.set_title(r'$\Delta\delta$ Profile', fontsize=10)
    ax9.legend(fontsize=7)

    fig.suptitle('Abaqus Simulation Dataset: Complete Summary Dashboard\n'
                 'YSZ/GDC/LSCF Mixed-Mode Fracture Analysis',
                 fontsize=14, fontweight='bold', y=1.02)

    save_fig(fig, "fig20_summary_dashboard")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    T_grid = np.array([25, 100, 200, 300, 400, 500, 600, 700, 800])

    print("=" * 60)
    print("  FIGURE GENERATOR - Publication Quality")
    print("=" * 60)

    fig01_layer_schematic()
    fig02_mesh_sensitivity()
    fig03_rve_convergence()
    fig04_youngs_modulus()
    fig05_poissons_ratio()
    fig06_thermal_expansion()
    fig07_gdc_chemical_expansion()
    fig08_lscf_chemical_expansion()
    fig09_nonstoichiometry_profile()
    fig10_bulk_fracture_toughness()
    fig11_interface_toughness()
    fig12_mixed_mode_envelope()
    fig13_weibull_distribution()
    fig14_curvature_evolution()
    fig15_crack_path()
    fig16_umat_summary_heatmap()
    fig17_parametric_sweep()
    fig18_energy_functional()
    fig19_traction_separation()
    fig20_summary_dashboard()

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {FIG_DIR}")
    print("=" * 60)
    fig_files = sorted([f for f in os.listdir(FIG_DIR)])
    for f in fig_files:
        size = os.path.getsize(os.path.join(FIG_DIR, f))
        print(f"  {f:55s} ({size:>10,d} bytes)")
    print(f"\n  Total: {len(fig_files)} figure files")
