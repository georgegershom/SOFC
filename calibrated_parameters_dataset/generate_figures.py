#!/usr/bin/env python3
"""
Phase-Field Fracture Modeling Dataset Visualization
Generates comprehensive figures for calibrated parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Paths
csv_dir = Path('csv_files')
fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

# Color palette
colors = {
    'YSZ': '#1f77b4',
    'GDC': '#ff7f0e', 
    'LSCF': '#2ca02c',
    'Interface': '#d62728'
}

def create_figure_1_main_parameters():
    """Figure 1: Main Calibrated Parameters Overview"""
    df = pd.read_csv(csv_dir / '01_main_calibrated_parameters.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Main Calibrated Parameters for Phase-Field Fracture Model', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Fracture Energies
    ax = axes[0, 0]
    fracture_params = df[df['Parameter'].str.contains('Fracture Energy')]
    materials = ['LSCF', 'YSZ', 'GDC']
    values = fracture_params['Calibrated_Value'].values[:3]
    lower = fracture_params['Lower_Bound'].values[:3]
    upper = fracture_params['Upper_Bound'].values[:3]
    errors = np.array([[values[i] - lower[i], upper[i] - values[i]] for i in range(3)]).T
    
    bars = ax.bar(materials, values, yerr=errors, capsize=5, 
                   color=[colors[m] for m in materials], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Fracture Energy $G_c$ (J/m²)', fontweight='bold')
    ax.set_title('(a) Bulk Fracture Energies', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Phase-field and Regularization Parameters
    ax = axes[0, 1]
    param_names = ['$l_0$ min', '$l_0$ max', '$l$', '$\\eta$']
    param_values = [5, 20, 500, 2.1]  # nm, nm, nm (converted from µm), dimensionless
    param_colors = ['#ff7f0e', '#ff7f0e', '#1f77b4', '#2ca02c']
    
    bars = ax.barh(param_names, param_values, color=param_colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Value', fontweight='bold')
    ax.set_title('(b) Phase-Field Length Scales & BK Exponent', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, param_values)):
        if i < 2:
            unit = ' nm'
        elif i == 2:
            unit = ' nm'
        else:
            unit = ''
        ax.text(val + 10, bar.get_y() + bar.get_height()/2, 
                f'{val}{unit}', va='center', fontweight='bold')
    
    # Plot 3: Interface Adhesion Range
    ax = axes[1, 0]
    conditions = ['Min\n(Sr-seg.)', 'Baseline', 'Max\n(Interdiff.)']
    adhesion_values = [0.2, 1.5, 3.2]
    bar_colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(conditions, adhesion_values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Interface Adhesion $\\Gamma_i$ (J/m²)', fontweight='bold')
    ax.set_title('(c) Interface Adhesion Energy Range', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Penalty Parameter Range
    ax = axes[1, 1]
    x = np.logspace(2, 4, 100)
    y = 1 / x  # Inverse relationship for demonstration
    
    ax.fill_between(x, 0, y, alpha=0.3, color='#1f77b4', label='Acceptable Range')
    ax.axvline(1000, color='red', linestyle='--', linewidth=2, label='Recommended: 1000 GPa/m')
    ax.set_xscale('log')
    ax.set_xlabel('Penalty Parameter $\\beta_{pen}$ (GPa/m)', fontweight='bold')
    ax.set_ylabel('Relative Interface Compliance', fontweight='bold')
    ax.set_title('(d) Penalty Parameter Selection', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_01_Main_Parameters.png', bbox_inches='tight')
    print("✓ Generated Figure 1: Main Parameters")
    plt.close()

def create_figure_2_interface_properties():
    """Figure 2: Interface Fracture Properties"""
    df = pd.read_csv(csv_dir / '02_interface_fracture_properties.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Interface Fracture Properties: YSZ/GDC and GDC/LSCF', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: YSZ/GDC Interface Energy
    ax = axes[0, 0]
    ysz_gdc = df[df['Interface'] == 'YSZ/GDC'].head(7)
    conditions = ['Baseline', 'Interdiff.', '100nm', '500nm', '1µm']
    values = [2.15, 2.85, 2.1, 2.3, 2.6]
    colors_bar = ['#1f77b4', '#2ca02c', '#ff7f0e', '#ff7f0e', '#ff7f0e']
    
    bars = ax.bar(range(len(conditions)), values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('$G_{c,int}$ (J/m²)', fontweight='bold')
    ax.set_title('(a) YSZ/GDC Interface: Effect of Interdiffusion & Thickness', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3.5])
    
    # Plot 2: GDC/LSCF Degradation
    ax = axes[0, 1]
    gdc_lscf = df[df['Interface'] == 'GDC/LSCF'].tail(4)
    time_points = [0, 100, 500, 1000]
    gc_values = gdc_lscf['Gc_int'].values
    
    ax.plot(time_points, gc_values, 'o-', linewidth=2.5, markersize=8, 
            color='#d62728', label='$G_{c,int}$')
    ax.fill_between(time_points, gdc_lscf['Gc_min'].values, gdc_lscf['Gc_max'].values, 
                     alpha=0.2, color='#d62728')
    ax.set_xlabel('Operation Time (h)', fontweight='bold')
    ax.set_ylabel('$G_{c,int}$ (J/m²)', fontweight='bold')
    ax.set_title('(b) GDC/LSCF Degradation: Sr-Segregation Effect', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Critical Strength Comparison
    ax = axes[1, 0]
    interfaces = ['YSZ/GDC\nBaseline', 'YSZ/GDC\nInterdiff.', 
                  'GDC/LSCF\nFresh', 'GDC/LSCF\n1000h']
    sigma_values = [222.5, 240.0, 200.0, 105.0]
    colors_sigma = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    bars = ax.bar(interfaces, sigma_values, color=colors_sigma, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Critical Strength $\\sigma_{max}$ (MPa)', fontweight='bold')
    ax.set_title('(c) Interface Critical Strength Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Characteristic Length vs Gc
    ax = axes[1, 1]
    gc_all = df['Gc_int'].values
    length_all = df['Characteristic_Length'].values
    interface_type = ['YSZ/GDC' if 'YSZ' in i else 'GDC/LSCF' for i in df['Interface']]
    
    for itype, color in [('YSZ/GDC', '#1f77b4'), ('GDC/LSCF', '#d62728')]:
        mask = [it == itype for it in interface_type]
        ax.scatter(np.array(gc_all)[mask], np.array(length_all)[mask], 
                  s=100, alpha=0.6, color=color, label=itype, edgecolors='black')
    
    ax.set_xlabel('Fracture Energy $G_{c,int}$ (J/m²)', fontweight='bold')
    ax.set_ylabel('Characteristic Length (µm)', fontweight='bold')
    ax.set_title('(d) Process Zone Size vs Fracture Energy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_02_Interface_Properties.png', bbox_inches='tight')
    print("✓ Generated Figure 2: Interface Properties")
    plt.close()

def create_figure_3_lscf_nonstoichiometry():
    """Figure 3: LSCF Non-stoichiometry and Chemical Strain"""
    df = pd.read_csv(csv_dir / '03_LSCF_nonstoichiometry_data.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LSCF Cathode: Non-stoichiometry and Chemical Expansion', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Delta-delta vs Temperature (different pO2)
    ax = axes[0, 0]
    pO2_levels = df['pO2_atm'].unique()
    colors_po2 = plt.cm.viridis(np.linspace(0, 1, len(pO2_levels)))
    
    for i, po2 in enumerate(pO2_levels):
        data = df[df['pO2_atm'] == po2]
        ax.plot(data['Temperature_C'], data['Delta_delta'], 'o-', 
               linewidth=2, markersize=6, label=f'$pO_2$ = {po2:.0e} atm', 
               color=colors_po2[i])
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Non-stoichiometry $\\Delta\\delta$', fontweight='bold')
    ax.set_title('(a) LSCF Non-stoichiometry vs Temperature', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Delta-delta vs pO2 (different temperatures)
    ax = axes[0, 1]
    temps = df['Temperature_C'].unique()
    colors_temp = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))
    
    for i, temp in enumerate(temps):
        data = df[df['Temperature_C'] == temp]
        ax.plot(data['log_pO2'], data['Delta_delta'], 's-', 
               linewidth=2, markersize=6, label=f'{temp}°C', 
               color=colors_temp[i])
    
    ax.set_xlabel('log($pO_2$/atm)', fontweight='bold')
    ax.set_ylabel('Non-stoichiometry $\\Delta\\delta$', fontweight='bold')
    ax.set_title('(b) LSCF Non-stoichiometry vs Oxygen Partial Pressure', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Chemical Strain Components
    ax = axes[1, 0]
    data_800C = df[df['Temperature_C'] == 800]
    
    ax.plot(data_800C['Delta_delta'], data_800C['Chemical_Strain_xx'] * 100, 
           'o-', linewidth=2.5, markersize=7, label='$\\varepsilon_{11}$ (in-plane)', color='#1f77b4')
    ax.plot(data_800C['Delta_delta'], data_800C['Chemical_Strain_zz'] * 100, 
           's-', linewidth=2.5, markersize=7, label='$\\varepsilon_{33}$ (out-of-plane)', color='#d62728')
    
    ax.set_xlabel('Non-stoichiometry $\\Delta\\delta$', fontweight='bold')
    ax.set_ylabel('Chemical Strain (%)', fontweight='bold')
    ax.set_title('(c) Anisotropic Chemical Strain at 800°C', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: 3D Surface plot simulation (heatmap)
    ax = axes[1, 1]
    pivot_data = df.pivot_table(values='Delta_delta', 
                                 index='Temperature_C', 
                                 columns='log_pO2')
    
    im = ax.contourf(pivot_data.columns, pivot_data.index, pivot_data.values, 
                     levels=20, cmap='RdYlBu_r')
    ax.set_xlabel('log($pO_2$/atm)', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('(d) LSCF Non-stoichiometry Map', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\\Delta\\delta$', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_03_LSCF_Nonstoichiometry.png', bbox_inches='tight')
    print("✓ Generated Figure 3: LSCF Non-stoichiometry")
    plt.close()

def create_figure_4_gdc_expansion():
    """Figure 4: GDC Chemical Expansion Dataset"""
    df = pd.read_csv(csv_dir / '04_GDC_chemical_expansion_22delta_4T.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GDC Interlayer: Chemical Expansion Dataset (22δ × 4T)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Chemical expansion vs delta at different temperatures
    ax = axes[0, 0]
    temps = ['600C', '700C', '800C', '900C']
    colors_t = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for temp, color in zip(temps, colors_t):
        col_name = f'T_{temp}'
        ax.plot(df['delta'], df[col_name] * 1e6, 'o-', 
               linewidth=2, markersize=5, label=f'{temp[:-1]}°C', color=color)
    
    ax.set_xlabel('Non-stoichiometry $\\delta$', fontweight='bold')
    ax.set_ylabel('Linear Strain (ppm)', fontweight='bold')
    ax.set_title('(a) GDC Chemical Expansion vs $\\delta$', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Chemical expansion coefficient vs delta
    ax = axes[0, 1]
    for temp, color in zip(temps, colors_t):
        col_name = f'alpha_chem_{temp}'
        ax.plot(df['delta'], df[col_name], 's-', 
               linewidth=2, markersize=5, label=f'{temp[:-1]}°C', color=color)
    
    ax.set_xlabel('Non-stoichiometry $\\delta$', fontweight='bold')
    ax.set_ylabel('$\\alpha_{chem}$ (strain/δ)', fontweight='bold')
    ax.set_title('(b) Chemical Expansion Coefficient', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: pO2 vs delta (Brouwer diagram style)
    ax = axes[1, 0]
    for temp, color in zip(temps, colors_t):
        col_name = f'pO2_{temp}_atm'
        valid_data = df[df[col_name] > 0]
        ax.semilogy(valid_data['delta'], valid_data[col_name], 'o-', 
                   linewidth=2, markersize=5, label=f'{temp[:-1]}°C', color=color)
    
    ax.set_xlabel('Non-stoichiometry $\\delta$', fontweight='bold')
    ax.set_ylabel('$pO_2$ (atm)', fontweight='bold')
    ax.set_title('(c) Defect Chemistry: $pO_2$ vs $\\delta$', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Heatmap of expansion at different conditions
    ax = axes[1, 1]
    expansion_matrix = df[['T_600C', 'T_700C', 'T_800C', 'T_900C']].values * 1e6
    
    im = ax.imshow(expansion_matrix.T, aspect='auto', cmap='YlOrRd', 
                   interpolation='bilinear', extent=[0, 22, 600, 900])
    ax.set_xlabel('Data Point Index (increasing $\\delta$)', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('(d) GDC Expansion Heatmap', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Strain (ppm)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_04_GDC_Chemical_Expansion.png', bbox_inches='tight')
    print("✓ Generated Figure 4: GDC Chemical Expansion")
    plt.close()

def create_figure_5_cohesive_zone():
    """Figure 5: Cohesive Zone Model Parameters"""
    df = pd.read_csv(csv_dir / '08_cohesive_zone_model_parameters.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cohesive Zone Model Parameters for Interface Fracture', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Mode-I fracture energy comparison
    ax = axes[0, 0]
    mode_i = df[df['Mode'] == 'Mode-I']
    ysz_gdc_i = mode_i[mode_i['Interface'] == 'YSZ/GDC'].head(4)
    gdc_lscf_i = mode_i[mode_i['Interface'] == 'GDC/LSCF'].head(4)
    
    x = np.arange(4)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ysz_gdc_i['Gc'].values, width, 
                   label='YSZ/GDC', color='#1f77b4', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, gdc_lscf_i['Gc'].values, width, 
                   label='GDC/LSCF', color='#d62728', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Mode-I $G_c$ (J/m²)', fontweight='bold')
    ax.set_xlabel('Interface Condition', fontweight='bold')
    ax.set_title('(a) Mode-I Fracture Energy Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Degraded', 'Intermediate', 'Extreme'], 
                       rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Traction-separation curves (schematic)
    ax = axes[1, 0]
    delta = np.linspace(0, 0.15, 100)
    
    # Bilinear model
    T_bilinear = np.where(delta < 0.03, delta/0.03 * 220, 
                          np.maximum(0, 220 * (0.15 - delta) / (0.15 - 0.03)))
    ax.plot(delta, T_bilinear, '-', linewidth=2.5, label='Bilinear (YSZ/GDC)', color='#1f77b4')
    
    # Exponential model
    T_exp = 180 * (delta/0.02) * np.exp(1 - delta/0.02)
    ax.plot(delta, T_exp, '-', linewidth=2.5, label='Exponential (GDC/LSCF)', color='#d62728')
    
    ax.set_xlabel('Separation $\\delta$ (µm)', fontweight='bold')
    ax.set_ylabel('Traction $T$ (MPa)', fontweight='bold')
    ax.set_title('(b) Traction-Separation Laws', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.15])
    ax.set_ylim([0, 250])
    
    # Plot 3: Mixed-mode fracture envelope
    ax = axes[0, 1]
    
    # Power law criterion
    mode_i_ratio = np.linspace(0, 1, 100)
    mode_ii_ratio = (1 - mode_i_ratio**2)**0.5  # Circular criterion
    
    ax.plot(mode_i_ratio, mode_ii_ratio, '-', linewidth=3, color='#2ca02c', 
           label='Mixed-mode criterion')
    ax.fill_between(mode_i_ratio, 0, mode_ii_ratio, alpha=0.3, color='#2ca02c', 
                    label='Safe region')
    
    # Add some data points
    points_i = [0.3, 0.6, 0.8, 0.5]
    points_ii = [0.9, 0.7, 0.4, 0.8]
    ax.scatter(points_i, points_ii, s=100, c='red', marker='x', linewidths=3, 
              label='Example loading states', zorder=5)
    
    ax.set_xlabel('Mode-I Ratio $G_I/G_{Ic}$', fontweight='bold')
    ax.set_ylabel('Mode-II Ratio $G_{II}/G_{IIc}$', fontweight='bold')
    ax.set_title('(c) Mixed-Mode Fracture Criterion', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    
    # Plot 4: Penalty stiffness vs interface compliance
    ax = axes[1, 1]
    interfaces_all = df.groupby(['Interface', 'Condition']).first().reset_index()
    
    scatter_data = []
    for _, row in interfaces_all.iterrows():
        color = '#1f77b4' if 'YSZ' in row['Interface'] else '#d62728'
        ax.scatter(row['K_penalty'], row['Gc'], s=100, alpha=0.6, 
                  color=color, edgecolors='black')
    
    ax.set_xlabel('Penalty Stiffness $K$ (GPa/m)', fontweight='bold')
    ax.set_ylabel('Fracture Energy $G_c$ (J/m²)', fontweight='bold')
    ax.set_title('(d) Interface Stiffness vs Fracture Energy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', alpha=0.6, label='YSZ/GDC'),
                      Patch(facecolor='#d62728', alpha=0.6, label='GDC/LSCF')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_05_Cohesive_Zone_Model.png', bbox_inches='tight')
    print("✓ Generated Figure 5: Cohesive Zone Model")
    plt.close()

def create_figure_6_verification():
    """Figure 6: Verification and QA Parameters"""
    df = pd.read_csv(csv_dir / '05_verification_QA_parameters.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Verification and Quality Assurance Parameters', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Mesh convergence study (simulated)
    ax = axes[0, 0]
    h_over_l0 = np.array([0.1, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5])
    energy_error = np.array([0.08, 0.05, 0.02, 0.015, 0.025, 0.05, 0.12])
    
    ax.semilogy(h_over_l0, energy_error, 'o-', linewidth=2.5, markersize=8, 
               color='#1f77b4', label='Energy error')
    ax.axvspan(0.125, 0.5, alpha=0.2, color='green', label='Recommended range')
    ax.axhline(0.01, color='red', linestyle='--', linewidth=2, label='Target accuracy')
    
    ax.set_xlabel('Mesh size ratio $h/l_0$', fontweight='bold')
    ax.set_ylabel('Relative Energy Error', fontweight='bold')
    ax.set_title('(a) Mesh Objectivity Study', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Degradation function
    ax = axes[0, 1]
    phi = np.linspace(0, 1, 200)
    k_res_values = [1e-6, 1e-5, 1e-4]
    colors_k = ['#2ca02c', '#ff7f0e', '#d62728']
    
    for k_res, color in zip(k_res_values, colors_k):
        g_phi = (1 - phi)**2 + k_res
        ax.plot(phi, g_phi, linewidth=2.5, label=f'$k_{{res}}={k_res:.0e}$', color=color)
    
    ax.set_xlabel('Phase-field $\\phi$', fontweight='bold')
    ax.set_ylabel('Degradation function $g(\\phi)$', fontweight='bold')
    ax.set_title('(b) Degradation Function with Residual Stiffness', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.1])
    
    # Plot 3: Convergence tolerance comparison
    ax = axes[1, 0]
    tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    iterations = [8, 12, 18, 25, 35]
    comp_time = [0.5, 0.8, 1.2, 1.8, 2.8]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(tolerances, iterations, 's-', linewidth=2.5, markersize=8, 
                    color='#1f77b4', label='Iterations')
    line2 = ax2.plot(tolerances, comp_time, 'o-', linewidth=2.5, markersize=8, 
                     color='#d62728', label='CPU time')
    
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('Newton-Raphson Tolerance $\\varepsilon_{tol}$', fontweight='bold')
    ax.set_ylabel('Average Iterations per Step', fontweight='bold', color='#1f77b4')
    ax2.set_ylabel('Relative CPU Time', fontweight='bold', color='#d62728')
    ax.set_title('(c) Convergence Tolerance Selection', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    # Plot 4: Load step size vs crack increment
    ax = axes[1, 1]
    delta_lambda = np.logspace(-4, -1, 50)
    delta_a = 0.05 / delta_lambda**0.5  # Inverse square root relationship
    
    ax.loglog(delta_lambda, delta_a, linewidth=2.5, color='#2ca02c')
    ax.axhspan(0.01, 0.1, alpha=0.2, color='green', label='Stable crack growth')
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, 
              label='Recommended: 0.05 µm')
    
    ax.set_xlabel('Load Step Size $\\Delta\\lambda$', fontweight='bold')
    ax.set_ylabel('Crack Increment $\\Delta a$ (µm)', fontweight='bold')
    ax.set_title('(d) Load Step vs Crack Propagation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_06_Verification_QA.png', bbox_inches='tight')
    print("✓ Generated Figure 6: Verification & QA")
    plt.close()

def create_figure_7_material_properties():
    """Figure 7: Material Properties Summary"""
    df = pd.read_csv(csv_dir / '06_material_properties.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Material Properties: LSCF, YSZ, and GDC', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Young's Modulus
    ax = axes[0, 0]
    materials = ['LSCF', 'YSZ', 'GDC']
    E_values = [40, 200, 160]
    colors_mat = ['#2ca02c', '#1f77b4', '#ff7f0e']
    
    bars = ax.bar(materials, E_values, color=colors_mat, alpha=0.7, edgecolor='black')
    ax.set_ylabel("Young's Modulus (GPa)", fontweight='bold')
    ax.set_title("(a) Elastic Modulus at 800°C", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, E_values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val} GPa', 
               ha='center', fontweight='bold')
    
    # Plot 2: Thermal vs Chemical Expansion
    ax = axes[0, 1]
    materials_exp = ['LSCF', 'YSZ', 'GDC']
    thermal_exp = [13.5, 10.5, 12.0]
    chemical_exp = [8.5, 0.0, 10.0]  # Effective chemical expansion coefficient
    
    x = np.arange(len(materials_exp))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, thermal_exp, width, label='Thermal', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, chemical_exp, width, label='Chemical', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Expansion Coefficient (ppm/K)', fontweight='bold')
    ax.set_title('(b) Thermal vs Chemical Expansion', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(materials_exp)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fracture Toughness
    ax = axes[1, 0]
    K_IC = [0.9, 2.8, 1.2]
    
    bars = ax.bar(materials, K_IC, color=colors_mat, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Fracture Toughness $K_{IC}$ (MPa·m$^{0.5}$)', fontweight='bold')
    ax.set_title('(c) Fracture Toughness at 800°C', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Conductivity comparison
    ax = axes[1, 1]
    ionic_cond = [0.15, 0.15, 0.025]  # YSZ approximation
    electronic_cond = [320, 0.001, 0.0008]
    
    x = np.arange(len(materials))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ionic_cond, width, label='Ionic', 
                   color='#9b59b6', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, electronic_cond, width, label='Electronic', 
                   color='#f39c12', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Conductivity (S/cm)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(d) Ionic vs Electronic Conductivity at 800°C', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(materials)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_07_Material_Properties.png', bbox_inches='tight')
    print("✓ Generated Figure 7: Material Properties")
    plt.close()

def create_figure_8_operating_conditions():
    """Figure 8: Operating Conditions and Stress States"""
    df = pd.read_csv(csv_dir / '07_operating_conditions.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Operating Conditions and Electrochemical Environment', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Voltage vs Current Density
    ax = axes[0, 0]
    load_conditions = df[df['Condition'].str.contains('Load')]
    
    ax.plot(load_conditions['Current_Density_A_cm2'], load_conditions['Voltage_V'], 
           'o-', linewidth=2.5, markersize=10, color='#e74c3c')
    ax.axhline(1.1, color='green', linestyle='--', linewidth=2, label='OCV')
    ax.fill_between([0, 1.2], 0.7, 0.85, alpha=0.2, color='green', 
                    label='Optimal range')
    
    ax.set_xlabel('Current Density (A/cm²)', fontweight='bold')
    ax.set_ylabel('Cell Voltage (V)', fontweight='bold')
    ax.set_title('(a) Polarization Curve (800°C)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0.5, 1.2])
    
    # Plot 2: Temperature cycling profile
    ax = axes[0, 1]
    time_profile = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    temp_profile = [600, 600, 800, 800, 900, 900, 800, 650, 600]
    
    ax.plot(time_profile, temp_profile, linewidth=3, color='#e74c3c')
    ax.fill_between(time_profile, temp_profile, 600, alpha=0.3, color='#f39c12')
    ax.axhline(800, color='blue', linestyle='--', linewidth=2, label='Operating T')
    
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('(b) Thermal Cycling Profile', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: pO2 gradient across cell
    ax = axes[1, 0]
    position = np.linspace(0, 1, 100)
    pO2_gradient = 0.21 * np.exp(-20 * position)
    
    ax.semilogy(position, pO2_gradient, linewidth=3, color='#3498db')
    ax.axvline(0.7, color='red', linestyle='--', linewidth=2, 
              label='Cathode/Electrolyte interface')
    ax.axvline(0.8, color='orange', linestyle='--', linewidth=2, 
              label='GDC interlayer')
    
    ax.set_xlabel('Normalized Position (Cathode → Anode)', fontweight='bold')
    ax.set_ylabel('$pO_2$ (atm)', fontweight='bold')
    ax.set_title('(c) Oxygen Partial Pressure Profile', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Operating time vs degradation
    ax = axes[1, 1]
    time_points = np.array([0, 100, 200, 500, 1000, 2000])
    degradation = 100 * (1 - np.exp(-time_points / 800))
    
    ax.plot(time_points, degradation, 'o-', linewidth=2.5, markersize=8, 
           color='#d62728')
    ax.axhline(20, color='orange', linestyle='--', linewidth=2, 
              label='Significant degradation')
    ax.fill_between(time_points, 0, degradation, alpha=0.3, color='#d62728')
    
    ax.set_xlabel('Operation Time (h)', fontweight='bold')
    ax.set_ylabel('Interface Degradation (%)', fontweight='bold')
    ax.set_title('(d) Long-term Degradation Evolution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_08_Operating_Conditions.png', bbox_inches='tight')
    print("✓ Generated Figure 8: Operating Conditions")
    plt.close()

def main():
    """Main function to generate all figures"""
    print("\n" + "="*60)
    print("Generating Phase-Field Fracture Dataset Visualizations")
    print("="*60 + "\n")
    
    try:
        create_figure_1_main_parameters()
        create_figure_2_interface_properties()
        create_figure_3_lscf_nonstoichiometry()
        create_figure_4_gdc_expansion()
        create_figure_5_cohesive_zone()
        create_figure_6_verification()
        create_figure_7_material_properties()
        create_figure_8_operating_conditions()
        
        print("\n" + "="*60)
        print("All figures generated successfully!")
        print(f"Location: {fig_dir.absolute()}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
