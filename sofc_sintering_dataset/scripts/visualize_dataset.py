#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC??????????
????????????
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns

# ??????
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

# ????
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def plot_green_state_properties():
    """???????"""
    df = pd.read_csv(RAW_DATA_DIR / "green_state_properties.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ????
    ax = axes[0, 0]
    x = np.arange(len(df['Material']))
    width = 0.35
    ax.bar(x - width/2, df['Green_Density_kg_m3'], width, label='????', alpha=0.8)
    ax.bar(x + width/2, df['Theoretical_Density_kg_m3'], width, label='????', alpha=0.8)
    ax.set_xlabel('??')
    ax.set_ylabel('?? (kg/m?)')
    ax.set_title('???? vs ????')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Material'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ???
    ax = axes[0, 1]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    ax.barh(df['Material'], df['Initial_Porosity_fraction'] * 100, color=colors, alpha=0.8)
    ax.set_xlabel('????? (%)')
    ax.set_title('????????')
    ax.grid(True, alpha=0.3, axis='x')
    
    # ?????????
    ax = axes[1, 0]
    x = np.arange(len(df['Material']))
    width = 0.35
    ax.bar(x - width/2, df['Binder_Content_wt_percent'], width, label='??? (wt%)', alpha=0.8)
    ax.bar(x + width/2, df['Porogen_Content_vol_percent'], width, label='??? (vol%)', alpha=0.8)
    ax.set_xlabel('??')
    ax.set_ylabel('?? (%)')
    ax.set_title('?????????')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Material'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ????
    ax = axes[1, 1]
    ax2 = ax.twinx()
    x = np.arange(len(df['Material']))
    p1 = ax.bar(x - 0.2, df['Particle_Size_D50_um'], 0.4, label='D50?? (?m)', alpha=0.8, color='steelblue')
    p2 = ax2.bar(x + 0.2, df['Specific_Surface_Area_m2_g'], 0.4, label='???? (m?/g)', alpha=0.8, color='coral')
    ax.set_xlabel('??')
    ax.set_ylabel('D50?? (?m)', color='steelblue')
    ax2.set_ylabel('???? (m?/g)', color='coral')
    ax.set_title('?????????')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Material'], rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_green_state_properties.png", dpi=300, bbox_inches='tight')
    print("? ???: ??????")
    plt.close()


def plot_sintering_kinetics():
    """????????"""
    materials = {
        'Anode': 'anode_dilatometry.csv',
        'Electrolyte': 'electrolyte_dilatometry.csv', 
        'Cathode': 'cathode_dilatometry.csv'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Anode': '#ff6b6b', 'Electrolyte': '#4ecdc4', 'Cathode': '#95e1d3'}
    
    # ???? vs ??
    ax = axes[0, 0]
    for name, file in materials.items():
        df = pd.read_csv(RAW_DATA_DIR / "sintering_kinetics" / file)
        ax.plot(df['Temperature_C'], df['Linear_Shrinkage_percent'], 
                label=name, linewidth=2.5, marker='o', markersize=4, 
                color=colors[name], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('????? (%)')
    ax.set_title('??????')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ???? vs ??
    ax = axes[0, 1]
    for name, file in materials.items():
        df = pd.read_csv(RAW_DATA_DIR / "sintering_kinetics" / file)
        # ??????????
        df_heating = df[df['Time_min'] <= 240]
        ax.plot(df_heating['Temperature_C'], df_heating['Shrinkage_Rate_percent_min'], 
                label=name, linewidth=2.5, marker='s', markersize=4,
                color=colors[name], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('???? (%/min)')
    ax.set_title('??????')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # ???? vs ??
    ax = axes[1, 0]
    for name, file in materials.items():
        df = pd.read_csv(RAW_DATA_DIR / "sintering_kinetics" / file)
        ax.plot(df['Temperature_C'], df['Relative_Density_fraction'], 
                label=name, linewidth=2.5, marker='^', markersize=4,
                color=colors[name], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('????')
    ax.set_title('?????')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ????????
    ax = axes[1, 1]
    df_bilayer = pd.read_csv(RAW_DATA_DIR / "sintering_kinetics" / "bilayer_anode_electrolyte.csv")
    ax.plot(df_bilayer['Temperature_C'], df_bilayer['Differential_Shrinkage_percent'], 
            linewidth=2.5, marker='D', markersize=5, color='#e74c3c', label='????', alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('???? (%)')
    ax.set_title('??-?????????')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ????????
    ax.axvspan(1100, 1300, alpha=0.2, color='yellow', label='????')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_sintering_kinetics.png", dpi=300, bbox_inches='tight')
    print("? ???: ???????")
    plt.close()


def plot_cte_data():
    """????????"""
    df = pd.read_csv(RAW_DATA_DIR / "cte_data" / "thermal_expansion_coefficients.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    materials = df['Material'].unique()
    colors = {'Ni-YSZ': '#e74c3c', '8YSZ': '#3498db', 'LSM-YSZ': '#2ecc71'}
    
    # CTE vs ??
    ax = axes[0]
    for material in materials:
        df_mat = df[df['Material'] == material]
        ax.plot(df_mat['Temperature_C'], df_mat['CTE_1E-6_K'], 
                label=material, linewidth=2.5, marker='o', markersize=6,
                color=colors[material], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('????? (10??/K)')
    ax.set_title('??????????')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ???? vs ??
    ax = axes[1]
    for material in materials:
        df_mat = df[df['Material'] == material]
        ax.plot(df_mat['Temperature_C'], df_mat['Linear_Expansion_percent'], 
                label=material, linewidth=2.5, marker='s', markersize=6,
                color=colors[material], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('???? (%)')
    ax.set_title('?????')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_thermal_expansion.png", dpi=300, bbox_inches='tight')
    print("? ???: ???????")
    plt.close()


def plot_creep_data():
    """?????????"""
    # ??????
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    temps_files = [
        ('800?C', 'anode_creep_800C.csv', 0),
        ('1000?C', 'anode_creep_1000C.csv', 1),
        ('1200?C', 'anode_creep_1200C.csv', 2)
    ]
    
    for temp_label, file, idx in temps_files:
        df = pd.read_csv(RAW_DATA_DIR / "creep_tests" / file)
        
        # ?? vs ??
        ax = axes[0, idx]
        for stress in df['Stress_MPa'].unique():
            df_stress = df[df['Stress_MPa'] == stress]
            ax.plot(df_stress['Time_s'], df_stress['Strain'], 
                    label=f'{stress} MPa', linewidth=2.5, marker='o', markersize=4)
        ax.set_xlabel('?? (s)')
        ax.set_ylabel('??')
        ax.set_title(f'?????? - {temp_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ??? vs ?? (????)
        ax = axes[1, idx]
        stresses = df['Stress_MPa'].unique()
        strain_rates = [df[df['Stress_MPa'] == s]['Strain_Rate_s_inv'].mean() for s in stresses]
        ax.loglog(stresses, strain_rates, 'o-', linewidth=2.5, markersize=8, color='#e74c3c')
        
        # ??Norton?? (log-log???????)
        log_stress = np.log10(stresses)
        log_rate = np.log10(strain_rates)
        coeffs = np.polyfit(log_stress, log_rate, 1)
        n = coeffs[0]
        fit_stress = np.logspace(np.log10(stresses.min()), np.log10(stresses.max()), 50)
        fit_rate = 10**(coeffs[1]) * fit_stress**n
        ax.loglog(fit_stress, fit_rate, '--', linewidth=2, color='blue', alpha=0.7, 
                 label=f'Norton??: n={n:.2f}')
        
        ax.set_xlabel('?? (MPa)')
        ax.set_ylabel('??? (s??)')
        ax.set_title(f'Norton???? - {temp_label}')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_anode_creep_analysis.png", dpi=300, bbox_inches='tight')
    print("? ???: ????????")
    plt.close()


def plot_elastic_properties():
    """???????"""
    df = pd.read_csv(RAW_DATA_DIR / "elastic_properties" / "youngs_modulus_temperature.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    materials = df['Material'].unique()
    colors = {'Ni-YSZ': '#e74c3c', '8YSZ': '#3498db', 'LSM-YSZ': '#2ecc71'}
    
    # ???? vs ?? (????0.9)
    ax = axes[0, 0]
    for material in materials:
        df_mat = df[(df['Material'] == material) & (np.abs(df['Relative_Density'] - 0.90) < 0.02)]
        if len(df_mat) > 0:
            ax.plot(df_mat['Temperature_C'], df_mat['Youngs_Modulus_GPa'], 
                    label=material, linewidth=2.5, marker='o', markersize=6,
                    color=colors[material], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('???? (GPa)')
    ax.set_title('????????? (?=0.90)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ???? vs ???? (??)
    ax = axes[0, 1]
    for material in materials:
        df_mat = df[(df['Material'] == material) & (df['Temperature_C'] == 25)]
        ax.plot(df_mat['Relative_Density'], df_mat['Youngs_Modulus_GPa'], 
                label=material, linewidth=2.5, marker='s', markersize=8,
                color=colors[material], alpha=0.8)
    ax.set_xlabel('????')
    ax.set_ylabel('???? (GPa)')
    ax.set_title('????????? (25?C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ??? vs ??
    ax = axes[1, 0]
    for material in materials:
        df_mat = df[(df['Material'] == material) & (np.abs(df['Relative_Density'] - 0.90) < 0.02)]
        if len(df_mat) > 0:
            ax.plot(df_mat['Temperature_C'], df_mat['Poissons_Ratio'], 
                    label=material, linewidth=2.5, marker='^', markersize=6,
                    color=colors[material], alpha=0.8)
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('???')
    ax.set_title('???????? (?=0.90)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3D??: ???? vs ????? (?????)
    ax = axes[1, 1]
    df_anode = df[df['Material'] == 'Ni-YSZ']
    pivot_table = df_anode.pivot(index='Relative_Density', 
                                   columns='Temperature_C', 
                                   values='Youngs_Modulus_GPa')
    im = ax.contourf(pivot_table.columns, pivot_table.index, pivot_table.values, 
                     levels=15, cmap='RdYlBu_r')
    ax.set_xlabel('?? (?C)')
    ax.set_ylabel('????')
    ax.set_title('??????????')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('???? (GPa)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_elastic_properties.png", dpi=300, bbox_inches='tight')
    print("? ???: ??????")
    plt.close()


def plot_constitutive_summary():
    """???????????"""
    with open(BASE_DIR / "processed_data" / "constitutive_models.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    materials = ['anode', 'electrolyte', 'cathode']
    material_names = {'anode': 'Ni-YSZ??', 'electrolyte': '8YSZ???', 'cathode': 'LSM-YSZ??'}
    colors_map = {'anode': '#e74c3c', 'electrolyte': '#3498db', 'cathode': '#2ecc71'}
    
    # ????n
    ax = axes[0, 0]
    n_values = []
    labels = []
    colors_list = []
    for mat in materials:
        for temp_range in data[mat]['creep_parameters']['temperature_ranges']:
            n_values.append(temp_range['n_stress_exponent'])
            labels.append(f"{material_names[mat]}\n{temp_range['range_C']}?C")
            colors_list.append(colors_map[mat])
    
    x_pos = np.arange(len(n_values))
    ax.bar(x_pos, n_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('???? n')
    ax.set_title('Norton??????')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='????2')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ???Q
    ax = axes[0, 1]
    Q_values = []
    labels = []
    colors_list = []
    for mat in materials:
        for temp_range in data[mat]['creep_parameters']['temperature_ranges']:
            Q_values.append(temp_range['Q_activation_energy_kJ_mol'])
            labels.append(f"{material_names[mat]}\n{temp_range['range_C']}?C")
            colors_list.append(colors_map[mat])
    
    x_pos = np.arange(len(Q_values))
    ax.bar(x_pos, Q_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('??? Q (kJ/mol)')
    ax.set_title('?????')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # CTE??
    ax = axes[1, 0]
    cte_values = [data[mat]['thermal_properties']['CTE_base_1E-6_K'] for mat in materials]
    mat_labels = [material_names[mat] for mat in materials]
    colors_list = [colors_map[mat] for mat in materials]
    ax.barh(mat_labels, cte_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('CTE (10??/K)')
    ax.set_title('??????????')
    ax.grid(True, alpha=0.3, axis='x')
    
    # CTE????
    cte_diff_ae = abs(cte_values[0] - cte_values[1])
    cte_diff_ce = abs(cte_values[2] - cte_values[1])
    ax.text(0.95, 0.95, f'??-??? ?CTE: {cte_diff_ae:.1f}?10??/K\n??-??? ?CTE: {cte_diff_ce:.1f}?10??/K',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ????????
    ax = axes[1, 1]
    E_values = [data[mat]['elastic_properties']['reference_modulus_GPa'] for mat in materials]
    mat_labels = [material_names[mat] for mat in materials]
    colors_list = [colors_map[mat] for mat in materials]
    ax.bar(mat_labels, E_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('???? (GPa)')
    ax.set_title('???????? (???, 25?C)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_constitutive_model_summary.png", dpi=300, bbox_inches='tight')
    print("? ???: ??????????")
    plt.close()


def main():
    """???"""
    print("\n" + "="*60)
    print("  SOFC??????????")
    print("="*60 + "\n")
    
    print("?????????...\n")
    
    plot_green_state_properties()
    plot_sintering_kinetics()
    plot_cte_data()
    plot_creep_data()
    plot_elastic_properties()
    plot_constitutive_summary()
    
    print("\n" + "="*60)
    print(f"  ????????: {FIGURES_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
