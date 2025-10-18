#!/usr/bin/env python3
"""
Visualization Script for Thermo-Mechanical Dataset
Generates comprehensive plots for validation and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_thermal_properties(df: pd.DataFrame, output_dir: Path):
    """Plot thermal properties vs temperature for all mixes"""
    cal_data = df[df['Data_Type'] == 'Calibration']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Thermal Properties Evolution with Temperature', fontsize=16, fontweight='bold')
    
    mix_ids = cal_data['Mix_ID'].unique()
    colors = sns.color_palette("husl", len(mix_ids))
    
    # Thermal conductivity
    ax = axes[0, 0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Thermal_Conductivity_W_mK'], 
                marker='o', linewidth=2, label=mix_id, color=colors[i])
        # Add uncertainty band
        ax.fill_between(mix_data['Temperature_C'],
                        mix_data['Thermal_Conductivity_W_mK'] - mix_data['Thermal_Conductivity_Std'],
                        mix_data['Thermal_Conductivity_W_mK'] + mix_data['Thermal_Conductivity_Std'],
                        alpha=0.2, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Specific heat
    ax = axes[0, 1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Specific_Heat_J_kgK'], 
                marker='s', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Specific Heat (J/kg·K)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(100, color='red', linestyle='--', alpha=0.5, label='Moisture evaporation')
    ax.axvline(450, color='orange', linestyle='--', alpha=0.5, label='Dehydration')
    
    # Density
    ax = axes[1, 0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Density_kg_m3'], 
                marker='^', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Density (kg/m³)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Thermal diffusivity (calculated)
    ax = axes[1, 1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        alpha_thermal = (mix_data['Thermal_Conductivity_W_mK'] / 
                        (mix_data['Density_kg_m3'] * mix_data['Specific_Heat_J_kgK'])) * 1e6
        ax.plot(mix_data['Temperature_C'], alpha_thermal, 
                marker='d', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Thermal Diffusivity (mm²/s)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'thermal_properties.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated thermal_properties.png")


def plot_mechanical_properties(df: pd.DataFrame, output_dir: Path):
    """Plot mechanical properties vs temperature for all mixes"""
    cal_data = df[df['Data_Type'] == 'Calibration']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mechanical Properties Evolution with Temperature', fontsize=16, fontweight='bold')
    
    mix_ids = cal_data['Mix_ID'].unique()
    colors = sns.color_palette("husl", len(mix_ids))
    
    # Compressive strength
    ax = axes[0, 0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Compressive_Strength_MPa'], 
                marker='o', linewidth=2, label=mix_id, color=colors[i])
        ax.fill_between(mix_data['Temperature_C'],
                        mix_data['Compressive_Strength_MPa'] - mix_data['Compressive_Strength_Std'],
                        mix_data['Compressive_Strength_MPa'] + mix_data['Compressive_Strength_Std'],
                        alpha=0.2, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Compressive Strength (MPa)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Tensile strength
    ax = axes[0, 1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Tensile_Strength_MPa'], 
                marker='s', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Tensile Strength (MPa)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Elastic modulus
    ax = axes[1, 0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Elastic_Modulus_MPa'], 
                marker='^', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Elastic Modulus (MPa)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Poisson's ratio
    ax = axes[1, 1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Poisson_Ratio'], 
                marker='d', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel("Poisson's Ratio", fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Incompressible limit')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mechanical_properties.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated mechanical_properties.png")


def plot_retention_factors(thermal_df: pd.DataFrame, mechanical_df: pd.DataFrame, output_dir: Path):
    """Plot retention factors (property/property_20C) vs temperature"""
    cal_mech = mechanical_df[mechanical_df['Data_Type'] == 'Calibration']
    cal_thermal = thermal_df[thermal_df['Data_Type'] == 'Calibration']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Property Retention Factors (Normalized to 20°C)', fontsize=16, fontweight='bold')
    
    mix_ids = cal_mech['Mix_ID'].unique()
    colors = sns.color_palette("husl", len(mix_ids))
    
    # Compressive strength retention
    ax = axes[0, 0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_mech[cal_mech['Mix_ID'] == mix_id]
        fc_20 = mix_data[mix_data['Temperature_C'] == 20]['Compressive_Strength_MPa'].values[0]
        retention = mix_data['Compressive_Strength_MPa'] / fc_20
        ax.plot(mix_data['Temperature_C'], retention, 
                marker='o', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('$f_c(T) / f_c(20°C)$', fontsize=11)
    ax.set_title('Compressive Strength Retention', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Elastic modulus retention
    ax = axes[0, 1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_mech[cal_mech['Mix_ID'] == mix_id]
        E_20 = mix_data[mix_data['Temperature_C'] == 20]['Elastic_Modulus_MPa'].values[0]
        retention = mix_data['Elastic_Modulus_MPa'] / E_20
        ax.plot(mix_data['Temperature_C'], retention, 
                marker='s', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('$E(T) / E(20°C)$', fontsize=11)
    ax.set_title('Elastic Modulus Retention', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Thermal conductivity retention
    ax = axes[1, 0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_thermal[cal_thermal['Mix_ID'] == mix_id]
        k_20 = mix_data[mix_data['Temperature_C'] == 20]['Thermal_Conductivity_W_mK'].values[0]
        retention = mix_data['Thermal_Conductivity_W_mK'] / k_20
        ax.plot(mix_data['Temperature_C'], retention, 
                marker='^', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('$k(T) / k(20°C)$', fontsize=11)
    ax.set_title('Thermal Conductivity Retention', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Rubber content effect at 600°C
    ax = axes[1, 1]
    rubber_contents = []
    fc_retention_600 = []
    E_retention_600 = []
    
    for mix_id in mix_ids:
        mix_data = cal_mech[cal_mech['Mix_ID'] == mix_id]
        if mix_id == 'C':
            rubber_contents.append(0)
        else:
            # Extract rubber percentage from ID
            rubber_contents.append(int(mix_id[1:3].replace('S', '').replace('L', '')))
        
        fc_20 = mix_data[mix_data['Temperature_C'] == 20]['Compressive_Strength_MPa'].values[0]
        fc_600 = mix_data[mix_data['Temperature_C'] == 600]['Compressive_Strength_MPa'].values[0]
        fc_retention_600.append(fc_600 / fc_20)
        
        E_20 = mix_data[mix_data['Temperature_C'] == 20]['Elastic_Modulus_MPa'].values[0]
        E_600 = mix_data[mix_data['Temperature_C'] == 600]['Elastic_Modulus_MPa'].values[0]
        E_retention_600.append(E_600 / E_20)
    
    ax.plot(rubber_contents, fc_retention_600, marker='o', linewidth=2, 
            label='Compressive Strength', markersize=10)
    ax.plot(rubber_contents, E_retention_600, marker='s', linewidth=2, 
            label='Elastic Modulus', markersize=10)
    ax.set_xlabel('Rubber Content (%)', fontsize=11)
    ax.set_ylabel('Property Retention at 600°C', fontsize=11)
    ax.set_title('Rubber Content Effect on High-Temp Performance', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retention_factors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated retention_factors.png")


def plot_transport_properties(df: pd.DataFrame, output_dir: Path):
    """Plot transport properties vs temperature"""
    cal_data = df[df['Data_Type'] == 'Calibration']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Transport Properties Evolution with Temperature', fontsize=16, fontweight='bold')
    
    mix_ids = cal_data['Mix_ID'].unique()
    colors = sns.color_palette("husl", len(mix_ids))
    
    # Gas permeability
    ax = axes[0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        perm = pd.to_numeric(mix_data['Gas_Permeability_m2'])
        ax.semilogy(mix_data['Temperature_C'], perm, 
                    marker='o', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Gas Permeability (m²)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Moisture diffusivity
    ax = axes[1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        diff = pd.to_numeric(mix_data['Moisture_Diffusivity_m2_s'])
        ax.semilogy(mix_data['Temperature_C'], diff, 
                    marker='s', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Moisture Diffusivity (m²/s)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Porosity
    ax = axes[2]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Porosity'], 
                marker='^', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Porosity', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transport_properties.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated transport_properties.png")


def plot_deformation_properties(df: pd.DataFrame, output_dir: Path):
    """Plot deformation properties vs temperature"""
    cal_data = df[df['Data_Type'] == 'Calibration']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Deformation Properties Evolution with Temperature', fontsize=16, fontweight='bold')
    
    mix_ids = cal_data['Mix_ID'].unique()
    colors = sns.color_palette("husl", len(mix_ids))
    
    # Thermal strain
    ax = axes[0]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        strain = pd.to_numeric(mix_data['Thermal_Strain'])
        ax.plot(mix_data['Temperature_C'], strain * 1000, 
                marker='o', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Thermal Strain (×10⁻³)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Creep coefficient
    ax = axes[1]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        ax.plot(mix_data['Temperature_C'], mix_data['Creep_Coefficient'], 
                marker='s', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Creep Coefficient', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='Peak creep temperature')
    
    # Shrinkage strain
    ax = axes[2]
    for i, mix_id in enumerate(mix_ids):
        mix_data = cal_data[cal_data['Mix_ID'] == mix_id]
        shrinkage = pd.to_numeric(mix_data['Shrinkage_Strain'])
        ax.plot(mix_data['Temperature_C'], shrinkage * 1000, 
                marker='^', linewidth=2, label=mix_id, color=colors[i])
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Shrinkage Strain (×10⁻³)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deformation_properties.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated deformation_properties.png")


def plot_calibration_vs_validation(thermal_df: pd.DataFrame, mechanical_df: pd.DataFrame, 
                                   output_dir: Path):
    """Compare calibration and validation datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calibration vs Validation Dataset Comparison (Mix C)', 
                 fontsize=16, fontweight='bold')
    
    mix_id = 'C'
    
    # Thermal conductivity
    ax = axes[0, 0]
    for data_type, marker, alpha in [('Calibration', 'o', 1.0), ('Validation', 's', 0.6)]:
        data = thermal_df[(thermal_df['Mix_ID'] == mix_id) & 
                         (thermal_df['Data_Type'] == data_type)]
        ax.plot(data['Temperature_C'], data['Thermal_Conductivity_W_mK'],
                marker=marker, linewidth=2, label=data_type, alpha=alpha)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Compressive strength
    ax = axes[0, 1]
    for data_type, marker, alpha in [('Calibration', 'o', 1.0), ('Validation', 's', 0.6)]:
        data = mechanical_df[(mechanical_df['Mix_ID'] == mix_id) & 
                            (mechanical_df['Data_Type'] == data_type)]
        ax.plot(data['Temperature_C'], data['Compressive_Strength_MPa'],
                marker=marker, linewidth=2, label=data_type, alpha=alpha)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Compressive Strength (MPa)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Elastic modulus
    ax = axes[1, 0]
    for data_type, marker, alpha in [('Calibration', 'o', 1.0), ('Validation', 's', 0.6)]:
        data = mechanical_df[(mechanical_df['Mix_ID'] == mix_id) & 
                            (mechanical_df['Data_Type'] == data_type)]
        ax.plot(data['Temperature_C'], data['Elastic_Modulus_MPa'],
                marker=marker, linewidth=2, label=data_type, alpha=alpha)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Elastic Modulus (MPa)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Specific heat
    ax = axes[1, 1]
    for data_type, marker, alpha in [('Calibration', 'o', 1.0), ('Validation', 's', 0.6)]:
        data = thermal_df[(thermal_df['Mix_ID'] == mix_id) & 
                         (thermal_df['Data_Type'] == data_type)]
        ax.plot(data['Temperature_C'], data['Specific_Heat_J_kgK'],
                marker=marker, linewidth=2, label=data_type, alpha=alpha)
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Specific Heat (J/kg·K)', fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_vs_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated calibration_vs_validation.png")


def main():
    """Main visualization function"""
    print("\n" + "=" * 80)
    print("Generating Visualization Plots")
    print("=" * 80)
    
    data_dir = Path("/workspace/thermo_mechanical_dataset")
    csv_dir = data_dir / "csv"
    plot_dir = data_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    thermal_df = pd.read_csv(csv_dir / "thermal_properties.csv")
    mechanical_df = pd.read_csv(csv_dir / "mechanical_properties.csv")
    transport_df = pd.read_csv(csv_dir / "transport_properties.csv")
    deformation_df = pd.read_csv(csv_dir / "deformation_properties.csv")
    print("  ✓ Loaded all property datasets")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_thermal_properties(thermal_df, plot_dir)
    plot_mechanical_properties(mechanical_df, plot_dir)
    plot_retention_factors(thermal_df, mechanical_df, plot_dir)
    plot_transport_properties(transport_df, plot_dir)
    plot_deformation_properties(deformation_df, plot_dir)
    plot_calibration_vs_validation(thermal_df, mechanical_df, plot_dir)
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {plot_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
