#!/usr/bin/env python3
"""
Example Usage Scripts for Thermo-Mechanical Dataset
Demonstrates various ways to load, analyze, and use the generated data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def example_1_load_and_filter():
    """Example 1: Load and filter data for specific conditions"""
    print("=" * 80)
    print("EXAMPLE 1: Load and Filter Data")
    print("=" * 80)
    
    # Load complete dataset
    df = pd.read_csv('thermo_mechanical_dataset/complete_dataset.csv')
    
    print(f"\nTotal records: {len(df):,}")
    print(f"Mix designs: {df['Mix_ID'].unique()}")
    print(f"Property types: {df['Property_Type'].unique()}")
    
    # Extract calibration data for R10S mix at 600°C
    target_data = df[
        (df['Mix_ID'] == 'R10S') & 
        (df['Temperature_C'] == 600) & 
        (df['Data_Type'] == 'Calibration')
    ]
    
    print("\n" + "-" * 80)
    print("Data for R10S mix at 600°C (Calibration):")
    print("-" * 80)
    
    # Thermal properties
    thermal = target_data[target_data['Property_Type'] == 'Thermal']
    if not thermal.empty:
        print(f"\nThermal Conductivity: {thermal['Thermal_Conductivity_W_mK'].values[0]:.4f} W/m·K")
        print(f"Specific Heat: {thermal['Specific_Heat_J_kgK'].values[0]:.2f} J/kg·K")
        print(f"Density: {thermal['Density_kg_m3'].values[0]:.2f} kg/m³")
    
    # Mechanical properties
    mechanical = target_data[target_data['Property_Type'] == 'Mechanical']
    if not mechanical.empty:
        print(f"\nCompressive Strength: {mechanical['Compressive_Strength_MPa'].values[0]:.2f} MPa")
        print(f"Elastic Modulus: {mechanical['Elastic_Modulus_MPa'].values[0]:.1f} MPa")
        print(f"Poisson's Ratio: {mechanical['Poisson_Ratio'].values[0]:.4f}")
    
    print("\n")


def example_2_retention_analysis():
    """Example 2: Analyze property retention factors"""
    print("=" * 80)
    print("EXAMPLE 2: Property Retention Analysis")
    print("=" * 80)
    
    # Load mechanical properties
    df = pd.read_csv('thermo_mechanical_dataset/csv/mechanical_properties.csv')
    
    # Focus on calibration data
    cal_df = df[df['Data_Type'] == 'Calibration']
    
    # Compare different mixes at 600°C
    print("\nStrength Retention at 600°C (relative to 20°C):")
    print("-" * 80)
    print(f"{'Mix ID':<10} {'fc(20°C)':<12} {'fc(600°C)':<12} {'Retention':<12} {'Improvement'}")
    print("-" * 80)
    
    control_retention = None
    for mix_id in ['C', 'R5S', 'R10S', 'R15S', 'R20S']:
        mix_data = cal_df[cal_df['Mix_ID'] == mix_id]
        
        fc_20 = mix_data[mix_data['Temperature_C'] == 20]['Compressive_Strength_MPa'].values[0]
        fc_600 = mix_data[mix_data['Temperature_C'] == 600]['Compressive_Strength_MPa'].values[0]
        retention = fc_600 / fc_20
        
        if mix_id == 'C':
            control_retention = retention
            improvement = "-"
        else:
            improvement = f"+{((retention/control_retention - 1) * 100):.1f}%"
        
        print(f"{mix_id:<10} {fc_20:<12.2f} {fc_600:<12.2f} {retention*100:<11.1f}% {improvement}")
    
    print("\n")


def example_3_thermal_analysis():
    """Example 3: Calculate thermal diffusivity and thermal inertia"""
    print("=" * 80)
    print("EXAMPLE 3: Thermal Analysis")
    print("=" * 80)
    
    # Load thermal properties
    df = pd.read_csv('thermo_mechanical_dataset/csv/thermal_properties.csv')
    
    # Calculate derived thermal properties for R10S
    mix_data = df[(df['Mix_ID'] == 'R10S') & (df['Data_Type'] == 'Calibration')]
    
    print("\nThermal Properties Evolution for R10S Mix:")
    print("-" * 80)
    print(f"{'T (°C)':<10} {'k (W/m·K)':<12} {'α (mm²/s)':<12} {'ρcp (MJ/m³·K)':<15} {'Inertia'}")
    print("-" * 80)
    
    for _, row in mix_data[::4].iterrows():  # Every 4th row (80°C increments)
        T = row['Temperature_C']
        k = row['Thermal_Conductivity_W_mK']
        rho = row['Density_kg_m3']
        cp = row['Specific_Heat_J_kgK']
        
        # Thermal diffusivity (mm²/s)
        alpha = (k / (rho * cp)) * 1e6
        
        # Volumetric heat capacity (MJ/m³·K)
        rho_cp = (rho * cp) / 1e6
        
        # Thermal inertia index (relative to 20°C)
        if T == 20:
            rho_cp_20 = rho_cp
        inertia = rho_cp / rho_cp_20
        
        print(f"{T:<10.0f} {k:<12.4f} {alpha:<12.4f} {rho_cp:<15.4f} {inertia:<8.4f}")
    
    print("\n")


def example_4_compare_validation():
    """Example 4: Compare calibration vs validation datasets"""
    print("=" * 80)
    print("EXAMPLE 4: Calibration vs Validation Comparison")
    print("=" * 80)
    
    # Load mechanical properties
    df = pd.read_csv('thermo_mechanical_dataset/csv/mechanical_properties.csv')
    
    # Compare for control mix at selected temperatures
    print("\nControl Mix (C) - Compressive Strength Comparison:")
    print("-" * 80)
    print(f"{'Temperature':<15} {'Calibration (MPa)':<20} {'Validation (MPa)':<20} {'Difference'}")
    print("-" * 80)
    
    for T in [20, 200, 400, 600, 800]:
        cal_data = df[(df['Mix_ID'] == 'C') & 
                     (df['Temperature_C'] == T) & 
                     (df['Data_Type'] == 'Calibration')]
        val_data = df[(df['Mix_ID'] == 'C') & 
                     (df['Temperature_C'] == T) & 
                     (df['Data_Type'] == 'Validation')]
        
        fc_cal = cal_data['Compressive_Strength_MPa'].values[0]
        fc_val = val_data['Compressive_Strength_MPa'].values[0]
        diff = ((fc_val - fc_cal) / fc_cal) * 100
        
        print(f"{T}°C{'':<11} {fc_cal:<20.3f} {fc_val:<20.3f} {diff:+.2f}%")
    
    print("\n")


def example_5_json_access():
    """Example 5: Access JSON format data"""
    print("=" * 80)
    print("EXAMPLE 5: JSON Data Access")
    print("=" * 80)
    
    # Load JSON data for specific mix
    with open('thermo_mechanical_dataset/json/dataset_R15S.json', 'r') as f:
        data = json.load(f)
    
    print("\nAvailable property types in JSON:")
    print("-" * 80)
    for prop_type in data.keys():
        count = len(data[prop_type])
        print(f"{prop_type}: {count} records")
    
    # Extract specific temperature data
    print("\n\nR15S Mix Properties at 400°C (Calibration):")
    print("-" * 80)
    
    for prop_type, records in data.items():
        for record in records:
            if record['Temperature_C'] == 400 and record['Data_Type'] == 'Calibration':
                print(f"\n{prop_type.upper()}:")
                for key, value in record.items():
                    if key not in ['Mix_ID', 'Temperature_C', 'Data_Type', 'Property_Type']:
                        print(f"  {key}: {value}")
                break
    
    print("\n")


def example_6_create_input_function():
    """Example 6: Create temperature-dependent input functions"""
    print("=" * 80)
    print("EXAMPLE 6: Create Temperature-Dependent Functions")
    print("=" * 80)
    
    # Load thermal properties
    df = pd.read_csv('thermo_mechanical_dataset/csv/thermal_properties.csv')
    
    # Get calibration data for R10S
    mix_data = df[(df['Mix_ID'] == 'R10S') & (df['Data_Type'] == 'Calibration')]
    
    T = mix_data['Temperature_C'].values
    k = mix_data['Thermal_Conductivity_W_mK'].values
    cp = mix_data['Specific_Heat_J_kgK'].values
    
    print("\nCreating interpolation functions for FEA input...")
    print("-" * 80)
    
    # Test interpolation at arbitrary temperatures
    test_temps = [50, 150, 350, 550, 750]
    
    print(f"\n{'T (°C)':<10} {'k (W/m·K)':<15} {'cp (J/kg·K)':<15}")
    print("-" * 80)
    
    for T_test in test_temps:
        k_interp = np.interp(T_test, T, k)
        cp_interp = np.interp(T_test, T, cp)
        print(f"{T_test:<10.0f} {k_interp:<15.4f} {cp_interp:<15.2f}")
    
    print("\nThese functions can be used directly in FEA software!")
    print("ABAQUS: *CONDUCTIVITY, DEPENDENCIES=1")
    print("ANSYS: MPTEMP / MPDATA")
    print("COMSOL: Material properties → Temperature-dependent\n")


def example_7_uncertainty_quantification():
    """Example 7: Uncertainty quantification for probabilistic analysis"""
    print("=" * 80)
    print("EXAMPLE 7: Uncertainty Quantification")
    print("=" * 80)
    
    # Load mechanical properties
    df = pd.read_csv('thermo_mechanical_dataset/csv/mechanical_properties.csv')
    
    # Get data for R10S at 600°C (calibration)
    data = df[(df['Mix_ID'] == 'R10S') & 
             (df['Temperature_C'] == 600) & 
             (df['Data_Type'] == 'Calibration') &
             (df['Property_Type'] == 'Mechanical')]
    
    print("\nUncertainty Bounds for R10S at 600°C:")
    print("-" * 80)
    print(f"{'Property':<30} {'Mean':<15} {'Std Dev':<15} {'COV (%)':<15}")
    print("-" * 80)
    
    properties = {
        'Compressive Strength (MPa)': ('Compressive_Strength_MPa', 'Compressive_Strength_Std'),
        'Tensile Strength (MPa)': ('Tensile_Strength_MPa', 'Tensile_Strength_Std'),
        'Elastic Modulus (MPa)': ('Elastic_Modulus_MPa', 'Elastic_Modulus_Std'),
        'Poisson Ratio': ('Poisson_Ratio', 'Poisson_Ratio_Std'),
    }
    
    for prop_name, (mean_col, std_col) in properties.items():
        mean_val = data[mean_col].values[0]
        std_val = data[std_col].values[0]
        cov = (std_val / mean_val) * 100
        print(f"{prop_name:<30} {mean_val:<15.4f} {std_val:<15.4f} {cov:<15.2f}")
    
    print("\nFor Monte Carlo simulation:")
    print("  - Use normal distribution: N(μ, σ)")
    print("  - Number of samples: 1000-10000 recommended")
    print("  - Check physical bounds (e.g., E > 0, 0 < ν < 0.5)\n")


def example_8_rubber_optimization():
    """Example 8: Rubber content optimization analysis"""
    print("=" * 80)
    print("EXAMPLE 8: Rubber Content Optimization")
    print("=" * 80)
    
    # Load mechanical properties
    mech_df = pd.read_csv('thermo_mechanical_dataset/csv/mechanical_properties.csv')
    thermal_df = pd.read_csv('thermo_mechanical_dataset/csv/thermal_properties.csv')
    
    print("\nPerformance Metrics vs Rubber Content:")
    print("-" * 80)
    print(f"{'Mix':<8} {'Rubber %':<10} {'fc(20°C)':<12} {'Retention':<12} "
          f"{'k(20°C)':<12} {'Weight':<10}")
    print("-" * 80)
    
    for mix_id in ['C', 'R5S', 'R10S', 'R15S', 'R20S']:
        # Get mechanical data
        mech_cal = mech_df[(mech_df['Mix_ID'] == mix_id) & 
                          (mech_df['Data_Type'] == 'Calibration')]
        
        # Get thermal data
        thermal_cal = thermal_df[(thermal_df['Mix_ID'] == mix_id) & 
                                (thermal_df['Data_Type'] == 'Calibration')]
        
        # Extract rubber content
        rubber_pct = 0 if mix_id == 'C' else int(mix_id[1:3].replace('S', '').replace('L', ''))
        
        # Compressive strength at 20°C and 600°C
        fc_20 = mech_cal[mech_cal['Temperature_C'] == 20]['Compressive_Strength_MPa'].values[0]
        fc_600 = mech_cal[mech_cal['Temperature_C'] == 600]['Compressive_Strength_MPa'].values[0]
        retention = (fc_600 / fc_20) * 100
        
        # Thermal conductivity at 20°C
        k_20 = thermal_cal[thermal_cal['Temperature_C'] == 20]['Thermal_Conductivity_W_mK'].values[0]
        
        # Density (weight indicator)
        rho_20 = thermal_cal[thermal_cal['Temperature_C'] == 20]['Density_kg_m3'].values[0]
        weight_index = rho_20 / 2400  # Normalized to control mix
        
        print(f"{mix_id:<8} {rubber_pct:<10} {fc_20:<12.2f} {retention:<11.1f}% "
              f"{k_20:<12.4f} {weight_index:<10.4f}")
    
    print("\nRecommendation: 10-15% rubber content provides optimal balance")
    print("  - Acceptable ambient strength reduction (~20%)")
    print("  - Significant high-temperature retention improvement (~25%)")
    print("  - Good thermal insulation (~25% reduction in k)")
    print("  - Moderate weight reduction (~10%)\n")


def main():
    """Run all examples"""
    examples = [
        example_1_load_and_filter,
        example_2_retention_analysis,
        example_3_thermal_analysis,
        example_4_compare_validation,
        example_5_json_access,
        example_6_create_input_function,
        example_7_uncertainty_quantification,
        example_8_rubber_optimization,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")
        
        input("Press Enter to continue to next example...")
        print("\n\n")


if __name__ == "__main__":
    main()
