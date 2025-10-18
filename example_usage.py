#!/usr/bin/env python3
"""
Example usage of the Thermo-Mechanical Dataset Generator
Demonstrates how to generate, access, and analyze the dataset
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from thermo_mechanical_dataset import ThermoMechanicalDataset

def example_basic_generation():
    """Example 1: Basic dataset generation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Dataset Generation")
    print("="*60)
    
    # Initialize the dataset generator
    dataset = ThermoMechanicalDataset(
        output_dir='./example_output',
        seed=42,
        uncertainty_level=0.05
    )
    
    # Generate complete dataset
    calibration_data, validation_data = dataset.generate_complete_dataset(
        calibration_ratio=0.7,
        n_validation_sets=3
    )
    
    print("\nDataset generated successfully!")
    print(f"Number of mixes: {len(calibration_data.keys())}")
    print(f"Mixes included: {', '.join(calibration_data.keys())}")
    print(f"Validation sets: {len(validation_data)}")
    
    return calibration_data, validation_data

def example_access_properties(calibration_data):
    """Example 2: Accessing specific properties"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Accessing Properties")
    print("="*60)
    
    # Access thermal properties for R10S mix
    mix_id = 'R10S'
    thermal_props = calibration_data[mix_id]['thermal']
    
    print(f"\nThermal properties for {mix_id}:")
    print(f"Temperature points: {len(thermal_props['temperature'])}")
    print(f"Temperature range: {min(thermal_props['temperature']):.0f}°C - {max(thermal_props['temperature']):.0f}°C")
    
    # Get conductivity at specific temperatures
    temps = thermal_props['temperature']
    conductivity = thermal_props['thermal_conductivity']
    
    # Find conductivity at 20°C and 400°C
    idx_20 = 0
    idx_400 = np.argmin(np.abs(np.array(temps) - 400))
    
    print(f"\nThermal conductivity:")
    print(f"  At 20°C: {conductivity[idx_20]:.3f} W/(m·K)")
    print(f"  At 400°C: {conductivity[idx_400]:.3f} W/(m·K)")
    print(f"  Reduction: {(1 - conductivity[idx_400]/conductivity[idx_20])*100:.1f}%")
    
    # Access mechanical properties
    mech_props = calibration_data[mix_id]['mechanical']
    E_20 = mech_props['elastic']['elastic_modulus'][idx_20]
    E_400 = mech_props['elastic']['elastic_modulus'][idx_400]
    
    print(f"\nElastic modulus:")
    print(f"  At 20°C: {E_20:.1f} GPa")
    print(f"  At 400°C: {E_400:.1f} GPa")
    print(f"  Reduction: {(1 - E_400/E_20)*100:.1f}%")

def example_compare_mixes(calibration_data):
    """Example 3: Compare properties between mixes"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Comparing Mixes")
    print("="*60)
    
    # Compare compressive strength at room temperature
    print("\nCompressive strength at 20°C:")
    print("-" * 40)
    
    for mix_id in ['C', 'R5S', 'R10S', 'R15S', 'R20S']:
        if mix_id in calibration_data:
            fc = calibration_data[mix_id]['mechanical']['strength']['compressive_strength'][0]
            print(f"{mix_id:6s}: {fc:.1f} MPa")
    
    # Compare thermal conductivity
    print("\nThermal conductivity at 20°C:")
    print("-" * 40)
    
    for mix_id in ['C', 'R5S', 'R10S', 'R15S', 'R20S']:
        if mix_id in calibration_data:
            k = calibration_data[mix_id]['thermal']['thermal_conductivity'][0]
            print(f"{mix_id:6s}: {k:.3f} W/(m·K)")

def example_plot_properties(calibration_data):
    """Example 4: Plotting property evolution"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Plotting Properties")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define colors for each mix
    colors = {'C': 'black', 'R5S': 'blue', 'R10S': 'green', 
              'R15S': 'orange', 'R20S': 'red', 'R10L': 'purple'}
    
    # Plot thermal conductivity
    for mix_id in ['C', 'R10S', 'R20S']:
        if mix_id in calibration_data:
            thermal = calibration_data[mix_id]['thermal']
            axes[0, 0].plot(thermal['temperature'], 
                          thermal['thermal_conductivity'],
                          label=mix_id, color=colors[mix_id], linewidth=2)
    
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Thermal Conductivity (W/m·K)')
    axes[0, 0].set_title('Thermal Conductivity Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot elastic modulus
    for mix_id in ['C', 'R10S', 'R20S']:
        if mix_id in calibration_data:
            elastic = calibration_data[mix_id]['mechanical']['elastic']
            axes[0, 1].plot(elastic['temperature'],
                          elastic['elastic_modulus'],
                          label=mix_id, color=colors[mix_id], linewidth=2)
    
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Elastic Modulus (GPa)')
    axes[0, 1].set_title('Elastic Modulus Degradation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot compressive strength
    for mix_id in ['C', 'R10S', 'R20S']:
        if mix_id in calibration_data:
            strength = calibration_data[mix_id]['mechanical']['strength']
            axes[1, 0].plot(strength['temperature'],
                          strength['compressive_strength'],
                          label=mix_id, color=colors[mix_id], linewidth=2)
    
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Compressive Strength (MPa)')
    axes[1, 0].set_title('Compressive Strength Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot porosity
    for mix_id in ['C', 'R10S', 'R20S']:
        if mix_id in calibration_data:
            porosity = calibration_data[mix_id]['transport']['porosity']
            axes[1, 1].plot(porosity['temperature'],
                          porosity['total_porosity'],
                          label=mix_id, color=colors[mix_id], linewidth=2)
    
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Porosity (fraction)')
    axes[1, 1].set_title('Porosity Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./example_output/example_plots.png', dpi=150)
    print("\nPlots saved to: ./example_output/example_plots.png")
    plt.show()

def example_export_to_csv(calibration_data):
    """Example 5: Export data to CSV for external analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Exporting to CSV")
    print("="*60)
    
    # Export R10S thermal properties to CSV
    mix_id = 'R10S'
    thermal = calibration_data[mix_id]['thermal']
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature_C': thermal['temperature'],
        'Thermal_Conductivity_W_mK': thermal['thermal_conductivity'],
        'Specific_Heat_J_kgK': thermal['specific_heat_capacity'],
        'Density_kg_m3': thermal['density'],
        'Thermal_Diffusivity_mm2_s': thermal['thermal_diffusivity']
    })
    
    # Save to CSV
    csv_path = './example_output/R10S_thermal_properties.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nExported thermal properties to: {csv_path}")
    
    # Show first few rows
    print("\nFirst 5 rows of exported data:")
    print(df.head())

def example_fea_input_generation(calibration_data):
    """Example 6: Generate FEA input snippet"""
    print("\n" + "="*60)
    print("EXAMPLE 6: FEA Input Generation")
    print("="*60)
    
    mix_id = 'R10S'
    thermal = calibration_data[mix_id]['thermal']
    elastic = calibration_data[mix_id]['mechanical']['elastic']
    
    print(f"\nABAQUS Material Input for {mix_id}:")
    print("-" * 40)
    print(f"*Material, name=CONCRETE_{mix_id}")
    print("*Density")
    
    # Sample a few temperature points
    for i in range(0, len(thermal['temperature']), 10):
        temp = thermal['temperature'][i]
        rho = thermal['density'][i]
        print(f" {rho:.1f}, {temp:.1f}")
    
    print("*Elastic, type=ISOTROPIC")
    for i in range(0, len(elastic['temperature']), 10):
        temp = elastic['temperature'][i]
        E = elastic['elastic_modulus'][i] * 1e9
        nu = elastic['poisson_ratio'][i]
        print(f" {E:.3e}, {nu:.4f}, {temp:.1f}")
    
    print("\n(Truncated for brevity...)")

def example_validation_comparison(calibration_data, validation_data):
    """Example 7: Compare calibration vs validation data"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Calibration vs Validation")
    print("="*60)
    
    if len(validation_data) == 0:
        print("No validation data available")
        return
    
    # Get first validation set
    val_set_1 = list(validation_data.values())[0]
    mix_id = 'R10S'
    
    # Compare thermal conductivity at 20°C
    cal_k = calibration_data[mix_id]['thermal']['thermal_conductivity'][0]
    val_k = val_set_1[mix_id]['thermal']['thermal_conductivity'][0]
    
    print(f"\nThermal conductivity at 20°C for {mix_id}:")
    print(f"  Calibration: {cal_k:.4f} W/(m·K)")
    print(f"  Validation:  {val_k:.4f} W/(m·K)")
    print(f"  Difference:  {abs(cal_k - val_k):.4f} ({abs(cal_k - val_k)/cal_k*100:.1f}%)")
    
    # Statistical analysis
    cal_values = calibration_data[mix_id]['thermal']['thermal_conductivity']
    val_values = val_set_1[mix_id]['thermal']['thermal_conductivity']
    
    print(f"\nStatistical comparison (full temperature range):")
    print(f"  Calibration mean: {np.mean(cal_values):.4f}")
    print(f"  Validation mean:  {np.mean(val_values):.4f}")
    print(f"  Calibration std:  {np.std(cal_values):.4f}")
    print(f"  Validation std:   {np.std(val_values):.4f}")

def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("THERMO-MECHANICAL DATASET GENERATOR - EXAMPLES")
    print("="*80)
    
    # Run examples
    print("\nRunning examples...")
    
    # Generate dataset
    calibration_data, validation_data = example_basic_generation()
    
    # Access properties
    example_access_properties(calibration_data)
    
    # Compare mixes
    example_compare_mixes(calibration_data)
    
    # Plot properties
    example_plot_properties(calibration_data)
    
    # Export to CSV
    example_export_to_csv(calibration_data)
    
    # Generate FEA input
    example_fea_input_generation(calibration_data)
    
    # Compare calibration vs validation
    example_validation_comparison(calibration_data, validation_data)
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nCheck the './example_output' directory for generated files.")

if __name__ == "__main__":
    main()