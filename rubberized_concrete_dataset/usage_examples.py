#!/usr/bin/env python3
"""
Usage Examples for Rubberized Concrete Dataset
Demonstrates how to load, analyze, and visualize the dataset

Author: AI Assistant
Date: 2025-10-18
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def example_1_load_material_properties():
    """Example 1: Load and interpolate material properties"""
    print("Example 1: Loading Material Properties")
    
    # Load thermal properties
    with open('thermal_properties.json', 'r') as f:
        thermal_data = json.load(f)
    
    # Extract data for 10% rubber content
    rubber_10 = thermal_data['rubber_10pct']
    temperatures = np.array(rubber_10['temperature'])
    conductivity = np.array(rubber_10['thermal_conductivity'])
    
    # Create interpolation function
    k_interp = interpolate.interp1d(temperatures, conductivity, kind='cubic')
    
    # Get thermal conductivity at specific temperature
    T_query = 300  # °C
    k_at_300 = k_interp(T_query)
    print(f"Thermal conductivity at {T_query}°C: {k_at_300:.3f} W/m·K")
    
    return k_interp

def example_2_compare_rubber_effects():
    """Example 2: Compare effects of rubber content"""
    print("\nExample 2: Comparing Rubber Content Effects")
    
    # Load mechanical properties CSV
    df = pd.read_csv('mechanical_properties.csv')
    
    # Compare compressive strength at 400°C for different rubber contents
    temp_target = 400
    df_400 = df[df['temperature_C'].round() == temp_target]
    
    print(f"Compressive strength at {temp_target}°C:")
    for rubber in sorted(df_400['rubber_content_pct'].unique()):
        strength = df_400[df_400['rubber_content_pct'] == rubber]['compressive_strength_MPa'].iloc[0]
        print(f"  {rubber}% rubber: {strength:.1f} MPa")

def example_3_analyze_temperature_evolution():
    """Example 3: Analyze temperature evolution validation data"""
    print("\nExample 3: Temperature Evolution Analysis")
    
    # Load temperature evolution data
    df = pd.read_csv('temperature_evolution_validation.csv')
    
    # Filter for specific conditions
    conditions = (
        (df['specimen_type'] == 'small_cube') &
        (df['fire_curve'] == 'ISO834') &
        (df['rubber_content_pct'] == 10) &
        (df['thermocouple_location'] == 'center')
    )
    data = df[conditions]
    
    # Find time to reach 500°C at center
    temp_500_data = data[data['temperature_C'] >= 500]
    if not temp_500_data.empty:
        time_to_500 = temp_500_data['time_hours'].iloc[0]
        print(f"Time to reach 500°C at center (10% rubber): {time_to_500:.2f} hours")

def example_4_strain_decomposition():
    """Example 4: Analyze strain components"""
    print("\nExample 4: Strain Component Analysis")
    
    # Load strain data
    df = pd.read_csv('deformation_strain_validation.csv')
    
    # Filter for medium load scenario with 15% rubber
    conditions = (
        (df['rubber_content_pct'] == 15) &
        (df['loading_scenario'] == 'medium_load')
    )
    data = df[conditions]
    
    # Find maximum strain components at end of test
    final_data = data.iloc[-1]
    print(f"Final strain components (15% rubber, medium load):")
    print(f"  Total strain: {final_data['total_strain']*1000:.2f} ×10⁻³")
    print(f"  Thermal strain: {final_data['thermal_strain']*1000:.2f} ×10⁻³")
    print(f"  Mechanical strain: {final_data['mechanical_strain']*1000:.2f} ×10⁻³")
    print(f"  Creep strain: {final_data['creep_strain']*1000:.2f} ×10⁻³")

def example_5_spalling_analysis():
    """Example 5: Spalling behavior analysis"""
    print("\nExample 5: Spalling Analysis")
    
    # Load spalling summary data
    df = pd.read_csv('spalling_failure_summary.csv')
    
    # Calculate spalling occurrence rate by rubber content
    spall_rates = df.groupby('rubber_content_pct')['spalling_occurred'].mean()
    
    print("Spalling occurrence rates:")
    for rubber, rate in spall_rates.items():
        print(f"  {rubber}% rubber: {rate:.1%}")
    
    # Find conditions with highest spalling risk
    high_risk = df[df['spalling_occurred'] == True]
    if not high_risk.empty:
        worst_condition = high_risk.loc[high_risk['max_spalling_depth_mm'].idxmax()]
        print(f"\nWorst spalling case:")
        print(f"  Rubber content: {worst_condition['rubber_content_pct']}%")
        print(f"  Test condition: {worst_condition['test_condition']}")
        print(f"  Max depth: {worst_condition['max_spalling_depth_mm']:.1f} mm")

def example_6_create_material_model():
    """Example 6: Create simple material property model"""
    print("\nExample 6: Material Property Modeling")
    
    # Load thermal properties
    df = pd.read_csv('thermal_properties.csv')
    
    def thermal_conductivity_model(T, rubber_content):
        """
        Simple model for thermal conductivity
        k(T, rubber) = k0 * (1 - a*rubber/100) * exp(-b*T/1000)
        """
        k0 = 1.8  # Base conductivity
        a = 0.3   # Rubber reduction factor
        b = 0.8   # Temperature degradation factor
        
        return k0 * (1 - a * rubber_content / 100) * np.exp(-b * T / 1000)
    
    # Test the model
    T_test = 500  # °C
    rubber_test = 10  # %
    k_model = thermal_conductivity_model(T_test, rubber_test)
    
    # Compare with dataset
    data_point = df[(df['temperature_C'].round() == T_test) & 
                   (df['rubber_content_pct'] == rubber_test)]
    if not data_point.empty:
        k_data = data_point['thermal_conductivity_W_m_K'].iloc[0]
        error = abs(k_model - k_data) / k_data * 100
        print(f"Model validation at {T_test}°C, {rubber_test}% rubber:")
        print(f"  Model prediction: {k_model:.3f} W/m·K")
        print(f"  Dataset value: {k_data:.3f} W/m·K")
        print(f"  Error: {error:.1f}%")

def example_7_visualization():
    """Example 7: Create custom visualizations"""
    print("\nExample 7: Custom Visualization")
    
    # Load mechanical properties
    df = pd.read_csv('mechanical_properties.csv')
    
    # Create strength degradation plot
    plt.figure(figsize=(10, 6))
    
    rubber_contents = [0, 10, 20]
    colors = ['red', 'blue', 'green']
    
    for i, rubber in enumerate(rubber_contents):
        data = df[df['rubber_content_pct'] == rubber]
        plt.plot(data['temperature_C'], data['compressive_strength_MPa'], 
                label=f'{rubber}% rubber', color=colors[i], linewidth=2)
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Compressive Strength (MPa)')
    plt.title('Compressive Strength Degradation with Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('custom_strength_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Custom plot saved as 'custom_strength_plot.png'")

if __name__ == "__main__":
    print("Rubberized Concrete Dataset - Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_1_load_material_properties()
    example_2_compare_rubber_effects()
    example_3_analyze_temperature_evolution()
    example_4_strain_decomposition()
    example_5_spalling_analysis()
    example_6_create_material_model()
    example_7_visualization()
    
    print("\nAll examples completed successfully!")
