#!/usr/bin/env python3
"""
Example usage of the YSZ Material Properties Dataset
Demonstrates how to use the data for FEM preprocessing
"""

import pandas as pd
import numpy as np
from material_properties_analysis import YSZMaterialProperties

def example_basic_usage():
    """Basic example of loading and using the dataset"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Data Loading and Access")
    print("=" * 60)
    
    # Method 1: Direct CSV loading
    df = pd.read_csv('ysz_material_properties.csv')
    print(f"\nDataset shape: {df.shape}")
    print(f"Temperature range: {df['Temperature_C'].min()}°C to {df['Temperature_C'].max()}°C")
    print(f"\nFirst 3 rows of data:")
    print(df.head(3))
    
    # Method 2: Using the analysis class
    ysz = YSZMaterialProperties()
    
    # Get properties at specific temperature
    temp_c = 800  # Operating temperature
    temp_k = temp_c + 273.15
    
    print(f"\n\nProperties at {temp_c}°C:")
    print("-" * 40)
    print(f"Young's Modulus: {ysz.get_property('Youngs_Modulus_GPa', temp_k):.2f} GPa")
    print(f"CTE: {ysz.get_property('CTE_1e-6_per_K', temp_k):.2f} × 10⁻⁶/K")
    print(f"Poisson's Ratio: {ysz.get_property('Poissons_Ratio', temp_k):.3f}")

def example_thermal_stress_calculation():
    """Example: Calculate thermal stress for cooling from sintering"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Thermal Stress Calculation")
    print("=" * 60)
    
    ysz = YSZMaterialProperties()
    
    # Temperature change scenario
    T_sintering = 1500  # °C
    T_room = 25  # °C
    
    # Get average properties (simplified - should integrate properly)
    T_avg_k = ((T_sintering + T_room) / 2) + 273.15
    
    E_avg = ysz.get_property('Youngs_Modulus_GPa', T_avg_k) * 1e9  # Convert to Pa
    alpha_avg = ysz.get_property('CTE_1e-6_per_K', T_avg_k) * 1e-6  # Convert to 1/K
    nu = ysz.get_property('Poissons_Ratio', T_avg_k)
    
    # Temperature change
    delta_T = T_room - T_sintering
    
    # Thermal strain
    thermal_strain = alpha_avg * delta_T
    
    # Thermal stress (constrained case)
    thermal_stress = E_avg * thermal_strain / (1 - nu)
    
    print(f"\nCooling from {T_sintering}°C to {T_room}°C:")
    print(f"Temperature change: {delta_T}°C")
    print(f"Average CTE: {alpha_avg*1e6:.2f} × 10⁻⁶/K")
    print(f"Average E: {E_avg/1e9:.1f} GPa")
    print(f"Thermal strain: {thermal_strain*1e6:.1f} × 10⁻⁶")
    print(f"Thermal stress (fully constrained): {abs(thermal_stress/1e6):.1f} MPa (tensile)")
    
    # Check against strength
    strength_room = ysz.get_property('Characteristic_Strength_MPa', 25 + 273.15)
    print(f"\nCharacteristic strength at room temp: {strength_room:.0f} MPa")
    print(f"Safety factor: {strength_room / abs(thermal_stress/1e6):.2f}")

def example_creep_rate_calculation():
    """Example: Calculate creep rate at high temperature"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Creep Rate Calculation")
    print("=" * 60)
    
    ysz = YSZMaterialProperties()
    
    # Conditions
    T_c = 1200  # °C
    T_k = T_c + 273.15
    stress_MPa = 10  # Applied stress in MPa
    
    # Get creep parameters
    A = ysz.get_property('Creep_Rate_Coefficient_A', T_k)
    n = ysz.get_property('Creep_Stress_Exponent_n', T_k)
    Q = ysz.get_property('Creep_Activation_Energy_kJ_mol', T_k) * 1000  # Convert to J/mol
    
    # Gas constant
    R = 8.314  # J/(mol·K)
    
    # Norton power law: ε_dot = A * σ^n * exp(-Q/RT)
    # Note: This is simplified - actual implementation depends on units of A
    creep_rate = A * (stress_MPa ** n) * np.exp(-Q / (R * T_k))
    
    print(f"\nCreep conditions:")
    print(f"Temperature: {T_c}°C")
    print(f"Applied stress: {stress_MPa} MPa")
    print(f"\nCreep parameters:")
    print(f"A: {A:.2e}")
    print(f"n: {n:.2f}")
    print(f"Q: {Q/1000:.1f} kJ/mol")
    print(f"\nEstimated creep rate: {creep_rate:.2e} /s")
    
    # Time to 1% strain
    if creep_rate > 0:
        time_to_1pct = 0.01 / creep_rate
        print(f"Time to 1% strain: {time_to_1pct/3600:.1f} hours")

def example_weibull_failure_probability():
    """Example: Calculate failure probability using Weibull statistics"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Weibull Failure Probability")
    print("=" * 60)
    
    ysz = YSZMaterialProperties()
    
    # Conditions
    T_c = 800  # °C
    T_k = T_c + 273.15
    
    # Get Weibull parameters
    m = ysz.get_property('Weibull_Modulus', T_k)  # Weibull modulus
    sigma_0 = ysz.get_property('Characteristic_Strength_MPa', T_k)  # Characteristic strength
    
    # Applied stress levels to evaluate
    stress_levels = np.array([50, 100, 150, 200, 250])  # MPa
    
    print(f"\nWeibull parameters at {T_c}°C:")
    print(f"Weibull modulus (m): {m:.1f}")
    print(f"Characteristic strength (σ₀): {sigma_0:.1f} MPa")
    
    print(f"\nFailure probability vs applied stress:")
    print("-" * 40)
    print("Stress (MPa) | Failure Probability | Survival")
    print("-" * 40)
    
    for stress in stress_levels:
        # Weibull failure probability: P_f = 1 - exp(-(σ/σ₀)^m)
        P_failure = 1 - np.exp(-(stress/sigma_0)**m)
        P_survival = 1 - P_failure
        print(f"{stress:12.0f} | {P_failure:18.4f} | {P_survival:.4f}")
    
    # Find stress for 1% failure probability
    P_target = 0.01
    stress_1pct = sigma_0 * (-np.log(1 - P_target))**(1/m)
    print(f"\nStress for 1% failure probability: {stress_1pct:.1f} MPa")

def example_fem_preprocessing():
    """Example: Generate temperature-dependent property tables for FEM"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: FEM Preprocessing - Property Tables")
    print("=" * 60)
    
    ysz = YSZMaterialProperties()
    
    # Generate property table for FEM input
    temps_c = np.array([25, 200, 400, 600, 800, 1000, 1200, 1400])
    
    print("\nTemperature-Dependent Property Table for FEM:")
    print("-" * 80)
    print("T(°C) | E(GPa) | ν    | α(10⁻⁶/K) | k(W/mK) | ρ(kg/m³)")
    print("-" * 80)
    
    for T in temps_c:
        T_k = T + 273.15
        E = ysz.get_property('Youngs_Modulus_GPa', T_k)
        nu = ysz.get_property('Poissons_Ratio', T_k)
        alpha = ysz.get_property('CTE_1e-6_per_K', T_k)
        k = ysz.get_property('Thermal_Conductivity_W_mK', T_k)
        rho = ysz.get_property('Density_kg_m3', T_k)
        
        print(f"{T:5.0f} | {E:6.1f} | {nu:.3f} | {alpha:9.2f} | {k:7.3f} | {rho:8.1f}")
    
    print("\nNote: These values can be directly input into FEM software")
    print("      temperature-dependent material property definitions.")

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" YSZ MATERIAL PROPERTIES DATASET - USAGE EXAMPLES")
    print("=" * 60)
    
    # Run all examples
    example_basic_usage()
    example_thermal_stress_calculation()
    example_creep_rate_calculation()
    example_weibull_failure_probability()
    example_fem_preprocessing()
    
    print("\n" + "=" * 60)
    print(" Examples completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()