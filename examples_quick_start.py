#!/usr/bin/env python3
"""
Quick Start Examples for YSZ Material Properties Dataset

This script demonstrates common use cases for the dataset.
Run this file to see practical examples of how to use the data.
"""

from material_properties_loader import YSZMaterialProperties
import numpy as np

def example_1_basic_property_lookup():
    """Example 1: Look up a single property at a specific temperature."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Property Lookup")
    print("="*70)
    
    ysz = YSZMaterialProperties()
    
    # Get Young's modulus at SOFC operating temperature
    T_operating = 800  # °C
    E = ysz.get_property('Youngs_Modulus_GPa', T_operating)
    
    print(f"\nYoung's Modulus at {T_operating}°C: {E:.2f} GPa")
    print(f"This represents a {((205-E)/205*100):.1f}% decrease from room temperature!")
    

def example_2_thermal_stress_calculation():
    """Example 2: Calculate thermal stress during heat-up."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Thermal Stress During SOFC Startup")
    print("="*70)
    
    ysz = YSZMaterialProperties()
    
    T_room = 25  # °C
    T_operating = 800  # °C
    
    # Use properties at mean temperature
    T_mean = (T_room + T_operating) / 2
    
    E = ysz.get_property('Youngs_Modulus_GPa', T_mean)  # GPa
    alpha = ysz.get_property('CTE_1e-6_K', T_mean)  # 10^-6 / K
    
    # Thermal stress (simplified, assuming full constraint)
    delta_T = T_operating - T_room  # K
    sigma_thermal = E * 1000 * alpha * 1e-6 * delta_T  # MPa
    
    print(f"\nHeating from {T_room}°C to {T_operating}°C:")
    print(f"  ΔT = {delta_T} K")
    print(f"  E (at {T_mean}°C) = {E:.2f} GPa")
    print(f"  α (at {T_mean}°C) = {alpha:.2f} × 10⁻⁶ K⁻¹")
    print(f"  → Thermal stress: {sigma_thermal:.1f} MPa")
    
    # Check against strength
    sigma_char = ysz.get_property('Characteristic_Strength_MPa', T_operating)
    safety_factor = sigma_char / sigma_thermal
    print(f"\nCharacteristic strength at {T_operating}°C: {sigma_char:.1f} MPa")
    print(f"Safety factor: {safety_factor:.2f}")
    
    if safety_factor > 2.0:
        print("✓ Design is safe (SF > 2.0)")
    else:
        print("⚠ WARNING: Low safety factor! Consider stress relief.")


def example_3_creep_analysis():
    """Example 3: Evaluate creep during sintering."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Creep Analysis During Sintering")
    print("="*70)
    
    ysz = YSZMaterialProperties()
    
    T_sinter = 1400  # °C
    stress = 15  # MPa (from weight of green body)
    grain_size = 0.8  # μm (fine powder)
    time_hours = 2  # Sintering dwell time
    
    # Calculate creep rate
    creep_rate = ysz.get_creep_rate(stress, T_sinter, grain_size)
    
    # Total creep strain
    time_seconds = time_hours * 3600
    creep_strain = creep_rate * time_seconds
    creep_strain_percent = creep_strain * 100
    
    print(f"\nSintering conditions:")
    print(f"  Temperature: {T_sinter}°C")
    print(f"  Applied stress: {stress} MPa")
    print(f"  Grain size: {grain_size} μm")
    print(f"  Dwell time: {time_hours} hours")
    
    print(f"\nCreep analysis:")
    print(f"  Creep rate: {creep_rate:.3e} s⁻¹")
    print(f"  Total creep strain: {creep_strain:.4f} ({creep_strain_percent:.2f}%)")
    
    if creep_strain_percent > 5:
        print("  ⚠ Significant creep! Expect dimensional changes.")
    elif creep_strain_percent > 1:
        print("  → Moderate creep, may need tolerance adjustment.")
    else:
        print("  ✓ Minimal creep, dimensions stable.")


def example_4_weibull_reliability():
    """Example 4: Calculate failure probability using Weibull statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Weibull Reliability Analysis")
    print("="*70)
    
    ysz = YSZMaterialProperties()
    
    T = 25  # °C (room temperature test)
    
    # Get Weibull parameters
    m = ysz.get_property('Weibull_Modulus_m', T)
    sigma_0 = ysz.get_property('Characteristic_Strength_MPa', T)
    
    print(f"\nWeibull parameters at {T}°C:")
    print(f"  Modulus (m): {m:.2f}")
    print(f"  Characteristic strength (σ₀): {sigma_0:.1f} MPa")
    
    # Calculate failure probability for different stress levels
    print(f"\nFailure probability vs applied stress:")
    print(f"{'Stress (MPa)':>15} {'P_fail (%)':>15} {'Reliability (%)':>18}")
    print("-" * 50)
    
    for stress in [200, 250, 300, 350, 400, 450]:
        P_fail = 1 - np.exp(-((stress / sigma_0) ** m))
        reliability = (1 - P_fail) * 100
        print(f"{stress:>15d} {P_fail*100:>15.2f} {reliability:>18.2f}")
    
    print("\nInterpretation:")
    print("  - At 63.2% failure: stress ≈ σ₀ (characteristic strength)")
    print("  - Higher m → less scatter → more predictable")
    print("  - m = 10.5 is typical for well-processed ceramics")


def example_5_temperature_profile():
    """Example 5: Generate properties along a temperature profile."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Properties Along Thermal Gradient")
    print("="*70)
    
    ysz = YSZMaterialProperties()
    
    # Simulate a thermal gradient (e.g., through electrolyte thickness)
    print("\nThermal gradient from hot side to cold side:")
    print(f"{'Position (mm)':>15} {'Temp (°C)':>12} {'E (GPa)':>12} {'α (10⁻⁶/K)':>15}")
    print("-" * 56)
    
    # Linear temperature gradient
    positions = np.linspace(0, 0.5, 6)  # 0 to 0.5 mm (electrolyte thickness)
    T_hot = 850  # °C
    T_cold = 750  # °C
    
    for x in positions:
        T = T_hot - (T_hot - T_cold) * x / 0.5
        E = ysz.get_property('Youngs_Modulus_GPa', T)
        alpha = ysz.get_property('CTE_1e-6_K', T)
        print(f"{x:>15.3f} {T:>12.1f} {E:>12.2f} {alpha:>15.2f}")
    
    print("\nNote: Property gradients create additional stress!")
    print("      Use thermo-mechanical coupling in FEM for accuracy.")


def example_6_custom_export():
    """Example 6: Export custom dataset for FEM."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Custom Dataset for FEM")
    print("="*70)
    
    ysz = YSZMaterialProperties()
    
    # Create custom temperature points for your specific analysis
    # Example: SOFC thermal cycle
    temp_cycle = np.array([25, 100, 200, 300, 400, 500, 600, 700, 
                          750, 800, 800, 750, 700, 600, 400, 200, 25])
    
    # Export to CSV
    output_file = 'fem_thermal_cycle.csv'
    ysz.export_for_fem(output_file, temperature_points=temp_cycle)
    
    print(f"\nCustom thermal cycle exported to: {output_file}")
    print(f"Temperature points: {len(temp_cycle)}")
    print(f"Range: {temp_cycle.min()}°C to {temp_cycle.max()}°C")
    print("\nThis file can be directly imported into:")
    print("  - ANSYS (MPTEMP/MPDATA)")
    print("  - COMSOL (Interpolation function)")
    print("  - Abaqus (*MATERIAL)")
    print("  - Custom FEM codes")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("YSZ MATERIAL PROPERTIES DATASET - QUICK START EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates 6 common use cases:")
    print("  1. Basic property lookup")
    print("  2. Thermal stress calculation")
    print("  3. Creep analysis during sintering")
    print("  4. Weibull reliability analysis")
    print("  5. Properties along thermal gradient")
    print("  6. Custom dataset export for FEM")
    
    try:
        example_1_basic_property_lookup()
        example_2_thermal_stress_calculation()
        example_3_creep_analysis()
        example_4_weibull_reliability()
        example_5_temperature_profile()
        example_6_custom_export()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Modify these examples for your specific case")
        print("  2. Read USAGE_GUIDE.md for more advanced usage")
        print("  3. Run validate_dataset.py to verify data integrity")
        print("  4. Generate custom datasets with generate_custom_dataset.py")
        print("\nHappy modeling! 🚀")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("Make sure all required files are in the same directory.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())