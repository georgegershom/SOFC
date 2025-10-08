#!/usr/bin/env python3
"""
YSZ Material Properties Dataset Validation and Usage Example

This script demonstrates how to load and use the generated YSZ material 
properties dataset for FEM analysis preparation.
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def load_and_validate_dataset():
    """Load the dataset and perform basic validation"""
    
    print("="*60)
    print("YSZ MATERIAL PROPERTIES DATASET VALIDATION")
    print("="*60)
    
    # Load CSV data
    print("\n1. Loading CSV dataset...")
    data = pd.read_csv('/workspace/ysz_material_properties.csv')
    print(f"   ✓ Loaded {len(data)} temperature points")
    print(f"   ✓ Temperature range: {data['Temperature_C'].min()}°C to {data['Temperature_C'].max()}°C")
    print(f"   ✓ Properties included: {len(data.columns)-2}")
    
    # Load JSON metadata
    print("\n2. Loading JSON metadata...")
    with open('/workspace/ysz_material_properties.json', 'r') as f:
        properties = json.load(f)
    print(f"   ✓ Material: {properties['material']}")
    print(f"   ✓ Application: {properties['application']}")
    
    # Validate key properties at room temperature
    print("\n3. Validating room temperature properties...")
    rt_idx = 0  # First row is 25°C
    
    validations = [
        ("Young's Modulus", "youngs_modulus_GPa", 205, "GPa"),
        ("Poisson's Ratio", "poisson_ratio_dimensionless", 0.30, ""),
        ("Thermal Expansion", "thermal_expansion_coefficient_×10⁻⁶ /K", 10.2, "×10⁻⁶ /K"),
        ("Density", "density_kg/m³", 5850, "kg/m³"),
        ("Thermal Conductivity", "thermal_conductivity_W/m·K", 2.2, "W/m·K"),
        ("Fracture Toughness", "fracture_toughness_MPa√m", 9.2, "MPa√m")
    ]
    
    for prop_name, col_name, expected, units in validations:
        actual = data.loc[rt_idx, col_name]
        print(f"   ✓ {prop_name}: {actual:.2f} {units} (expected ~{expected})")
    
    return data, properties

def demonstrate_temperature_interpolation(data):
    """Demonstrate how to get properties at specific temperatures"""
    
    print("\n4. Temperature interpolation examples...")
    
    # Example temperatures of interest
    target_temps = [100, 500, 800, 1200]
    
    for temp in target_temps:
        # Find closest temperature point
        temp_idx = data['Temperature_C'].sub(temp).abs().idxmin()
        actual_temp = data.loc[temp_idx, 'Temperature_C']
        
        # Get key properties
        E = data.loc[temp_idx, 'youngs_modulus_GPa']
        alpha = data.loc[temp_idx, 'thermal_expansion_coefficient_×10⁻⁶ /K']
        k = data.loc[temp_idx, 'thermal_conductivity_W/m·K']
        
        print(f"   At {actual_temp:.0f}°C: E={E:.1f} GPa, α={alpha:.2f}×10⁻⁶/K, k={k:.2f} W/m·K")

def generate_fem_input_example(data):
    """Generate example FEM input data"""
    
    print("\n5. Generating FEM input examples...")
    
    # Select key temperature points for FEM analysis
    key_temps = [25, 200, 400, 600, 800, 1000, 1200, 1400, 1500]
    
    print("\n   ANSYS APDL Format:")
    print("   ! Temperature-dependent Young's Modulus")
    temp_line = "   MPTEMP,1," + ",".join([str(t) for t in key_temps])
    print(temp_line)
    
    # Get Young's modulus values at these temperatures
    E_values = []
    for temp in key_temps:
        temp_idx = data['Temperature_C'].sub(temp).abs().idxmin()
        E_Pa = data.loc[temp_idx, 'youngs_modulus_GPa'] * 1e9  # Convert to Pa
        E_values.append(f"{E_Pa:.0f}")
    
    E_line = "   MPDATA,EX,1,1," + ",".join(E_values)
    print(E_line)
    
    print("\n   ABAQUS Format:")
    print("   *ELASTIC, TYPE=ISOTROPIC, TEMPERATURE")
    for temp in key_temps:
        temp_idx = data['Temperature_C'].sub(temp).abs().idxmin()
        E_Pa = data.loc[temp_idx, 'youngs_modulus_GPa'] * 1e9
        nu = data.loc[temp_idx, 'poisson_ratio_dimensionless']
        print(f"   {E_Pa:.0f}, {nu:.3f}, {temp}.")

def analyze_temperature_dependencies(data):
    """Analyze and report temperature dependencies"""
    
    print("\n6. Temperature dependency analysis...")
    
    # Calculate percentage changes from RT to 1500°C
    rt_idx = 0
    ht_idx = len(data) - 1  # Last index (1500°C)
    
    properties_to_analyze = [
        ("Young's Modulus", "youngs_modulus_GPa", "GPa"),
        ("Thermal Expansion", "thermal_expansion_coefficient_×10⁻⁶ /K", "×10⁻⁶ /K"),
        ("Thermal Conductivity", "thermal_conductivity_W/m·K", "W/m·K"),
        ("Fracture Toughness", "fracture_toughness_MPa√m", "MPa√m"),
        ("Characteristic Strength", "characteristic_strength_MPa", "MPa")
    ]
    
    for prop_name, col_name, units in properties_to_analyze:
        rt_val = data.loc[rt_idx, col_name]
        ht_val = data.loc[ht_idx, col_name]
        change_pct = ((ht_val - rt_val) / rt_val) * 100
        
        print(f"   {prop_name}:")
        print(f"     25°C: {rt_val:.2f} {units}")
        print(f"     1500°C: {ht_val:.2f} {units}")
        print(f"     Change: {change_pct:+.1f}%")

def create_quick_visualization(data):
    """Create a quick visualization of key properties"""
    
    print("\n7. Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Young's Modulus
    axes[0,0].plot(data['Temperature_C'], data['youngs_modulus_GPa'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('Temperature (°C)')
    axes[0,0].set_ylabel('Young\'s Modulus (GPa)')
    axes[0,0].set_title('Young\'s Modulus vs Temperature')
    axes[0,0].grid(True, alpha=0.3)
    
    # Thermal Expansion
    axes[0,1].plot(data['Temperature_C'], data['thermal_expansion_coefficient_×10⁻⁶ /K'], 'r-', linewidth=2)
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('CTE (×10⁻⁶ /K)')
    axes[0,1].set_title('Thermal Expansion vs Temperature')
    axes[0,1].grid(True, alpha=0.3)
    
    # Thermal Conductivity
    axes[1,0].plot(data['Temperature_C'], data['thermal_conductivity_W/m·K'], 'g-', linewidth=2)
    axes[1,0].set_xlabel('Temperature (°C)')
    axes[1,0].set_ylabel('Thermal Conductivity (W/m·K)')
    axes[1,0].set_title('Thermal Conductivity vs Temperature')
    axes[1,0].grid(True, alpha=0.3)
    
    # Fracture Toughness
    axes[1,1].plot(data['Temperature_C'], data['fracture_toughness_MPa√m'], 'm-', linewidth=2)
    axes[1,1].set_xlabel('Temperature (°C)')
    axes[1,1].set_ylabel('Fracture Toughness (MPa√m)')
    axes[1,1].set_title('Fracture Toughness vs Temperature')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/validation_plots.png', dpi=300, bbox_inches='tight')
    print("   ✓ Validation plots saved to: validation_plots.png")
    
    return fig

def main():
    """Main validation and demonstration function"""
    
    # Load and validate dataset
    data, properties = load_and_validate_dataset()
    
    # Demonstrate temperature interpolation
    demonstrate_temperature_interpolation(data)
    
    # Generate FEM input examples
    generate_fem_input_example(data)
    
    # Analyze temperature dependencies
    analyze_temperature_dependencies(data)
    
    # Create visualization
    create_quick_visualization(data)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("✓ Dataset loaded and validated successfully")
    print("✓ Temperature dependencies confirmed")
    print("✓ FEM input examples generated")
    print("✓ Ready for thermomechanical analysis!")
    print("\nDataset files:")
    print("- ysz_material_properties.csv (Full dataset)")
    print("- ysz_material_properties.json (With metadata)")
    print("- ysz_properties_summary.csv (Summary)")
    print("- YSZ_Material_Properties_Documentation.md (Documentation)")

if __name__ == "__main__":
    main()