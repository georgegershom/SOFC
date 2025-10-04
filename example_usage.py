#!/usr/bin/env python3
"""
Example usage of the Synthetic Synchrotron X-ray Data Generator

This script demonstrates how to generate realistic SOFC creep deformation data
for validation of computational models and analysis algorithms.
"""

import numpy as np
from synchrotron_data_generator import (
    SynchrotronDataGenerator, 
    MaterialProperties, 
    OperationalParameters, 
    SampleGeometry
)

def create_example_sofc_interconnect():
    """Create example SOFC interconnect material properties"""
    
    # Typical ferritic stainless steel for SOFC interconnects (e.g., Crofer 22 APU)
    material_props = MaterialProperties(
        alloy_composition={
            'Fe': 76.8,   # Iron (balance)
            'Cr': 22.0,   # Chromium (for oxidation resistance)
            'Mn': 0.5,    # Manganese
            'Ti': 0.08,   # Titanium (grain refinement)
            'La': 0.04,   # Lanthanum (oxidation resistance)
            'C': 0.03     # Carbon
        },
        grain_size_mean=15.0,      # micrometers
        grain_size_std=5.0,        # micrometers
        initial_porosity=0.002,    # 0.2% initial porosity
        elastic_modulus=200.0,     # GPa (typical for ferritic steel)
        poisson_ratio=0.3,
        thermal_expansion_coeff=12.0e-6,  # 1/K
        creep_exponent=5.0,        # Norton creep law exponent
        activation_energy=300.0    # kJ/mol
    )
    
    return material_props

def create_example_operating_conditions():
    """Create example SOFC operating conditions for creep testing"""
    
    # Typical intermediate temperature SOFC conditions
    op_params = OperationalParameters(
        temperature=700.0,         # Celsius (intermediate temperature)
        mechanical_stress=50.0,    # MPa (typical compressive load)
        time_points=[0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0],  # hours
        atmosphere="Air",
        heating_rate=5.0,          # K/min
        cooling_rate=2.0           # K/min
    )
    
    return op_params

def create_example_sample_geometry():
    """Create example sample geometry"""
    
    sample_geom = SampleGeometry(
        length=5.0,      # mm
        width=2.0,       # mm  
        thickness=0.5,   # mm
        shape="rectangular",
        volume=5.0       # mm³
    )
    
    return sample_geom

def generate_high_stress_scenario():
    """Generate data for high-stress accelerated creep scenario"""
    
    print("\n=== HIGH STRESS ACCELERATED CREEP SCENARIO ===")
    
    # High stress conditions for accelerated testing
    material_props = create_example_sofc_interconnect()
    
    op_params = OperationalParameters(
        temperature=750.0,         # Higher temperature
        mechanical_stress=100.0,   # Higher stress
        time_points=[0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],  # hours
        atmosphere="Air + 3% H2O",
        heating_rate=5.0,
        cooling_rate=2.0
    )
    
    sample_geom = create_example_sample_geometry()
    
    # Generate with smaller voxels for higher resolution
    generator = SynchrotronDataGenerator(
        voxel_size=1.0,  # micrometers (larger for demo)
        image_dimensions=(64, 64, 32),  # Smaller for faster demo generation
        seed=42
    )
    
    dataset = generator.generate_complete_dataset(
        material_props, op_params, sample_geom, 
        output_dir="high_stress_creep_data"
    )
    
    return dataset

def generate_long_term_scenario():
    """Generate data for long-term service conditions"""
    
    print("\n=== LONG-TERM SERVICE CONDITIONS SCENARIO ===")
    
    material_props = create_example_sofc_interconnect()
    
    # Lower stress, longer time - more realistic service conditions
    op_params = OperationalParameters(
        temperature=650.0,         # Lower temperature
        mechanical_stress=25.0,    # Lower stress
        time_points=[0.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],  # hours
        atmosphere="Air",
        heating_rate=2.0,          # Slower heating
        cooling_rate=1.0           # Slower cooling
    )
    
    sample_geom = create_example_sample_geometry()
    
    generator = SynchrotronDataGenerator(
        voxel_size=1.0,  # micrometers (larger for demo)
        image_dimensions=(64, 64, 32),  # Smaller for demo
        seed=123
    )
    
    dataset = generator.generate_complete_dataset(
        material_props, op_params, sample_geom,
        output_dir="long_term_service_data"
    )
    
    return dataset

def generate_thermal_cycling_scenario():
    """Generate data with thermal cycling effects"""
    
    print("\n=== THERMAL CYCLING SCENARIO ===")
    
    material_props = create_example_sofc_interconnect()
    
    # Simulate thermal cycling by varying temperature
    op_params = OperationalParameters(
        temperature=700.0,         # Base temperature
        mechanical_stress=40.0,    
        time_points=[0.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0],  # hours
        atmosphere="Air + 10% H2O",  # Humid conditions
        heating_rate=10.0,         # Faster thermal cycling
        cooling_rate=8.0
    )
    
    sample_geom = create_example_sample_geometry()
    
    generator = SynchrotronDataGenerator(
        voxel_size=1.0,  # micrometers (larger for demo)
        image_dimensions=(64, 64, 32),  # Smaller for demo
        seed=456
    )
    
    dataset = generator.generate_complete_dataset(
        material_props, op_params, sample_geom,
        output_dir="thermal_cycling_data"
    )
    
    return dataset

def main():
    """Main function to demonstrate different data generation scenarios"""
    
    print("SYNTHETIC SYNCHROTRON X-RAY DATA GENERATOR")
    print("=" * 50)
    print("Generating realistic SOFC creep deformation datasets...")
    
    # Generate different scenarios
    scenarios = []
    
    try:
        # Scenario 1: High stress accelerated testing
        dataset1 = generate_high_stress_scenario()
        scenarios.append(("High Stress", dataset1))
        
        # Scenario 2: Long-term service conditions
        dataset2 = generate_long_term_scenario()
        scenarios.append(("Long Term", dataset2))
        
        # Scenario 3: Thermal cycling
        dataset3 = generate_thermal_cycling_scenario()
        scenarios.append(("Thermal Cycling", dataset3))
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # Print summary of all scenarios
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE - SUMMARY OF ALL SCENARIOS")
    print("=" * 60)
    
    for scenario_name, dataset in scenarios:
        print(f"\n{scenario_name} Scenario:")
        print(f"  Output Directory: {scenario_name.lower().replace(' ', '_')}_data")
        
        # Extract key metrics
        metadata = dataset['metadata']
        metrics = dataset['analysis_metrics']
        
        print(f"  Temperature: {metadata['operational_parameters']['temperature']}°C")
        print(f"  Stress: {metadata['operational_parameters']['mechanical_stress']} MPa")
        print(f"  Duration: {max(metadata['operational_parameters']['time_points'])} hours")
        print(f"  Time Points: {len(metadata['operational_parameters']['time_points'])}")
        
        # Damage progression
        initial_damage = list(metrics['damage_evolution'].values())[0]
        final_damage = list(metrics['damage_evolution'].values())[-1]
        print(f"  Damage Progression: {initial_damage:.4f} → {final_damage:.4f}")
        
        # Porosity evolution
        initial_porosity = list(metrics['porosity_evolution'].values())[0]
        final_porosity = list(metrics['porosity_evolution'].values())[-1]
        print(f"  Porosity Evolution: {initial_porosity:.4f} → {final_porosity:.4f}")
    
    print("\n" + "=" * 60)
    print("DATA USAGE NOTES:")
    print("=" * 60)
    print("1. Each dataset contains:")
    print("   - 4D tomography data (HDF5 format)")
    print("   - X-ray diffraction patterns and strain/stress maps")
    print("   - Complete metadata and analysis metrics")
    print("   - Summary reports")
    print("\n2. Use the visualization tools to explore the data:")
    print("   python visualize_data.py")
    print("\n3. Data can be used for:")
    print("   - Model validation and calibration")
    print("   - Algorithm development and testing")
    print("   - Creep mechanism analysis")
    print("   - Failure prediction studies")
    
if __name__ == "__main__":
    main()