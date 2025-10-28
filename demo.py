#!/usr/bin/env python3
"""
Quick demonstration of the Synthetic Synchrotron X-ray Data Generator
"""

from synchrotron_data_generator import (
    SynchrotronDataGenerator, 
    MaterialProperties, 
    OperationalParameters, 
    SampleGeometry
)

def main():
    """Generate a small demonstration dataset"""
    
    print("SYNTHETIC SYNCHROTRON X-RAY DATA GENERATOR - DEMO")
    print("=" * 50)
    
    # Define SOFC interconnect material (Crofer 22 APU-like)
    material_props = MaterialProperties(
        alloy_composition={'Fe': 76.8, 'Cr': 22.0, 'Mn': 0.5, 'Ti': 0.08, 'C': 0.03},
        grain_size_mean=15.0,      # micrometers
        grain_size_std=5.0,        # micrometers
        initial_porosity=0.002,    # 0.2% initial porosity
        elastic_modulus=200.0,     # GPa
        poisson_ratio=0.3,
        thermal_expansion_coeff=12.0e-6,  # 1/K
        creep_exponent=5.0,        # Norton creep law exponent
        activation_energy=300.0    # kJ/mol
    )
    
    # Define operating conditions
    op_params = OperationalParameters(
        temperature=700.0,         # Celsius
        mechanical_stress=50.0,    # MPa
        time_points=[0.0, 25.0, 100.0, 250.0],  # hours (shorter for demo)
        atmosphere="Air",
        heating_rate=5.0,          # K/min
        cooling_rate=2.0           # K/min
    )
    
    # Define sample geometry
    sample_geom = SampleGeometry(
        length=3.0,      # mm
        width=2.0,       # mm  
        thickness=0.5,   # mm
        shape="rectangular",
        volume=3.0       # mm³
    )
    
    # Create generator with small dimensions for demo
    generator = SynchrotronDataGenerator(
        voxel_size=1.0,  # micrometers
        image_dimensions=(48, 48, 24),  # Small for demo
        seed=42
    )
    
    # Generate dataset
    print("Generating synthetic SOFC creep dataset...")
    dataset = generator.generate_complete_dataset(
        material_props, op_params, sample_geom,
        output_dir="demo_sofc_data"
    )
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETE!")
    print("=" * 50)
    
    # Print summary
    metadata = dataset['metadata']
    metrics = dataset['analysis_metrics']
    
    print(f"\nGenerated Dataset Summary:")
    print(f"- Temperature: {metadata['operational_parameters']['temperature']}°C")
    print(f"- Stress: {metadata['operational_parameters']['mechanical_stress']} MPa")
    print(f"- Duration: {max(metadata['operational_parameters']['time_points'])} hours")
    print(f"- Time Points: {len(metadata['operational_parameters']['time_points'])}")
    print(f"- Voxel Size: {metadata['voxel_size_um']} μm")
    print(f"- Image Dimensions: {metadata['image_dimensions']}")
    
    # Damage progression
    initial_damage = list(metrics['damage_evolution'].values())[0]
    final_damage = list(metrics['damage_evolution'].values())[-1]
    print(f"- Damage Progression: {initial_damage:.4f} → {final_damage:.4f}")
    
    # Porosity evolution
    initial_porosity = list(metrics['porosity_evolution'].values())[0]
    final_porosity = list(metrics['porosity_evolution'].values())[-1]
    print(f"- Porosity Evolution: {initial_porosity:.4f} → {final_porosity:.4f}")
    
    print(f"\nFiles created in 'demo_sofc_data/' directory:")
    print("- tomography_4d.h5: 4D microstructure evolution")
    print("- xrd_data.h5: X-ray diffraction patterns and strain/stress maps")
    print("- metadata.json: Complete experimental parameters")
    print("- analysis_metrics.json: Quantitative damage metrics")
    print("- dataset_summary.txt: Human-readable summary")
    
    print(f"\nTo visualize the data, run:")
    print(f"python3 visualize_data.py demo_sofc_data --report")

if __name__ == "__main__":
    main()