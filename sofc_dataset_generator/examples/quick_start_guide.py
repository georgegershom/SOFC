#!/usr/bin/env python3
"""
Quick Start Guide for SOFC Dataset Generator

This example demonstrates the basic workflow for generating a synthetic
SOFC warp-stress dataset for machine learning applications.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    """Quick start example for SOFC dataset generation."""
    
    print("SOFC Dataset Generator - Quick Start Guide")
    print("=" * 50)
    
    # Step 1: Create parameter space and DOE matrix
    print("\n1. Creating parameter space and DOE matrix...")
    
    from doe.parameter_space import create_sofc_parameter_space
    from doe.doe_generator import create_doe_generator, DOEConfiguration
    
    # Create parameter space
    param_space = create_sofc_parameter_space()
    print(f"   Parameter space created with {len(param_space.parameters)} parameters")
    
    # Generate small DOE matrix for demonstration
    doe_config = DOEConfiguration(n_samples=5, sampling_method='latin_hypercube', seed=42)
    
    try:
        # Try full DOE generator
        doe_generator = create_doe_generator(doe_config=doe_config)
    except ImportError:
        # Fallback to simple DOE generator
        from doe.simple_doe import create_simple_doe_generator, SimpleDOEConfiguration
        simple_config = SimpleDOEConfiguration(n_samples=5, sampling_method='latin_hypercube', seed=42)
        doe_generator = create_simple_doe_generator(param_space, simple_config)
    
    doe_matrix = doe_generator.generate_doe_matrix()
    print(f"   DOE matrix generated: {doe_matrix.shape}")
    
    # Step 2: Create material models
    print("\n2. Setting up material models...")
    
    from materials.material_models import create_sofc_material_model
    
    material_model = create_sofc_material_model()
    
    # Test material properties at different temperatures
    test_params = doe_matrix.iloc[0].to_dict()
    temperatures = [25, 1000, 1400]  # Room temp, operating temp, sintering temp
    
    print("   Material properties at different temperatures:")
    for temp in temperatures:
        props = material_model.get_effective_properties(temp, test_params)
        print(f"     {temp}°C: E = {props.elastic_modulus/1e9:.1f} GPa, "
              f"α = {props.thermal_expansion*1e6:.1f} ppm/K")
    
    # Step 3: Generate mesh
    print("\n3. Generating mesh...")
    
    from geometry.mesh_generator import create_mesh_generator
    
    mesh_generator = create_mesh_generator(mesh_resolution=5e-3)  # 5mm elements for speed
    
    # Use first DOE point for mesh generation
    sample_params = doe_matrix.iloc[0].to_dict()
    
    geometry = mesh_generator.create_geometry(sample_params)
    print(f"   Geometry: {geometry.length*1000:.1f} x {geometry.width*1000:.1f} mm")
    print(f"   Thickness: {geometry.total_thickness*1e6:.1f} μm")
    
    mesh = mesh_generator.generate_mesh(sample_params)
    stats = mesh_generator.get_mesh_statistics()
    print(f"   Mesh: {stats['n_points']} nodes, {stats['n_cells']} elements")
    
    # Step 4: Run simplified FEA simulation
    print("\n4. Running FEA simulation...")
    
    from fea.fea_solver import create_fea_solver, FEAConfiguration
    
    fea_config = FEAConfiguration(
        mesh_resolution=5e-3,
        n_time_steps=10,  # Reduced for speed
        solver_type='simplified'
    )
    
    fea_solver = create_fea_solver(fea_config)
    fea_solver.setup_problem(sample_params, material_model, mesh_generator)
    
    simulation_results = fea_solver.solve()
    print(f"   Simulation completed in {simulation_results.simulation_time:.2f} seconds")
    print(f"   Final displacement range: {np.ptp(simulation_results.final_displacement):.2e} m")
    print(f"   Final stress range: {np.ptp(simulation_results.final_stress):.2e} Pa")
    
    # Step 5: Extract warp field
    print("\n5. Extracting warp field...")
    
    from extraction.warp_extractor import create_warp_extractor
    
    warp_extractor = create_warp_extractor(grid_resolution=(32, 32))
    warp_data = warp_extractor.extract_warp_field(
        mesh, simulation_results.final_displacement, sample_params
    )
    
    print(f"   Warp field extracted:")
    print(f"     Max warp: {warp_data.max_warp*1e6:.2f} μm")
    print(f"     RMS warp: {warp_data.rms_warp*1e6:.2f} μm")
    print(f"     Grid resolution: {warp_data.grid_resolution}")
    
    # Step 6: Extract stress field
    print("\n6. Extracting stress field...")
    
    from extraction.stress_extractor import create_stress_extractor
    
    stress_extractor = create_stress_extractor(voxel_resolution=(16, 16, 8))
    stress_data = stress_extractor.extract_stress_field(
        mesh, simulation_results.final_stress, sample_params
    )
    
    print(f"   Stress field extracted:")
    print(f"     Max von Mises: {stress_data.max_von_mises/1e6:.2f} MPa")
    print(f"     Max principal: {stress_data.max_principal_stress/1e6:.2f} MPa")
    print(f"     Voxel resolution: {stress_data.voxel_resolution}")
    
    # Step 7: Visualize results
    print("\n7. Creating visualizations...")
    
    # Create simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Warp field
    if warp_data.warp_height_map is not None:
        im1 = axes[0, 0].imshow(warp_data.warp_height_map * 1e6, cmap='RdBu_r')
        axes[0, 0].set_title('Warp Field (μm)')
        plt.colorbar(im1, ax=axes[0, 0])
    
    # Stress distribution (von Mises)
    coords = stress_data.element_coordinates
    von_mises = stress_data.von_mises_stress / 1e6  # Convert to MPa
    
    scatter = axes[0, 1].scatter(coords[:, 0]*1000, coords[:, 1]*1000, 
                                c=von_mises, cmap='viridis', s=1)
    axes[0, 1].set_title('von Mises Stress (MPa)')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # DOE parameter distribution (first few parameters)
    param_names = list(doe_matrix.columns)[:4]
    for i, param_name in enumerate(param_names):
        if i < 2:
            row, col = 1, i
            axes[row, col].hist(doe_matrix[param_name], bins=10, alpha=0.7)
            axes[row, col].set_title(f'{param_name.split(".")[-1]}')
            axes[row, col].axvline(sample_params[param_name], color='red', 
                                  linestyle='--', label='Current sample')
            axes[row, col].legend()
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("./quick_start_output")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "quick_start_results.png", dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to: {output_dir / 'quick_start_results.png'}")
    
    # Step 8: Save data
    print("\n8. Saving data...")
    
    # Save warp data
    warp_extractor.export_warp_data(warp_data, output_dir / "warp_data", format='hdf5')
    print(f"   Warp data saved to: {output_dir / 'warp_data.h5'}")
    
    # Save stress data  
    stress_extractor.export_stress_data(stress_data, output_dir / "stress_data", format='hdf5')
    print(f"   Stress data saved to: {output_dir / 'stress_data.h5'}")
    
    # Save DOE matrix
    doe_matrix.to_csv(output_dir / "doe_matrix.csv", index=False)
    print(f"   DOE matrix saved to: {output_dir / 'doe_matrix.csv'}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Quick Start Guide Completed Successfully!")
    print("\nWhat you've accomplished:")
    print("✓ Generated a Design of Experiments matrix")
    print("✓ Created realistic SOFC material models")
    print("✓ Generated a 3D finite element mesh")
    print("✓ Ran a thermo-mechanical FEA simulation")
    print("✓ Extracted warp field as 2.5D height maps")
    print("✓ Extracted stress field as 3D voxelized data")
    print("✓ Created visualizations of the results")
    print("✓ Saved data in ML-ready formats")
    
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("- Scale up to generate full dataset (100-1000+ samples)")
    print("- Use the dataset to train ML models for inverse stress prediction")
    print("- Experiment with different DOE strategies and mesh resolutions")
    print("- Validate ML models against experimental data")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\nQuick start guide interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\nQuick start guide failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)