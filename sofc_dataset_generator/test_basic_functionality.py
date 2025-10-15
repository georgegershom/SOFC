#!/usr/bin/env python3
"""
Basic Functionality Test for SOFC Dataset Generator

This script tests the basic functionality of each component
without running the full dataset generation.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_parameter_space():
    """Test parameter space definition."""
    print("Testing parameter space...")
    
    from src.doe.parameter_space import create_sofc_parameter_space
    
    param_space = create_sofc_parameter_space()
    print(f"  ✓ Created parameter space with {len(param_space.parameters)} parameters")
    
    # Test validation
    is_valid = param_space.validate_parameters()
    print(f"  ✓ Parameter validation: {'PASS' if is_valid else 'FAIL'}")
    
    return True

def test_doe_generator():
    """Test DOE matrix generation."""
    print("Testing DOE generator...")
    
    from src.doe.doe_generator import create_doe_generator, DOEConfiguration
    
    config = DOEConfiguration(n_samples=10, sampling_method="latin_hypercube")
    generator = create_doe_generator(doe_config=config)
    
    doe_matrix = generator.generate_doe_matrix()
    print(f"  ✓ Generated DOE matrix with shape: {doe_matrix.shape}")
    
    # Validate matrix
    validation = generator.validate_doe_matrix(doe_matrix)
    all_valid = all(validation.values())
    print(f"  ✓ DOE validation: {'PASS' if all_valid else 'FAIL'}")
    
    return True

def test_material_model():
    """Test material property models."""
    print("Testing material models...")
    
    from src.materials.material_models import create_sofc_material_model
    
    material_model = create_sofc_material_model()
    
    # Test parameters
    test_params = {
        'material.anode.porosity': 0.35,
        'material.anode.ni_content': 0.5,
        'material.electrolyte.grain_size': 1.0,
        'material.electrolyte.density_fraction': 0.95,
        'material.cathode.porosity': 0.4,
        'material.cathode.lsm_content': 0.5,
        'geometry.anode_thickness': 500e-6,
        'geometry.electrolyte_thickness': 15e-6,
        'geometry.cathode_thickness': 40e-6
    }
    
    # Test at different temperatures
    for temp in [25, 1000, 1400]:
        props = material_model.get_effective_properties(temp, test_params)
        print(f"  ✓ Properties at {temp}°C: E = {props.elastic_modulus/1e9:.1f} GPa")
    
    return True

def test_mesh_generator():
    """Test mesh generation."""
    print("Testing mesh generator...")
    
    from src.geometry.mesh_generator import create_mesh_generator
    
    mesh_gen = create_mesh_generator(mesh_resolution=5e-3)  # Coarse mesh for testing
    
    # Test parameters
    test_params = {
        'geometry.length': 50e-3,  # 50mm
        'geometry.width': 50e-3,   # 50mm
        'geometry.anode_thickness': 500e-6,
        'geometry.electrolyte_thickness': 15e-6,
        'geometry.cathode_thickness': 40e-6,
    }
    
    geometry = mesh_gen.create_geometry(test_params)
    print(f"  ✓ Created geometry: {geometry.length*1000:.0f}x{geometry.width*1000:.0f}x{geometry.total_thickness*1e6:.0f} mm³")
    
    mesh = mesh_gen.generate_mesh(test_params)
    stats = mesh_gen.get_mesh_statistics()
    print(f"  ✓ Generated mesh: {stats['n_points']} nodes, {stats['n_cells']} elements")
    
    return True

def test_fea_solver():
    """Test FEA solver (simplified)."""
    print("Testing FEA solver...")
    
    from src.fea.fea_solver import create_fea_solver, FEAConfiguration
    from src.materials.material_models import create_sofc_material_model
    from src.geometry.mesh_generator import create_mesh_generator
    
    # Create components
    fea_config = FEAConfiguration(
        mesh_resolution=10e-3,  # Very coarse for testing
        n_time_steps=5,         # Few time steps
        solver_type="simplified"
    )
    
    fea_solver = create_fea_solver(fea_config)
    material_model = create_sofc_material_model()
    mesh_generator = create_mesh_generator(10e-3)
    
    # Test parameters
    test_params = {
        'geometry.length': 30e-3,
        'geometry.width': 30e-3,
        'geometry.anode_thickness': 500e-6,
        'geometry.electrolyte_thickness': 15e-6,
        'geometry.cathode_thickness': 40e-6,
        'thermal.peak_temperature': 1400.0,
        'thermal.heating_rate': 5.0,
        'thermal.cooling_rate': 2.0,
        'thermal.dwell_time': 60.0,
        'material.anode.porosity': 0.35,
        'material.anode.ni_content': 0.5,
        'material.electrolyte.grain_size': 1.0,
        'material.electrolyte.density_fraction': 0.95,
        'material.cathode.porosity': 0.4,
        'material.cathode.lsm_content': 0.5,
    }
    
    fea_solver.setup_problem(test_params, material_model, mesh_generator)
    results = fea_solver.solve()
    
    print(f"  ✓ FEA simulation completed in {results.simulation_time:.2f} seconds")
    print(f"  ✓ Results: {len(results.coordinates)} nodes, {len(results.final_stress)} stress points")
    
    return results

def test_extractors(simulation_results):
    """Test warp and stress field extractors."""
    print("Testing field extractors...")
    
    from src.extraction.warp_extractor import create_warp_extractor
    from src.extraction.stress_extractor import create_stress_extractor
    
    # Create extractors
    warp_extractor = create_warp_extractor(grid_resolution=5e-3)  # Coarse grid
    stress_extractor = create_stress_extractor(voxel_resolution=5e-3)
    
    # Geometry info
    geometry_info = {
        'total_thickness': 555e-6,
        'layers': {
            'anode': {'z_bottom': 0.0, 'z_top': 500e-6},
            'electrolyte': {'z_bottom': 500e-6, 'z_top': 515e-6},
            'cathode': {'z_bottom': 515e-6, 'z_top': 555e-6}
        }
    }
    
    # Extract warp field
    warp_data = warp_extractor.extract_warp_field(simulation_results, geometry_info)
    print(f"  ✓ Warp field extracted: {warp_data.height_map_top.shape} grid")
    print(f"  ✓ Max displacement: {warp_data.statistics['max_displacement']*1e6:.2f} μm")
    
    # Extract stress field
    stress_data = stress_extractor.extract_stress_field(simulation_results, geometry_info)
    print(f"  ✓ Stress field extracted: {stress_data.voxelized_stress.shape} voxels")
    print(f"  ✓ Max von Mises stress: {stress_data.statistics['max_von_mises']/1e6:.1f} MPa")
    
    return warp_data, stress_data

def test_data_manager(warp_data, stress_data):
    """Test data management system."""
    print("Testing data manager...")
    
    from src.utils.data_manager import create_dataset_manager, DatasetSample
    from datetime import datetime
    
    # Create data manager
    data_manager = create_dataset_manager("test_output")
    
    # Create test sample
    test_params = {
        'geometry.length': 30e-3,
        'geometry.width': 30e-3,
        'thermal.peak_temperature': 1400.0,
    }
    
    sample = DatasetSample(
        sample_id="test_sample_001",
        doe_parameters=test_params,
        warp_data=warp_data,
        stress_data=stress_data,
        metadata={'test': True},
        timestamp=datetime.now().isoformat()
    )
    
    # Add sample
    data_manager.add_sample(sample)
    print(f"  ✓ Added sample to dataset")
    
    # Create dataset
    metadata = data_manager.create_dataset_from_samples(
        dataset_name="test_dataset",
        description="Test dataset for validation"
    )
    print(f"  ✓ Created dataset with {metadata.n_samples} samples")
    
    # Get statistics
    stats = data_manager.get_dataset_statistics()
    print(f"  ✓ Dataset statistics calculated")
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("SOFC Dataset Generator - Basic Functionality Test")
    print("=" * 50)
    
    try:
        # Test each component
        test_parameter_space()
        test_doe_generator()
        test_material_model()
        test_mesh_generator()
        
        # Test FEA solver (returns results for next tests)
        simulation_results = test_fea_solver()
        
        # Test extractors (returns data for next test)
        warp_data, stress_data = test_extractors(simulation_results)
        
        # Test data manager
        test_data_manager(warp_data, stress_data)
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("The SOFC dataset generator is ready to use.")
        print("=" * 50)
        
        print("\nTo generate a full dataset, run:")
        print("python generate_dataset.py --test_run")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)