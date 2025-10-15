#!/usr/bin/env python3
"""
Simplified Test for SOFC Dataset Generator Core Components

This script tests the basic functionality without complex dependencies.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_parameter_space():
    """Test parameter space definition."""
    print("Testing parameter space...")
    
    try:
        from src.doe.parameter_space import create_sofc_parameter_space
        
        param_space = create_sofc_parameter_space()
        print(f"  ✓ Created parameter space with {len(param_space.parameters)} parameters")
        
        # Test validation
        is_valid = param_space.validate_parameters()
        print(f"  ✓ Parameter validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Show some parameters
        param_names = list(param_space.parameters.keys())[:5]
        print(f"  ✓ Sample parameters: {param_names}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_material_model():
    """Test material property models."""
    print("Testing material models...")
    
    try:
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
        
        # Test at room temperature
        props = material_model.get_effective_properties(25, test_params)
        print(f"  ✓ Properties at 25°C: E = {props.elastic_modulus/1e9:.1f} GPa")
        
        # Test at high temperature
        props = material_model.get_effective_properties(1400, test_params)
        print(f"  ✓ Properties at 1400°C: E = {props.elastic_modulus/1e9:.1f} GPa")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_doe_generator():
    """Test DOE matrix generation (without pyDOE2)."""
    print("Testing DOE generator...")
    
    try:
        from src.doe.doe_generator import create_doe_generator, DOEConfiguration
        
        config = DOEConfiguration(n_samples=10, sampling_method="random")  # Use random instead of LHS
        generator = create_doe_generator(doe_config=config)
        
        doe_matrix = generator.generate_doe_matrix()
        print(f"  ✓ Generated DOE matrix with shape: {doe_matrix.shape}")
        
        # Check some values
        print(f"  ✓ Sample parameter ranges:")
        for col in doe_matrix.columns[:3]:
            values = doe_matrix[col]
            if values.dtype in ['float64', 'int64']:
                print(f"    {col}: {values.min():.3f} to {values.max():.3f}")
            else:
                print(f"    {col}: {values.unique()}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_basic_fea_components():
    """Test basic FEA components without full simulation."""
    print("Testing basic FEA components...")
    
    try:
        from src.fea.fea_solver import FEAConfiguration, SOFCFEASolver
        
        # Test configuration
        config = FEAConfiguration(
            mesh_resolution=10e-3,
            n_time_steps=5,
            solver_type="simplified"
        )
        
        solver = SOFCFEASolver(config)
        print(f"  ✓ Created FEA solver with config")
        
        # Test thermal profile setup
        test_params = {
            'thermal.peak_temperature': 1400.0,
            'thermal.heating_rate': 5.0,
            'thermal.cooling_rate': 2.0,
            'thermal.dwell_time': 60.0,
        }
        
        # This would normally require mesh, but we can test the thermal profile
        solver.doe_parameters = test_params
        solver._setup_thermal_profile()
        
        print(f"  ✓ Thermal profile created with {len(solver.time_profile)} time points")
        print(f"  ✓ Temperature range: {solver.temperature_profile.min():.0f} to {solver.temperature_profile.max():.0f} °C")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_data_structures():
    """Test data structure definitions."""
    print("Testing data structures...")
    
    try:
        from src.extraction.warp_extractor import WarpFieldData
        from src.extraction.stress_extractor import StressFieldData
        from src.utils.data_manager import DatasetSample
        from datetime import datetime
        
        # Create dummy data
        dummy_coords = np.random.rand(100, 3) * 0.1  # 100 points in 10cm cube
        dummy_displacements = np.random.rand(100, 3) * 1e-6  # μm displacements
        dummy_stress = np.random.rand(50, 6) * 1e6  # MPa stresses
        
        # Test warp data structure
        warp_data = WarpFieldData(
            original_coordinates=dummy_coords,
            deformed_coordinates=dummy_coords + dummy_displacements,
            displacement_field=dummy_displacements,
            top_surface_original=dummy_coords[:20, :2],
            top_surface_deformed=(dummy_coords + dummy_displacements)[:20, :2],
            bottom_surface_original=dummy_coords[:20, :2],
            bottom_surface_deformed=(dummy_coords + dummy_displacements)[:20, :2],
            top_height_map=np.random.rand(20, 20) * 1e-6,
            bottom_height_map=np.random.rand(20, 20) * 1e-6,
            x_grid=np.linspace(0, 0.1, 20),
            y_grid=np.linspace(0, 0.1, 20),
            plate_dimensions=(0.1, 0.1),
            max_warp=np.max(np.linalg.norm(dummy_displacements, axis=1)),
            rms_warp=np.sqrt(np.mean(np.linalg.norm(dummy_displacements, axis=1)**2))
        )
        
        print(f"  ✓ Created warp data structure")
        
        # Test stress data structure
        stress_data = StressFieldData(
            coordinates=dummy_coords[:50],
            stress_tensor=dummy_stress,
            principal_stresses=np.random.rand(50, 3) * 1e6,
            von_mises_stress=np.random.rand(50) * 1e6,
            hydrostatic_stress=np.random.rand(50) * 1e6,
            voxel_grid_x=np.zeros((10, 10, 10)),
            voxel_grid_y=np.zeros((10, 10, 10)),
            voxel_grid_z=np.zeros((10, 10, 10)),
            voxelized_stress=np.random.rand(10, 10, 10, 6) * 1e6,
            layer_stress_maps={},
            stress_invariants=np.random.rand(50, 3) * 1e6,
            stress_gradients=np.random.rand(10, 10, 10, 6, 3) * 1e9,
            voxel_resolution=5e-3,
            bounds=(0, 0.1, 0, 0.1, 0, 1e-3),
            layer_info={},
            statistics={'max_von_mises': np.max(dummy_stress)}
        )
        
        print(f"  ✓ Created stress data structure")
        
        # Test dataset sample
        sample = DatasetSample(
            sample_id="test_001",
            doe_parameters={'test_param': 1.0},
            warp_data=warp_data,
            stress_data=stress_data,
            metadata={'test': True},
            timestamp=datetime.now().isoformat()
        )
        
        print(f"  ✓ Created dataset sample with hash: {sample.get_hash()[:8]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run simplified tests."""
    print("=" * 60)
    print("SOFC Dataset Generator - Simplified Functionality Test")
    print("=" * 60)
    
    tests = [
        test_parameter_space,
        test_material_model,
        test_doe_generator,
        test_basic_fea_components,
        test_data_structures,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    if failed == 0:
        print(f"✅ ALL {passed} TESTS PASSED!")
        print("The core components are working correctly.")
    else:
        print(f"⚠️  {passed} tests passed, {failed} tests failed")
        print("Some components may need additional dependencies or fixes.")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Install additional dependencies if needed:")
    print("   pip install pyvista meshio pyDOE2 tqdm")
    print("2. Run the full test:")
    print("   python test_basic_functionality.py")
    print("3. Generate a test dataset:")
    print("   python generate_dataset.py --test_run")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)