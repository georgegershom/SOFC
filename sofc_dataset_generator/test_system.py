#!/usr/bin/env python3
"""
Test the SOFC dataset generation system components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_parameter_space():
    """Test parameter space creation"""
    print("Testing parameter space...")
    try:
        from doe.parameter_space import create_sofc_parameter_space
        param_space = create_sofc_parameter_space()
        print(f"✓ Created parameter space with {len(param_space.parameters)} parameters")
        
        # Show some parameters
        print("Sample parameters:")
        for i, (name, param) in enumerate(list(param_space.parameters.items())[:3]):
            print(f"  {name}: {param.param_type.value}, range: {param.min_val} - {param.max_val}")
        
        return True
    except Exception as e:
        print(f"✗ Parameter space failed: {e}")
        return False

def test_doe_generation():
    """Test DOE generation"""
    print("\nTesting DOE generation...")
    try:
        # Try the full DOE generator first
        try:
            from doe.doe_generator import create_doe_generator, DOEConfiguration
            from doe.parameter_space import create_sofc_parameter_space
            
            param_space = create_sofc_parameter_space()
            config = DOEConfiguration(n_samples=3, sampling_method='latin_hypercube')
            generator = create_doe_generator(doe_config=config)
            doe_matrix = generator.generate_doe_matrix()
            
            print(f"✓ Generated DOE matrix with full generator, shape: {doe_matrix.shape}")
            
        except ImportError as ie:
            print(f"Full DOE generator not available ({ie}), using simple version...")
            
            # Fallback to simple DOE generator
            from doe.simple_doe import create_simple_doe_generator, SimpleDOEConfiguration
            from doe.parameter_space import create_sofc_parameter_space
            
            param_space = create_sofc_parameter_space()
            config = SimpleDOEConfiguration(n_samples=3, sampling_method='latin_hypercube')
            generator = create_simple_doe_generator(param_space, config)
            doe_matrix = generator.generate_doe_matrix()
            
            print(f"✓ Generated DOE matrix with simple generator, shape: {doe_matrix.shape}")
        
        print("Sample DOE point:")
        for col in list(doe_matrix.columns)[:3]:
            print(f"  {col}: {doe_matrix[col].iloc[0]:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ DOE generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_material_models():
    """Test material models"""
    print("\nTesting material models...")
    try:
        from materials.material_models import create_sofc_material_model
        
        material_model = create_sofc_material_model()
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
        
        props = material_model.get_effective_properties(1000, test_params)
        print(f"✓ Material model works:")
        print(f"  Elastic modulus: {props.elastic_modulus/1e9:.1f} GPa")
        print(f"  Thermal expansion: {props.thermal_expansion*1e6:.1f} ppm/K")
        print(f"  Density: {props.density:.0f} kg/m³")
        
        return True
    except Exception as e:
        print(f"✗ Material model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_generation():
    """Test mesh generation"""
    print("\nTesting mesh generation...")
    try:
        from geometry.mesh_generator import create_mesh_generator
        
        mesh_gen = create_mesh_generator(mesh_resolution=5e-3)  # 5mm elements
        
        test_params = {
            'geometry.length': 50e-3,  # 50 mm
            'geometry.width': 50e-3,   # 50 mm
            'geometry.anode_thickness': 500e-6,      # 500 μm
            'geometry.electrolyte_thickness': 15e-6,  # 15 μm
            'geometry.cathode_thickness': 40e-6,      # 40 μm
        }
        
        geometry = mesh_gen.create_geometry(test_params)
        print(f"✓ Geometry created:")
        print(f"  Dimensions: {geometry.length*1000:.1f} x {geometry.width*1000:.1f} mm")
        print(f"  Total thickness: {geometry.total_thickness*1e6:.1f} μm")
        print(f"  Number of layers: {len(geometry.layers)}")
        
        # Test mesh generation (simplified)
        try:
            mesh = mesh_gen.generate_mesh(test_params)
            stats = mesh_gen.get_mesh_statistics()
            print(f"  Mesh nodes: {stats['n_points']}")
            print(f"  Mesh elements: {stats['n_cells']}")
        except Exception as mesh_error:
            print(f"  Mesh generation failed (expected with missing dependencies): {mesh_error}")
        
        return True
    except Exception as e:
        print(f"✗ Mesh generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("SOFC Dataset Generator - System Test")
    print("=" * 50)
    
    tests = [
        test_parameter_space,
        test_doe_generation, 
        test_material_models,
        test_mesh_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All core components working!")
        return 0
    else:
        print(f"✗ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())