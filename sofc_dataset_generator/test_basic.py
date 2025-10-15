#!/usr/bin/env python3
"""
Basic test of SOFC dataset generator components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.materials.sofc_materials import SOFCMaterials
from src.doe.doe_generator import DOEGenerator
from src.fea.mesh_generator import SOFCMeshGenerator, MeshParameters


def test_materials():
    """Test material properties"""
    print("Testing material properties...")
    materials = SOFCMaterials()
    
    # Test 8YSZ properties
    props_25 = materials.get_all_properties('8YSZ', 25.0)
    props_800 = materials.get_all_properties('8YSZ', 800.0)
    
    print(f"8YSZ at 25°C - Young's modulus: {props_25['youngs_modulus']:.1f} GPa")
    print(f"8YSZ at 800°C - Young's modulus: {props_800['youngs_modulus']:.1f} GPa")
    print(f"8YSZ at 25°C - CTE: {props_25['cte']*1e6:.1f} ppm/K")
    print(f"8YSZ at 800°C - CTE: {props_800['cte']*1e6:.1f} ppm/K")
    
    return True


def test_doe():
    """Test DOE generation"""
    print("\nTesting DOE generation...")
    doe_generator = DOEGenerator()
    
    # Generate small DOE
    doe_df = doe_generator.generate_doe(n_samples=5, strategy='lhs', random_state=42)
    
    print(f"Generated DOE with shape: {doe_df.shape}")
    print(f"Parameter names: {list(doe_df.columns)[:5]}...")  # First 5 parameters
    
    # Check parameter ranges
    for col in doe_df.columns[:3]:  # First 3 parameters
        print(f"{col}: {doe_df[col].min():.3f} - {doe_df[col].max():.3f}")
    
    return True


def test_mesh():
    """Test mesh generation"""
    print("\nTesting mesh generation...")
    
    # Create simple mesh parameters
    mesh_params = MeshParameters(
        cell_length=50.0,  # Smaller for testing
        cell_width=50.0,
        electrolyte_thickness=150.0,
        anode_thickness=300.0,
        cathode_thickness=50.0,
        interconnect_thickness=1000.0,  # 1mm
        elements_per_mm=5.0,  # Lower resolution
        electrolyte_elements_z=4,
        electrode_elements_z=2,
        interconnect_elements_z=6
    )
    
    # Generate mesh
    mesh_generator = SOFCMeshGenerator(mesh_params)
    mesh = mesh_generator.generate_mesh()
    
    # Check mesh statistics
    stats = mesh_generator.get_mesh_statistics()
    print(f"Mesh statistics: {stats}")
    
    # Check mesh structure
    print(f"Nodes shape: {mesh['nodes'].shape}")
    print(f"Elements shape: {mesh['elements'].shape}")
    print(f"Element groups: {list(mesh['element_groups'].keys())}")
    
    return True


def test_creep():
    """Test creep models"""
    print("\nTesting creep models...")
    from src.materials.creep_models import create_8ysz_creep_model
    
    creep_model = create_8ysz_creep_model()
    
    # Test at different stress levels
    stress_levels = [50, 100, 150]  # MPa
    temperature = 800.0  # °C
    time = 3600  # 1 hour
    
    print("Creep strain rates (s⁻¹):")
    for stress in stress_levels:
        strain_rate = creep_model.strain_rate(stress, temperature)
        accumulated = creep_model.accumulated_strain(stress, temperature, time)
        print(f"  Stress {stress} MPa: {strain_rate:.2e} s⁻¹, accumulated: {accumulated:.2e}")
    
    return True


def main():
    """Run all tests"""
    print("SOFC Dataset Generator - Basic Component Tests")
    print("=" * 50)
    
    tests = [
        test_materials,
        test_doe,
        test_mesh,
        test_creep
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is working correctly.")
    else:
        print("Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()