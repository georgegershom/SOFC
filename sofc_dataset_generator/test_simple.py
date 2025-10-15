#!/usr/bin/env python3
"""
Simple test of SOFC dataset generation components
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.materials.sofc_materials import SOFCMaterials
from src.doe.doe_generator import DOEGenerator
from src.doe.sofc_parameters import SOFCParameters

def test_materials():
    """Test material properties"""
    print("Testing material properties...")
    materials = SOFCMaterials()
    
    # Test at room temperature
    props_25 = materials.get_all_properties('8YSZ', 25.0)
    print(f"8YSZ at 25°C - Young's Modulus: {props_25['youngs_modulus']:.1f} GPa")
    
    # Test at operating temperature
    props_800 = materials.get_all_properties('8YSZ', 800.0)
    print(f"8YSZ at 800°C - Young's Modulus: {props_800['youngs_modulus']:.1f} GPa")
    
    print("✓ Material properties working")

def test_doe():
    """Test Design of Experiments"""
    print("\nTesting Design of Experiments...")
    
    # Create DOE generator
    doe_generator = DOEGenerator()
    
    # Generate small DOE matrix
    doe_df = doe_generator.generate_doe(
        n_samples=5,
        strategy='lhs',
        random_state=42
    )
    
    print(f"Generated DOE matrix shape: {doe_df.shape}")
    print(f"Parameter names: {list(doe_df.columns)[:5]}...")  # First 5 parameters
    
    # Generate manufacturing parameters
    mfg_params = doe_generator.generate_parameter_combinations(3, strategy='lhs', random_state=42)
    print(f"Generated {len(mfg_params)} manufacturing parameter sets")
    
    # Show first parameter set
    first_params = mfg_params[0]
    geometric_params = first_params.get_geometric_values()
    print(f"First sample geometric parameters:")
    for name, value in list(geometric_params.items())[:3]:  # First 3
        print(f"  {name}: {value:.2f}")
    
    print("✓ Design of Experiments working")

def test_parameter_ranges():
    """Test parameter ranges"""
    print("\nTesting parameter ranges...")
    
    sofc_params = SOFCParameters()
    param_ranges = sofc_params.get_all_parameters()
    
    print(f"Total parameters: {len(param_ranges)}")
    print(f"Geometric parameters: {len(sofc_params.get_geometric_parameters())}")
    print(f"Material parameters: {len(sofc_params.get_material_parameters())}")
    print(f"Process parameters: {len(sofc_params.get_process_parameters())}")
    print(f"Environmental parameters: {len(sofc_params.get_environmental_parameters())}")
    
    # Test parameter sampling
    cell_length_range = sofc_params.cell_length
    samples = cell_length_range.sample(5, random_state=42)
    print(f"Cell length samples: {samples}")
    
    print("✓ Parameter ranges working")

def main():
    """Run all tests"""
    print("SOFC Dataset Generator - Component Tests")
    print("=" * 50)
    
    try:
        test_materials()
        test_parameter_ranges()
        test_doe()
        
        print("\n" + "=" * 50)
        print("✓ All components working correctly!")
        print("\nThe SOFC dataset generation system is ready to use.")
        print("To generate a full dataset, run:")
        print("  python3 run_full_dataset.py --samples 100")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()