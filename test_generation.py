#!/usr/bin/env python3
"""
Test Script for SOFC Dataset Generation
=======================================

This script tests the dataset generation system with a small sample
to ensure everything is working correctly.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from sofc_dataset_generator import SOFCDatasetGenerator
from data_augmentation import SOFCDataAugmenter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_analytical_generation():
    """Test analytical dataset generation"""
    logger.info("Testing analytical dataset generation...")
    
    try:
        # Initialize generator
        generator = SOFCDatasetGenerator(output_dir="test_output")
        
        # Generate small dataset
        dataset = generator.generate_dataset(n_samples=100, method='lhs', save_results=True)
        
        # Check basic properties
        assert len(dataset) == 100, f"Expected 100 samples, got {len(dataset)}"
        assert 'max_principal_stress_elastic' in dataset.columns, "Missing stress column"
        assert 'safety_factor_elastic' in dataset.columns, "Missing safety factor column"
        
        # Check data ranges
        stress_values = dataset['max_principal_stress_elastic']
        assert stress_values.min() >= 0, "Negative stress values found"
        assert stress_values.max() < 1000, "Unrealistically high stress values found"
        
        logger.info("✅ Analytical generation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Analytical generation test failed: {e}")
        return False

def test_data_augmentation():
    """Test data augmentation"""
    logger.info("Testing data augmentation...")
    
    try:
        # Generate small analytical dataset first
        generator = SOFCDatasetGenerator(output_dir="test_output")
        dataset = generator.generate_dataset(n_samples=50, method='lhs', save_results=False)
        
        # Test augmentation
        augmenter = SOFCDataAugmenter(dataset)
        
        # Test noise augmentation
        noisy_data = augmenter.augment_with_noise(noise_level=0.05, n_samples=20)
        assert len(noisy_data) == 20, f"Expected 20 samples, got {len(noisy_data)}"
        
        # Test interpolation augmentation
        interpolated_data = augmenter.augment_with_interpolation(n_samples=20)
        assert len(interpolated_data) == 20, f"Expected 20 samples, got {len(interpolated_data)}"
        
        # Test physics-informed augmentation
        physics_data = augmenter.augment_with_physics_informed_synthesis(n_samples=20)
        assert len(physics_data) == 20, f"Expected 20 samples, got {len(physics_data)}"
        
        logger.info("✅ Data augmentation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data augmentation test failed: {e}")
        return False

def test_parameter_constraints():
    """Test parameter constraint validation"""
    logger.info("Testing parameter constraints...")
    
    try:
        generator = SOFCDatasetGenerator()
        
        # Check parameter ranges are reasonable
        param_ranges = generator.param_ranges
        
        # Check thickness ranges
        assert param_ranges['electrolyte_thickness'][0] > 0, "Negative thickness range"
        assert param_ranges['electrolyte_thickness'][1] < 1.0, "Unrealistically thick electrolyte"
        
        # Check temperature ranges
        assert param_ranges['max_temperature'][0] > 1000, "Temperature too low"
        assert param_ranges['max_temperature'][1] < 1500, "Temperature too high"
        
        # Check material property ranges
        assert param_ranges['electrolyte_E_25C'][0] > 100, "Young's modulus too low"
        assert param_ranges['electrolyte_E_25C'][1] < 300, "Young's modulus too high"
        
        logger.info("✅ Parameter constraints test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter constraints test failed: {e}")
        return False

def test_stress_calculations():
    """Test stress calculation accuracy"""
    logger.info("Testing stress calculations...")
    
    try:
        generator = SOFCDatasetGenerator()
        
        # Generate single sample
        dataset = generator.generate_dataset(n_samples=1, method='lhs', save_results=False)
        
        # Check stress calculations are reasonable
        stress = dataset['max_principal_stress_elastic'].iloc[0]
        safety_factor = dataset['safety_factor_elastic'].iloc[0]
        fracture_risk = dataset['fracture_risk_elastic'].iloc[0]
        
        # Basic sanity checks
        assert 0 < stress < 500, f"Unrealistic stress value: {stress}"
        assert 0 < safety_factor < 10, f"Unrealistic safety factor: {safety_factor}"
        assert 0 < fracture_risk < 2, f"Unrealistic fracture risk: {fracture_risk}"
        
        # Check relationships
        assert abs(safety_factor - 165.0 / stress) < 1e-6, "Safety factor calculation error"
        assert abs(fracture_risk - 1.0 / safety_factor) < 1e-6, "Fracture risk calculation error"
        
        logger.info("✅ Stress calculations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Stress calculations test failed: {e}")
        return False

def test_file_operations():
    """Test file saving and loading"""
    logger.info("Testing file operations...")
    
    try:
        import pandas as pd
        import json
        
        # Generate small dataset
        generator = SOFCDatasetGenerator(output_dir="test_output")
        dataset = generator.generate_dataset(n_samples=10, method='lhs', save_results=True)
        
        # Check files were created
        csv_file = Path("test_output/sofc_residual_stress_dataset.csv")
        json_file = Path("test_output/dataset_metadata.json")
        
        assert csv_file.exists(), "CSV file not created"
        assert json_file.exists(), "JSON metadata file not created"
        
        # Test loading
        loaded_dataset = pd.read_csv(csv_file)
        assert len(loaded_dataset) == 10, "Loaded dataset has wrong number of samples"
        
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        assert metadata['n_samples'] == 10, "Metadata has wrong sample count"
        
        logger.info("✅ File operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ File operations test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    
    test_dirs = ["test_output", "sofc_dataset", "analytical", "augmentation_results"]
    for dir_name in test_dirs:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            logger.info(f"Cleaned up {dir_name}")

def main():
    """Run all tests"""
    logger.info("Starting SOFC Dataset Generation Tests")
    logger.info("="*50)
    
    tests = [
        ("Parameter Constraints", test_parameter_constraints),
        ("Analytical Generation", test_analytical_generation),
        ("Stress Calculations", test_stress_calculations),
        ("Data Augmentation", test_data_augmentation),
        ("File Operations", test_file_operations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            failed += 1
    
    # Cleanup
    cleanup_test_files()
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The dataset generation system is working correctly.")
        return True
    else:
        logger.error(f"❌ {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)