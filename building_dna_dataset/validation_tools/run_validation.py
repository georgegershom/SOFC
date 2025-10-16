#!/usr/bin/env python3
"""
Building DNA Dataset Validation Runner
=====================================

Simple script to run validation on the building DNA dataset with 
pre-configured settings for the Dynamic Digital Twin Framework.

Usage:
    python run_validation.py

Author: AI Assistant
Date: 2025-10-16
Version: 1.0.0
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from data_validator import BuildingDNAValidator
except ImportError:
    print("Error: Could not import data_validator module.")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Run validation with default settings"""
    
    # Determine dataset path (parent directory of validation_tools)
    dataset_path = current_dir.parent
    
    print("Building DNA Dataset Validation")
    print("=" * 50)
    print(f"Dataset Path: {dataset_path}")
    print(f"Validation Tool: {current_dir}")
    print()
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Initialize validator
    print("Initializing validator...")
    validator = BuildingDNAValidator(str(dataset_path))
    
    # Run validation
    print("Running comprehensive validation...")
    results = validator.validate_dataset()
    
    # Save detailed report
    report_path = validator.save_validation_report()
    print(f"\nDetailed report saved to: {report_path}")
    
    # Print summary
    validator.print_summary()
    
    # Additional output for CI/CD integration
    print(f"\nValidation Status: {results['overall_status']}")
    print(f"Quality Score: {results['validation_summary'].get('overall_quality_score', 0):.3f}")
    
    # Return appropriate exit code
    if results['overall_status'] in ['PASSED', 'PASSED_WITH_WARNINGS']:
        print("\n✅ Validation completed successfully!")
        return 0
    else:
        print("\n❌ Validation failed!")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)