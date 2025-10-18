#!/usr/bin/env python3
"""
Main script to run the complete dataset generation for Fire-Resistant Rubberized Concrete
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'scipy', 'scikit-learn', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'data_generation_scripts/requirements.txt'])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_data_generation():
    """Run the complete data generation process"""
    print("=" * 80)
    print("FIRE-RESISTANT RUBBERIZED CONCRETE DATASET GENERATION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not os.path.exists('data_generation_scripts'):
        print("Error: data_generation_scripts directory not found!")
        print("Please run this script from the project root directory.")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_dependencies():
            print("Failed to install dependencies. Please install manually.")
            return False
    
    # Change to data_generation_scripts directory
    os.chdir('data_generation_scripts')
    
    try:
        # Run the master data generator
        from master_data_generator import main as run_master_generator
        run_master_generator()
        
        print("\n" + "=" * 80)
        print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Generated files:")
        print("  - generated_datasets/material_properties_data.json")
        print("  - generated_datasets/mix_design_data.json")
        print("  - generated_datasets/thermo_mechanical_testing_data.json")
        print("  - generated_datasets/analysis/ (comprehensive analysis)")
        print("  - generated_datasets/README.md (usage instructions)")
        print("  - generated_datasets/summary_report.json (detailed summary)")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        return False
    finally:
        # Change back to original directory
        os.chdir('..')

def main():
    """Main function"""
    print("Fire-Resistant Rubberized Concrete Dataset Generator")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run data generation
    success = run_data_generation()
    
    if success:
        print("\n✅ Dataset generation completed successfully!")
        print("You can now use the generated datasets for your research.")
    else:
        print("\n❌ Dataset generation failed!")
        print("Please check the error messages above and try again.")
    
    return success

if __name__ == "__main__":
    main()