#!/usr/bin/env python3
"""
Simple script to run the SOFC Digital Twin Dataset generation.

This script will generate the complete dataset and create all necessary files.
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def generate_dataset():
    """Generate the complete dataset."""
    print("Generating SOFC Digital Twin Dataset...")
    try:
        from sofc_dataset_generator import main
        dataset = main()
        print("Dataset generation completed successfully!")
        return True
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return False

def main():
    """Main function."""
    print("SOFC Digital Twin Dataset Generation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("sofc_dataset_generator.py"):
        print("Error: Please run this script from the sofc_digital_twin_dataset directory")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Generate dataset
    if not generate_dataset():
        return False
    
    print("\n" + "=" * 50)
    print("Dataset generation completed successfully!")
    print("Check the 'data' directory for generated files.")
    print("Check the 'examples' directory for usage examples.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)