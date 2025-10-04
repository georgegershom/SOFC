#!/usr/bin/env python3
"""
Complete Atomic-Scale Simulation Dataset Generator
Master script to generate the entire dataset with all components
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        end_time = time.time()
        print(f"✅ Completed successfully in {end_time - start_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Script {script_name} not found!")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def create_directory_structure():
    """Create the necessary directory structure"""
    directories = [
        'atomic_simulation_dataset',
        'atomic_simulation_dataset/md_data',
        'atomic_simulation_dataset/figures'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def generate_requirements_file():
    """Generate requirements.txt file"""
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("📄 Generated requirements.txt")

def main():
    """Main function to generate the complete dataset"""
    print("🚀 Atomic-Scale Simulation Dataset Generator")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return False
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Generate requirements file
    generate_requirements_file()
    
    # List of scripts to run in order
    scripts = [
        ("generate_atomic_simulation_dataset.py", "DFT Calculation Data Generation"),
        ("generate_md_simulation_data.py", "MD Simulation Data Generation"), 
        ("analyze_dataset.py", "Dataset Analysis and Visualization")
    ]
    
    # Track success/failure
    results = []
    total_start_time = time.time()
    
    # Run each script
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results.append((script_name, description, success))
        
        if not success:
            print(f"\n⚠️  Warning: {script_name} failed, but continuing with remaining scripts...")
    
    # Summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("📊 GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"\nScript Results ({successful}/{total} successful):")
    for script_name, description, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {description}")
    
    # Check if dataset was created
    dataset_files = [
        'atomic_simulation_dataset/dft_formation_energies.csv',
        'atomic_simulation_dataset/activation_barriers.csv',
        'atomic_simulation_dataset/surface_energies.csv',
        'atomic_simulation_dataset/md_data/grain_boundary_sliding.csv',
        'atomic_simulation_dataset/md_data/dislocation_mobility.csv'
    ]
    
    created_files = []
    for file_path in dataset_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            created_files.append((file_path, size))
    
    if created_files:
        print(f"\n📄 Generated Dataset Files:")
        for file_path, size in created_files:
            print(f"  📄 {file_path} ({size:,} bytes)")
    
    # Final status
    if successful == total:
        print(f"\n🎉 Dataset generation completed successfully!")
        print("📖 See README_atomic_simulation_dataset.md for usage instructions")
        return True
    else:
        print(f"\n⚠️  Dataset generation completed with {total - successful} errors")
        print("Some components may be missing. Check error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)