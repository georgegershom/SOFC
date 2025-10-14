#!/usr/bin/env python3
"""
Master script to generate the complete building retrofit dataset.
Runs all data generation scripts and integrates the data.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status."""
    
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.path.dirname(script_path),
                              capture_output=True, text=True, check=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Return code: {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    """Generate the complete building retrofit dataset."""
    
    print("🏗️  Building Retrofit Dataset Generator")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Define scripts to run
    scripts = [
        {
            'path': 'generate_iot_data.py',
            'description': 'IoT Sensor Data Generation'
        },
        {
            'path': 'generate_building_attributes.py', 
            'description': 'Building Attributes Generation'
        },
        {
            'path': 'generate_energy_performance.py',
            'description': 'Energy Performance Data Generation'
        },
        {
            'path': 'generate_lca_data.py',
            'description': 'LCA Data Generation'
        },
        {
            'path': 'integrate_data.py',
            'description': 'Data Integration Pipeline'
        }
    ]
    
    # Track success/failure
    results = {}
    
    # Run each script
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script['path'])
        success = run_script(script_path, script['description'])
        results[script['description']] = success
        
        if not success:
            print(f"\n⚠️  Warning: {script['description']} failed!")
            print("Continuing with remaining scripts...")
    
    # Summary
    print(f"\n{'='*60}")
    print("DATASET GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Scripts completed successfully: {successful}/{total}")
    print()
    
    for description, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {description}")
    
    print()
    
    if successful == total:
        print("🎉 All scripts completed successfully!")
        print("Your building retrofit dataset is ready for analysis.")
        print()
        print("Dataset structure:")
        print("  📁 raw_data/          - Raw generated data")
        print("  📁 processed_data/     - Integrated and processed data")
        print("  📁 scripts/           - Data generation scripts")
        print("  📁 documentation/     - Dataset documentation")
        print()
        print("Next steps:")
        print("  1. Explore the data in processed_data/analysis_ready/")
        print("  2. Use the integrated datasets for your PhD research")
        print("  3. Refer to README.md for detailed documentation")
    else:
        print("⚠️  Some scripts failed. Please check the error messages above.")
        print("You may need to run individual scripts manually to fix issues.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()