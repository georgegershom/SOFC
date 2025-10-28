#!/usr/bin/env python
"""
Main script to generate complete synthetic synchrotron X-ray dataset for SOFC creep analysis.
This generates tomography, diffraction, and metadata for validation of creep models.
"""

import sys
import os
import time
import subprocess

def run_script(script_path, description):
    """Run a Python script and track execution time."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_path}:")
        print(e.stderr)
        return False

def main():
    """Generate all synthetic synchrotron data."""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  SYNCHROTRON X-RAY DATA GENERATION FOR SOFC CREEP ANALYSIS    ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  This script generates synthetic but realistic synchrotron     ║
    ║  X-ray tomography and diffraction data for validating creep    ║
    ║  deformation models in SOFC metallic interconnects.           ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("This will generate:")
    print("  • 3D/4D Tomography data (~500 MB)")
    print("  • X-ray Diffraction patterns")
    print("  • Residual stress/strain maps")
    print("  • Comprehensive metadata")
    print("  • Visualization outputs")
    print("\nEstimated time: 2-5 minutes")
    
    response = input("\nProceed with data generation? (y/n): ")
    if response.lower() != 'y':
        print("Data generation cancelled.")
        return
    
    # Track overall progress
    start_total = time.time()
    success_count = 0
    scripts = [
        ("synchrotron_data/scripts/generate_tomography_data.py", 
         "Generating 3D/4D Tomography Data"),
        ("synchrotron_data/scripts/generate_diffraction_data.py",
         "Generating X-ray Diffraction Data"),
        ("synchrotron_data/scripts/generate_metadata.py",
         "Generating Experimental Metadata"),
        ("synchrotron_data/scripts/visualization_tools.py",
         "Creating Visualizations")
    ]
    
    for script_path, description in scripts:
        if os.path.exists(script_path):
            if run_script(script_path, description):
                success_count += 1
        else:
            print(f"\n✗ Script not found: {script_path}")
    
    # Summary
    total_time = time.time() - start_total
    
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print('='*60)
    print(f"Scripts executed: {success_count}/{len(scripts)}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if success_count == len(scripts):
        print("\n✓ All data successfully generated!")
        print("\nGenerated files are located in:")
        print("  • synchrotron_data/tomography/    - 3D/4D microstructure data")
        print("  • synchrotron_data/diffraction/   - XRD patterns and strain maps")
        print("  • synchrotron_data/metadata/      - Experimental parameters")
        print("  • synchrotron_data/visualizations/ - Generated plots")
        
        print("\nTo explore the data:")
        print("  1. Load HDF5 files with h5py or HDFView")
        print("  2. Review metadata JSON/YAML files")
        print("  3. Check visualization outputs")
        
    else:
        print("\n⚠ Some scripts failed. Check error messages above.")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()