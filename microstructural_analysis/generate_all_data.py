#!/usr/bin/env python3
"""
Master Script for Generating Complete Microstructural Analysis Dataset
PhD Research: Thermo-Mechanical Model for Fire-Resistant Rubberized Concrete
"""

import os
import sys
import time
from datetime import datetime
import subprocess

def run_generator(script_path, name):
    """Run a data generator script"""
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        elapsed = time.time() - start_time
        print(f"✅ {name} completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {name}:")
        print(e.stderr)
        return False

def main():
    """Generate all microstructural analysis datasets"""
    
    print("\n" + "="*70)
    print(" MICROSTRUCTURAL & CHEMICAL ANALYSIS DATASET GENERATION")
    print(" PhD Research: Fire-Resistant Rubberized Concrete")
    print("="*70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to the correct directory
    os.chdir('/workspace/microstructural_analysis')
    
    # Create necessary directories
    directories = [
        'SEM_Analysis', 'XRD_Analysis', 'TGA_DTA_Analysis', 
        'MicroCT_Analysis', 'figures', 'processed_data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\n📁 Directory structure created")
    
    # List of generators to run
    generators = [
        ('SEM_Analysis/sem_data_generator.py', 'SEM Analysis Generator'),
        ('XRD_Analysis/xrd_data_generator.py', 'XRD Analysis Generator'),
        ('TGA_DTA_Analysis/tga_dta_generator.py', 'TGA/DTA Analysis Generator'),
        ('MicroCT_Analysis/microct_data_generator.py', 'Micro-CT Analysis Generator')
    ]
    
    success_count = 0
    failed = []
    
    # Run each generator
    for script, name in generators:
        # Change to the appropriate directory
        script_dir = os.path.dirname(script)
        if script_dir:
            os.chdir(script_dir)
            script_name = os.path.basename(script)
        else:
            script_name = script
            
        if os.path.exists(script_name):
            if run_generator(script_name, name):
                success_count += 1
            else:
                failed.append(name)
        else:
            print(f"⚠️ Script not found: {script}")
            failed.append(name)
        
        # Return to main directory
        os.chdir('/workspace/microstructural_analysis')
    
    # Summary
    print("\n" + "="*70)
    print(" DATASET GENERATION SUMMARY")
    print("="*70)
    print(f"\n✅ Successfully generated: {success_count}/{len(generators)} datasets")
    
    if failed:
        print(f"❌ Failed generators: {', '.join(failed)}")
    
    # List all generated CSV files
    print("\n📊 Generated Data Files:")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.csv', '.json')):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  • {file_path} ({file_size:.1f} KB)")
    
    print(f"\n🎯 Total completion time: {time.time() - time.time():.1f} seconds")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Run integrated analysis:")
    print("   python scripts/integrated_analysis.py")
    print("\n2. View generated visualizations in 'figures' directory")
    print("\n3. Review PhD insights in 'figures/phd_insights.json'")
    print("\n4. Use data for thermo-mechanical model calibration")
    
    return success_count == len(generators)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)