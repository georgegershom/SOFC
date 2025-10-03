#!/usr/bin/env python3
"""
SOFC Simulation Runner
Executes Abaqus simulations for all heating rates and processes results
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_abaqus_simulation(input_file, job_name, output_dir):
    """Run Abaqus simulation"""
    
    print(f"Running simulation: {job_name}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Abaqus command
    cmd = [
        'abaqus',
        'job=' + job_name,
        'input=' + input_file,
        'interactive',
        'ask_delete=OFF'
    ]
    
    try:
        # Run Abaqus
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
        
        if result.returncode == 0:
            print(f"✓ Simulation {job_name} completed successfully")
            return True
        else:
            print(f"✗ Simulation {job_name} failed")
            print(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("✗ Abaqus not found. Please ensure Abaqus is installed and in PATH")
        return False
    except Exception as e:
        print(f"✗ Error running simulation: {e}")
        return False

def run_all_simulations():
    """Run all SOFC simulations"""
    
    base_dir = Path("/workspace/sofc_simulation")
    input_files = [
        "sofc_hr1.inp",
        "sofc_hr4.inp", 
        "sofc_hr10.inp"
    ]
    
    results = {}
    
    for input_file in input_files:
        job_name = input_file.replace('.inp', '')
        output_dir = base_dir / "outputs" / job_name
        
        print(f"\n{'='*60}")
        print(f"Running {job_name.upper()} simulation")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = run_abaqus_simulation(
            str(base_dir / input_file),
            job_name,
            str(output_dir)
        )
        end_time = time.time()
        
        results[job_name] = {
            'success': success,
            'runtime': end_time - start_time,
            'output_dir': str(output_dir)
        }
        
        if success:
            print(f"✓ {job_name} completed in {results[job_name]['runtime']:.1f} seconds")
        else:
            print(f"✗ {job_name} failed")
    
    # Summary
    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Successful simulations: {successful}/{total}")
    
    for job_name, result in results.items():
        status = "✓" if result['success'] else "✗"
        runtime = f"{result['runtime']:.1f}s" if result['success'] else "N/A"
        print(f"{status} {job_name}: {runtime}")
    
    return results

def main():
    """Main function"""
    
    print("SOFC Multi-Physics Simulation Runner")
    print("="*50)
    
    # Check if Abaqus is available
    try:
        result = subprocess.run(['abaqus', 'information=system'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Warning: Abaqus may not be properly configured")
    except FileNotFoundError:
        print("Error: Abaqus not found in PATH")
        print("Please install Abaqus and ensure it's accessible from command line")
        return
    
    # Run simulations
    results = run_all_simulations()
    
    # Check for output files
    print(f"\n{'='*60}")
    print("OUTPUT FILES CHECK")
    print(f"{'='*60}")
    
    for job_name, result in results.items():
        if result['success']:
            output_dir = Path(result['output_dir'])
            odb_file = output_dir / f"{job_name}.odb"
            dat_file = output_dir / f"{job_name}.dat"
            
            print(f"\n{job_name}:")
            print(f"  ODB file: {'✓' if odb_file.exists() else '✗'} {odb_file}")
            print(f"  DAT file: {'✓' if dat_file.exists() else '✗'} {dat_file}")

if __name__ == "__main__":
    main()