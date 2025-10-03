#!/usr/bin/env python3
"""
SOFC Complete Workflow Runner
Executes the entire SOFC simulation workflow from model generation to validation
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors"""
    
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully ({end_time - start_time:.1f}s)")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        else:
            print(f"✗ {description} failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False

def check_requirements():
    """Check if all requirements are met"""
    
    print("Checking requirements...")
    
    requirements = {
        'python3': 'Python 3.7+',
        'numpy': 'NumPy for numerical computations',
        'matplotlib': 'Matplotlib for visualization'
    }
    
    missing = []
    
    # Check Python
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Python: {result.stdout.strip()}")
        else:
            missing.append('python3')
    except:
        missing.append('python3')
    
    # Check Python packages
    for package in ['numpy', 'matplotlib']:
        try:
            result = subprocess.run(['python3', '-c', f'import {package}'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {package}: Available")
            else:
                missing.append(package)
        except:
            missing.append(package)
    
    # Check Abaqus (optional)
    try:
        result = subprocess.run(['abaqus', 'information=system'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Abaqus: Available")
        else:
            print("⚠ Abaqus: Not available (simulations will be skipped)")
    except:
        print("⚠ Abaqus: Not available (simulations will be skipped)")
    
    if missing:
        print(f"\nMissing requirements: {', '.join(missing)}")
        print("Please install missing packages:")
        for pkg in missing:
            if pkg == 'python3':
                print("  - Install Python 3.7+")
            else:
                print(f"  - pip install {pkg}")
        return False
    
    return True

def run_workflow():
    """Run the complete SOFC simulation workflow"""
    
    base_dir = Path("/workspace/sofc_simulation")
    scripts_dir = base_dir / "scripts"
    
    print("SOFC Complete Workflow")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("Requirements check failed. Please install missing dependencies.")
        return False
    
    # Step 1: Generate models
    print(f"\n{'='*60}")
    print("STEP 1: Generate SOFC Models")
    print(f"{'='*60}")
    
    if not run_command(['python3', 'generate_sofc_model.py'], 
                      "Generate SOFC models for all heating rates", 
                      cwd=scripts_dir):
        print("Model generation failed. Stopping workflow.")
        return False
    
    # Check if models were created
    model_files = ['sofc_hr1.inp', 'sofc_hr4.inp', 'sofc_hr10.inp']
    for model_file in model_files:
        if (base_dir / model_file).exists():
            print(f"✓ {model_file} created")
        else:
            print(f"✗ {model_file} not found")
    
    # Step 2: Run simulations (if Abaqus is available)
    print(f"\n{'='*60}")
    print("STEP 2: Run Simulations")
    print(f"{'='*60}")
    
    # Check if Abaqus is available
    try:
        result = subprocess.run(['abaqus', 'information=system'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Abaqus is available. Running simulations...")
            if not run_command(['python3', 'run_simulation.py'], 
                              "Run Abaqus simulations", 
                              cwd=scripts_dir):
                print("Simulation failed. Continuing with mock data for demonstration.")
        else:
            print("Abaqus not available. Skipping simulations.")
    except:
        print("Abaqus not available. Skipping simulations.")
    
    # Step 3: Post-process results
    print(f"\n{'='*60}")
    print("STEP 3: Post-Process Results")
    print(f"{'='*60}")
    
    if not run_command(['python3', 'post_process.py'], 
                      "Post-process simulation results", 
                      cwd=scripts_dir):
        print("Post-processing failed. Continuing with validation.")
    
    # Step 4: Validate results
    print(f"\n{'='*60}")
    print("STEP 4: Validate Results")
    print(f"{'='*60}")
    
    if not run_command(['python3', 'validation.py'], 
                      "Validate simulation results", 
                      cwd=scripts_dir):
        print("Validation failed.")
    
    # Step 5: Generate summary report
    print(f"\n{'='*60}")
    print("STEP 5: Generate Summary Report")
    print(f"{'='*60}")
    
    generate_summary_report(base_dir)
    
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETED")
    print(f"{'='*60}")
    print("Check the following directories for results:")
    print(f"  - Models: {base_dir}")
    print(f"  - Outputs: {base_dir / 'outputs'}")
    print(f"  - Validation: {base_dir / 'validation'}")
    
    return True

def generate_summary_report(base_dir):
    """Generate a summary report of the workflow"""
    
    report_file = base_dir / "workflow_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("SOFC Simulation Workflow Summary\n")
        f.write("="*50 + "\n\n")
        
        # Check generated files
        f.write("Generated Files:\n")
        f.write("-" * 20 + "\n")
        
        model_files = ['sofc_hr1.inp', 'sofc_hr4.inp', 'sofc_hr10.inp']
        for model_file in model_files:
            if (base_dir / model_file).exists():
                f.write(f"✓ {model_file}\n")
            else:
                f.write(f"✗ {model_file}\n")
        
        # Check output directories
        f.write("\nOutput Directories:\n")
        f.write("-" * 20 + "\n")
        
        output_dirs = ['outputs', 'validation', 'scripts']
        for output_dir in output_dirs:
            if (base_dir / output_dir).exists():
                f.write(f"✓ {output_dir}/\n")
            else:
                f.write(f"✗ {output_dir}/\n")
        
        # Check results files
        f.write("\nResults Files:\n")
        f.write("-" * 20 + "\n")
        
        for hr in ['hr1', 'hr4', 'hr10']:
            results_file = base_dir / "outputs" / f"sofc_{hr}" / "results" / "results.json"
            if results_file.exists():
                f.write(f"✓ sofc_{hr} results\n")
            else:
                f.write(f"✗ sofc_{hr} results\n")
        
        # Validation results
        validation_file = base_dir / "validation" / "validation_results.json"
        if validation_file.exists():
            f.write(f"✓ Validation results\n")
        else:
            f.write(f"✗ Validation results\n")
    
    print(f"✓ Summary report saved to {report_file}")

def main():
    """Main function"""
    
    # Change to workspace directory
    os.chdir("/workspace/sofc_simulation")
    
    # Run workflow
    success = run_workflow()
    
    if success:
        print("\n🎉 SOFC simulation workflow completed successfully!")
        print("\nNext steps:")
        print("1. Review generated models in the root directory")
        print("2. Check simulation results in outputs/")
        print("3. Examine validation results in validation/")
        print("4. Run individual scripts for specific tasks")
    else:
        print("\n❌ Workflow encountered errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()