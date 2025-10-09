#!/usr/bin/env python3
"""
SOFC Thermal Data Generation and Analysis Runner - Optimized Version
This script runs the complete SOFC thermal analysis pipeline with memory optimization
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def run_data_generator():
    """Run the thermal data generator"""
    print("\n" + "="*60)
    print("GENERATING SOFC THERMAL HISTORY DATA")
    print("="*60)
    
    try:
        import sofc_thermal_data_generator_optimized
        generator = sofc_thermal_data_generator_optimized.SOFCThermalDataGeneratorOptimized()
        sintering_data, cycling_data, steady_state_data, stress_data = generator.generate_all_data()
        print("✓ Data generation completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return False

def run_analysis():
    """Run the thermal analysis"""
    print("\n" + "="*60)
    print("RUNNING SOFC THERMAL ANALYSIS")
    print("="*60)
    
    try:
        import sofc_thermal_analysis_optimized
        sofc_thermal_analysis_optimized.main()
        print("✓ Analysis completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error running analysis: {e}")
        return False

def main():
    """Main execution function"""
    print("SOFC Thermal History Data Generation and Analysis - Optimized")
    print("=" * 70)
    print("This script will generate comprehensive thermal data for SOFC analysis")
    print("including sintering, thermal cycling, steady-state operation, and residual stresses.")
    print("(Memory-optimized version for better performance)")
    print()
    
    # Check if we're in the right directory
    if not Path("sofc_thermal_data_generator_optimized.py").exists():
        print("✗ Error: sofc_thermal_data_generator_optimized.py not found in current directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Generate data
    if not run_data_generator():
        return
    
    # Run analysis
    if not run_analysis():
        return
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("📊 Data Files:")
    print("  - sofc_sintering_data.csv")
    print("  - sofc_thermal_cycling_data.csv") 
    print("  - sofc_steady_state_data.csv")
    print("  - sofc_residual_stress_data.csv")
    print("  - sofc_thermal_data_summary.json")
    print()
    print("📈 Visualization Files:")
    print("  - sofc_sintering_analysis.png")
    print("  - sofc_thermal_cycling_analysis.png")
    print("  - sofc_steady_state_analysis.png")
    print("  - sofc_residual_stress_analysis.png")
    print()
    print("📋 Report:")
    print("  - sofc_thermal_analysis_report.md")
    print()
    print("All thermal history data has been generated and analyzed!")
    print("Check the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main()