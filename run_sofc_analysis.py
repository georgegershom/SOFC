#!/usr/bin/env python3
"""
Complete SOFC Thermal History Analysis Pipeline
Generates data and runs comprehensive analysis in one go.
"""

import subprocess
import sys
import os

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

def run_data_generation():
    """Run the thermal data generator."""
    print("\nRunning SOFC thermal data generation...")
    try:
        subprocess.check_call([sys.executable, "sofc_thermal_data_generator.py"])
        print("Data generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during data generation: {e}")
        return False
    return True

def run_analysis():
    """Run the thermal analysis."""
    print("\nRunning SOFC thermal analysis...")
    try:
        subprocess.check_call([sys.executable, "sofc_thermal_analyzer.py"])
        print("Analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during analysis: {e}")
        return False
    return True

def main():
    """Main pipeline execution."""
    print("SOFC THERMAL HISTORY ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Failed to install requirements. Exiting.")
        return
    
    # Step 2: Generate thermal data
    if not run_data_generation():
        print("Failed to generate thermal data. Exiting.")
        return
    
    # Step 3: Run analysis
    if not run_analysis():
        print("Failed to run analysis. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nGenerated files:")
    print("📁 sofc_thermal_data/")
    print("  ├── sintering_thermal_data.csv")
    print("  ├── thermal_cycling_data.csv")
    print("  ├── steady_state_thermal_data.csv")
    print("  ├── spatial_thermal_data.npz")
    print("  └── metadata.json")
    print("\n📁 thermal_analysis_plots/")
    print("  ├── sintering_analysis.png")
    print("  ├── sintering_critical_phases.png")
    print("  ├── thermal_cycling_analysis.png")
    print("  ├── steady_state_analysis.png")
    print("  ├── spatial_distributions.png")
    print("  ├── analysis_summary.json")
    print("  └── analysis_summary.txt")
    
    print("\n🎯 Key Results:")
    if os.path.exists("thermal_analysis_plots/analysis_summary.txt"):
        with open("thermal_analysis_plots/analysis_summary.txt", "r") as f:
            lines = f.readlines()
            # Print critical findings section
            in_critical = False
            for line in lines:
                if "CRITICAL FINDINGS:" in line:
                    in_critical = True
                    continue
                if in_critical and line.strip():
                    print(line.strip())

if __name__ == "__main__":
    main()