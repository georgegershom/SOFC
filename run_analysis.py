#!/usr/bin/env python3
"""
Main execution script for the Vulnerability Analysis
This script runs the complete analysis pipeline
"""

import subprocess
import sys
import os

def check_and_install_packages():
    """Check and install required packages"""
    print("Checking required packages...")
    
    required_packages = [
        'matplotlib', 'seaborn', 'plotly', 'numpy', 'pandas', 'scipy', 'kaleido'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def run_analysis():
    """Run the main vulnerability analysis"""
    print("\n" + "="*60)
    print("SENCE FRAMEWORK VULNERABILITY ANALYSIS")
    print("Niger Delta Petroleum Cities")
    print("="*60)
    
    # Check if the main script exists
    if not os.path.exists('vulnerability_radar_chart.py'):
        print("Error: vulnerability_radar_chart.py not found!")
        return
    
    # Run the analysis
    print("\nExecuting analysis pipeline...")
    try:
        subprocess.run([sys.executable, 'vulnerability_radar_chart.py'], check=True)
        print("\n✓ Analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during analysis: {e}")
        return
    
    # Check generated outputs
    print("\n" + "-"*40)
    print("Generated Outputs:")
    print("-"*40)
    
    outputs = [
        ('radar_chart_advanced.png', 'Static radar chart visualization'),
        ('radar_chart_interactive.html', 'Interactive Plotly visualization'),
        ('statistical_report.txt', 'Statistical validation report'),
        ('sence_framework.mmd', 'Mermaid diagram of SENCE framework'),
        ('vulnerability_assessment.puml', 'PlantUML workflow diagram')
    ]
    
    for filename, description in outputs:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename:<35} ({size:,} bytes)")
            print(f"  {description}")
        else:
            print(f"✗ {filename:<35} (not generated)")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    # Provide instructions for viewing
    print("\nTo view the results:")
    print("1. Open 'radar_chart_advanced.png' for the static visualization")
    print("2. Open 'radar_chart_interactive.html' in a web browser for interactive charts")
    print("3. View 'statistical_report.txt' for detailed statistics")
    print("4. Use Mermaid Live Editor for 'sence_framework.mmd'")
    print("5. Use PlantUML viewer for 'vulnerability_assessment.puml'")

if __name__ == "__main__":
    # Check and install packages
    check_and_install_packages()
    
    # Run the analysis
    run_analysis()