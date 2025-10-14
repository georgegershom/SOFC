"""
Master Script to Generate All Building Retrofit Datasets
Run this script to generate the complete dataset suite.
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 80)

def run_script(script_name):
    """Run a Python script and handle errors."""
    start_time = time.time()
    
    try:
        print(f"Executing {script_name}...")
        exec(open(script_name).read(), {'__name__': '__main__'})
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.2f} seconds")
        return True
    except Exception as e:
        print(f"\n❌ Error executing {script_name}: {str(e)}")
        return False

def main():
    """Main execution function."""
    print_header("BUILDING RETROFIT DATASET GENERATOR")
    print(f"Generation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if data directory exists
    data_dir = '../data'
    if not os.path.exists(data_dir):
        print(f"\nCreating data directory: {data_dir}")
        os.makedirs(data_dir)
    
    # List of scripts to run in order
    scripts = [
        ('generate_building_attributes.py', 'Building Attributes Dataset'),
        ('generate_iot_sensor_data.py', 'IoT Sensor Data (Time Series)'),
        ('generate_energy_performance.py', 'Energy Performance & Retrofit Scenarios'),
        ('generate_lca_data.py', 'Lifecycle Assessment (LCA) Data'),
        ('integrate_datasets.py', 'Data Integration & ML-Ready Datasets')
    ]
    
    total_steps = len(scripts)
    successful = 0
    failed = 0
    
    overall_start = time.time()
    
    # Execute each script
    for i, (script, description) in enumerate(scripts, 1):
        print_step(i, total_steps, description)
        
        if run_script(script):
            successful += 1
        else:
            failed += 1
            response = input("\nContinue with remaining steps? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Final summary
    overall_time = time.time() - overall_start
    
    print_header("GENERATION COMPLETE")
    print(f"\nTotal time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Successful: {successful}/{total_steps}")
    print(f"Failed: {failed}/{total_steps}")
    
    if failed == 0:
        print("\n🎉 All datasets generated successfully!")
        print("\n📁 Your datasets are available in: ../data/")
        print("\nKey files:")
        print("  • building_attributes.csv - Building fabric and properties")
        print("  • iot_sensor_data.parquet - IoT sensor time series (438,000 records)")
        print("  • energy_performance_historical.csv - Historical energy consumption")
        print("  • retrofit_scenarios.csv - Retrofit analysis and ROI")
        print("  • lca_building_baseline.csv - Building LCA baselines")
        print("  • lca_retrofit_measures.csv - Retrofit measure LCA")
        print("  • integrated_building_master.csv - Comprehensive building profiles")
        print("  • integrated_retrofit_analysis.csv - Complete retrofit analysis")
        print("  • ml_ready_dataset.parquet - ML-ready feature dataset")
        print("\nReference databases:")
        print("  • material_properties_database.json")
        print("  • retrofit_measures_database.json")
        print("  • epd_database.json")
        print("  • hvac_lca_database.json")
        
        print("\n📖 Next steps:")
        print("  1. Explore the integrated datasets")
        print("  2. Review the data quality report (data_quality_report.json)")
        print("  3. Use ml_ready_dataset.parquet for machine learning models")
        print("  4. Analyze retrofit scenarios in integrated_retrofit_analysis.csv")
    else:
        print(f"\n⚠️  {failed} step(s) failed. Please review errors above.")
    
    print(f"\nGeneration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
