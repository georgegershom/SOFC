#!/usr/bin/env python3
"""
Master script to generate all contextual and external data
for the Dynamic Digital Twin Framework
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

def run_generator(script_path, description):
    """Run a data generator script and handle errors"""
    
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Change to the script's directory
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description} ({elapsed_time:.1f}s)")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 characters
        else:
            print(f"❌ FAILED: {description} ({elapsed_time:.1f}s)")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {str(e)}")
        return False
    
    return True

def main():
    """Generate all contextual and external data"""
    
    print("🚀 Starting Dynamic Digital Twin Framework Data Generation")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Define all data generators in order of execution
    generators = [
        # Weather & Climate Data
        {
            'script': 'weather_climate/tmy_data_generator.py',
            'description': 'TMY Weather Data Generation'
        },
        {
            'script': 'weather_climate/climate_projections_generator.py',
            'description': 'Climate Change Projections'
        },
        
        # Economic & Market Data
        {
            'script': 'economic_market/energy_prices_generator.py',
            'description': 'Energy Pricing Data'
        },
        {
            'script': 'economic_market/material_technology_costs.py',
            'description': 'Material & Technology Costs'
        },
        {
            'script': 'economic_market/labor_costs_generator.py',
            'description': 'Labor Costs Data'
        },
        {
            'script': 'economic_market/financial_parameters_generator.py',
            'description': 'Financial Parameters'
        },
        
        # Geospatial & Regulatory Data
        {
            'script': 'geospatial_regulatory/location_data_generator.py',
            'description': 'Location & Geospatial Data'
        },
        {
            'script': 'geospatial_regulatory/carbon_intensity_generator.py',
            'description': 'Grid Carbon Intensity Data'
        },
        {
            'script': 'geospatial_regulatory/building_codes_generator.py',
            'description': 'Building Codes & Standards'
        },
        
        # Data Integration
        {
            'script': 'data_integration/unified_data_access.py',
            'description': 'Unified Data Access Layer'
        }
    ]
    
    # Track results
    total_generators = len(generators)
    successful_generators = 0
    failed_generators = []
    
    overall_start_time = time.time()
    
    # Run each generator
    for i, generator in enumerate(generators, 1):
        print(f"\n📊 Progress: {i}/{total_generators}")
        
        script_path = os.path.join('contextual_external_data', generator['script'])
        
        if run_generator(script_path, generator['description']):
            successful_generators += 1
        else:
            failed_generators.append(generator['description'])
    
    # Summary
    overall_elapsed = time.time() - overall_start_time
    
    print(f"\n{'='*80}")
    print("🎯 DATA GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total generators: {total_generators}")
    print(f"Successful: {successful_generators}")
    print(f"Failed: {len(failed_generators)}")
    print(f"Success rate: {successful_generators/total_generators*100:.1f}%")
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    print(f"End time: {datetime.now().isoformat()}")
    
    if failed_generators:
        print(f"\n❌ Failed generators:")
        for failed in failed_generators:
            print(f"  - {failed}")
    
    # Generate final summary report
    generate_summary_report(successful_generators, failed_generators, overall_elapsed)
    
    print(f"\n🏁 Data generation completed!")
    
    return len(failed_generators) == 0  # Return True if all succeeded

def generate_summary_report(successful_count, failed_generators, elapsed_time):
    """Generate a comprehensive summary report"""
    
    report = {
        'generation_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_generators': successful_count + len(failed_generators),
            'successful_generators': successful_count,
            'failed_generators': len(failed_generators),
            'success_rate': successful_count / (successful_count + len(failed_generators)),
            'total_time_minutes': elapsed_time / 60
        },
        'data_categories_generated': {
            'weather_climate': {
                'tmy_data': 'Typical Meteorological Year data for 5 major cities',
                'climate_projections': 'Future climate scenarios (2030-2080) for 5 IPCC pathways'
            },
            'economic_market': {
                'energy_prices': 'Historical and forecast energy prices for 5 regions',
                'material_costs': 'Comprehensive material and technology cost database',
                'labor_costs': 'Trade-specific labor rates for 6 US regions',
                'financial_parameters': 'Discount rates, inflation, incentives, financing options'
            },
            'geospatial_regulatory': {
                'location_data': 'Detailed location data for 20 major US cities',
                'carbon_intensity': 'Hourly grid carbon intensity for 5 ISO/RTO regions',
                'building_codes': 'IECC requirements and local emissions standards'
            },
            'data_integration': {
                'unified_access': 'SQLite database with unified data access layer'
            }
        },
        'key_features': [
            'Hourly resolution weather and carbon intensity data',
            'Multiple climate change scenarios (SSP1-1.9 to SSP5-8.5)',
            'Time-of-use energy pricing with seasonal variations',
            'Regional cost variations and bulk pricing tiers',
            'Building code requirements by climate zone',
            'Local emissions standards (NYC Local Law 97, Boston BERDO)',
            'Comprehensive financial analysis parameters',
            'Unified SQLite database for fast data access'
        ],
        'data_volume': {
            'estimated_total_records': '> 1 million data points',
            'estimated_total_size': '> 100 MB',
            'time_coverage': '2015-2040 (historical and projections)',
            'geographic_coverage': 'Major US cities and regions'
        },
        'failed_generators': failed_generators if failed_generators else None,
        'next_steps': [
            'Run data validation scripts',
            'Initialize unified data access layer',
            'Test integration with Digital Twin Framework',
            'Set up automated data refresh procedures'
        ]
    }
    
    # Save report
    import json
    with open('contextual_external_data/data_generation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Summary report saved to: contextual_external_data/data_generation_report.json")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)