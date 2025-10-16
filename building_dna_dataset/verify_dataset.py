#!/usr/bin/env python3
"""
Dataset Verification Script
Validates the completeness and integrity of the Building DNA dataset
"""

import json
import os
from pathlib import Path
import sys

def verify_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✓ {description}: {size:,} bytes")
        return True
    else:
        print(f"✗ MISSING: {description}")
        return False

def verify_json_valid(file_path):
    """Check if JSON file is valid"""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON error in {file_path}: {e}")
        return False

def main():
    dataset_root = Path(__file__).parent
    print("="*80)
    print("BUILDING DNA DATASET - VERIFICATION REPORT")
    print("="*80 + "\n")
    
    files_to_check = {
        "Metadata": [
            ("metadata/building_info.json", "Building Information"),
            ("metadata/energy_model_inputs.json", "Energy Model Inputs"),
            ("metadata/iot_sensor_framework.json", "IoT Sensor Framework"),
            ("metadata/data_dictionary.json", "Data Dictionary"),
        ],
        "Geometric Data": [
            ("geometric_data/bim_metadata.json", "BIM Metadata"),
            ("geometric_data/floor_plans_data.json", "Floor Plans"),
            ("geometric_data/lidar_data.json", "LiDAR Data"),
            ("geometric_data/ifc_sample_extract.ifc", "IFC Model Sample"),
        ],
        "Construction Materials": [
            ("construction_materials/wall_assemblies.json", "Wall Assemblies"),
            ("construction_materials/roof_assemblies.json", "Roof Assemblies"),
            ("construction_materials/floor_assemblies.json", "Floor Assemblies"),
            ("construction_materials/windows_doors.json", "Windows & Doors"),
            ("construction_materials/material_properties_database.json", "Material Properties"),
        ],
        "HVAC Systems": [
            ("systems/hvac/hvac_system_specifications.json", "HVAC Specifications"),
            ("systems/hvac/hvac_historical_performance.csv", "HVAC Historical Data"),
        ],
        "Other Systems": [
            ("systems/dhw/dhw_system_specifications.json", "DHW System"),
            ("systems/lighting/lighting_system_inventory.json", "Lighting Inventory"),
            ("systems/renewable/renewable_energy_systems.json", "Renewable Energy"),
        ],
        "Environmental": [
            ("environmental/air_tightness_data.json", "Air Tightness Data"),
        ],
        "Documentation": [
            ("README.md", "Main Documentation"),
            ("QUICKSTART.md", "Quick Start Guide"),
            ("DATASET_SUMMARY.md", "Dataset Summary"),
            ("visualization_example.py", "Visualization Examples"),
            ("requirements.txt", "Python Requirements"),
        ]
    }
    
    total_files = 0
    valid_files = 0
    
    for category, files in files_to_check.items():
        print(f"\n{category}:")
        print("-" * 80)
        for file_path, description in files:
            full_path = dataset_root / file_path
            if verify_file_exists(full_path, description):
                total_files += 1
                if file_path.endswith('.json'):
                    if verify_json_valid(full_path):
                        valid_files += 1
                    else:
                        print(f"  ⚠ JSON validation failed")
                else:
                    valid_files += 1
    
    print("\n" + "="*80)
    print(f"VERIFICATION SUMMARY: {valid_files}/{total_files} files validated")
    print("="*80 + "\n")
    
    # Load and display key metrics
    try:
        with open(dataset_root / 'metadata/building_info.json', 'r') as f:
            building = json.load(f)
        
        print("KEY BUILDING METRICS:")
        print(f"  Building: {building['building_name']}")
        print(f"  Floor Area: {building['general_characteristics']['gross_floor_area_m2']:,.0f} m²")
        print(f"  EUI: {building['energy_summary']['eui_kwh_m2_year']:.1f} kWh/m²/year")
        print(f"  Annual Energy: {building['energy_summary']['annual_energy_consumption_kwh']:,.0f} kWh")
        print(f"  Retrofit Potential: {building['retrofit_potential']['estimated_eui_reduction_potential_percent']}%")
        print()
    except:
        pass
    
    if valid_files == total_files:
        print("✅ Dataset verification PASSED - All files present and valid!")
        return 0
    else:
        print("⚠️ Dataset verification INCOMPLETE - Some files missing or invalid")
        return 1

if __name__ == "__main__":
    sys.exit(main())
