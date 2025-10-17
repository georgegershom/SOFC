#!/usr/bin/env python3
"""
Dataset Validation Script
Validates the generated building DNA datasets for completeness and quality
"""

import json
import os
from pathlib import Path

def validate_building_dataset(filepath):
    """Validate a single building dataset"""
    print(f"Validating: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return False
    
    validation_results = {
        'file_size': os.path.getsize(filepath),
        'has_metadata': 'metadata' in data,
        'has_building_dna': 'building_dna' in data,
        'has_bim_model': 'bim_model' in data,
        'has_lidar_data': 'lidar_point_cloud' in data,
        'has_iot_data': 'iot_sensor_data' in data,
        'has_lca_data': 'lca_assessment' in data,
        'has_performance_metrics': 'performance_metrics' in data
    }
    
    # Check metadata
    if 'metadata' in data:
        metadata = data['metadata']
        validation_results['has_building_id'] = 'building_id' in metadata
        validation_results['has_building_type'] = 'building_type' in metadata
        validation_results['has_generated_at'] = 'generated_at' in metadata
    
    # Check building DNA
    if 'building_dna' in data:
        building_dna = data['building_dna']
        validation_results['has_geometry'] = 'geometry' in building_dna
        validation_results['has_construction_materials'] = 'construction_materials' in building_dna
        validation_results['has_systems'] = 'systems' in building_dna
    
    # Check BIM model
    if 'bim_model' in data:
        bim_model = data['bim_model']
        validation_results['has_elements'] = 'elements_count' in bim_model or 'elements' in bim_model
        validation_results['has_materials'] = 'materials' in bim_model
    
    # Check LiDAR data
    if 'lidar_point_cloud' in data:
        lidar_data = data['lidar_point_cloud']
        validation_results['has_points'] = 'total_points' in lidar_data or 'points' in lidar_data
    
    # Check IoT data
    if 'iot_sensor_data' in data:
        iot_data = data['iot_sensor_data']
        validation_results['has_zones'] = 'zones_count' in iot_data or 'zones' in iot_data
        validation_results['has_sensors'] = 'sensors_count' in iot_data or 'sensor_configurations' in iot_data
    
    # Check LCA data
    if 'lca_assessment' in data:
        lca_data = data['lca_assessment']
        validation_results['has_carbon_intensity'] = 'carbon_intensity' in lca_data
        validation_results['has_energy_intensity'] = 'energy_intensity' in lca_data
        validation_results['has_recommendations'] = 'recommendations_count' in lca_data or 'recommendations' in lca_data
    
    # Print validation results
    print(f"  File size: {validation_results['file_size']:,} bytes")
    print(f"  Metadata: {'✅' if validation_results['has_metadata'] else '❌'}")
    print(f"  Building DNA: {'✅' if validation_results['has_building_dna'] else '❌'}")
    print(f"  BIM Model: {'✅' if validation_results['has_bim_model'] else '❌'}")
    print(f"  LiDAR Data: {'✅' if validation_results['has_lidar_data'] else '❌'}")
    print(f"  IoT Data: {'✅' if validation_results['has_iot_data'] else '❌'}")
    print(f"  LCA Data: {'✅' if validation_results['has_lca_data'] else '❌'}")
    print(f"  Performance Metrics: {'✅' if validation_results['has_performance_metrics'] else '❌'}")
    
    # Calculate overall score
    total_checks = len([k for k in validation_results.keys() if k != 'file_size'])
    passed_checks = sum(1 for k, v in validation_results.items() if k != 'file_size' and v)
    score = (passed_checks / total_checks) * 100
    
    print(f"  Overall Score: {score:.1f}% ({passed_checks}/{total_checks})")
    
    return score >= 80  # Consider valid if 80% or more checks pass

def main():
    """Validate all generated datasets"""
    print("Building DNA Dataset Validation")
    print("=" * 40)
    
    # Find all JSON files
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"Found {len(json_files)} JSON files to validate\n")
    
    validation_results = []
    
    for json_file in sorted(json_files):
        is_valid = validate_building_dataset(json_file)
        validation_results.append((json_file, is_valid))
        print()
    
    # Summary
    print("=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    
    valid_files = [f for f, v in validation_results if v]
    invalid_files = [f for f, v in validation_results if not v]
    
    print(f"Total files: {len(json_files)}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    print(f"Success rate: {len(valid_files)/len(json_files)*100:.1f}%")
    
    if invalid_files:
        print(f"\nInvalid files:")
        for file in invalid_files:
            print(f"  - {file}")
    
    if valid_files:
        print(f"\nValid files:")
        for file in valid_files:
            print(f"  ✅ {file}")
    
    print(f"\nDataset validation complete!")

if __name__ == "__main__":
    main()