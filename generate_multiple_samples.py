#!/usr/bin/env python3
"""
Generate Multiple Sample Building DNA Datasets
Creates sample datasets for different building types
"""

import json
import uuid
from datetime import datetime
from building_dna_generator import BuildingDNAGenerator
from bim_generator import BIMGenerator
from iot_sensor_generator import IoTSensorGenerator
from lca_integration import LCAIntegration

def generate_building_dataset(building_type, index):
    """Generate a single building dataset"""
    print(f"\nGenerating {building_type} building {index}...")
    
    # Initialize generators
    dna_generator = BuildingDNAGenerator(seed=42 + index)
    bim_generator = BIMGenerator(seed=42 + index)
    iot_generator = IoTSensorGenerator(seed=42 + index)
    lca_integration = LCAIntegration(seed=42 + index)
    
    # Generate building DNA
    building_dna = dna_generator.generate_complete_building_dna(building_type)
    building_id = building_dna['metadata']['building_id']
    
    # Generate BIM model
    bim_model = bim_generator.generate_complete_bim_model(building_dna['geometry'])
    
    # Generate LiDAR point cloud (smaller for samples)
    lidar_points = bim_generator.generate_lidar_point_cloud(bim_model, point_density=200)
    
    # Generate IoT sensor data
    zones = iot_generator.generate_building_zones(building_dna['geometry'])
    sensor_configs = iot_generator.generate_sensor_configurations(zones)
    
    # Generate LCA assessment
    lca_assessment = lca_integration.generate_building_lca(building_dna, assessment_period=50)
    
    # Compile dataset
    dataset = {
        'metadata': {
            'dataset_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'building_id': building_id,
            'building_type': building_type,
            'generator_version': '1.0.0'
        },
        'building_dna': building_dna,
        'bim_model': {
            'model_id': bim_model.model_id,
            'elements_count': len(bim_model.elements),
            'materials': list(bim_model.materials.keys()),
            'spaces_count': len(bim_model.spaces),
            'systems_count': len(bim_model.systems)
        },
        'lidar_point_cloud': {
            'total_points': len(lidar_points),
            'sample_points': [{'x': p.x, 'y': p.y, 'z': p.z, 'intensity': p.intensity} for p in lidar_points[:100]]
        },
        'iot_sensor_data': {
            'zones_count': len(zones),
            'sensors_count': len(sensor_configs),
            'sensor_types': list(set(config.sensor_type for config in sensor_configs))
        },
        'lca_assessment': {
            'carbon_intensity': lca_assessment.carbon_intensity,
            'energy_intensity': lca_assessment.energy_intensity,
            'water_intensity': lca_assessment.water_intensity,
            'waste_intensity': lca_assessment.waste_intensity,
            'recommendations_count': len(lca_assessment.recommendations)
        },
        'performance_metrics': {
            'building_area_m2': building_dna['geometry']['total_area'],
            'building_volume_m3': building_dna['geometry']['total_volume'],
            'number_of_floors': building_dna['geometry']['number_of_floors'],
            'aspect_ratio': building_dna['geometry']['aspect_ratio'],
            'window_to_wall_ratio': building_dna['geometry']['window_to_wall_ratio']
        }
    }
    
    return dataset

def main():
    """Generate multiple building datasets"""
    print("Generating Multiple Building DNA Datasets")
    print("=" * 50)
    
    building_types = ['residential', 'commercial', 'office', 'industrial']
    num_buildings_per_type = 2
    
    all_datasets = []
    
    for building_type in building_types:
        print(f"\n{'='*20} {building_type.upper()} BUILDINGS {'='*20}")
        for i in range(num_buildings_per_type):
            dataset = generate_building_dataset(building_type, i)
            all_datasets.append(dataset)
            
            # Save individual dataset
            filename = f"{building_type}_building_{i+1}_{dataset['metadata']['building_id'][:8]}.json"
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
            print(f"   Saved: {filename}")
    
    # Create summary dataset
    summary = {
        'metadata': {
            'total_buildings': len(all_datasets),
            'building_types': building_types,
            'buildings_per_type': num_buildings_per_type,
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0'
        },
        'buildings': all_datasets
    }
    
    with open('complete_building_dna_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("DATASET GENERATION COMPLETE!")
    print(f"{'='*50}")
    print(f"Total buildings generated: {len(all_datasets)}")
    print(f"Summary file: complete_building_dna_summary.json")
    
    # Print statistics
    print("\nBuilding Type Statistics:")
    print("-" * 30)
    for building_type in building_types:
        buildings = [b for b in all_datasets if b['metadata']['building_type'] == building_type]
        avg_area = sum(b['performance_metrics']['building_area_m2'] for b in buildings) / len(buildings)
        avg_floors = sum(b['performance_metrics']['number_of_floors'] for b in buildings) / len(buildings)
        avg_carbon = sum(b['lca_assessment']['carbon_intensity'] for b in buildings) / len(buildings)
        
        print(f"{building_type.capitalize()}:")
        print(f"  Count: {len(buildings)}")
        print(f"  Avg Area: {avg_area:.0f} m²")
        print(f"  Avg Floors: {avg_floors:.1f}")
        print(f"  Avg Carbon: {avg_carbon:.1f} kg CO2/m²/year")
    
    print(f"\nTotal files generated: {len(all_datasets) + 1}")
    print("Each building dataset contains comprehensive data for:")
    print("  - Building DNA (static & fabric data)")
    print("  - 3D BIM model")
    print("  - LiDAR point cloud")
    print("  - IoT sensor data")
    print("  - LCA assessment")
    print("  - Performance metrics")

if __name__ == "__main__":
    main()