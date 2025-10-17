#!/usr/bin/env python3
"""
Sample Building DNA Dataset Generator
Generates a simplified sample dataset for demonstration
"""

import json
import uuid
from datetime import datetime
from building_dna_generator import BuildingDNAGenerator
from bim_generator import BIMGenerator
from iot_sensor_generator import IoTSensorGenerator
from lca_integration import LCAIntegration

def generate_sample_dataset():
    """Generate a sample building dataset"""
    print("Generating Sample Building DNA Dataset")
    print("=" * 40)
    
    # Initialize generators
    dna_generator = BuildingDNAGenerator(seed=42)
    bim_generator = BIMGenerator(seed=42)
    iot_generator = IoTSensorGenerator(seed=42)
    lca_integration = LCAIntegration(seed=42)
    
    # Generate building DNA
    print("1. Generating Building DNA...")
    building_dna = dna_generator.generate_complete_building_dna('office')
    building_id = building_dna['metadata']['building_id']
    print(f"   Building ID: {building_id}")
    
    # Generate BIM model
    print("2. Generating 3D BIM Model...")
    bim_model = bim_generator.generate_complete_bim_model(building_dna['geometry'])
    print(f"   BIM Model ID: {bim_model.model_id}")
    print(f"   Elements: {len(bim_model.elements)}")
    
    # Generate LiDAR point cloud
    print("3. Generating LiDAR Point Cloud...")
    lidar_points = bim_generator.generate_lidar_point_cloud(bim_model, point_density=500)
    print(f"   Points: {len(lidar_points)}")
    
    # Generate IoT sensor data
    print("4. Generating IoT Sensor Data...")
    zones = iot_generator.generate_building_zones(building_dna['geometry'])
    sensor_configs = iot_generator.generate_sensor_configurations(zones)
    print(f"   Zones: {len(zones)}")
    print(f"   Sensors: {len(sensor_configs)}")
    
    # Generate LCA assessment
    print("5. Generating LCA Assessment...")
    lca_assessment = lca_integration.generate_building_lca(building_dna, assessment_period=50)
    print(f"   Carbon Intensity: {lca_assessment.carbon_intensity:.1f} kg CO2/m²/year")
    
    # Compile complete dataset
    complete_dataset = {
        'metadata': {
            'dataset_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'building_id': building_id,
            'building_type': 'office',
            'generator_version': '1.0.0',
            'components': [
                'building_dna',
                'bim_model',
                'lidar_point_cloud',
                'iot_sensor_data',
                'lca_assessment'
            ]
        },
        'building_dna': building_dna,
        'bim_model': {
            'model_id': bim_model.model_id,
            'building_id': bim_model.building_id,
            'elements': [{
                'id': elem.id,
                'element_type': elem.element_type,
                'material_id': elem.material_id,
                'location': elem.location,
                'dimensions': elem.dimensions,
                'level': elem.level,
                'zone': elem.zone
            } for elem in bim_model.elements],
            'materials': bim_model.materials,
            'spaces': bim_model.spaces,
            'systems': bim_model.systems,
            'metadata': bim_model.metadata
        },
        'lidar_point_cloud': {
            'points': [{
                'x': point.x,
                'y': point.y,
                'z': point.z,
                'intensity': point.intensity,
                'classification': point.classification
            } for point in lidar_points[:1000]],  # Limit to 1000 points for sample
            'metadata': {
                'total_points': len(lidar_points),
                'sample_points': 1000,
                'generated_at': datetime.now().isoformat()
            }
        },
        'iot_sensor_data': {
            'zones': [{
                'zone_id': zone.zone_id,
                'name': zone.name,
                'level': zone.level,
                'area': zone.area,
                'zone_type': zone.zone_type,
                'occupancy': zone.occupancy
            } for zone in zones],
            'sensor_configurations': [{
                'sensor_id': config.sensor_id,
                'sensor_type': config.sensor_type,
                'location': config.location,
                'zone': config.zone,
                'manufacturer': config.manufacturer,
                'model': config.model,
                'status': config.status
            } for config in sensor_configs],
            'metadata': {
                'total_sensors': len(sensor_configs),
                'total_zones': len(zones),
                'generated_at': datetime.now().isoformat()
            }
        },
        'lca_assessment': {
            'building_id': lca_assessment.building_id,
            'assessment_date': lca_assessment.assessment_date,
            'assessment_period': lca_assessment.assessment_period,
            'total_embodied_carbon': lca_assessment.total_embodied_carbon,
            'total_operational_carbon': lca_assessment.total_operational_carbon,
            'carbon_intensity': lca_assessment.carbon_intensity,
            'energy_intensity': lca_assessment.energy_intensity,
            'water_intensity': lca_assessment.water_intensity,
            'waste_intensity': lca_assessment.waste_intensity,
            'recommendations': lca_assessment.recommendations
        },
        'performance_metrics': {
            'building_area_m2': building_dna['geometry']['total_area'],
            'building_volume_m3': building_dna['geometry']['total_volume'],
            'number_of_floors': building_dna['geometry']['number_of_floors'],
            'carbon_intensity_kg_co2_m2_year': lca_assessment.carbon_intensity,
            'energy_intensity_mj_m2_year': lca_assessment.energy_intensity,
            'total_sensors': len(sensor_configs),
            'total_zones': len(zones),
            'bim_elements': len(bim_model.elements),
            'lidar_points': len(lidar_points)
        }
    }
    
    # Save dataset
    output_file = f"sample_building_dna_dataset_{building_id[:8]}.json"
    with open(output_file, 'w') as f:
        json.dump(complete_dataset, f, indent=2, default=str)
    
    print(f"\nSample dataset saved to: {output_file}")
    
    # Print summary
    print("\nDataset Summary:")
    print("-" * 30)
    print(f"Building Type: {complete_dataset['metadata']['building_type']}")
    print(f"Building Area: {complete_dataset['performance_metrics']['building_area_m2']:.0f} m²")
    print(f"Building Volume: {complete_dataset['performance_metrics']['building_volume_m3']:.0f} m³")
    print(f"Number of Floors: {complete_dataset['performance_metrics']['number_of_floors']}")
    print(f"BIM Elements: {complete_dataset['performance_metrics']['bim_elements']}")
    print(f"LiDAR Points: {complete_dataset['performance_metrics']['lidar_points']}")
    print(f"Sensors: {complete_dataset['performance_metrics']['total_sensors']}")
    print(f"Zones: {complete_dataset['performance_metrics']['total_zones']}")
    print(f"Carbon Intensity: {complete_dataset['lca_assessment']['carbon_intensity']:.1f} kg CO2/m²/year")
    print(f"Energy Intensity: {complete_dataset['lca_assessment']['energy_intensity']:.1f} MJ/m²/year")
    print(f"LCA Recommendations: {len(complete_dataset['lca_assessment']['recommendations'])}")
    
    return complete_dataset

if __name__ == "__main__":
    generate_sample_dataset()