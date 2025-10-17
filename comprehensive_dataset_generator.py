#!/usr/bin/env python3
"""
Comprehensive Building DNA Dataset Generator
Integrates all components to create a complete building DNA dataset
for Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization
"""

import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import zipfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Import all generator modules
from building_dna_generator import BuildingDNAGenerator
from bim_generator import BIMGenerator
from iot_sensor_generator import IoTSensorGenerator
from lca_integration import LCAIntegration

class ComprehensiveDatasetGenerator:
    """Main class for generating comprehensive building DNA datasets"""
    
    def __init__(self, seed: int = 42):
        """Initialize the comprehensive dataset generator"""
        self.seed = seed
        self.dna_generator = BuildingDNAGenerator(seed=seed)
        self.bim_generator = BIMGenerator(seed=seed)
        self.iot_generator = IoTSensorGenerator(seed=seed)
        self.lca_integration = LCAIntegration(seed=seed)
        
    def generate_complete_building_dataset(self, building_type: str = 'residential', 
                                         building_id: str = None) -> Dict[str, Any]:
        """Generate complete building dataset with all components"""
        print(f"Generating complete building dataset for {building_type} building...")
        
        if building_id is None:
            building_id = str(uuid.uuid4())
        
        # 1. Generate Building DNA (Static & Fabric Data)
        print("  - Generating building DNA...")
        building_dna = self.dna_generator.generate_complete_building_dna(building_type)
        building_dna['metadata']['building_id'] = building_id
        
        # 2. Generate 3D BIM Model
        print("  - Generating 3D BIM model...")
        bim_model = self.bim_generator.generate_complete_bim_model(building_dna['geometry'])
        
        # 3. Generate LiDAR Point Cloud
        print("  - Generating LiDAR point cloud...")
        lidar_points = self.bim_generator.generate_lidar_point_cloud(bim_model, point_density=1000)
        
        # 4. Generate IoT Sensor Data
        print("  - Generating IoT sensor data...")
        zones = self.iot_generator.generate_building_zones(building_dna['geometry'])
        sensor_configs = self.iot_generator.generate_sensor_configurations(zones)
        
        # Generate 7 days of sensor data
        start_time = datetime.now() - timedelta(days=7)
        sensor_readings = self.iot_generator.generate_sensor_readings(
            sensor_configs, zones, start_time, 168  # 7 days * 24 hours
        )
        
        # Generate energy and occupancy data
        energy_data = self.iot_generator.generate_energy_consumption_data(zones, start_time, 168)
        occupancy_data = self.iot_generator.generate_occupancy_patterns(zones, start_time, 168)
        
        # 5. Generate LCA Assessment
        print("  - Generating LCA assessment...")
        lca_assessment = self.lca_integration.generate_building_lca(building_dna, assessment_period=50)
        
        # 6. Compile complete dataset
        complete_dataset = {
            'metadata': {
                'dataset_id': str(uuid.uuid4()),
                'generated_at': datetime.now().isoformat(),
                'building_id': building_id,
                'building_type': building_type,
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
                'elements': [self._element_to_dict(elem) for elem in bim_model.elements],
                'materials': bim_model.materials,
                'spaces': bim_model.spaces,
                'systems': bim_model.systems,
                'metadata': bim_model.metadata
            },
            'lidar_point_cloud': {
                'points': [self._point_to_dict(point) for point in lidar_points],
                'metadata': {
                    'total_points': len(lidar_points),
                    'generated_at': datetime.now().isoformat(),
                    'coordinate_system': 'WGS84',
                    'units': 'meters'
                }
            },
            'iot_sensor_data': {
                'zones': [self._zone_to_dict(zone) for zone in zones],
                'sensor_configurations': [self._sensor_config_to_dict(config) for config in sensor_configs],
                'sensor_readings': [self._sensor_reading_to_dict(reading) for reading in sensor_readings],
                'energy_consumption': energy_data.to_dict('records'),
                'occupancy_patterns': occupancy_data.to_dict('records'),
                'metadata': {
                    'total_readings': len(sensor_readings),
                    'data_period': '7 days',
                    'generated_at': datetime.now().isoformat()
                }
            },
            'lca_assessment': self._lca_to_dict(lca_assessment),
            'performance_metrics': self._calculate_performance_metrics(
                building_dna, lca_assessment, energy_data, occupancy_data
            )
        }
        
        return complete_dataset
    
    def _element_to_dict(self, element):
        """Convert BIM element to dictionary"""
        return {
            'id': element.id,
            'element_type': element.element_type,
            'material_id': element.material_id,
            'geometry': element.geometry,
            'properties': element.properties,
            'location': element.location,
            'orientation': element.orientation,
            'dimensions': element.dimensions,
            'level': element.level,
            'zone': element.zone,
            'construction_id': element.construction_id
        }
    
    def _point_to_dict(self, point):
        """Convert LiDAR point to dictionary"""
        return {
            'x': point.x,
            'y': point.y,
            'z': point.z,
            'intensity': point.intensity,
            'return_number': point.return_number,
            'number_of_returns': point.number_of_returns,
            'classification': point.classification,
            'scan_angle': point.scan_angle,
            'gps_time': point.gps_time,
            'red': point.red,
            'green': point.green,
            'blue': point.blue
        }
    
    def _zone_to_dict(self, zone):
        """Convert building zone to dictionary"""
        return {
            'zone_id': zone.zone_id,
            'name': zone.name,
            'level': zone.level,
            'area': zone.area,
            'volume': zone.volume,
            'occupancy': zone.occupancy,
            'zone_type': zone.zone_type,
            'setpoint_temperature': zone.setpoint_temperature,
            'setpoint_humidity': zone.setpoint_humidity,
            'ventilation_rate': zone.ventilation_rate,
            'lighting_level': zone.lighting_level,
            'equipment_load': zone.equipment_load
        }
    
    def _sensor_config_to_dict(self, config):
        """Convert sensor configuration to dictionary"""
        return {
            'sensor_id': config.sensor_id,
            'sensor_type': config.sensor_type,
            'location': config.location,
            'zone': config.zone,
            'measurement_range': config.measurement_range,
            'accuracy': config.accuracy,
            'resolution': config.resolution,
            'sampling_rate': config.sampling_rate,
            'calibration_interval': config.calibration_interval,
            'last_calibration': config.last_calibration,
            'next_maintenance': config.next_maintenance,
            'manufacturer': config.manufacturer,
            'model': config.model,
            'installation_date': config.installation_date,
            'status': config.status
        }
    
    def _sensor_reading_to_dict(self, reading):
        """Convert sensor reading to dictionary"""
        return {
            'sensor_id': reading.sensor_id,
            'timestamp': reading.timestamp,
            'value': reading.value,
            'unit': reading.unit,
            'quality': reading.quality,
            'confidence': reading.confidence,
            'location': reading.location,
            'zone': reading.zone,
            'sensor_type': reading.sensor_type,
            'calibration_date': reading.calibration_date,
            'maintenance_due': reading.maintenance_due
        }
    
    def _lca_to_dict(self, lca):
        """Convert LCA assessment to dictionary"""
        return {
            'building_id': lca.building_id,
            'assessment_date': lca.assessment_date,
            'assessment_period': lca.assessment_period,
            'total_embodied_carbon': lca.total_embodied_carbon,
            'total_operational_carbon': lca.total_operational_carbon,
            'total_embodied_energy': lca.total_embodied_energy,
            'total_operational_energy': lca.total_operational_energy,
            'total_water_consumption': lca.total_water_consumption,
            'total_waste_generation': lca.total_waste_generation,
            'carbon_intensity': lca.carbon_intensity,
            'energy_intensity': lca.energy_intensity,
            'water_intensity': lca.water_intensity,
            'waste_intensity': lca.waste_intensity,
            'materials_breakdown': lca.materials_breakdown,
            'processes_breakdown': lca.processes_breakdown,
            'impact_categories': lca.impact_categories,
            'recommendations': lca.recommendations
        }
    
    def _calculate_performance_metrics(self, building_dna, lca_assessment, energy_data, occupancy_data):
        """Calculate overall building performance metrics"""
        # Energy performance
        total_energy_consumption = energy_data['energy_consumption_kwh'].sum()
        avg_energy_intensity = energy_data['energy_consumption_kwh'].mean() / energy_data['area'].mean()
        
        # Occupancy performance
        avg_occupancy = occupancy_data['occupancy_count'].mean()
        occupancy_efficiency = avg_occupancy / occupancy_data['area'].mean() if occupancy_data['area'].mean() > 0 else 0
        
        # Environmental performance
        carbon_intensity = lca_assessment.carbon_intensity
        energy_intensity = lca_assessment.energy_intensity
        
        # Building efficiency metrics
        building_area = building_dna['geometry']['total_area']
        building_volume = building_dna['geometry']['total_volume']
        
        # Calculate efficiency ratios
        energy_per_m2 = total_energy_consumption / building_area if building_area > 0 else 0
        carbon_per_m2 = carbon_intensity
        
        # Performance rating (0-100)
        performance_score = self._calculate_performance_score(
            energy_per_m2, carbon_per_m2, avg_occupancy, building_type=building_dna['metadata']['building_type']
        )
        
        return {
            'total_energy_consumption_kwh': total_energy_consumption,
            'avg_energy_intensity_kwh_m2': avg_energy_intensity,
            'avg_occupancy': avg_occupancy,
            'occupancy_efficiency_persons_m2': occupancy_efficiency,
            'carbon_intensity_kg_co2_m2_year': carbon_intensity,
            'energy_intensity_mj_m2_year': energy_intensity,
            'building_area_m2': building_area,
            'building_volume_m3': building_volume,
            'performance_score': performance_score,
            'efficiency_rating': self._get_efficiency_rating(performance_score)
        }
    
    def _calculate_performance_score(self, energy_per_m2, carbon_per_m2, occupancy, building_type):
        """Calculate overall performance score (0-100)"""
        # Base scores for different building types
        base_scores = {
            'residential': 70,
            'commercial': 60,
            'office': 65,
            'industrial': 50
        }
        
        base_score = base_scores.get(building_type, 60)
        
        # Energy efficiency factor
        if energy_per_m2 < 50:
            energy_factor = 1.2
        elif energy_per_m2 < 100:
            energy_factor = 1.0
        elif energy_per_m2 < 200:
            energy_factor = 0.8
        else:
            energy_factor = 0.6
        
        # Carbon efficiency factor
        if carbon_per_m2 < 20:
            carbon_factor = 1.2
        elif carbon_per_m2 < 50:
            carbon_factor = 1.0
        elif carbon_per_m2 < 100:
            carbon_factor = 0.8
        else:
            carbon_factor = 0.6
        
        # Occupancy factor
        if occupancy > 0:
            occupancy_factor = 1.1
        else:
            occupancy_factor = 0.9
        
        # Calculate final score
        final_score = base_score * energy_factor * carbon_factor * occupancy_factor
        return min(100, max(0, final_score))
    
    def _get_efficiency_rating(self, performance_score):
        """Get efficiency rating based on performance score"""
        if performance_score >= 90:
            return 'A+'
        elif performance_score >= 80:
            return 'A'
        elif performance_score >= 70:
            return 'B'
        elif performance_score >= 60:
            return 'C'
        elif performance_score >= 50:
            return 'D'
        else:
            return 'F'
    
    def generate_multiple_buildings(self, building_types: List[str], 
                                  num_buildings_per_type: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple building datasets"""
        all_buildings = []
        
        for building_type in building_types:
            print(f"\nGenerating {num_buildings_per_type} {building_type} buildings...")
            for i in range(num_buildings_per_type):
                print(f"  Building {i+1}/{num_buildings_per_type}")
                building_dataset = self.generate_complete_building_dataset(building_type)
                all_buildings.append(building_dataset)
        
        return all_buildings
    
    def export_dataset(self, dataset: Dict[str, Any], output_dir: str = "building_dna_dataset") -> str:
        """Export complete dataset to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        building_id = dataset['metadata']['building_id']
        building_type = dataset['metadata']['building_type']
        
        # Create building-specific directory
        building_dir = output_path / f"{building_type}_{building_id}"
        building_dir.mkdir(exist_ok=True)
        
        # Export individual components
        print(f"Exporting dataset to: {building_dir}")
        
        # 1. Building DNA
        with open(building_dir / "building_dna.json", 'w') as f:
            json.dump(dataset['building_dna'], f, indent=2, default=str)
        
        # 2. BIM Model
        with open(building_dir / "bim_model.json", 'w') as f:
            json.dump(dataset['bim_model'], f, indent=2, default=str)
        
        # 3. LiDAR Point Cloud
        with open(building_dir / "lidar_point_cloud.json", 'w') as f:
            json.dump(dataset['lidar_point_cloud'], f, indent=2, default=str)
        
        # 4. IoT Sensor Data
        with open(building_dir / "iot_sensor_data.json", 'w') as f:
            json.dump(dataset['iot_sensor_data'], f, indent=2, default=str)
        
        # 5. LCA Assessment
        with open(building_dir / "lca_assessment.json", 'w') as f:
            json.dump(dataset['lca_assessment'], f, indent=2, default=str)
        
        # 6. Performance Metrics
        with open(building_dir / "performance_metrics.json", 'w') as f:
            json.dump(dataset['performance_metrics'], f, indent=2, default=str)
        
        # 7. Complete Dataset
        with open(building_dir / "complete_dataset.json", 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        # 8. Generate visualizations
        self._generate_visualizations(dataset, building_dir)
        
        # 9. Create README
        self._create_readme(building_dir, dataset)
        
        return str(building_dir)
    
    def _generate_visualizations(self, dataset: Dict[str, Any], output_dir: Path):
        """Generate visualization files"""
        try:
            # BIM 3D visualization
            self.bim_generator.visualize_bim_model(
                self._dict_to_bim_model(dataset['bim_model']),
                str(output_dir / "bim_visualization.html")
            )
            
            # LiDAR visualization
            self.bim_generator.visualize_lidar_points(
                self._dict_to_lidar_points(dataset['lidar_point_cloud']),
                str(output_dir / "lidar_visualization.html")
            )
            
            # IoT sensor data visualization
            self.iot_generator.visualize_sensor_data(
                self._dict_to_sensor_readings(dataset['iot_sensor_data']['sensor_readings']),
                str(output_dir / "sensor_data_visualization.html")
            )
            
            # Energy consumption visualization
            energy_df = pd.DataFrame(dataset['iot_sensor_data']['energy_consumption'])
            self.iot_generator.visualize_energy_consumption(
                energy_df,
                str(output_dir / "energy_consumption_visualization.html")
            )
            
            # LCA impacts visualization
            self.lca_integration.visualize_lca_impacts(
                self._dict_to_lca_assessment(dataset['lca_assessment']),
                str(output_dir / "lca_impacts_visualization.html")
            )
            
        except Exception as e:
            print(f"Warning: Could not generate some visualizations: {e}")
    
    def _dict_to_bim_model(self, bim_dict):
        """Convert dictionary back to BIM model object"""
        from bim_generator import BIMModel, BIMElement
        # This is a simplified conversion - in practice, you'd need to handle all fields
        return BIMModel(
            model_id=bim_dict['model_id'],
            building_id=bim_dict['building_id'],
            elements=[],  # Simplified
            materials=bim_dict['materials'],
            spaces=bim_dict['spaces'],
            systems=bim_dict['systems'],
            metadata=bim_dict['metadata']
        )
    
    def _dict_to_lidar_points(self, lidar_dict):
        """Convert dictionary back to LiDAR points"""
        from bim_generator import LiDARPoint
        points = []
        for point_dict in lidar_dict['points']:
            point = LiDARPoint(
                x=point_dict['x'],
                y=point_dict['y'],
                z=point_dict['z'],
                intensity=point_dict['intensity'],
                return_number=point_dict['return_number'],
                number_of_returns=point_dict['number_of_returns'],
                classification=point_dict['classification'],
                scan_angle=point_dict['scan_angle'],
                gps_time=point_dict['gps_time'],
                red=point_dict['red'],
                green=point_dict['green'],
                blue=point_dict['blue']
            )
            points.append(point)
        return points
    
    def _dict_to_sensor_readings(self, readings_dict):
        """Convert dictionary back to sensor readings"""
        from iot_sensor_generator import SensorReading
        readings = []
        for reading_dict in readings_dict:
            reading = SensorReading(
                sensor_id=reading_dict['sensor_id'],
                timestamp=reading_dict['timestamp'],
                value=reading_dict['value'],
                unit=reading_dict['unit'],
                quality=reading_dict['quality'],
                confidence=reading_dict['confidence'],
                location=tuple(reading_dict['location']),
                zone=reading_dict['zone'],
                sensor_type=reading_dict['sensor_type'],
                calibration_date=reading_dict['calibration_date'],
                maintenance_due=reading_dict['maintenance_due']
            )
            readings.append(reading)
        return readings
    
    def _dict_to_lca_assessment(self, lca_dict):
        """Convert dictionary back to LCA assessment"""
        from lca_integration import LCABuilding
        return LCABuilding(
            building_id=lca_dict['building_id'],
            assessment_date=lca_dict['assessment_date'],
            assessment_period=lca_dict['assessment_period'],
            total_embodied_carbon=lca_dict['total_embodied_carbon'],
            total_operational_carbon=lca_dict['total_operational_carbon'],
            total_embodied_energy=lca_dict['total_embodied_energy'],
            total_operational_energy=lca_dict['total_operational_energy'],
            total_water_consumption=lca_dict['total_water_consumption'],
            total_waste_generation=lca_dict['total_waste_generation'],
            carbon_intensity=lca_dict['carbon_intensity'],
            energy_intensity=lca_dict['energy_intensity'],
            water_intensity=lca_dict['water_intensity'],
            waste_intensity=lca_dict['waste_intensity'],
            materials_breakdown=lca_dict['materials_breakdown'],
            processes_breakdown=lca_dict['processes_breakdown'],
            impact_categories=lca_dict['impact_categories'],
            recommendations=lca_dict['recommendations']
        )
    
    def _create_readme(self, output_dir: Path, dataset: Dict[str, Any]):
        """Create README file for the dataset"""
        readme_content = f"""# Building DNA Dataset

## Building Information
- **Building ID**: {dataset['metadata']['building_id']}
- **Building Type**: {dataset['metadata']['building_type']}
- **Generated At**: {dataset['metadata']['generated_at']}
- **Generator Version**: {dataset['metadata']['generator_version']}

## Dataset Components

### 1. Building DNA (building_dna.json)
Static and fabric data including:
- Geometric data (3D BIM, floor plans, LiDAR data)
- Construction & material data (wall assemblies, windows, doors)
- System data (HVAC, lighting, renewable energy)
- Air tightness data

### 2. BIM Model (bim_model.json)
3D Building Information Model including:
- Building elements (walls, floors, columns, windows)
- Material properties
- Spatial relationships
- System integration

### 3. LiDAR Point Cloud (lidar_point_cloud.json)
Point cloud data including:
- 3D coordinates (x, y, z)
- Intensity values
- Classification data
- Color information

### 4. IoT Sensor Data (iot_sensor_data.json)
Real-time monitoring data including:
- Sensor configurations
- Time-series readings
- Energy consumption data
- Occupancy patterns

### 5. LCA Assessment (lca_assessment.json)
Life-cycle assessment data including:
- Embodied carbon and energy
- Operational impacts
- Material breakdown
- Environmental recommendations

### 6. Performance Metrics (performance_metrics.json)
Building performance indicators:
- Energy intensity
- Carbon intensity
- Occupancy efficiency
- Overall performance score

## Visualizations
- `bim_visualization.html`: 3D BIM model visualization
- `lidar_visualization.html`: LiDAR point cloud visualization
- `sensor_data_visualization.html`: IoT sensor data over time
- `energy_consumption_visualization.html`: Energy consumption patterns
- `lca_impacts_visualization.html`: LCA impact assessment

## Usage
This dataset is designed for use in Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization. It provides comprehensive building data for:
- Building performance analysis
- Retrofit optimization
- Energy efficiency improvements
- Environmental impact assessment
- Real-time monitoring and control

## Data Format
All data is provided in JSON format for easy integration with various analysis tools and frameworks.
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    def create_dataset_archive(self, output_dir: str, archive_name: str = None) -> str:
        """Create a ZIP archive of the complete dataset"""
        if archive_name is None:
            archive_name = f"building_dna_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        output_path = Path(output_dir)
        archive_path = output_path.parent / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_path.parent)
                    zipf.write(file_path, arcname)
        
        print(f"Dataset archive created: {archive_path}")
        return str(archive_path)

def main():
    """Main function to demonstrate comprehensive dataset generation"""
    print("Comprehensive Building DNA Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = ComprehensiveDatasetGenerator(seed=42)
    
    # Generate multiple building types
    building_types = ['residential', 'commercial', 'office', 'industrial']
    all_buildings = generator.generate_multiple_buildings(building_types, num_buildings_per_type=2)
    
    print(f"\nGenerated {len(all_buildings)} complete building datasets")
    
    # Export all datasets
    output_dir = "comprehensive_building_dna_dataset"
    exported_paths = []
    
    for i, building_dataset in enumerate(all_buildings):
        print(f"\nExporting building {i+1}/{len(all_buildings)}...")
        building_path = generator.export_dataset(building_dataset, output_dir)
        exported_paths.append(building_path)
    
    # Create summary dataset
    summary_data = {
        'metadata': {
            'total_buildings': len(all_buildings),
            'building_types': building_types,
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0'
        },
        'buildings': all_buildings
    }
    
    with open(f"{output_dir}/complete_dataset_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # Create dataset archive
    archive_path = generator.create_dataset_archive(output_dir)
    
    print(f"\nDataset generation complete!")
    print(f"Total buildings: {len(all_buildings)}")
    print(f"Output directory: {output_dir}")
    print(f"Archive created: {archive_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print("-" * 30)
    for building_type in building_types:
        count = sum(1 for b in all_buildings if b['metadata']['building_type'] == building_type)
        print(f"{building_type.capitalize()}: {count} buildings")
    
    print(f"\nTotal files generated: {len(exported_paths)} building directories")
    print("Each building directory contains:")
    print("  - building_dna.json")
    print("  - bim_model.json") 
    print("  - lidar_point_cloud.json")
    print("  - iot_sensor_data.json")
    print("  - lca_assessment.json")
    print("  - performance_metrics.json")
    print("  - complete_dataset.json")
    print("  - Multiple HTML visualizations")
    print("  - README.md")

if __name__ == "__main__":
    main()