#!/usr/bin/env python3
"""
Integrated Building Retrofit Dataset Generator

This script generates a comprehensive multi-faceted dataset for AI- and IoT-driven 
optimization of building retrofits, including:
- IoT sensor data (energy, environmental, weather, occupancy)
- Building attributes and fabric data
- Energy performance data
- Lifecycle assessment (LCA) data

Author: AI Assistant
Version: 1.0.0
Date: 2024-10-14
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import h5py
from tqdm import tqdm
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.iot_data_generator import IoTDataGenerator
from scripts.building_attributes_generator import BuildingAttributesGenerator
from scripts.energy_performance_generator import EnergyPerformanceGenerator
from scripts.lca_data_generator import LCADataGenerator
from scripts.data_integrator import DataIntegrator
from scripts.data_validator import DataValidator

class BuildingRetrofitDatasetGenerator:
    """Main class for generating the integrated building retrofit dataset."""
    
    def __init__(self, config_path: str):
        """Initialize the dataset generator with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()
        self.setup_directories()
        
        # Initialize component generators
        self.iot_generator = IoTDataGenerator(self.config)
        self.building_generator = BuildingAttributesGenerator(self.config)
        self.energy_generator = EnergyPerformanceGenerator(self.config)
        self.lca_generator = LCADataGenerator(self.config)
        self.integrator = DataIntegrator(self.config)
        self.validator = DataValidator(self.config)
        
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories for data storage."""
        base_path = project_root / "data"
        directories = [
            "iot_sensors", "building_attributes", "energy_performance", 
            "lca_data", "integrated", "exports"
        ]
        
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
            
    def generate_building_registry(self) -> pd.DataFrame:
        """Generate a registry of all buildings in the dataset."""
        self.logger.info("Generating building registry...")
        
        num_buildings = self.config['scale']['num_buildings']
        cities = self.config['geographic']['cities']
        building_types = self.config['dataset_info']['building_types']
        
        buildings = []
        for i in tqdm(range(num_buildings), desc="Creating building registry"):
            city = np.random.choice(cities)
            building_type = np.random.choice(building_types)
            
            # Generate unique building ID
            building_id = f"BLD_{i+1:06d}"
            
            # Generate basic location data with some spatial clustering
            lat_offset = np.random.normal(0, 0.1)  # ~11km radius
            lon_offset = np.random.normal(0, 0.1)
            
            building = {
                'building_id': building_id,
                'city': city['name'],
                'country': city['country'],
                'latitude': city['latitude'] + lat_offset,
                'longitude': city['longitude'] + lon_offset,
                'climate_zone': city['climate_zone'],
                'heating_degree_days': city['heating_degree_days'],
                'cooling_degree_days': city['cooling_degree_days'],
                'building_type': building_type,
                'created_timestamp': datetime.now().isoformat()
            }
            buildings.append(building)
            
        registry_df = pd.DataFrame(buildings)
        
        # Save registry
        registry_path = project_root / "data" / "building_registry.csv"
        registry_df.to_csv(registry_path, index=False)
        self.logger.info(f"Building registry saved to {registry_path}")
        
        return registry_df
    
    def generate_complete_dataset(self):
        """Generate the complete integrated dataset."""
        self.logger.info("Starting complete dataset generation...")
        
        try:
            # Step 1: Generate building registry
            building_registry = self.generate_building_registry()
            
            # Step 2: Generate building attributes and fabric data
            self.logger.info("Generating building attributes...")
            building_attributes = self.building_generator.generate_attributes(building_registry)
            
            # Step 3: Generate energy performance data
            self.logger.info("Generating energy performance data...")
            energy_performance = self.energy_generator.generate_performance_data(
                building_registry, building_attributes
            )
            
            # Step 4: Generate LCA data
            self.logger.info("Generating LCA data...")
            lca_data = self.lca_generator.generate_lca_data(
                building_registry, building_attributes
            )
            
            # Step 5: Generate IoT sensor data (time series)
            self.logger.info("Generating IoT sensor data...")
            iot_data = self.iot_generator.generate_sensor_data(
                building_registry, building_attributes, energy_performance
            )
            
            # Step 6: Integrate all datasets
            self.logger.info("Integrating datasets...")
            integrated_dataset = self.integrator.integrate_datasets(
                building_registry, building_attributes, energy_performance, 
                lca_data, iot_data
            )
            
            # Step 7: Validate data quality
            self.logger.info("Validating data quality...")
            validation_report = self.validator.validate_dataset(integrated_dataset)
            
            # Step 8: Export datasets in multiple formats
            self.logger.info("Exporting datasets...")
            self.export_datasets(integrated_dataset, validation_report)
            
            self.logger.info("Dataset generation completed successfully!")
            return integrated_dataset
            
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {e}")
            raise
    
    def export_datasets(self, integrated_dataset: dict, validation_report: dict):
        """Export datasets in multiple formats for different use cases."""
        export_dir = project_root / "exports"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export as CSV files
        csv_dir = export_dir / f"csv_{timestamp}"
        csv_dir.mkdir(exist_ok=True)
        
        for dataset_name, data in integrated_dataset.items():
            if isinstance(data, pd.DataFrame):
                csv_path = csv_dir / f"{dataset_name}.csv"
                data.to_csv(csv_path, index=False)
                self.logger.info(f"Exported {dataset_name} to {csv_path}")
        
        # Export as Excel workbook
        excel_path = export_dir / f"building_retrofit_dataset_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for dataset_name, data in integrated_dataset.items():
                if isinstance(data, pd.DataFrame):
                    # Truncate sheet names to 31 characters (Excel limit)
                    sheet_name = dataset_name[:31]
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Exported complete dataset to {excel_path}")
        
        # Export as HDF5 for large time series data
        h5_path = export_dir / f"building_retrofit_dataset_{timestamp}.h5"
        with h5py.File(h5_path, 'w') as h5f:
            for dataset_name, data in integrated_dataset.items():
                if isinstance(data, pd.DataFrame):
                    # Convert to numpy arrays for HDF5 storage
                    grp = h5f.create_group(dataset_name)
                    
                    # Store numeric columns
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        numeric_data = data[numeric_cols].values
                        grp.create_dataset('numeric_data', data=numeric_data)
                        grp.attrs['numeric_columns'] = [col.encode() for col in numeric_cols]
                    
                    # Store string columns
                    string_cols = data.select_dtypes(include=['object']).columns
                    if len(string_cols) > 0:
                        for col in string_cols:
                            grp.create_dataset(f'string_{col}', 
                                             data=[str(val).encode() for val in data[col].values])
        
        self.logger.info(f"Exported HDF5 dataset to {h5_path}")
        
        # Export validation report
        report_path = export_dir / f"validation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        self.logger.info(f"Exported validation report to {report_path}")
        
        # Create dataset documentation
        self.create_dataset_documentation(export_dir, timestamp, validation_report)
    
    def create_dataset_documentation(self, export_dir: Path, timestamp: str, validation_report: dict):
        """Create comprehensive dataset documentation."""
        doc_content = f"""# Building Retrofit Dataset Documentation

## Dataset Information
- **Name**: {self.config['dataset_info']['name']}
- **Version**: {self.config['dataset_info']['version']}
- **Generated**: {datetime.now().isoformat()}
- **Geographic Focus**: {self.config['dataset_info']['geographic_focus']}
- **Building Types**: {', '.join(self.config['dataset_info']['building_types'])}

## Dataset Scale
- **Number of Buildings**: {self.config['scale']['num_buildings']:,}
- **Number of Cities**: {self.config['scale']['num_cities']}
- **Time Series Duration**: {self.config['scale']['time_series_duration_months']} months
- **Sensor Frequency**: {self.config['scale']['sensor_frequency_minutes']} minutes

## Data Categories

### 1. IoT Sensor Data
Real-time sensor measurements including:
- Energy consumption (whole building and end-use)
- Indoor environmental parameters (CO₂, TVOC, PM2.5, temperature, humidity)
- Outdoor weather conditions
- Occupancy patterns

### 2. Building Attributes & Fabric
Comprehensive building characteristics:
- Geometric and structural data
- Construction materials and thermal properties
- Building function and architectural style
- HVAC system information

### 3. Energy Performance
Historical and projected energy data:
- Annual energy consumption by fuel type
- EU energy efficiency ratings
- Post-retrofit performance projections
- Energy savings potential

### 4. Lifecycle Assessment (LCA)
Environmental impact data:
- Embodied carbon and energy of materials
- Construction process impacts
- End-of-life considerations
- Recyclability and durability metrics

## Data Quality Metrics
{json.dumps(validation_report.get('summary', {}), indent=2)}

## File Formats
- **CSV**: Individual datasets for easy analysis
- **Excel**: Complete workbook with all datasets
- **HDF5**: Optimized for large time series data
- **JSON**: Validation reports and metadata

## Usage Guidelines
1. **Data Integration**: Use building_id as the primary key to join datasets
2. **Time Series**: IoT data is indexed by timestamp and building_id
3. **Geographic Analysis**: Use latitude/longitude for spatial analysis
4. **Energy Modeling**: Combine building attributes with performance data
5. **LCA Studies**: Link material data with building attributes

## Citation
If you use this dataset in your research, please cite:
```
Building Retrofit Dataset v{self.config['dataset_info']['version']} ({datetime.now().year}). 
AI- and IoT-driven optimization of building retrofits. 
Generated dataset for PhD research.
```

## Contact
For questions about this dataset, please refer to the generation logs and validation reports.
"""
        
        doc_path = export_dir / f"dataset_documentation_{timestamp}.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        self.logger.info(f"Created dataset documentation at {doc_path}")

def main():
    """Main function to run the dataset generator."""
    config_path = project_root / "config" / "dataset_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    generator = BuildingRetrofitDatasetGenerator(str(config_path))
    dataset = generator.generate_complete_dataset()
    
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Generated dataset with {len(dataset.get('building_registry', []))} buildings")
    print(f"Export directory: {project_root / 'exports'}")
    print("="*50)

if __name__ == "__main__":
    main()