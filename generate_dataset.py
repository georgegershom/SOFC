"""
Main script to generate the complete Building Retrofit Dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from data_generators import (
    IoTDataGenerator, 
    BuildingAttributesGenerator, 
    EnergyPerformanceGenerator, 
    LCADataGenerator
)
from dataset_schema import schema

class BuildingRetrofitDatasetGenerator:
    """Main class to generate the complete building retrofit dataset"""
    
    def __init__(self, num_buildings: int = 100, start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
        self.num_buildings = num_buildings
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize generators
        self.iot_generator = IoTDataGenerator(num_buildings, start_date, end_date)
        self.building_generator = BuildingAttributesGenerator(num_buildings)
        self.energy_generator = EnergyPerformanceGenerator(num_buildings)
        self.lca_generator = LCADataGenerator()
        
        # Create output directory
        self.output_dir = "/workspace/dataset"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Metadata
        self.metadata = {
            "dataset_name": "Building Retrofit Optimization Dataset",
            "description": "Multi-faceted dataset for AI- and IoT-driven optimization of building retrofits",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "num_buildings": num_buildings,
            "date_range": f"{start_date} to {end_date}",
            "data_categories": [
                "IoT Sensor Data",
                "Building Attributes & Fabric", 
                "Energy Performance",
                "Lifecycle Assessment (LCA)"
            ],
            "total_records": 0,
            "file_sizes": {}
        }
    
    def generate_iot_data(self):
        """Generate all IoT sensor data"""
        print("Generating IoT sensor data...")
        
        # Energy consumption data
        print("  - Energy consumption data...")
        energy_data = self.iot_generator.generate_energy_consumption_data()
        energy_data.to_csv(f"{self.output_dir}/iot_energy_consumption.csv", index=False)
        self.metadata["file_sizes"]["iot_energy_consumption.csv"] = os.path.getsize(f"{self.output_dir}/iot_energy_consumption.csv")
        
        # Environmental parameters
        print("  - Environmental parameters...")
        env_data = self.iot_generator.generate_environmental_data()
        env_data.to_csv(f"{self.output_dir}/iot_environmental_parameters.csv", index=False)
        self.metadata["file_sizes"]["iot_environmental_parameters.csv"] = os.path.getsize(f"{self.output_dir}/iot_environmental_parameters.csv")
        
        # Weather conditions
        print("  - Weather conditions...")
        weather_data = self.iot_generator.generate_weather_data()
        weather_data.to_csv(f"{self.output_dir}/iot_weather_conditions.csv", index=False)
        self.metadata["file_sizes"]["iot_weather_conditions.csv"] = os.path.getsize(f"{self.output_dir}/iot_weather_conditions.csv")
        
        # Occupancy patterns
        print("  - Occupancy patterns...")
        occupancy_data = self.iot_generator.generate_occupancy_data()
        occupancy_data.to_csv(f"{self.output_dir}/iot_occupancy_patterns.csv", index=False)
        self.metadata["file_sizes"]["iot_occupancy_patterns.csv"] = os.path.getsize(f"{self.output_dir}/iot_occupancy_patterns.csv")
        
        # Calculate total records
        total_records = len(energy_data) + len(env_data) + len(weather_data) + len(occupancy_data)
        self.metadata["total_records"] += total_records
        
        print(f"  Generated {total_records:,} IoT sensor records")
    
    def generate_building_attributes(self):
        """Generate building attributes and fabric data"""
        print("Generating building attributes data...")
        
        # Basic building information
        print("  - Basic building information...")
        basic_info = self.building_generator.generate_basic_info()
        basic_info.to_csv(f"{self.output_dir}/building_basic_info.csv", index=False)
        self.metadata["file_sizes"]["building_basic_info.csv"] = os.path.getsize(f"{self.output_dir}/building_basic_info.csv")
        
        # Geometric data
        print("  - Geometric data...")
        geometric_data = self.building_generator.generate_geometric_data()
        geometric_data.to_csv(f"{self.output_dir}/building_geometric_data.csv", index=False)
        self.metadata["file_sizes"]["building_geometric_data.csv"] = os.path.getsize(f"{self.output_dir}/building_geometric_data.csv")
        
        # Thermal properties
        print("  - Thermal properties...")
        thermal_data = self.building_generator.generate_thermal_properties()
        thermal_data.to_csv(f"{self.output_dir}/building_thermal_properties.csv", index=False)
        self.metadata["file_sizes"]["building_thermal_properties.csv"] = os.path.getsize(f"{self.output_dir}/building_thermal_properties.csv")
        
        # Construction materials
        print("  - Construction materials...")
        materials_data = self.building_generator.generate_construction_materials()
        materials_data.to_csv(f"{self.output_dir}/building_construction_materials.csv", index=False)
        self.metadata["file_sizes"]["building_construction_materials.csv"] = os.path.getsize(f"{self.output_dir}/building_construction_materials.csv")
        
        # Calculate total records
        total_records = len(basic_info) + len(geometric_data) + len(thermal_data) + len(materials_data)
        self.metadata["total_records"] += total_records
        
        print(f"  Generated {total_records:,} building attribute records")
    
    def generate_energy_performance(self):
        """Generate energy performance data"""
        print("Generating energy performance data...")
        
        # Historical consumption
        print("  - Historical consumption...")
        historical_data = self.energy_generator.generate_historical_consumption()
        historical_data.to_csv(f"{self.output_dir}/energy_historical_consumption.csv", index=False)
        self.metadata["file_sizes"]["energy_historical_consumption.csv"] = os.path.getsize(f"{self.output_dir}/energy_historical_consumption.csv")
        
        # Efficiency ratings
        print("  - Efficiency ratings...")
        ratings_data = self.energy_generator.generate_efficiency_ratings()
        ratings_data.to_csv(f"{self.output_dir}/energy_efficiency_ratings.csv", index=False)
        self.metadata["file_sizes"]["energy_efficiency_ratings.csv"] = os.path.getsize(f"{self.output_dir}/energy_efficiency_ratings.csv")
        
        # Retrofit impact
        print("  - Retrofit impact...")
        retrofit_data = self.energy_generator.generate_retrofit_impact()
        retrofit_data.to_csv(f"{self.output_dir}/energy_retrofit_impact.csv", index=False)
        self.metadata["file_sizes"]["energy_retrofit_impact.csv"] = os.path.getsize(f"{self.output_dir}/energy_retrofit_impact.csv")
        
        # Calculate total records
        total_records = len(historical_data) + len(ratings_data) + len(retrofit_data)
        self.metadata["total_records"] += total_records
        
        print(f"  Generated {total_records:,} energy performance records")
    
    def generate_lca_data(self):
        """Generate Lifecycle Assessment data"""
        print("Generating LCA data...")
        
        # Material EPDs
        print("  - Material EPDs...")
        epd_data = self.lca_generator.generate_material_epds()
        epd_data.to_csv(f"{self.output_dir}/lca_material_epds.csv", index=False)
        self.metadata["file_sizes"]["lca_material_epds.csv"] = os.path.getsize(f"{self.output_dir}/lca_material_epds.csv")
        
        # Building LCA
        print("  - Building LCA...")
        building_lca_data = self.lca_generator.generate_building_lca(self.iot_generator.building_ids)
        building_lca_data.to_csv(f"{self.output_dir}/lca_building_lca.csv", index=False)
        self.metadata["file_sizes"]["lca_building_lca.csv"] = os.path.getsize(f"{self.output_dir}/lca_building_lca.csv")
        
        # Calculate total records
        total_records = len(epd_data) + len(building_lca_data)
        self.metadata["total_records"] += total_records
        
        print(f"  Generated {total_records:,} LCA records")
    
    def create_data_integration_script(self):
        """Create a data integration script for merging datasets"""
        integration_script = '''
"""
Data Integration Script for Building Retrofit Dataset
This script demonstrates how to merge and integrate the various dataset components
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BuildingRetrofitDataIntegrator:
    """Class for integrating and analyzing the building retrofit dataset"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.building_data = {}
        self.iot_data = {}
        self.energy_data = {}
        self.lca_data = {}
    
    def load_all_data(self):
        """Load all dataset components"""
        print("Loading dataset components...")
        
        # Load building attributes
        self.building_data['basic_info'] = pd.read_csv(f"{self.data_dir}/building_basic_info.csv")
        self.building_data['geometric'] = pd.read_csv(f"{self.data_dir}/building_geometric_data.csv")
        self.building_data['thermal'] = pd.read_csv(f"{self.data_dir}/building_thermal_properties.csv")
        self.building_data['materials'] = pd.read_csv(f"{self.data_dir}/building_construction_materials.csv")
        
        # Load IoT data
        self.iot_data['energy'] = pd.read_csv(f"{self.data_dir}/iot_energy_consumption.csv")
        self.iot_data['environmental'] = pd.read_csv(f"{self.data_dir}/iot_environmental_parameters.csv")
        self.iot_data['weather'] = pd.read_csv(f"{self.data_dir}/iot_weather_conditions.csv")
        self.iot_data['occupancy'] = pd.read_csv(f"{self.data_dir}/iot_occupancy_patterns.csv")
        
        # Load energy performance data
        self.energy_data['historical'] = pd.read_csv(f"{self.data_dir}/energy_historical_consumption.csv")
        self.energy_data['ratings'] = pd.read_csv(f"{self.data_dir}/energy_efficiency_ratings.csv")
        self.energy_data['retrofit'] = pd.read_csv(f"{self.data_dir}/energy_retrofit_impact.csv")
        
        # Load LCA data
        self.lca_data['epds'] = pd.read_csv(f"{self.data_dir}/lca_material_epds.csv")
        self.lca_data['building_lca'] = pd.read_csv(f"{self.data_dir}/lca_building_lca.csv")
        
        print("All data loaded successfully!")
    
    def create_integrated_building_dataset(self):
        """Create an integrated dataset with all building information"""
        print("Creating integrated building dataset...")
        
        # Start with basic info
        integrated = self.building_data['basic_info'].copy()
        
        # Merge geometric data
        integrated = integrated.merge(
            self.building_data['geometric'], 
            on='building_id', 
            how='left'
        )
        
        # Merge thermal properties
        integrated = integrated.merge(
            self.building_data['thermal'], 
            on='building_id', 
            how='left'
        )
        
        # Merge materials data
        integrated = integrated.merge(
            self.building_data['materials'], 
            on='building_id', 
            how='left'
        )
        
        # Add energy performance summary
        energy_summary = self.energy_data['historical'].groupby('building_id').agg({
            'total_energy_kwh': 'mean',
            'energy_intensity_kwhm2': 'mean'
        }).reset_index()
        energy_summary.columns = ['building_id', 'avg_annual_energy_kwh', 'avg_energy_intensity_kwhm2']
        
        integrated = integrated.merge(energy_summary, on='building_id', how='left')
        
        # Add latest efficiency rating
        latest_ratings = self.energy_data['ratings'].loc[
            self.energy_data['ratings'].groupby('building_id')['rating_year'].idxmax()
        ][['building_id', 'eu_energy_rating', 'energy_performance_index']]
        
        integrated = integrated.merge(latest_ratings, on='building_id', how='left')
        
        return integrated
    
    def create_time_series_dataset(self, building_id):
        """Create a time series dataset for a specific building"""
        print(f"Creating time series dataset for building {building_id}...")
        
        # Get all IoT data for the building
        energy = self.iot_data['energy'][self.iot_data['energy']['building_id'] == building_id].copy()
        environmental = self.iot_data['environmental'][self.iot_data['environmental']['building_id'] == building_id].copy()
        weather = self.iot_data['weather'][self.iot_data['weather']['building_id'] == building_id].copy()
        occupancy = self.iot_data['occupancy'][self.iot_data['occupancy']['building_id'] == building_id].copy()
        
        # Convert timestamps
        for df in [energy, environmental, weather, occupancy]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Merge all time series data
        time_series = energy[['total_consumption_kwh', 'heating_kwh', 'cooling_kwh']].copy()
        
        time_series = time_series.join(environmental[['co2_ppm', 'temperature_c', 'humidity_percent']])
        time_series = time_series.join(weather[['outdoor_temp_c', 'solar_irradiance_wm2']])
        time_series = time_series.join(occupancy[['occupancy_count', 'activity_level']])
        
        return time_series.dropna()
    
    def analyze_energy_patterns(self):
        """Analyze energy consumption patterns"""
        print("Analyzing energy patterns...")
        
        # Monthly energy consumption by building type
        energy_with_type = self.iot_data['energy'].merge(
            self.building_data['basic_info'][['building_id', 'building_type']], 
            on='building_id'
        )
        
        energy_with_type['month'] = pd.to_datetime(energy_with_type['timestamp']).dt.month
        monthly_consumption = energy_with_type.groupby(['building_type', 'month'])['total_consumption_kwh'].mean().reset_index()
        
        return monthly_consumption
    
    def identify_retrofit_candidates(self, energy_threshold=200):
        """Identify buildings that are good candidates for retrofit"""
        print("Identifying retrofit candidates...")
        
        # Calculate average energy intensity
        avg_energy = self.energy_data['historical'].groupby('building_id')['energy_intensity_kwhm2'].mean().reset_index()
        avg_energy.columns = ['building_id', 'avg_energy_intensity']
        
        # Get building info
        candidates = avg_energy.merge(
            self.building_data['basic_info'][['building_id', 'building_type', 'construction_year']], 
            on='building_id'
        )
        
        # Filter for high energy intensity and older buildings
        retrofit_candidates = candidates[
            (candidates['avg_energy_intensity'] > energy_threshold) & 
            (candidates['construction_year'] < 2000)
        ].sort_values('avg_energy_intensity', ascending=False)
        
        return retrofit_candidates

# Example usage
if __name__ == "__main__":
    integrator = BuildingRetrofitDataIntegrator()
    integrator.load_all_data()
    
    # Create integrated dataset
    integrated_data = integrator.create_integrated_building_dataset()
    print(f"Integrated dataset shape: {integrated_data.shape}")
    
    # Analyze energy patterns
    energy_patterns = integrator.analyze_energy_patterns()
    print("Energy patterns by building type:")
    print(energy_patterns.head())
    
    # Identify retrofit candidates
    candidates = integrator.identify_retrofit_candidates()
    print(f"Found {len(candidates)} retrofit candidates")
    print(candidates.head())
'''
        
        with open(f"{self.output_dir}/data_integration.py", "w") as f:
            f.write(integration_script)
    
    def create_documentation(self):
        """Create comprehensive documentation"""
        documentation = f"""
# Building Retrofit Optimization Dataset

## Overview
This dataset is designed for PhD research on AI- and IoT-driven optimization of building retrofits. It integrates multiple data sources to provide a comprehensive view of building performance, energy consumption, and environmental impact.

## Dataset Statistics
- **Total Buildings**: {self.metadata['num_buildings']:,}
- **Date Range**: {self.metadata['date_range']}
- **Total Records**: {self.metadata['total_records']:,}
- **Generated**: {self.metadata['created_date']}

## Data Categories

### 1. IoT Sensor Data
Real-time sensor data collected from building monitoring systems.

#### Files:
- `iot_energy_consumption.csv` - Energy consumption data by end use
- `iot_environmental_parameters.csv` - Indoor environmental quality metrics
- `iot_weather_conditions.csv` - Outdoor weather conditions
- `iot_occupancy_patterns.csv` - Building occupancy and activity patterns

#### Key Parameters:
- **Energy Consumption**: Total, heating, cooling, lighting, appliances, HVAC
- **Environmental**: CO₂, TVOC, PM2.5, temperature, humidity, air quality index
- **Weather**: Temperature, humidity, wind, solar irradiance, precipitation
- **Occupancy**: Count, density, activity level, occupancy type

### 2. Building Attributes & Fabric
Static building characteristics and construction details.

#### Files:
- `building_basic_info.csv` - Basic building information and location
- `building_geometric_data.csv` - Geometric and structural measurements
- `building_thermal_properties.csv` - Thermal properties of building envelope
- `building_construction_materials.csv` - Construction materials and specifications

#### Key Parameters:
- **Basic Info**: Location, construction year, building type, architectural style
- **Geometric**: Floor area, height, volume, window area, aspect ratio
- **Thermal**: U-values, R-values, thermal mass, air tightness
- **Materials**: Wall, roof, floor, window materials, insulation details

### 3. Energy Performance
Historical energy consumption and efficiency metrics.

#### Files:
- `energy_historical_consumption.csv` - Monthly energy consumption history
- `energy_efficiency_ratings.csv` - Energy efficiency ratings and certifications
- `energy_retrofit_impact.csv` - Post-retrofit performance improvements

#### Key Parameters:
- **Historical**: Monthly energy consumption by end use, energy intensity
- **Ratings**: EU energy ratings, performance indices, CO₂ emissions
- **Retrofit Impact**: Energy savings, cost, payback period, lifetime savings

### 4. Lifecycle Assessment (LCA)
Environmental impact data for materials and buildings.

#### Files:
- `lca_material_epds.csv` - Environmental Product Declarations for materials
- `lca_building_lca.csv` - Building-level lifecycle assessment data

#### Key Parameters:
- **Material EPDs**: GWP, acidification, eutrophication, ozone depletion, energy demand
- **Building LCA**: Total environmental impact by lifecycle stage, recycling potential

## Data Integration

The `data_integration.py` script provides tools for:
- Loading and merging all dataset components
- Creating integrated building datasets
- Generating time series data for specific buildings
- Analyzing energy patterns and identifying retrofit candidates

## Usage Examples

```python
from data_integration import BuildingRetrofitDataIntegrator

# Initialize integrator
integrator = BuildingRetrofitDataIntegrator()

# Load all data
integrator.load_all_data()

# Create integrated dataset
integrated_data = integrator.create_integrated_building_dataset()

# Analyze energy patterns
energy_patterns = integrator.analyze_energy_patterns()

# Identify retrofit candidates
candidates = integrator.identify_retrofit_candidates()
```

## Data Quality Notes

- All data is synthetically generated for research purposes
- Data follows realistic patterns based on building science literature
- Missing data is handled using appropriate imputation methods
- Temporal consistency is maintained across all time series data

## Research Applications

This dataset supports research in:
- Building energy performance prediction
- Retrofit optimization algorithms
- IoT data integration and analysis
- Lifecycle assessment and sustainability
- Machine learning for building science
- Digital twin development for buildings

## File Sizes
"""
        
        # Add file sizes to documentation
        for filename, size in self.metadata["file_sizes"].items():
            size_mb = size / (1024 * 1024)
            documentation += f"- `{filename}`: {size_mb:.2f} MB\n"
        
        documentation += f"""
## Citation

If you use this dataset in your research, please cite:

```
Building Retrofit Optimization Dataset v{self.metadata['version']}
Generated for PhD research on AI- and IoT-driven optimization of building retrofits
Created: {self.metadata['created_date']}
```

## Contact

For questions about this dataset, please refer to the data integration script and documentation provided.
"""
        
        with open(f"{self.output_dir}/README.md", "w") as f:
            f.write(documentation)
    
    def generate_complete_dataset(self):
        """Generate the complete dataset"""
        print("=" * 60)
        print("BUILDING RETROFIT DATASET GENERATOR")
        print("=" * 60)
        print(f"Generating dataset for {self.num_buildings:,} buildings")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print("=" * 60)
        
        # Generate all data components
        self.generate_iot_data()
        print()
        
        self.generate_building_attributes()
        print()
        
        self.generate_energy_performance()
        print()
        
        self.generate_lca_data()
        print()
        
        # Create integration script and documentation
        print("Creating data integration tools...")
        self.create_data_integration_script()
        self.create_documentation()
        
        # Save metadata
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print("=" * 60)
        print("DATASET GENERATION COMPLETE!")
        print("=" * 60)
        print(f"Total records generated: {self.metadata['total_records']:,}")
        print(f"Output directory: {self.output_dir}")
        print("Files created:")
        for filename in sorted(self.metadata["file_sizes"].keys()):
            size_mb = self.metadata["file_sizes"][filename] / (1024 * 1024)
            print(f"  - {filename}: {size_mb:.2f} MB")
        print("=" * 60)

if __name__ == "__main__":
    # Generate dataset with 100 buildings and 4 years of data
    generator = BuildingRetrofitDatasetGenerator(
        num_buildings=100,
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    generator.generate_complete_dataset()