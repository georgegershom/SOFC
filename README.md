# Building DNA Dataset Generator

A comprehensive tool for generating realistic building static and fabric data for Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization.

## Overview

This project generates a complete "Building DNA" dataset that includes all the foundational, non-changing data about a building's physical characteristics, as well as dynamic IoT sensor data and life-cycle assessment information. The dataset is designed to support advanced building retrofit optimization using digital twin technology.

## Features

### 🏗️ **Geometric Data**
- Detailed 3D BIM (Building Information Model) files with geometry, surfaces, volumes, and spatial relationships
- Digital floor plans with precise dimensions
- Aerial/LiDAR data for roof shape, surrounding context, and solar exposure analysis

### 🧱 **Construction & Material Data**
- Wall assemblies with layer-by-layer composition (brick, insulation, cladding)
- Material properties including U-value, R-value, thermal mass, density
- Roof & floor assemblies with detailed specifications
- Window & door specifications with U-value, SHGC, visible transmittance, frame type, gas fill, age, and condition
- Air tightness data from historical blower door test results

### ⚙️ **System Data**
- HVAC system specifications (make, model, fuel type, age, efficiency ratings)
- DHW (Domestic Hot Water) system details
- Lighting system inventory with fixture types, lamp types, wattages, and control systems
- Renewable energy systems (solar PV, solar thermal, wind, geothermal)

### 📡 **IoT Sensor Data**
- Real-time sensor readings (temperature, humidity, CO2, pressure, air velocity, light, occupancy, energy, water flow, vibration)
- Sensor configurations and maintenance schedules
- Energy consumption patterns
- Occupancy patterns and trends

### 🌱 **Life-Cycle Assessment (LCA)**
- Embodied carbon and energy calculations
- Operational impact assessments
- Material and process breakdowns
- Environmental impact categories (GWP, AP, EP, ODP, POCP)
- Sustainability recommendations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd building-dna-dataset-generator
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Generate a Single Building Dataset

```python
from comprehensive_dataset_generator import ComprehensiveDatasetGenerator

# Initialize generator
generator = ComprehensiveDatasetGenerator(seed=42)

# Generate complete building dataset
building_dataset = generator.generate_complete_building_dataset('office')

# Export dataset
output_path = generator.export_dataset(building_dataset, 'output_directory')
```

### Generate Multiple Building Datasets

```python
# Generate multiple building types
building_types = ['residential', 'commercial', 'office', 'industrial']
all_buildings = generator.generate_multiple_buildings(building_types, num_buildings_per_type=3)

# Export all datasets
for building_dataset in all_buildings:
    generator.export_dataset(building_dataset, 'output_directory')
```

### Run Complete Dataset Generation

```bash
python run_dataset_generation.py
```

## Dataset Structure

Each building dataset contains the following files:

### Core Data Files
- **`building_dna.json`** - Static & fabric data (geometry, materials, systems)
- **`bim_model.json`** - 3D Building Information Model
- **`lidar_point_cloud.json`** - LiDAR point cloud data
- **`iot_sensor_data.json`** - IoT sensor configurations and readings
- **`lca_assessment.json`** - Life-cycle assessment data
- **`performance_metrics.json`** - Building performance indicators
- **`complete_dataset.json`** - Complete integrated dataset

### Visualizations
- **`bim_visualization.html`** - Interactive 3D BIM model
- **`lidar_visualization.html`** - LiDAR point cloud visualization
- **`sensor_data_visualization.html`** - IoT sensor data over time
- **`energy_consumption_visualization.html`** - Energy consumption patterns
- **`lca_impacts_visualization.html`** - LCA impact assessment charts

### Documentation
- **`README.md`** - Building-specific documentation

## Building Types Supported

### 🏠 **Residential**
- Single-family homes, apartments, condominiums
- Typical residential systems and materials
- Occupancy patterns for residential use

### 🏢 **Commercial**
- Retail spaces, restaurants, shopping centers
- Commercial HVAC and lighting systems
- High occupancy and energy usage patterns

### 🏢 **Office**
- Corporate offices, co-working spaces
- Advanced HVAC and lighting controls
- Professional occupancy patterns

### 🏭 **Industrial**
- Manufacturing facilities, warehouses
- Heavy-duty systems and equipment
- Industrial energy consumption patterns

## Data Quality Features

### Realistic Data Generation
- Material properties based on real-world specifications
- Realistic building geometries and proportions
- Accurate system specifications and performance data
- Time-series data with realistic patterns and noise

### Data Validation
- Range checking for all parameters
- Consistency validation across related data
- Quality indicators for sensor data
- Confidence scores for measurements

### Comprehensive Coverage
- All major building components included
- Multiple data sources integrated
- Historical and real-time data
- Performance and environmental metrics

## Usage Examples

### Building Performance Analysis
```python
# Load building dataset
with open('building_dna.json', 'r') as f:
    building_data = json.load(f)

# Access performance metrics
performance = building_data['performance_metrics']
print(f"Energy intensity: {performance['avg_energy_intensity_kwh_m2']} kWh/m²")
print(f"Carbon intensity: {performance['carbon_intensity_kg_co2_m2_year']} kg CO2/m²/year")
print(f"Performance score: {performance['performance_score']}/100")
```

### IoT Sensor Data Analysis
```python
# Load sensor data
with open('iot_sensor_data.json', 'r') as f:
    sensor_data = json.load(f)

# Analyze temperature readings
temp_readings = [r for r in sensor_data['sensor_readings'] if r['sensor_type'] == 'temperature']
avg_temp = sum(r['value'] for r in temp_readings) / len(temp_readings)
print(f"Average temperature: {avg_temp:.1f}°C")
```

### LCA Impact Assessment
```python
# Load LCA data
with open('lca_assessment.json', 'r') as f:
    lca_data = json.load(f)

# View environmental impacts
print(f"Total embodied carbon: {lca_data['total_embodied_carbon']:.0f} kg CO2")
print(f"Total operational carbon: {lca_data['total_operational_carbon']:.0f} kg CO2/year")
print(f"Recommendations: {len(lca_data['recommendations'])} generated")
```

## Advanced Features

### Custom Building Generation
```python
# Generate building with specific parameters
building_dna = generator.dna_generator.generate_complete_building_dna('office')
# Modify specific parameters
building_dna['geometry']['total_area'] = 5000  # 5000 m²
building_dna['geometry']['number_of_floors'] = 10

# Generate complete dataset with custom building
dataset = generator.generate_complete_building_dataset('office', building_id='custom_office_001')
```

### Sensor Data Customization
```python
# Generate custom sensor configurations
zones = generator.iot_generator.generate_building_zones(building_geometry)
sensor_configs = generator.iot_generator.generate_sensor_configurations(zones)

# Modify sensor parameters
for config in sensor_configs:
    if config.sensor_type == 'temperature':
        config.accuracy = 0.05  # High accuracy temperature sensors
        config.sampling_rate = 0.2  # 5-second sampling
```

### LCA Assessment Customization
```python
# Generate LCA with custom assessment period
lca_assessment = generator.lca_integration.generate_building_lca(
    building_dna, 
    assessment_period=100  # 100-year assessment
)
```

## Output Formats

### JSON Format
All data is provided in JSON format for easy integration with various analysis tools and frameworks.

### CSV Format
Time-series data (energy consumption, occupancy patterns) is also available in CSV format for spreadsheet analysis.

### HTML Visualizations
Interactive visualizations are provided in HTML format for easy viewing in web browsers.

## Integration with Digital Twin Frameworks

This dataset is designed for seamless integration with:
- Building Information Modeling (BIM) software
- Building Energy Modeling (BEM) tools
- IoT platforms and sensor networks
- Machine learning and AI frameworks
- Optimization algorithms for retrofit planning

## Performance Metrics

The generated datasets include comprehensive performance metrics:
- Energy intensity (kWh/m²/year)
- Carbon intensity (kg CO2/m²/year)
- Water intensity (L/m²/year)
- Waste intensity (kg/m²/year)
- Overall performance score (0-100)
- Efficiency rating (A+ to F)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional building types
- New sensor types
- Enhanced material databases
- Improved visualization tools
- Additional LCA impact categories

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@software{building_dna_dataset_generator,
  title={Building DNA Dataset Generator for Dynamic Digital Twin Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/building-dna-dataset-generator}
}
```

## Support

For questions, issues, or support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Contact the maintainers

## Changelog

### Version 1.0.0
- Initial release
- Complete building DNA dataset generation
- 3D BIM model generation
- LiDAR point cloud generation
- IoT sensor data simulation
- Life-cycle assessment integration
- Interactive visualizations
- Multiple building type support

---

**Note**: This tool generates synthetic data for research and development purposes. While the data is realistic and based on real-world parameters, it should not be used for actual building design or construction without proper validation and verification.