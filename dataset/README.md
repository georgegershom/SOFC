
# Building Retrofit Optimization Dataset

## Overview
This dataset is designed for PhD research on AI- and IoT-driven optimization of building retrofits. It integrates multiple data sources to provide a comprehensive view of building performance, energy consumption, and environmental impact.

## Dataset Statistics
- **Total Buildings**: 100
- **Date Range**: 2020-01-01 to 2023-12-31
- **Total Records**: 591,352
- **Generated**: 2025-10-14T01:51:42.143650

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
- `iot_energy_consumption.csv`: 8.15 MB
- `iot_environmental_parameters.csv`: 6.89 MB
- `iot_weather_conditions.csv`: 7.57 MB
- `iot_occupancy_patterns.csv`: 5.64 MB
- `building_basic_info.csv`: 0.01 MB
- `building_geometric_data.csv`: 0.01 MB
- `building_thermal_properties.csv`: 0.01 MB
- `building_construction_materials.csv`: 0.01 MB
- `energy_historical_consumption.csv`: 0.30 MB
- `energy_efficiency_ratings.csv`: 0.00 MB
- `energy_retrofit_impact.csv`: 0.00 MB
- `lca_material_epds.csv`: 0.00 MB
- `lca_building_lca.csv`: 0.02 MB

## Citation

If you use this dataset in your research, please cite:

```
Building Retrofit Optimization Dataset v1.0.0
Generated for PhD research on AI- and IoT-driven optimization of building retrofits
Created: 2025-10-14T01:51:42.143650
```

## Contact

For questions about this dataset, please refer to the data integration script and documentation provided.
