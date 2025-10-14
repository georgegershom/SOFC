# Building Retrofit Dataset for AI- and IoT-driven Optimization

## Overview
This comprehensive dataset integrates real-time IoT sensor data, detailed building attributes, historical energy performance, and lifecycle assessment (LCA) information for PhD research on AI- and IoT-driven optimization of building retrofits.

## Dataset Structure

### Raw Data Categories

#### 1. IoT Sensor Data (`raw_data/iot_sensors/`)
- **Energy consumption**: Whole building & end-use consumption data
- **Indoor environmental parameters**: CO₂, TVOC, PM2.5, temperature, humidity
- **Outdoor weather conditions**: Temperature, humidity, wind speed, solar radiation
- **Occupancy patterns**: People count, activity levels, space utilization

#### 2. Building Attributes & Fabric (`raw_data/building_attributes/`)
- **Geometric & structural data**: Rooftop area, height, volume, floor area
- **Construction details**: Year, materials, function, style, quality
- **Thermal properties**: U-value, R-value, thermal mass, air tightness

#### 3. Energy Performance (`raw_data/energy_performance/`)
- **Historical consumption**: Monthly/yearly energy usage patterns
- **Efficiency ratings**: EU A-G ratings, ENERGY STAR scores
- **Post-retrofit data**: Energy savings, performance improvements

#### 4. Lifecycle Assessment (`raw_data/lca_data/`)
- **Environmental Product Declarations (EPDs)**: Material environmental data
- **Carbon footprint**: Construction materials and processes
- **LCA databases**: Building and civil engineering works data

### Processed Data (`processed_data/`)
- **Integrated**: Merged datasets with spatial and temporal alignment
- **Validated**: Quality-checked and cleaned data
- **Analysis-ready**: Preprocessed data optimized for ML/AI models

## Usage
1. Start with `processed_data/analysis_ready/` for immediate analysis
2. Use `scripts/` for data processing and integration
3. Refer to `documentation/` for detailed data descriptions

## Citation
If you use this dataset in your research, please cite:
```
Building Retrofit Dataset for AI- and IoT-driven Optimization
[Your Name], [Institution], [Year]
```

## License
This dataset is provided for academic research purposes.