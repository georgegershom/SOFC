# Building Retrofit Dataset - Overview

## Dataset Summary

This comprehensive dataset integrates real-time IoT sensor data, detailed building attributes, historical energy performance, and lifecycle assessment (LCA) information for PhD research on AI- and IoT-driven optimization of building retrofits.

## Dataset Statistics

- **Total Buildings**: 15 buildings across multiple types
- **Time Range**: 2020-01-01 to 2023-12-31 (4 years)
- **IoT Data Points**: 350,410 records per building (hourly data)
- **Building Types**: Residential, Office, Retail, Educational, Healthcare, Industrial
- **Construction Periods**: Pre-1970 to Post-2020
- **Data Categories**: 4 main categories with 20+ subcategories

## Dataset Structure

```
building_retrofit_dataset/
├── raw_data/                    # Raw generated data
│   ├── iot_sensors/            # IoT sensor data (40 files)
│   ├── building_attributes/    # Building characteristics (1 file)
│   ├── energy_performance/     # Energy data (3 files)
│   └── lca_data/              # LCA data (4 files)
├── processed_data/             # Integrated and processed data
│   ├── integrated/            # Merged datasets
│   ├── validated/             # Quality-checked data
│   └── analysis_ready/        # ML-ready datasets
├── scripts/                   # Data generation scripts
└── documentation/             # Dataset documentation
```

## Data Categories

### 1. IoT Sensor Data
- **Energy Consumption**: Whole building & end-use consumption
- **Environmental Parameters**: CO₂, TVOC, PM2.5, temperature, humidity
- **Weather Conditions**: Outdoor temperature, humidity, wind, solar radiation
- **Occupancy Patterns**: People count, activity levels, space utilization

### 2. Building Attributes & Fabric
- **Geometric Data**: Floor area, height, volume, rooftop area
- **Structural Data**: Construction year, materials, quality, style
- **Thermal Properties**: U-values, R-values, air tightness, thermal mass

### 3. Energy Performance
- **Historical Consumption**: 6 years of monthly/yearly data
- **Efficiency Ratings**: EU A-G ratings, ENERGY STAR scores, LEED certification
- **Retrofit Data**: Energy savings, costs, payback periods

### 4. Lifecycle Assessment (LCA)
- **Material EPDs**: Environmental Product Declarations for 11 materials
- **Construction Processes**: LCA data for 13 construction processes
- **Building LCA**: Complete building lifecycle impacts
- **Retrofit Scenarios**: 4 retrofit scenarios per building

## Key Features

### Realistic Data Generation
- **Temporal Patterns**: Seasonal and daily variations
- **Building-Specific**: Data tailored to building type and age
- **Correlated Variables**: Realistic relationships between parameters
- **Missing Data**: Simulated incomplete information for older buildings

### Multi-Scale Integration
- **Building Level**: Static attributes and characteristics
- **Time Series**: Hourly IoT sensor data
- **Historical**: Multi-year energy performance
- **Scenario-Based**: Multiple retrofit options

### Research-Ready Format
- **ML Features**: Engineered features for machine learning
- **Analysis Datasets**: Pre-configured for different analysis types
- **Validation**: Quality checks and completeness reports
- **Documentation**: Comprehensive metadata and usage guides

## Usage Recommendations

### For Energy Analysis
Use `energy_analysis.csv` - Contains building attributes with latest energy performance data.

### For Retrofit Optimization
Use `retrofit_analysis.csv` - Includes retrofit scenarios with cost-benefit analysis.

### For Machine Learning
Use `ml_ready.csv` - Features engineered for ML algorithms with target variables.

### For Time Series Analysis
Use individual IoT sensor files in `raw_data/iot_sensors/` for detailed temporal patterns.

## Data Quality

- **Completeness**: 95%+ data completeness across all categories
- **Validation**: Automated quality checks and outlier detection
- **Consistency**: Cross-validated relationships between datasets
- **Documentation**: Comprehensive metadata and data dictionaries

## Citation

If you use this dataset in your research, please cite:

```
Building Retrofit Dataset for AI- and IoT-driven Optimization
[Your Name], [Institution], 2024
```

## License

This dataset is provided for academic research purposes. Please refer to the license terms for commercial use restrictions.