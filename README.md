# IoT & Real-Time Monitoring Dataset for Building Digital Twin

## Overview

This repository contains a comprehensive IoT sensor dataset generated for the **Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization**. The dataset simulates one year of high-frequency building sensor data with realistic correlations, seasonal variations, and operational patterns.

## Dataset Characteristics

- **Time Period**: January 1, 2023 - December 31, 2023
- **Sampling Frequency**: 15 minutes
- **Total Data Points**: 34,945 records
- **Building Area**: 50,000 sq ft (8 floors, 32 zones)
- **Location**: Eastern Time Zone (US)

## Data Categories

### 1. Energy Consumption Data
- **Whole-Building**: Total electricity consumption (kWh)
- **Sub-Metered Systems**:
  - HVAC electricity consumption
  - Lighting electricity consumption
  - Plug loads (computers, equipment)
  - Water heating
  - Base load

### 2. Indoor Environmental Quality (IEQ)
- **Thermal**: Zone-specific air temperatures (32 zones)
- **Air Quality**: 
  - CO₂ levels (ppm)
  - PM2.5 particulate matter (μg/m³)
  - Total Volatile Organic Compounds (TVOCs) (ppb)
- **Lighting**: Illuminance levels (lux)
- **Acoustics**: Noise levels (dB)
- **Humidity**: Indoor relative humidity (%)

### 3. Occupancy & Usage Patterns
- **Occupant Count**: People counters and Wi-Fi-derived occupancy
- **Space Utilization**: Percentage of space being used
- **Window Operation**: Smart window sensors (open/closed)
- **Blind Operation**: Smart blind sensors (open/closed)

### 4. External Weather Conditions
- **Temperature**: Outdoor air temperature (°F)
- **Humidity**: Outdoor relative humidity (%)
- **Solar Irradiance**: Solar radiation (W/m²)
- **Wind**: Speed (m/s) and direction (degrees)
- **Precipitation**: Rainfall (mm)

### 5. Building System Operation
- **HVAC Systems**:
  - Supply and return air temperatures
  - Damper positions (%)
  - Fan speeds (%)
  - Valve positions (%)
  - Chiller and boiler status
- **Setpoints**: Heating and cooling setpoints (°F)

## File Formats

The dataset is provided in multiple formats for different use cases:

- **CSV**: `iot_building_dataset.csv` - Universal format for data analysis
- **Parquet**: `iot_building_dataset.parquet` - Efficient binary format for large datasets
- **Excel**: `iot_building_dataset.xlsx` - Human-readable format with separate sheets:
  - Complete_Dataset: All data in one sheet
  - Weather_Data: External weather conditions
  - Energy_Data: Energy consumption metrics
  - IEQ_Data: Indoor environmental quality
  - Occupancy_Data: Occupancy and usage patterns
  - HVAC_Data: Building system operation

## Data Quality Features

### Realistic Correlations
- HVAC energy consumption correlates with outdoor temperature
- CO₂ levels correlate with occupancy
- Illuminance correlates with solar irradiance
- Indoor temperature responds to HVAC operation

### Seasonal Variations
- Temperature follows realistic seasonal patterns
- Energy consumption peaks during extreme weather
- Occupancy patterns vary by season
- Solar irradiance follows seasonal cycles

### Operational Patterns
- Business hours occupancy patterns (8 AM - 6 PM)
- Weekend vs. weekday differences
- HVAC operation responds to temperature setpoints
- Window/blind operation based on comfort conditions

## Usage for AI Model Training

This dataset is specifically designed for training AI models in building retrofit optimization:

### Deep Reinforcement Learning
- **State Space**: All sensor readings as state variables
- **Action Space**: HVAC setpoints, damper positions, window/blind control
- **Reward Function**: Energy efficiency, comfort, air quality

### Multi-Objective Optimization
- **Objectives**: Energy consumption, occupant comfort, air quality
- **Constraints**: Temperature ranges, humidity limits, equipment capacity

### Time Series Analysis
- **Forecasting**: Energy consumption, temperature, occupancy
- **Anomaly Detection**: Equipment failures, unusual patterns
- **Pattern Recognition**: Occupancy patterns, energy usage trends

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Generate New Dataset
```python
from iot_dataset_generator import IoTDatasetGenerator

# Create generator
generator = IoTDatasetGenerator(
    start_date='2023-01-01',
    end_date='2023-12-31',
    timezone='US/Eastern',
    building_area=50000
)

# Generate dataset
dataset = generator.generate_complete_dataset()

# Save in multiple formats
output_dir = generator.save_dataset(dataset)
```

### Validate Dataset Quality
```python
from data_validation import IoTDataValidator

# Validate dataset
validator = IoTDataValidator('iot_dataset/iot_building_dataset.csv')
report = validator.generate_quality_report()
```

## Dataset Statistics

### Energy Consumption
- **Average Daily Consumption**: ~2,500 kWh
- **Peak Consumption**: ~4,000 kWh (summer cooling)
- **Minimum Consumption**: ~1,500 kWh (mild weather)

### Occupancy Patterns
- **Peak Occupancy**: ~150 people (business hours)
- **Off-Hours**: ~20 people (evenings/weekends)
- **Weekend**: ~10 people

### Environmental Conditions
- **Indoor Temperature Range**: 68-76°F
- **CO₂ Levels**: 400-1,500 ppm
- **Illuminance**: 0-1,500 lux
- **Noise Levels**: 30-80 dB

## Visualization

The dataset includes comprehensive visualizations:

- **Energy Analysis**: Consumption patterns, correlations, seasonal trends
- **IEQ Analysis**: Temperature, air quality, lighting, acoustics
- **Occupancy & HVAC**: Usage patterns and system operation
- **Correlation Heatmaps**: Sensor data relationships

## Applications

### Research Applications
- Building energy modeling
- Occupant behavior analysis
- HVAC system optimization
- Indoor air quality studies
- Smart building control algorithms

### Industry Applications
- Building management systems
- Energy efficiency consulting
- HVAC system design
- Smart building automation
- Retrofit planning and optimization

## Data Validation

The dataset includes comprehensive quality checks:

- **Completeness**: Missing data analysis
- **Range Validation**: Sensor value bounds checking
- **Temporal Consistency**: Time series integrity
- **Correlation Validation**: Expected relationships
- **Seasonal Pattern Validation**: Realistic seasonal variations

## Citation

If you use this dataset in your research, please cite:

```
IoT & Real-Time Monitoring Dataset for Building Digital Twin Framework
Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization
Generated: 2024
```

## License

This dataset is provided for research and educational purposes. Please ensure appropriate attribution when using in publications or commercial applications.

## Contact

For questions about the dataset or to request modifications, please refer to the project documentation or create an issue in the repository.

---

**Note**: This is a synthetic dataset generated for research purposes. While it includes realistic patterns and correlations, it should not be used as a substitute for real building sensor data in production systems.