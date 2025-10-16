# IoT Building Digital Twin Dataset - Executive Summary

## Overview

A comprehensive, one-year IoT sensor dataset for building performance analysis, digital twin development, and deep reinforcement learning applications in building retrofit optimization.

**Status: ✅ Complete and Ready for Use**

---

## Quick Facts

| Property | Value |
|----------|-------|
| **Dataset Period** | January 1, 2024 - December 31, 2024 (365 days) |
| **Sampling Rate** | 15 minutes |
| **Total Records** | 981,120 across all data streams |
| **Dataset Size** | ~100 MB (CSV format) |
| **Building Type** | Commercial Office Building |
| **Location** | New York City, USA |
| **Building Area** | 5,000 m² (53,820 sq ft) |
| **Zones** | 12 HVAC zones |
| **Max Occupancy** | 250 people |

---

## Data Streams (6 Categories, 50+ Metrics)

### 1. ⚡ Energy Consumption
- **Whole-Building**: Electricity, Gas, Water, District Heating/Cooling
- **Sub-Metered**: HVAC, Lighting, Plug Loads, Other
- **Records**: 70,080 (35,040 each file)
- **Files**: `energy/whole_building_energy.csv`, `energy/sub_metered_energy.csv`

### 2. 🌡️ Indoor Environmental Quality (IEQ)
- **Thermal**: Temperature, Humidity (12 zones)
- **Air Quality**: CO₂, PM2.5, PM10, TVOCs (12 zones)
- **Comfort**: Illuminance, Noise levels (12 zones)
- **Records**: 420,480 (35,040 × 12 zones)
- **File**: `ieq/indoor_environmental_quality.csv`

### 3. 👥 Occupancy & Usage
- **People Count**: Real-time occupancy (0-250)
- **Space Utilization**: Desk and meeting room usage
- **Passive Systems**: Window and blind operation
- **Records**: 35,040
- **File**: `occupancy/occupancy_usage.csv`

### 4. 🌤️ Weather Station
- **Solar**: Irradiance (0-1000 W/m²)
- **Meteorological**: Temperature, humidity, wind, rainfall
- **Records**: 35,040
- **File**: `weather/weather_station.csv`

### 5. 🔧 HVAC Systems
- **Air Handling**: Supply/return temps, dampers, fan speeds (12 zones)
- **Plant**: Chiller/boiler status, valve positions (12 zones)
- **Setpoints**: Heating/cooling setpoints (12 zones)
- **Records**: 420,480 (35,040 × 12 zones)
- **File**: `hvac_systems/hvac_operation.csv`

### 6. 📋 Metadata & Configuration
- **Building Config**: `building_config.json`
- **Dataset Metadata**: `dataset_metadata.json`
- **Documentation**: README.md, DATA_DICTIONARY.md

---

## Key Features

### ✅ Data Quality
- **100% Complete**: No missing values
- **Consistent Timestamps**: Exact 15-minute intervals
- **Realistic Patterns**: Seasonal, daily, and weekly cycles
- **Physical Constraints**: All values within realistic bounds
- **Correlated Variables**: Physically meaningful relationships

### 📊 Temporal Patterns
- **Seasonal**: Full year captures winter/summer extremes
- **Daily**: Business hour patterns (8 AM - 6 PM weekdays)
- **Weekly**: Weekday/weekend differences, reduced Mon/Fri occupancy
- **Hourly**: Peak loads, occupancy, and environmental conditions

### 🔗 Data Relationships
- Energy ↔ Weather: Strong temperature-dependent HVAC loads
- IEQ ↔ Occupancy: CO₂ and comfort metrics track occupancy
- Solar ↔ Lighting: Daylight harvesting patterns
- Temperature ↔ Humidity: Inverse psychrometric relationships

---

## Research Applications

### 🏗️ Digital Twin Development
- Real-time building state representation
- Physics-based model validation
- Predictive maintenance and diagnostics
- Virtual retrofit scenario testing

### 🤖 Deep Reinforcement Learning
- **State Space**: IEQ metrics, occupancy, weather, time-of-day
- **Action Space**: HVAC setpoints, damper/valve positions, fan speeds
- **Reward Function**: Energy minimization + comfort constraints
- **Training**: 1 year = 35,040 episodes at 15-min resolution

### 🔄 Building Retrofit Optimization
- Pre-retrofit baseline establishment
- Multi-objective optimization (energy, comfort, cost, carbon)
- Life-cycle assessment integration
- Cost-benefit analysis

### 📈 Energy Analytics
- Consumption pattern analysis
- Peak demand forecasting
- Energy efficiency measure evaluation
- Utility cost optimization

### 🌡️ Indoor Environmental Quality
- Thermal comfort assessment (PMV/PPD)
- Air quality monitoring and prediction
- Occupant health and productivity analysis
- Ventilation effectiveness evaluation

---

## Dataset Structure

```
iot_building_dataset/
│
├── 📁 energy/
│   ├── whole_building_energy.csv       (2.8 MB, 35,040 records)
│   └── sub_metered_energy.csv          (3.1 MB, 35,040 records)
│
├── 📁 ieq/
│   └── indoor_environmental_quality.csv (46.5 MB, 420,480 records)
│
├── 📁 occupancy/
│   └── occupancy_usage.csv             (3.2 MB, 35,040 records)
│
├── 📁 weather/
│   └── weather_station.csv             (3.7 MB, 35,040 records)
│
├── 📁 hvac_systems/
│   └── hvac_operation.csv              (38.1 MB, 420,480 records)
│
├── 📁 analysis_plots/                  (Visualizations)
│   ├── energy_consumption.png
│   ├── ieq_conditions.png
│   ├── weather_conditions.png
│   ├── occupancy_patterns.png
│   └── correlation_matrix.png
│
├── 📄 building_config.json             (Building parameters)
├── 📄 dataset_metadata.json            (Complete metadata)
├── 📄 README.md                        (Comprehensive guide)
├── 📄 DATA_DICTIONARY.md               (Variable definitions)
├── 📄 DATASET_SUMMARY.md               (This file)
│
├── 🐍 generate_iot_dataset.py          (Data generation script)
├── 🐍 analyze_dataset.py               (Analysis script)
├── 🐍 quick_start_example.py           (Quick start guide)
└── 📋 requirements.txt                 (Python dependencies)
```

---

## Getting Started

### 1. Quick Start (5 minutes)
```bash
# Install dependencies
pip install -r requirements.txt

# Run quick start example
python quick_start_example.py
```

### 2. Comprehensive Analysis (10 minutes)
```bash
# Run full dataset analysis with visualizations
python analyze_dataset.py
```

### 3. Custom Analysis
```python
import pandas as pd

# Load data
energy = pd.read_csv('energy/whole_building_energy.csv', parse_dates=['timestamp'])
weather = pd.read_csv('weather/weather_station.csv', parse_dates=['timestamp'])
ieq = pd.read_csv('ieq/indoor_environmental_quality.csv', parse_dates=['timestamp'])

# Merge and analyze
df = energy.merge(weather, on='timestamp')
print(df['electricity_kw'].corr(df['ambient_temperature_c']))
```

---

## Key Statistics

### Energy Consumption
| Metric | Value |
|--------|-------|
| **Electricity** | 186 kW average, 94-330 kW range |
| **Gas** | 134 kW average (heating season) |
| **Peak Demand** | ~330 kW (summer afternoon) |
| **Annual Energy** | ~1,630 MWh electricity, ~1,173 MWh gas |
| **EUI** | ~326 kWh/m²/year (electricity) |

### Indoor Environment
| Metric | Value |
|--------|-------|
| **Temperature** | 22.3°C average, 16.5-29°C range |
| **Humidity** | 47% average, 25-75% range |
| **CO₂** | 594 ppm average, 400-1200 ppm range |
| **Illuminance** | 360 lux average, 50-800 lux range |

### Occupancy
| Metric | Value |
|--------|-------|
| **Average** | 38.5 people (15% of max) |
| **Peak** | 183 people (73% of max) |
| **Business Hours** | 8 AM - 6 PM weekdays |
| **Desk Utilization** | 30-80% during occupied hours |

### Weather
| Metric | Value |
|--------|-------|
| **Temperature** | 15°C average, -5 to 35°C range |
| **Solar Irradiance** | 316 W/m² average, 0-1000 W/m² range |
| **Annual Rainfall** | ~1,100 mm total |

---

## Key Correlations

| Variables | Correlation | Interpretation |
|-----------|------------|----------------|
| Electricity ↔ Occupancy | +0.75 | Strong: Energy follows occupancy |
| CO₂ ↔ Occupancy | +0.85 | Very Strong: Ventilation indicator |
| Electricity ↔ Outdoor Temp | +0.38 | Moderate: HVAC weather-dependent |
| Gas ↔ Outdoor Temp | -0.55 | Strong: Heating in cold weather |
| Cooling Valve ↔ Outdoor Temp | +0.80 | Very Strong: Cooling on hot days |
| Heating Valve ↔ Outdoor Temp | -0.85 | Very Strong: Heating on cold days |

---

## Validation & Quality Checks

### ✅ Completeness
- All 35,040 timestamps present (no gaps)
- All variables populated (no missing values)
- All zones represented equally

### ✅ Consistency
- Energy balance: sub-metered = total
- Psychrometric: temp-humidity relationships valid
- Physics: cause-effect relationships maintained

### ✅ Realism
- Values within equipment capacity limits
- Seasonal patterns match climate zone
- Occupancy patterns match office building type
- HVAC operation follows control logic

### ✅ Usability
- Standard CSV format (UTF-8)
- ISO 8601 timestamps
- Clear variable naming
- Comprehensive documentation

---

## Use Case Examples

### Example 1: Energy Baseline Model
```python
# Create energy model with weather and occupancy
from sklearn.ensemble import RandomForestRegressor

# Load and merge data
df = energy.merge(weather).merge(occupancy, on='timestamp')

# Features and target
X = df[['ambient_temperature_c', 'solar_irradiance_wm2', 'occupant_count', 'hour', 'dayofweek']]
y = df['electricity_kw']

# Train model
model = RandomForestRegressor()
model.fit(X, y)
```

### Example 2: DRL HVAC Control
```python
# State: [temp, humidity, co2, outdoor_temp, occupancy, hour]
# Action: [heating_setpoint, cooling_setpoint, damper_position]
# Reward: -energy_cost - comfort_penalty

state = [zone_temp, zone_humidity, zone_co2, outdoor_temp, occupancy, hour]
action = agent.select_action(state)
reward = calculate_reward(energy_cost, comfort_penalty)
```

### Example 3: Retrofit ROI Analysis
```python
# Compare pre/post retrofit energy
baseline_energy = df['electricity_kw'].sum() * 0.25  # kWh
post_retrofit_energy = baseline_energy * 0.75  # 25% savings

savings = (baseline_energy - post_retrofit_energy) * energy_rate
roi = savings / retrofit_cost
payback = retrofit_cost / savings
```

---

## Seasonal Highlights

### ❄️ Winter (Dec-Feb)
- High heating demand (gas 215 kW avg)
- Low cooling (district cooling 20 kW avg)
- Lower solar irradiance (200-600 W/m²)
- Electricity: 117 kW average

### 🌸 Spring (Mar-May)
- Moderate loads
- Increasing solar (400-800 W/m²)
- Free cooling opportunities
- Electricity: 206 kW average

### ☀️ Summer (Jun-Aug)
- Peak cooling demand
- Maximum solar (600-1000 W/m²)
- Low heating
- Electricity: 255 kW average (peak)

### 🍂 Fall (Sep-Nov)
- Decreasing loads
- Moderate solar (300-700 W/m²)
- Transition season
- Electricity: 165 kW average

---

## Technical Specifications

### Data Generation
- **Base Models**: Sinusoidal functions for seasonal/daily cycles
- **Occupancy**: Rule-based schedules (business hours)
- **Weather**: Physical solar geometry + seasonal temperature model
- **HVAC**: Control-based operation (PID-like setpoint tracking)
- **Noise**: Gaussian (5% of mean) for sensor realism
- **Correlations**: Physics-based variable relationships

### Limitations
1. Synthetic data (not actual measurements)
2. No equipment failures or anomalies
3. Simplified occupancy distribution
4. Ideal HVAC control (perfect setpoint tracking)
5. Modeled weather (not actual observations)
6. No extreme weather events

### Future Enhancements
- Equipment fault injection
- More granular occupancy (by zone)
- Actual weather data integration
- Higher frequency sampling (5-min or 1-min)
- Additional end-uses (EV charging, renewables)

---

## Citation

```bibtex
@dataset{iot_building_dataset_2024,
  title={IoT Building Digital Twin Dataset},
  author={Generated for Building Retrofit DRL Research},
  year={2024},
  publisher={Research Dataset},
  version={1.0},
  description={One-year IoT sensor data for building digital twin and 
               deep reinforcement learning applications},
  url={./iot_building_dataset/}
}
```

---

## Support & Resources

### 📚 Documentation
- `README.md` - Comprehensive user guide
- `DATA_DICTIONARY.md` - All variable definitions
- `dataset_metadata.json` - Complete metadata
- `building_config.json` - Building parameters

### 🔧 Tools & Scripts
- `generate_iot_dataset.py` - Data generation source code
- `analyze_dataset.py` - Comprehensive analysis tool
- `quick_start_example.py` - Getting started guide
- `requirements.txt` - Python dependencies

### 📊 Visualizations
- Energy consumption patterns
- IEQ conditions by zone
- Weather conditions
- Occupancy patterns
- Correlation matrices

### 📖 Additional References
- ASHRAE Standards (55, 62.1, 90.1)
- ISO 7730 (Thermal Environment)
- EPA Air Quality Standards
- WHO Air Quality Guidelines

---

## Performance Benchmarks

### Dataset Loading Performance
| Operation | Time (approx) |
|-----------|--------------|
| Load single CSV | < 1 second |
| Load all datasets | < 5 seconds |
| Merge 3 datasets | < 2 seconds |
| Generate visualizations | < 30 seconds |
| Full analysis | < 1 minute |

### Computational Requirements
| Requirement | Specification |
|-------------|--------------|
| RAM | 2 GB minimum, 4 GB recommended |
| Storage | 150 MB (data + plots) |
| Python | 3.8+ |
| Pandas | 2.0+ |
| Processing | Single core sufficient |

---

## License & Usage Terms

This dataset is generated for **research and educational purposes**. 

- ✅ Free to use for academic research
- ✅ Can be modified and extended
- ✅ Can be used in publications (with citation)
- ✅ Can be shared with attribution
- ⚠️ Synthetic data - not actual building measurements
- ⚠️ No warranty or guarantee of accuracy

---

## Acknowledgments

This dataset was generated to support research in:
- **Dynamic Digital Twin Frameworks**
- **Multi-Objective Building Retrofit Optimization**
- **Real-Time IoT Integration**
- **Life-Cycle Assessment**
- **Deep Reinforcement Learning for Building Control**

---

## Contact & Contributions

For questions, issues, or contributions:
- Check documentation files first
- Review `dataset_metadata.json` for specifications
- Examine `generate_iot_dataset.py` for methodology
- See `analyze_dataset.py` for usage examples

---

**Dataset Version:** 1.0  
**Generated:** October 2024  
**Status:** ✅ Production Ready  
**Total Size:** ~100 MB  
**Total Records:** 981,120

---

*This dataset represents a complete, realistic, and comprehensive IoT data foundation for advanced building performance research, digital twin development, and AI-driven optimization applications.*
