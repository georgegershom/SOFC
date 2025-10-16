# IoT Building Digital Twin Dataset

## Overview

This comprehensive dataset contains one full year (365 days) of high-frequency, time-series IoT sensor data from a commercial office building. The dataset is designed for building performance optimization, digital twin development, and deep reinforcement learning applications in the context of building retrofit optimization.

**Dataset Period:** January 1, 2024 - December 31, 2024  
**Sampling Interval:** 15 minutes  
**Total Records:** ~981,120 across all data streams  
**Dataset Size:** ~100 MB

## Research Application

This dataset supports the research topic:
> **"A Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization: Integrating Real-Time IoT, Life-Cycle Assessment, and Deep Reinforcement Learning"**

The data captures seasonal variations, occupancy patterns, weather conditions, and building system operations essential for:
- Energy consumption analysis and prediction
- Indoor environmental quality optimization
- HVAC system control optimization
- Occupancy-driven building operations
- Multi-objective retrofit scenario evaluation
- Deep reinforcement learning model training

## Building Configuration

- **Type:** Commercial Office Building
- **Total Area:** 5,000 m²
- **Number of Floors:** 5
- **Number of Zones:** 12 (multi-zone HVAC)
- **Maximum Occupancy:** 250 people
- **Location:** New York City, USA (40.7128°N, 74.0060°W)
- **Year Built:** 1995

## Dataset Structure

```
iot_building_dataset/
├── energy/                          # Energy consumption data
│   ├── whole_building_energy.csv   # Whole-building meters (35,040 records)
│   └── sub_metered_energy.csv      # Circuit-level sub-metering (35,040 records)
├── ieq/                            # Indoor Environmental Quality
│   └── indoor_environmental_quality.csv  # Multi-zone IEQ (420,480 records)
├── occupancy/                      # Occupancy and usage patterns
│   └── occupancy_usage.csv         # People count & space utilization (35,040 records)
├── weather/                        # External conditions
│   └── weather_station.csv         # On-site weather station (35,040 records)
├── hvac_systems/                   # HVAC operation data
│   └── hvac_operation.csv          # Multi-zone HVAC systems (420,480 records)
├── building_config.json            # Building configuration metadata
├── dataset_metadata.json           # Complete dataset metadata
└── README.md                       # This file
```

## Data Streams

### 1. Energy Consumption (The "Utility Backbone")

#### Whole-Building Energy (`energy/whole_building_energy.csv`)
- **Records:** 35,040 (one year at 15-min intervals)
- **Metrics:**
  - `electricity_kw`: Total building electricity consumption (kW)
  - `gas_kw`: Natural gas consumption (kW thermal)
  - `water_m3`: Water consumption (cubic meters per interval)
  - `district_heating_kw`: District heating consumption (kW)
  - `district_cooling_kw`: District cooling consumption (kW)

#### Sub-Metered Energy (`energy/sub_metered_energy.csv`)
- **Records:** 35,040
- **Metrics:**
  - `hvac_electricity_kw`: HVAC system electricity (typically 40-55% of total)
  - `lighting_electricity_kw`: Lighting loads (25-30% of total during occupied hours)
  - `plug_loads_kw`: Plug loads and equipment (20-25% of total)
  - `other_loads_kw`: Other electrical loads (balance)

### 2. Indoor Environmental Quality (`ieq/indoor_environmental_quality.csv`)

- **Records:** 420,480 (35,040 per zone × 12 zones)
- **Metrics:**
  - `zone_id`: Zone identifier (Zone_01 through Zone_12)
  - `temperature_c`: Air temperature (°C)
  - `relative_humidity_pct`: Relative humidity (%)
  - `co2_ppm`: CO₂ concentration (ppm) - key ventilation indicator
  - `pm25_ugm3`: Particulate Matter 2.5 (μg/m³)
  - `pm10_ugm3`: Particulate Matter 10 (μg/m³)
  - `tvoc_ppb`: Total Volatile Organic Compounds (ppb)
  - `illuminance_lux`: Light levels (lux)
  - `noise_db`: Noise levels (dB)

**Patterns:**
- Temperature: 18-26°C with seasonal and occupancy variations
- CO₂: 400-1200 ppm (400 baseline, up to 1200 during high occupancy)
- Humidity: 25-75% inversely related to temperature
- Illuminance: Natural + artificial light (50-800 lux)

### 3. Occupancy & Usage Patterns (`occupancy/occupancy_usage.csv`)

- **Records:** 35,040
- **Metrics:**
  - `occupant_count`: Number of people in building (0-250)
  - `desk_utilization_pct`: Desk/workstation utilization (%)
  - `meeting_room_utilization_pct`: Meeting room booking rate (%)
  - `windows_open_pct`: Percentage of operable windows open
  - `blinds_closed_pct`: Percentage of blinds/shades closed

**Patterns:**
- Business hours: 8 AM - 6 PM weekdays
- Peak occupancy: 2-3 PM (60-90% of max capacity)
- Reduced occupancy: Mondays (85%), Fridays (80%)
- Weekend/night: 5% (security/cleaning)

### 4. External Weather Conditions (`weather/weather_station.csv`)

- **Records:** 35,040
- **Metrics:**
  - `solar_irradiance_wm2`: Solar irradiance (W/m²) - 0-1000
  - `wind_speed_ms`: Wind speed (m/s)
  - `wind_direction_deg`: Wind direction (0-360°)
  - `ambient_temperature_c`: Outdoor temperature (°C)
  - `relative_humidity_pct`: Outdoor humidity (%)
  - `rainfall_mm`: Precipitation per interval (mm)

**Seasonal Patterns:**
- Winter (Dec-Feb): -5°C to 10°C
- Spring (Mar-May): 5°C to 20°C
- Summer (Jun-Aug): 20°C to 35°C
- Fall (Sep-Nov): 10°C to 20°C

### 5. HVAC System Operation (`hvac_systems/hvac_operation.csv`)

- **Records:** 420,480 (35,040 per zone × 12 zones)
- **Metrics:**
  - `zone_id`: Zone identifier
  - `supply_air_temp_c`: Supply air temperature (°C)
  - `return_air_temp_c`: Return air temperature (°C)
  - `damper_position_pct`: Damper position (0-100%)
  - `fan_speed_pct`: Fan speed (0-100%)
  - `heating_valve_pct`: Heating valve position (0-100%)
  - `cooling_valve_pct`: Cooling valve position (0-100%)
  - `chiller_status`: Chiller on/off (0/1)
  - `boiler_status`: Boiler on/off (0/1)
  - `heating_setpoint_c`: Heating setpoint temperature (°C)
  - `cooling_setpoint_c`: Cooling setpoint temperature (°C)

**Control Logic:**
- Occupied setpoints: 21°C heating, 24°C cooling
- Unoccupied setpoints: 18°C heating, 26°C cooling
- Supply air temp: 16-20°C (varies with load)
- Fan speed: 30% minimum, 60-80% during occupied hours

## Data Characteristics

### Temporal Patterns

1. **Seasonal Variations** (Annual Cycle)
   - Energy consumption varies by ±40% seasonally
   - HVAC loads peak in summer (cooling) and winter (heating)
   - Solar irradiance varies 0-1000 W/m² with seasonal envelope

2. **Daily Cycles** (24-hour)
   - Occupancy-driven patterns (weekday vs. weekend)
   - Temperature swings: ±5°C daily
   - Lighting follows daylight and occupancy

3. **Weekly Patterns**
   - Weekday operations: high activity
   - Weekend: minimal occupancy (~5%)
   - Monday/Friday: reduced occupancy

### Correlations & Dependencies

- **Energy-Weather:** Strong correlation between outdoor temperature and HVAC energy
- **IEQ-Occupancy:** CO₂, temperature, and noise correlate with occupant count
- **Solar-Lighting:** Artificial lighting inversely proportional to solar irradiance
- **Temperature-Humidity:** Inverse relationship in both indoor and outdoor
- **HVAC-Setpoints:** System operation follows setpoint schedules

### Data Quality

- **Completeness:** 100% - no missing values
- **Sampling Rate:** Consistent 15-minute intervals
- **Noise Level:** ~5% Gaussian noise added for realism
- **Outliers:** Physically realistic bounds applied to all sensors
- **Anomalies:** Natural variations and operational patterns (no synthetic faults injected)

## Use Cases

### 1. Energy Optimization
- Baseline energy consumption modeling
- Peak demand analysis and reduction
- Load forecasting and prediction
- Energy efficiency measure evaluation

### 2. Digital Twin Development
- Real-time building state representation
- Predictive maintenance scheduling
- System performance benchmarking
- Retrofit scenario simulation

### 3. Deep Reinforcement Learning
- HVAC control optimization
- Multi-objective reward functions (comfort + energy)
- State representation: IEQ + occupancy + weather
- Action space: Setpoints, damper positions, fan speeds

### 4. Indoor Environmental Quality
- Thermal comfort analysis (PMV, PPD)
- Air quality monitoring and prediction
- Ventilation effectiveness evaluation
- Occupant satisfaction modeling

### 5. Building Retrofit Analysis
- Pre-retrofit baseline establishment
- Post-retrofit performance comparison
- Life-cycle cost-benefit analysis
- Multi-objective optimization (energy, comfort, cost, carbon)

## Getting Started

### Quick Start with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load whole-building energy data
energy = pd.read_csv('energy/whole_building_energy.csv', parse_dates=['timestamp'])

# Basic analysis
print(energy.describe())

# Plot daily energy profile
energy.set_index('timestamp')['electricity_kw'].resample('D').mean().plot()
plt.title('Daily Average Electricity Consumption')
plt.ylabel('kW')
plt.show()
```

### Loading Multi-Zone Data

```python
# Load IEQ data for all zones
ieq = pd.read_csv('ieq/indoor_environmental_quality.csv', parse_dates=['timestamp'])

# Analyze specific zone
zone_5 = ieq[ieq['zone_id'] == 'Zone_05']
print(zone_5.groupby(zone_5['timestamp'].dt.hour)['temperature_c'].mean())
```

### Merging Data Streams

```python
# Combine energy, weather, and occupancy
energy = pd.read_csv('energy/whole_building_energy.csv', parse_dates=['timestamp'])
weather = pd.read_csv('weather/weather_station.csv', parse_dates=['timestamp'])
occupancy = pd.read_csv('occupancy/occupancy_usage.csv', parse_dates=['timestamp'])

# Merge on timestamp
df = energy.merge(weather, on='timestamp').merge(occupancy, on='timestamp')
print(df.head())
```

## Data Analysis Examples

### 1. Seasonal Energy Patterns
```python
energy = pd.read_csv('energy/whole_building_energy.csv', parse_dates=['timestamp'])
energy['month'] = energy['timestamp'].dt.month
monthly_avg = energy.groupby('month')['electricity_kw'].mean()
monthly_avg.plot(kind='bar', title='Average Electricity by Month')
```

### 2. Occupancy-Energy Correlation
```python
import pandas as pd
import numpy as np

energy = pd.read_csv('energy/whole_building_energy.csv', parse_dates=['timestamp'])
occupancy = pd.read_csv('occupancy/occupancy_usage.csv', parse_dates=['timestamp'])

df = energy.merge(occupancy, on='timestamp')
correlation = df[['electricity_kw', 'occupant_count']].corr()
print(f"Energy-Occupancy Correlation: {correlation.iloc[0,1]:.3f}")
```

### 3. Temperature-Dependent HVAC Operation
```python
weather = pd.read_csv('weather/weather_station.csv', parse_dates=['timestamp'])
energy = pd.read_csv('energy/sub_metered_energy.csv', parse_dates=['timestamp'])

df = weather.merge(energy, on='timestamp')
df.plot.scatter(x='ambient_temperature_c', y='hvac_electricity_kw', alpha=0.3)
plt.title('HVAC Energy vs. Outdoor Temperature')
```

## Data Format Specifications

All CSV files follow these conventions:
- **Timestamp:** ISO 8601 format (YYYY-MM-DD HH:MM:SS)
- **Numeric precision:** 2 decimal places for most metrics
- **Missing values:** None (100% complete dataset)
- **Delimiter:** Comma (,)
- **Encoding:** UTF-8
- **Headers:** First row contains column names

## Metadata Files

### `building_config.json`
Contains building physical and operational parameters:
- Building geometry and area
- Location coordinates
- Number of zones and floors
- Maximum occupancy

### `dataset_metadata.json`
Complete dataset documentation including:
- Generation parameters
- Data stream descriptions
- File locations and record counts
- Metric definitions

## Citation

If you use this dataset in your research, please cite:

```
IoT Building Digital Twin Dataset v1.0
Generated for: "A Dynamic Digital Twin Framework for Multi-Objective Building Retrofit 
Optimization: Integrating Real-Time IoT, Life-Cycle Assessment, and Deep Reinforcement Learning"
Year: 2024
```

## License & Usage

This dataset is generated for research and educational purposes. It represents realistic building operations but is synthetically created with physically-based models and stochastic patterns.

## Technical Notes

### Data Generation Methodology
- **Base patterns:** Sinusoidal functions for seasonal/daily cycles
- **Occupancy:** Rule-based weekday/weekend schedules
- **Weather:** Seasonal temperature model + solar geometry
- **Noise:** Gaussian noise (5% of signal mean) for realism
- **Correlations:** Physics-based relationships between variables

### Assumptions & Limitations
- Assumes typical office building operations in NYC climate zone
- No equipment failures or anomalies included
- Uniform occupancy distribution across zones (simplified)
- Weather data modeled, not actual observations
- HVAC system assumed to maintain setpoints perfectly

## Support & Questions

For questions about the dataset or suggestions for improvements:
- Check `dataset_metadata.json` for complete specifications
- Review `generate_iot_dataset.py` for generation logic
- Examine sample data files for format examples

## Version History

**v1.0** (2024)
- Initial release
- 365-day dataset at 15-minute resolution
- 6 data streams with 50+ metrics
- Multi-zone IEQ and HVAC data
- Comprehensive metadata and documentation

---

**Generated:** October 2024  
**Dataset Version:** 1.0  
**Total Size:** ~100 MB (CSV format)  
**Records:** 981,120 across all streams
