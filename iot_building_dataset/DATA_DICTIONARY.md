# Data Dictionary

## IoT Building Dataset - Variable Definitions

This document provides detailed descriptions of all variables in the IoT Building Dataset.

---

## 1. Whole-Building Energy (`energy/whole_building_energy.csv`)

| Variable | Unit | Type | Range | Description |
|----------|------|------|-------|-------------|
| `timestamp` | - | datetime | 2024-01-01 to 2024-12-31 | ISO 8601 timestamp at 15-minute intervals |
| `electricity_kw` | kW | float | 50-330 | Total building electrical power consumption |
| `gas_kw` | kW | float | 10-250 | Natural gas consumption (thermal power equivalent) |
| `water_m3` | m³ | float | 0.1-1.2 | Water consumption per 15-minute interval |
| `district_heating_kw` | kW | float | 0-200 | District heating thermal power consumption |
| `district_cooling_kw` | kW | float | 0-150 | District cooling thermal power consumption |

**Notes:**
- Electricity includes all building loads (HVAC, lighting, plug loads, equipment)
- Gas is primarily for heating in winter months
- Water includes domestic use, cooling tower makeup, and landscaping

---

## 2. Sub-Metered Energy (`energy/sub_metered_energy.csv`)

| Variable | Unit | Type | Range | Description |
|----------|------|------|-------|-------------|
| `timestamp` | - | datetime | 2024-01-01 to 2024-12-31 | ISO 8601 timestamp |
| `hvac_electricity_kw` | kW | float | 40-180 | HVAC system electrical consumption (fans, pumps, chillers) |
| `lighting_electricity_kw` | kW | float | 10-80 | Lighting system consumption |
| `plug_loads_kw` | kW | float | 10-70 | Plug loads (computers, equipment, appliances) |
| `other_loads_kw` | kW | float | 20-100 | Other electrical loads (elevators, security, IT, etc.) |

**Notes:**
- Sum of sub-metered loads equals total electricity consumption
- HVAC typically 40-55% of total electricity
- Lighting varies with occupancy and daylight availability

---

## 3. Indoor Environmental Quality (`ieq/indoor_environmental_quality.csv`)

| Variable | Unit | Type | Range | Description |
|----------|------|------|-------|-------------|
| `timestamp` | - | datetime | 2024-01-01 to 2024-12-31 | ISO 8601 timestamp |
| `zone_id` | - | string | Zone_01 to Zone_12 | Building zone identifier |
| `temperature_c` | °C | float | 16-29 | Air temperature |
| `relative_humidity_pct` | % | float | 25-75 | Relative humidity |
| `co2_ppm` | ppm | float | 400-1200 | Carbon dioxide concentration |
| `pm25_ugm3` | μg/m³ | float | 0-35 | Particulate Matter 2.5 concentration |
| `pm10_ugm3` | μg/m³ | float | 0-65 | Particulate Matter 10 concentration |
| `tvoc_ppb` | ppb | float | 30-500 | Total Volatile Organic Compounds |
| `illuminance_lux` | lux | float | 50-800 | Illuminance (light level) |
| `noise_db` | dB | float | 35-60 | Noise level |

**Notes:**
- Temperature setpoints: 21°C heating, 24°C cooling (occupied hours)
- CO₂ baseline: 400 ppm (outdoor), increases with occupancy
- PM2.5 and PM10: WHO guidelines are 15 and 45 μg/m³ respectively
- TVOC increases during occupied hours due to human activity
- Illuminance: Natural + artificial light, ~300-500 lux typical for office work
- Noise: 35-45 dB quiet, 45-60 dB occupied periods

### Zone Information

| Zone ID | Floor | Location | Notes |
|---------|-------|----------|-------|
| Zone_01-06 | 1-3 | Lower floors | Higher solar gain, more windows |
| Zone_07-12 | 4-5 | Upper floors | Lower solar gain, warmer (heat rise) |

---

## 4. Occupancy & Usage Patterns (`occupancy/occupancy_usage.csv`)

| Variable | Unit | Type | Range | Description |
|----------|------|------|-------|-------------|
| `timestamp` | - | datetime | 2024-01-01 to 2024-12-31 | ISO 8601 timestamp |
| `occupant_count` | people | integer | 0-250 | Number of occupants in building |
| `desk_utilization_pct` | % | float | 0-100 | Percentage of desks occupied |
| `meeting_room_utilization_pct` | % | float | 0-100 | Percentage of meeting rooms booked/in-use |
| `windows_open_pct` | % | float | 0-100 | Percentage of operable windows that are open |
| `blinds_closed_pct` | % | float | 0-100 | Percentage of window blinds/shades closed |

**Notes:**
- Occupancy patterns: 8 AM - 6 PM weekdays (business hours)
- Peak occupancy: 2-3 PM at 60-90% capacity
- Weekend/night: ~5% occupancy (security/cleaning)
- Windows open more frequently when outdoor temperature is comfortable (18-24°C)
- Blinds closed more when solar irradiance is high (glare control)

### Occupancy Schedules

| Period | Typical Occupancy | Notes |
|--------|------------------|-------|
| Weekday 8-18h | 60-90% | Normal business hours |
| Weekday other | 5% | After hours |
| Weekend | 5% | Minimal staff |
| Monday | 85% of normal | Reduced (work from home) |
| Friday | 80% of normal | Reduced (work from home) |

---

## 5. Weather Station (`weather/weather_station.csv`)

| Variable | Unit | Type | Range | Description |
|----------|------|------|-------|-------------|
| `timestamp` | - | datetime | 2024-01-01 to 2024-12-31 | ISO 8601 timestamp |
| `solar_irradiance_wm2` | W/m² | float | 0-1000 | Solar irradiance on horizontal surface |
| `wind_speed_ms` | m/s | float | 0-15 | Wind speed |
| `wind_direction_deg` | degrees | float | 0-360 | Wind direction (0=North, 90=East, 180=South, 270=West) |
| `ambient_temperature_c` | °C | float | -5 to 35 | Outdoor air temperature |
| `relative_humidity_pct` | % | float | 20-95 | Outdoor relative humidity |
| `rainfall_mm` | mm | float | 0-15 | Precipitation per 15-minute interval |

**Notes:**
- Location: New York City (40.7128°N, 74.0060°W)
- Solar irradiance: 0 at night, up to 1000 W/m² peak midday
- Seasonal temperature patterns typical for humid subtropical/continental climate
- Rainfall events occur ~15% of time intervals

### Seasonal Climate Patterns

| Season | Temperature Range | Solar Irradiance | Precipitation |
|--------|------------------|------------------|---------------|
| Winter (Dec-Feb) | -5 to 10°C | 200-600 W/m² peak | Moderate (snow/rain) |
| Spring (Mar-May) | 5 to 20°C | 400-800 W/m² peak | High (rainy) |
| Summer (Jun-Aug) | 20 to 35°C | 600-1000 W/m² peak | Moderate (thunderstorms) |
| Fall (Sep-Nov) | 10 to 20°C | 300-700 W/m² peak | High (rainy) |

---

## 6. HVAC System Operation (`hvac_systems/hvac_operation.csv`)

| Variable | Unit | Type | Range | Description |
|----------|------|------|-------|-------------|
| `timestamp` | - | datetime | 2024-01-01 to 2024-12-31 | ISO 8601 timestamp |
| `zone_id` | - | string | Zone_01 to Zone_12 | Building zone identifier |
| `supply_air_temp_c` | °C | float | 14-22 | Supply air temperature from AHU |
| `return_air_temp_c` | °C | float | 20-30 | Return air temperature to AHU |
| `damper_position_pct` | % | float | 0-100 | Outside air damper position (0=closed, 100=fully open) |
| `fan_speed_pct` | % | float | 0-100 | Supply fan speed (VFD controlled) |
| `heating_valve_pct` | % | float | 0-100 | Heating coil valve position |
| `cooling_valve_pct` | % | float | 0-100 | Cooling coil valve position |
| `chiller_status` | - | binary | 0 or 1 | Chiller on/off status (0=off, 1=on) |
| `boiler_status` | - | binary | 0 or 1 | Boiler on/off status (0=off, 1=on) |
| `heating_setpoint_c` | °C | float | 18-21 | Zone heating setpoint |
| `cooling_setpoint_c` | °C | float | 24-26 | Zone cooling setpoint |

**Notes:**
- Supply air temp: Typically 16-18°C for cooling, 18-20°C for heating
- Damper position: Higher during occupied hours for fresh air (ventilation)
- Fan speed: 30% minimum, 60-80% during occupied hours
- Valves: Modulate to maintain setpoint temperatures
- Chiller: Operates primarily in summer (outdoor temp > 20°C)
- Boiler: Operates primarily in winter (outdoor temp < 15°C)
- Setpoints: Lower during unoccupied hours for energy savings

### HVAC Operating Modes

| Mode | Outdoor Temp | Chiller | Boiler | Typical Valves |
|------|-------------|---------|--------|----------------|
| Cooling | > 22°C | ON | OFF | Cooling valve open |
| Heating | < 15°C | OFF | ON | Heating valve open |
| Economizer | 15-22°C | OFF | OFF | High damper position |
| Free cooling | 10-18°C | OFF | OFF | 100% outside air |

### Setpoint Schedules

| Period | Heating Setpoint | Cooling Setpoint | Deadband |
|--------|-----------------|------------------|----------|
| Occupied (8-18h weekdays) | 21°C | 24°C | 3°C |
| Unoccupied | 18°C | 26°C | 8°C |
| Night setback | 16°C | 28°C | 12°C |

---

## Data Quality Indicators

### Completeness
- **100% complete** - No missing values in any dataset
- All timestamps present at exact 15-minute intervals
- No data gaps or anomalies

### Consistency
- All timestamps aligned across datasets (can be merged on timestamp)
- Physical relationships maintained (energy balance, psychrometrics)
- Realistic operational constraints applied

### Accuracy
- ±5% Gaussian noise for realistic sensor variation
- Physical bounds enforced (e.g., humidity 0-100%, temperature realistic)
- Correlations maintained (e.g., occupancy-CO2, temperature-HVAC)

---

## Units & Standards

### Energy
- All power values in kilowatts (kW)
- Energy = Power × Time, 15-min interval = 0.25 hours
- Annual energy (kWh) = sum(kW) × 0.25

### Temperature
- All temperatures in Celsius (°C)
- Convert to Fahrenheit: F = (C × 9/5) + 32
- Convert to Kelvin: K = C + 273.15

### Air Quality Standards

| Metric | Unit | Good | Moderate | Poor | Source |
|--------|------|------|----------|------|--------|
| CO₂ | ppm | < 800 | 800-1000 | > 1000 | ASHRAE 62.1 |
| PM2.5 | μg/m³ | < 12 | 12-35 | > 35 | EPA/WHO |
| PM10 | μg/m³ | < 45 | 45-150 | > 150 | WHO |
| TVOC | ppb | < 300 | 300-500 | > 500 | Various |

### Thermal Comfort Standards

| Standard | Temperature Range | Humidity Range | Notes |
|----------|------------------|----------------|-------|
| ASHRAE 55 | 20-26°C (68-79°F) | 30-60% | Summer/Winter |
| EN 15251 | 20-26°C | 30-70% | Category II |
| ISO 7730 | 20-24°C | 30-70% | PMV ±0.5 |

---

## Derived Variables & Calculations

### Energy Metrics
```python
# Total annual energy consumption (kWh)
annual_energy = df['electricity_kw'].sum() * 0.25

# Energy Use Intensity (kWh/m²/year)
EUI = annual_energy / building_area_m2

# Peak demand (kW)
peak_demand = df['electricity_kw'].max()

# Load factor
load_factor = df['electricity_kw'].mean() / df['electricity_kw'].max()
```

### Thermal Comfort (PMV/PPD)
```python
# Simplified PMV calculation (requires more parameters in practice)
# PMV ≈ (temperature - 23.5) / 3  # Very simplified
# PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)
```

### Occupancy-Based Metrics
```python
# CO2-derived occupancy (simplified)
occupancy_estimate = (co2_ppm - 400) / 6  # ~6 ppm per person

# Fresh air rate (L/s/person)
ventilation_rate = (outdoor_air_flow / occupant_count) * 1000
```

### HVAC Efficiency
```python
# Coefficient of Performance (simplified)
COP_cooling = cooling_output_kw / chiller_electricity_kw

# System Efficiency
hvac_efficiency = (heating_output + cooling_output) / total_hvac_electricity
```

---

## Variable Relationships & Correlations

### Strong Positive Correlations (r > 0.6)
- `electricity_kw` ↔ `occupant_count` (r ≈ 0.75)
- `co2_ppm` ↔ `occupant_count` (r ≈ 0.85)
- `cooling_valve_pct` ↔ `ambient_temperature_c` (r ≈ 0.80)
- `illuminance_lux` ↔ `solar_irradiance_wm2` (r ≈ 0.70)

### Strong Negative Correlations (r < -0.6)
- `heating_valve_pct` ↔ `ambient_temperature_c` (r ≈ -0.85)
- `temperature_c` ↔ `relative_humidity_pct` (r ≈ -0.65)
- `artificial_lighting` ↔ `solar_irradiance_wm2` (r ≈ -0.70)

### Moderate Correlations (0.3 < |r| < 0.6)
- `electricity_kw` ↔ `ambient_temperature_c` (r ≈ 0.38)
- `gas_kw` ↔ `ambient_temperature_c` (r ≈ -0.55)
- `hvac_electricity_kw` ↔ `ambient_temperature_c` (r ≈ 0.45)

---

## Temporal Patterns

### Daily Cycles
- **Occupancy**: Low 0-7h, rising 7-9h, high 9-17h, falling 17-19h
- **Energy**: Follows occupancy with thermal lag
- **Temperature**: Minimum 6-7h, maximum 15-16h
- **Solar**: 0 at night, peak ~13h local solar noon

### Weekly Cycles
- **Weekday**: High activity, normal patterns
- **Weekend**: Low occupancy (~5%), reduced energy
- **Monday**: 85% normal occupancy (WFH)
- **Friday**: 80% normal occupancy (WFH)

### Seasonal Cycles
- **Winter**: High heating, low cooling, lower solar
- **Spring**: Moderate loads, increasing solar
- **Summer**: High cooling, peak solar, low heating
- **Fall**: Decreasing loads and solar

---

## Usage Notes

### For Energy Modeling
- Use whole-building energy for baseline models
- Sub-metered data for detailed end-use analysis
- Weather data as independent variables
- Consider time lags (thermal mass ~2-4 hours)

### For Control Optimization (DRL)
- **State space**: IEQ metrics, occupancy, weather, time
- **Action space**: Setpoints, damper positions, valve positions
- **Reward function**: Energy cost + comfort penalty
- **Constraints**: Temperature bounds, CO2 limits, equipment capacity

### For Retrofit Analysis
- Establish pre-retrofit baseline from this data
- Simulate post-retrofit scenarios
- Calculate energy savings, comfort improvements
- Perform life-cycle cost analysis

### For Digital Twin
- Use for real-time state estimation
- Validate physics-based models
- Train surrogate/reduced-order models
- Anomaly detection and diagnostics

---

## Data Limitations

1. **Synthetic Data**: Generated with physics-based models, not actual measurements
2. **No Anomalies**: Equipment failures and faults not included
3. **Simplified Occupancy**: Uniform distribution across zones
4. **Ideal Control**: HVAC maintains setpoints perfectly
5. **Weather Model**: Simplified seasonal patterns, not actual observations
6. **No Extreme Events**: Heat waves, cold snaps simplified

---

## References & Standards

- **ASHRAE 55**: Thermal Environmental Conditions for Human Occupancy
- **ASHRAE 62.1**: Ventilation for Acceptable Indoor Air Quality
- **ASHRAE 90.1**: Energy Standard for Buildings
- **ISO 7730**: Ergonomics of the thermal environment
- **EN 15251**: Indoor environmental input parameters
- **EPA**: Air Quality Standards
- **WHO**: Air Quality Guidelines

---

*Last Updated: October 2024*  
*Dataset Version: 1.0*
