# Building DNA Dataset - Data Dictionary

## Overview

This data dictionary provides comprehensive definitions for all fields, data types, units, and valid ranges used throughout the Building DNA Dataset. The dataset follows SI (metric) units as the primary standard, with imperial conversions provided where applicable.

## General Conventions

### Data Types
- **String**: Text values, UTF-8 encoded
- **Number**: Numeric values (integer or floating-point)
- **Boolean**: True/false values
- **Array**: Ordered list of values
- **Object**: Nested JSON object with key-value pairs
- **Date**: ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)

### Units and Conversions
- **Length**: meters (m), millimeters (mm)
- **Area**: square meters (m²)
- **Volume**: cubic meters (m³)
- **Temperature**: Celsius (°C)
- **Pressure**: Pascals (Pa), kilopascals (kPa)
- **Energy**: kilowatt-hours (kWh), joules (J)
- **Power**: watts (W), kilowatts (kW)
- **Thermal Resistance**: m²K/W (R-value)
- **Thermal Transmittance**: W/m²K (U-value)

## File-by-File Field Definitions

## 1. building_metadata.json

### Root Level Fields

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `building_id` | String | - | Unique building identifier | Alphanumeric, 3-50 chars |
| `project_name` | String | - | Descriptive project name | 1-200 chars |
| `building_type` | String | - | Primary building use type | Predefined list |
| `location` | Object | - | Geographic and climate information | - |
| `building_characteristics` | Object | - | Physical building properties | - |
| `dataset_metadata` | Object | - | Dataset version and quality info | - |

### location Object

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `address` | String | - | Street address | 1-200 chars |
| `coordinates.latitude` | Number | degrees | Latitude coordinate | -90 to 90 |
| `coordinates.longitude` | Number | degrees | Longitude coordinate | -180 to 180 |
| `climate_zone` | String | - | ASHRAE climate zone | 1A-8B |
| `elevation_m` | Number | meters | Elevation above sea level | -500 to 5000 |
| `orientation` | Number | degrees | Building orientation from north | 0 to 360 |

### building_characteristics Object

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `year_built` | Number | year | Original construction year | 1800-2030 |
| `year_renovated` | Number | year | Last major renovation year | 1800-2030 |
| `total_floor_area_m2` | Number | m² | Total floor area | 100-1000000 |
| `conditioned_floor_area_m2` | Number | m² | Conditioned (heated/cooled) area | 50-1000000 |
| `number_of_floors` | Number | - | Number of floors | 1-200 |
| `building_height_m` | Number | meters | Total building height | 3-1000 |
| `occupancy_type` | String | - | Primary occupancy classification | IBC classifications |
| `max_occupancy` | Number | people | Maximum occupant load | 1-50000 |
| `typical_occupancy` | Number | people | Typical occupant count | 1-50000 |

## 2. geometric_data/bim_geometry.json

### bim_model_info Object

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `software` | String | - | BIM software used | - |
| `model_version` | String | - | Model version number | - |
| `level_of_detail` | String | - | LOD classification | LOD_100 to LOD_500 |
| `coordinate_system` | String | - | Coordinate reference system | - |
| `units` | String | - | Model units | meters, feet, etc. |
| `accuracy` | String | - | Model accuracy specification | - |

### Geometric Elements

#### Vertices
| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `x` | Number | meters | X coordinate | -10000 to 10000 |
| `y` | Number | meters | Y coordinate | -10000 to 10000 |
| `z` | Number | meters | Z coordinate (elevation) | -100 to 1000 |

#### Wall Elements
| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `wall_id` | String | - | Unique wall identifier | 3-20 chars |
| `wall_type` | String | - | Wall construction type | Predefined types |
| `area_m2` | Number | m² | Wall surface area | 1-10000 |
| `height_m` | Number | meters | Wall height | 2-20 |
| `length_m` | Number | meters | Wall length | 1-500 |
| `thickness_m` | Number | meters | Wall thickness | 0.05-2.0 |
| `orientation_degrees` | Number | degrees | Wall orientation from north | 0-360 |

## 3. construction_materials/wall_assemblies.json

### Wall Assembly Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `assembly_id` | String | - | Unique assembly identifier | 3-20 chars |
| `assembly_name` | String | - | Descriptive assembly name | 1-100 chars |
| `wall_type` | String | - | Wall construction category | Predefined types |
| `total_thickness_m` | Number | meters | Total assembly thickness | 0.05-2.0 |
| `total_r_value_m2k_w` | Number | m²K/W | Total thermal resistance | 0.5-20.0 |
| `total_u_value_w_m2k` | Number | W/m²K | Total thermal transmittance | 0.05-2.0 |
| `thermal_mass_kg_m2` | Number | kg/m² | Thermal mass per unit area | 10-2000 |
| `fire_rating_hours` | Number | hours | Fire resistance rating | 0-4 |
| `acoustic_rating_stc` | Number | - | Sound transmission class | 25-65 |

### Layer Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `layer_id` | String | - | Unique layer identifier | 3-20 chars |
| `layer_name` | String | - | Layer description | 1-100 chars |
| `material` | String | - | Material type | Predefined materials |
| `thickness_m` | Number | meters | Layer thickness | 0.0001-1.0 |
| `thermal_conductivity_w_mk` | Number | W/mK | Thermal conductivity | 0.01-500 |
| `density_kg_m3` | Number | kg/m³ | Material density | 1-10000 |
| `specific_heat_j_kgk` | Number | J/kgK | Specific heat capacity | 500-5000 |
| `r_value_m2k_w` | Number | m²K/W | Layer thermal resistance | 0.001-10.0 |
| `u_value_w_m2k` | Number | W/m²K | Layer thermal transmittance | 0.1-1000 |
| `position_from_exterior` | Number | - | Layer position (1=exterior) | 1-20 |

### Window and Door Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `opening_id` | String | - | Unique opening identifier | 3-20 chars |
| `opening_type` | String | - | Window or Door | Window, Door |
| `manufacturer` | String | - | Manufacturer name | 1-100 chars |
| `model` | String | - | Product model number | 1-50 chars |
| `installation_year` | Number | year | Installation year | 1900-2030 |
| `u_value_w_m2k` | Number | W/m²K | Overall U-value | 0.5-8.0 |
| `solar_heat_gain_coefficient` | Number | - | SHGC (0-1) | 0.1-1.0 |
| `visible_transmittance` | Number | - | VT (0-1) | 0.1-1.0 |
| `air_leakage_l_sm2` | Number | L/s/m² | Air leakage rate | 0.01-5.0 |

## 4. system_data/hvac_systems.json

### HVAC System Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `system_id` | String | - | Unique system identifier | 3-20 chars |
| `system_name` | String | - | Descriptive system name | 1-100 chars |
| `system_type` | String | - | HVAC system type | Predefined types |
| `manufacturer` | String | - | Equipment manufacturer | 1-100 chars |
| `model` | String | - | Equipment model | 1-50 chars |
| `installation_date` | String | Date | Installation date | ISO 8601 |
| `warranty_expiration` | String | Date | Warranty expiration | ISO 8601 |

### Capacity Data

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `cooling_capacity_kw` | Number | kW | Cooling capacity | 5-5000 |
| `cooling_capacity_tons` | Number | tons | Cooling capacity (imperial) | 1.5-1400 |
| `heating_capacity_kw` | Number | kW | Heating capacity | 5-5000 |
| `supply_airflow_m3s` | Number | m³/s | Supply air flow rate | 0.5-500 |
| `supply_airflow_cfm` | Number | CFM | Supply air flow (imperial) | 1000-1000000 |
| `outside_air_m3s` | Number | m³/s | Outside air flow rate | 0.1-100 |

### Efficiency Ratings

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `cooling_eer` | Number | - | Energy efficiency ratio | 8-25 |
| `cooling_cop` | Number | - | Coefficient of performance | 2-8 |
| `heating_afue` | Number | - | Annual fuel utilization efficiency | 0.7-0.98 |
| `heating_cop` | Number | - | Heating COP | 1.5-6.0 |
| `integrated_part_load_value_iplv` | Number | - | IPLV rating | 8-30 |

## 5. system_data/dhw_systems.json

### DHW System Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `system_id` | String | - | Unique system identifier | 3-20 chars |
| `system_name` | String | - | Descriptive system name | 1-100 chars |
| `system_type` | String | - | DHW system type | Predefined types |
| `storage_capacity_liters` | Number | liters | Storage tank capacity | 50-5000 |
| `storage_capacity_gallons` | Number | gallons | Storage capacity (imperial) | 15-1300 |
| `input_capacity_kw` | Number | kW | Input heating capacity | 5-500 |
| `input_capacity_btu_hr` | Number | BTU/hr | Input capacity (imperial) | 17000-1700000 |

### Efficiency Ratings

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `energy_factor_ef` | Number | - | Energy factor | 0.4-0.95 |
| `thermal_efficiency_percent` | Number | % | Thermal efficiency | 60-98 |
| `uniform_energy_factor_uef` | Number | - | Uniform energy factor | 0.4-0.95 |
| `standby_loss_percent_hr` | Number | %/hr | Standby heat loss rate | 0.5-5.0 |

## 6. system_data/lighting_systems.json

### Lighting System Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `system_id` | String | - | Unique system identifier | 3-20 chars |
| `system_name` | String | - | Descriptive system name | 1-100 chars |
| `space_served` | String | - | Space or zone served | 1-100 chars |
| `total_area_m2` | Number | m² | Total area served | 10-10000 |
| `design_illuminance_lux` | Number | lux | Design illuminance level | 50-2000 |
| `design_illuminance_fc` | Number | fc | Design illuminance (imperial) | 5-200 |
| `lighting_power_density_w_m2` | Number | W/m² | Lighting power density | 2-25 |
| `lighting_power_density_w_sqft` | Number | W/ft² | LPD (imperial) | 0.2-2.5 |

### Fixture Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `fixture_id` | String | - | Unique fixture identifier | 3-20 chars |
| `fixture_type` | String | - | Fixture type | Predefined types |
| `manufacturer` | String | - | Fixture manufacturer | 1-100 chars |
| `model` | String | - | Fixture model | 1-50 chars |
| `quantity` | Number | - | Number of fixtures | 1-1000 |
| `wattage_per_fixture` | Number | W | Power per fixture | 5-1000 |
| `lumens_per_fixture` | Number | lm | Light output per fixture | 500-100000 |
| `efficacy_lm_w` | Number | lm/W | Luminous efficacy | 50-200 |
| `rated_life_hours` | Number | hours | Rated lamp life | 1000-100000 |
| `color_temperature_k` | Number | K | Color temperature | 2700-6500 |
| `cri` | Number | - | Color rendering index | 70-100 |

## 7. system_data/renewable_energy_systems.json

### Solar PV System Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `system_id` | String | - | Unique system identifier | 3-20 chars |
| `system_name` | String | - | Descriptive system name | 1-100 chars |
| `system_type` | String | - | Renewable energy type | Predefined types |
| `installation_date` | String | Date | Installation date | ISO 8601 |
| `dc_capacity_kw` | Number | kW | DC system capacity | 1-10000 |
| `ac_capacity_kw` | Number | kW | AC system capacity | 1-10000 |
| `number_of_modules` | Number | - | Total module count | 1-50000 |
| `module_capacity_w` | Number | W | Individual module capacity | 100-700 |

### Solar Module Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `manufacturer` | String | - | Module manufacturer | 1-100 chars |
| `model` | String | - | Module model | 1-50 chars |
| `technology` | String | - | Cell technology | Predefined types |
| `rated_power_w` | Number | W | Rated power output | 100-700 |
| `efficiency_percent` | Number | % | Module efficiency | 15-25 |
| `temperature_coefficient_percent_c` | Number | %/°C | Temperature coefficient | -0.5 to -0.2 |
| `vmp_v` | Number | V | Voltage at maximum power | 20-80 |
| `imp_a` | Number | A | Current at maximum power | 5-15 |

### Performance Data

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `design_annual_production_kwh` | Number | kWh | Design annual energy production | 1000-20000000 |
| `actual_annual_production_kwh` | Number | kWh | Actual annual production | 1000-20000000 |
| `performance_ratio` | Number | - | Performance ratio | 0.6-0.9 |
| `capacity_factor_percent` | Number | % | Capacity factor | 10-30 |
| `specific_yield_kwh_kw` | Number | kWh/kW | Specific yield | 800-2000 |
| `system_efficiency_percent` | Number | % | Overall system efficiency | 12-22 |

## 8. environmental_data/lidar_aerial_data.json

### LiDAR Survey Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `survey_date` | String | Date | Survey date | ISO 8601 |
| `survey_company` | String | - | Survey company name | 1-100 chars |
| `equipment` | String | - | LiDAR equipment used | 1-100 chars |
| `point_density` | String | - | Point cloud density | 1-50 chars |
| `accuracy` | String | - | Survey accuracy | 1-20 chars |
| `coordinate_system` | String | - | Coordinate reference system | 1-50 chars |

### Point Cloud Data

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `x` | Number | meters | X coordinate | -10000 to 10000 |
| `y` | Number | meters | Y coordinate | -10000 to 10000 |
| `z` | Number | meters | Z coordinate (elevation) | -100 to 1000 |
| `intensity` | Number | - | Point intensity value | 0-255 |

### Solar Analysis Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `global_horizontal_kwh_m2` | Number | kWh/m² | Annual global horizontal irradiation | 800-2500 |
| `direct_normal_kwh_m2` | Number | kWh/m² | Annual direct normal irradiation | 1000-3000 |
| `diffuse_horizontal_kwh_m2` | Number | kWh/m² | Annual diffuse horizontal irradiation | 400-1200 |
| `tilt_degrees` | Number | degrees | Surface tilt angle | 0-90 |
| `azimuth_degrees` | Number | degrees | Surface azimuth angle | 0-360 |
| `annual_irradiation_kwh_m2` | Number | kWh/m² | Annual irradiation on surface | 600-2200 |
| `shading_factor` | Number | - | Shading reduction factor | 0-1 |

## 9. environmental_data/air_tightness_data.json

### Blower Door Test Properties

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `test_id` | String | - | Unique test identifier | 3-20 chars |
| `test_date` | String | Date | Test date | ISO 8601 |
| `test_standard` | String | - | Test standard used | ASTM, ISO, etc. |
| `testing_company` | String | - | Testing company name | 1-100 chars |
| `technician` | String | - | Technician name and certification | 1-100 chars |

### Test Results

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `air_changes_per_hour_50pa` | Number | ACH | Air changes per hour at 50 Pa | 0.1-20.0 |
| `air_leakage_rate_50pa_m3h` | Number | m³/h | Air leakage rate at 50 Pa | 100-100000 |
| `air_leakage_rate_50pa_cfm` | Number | CFM | Air leakage rate (imperial) | 60-60000 |
| `specific_leakage_area_cm2_m2` | Number | cm²/m² | Specific leakage area | 1-20 |
| `normalized_leakage_area_cm2` | Number | cm² | Normalized leakage area | 100-10000 |
| `effective_leakage_area_cm2` | Number | cm² | Effective leakage area | 50-8000 |

### Building Leakage Curve

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `flow_coefficient_c` | Number | - | Flow coefficient | 100-5000 |
| `pressure_exponent_n` | Number | - | Pressure exponent | 0.5-1.0 |
| `correlation_coefficient_r2` | Number | - | R-squared correlation | 0.9-1.0 |

### Weather Conditions

| Field | Type | Units | Description | Valid Range |
|-------|------|-------|-------------|-------------|
| `outdoor_temperature_c` | Number | °C | Outdoor temperature | -40 to 50 |
| `indoor_temperature_c` | Number | °C | Indoor temperature | 15 to 30 |
| `wind_speed_ms` | Number | m/s | Wind speed | 0-20 |
| `wind_direction_degrees` | Number | degrees | Wind direction | 0-360 |
| `barometric_pressure_pa` | Number | Pa | Barometric pressure | 95000-105000 |

## Data Validation Rules

### General Rules
1. All required fields must be present and non-null
2. Numeric values must be within specified valid ranges
3. String values must not exceed maximum length limits
4. Date values must follow ISO 8601 format
5. Identifiers must be unique within their scope

### Cross-Reference Rules
1. Building areas must be consistent across files
2. R-values and U-values must be mathematically consistent (U ≈ 1/R)
3. System capacities must be appropriate for building size
4. Energy consumption must align with system specifications
5. Geometric coordinates must form valid shapes

### Physical Constraint Rules
1. Thermal properties must be within realistic ranges
2. System efficiencies must meet minimum standards
3. Air leakage rates must be achievable
4. Solar irradiation must match climate zone
5. Material properties must be physically possible

## Units and Conversions

### Length Conversions
- 1 meter = 3.28084 feet
- 1 millimeter = 0.0393701 inches
- 1 kilometer = 0.621371 miles

### Area Conversions
- 1 m² = 10.7639 ft²
- 1 hectare = 2.47105 acres

### Volume Conversions
- 1 m³ = 35.3147 ft³
- 1 liter = 0.264172 gallons (US)

### Temperature Conversions
- °C = (°F - 32) × 5/9
- K = °C + 273.15

### Energy Conversions
- 1 kWh = 3,412.14 BTU
- 1 kWh = 3.6 MJ
- 1 therm = 29.3071 kWh

### Power Conversions
- 1 kW = 3,412.14 BTU/hr
- 1 ton (cooling) = 3.51685 kW

### Pressure Conversions
- 1 Pa = 0.000145038 psi
- 1 kPa = 0.145038 psi
- 1 atm = 101,325 Pa

This data dictionary serves as the authoritative reference for all field definitions and validation rules used throughout the Building DNA Dataset.