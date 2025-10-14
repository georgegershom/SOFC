# Building Retrofit Dataset - Data Dictionary

## Building Master Table (`building_master.csv`)

### Building Identification
- `building_id`: Unique building identifier (e.g., B001, B002)
- `building_type`: Type of building (residential, office, retail, educational, healthcare, industrial)

### Geometric & Structural Data
- `floor_area_m2`: Floor area in square meters
- `total_floor_area_m2`: Total floor area across all floors
- `num_floors`: Number of floors
- `floor_height_m`: Height of each floor in meters
- `building_height_m`: Total building height in meters
- `footprint_length_m`: Building footprint length
- `footprint_width_m`: Building footprint width
- `rooftop_area_m2`: Rooftop area in square meters
- `volume_m3`: Total building volume in cubic meters
- `window_area_m2`: Total window area in square meters
- `window_to_wall_ratio`: Ratio of window area to wall area

### Construction Data
- `construction_year`: Year of construction
- `construction_period`: Construction period category (pre-1970, 1970-1980, etc.)
- `building_age_years`: Current building age in years
- `construction_quality`: Quality rating (poor, fair, good, excellent)
- `architectural_style`: Architectural style (traditional, modern, brutalist, etc.)
- `primary_material`: Primary construction material
- `secondary_material`: Secondary construction material
- `roof_material`: Roof material type
- `window_material`: Window material type
- `insulation_present`: Boolean indicating presence of insulation
- `insulation_type`: Type of insulation material

### Thermal Properties
- `u_value_wall_w_m2k`: Wall U-value in W/m²K
- `u_value_roof_w_m2k`: Roof U-value in W/m²K
- `u_value_floor_w_m2k`: Floor U-value in W/m²K
- `u_value_window_w_m2k`: Window U-value in W/m²K
- `r_value_wall_m2k_w`: Wall R-value in m²K/W
- `r_value_roof_m2k_w`: Roof R-value in m²K/W
- `r_value_floor_m2k_w`: Floor R-value in m²K/W
- `r_value_window_m2k_w`: Window R-value in m²K/W
- `air_tightness_ach50`: Air tightness at 50 Pa in ACH
- `thermal_mass_kj_m2k`: Thermal mass in kJ/m²K

### Energy Performance
- `eu_rating`: EU energy efficiency rating (A-G)
- `energy_star_score`: ENERGY STAR score (1-100)
- `leed_certified`: Boolean indicating LEED certification
- `leed_level`: LEED certification level (Certified, Silver, Gold, Platinum)

### Retrofit Information
- `has_retrofit`: Boolean indicating if building has been retrofitted
- `retrofit_year`: Year of retrofit (if applicable)
- `retrofit_type`: Type of retrofit performed
- `retrofit_cost_eur`: Retrofit cost in Euros
- `energy_savings_percent`: Energy savings percentage from retrofit

### LCA Data
- `total_gwp_kg_co2e`: Total global warming potential in kg CO2e
- `total_energy_mj`: Total energy consumption in MJ
- `total_water_l`: Total water consumption in liters
- `gwp_per_m2_kg_co2e`: GWP per square meter in kg CO2e/m²
- `renewable_energy_percent`: Percentage of renewable energy
- `recycled_materials_percent`: Percentage of recycled materials

## IoT Sensor Data

### Energy Consumption Data (`*_energy.csv`)
- `timestamp`: Hourly timestamp
- `building_id`: Building identifier
- `building_type`: Building type
- `total_consumption`: Total energy consumption in kWh
- `heating_consumption`: Heating energy consumption in kWh
- `cooling_consumption`: Cooling energy consumption in kWh
- `lighting_consumption`: Lighting energy consumption in kWh
- `equipment_consumption`: Equipment energy consumption in kWh
- `hot_water_consumption`: Hot water energy consumption in kWh

### Environmental Data (`*_environmental.csv`)
- `timestamp`: Hourly timestamp
- `building_id`: Building identifier
- `co2_ppm`: CO2 concentration in ppm
- `tvoc_ug_m3`: TVOC concentration in μg/m³
- `pm25_ug_m3`: PM2.5 concentration in μg/m³
- `temperature_c`: Indoor temperature in °C
- `humidity_percent`: Indoor humidity in %

### Weather Data (`*_weather.csv`)
- `timestamp`: Hourly timestamp
- `building_id`: Building identifier
- `outdoor_temp_c`: Outdoor temperature in °C
- `outdoor_humidity_percent`: Outdoor humidity in %
- `wind_speed_ms`: Wind speed in m/s
- `solar_radiation_w_m2`: Solar radiation in W/m²
- `precipitation_mmh`: Precipitation in mm/h

### Occupancy Data (`*_occupancy.csv`)
- `timestamp`: Hourly timestamp
- `building_id`: Building identifier
- `building_type`: Building type
- `occupancy_ratio`: Occupancy ratio (0-1)
- `people_count`: Number of people in building

## Energy Performance Data

### Historical Consumption (`historical_consumption.csv`)
- `building_id`: Building identifier
- `year`: Year of data
- `total_consumption_kwh`: Total annual consumption in kWh
- `energy_intensity_kwh_m2`: Energy intensity in kWh/m²
- `electricity_kwh`: Electricity consumption in kWh
- `natural_gas_kwh`: Natural gas consumption in kWh
- `oil_kwh`: Oil consumption in kWh
- `district_heating_kwh`: District heating consumption in kWh
- `renewable_kwh`: Renewable energy consumption in kWh
- `jan_consumption` to `dec_consumption`: Monthly consumption breakdown

### Efficiency Ratings (`efficiency_ratings.csv`)
- `building_id`: Building identifier
- `eu_rating`: EU energy efficiency rating (A-G)
- `energy_star_score`: ENERGY STAR score (1-100)
- `leed_certified`: LEED certification status
- `leed_level`: LEED certification level
- `energy_intensity_kwh_m2`: Energy intensity in kWh/m²

### Retrofit Data (`retrofit_data.csv`)
- `building_id`: Building identifier
- `has_retrofit`: Retrofit status
- `retrofit_year`: Year of retrofit
- `retrofit_type`: Type of retrofit
- `retrofit_cost_eur`: Retrofit cost in Euros
- `energy_savings_percent`: Energy savings percentage
- `payback_period_years`: Payback period in years
- `co2_reduction_percent`: CO2 reduction percentage

## LCA Data

### Material EPDs (`material_epds.csv`)
- `material_id`: Unique material identifier
- `material_name`: Material name
- `material_category`: Material category
- `epd_program`: EPD program (EN 15804, ISO 14025, etc.)
- `epd_valid_from`: EPD validity start date
- `epd_valid_until`: EPD validity end date
- `functional_unit`: Functional unit (typically 1 m³)
- `density_kg_m3`: Material density in kg/m³
- `gwp_kg_co2e_m3`: Global warming potential in kg CO2e/m³
- `odp_kg_cfc11e_m3`: Ozone depletion potential in kg CFC11e/m³
- `pocp_kg_ethene_m3`: Photochemical ozone creation potential in kg ethene/m³
- `ap_kg_so2e_m3`: Acidification potential in kg SO2e/m³
- `ep_kg_po4e_m3`: Eutrophication potential in kg PO4e/m³
- `adp_kg_sbe_m3`: Abiotic depletion potential in kg Sbe/m³
- `renewable_energy_percent`: Percentage of renewable energy
- `recycled_content_percent`: Percentage of recycled content
- `recyclability_percent`: Percentage recyclability

### Construction Process LCA (`construction_process_lca.csv`)
- `process_id`: Unique process identifier
- `process_name`: Process name
- `process_category`: Process category (foundation, structure, envelope, etc.)
- `unit`: Unit of measurement (per m²)
- `gwp_kg_co2e_m2`: GWP per m² in kg CO2e
- `energy_consumption_mj_m2`: Energy consumption per m² in MJ
- `water_consumption_l_m2`: Water consumption per m² in liters
- `waste_generation_kg_m2`: Waste generation per m² in kg
- `noise_level_db`: Noise level in dB
- `dust_emission_kg_m2`: Dust emission per m² in kg

### Building LCA Database (`building_lca_database.csv`)
- `building_id`: Building identifier
- `building_type`: Building type
- `construction_year`: Construction year
- `floor_area_m2`: Floor area in m²
- `total_gwp_kg_co2e`: Total GWP in kg CO2e
- `total_energy_mj`: Total energy in MJ
- `total_water_l`: Total water consumption in liters
- `gwp_per_m2_kg_co2e`: GWP per m² in kg CO2e/m²
- `energy_per_m2_mj`: Energy per m² in MJ/m²
- `water_per_m2_l`: Water per m² in L/m²
- `renewable_energy_percent`: Renewable energy percentage
- `recycled_materials_percent`: Recycled materials percentage
- `waste_generation_kg`: Waste generation in kg
- `hazardous_waste_kg`: Hazardous waste in kg

### Retrofit LCA Scenarios (`retrofit_lca_scenarios.csv`)
- `building_id`: Building identifier
- `retrofit_scenario`: Retrofit scenario (no_retrofit, light_retrofit, deep_retrofit, comprehensive_retrofit)
- `gwp_reduction_percent`: GWP reduction percentage
- `energy_reduction_percent`: Energy reduction percentage
- `retrofit_cost_eur_m2`: Retrofit cost per m² in Euros
- `payback_period_years`: Payback period in years
- `co2_savings_kg_co2e_m2`: CO2 savings per m² in kg CO2e
- `energy_savings_kwh_m2`: Energy savings per m² in kWh
- `renewable_energy_addition_percent`: Additional renewable energy percentage
- `material_recycling_percent`: Material recycling percentage

## Analysis-Ready Datasets

### Energy Analysis Dataset (`energy_analysis.csv`)
Combines building attributes with latest energy performance data for energy analysis.

### Retrofit Analysis Dataset (`retrofit_analysis.csv`)
Includes building attributes with retrofit scenarios and cost-benefit analysis.

### ML-Ready Dataset (`ml_ready.csv`)
Features engineered for machine learning with additional derived features:
- `age_category`: Age category (new, modern, old, historic)
- `size_category`: Size category (small, medium, large, very_large, mega)
- `thermal_score`: Average thermal performance score
- `window_performance`: Window performance metric
- `quality_score`: Construction quality score (1-4)
- `insulation_score`: Insulation performance score
- `energy_efficiency_category`: Energy efficiency category

## Data Quality Notes

- All timestamps are in UTC
- Missing values are indicated by NaN
- Energy consumption data is in kWh unless otherwise specified
- Environmental data follows standard units (ppm, μg/m³, °C, %)
- LCA data follows EN 15804 and ISO 14025 standards
- All monetary values are in Euros (EUR)