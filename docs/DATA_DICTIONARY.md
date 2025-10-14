### Dataset Data Dictionary

This dictionary describes the CSV artifacts produced in `data/processed` by `scripts/generate_dataset.py`.

- **buildings.csv**
  - **building_id**: Unique identifier (string, e.g., B00001)
  - **latitude, longitude**: Coordinates (float)
  - **function_type**: {residential, office, school, retail, hospital, warehouse}
  - **style**: Architectural style label (string)
  - **build_quality**: {poor, average, good, excellent}
  - **construction_year**: Year built (int)
  - **num_floors**: Count of floors (int)
  - **floor_height_m**: Floor-to-floor height (m, float)
  - **rooftop_area_m2, footprint_area_m2**: Areas (m², float)
  - **gross_floor_area_m2**: Total GFA (m², float)
  - **height_m**: Approximate building height (m, float)
  - **volume_m3**: Approximate internal volume (m³, float)
  - **wall_material, roof_material**: Primary materials (string)
  - **window_glazing**: {single, double, triple}
  - **wall_u_value_w_m2k, roof_u_value_w_m2k, window_u_value_w_m2k**: Thermal transmittance (float)
  - **baseline_energy_intensity_kwh_m2y**: Baseline energy use intensity (kWh/m²·yr, float)
  - **energy_rating**: EU-like label {A..G}, derived from EUI
  - **monitored_iot**: Boolean whether IoT timeseries is generated

- **epd_factors.csv**
  - **material**: Material category
  - **kgco2e_per_kg**: Embodied carbon factor (kgCO2e/kg)
  - **source_note**: Text note (illustrative only)

- **lca_material_inventory.csv**
  - **building_id**
  - **component**: {walls, roof, windows, envelope_insulation}
  - **material**: Material name
  - **mass_kg**: Estimated mass (kg)

- **lca_results.csv**
  - **building_id**
  - **embodied_carbon_kgco2e**: Sum over inventory using `epd_factors.csv`

- **energy_performance_monthly.csv**
  - **building_id**
  - **period**: End-of-month date (YYYY-MM-DD)
  - **year, month**: Integers
  - **monthly_energy_kwh**: Modeled whole-building monthly energy (kWh)
  - **retrofit**: Boolean flag
  - **retrofit_date**: End-of-month retrofit completion date (YYYY-MM-DD or blank)
  - **retrofit_scopes**: Comma-separated subset of {envelope,hvac,lighting,controls}
  - **savings_fraction_post**: Applied fraction after retrofit for post-periods

- **iot_timeseries.csv.gz** (gzip CSV)
  - **timestamp**: ISO-8601 timestamp (UTC naive)
  - **building_id**
  - **energy_wh_total**: Total electricity per interval (Wh)
  - **energy_wh_hvac, energy_wh_lighting**: End-use splits (Wh)
  - **co2_ppm**: Indoor CO₂ concentration (ppm)
  - **tvoc_ppb**: Total VOC (ppb)
  - **pm25_ugm3**: PM2.5 (µg/m³)
  - **indoor_temp_c, indoor_rh_pct**
  - **outdoor_temp_c, outdoor_rh_pct**
  - **occupancy_count**: Modeled occupants

Notes:
- Values are synthetic and for research prototyping only; not suitable for design or compliance.
- Thermal and embodied factors are coarse approximations to support ML experimentation.
