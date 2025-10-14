# Building Retrofit Research Dataset Suite

## AI- and IoT-Driven Optimization of Building Retrofits

This comprehensive dataset suite has been specifically designed for PhD research on AI- and IoT-driven optimization of building retrofits. It integrates multiple data dimensions essential for studying building energy performance, retrofit strategies, and lifecycle environmental impacts.

---

## 📊 Dataset Overview

The suite consists of **7 core datasets** and **4 reference databases**, covering 50 buildings over a full year with hourly IoT sensor data (438,000 records).

### Core Datasets

| Dataset | Records | Description |
|---------|---------|-------------|
| **building_attributes.csv** | 50 | Comprehensive building fabric, geometric, structural, and thermal properties |
| **iot_sensor_data.parquet** | 438,000 | Hourly IoT sensor readings (energy, environmental quality, weather, occupancy) |
| **energy_performance_historical.csv** | 250 | 5 years of annual energy consumption data |
| **retrofit_scenarios.csv** | 150 | Retrofit measure combinations with costs, savings, and ROI |
| **lca_building_baseline.csv** | 50 | Lifecycle assessment baseline for all buildings |
| **lca_retrofit_measures.csv** | 450+ | LCA for individual retrofit measures |
| **lca_carbon_payback.csv** | 450+ | Carbon payback analysis for retrofit measures |

### Integrated Datasets (Generated)

| Dataset | Description |
|---------|-------------|
| **integrated_building_master.csv** | Complete building profiles with all attributes, energy stats, and best retrofit options |
| **integrated_retrofit_analysis.csv** | Comprehensive retrofit analysis combining energy, cost, and LCA data |
| **ml_ready_dataset.parquet** | ML-ready feature dataset for predictive modeling |
| **timeseries_daily.csv** | Daily aggregated IoT sensor data |
| **timeseries_monthly.csv** | Monthly aggregated energy and environmental data |
| **hourly_patterns.csv** | Average hourly consumption patterns by building |

### Reference Databases (JSON)

- **material_properties_database.json** - Wall, roof, and window material properties
- **retrofit_measures_database.json** - Retrofit measure specifications and impacts
- **epd_database.json** - Environmental Product Declarations for construction materials
- **hvac_lca_database.json** - HVAC system lifecycle assessment data

---

## 🎯 Research Applications

This dataset suite supports:

1. **Energy Performance Prediction** - ML models for forecasting building energy consumption
2. **Retrofit Optimization** - Multi-objective optimization considering cost, energy, and carbon
3. **Indoor Environmental Quality Analysis** - Correlation between IEQ parameters and energy use
4. **Lifecycle Assessment** - Comprehensive environmental impact analysis
5. **Occupancy Pattern Recognition** - Understanding energy use patterns
6. **Building Stock Analysis** - Large-scale retrofit potential assessment
7. **Smart Building Control** - IoT-driven HVAC and lighting optimization

---

## 📁 Data Categories and Parameters

### 1. IoT Sensor Data
**Temporal Resolution:** Hourly (8,760 hours/year)

**Parameters:**
- **Energy Consumption**
  - Total building consumption (kWh)
  - HVAC consumption (kWh)
  - Lighting consumption (kWh)
  - Equipment consumption (kWh)

- **Indoor Environmental Quality (IEQ)**
  - Indoor temperature (°C)
  - Indoor humidity (%)
  - CO₂ levels (ppm)
  - TVOC - Total Volatile Organic Compounds (µg/m³)
  - PM2.5 particulate matter (µg/m³)

- **Outdoor Weather**
  - Outdoor temperature (°C)
  - Outdoor humidity (%)
  - Solar radiation (W/m²)
  - Wind speed (m/s)

- **Occupancy**
  - Occupancy ratio (0-1)

### 2. Building Attributes & Fabric

**Geometric Properties:**
- Number of floors
- Floor height, total height (m)
- Footprint area, total floor area (m²)
- Building dimensions (length, width)
- Rooftop area (m²)
- Wall area, window area (m²)
- Window-to-wall ratio
- Volume (m³)

**Construction Details:**
- Construction year
- Building age
- Building type (residential, commercial, industrial, educational)
- Building function
- Architectural style
- Quality rating

**Thermal Properties:**
- Wall U-value and R-value (W/m²K, m²K/W)
- Roof U-value and R-value
- Window U-value and SHGC (Solar Heat Gain Coefficient)
- Envelope average U-value
- Heat loss coefficient (W/K)
- Thermal mass (kJ/K)

**Materials:**
- Wall material type
- Roof material type
- Window type (single, double, triple glazed)
- HVAC system type

### 3. Energy Performance

**Historical Data (5 years):**
- Annual total consumption (kWh)
- Consumption by end-use (heating/cooling, lighting, equipment)
- Energy costs (EUR)
- Carbon emissions (kg CO₂)

**Efficiency Ratings:**
- EPC rating (EU A-G scale)

**Retrofit Scenarios:**
- Minimal, Standard, and Deep Retrofit options
- List of retrofit measures
- Investment costs (EUR)
- Annual energy savings (kWh and %)
- Annual cost savings (EUR)
- Annual carbon savings (kg CO₂)
- Simple payback period (years)
- NPV over 20 years (EUR)
- ROI (%)
- Post-retrofit EPC rating

### 4. Lifecycle Assessment (LCA)

**Building Baseline:**
- Total embodied carbon (kg CO₂eq)
- Embodied carbon per m² (kg CO₂eq/m²)
- Total embodied energy (MJ)
- Embodied energy per m² (MJ/m²)
- Water usage (liters)
- Construction process impacts
- Transportation impacts
- HVAC manufacturing impacts

**Retrofit Measures:**
- Material quantities (kg)
- Global Warming Potential (kg CO₂eq)
- Embodied energy (MJ)
- Water usage (liters)
- Recyclability (%)
- Lifespan (years)
- Installation impacts
- Demolition impacts (where applicable)

**Carbon Payback:**
- Embodied carbon of retrofit (kg CO₂eq)
- Annual carbon savings (kg CO₂eq/year)
- Carbon payback period (years)
- Lifetime carbon benefit (kg CO₂eq)
- Benefit-to-impact ratio

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
cd building_retrofit_datasets

# Install dependencies
pip install -r requirements.txt
```

### Generate All Datasets

```bash
# Navigate to scripts directory
cd scripts

# Run the master generation script
python generate_all_datasets.py
```

This will generate all datasets in sequence:
1. Building attributes (~ 5 seconds)
2. IoT sensor data (~ 45-60 seconds)
3. Energy performance & retrofit scenarios (~ 10 seconds)
4. LCA data (~ 10 seconds)
5. Integrated datasets (~ 5 seconds)

**Total generation time:** ~1.5-2 minutes

### Generate Individual Datasets

You can also run individual generators:

```bash
# Building attributes only
python generate_building_attributes.py

# IoT sensor data only
python generate_iot_sensor_data.py

# Energy performance only
python generate_energy_performance.py

# LCA data only
python generate_lca_data.py

# Integration only (requires other datasets)
python integrate_datasets.py
```

---

## 📖 Usage Examples

### Loading Data in Python

```python
import pandas as pd
import numpy as np

# Load building attributes
buildings = pd.read_csv('../data/building_attributes.csv')

# Load IoT sensor data (Parquet is faster than CSV)
iot_data = pd.read_parquet('../data/iot_sensor_data.parquet')

# Load retrofit scenarios
retrofits = pd.read_csv('../data/retrofit_scenarios.csv')

# Load integrated building master
master = pd.read_csv('../data/integrated_building_master.csv')

# Load ML-ready dataset
ml_data = pd.read_parquet('../data/ml_ready_dataset.parquet')
```

### Example Analysis: Energy Consumption by Building Type

```python
import matplotlib.pyplot as plt

# Aggregate energy consumption by building type
energy_by_type = iot_data.groupby('building_type')['total_energy_consumption_kwh'].sum()

# Plot
energy_by_type.plot(kind='bar', title='Total Energy Consumption by Building Type')
plt.ylabel('Energy (kWh)')
plt.show()
```

### Example Analysis: Retrofit ROI

```python
# Find best retrofit scenario for each building
best_retrofits = retrofits.loc[
    retrofits.groupby('building_id')['roi_percent'].idxmax()
]

# Plot payback vs savings
plt.scatter(best_retrofits['simple_payback_years'], 
            best_retrofits['total_energy_saving_pct'])
plt.xlabel('Simple Payback (years)')
plt.ylabel('Energy Savings (%)')
plt.title('Retrofit Investment Analysis')
plt.show()
```

### Example ML: Energy Consumption Prediction

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load ML-ready dataset
ml_data = pd.read_parquet('../data/ml_ready_dataset.parquet')

# Select features and target
features = ['outdoor_temperature', 'indoor_temperature', 'occupancy_ratio',
            'solar_radiation', 'total_floor_area_m2', 'envelope_avg_u_value',
            'day_of_week', 'month']

X = pd.get_dummies(ml_data[features + ['building_type']], drop_first=True)
y = ml_data['total_energy_consumption_kwh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10))
```

---

## 🔬 Dataset Characteristics

### Building Stock Composition

- **50 buildings** across 4 types:
  - Residential (~30%)
  - Commercial (~30%)
  - Industrial (~20%)
  - Educational (~20%)

- **Construction years:** 1920-2023
- **EPC ratings:** Distributed A-G (realistic distribution favoring D-E)
- **Floor areas:** 500-5,000 m²
- **Building ages:** 1-104 years

### Data Realism

This is a **fabricated/synthetic dataset** generated using realistic models and distributions based on:
- European building stock characteristics
- Standard thermal properties (ISO 13790, EN 15978)
- Real-world retrofit measure performance
- Validated EPD databases (EN 15804)
- Industry-standard LCA methodologies

**Use cases:**
- ✅ Algorithm development and testing
- ✅ Proof-of-concept research
- ✅ Educational purposes
- ✅ Methodology validation
- ⚠️  Not for policy recommendations without validation
- ⚠️  Should be supplemented with real data for production applications

### Quality Metrics

- **Completeness:** 100% (no missing values in core attributes)
- **Consistency:** Cross-validated across datasets
- **Temporal coverage:** Full year (2023)
- **Temporal resolution:** Hourly for IoT, annual for energy history
- **Spatial coverage:** 50 diverse buildings

---

## 📚 Methodology & Assumptions

### IoT Sensor Data Generation

1. **Energy Consumption:**
   - Base consumption per m² varies by building type
   - Adjusted for EPC rating (efficiency factor)
   - Occupancy-driven variation
   - Weather-dependent HVAC loads
   - Realistic daily and seasonal patterns

2. **Indoor Environmental Quality:**
   - CO₂ levels correlated with occupancy
   - Temperature affected by outdoor weather and HVAC efficiency
   - TVOC and PM2.5 with realistic ranges
   - Humidity based on outdoor conditions and occupancy

3. **Weather Data:**
   - Seasonal temperature variation (sinusoidal)
   - Daily temperature cycles
   - Solar radiation (daylight hours only)
   - Correlated humidity

4. **Occupancy Patterns:**
   - Building-type specific patterns
   - Weekday vs. weekend differences
   - Realistic working hours

### Retrofit Analysis

- **Measures considered:** 8 categories (insulation, windows, HVAC, solar PV, etc.)
- **Cost estimation:** Based on industry averages (EUR, 2023)
- **Savings calculation:** Physics-based with diminishing returns for multiple measures
- **Financial metrics:** NPV at 3% discount rate over 20 years
- **EPC improvement:** Based on energy savings thresholds

### Lifecycle Assessment

- **EPD data:** Based on EN 15804 compliant databases
- **System boundaries:** Cradle-to-grave
- **Impact categories:** GWP (Global Warming Potential), embodied energy, water
- **Functional unit:** Per kg of material, per m² of building element
- **Construction impacts:** Transportation (50-200 km), installation labor
- **End-of-life:** Recycling benefits included

---

## 🗂️ File Structure

```
building_retrofit_datasets/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Generated datasets (created after running scripts)
│   ├── building_attributes.csv
│   ├── building_attributes.xlsx
│   ├── iot_sensor_data.csv
│   ├── iot_sensor_data.parquet
│   ├── energy_performance_historical.csv
│   ├── retrofit_scenarios.csv
│   ├── retrofit_scenarios.xlsx
│   ├── lca_building_baseline.csv
│   ├── lca_retrofit_measures.csv
│   ├── lca_carbon_payback.csv
│   ├── integrated_building_master.csv
│   ├── integrated_building_master.xlsx
│   ├── integrated_retrofit_analysis.csv
│   ├── integrated_retrofit_analysis.xlsx
│   ├── ml_ready_dataset.csv
│   ├── ml_ready_dataset.parquet
│   ├── timeseries_daily.csv
│   ├── timeseries_monthly.csv
│   ├── hourly_patterns.csv
│   ├── material_properties_database.json
│   ├── retrofit_measures_database.json
│   ├── epd_database.json
│   ├── hvac_lca_database.json
│   ├── data_quality_report.json
│   └── integration_summary.json
│
├── scripts/                           # Dataset generators
│   ├── generate_all_datasets.py      # Master script - run this!
│   ├── generate_building_attributes.py
│   ├── generate_iot_sensor_data.py
│   ├── generate_energy_performance.py
│   ├── generate_lca_data.py
│   └── integrate_datasets.py
│
└── docs/                              # Additional documentation (optional)
```

---

## 🎓 Citation & Attribution

If you use this dataset in your research, please cite as:

```
Building Retrofit Research Dataset Suite (2024)
AI- and IoT-Driven Optimization of Building Retrofits
Generated synthetic dataset for building energy performance and retrofit analysis
Version 1.0
```

BibTeX:
```bibtex
@dataset{building_retrofit_2024,
  title={Building Retrofit Research Dataset Suite},
  subtitle={AI- and IoT-Driven Optimization of Building Retrofits},
  year={2024},
  version={1.0},
  type={Synthetic Dataset},
  keywords={Building retrofit, Energy efficiency, IoT, Machine learning, LCA}
}
```

---

## 📋 Data Dictionary

### Building Attributes (building_attributes.csv)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| building_id | string | - | Unique building identifier (BLD_001 to BLD_050) |
| building_type | string | - | Type: residential, commercial, industrial, educational |
| building_function | string | - | Specific function (e.g., Office, School, Factory) |
| construction_year | integer | year | Year of construction |
| building_age_years | integer | years | Age of building (2024 - construction_year) |
| num_floors | integer | - | Number of floors |
| floor_height_m | float | m | Average floor height |
| total_height_m | float | m | Total building height |
| footprint_area_m2 | float | m² | Building footprint |
| total_floor_area_m2 | float | m² | Total floor area (gross) |
| rooftop_area_m2 | float | m² | Roof area |
| wall_area_m2 | float | m² | External wall area |
| window_area_m2 | float | m² | Total window area |
| window_wall_ratio | float | - | Window-to-wall ratio |
| volume_m3 | float | m³ | Building volume |
| wall_material | string | - | Wall construction type |
| roof_material | string | - | Roof construction type |
| window_type | string | - | Window type (single/double/triple glazed) |
| hvac_system | string | - | HVAC system type |
| wall_u_value | float | W/m²K | Wall thermal transmittance |
| wall_r_value | float | m²K/W | Wall thermal resistance |
| roof_u_value | float | W/m²K | Roof thermal transmittance |
| roof_r_value | float | m²K/W | Roof thermal resistance |
| window_u_value | float | W/m²K | Window thermal transmittance |
| window_shgc | float | - | Solar heat gain coefficient |
| envelope_avg_u_value | float | W/m²K | Weighted average envelope U-value |
| heat_loss_coefficient_w_k | float | W/K | Building heat loss coefficient |
| thermal_mass_kj_k | float | kJ/K | Building thermal mass |
| epc_rating | string | - | Energy Performance Certificate (A-G) |
| quality_rating | string | - | Construction quality (poor/fair/good/excellent) |
| architectural_style | string | - | Architectural style |

### IoT Sensor Data (iot_sensor_data.parquet)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| timestamp | datetime | - | Recording timestamp (hourly) |
| building_id | string | - | Building identifier |
| building_type | string | - | Building type |
| occupancy_ratio | float | 0-1 | Occupancy ratio (0=empty, 1=full) |
| outdoor_temperature | float | °C | Outdoor air temperature |
| outdoor_humidity | float | % | Outdoor relative humidity |
| solar_radiation | float | W/m² | Solar radiation |
| wind_speed | float | m/s | Wind speed |
| indoor_temperature | float | °C | Indoor air temperature |
| indoor_humidity | float | % | Indoor relative humidity |
| co2_level | float | ppm | CO₂ concentration |
| tvoc | float | µg/m³ | Total volatile organic compounds |
| pm25 | float | µg/m³ | Particulate matter (PM2.5) |
| total_energy_consumption_kwh | float | kWh | Total hourly energy consumption |
| hvac_consumption_kwh | float | kWh | HVAC energy consumption |
| lighting_consumption_kwh | float | kWh | Lighting energy consumption |
| equipment_consumption_kwh | float | kWh | Equipment energy consumption |

### Retrofit Scenarios (retrofit_scenarios.csv)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| building_id | string | - | Building identifier |
| scenario_name | string | - | Retrofit scenario (Minimal/Standard/Deep Retrofit) |
| baseline_consumption_kwh | float | kWh/year | Current annual energy consumption |
| num_measures | integer | - | Number of retrofit measures |
| measure_list | string | - | Comma-separated list of measures |
| total_cost_eur | float | EUR | Total retrofit investment cost |
| annual_energy_saving_kwh | float | kWh/year | Annual energy savings |
| annual_cost_saving_eur | float | EUR/year | Annual cost savings |
| annual_carbon_saving_kgco2 | float | kg CO₂/year | Annual carbon savings |
| total_energy_saving_pct | float | % | Energy savings percentage |
| total_carbon_saving_pct | float | % | Carbon savings percentage |
| simple_payback_years | float | years | Simple payback period |
| npv_20years_eur | float | EUR | Net present value (20 years, 3% discount) |
| roi_percent | float | % | Return on investment |
| pre_retrofit_epc | string | - | Current EPC rating |
| post_retrofit_epc | string | - | Estimated post-retrofit EPC rating |

### LCA Building Baseline (lca_building_baseline.csv)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| building_id | string | - | Building identifier |
| total_gwp_materials_kgco2eq | float | kg CO₂eq | Total GWP from materials |
| total_embodied_energy_mj | float | MJ | Total embodied energy |
| total_water_usage_l | float | liters | Total water usage |
| gwp_per_m2_kgco2eq | float | kg CO₂eq/m² | GWP per square meter |
| embodied_energy_per_m2_mj | float | MJ/m² | Embodied energy per square meter |
| construction_gwp_kgco2eq | float | kg CO₂eq | Construction process GWP |
| transport_gwp_kgco2eq | float | kg CO₂eq | Transportation GWP |
| transport_distance_km | float | km | Average transport distance |
| hvac_manufacturing_gwp_kgco2eq | float | kg CO₂eq | HVAC manufacturing GWP |
| hvac_operational_gwp_kgco2eq_year | float | kg CO₂eq/year | HVAC annual operational GWP |
| hvac_lifespan_years | integer | years | HVAC system lifespan |
| total_embodied_carbon_kgco2eq | float | kg CO₂eq | Total embodied carbon |

---

## 🔧 Customization

You can customize the dataset generation by modifying the generator scripts:

### Change Number of Buildings

In each generator script, modify the `num_buildings` parameter:

```python
# In generate_building_attributes.py
generator = BuildingAttributesGenerator(num_buildings=100)  # Change from 50 to 100

# In generate_iot_sensor_data.py
generator = IoTSensorDataGenerator(num_buildings=100, days=365)
```

### Change Time Period

In `generate_iot_sensor_data.py`:

```python
# Change duration
generator = IoTSensorDataGenerator(num_buildings=50, days=730)  # 2 years instead of 1

# Change start date
self.start_date = datetime(2022, 1, 1)  # Start from 2022
```

### Modify Material Properties

Edit the material databases in the generators or modify the JSON files after generation:
- `material_properties_database.json`
- `epd_database.json`
- `hvac_lca_database.json`

### Adjust Retrofit Measures

In `generate_energy_performance.py`, modify the `retrofit_measures` dictionary:

```python
self.retrofit_measures = {
    'wall_insulation': {
        'cost_per_m2': 85,  # Adjust cost
        'energy_saving_percent': 22,  # Adjust savings
        # ...
    },
    # Add new measures...
}
```

---

## ⚠️ Limitations & Disclaimers

1. **Synthetic Data:** This is fabricated data for research purposes. While based on realistic models, it should not replace real measurements for production applications.

2. **Geographic Specificity:** Patterns are based on European/temperate climate assumptions. Adjust for other climates.

3. **Simplifications:**
   - Thermal calculations are simplified compared to full building simulation
   - Some interdependencies between retrofit measures are modeled with reduction factors
   - LCA data is representative but not building-specific

4. **Validation:** Always validate models and findings with real-world data before deployment.

5. **Updates:** Building codes, material properties, and costs evolve. Update reference databases periodically.

---

## 🤝 Contributing

To extend or improve this dataset:

1. **Add new building types:** Modify building type distributions and patterns
2. **Include additional sensors:** Extend IoT data generation with new parameters
3. **Add retrofit measures:** Update retrofit measures database
4. **Improve LCA data:** Add more materials to EPD database
5. **Regional variations:** Create region-specific parameter sets

---

## 📞 Support & Contact

For questions, issues, or collaboration:

- **Issues:** Please document any data quality issues or suggestions
- **Research collaboration:** This dataset is designed to support reproducible research
- **Extensions:** Feel free to build upon and extend this dataset for your research

---

## 📄 License

This dataset is provided for research and educational purposes.

**Usage Terms:**
- ✅ Free to use for academic research
- ✅ Free to modify and extend
- ✅ Free to share with attribution
- ⚠️  Not for commercial use without permission
- ⚠️  Provided "as is" without warranty

---

## 🎯 Roadmap for PhD Research

### Phase 1: Data Exploration (Weeks 1-4)
- [ ] Load and explore all datasets
- [ ] Visualize key patterns and relationships
- [ ] Validate data quality and consistency
- [ ] Identify interesting correlations

### Phase 2: Feature Engineering (Weeks 5-8)
- [ ] Create derived features from time series
- [ ] Engineer building-specific features
- [ ] Normalize and scale features
- [ ] Handle temporal features

### Phase 3: Model Development (Weeks 9-20)
- [ ] Energy consumption prediction models
- [ ] Retrofit recommendation systems
- [ ] Multi-objective optimization
- [ ] Carbon payback optimization

### Phase 4: Validation & Analysis (Weeks 21-28)
- [ ] Model validation and testing
- [ ] Sensitivity analysis
- [ ] Case study analysis
- [ ] Results interpretation

### Phase 5: Integration & Deployment (Weeks 29-36)
- [ ] Real-time decision support system
- [ ] Dashboard development
- [ ] API development
- [ ] Documentation and publication

---

## 📚 Recommended Reading

### Building Energy Modeling
- ISO 13790: Energy performance of buildings
- EN 15978: Sustainability of construction works
- ASHRAE Handbook - Fundamentals

### Lifecycle Assessment
- EN 15804: Environmental product declarations
- ISO 14040/14044: LCA principles and framework
- ILCD Handbook: LCA methodology

### Machine Learning for Buildings
- Literature on energy prediction models
- IoT and smart building systems
- Multi-objective optimization methods

---

**Version:** 1.0  
**Last Updated:** October 2024  
**Generated:** Automated dataset generation  
**Format:** CSV, Parquet, Excel, JSON  

---

*Happy researching! 🔬🏗️📊*
