# Building Static & Fabric Data (The "DNA" of the Building)
## Digital Twin Framework for Multi-Objective Building Retrofit Optimization

### Dataset Overview

This comprehensive dataset represents the complete static and fabric data for **Riverside Office Complex - Building A**, a 6-story commercial office building located in Cambridge, MA. The dataset is designed to support a Dynamic Digital Twin Framework that integrates real-time IoT monitoring, life-cycle assessment, and deep reinforcement learning for optimal building retrofit strategies.

**Building Profile:**
- **Building ID:** BLD-DT-2025-001
- **Type:** Commercial Office
- **Year Built:** 1985
- **Last Major Retrofit:** 2010
- **Gross Floor Area:** 8,450 m²
- **Number of Floors:** 6 + Basement
- **Current EUI:** 163.8 kWh/m²/year
- **Retrofit Potential:** 45% energy reduction

---

## 📁 Dataset Structure

```
building_dna_dataset/
├── metadata/
│   ├── building_info.json                    # General building information
│   ├── energy_model_inputs.json              # EnergyPlus simulation parameters
│   └── iot_sensor_framework.json             # IoT sensor deployment plan
│
├── geometric_data/
│   ├── bim_metadata.json                      # BIM model structure and metadata
│   ├── floor_plans_data.json                 # Detailed floor plan information
│   ├── lidar_data.json                        # LiDAR scan results and roof analysis
│   └── ifc_sample_extract.ifc                # Sample IFC building model extract
│
├── construction_materials/
│   ├── wall_assemblies.json                   # Wall construction details
│   ├── roof_assemblies.json                   # Roof system specifications
│   ├── floor_assemblies.json                  # Floor/foundation assemblies
│   ├── windows_doors.json                     # Fenestration specifications
│   └── material_properties_database.json     # Thermal & physical properties
│
├── systems/
│   ├── hvac/
│   │   ├── hvac_system_specifications.json   # HVAC equipment details
│   │   └── hvac_historical_performance.csv   # 12 months of operational data
│   ├── dhw/
│   │   └── dhw_system_specifications.json    # Domestic hot water systems
│   ├── lighting/
│   │   └── lighting_system_inventory.json    # Complete lighting inventory
│   └── renewable/
│       └── renewable_energy_systems.json     # Solar PV/thermal potential
│
└── environmental/
    └── air_tightness_data.json               # Blower door tests & IR surveys
```

---

## 🔑 Key Data Categories

### 1. **Geometric Data**

#### BIM Metadata (`bim_metadata.json`)
- Building volume and surface areas
- Floor-by-floor spatial data
- Thermal zone definitions
- Solar exposure analysis by orientation
- Coordinate system and georeferencing

#### Floor Plans (`floor_plans_data.json`)
- Detailed space-by-space breakdown
- Room dimensions and functions
- Occupancy capacities
- Ceiling heights

#### LiDAR Data (`lidar_data.json`)
- High-resolution roof geometry
- Equipment locations (HVAC, exhaust fans)
- Solar exposure analysis (shading, optimal PV placement)
- Surrounding context (adjacent buildings, vegetation)
- Facade analysis with thermal anomalies

**Key Metrics:**
- Total building volume: 32,490 m³
- Envelope surface area: 6,540 m²
- Annual solar radiation on south facade: 980 kWh/m²

---

### 2. **Construction & Material Data**

#### Wall Assemblies (`wall_assemblies.json`)
Three detailed wall types:
- **EXT-WALL-01:** Original exterior wall (North/East/West) - U-value: 0.61 W/m²K
- **EXT-WALL-02:** Renovated south facade - U-value: 0.36 W/m²K
- **INT-WALL-01:** Interior partitions

Each assembly includes:
- Layer-by-layer composition (6+ layers)
- Material thermal properties (conductivity, density, specific heat)
- Thermal bridging factors
- Embodied carbon and energy
- Current condition assessment

#### Roof Assembly (`roof_assemblies.json`)
- EPDM membrane with polyisocyanurate insulation (R-43)
- U-value: 0.162 W/m²K
- Drainage analysis
- Roof penetrations catalog (26 penetrations)

#### Floor Assemblies (`floor_assemblies.json`)
- Ground floor slab-on-grade with perimeter insulation
- Intermediate floors (post-tensioned concrete)
- Basement floor with ground coupling analysis

#### Windows & Doors (`windows_doors.json`)
- **Original windows** (145 units): U-value 3.6 W/m²K, SHGC 0.76 → Priority retrofit
- **Renovated windows** (68 units): U-value 1.8 W/m²K, SHGC 0.38
- Detailed glazing specifications (gas fill, coatings, spacers)
- Frame thermal properties
- Condition assessment and remaining life

#### Material Properties Database (`material_properties_database.json`)
45+ materials with comprehensive data:
- Thermal: conductivity, specific heat, thermal mass
- Physical: density, strength, moisture properties
- Environmental: embodied carbon, recyclability, service life
- Fire & acoustic properties where applicable

---

### 3. **HVAC Systems**

#### System Overview
- **Type:** Multi-zone VAV with hot water reheat
- **Heating:** Natural gas furnaces (AFUE 0.72, degraded from 0.80)
- **Cooling:** DX coils (Current EER 8.5, degraded from 10.2)
- **4x Rooftop Units** (20 years old, at end of design life)

#### Historical Performance Data (`hvac_historical_performance.csv`)
12 months of detailed operational data (2023-2024):
- Runtime hours (heating/cooling)
- Energy consumption
- Airflow rates
- Outdoor/supply/return temperatures
- Filter pressure drop trends
- Refrigerant pressures
- Compressor current
- Maintenance events

**Key Insights:**
- RTU-03 has highest energy consumption (efficiency degradation)
- Significant economizer usage opportunities
- Filter replacement every 3 months

#### Controls & Automation
- BACnet-based BAS (Johnson Controls Metasys)
- 385 control points
- DDC with demand-based ventilation
- Night setback and optimal start strategies

---

### 4. **Domestic Hot Water (DHW)**

2x Gas-fired storage water heaters (17 years old):
- **WH-01:** Fair condition, efficiency degraded to 0.56 (from 0.62)
- **WH-02:** Poor condition, tank leak detected, priority replacement
- Total capacity: 1,135 L
- Annual consumption: 125,000 kWh gas
- Significant standby losses (35% of energy)
- Recirculation system with timer controls

**Retrofit Opportunities:**
- Condensing water heaters: 52,500 kWh/year savings
- Solar thermal pre-heat: 56,250 kWh/year savings
- Heat pump integration: 78,125 kWh/year savings

---

### 5. **Lighting Systems**

#### Current Inventory (1,850 fixtures)
- **LED (40%):** 485 troffers + 285 downlights (good condition)
- **Fluorescent T8 (33%):** 620 fixtures (high maintenance, retrofit priority)
- **Incandescent (7%):** 125 fixtures (obsolete, immediate replacement)
- **Current LPD:** 19.8 W/m² (104% above ASHRAE 90.1-2019)

#### Controls
- Occupancy sensors: 42% coverage
- Daylight harvesting: South facade only (35% savings)
- BAS-integrated scheduling

#### Retrofit Potential
- T8 to LED retrofit: 244,070 kWh/year savings ($24,407, 2.1 year payback)
- Networked controls: Additional 30% savings
- Total lighting energy reduction: 72% achievable

---

### 6. **Renewable Energy Systems**

#### Solar PV Potential (Approved - Awaiting Funding)
- **System Size:** 161.5 kWp DC (150 kWp AC)
- **Annual Production:** 215,000 kWh (Year 1)
- **Technology:** 394x 410W monocrystalline PERC modules
- **Inverters:** 3x SolarEdge 50kW with power optimizers
- **Available Roof Area:** 950 m² (optimal), 1,060 m² (usable)
- **Performance Ratio:** 0.82
- **Shading Losses:** 3.2%
- **Financial:**
  - Net cost (after incentives): $179,175
  - Annual savings: $25,800
  - Simple payback: 6.9 years
  - 25-year NPV (7%): $185,000

#### Solar Thermal DHW Pre-heat
- **System Size:** 45 m² flat plate collectors
- **Annual Output:** 56,250 kWh thermal
- **Solar Fraction:** 45%
- **Payback:** 8.8 years

#### Battery Storage (300 kWh / 150 kW)
- Peak shaving and demand charge reduction
- $22,700/year savings
- Enhanced with solar for 35% self-consumption increase

#### Geothermal (Feasibility: Moderate)
- Vertical borehole system (48 boreholes × 150m)
- COP 4.2 (heating), EER 22 (cooling)
- High upfront cost ($1.25M), long payback (32.7 years)

---

### 7. **Air Tightness & Envelope Performance**

#### Blower Door Test Results (Sept 2022)
- **ACH50:** 2.84 (does not meet ASHRAE 189.1 requirement of 1.25)
- **Air flow at 50 Pa:** 92,400 m³/h
- **Equivalent leakage area:** 8,450 cm²
- **Annual infiltration heat loss:** 45,600 kWh

#### Major Leakage Paths (from IR thermography):
1. **Window perimeters** (74% of total leakage)
   - North facade: 28% contribution
   - West facade: 24% contribution  
   - East facade: 22% contribution
2. **Electrical penetrations:** 12%
3. **HVAC roof penetrations:** 8%
4. **Loading dock doors:** 4%

#### Thermal Anomalies (IR Survey Nov 2023)
- 47 anomalies detected
- 15 high severity (primarily west facade windows)
- Estimated total heat loss: 4,495 W
- Annual energy loss: 39,400 kWh ($4,728)

#### Air Sealing Potential
- **Achievable ACH50:** 0.89 (69% improvement)
- **Total investment:** $334,000
- **Annual savings:** 54,500 kWh
- **Payback:** 6.1 years

---

### 8. **Energy Model Integration**

The dataset includes complete inputs for energy simulation:

#### EnergyPlus Model Parameters (`energy_model_inputs.json`)
- **Calibrated baseline model** (CV-RMSE: 8.5% electric, 12.3% gas)
- Detailed envelope constructions
- HVAC system curves and efficiencies
- Internal loads and schedules
- 6 thermal zones

#### Retrofit Scenarios Defined
1. **Envelope:** Window + insulation (28% heating, 18% cooling reduction)
2. **HVAC:** High-efficiency RTUs (42% HVAC energy reduction)
3. **Lighting:** LED + controls (72% lighting reduction)
4. **Renewables:** 161.5 kWp solar PV
5. **Integrated Deep Retrofit:** 78% total energy reduction, net-zero feasible

---

### 9. **IoT Sensor Framework**

#### Planned Deployment: 385 Sensors

**Environmental Monitoring:**
- 48× Temperature/humidity sensors (Sensirion SHT31)
- 24× CO₂ sensors (Vaisala GMP252)
- 1× Weather station (roof-mounted)

**Energy Metering:**
- 18× Electrical submeters (Schneider PowerLogic PM8000)
- 1× Ultrasonic gas meter
- 8× Thermal energy meters (BTU meters)

**HVAC Monitoring:**
- 32× Air flow sensors
- 12× Static pressure sensors
- 16× Refrigerant pressure/temperature
- 8× Filter differential pressure

**Occupancy & Comfort:**
- 185× PIR occupancy sensors
- 12× People counting (thermal imaging)
- 240× Desk occupancy sensors
- 68× Daylight sensors

**Indoor Air Quality:**
- 24× VOC sensors
- 18× PM2.5/PM10 sensors

**Envelope Performance:**
- 45× Surface temperature sensors
- 24× Condensation detectors

#### Data Architecture
- **Edge Computing:** 6× Dell Edge Gateway 5100 (one per floor)
- **Communication:** MQTT over TLS, BACnet/IP, Modbus TCP
- **Storage:** InfluxDB (time-series), PostgreSQL (metadata)
- **Analytics:** Apache Spark stream processing
- **ML Integration:** Fault detection (Isolation Forest + LSTM)

#### Digital Twin Synchronization
- Update frequency: 60 seconds
- 385 state variables tracked
- EnergyPlus co-simulation
- Bayesian model calibration every 7 days
- 72-hour forecast horizon

---

## 📊 Key Performance Indicators

### Current Performance
- **Total Energy Consumption:** 1,250,000 kWh/year
- **Heating Load:** 420,000 kWh/year (33.6%)
- **Cooling Load:** 315,000 kWh/year (25.2%)
- **Lighting Load:** 285,000 kWh/year (22.8%)
- **Plug Loads:** 230,000 kWh/year (18.4%)
- **EUI:** 163.8 kWh/m²/year
- **Peak Demand:** 385 kW (heating), 295 kW (cooling)

### Envelope Performance
- **Average Wall U-value:** 0.45 W/m²K
- **Average Roof U-value:** 0.28 W/m²K (after 2015 upgrade)
- **Average Window U-value:** 2.8 W/m²K (mixed original/renovated)
- **Window-to-Wall Ratio:** 0.32 (32%)
- **Infiltration:** ACH50 2.84 (requires improvement)

### System Efficiencies (Current/Degraded)
- **HVAC Heating:** AFUE 0.72 (was 0.80 rated)
- **HVAC Cooling:** EER 8.5 (was 10.2 rated)
- **DHW:** Energy Factor 0.55 (was 0.62 rated)
- **Lighting:** 19.8 W/m² LPD (vs. 9.7 W/m² ASHRAE 90.1-2019)

---

## 🎯 Retrofit Optimization Objectives

This dataset supports multi-objective optimization for:

1. **Energy Reduction:** Minimize operational energy consumption
2. **Carbon Reduction:** Minimize embodied + operational carbon
3. **Cost Optimization:** Minimize lifecycle cost (capital + operating)
4. **Comfort Improvement:** Maximize thermal comfort and IAQ
5. **Resilience:** Improve building resilience to climate change

### Recommended Integrated Retrofit Package

| Measure | Investment | Annual Savings | Payback | Energy Reduction |
|---------|-----------|----------------|---------|------------------|
| Window Replacement (N/E/W) | $285,000 | $7,000 | 40.7 yr | 35,000 kWh |
| HVAC RTU Replacement | $380,000 | $19,000 | 20.0 yr | 95,000 kWh |
| Lighting LED Retrofit | $127,750 | $30,165 | 4.2 yr | 251,720 kWh |
| Air Sealing Package | $334,000 | $6,540 | 51.1 yr | 54,500 kWh |
| Solar PV (161.5 kW) | $179,175 | $25,800 | 6.9 yr | 215,000 kWh* |
| **TOTAL** | **$1,305,925** | **$88,505** | **14.8 yr** | **651,220 kWh (52%)** |

*Solar PV is generation, not reduction

**Combined Performance:**
- **New EUI:** 78.5 kWh/m²/year (52% reduction)
- **Net Energy:** -136,220 kWh/year (with solar → 17% net positive)
- **CO₂ Reduction:** 280,000 kg/year
- **Path to Net Zero:** Feasible with additional 50 kWp solar + 100 kWh storage

---

## 🔬 Research Applications

This dataset is designed for:

### 1. **Building Energy Modeling**
- Calibrated baseline creation
- Retrofit scenario simulation
- Uncertainty quantification
- Model predictive control

### 2. **Life-Cycle Assessment (LCA)**
- Embodied carbon analysis
- Operational carbon tracking
- Whole-building LCA
- Circular economy strategies

### 3. **Machine Learning & AI**
- Deep reinforcement learning for control optimization
- Fault detection and diagnostics
- Predictive maintenance
- Occupancy prediction
- Energy forecasting

### 4. **Digital Twin Development**
- Physics-based model integration
- Real-time state estimation
- What-if scenario analysis
- Performance degradation modeling

### 5. **Multi-Objective Optimization**
- Genetic algorithms (NSGA-II, NSGA-III)
- Particle swarm optimization
- Bayesian optimization
- Cost-optimal retrofit packages

---

## 📖 Data Standards & Formats

### File Formats
- **JSON:** Structured building data, specifications, metadata
- **CSV:** Time-series performance data
- **IFC:** Building Information Modeling (IFC4 schema)

### Units
- **Length:** meters (m)
- **Area:** square meters (m²)
- **Volume:** cubic meters (m³)
- **Temperature:** Celsius (°C)
- **Energy:** kilowatt-hours (kWh)
- **Power:** kilowatts (kW) or watts (W)
- **Pressure:** Pascals (Pa)
- **Thermal Resistance:** m²K/W
- **Thermal Conductivity:** W/mK

### Coordinate System
- **EPSG:26986** - NAD83 / Massachusetts Mainland
- Building origin: (42.3736°N, 71.1097°W)
- Elevation: 15.2m above sea level

---

## 🔍 Data Quality & Validation

### Data Sources
- **BIM Model:** Autodesk Revit 2024 (LOD 350)
- **Energy Audit:** Conducted June 2023
- **Blower Door Test:** ASTM E779-19 compliant (Sept 2022)
- **LiDAR Scan:** Leica RTC360 (March 2024, 3mm accuracy)
- **IR Thermography:** FLIR T1020 (Nov 2023, ΔT=18.5°C)
- **Utility Bills:** 24 months validated data
- **Maintenance Records:** 10 years CMMS data

### Calibration & Validation
- Energy model CV-RMSE: 8.5% (electric), 12.3% (gas)
- NMBE: -3.2% (electric), 5.8% (gas)
- Meets ASHRAE Guideline 14 criteria

---

## 📚 References & Standards

### Standards Referenced
- **ASHRAE 90.1-2019:** Energy Standard for Buildings
- **ASHRAE 189.1-2020:** High-Performance Green Buildings
- **ASHRAE Guideline 14:** Measurement of Energy Demand and Savings
- **ASTM E779-19:** Air Leakage Testing
- **IFC4:** Industry Foundation Classes (ISO 16739)
- **ISO 10456:2007:** Building materials - Thermal properties

### Related Research
- Digital Twin frameworks for building optimization
- Deep reinforcement learning for HVAC control
- Multi-objective retrofit optimization (Pareto frontiers)
- Building-to-grid integration strategies
- Embodied vs. operational carbon tradeoffs

---

## 💡 Usage Examples

### Example 1: Energy Model Creation
```python
import json

# Load building geometry
with open('geometric_data/bim_metadata.json') as f:
    geometry = json.load(f)

# Load envelope properties
with open('construction_materials/wall_assemblies.json') as f:
    walls = json.load(f)

# Extract U-values for energy model
wall_u_values = {
    assembly['assembly_id']: assembly['performance_characteristics']['u_value_with_thermal_bridging_w_m2k']
    for assembly in walls['wall_assemblies']
}
```

### Example 2: HVAC Performance Analysis
```python
import pandas as pd

# Load HVAC historical data
hvac_data = pd.read_csv('systems/hvac/hvac_historical_performance.csv')

# Calculate monthly efficiency
hvac_data['heating_efficiency'] = (
    hvac_data['heating_energy_kwh'] / 
    (hvac_data['gas_consumption_m3'] * 10.55)  # kWh per m³ gas
)

# Identify degradation trends
efficiency_trend = hvac_data.groupby('rtu_id')['heating_efficiency'].mean()
```

### Example 3: Retrofit NPV Calculation
```python
# Load retrofit scenarios
with open('systems/renewable/renewable_energy_systems.json') as f:
    renewable = json.load(f)

solar_pv = renewable['solar_pv_potential']['financial_analysis']
npv = solar_pv['npv_usd_7percent']
irr = solar_pv['irr_percent']
payback = solar_pv['simple_payback_years']

print(f"Solar PV: NPV=${npv:,}, IRR={irr}%, Payback={payback} years")
```

---

## 📝 License & Citation

### License
This dataset is provided for research and educational purposes. Commercial use requires permission.

### Citation
If you use this dataset in your research, please cite:

```
Riverside Office Complex Building DNA Dataset (2024)
Digital Twin Framework for Multi-Objective Building Retrofit Optimization
Version 2024.1
https://github.com/building-digital-twin/building-dna-dataset
```

---

## 📧 Contact & Support

For questions, issues, or collaboration:
- **Dataset Maintainer:** Building Digital Twin Research Team
- **Institution:** Advanced Building Systems Laboratory
- **Email:** building-dt-support@example.edu
- **GitHub Issues:** https://github.com/building-digital-twin/building-dna-dataset/issues

---

## 🔄 Version History

### Version 2024.1 (Current)
- Initial comprehensive dataset release
- 385 IoT sensor framework defined
- Complete static building DNA
- Calibrated energy model inputs
- 5 retrofit scenarios with LCA data

### Planned Updates
- **2024.2:** Real-time IoT data integration (6 months)
- **2024.3:** Reinforcement learning control results
- **2025.1:** Post-retrofit performance validation

---

## 🎓 Educational Use

This dataset is ideal for:
- **Graduate courses:** Building energy modeling, retrofit optimization, digital twins
- **Workshops:** Hands-on building physics, LCA, ML for buildings
- **Hackathons:** Energy prediction challenges, control optimization
- **Research projects:** Multi-objective optimization, uncertainty analysis

Sample assignments, tutorials, and Jupyter notebooks available in `/examples` directory.

---

**Last Updated:** October 16, 2025  
**Dataset Version:** 2024.1  
**Total Dataset Size:** ~15 MB (JSON/CSV), ~500 GB (with full point cloud data)
