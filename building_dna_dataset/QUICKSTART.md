# Quick Start Guide - Building DNA Dataset

## 🚀 Getting Started in 5 Minutes

### Step 1: Explore the Dataset Structure

```bash
cd building_dna_dataset
ls -la
```

You'll see the following structure:
- `metadata/` - Building information and model inputs
- `geometric_data/` - BIM, floor plans, LiDAR scans
- `construction_materials/` - Wall, roof, floor, window assemblies
- `systems/` - HVAC, DHW, lighting, renewable systems
- `environmental/` - Air tightness and blower door tests

### Step 2: Load Building Metadata

```python
import json

# Load main building information
with open('metadata/building_info.json', 'r') as f:
    building = json.load(f)

print(f"Building: {building['building_name']}")
print(f"Floor Area: {building['general_characteristics']['gross_floor_area_m2']} m²")
print(f"EUI: {building['energy_summary']['eui_kwh_m2_year']} kWh/m²/year")
```

### Step 3: Analyze HVAC Performance

```python
import pandas as pd

# Load HVAC historical data
hvac_data = pd.read_csv('systems/hvac/hvac_historical_performance.csv')

# Calculate average monthly energy by unit
monthly_energy = hvac_data.groupby('rtu_id')[['heating_energy_kwh', 'cooling_energy_kwh']].sum()
print(monthly_energy)
```

### Step 4: Run Visualizations

```bash
# Install dependencies
pip install pandas matplotlib seaborn plotly numpy

# Run visualization script
python visualization_example.py
```

This will generate 4 PNG visualizations:
- `envelope_performance.png` - Window U-values and solar heat gain by orientation
- `hvac_performance.png` - HVAC efficiency degradation over time
- `energy_breakdown.png` - Energy consumption by end-use
- `retrofit_scenarios.png` - Cost-benefit analysis of retrofit measures

### Step 5: Access Specific Data

#### Get Window Specifications
```python
with open('construction_materials/windows_doors.json', 'r') as f:
    windows = json.load(f)

# Original windows (poor performance)
original = windows['window_types'][0]
print(f"U-value: {original['performance_characteristics']['u_value_total_w_m2k']} W/m²K")
print(f"SHGC: {original['performance_characteristics']['shgc']}")
```

#### Get Material Thermal Properties
```python
with open('construction_materials/material_properties_database.json', 'r') as f:
    materials = json.load(f)

# Fiberglass insulation properties
fiberglass = materials['insulation_materials'][0]
print(f"Conductivity: {fiberglass['thermal_properties']['conductivity_w_mk']} W/mK")
print(f"R-value per inch: {fiberglass['thermal_properties']['r_value_per_inch']}")
```

#### Get Solar PV Potential
```python
with open('systems/renewable/renewable_energy_systems.json', 'r') as f:
    renewable = json.load(f)

pv = renewable['solar_pv_potential']
print(f"System Size: {pv['system_sizing']['system_capacity_dc_kw']} kW")
print(f"Annual Production: {pv['energy_production_estimate']['year_1_production_kwh']} kWh")
print(f"Payback: {pv['financial_analysis']['simple_payback_years']} years")
```

---

## 📊 Common Use Cases

### Use Case 1: Energy Model Input Generation

```python
import json

# Load energy model inputs
with open('metadata/energy_model_inputs.json', 'r') as f:
    model_inputs = json.load(f)

# Extract envelope U-values for simulation
envelope = model_inputs['envelope_inputs']
print("Envelope U-values:")
for component, data in envelope.items():
    if isinstance(data, dict) and 'u_value_w_m2k' in data:
        print(f"  {component}: {data['u_value_w_m2k']} W/m²K")
```

### Use Case 2: Retrofit Optimization Analysis

```python
# Compare all retrofit scenarios
scenarios = {}

# Windows
with open('construction_materials/windows_doors.json', 'r') as f:
    windows = json.load(f)
    scenarios['Windows'] = {
        'cost': windows['retrofit_recommendations']['estimated_cost_usd'],
        'savings_kwh': windows['retrofit_recommendations']['estimated_energy_savings_kwh_year'],
        'payback': windows['retrofit_recommendations']['simple_payback_years']
    }

# HVAC
with open('systems/hvac/hvac_system_specifications.json', 'r') as f:
    hvac = json.load(f)
    scenarios['HVAC'] = {
        'cost': hvac['retrofit_opportunities']['priority_1']['estimated_cost_usd'],
        'savings_kwh': hvac['retrofit_opportunities']['priority_1']['estimated_savings_kwh_year'],
        'payback': hvac['retrofit_opportunities']['priority_1']['simple_payback_years']
    }

# Sort by payback period
sorted_scenarios = sorted(scenarios.items(), key=lambda x: x[1]['payback'])
for name, data in sorted_scenarios:
    print(f"{name}: ${data['cost']:,} → {data['savings_kwh']:,} kWh/year → {data['payback']:.1f} yr payback")
```

### Use Case 3: Life-Cycle Carbon Analysis

```python
# Calculate embodied carbon for wall assemblies
with open('construction_materials/wall_assemblies.json', 'r') as f:
    walls = json.load(f)

for assembly in walls['wall_assemblies']:
    if assembly['wall_type'] == 'External':
        total_carbon = assembly['performance_characteristics']['total_embodied_carbon_kgco2_m2']
        area = walls['wall_inventory']['exterior_walls'][f"assembly_{assembly['assembly_id'].replace('-', '_')}"]["area_m2"]
        print(f"{assembly['assembly_name']}: {total_carbon * area / 1000:.1f} tonnes CO2e")
```

### Use Case 4: IoT Sensor Deployment Planning

```python
# Get sensor counts and locations
with open('metadata/iot_sensor_framework.json', 'r') as f:
    iot = json.load(f)

print("Planned IoT Deployment:")
for category, sensors in iot['sensor_categories'].items():
    print(f"\n{category.replace('_', ' ').title()}:")
    for sensor_type, details in sensors.items():
        if 'quantity' in details:
            print(f"  - {details['quantity']}x {details.get('sensor_type', sensor_type)}")
```

---

## 🔍 Data Exploration Tips

### 1. Check Data Completeness
All files are in JSON format (except HVAC historical CSV). Use Python's `json` library or any text editor with JSON support.

### 2. Understand Units
All units follow SI standards:
- Length: meters (m)
- Energy: kilowatt-hours (kWh)
- Temperature: Celsius (°C)
- Thermal properties: W/mK, W/m²K, m²K/W

See `metadata/data_dictionary.json` for complete unit definitions.

### 3. Cross-Reference Data
Many files reference each other. For example:
- `building_info.json` → Overall building characteristics
- `bim_metadata.json` → Thermal zones referenced in HVAC
- `material_properties_database.json` → Materials used in wall assemblies

### 4. Time-Series Data
HVAC historical performance CSV includes:
- 12 months of operational data
- 4 RTU units tracked
- Monthly energy consumption, temperatures, pressures
- Maintenance events logged

---

## 📚 Key Files Reference

| File | Description | Key Data |
|------|-------------|----------|
| `metadata/building_info.json` | Master building info | EUI, floor area, energy summary |
| `geometric_data/bim_metadata.json` | BIM structure | Thermal zones, volumes, solar exposure |
| `construction_materials/wall_assemblies.json` | Wall details | U-values, layers, embodied carbon |
| `construction_materials/windows_doors.json` | Fenestration | U-values, SHGC, VT, condition |
| `systems/hvac/hvac_historical_performance.csv` | HVAC time-series | Energy, efficiency, maintenance |
| `systems/renewable/renewable_energy_systems.json` | Solar PV/thermal | Capacity, production, payback |
| `environmental/air_tightness_data.json` | Blower door tests | ACH50, leakage areas, IR survey |

---

## 💡 Next Steps

1. **Review the README.md** for comprehensive dataset documentation
2. **Check data_dictionary.json** for field definitions and units
3. **Run visualization_example.py** to generate charts
4. **Explore energy_model_inputs.json** for EnergyPlus simulation parameters
5. **Review iot_sensor_framework.json** for digital twin IoT integration

---

## 🐛 Troubleshooting

**Q: JSON file won't load**
```python
# Use this pattern for robust loading
import json
try:
    with open('path/to/file.json', 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
```

**Q: Missing pandas/matplotlib**
```bash
pip install -r requirements.txt
```

**Q: Can't find a specific data point**
Check `metadata/data_dictionary.json` for field definitions and search across files:
```bash
grep -r "u_value" building_dna_dataset/
```

---

## 📧 Support

For questions or issues:
- Check the main **README.md** for detailed documentation
- Review **data_dictionary.json** for field definitions
- See example code in **visualization_example.py**

Happy analyzing! 🏢📊
