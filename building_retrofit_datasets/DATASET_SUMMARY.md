# Dataset Generation Summary

**Generated:** October 14, 2024  
**Total Size:** 71 MB  
**Buildings:** 50  
**Time Period:** Full year 2023 (Jan 1 - Dec 31)  
**IoT Records:** 438,000 (hourly resolution)

---

## ✅ Generated Datasets

### Core Datasets

| File | Size | Records | Description |
|------|------|---------|-------------|
| `building_attributes.csv` | 15 KB | 50 | Complete building fabric, geometric, and thermal properties |
| `iot_sensor_data.parquet` | 14 MB | 438,000 | Hourly IoT sensor readings (energy, IEQ, weather, occupancy) |
| `iot_sensor_data.csv` | 53 MB | 438,000 | Same as above in CSV format |
| `energy_performance_historical.csv` | 20 KB | 250 | 5 years of annual energy consumption (2019-2023) |
| `retrofit_scenarios.csv` | 103 KB | 150 | Retrofit analysis with costs, savings, and ROI |
| `lca_building_baseline.csv` | 95 KB | 50 | Building lifecycle assessment baselines |
| `lca_retrofit_measures.csv` | 49 KB | 404 | LCA for individual retrofit measures |
| `lca_carbon_payback.csv` | 31 KB | 404 | Carbon payback analysis |

### Integrated Datasets

| File | Size | Records | Description |
|------|------|---------|-------------|
| `integrated_building_master.csv` | 23 KB | 50 | Comprehensive building profiles with all metrics |
| `integrated_retrofit_analysis.csv` | 118 KB | 150 | Complete retrofit analysis (energy + cost + LCA) |
| `ml_ready_dataset.parquet` | 486 KB | 18,250 | ML-ready feature dataset (daily aggregation) |
| `timeseries_daily.csv` | 1.7 MB | 18,250 | Daily aggregated IoT data |
| `timeseries_monthly.csv` | 59 KB | 600 | Monthly aggregated data |
| `hourly_patterns.csv` | 57 KB | 1,200 | Average hourly consumption patterns |

### Reference Databases (JSON)

| File | Size | Description |
|------|------|-------------|
| `material_properties_database.json` | 2.9 KB | Wall, roof, window material properties |
| `retrofit_measures_database.json` | 1.4 KB | Retrofit measure specifications |
| `epd_database.json` | 3.0 KB | Environmental Product Declarations |
| `hvac_lca_database.json` | 1.4 KB | HVAC system LCA data |

### Metadata

| File | Description |
|------|-------------|
| `data_quality_report.json` | Data quality metrics and statistics |
| `integration_summary.json` | Integration metadata and coverage info |

---

## 📊 Dataset Statistics

### Building Stock Composition

- **Industrial:** 15 buildings (30%)
- **Commercial:** 14 buildings (28%)
- **Educational:** 13 buildings (26%)
- **Residential:** 8 buildings (16%)

### Energy Performance Certificate (EPC) Distribution

| Rating | Count | Percentage |
|--------|-------|------------|
| A | 7 | 14% |
| B | 5 | 10% |
| C | 3 | 6% |
| D | 4 | 8% |
| E | 3 | 6% |
| F | 1 | 2% |
| G | 27 | 54% |

### Construction Era

- **Oldest building:** 1923 (101 years old)
- **Newest building:** 2020 (4 years old)
- **Average age:** 53.7 years

### Thermal Performance

- **Average envelope U-value:** 1.648 W/m²K
- **Best U-value:** 0.274 W/m²K (modern, well-insulated)
- **Worst U-value:** 3.523 W/m²K (old, uninsulated)

### Energy Consumption (2023)

- **Average annual consumption:** 530,572 kWh per building
- **Range:** 8,156 kWh to 2,010,248 kWh
- **Total dataset energy:** ~26.5 GWh/year

### Retrofit Potential

**Deep Retrofit Scenario:**
- **Average energy savings:** 62.4%
- **Average carbon savings:** 64.8%
- **Average investment cost:** €286,721
- **Average ROI:** 171%
- **Average payback period:** 8.1 years

### Lifecycle Assessment

**Building Baseline:**
- **Average embodied carbon:** 372,634 kg CO₂eq per building
- **Average embodied carbon per m²:** 175.2 kg CO₂eq/m²
- **Average embodied energy:** 3,244,856 MJ per building

**Carbon Payback by Measure:**
- Roof insulation: 0.37 years
- Wall insulation: 0.40 years
- HVAC upgrade (heat pump): 0.79 years
- Window replacement: 1.62 years
- Solar PV: 3.80 years

---

## 🎯 Key Research Applications

This comprehensive dataset enables research in:

1. **Energy Performance Prediction**
   - Time series forecasting
   - Weather-normalized consumption
   - Building-specific consumption patterns

2. **Retrofit Optimization**
   - Cost-benefit analysis
   - Multi-objective optimization (energy, carbon, cost)
   - Measure prioritization

3. **Indoor Environmental Quality**
   - CO₂, TVOC, PM2.5 monitoring
   - Correlation with occupancy and energy use
   - Thermal comfort analysis

4. **Lifecycle Assessment**
   - Embodied vs operational carbon
   - Carbon payback calculations
   - Material selection optimization

5. **Machine Learning Applications**
   - Energy consumption prediction
   - Anomaly detection
   - Retrofit recommendation systems
   - Building performance classification

6. **IoT Analytics**
   - Sensor data fusion
   - Occupancy pattern recognition
   - Real-time optimization
   - Predictive maintenance

---

## 📖 Quick Start Examples

### Load and Explore Data

```python
import pandas as pd
import numpy as np

# Load integrated building master
buildings = pd.read_csv('data/integrated_building_master.csv')
print(f"Total buildings: {len(buildings)}")
print(f"Average U-value: {buildings['envelope_avg_u_value'].mean():.3f} W/m²K")

# Load IoT data (use Parquet for speed)
iot = pd.read_parquet('data/iot_sensor_data.parquet')
iot['timestamp'] = pd.to_datetime(iot['timestamp'])
print(f"Total IoT records: {len(iot):,}")

# Load retrofit analysis
retrofits = pd.read_csv('data/integrated_retrofit_analysis.csv')
print(f"Average ROI: {retrofits['roi_percent'].mean():.1f}%")
```

### Analyze Energy Consumption

```python
# Energy consumption by building type
consumption_by_type = iot.groupby('building_type')['total_energy_consumption_kwh'].sum()
print(consumption_by_type.sort_values(ascending=False))

# Monthly patterns
iot['month'] = iot['timestamp'].dt.month
monthly = iot.groupby('month')['total_energy_consumption_kwh'].mean()
monthly.plot(kind='line', title='Average Hourly Consumption by Month')
```

### Retrofit Analysis

```python
# Best retrofit for each building
best_retrofits = retrofits.loc[
    retrofits.groupby('building_id')['roi_percent'].idxmax()
]

print(f"Average investment: €{best_retrofits['total_cost_eur'].mean():,.0f}")
print(f"Average savings: {best_retrofits['total_energy_saving_pct'].mean():.1f}%")
print(f"Average payback: {best_retrofits['simple_payback_years'].mean():.1f} years")
```

### Machine Learning

```python
# Load ML-ready dataset
ml_data = pd.read_parquet('data/ml_ready_dataset.parquet')

# Features and target
features = ['outdoor_temperature', 'occupancy_ratio', 'envelope_avg_u_value']
X = ml_data[features]
y = ml_data['total_energy_consumption_kwh']

# Train-test split and model training...
```

---

## 📁 File Locations

All datasets are located in: `/workspace/building_retrofit_datasets/data/`

### Essential Files for Most Research:

1. **`integrated_building_master.xlsx`** - Start here for building overview
2. **`ml_ready_dataset.parquet`** - For machine learning projects
3. **`integrated_retrofit_analysis.xlsx`** - For retrofit optimization
4. **`iot_sensor_data.parquet`** - For time series analysis

### File Format Recommendations:

- **For Python/ML:** Use `.parquet` files (faster, smaller)
- **For Excel/manual inspection:** Use `.xlsx` or `.csv` files
- **For reference:** Check `.json` files for material/measure properties

---

## ⚠️ Important Notes

1. **Synthetic Data:** This is fabricated data for research purposes. While based on realistic models and distributions, it should be validated with real data for production applications.

2. **Data Quality:** All core datasets have 100% completeness (no missing values). Quality metrics available in `data_quality_report.json`.

3. **Temporal Coverage:** IoT data covers full year 2023 at hourly resolution. Energy historical data covers 2019-2023 annually.

4. **Consistency:** All datasets are cross-referenced by `building_id` and have been validated for consistency.

5. **Updates:** Material properties, costs, and LCA values are based on 2023 European standards.

---

## 🚀 Next Steps

1. **Explore the data:**
   ```bash
   cd /workspace/building_retrofit_datasets
   python3  # or jupyter notebook
   ```

2. **Review the main README:**
   - Open `README.md` for comprehensive documentation
   - Check usage examples and API references

3. **Start your analysis:**
   - Load `integrated_building_master.csv` for overview
   - Use `ml_ready_dataset.parquet` for ML models
   - Analyze `integrated_retrofit_analysis.csv` for retrofit insights

4. **Customize as needed:**
   - Modify generator scripts in `scripts/` directory
   - Adjust building count, time period, or parameters
   - Re-run `python3 scripts/generate_all_datasets.py`

---

## 📞 Support

For questions or issues:
- Check `README.md` for detailed documentation
- Review `data_quality_report.json` for data statistics
- Examine generator scripts in `scripts/` for methodology

---

**Dataset Version:** 1.0  
**Generation Date:** October 14, 2024  
**Total Generation Time:** ~2 minutes  
**Status:** ✅ Complete and ready for use

*Happy researching! 🔬📊🏗️*
