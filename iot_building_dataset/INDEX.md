# 📁 IoT Building Dataset - File Index

## 📊 Data Files (CSV)

### Energy Consumption
- `energy/whole_building_energy.csv` (2.8 MB) - Whole-building energy meters
- `energy/sub_metered_energy.csv` (3.1 MB) - Circuit-level sub-metering

### Indoor Environmental Quality  
- `ieq/indoor_environmental_quality.csv` (46.5 MB) - Multi-zone IEQ sensors

### Occupancy & Usage
- `occupancy/occupancy_usage.csv` (3.2 MB) - People count and space utilization

### Weather Station
- `weather/weather_station.csv` (3.7 MB) - On-site meteorological data

### HVAC Systems
- `hvac_systems/hvac_operation.csv` (38.1 MB) - Multi-zone HVAC operation

## 📋 Configuration Files (JSON)

- `building_config.json` - Building physical parameters
- `dataset_metadata.json` - Complete dataset metadata

## 📖 Documentation Files (Markdown)

- `README.md` - Comprehensive user guide and dataset overview
- `DATA_DICTIONARY.md` - Detailed variable definitions and units
- `DATASET_SUMMARY.md` - Executive summary and quick reference
- `INDEX.md` - This file (file directory)

## 🐍 Python Scripts

- `generate_iot_dataset.py` - Data generation source code
- `analyze_dataset.py` - Comprehensive analysis and visualization tool
- `quick_start_example.py` - Getting started tutorial
- `requirements.txt` - Python package dependencies

## 📊 Visualizations (PNG)

- `analysis_plots/energy_consumption.png` - Energy patterns over time
- `analysis_plots/ieq_conditions.png` - IEQ metrics by zone
- `analysis_plots/weather_conditions.png` - Weather patterns
- `analysis_plots/occupancy_patterns.png` - Occupancy and utilization
- `analysis_plots/correlation_matrix.png` - Variable correlations
- `quick_start_example.png` - Quick start visualization

## 🗂️ Directory Structure

```
iot_building_dataset/
├── energy/              # Energy consumption data
├── ieq/                 # Indoor environmental quality
├── occupancy/           # Occupancy and usage patterns  
├── weather/             # Weather station data
├── hvac_systems/        # HVAC operation data
├── analysis_plots/      # Generated visualizations
├── raw/                 # (Reserved for raw data)
└── processed/           # (Reserved for processed data)
```

## 🚀 Quick Start Guide

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run quick example:**
   ```bash
   python quick_start_example.py
   ```

3. **Full analysis:**
   ```bash
   python analyze_dataset.py
   ```

4. **Read documentation:**
   - Start with `README.md`
   - Reference `DATA_DICTIONARY.md` for variables
   - See `DATASET_SUMMARY.md` for overview

## 📈 Dataset Statistics

- **Total Records:** 981,120
- **Total Size:** ~100 MB
- **Duration:** 365 days (2024)
- **Sampling:** 15-minute intervals
- **Data Streams:** 6 categories
- **Metrics:** 50+ variables
- **Zones:** 12 HVAC zones

## ✅ All Files Generated Successfully!
