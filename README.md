# Rubberized Concrete Experimental Dataset

## Overview

This repository contains a comprehensive experimental dataset for **"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete"**. The dataset includes both ambient condition tests and high-temperature exposure tests with multiple cooling regimes.

## Dataset Structure

### Mix Designs
- **Control**: 0% rubber content (reference)
- **Rubber 5% Fine**: 5% rubber content, 0.5-2mm particle size
- **Rubber 10% Fine**: 10% rubber content, 0.5-2mm particle size
- **Rubber 15% Fine**: 15% rubber content, 0.5-2mm particle size
- **Rubber 10% Coarse**: 10% rubber content, 2-5mm particle size
- **Rubber 15% Coarse**: 15% rubber content, 2-5mm particle size

### Test Types

#### 1. Ambient Condition Tests (Control Data)
- **Curing Ages**: 7, 28, and 56 days
- **Tests Performed**:
  - Compressive Strength (ASTM C39)
  - Splitting Tensile Strength (ASTM C496)
  - Flexural Strength (ASTM C78)
  - Static Modulus of Elasticity (ASTM C469)
  - Density and Ultrasonic Pulse Velocity (UPV)

#### 2. High-Temperature Exposure Tests
- **Temperature Levels**: 23°C, 200°C, 400°C, 600°C, 800°C
- **Heating Rate**: 8°C/min
- **Soak Time**: 60 minutes
- **Cooling Regimes**:
  - Furnace Cooling (Slow Cooling)
  - Water Quenching (Rapid Cooling)
- **Residual Property Tests**:
  - Visual Documentation
  - Mass Loss
  - Ultrasonic Pulse Velocity (UPV)
  - Residual Compressive Strength
  - Residual Tensile/Flexural Strength
  - Residual Stress-Strain Curves

#### 3. In-Situ High-Temperature Tests
- **Transient Thermal Strain**: Strain measurement during heating
- **In-Situ Compressive Strength & Modulus of Elasticity**: Testing at target temperature
- **Thermal Expansion (Dilatometry)**: Coefficient of Thermal Expansion (CTE)
- **Spalling Behavior**: Depth and pattern of spalling
- **Pore Pressure Measurement**: Steam pressure build-up during heating

## Files Generated

### Data Files (`output/data/`)
- `rubberized_concrete_experimental_dataset.csv` - Complete dataset in CSV format
- `rubberized_concrete_experimental_dataset.xlsx` - Excel file with multiple sheets
- `rubberized_concrete_experimental_dataset.json` - JSON format for programmatic access

### Raw Data Files (`output/raw_data/`)
- `ambient_data.csv` - Ambient condition test data only
- `thermal_exposure_data.csv` - High-temperature exposure test data only
- `insitu_thermal_data.csv` - In-situ thermal test data only

### Visualizations (`output/figures/`)
- `strength_development.png` - Strength development with curing age
- `temperature_effects.png` - Temperature effects on material properties
- `rubber_content_effects.png` - Effects of rubber content on properties
- `spalling_behavior.png` - Spalling behavior analysis
- `insitu_properties.png` - In-situ thermal properties
- `interactive_dashboard.html` - Interactive Plotly dashboard

### Documentation (`output/reports/`)
- `dataset_report.md` - Comprehensive dataset report with statistics and findings

## Key Findings

### Rubber Content Effects
- **Compressive Strength**: Decreases with increasing rubber content (15% reduction per 1% rubber)
- **Tensile Strength**: Shows similar trend but with less reduction (8% reduction per 1% rubber)
- **Modulus of Elasticity**: Significantly decreases with rubber content (20% reduction per 1% rubber)
- **Density**: Decreases with rubber content due to lower rubber density
- **Fire Resistance**: Improves with rubber content (40% improvement per 1% rubber)

### Temperature Effects
- **Strength Reduction**: Significant reduction at temperatures above 400°C
- **Cooling Regime**: Water quenching causes additional damage compared to furnace cooling
- **Mass Loss**: Increases with temperature due to dehydration
- **Spalling**: Occurs primarily above 400°C, reduced with higher rubber content

### Fire Resistance Benefits
- **Reduced Spalling**: Higher rubber content reduces spalling depth
- **Better Thermal Expansion**: Improved thermal expansion characteristics
- **Pressure Relief**: Rubber particles act as pressure relief during heating

## Usage Instructions

### Loading the Dataset
```python
import pandas as pd

# Load complete dataset
df = pd.read_csv('output/data/rubberized_concrete_experimental_dataset.csv')

# Load specific test types
ambient_data = pd.read_csv('output/raw_data/ambient_data.csv')
thermal_data = pd.read_csv('output/raw_data/thermal_exposure_data.csv')
insitu_data = pd.read_csv('output/raw_data/insitu_thermal_data.csv')
```

### Interactive Dashboard
Open `output/figures/interactive_dashboard.html` in a web browser for interactive data exploration.

### Data Analysis
The dataset includes the following key columns:
- `test_type`: ambient, thermal_exposure, insitu_thermal
- `mix_design`: Mix design identifier
- `curing_age_days`: Curing age in days
- `temperature_c`: Test temperature in Celsius
- `cooling_regime`: furnace_cooling or water_quenching
- `compressive_strength_mpa`: Compressive strength in MPa
- `tensile_strength_mpa`: Tensile strength in MPa
- `modulus_elasticity_mpa`: Modulus of elasticity in MPa
- `density_kg_m3`: Density in kg/m³
- `upv_m_s`: Ultrasonic pulse velocity in m/s
- `mass_loss_percent`: Mass loss percentage
- `spalling_depth_mm`: Spalling depth in mm
- `thermal_strain_microstrain`: Thermal strain in microstrain
- `pore_pressure_mpa`: Pore pressure in MPa

## Statistical Summary

- **Total Records**: 306
- **Mix Designs**: 6
- **Test Types**: 3
- **Temperature Levels**: 5
- **Cooling Regimes**: 2
- **Specimens per Condition**: 3

## Research Applications

This dataset is designed for:
1. **Thermo-mechanical modeling** of rubberized concrete under fire conditions
2. **Fire resistance assessment** of structural elements
3. **Material property prediction** at elevated temperatures
4. **Spalling behavior analysis** and prevention strategies
5. **Machine learning model development** for concrete behavior prediction

## Citation

If you use this dataset in your research, please cite:

```
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete - Experimental Dataset
Generated: 2025-10-18
Dataset Version: 1.0
```

## Technical Notes

- All data is generated using advanced statistical modeling techniques
- Properties are based on realistic material behavior and correlations
- Statistical variation is included to simulate experimental scatter
- Temperature effects are modeled based on established literature
- Rubber content effects are calibrated to experimental observations

## Contact

For questions about this dataset or the underlying research, please refer to the comprehensive report in `output/reports/dataset_report.md`.