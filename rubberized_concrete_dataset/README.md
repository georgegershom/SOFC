# Fire-Resistant Rubberized Concrete Dataset

## Overview

This comprehensive dataset supports the development and validation of thermo-mechanical models for fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset includes both material property data for model calibration and experimental validation data for model verification.

## Dataset Description

**Topic**: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Generated**: 2025-10-18

**Scope**: Temperature range 20-1000°C, Rubber content 0-20% by volume

## Dataset Structure

### 1. Material Properties (Model Input Data)

#### 1.1 Thermal Properties
- **File**: `thermal_properties.json`, `thermal_properties.csv`
- **Parameters**: 
  - Thermal conductivity (W/m·K)
  - Specific heat (J/kg·K) 
  - Density (kg/m³)
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

#### 1.2 Mechanical Properties
- **File**: `mechanical_properties.json`, `mechanical_properties.csv`
- **Parameters**:
  - Compressive strength (MPa)
  - Tensile strength (MPa)
  - Elastic modulus (GPa)
  - Poisson's ratio
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

#### 1.3 Deformation Properties
- **File**: `deformation_properties.json`, `deformation_properties.csv`
- **Parameters**:
  - Coefficient of thermal expansion (1/K)
  - Transient thermal strain
  - Creep strain data (time-dependent)
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

#### 1.4 Poro-mechanical Properties
- **File**: `poromechanical_properties.json`, `poromechanical_properties.csv`
- **Parameters**:
  - Porosity (temperature and damage dependent)
  - Permeability (m²)
  - Damage parameter (0-1)
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

### 2. Validation Data (Experimental Measurements)

#### 2.1 Temperature Evolution Data
- **File**: `temperature_evolution_validation.json`, `temperature_evolution_validation.csv`
- **Description**: Thermocouple measurements during fire exposure
- **Specimen Types**: Small cube (100mm), Cylinder (φ100×200mm), Beam (100×100×400mm), Slab (500×500×100mm)
- **Fire Curves**: ISO834, ASTM E119, Hydrocarbon, Parametric
- **Measurement Points**: Surface, 25mm depth, center
- **Duration**: 0-4 hours

#### 2.2 Deformation/Strain History
- **File**: `deformation_strain_validation.json`, `deformation_strain_validation.csv`
- **Description**: Strain measurements under combined thermal-mechanical loading
- **Loading Scenarios**: 
  - Thermal only (free expansion)
  - Low load (20% of strength)
  - Medium load (40% of strength)
  - High load (60% of strength)
- **Strain Components**: Total, thermal, mechanical, creep, transient
- **Duration**: 0-3 hours

#### 2.3 Spalling and Failure Data
- **File**: `spalling_failure_validation.json`, `spalling_failure_summary.csv`, `spalling_failure_detailed.csv`
- **Description**: Spalling patterns and failure analysis
- **Test Conditions**:
  - Standard fire (ISO834, moderate load, 4.5% moisture)
  - Rapid heating (hydrocarbon curve, higher load, 6% moisture)
  - High load (ISO834, 60% load, 4% moisture)
  - High moisture (ISO834, moderate load, 8% moisture)
- **Measurements**: Spalling occurrence, depth progression, mass loss, failure time

## Key Features

### Realistic Material Behavior
- Temperature-dependent property degradation
- Rubber content effects on all properties
- Moisture-related phenomena (transient strain, spalling)
- Damage evolution with temperature

### Comprehensive Validation Data
- Multi-point temperature measurements
- Strain decomposition (thermal, mechanical, creep, transient)
- Spalling risk assessment
- Failure mode identification

### Multiple Data Formats
- JSON for hierarchical data structure
- CSV for tabular analysis
- Comprehensive metadata
- Visualization tools included

## Usage Examples

### Loading Thermal Properties
```python
import json
import pandas as pd

# Load JSON data
with open('thermal_properties.json', 'r') as f:
    thermal_data = json.load(f)

# Access data for 10% rubber content
rubber_10_data = thermal_data['rubber_10pct']
temperatures = rubber_10_data['temperature']
conductivity = rubber_10_data['thermal_conductivity']

# Or load CSV data
df = pd.read_csv('thermal_properties.csv')
rubber_10_df = df[df['rubber_content_pct'] == 10]
```

### Analyzing Validation Data
```python
# Load temperature evolution data
df_temp = pd.read_csv('temperature_evolution_validation.csv')

# Filter for specific conditions
iso834_data = df_temp[
    (df_temp['fire_curve'] == 'ISO834') & 
    (df_temp['specimen_type'] == 'small_cube') &
    (df_temp['rubber_content_pct'] == 10)
]

# Plot temperature evolution
import matplotlib.pyplot as plt
for location in ['surface', '25mm', 'center']:
    data = iso834_data[iso834_data['thermocouple_location'] == location]
    plt.plot(data['time_hours'], data['temperature_C'], label=location)
plt.legend()
plt.show()
```

### Spalling Analysis
```python
# Load spalling data
df_spall = pd.read_csv('spalling_failure_summary.csv')

# Analyze spalling occurrence by rubber content
spall_rate = df_spall.groupby('rubber_content_pct')['spalling_occurred'].mean()
print("Spalling occurrence rate by rubber content:")
print(spall_rate)
```

## Model Implementation Guidelines

### Material Property Functions
The dataset can be used to develop temperature-dependent material property functions:

```python
def thermal_conductivity(T, rubber_content):
    # Fit polynomial or exponential functions to data
    # Example: k(T) = k0 * (1 - a*rubber_content/100) * exp(-b*T)
    pass

def compressive_strength(T, rubber_content):
    # Implement strength degradation model
    # Consider both temperature and rubber content effects
    pass
```

### Validation Methodology
1. **Calibration**: Use material property data to determine model parameters
2. **Validation**: Compare model predictions with experimental validation data
3. **Verification**: Ensure model captures key phenomena (spalling, strain evolution)

## Data Quality and Limitations

### Strengths
- Comprehensive temperature range (20-1000°C)
- Multiple rubber content levels (0-20%)
- Realistic material behavior modeling
- Multiple validation scenarios
- Consistent data structure

### Limitations
- Synthetic data based on literature and engineering judgment
- Limited to specific rubber type and concrete matrix
- No consideration of aggregate type variations
- Simplified fire exposure conditions

## File Structure
```
rubberized_concrete_dataset/
├── thermal_properties.json
├── thermal_properties.csv
├── mechanical_properties.json
├── mechanical_properties.csv
├── deformation_properties.json
├── deformation_properties.csv
├── poromechanical_properties.json
├── poromechanical_properties.csv
├── temperature_evolution_validation.json
├── temperature_evolution_validation.csv
├── deformation_strain_validation.json
├── deformation_strain_validation.csv
├── spalling_failure_validation.json
├── spalling_failure_summary.csv
├── spalling_failure_detailed.csv
├── visualizations/
│   ├── thermal_properties.png
│   ├── mechanical_properties.png
│   ├── temperature_evolution.png
│   ├── strain_evolution.png
│   ├── spalling_analysis.png
│   └── summary_dashboard.png
├── metadata.json
├── README.md
├── data_dictionary.json
└── usage_examples.py
```

## Citation

If you use this dataset in your research, please cite:

```
Fire-Resistant Rubberized Concrete Dataset for Thermo-Mechanical Modeling
Generated: 2025-10-18
Topic: Development and Validation of a Thermo-Mechanical Model for 
       Fire-Resistant Structural Elements Utilizing High-Performance 
       Rubberized Concrete
```

## Contact and Support

This dataset was generated for research and educational purposes. For questions about the data structure or usage, refer to the included documentation and example scripts.

## Version History

- v1.0 (2025-10-18): Initial dataset release
  - Complete material property database
  - Comprehensive validation datasets
  - Visualization tools
  - Documentation and examples
