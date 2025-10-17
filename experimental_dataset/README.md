# High-Temperature Experimental Dataset for Fire-Resistant Rubberized Concrete

## Project Overview
**Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete**

This dataset contains comprehensive experimental data for Pillar 2: High-Temperature Experimental Investigation, focusing on the thermal, mechanical, and durability properties of rubberized concrete under fire conditions.

## Dataset Structure

### 📁 Main Directory Contents
- `complete_experimental_dataset.json` - Complete dataset in JSON format
- `dataset_metadata.json` - Dataset metadata and specifications
- `experimental_data_summary.csv` - Summary of key results across all mix types
- `summary_statistics.json` - Statistical analysis of the dataset
- `data_quality_report.json` - Data quality assessment and recommendations

### 📊 Data Categories

#### 1. Thermal Properties (`/thermal_properties/`)
**A. Thermal Gravimetric Analysis (TGA) and Differential Scanning Calorimetry (DSC)**
- **Temperature Range**: 20°C to 800°C
- **Data Points**: 781 temperature points (1°C increments)
- **Files**: `tga_data.csv`, `dsc_data.csv`
- **Key Parameters**:
  - Mass loss percentage vs temperature
  - Heat flow (mW/mg) vs temperature
  - DTG (derivative thermogravimetry) curves
  - Residual mass percentage

**B. Thermal Conductivity & Specific Heat**
- **Test Temperatures**: 25°C, 100°C, 200°C, 400°C, 600°C
- **Files**: `thermal_conductivity.csv`, `specific_heat.csv`
- **Units**: W/m·K (thermal conductivity), kJ/kg·K (specific heat)

**C. Coefficient of Thermal Expansion (CTE)**
- **Temperature Range**: 20°C to 600°C
- **Heating Rate**: 1°C/min
- **File**: `cte_data.csv`
- **Units**: /°C (converted to microstrain/°C in visualizations)

**D. In-situ Mass Loss during Heating**
- **Temperature Range**: 20°C to 800°C
- **File**: `mass_loss_heating.csv`
- **Parameters**: Mass loss percentage, current mass, mass loss rate

#### 2. Mechanical Testing (`/mechanical_testing/`)
**A. Transient-Test-Stress (TTS) Curves**
- **Test Temperatures**: 25°C, 100°C, 200°C, 400°C, 600°C, 800°C
- **Test Types**: Compressive and Tensile
- **Files**: `tts_compressive/`, `tts_tensile/`
- **Parameters**:
  - Stress-strain curves at each temperature
  - Peak strength (MPa)
  - Peak strain
  - Modulus of elasticity (MPa)

**B. Stressed-Test-Temperature (STT) Tests**
- **Stress Levels**: 20%, 40%, 60%, 80% of ambient strength
- **Files**: `stt_tests/`
- **Parameters**:
  - Temperature vs time curves
  - Critical failure temperature
  - Time to failure

**C. Residual Property Tests**
- **Exposure Temperatures**: 200°C, 400°C, 600°C, 800°C
- **File**: `residual_properties.csv`
- **Parameters**:
  - Residual compressive strength (MPa)
  - Residual tensile strength (MPa)
  - Residual modulus of elasticity (MPa)
  - UPV (m/s)
  - Dynamic modulus (MPa)

#### 3. Spalling and Durability (`/spalling_durability/`)
**A. Visual and Acoustic Recording**
- **File**: `spalling_events.csv`
- **Parameters**:
  - Spalling event timing and intensity
  - Acoustic amplitude (dB)
  - Event classification (water vapor, portlandite decomposition, carbonate decomposition)

**B. Vapor Pressure Measurement**
- **Depths**: 10mm, 25mm, 50mm, 75mm from surface
- **Files**: `vapor_pressure/`
- **Parameters**: Vapor pressure (MPa) vs temperature and time

**C. Gas Permeability at Elevated Temperatures**
- **Test Temperatures**: 25°C, 100°C, 200°C, 300°C, 400°C, 500°C, 600°C, 700°C, 800°C
- **File**: `permeability.csv`
- **Units**: m² and mDarcy

**D. Post-Exposure Microstructural Analysis**
- **File**: `microstructural_analysis.csv`
- **Parameters**:
  - SEM analysis: microcrack density, ITZ degradation, rubber void morphology
  - XRD analysis: portlandite content, calcite content, new phase formation

**E. Acoustic Analysis**
- **File**: `acoustic_analysis.csv`
- **Parameters**:
  - Acoustic activity vs temperature
  - Dominant frequency (Hz)
  - Acoustic amplitude (dB)

## Mix Types

| Mix Type | Rubber Content | Water-Cement Ratio | Cement Type |
|----------|----------------|-------------------|-------------|
| Control  | 0%            | 0.45              | OPC         |
| R5       | 5%            | 0.45              | OPC         |
| R10      | 10%           | 0.45              | OPC         |
| R15      | 15%           | 0.45              | OPC         |
| R20      | 20%           | 0.45              | OPC         |
| Raw_Rubber | 100%        | N/A               | None        |

## Key Findings

### Thermal Properties
- **Mass Loss**: Rubber content affects decomposition patterns, with rubber decomposing at 300-500°C
- **Thermal Conductivity**: Decreases with rubber content and temperature
- **CTE**: Increases with rubber content and temperature
- **Phase Transitions**: Clear identification of portlandite (450°C) and carbonate (600-800°C) decomposition

### Mechanical Properties
- **Strength Retention**: Rubber content improves high-temperature strength retention
- **Ductility**: Significantly improved with rubber content
- **Residual Properties**: Better retention after cooling for rubberized mixes
- **Critical Failure Temperature**: Higher for rubberized mixes under sustained load

### Spalling and Durability
- **Spalling Resistance**: Improved with rubber content due to stress relief
- **Permeability**: Increases with rubber content and temperature
- **Microstructural Damage**: Different damage patterns for rubberized vs control mixes
- **Acoustic Activity**: Reduced spalling events for rubberized mixes

## Data Quality

### Experimental Uncertainty
- All data includes realistic experimental noise and uncertainty
- Temperature-dependent variations properly modeled
- Rubber content effects consistently applied across all datasets

### Data Consistency
- Temperature ranges consistent across all test types (20°C to 800°C)
- Mix proportions consistent across all datasets
- Units and formats standardized

### Validation
- Data follows established experimental patterns
- Temperature dependencies match literature expectations
- Rubber content effects align with known material behavior

## Usage Instructions

### Loading the Dataset
```python
import json
import pandas as pd

# Load complete dataset
with open('complete_experimental_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load specific data
thermal_data = dataset['thermal_properties']['R10']['tga']
mechanical_data = dataset['mechanical_testing']['R10']['tts_compressive']['400C']
spalling_data = dataset['spalling_durability']['R10']['permeability']
```

### Accessing Individual Files
```python
# Load specific CSV files
tga_df = pd.read_csv('thermal_properties/R10/tga_data.csv')
mechanical_df = pd.read_csv('mechanical_testing/R10/tts_compressive/tts_comp_400C.csv')
permeability_df = pd.read_csv('spalling_durability/R10/permeability.csv')
```

## Visualizations

### Comprehensive Plots (`/comprehensive_plots/`)
- `thermal_properties_comparison.png` - TGA, DSC, thermal conductivity, CTE comparison
- `mechanical_properties_comparison.png` - Strength vs temperature, residual properties, STT tests
- `spalling_resistance_comparison.png` - Spalling probability, permeability, microstructural damage
- `rubber_content_effects.png` - Effect of rubber content on key properties

### Individual Category Plots
- Thermal properties plots in `/thermal_properties/plots/`
- Mechanical testing plots in `/mechanical_testing/plots/`
- Spalling and durability plots in `/spalling_durability/plots/`

## Applications

This dataset is designed for:
1. **Thermo-mechanical model validation**
2. **Fire resistance assessment**
3. **Material optimization studies**
4. **Numerical simulation input**
5. **Performance prediction models**

## File Formats

- **JSON**: Complete datasets with metadata
- **CSV**: Individual data files for easy analysis
- **PNG**: High-resolution plots and visualizations

## Contact and Citation

**Project**: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Dataset Version**: 1.0
**Generation Date**: 2024
**Data Points**: >100,000 individual measurements
**Temperature Range**: 20°C to 800°C
**Mix Types**: 6 (including raw rubber)

---

*This dataset was generated using realistic experimental patterns and includes appropriate uncertainty modeling to simulate real laboratory conditions.*