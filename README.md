# Fire-Resistant Rubberized Concrete Synthetic Dataset

## Research Title
**Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete**

## Overview
This repository contains a comprehensive synthetic dataset for studying the thermo-mechanical behavior of rubberized concrete under fire conditions. The dataset captures complex degradation pathways including the effects of:
- Heating rate
- Peak temperature
- Cooling regime
- Rubber content and size

## Dataset Components

### Phase A: Ambient Condition Tests
- **File**: `ambient_properties.csv`
- **Content**: 54 specimens tested at ambient conditions
- **Variables**: Compressive strength, tensile strength, flexural strength, elastic modulus, density, UPV, porosity
- **Mix Types**: 6 (Control, R5S, R10S, R15S, R20S, R10L)
- **Ages**: 7, 28, 56 days

### Phase B: High-Temperature Residual Properties
- **File**: `residual_properties_high_temp.csv`
- **Content**: 234 specimens tested after high-temperature exposure
- **Peak Temperatures**: 23, 200, 400, 600, 800°C
- **Cooling Methods**: Furnace (slow), Quench (water)
- **Heating Rates**: 5°C/min (standard), 10°C/min (rapid for spalling study)
- **Key Measurements**: 
  - Residual strength and stiffness
  - Mass loss
  - UPV measurements
  - Spalling occurrence and depth
  - Visual damage assessment

### Phase C: In-Situ High-Temperature Tests
- **Files**: 
  - `in_situ_properties.csv` - Summary data
  - `stress_strain_curves.json` - Complete stress-strain curves
- **Content**: 24 specimens tested at elevated temperature
- **Test Temperatures**: 23, 200, 400, 600°C
- **Key Measurements**:
  - In-situ compressive strength
  - In-situ elastic modulus
  - Peak and ultimate strain
  - Toughness
  - Poisson's ratio
  - Thermal strain

### Phase D: Advanced Measurements
- **Files**:
  - `pore_pressure_summary.csv` - Peak values
  - `pore_pressure_profiles.json` - Time series data
  - `transient_strain_summary.csv` - Summary statistics
  - `transient_strain_profiles.json` - Full strain profiles
- **Content**: 
  - 18 pore pressure configurations
  - 8 transient strain experiments
- **Variables**:
  - Pore pressure evolution at different depths
  - Load-induced thermal strain (LITS)
  - Transient creep behavior

## Visualizations
The dataset includes comprehensive analysis plots:
1. **`comprehensive_analysis.png`** - 12-panel overview including:
   - Strength development with age
   - Temperature degradation curves
   - Mass loss evolution
   - Cooling method effects
   - Spalling risk assessment
   - In-situ vs residual comparison
   - Correlations and statistical analysis

2. **`stress_strain_curves.png`** - In-situ stress-strain behavior at different temperatures

3. **`pore_pressure_evolution.png`** - Pore pressure development during heating

## Mix Design Nomenclature
- **C**: Control mix (no rubber)
- **R5S**: 5% rubber, small particles
- **R10S**: 10% rubber, small particles
- **R15S**: 15% rubber, small particles
- **R20S**: 20% rubber, small particles
- **R10L**: 10% rubber, large particles

## Usage Instructions

### Installation
```bash
pip install -r requirements.txt
```

### Generate Dataset
```bash
python3 generate_fire_resistance_dataset.py
```

### Load Data in Python
```python
import pandas as pd
import json

# Load CSV data
df_ambient = pd.read_csv('fire_resistance_dataset/ambient_properties.csv')
df_residual = pd.read_csv('fire_resistance_dataset/residual_properties_high_temp.csv')
df_in_situ = pd.read_csv('fire_resistance_dataset/in_situ_properties.csv')

# Load JSON data
with open('fire_resistance_dataset/stress_strain_curves.json', 'r') as f:
    stress_strain_curves = json.load(f)
```

## Key Features
1. **Physical Consistency**: Degradation trends follow established patterns from fire resistance literature
2. **Stochastic Realism**: Controlled scatter mimics natural material variability
3. **Multi-Scale Linkage**: Clear correlations between non-destructive and mechanical tests
4. **Rubber-Specific Behaviors**: Models melting zones, pore pressure relief, and spalling mitigation

## Scientific Basis
The synthetic data generation incorporates:
- Popovics stress-strain model for concrete
- Temperature-dependent degradation functions
- Pore pressure development based on moisture evaporation
- Thermal shock effects from rapid cooling
- Rubber decomposition kinetics

## Applications
This dataset is designed for:
- Calibration of thermo-mechanical constitutive models
- Validation of numerical simulations
- Development of machine learning models for fire resistance prediction
- Optimization of rubber content for fire performance
- Risk assessment of spalling in high-performance concrete

## Reproducibility
The dataset generation uses a fixed random seed (42) for reproducibility. Re-running the generator will produce identical results.

## Citation
If you use this dataset in your research, please cite:
```
Title: Development and Validation of a Thermo-Mechanical Model for 
       Fire-Resistant Structural Elements Utilizing High-Performance 
       Rubberized Concrete
Dataset: Comprehensive Synthetic Dataset for Fire-Resistant Rubberized Concrete
Year: 2024
```

## File Structure
```
fire_resistance_dataset/
├── ambient_properties.csv              # Ambient condition tests
├── residual_properties_high_temp.csv   # Post-fire residual properties
├── in_situ_properties.csv              # In-situ high-temp tests
├── stress_strain_curves.json           # Complete stress-strain data
├── pore_pressure_summary.csv           # Peak pore pressures
├── pore_pressure_profiles.json         # Full pore pressure time series
├── transient_strain_summary.csv        # Transient strain statistics
├── transient_strain_profiles.json      # Complete strain profiles
├── comprehensive_analysis.png          # Main analysis visualization
├── stress_strain_curves.png            # Stress-strain behavior plots
└── pore_pressure_evolution.png         # Pore pressure development plots
```

## Data Statistics
- **Total Data Points**: 338 experimental specimens
- **Temperature Range**: 23-800°C
- **Mix Designs**: 6 variations
- **Test Types**: Ambient, Residual, In-Situ
- **Total File Size**: ~2.7 MB

## Notes
- All strength values are in MPa
- Temperatures are in Celsius
- Mass loss and strains are in percentages
- UPV (Ultrasonic Pulse Velocity) is in m/s
- Density is in kg/m³

## License
This synthetic dataset is provided for research purposes.