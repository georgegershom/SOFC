# Comprehensive Synthetic Dataset for Fire-Resistant Structural Elements Research

## Research Title
**"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete"**

## Dataset Overview

This comprehensive synthetic dataset has been generated to support the experimental phase of research on thermo-mechanical degradation pathways in fire-resistant rubberized concrete. The dataset captures complex interactions between heating rate, peak temperature, cooling regime, and rubber content/size effects.

### Generation Details
- **Generation Date**: 2025-10-18 17:12:51
- **Random Seed**: 42 (for reproducibility)
- **Total Specimens**: 360 specimens across all test types
- **Total Data Points**: Over 10,000 individual measurements

## Dataset Components

### 1. Ambient Condition Tests (`ambient_properties.csv`)
- **Specimens**: 54 specimens
- **Matrix**: 6 mixes × 3 ages × 3 specimens
- **Purpose**: Baseline properties for normalization and comparison

**Variables Measured**:
- Compressive Strength (MPa)
- Tensile Strength (MPa) 
- Modulus of Elasticity (MPa)
- Dry Density (kg/m³)
- Ultrasonic Pulse Velocity - UPV (m/s)

### 2. Residual Properties After High-Temperature Exposure (`residual_properties_high_temp.csv`)
- **Specimens**: 270 specimens
- **Matrix**: 6 mixes × 5 temperatures × 2 cooling methods × 2 heating rates × 3 specimens
- **Purpose**: Main dataset for model calibration

**Variables Measured**:
- Peak Temperature (°C): 23, 200, 400, 600, 800
- Heating Rate: 5°C/min (standard), 10°C/min (rapid - for spalling study)
- Cooling Method: Furnace (slow), Quench (water)
- Mass Loss (%)
- Residual Compressive Strength (MPa)
- UPV after heating (m/s)
- Spalling occurrence and depth (mm)
- Visual cracking rating

### 3. In-Situ High-Temperature Tests (`in_situ_properties.csv`)
- **Specimens**: 24 specimens
- **Matrix**: 3 key mixes × 4 temperatures × 2 specimens
- **Purpose**: Hot strength behavior and stress-strain relationships

**Variables Measured**:
- In-Situ Compressive Strength (MPa)
- In-Situ Modulus of Elasticity (MPa)
- Peak Strain
- Poisson's Ratio at temperature
- Complete stress-strain curves (stored in JSON)

### 4. Advanced Measurements

#### Pore Pressure Data (`pore_pressure_summary.csv`, `pore_pressure_time_series.json`)
- **Configurations**: 12 experimental setups
- **Matrix**: 2 mixes × 3 depths × 2 runs
- **Purpose**: Spalling mechanism investigation

**Variables Measured**:
- Peak pore pressure (MPa)
- Time of peak pressure (min)
- Complete pressure evolution curves

#### Stress-Strain Curves (`stress_strain_curves.json`)
- **Curves**: 24 complete curves
- **Resolution**: 50 data points per curve
- **Purpose**: Constitutive model development

## Mix Compositions

| Mix ID | Description | Expected Behavior |
|--------|-------------|-------------------|
| C | Control concrete (no rubber) | High strength, spalling risk |
| R5S | 5% small rubber particles | Slight strength reduction, improved fire resistance |
| R10S | 10% small rubber particles | Moderate strength reduction, good fire resistance |
| R15S | 15% small rubber particles | Notable strength reduction, excellent fire resistance |
| R20S | 20% small rubber particles | Significant strength reduction, superior fire resistance |
| R10L | 10% large rubber particles | Different microstructure effects |

## Key Scientific Features

### 1. Physically Consistent Degradation Patterns
- **Plateau Phase**: Minimal degradation up to 300°C
- **Critical Zone**: Sharp strength drop at 400-600°C
- **Severe Degradation**: Near-complete loss by 800°C
- **Rubber Effects**: Melting zone behavior, pore pressure reduction

### 2. Stochastic Realism
- **Coefficient of Variation**: 5-8% for strength properties
- **Material Variability**: Realistic scatter in all measurements
- **Measurement Uncertainty**: Appropriate noise levels

### 3. Multi-Scale Correlations
- **NDT-Strength Relationships**: UPV correlates with damage
- **Mass Loss-Strength**: Clear degradation relationships
- **Microstructure-Performance**: Rubber content effects

### 4. Advanced Phenomena Modeling
- **Spalling Risk Assessment**: Rapid heating effects
- **Thermal Shock**: Quenching vs. furnace cooling
- **Pore Pressure Evolution**: Time-dependent pressure buildup
- **Constitutive Behavior**: Complete stress-strain relationships

## Data Quality Assurance

### Internal Consistency Checks
✅ Strength decreases with increasing temperature  
✅ Mass loss increases with temperature and rubber content  
✅ UPV degradation correlates with strength loss  
✅ Spalling occurs primarily in control mix under rapid heating  
✅ Rubber mixes show improved fire resistance  

### Statistical Validation
✅ Appropriate variability (COV 5-8%)  
✅ Normal distribution of residuals  
✅ No systematic biases  
✅ Realistic property ranges  

## Usage Guidelines

### Model Calibration
1. Use ambient data for baseline property establishment
2. Use residual data for temperature degradation functions
3. Use in-situ data for hot strength relationships
4. Use pore pressure data for spalling risk models

### Model Validation
1. Reserve 20-30% of data for validation
2. Use cross-validation techniques
3. Validate across different temperature ranges
4. Check predictions against physical limits

### Statistical Analysis
1. Account for specimen-to-specimen variability
2. Use appropriate statistical distributions
3. Consider temperature-dependent uncertainty
4. Include measurement error in uncertainty analysis

## File Structure

```
datasets/
├── ambient_properties.csv              # Baseline properties
├── residual_properties_high_temp.csv   # Main degradation dataset
├── in_situ_properties.csv              # Hot strength data
├── pore_pressure_summary.csv           # Peak pressure data
├── pore_pressure_time_series.json      # Complete pressure evolution
├── stress_strain_curves.json           # Complete stress-strain curves
└── dataset_metadata.json               # Comprehensive metadata

plots/
├── comprehensive_high_temperature_analysis.png
├── temperature_degradation_trends.png
├── in_situ_stress_strain_curves.png
└── pore_pressure_evolution.png
```

## Research Applications

### Primary Applications
1. **Thermo-mechanical model development**
2. **Fire resistance prediction models**
3. **Spalling risk assessment tools**
4. **Design code development**

### Secondary Applications
1. **Material optimization studies**
2. **Probabilistic analysis methods**
3. **Multi-physics simulation validation**
4. **Educational case studies**

## Technical Specifications

### Data Format
- **CSV Files**: UTF-8 encoding, comma-separated
- **JSON Files**: UTF-8 encoding, pretty-printed
- **Numerical Precision**: 6 significant figures
- **Missing Values**: None (complete dataset)

### Coordinate Systems
- **Temperature**: Celsius (°C)
- **Pressure**: Megapascals (MPa)
- **Time**: Minutes (min)
- **Length**: Millimeters (mm)
- **Density**: kg/m³

## Citation and Usage

This synthetic dataset was generated for research purposes. When using this data, please cite:

*"Synthetic Dataset for Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete, Generated 2025-10-18"*

## Contact and Support

For questions about the dataset structure, generation methodology, or usage guidelines, please refer to the accompanying documentation and metadata files.

---

**Dataset Generation Complete**: All 360 specimens with comprehensive measurements are ready for immediate use in model calibration and validation studies.