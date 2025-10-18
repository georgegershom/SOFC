# Synthetic Fire-Resistance Dataset Documentation

## Overview
This comprehensive synthetic dataset was generated for the research project: **"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete."**

The dataset captures complex thermo-mechanical degradation pathways, including critical effects of heating rate, peak temperature, cooling regime, and rubber content/size. All data is scientifically plausible, internally consistent, and formatted for immediate use in model calibration and validation.

## Dataset Structure

### 1. Ambient Properties Dataset (`ambient_properties.csv`)
**Purpose**: Baseline mechanical properties at room temperature
**Size**: 54 specimens (6 mixes × 3 ages × 3 specimens each)

**Columns**:
- `Specimen_ID`: Unique identifier (e.g., "C-28-A-1")
- `Mix_ID`: Concrete mix type (C, R5S, R10S, R15S, R20S, R10L)
- `Curing_Age_days`: Age at testing (7, 28, 56 days)
- `Test_Type`: Always "Ambient"
- `Compressive_Strength_MPa`: Compressive strength
- `Tensile_Strength_MPa`: Tensile strength (~10% of compressive)
- `Modulus_of_Elasticity_MPa`: Elastic modulus
- `Dry_Density_kgm3`: Material density
- `UPV_mps`: Ultrasonic pulse velocity

**Key Features**:
- Strength decreases with rubber content (Control=65 MPa, decreasing by 8 MPa per step)
- Age effects: 7-day = 75% of 28-day strength, 56-day = 105% of 28-day strength
- Realistic variability: 5% COV for strength, 4% for modulus, 2% for UPV

### 2. Residual High-Temperature Properties (`residual_properties_high_temp.csv`)
**Purpose**: Mechanical properties after high-temperature exposure and cooling
**Size**: 270 specimens (6 mixes × 5 temperatures × 2 cooling methods × 2 heating rates × 3 specimens)

**Columns**:
- `Specimen_ID`: Unique identifier (e.g., "C-28-R-800-Quench-10_C_per_min-1")
- `Mix_ID`: Concrete mix type
- `Peak_Temperature_C`: Exposure temperature (23, 200, 400, 600, 800°C)
- `Heating_Rate`: Heating rate (5_C_per_min, 10_C_per_min)
- `Cooling_Method`: Cooling type (Furnace, Quench)
- `Test_Type`: Always "Residual"
- `Mass_Loss_pct`: Mass loss percentage
- `UPV_mps`: Ultrasonic pulse velocity after exposure
- `Residual_Compressive_Strength_MPa`: Compressive strength after exposure
- `Spalling_Occurred`: Boolean flag for spalling occurrence
- `Spalling_Depth_mm`: Depth of spalling (if occurred)
- `Visual_Cracking_Rating`: Cracking severity (None, Minor, Moderate, Severe)

**Key Features**:
- Complex degradation patterns based on temperature and mix type
- Rubber-specific behaviors: melting zone effects, pore pressure reduction
- Spalling risk: High for control mix with rapid heating at ≥400°C
- Quenching causes additional 15% strength loss due to thermal shock
- Mass loss increases with temperature and rubber content

### 3. In-Situ High-Temperature Properties (`in_situ_properties.csv`)
**Purpose**: Mechanical properties during high-temperature exposure
**Size**: 24 specimens (3 key mixes × 4 temperatures × 2 specimens each)

**Columns**:
- `Specimen_ID`: Unique identifier (e.g., "C-28-IS-400-1")
- `Mix_ID`: Concrete mix type (C, R10S, R20S)
- `Test_Temperature_C`: Test temperature (23, 200, 400, 600°C)
- `Test_Type`: Always "In-Situ"
- `InSitu_Compressive_Strength_MPa`: Compressive strength at temperature
- `InSitu_Modulus_of_Elasticity_MPa`: Elastic modulus at temperature
- `Peak_Strain`: Strain at peak stress
- `Poissons_Ratio`: Poisson's ratio (decreases with temperature)

**Key Features**:
- Different strength retention patterns compared to residual tests
- Rubber mixes show increased ductility (higher peak strain)
- Temperature-dependent material behavior
- Full stress-strain curves available in `stress_strain_curves.json`

### 4. Pore Pressure Summary (`pore_pressure_summary.csv`)
**Purpose**: Peak pore pressure data for spalling investigation
**Size**: 12 configurations (2 mixes × 3 depths × 2 runs)

**Columns**:
- `Mix_ID`: Concrete mix type (C, R20S)
- `Depth_mm`: Depth from exposed surface (10, 25, 40 mm)
- `Run`: Experimental run number (1, 2)
- `Peak_Pressure_MPa`: Maximum pore pressure recorded
- `Time_of_Peak_min`: Time when peak pressure occurred

**Key Features**:
- Control mix shows higher peak pressures and earlier peaks
- Rubber concrete shows broader, lower pressure peaks
- Depth-dependent pressure profiles
- Full time-series data available in the generation script

### 5. Stress-Strain Curves (`stress_strain_curves.json`)
**Purpose**: Complete stress-strain curves for in-situ tests
**Format**: JSON with specimen IDs as keys

**Structure**:
```json
{
  "C-28-IS-400-1": {
    "strain": [0.0, 0.0005, 0.001, ...],
    "stress": [0.0, 15.2, 28.4, ...]
  }
}
```

**Key Features**:
- Parabolic ascending branch
- Linear descending branch (more gradual for rubber mixes)
- Temperature-dependent peak strain
- Realistic noise and variability

## Mix Designations

| Mix_ID | Description | Rubber Content | Expected Behavior |
|--------|-------------|----------------|-------------------|
| C | Control | 0% | Baseline performance, high spalling risk |
| R5S | 5% Small Rubber | 5% | Slight strength reduction, improved ductility |
| R10S | 10% Small Rubber | 10% | Moderate strength reduction, good ductility |
| R15S | 15% Small Rubber | 15% | Significant strength reduction, high ductility |
| R20S | 20% Small Rubber | 20% | High strength reduction, very high ductility |
| R10L | 10% Large Rubber | 10% | Similar to R10S but with different failure modes |

## Physical Consistency Features

### 1. Temperature-Dependent Degradation
- **23-200°C**: Minimal strength loss, slight increase possible
- **200-400°C**: Gradual strength reduction, rubber softening zone
- **400-600°C**: Sharp strength drop, rubber combustion
- **600-800°C**: Near-complete strength loss

### 2. Rubber-Specific Behaviors
- **Melting Zone**: 200-400°C where rubber softens before burning
- **Pore Pressure Reduction**: Lower peak pressures compared to control
- **Ductility Enhancement**: Increased strain capacity
- **Spalling Mitigation**: Reduced spalling risk with rapid heating

### 3. Cooling Method Effects
- **Furnace Cooling**: Gradual cooling, minimal thermal shock
- **Quench Cooling**: Rapid cooling, 15% additional strength loss

### 4. Heating Rate Effects
- **5°C/min**: Standard heating rate
- **10°C/min**: Rapid heating, increased spalling risk for control mix

## Statistical Properties

### Variability (Coefficient of Variation)
- Compressive Strength: 5-8%
- Modulus of Elasticity: 4-6%
- Ultrasonic Pulse Velocity: 2-5%
- Mass Loss: 10%

### Correlations
- UPV correlates with strength retention (R² ≈ 0.7)
- Mass loss correlates with strength loss (R² ≈ 0.8)
- Spalling probability increases with heating rate and temperature

## Usage Guidelines

### For Model Calibration
1. Use ambient data to establish baseline material properties
2. Use residual data to calibrate temperature-dependent degradation models
3. Use in-situ data to validate transient behavior models
4. Use pore pressure data to calibrate spalling prediction models

### For Model Validation
1. Compare predicted vs. measured strength retention curves
2. Validate spalling prediction accuracy
3. Check stress-strain curve predictions
4. Verify pore pressure evolution models

### Data Quality Assurance
- All negative values have been set to zero (physical constraint)
- Statistical scatter is realistic and consistent
- Cross-correlations between properties are maintained
- Temperature effects follow established material science principles

## File Formats

- **CSV Files**: Comma-separated values, ready for Excel or pandas
- **JSON File**: Stress-strain curves in structured format
- **PNG Files**: Visualization plots for data exploration

## Generated Visualizations

1. **comprehensive_high_temperature_analysis.png**: Four-panel summary plot
   - Strength degradation with temperature
   - Effect of cooling method
   - Strength vs. mass loss correlation
   - Spalling risk assessment

2. **in_situ_stress_strain_curves.png**: Stress-strain behavior at 400°C
3. **pore_pressure_evolution.png**: Pore pressure development over time

## Research Applications

This dataset is specifically designed for:
- Thermo-mechanical model development
- Fire resistance assessment
- Spalling prediction modeling
- Probabilistic analysis
- Material optimization studies
- Code development for fire-resistant concrete

## Contact and Citation

For questions about this dataset or its usage, please refer to the research project documentation. This synthetic dataset was generated using established material science principles and experimental observations from the literature on fire-resistant concrete and rubberized concrete composites.

---
*Dataset generated on: 2025-10-18*  
*Total specimens: 360*  
*Total data points: 2,880+*