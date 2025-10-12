# Stratified Flow Attenuation Mechanisms Dataset

## PhD Thesis Dataset
**Topic:** "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"

## Overview

This comprehensive dataset is designed to support research on acoustic attenuation mechanisms in stratified multiphase flows. The dataset combines theoretical models, empirical relationships, and synthetic experimental data to provide a robust foundation for understanding how acoustic waves propagate and attenuate in complex stratified flow systems.

## Dataset Components

### 1. Fluid Properties (`fluid_properties.csv`)
- **Water, Oil, and Gas phase properties**
- Temperature and pressure dependencies
- Physical properties: density, viscosity, sound speed, thermal conductivity, specific heat, bulk modulus
- **Samples:** 3,000 (1,000 per fluid type)

### 2. Flow Geometry (`flow_geometry.csv`)
- **Stratified flow parameters**
- Pipe/duct dimensions and geometry
- Layer thickness ratios and interface characteristics
- Flow velocities and Reynolds numbers
- Flow regime classification
- **Samples:** 1,000

### 3. Acoustic Properties (`acoustic_properties.csv`)
- **Frequency-dependent attenuation mechanisms**
- Viscous, thermal, scattering, and interface attenuation
- Acoustic impedance and reflection/transmission coefficients
- Wavelength and power level data
- **Samples:** 20,000 (1,000 samples × 20 frequencies)

### 4. Experimental Conditions (`experimental_conditions.csv`)
- **Measurement setup parameters**
- Environmental conditions (temperature, pressure, humidity)
- Transducer specifications and positioning
- Signal processing parameters
- Noise characteristics and measurement uncertainty
- **Samples:** 1,000

### 5. Theoretical Models (`theoretical_models.csv`)
- **Classical and advanced attenuation models**
- Stokes-Kirchhoff, Navier-Stokes, Thermoacoustic models
- Multiphase flow and interface scattering models
- Mode conversion and advanced mechanisms
- **Samples:** 50 frequency points

### 6. Correlation Data (`correlation_data.csv`)
- **Parameter correlation analysis**
- Temperature, pressure, and viscosity effects
- Frequency-dependent relationships
- Combined effect modeling
- **Samples:** 1,000

## Key Features

### Attenuation Mechanisms Covered
1. **Viscous Attenuation** - Energy dissipation due to fluid viscosity
2. **Thermal Attenuation** - Heat conduction effects on acoustic waves
3. **Scattering Attenuation** - Wave scattering from interfaces and particles
4. **Interface Attenuation** - Mode conversion and reflection at phase boundaries

### Flow Regimes Included
- Laminar-Laminar stratified flows
- Turbulent-Turbulent stratified flows
- Mixed regime flows
- Transitional flow conditions

### Frequency Range
- **10 Hz to 100 kHz** (acoustic to ultrasonic range)
- Logarithmically spaced frequency points
- Covers practical measurement ranges

### Physical Parameters
- **Temperature:** 273-373 K (0-100°C)
- **Pressure:** 1-10 bar
- **Pipe diameters:** 0.01-0.5 m
- **Flow velocities:** 0.1-10 m/s
- **Reynolds numbers:** 1,000-100,000

## Usage Instructions

### 1. Generate the Dataset
```bash
python stratified_flow_attenuation_dataset.py
```

### 2. Load and Analyze Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load specific datasets
fluid_props = pd.read_csv('stratified_flow_dataset/fluid_properties.csv')
acoustic_props = pd.read_csv('stratified_flow_dataset/acoustic_properties.csv')
flow_geometry = pd.read_csv('stratified_flow_dataset/flow_geometry.csv')

# Example analysis
frequencies = acoustic_props['frequency'].unique()
attenuation_by_freq = acoustic_props.groupby('frequency')['total_attenuation'].mean()

plt.loglog(frequencies, attenuation_by_freq)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (Np/m)')
plt.title('Frequency-Dependent Attenuation')
plt.grid(True)
plt.show()
```

### 3. Theoretical Model Comparison
```python
# Compare different theoretical models
theoretical = pd.read_csv('stratified_flow_dataset/theoretical_models.csv')

plt.loglog(theoretical['frequency'], theoretical['stokes_kirchhoff_attenuation'], 
           label='Stokes-Kirchhoff')
plt.loglog(theoretical['frequency'], theoretical['multiphase_attenuation'], 
           label='Multiphase Model')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (Np/m)')
plt.legend()
plt.grid(True)
plt.show()
```

## Dataset Statistics

### File Sizes
- `fluid_properties.csv`: ~0.5 MB
- `flow_geometry.csv`: ~0.2 MB
- `acoustic_properties.csv`: ~3.5 MB
- `experimental_conditions.csv`: ~0.3 MB
- `theoretical_models.csv`: ~0.1 MB
- `correlation_data.csv`: ~0.2 MB
- **Total dataset size:** ~5 MB

### Data Quality
- **Completeness:** 100% (no missing values)
- **Consistency:** All units in SI system
- **Reproducibility:** Fixed random seed (42)
- **Validation:** Physics-based constraints applied

## Research Applications

### 1. Model Development
- Validate new attenuation models against comprehensive data
- Compare theoretical predictions with synthetic experimental data
- Develop machine learning models for attenuation prediction

### 2. Parameter Sensitivity Analysis
- Study effects of temperature, pressure, and flow conditions
- Analyze frequency-dependent behavior
- Investigate interface effects on attenuation

### 3. Experimental Design
- Optimize measurement parameters
- Select appropriate frequency ranges
- Design stratified flow experiments

### 4. Industrial Applications
- Pipeline leak detection systems
- Multiphase flow metering
- Acoustic monitoring of industrial processes

## Theoretical Background

### Attenuation Mechanisms

#### 1. Viscous Attenuation
The classical Stokes-Kirchhoff formula describes viscous attenuation:
```
α_v = (2π²f²)/(3ρc³) × (4μ/3 + μ_B + (γ-1)κ/(γc_p))
```

#### 2. Thermal Attenuation
Thermal relaxation effects:
```
α_t = (2π²f²)/(2ρc³) × (γ-1)κ/c_p
```

#### 3. Interface Scattering
Mode conversion at phase boundaries:
```
α_s = f^4 × (scattering cross-section)
```

#### 4. Multiphase Effects
Additional attenuation due to phase interactions:
```
α_m = f^1.5 × (interface effects)
```

## Validation and Quality Assurance

### Physical Constraints
- All attenuation coefficients are positive
- Frequency dependencies follow expected power laws
- Temperature and pressure effects are physically reasonable
- Reynolds numbers correspond to appropriate flow regimes

### Statistical Validation
- Parameter distributions match expected ranges
- Correlations between parameters are physically meaningful
- Outliers are identified and handled appropriately

## Future Enhancements

### Planned Additions
1. **Experimental Data Integration** - Real experimental data from literature
2. **Advanced Models** - Machine learning-based attenuation models
3. **3D Effects** - Three-dimensional flow geometry considerations
4. **Transient Effects** - Time-dependent attenuation behavior
5. **Non-Newtonian Fluids** - Complex fluid rheology effects

### Data Updates
- Regular updates with new theoretical developments
- Integration of experimental validation data
- Expansion of parameter ranges based on research needs

## Citation

If you use this dataset in your research, please cite:

```
Stratified Flow Attenuation Mechanisms Dataset (2024)
PhD Thesis: "Study on the Attenuation Mechanisms in Stratified Flows: 
Beyond Single Phase Leakage Acoustics"
Dataset Version 1.0
```

## Contact and Support

For questions about the dataset or suggestions for improvements, please refer to the PhD thesis documentation or contact the research team.

## License

This dataset is provided for academic research purposes. Please ensure proper attribution and follow academic integrity guidelines when using this data.

---

**Generated on:** 2024
**Dataset Version:** 1.0
**Total Samples:** 26,050
**Files:** 7 (6 CSV + 1 metadata JSON)
**Documentation:** Complete with examples and theoretical background