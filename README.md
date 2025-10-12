# Stratified Flow Attenuation Dataset

## PhD Research Topic
**"Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"**

## Overview

This comprehensive dataset has been generated to support advanced research in stratified flow acoustics, specifically focusing on attenuation mechanisms that extend beyond traditional single-phase leakage acoustics. The dataset encompasses multiple aspects of multiphase flow acoustics, providing a rich foundation for theoretical development, experimental validation, and machine learning applications.

## Dataset Composition

### 📊 Total Dataset Size: 5,800 samples across 5 specialized datasets

### 1. Frequency Domain Dataset (2,000 samples)
**File**: `frequency_domain.csv`

**Description**: Frequency-dependent attenuation measurements across different stratified flow configurations.

**Key Features**:
- `frequency_hz`: Acoustic frequency (10 Hz - 100 kHz)
- `flow_configuration`: Flow pattern type (horizontal_stratified, wavy_interface, slug_flow, etc.)
- `attenuation_db_per_m`: Measured attenuation coefficient
- `gas_fraction`: Volume fraction of gas phase
- `interface_roughness_mm`: Interface roughness parameter
- `reynolds_number_gas/liquid`: Flow Reynolds numbers
- `weber_number`: Weber number (surface tension effects)
- `froude_number`: Froude number (gravitational effects)
- `temperature_c`: Operating temperature
- `pressure_bar`: Operating pressure

**Research Applications**:
- Frequency-dependent attenuation modeling
- Flow pattern classification
- Interface scattering analysis
- Dimensionless parameter correlations

### 2. Experimental Conditions Dataset (500 samples)
**File**: `experimental_conditions.csv`

**Description**: Realistic experimental setup parameters and operating conditions.

**Key Features**:
- `experiment_id`: Unique experiment identifier
- `pipe_diameter_m`: Test section diameter
- `pipe_length_m`: Test section length
- `inclination_angle_deg`: Pipe inclination angle
- `gas/liquid_superficial_velocity_ms`: Phase velocities
- `predicted_flow_pattern`: Flow regime classification
- `transducer_frequency_hz`: Acoustic sensor frequency
- `measurement_distance_m`: Sensor positioning
- `ambient_conditions`: Environmental parameters

**Research Applications**:
- Experimental design optimization
- Flow pattern mapping
- Scaling analysis
- Sensor placement studies

### 3. Attenuation Models Dataset (1,500 samples)
**File**: `attenuation_models.csv`

**Description**: Theoretical predictions from various physical attenuation models.

**Key Features**:
- `model_type`: Physical model (Rayleigh scattering, Mie scattering, viscous losses, etc.)
- `frequency_hz`: Acoustic frequency
- `predicted_attenuation_db_per_m`: Model prediction
- `particle_radius_m`: Scatterer size (where applicable)
- `particle_concentration_per_m3`: Scatterer density
- `viscosity_pas`: Fluid viscosity
- `interface_roughness_m`: Surface roughness

**Research Applications**:
- Model validation and comparison
- Physical mechanism identification
- Parameter sensitivity analysis
- Hybrid model development

### 4. Multiphase Flow Dataset (1,000 samples)
**File**: `multiphase_flow.csv`

**Description**: Comprehensive multiphase flow characterization with acoustic properties.

**Key Features**:
- `flow_regime`: Flow pattern classification
- `gas/liquid_fraction`: Phase volume fractions
- `bubble/droplet_diameter_m`: Dispersed phase characteristics
- `mixture_density_kgm3`: Effective mixture density
- `mixture_sound_speed_ms`: Mixture acoustic velocity
- `attenuation_components`: Breakdown by physical mechanism
- `interface_area_density_m2m3`: Interfacial area concentration
- `turbulent_kinetic_energy_m2s2`: Turbulence parameters

**Research Applications**:
- Mixture property modeling
- Multi-mechanism attenuation analysis
- Flow regime characterization
- Turbulence-acoustics coupling

### 5. Time Series Features Dataset (800 samples)
**File**: `time_series_features.csv`

**Description**: Time-domain acoustic signal features for flow pattern recognition.

**Key Features**:
- `flow_pattern`: Time-varying flow pattern
- `rms_amplitude`: Signal RMS amplitude
- `dominant_frequency_hz`: Primary frequency component
- `spectral_centroid_hz`: Spectral centroid
- `zero_crossings_per_sec`: Zero crossing rate
- `spectral_bandwidth_hz`: Frequency spread
- `mfcc_coefficients`: Mel-frequency cepstral coefficients

**Research Applications**:
- Real-time flow monitoring
- Pattern recognition algorithms
- Signal processing development
- Machine learning training

## Research Focus Areas

### 🔬 Beyond Single Phase Acoustics

This dataset specifically addresses advanced topics in stratified flow acoustics:

1. **Multi-Scale Attenuation Mechanisms**
   - Molecular-level viscous losses
   - Bubble/droplet scattering
   - Interface wave interactions
   - Turbulent mixing effects

2. **Complex Flow Configurations**
   - Wavy stratified interfaces
   - Intermittent slug flows
   - Annular liquid films
   - Dispersed phase flows

3. **Advanced Physical Models**
   - Rayleigh and Mie scattering theories
   - Viscous boundary layer effects
   - Thermal conduction losses
   - Mode conversion phenomena

4. **Multi-Modal Sensing**
   - Frequency-domain analysis
   - Time-domain features
   - Statistical characterization
   - Machine learning integration

## Dataset Quality and Validation

### 🎯 Data Generation Methodology

- **Physics-Based Models**: All synthetic data generated using established physical principles
- **Realistic Parameter Ranges**: Based on literature review of experimental studies
- **Statistical Validation**: Comprehensive statistical analysis and correlation studies
- **Noise Modeling**: Realistic measurement uncertainties included

### 📈 Quality Metrics

- **Coverage**: Comprehensive parameter space coverage
- **Diversity**: Multiple flow regimes and operating conditions
- **Consistency**: Cross-dataset parameter consistency
- **Completeness**: Minimal missing data (<1%)

## Usage Examples

### Python Data Loading
```python
import pandas as pd
import numpy as np

# Load frequency domain data
freq_data = pd.read_csv('frequency_domain.csv')

# Filter for slug flow configuration
slug_data = freq_data[freq_data['flow_configuration'] == 'slug_flow']

# Analyze attenuation vs frequency
import matplotlib.pyplot as plt
plt.loglog(slug_data['frequency_hz'], slug_data['attenuation_db_per_m'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (dB/m)')
plt.title('Slug Flow Attenuation Characteristics')
```

### Machine Learning Applications
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load multiphase flow data
multi_data = pd.read_csv('multiphase_flow.csv')

# Prepare features and target
features = ['gas_fraction', 'bubble_diameter_m', 'mixture_density_kgm3', 
           'turbulent_kinetic_energy_m2s2']
target = 'total_attenuation_db_per_m_at_1khz'

X = multi_data[features].dropna()
y = multi_data[target].loc[X.index]

# Train attenuation prediction model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model performance
score = model.score(X_test, y_test)
print(f"Model R² score: {score:.3f}")
```

## File Structure

```
stratified_flow_datasets/
├── frequency_domain.csv              # Frequency-dependent attenuation data
├── experimental_conditions.csv       # Experimental setup parameters
├── attenuation_models.csv           # Theoretical model predictions
├── multiphase_flow.csv              # Multiphase flow characterization
├── time_series_features.csv         # Time-domain acoustic features
├── dataset_summary.txt              # Statistical summary
├── statistical_summary.txt          # Detailed statistics
├── research_insights.md             # Research findings and recommendations
└── visualizations/                  # Analysis plots and figures
    ├── frequency_domain_analysis.png
    ├── flow_pattern_analysis.png
    ├── attenuation_models.png
    ├── multiphase_characteristics.png
    └── time_series_features.png
```

## Research Applications

### 🎓 PhD Research Support

This dataset is specifically designed to support PhD-level research in:

- **Acoustic Flow Measurement**: Advanced sensing techniques for multiphase flows
- **Physical Modeling**: Development of comprehensive attenuation theories
- **Machine Learning**: Data-driven flow characterization and prediction
- **Experimental Design**: Optimization of measurement systems and protocols
- **Industrial Applications**: Real-world flow monitoring and control systems

### 📚 Potential Research Questions

1. How do interface waves affect acoustic propagation in stratified flows?
2. What are the dominant attenuation mechanisms at different frequency ranges?
3. Can machine learning improve flow pattern recognition accuracy?
4. How do operating conditions influence acoustic measurement sensitivity?
5. What are the optimal sensor configurations for different flow regimes?

## Citation and Usage

If you use this dataset in your research, please cite:

```
Stratified Flow Attenuation Dataset (2025)
"Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"
Generated for PhD Research in Multiphase Flow Acoustics
```

## Dataset Limitations

### ⚠️ Important Considerations

- **Synthetic Nature**: Data generated from theoretical models, requires experimental validation
- **Parameter Ranges**: Limited to typical industrial flow conditions
- **Model Assumptions**: Based on established theories with inherent limitations
- **Measurement Noise**: Simplified noise models may not capture all real-world effects

## Future Enhancements

### 🚀 Planned Additions

1. **Experimental Validation Data**: Real measurement data for model validation
2. **Extended Parameter Ranges**: Extreme operating conditions
3. **Advanced Flow Regimes**: Complex transitional flows
4. **Multi-Component Fluids**: Non-Newtonian and reactive fluids
5. **Sensor Response Models**: Realistic transducer characteristics

## Contact and Support

For questions, suggestions, or collaboration opportunities related to this dataset:

- **Research Focus**: Stratified Flow Acoustics
- **Application Domain**: Multiphase Flow Measurement
- **Technical Level**: PhD Research and Advanced Engineering

## License

This dataset is provided for research and educational purposes. Please ensure appropriate citation when using in publications or presentations.

---

**Generated**: October 12, 2025  
**Version**: 1.0  
**Total Samples**: 5,800  
**Research Domain**: Multiphase Flow Acoustics  
**Focus**: Beyond Single Phase Leakage Acoustics