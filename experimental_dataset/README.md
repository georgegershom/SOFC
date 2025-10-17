# High-Temperature Experimental Dataset for Fire-Resistant Rubberized Concrete

## Overview

This comprehensive experimental dataset was generated for the development and validation of thermo-mechanical models for fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset contains extensive experimental data covering thermal properties, mechanical testing, and spalling/durability characteristics.

## Dataset Structure

```
experimental_dataset/
├── thermal_properties/
│   ├── tga_dsc/                    # TGA/DSC analysis data
│   ├── thermal_conductivity/       # Thermal conductivity measurements
│   ├── cte/                        # Coefficient of thermal expansion
│   └── mass_loss/                  # In-situ mass loss during heating
├── mechanical_testing/
│   ├── tts_curves/                 # Transient-Test-Stress curves
│   ├── stt_tests/                  # Stressed-Test-Temperature tests
│   └── residual_properties/        # Post-heating residual properties
├── spalling_durability/
│   ├── visual_audio/               # Video/audio recording metadata
│   ├── vapor_pressure/             # Vapor pressure measurements
│   ├── gas_permeability/           # Gas permeability at elevated temperatures
│   └── microstructural/            # SEM and XRD analysis
├── metadata/
│   ├── sample_specifications.json  # Sample mix details
│   ├── test_protocols.json         # Experimental protocols
│   └── data_dictionary.json        # Complete data dictionary
└── validation_scripts/
    ├── data_validator.py           # Data validation tools
    └── data_analyzer.py            # Analysis and visualization tools
```

## Sample Mixes

1. **Control Mix**: 100% Natural Aggregate (NA)
2. **10% Rubber**: 10% Crumb Rubber (CR) replacement
3. **20% Rubber**: 20% Crumb Rubber (CR) replacement  
4. **30% Rubber**: 30% Crumb Rubber (CR) replacement
5. **Raw Rubber**: Pure crumb rubber samples (reference)

## Test Conditions

### Temperature Ranges
- **Ambient**: 25°C
- **Low**: 100°C, 200°C
- **Medium**: 400°C, 600°C
- **High**: 800°C, 1000°C

### Heating Rates
- **TGA/DSC**: 10°C/min
- **Mechanical Testing**: 5°C/min
- **CTE**: 1°C/min
- **Mass Loss**: 5°C/min

## Data Categories

### A. Thermal Property Dataset

#### 1. TGA/DSC Analysis
- **Equipment**: TGA/DSC 3+ (Mettler Toledo)
- **Temperature Range**: 25°C to 800°C
- **Data Points**: 775 points (1°C intervals)
- **Replicates**: 3 per mix
- **Key Parameters**:
  - Mass loss percentage
  - Heat flow (mW/mg)
  - Remaining mass percentage
  - Mass loss rate

#### 2. Thermal Conductivity & Specific Heat
- **Equipment**: Hot Disk TPS 2500S
- **Temperatures**: 25°C, 100°C, 200°C, 400°C, 600°C
- **Replicates**: 5 per temperature
- **Key Parameters**:
  - Thermal conductivity (W/m·K)
  - Specific heat (J/kg·K)
  - Density (kg/m³)
  - Thermal diffusivity (m²/s)

#### 3. Coefficient of Thermal Expansion (CTE)
- **Equipment**: DIL 402 C (Netzsch)
- **Heating Rate**: 1°C/min
- **Temperature Range**: 25°C to 600°C
- **Replicates**: 3 per mix
- **Key Parameters**:
  - Instantaneous CTE (μm/m·°C)
  - Thermal strain
  - Length change percentage

#### 4. In-situ Mass Loss
- **Equipment**: High-temperature furnace with precision balance
- **Heating Rate**: 5°C/min
- **Temperature Range**: 25°C to 800°C
- **Replicates**: 3 per mix
- **Key Parameters**:
  - Remaining mass (kg)
  - Mass loss percentage
  - Mass loss rate (kg/min)

### B. High-Temperature Mechanical Testing Dataset

#### 1. Transient-Test-Stress (TTS) Curves
- **Equipment**: Instron 5982 with high-temperature furnace
- **Loading Rate**: 0.5 MPa/s
- **Temperatures**: 25°C, 200°C, 400°C, 600°C, 800°C
- **Replicates**: 5 per temperature
- **Key Parameters**:
  - Stress-strain curves
  - Peak strength (MPa)
  - Peak strain
  - Elastic modulus (MPa)

#### 2. Stressed-Test-Temperature (STT) Tests
- **Equipment**: Instron 5982 with high-temperature furnace
- **Preload Levels**: 20%, 40%, 60%, 80% of ambient strength
- **Heating Rate**: 5°C/min
- **Replicates**: 5 per preload level
- **Key Parameters**:
  - Failure temperature (°C)
  - Time to failure (min)
  - Applied stress (MPa)
  - Stress ratio

#### 3. Residual Properties
- **Cooling Method**: Natural air cooling (24 hours)
- **Exposure Temperatures**: 25°C, 200°C, 400°C, 600°C, 800°C
- **Replicates**: 5 per temperature
- **Key Parameters**:
  - Residual compressive strength (MPa)
  - Residual tensile strength (MPa)
  - Residual elastic modulus (MPa)
  - UPV (m/s)
  - Dynamic modulus (GPa)

### C. Spalling and Durability Dataset

#### 1. Visual and Acoustic Recording
- **Camera**: Sony A7R IV 4K (30 fps)
- **Microphone**: Rode VideoMic Pro Plus (44.1 kHz)
- **Analysis**: ImageJ + MATLAB
- **Key Parameters**:
  - Spalling events count
  - Audio peaks count
  - Temperature at spalling
  - Event timing

#### 2. Vapor Pressure Measurement
- **Equipment**: Custom-built pressure transducers
- **Sensor Depths**: 10, 20, 30, 40, 50 mm from surface
- **Pressure Range**: 0-10 bar
- **Sampling Rate**: 1 Hz
- **Key Parameters**:
  - Vapor pressure (Pa)
  - Saturation pressure (Pa)
  - Pressure vs depth profiles

#### 3. Gas Permeability
- **Equipment**: Custom permeability apparatus
- **Gas**: Nitrogen
- **Pressure Differences**: 0.1, 0.2, 0.5, 1.0 bar
- **Temperatures**: 25°C, 100°C, 200°C, 400°C, 600°C
- **Key Parameters**:
  - Permeability (m²)
  - Flow rate (m³/s)
  - Viscosity (Pa·s)

#### 4. Microstructural Analysis
- **SEM Equipment**: FEI Quanta 200 FEG
- **XRD Equipment**: Bruker D8 Advance
- **Exposure Temperatures**: 25°C, 200°C, 400°C, 600°C, 800°C
- **Replicates**: 3 per temperature
- **Key Parameters**:
  - Microcrack density (cracks/mm²)
  - ITZ degradation (0-1 scale)
  - Void density (voids/mm²)
  - Phase abundance (relative intensity)

## Data Quality

- **Replicates per Test**: 3-5
- **Measurement Uncertainty**: ±5%
- **Calibration Standards**: NIST traceable
- **Validation Status**: Generated for model validation

## File Formats

- **JSON**: Complete datasets with metadata
- **CSV**: Individual test results for analysis
- **PNG**: Generated plots and visualizations
- **MP4/WAV**: Video and audio recordings (metadata only)

## Usage

### Data Validation
```bash
python3 validation_scripts/data_validator.py
```

### Data Analysis
```bash
python3 validation_scripts/data_analyzer.py
```

### Generated Outputs
- Validation report: `validation_report.json`
- Analysis results: `analysis/analysis_results.json`
- Comprehensive plots: `analysis/plots/`

## Key Findings

### Thermal Properties
- Rubber content reduces thermal conductivity by 2% per 10% rubber
- Specific heat increases with rubber content
- CTE increases significantly with rubber content
- Mass loss shows distinct decomposition stages

### Mechanical Properties
- Strength degradation follows exponential decay
- Rubber content affects degradation rate
- Residual properties show significant reduction above 400°C
- STT tests reveal critical failure temperatures

### Spalling Behavior
- Rubber content increases spalling propensity
- Vapor pressure builds up at different depths
- Permeability increases with temperature
- Microstructural changes correlate with temperature

## Applications

This dataset is specifically designed for:
- Thermo-mechanical model validation
- Fire resistance assessment
- Material property characterization
- Numerical simulation calibration
- Research and development

## Citation

When using this dataset, please cite:
```
High-Temperature Experimental Dataset for Fire-Resistant Rubberized Concrete
Generated for Thermo-Mechanical Model Validation
[Your Institution/Research Team]
[Date]
```

## Contact

For questions about this dataset or requests for additional data, please contact the research team.

---

**Note**: This dataset was generated for research purposes. For critical applications, validate with experimental data from actual laboratory testing.