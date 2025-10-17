# Pillar 3: Numerical Modeling Dataset

This dataset contains comprehensive data for finite element modeling of concrete fire behavior, including material properties, model parameters, and validation data.

## Dataset Contents

### 1. Main Documentation
- `pillar3_numerical_modeling_dataset.md` - Complete dataset documentation

### 2. Temperature Data
- `temp_curves_experimental.csv` - Experimental temperature vs time curves
- `temp_curves_model.csv` - Model-predicted temperature vs time curves

### 3. Pore Pressure Data
- `pore_pressure_experimental.csv` - Experimental pore pressure vs time curves
- `pore_pressure_model.csv` - Model-predicted pore pressure vs time curves

### 4. Mechanical Properties
- `stress_strain_tts.csv` - Stress-strain curves from TTS tests
- `failure_data_stt.csv` - Time/temperature to failure from STT tests
- `spalling_comparison.csv` - Spalling occurrence comparison data

### 5. Model Parameters
- `material_properties.json` - Temperature-dependent material properties
- `model_parameters.json` - Plasticity/damage model parameters
- `validation_metrics.json` - Model validation metrics

### 6. Finite Element Model
- `abaqus_input_template.inp` - Abaqus input file template

## Usage Instructions

### For Abaqus Users
1. Use the `abaqus_input_template.inp` as a starting point
2. Import material properties from `material_properties.json`
3. Apply model parameters from `model_parameters.json`
4. Validate against experimental data in CSV files

### For Other FE Software
1. Convert material properties from `material_properties.json` to your software format
2. Apply CDP model parameters from `model_parameters.json`
3. Use validation data for model verification

### Data Import
- All CSV files use comma-separated values
- JSON files contain structured parameter data
- Temperature data is in Celsius
- Pressure data is in MPa
- Time data is in minutes

## Model Validation

The dataset includes comprehensive validation metrics:
- Temperature prediction R² = 0.987
- Pore pressure prediction R² = 0.982
- Stress-strain prediction R² = 0.995
- Failure time prediction R² = 0.985
- Spalling prediction accuracy = 100%

## Limitations

- Validated for temperatures up to 800°C
- Quasi-static loading conditions only
- Normal strength concrete (40 MPa)
- 100mm diameter specimens

## Contact

For questions about this dataset, refer to the main documentation file.