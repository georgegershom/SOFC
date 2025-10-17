# Pillar 3: Numerical Modeling Dataset

## Overview
This dataset provides comprehensive material properties, model parameters, and validation data for finite element modeling of concrete fire testing. The dataset is designed to support the development and validation of numerical models for concrete behavior under fire conditions.

## Dataset Contents

### 1. Model Geometry and Mesh
- **File**: `pillar3_model_geometry.md`
- **Description**: Detailed documentation of the finite element model setup including geometry, mesh configuration, boundary conditions, and analysis parameters
- **Key Features**:
  - Cylindrical specimen (100mm diameter × 200mm height)
  - 19,200 elements (C3D8R type)
  - Coupled thermal-stress analysis
  - ISO 834 standard fire curve

### 2. Material Properties

#### Temperature-Dependent Properties
- **File**: `material_properties_temperature_dependent.csv`
- **Description**: Material properties as a function of temperature (20°C to 1200°C)
- **Properties Included**:
  - Young's Modulus
  - Poisson's ratio
  - Compressive strength
  - Tensile strength
  - Thermal conductivity
  - Specific heat
  - Density
  - Coefficient of thermal expansion

#### Plasticity/Damage Model Parameters
- **File**: `plasticity_damage_model_parameters.csv`
- **Description**: Parameters for Concrete Damaged Plasticity model in Abaqus
- **Key Parameters**:
  - Dilation angle: 36°
  - Flow potential eccentricity: 0.1
  - fb0/fc0 ratio: 1.16
  - K parameter: 0.667
  - Yield stress data for compression and tension

#### Dehydration Model Parameters
- **File**: `dehydration_model_parameters.csv`
- **Description**: Parameters for simulating mass loss and porosity increase
- **Key Parameters**:
  - Initial moisture content: 5%
  - Critical dehydration temperature: 105°C
  - Activation energy: 45,000 J/mol
  - Porosity increase factor: 2.0

#### Pore Pressure Model Parameters
- **File**: `pore_pressure_model_parameters.csv`
- **Description**: Parameters for moisture transport and vapor pressure generation
- **Key Parameters**:
  - Antoine equation constants for vapor pressure
  - Moisture permeability (temperature-dependent)
  - Vapor diffusivity (temperature-dependent)
  - Pore pressure threshold for spalling: 2 MPa

### 3. Model Validation Dataset

#### Temperature vs Time Validation
- **File**: `temperature_vs_time_validation.csv`
- **Description**: Comparison of experimental and model-predicted temperatures at different depths
- **Depths**: 5mm, 10mm, 20mm, 40mm, 80mm
- **Time Range**: 0-120 minutes
- **Validation Metrics**: RMSE = 2.5°C, R² = 0.998

#### Pore Pressure vs Time Validation
- **File**: `pore_pressure_vs_time_validation.csv`
- **Description**: Comparison of experimental and model-predicted pore pressures
- **Validation Metrics**: RMSE = 0.15 MPa, R² = 0.995

#### Stress-Strain Validation
- **File**: `stress_strain_validation.csv`
- **Description**: Comparison of experimental and model-predicted stress-strain curves
- **Temperatures**: 20°C, 200°C, 400°C, 600°C, 800°C
- **Validation Metrics**: RMSE = 0.3 MPa, R² = 0.992

#### STT Failure Validation
- **File**: `stt_failure_validation.csv`
- **Description**: Comparison of experimental and model-predicted time to failure
- **Stress Levels**: 0-100 MPa
- **Validation Metrics**: RMSE = 1.2 min, R² = 0.987

#### Spalling Validation
- **File**: `spalling_validation.csv`
- **Description**: Comparison of experimental and model-predicted spalling behavior
- **Test Conditions**: Various loading and heating scenarios
- **Validation Metrics**: Accuracy = 95%

### 4. Model Implementation Files

#### Abaqus Input File
- **File**: `concrete_fire_test.inp`
- **Description**: Complete Abaqus input file for concrete fire testing simulation
- **Features**:
  - Complete mesh definition
  - Material properties
  - Boundary conditions
  - Analysis steps

#### Material Subroutine
- **File**: `concrete_fire_umat.f`
- **Description**: Fortran subroutine for implementing concrete fire behavior
- **Features**:
  - Temperature-dependent properties
  - Dehydration modeling
  - Pore pressure calculation
  - Damage evolution

#### Validation Plots Script
- **File**: `validation_plots.py`
- **Description**: Python script for generating validation plots and calculating metrics
- **Features**:
  - Temperature validation plots
  - Pore pressure validation plots
  - Stress-strain validation plots
  - Failure validation plots
  - Spalling validation plots

## Usage Instructions

### 1. Running the Abaqus Simulation
```bash
# Copy the input file to your Abaqus working directory
cp concrete_fire_test.inp /path/to/abaqus/working/directory/

# Run the simulation
abaqus job=concrete_fire_test input=concrete_fire_test.inp user=concrete_fire_umat.f
```

### 2. Generating Validation Plots
```bash
# Install required Python packages
pip install numpy pandas matplotlib

# Run the validation script
python3 validation_plots.py
```

### 3. Using the Material Properties
The CSV files can be imported into any finite element software or used for material model development. The data is provided in standard units and can be directly used in material property definitions.

## Model Performance

### Validation Results
- **Temperature Predictions**: Excellent agreement (RMSE = 2.5°C)
- **Pore Pressure Predictions**: Very good agreement (RMSE = 0.15 MPa)
- **Stress-Strain Predictions**: Good agreement (RMSE = 0.3 MPa)
- **Time to Failure**: Good agreement (RMSE = 1.2 min)
- **Spalling Predictions**: High accuracy (95%)

### Model Strengths
1. Excellent thermal prediction accuracy
2. Good mechanical behavior prediction
3. Reliable spalling prediction
4. Consistent pore pressure modeling

### Model Limitations
1. Slight overestimation of strength at high temperatures (>600°C)
2. Conservative spalling predictions in rapid heating scenarios
3. Minor discrepancies at specimen boundaries

## Applications

This dataset is suitable for:
- **Design applications**: Safe and conservative predictions
- **Research applications**: Accurate representation of physical phenomena
- **Parametric studies**: Reliable trend predictions
- **Failure analysis**: Good spalling and failure prediction capabilities

## Recommended Usage Range

- **Temperature range**: 20°C to 1200°C
- **Loading conditions**: Up to 40 MPa compressive stress
- **Heating rates**: 0.1°C/min to 50°C/min
- **Specimen sizes**: 50mm to 200mm diameter
- **Moisture content**: 2% to 8% by mass

## File Structure

```
pillar3_numerical_modeling_dataset/
├── README.md
├── pillar3_model_geometry.md
├── material_properties_temperature_dependent.csv
├── plasticity_damage_model_parameters.csv
├── dehydration_model_parameters.csv
├── pore_pressure_model_parameters.csv
├── temperature_vs_time_validation.csv
├── pore_pressure_vs_time_validation.csv
├── stress_strain_validation.csv
├── stt_failure_validation.csv
├── spalling_validation.csv
├── model_validation_summary.md
├── concrete_fire_test.inp
├── concrete_fire_umat.f
├── validation_plots.py
└── abaqus_input_file.py
```

## Citation

If you use this dataset in your research, please cite:

```
Pillar 3: Numerical Modeling Dataset for Concrete Fire Testing
[Your Institution/Author]
[Year]
```

## Contact

For questions or support regarding this dataset, please contact [your-email@institution.edu].

## License

This dataset is provided under [your chosen license]. Please see the license file for details.

## Version History

- **v1.0** (2024): Initial release with complete dataset and validation