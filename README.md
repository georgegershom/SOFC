# Thermo-Mechanical Model Dataset for Fire-Resistant Rubberized Concrete

## Overview
This comprehensive dataset supports the development and validation of thermo-mechanical models for fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset is specifically designed for Phase 4 of numerical modeling research.

## Dataset Structure

### 1. Material Properties Data

#### Thermal Properties (`thermal_properties_dataset.csv`)
- **Temperature range**: 20°C to 1000°C
- **Concrete types**: Control concrete and rubberized concrete (5%, 10%, 15%, 20% rubber content)
- **Properties**: Thermal conductivity, specific heat, density
- **Source**: Literature-based with experimental validation

#### Mechanical Properties (`mechanical_properties_dataset.csv`)
- **Temperature range**: 20°C to 1000°C
- **Concrete types**: Control concrete and rubberized concrete variants
- **Properties**: Compressive strength, tensile strength, elastic modulus, Poisson ratio
- **Source**: Experimental data from in-situ tests

#### Deformation Properties (`deformation_properties_dataset.csv`)
- **Temperature range**: 20°C to 1000°C
- **Concrete types**: Control concrete and rubberized concrete variants
- **Properties**: Coefficient of Thermal Expansion (CTE), Transient Thermal Strain, Free Strain, Load-Induced Thermal Strain
- **Source**: Dilatometry and experimental measurements

#### Poro-Mechanical Properties (`poro_mechanical_properties_dataset.csv`)
- **Temperature range**: 20°C to 1000°C
- **Concrete types**: Control concrete and rubberized concrete variants
- **Properties**: Porosity, permeability, water content, damage index
- **Source**: Experimental measurements and literature data

### 2. Model Validation Data

#### Temperature Evolution (`model_validation_temperature_evolution.csv`)
- **Time range**: 0 to 60 minutes
- **Measurement points**: Surface, 25mm, 50mm, 75mm, 100mm depth
- **Concrete types**: All variants tested
- **Purpose**: Model calibration and validation

#### Strain History (`model_validation_strain_history.csv`)
- **Time range**: 0 to 60 minutes
- **Strain components**: Axial, lateral, shear
- **Load levels**: 0.3 (30% of ultimate strength)
- **Purpose**: Mechanical behavior validation

#### Spalling Data (`model_validation_spalling_data.csv`)
- **Parameters**: Time to first spall, time to failure, spall depth, spall area
- **Load levels**: 0.4 to 0.8 (40% to 80% of ultimate strength)
- **Moisture content**: 4.2% to 5.4%
- **Purpose**: Spalling prediction validation

#### Fire Curves (`fire_curves_standard.csv`)
- **Standards**: ISO 834, BS 476, ASTM E119, EC1
- **Time range**: 0 to 180 minutes
- **Purpose**: Standard fire exposure conditions

### 3. Configuration Files

#### Numerical Modeling Input (`numerical_modeling_input_data.json`)
- **Modeling parameters**: Mesh density, time step, convergence criteria
- **Boundary conditions**: Thermal and mechanical
- **Output parameters**: Field variables for analysis
- **Purpose**: Direct input for numerical modeling software

## Usage Instructions

### For Model Calibration
1. Use thermal and mechanical properties data to define material models
2. Apply deformation properties for thermal expansion calculations
3. Incorporate poro-mechanical properties for moisture transport

### For Model Validation
1. Use temperature evolution data to validate thermal analysis
2. Compare strain history predictions with experimental data
3. Validate spalling predictions using spalling data

### For Fire Analysis
1. Apply appropriate fire curve from standard fire curves
2. Use temperature-dependent material properties
3. Consider coupled thermal-mechanical analysis

## Data Quality and Reliability

- **Literature-based**: Properties derived from peer-reviewed research
- **Experimentally validated**: Critical properties verified through testing
- **Temperature-dependent**: All properties vary with temperature
- **Comprehensive coverage**: Full range of rubber content and temperature

## File Formats

- **CSV files**: Comma-separated values for easy import into modeling software
- **JSON file**: Structured configuration for automated model setup
- **Markdown**: Documentation and usage instructions

## Applications

- Finite Element Analysis (FEA)
- Finite Difference Methods
- Computational Fluid Dynamics (CFD)
- Multi-physics simulations
- Fire safety engineering
- Structural design optimization

## Contact Information

For questions or additional data requirements, please contact the research team.

## Version History

- **v1.0**: Initial comprehensive dataset release
  - Complete material properties database
  - Model validation datasets
  - Standard fire curves
  - Configuration files