# Thermo-Mechanical Modeling Dataset for Fire-Resistant Structural Elements

## Dataset Overview

This comprehensive numerical modeling dataset supports the research titled: **"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete."**

The dataset provides complete, structured, and physically consistent input parameters for finite element model development while maintaining independent validation datasets for rigorous model verification.

## Dataset Structure

```
thermo_mechanical_dataset/
├── README.md                           # This file
├── data_dictionary.md                  # Comprehensive data descriptions
├── thermal_properties/                 # Temperature-dependent thermal properties
│   ├── thermal_conductivity.csv
│   ├── specific_heat.csv
│   ├── thermal_diffusivity.csv
│   └── thermal_expansion.csv
├── mechanical_properties/              # Mechanical properties with degradation
│   ├── compressive_strength.csv
│   ├── tensile_strength.csv
│   ├── elastic_modulus.csv
│   ├── poissons_ratio.csv
│   └── fracture_properties.csv
├── transport_properties/               # Transport and poro-mechanical properties
│   ├── permeability.csv
│   ├── porosity.csv
│   ├── moisture_transport.csv
│   └── gas_transport.csv
├── deformation_properties/             # Time and temperature dependent deformation
│   ├── creep_parameters.csv
│   ├── shrinkage_parameters.csv
│   ├── thermal_strain.csv
│   └── damage_evolution.csv
├── microstructural_data/               # Multi-scale linking parameters
│   ├── pore_structure.csv
│   ├── fiber_distribution.csv
│   ├── matrix_properties.csv
│   └── interface_properties.csv
├── calibration_data/                   # Data for model parameter calibration
│   ├── experimental_curves/
│   ├── temperature_profiles/
│   └── stress_strain_data/
├── validation_data/                    # Independent validation datasets
│   ├── fire_tests/
│   ├── structural_tests/
│   └── durability_tests/
├── fea_formats/                        # FEA software ready formats
│   ├── abaqus/
│   ├── ansys/
│   └── comsol/
└── statistical_analysis/               # Statistical bounds and uncertainty
    ├── parameter_distributions.csv
    ├── correlation_matrices.csv
    └── uncertainty_bounds.csv
```

## Mix Compositions

The dataset covers six concrete mix types with varying rubber content:

- **C**: Control concrete (0% rubber)
- **R5S**: 5% small rubber particles
- **R10S**: 10% small rubber particles  
- **R15S**: 15% small rubber particles
- **R20S**: 20% small rubber particles
- **R10L**: 10% large rubber particles

## Temperature Range

All properties are provided as continuous functions from 20°C to 800°C with:
- 10°C increments for detailed analysis
- Physically-based degradation functions
- Smooth interpolation between data points

## Data Types

- **Calibration**: 70% of data for model parameter fitting
- **Validation**: 30% of data for independent model verification
- **Statistical**: Mean ± standard deviation for all parameters

## Usage Guidelines

1. **Model Development**: Use calibration data for parameter fitting
2. **Model Validation**: Use validation data for independent verification
3. **Uncertainty Analysis**: Use statistical bounds for probabilistic modeling
4. **Multi-Physics Coupling**: Ensure consistent property relationships across domains

## Quality Assurance

- Physical consistency checks across all property domains
- Temperature continuity verification
- Statistical distribution validation
- Multi-scale parameter linking verification

## Citation

When using this dataset, please cite:
```
[Research Title]: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete
Dataset Version: 1.0
Generated: October 2024
```

## Contact Information

For questions regarding this dataset, please contact the research team.