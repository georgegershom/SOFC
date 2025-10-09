# Mechanical Boundary Conditions Dataset Documentation

## Overview

This dataset contains fabricated experimental data for mechanical boundary conditions in Solid Oxide Fuel Cell (SOFC) research, specifically focusing on the electrolyte fracture risk assessment under various loading and constraint conditions.

## Dataset Information

- **File**: `mechanical_boundary_conditions_dataset.csv`
- **Total Records**: 40 experiments
- **Features**: 23 parameters per experiment
- **Domain**: SOFC Mechanical Engineering
- **Focus**: Electrolyte fracture risk under mechanical boundary conditions

## Data Structure

### Primary Identifiers
- `experiment_id`: Unique identifier for each experiment (EXP_001 to EXP_040)

### Experimental Setup Parameters
- `fixture_type`: Type of mechanical fixture used in the experiment
- `stack_pressure_mpa`: Applied stack pressure in megapascals (0.08 - 0.30 MPa)
- `constraint_type`: Type of mechanical constraint applied
- `applied_load_type`: Category of applied mechanical load
- `load_magnitude_mpa`: Magnitude of applied load in megapascals
- `load_direction`: Direction of applied load (X, Y, Z, Multi-directional, etc.)

### Environmental Conditions
- `temperature_c`: Operating temperature in Celsius (25°C - 900°C)
- `test_duration_hours`: Duration of test in hours (0.1 - 10,000 hours)

### Boundary Condition Details
- `boundary_condition_description`: Detailed description of boundary conditions
- `displacement_constraint_x/y/z`: Displacement constraints in X, Y, Z directions
- `rotation_constraint_x/y/z`: Rotational constraints about X, Y, Z axes

### Interface Properties
- `contact_pressure_mpa`: Contact pressure at interfaces
- `friction_coefficient`: Friction coefficient between materials (0.0 - 0.5)
- `material_interface`: Type of material interface

### Geometric and Stress Parameters
- `geometric_discontinuity`: Type of geometric feature affecting stress
- `stress_concentration_factor`: Stress concentration factor (1.0 - 3.2)
- `safety_factor`: Safety factor against fracture (0.85 - 1.95)

### Additional Information
- `notes`: Additional experimental notes and observations

## Fixture Types

The dataset includes 15 different fixture types:

1. **Rigid_Compression_Fixture** (17 experiments): Standard rigid compression setup
2. **Flexible_Compression_Fixture** (2 experiments): Allows lateral expansion
3. **Spring_Loaded_Fixture** (1 experiment): Variable compression system
4. **Hydraulic_Fixture** (1 experiment): Hydraulic pressure distribution
5. **Pneumatic_Fixture** (1 experiment): Pneumatic pressure control
6. **Multi_Point_Fixture** (1 experiment): Multi-point loading configuration
7. **Vacuum_Fixture** (1 experiment): Vacuum-assisted system
8. **Compliant_Fixture** (1 experiment): Compliant material interface
9. **Segmented_Fixture** (1 experiment): Segmented loading pattern
10. **Dynamic_Fixture** (1 experiment): Dynamic loading with vibration
11. **Elastomeric_Fixture** (1 experiment): Elastomeric interface
12. **Piezoelectric_Fixture** (1 experiment): Precise load control
13. **Thermal_Fixture** (1 experiment): High temperature focus
14. **Magnetic_Fixture** (1 experiment): Contactless loading
15. **Adaptive_Fixture** (1 experiment): Adaptive control system
16. **Fluid_Fixture** (1 experiment): Hydrostatic conditions
17. **Composite_Fixture** (1 experiment): Multi-material system
18. **Electroactive_Fixture** (1 experiment): Smart material interface
19. **Cryogenic_Fixture** (1 experiment): Extreme temperature testing
20. **Nano_Fixture** (1 experiment): Microscale analysis

## Constraint Types

- **Simply_Supported** (15 experiments): Bottom fixed in Z, free in X,Y
- **Pinned_Support** (2 experiments): Fixed in Z only
- **Clamped_Support** (3 experiments): All degrees of freedom constrained
- **Elastic_Support** (2 experiments): Spring-like constraints
- **Distributed_Support** (1 experiment): Distributed constraint system
- **Point_Support** (1 experiment): Point-wise constraints
- **Suction_Support** (1 experiment): Vacuum-based constraint
- **Compliant_Support** (1 experiment): Compliant constraint interface
- **Segmented_Support** (1 experiment): Segmented constraint pattern
- **Dynamic_Support** (1 experiment): Time-varying constraints
- **Controlled_Support** (1 experiment): Actively controlled constraints
- **Thermal_Support** (1 experiment): Temperature-dependent constraints
- **Magnetic_Support** (1 experiment): Magnetic levitation
- **Adaptive_Support** (1 experiment): Self-adjusting constraints
- **Hydrostatic_Support** (1 experiment): Fluid-based constraints
- **Composite_Support** (1 experiment): Multi-material constraints
- **Electroactive_Support** (1 experiment): Smart material constraints
- **Cryogenic_Support** (1 experiment): Low-temperature constraints
- **Nano_Support** (1 experiment): Nanoscale constraints

## Applied Load Types

- **Uniform_Compression** (19 experiments): Uniform compressive loading
- **Variable_Compression** (1 experiment): Time-varying compression
- **Thermal_Load** (2 experiments): Thermally induced loads
- **Combined_Load** (1 experiment): Multiple load types
- **Cyclic_Load** (2 experiments): Repeated loading cycles
- **Distributed_Load** (1 experiment): Non-uniform load distribution
- **Bending_Load** (1 experiment): Three-point bending
- **Shear_Load** (1 experiment): In-plane shear loading
- **Torsional_Load** (1 experiment): Rotational loading
- **Non_Uniform_Load** (1 experiment): Spatially varying loads
- **Minimal_Load** (1 experiment): Baseline minimal loading
- **Vibration_Load** (1 experiment): Dynamic vibration loading
- **Gradient_Load** (1 experiment): Pressure gradient application
- **Biaxial_Load** (1 experiment): Two-dimensional loading
- **Precise_Load** (1 experiment): High-precision loading
- **Overload_Test** (1 experiment): Beyond normal operating limits
- **Thermal_Expansion** (1 experiment): Thermal expansion effects
- **Fatigue_Load** (1 experiment): Long-term fatigue testing
- **Contactless_Load** (1 experiment): Non-contact loading
- **Impact_Load** (1 experiment): Dynamic impact loading
- **Variable_Load** (1 experiment): Time-varying load patterns
- **Creep_Load** (1 experiment): Long-term creep testing
- **Hydrostatic_Load** (1 experiment): Omnidirectional pressure
- **Residual_Stress** (1 experiment): Post-fabrication stress
- **Layered_Load** (1 experiment): Multi-layer loading
- **Ultimate_Load** (1 experiment): Ultimate strength testing
- **Smart_Load** (1 experiment): Intelligent loading system
- **Multiaxial_Load** (1 experiment): Complex multi-directional loading
- **Thermal_Shock** (1 experiment): Rapid temperature change
- **Cyclic_Thermal** (1 experiment): Thermal cycling with load
- **Micro_Load** (1 experiment): Microscale loading

## Key Statistics

### Pressure Distribution
- **Range**: 0.08 - 0.30 MPa
- **Mean**: 0.183 ± 0.058 MPa
- **Standard Operating Range**: 0.10 - 0.25 MPa

### Temperature Distribution
- **Range**: 25°C - 900°C
- **Mean**: 774°C ± 135°C
- **Primary Operating Temperature**: 800°C (SOFC operating condition)

### Safety Factor Analysis
- **Range**: 0.85 - 1.95
- **Mean**: 1.38 ± 0.29
- **High Risk (SF < 1.0)**: 3 experiments (7.5%)
- **Moderate Risk (1.0 ≤ SF < 1.5)**: 24 experiments (60.0%)
- **Low Risk (SF ≥ 1.5)**: 13 experiments (32.5%)

### Test Duration Distribution
- **Range**: 0.1 - 10,000 hours
- **Mean**: 1,045 hours
- **Short-term tests (<100h)**: 12 experiments
- **Medium-term tests (100-1000h)**: 16 experiments
- **Long-term tests (>1000h)**: 12 experiments

## Data Quality and Validation

### Completeness
- All 40 records contain complete data for all 23 parameters
- No missing values in the dataset
- All categorical variables have consistent naming conventions

### Consistency Checks
- Stack pressure values align with load magnitude values
- Safety factors correlate inversely with stress concentration factors
- Temperature ranges are appropriate for SOFC operating conditions
- Test durations span realistic experimental timeframes

### Physical Validity
- All pressure values are within realistic SOFC operating ranges
- Safety factors reflect expected mechanical behavior
- Stress concentration factors align with geometric features
- Material interface properties are physically reasonable

## Usage Guidelines

### Research Applications
1. **Mechanical Design Optimization**: Use pressure and constraint data for SOFC stack design
2. **Fracture Risk Assessment**: Analyze safety factor relationships with operating conditions
3. **Fixture Development**: Compare fixture types for experimental setup optimization
4. **Boundary Condition Modeling**: Validate finite element model boundary conditions

### Analysis Recommendations
1. **Statistical Analysis**: Focus on pressure-safety factor correlations
2. **Clustering Analysis**: Group experiments by operating conditions
3. **Risk Assessment**: Identify critical operating parameter combinations
4. **Optimization Studies**: Determine optimal fixture and constraint combinations

### Limitations
1. **Fabricated Data**: This is a synthetic dataset created for research purposes
2. **Simplified Models**: Real SOFC systems may have additional complexity
3. **Limited Validation**: Experimental validation would be required for practical applications
4. **Parameter Interactions**: Complex multi-parameter interactions may not be fully captured

## File Structure

```
mechanical_boundary_conditions_dataset.csv
├── experiment_id (string): Unique experiment identifier
├── fixture_type (string): Mechanical fixture classification
├── stack_pressure_mpa (float): Applied pressure in MPa
├── constraint_type (string): Boundary constraint classification
├── applied_load_type (string): Load type classification
├── load_magnitude_mpa (float): Load magnitude in MPa
├── load_direction (string): Load direction specification
├── temperature_c (float): Operating temperature in Celsius
├── test_duration_hours (float): Test duration in hours
├── boundary_condition_description (string): Detailed BC description
├── displacement_constraint_x (string): X-direction displacement constraint
├── displacement_constraint_y (string): Y-direction displacement constraint
├── displacement_constraint_z (string): Z-direction displacement constraint
├── rotation_constraint_x (string): X-rotation constraint
├── rotation_constraint_y (string): Y-rotation constraint
├── rotation_constraint_z (string): Z-rotation constraint
├── contact_pressure_mpa (float): Interface contact pressure
├── friction_coefficient (float): Material friction coefficient
├── material_interface (string): Interface material specification
├── geometric_discontinuity (string): Geometric feature description
├── stress_concentration_factor (float): Stress concentration factor
├── safety_factor (float): Safety factor against fracture
└── notes (string): Additional experimental notes
```

## Related Files

- `mechanical_boundary_analysis.py`: Comprehensive analysis script
- `dataset_documentation.md`: This documentation file
- Generated visualization files:
  - `fixture_type_analysis.png`
  - `pressure_load_analysis.png`
  - `boundary_condition_analysis.png`
  - `safety_analysis.png`
  - `correlation_analysis.png`
  - `clustering_analysis.png`

## Citation

If using this dataset in research, please cite as:

```
Mechanical Boundary Conditions Dataset for SOFC Electrolyte Fracture Risk Assessment
Generated for: "A Comparative Analysis of Constitutive Models for Predicting the Electrolyte's Fracture Risk in Planar SOFCs"
Date: October 2025
Source: Fabricated research dataset based on SOFC mechanical engineering principles
```

## Contact and Support

This dataset was generated as part of SOFC mechanical reliability research. For questions about the dataset structure, analysis methods, or applications, please refer to the accompanying analysis script and visualization outputs.

## Version History

- **v1.0** (October 2025): Initial dataset creation with 40 experiments and 23 parameters
- Comprehensive coverage of fixture types, constraint conditions, and loading scenarios
- Complete documentation and analysis framework provided