# Mechanical Boundary Conditions Dataset

## Overview
This dataset contains comprehensive information about mechanical boundary conditions used in experimental testing setups. It includes fixture specifications, applied loads, constraints, and time-series data from various mechanical tests.

## Dataset Structure

### 1. Main Dataset Files

#### mechanical_boundary_conditions.csv
Primary dataset containing 30 experimental records with the following information:
- **Experiment Identification**: experiment_id, timestamp
- **Fixture Information**: fixture_type, fixture_material, clamping_force_N
- **Pressure Conditions**: stack_pressure_MPa
- **Environmental Conditions**: temperature_C, humidity_percent
- **Loading Conditions**: preload_N, applied_load_type, applied_load_magnitude, load_direction, load_frequency_Hz
- **Boundary Constraints**: constraint_x/y/z, rotation_x/y/z_constrained
- **Sample Properties**: sample_thickness_mm, sample_area_mm2
- **Additional Notes**: Test descriptions and special conditions

#### fixture_specifications.json
Detailed specifications for 10 different fixture types including:
- Fixture materials and dimensions
- Maximum load capacities
- Temperature ranges
- Applicable test types
- Compatible sample geometries
- Mechanical properties (stiffness, tolerance, surface finish)

#### applied_loads_timeseries.csv
Time-series data showing load evolution during experiments:
- Time stamps in seconds
- Applied forces (N)
- Displacements (mm)
- Strain and stress values
- Temperature variations
- Cycle numbers for fatigue tests
- Load stages (Initial, Loading, Peak, Unloading, etc.)

#### constraint_conditions.json
Comprehensive constraint and boundary condition definitions:
- Degrees of freedom specifications
- Stiffness matrices
- Contact properties
- Friction coefficients
- Damping characteristics
- Preload conditions

## Key Features

### Fixture Types Included
- Fixed-Fixed Clamps
- Hydraulic Grips
- Four-Point Bend Fixtures
- Compression Platens
- Vacuum Chucks
- Pin Joints
- Electromagnetic Fixtures
- Collet Chucks
- Environmental Chamber Fixtures
- Wedge Grips

### Load Types Covered
- **Static**: Constant loads
- **Cyclic**: Sinusoidal varying loads with frequencies 0.1-1000 Hz
- **Ramp**: Linearly increasing loads
- **Stepped**: Discrete load increments
- **Random**: Stochastic loading patterns
- **Impulse**: Short duration high magnitude loads

### Boundary Condition Types
- **Fixed**: All DOF constrained
- **Pinned**: Translations constrained, rotations free
- **Roller**: One translation free
- **Free**: No constraints
- **Elastic**: Spring-like boundaries
- **Magnetic**: Electromagnetic holding
- **Vacuum**: Vacuum-based holding

## Data Ranges

### Typical Operating Conditions
- **Temperature Range**: -180°C to 450°C
- **Pressure Range**: 1.0 to 8.5 MPa
- **Load Range**: 0 to 5000 N
- **Frequency Range**: 0.1 to 1000 Hz
- **Sample Thickness**: 2.8 to 25.0 mm
- **Sample Area**: 180 to 2000 mm²

### Material Properties
- Various fixture materials including:
  - Stainless Steel (304, 316)
  - Aluminum Alloys (6061, 7075)
  - Tool Steel
  - Titanium Alloys
  - Specialized materials (Invar, Inconel)

## Use Cases

This dataset is suitable for:
1. **Finite Element Analysis (FEA)**: Setting up boundary conditions for simulations
2. **Test Planning**: Selecting appropriate fixtures and constraints
3. **Data Analysis**: Understanding load-displacement relationships
4. **Machine Learning**: Training models for predicting material behavior
5. **Standards Compliance**: Verifying test setups against ASTM/ISO standards
6. **Educational Purposes**: Teaching mechanical testing concepts

## Data Format

### CSV Files
- Comma-separated values
- Headers included
- Timestamps in ISO format (YYYY-MM-DD HH:MM:SS)
- Numerical values with appropriate precision

### JSON Files
- Structured hierarchical data
- Arrays for multiple entries
- Nested objects for complex relationships
- Standard JSON formatting

## Quality Assurance

All data has been generated with:
- Realistic value ranges based on typical experimental setups
- Consistent units (SI units primarily)
- Proper relationships between related parameters
- Physical constraints respected (e.g., compression tests have negative Z-direction loads)

## Notes on Usage

1. **Data is Fabricated**: This is synthetic data created for demonstration and testing purposes
2. **Physical Consistency**: Values follow realistic physical relationships
3. **Standards Alignment**: Test configurations align with common ASTM and ISO standards
4. **Extensibility**: JSON structure allows easy addition of new fixtures and conditions

## File Summary

| File Name | Type | Records/Entries | Description |
|-----------|------|-----------------|-------------|
| mechanical_boundary_conditions.csv | CSV | 30 experiments | Main experimental dataset |
| fixture_specifications.json | JSON | 10 fixtures | Detailed fixture specifications |
| applied_loads_timeseries.csv | CSV | 100+ time points | Time-series load data |
| constraint_conditions.json | JSON | 10 constraints | Boundary condition definitions |

## Version Information
- **Created**: 2024
- **Version**: 1.0
- **Format Version**: 1.0

## Contact
For questions or additional information about this dataset, please refer to the experimental setup documentation or contact the laboratory personnel.

---

*This dataset represents typical mechanical boundary conditions found in material testing laboratories and can be used as reference data for simulation setup, test planning, and educational purposes.*