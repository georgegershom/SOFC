# Mechanical Boundary Conditions Dataset

## Overview

This dataset contains comprehensive information on mechanical boundary conditions for Solid Oxide Fuel Cell (SOFC) experimental setups and simulations. The dataset is designed to support fracture risk assessment, stress analysis, and durability prediction of planar SOFC electrolytes under various loading conditions.

## Dataset Information

- **Version:** 1.0
- **Date Created:** 2025-10-09
- **Total Records:** 50 test configurations
- **Format:** CSV and JSON

## Source

The dataset is derived from experimental setup details and simulation parameters for planar SOFC electrolyte fracture risk assessment. It encompasses:

1. **Experimental Setup Details:**
   - Fixture types and materials
   - Applied stack pressure configurations
   - Assembly methods and clamping systems

2. **External Constraints:**
   - Support configurations (simply supported, clamped, pinned, etc.)
   - Boundary condition types
   - Geometric constraints

3. **Applied Loads:**
   - Compressive loads
   - Flexural loads
   - Thermal cycling loads
   - Combined multi-physics loading

## Data Structure

### Fields Description

| Field Name | Description | Units |
|------------|-------------|-------|
| `Test_ID` | Unique identifier for each test configuration | - |
| `Fixture_Type` | Type of mechanical fixture used | - |
| `Fixture_Material` | Material composition of the fixture | - |
| `Applied_Stack_Pressure_MPa` | Stack pressure applied | MPa |
| `Pressure_Distribution` | Spatial distribution pattern of pressure | - |
| `Constraint_Type` | Type of mechanical constraint | - |
| `Support_Location` | Location where support is applied | - |
| `Support_Configuration` | Detailed configuration of support system | - |
| `Applied_Load_Type` | Nature of applied load | - |
| `Load_Magnitude_N` | Magnitude of applied load | N |
| `Load_Direction` | Direction of load application | - |
| `Temperature_C` | Operating temperature | °C |
| `Contact_Area_cm2` | Contact area between fixture and cell | cm² |
| `Clamping_Force_N` | Total clamping force | N |
| `Boundary_Condition_Category` | Category classification | - |
| `Assembly_Method` | Method used to assemble the stack | - |
| `Gasket_Type` | Type of gasket material | - |
| `Gasket_Thickness_mm` | Thickness of gasket | mm |
| `Sealing_Force_N` | Force applied for sealing | N |
| `Cell_Orientation` | Orientation of cell during testing | - |
| `Fixture_Compliance_mm_N` | Compliance of the fixture | mm/N |
| `Notes` | Additional notes and observations | - |

## Key Parameters

### Pressure Ranges
- **Minimum Sealing Pressure:** 0.05 MPa
- **Typical Operating Pressure:** 0.20 MPa
- **Maximum Test Pressure:** 0.35 MPa
- **Recommended Range:** 0.10 - 0.30 MPa

### Temperature Conditions
- **Room Temperature:** 25°C
- **Standard Operating Temperature:** 800°C
- **Sintering Temperature:** 1350°C
- **Low Operating Temperature:** 600°C
- **High Operating Temperature:** 1000°C

### Fixture Types
1. Rigid Compression Fixture
2. Spring Loaded Fixture
3. Four-Point Bending Fixture
4. Clamped Edges Fixture
5. Hydraulic Compression Fixture
6. Pneumatic Compression Fixture
7. Compliant Fixture

### Constraint Types
1. Simply Supported
2. Four-Point Support
3. Fully Clamped
4. Symmetry Planes
5. Pinned Support
6. Roller Support
7. Line Support
8. Fully Fixed

### Load Types
1. Compressive
2. Flexural
3. Thermal Cycling
4. Combined Thermal-Mechanical
5. Compressive with Shear

### Gasket Materials
1. Mica
2. Vermiculite
3. Glass-Ceramic
4. None (for testing fixtures)

## Usage Examples

### Standard Operating Condition
- Test ID: MBC_001
- Fixture: Rigid Compression Fixture (Alumina Ceramic)
- Pressure: 0.20 MPa (uniform distribution)
- Constraint: Simply Supported
- Temperature: 800°C
- Load: 200 N compressive

### Thermal Cycling Test
- Test IDs: MBC_018, MBC_019
- Temperature Range: 25°C to 800°C
- Heating/Cooling Cycles
- Same mechanical constraints as operational

### High Pressure Test
- Test ID: MBC_003
- Pressure: 0.30 MPa
- Purpose: Upper limit stress assessment

### Low Pressure Test
- Test ID: MBC_004
- Pressure: 0.10 MPa
- Purpose: Minimum sealing requirement

### Compliant Fixture
- Test ID: MBC_005
- Spring-loaded system
- Compensates for thermal expansion
- Higher fixture compliance (0.005 mm/N)

## Applications

This dataset is suitable for:

1. **Finite Element Analysis (FEA)**
   - Boundary condition setup for SOFC simulations
   - Stress analysis and fracture risk assessment
   - Multi-physics coupled simulations

2. **Experimental Design**
   - Test fixture design and selection
   - Operating parameter optimization
   - Validation of simulation models

3. **Durability Assessment**
   - Long-term stress evolution prediction
   - Thermal cycling fatigue analysis
   - Creep and stress relaxation studies

4. **Design Optimization**
   - Stack assembly optimization
   - Gasket selection and sizing
   - Pressure and temperature optimization

## Data Quality

- All configurations are based on realistic SOFC operating conditions
- Pressure values range from minimum sealing (0.05 MPa) to extreme testing (0.35 MPa)
- Temperature conditions span from room temperature to sintering conditions
- Multiple fixture types and constraint configurations for comprehensive coverage
- Repeat tests included for statistical validation (MBC_046, MBC_047, MBC_048)

## Related Files

1. **mechanical_boundary_conditions_dataset.csv** - Complete dataset in CSV format
2. **mechanical_boundary_conditions_dataset.json** - Complete dataset in JSON format with metadata
3. **mechanical_boundary_conditions_README.md** - This documentation file

## References

- Research Article: "A Comparative Analysis of Constitutive Models for Predicting the Electrolyte's Fracture Risk in Planar SOFCs"
- Section 2.3: Boundary Conditions and Load Cases

## Contact & Citation

When using this dataset, please cite the source research article and mention the mechanical boundary conditions dataset version 1.0.

## License

This dataset is provided for research and educational purposes.

---

**Last Updated:** 2025-10-09
**Version:** 1.0
