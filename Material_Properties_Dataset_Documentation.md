# Material Properties & Calibration Dataset Documentation

## Overview
This dataset provides comprehensive material properties data for building and calibrating multi-physics models, including traditional finite element analysis and AI-enhanced models. The data is structured for easy integration into simulation workflows and machine learning pipelines.

## Dataset Files

### 1. `material_properties_dataset.csv`
**Primary dataset containing fundamental material properties**

#### Structure:
- **Material_ID**: Unique identifier for each material (MAT001, MAT002)
- **Temperature_C**: Temperature in Celsius (°C)
- **Stress_MPa**: Applied stress in MPa (0 for non-mechanical properties)
- **Time_hours**: Time duration in hours (0 for instantaneous properties)
- **Property_Type**: Category of property (Mechanical, ThermoPhysical, Electrochemical)
- **Property_Name**: Specific property measured
- **Property_Value**: Measured value
- **Property_Unit**: Units of measurement
- **Test_Method**: ASTM standard or test method used
- **Notes**: Additional context or conditions

#### Properties Included:

**Mechanical Properties:**
- Tensile Strength (MPa) - Temperature dependent
- Young's Modulus (MPa) - Temperature dependent  
- Poisson's Ratio (dimensionless) - Temperature dependent

**Thermo-Physical Properties:**
- Coefficient of Thermal Expansion (CTE) (1/K) - Temperature dependent
- Thermal Conductivity (W/m·K) - Temperature dependent
- Specific Heat Capacity (J/kg·K) - Temperature dependent

**Electrochemical Properties:**
- Ionic Conductivity (S/m) - Temperature dependent
- Electronic Conductivity (S/m) - Temperature dependent

### 2. `creep_data.csv`
**Creep strain vs time curves for model calibration**

#### Structure:
- **Material_ID**: Material identifier
- **Temperature_C**: Test temperature (°C)
- **Stress_MPa**: Applied stress level (MPa)
- **Time_hours**: Time from test start (hours)
- **Creep_Strain_percent**: Accumulated creep strain (%)
- **Strain_Rate_per_hour**: Instantaneous strain rate (%/hour)
- **Test_Standard**: ASTM E139 standard
- **Notes**: Creep region classification

#### Creep Regions Covered:
- **Primary Creep**: Initial rapid strain accumulation
- **Secondary Creep**: Steady-state creep with constant strain rate
- **Transition**: Region between primary and secondary creep

## Temperature Range Coverage
- **Room Temperature**: 25°C (baseline)
- **Elevated Temperatures**: 100°C, 200°C, 300°C, 400°C, 500°C
- **Stress Levels**: 100 MPa, 150 MPa (for creep tests)
- **Time Range**: Up to 1000 hours for creep data

## Materials Included

### MAT001 (High-Performance Alloy)
- Higher strength and thermal conductivity
- Better creep resistance
- Suitable for high-temperature applications

### MAT002 (Standard Alloy)
- Moderate properties across all categories
- Good balance of cost and performance
- General-purpose applications

## Data Quality & Standards

### Test Standards Used:
- **ASTM E8**: Tensile testing
- **ASTM E831**: Thermal expansion
- **ASTM E1461**: Thermal conductivity
- **ASTM E1269**: Specific heat capacity
- **ASTM E139**: Creep testing
- **Electrochemical Impedance**: Ionic conductivity
- **Four-Point Probe**: Electronic conductivity

### Data Characteristics:
- **Temperature Dependencies**: All properties show realistic temperature sensitivity
- **Physical Consistency**: Properties follow expected physical relationships
- **Creep Behavior**: Realistic primary/secondary creep transitions
- **Units**: SI units throughout for consistency

## Usage Guidelines

### For Traditional Multi-Physics Models:
1. **Mechanical Models**: Use temperature-dependent elastic properties
2. **Thermal Models**: Apply temperature-dependent thermal properties
3. **Coupled Models**: Combine mechanical and thermal properties
4. **Creep Models**: Calibrate using creep strain vs time data

### For AI-Enhanced Models:
1. **Feature Engineering**: Temperature, stress, and time as input features
2. **Property Prediction**: Train models to predict properties at intermediate temperatures
3. **Creep Modeling**: Use strain rate data for time-dependent behavior prediction
4. **Multi-Physics Integration**: Combine different property types for comprehensive models

### Data Preprocessing Recommendations:
1. **Normalization**: Normalize temperature to [0,1] range
2. **Log Transformation**: Apply to creep strain rates for better model training
3. **Feature Scaling**: Standardize property values for neural networks
4. **Cross-Validation**: Use temperature-based splits for validation

## Integration Examples

### Python/Pandas:
```python
import pandas as pd

# Load material properties
props = pd.read_csv('material_properties_dataset.csv')

# Filter for specific material and property type
mat001_mechanical = props[(props['Material_ID'] == 'MAT001') & 
                         (props['Property_Type'] == 'Mechanical')]

# Load creep data
creep = pd.read_csv('creep_data.csv')
```

### MATLAB:
```matlab
% Load data
props = readtable('material_properties_dataset.csv');
creep = readtable('creep_data.csv');

% Filter data
mat001_mech = props(strcmp(props.Material_ID, 'MAT001') & ...
                   strcmp(props.Property_Type, 'Mechanical'), :);
```

## Model Calibration Workflow

1. **Data Validation**: Check temperature dependencies and physical consistency
2. **Property Interpolation**: Create continuous functions for intermediate temperatures
3. **Creep Model Fitting**: Fit Norton's law or other creep models to strain rate data
4. **Multi-Physics Coupling**: Establish relationships between different property types
5. **Validation**: Compare model predictions with experimental data
6. **Uncertainty Quantification**: Assess confidence intervals for predictions

## Notes & Limitations

- **Temperature Range**: Data limited to 25-500°C range
- **Stress Levels**: Creep data available for 100 and 150 MPa only
- **Time Range**: Creep tests limited to 1000 hours
- **Material Count**: Only two materials included (can be extended)
- **Property Interactions**: Some coupling effects may not be captured

## Future Extensions

- Additional materials and temperature ranges
- More stress levels for creep testing
- Fatigue properties and cyclic behavior
- Environmental effects (oxidation, corrosion)
- Microstructural dependencies
- Strain rate sensitivity data

## Contact & Support

For questions about this dataset or requests for additional data, please refer to the multi-physics modeling documentation or contact the materials engineering team.