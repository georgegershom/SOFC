# Usage Guidelines for Thermo-Mechanical Modeling Dataset

## Overview

This document provides comprehensive guidelines for effectively using the thermo-mechanical modeling dataset for fire-resistant structural elements utilizing high-performance rubberized concrete. It covers model development workflows, best practices, and application-specific recommendations.

## Getting Started

### Dataset Structure Navigation
```
thermo_mechanical_dataset/
├── README.md                    # Start here for overview
├── data_dictionary.md           # Detailed variable descriptions
├── validation_procedures.md     # Model validation guidelines
├── usage_guidelines.md          # This document
├── thermal_properties/          # Temperature-dependent thermal data
├── mechanical_properties/       # Mechanical behavior data
├── transport_properties/        # Poro-mechanical transport data
├── deformation_properties/      # Time-dependent deformation data
├── statistical_analysis/        # Uncertainty and correlation data
├── fea_formats/                 # Ready-to-use FEA software files
└── calibration_data/           # Experimental calibration datasets
```

### Quick Start Workflow
1. **Identify Application**: Determine your modeling objectives and required properties
2. **Select Mix Type**: Choose appropriate concrete mix based on rubber content requirements
3. **Load Data**: Import relevant datasets for your temperature range
4. **Validate Bounds**: Check that your application falls within dataset validity ranges
5. **Implement Model**: Use provided FEA formats or extract data for custom implementation
6. **Validate Results**: Follow validation procedures to ensure model reliability

## Application-Specific Guidelines

### Fire Resistance Modeling

#### Suitable Applications
- Building fire scenarios (ISO 834, ASTM E119)
- Hydrocarbon fire exposure (ISO 22899)
- Structural element fire testing
- Fire safety design optimization

#### Recommended Approach
```python
# Example workflow for fire resistance modeling
def setup_fire_resistance_model(mix_type, fire_curve_type):
    """
    Setup model for fire resistance analysis.
    """
    # Load thermal properties
    thermal_data = load_thermal_properties(mix_type)
    
    # Load mechanical properties
    mechanical_data = load_mechanical_properties(mix_type)
    
    # Load deformation properties for thermal stress analysis
    deformation_data = load_deformation_properties(mix_type)
    
    # Configure temperature-dependent properties
    properties = configure_temperature_dependence(
        thermal_data, mechanical_data, deformation_data
    )
    
    # Apply fire boundary conditions
    if fire_curve_type == 'ISO834':
        boundary_conditions = setup_iso834_fire()
    elif fire_curve_type == 'hydrocarbon':
        boundary_conditions = setup_hydrocarbon_fire()
    
    return properties, boundary_conditions
```

#### Key Considerations
- **Temperature Range**: Ensure fire temperatures don't exceed 800°C dataset limit
- **Heating Rate**: Consider transient effects for rapid heating scenarios
- **Moisture Effects**: Include moisture transport for accurate spalling prediction
- **Damage Coupling**: Link thermal damage to mechanical property degradation

### Structural Analysis at Elevated Temperatures

#### Suitable Applications
- Post-fire structural assessment
- High-temperature industrial applications
- Thermal stress analysis
- Long-term durability studies

#### Material Selection Criteria
| Application | Recommended Mix | Reasoning |
|-------------|----------------|-----------|
| High strength requirement | C or R5S | Minimal strength reduction |
| Improved toughness | R10S or R15S | Enhanced fracture resistance |
| Thermal insulation | R20S | Lower thermal conductivity |
| Large particle effects | R10L | Different microstructural response |

#### Implementation Strategy
```python
def structural_analysis_setup(mix_type, temperature_profile, load_history):
    """
    Setup for structural analysis at elevated temperatures.
    """
    # Load mechanical properties with temperature dependence
    mechanical_props = load_mechanical_properties(mix_type)
    
    # Configure creep behavior for long-term loading
    creep_params = load_creep_parameters(mix_type)
    
    # Setup damage evolution
    damage_params = load_damage_evolution(mix_type)
    
    # Create constitutive model
    constitutive_model = create_thermo_mechanical_model(
        mechanical_props, creep_params, damage_params
    )
    
    return constitutive_model
```

### Durability and Service Life Prediction

#### Suitable Applications
- Long-term performance assessment
- Maintenance scheduling optimization
- Life cycle cost analysis
- Climate change impact studies

#### Time-Dependent Modeling Approach
```python
def durability_analysis(mix_type, service_conditions, analysis_period):
    """
    Setup durability analysis with time-dependent degradation.
    """
    # Load transport properties for environmental ingress
    transport_props = load_transport_properties(mix_type)
    
    # Load shrinkage and creep for long-term deformation
    deformation_props = load_deformation_properties(mix_type)
    
    # Configure environmental exposure
    exposure_model = setup_environmental_exposure(service_conditions)
    
    # Time integration for long-term analysis
    time_steps = generate_time_steps(analysis_period)
    
    return transport_props, deformation_props, exposure_model, time_steps
```

## Multi-Physics Modeling Guidelines

### Coupled Thermo-Mechanical Analysis

#### Coupling Strategy
1. **Weak Coupling**: Sequential solution of thermal and mechanical problems
2. **Strong Coupling**: Simultaneous solution with full coupling matrix
3. **Staggered Coupling**: Iterative solution with convergence checking

#### Implementation Considerations
```python
def setup_thermo_mechanical_coupling(mix_type, coupling_type='strong'):
    """
    Configure thermo-mechanical coupling.
    """
    # Thermal properties
    thermal_props = {
        'conductivity': load_thermal_conductivity(mix_type),
        'specific_heat': load_specific_heat(mix_type),
        'density': load_density(mix_type)
    }
    
    # Mechanical properties with temperature dependence
    mechanical_props = {
        'elastic_modulus': load_elastic_modulus(mix_type),
        'poissons_ratio': load_poissons_ratio(mix_type),
        'thermal_expansion': load_thermal_expansion(mix_type)
    }
    
    # Coupling parameters
    coupling_params = {
        'mechanical_heat_generation': True,  # Plastic dissipation
        'thermal_stress_coupling': True,     # Thermal expansion
        'damage_thermal_coupling': True      # Damage affects conductivity
    }
    
    return thermal_props, mechanical_props, coupling_params
```

### Poro-Mechanical Analysis

#### Applications
- Moisture-induced damage
- Freeze-thaw cycling
- Chemical attack modeling
- Permeability evolution

#### Setup Procedure
```python
def setup_poro_mechanical_model(mix_type):
    """
    Configure poro-mechanical analysis.
    """
    # Transport properties
    transport_props = {
        'permeability': load_permeability(mix_type),
        'porosity': load_porosity(mix_type),
        'moisture_diffusivity': load_moisture_diffusivity(mix_type)
    }
    
    # Mechanical coupling
    mechanical_coupling = {
        'effective_stress_principle': True,
        'porosity_strain_coupling': True,
        'permeability_damage_coupling': True
    }
    
    return transport_props, mechanical_coupling
```

## Data Interpolation and Extrapolation

### Temperature Interpolation
```python
def interpolate_temperature_dependent_property(property_data, target_temperature):
    """
    Interpolate property values for intermediate temperatures.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Extract temperature and property values
    temperatures = property_data['Temperature_C'].values
    mean_values = property_data[property_data.columns[property_data.columns.str.contains('_Mean')][0]].values
    
    # Create interpolation function
    interp_func = interp1d(temperatures, mean_values, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    
    # Interpolate for target temperature
    interpolated_value = interp_func(target_temperature)
    
    # Warning for extrapolation
    if target_temperature < temperatures.min() or target_temperature > temperatures.max():
        print(f"Warning: Extrapolating beyond data range for T={target_temperature}°C")
    
    return interpolated_value
```

### Uncertainty Propagation
```python
def propagate_uncertainty(property_data, target_temperature, confidence_level=0.95):
    """
    Propagate uncertainty through interpolation.
    """
    from scipy import stats
    
    # Interpolate mean and standard deviation
    mean_value = interpolate_temperature_dependent_property(property_data, target_temperature)
    
    # Find nearest data points for uncertainty estimation
    temp_diff = np.abs(property_data['Temperature_C'] - target_temperature)
    nearest_idx = temp_diff.idxmin()
    
    std_col = property_data.columns[property_data.columns.str.contains('_Std')][0]
    std_value = property_data.loc[nearest_idx, std_col]
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=10)  # Assume 10 DOF
    
    confidence_interval = (
        mean_value - t_value * std_value,
        mean_value + t_value * std_value
    )
    
    return mean_value, std_value, confidence_interval
```

## Software-Specific Implementation

### ABAQUS Implementation
```python
def generate_abaqus_input(mix_type, analysis_type='thermal_stress'):
    """
    Generate ABAQUS input file sections.
    """
    # Load material data
    material_data = load_material_data(mix_type)
    
    # Generate material definition
    abaqus_material = f"""
*MATERIAL, NAME={mix_type}_CONCRETE
*DENSITY
{material_data['density']:.1f},
*ELASTIC, TYPE=ISOTROPIC
"""
    
    # Add temperature-dependent elastic properties
    for temp, E, nu in zip(material_data['temperature'], 
                          material_data['elastic_modulus'],
                          material_data['poissons_ratio']):
        abaqus_material += f"{E:.0f}, {nu:.4f}, {temp:.0f}\n"
    
    # Add thermal properties
    abaqus_material += "*CONDUCTIVITY\n"
    for temp, k in zip(material_data['temperature'], 
                      material_data['thermal_conductivity']):
        abaqus_material += f"{k:.4f}, {temp:.0f}\n"
    
    return abaqus_material
```

### ANSYS Implementation
```python
def generate_ansys_commands(mix_type, material_id=1):
    """
    Generate ANSYS APDL commands for material definition.
    """
    material_data = load_material_data(mix_type)
    
    ansys_commands = f"""
! Material {material_id}: {mix_type}
MP,DELETE,ALL,{material_id}
MP,DENS,{material_id},{material_data['density']:.1f}
"""
    
    # Temperature-dependent properties
    for i, (temp, E, nu, k, cp) in enumerate(zip(
        material_data['temperature'],
        material_data['elastic_modulus'],
        material_data['poissons_ratio'],
        material_data['thermal_conductivity'],
        material_data['specific_heat']
    )):
        ansys_commands += f"""
MPTEMP,{i+1},{temp}
MPDATA,EX,{material_id},{i+1},{E*1e6:.0f}
MPDATA,PRXY,{material_id},{i+1},{nu:.4f}
MPDATA,KXX,{material_id},{i+1},{k:.4f}
MPDATA,C,{material_id},{i+1},{cp:.1f}
"""
    
    return ansys_commands
```

### COMSOL Implementation
```python
def generate_comsol_functions(mix_type):
    """
    Generate COMSOL material property functions.
    """
    material_data = load_material_data(mix_type)
    
    comsol_functions = {}
    
    # Elastic modulus function
    comsol_functions['elastic_modulus'] = create_piecewise_function(
        material_data['temperature'], 
        material_data['elastic_modulus'] * 1e6,  # Convert to Pa
        f"E_{mix_type}"
    )
    
    # Thermal conductivity function
    comsol_functions['thermal_conductivity'] = create_piecewise_function(
        material_data['temperature'],
        material_data['thermal_conductivity'],
        f"k_{mix_type}"
    )
    
    return comsol_functions

def create_piecewise_function(x_data, y_data, function_name):
    """
    Create COMSOL piecewise function definition.
    """
    function_def = f"{function_name}(T) = piecewise(\n"
    
    for i in range(len(x_data) - 1):
        x1, x2 = x_data[i], x_data[i+1]
        y1, y2 = y_data[i], y_data[i+1]
        slope = (y2 - y1) / (x2 - x1)
        
        function_def += f"  (T >= {x1}[degC]) && (T < {x2}[degC]), {y1:.2e} + {slope:.2e}*(T-{x1}[degC]),\n"
    
    function_def += f"  T >= {x_data[-1]}[degC], {y_data[-1]:.2e})"
    
    return function_def
```

## Best Practices and Common Pitfalls

### Best Practices

#### Data Handling
1. **Always check data bounds** before using properties
2. **Use calibration data for fitting**, validation data for verification
3. **Propagate uncertainties** through your analysis
4. **Document assumptions** and limitations clearly

#### Model Development
1. **Start simple** - implement basic temperature dependence first
2. **Validate incrementally** - test each physics module separately
3. **Use consistent units** throughout your analysis
4. **Implement convergence checks** for coupled analyses

#### Results Interpretation
1. **Compare with experimental data** when available
2. **Check physical reasonableness** of results
3. **Perform sensitivity analysis** on key parameters
4. **Document validation metrics** achieved

### Common Pitfalls

#### Pitfall 1: Extrapolation Beyond Data Range
**Problem**: Using properties outside 20-800°C range
**Solution**: Implement bounds checking and warnings
```python
def check_temperature_bounds(temperature, property_name):
    if temperature < 20 or temperature > 800:
        raise ValueError(f"Temperature {temperature}°C outside valid range (20-800°C) for {property_name}")
```

#### Pitfall 2: Ignoring Statistical Uncertainty
**Problem**: Using only mean values without considering variability
**Solution**: Implement uncertainty quantification
```python
def sample_property_with_uncertainty(mean_value, std_value, n_samples=1000):
    """Sample property considering uncertainty."""
    return np.random.normal(mean_value, std_value, n_samples)
```

#### Pitfall 3: Inconsistent Multi-Physics Coupling
**Problem**: Using properties from different datasets without checking consistency
**Solution**: Use pre-validated property combinations
```python
def load_consistent_property_set(mix_type, temperature):
    """Load consistent set of properties for given conditions."""
    properties = {}
    
    # Load all properties for same mix and temperature
    for prop_type in ['thermal', 'mechanical', 'transport', 'deformation']:
        properties[prop_type] = load_properties(mix_type, temperature, prop_type)
    
    # Verify consistency
    verify_property_consistency(properties)
    
    return properties
```

#### Pitfall 4: Inadequate Mesh Resolution
**Problem**: Using coarse meshes that don't capture steep gradients
**Solution**: Implement adaptive mesh refinement
```python
def check_mesh_adequacy(temperature_gradient, element_size):
    """Check if mesh is adequate for temperature gradients."""
    critical_gradient = 50  # °C/mm
    max_element_size = 2    # mm
    
    if temperature_gradient > critical_gradient and element_size > max_element_size:
        print("Warning: Mesh may be too coarse for steep temperature gradients")
```

## Troubleshooting Guide

### Issue: Convergence Problems
**Symptoms**: Analysis fails to converge, excessive iterations
**Diagnosis Steps**:
1. Check material property continuity
2. Verify boundary condition implementation
3. Examine time step size
4. Review mesh quality

**Solutions**:
- Smooth property transitions using interpolation
- Implement proper constraint handling
- Use adaptive time stepping
- Refine mesh in critical regions

### Issue: Unphysical Results
**Symptoms**: Negative temperatures, excessive stresses, damage > 1.0
**Diagnosis Steps**:
1. Verify property bounds
2. Check unit consistency
3. Review coupling implementation
4. Examine boundary conditions

**Solutions**:
- Implement physical bounds checking
- Use consistent unit system throughout
- Validate coupling terms separately
- Apply realistic boundary conditions

### Issue: Poor Experimental Agreement
**Symptoms**: Simulation results don't match experimental data
**Diagnosis Steps**:
1. Compare individual property predictions
2. Check experimental conditions reproduction
3. Verify calibration data usage
4. Examine model assumptions

**Solutions**:
- Recalibrate critical parameters
- Include missing physics (e.g., moisture effects)
- Use more representative experimental data
- Refine model assumptions

## Support and Community

### Getting Help
- Review validation procedures for similar applications
- Check common pitfalls section
- Consult data dictionary for property definitions
- Examine provided example implementations

### Contributing Improvements
- Report issues with dataset or documentation
- Share validation results for new applications
- Contribute additional experimental data
- Suggest model enhancements

### Version Control
- Track dataset version used in your analysis
- Document any modifications made to properties
- Maintain reproducible analysis workflows
- Archive calibrated parameter sets

This comprehensive usage guide ensures effective and reliable application of the thermo-mechanical dataset for diverse fire-resistant structural modeling applications.