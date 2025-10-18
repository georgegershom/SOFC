# Validation Procedures and Guidelines

## Overview

This document provides comprehensive validation procedures for using the thermo-mechanical modeling dataset for fire-resistant structural elements. It includes model validation strategies, quality assurance checks, and best practices for ensuring reliable simulation results.

## Dataset Validation Hierarchy

### Level 1: Data Consistency Validation
**Purpose:** Verify internal consistency and physical plausibility of the dataset.

#### Physical Bounds Checking
```python
# Example validation checks
def validate_physical_bounds(df):
    checks = {
        'elastic_modulus': (1000, 50000),  # MPa
        'poissons_ratio': (0.05, 0.45),   # dimensionless
        'thermal_conductivity': (0.1, 5.0), # W/m·K
        'porosity': (0.05, 0.5),           # dimensionless
        'damage_level': (0.0, 1.0)         # dimensionless
    }
    
    for property_name, (min_val, max_val) in checks.items():
        if property_name in df.columns:
            violations = df[(df[property_name] < min_val) | 
                           (df[property_name] > max_val)]
            if len(violations) > 0:
                print(f"Warning: {len(violations)} violations in {property_name}")
```

#### Temperature Monotonicity Checks
- Thermal conductivity: Should decrease with temperature
- Specific heat: Should increase with temperature  
- Elastic modulus: Should decrease with temperature
- Damage level: Should increase with temperature (above threshold)

#### Multi-Physics Consistency
- Thermal diffusivity = k/(ρ·cp) within ±5%
- Higher porosity → higher permeability
- Higher damage → lower residual strength/stiffness

### Level 2: Statistical Validation
**Purpose:** Verify statistical properties and uncertainty bounds.

#### Distribution Analysis
```python
def validate_distributions(df):
    # Check for normal distribution of residuals
    from scipy import stats
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if '_Mean' in col:
            std_col = col.replace('_Mean', '_Std')
            if std_col in df.columns:
                # Check coefficient of variation
                cov = df[std_col] / df[col]
                if cov.mean() > 0.5:
                    print(f"High variability in {col}: CoV = {cov.mean():.3f}")
```

#### Calibration/Validation Split Verification
- Verify 70/30 split ratio (±5%)
- Check temperature range coverage in both sets
- Ensure all mix types represented in both sets

### Level 3: Model Validation
**Purpose:** Validate FEA model predictions against experimental data.

#### Benchmark Validation Cases

##### Case 1: ISO 834 Fire Curve Validation
**Objective:** Validate thermal response under standard fire conditions.

**Setup:**
- Concrete slab: 150mm thick
- Boundary conditions: ISO 834 fire curve on one side
- Validation metrics: Temperature profiles at 25mm, 75mm, 125mm depth

**Acceptance Criteria:**
- Temperature predictions within ±50°C at all measurement points
- Time to reach 300°C within ±10 minutes
- Maximum temperature gradient within ±20%

```python
def validate_iso834_response(simulation_results, experimental_data):
    """
    Validate thermal response against ISO 834 fire test data.
    """
    temperature_tolerance = 50  # °C
    time_tolerance = 600       # seconds (10 minutes)
    
    for depth in [25, 75, 125]:  # mm
        sim_temp = simulation_results[f'temp_at_{depth}mm']
        exp_temp = experimental_data[f'temp_at_{depth}mm']
        
        # Temperature accuracy
        temp_error = abs(sim_temp - exp_temp)
        if temp_error.max() > temperature_tolerance:
            print(f"Temperature validation failed at {depth}mm depth")
        
        # Time to 300°C
        sim_time_300 = find_time_to_temperature(sim_temp, 300)
        exp_time_300 = find_time_to_temperature(exp_temp, 300)
        
        if abs(sim_time_300 - exp_time_300) > time_tolerance:
            print(f"Time to 300°C validation failed at {depth}mm depth")
```

##### Case 2: Mechanical Loading Validation
**Objective:** Validate mechanical response under elevated temperatures.

**Setup:**
- Compression test at various temperatures (20°C, 200°C, 400°C, 600°C)
- Loading rate: 0.5 MPa/s
- Validation metrics: Peak strength, elastic modulus, failure strain

**Acceptance Criteria:**
- Strength predictions within ±15%
- Modulus predictions within ±20%
- Failure strain within ±25%

##### Case 3: Multi-Physics Coupling Validation
**Objective:** Validate coupled thermo-mechanical response.

**Setup:**
- Restrained thermal expansion test
- Temperature ramp: 20°C to 600°C at 5°C/min
- Validation metrics: Thermal stress development, cracking temperature

**Acceptance Criteria:**
- Thermal stress within ±20%
- Cracking temperature within ±30°C

### Level 4: Sensitivity Analysis
**Purpose:** Assess model sensitivity to parameter variations.

#### Parameter Sensitivity Study
```python
def parameter_sensitivity_analysis(base_parameters, variations):
    """
    Perform sensitivity analysis on key parameters.
    """
    results = {}
    
    for param_name, variation_range in variations.items():
        param_results = []
        
        for variation in variation_range:
            modified_params = base_parameters.copy()
            modified_params[param_name] *= (1 + variation)
            
            # Run simulation with modified parameters
            result = run_simulation(modified_params)
            param_results.append(result)
        
        # Calculate sensitivity coefficient
        sensitivity = calculate_sensitivity_coefficient(param_results, variation_range)
        results[param_name] = sensitivity
    
    return results

# Example sensitivity study
variations = {
    'thermal_conductivity': [-0.2, -0.1, 0.0, 0.1, 0.2],
    'elastic_modulus': [-0.2, -0.1, 0.0, 0.1, 0.2],
    'creep_coefficient': [-0.3, -0.15, 0.0, 0.15, 0.3]
}
```

#### Critical Parameter Identification
- Parameters with sensitivity coefficient > 0.5 are critical
- Focus calibration efforts on critical parameters
- Use higher-quality data for critical parameters

## Model Calibration Procedures

### Step 1: Parameter Initialization
```python
def initialize_parameters(mix_type, temperature_range):
    """
    Initialize parameters from calibration dataset.
    """
    cal_data = load_calibration_data(mix_type)
    
    # Filter by temperature range
    temp_data = cal_data[
        (cal_data['Temperature_C'] >= temperature_range[0]) &
        (cal_data['Temperature_C'] <= temperature_range[1])
    ]
    
    # Extract mean values for initialization
    initial_params = {}
    for col in temp_data.columns:
        if '_Mean' in col:
            param_name = col.replace('_Mean', '').replace('_MPa', '').replace('_W_m_K', '')
            initial_params[param_name] = temp_data[col].values
    
    return initial_params
```

### Step 2: Objective Function Definition
```python
def calibration_objective_function(params, experimental_data, weights=None):
    """
    Objective function for parameter calibration.
    """
    # Run simulation with current parameters
    simulation_results = run_simulation(params)
    
    # Calculate weighted residuals
    residuals = []
    
    for exp_point in experimental_data:
        sim_value = interpolate_simulation_result(
            simulation_results, 
            exp_point['temperature'], 
            exp_point['time']
        )
        
        residual = (sim_value - exp_point['value']) / exp_point['uncertainty']
        residuals.append(residual)
    
    # Apply weights if provided
    if weights is not None:
        residuals = np.array(residuals) * np.array(weights)
    
    return np.sum(np.array(residuals)**2)
```

### Step 3: Optimization Strategy
```python
from scipy.optimize import minimize

def calibrate_parameters(initial_params, experimental_data, bounds=None):
    """
    Calibrate parameters using optimization.
    """
    # Define parameter bounds based on physical constraints
    if bounds is None:
        bounds = generate_physical_bounds(initial_params)
    
    # Optimization
    result = minimize(
        calibration_objective_function,
        initial_params,
        args=(experimental_data,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    return result.x, result.fun
```

### Step 4: Uncertainty Quantification
```python
def quantify_parameter_uncertainty(calibrated_params, experimental_data):
    """
    Quantify uncertainty in calibrated parameters.
    """
    # Calculate Hessian matrix
    hessian = calculate_hessian(calibrated_params, experimental_data)
    
    # Parameter covariance matrix
    param_covariance = np.linalg.inv(hessian)
    
    # Parameter standard deviations
    param_std = np.sqrt(np.diag(param_covariance))
    
    # Correlation matrix
    correlation_matrix = param_covariance / np.outer(param_std, param_std)
    
    return param_std, correlation_matrix
```

## Validation Metrics and Acceptance Criteria

### Thermal Properties Validation

| Property | Metric | Acceptance Criteria |
|----------|--------|-------------------|
| Thermal Conductivity | RMSE | < 0.2 W/m·K |
| Specific Heat | RMSE | < 50 J/kg·K |
| Thermal Diffusivity | Relative Error | < 15% |

### Mechanical Properties Validation

| Property | Metric | Acceptance Criteria |
|----------|--------|-------------------|
| Compressive Strength | Relative Error | < 15% |
| Elastic Modulus | Relative Error | < 20% |
| Poisson's Ratio | Absolute Error | < 0.05 |

### Transport Properties Validation

| Property | Metric | Acceptance Criteria |
|----------|--------|-------------------|
| Permeability | Log Error | < 0.5 |
| Porosity | Absolute Error | < 0.03 |
| Diffusivity | Relative Error | < 30% |

## Quality Assurance Checklist

### Pre-Simulation Checks
- [ ] Material properties within physical bounds
- [ ] Temperature dependencies monotonic where expected
- [ ] Calibration/validation data split verified
- [ ] Statistical distributions reasonable (CoV < 0.5)
- [ ] Multi-physics consistency checked

### During Simulation
- [ ] Convergence criteria met
- [ ] Mass/energy conservation verified
- [ ] Time step size appropriate
- [ ] Mesh independence confirmed
- [ ] Boundary conditions correctly applied

### Post-Simulation Validation
- [ ] Results within expected physical ranges
- [ ] Temperature profiles realistic
- [ ] Stress distributions reasonable
- [ ] Damage evolution consistent
- [ ] Validation metrics met

## Common Validation Issues and Solutions

### Issue 1: Non-Physical Temperature Profiles
**Symptoms:** Temperature overshoots, negative gradients
**Solutions:**
- Check thermal conductivity temperature dependence
- Verify specific heat values at high temperatures
- Ensure proper boundary condition implementation

### Issue 2: Excessive Thermal Stresses
**Symptoms:** Stresses exceeding material strength by large margins
**Solutions:**
- Verify thermal expansion coefficient values
- Check elastic modulus temperature dependence
- Consider creep/relaxation effects

### Issue 3: Unrealistic Damage Evolution
**Symptoms:** Damage levels > 1.0, negative damage rates
**Solutions:**
- Check damage threshold temperatures
- Verify damage evolution rate parameters
- Ensure proper damage-property coupling

## Reporting and Documentation

### Validation Report Structure
1. **Executive Summary**
   - Key findings and recommendations
   - Overall validation status

2. **Dataset Description**
   - Mix types and properties covered
   - Temperature and time ranges
   - Statistical characteristics

3. **Validation Results**
   - Level 1-4 validation outcomes
   - Acceptance criteria compliance
   - Identified limitations

4. **Calibration Results**
   - Optimized parameter values
   - Uncertainty quantification
   - Sensitivity analysis results

5. **Recommendations**
   - Suitable application ranges
   - Required experimental validation
   - Future dataset improvements

### Documentation Requirements
- All validation scripts and data
- Calibration optimization history
- Sensitivity analysis results
- Comparison with literature data
- Uncertainty propagation analysis

## Continuous Validation

### Dataset Updates
- Regular comparison with new experimental data
- Incorporation of improved measurement techniques
- Extension to new temperature ranges or mix types

### Model Improvements
- Integration of new physical mechanisms
- Enhanced multi-physics coupling
- Improved numerical methods

### Community Validation
- Peer review of validation procedures
- Round-robin validation exercises
- Benchmark problem development

This validation framework ensures reliable and robust use of the thermo-mechanical dataset for fire-resistant structural element modeling.