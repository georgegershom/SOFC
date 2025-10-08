# YSZ Material Properties Dataset - Usage Guide

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from material_properties_loader import YSZMaterialProperties

# Load the dataset
ysz = YSZMaterialProperties()

# Get a property at specific temperature
E_800 = ysz.get_property('Youngs_Modulus_GPa', 800)
print(f"Young's Modulus at 800°C: {E_800:.2f} GPa")

# Get all properties at once
props_600 = ysz.get_all_properties(600)
print(props_600)
```

### 3. Run Validation
```bash
python3 validate_dataset.py
```

## Available Scripts

### `material_properties_loader.py`
Main data loading and interpolation module.

**Run example:**
```bash
python3 material_properties_loader.py
```

**What it does:**
- Loads all CSV datasets
- Creates interpolation functions
- Prints property summaries at 25°C and 800°C
- Generates visualization plots
- Exports FEM-ready data

### `validate_dataset.py`
Dataset integrity checker.

**Run:**
```bash
python3 validate_dataset.py
```

**Checks performed:**
- Temperature range adequacy
- Physical monotonicity (E↓, CTE↑, ρ↓)
- Realistic value ranges
- Poisson's ratio bounds
- Weibull parameter validity
- Data completeness
- Temperature spacing uniformity

### `generate_custom_dataset.py`
Generate datasets at custom temperature points.

**Examples:**
```bash
# SOFC operating range (600-900°C)
python3 generate_custom_dataset.py --tmin 600 --tmax 900 --npoints 25 -o sofc_operating.csv

# Full sintering cycle (RT to 1500°C)
python3 generate_custom_dataset.py --tmin 25 --tmax 1500 --npoints 100 -o sintering_full.csv

# High-resolution dataset for gradient analysis
python3 generate_custom_dataset.py --tmin 20 --tmax 1200 --npoints 200 -o high_resolution.csv
```

## Working with the Data

### Available Properties

| Property Name (use in code) | Units | Description |
|------------------------------|-------|-------------|
| `Youngs_Modulus_GPa` | GPa | Elastic modulus |
| `Poissons_Ratio` | - | Lateral/axial strain ratio |
| `CTE_1e-6_K` | 10⁻⁶ K⁻¹ | Coefficient of thermal expansion |
| `Density_kg_m3` | kg/m³ | Material density |
| `Thermal_Conductivity_W_mK` | W/m·K | Thermal conductivity |
| `Fracture_Toughness_MPa_m0.5` | MPa·m⁰·⁵ | Fracture resistance |
| `Creep_Exponent_n` | - | Power-law creep exponent |
| `Creep_Activation_Energy_kJ_mol` | kJ/mol | Creep activation energy |
| `Weibull_Modulus_m` | - | Statistical shape parameter |
| `Characteristic_Strength_MPa` | MPa | Weibull characteristic strength |

### Interpolation Example

```python
import numpy as np
from material_properties_loader import YSZMaterialProperties

ysz = YSZMaterialProperties()

# Generate smooth temperature curve
temp_range = np.linspace(25, 1500, 500)
E_curve = ysz.get_property('Youngs_Modulus_GPa', temp_range)
CTE_curve = ysz.get_property('CTE_1e-6_K', temp_range)

# Use in thermal stress calculation
# σ_thermal = E * α * ΔT (simplified)
delta_T = 100  # K
thermal_stress = E_curve * CTE_curve * 1e-6 * delta_T * 1000  # MPa
```

### Calculate Creep Rate

```python
ysz = YSZMaterialProperties()

# Power-law creep: ε̇ = A·σⁿ·d⁻ᵐ·exp(-Q/RT)
creep_rate = ysz.get_creep_rate(
    stress_mpa=50,           # Applied stress
    temperature_c=1200,      # Temperature
    grain_size_um=2.0        # Grain size
)

print(f"Creep strain rate: {creep_rate:.3e} s⁻¹")
```

### Weibull Probability Calculation

```python
import numpy as np
from material_properties_loader import YSZMaterialProperties

ysz = YSZMaterialProperties()

def weibull_failure_probability(stress, temperature):
    """Calculate failure probability at given stress and temperature."""
    m = ysz.get_property('Weibull_Modulus_m', temperature)
    sigma_0 = ysz.get_property('Characteristic_Strength_MPa', temperature)
    
    # Weibull CDF: P_f = 1 - exp(-(σ/σ₀)^m)
    P_f = 1 - np.exp(-((stress / sigma_0) ** m))
    return P_f

# Example: Probability of failure at 300 MPa and 25°C
P_fail = weibull_failure_probability(300, 25)
print(f"Failure probability: {P_fail*100:.1f}%")
```

## FEM Software Integration

### ANSYS Mechanical APDL

```apdl
! Define material properties with temperature dependence
MP, EX, 1, 205e9        ! Young's modulus at RT (Pa)
MPTEMP, 1, 25, 100, 200, 300, 400, 500, 600, 700, 800
MPDATA, EX, 1, 1, 205e9, 198.5e9, 190e9, 180.5e9, 170e9, 158.5e9, 146e9, 132.5e9, 118e9

! CTE definition
MPTEMP, 1, 25, 100, 200, 300, 400, 500, 600, 700, 800
MPDATA, ALPX, 1, 1, 10.2e-6, 10.4e-6, 10.6e-6, 10.8e-6, 11.0e-6, 11.2e-6, 11.4e-6, 11.6e-6, 11.8e-6

! Creep definition (power-law)
TB, CREEP, 1, 1, 4      ! Activate creep table
TBTEMP, 1000            ! Temperature (C)
TBDATA, 1, 2.5e-15, 2.5, 380e3, 0    ! C1, C2, C3, C4
```

### COMSOL Multiphysics

1. **Import CSV directly:**
   - Right-click **Material** node → **Functions** → **Interpolation**
   - Load `ysz_fem_input.csv`
   - Set interpolation method to "Cubic spline"

2. **Define temperature-dependent properties:**
   ```
   E = interp1(T, 'ysz_fem_input.csv', 'Youngs_Modulus_GPa') * 1e9
   alpha = interp1(T, 'ysz_fem_input.csv', 'CTE_1e-6_K') * 1e-6
   ```

3. **Enable physics:**
   - Add **Solid Mechanics** module
   - Add **Heat Transfer** module
   - Enable **Multiphysics** → **Thermal Expansion**
   - Add **Creep** subnodes if needed

### Abaqus

Create material input file (`ysz_material.inp`):

```
*MATERIAL, NAME=YSZ_8mol
**
** Temperature-dependent elastic properties
*ELASTIC, TYPE=ISOTROPIC
205.0E9, 0.31, 25.0
198.5E9, 0.31, 100.0
190.0E9, 0.31, 200.0
180.5E9, 0.32, 300.0
...
**
** Thermal expansion
*EXPANSION, TYPE=ISOTROPIC
10.2E-6, 25.0
10.4E-6, 100.0
10.6E-6, 200.0
...
**
** Density
*DENSITY
6050.0
**
** Creep (power-law)
*CREEP, LAW=POWER
2.5E-15, 2.5, 380E3
```

### Python/FEniCS

```python
from dolfin import *
import pandas as pd

# Load material data
props = pd.read_csv('ysz_fem_input.csv')

# Create interpolation function for temperature-dependent E
class TemperatureDependentE(UserExpression):
    def __init__(self, props_df, **kwargs):
        super().__init__(**kwargs)
        self.props = props_df
    
    def eval(self, values, x):
        # Get temperature at point x (from temperature field)
        T = temperature(x)  # Your temperature solution
        
        # Interpolate Young's modulus
        values[0] = np.interp(T, self.props['Temperature_C'], 
                             self.props['Youngs_Modulus_GPa']) * 1e9

# Define material
E_expr = TemperatureDependentE(props, degree=2)
nu = Constant(0.32)
mu = E_expr / (2*(1 + nu))
lambda_ = E_expr*nu / ((1+nu)*(1-2*nu))
```

## Best Practices

### 1. Temperature Range Selection
- **SOFC Operating:** 600-900°C (most stable operation)
- **Sintering:** 25-1500°C (full thermal cycle)
- **Thermal Shock:** 25-800°C (rapid heating scenarios)

### 2. Number of Data Points
- **Coarse mesh (fast):** 10-20 points
- **Standard mesh:** 30-50 points
- **Fine mesh (accurate gradients):** 100+ points
- **Adaptive:** Use more points where properties change rapidly (e.g., 700-1200°C)

### 3. Validation Steps
Always validate before production use:
1. Run `validate_dataset.py` to check integrity
2. Compare key values against literature (see README)
3. Perform mesh convergence study in FEM
4. Verify against experimental stress-strain curves if available
5. Check sensitivity to property uncertainties (±10% variation)

### 4. Common Pitfalls

❌ **Don't:**
- Use this data for flight-critical or safety-critical applications
- Extrapolate far beyond 1500°C (melting point ~2700°C, but properties diverge)
- Ignore grain size effects on creep and strength
- Assume properties are independent of atmosphere (O₂ partial pressure matters)

✅ **Do:**
- Cite this as "fabricated data for demonstration"
- Perform sensitivity analysis on uncertain parameters
- Validate with at least one experimental data point
- Consider porosity effects (reduce E and k proportionally)
- Account for thermal gradients in stress calculations

## Advanced Usage

### Custom Property Functions

```python
from material_properties_loader import YSZMaterialProperties
import numpy as np

class CustomYSZ(YSZMaterialProperties):
    """Extended YSZ class with custom properties."""
    
    def get_stress_intensity_factor(self, stress, crack_length, temperature):
        """
        Calculate stress intensity factor.
        K_I = Y·σ·√(πa)
        """
        Y = 1.12  # Geometry factor for edge crack
        K_I = Y * stress * np.sqrt(np.pi * crack_length)
        return K_I
    
    def is_safe(self, stress, crack_length, temperature):
        """Check if crack will propagate."""
        K_I = self.get_stress_intensity_factor(stress, crack_length, temperature)
        K_IC = self.get_property('Fracture_Toughness_MPa_m0.5', temperature)
        
        safety_factor = K_IC / K_I
        return safety_factor > 1.0, safety_factor

# Usage
ysz = CustomYSZ()
is_safe, SF = ysz.is_safe(stress=100, crack_length=0.001, temperature=800)
print(f"Safe: {is_safe}, Safety Factor: {SF:.2f}")
```

### Batch Processing

```python
from material_properties_loader import YSZMaterialProperties
import numpy as np
import pandas as pd

ysz = YSZMaterialProperties()

# Generate data for multiple conditions
results = []
for temp in [25, 200, 400, 600, 800, 1000, 1200]:
    for stress in [50, 100, 150, 200]:
        creep = ysz.get_creep_rate(stress, temp, grain_size_um=1.5)
        results.append({
            'Temperature_C': temp,
            'Stress_MPa': stress,
            'Creep_Rate_s-1': creep
        })

results_df = pd.DataFrame(results)
results_df.to_csv('creep_matrix.csv', index=False)
print(results_df.pivot(index='Temperature_C', columns='Stress_MPa', values='Creep_Rate_s-1'))
```

## Troubleshooting

### Issue: "Property not found" error
**Solution:** Check exact property name spelling. Use `print(ysz.interpolators.keys())` to see available properties.

### Issue: Extrapolation warnings
**Solution:** Stay within 25-1500°C range, or generate custom dataset with extended range (use with caution).

### Issue: Unrealistic creep rates
**Solution:** Verify stress units (MPa), temperature (Celsius), and grain size (micrometers). Creep rates should be O(1e-8 to 1e-2) s⁻¹ at SOFC conditions.

### Issue: FEM convergence problems
**Solution:** 
- Use finer temperature mesh near steep property gradients
- Enable automatic remeshing for large deformations
- Check for unrealistic thermal gradients (>100 K/mm)
- Reduce time step size in transient analysis

## Support & Further Information

- **Dataset Documentation:** See `README.md`
- **Validation Results:** Run `python3 validate_dataset.py`
- **Example Notebook:** See `examples/` directory (if available)
- **Literature References:** Listed in `README.md`

## License

MIT License - Educational use only. See LICENSE file for details.

---

**Version:** 1.0  
**Last Updated:** October 2025