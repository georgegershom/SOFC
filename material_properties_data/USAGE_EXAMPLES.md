# Usage Examples for Material Properties Dataset

## Quick Start Guide

### Python (Pandas)

```python
import pandas as pd
import numpy as np

# Load datasets
mech = pd.read_csv("mechanical_properties.csv")
creep = pd.read_csv("creep_properties.csv")
thermo = pd.read_csv("thermophysical_properties.csv")
electro = pd.read_csv("electrochemical_properties.csv")

# Example 1: Get Young's modulus for 8YSZ at 800°C
E = mech[(mech['Material'] == '8YSZ') & 
         (mech['Temperature_C'] == 800)]['Youngs_Modulus_GPa'].values[0]
print(f"Young's Modulus at 800°C: {E:.2f} GPa")

# Example 2: Get ionic conductivity vs temperature for 8YSZ
ysz_data = electro[electro['Material'] == '8YSZ']
temps = ysz_data['Temperature_C'].values
conductivity = ysz_data['Ionic_Conductivity_Sperm'].values

# Example 3: Interpolate property at arbitrary temperature
from scipy.interpolate import interp1d

ysz_mech = mech[mech['Material'] == '8YSZ']
T = ysz_mech['Temperature_C'].values
E = ysz_mech['Youngs_Modulus_GPa'].values
f = interp1d(T, E, kind='linear')

E_650 = f(650)  # Young's modulus at 650°C
print(f"Young's Modulus at 650°C: {E_650:.2f} GPa")

# Example 4: Get creep strain at specific conditions
creep_data = creep[(creep['Material'] == 'Ni-YSZ') &
                   (creep['Temperature_C'] == 800) &
                   (creep['Stress_MPa'] == 100)]
time = creep_data['Time_hours'].values
strain = creep_data['Creep_Strain_percent'].values
```

### MATLAB

```matlab
% Load datasets
mech = readtable('mechanical_properties.csv');
creep = readtable('creep_properties.csv');
thermo = readtable('thermophysical_properties.csv');
electro = readtable('electrochemical_properties.csv');

% Example 1: Get CTE for all materials at 800°C
idx = thermo.Temperature_C == 800;
materials = thermo.Material(idx);
cte = thermo.CTE_1perK(idx);

% Example 2: Plot ionic conductivity vs temperature
idx = strcmp(electro.Material, '8YSZ');
plot(electro.Temperature_C(idx), electro.Ionic_Conductivity_Sperm(idx));
xlabel('Temperature (°C)');
ylabel('Ionic Conductivity (S/m)');
title('8YSZ Ionic Conductivity');

% Example 3: Interpolate property
ysz_idx = strcmp(mech.Material, '8YSZ');
E_650 = interp1(mech.Temperature_C(ysz_idx), ...
                mech.Youngs_Modulus_GPa(ysz_idx), 650);
```

### COMSOL Multiphysics

```plaintext
1. Import CSV files as Interpolation Functions:
   - Right-click Global Definitions > Functions > Interpolation
   - Select the CSV file
   - Set arguments (Temperature_C) and function column

2. Define material properties:
   - Young's Modulus: E(T) [Pa] = YSZ_E_interp(T[K]-273.15) * 1e9
   - Thermal Conductivity: k(T) [W/(m·K)] = YSZ_k_interp(T[K]-273.15)
   - Ionic Conductivity: sigma_ion(T) [S/m] = YSZ_sigma_interp(T[K]-273.15)

3. For creep:
   - Import creep_properties.csv
   - Use Norton-Bailey creep model
   - Extract parameters by fitting to data
```

### ANSYS

```plaintext
1. Create material properties:
   /PREP7
   MP,EX,1,func_E(TEMP)    ! Young's Modulus
   MP,NUXY,1,func_nu(TEMP)  ! Poisson's Ratio
   MP,ALPX,1,func_CTE(TEMP) ! CTE
   MP,KXX,1,func_k(TEMP)    ! Thermal Conductivity
   MP,C,1,func_cp(TEMP)     ! Specific Heat

2. Define temperature-dependent properties:
   MPTEMP,1,20,100,200,300,400,500,600,700,800,900,1000
   MPDATA,EX,1,1,200e9,196e9,192e9,...

3. Define creep:
   TB,CREEP,1,1,5,100  ! Norton-Bailey law
   TBDATA,1,A,n,m,Q/R  ! Parameters from creep_properties.csv
```

### Abaqus

```python
# Material definition in Abaqus input file
*Material, name=8YSZ
*Elastic, type=ISOTROPIC
<temperature>, <E>, <nu>
20., 200e9, 0.31
100., 196e9, 0.312
...

*Expansion, type=ISO
<temperature>, <CTE>
20., 10.5e-6
100., 10.6e-6
...

*Conductivity
<temperature>, <k>
20., 2.7
100., 2.68
...

*Creep, law=TIME
<A>, <n>, <m>
```

## Use Cases

### 1. Thermal Stress Analysis

```python
import pandas as pd

# Load data
mech = pd.read_csv("mechanical_properties.csv")
thermo = pd.read_csv("thermophysical_properties.csv")

# Operating conditions
T_cold = 600  # °C
T_hot = 800   # °C

# Get CTE values
material = '8YSZ'
cte_cold = thermo[(thermo['Material'] == material) & 
                  (thermo['Temperature_C'] == T_cold)]['CTE_1perK'].values[0]
cte_hot = thermo[(thermo['Material'] == material) & 
                 (thermo['Temperature_C'] == T_hot)]['CTE_1perK'].values[0]
cte_avg = (cte_cold + cte_hot) / 2

# Thermal strain
delta_T = T_hot - T_cold
epsilon_thermal = cte_avg * delta_T

# Thermal stress (constrained)
E_avg = mech[(mech['Material'] == material) & 
             (mech['Temperature_C'] == (T_cold + T_hot)/2)]['Youngs_Modulus_GPa'].values[0]
sigma_thermal = E_avg * 1e9 * epsilon_thermal / 1e6  # MPa

print(f"Thermal Stress: {sigma_thermal:.2f} MPa")
```

### 2. Creep Life Prediction

```python
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load creep data
creep = pd.read_csv("creep_properties.csv")

# Get data for specific conditions
data = creep[(creep['Material'] == '8YSZ') &
             (creep['Temperature_C'] == 800) &
             (creep['Stress_MPa'] == 100)]

time = data['Time_hours'].values
strain = data['Creep_Strain_percent'].values

# Fit Norton-Bailey equation: ε = A * t^m
def power_law(t, A, m):
    return A * np.power(t, m)

# Only fit non-zero time points
mask = time > 0
params, _ = curve_fit(power_law, time[mask], strain[mask])
A, m = params

# Predict failure time (assuming failure at 0.2% strain)
epsilon_f = 0.2
t_failure = np.power(epsilon_f / A, 1/m)

print(f"Predicted failure time: {t_failure:.2f} hours")
```

### 3. Electrochemical Performance

```python
import pandas as pd
import numpy as np

# Load data
electro = pd.read_csv("electrochemical_properties.csv")

# Operating temperature
T = 800  # °C

# Electrolyte properties
ysz = electro[(electro['Material'] == '8YSZ') & 
              (electro['Temperature_C'] == T)]
sigma_ion = ysz['Ionic_Conductivity_Sperm'].values[0]

# Ohmic resistance
thickness = 10e-6  # 10 μm electrolyte
area = 1e-4  # 1 cm²
R_ohm = thickness / (sigma_ion * area)

# Voltage loss at 1 A/cm²
i = 1.0  # A/cm²
V_ohm = i * R_ohm

print(f"Ohmic resistance: {R_ohm:.4f} Ω")
print(f"Ohmic loss at 1 A/cm²: {V_ohm:.4f} V")
```

### 4. Multi-Physics Coupling

```python
"""
Simplified multi-physics coupling workflow
"""

import pandas as pd
import numpy as np

# Load all datasets
mech = pd.read_csv("mechanical_properties.csv")
thermo = pd.read_csv("thermophysical_properties.csv")
electro = pd.read_csv("electrochemical_properties.csv")

def solve_coupled_physics(T_initial, current_density):
    """
    Simplified coupled solver
    """
    T = T_initial  # °C
    
    # Step 1: Electrochemical model
    sigma = electro[(electro['Material'] == '8YSZ') & 
                    (electro['Temperature_C'] == T)]['Ionic_Conductivity_Sperm'].values[0]
    
    # Joule heating (W/m³)
    Q_joule = current_density**2 / sigma
    
    # Step 2: Thermal model
    k = thermo[(thermo['Material'] == '8YSZ') & 
               (thermo['Temperature_C'] == T)]['Thermal_Conductivity_WperMK'].values[0]
    
    # Temperature rise (simplified)
    thickness = 10e-6
    delta_T = Q_joule * thickness**2 / (2 * k)
    
    T_new = T + delta_T
    
    # Step 3: Mechanical model
    cte = thermo[(thermo['Material'] == '8YSZ') & 
                 (thermo['Temperature_C'] == T_new)]['CTE_1perK'].values[0]
    E = mech[(mech['Material'] == '8YSZ') & 
             (mech['Temperature_C'] == T_new)]['Youngs_Modulus_GPa'].values[0]
    
    # Thermal stress
    sigma_thermal = E * 1e9 * cte * delta_T / 1e6  # MPa
    
    return {
        'Temperature': T_new,
        'Joule_Heating': Q_joule,
        'Thermal_Stress': sigma_thermal,
        'Conductivity': sigma
    }

# Example
result = solve_coupled_physics(T_initial=800, current_density=1e4)  # A/m²
print(f"Temperature: {result['Temperature']:.2f} °C")
print(f"Thermal Stress: {result['Thermal_Stress']:.2f} MPa")
```

## Data Interpolation Best Practices

### Linear Interpolation (Recommended)

```python
from scipy.interpolate import interp1d

def get_property(material, property_name, temperature, dataset):
    """
    Get interpolated property value
    """
    data = dataset[dataset['Material'] == material]
    T = data['Temperature_C'].values
    prop = data[property_name].values
    
    f = interp1d(T, prop, kind='linear', fill_value='extrapolate')
    return f(temperature)

# Usage
E = get_property('8YSZ', 'Youngs_Modulus_GPa', 750, mech)
```

### Cubic Spline (For Smooth Derivatives)

```python
from scipy.interpolate import CubicSpline

data = mech[mech['Material'] == '8YSZ']
T = data['Temperature_C'].values
E = data['Youngs_Modulus_GPa'].values

cs = CubicSpline(T, E)
E_interp = cs(750)  # Value at 750°C
dE_dT = cs(750, 1)  # Derivative at 750°C
```

## Validation & Uncertainty

All data includes realistic experimental noise (2-5%). For production use:

1. **Validate with experiments**: Compare predictions with measured data
2. **Sensitivity analysis**: Vary properties within uncertainty bounds
3. **Compare with literature**: Check against published values
4. **Calibrate if needed**: Adjust parameters based on your specific materials

## Contact & Support

For questions about:
- Data generation methodology
- Specific material properties
- Multi-physics modeling approaches
- Custom dataset generation

Please refer to the main README.md file.
