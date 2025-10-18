# Quick Start Guide

## Get Started in 5 Minutes

### Installation

```bash
# Navigate to the scripts directory
cd phase4_numerical_modeling_dataset/scripts

# Install dependencies
pip install -r requirements.txt
```

### Load and Explore Data

```python
from data_loader import RubberizedConcreteDataLoader

# Initialize and load all data
loader = RubberizedConcreteDataLoader()
data = loader.load_all_data()

# Explore what's available
print(f"Thermal datasets: {list(data['thermal'].keys())}")
print(f"Mechanical datasets: {list(data['mechanical'].keys())}")
print(f"Validation datasets: {list(data['validation'].keys())}")
```

### Example 1: Plot Compressive Strength Degradation

```python
import matplotlib.pyplot as plt

# Get compressive strength data
strength = loader.mechanical_data['compressive_strength']

# Plot for all rubber contents
for rc in [0, 10, 20, 30]:
    data = strength[strength['Rubber_Content_Percent'] == rc]
    plt.plot(data['Temperature_C'], data['Compressive_Strength_MPa'], 
             'o-', label=f'{rc}% Rubber', linewidth=2)

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Compressive Strength (MPa)', fontsize=12)
plt.title('Strength Degradation with Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: Get Material Properties at Specific Temperature

```python
# Get properties at 500°C for 20% rubber concrete
props = loader.get_material_properties_at_temperature(
    rubber_content=20,
    temperature=500
)

print(f"Thermal Conductivity at 500°C: {props['thermal_conductivity']:.3f} W/mK")
```

### Example 3: Load Validation Data for Model Comparison

```python
# Get validation data for ISO 834 fire with 10% rubber
validation = loader.get_validation_data_for_model(
    fire_curve='ISO834',
    rubber_content=10
)

# Extract core temperature evolution
core_temps = validation['temperature'][['Time_min', 'TC6_Center_C']]
print("\nCore Temperature Evolution:")
print(core_temps.head(10))

# Extract strain evolution
strains = validation['strain'][['Time_min', 'Temperature_C', 'Axial_Strain_microstrain']]
print("\nStrain Evolution:")
print(strains.head(10))
```

### Example 4: Visualize All Data

```python
from data_loader import RubberizedConcreteDataLoader
from visualize_data import DataVisualizer

# Load data
loader = RubberizedConcreteDataLoader()
loader.load_all_data()

# Generate all plots
visualizer = DataVisualizer(loader)
visualizer.generate_all_plots()

# Plots will be saved in phase4_numerical_modeling_dataset/visualizations/
print("\nVisualizations saved!")
```

### Example 5: Fit Models and Export for FEA

```python
from model_calibration_helper import ModelCalibrationHelper

# Create helper
helper = ModelCalibrationHelper(loader)

# Fit thermal conductivity model for 0% rubber
result = helper.fit_thermal_conductivity_model(rubber_content=0)
print(f"\nModel Parameters: {result['parameters']}")
print(f"R-squared: {result['r_squared']:.4f}")
print(f"RMSE: {result['rmse']:.4f} W/mK")

# Create ABAQUS material input for 20% rubber
abaqus_input = helper.create_material_input_file(
    rubber_content=20,
    output_format='ABAQUS'
)

# Save to file
with open('material_RC20.inp', 'w') as f:
    f.write(abaqus_input)

print("\nABAQUS material file created: material_RC20.inp")
```

### Example 6: Analyze Spalling Behavior

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load spalling data
spalling = loader.validation_data['spalling_iso834']

# Filter for 10 MPa load
data = spalling[spalling['Applied_Load_MPa'] == 10.0]

# Plot time to first spall vs rubber content
plt.figure(figsize=(10, 6))
for rc in [0, 10, 20, 30]:
    rc_data = data[data['Rubber_Content_Percent'] == rc]
    avg_time = rc_data['First_Spall_Time_min'].mean()
    std_time = rc_data['First_Spall_Time_min'].std()
    plt.bar(rc, avg_time, yerr=std_time, capsize=5, alpha=0.7, 
            label=f'{rc}% Rubber')

plt.xlabel('Rubber Content (%)', fontsize=12)
plt.ylabel('Time to First Spall (min)', fontsize=12)
plt.title('Spalling Resistance (ISO834, 10 MPa Load)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

### Example 7: Compare Fire Curves

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Furnace temperature comparison
for fire_curve in ['iso834_temps', 'astm_e119_temps', 'hydrocarbon_temps']:
    data = loader.validation_data[fire_curve]
    data_0 = data[data['Rubber_Content_Percent'] == 0]
    ax1.plot(data_0['Time_min'], data_0['Furnace_Temp_C'], 
             linewidth=2.5, label=fire_curve.replace('_temps', '').upper())

ax1.set_xlabel('Time (min)', fontsize=12)
ax1.set_ylabel('Furnace Temperature (°C)', fontsize=12)
ax1.set_title('Fire Curve Comparison', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Core temperature for 20% rubber in all fire curves
for fire_curve, label in [('iso834_temps', 'ISO834'), 
                           ('astm_e119_temps', 'ASTM E119'),
                           ('hydrocarbon_temps', 'Hydrocarbon')]:
    data = loader.validation_data[fire_curve]
    data_20 = data[data['Rubber_Content_Percent'] == 20]
    ax2.plot(data_20['Time_min'], data_20['TC6_Center_C'], 
             linewidth=2.5, label=label)

ax2.set_xlabel('Time (min)', fontsize=12)
ax2.set_ylabel('Core Temperature (°C)', fontsize=12)
ax2.set_title('Core Temperature (20% Rubber)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Common Tasks

### Task 1: Extract Data for Specific Conditions

```python
# Get thermal conductivity for 30% rubber at all temperatures
cond_data = loader.thermal_data['conductivity']
cond_30 = cond_data[cond_data['Rubber_Content_Percent'] == 30]
print(cond_30[['Temperature_C', 'Thermal_Conductivity_W_mK']])
```

### Task 2: Calculate Thermal Diffusivity

```python
import numpy as np

rubber_content = 10

# Get required data
k = loader.thermal_data['conductivity']
cp = loader.thermal_data['specific_heat']
rho = loader.thermal_data['density']

# Filter for specific rubber content
k = k[k['Rubber_Content_Percent'] == rubber_content]
cp = cp[cp['Rubber_Content_Percent'] == rubber_content]
rho = rho[rho['Rubber_Content_Percent'] == rubber_content]

# Calculate thermal diffusivity: alpha = k / (rho * cp)
alpha = (k['Thermal_Conductivity_W_mK'].values / 
         (rho['Density_kg_m3'].values * cp['Specific_Heat_J_kgK'].values))

# Convert to mm²/s
alpha_mm2_s = alpha * 1e6

print(f"\nThermal Diffusivity for {rubber_content}% Rubber:")
for i, T in enumerate(k['Temperature_C'].values):
    print(f"  {T}°C: {alpha_mm2_s[i]:.4f} mm²/s")
```

### Task 3: Export Subset of Data

```python
# Export validation data for specific condition
validation = loader.get_validation_data_for_model(
    fire_curve='ISO834',
    rubber_content=20
)

# Save temperature data
validation['temperature'].to_csv('validation_temp_ISO834_20pct.csv', index=False)

# Save strain data
validation['strain'].to_csv('validation_strain_ISO834_20pct.csv', index=False)

print("Validation data exported!")
```

### Task 4: Compare Residual Strength Ratios

```python
import matplotlib.pyplot as plt

strength = loader.mechanical_data['compressive_strength']

plt.figure(figsize=(10, 6))
for rc in [0, 10, 20, 30]:
    data = strength[strength['Rubber_Content_Percent'] == rc]
    plt.plot(data['Temperature_C'], data['Residual_Strength_Ratio'], 
             'o-', linewidth=2, markersize=6, label=f'{rc}% Rubber')

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Residual Strength Ratio (f_T / f_20)', fontsize=12)
plt.title('Residual Compressive Strength Ratio', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
plt.show()
```

## Tips & Best Practices

### 1. Always Check Data Units
```python
# Units are specified in column names and metadata
# Temperature: °C
# Thermal conductivity: W/mK
# Specific heat: J/kgK
# Density: kg/m³
# Strength: MPa
# Modulus: GPa
# Strain: microstrain (×10⁻⁶)
```

### 2. Interpolation for Intermediate Values
```python
from scipy.interpolate import interp1d

# Get data
cond = loader.thermal_data['conductivity']
cond_0 = cond[cond['Rubber_Content_Percent'] == 0]

# Create interpolation function
T = cond_0['Temperature_C'].values
k = cond_0['Thermal_Conductivity_W_mK'].values
f_interp = interp1d(T, k, kind='linear', fill_value='extrapolate')

# Get value at arbitrary temperature
k_at_450C = f_interp(450)
print(f"Thermal conductivity at 450°C: {k_at_450C:.4f} W/mK")
```

### 3. Handle Missing Data
```python
# Check for missing values
cond = loader.thermal_data['conductivity']
print(f"Missing values: {cond.isnull().sum().sum()}")

# Data is complete for 20-1000°C range
# For extrapolation beyond this range, use with caution
```

### 4. Statistical Analysis
```python
# Calculate statistics for replicate tests
spalling = loader.validation_data['spalling_iso834']
stats = spalling.groupby(['Rubber_Content_Percent', 'Applied_Load_MPa']).agg({
    'First_Spall_Time_min': ['mean', 'std', 'min', 'max'],
    'Total_Spalled_Mass_g': ['mean', 'std']
})

print("\nSpalling Statistics:")
print(stats)
```

## Troubleshooting

**Problem:** Import errors
```bash
# Solution: Make sure you're in the scripts directory
cd phase4_numerical_modeling_dataset/scripts
python data_loader.py
```

**Problem:** Cannot find data files
```python
# Solution: Check your base_path in data_loader
loader = RubberizedConcreteDataLoader(base_path="../")
# Adjust path as needed based on your working directory
```

**Problem:** Plots not displaying
```python
# Solution: Add plt.show() or use inline plotting
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
# ... your plotting code ...
plt.show()
```

**Problem:** Need data for temperatures not in dataset
```python
# Solution: Use interpolation (see Tips section above)
# Be cautious extrapolating beyond 1000°C
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Check metadata/test_conditions.json** for test parameters
3. **Explore all validation datasets** to understand available data
4. **Develop your FE model** using model_calibration_helper.py
5. **Validate your model** against the validation datasets (don't use for calibration!)

## Getting Help

- **Documentation:** See README.md for comprehensive guide
- **Examples:** Check the scripts directory for working examples
- **Data questions:** Review metadata/test_conditions.json
- **Modeling questions:** See model_calibration_helper.py for templates

---

**Happy Modeling! 🔥🧱**
