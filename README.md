# 💻 Multi-Physics FEM Numerical Simulation Dataset

This repository contains a comprehensive numerical simulation dataset generated to mimic outputs from advanced multi-physics finite element modeling (FEM) software such as COMSOL Multiphysics and ABAQUS. The dataset includes thermal, mechanical, and electrochemical simulations for battery materials analysis.

## 📊 Dataset Overview

**Generated:** October 3, 2025  
**Simulation Duration:** 3,590 seconds (≈60 minutes)  
**Time Steps:** 20 sampled snapshots  
**Total Nodes:** 13,500  
**Total Elements:** 11,774  

### Physics Domains
- ✅ **Thermal Analysis**: Heat transfer, temperature distributions, thermal cycling
- ✅ **Mechanical Analysis**: Stress, strain, damage evolution, delamination
- ✅ **Electrochemical Analysis**: Voltage distributions, ionic transport

---

## 📁 Dataset Structure

```
fem_simulation_data/
├── dataset_metadata.json              # Complete metadata and input parameters
├── mesh_nodes.csv                     # FEM mesh node coordinates (13,500 nodes)
├── mesh_elements.csv                  # FEM element connectivity and properties
├── simulation_summary.csv             # Time series summary statistics
├── time_series_output/                # 20 timestep output files (~87MB)
│   ├── output_t000_time_0.0s.csv
│   ├── output_t001_time_180.0s.csv
│   └── ...
└── visualizations/                    # High-quality visualization plots
    ├── damage_evolution.png
    ├── stress_evolution.png
    ├── failure_mechanisms.png
    ├── temperature_distribution.png
    └── voltage_distribution.png
```

---

## 🔧 Input Parameters

### a. Mesh Data
- **Element Types**: C3D8 (linear brick), C3D8R (reduced integration brick)
- **Mesh Dimensions**: 30 × 30 × 15 elements
- **Domain Size**: 10 × 10 × 5 mm³
- **Element Size**: 0.1–0.2 mm (variable)
- **Interface Refinement Factor**: 2.5× (at z=2.5mm interface)

### b. Boundary Conditions

#### Temperature BC
- **Bottom Surface**: Fixed at 25°C (900 nodes)
- **Top Surface**: Time-dependent thermal cycling
- **Thermal Profile**: 5°C/min heating and cooling rates

#### Displacement BC
- **Bottom Surface**: Fixed in z-direction
- **Side Faces**: Symmetry conditions (x/y constrained)
- **Total Constrained Nodes**: 910

#### Voltage BC
- **Cathode (Top)**: 4.2 V
- **Anode (Bottom)**: 0.0 V (ground)

### c. Material Models

#### Cathode Material: NMC (LiNi₀.₈Mn₀.₁Co₀.₁O₂)
- **Elastic Properties**:
  - Young's Modulus: 150 GPa
  - Poisson's Ratio: 0.3
  - Density: 4,700 kg/m³
- **Plastic Properties**:
  - Yield Stress: 150 MPa
  - Hardening Modulus: 2 GPa
  - Hardening Exponent: 0.2
- **Thermal Properties**:
  - Thermal Expansion: 12 × 10⁻⁶ K⁻¹
  - Thermal Conductivity: 2.0 W/(m·K)
  - Specific Heat: 700 J/(kg·K)
- **Electrochemical Properties**:
  - Diffusion Coefficient: 1 × 10⁻¹⁴ m²/s
  - Ionic Conductivity: 0.1 S/m
  - Max Concentration: 51,765 mol/m³

#### Anode Material: Graphite
- **Elastic Properties**:
  - Young's Modulus: 15 GPa
  - Poisson's Ratio: 0.3
  - Density: 2,260 kg/m³
- **Plastic Properties**:
  - Yield Stress: 50 MPa
  - Hardening Modulus: 1 GPa
  - Hardening Exponent: 0.15
- **Thermal Properties**:
  - Thermal Expansion: 8 × 10⁻⁶ K⁻¹
  - Thermal Conductivity: 1.5 W/(m·K)
  - Specific Heat: 1,200 J/(kg·K)
- **Electrochemical Properties**:
  - Diffusion Coefficient: 3.9 × 10⁻¹⁴ m²/s
  - Ionic Conductivity: 0.05 S/m
  - Max Concentration: 30,555 mol/m³

#### Separator: Polymer
- **Elastic Properties**:
  - Young's Modulus: 0.5 GPa
  - Poisson's Ratio: 0.4
  - Density: 1,200 kg/m³
- **Thermal Properties**:
  - Thermal Expansion: 50 × 10⁻⁶ K⁻¹
  - Thermal Conductivity: 0.3 W/(m·K)
  - Specific Heat: 1,200 J/(kg·K)
- **Electrochemical Properties**:
  - Ionic Conductivity: 1.0 S/m
  - Porosity: 0.4

#### Creep Model: Power-Law Creep
- **Coefficient A**: 1 × 10⁻¹⁰ (Pa⁻ⁿ·s⁻¹)
- **Stress Exponent n**: 5.0
- **Activation Energy Q**: 120 kJ/mol
- **Gas Constant R**: 8.314 J/(mol·K)

### d. Transient Thermal Profiles

Generated 9 thermal cycling profiles with combinations of:
- **Heating Rates**: 1, 5, 10 °C/min
- **Cooling Rates**: 1, 5, 10 °C/min
- **Temperature Range**: 25–60 °C
- **Simulation Duration**: 3,600 seconds

---

## 📈 Output Data

### b. Output Variables (per node, per time step)

#### Stress Distributions
- **von Mises Stress** (Pa): Equivalent stress for yielding
- **Principal Stress σ₁, σ₂, σ₃** (Pa): Principal stress components
- **Interfacial Shear Stress** (Pa): Critical for delamination
- **Hydrostatic Stress** (Pa): Mean normal stress

#### Strain Fields
- **Elastic Strain** (dimensionless): Recoverable deformation
- **Plastic Strain** (dimensionless): Permanent deformation
- **Creep Strain** (dimensionless): Time-dependent deformation
- **Thermal Strain** (dimensionless): Temperature-induced expansion/contraction
- **Total Strain** (dimensionless): Sum of all strain components

#### Damage Variable (D)
- **Range**: 0 (undamaged) to 1 (fully damaged)
- **Evolution**: Based on stress, plastic strain, and interface location
- **Final Max Damage**: 0.0406 (4.06%)
- **Final Avg Damage**: ~0.01 (1%)

#### Temperature Distributions
- **Temperature** (°C): Nodal temperatures
- **Temperature Gradient** (°C/mm): Spatial temperature variation
- **Range**: 25–60 °C (base to peak cycling)

#### Voltage Distributions
- **Voltage** (V): Electrochemical potential
- **Current Density** (A/m²): Derived from voltage gradient
- **Range**: 0.0 V (anode) to 4.2 V (cathode)

#### Delamination Predictions
- **Delamination Risk** (dimensionless): Ratio of shear stress to critical shear
- **Delamination Initiated** (boolean): True if risk > 1.0
- **Delamination Area** (mm²): Total delaminated interface area
- **Final Delamination Area**: 107.84 mm²

#### Crack Initiation Predictions
- **Crack Risk** (dimensionless): Combined stress and damage criterion
- **Crack Initiated** (boolean): True if crack nucleated
- **Crack Count**: Number of crack initiation sites
- **Crack Propagation Angle** (rad): Direction based on principal stresses

---

## 📊 Key Results

### Simulation Summary Statistics

| Time (s) | Max von Mises (MPa) | Max Plastic Strain | Max Damage | Delamination Area (mm²) | Crack Count |
|----------|--------------------|--------------------|------------|------------------------|-------------|
| 0        | 235.3              | 0.000              | 0.0000     | 53.36                  | 0           |
| 180      | 343.9              | 0.022              | 0.0021     | 57.68                  | 0           |
| 370      | 340.6              | 0.044              | 0.0043     | 62.40                  | 0           |
| 560      | 339.6              | 0.066              | 0.0064     | 66.56                  | 0           |
| ...      | ...                | ...                | ...        | ...                    | ...         |
| 3590     | 331.9              | 0.396              | 0.0406     | 107.84                 | 0           |

### Failure Analysis
- **No Crack Initiation** observed during simulation
- **Delamination Growth**: From 53 mm² to 108 mm² (+103%)
- **Damage Accumulation**: Progressive increase to 4.06% max
- **Stress State**: Stable with cyclic variations

---

## 🚀 Usage Examples

### Python: Load and Analyze Data

```python
import pandas as pd
import numpy as np
import json

# Load metadata
with open('fem_simulation_data/dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load mesh
nodes = pd.read_csv('fem_simulation_data/mesh_nodes.csv')
elements = pd.read_csv('fem_simulation_data/mesh_elements.csv')

# Load time series data
t10 = pd.read_csv('fem_simulation_data/time_series_output/output_t010_time_1880.0s.csv')

# Analyze stress distribution
print(f"Max von Mises stress: {t10['von_mises_stress'].max()/1e6:.2f} MPa")
print(f"Average damage: {t10['damage'].mean():.4f}")
print(f"Delamination area: {t10['delamination_risk'].sum()*0.04:.2f} mm²")

# Filter high-stress regions
high_stress = t10[t10['von_mises_stress'] > 300e6]
print(f"Nodes with stress > 300 MPa: {len(high_stress)}")
```

### Python: Visualize Results

```python
import matplotlib.pyplot as plt

# Load summary
summary = pd.read_csv('fem_simulation_data/simulation_summary.csv')

# Plot damage evolution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(summary['time']/60, summary['max_damage']*100, 'r-', linewidth=2)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Maximum Damage (%)')
ax.set_title('Damage Evolution During Thermal Cycling')
ax.grid(True, alpha=0.3)
plt.show()
```

### MATLAB: Import Data

```matlab
% Load mesh nodes
nodes = readtable('fem_simulation_data/mesh_nodes.csv');

% Load time series
data = readtable('fem_simulation_data/time_series_output/output_t010_time_1880.0s.csv');

% Extract coordinates and stress
x = data.x;
y = data.y;
z = data.z;
stress = data.von_mises_stress / 1e6; % Convert to MPa

% Create 3D scatter plot
figure;
scatter3(x, y, z, 20, stress, 'filled');
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
zlabel('z (mm)');
title('von Mises Stress Distribution (MPa)');
```

---

## 📝 Data Format Specifications

### Mesh Files

#### `mesh_nodes.csv`
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| node_id | int | - | Unique node identifier |
| x | float | mm | x-coordinate |
| y | float | mm | y-coordinate |
| z | float | mm | z-coordinate |

#### `mesh_elements.csv`
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| element_id | int | - | Unique element identifier |
| 0-7 | int | - | Node IDs for 8-node hexahedral element |
| element_type | str | - | C3D8 or C3D8R |
| element_size | float | mm | Characteristic element length |

### Time Series Output Files

Each `output_t###_time_XXX.Xs.csv` contains:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| node_id | int | - | Node identifier |
| x, y, z | float | mm | Spatial coordinates |
| time | float | s | Simulation time |
| von_mises_stress | float | Pa | von Mises equivalent stress |
| principal_stress_1 | float | Pa | Maximum principal stress |
| principal_stress_2 | float | Pa | Intermediate principal stress |
| principal_stress_3 | float | Pa | Minimum principal stress |
| shear_stress | float | Pa | Interfacial shear stress |
| elastic_strain | float | - | Elastic strain component |
| plastic_strain | float | - | Plastic strain component |
| creep_strain | float | - | Creep strain component |
| thermal_strain | float | - | Thermal strain component |
| total_strain | float | - | Total strain |
| damage | float | - | Damage variable (0-1) |
| temperature | float | °C | Nodal temperature |
| voltage | float | V | Electrochemical potential |
| delamination_risk | float | - | Delamination risk factor |
| crack_risk | float | - | Crack initiation risk factor |

---

## 🔬 Applications

This dataset is suitable for:

1. **Machine Learning Training**: 
   - Stress prediction from boundary conditions
   - Damage evolution forecasting
   - Failure mode classification

2. **Multi-Physics Model Validation**:
   - Benchmark FEM solvers
   - Validate constitutive models
   - Compare numerical schemes

3. **Battery Material Design**:
   - Optimize electrode compositions
   - Design thermal management systems
   - Predict cycle life

4. **Educational Purposes**:
   - Teach FEM concepts
   - Demonstrate multi-physics coupling
   - Training in data analysis

---

## 🛠️ Regenerating the Dataset

To regenerate or customize the dataset:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the generator
python3 fem_simulation_data_generator.py
```

### Customization Options

Edit the `main()` function in `fem_simulation_data_generator.py`:

```python
# Modify mesh density
mesh_data = generator.generate_mesh_data(
    nx=50,  # Increase for finer mesh
    ny=50,
    nz=25,
    refinement_factor=3.0  # Higher = finer at interfaces
)

# Modify thermal profiles
thermal_profiles = generator.generate_transient_thermal_profiles(
    duration=7200,  # Longer simulation (2 hours)
    dt=5,           # Smaller time steps
    heating_rates=[1, 2, 5, 10, 15],  # More profiles
    cooling_rates=[1, 2, 5, 10, 15]
)
```

---

## 📚 References

### Material Properties Sources
- NMC Cathode: J. Electrochem. Soc. 167 (2020) 090526
- Graphite Anode: J. Power Sources 196 (2011) 3942-3948
- Polymer Separator: Energy Environ. Sci. 7 (2014) 1307-1338

### Modeling Approaches
- Damage Mechanics: Lemaitre & Chaboche (1990)
- Power-Law Creep: Frost & Ashby (1982)
- Interface Delamination: Mode II Fracture Mechanics

---

## 📄 License

This dataset is provided for research and educational purposes. Feel free to use, modify, and distribute with proper attribution.

---

## 🤝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{fem_simulation_2025,
  title={Multi-Physics FEM Numerical Simulation Dataset for Battery Materials},
  author={FEM Data Generator},
  year={2025},
  version={1.0},
  url={https://github.com/your-repo/fem-simulation-dataset}
}
```

---

## 📧 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Last Updated:** October 3, 2025  
**Dataset Version:** 1.0  
**Generator Version:** 1.0
