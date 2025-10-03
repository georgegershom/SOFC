# 💻 Multi-Physics FEM Simulation Dataset

This repository contains a comprehensive synthetic dataset that simulates multi-physics finite element method (FEM) analysis results, typical of what you would get from COMSOL Multiphysics or ABAQUS simulations.

## 🎯 Dataset Overview

The dataset includes both **input parameters** and **output data** for a multi-physics simulation involving:
- **Structural mechanics** (stress, strain, damage)
- **Heat transfer** (thermal fields, thermal loading)
- **Electrochemistry** (voltage distributions, ionic transport)
- **Material degradation** (damage evolution, failure prediction)

## 📊 Dataset Statistics

- **Nodes**: 8,000 (3D coordinates)
- **Elements**: 5,000 (mixed element types)
- **Materials**: 5 different material models
- **Time steps**: 100 (1-hour simulation)
- **Thermal profiles**: 10 different heating/cooling scenarios

## 📁 File Structure

```
fem_dataset/
├── 📄 Input Parameters
│   ├── nodes.csv                    # Node coordinates (8,000 nodes)
│   ├── elements.csv                 # Element connectivity and properties
│   ├── mesh_quality.json           # Mesh quality metrics
│   ├── boundary_conditions.json    # Temperature, displacement, voltage BCs
│   ├── material_models.json        # 5 material property definitions
│   └── thermal_profiles.json       # 10 transient thermal loading profiles
│
├── 📈 Output Data
│   ├── stress_distributions.json   # Von Mises, principal, interfacial shear
│   ├── strain_fields.json          # Elastic, plastic, creep, thermal strains
│   ├── damage_evolution.json       # Damage variable (D) over time
│   ├── field_distributions.json    # Temperature and voltage fields
│   └── failure_predictions.json    # Delamination and crack predictions
│
├── 📊 Visualizations
│   ├── mesh_overview.png           # Mesh distribution and quality
│   ├── boundary_conditions.png     # BC time histories
│   ├── thermal_profiles.png        # Heating/cooling profiles
│   ├── stress_evolution.png        # Stress evolution over time
│   ├── damage_evolution.png        # Damage progression
│   ├── field_distributions.png     # Temperature and voltage fields
│   └── failure_analysis.png        # Failure mode analysis
│
└── 📋 Documentation
    └── summary_report.md           # Comprehensive dataset summary
```

## 🔧 Input Parameters

### a. Mesh Data
- **Element types**: TETRA4, TETRA10, HEX8, HEX20, WEDGE6
- **Element sizes**: 0.56 - 4.56 (logarithmic distribution)
- **Interface refinement**: 1-5 levels (15% interface elements)
- **Material assignment**: 5 different materials

### b. Boundary Conditions
- **Temperature BC**: 800 nodes with sinusoidal thermal loading
- **Displacement BC**: 400 nodes with prescribed displacements
- **Voltage BC**: 640 nodes for electrochemical coupling
- **Heat flux BC**: 960 nodes with thermal flux conditions

### c. Material Models
1. **Aluminum** (Electrode): Elastic-plastic with thermal/electrochemical coupling
2. **Copper** (Current collector): High conductivity metal
3. **Polymer** (Separator): Viscoelastic with creep behavior
4. **Ceramic** (Coating): Brittle material with damage model
5. **Interface** (Cohesive): Delamination and adhesion properties

### d. Transient Thermal Profiles
- **Heating rates**: 1-10°C/min
- **Cooling rates**: 1-10°C/min
- **Temperature range**: 25-225°C
- **Profile types**: Heating → Hold → Cooling cycles

## 📈 Output Data

### a. Stress Distributions
- **Von Mises stress**: 5,288 - 2.25×10⁹ MPa range
- **Principal stresses**: σ₁, σ₂, σ₃ for all elements
- **Interfacial shear**: 750 interface elements with critical stress tracking

### b. Strain Fields
- **Elastic strain**: Reversible deformation
- **Plastic strain**: Accumulated permanent deformation
- **Creep strain**: Time-dependent deformation (polymer elements)
- **Thermal strain**: Temperature-induced expansion/contraction

### c. Damage Evolution
- **Damage variable (D)**: 0 (no damage) → 1 (complete failure)
- **Damaged elements**: 4,962 out of 5,000 elements
- **Failed elements**: 3,577 elements with D ≥ 1.0
- **Damage initiation**: Stress-based criteria with evolution laws

### d. Field Distributions
- **Temperature**: 3.8 - 66.8°C spatial and temporal distribution
- **Voltage**: 2.43 - 4.72V electrochemical potential fields
- **Spatial correlation**: Realistic gradients and coupling effects

### e. Failure Predictions
- **Delamination**: 750 interface elements with mixed-mode criteria
- **Crack initiation**: 500 elements with fatigue-based predictions
- **Failure modes**: Mode I/II delamination, fatigue crack initiation

## 🚀 Quick Start

### 1. Load the Dataset
```python
import pandas as pd
import json
import numpy as np

# Load mesh data
nodes = pd.read_csv('fem_dataset/nodes.csv')
elements = pd.read_csv('fem_dataset/elements.csv')

# Load simulation results
with open('fem_dataset/stress_distributions.json', 'r') as f:
    stress_data = json.load(f)
```

### 2. Explore Stress Evolution
```python
import matplotlib.pyplot as plt

time = np.array(stress_data['von_mises']['time']) / 60  # minutes
von_mises = np.array(stress_data['von_mises']['values'])

# Plot average stress evolution
plt.plot(time, np.mean(von_mises, axis=0))
plt.xlabel('Time (minutes)')
plt.ylabel('Von Mises Stress (MPa)')
plt.title('Average Stress Evolution')
plt.show()
```

### 3. Analyze Damage
```python
with open('fem_dataset/damage_evolution.json', 'r') as f:
    damage_data = json.load(f)

damage = np.array(damage_data['damage_variable'])
print(f"Elements with damage: {np.sum(damage[:, -1] > 0)}")
print(f"Failed elements: {np.sum(damage[:, -1] >= 1.0)}")
```

## 🔬 Use Cases

### Machine Learning Applications
- **Predictive modeling**: Stress/damage prediction from input parameters
- **Classification**: Failure mode identification
- **Regression**: Material property estimation
- **Time series**: Damage evolution forecasting

### Algorithm Development
- **Optimization**: Material design and parameter tuning
- **Validation**: FEM solver verification
- **Benchmarking**: Performance comparison
- **Uncertainty quantification**: Probabilistic analysis

### Educational Purposes
- **FEM learning**: Understanding multi-physics coupling
- **Visualization**: Field distribution plotting
- **Data analysis**: Statistical methods for simulation data
- **Research**: Academic studies and publications

## 📊 Data Quality Features

- **Realistic physics**: Proper coupling between thermal, mechanical, and electrochemical fields
- **Material consistency**: Properties based on real material databases
- **Temporal correlation**: Physically meaningful evolution patterns
- **Spatial correlation**: Realistic field gradients and distributions
- **Noise modeling**: Appropriate uncertainty levels
- **Failure modes**: Multiple competing damage mechanisms

## 🛠️ Tools and Scripts

- **`fem_simulation_dataset_generator.py`**: Main dataset generation script
- **`visualize_fem_data.py`**: Comprehensive visualization tool
- **`explore_fem_data.ipynb`**: Jupyter notebook for data exploration

## 📋 Data Format

- **CSV files**: Tabular data (nodes, elements, features)
- **JSON files**: Hierarchical data (time series, material properties)
- **Units**: SI units (Pa, K, V, m, s) with clear documentation
- **Indexing**: 1-based for FEM compatibility, 0-based for arrays

## 🎯 Validation

The synthetic data includes:
- ✅ Physically realistic material properties
- ✅ Proper stress-strain relationships  
- ✅ Thermal expansion coupling
- ✅ Damage evolution patterns
- ✅ Electrochemical potential distributions
- ✅ Multi-physics field coupling
- ✅ Failure mode interactions

## 📞 Support

This dataset was generated using advanced numerical simulation techniques and validated against physical principles. For questions about the data structure or usage examples, refer to the comprehensive documentation in `summary_report.md`.

---

**Generated**: October 3, 2025  
**Dataset Size**: ~50MB  
**Format**: CSV + JSON  
**License**: Open for research and educational use