# Multi-Fidelity Digital Twin for SOFCs Dataset

## Overview

This repository contains a comprehensive multi-fidelity dataset for Solid Oxide Fuel Cell (SOFC) digital twin modeling, specifically designed for predicting thermo-mechanical degradation through multi-scale modeling and deep learning approaches.

## Dataset Description

### Multi-Scale Model Input Parameters & Operating Conditions

The dataset provides a comprehensive matrix of operating conditions and material parameters that can be varied across simulations/experiments for SOFC digital twin development.

### Fidelity Levels

- **LF (Low Fidelity)**: 1,000 samples - Basic parameter ranges for initial modeling
- **MF (Medium Fidelity)**: 1,000 samples - Intermediate parameter ranges for refined modeling  
- **HF (High Fidelity)**: 1,000 samples - Detailed parameter ranges for high-precision modeling

### Parameter Categories

#### System-Level Operating Conditions
- Fuel Utilization (Uf): 0.6 - 0.9
- Oxidant Utilization: 0.15 - 0.25
- Current Density: 0.1 - 1.0 A/cm²
- Voltage: 0.6 - 1.0 V
- Temperature: 973 - 1073 K (700-800°C)
- Pressure: 1.0 - 1.2 atm
- Flow Rates (Fuel/Air): 50-200 sccm / 200-800 sccm

#### Transient Cycles
- Start-up/Shut-down profiles
- Load-following ramps
- Thermal cycling frequency
- Load cycling frequency

#### Cell/Stack Geometry
- Cell active area: 25-100 cm²
- Layer thicknesses (Anode, Cathode, Electrolyte, Interconnects)
- Channel dimensions for full stack modeling

#### Material Properties

**Anode (Ni-YSZ)**:
- Porosity: 0.2 - 0.4
- Tortuosity: 2.0 - 6.0
- Ni particle size: 0.5 - 3.0 μm
- YSZ particle size: 0.3 - 1.0 μm
- TPB density: 1e12 - 1e15 m⁻²
- Ionic/Electronic conductivity
- Young's Modulus: 50-200 GPa
- Thermal expansion coefficient

**Cathode (LSCF)**:
- Porosity: 0.15 - 0.35
- Tortuosity: 2.5 - 7.0
- LSCF particle size: 0.2 - 2.0 μm
- GDC particle size: 0.1 - 0.8 μm
- Chemical expansion coefficients
- Young's Modulus: 80-150 GPa

**Electrolyte (YSZ)**:
- Ionic conductivity: 0.01 - 0.1 S/m
- Young's Modulus: 200-300 GPa
- Fracture toughness: 1.0 - 3.0 MPa·m^0.5
- Thermal expansion coefficient

**Interconnect (Crofer 22APU)**:
- Thermal expansion coefficient
- Young's Modulus: 200-250 GPa
- Oxide scale growth rate
- Electrical resistivity

#### Microstructural Properties (FIB-SEM/X-Ray Tomography)
- 3D voxel data of reconstructed microstructure
- Phase fractions (Ni, YSZ, Pore)
- Particle size distributions
- Specific surface area
- Connectivity parameters
- Tortuosity factors

#### Degradation Parameters
- Ni agglomeration rate
- Ni oxidation rate
- Cathode poisoning rate
- Electrolyte cracking rate
- Thermal stress accumulation
- Redox cycling damage

## Dataset Structure

```
sofc_dataset/
├── sofc_dataset_lf.csv          # Low fidelity dataset (1,000 samples)
├── sofc_dataset_mf.csv          # Medium fidelity dataset (1,000 samples)
├── sofc_dataset_hf.csv          # High fidelity dataset (1,000 samples)
├── sofc_dataset.h5              # HDF5 format with all fidelity levels
├── metadata.json                # Dataset metadata and parameter descriptions
├── summary_report.md            # Comprehensive dataset summary
├── usage_examples.md            # Usage examples and code snippets
├── visualizations/              # Dataset visualization plots
│   ├── parameter_distributions_*.png
│   ├── fidelity_comparison_*.png
│   └── correlation_heatmap_*.png
└── analysis/                    # Advanced parameter space analysis
    ├── parameter_ranges_comparison.png
    ├── multidimensional_scaling.png
    ├── parameter_importance.png
    └── fidelity_differences.png
```

## Usage

### Loading the Dataset

```python
import pandas as pd
import h5py

# Load CSV files
lf_data = pd.read_csv('sofc_dataset_lf.csv')
mf_data = pd.read_csv('sofc_dataset_mf.csv')
hf_data = pd.read_csv('sofc_dataset_hf.csv')

# Or load from HDF5
with h5py.File('sofc_dataset.h5', 'r') as f:
    lf_group = f['LF']
    mf_group = f['MF']
    hf_group = f['HF']
```

### Machine Learning Example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Combine all fidelity levels
all_data = pd.concat([lf_data, mf_data, hf_data], ignore_index=True)

# Prepare features and target
feature_cols = [col for col in all_data.columns if col not in ['fidelity_level']]
X = all_data[feature_cols]
y = all_data['efficiency']  # or any other target variable

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## Sampling Methodology

The dataset was generated using **Latin Hypercube Sampling (LHS)** to efficiently explore the multi-dimensional parameter space. This approach ensures:

- Uniform coverage of the parameter space
- Minimal correlation between parameters
- Efficient sampling for high-dimensional spaces
- Reproducible results (random seed: 42)

## Applications

This dataset is designed for:

1. **Multi-Fidelity Digital Twin Development**: Train models that can operate across different fidelity levels
2. **Thermo-Mechanical Degradation Prediction**: Model long-term performance and degradation
3. **Multi-Scale Modeling**: Bridge between different scales (system, cell, microstructure)
4. **Deep Learning Applications**: Neural networks for parameter optimization and prediction
5. **Design of Experiments**: Optimize SOFC design parameters
6. **Uncertainty Quantification**: Assess model uncertainty and parameter sensitivity

## Technical Specifications

- **Total Samples**: 3,000 (1,000 per fidelity level)
- **Total Parameters**: 68 parameters across all categories
- **File Formats**: CSV, HDF5, JSON metadata
- **Sampling Method**: Latin Hypercube Sampling
- **Reproducibility**: Fixed random seed (42)
- **Memory Usage**: ~2-3 MB per CSV file

## Citation

If you use this dataset in your research, please cite:

```
Multi-Fidelity Digital Twin for SOFCs: Multi-Scale Modeling and Deep Learning for Predicting Thermo-Mechanical Degradation
Dataset 1: Multi-Scale Model Input Parameters & Operating Conditions
Generated using Latin Hypercube Sampling for comprehensive parameter space exploration
```

## License

This dataset is provided for academic and research purposes. Please ensure proper attribution when using this data in publications or commercial applications.

## Contact

For questions about the dataset or collaboration opportunities, please refer to the documentation files included in the dataset directory.

---

**Note**: This dataset represents a comprehensive parameter space for SOFC modeling. The parameter ranges are based on literature values and experimental data, but users should validate against their specific application requirements.