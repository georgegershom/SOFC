# Atomic-Scale Simulation Dataset for Quantum-Enhanced Materials Modeling

This repository contains a comprehensive synthetic dataset designed for quantum-enhanced materials modeling, specifically targeting creep behavior and mechanical properties at the atomic scale.

## Dataset Overview

The dataset consists of computationally generated data that would typically come from Density Functional Theory (DFT) and Molecular Dynamics (MD) simulations. This synthetic data is designed to feed critical parameters into larger-scale materials models.

### Dataset Components

#### 1. DFT Calculation Data
- **Formation Energies** (`dft_formation_energies.csv`): 1,000 samples
  - Defect formation energies for vacancies, interstitials, substitutional atoms, dislocations, and grain boundaries
  - Materials: Al, Cu, Fe, Ni, Ti, Mg, Zn
  - Temperature range: 300-1200 K
  - Includes computational parameters (k-points, cutoff energy, exchange-correlation functional)

- **Activation Barriers** (`activation_barriers.csv`): 500 samples
  - Energy barriers for diffusion processes including vacancy migration, interstitial migration, grain boundary diffusion
  - Stress and grain size effects included
  - Attempt frequencies based on Debye model

- **Surface Energies** (`surface_energies.csv`): 300 samples
  - Surface energies for different crystallographic planes
  - Miller indices: (100), (110), (111), (210), (211), (310)
  - Temperature and atmosphere effects

#### 2. MD Simulation Data
- **Grain Boundary Sliding** (`md_data/grain_boundary_sliding.csv`): 200 simulations
  - Sliding resistance and rates for different grain boundary types (tilt, twist, mixed, twin)
  - Stress-dependent behavior with power-law relationships
  - Temperature effects on sliding mechanisms

- **Dislocation Mobility** (`md_data/dislocation_mobility.csv`): 150 simulations
  - Mobility data for edge, screw, and mixed dislocations
  - Temperature and stress dependencies
  - Dislocation density effects

- **Force Field Parameters** (`md_data/force_field_parameters.csv`)
  - EAM potential parameters for all materials
  - Lennard-Jones parameters, cohesive energies, bulk moduli

- **Sample Trajectories** (`md_data/sample_trajectories.json`): 10 trajectories
  - Atomic positions, forces, and energies over time
  - Thermodynamic properties (temperature, pressure)
  - Metadata including simulation parameters

## Data Format and Structure

### CSV Files
All tabular data is stored in CSV format with descriptive column headers. Key features include:
- Material identification
- Temperature and stress conditions
- Computational parameters
- Physical properties and derived quantities

### JSON Files
Complex trajectory data and metadata are stored in JSON format for easy parsing and analysis.

## Physical Realism

The synthetic data incorporates realistic physical relationships:

1. **Temperature Dependencies**: Arrhenius-type behavior for diffusion processes
2. **Stress Effects**: Power-law relationships for creep mechanisms
3. **Material Properties**: Based on experimental values from literature
4. **Computational Parameters**: Realistic DFT and MD simulation settings
5. **Statistical Variations**: Appropriate noise levels reflecting computational uncertainty

## Usage Examples

### Loading and Basic Analysis
```python
import pandas as pd
import numpy as np

# Load DFT formation energy data
dft_data = pd.read_csv('atomic_simulation_dataset/dft_formation_energies.csv')

# Filter for aluminum vacancy data
al_vacancies = dft_data[(dft_data['material'] == 'Al') & 
                       (dft_data['defect_type'] == 'vacancy')]

# Calculate temperature-dependent formation energy
temperature_effect = al_vacancies.groupby('temperature_K')['formation_energy_eV'].mean()
```

### Machine Learning Applications
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare features for formation energy prediction
features = ['temperature_K', 'defect_concentration', 'supercell_size', 'cutoff_energy_eV']
X = dft_data[features]
y = dft_data['formation_energy_eV']

# Train predictive model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

### Integration with Phase-Field Models
The dataset provides parameters for:
- Diffusion coefficients: `D = D0 * exp(-Q/kT)` where Q is from activation barriers
- Interface energies: From surface energy calculations
- Elastic properties: From force field parameters

## Dataset Statistics

- **Total Samples**: 2,160 across all datasets
- **Materials Covered**: 7 (Al, Cu, Fe, Ni, Ti, Mg, Zn)
- **Temperature Range**: 300-1200 K
- **Property Range**: 
  - Formation energies: 0.1-8.5 eV
  - Activation barriers: 0.1-2.5 eV
  - Surface energies: 0.1-15 J/m²

## Quality Assurance

The dataset includes:
- Physical constraint enforcement (positive energies, realistic ranges)
- Statistical validation (appropriate distributions, correlations)
- Computational parameter consistency
- Material property relationships based on experimental data

## Applications

This dataset is designed for:
1. **Surrogate Model Training**: Fast approximation of expensive DFT/MD calculations
2. **Multi-scale Model Parameterization**: Feeding atomic-scale data into continuum models
3. **Machine Learning Research**: Development of ML models for materials properties
4. **Quantum-Enhanced Modeling**: Integration with quantum mechanical methods
5. **Creep Mechanism Studies**: Understanding temperature and stress dependencies

## File Structure
```
atomic_simulation_dataset/
├── dft_formation_energies.csv
├── activation_barriers.csv
├── surface_energies.csv
├── md_data/
│   ├── grain_boundary_sliding.csv
│   ├── dislocation_mobility.csv
│   ├── force_field_parameters.csv
│   └── sample_trajectories.json
├── figures/
│   ├── formation_energies_by_material.png
│   ├── activation_barriers_vs_temperature.png
│   ├── surface_energies_by_miller.png
│   ├── grain_boundary_analysis.png
│   ├── dislocation_mobility_analysis.png
│   ├── dft_correlation_matrix.png
│   ├── material_property_comparison.png
│   └── ml_analysis.png
├── dataset_summary.json
├── analysis_summary.json
├── ml_analysis_results.json
└── data_quality_report.json
```

## Citation

If you use this dataset in your research, please cite:
```
Atomic-Scale Simulation Dataset for Quantum-Enhanced Materials Modeling
Generated using synthetic DFT and MD simulation methods
[Year] [Institution/Author]
```

## License

This dataset is provided for research and educational purposes. Please refer to the license file for specific terms and conditions.

## Contact

For questions about the dataset or requests for additional data, please contact [contact information].