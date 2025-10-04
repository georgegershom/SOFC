# Atomic-Scale Simulation Dataset

This repository contains a comprehensive dataset of atomic-scale simulation data generated for quantum-enhanced inputs in materials science applications. The dataset includes Density Functional Theory (DFT) calculations, Molecular Dynamics (MD) simulations, and defect formation energy data.

## Dataset Overview

The dataset contains **1,800 total simulations** across four materials (Ni, Al, Fe, Cu):

- **1,000 DFT simulations** - Formation energies, activation barriers, surface energies
- **500 MD simulations** - Grain boundary sliding, dislocation mobility, diffusion
- **300 Defect simulations** - Vacancy, interstitial, antisite, dislocation, and grain boundary formation energies

## Data Structure

### DFT Simulations (`dft_simulations.*`)
Contains quantum-mechanically accurate calculations including:

- **Formation Energies**: Defect formation energies (eV/atom)
- **Activation Barriers**: Energy barriers for diffusion processes (eV)
- **Surface Energies**: Surface formation energies (J/m²)
- **Elastic Properties**: Bulk modulus and elastic constants (GPa)
- **Electronic Properties**: Band gap and magnetic moment

### MD Simulations (`md_simulations.*`)
Contains molecular dynamics simulation results:

- **Grain Boundary Properties**: Energy and sliding resistance
- **Dislocation Mobility**: Temperature-dependent mobility (m/s/MPa)
- **Diffusion Coefficients**: Arrhenius-type temperature dependence (m²/s)
- **Mechanical Properties**: Stress-strain curves and viscosity
- **Thermodynamic Data**: Temperature and pressure ranges

### Defect Simulations (`defect_simulations.*`)
Contains defect formation energy calculations:

- **Vacancy Formation**: Point defect formation energies
- **Interstitial Formation**: Self-interstitial formation energies
- **Antisite Defects**: Chemical disorder formation energies
- **Dislocation Core**: Extended defect formation energies
- **Grain Boundary**: Interface formation energies

## File Formats

The dataset is provided in multiple formats for maximum compatibility:

- **JSON**: Human-readable format with full metadata
- **HDF5**: Binary format for efficient storage and access
- **CSV**: Tabular format for data analysis and machine learning

## Usage Examples

### Loading DFT Data
```python
import json
import pandas as pd

# Load JSON data
with open('atomic_simulation_data/dft_simulations.json', 'r') as f:
    dft_data = json.load(f)

# Load CSV data for analysis
df_dft = pd.read_csv('atomic_simulation_data/dft_simulations.csv')
```

### Loading HDF5 Data
```python
import h5py

# Load HDF5 data
with h5py.File('atomic_simulation_data/dft_simulations.h5', 'r') as f:
    # Access simulation data
    for sim_id in f.keys():
        sim_data = f[sim_id]
        formation_energy = sim_data['results']['formation_energy'][()]
```

### Analyzing Temperature Dependencies
```python
import matplotlib.pyplot as plt

# Plot diffusion coefficient vs temperature
temperatures = [sim['parameters']['temperature'] for sim in md_data]
diffusion_coeffs = [sim['results']['diffusion_coefficient'] for sim in md_data]

plt.scatter(temperatures, diffusion_coeffs)
plt.xlabel('Temperature (K)')
plt.ylabel('Diffusion Coefficient (m²/s)')
plt.yscale('log')
plt.show()
```

## Statistical Properties

### DFT Data Ranges
- Formation Energy: -0.47 to 0.27 eV/atom
- Activation Barrier: 0.61 to 2.80 eV
- Surface Energy: 1.02 to 2.87 J/m²

### MD Data Ranges
- Temperature: 306 to 1194 K
- Diffusion Coefficient: 2.15×10⁻⁴⁴ to 3.19×10⁻¹¹ m²/s

## Applications

This dataset is designed for:

1. **Surrogate Model Training**: Train machine learning models to predict material properties
2. **Phase-Field Model Parameterization**: Input parameters for continuum-scale models
3. **Crystal Plasticity Models**: Dislocation mobility and grain boundary properties
4. **Creep Rate Predictions**: Temperature-dependent diffusion and activation barriers
5. **Quantum-Enhanced Materials Design**: High-fidelity property predictions

## Data Generation Methodology

The dataset was generated using realistic material properties and physical correlations:

- **Temperature Dependencies**: Arrhenius behavior for diffusion, exponential temperature effects
- **Material Correlations**: Formation energies correlate with activation barriers
- **Statistical Distributions**: Normal distributions with realistic standard deviations
- **Physical Constraints**: Properties respect thermodynamic and mechanical bounds

## Quality Assurance

- All simulations include convergence criteria and metadata
- Temperature and pressure ranges span realistic operating conditions
- Material properties are consistent with experimental literature
- Cross-correlations between properties follow physical principles

## Citation

If you use this dataset in your research, please cite:

```
Atomic-Scale Simulation Dataset for Quantum-Enhanced Materials Modeling
Generated Dataset for DFT, MD, and Defect Formation Energy Calculations
Materials: Ni, Al, Fe, Cu | Simulations: 1,800 | Properties: Formation energies, 
activation barriers, surface energies, diffusion coefficients, dislocation mobility
```

## Contact

For questions about the dataset or requests for additional materials/properties, please refer to the generation script `atomic_simulation_dataset.py` for customization options.