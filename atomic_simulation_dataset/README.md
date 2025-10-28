# Atomic-Scale Simulation Dataset

## Overview
This dataset contains computational outputs from Density Functional Theory (DFT) and Molecular Dynamics (MD) simulations for quantum-enhanced material modeling. The data provides critical parameters for larger-scale models including formation energies, activation barriers, and dynamic behavior at the atomic scale.

## Directory Structure
```
atomic_simulation_dataset/
├── dft_calculations/       # DFT calculation outputs
│   ├── defect_energies/   # Vacancy, dislocation, grain boundary energies
│   ├── activation_barriers/# Diffusion and migration barriers
│   └── surface_energies/   # Surface and interface energies
├── md_simulations/         # MD simulation trajectories and results
│   ├── grain_boundary/    # Grain boundary sliding simulations
│   └── dislocation/       # Dislocation mobility studies
├── processed_data/         # Post-processed and analysis results
├── scripts/               # Analysis and visualization scripts
└── docs/                  # Documentation and metadata
```

## Data Format
- DFT outputs: JSON files with energies (eV), structures (POSCAR format), and electronic properties
- MD outputs: Trajectory files (LAMMPS format), time series data (CSV), and statistical analyses
- All energies are in eV, distances in Angstroms, forces in eV/Angstrom

## Usage
The data can be used to:
1. Train surrogate models for multiscale simulations
2. Parameterize phase-field models
3. Inform crystal plasticity models
4. Validate continuum-scale creep models

## Citation
Please cite this dataset as: "Atomic-Scale Simulation Dataset for Quantum-Enhanced Material Modeling, 2025"