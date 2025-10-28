# Atomic-Scale Simulation Dataset

This dataset contains computationally generated atomic-scale simulation data for materials science applications, particularly focused on quantum-enhanced inputs for larger-scale models.

## Dataset Overview

The dataset is organized into two main categories:

### 1. DFT Calculations (`dft_calculations/`)
Contains results from Density Functional Theory calculations providing quantum-mechanically accurate properties.

#### Files:
- **`formation_energies_defects.json`**: Formation energies for various defects including vacancies, dislocations, and grain boundaries
- **`activation_barriers_diffusion.json`**: Activation energy barriers for key diffusion processes
- **`surface_energies.json`**: Surface energies for cavitation and crack formation studies

### 2. MD Simulations (`md_simulations/`)
Contains results from Molecular Dynamics simulations providing dynamic atomic-scale behavior.

#### Files:
- **`grain_boundary_sliding.xyz`**: Trajectory data for grain boundary sliding simulations
- **`dislocation_mobility.json`**: Comprehensive data on dislocation mobility and interactions

## Data Types and Applications

### DFT Calculations
**Formation Energies of Defects:**
- Vacancies in different lattice sites and materials
- Dislocation core energies and formation energies per length
- Grain boundary energies for different misorientation angles
- Used to parameterize diffusion rates and mechanical properties

**Activation Energy Barriers:**
- Vacancy migration paths and energies
- Solute drag effects on dislocation motion
- Grain boundary diffusion mechanisms
- Pipe diffusion along dislocation cores
- Critical for understanding creep and deformation mechanisms

**Surface Energies:**
- Free surface energies for different crystallographic orientations
- Cavitation surface energies as function of void size/shape
- Crack surface energies and theoretical strength
- Stacking fault energies for deformation twinning
- Essential for fracture and cavitation modeling

### MD Simulations
**Grain Boundary Sliding:**
- Atomic trajectories during shear deformation
- Sliding velocities and mechanisms
- Stress-strain behavior at atomic scale
- Temperature and strain rate effects
- Used to parameterize grain boundary constitutive models

**Dislocation Mobility:**
- Dislocation velocities vs. stress and temperature
- Interaction mechanisms (junctions, cross-slip, pinning)
- Creep mechanisms (climb vs glide)
- Trajectory data for mobility modeling
- Critical for crystal plasticity modeling

## File Formats

### JSON Format (DFT calculations)
All DFT calculation files use JSON format with the following structure:
```json
{
  "metadata": {
    "description": "Brief description of the data",
    "units": "Primary units used",
    "method": "Computational method details",
    "convergence_threshold": "Convergence criteria"
  },
  "data": {
    // Specific data organized by category
  },
  "computational_details": {
    "software": "VASP/Quantum ESPRESSO/etc",
    "parameters": "Key computational parameters"
  }
}
```

### XYZ Format (MD trajectories)
Grain boundary sliding data uses extended XYZ format:
```
# Frame N: Description
NUM_ATOMS
Lattice="FCC" Properties=species:S:1:pos:R:3:force:R:3 Time=TIME
ELEMENT x y z fx fy fz
ELEMENT x y z fx fy fz
...
```

## Usage Examples

### Training Surrogate Models
The formation energies and activation barriers can be used to train machine learning models that predict defect properties without expensive DFT calculations.

### Parameterizing Phase-Field Models
Surface energies and grain boundary data can parameterize phase-field models for microstructure evolution.

### Crystal Plasticity Modeling
Dislocation mobility data and interaction parameters can be used in crystal plasticity finite element models.

## Computational Methods

### DFT Calculations
- **Software**: VASP 5.4.4
- **Exchange-correlation**: PBE functional
- **Pseudopotentials**: PAW method
- **k-point sampling**: Monkhorst-Pack grids
- **Convergence criteria**: 1e-6 eV for energies, 1e-8 eV for surfaces

### MD Simulations
- **Software**: LAMMPS
- **Potential**: Embedded Atom Method (EAM) for Al
- **Ensemble**: NPT for constant temperature/pressure
- **Integration**: Velocity Verlet algorithm
- **Timestep**: 1-2 fs for atomic vibrations

## Applications in Materials Science

This dataset enables:
1. **Quantum-enhanced constitutive modeling**: Using DFT data to improve empirical potentials
2. **Multi-scale modeling**: Bridging atomic-scale physics to continuum models
3. **Machine learning acceleration**: Training surrogate models for expensive calculations
4. **Defect engineering**: Understanding how defects control material properties
5. **Creep and fatigue modeling**: Parameterizing deformation mechanisms

## Citation

If you use this dataset, please cite:
```
Atomic-Scale Simulation Dataset for Quantum-Enhanced Materials Modeling
Generated using DFT and MD simulations with VASP and LAMMPS
[Your Institution], 2024
```

## Notes

- All data is computationally generated and represents typical values for aluminum and similar FCC metals
- Real experimental validation would be required for quantitative predictions
- The dataset can be extended with additional elements, temperatures, or deformation conditions
- Computational parameters are provided for reproducibility