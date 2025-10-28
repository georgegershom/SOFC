# Atomic-Scale Simulation Dataset
## Quantum-Enhanced Materials Modeling

This repository contains synthetically generated atomic-scale simulation data for use in quantum-enhanced materials modeling, specifically for creep simulation and materials property prediction.

---

## 📁 Dataset Overview

The dataset includes **2,150 samples** across **6 main datasets**, covering:
- **DFT (Density Functional Theory) calculations**: Defect energies, activation barriers, surface properties
- **MD (Molecular Dynamics) simulations**: Grain boundary sliding, dislocation mobility, trajectories

### Materials Covered
Al, Cr, Cu, Fe, Inconel-718, Ni, Steel-316, Ti

---

## 📊 DFT Calculation Datasets

### 1. Defect Formation Energies (`dft_defect_formation_energies.csv`)
**500 samples** of defect formation energies calculated using DFT.

**Columns:**
- `material`: Material system
- `defect_type`: Type of defect (vacancy, interstitial, substitutional, divacancy)
- `temperature_K`: Temperature in Kelvin
- `formation_energy_eV`: Formation energy in electron volts
- `lattice_parameter_A`: Lattice parameter in Angstroms
- `defect_volume_A3`: Volume of defect in cubic Angstroms
- `charge_state`: Charge state of defect
- `convergence_tolerance`: DFT convergence tolerance
- `k_points`: K-point mesh used
- `energy_cutoff_eV`: Energy cutoff for plane waves

**Key Statistics:**
- Vacancy formation: 0.5-3.0 eV
- Interstitial formation: 2.0-6.0 eV
- Temperature range: 300-1500 K

**Use Cases:**
- Parameterize vacancy diffusion models
- Train ML models for defect property prediction
- Calculate effective diffusion coefficients

---

### 2. Grain Boundary Energies (`dft_grain_boundary_energies.csv`)
**200 samples** of grain boundary energies for various misorientations.

**Columns:**
- `material`: Material system
- `gb_type`: Grain boundary type (tilt, twist, mixed, symmetric_tilt, asymmetric_tilt)
- `misorientation_deg`: Misorientation angle in degrees
- `gb_energy_J_m2`: GB energy in J/m²
- `gb_width_nm`: Width of grain boundary in nanometers
- `grain_size_um`: Grain size in micrometers
- `segregation_energy_eV`: Solute segregation energy
- `diffusivity_enhancement`: Enhancement factor for GB diffusion

**Key Statistics:**
- GB energy range: 0.2-2.0 J/m²
- Misorientation angles: 5-60 degrees

**Use Cases:**
- Parameterize grain boundary models
- Understand cavitation and crack formation
- Model grain boundary sliding resistance

---

### 3. Activation Energy Barriers (`dft_activation_barriers.csv`)
**400 samples** of activation energy barriers for various diffusion mechanisms.

**Columns:**
- `material`: Material system
- `mechanism`: Diffusion mechanism (vacancy_migration, interstitial_migration, substitutional_diffusion, grain_boundary_diffusion, dislocation_pipe_diffusion, solute_drag)
- `temperature_K`: Temperature in Kelvin
- `activation_energy_eV`: Activation energy in eV
- `pre_exponential_factor_m2_s`: Pre-exponential factor D₀ in m²/s
- `diffusion_coefficient_m2_s`: Diffusion coefficient D in m²/s
- `attempt_frequency_THz`: Attempt frequency in THz
- `migration_path_length_A`: Migration path length in Angstroms
- `saddle_point_energy_eV`: Saddle point energy

**Key Statistics:**
- Vacancy migration: 0.5-1.5 eV
- Interstitial migration: 0.1-0.8 eV
- Substitutional diffusion: 1.0-3.0 eV

**Use Cases:**
- Calculate temperature-dependent diffusion coefficients
- Parameterize creep rate equations
- Model solute drag effects

**Key Equation:**
```
D = D₀ × exp(-Q/kT)
```
where Q is the activation energy, k is Boltzmann's constant, and T is temperature.

---

### 4. Surface Energies (`dft_surface_energies.csv`)
**300 samples** of surface energies for different crystallographic orientations.

**Columns:**
- `material`: Material system
- `surface_orientation`: Crystallographic orientation ((100), (110), (111), (112), (210), (211))
- `surface_energy_J_m2`: Surface energy in J/m²
- `temperature_K`: Temperature in Kelvin
- `work_of_adhesion_J_m2`: Work of adhesion
- `surface_stress_N_m`: Surface stress in N/m
- `atomic_density_per_A2`: Atomic density per Å²
- `surface_relaxation_percent`: Surface relaxation percentage

**Key Statistics:**
- Surface energy range: 0.5-3.0 J/m²
- (111) surfaces typically have lowest energy for FCC metals

**Use Cases:**
- Model cavity nucleation and growth
- Calculate critical cavity size
- Predict crack propagation behavior

---

## 🔬 MD Simulation Datasets

### 5. Grain Boundary Sliding (`md_grain_boundary_sliding.csv`)
**350 samples** of grain boundary sliding simulation results.

**Columns:**
- `material`: Material system
- `gb_type`: Grain boundary type
- `temperature_K`: Temperature in Kelvin
- `misorientation_deg`: Misorientation angle
- `applied_shear_stress_MPa`: Applied shear stress in MPa
- `critical_shear_stress_MPa`: Critical stress for sliding in MPa
- `sliding_displacement_nm`: Sliding displacement in nanometers
- `sliding_velocity_nm_ps`: Sliding velocity in nm/ps
- `activation_energy_eV`: Activation energy for sliding
- `simulation_time_ps`: Simulation time in picoseconds
- `box_size_nm`: Simulation box size
- `num_atoms`: Number of atoms in simulation
- `timestep_fs`: Timestep in femtoseconds

**Key Insights:**
- Critical shear stress decreases with temperature
- Sliding velocity increases with applied stress above critical value
- GB type significantly affects sliding resistance

**Use Cases:**
- Parameterize grain boundary sliding models
- Understand creep mechanisms
- Predict high-temperature deformation

---

### 6. Dislocation Mobility (`md_dislocation_mobility.csv`)
**400 samples** of dislocation mobility simulation results.

**Columns:**
- `material`: Material system
- `dislocation_type`: Type (edge, screw, mixed, prismatic, basal)
- `slip_system`: Crystallographic slip system
- `temperature_K`: Temperature in Kelvin
- `applied_stress_MPa`: Applied stress in MPa
- `peierls_stress_MPa`: Peierls stress (resistance to motion) in MPa
- `dislocation_velocity_m_s`: Dislocation velocity in m/s
- `mobility_coefficient`: Mobility coefficient M
- `stress_exponent`: Stress exponent m
- `burgers_vector_A`: Burgers vector magnitude in Angstroms
- `dislocation_density_m2`: Dislocation density in m⁻²
- `line_tension_eV_A`: Line tension in eV/Å
- `core_energy_eV_A`: Core energy in eV/Å
- `interaction_energy_eV`: Interaction energy
- `simulation_time_ps`: Simulation time
- `num_atoms`: Number of atoms

**Key Relationships:**
```
v = M × (τ - τₚ)^m
```
where:
- v = dislocation velocity
- M = mobility coefficient
- τ = applied stress
- τₚ = Peierls stress
- m = stress exponent

**Use Cases:**
- Parameterize dislocation dynamics models
- Calculate creep strain rates
- Model work hardening behavior

---

### 7. Sample MD Trajectory (`md_sample_trajectory.json`)
A sample molecular dynamics trajectory with **1000 atoms** over **100 timesteps**.

**Data Includes:**
- `timesteps_fs`: Time array in femtoseconds
- `potential_energy_eV`: Potential energy evolution
- `kinetic_energy_eV`: Kinetic energy evolution
- `total_energy_eV`: Total energy (conservation check)
- `temperature_K`: Instantaneous temperature
- `pressure_GPa`: Instantaneous pressure
- `num_atoms`: Number of atoms
- `num_steps`: Number of timesteps

**Use Cases:**
- Understand MD simulation outputs
- Extract thermodynamic properties
- Validate energy conservation

---

## 🎯 Usage Guidelines

### Integration with Continuum Models

#### 1. **Creep Rate Models**
Use activation barriers to parameterize creep equations:
```
ε̇ = A × σⁿ × exp(-Q/RT)
```
where Q comes from `dft_activation_barriers.csv`

#### 2. **Phase-Field Models**
Use grain boundary and surface energies as input parameters:
- GB energy → grain boundary mobility
- Surface energy → cavity nucleation rate

#### 3. **Crystal Plasticity Models**
Use dislocation mobility data:
- Peierls stress → yield strength
- Mobility coefficients → strain rate sensitivity

#### 4. **Machine Learning Surrogate Models**
Train ML models on this data to:
- Predict material properties at new conditions
- Accelerate multiscale simulations
- Interpolate between simulation data points

---

## 📈 Visualization

Three comprehensive plots are provided:

1. **`dft_analysis.png`**: DFT calculation results
   - Defect formation energies by type
   - GB energy vs misorientation
   - Activation barriers by mechanism
   - Surface energies by orientation

2. **`md_analysis.png`**: MD simulation results
   - GB sliding stress vs displacement
   - Sliding velocity distribution
   - Dislocation velocity vs stress
   - Peierls stress by dislocation type

3. **`trajectory_analysis.png`**: MD trajectory analysis
   - Energy evolution
   - Temperature evolution
   - Pressure evolution
   - Energy conservation check

---

## 🔧 Data Generation

The data was generated using:
- `generate_atomic_simulation_data.py`: Main data generator
- `visualize_atomic_data.py`: Visualization and analysis

To regenerate the data:
```bash
python3 generate_atomic_simulation_data.py
```

To create visualizations:
```bash
python3 visualize_atomic_data.py
```

---

## 📐 Units Reference

| Quantity | Unit | Symbol |
|----------|------|--------|
| Energy | electron volt | eV |
| Surface Energy | Joules per square meter | J/m² |
| Temperature | Kelvin | K |
| Stress | Megapascal | MPa |
| Length | Angstrom / nanometer | Å / nm |
| Time | femtosecond / picosecond | fs / ps |
| Velocity | meters per second | m/s |
| Diffusivity | square meters per second | m²/s |

---

## 🔬 Physical Context

### Defect Types
- **Vacancy**: Missing atom in crystal lattice
- **Interstitial**: Extra atom between lattice sites
- **Substitutional**: Foreign atom replacing host atom
- **Divacancy**: Two adjacent vacancies

### Diffusion Mechanisms
- **Vacancy migration**: Atom jumping into vacancy
- **Interstitial migration**: Small atom moving between sites
- **Grain boundary diffusion**: Fast diffusion along GBs
- **Dislocation pipe diffusion**: Diffusion along dislocation cores
- **Solute drag**: Solutes pinning dislocations

### Dislocation Types
- **Edge**: Extra half-plane of atoms
- **Screw**: Helical distortion around dislocation line
- **Mixed**: Combination of edge and screw character

---

## 🚀 Next Steps

1. **Data Exploration**: Review the CSV files and visualizations
2. **Model Parameterization**: Extract parameters for your continuum models
3. **ML Training**: Use this data to train surrogate models
4. **Validation**: Compare predictions with experimental data
5. **Scaling**: Extend to additional materials and conditions

---

## 📚 References

This data is inspired by typical outputs from:
- **DFT codes**: VASP, Quantum ESPRESSO, CASTEP
- **MD codes**: LAMMPS, GROMACS, HOOMD-blue
- **Analysis tools**: OVITO, ASE, pymatgen

---

## ⚖️ License & Citation

This is **synthetic data** for research and development purposes. If you use this dataset, please cite:
```
Atomic-Scale Simulation Dataset for Quantum-Enhanced Materials Modeling
Generated: 2025-10-04
Materials: Al, Cr, Cu, Fe, Inconel-718, Ni, Steel-316, Ti
```

---

## 📧 Contact

For questions about data usage or to request additional datasets, please reach out.

---

**Last Updated**: 2025-10-04
**Version**: 1.0
**Total Samples**: 2,150
**File Format**: CSV (tabular data), JSON (trajectory data)
