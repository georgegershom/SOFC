# SOFC Thermo-Mechanical Simulation Package

## Overview

This package provides a complete finite element analysis (FEA) workflow for simulating Solid Oxide Fuel Cell (SOFC) behavior under thermal cycling conditions. The simulation uses Abaqus/Standard to perform sequential thermo-mechanical analysis with damage and delamination prediction.

## Features

- **2D Plane Stress Analysis**: Cross-sectional model of SOFC repeat unit (10mm × 1mm)
- **Multi-layer Structure**: Anode, Electrolyte, Cathode, and Interconnect layers
- **Temperature-Dependent Materials**: Including plasticity and creep models
- **Multiple Heating Rates**: HR1 (1°C/min), HR4 (4°C/min), HR10 (10°C/min)
- **Damage Prediction**: Interface delamination and bulk damage metrics
- **Automated Post-Processing**: Stress analysis, strain evolution, and visualization

## Directory Structure

```
sofc_simulation/
├── inp/                    # Abaqus input files
│   └── sofc_main.inp      # Main simulation input file
├── scripts/               # Automation scripts
│   └── create_sofc_model.py  # CAE model creation script
├── post_processing/       # Analysis scripts
│   ├── damage_analysis.py    # Damage and delamination analysis
│   └── visualize_results.py  # Results visualization
├── results/              # Simulation outputs (created at runtime)
├── docs/                 # Additional documentation
├── run_simulation.py     # Main runner script
└── README.md            # This file
```

## Material Properties

### Layer Configuration
- **Anode (Ni-YSZ)**: 0.0-0.4 mm
  - Johnson-Cook plasticity
  - Norton-Bailey creep
  - E: 140→91 GPa (298→1273K)
  
- **Electrolyte (8YSZ)**: 0.4-0.5 mm
  - Norton-Bailey creep
  - E: 210→170 GPa (298→1273K)
  
- **Cathode (LSM)**: 0.5-0.9 mm
  - E: 120→84 GPa (298→1273K)
  
- **Interconnect (Ferritic Steel)**: 0.9-1.0 mm
  - E: 205→150 GPa (298→1273K)

### Critical Interface Thresholds
- Anode-Electrolyte: τ_crit = 25 MPa
- Electrolyte-Cathode: τ_crit = 20 MPa
- Cathode-Interconnect: τ_crit = 30 MPa

## Usage

### Quick Start

```bash
# Run simulation with default heating rate (HR1)
python run_simulation.py

# Run with specific heating rate
python run_simulation.py --rate HR4

# Run all heating rates
python run_simulation.py --all

# Generate input files only (no execution)
python run_simulation.py --generate-only
```

### Manual Execution in Abaqus CAE

1. **Create Model**:
   ```python
   # In Abaqus CAE Python console
   execfile('scripts/create_sofc_model.py')
   model, job = create_sofc_model(heating_rate='HR1')
   ```

2. **Submit Job**:
   ```python
   mdb.jobs[job].submit(consistencyChecking=OFF)
   mdb.jobs[job].waitForCompletion()
   ```

3. **Post-Processing**:
   ```bash
   abaqus python post_processing/damage_analysis.py Job-1.odb
   abaqus python post_processing/visualize_results.py Job-1.odb --all
   ```

### Using Input Files Directly

```bash
# Run with INP file
abaqus job=SOFC_HR1_Job input=inp/sofc_main.inp interactive

# View results in CAE
abaqus cae database=SOFC_HR1_Job.odb
```

## Analysis Steps

### Step 1: Transient Heat Transfer
- Applies thermal boundary conditions
- Bottom edge: Temperature ramp/hold/cool cycle
- Top edge: Convection (h=25 W/m²K, T∞=25°C)
- Duration varies by heating rate

### Step 2: Thermo-Mechanical Analysis
- Sequential coupling using temperature field from Step 1
- Mechanical BCs: Roller constraints (symmetric)
- NLGEOM=ON for geometric nonlinearity
- Includes creep and plasticity effects

## Output Variables

### Field Outputs
- **Thermal**: NT (temperature), HFL (heat flux)
- **Mechanical**: S (stress), E/LE (strain), MISES (von Mises)
- **Inelastic**: PEEQ (plastic strain), CEEQ (creep strain)
- **Damage**: DAMAGE_D (custom damage variable)

### History Outputs
- Interface stresses (S11, S22, S12)
- Time-dependent evolution at critical locations

## Damage Model

The damage variable D evolves according to:

```
dD/dt = k_D * max(0, (σ_vm - σ_th)/σ_th)^p * (1 + 3*w_interface)
```

Where:
- k_D = 1.5×10⁻⁵ (damage rate constant)
- σ_th = 120 MPa (threshold stress)
- p = 2 (damage exponent)
- w_interface = proximity weight to interfaces

## Post-Processing Outputs

### Damage Report
- Interface shear stress analysis
- Delamination risk assessment
- Maximum damage locations
- Electrolyte cracking prediction

### Visualizations
- Temperature evolution plots
- Stress distribution contours
- Strain evolution (plastic, creep, total)
- Interface integrity analysis

### Data Export
- Node coordinates (nodes.csv)
- Element connectivity (elements.csv)
- Final stress state (final_stress.csv)

## Convergence Tips

1. **Mesh Refinement**: Focus on interfaces (±20 μm)
2. **Time Incrementation**: Start with Δt=1s, allow automatic adjustment
3. **Stability**: Use NLGEOM for large thermal strains
4. **Creep Integration**: Ensure time units consistency (seconds)

## Validation Metrics

Compare simulation results with experimental data:
- XRD-inferred crack depths in electrolyte
- DIC strain measurements
- Post-mortem microscopy of interfaces

## Requirements

- Abaqus/Standard (2020 or later recommended)
- Python 2.7 or 3.x (for Abaqus Python)
- NumPy, Matplotlib (for post-processing)
- 4+ GB RAM for typical mesh density
- ~500 MB disk space per simulation

## Troubleshooting

### Common Issues

1. **Convergence Problems**:
   - Reduce initial time increment
   - Check material property discontinuities
   - Verify temperature-dependent data interpolation

2. **Memory Errors**:
   - Reduce mesh density
   - Use selective output requests
   - Increase system swap space

3. **Post-Processing Failures**:
   - Ensure ODB contains requested fields
   - Check Python package compatibility
   - Verify file paths are correct

## Citation

If you use this simulation package in your research, please cite:
```
[Your thesis/paper citation here]
```

## Contact

For questions or support, contact:
[Your contact information]

## License

[Specify license - e.g., MIT, GPL, proprietary]

---

## Appendix: Modifying Parameters

### Changing Materials

Edit material properties in `scripts/create_sofc_model.py`:

```python
mat_anode.Elastic(temperatureDependency=ON,
                  table=((new_E_room, new_nu, 298.0),
                        (new_E_hot, new_nu, 1273.0)))
```

### Adjusting Geometry

Modify layer thicknesses:

```python
layers = {
    'anode': (0.0, 0.00035),      # Thinner anode
    'electrolyte': (0.00035, 0.00045),  # Adjusted
    # ...
}
```

### Custom Damage Criteria

In `post_processing/damage_analysis.py`:

```python
self.tau_crit = {
    'anode_electrolyte': 30.0e6,  # Increased threshold
    # ...
}
```

## Version History

- v1.0.0: Initial release with complete thermo-mechanical workflow
- Future: Cohesive zone implementation, electrical coupling

---

*Generated for SOFC simulation package - Last updated: 2025*