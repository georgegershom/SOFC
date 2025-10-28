# SOFC Multi-Physics Simulation

This repository contains a complete implementation of a Solid Oxide Fuel Cell (SOFC) multi-physics simulation using Abaqus/Standard. The simulation includes sequential heat transfer and thermo-mechanical analysis with temperature-dependent materials, creep, and plasticity.

## Overview

The simulation models a 2D cross-section of a single SOFC repeat unit with the following layers:
- **Anode (Ni-YSZ)**: 0.00–0.40 mm
- **Electrolyte (8YSZ)**: 0.40–0.50 mm  
- **Cathode (LSM)**: 0.50–0.90 mm
- **Interconnect (Ferritic Steel)**: 0.90–1.00 mm

## Features

- **Sequential Multi-Physics**: Heat transfer → Thermo-mechanical analysis
- **Temperature-Dependent Materials**: All properties vary with temperature
- **Advanced Material Models**: Johnson-Cook plasticity, Norton-Bailey creep
- **Multiple Heating Rates**: HR1 (1°C/min), HR4 (4°C/min), HR10 (10°C/min)
- **Damage & Delamination Proxies**: Post-processing analysis for failure prediction
- **Validation Framework**: Comparison with synthetic data

## Directory Structure

```
sofc_simulation/
├── geometry/           # Geometry files
├── materials/         # Material property files
├── mesh/             # Mesh generation scripts
├── analysis/         # Analysis setup files
├── outputs/          # Simulation results
├── scripts/          # Python automation scripts
├── validation/       # Validation results
├── sofc_hr1.inp      # Abaqus input for HR1
├── sofc_hr4.inp      # Abaqus input for HR4
├── sofc_hr10.inp     # Abaqus input for HR10
└── README.md         # This file
```

## Quick Start

### 1. Generate Models
```bash
cd /workspace/sofc_simulation
python3 scripts/generate_sofc_model.py
```

### 2. Run Simulations (requires Abaqus)
```bash
python3 scripts/run_simulation.py
```

### 3. Post-Process Results
```bash
python3 scripts/post_process.py
```

### 4. Validate Results
```bash
python3 scripts/validation.py
```

### 5. Run Complete Workflow
```bash
python3 scripts/run_complete_workflow.py
```

## Material Properties

### Ni-YSZ Anode
- **Elastic**: E = 140→91 GPa, ν = 0.30
- **CTE**: α = 12.5→13.5 ×10⁻⁶ K⁻¹
- **Thermal**: k = 6.0→4.0 W/m·K, cp = 450→570 J/kg·K
- **Plasticity**: Johnson-Cook model
- **Creep**: Norton-Bailey model

### 8YSZ Electrolyte
- **Elastic**: E = 210→170 GPa, ν = 0.28
- **CTE**: α = 10.5→11.2 ×10⁻⁶ K⁻¹
- **Thermal**: k = 2.6→2.0 W/m·K, cp = 400→600 J/kg·K
- **Creep**: Norton-Bailey model

### LSM Cathode
- **Elastic**: E = 120→84 GPa, ν = 0.30
- **CTE**: α = 11.5→12.4 ×10⁻⁶ K⁻¹
- **Thermal**: k = 2.0→1.8 W/m·K, cp = 480→610 J/kg·K

### Ferritic Steel Interconnect
- **Elastic**: E = 205→150 GPa, ν = 0.30
- **CTE**: α = 12.5→13.2 ×10⁻⁶ K⁻¹
- **Thermal**: k = 20→15 W/m·K, cp = 500→700 J/kg·K

## Analysis Steps

### Step 1: Heat Transfer
- **Type**: Transient heat transfer
- **BCs**: Prescribed temperature at bottom, film condition at top
- **Output**: Temperature field, heat flux

### Step 2: Thermo-Mechanical
- **Type**: Static general with NLGEOM
- **Predefined Field**: Temperature from Step 1
- **BCs**: Roller constraints on left and bottom edges
- **Output**: Stress, strain, plastic strain, creep strain

## Heating Rate Scenarios

| Scenario | Rate | Ramp Time | Hold Time | Cool Time | Total Time |
|----------|------|-----------|-----------|-----------|------------|
| HR1      | 1°C/min | 875 min | 10 min | 875 min | 1760 min |
| HR4      | 4°C/min | 218.75 min | 10 min | 218.75 min | 447.5 min |
| HR10     | 10°C/min | 87.5 min | 10 min | 87.5 min | 185 min |

## Damage & Delamination Analysis

### Damage Proxy
```
Ḋ = k_D [max(0, (σ_vm - σ_th)/σ_th)]^p · (1 + 3·w_iface)
```
- σ_th = 120 MPa (threshold stress)
- k_D = 1.5×10⁻⁵, p = 2
- w_iface = interface proximity weight

### Delamination Proxy
Critical shear stress thresholds:
- Anode-Electrolyte: 25 MPa
- Electrolyte-Cathode: 20 MPa  
- Cathode-Interconnect: 30 MPa

## Output Files

### Field Outputs
- **S**: Stress tensor components
- **Mises**: Von Mises stress
- **LE**: Logarithmic strain
- **PEEQ**: Equivalent plastic strain
- **CEEQ**: Equivalent creep strain
- **TEMP**: Temperature
- **HFL**: Heat flux

### History Outputs
- Interface stress components (S11, S22, S12)
- Path-wise outputs along interfaces

## Validation

The simulation results are validated against synthetic data with the following criteria:
- **Temperature**: ±5% tolerance
- **Stress**: ±30% tolerance  
- **Damage**: ±50% tolerance
- **Delamination**: ±50% tolerance

## Requirements

- **Abaqus/Standard**: For running simulations
- **Python 3.7+**: For automation scripts
- **NumPy**: For numerical computations
- **Matplotlib**: For visualization

## Troubleshooting

### Common Issues

1. **Abaqus not found**: Ensure Abaqus is installed and in PATH
2. **Memory issues**: Reduce mesh density or use automatic incrementation
3. **Convergence problems**: Check material properties and boundary conditions
4. **Long runtimes**: Use appropriate time incrementation settings

### Performance Tips

- Use automatic time incrementation for heat transfer
- Enable NLGEOM for mechanical step
- Refine mesh only at interfaces
- Use appropriate element types (DC2D4 for heat, CPS4 for mechanics)

## References

This simulation follows the methodology described in:
- Multi-physics SOFC modeling literature
- Abaqus/Standard documentation
- Material property databases for SOFC components

## License

This project is provided for educational and research purposes.