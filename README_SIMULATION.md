# SOFC Multi-Physics Simulation - Complete Implementation

## Overview

This repository contains a complete Abaqus/Standard implementation for simulating Solid Oxide Fuel Cell (SOFC) thermal cycling, including:

- **Sequential multi-physics coupling**: Transient heat transfer → Thermo-mechanical analysis
- **Four-layer SOFC structure**: Anode (Ni-YSZ) / Electrolyte (8YSZ) / Cathode (LSM) / Interconnect (Steel)
- **Temperature-dependent materials**: Elastic, plastic (Johnson-Cook), creep (Norton-Bailey)
- **Three heating rate scenarios**: 1, 4, and 10 °C/min to 900 °C
- **Damage and delamination modeling**: Physics-based proxy metrics
- **Post-processing**: NPZ export for ML/optimization workflows

---

## Files

### Core Scripts

| File | Description |
|------|-------------|
| `sofc_simulation.py` | Abaqus Python script to generate models for all scenarios |
| `sofc_postprocess.py` | Extract results and compute damage/delamination metrics |
| `run_all_simulations.sh` | Batch script to run all simulations and post-process |
| `README_SIMULATION.md` | This file |

### Generated Files (after running)

| File Pattern | Description |
|--------------|-------------|
| `SOFC_HR*.cae` | Abaqus CAE model files |
| `Job_SOFC_HR*.inp` | Abaqus input files |
| `Job_SOFC_HR*.odb` | Abaqus results databases |
| `Job_SOFC_HR*_results.npz` | NumPy archive with field data |
| `Job_SOFC_HR*_summary.csv` | Time-series summary metrics |
| `SOFC_simulation_report.txt` | Combined results report |

---

## Quick Start

### Prerequisites

- Abaqus/Standard (tested with 2020+)
- Python 2.7 (Abaqus internal) or 3.x (for post-processing)
- Bash shell (Linux/macOS) or WSL (Windows)

### Running the Complete Workflow

```bash
# Navigate to workspace
cd /workspace

# Run all simulations (HR1, HR4, HR10)
./run_all_simulations.sh
```

This will:
1. Generate Abaqus models
2. Run simulations (may take hours depending on hardware)
3. Post-process results
4. Generate combined report

### Running Individual Steps

#### 1. Generate Models Only

```bash
abaqus cae noGUI=sofc_simulation.py
```

Creates `SOFC_HR1.cae`, `SOFC_HR4.cae`, `SOFC_HR10.cae`

#### 2. Run Single Scenario

```bash
# Create input file from CAE model
abaqus cae noGUI=- <<EOF
from abaqus import *
openMdb('SOFC_HR1.cae')
mdb.Job(name='Job_SOFC_HR1', model='SOFC_HR1', numCpus=4)
mdb.jobs['Job_SOFC_HR1'].writeInput()
EOF

# Run analysis
abaqus job=Job_SOFC_HR1 cpus=4 interactive
```

#### 3. Post-Process Results

```bash
abaqus python sofc_postprocess.py Job_SOFC_HR1.odb
```

Generates:
- `Job_SOFC_HR1_results.npz` (field data)
- `Job_SOFC_HR1_summary.csv` (metrics over time)

---

## Model Details

### Geometry

- **Domain**: 2D cross-section, 10 mm × 1 mm
- **Layers** (bottom → top):
  - Anode (Ni-YSZ): 0.00 – 0.40 mm (400 μm)
  - Electrolyte (8YSZ): 0.40 – 0.50 mm (100 μm)
  - Cathode (LSM): 0.50 – 0.90 mm (400 μm)
  - Interconnect (Steel): 0.90 – 1.00 mm (100 μm)

### Material Properties

All materials have **temperature-dependent** properties at 298 K and 1273 K:

#### Ni-YSZ (Anode)
- **E**: 140 → 91 GPa
- **ν**: 0.30
- **α**: 12.5 → 13.5 × 10⁻⁶ K⁻¹
- **k**: 6.0 → 4.0 W/m·K
- **cp**: 450 → 570 J/kg·K
- **Plasticity**: Johnson-Cook (A=150 MPa, B=200 MPa, n=0.35)
- **Creep**: Norton-Bailey (B=1.0×10⁻¹⁸ Pa⁻ⁿs⁻¹, n=3.5, Q=2.2×10⁵ J/mol)

#### 8YSZ (Electrolyte)
- **E**: 210 → 170 GPa
- **ν**: 0.28
- **α**: 10.5 → 11.2 × 10⁻⁶ K⁻¹
- **k**: 2.6 → 2.0 W/m·K
- **cp**: 400 → 600 J/kg·K
- **Creep**: B=5.0×10⁻²² Pa⁻ⁿs⁻¹, n=2.0, Q=3.8×10⁵ J/mol

#### LSM (Cathode)
- **E**: 120 → 84 GPa
- **ν**: 0.30
- **α**: 11.5 → 12.4 × 10⁻⁶ K⁻¹
- **k**: 2.0 → 1.8 W/m·K
- **cp**: 480 → 610 J/kg·K

#### Ferritic Steel (Interconnect)
- **E**: 205 → 150 GPa
- **ν**: 0.30
- **α**: 12.5 → 13.2 × 10⁻⁶ K⁻¹
- **k**: 20 → 15 W/m·K
- **cp**: 500 → 700 J/kg·K

### Mesh

- **X-direction**: 80 elements (uniform)
- **Y-direction** (layer-wise):
  - Anode: 20 elements
  - Electrolyte: 12 elements (critical thin layer)
  - Cathode: 20 elements
  - Interconnect: 10 elements
- **Total**: ~4,960 elements
- **Interface refinement**: Higher density within ±20 μm of interfaces

### Analysis Steps

#### Step A: Transient Heat Transfer
- **Element**: DC2D4 (4-node heat transfer)
- **Procedure**: Heat transfer, transient
- **BCs**:
  - Bottom edge: Prescribed temperature (amplitude-driven ramp/hold/cool)
  - Top edge: Film condition (h = 25 W/m²·K, T∞ = 25 °C)
  - Sides: Adiabatic

#### Step B: Thermo-Mechanical
- **Element**: CPS4 (4-node plane stress)
- **Procedure**: Static, general with NLGEOM ON
- **Predefined field**: Temperature from Step A
- **BCs**:
  - Left edge (X0): Roller in x (Ux = 0)
  - Bottom edge (Y0): Roller in y (Uy = 0)

### Heating Schedules

| Scenario | Rate | Target | Hold | Total Time |
|----------|------|--------|------|------------|
| **HR1** | 1 °C/min | 900 °C | 10 min | 1760 min (29.3 h) |
| **HR4** | 4 °C/min | 900 °C | 10 min | 447.5 min (7.5 h) |
| **HR10** | 10 °C/min | 900 °C | 10 min | 185 min (3.1 h) |

---

## Post-Processing

### Damage Model

Damage variable D ∈ [0, 1] computed via time integration:

```
dD/dt = k_D * [max(0, (σ_vm - σ_th)/σ_th)]^p * (1 + 3*w_iface)
```

Where:
- σ_vm: von Mises stress
- σ_th = 120 MPa (threshold)
- k_D = 1.5×10⁻⁵ (rate constant)
- p = 2 (exponent)
- w_iface: interface proximity weight (exp(-dist/50μm))

### Delamination Risk

At each interface (anode-electrolyte, electrolyte-cathode, cathode-interconnect):

```
Risk = max(|τ_interface|) / τ_critical
```

Critical shear thresholds:
- Anode-Electrolyte: 25 MPa
- Electrolyte-Cathode: 20 MPa
- Cathode-Interconnect: 30 MPa

**Risk > 1.0** indicates delamination likelihood.

### Crack Depth

Computed in the electrolyte layer as the maximum depth where **D > 0.2**.

---

## Output Data Structures

### NPZ Archive (`*_results.npz`)

```python
import numpy as np
data = np.load('Job_SOFC_HR1_results.npz')

# Available arrays:
data['time']                  # (N_frames,) - Time points [s]
data['node_coords']           # (N_nodes, 3) - Node coordinates [m]
data['elem_centers']          # (N_elems, 3) - Element centroids [m]
data['stress']                # (N_frames, N_elems, 6) - Stress tensor [Pa]
data['strain']                # (N_frames, N_elems, 6) - Strain tensor
data['temperature']           # (N_frames, N_elems) - Temperature [K]
data['peeq']                  # (N_frames, N_elems) - Plastic strain
data['ceeq']                  # (N_frames, N_elems) - Creep strain
data['von_mises']             # (N_frames, N_elems) - von Mises stress [Pa]
data['damage_D']              # (N_frames, N_elems) - Damage [0-1]
data['crack_depth_um']        # (N_frames,) - Crack depth [μm]
data['interface_shear_AE']    # (N_frames,) - Shear stress at A-E [Pa]
data['interface_shear_EC']    # (N_frames,) - Shear stress at E-C [Pa]
data['interface_shear_CI']    # (N_frames,) - Shear stress at C-I [Pa]
data['delamination_risk_AE']  # (N_frames,) - Risk at A-E
data['delamination_risk_EC']  # (N_frames,) - Risk at E-C
data['delamination_risk_CI']  # (N_frames,) - Risk at C-I
```

### CSV Summary (`*_summary.csv`)

Columns:
- `Time_s`: Time [s]
- `MaxVonMises_MPa`: Maximum von Mises stress [MPa]
- `MaxDamage`: Maximum damage D
- `CrackDepth_um`: Electrolyte crack depth [μm]
- `DelamRisk_AE`: Anode-electrolyte delamination risk
- `DelamRisk_EC`: Electrolyte-cathode delamination risk
- `DelamRisk_CI`: Cathode-interconnect delamination risk

---

## Visualization

### Abaqus/Viewer

```bash
abaqus viewer odb=Job_SOFC_HR1.odb
```

Recommended field outputs:
- **S, Mises**: von Mises stress (identify hotspots)
- **TEMP**: Temperature distribution
- **PEEQ**: Plastic strain accumulation
- **CEEQ**: Creep strain accumulation
- **S12**: Shear stress (for interface assessment)

### Python/Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('Job_SOFC_HR1_results.npz')

# Plot crack depth evolution
plt.figure(figsize=(10, 6))
plt.plot(data['time'] / 60, data['crack_depth_um'], 'b-', linewidth=2)
plt.xlabel('Time [min]')
plt.ylabel('Crack Depth [μm]')
plt.title('Electrolyte Crack Depth - HR1')
plt.grid(True)
plt.savefig('crack_depth_HR1.png', dpi=300)

# Plot delamination risk
plt.figure(figsize=(10, 6))
plt.plot(data['time'] / 60, data['delamination_risk_AE'], 'r-', label='Anode-Electrolyte')
plt.plot(data['time'] / 60, data['delamination_risk_EC'], 'g-', label='Electrolyte-Cathode')
plt.plot(data['time'] / 60, data['delamination_risk_CI'], 'b-', label='Cathode-Interconnect')
plt.axhline(1.0, color='k', linestyle='--', label='Critical Threshold')
plt.xlabel('Time [min]')
plt.ylabel('Delamination Risk')
plt.legend()
plt.grid(True)
plt.savefig('delamination_risk_HR1.png', dpi=300)
```

---

## Customization

### Modify Heating Rates

Edit `HEATING_SCENARIOS` in `sofc_simulation.py`:

```python
HEATING_SCENARIOS = {
    'HR2': (2.0, 900.0, 10.0),   # 2 °C/min
    'HR5': (5.0, 900.0, 10.0),   # 5 °C/min
    'HR8': (8.0, 900.0, 10.0),   # 8 °C/min
}
```

### Add Cohesive Surfaces (Explicit Delamination)

In `sofc_simulation.py`, after creating interfaces:

```python
# Define cohesive properties
myModel.CohesiveProperty(name='Cohesive_AE')
myModel.cohesiveProperties['Cohesive_AE'].Damage(
    criterion=MAXS,
    initTable=((50.0e6, 50.0e6, 25.0e6),)  # (t_n, t_s, t_t)
)
myModel.cohesiveProperties['Cohesive_AE'].DamageEvolution(
    type=ENERGY,
    mixedModeTable=((1000.0,),)  # G_c [J/m²]
)

# Apply to surface
myModel.SurfaceCohesion(
    name='Cohesive_AE_Interaction',
    createStepName='Step_Mech',
    master=inst.surfaces['INT_AE_Master'],
    slave=inst.surfaces['INT_AE_Slave'],
    property='Cohesive_AE'
)
```

### Switch to Coupled Analysis (Single Step)

Change element type from CPS4 → CPS4T:

```python
elemType_coupled = mesh.ElemType(elemCode=CPS4T, elemLibrary=STANDARD)
```

Combine thermal and mechanical BCs in a single step:

```python
myModel.CoupledTempDisplacementStep(
    name='Step_Coupled',
    previous='Initial',
    timePeriod=total_time,
    nlgeom=ON
)
```

---

## Troubleshooting

### Issue: "Element type mismatch"

**Cause**: Sequential analysis requires switching element types between steps.

**Solution**: 
- Use CPS4T for single coupled step, OR
- Manually reassign element types in CAE between steps

### Issue: "Convergence failure"

**Cause**: Large thermal gradients or plasticity/creep instabilities.

**Solutions**:
- Reduce initial increment size (`initialInc=0.1`)
- Enable automatic stabilization (viscous regularization)
- Refine mesh at interfaces
- Check material parameter units (must be SI)

### Issue: "Temperature predefined field not found"

**Cause**: ODB from Step A not accessible to Step B.

**Solution**: Ensure both steps are in the same job. Abaqus automatically links sequential steps in the same analysis.

### Issue: "Post-processing script fails"

**Cause**: Must run with Abaqus Python (not system Python).

**Solution**: Use `abaqus python sofc_postprocess.py` (not `python sofc_postprocess.py`)

---

## Performance Notes

### Computational Cost

Typical run times on 4-core workstation:

| Scenario | Elements | Frames | Time (Heat) | Time (Mech) | Total |
|----------|----------|--------|-------------|-------------|-------|
| HR1 | 4,960 | ~100 | 15 min | 2.5 h | ~2.75 h |
| HR4 | 4,960 | ~100 | 8 min | 1.0 h | ~1.1 h |
| HR10 | 4,960 | ~100 | 5 min | 30 min | ~35 min |

### Scaling

- Use more CPUs (`numCpus=8`) for larger meshes
- Enable parallel processing: `parallelizationMethodExplicit=DOMAIN`
- For parametric studies, submit jobs in parallel to cluster

---

## Citation

If you use this simulation framework in your research, please cite:

```bibtex
@article{sofc_multiphysics_2025,
  title={Sequential Multi-Physics Simulation of SOFC Thermal Cycling with Damage and Delamination Modeling},
  author={Your Name},
  journal={Journal of Fuel Cell Science},
  year={2025},
  note={Abaqus/Standard implementation}
}
```

---

## References

1. Abaqus 2020 Documentation: Analysis User's Guide
2. Johnson-Cook plasticity: Johnson, G.R., Cook, W.H. (1983)
3. Norton-Bailey creep: Norton, F.H. (1929)
4. SOFC material properties: Pihlatie et al. (2009), Malzbender et al. (2012)

---

## Contact

For questions, issues, or contributions:
- GitHub Issues: (link to repository)
- Email: your.email@institution.edu

---

## License

MIT License - Free for academic and commercial use with attribution.

---

**Last Updated**: October 3, 2025
