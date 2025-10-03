
Synthetic Multi-Physics FEM-Like Dataset
=======================================

This dataset contains fabricated, physically inspired time-series fields and
input descriptors resembling outputs from multi-physics FEM solvers
(thermal, mechanical, electrochemical). No PDEs were solved. Values and
units are plausible but synthetic.

Directory Layout
----------------

root: /workspace/data/num-sim-001

- dataset.json: Machine-readable schema and units
- README.md: This file
- samples/
  - sample_XXX/
    - inputs.json: Mesh, BCs, materials, time, units
    - time.csv: per-step summary (t_min, T_top, T_bottom, V_top, V_bottom)
    - fields_timeseries.npz: compressed arrays

Key Arrays in fields_timeseries.npz
-----------------------------------

- temperature_C [t, y, x] (C)
- voltage_V [t, y, x] (V)
- stress_vm_MPa [t, y, x] (MPa)
- stress_principal1_MPa [t, y, x] (MPa)
- stress_principal2_MPa [t, y, x] (MPa)
- shear_tau_xy_MPa [t, y, x] (MPa)
- strain_elastic_xx [t, y, x] (-)
- strain_elastic_yy [t, y, x] (-)
- strain_thermal_iso [t, y, x] (-)  isotropic thermal strain
- strain_plastic_eq [t, y, x] (-)  equivalent plastic strain
- strain_creep_eq [t, y, x] (-)   equivalent creep strain
- damage_D [t, y, x] (0..1)
- interfacial_tau_xy_MPa [t, x] (MPa) shear at the material interface
- delamination_init_step_by_x [x] (int) first step where delamination predicted; -1 if none
- crack_init_step_mask [t, y, x] (bool) crack initiation prediction mask

Notes
-----
- Interface at y = inputs.json:mesh.interface_y_fraction (default 0.5).
- Delamination is predicted when interfacial shear exceeds a threshold reduced by damage.
- Crack initiation is predicted when principal stress exceeds a threshold reduced by damage.

How to Load with NumPy
----------------------


```python
import numpy as np
import json
from pathlib import Path

root = Path("." ) / "samples" / "sample_000"
with open(root / "inputs.json") as f:
    meta = json.load(f)
npz = np.load(root / "fields_timeseries.npz")
print(npz["temperature_C"].shape)
```
