Numerical Simulation Dataset (Synthetic FEM)

Structure under `data/simulations/sim_XXX/`:

- `mesh.json`: Mesh configuration metadata
- `nodes.csv`: Node list with columns: `node_id,x,y`
- `elements.csv`: Element connectivity. For `quad4`: `elem_id,n1,n2,n3,n4`; for `tri3`: `elem_id,n1,n2,n3`
- `boundary_conditions.json`: Temperature/displacement/voltage boundary values
- `material_models.json`: Elastic/plastic/creep/thermal/electrochemical parameters
- `thermal_profile.csv`: Transient times with `time_s,temperature_c,voltage_v`
- `snapshot_tXX/*.csv`: Per-snapshot fields with `id,value`
  - `temperature_node.csv`, `voltage_node.csv`: Nodal fields
  - `stress_vm_elem.csv`, `stress_p1_elem.csv`, `shear_interface_elem.csv`: Element stresses
  - `strain_el_elem.csv`, `strain_pl_elem.csv`, `strain_cr_elem.csv`, `strain_th_elem.csv`: Element strains
  - `damage_elem.csv`: Damage variable D in [0,1)
  - `delamination_elem.csv`, `crack_init_elem.csv`: Binary predictions per element (0/1)
- `metadata.json`: Random seed and notes
- `summary.json`: Basic statistics per field per snapshot

Notes
- Data are fabricated to be physically plausible but not solver-accurate.
- Heating/cooling rates targeted: 1–10 °C/min across the transient.
- Interface effects appear near `y ≈ 0.5` via elevated interfacial shear and damage.
