# PINN Training, Validation & Inverse Datasets

## ⚠️ SYNTHETIC DATA DISCLAIMER
All data in this directory are **synthetically generated** for research-scaffolding
purposes. They are NOT real experimental or simulation measurements.

## Files

| File | Description | Points |
|------|-------------|--------|
| supervised_collocation_points.csv | (x,y,z,t,conditions) → (T, σ_ij, D) training data | 50,000 (representative subset of 10⁸+) |
| sparse_surface_ir_dic.csv | IR temperature + DIC strain on cell top surface | 200,000 (10k pts × 20 timesteps) |
| sparse_thermocouple_timeseries.csv | 30 TC sensors time-resolved through 100 timesteps | 3,000 |
| inverse_dic_displacement_fields.csv | DIC displacement during thermal ramp (defect identification) | 100,000 (5k pts × 20 T-steps) |
| inverse_true_parameters.csv | Ground-truth fracture properties for inverse validation | 3 parameters |
| microct_3d_damage_map.csv | Synthetic voxelised 3D damage field (40×40×50 grid) | 80,000 voxels |
| validation_metrics_summary.csv | IoU, crack statistics comparing PINN vs micro-CT | 6 metrics |
| attention_pinn_papers_extended.csv | Additional attention/PINN papers (2021–2024) | 8 papers |

## Physical Models Used
- **Supervised data**: Arrhenius-ASR, Norton creep damage accumulation, linear thermal expansion
- **DIC/IR surface**: sinusoidal spatial temperature and strain distributions with cycle-dependent growth
- **Inverse problem**: CTE-mismatch–driven crack opening (exponential COD near defect)
- **Micro-CT damage**: interface-concentrated damage + edge effects (Gaussian decay)

## Usage in PINN Training
1. `supervised_collocation_points.csv` → Data loss (MSE on FE-predicted fields)
2. `sparse_surface_ir_dic.csv` + `sparse_thermocouple_timeseries.csv` → Boundary condition + validation loss
3. `inverse_dic_displacement_fields.csv` → Inverse identification of Gc, σ_c
4. `microct_3d_damage_map.csv` → End-of-life validation (IoU of predicted vs measured damage)
