## Sintering ANN/PINN Synthetic Dataset Generator

This folder contains a physics-inspired synthetic data generator for training ANN and PINN models on thin-film sintering and cooling-induced stresses.

### What it creates
- Train/Val/Test CSVs with 12,000 total samples by default
- Synthetic validation files mimicking DIC and XRD measurements
- A `manifest.json` summarizing outputs

### Run
```bash
python3 generate_dataset.py --n 12000 --outdir ./data
```

Optional flags:
- `--seed <int>`: RNG seed (default 42)
- `--no_validation`: Skip creating DIC/XRD files

### Files
- `data/train.csv`, `data/val.csv`, `data/test.csv`
- `data/validation/dic_synthetic.csv`, `data/validation/xrd_synthetic.csv`
- `data/manifest.json`

### Schema (selected columns)
- Input features
  - `sintering_temperature_c`, `cooling_rate_c_per_min`, `delta_alpha_k_inv`, `porosity_fraction`
  - `film_thickness_um`, `substrate_thickness_mm`
  - `film_youngs_modulus_gpa`, `film_poisson_ratio`, `substrate_youngs_modulus_gpa`, `substrate_poisson_ratio`
- Derived features
  - `delta_t_c`, `thermal_strain`, `film_effective_modulus_gpa`, `thermal_stress_mpa`, `gradient_factor`
- Output labels
  - `stress_hotspot_score` (0-1)
  - `crack_initiation_risk` (0-1), `crack_initiation_label` (0/1)
  - `delamination_probability` (0-1), `delamination_label` (0/1)

### Physical heuristics used
- Effective modulus vs porosity via a Gibson–Ashby-like relation
- Thermal stress from TEC mismatch under plane-stress approximation with cooling gradient factor
- Crack risk from a Griffith criterion proxy using `K_IC` and flaw size scaled by porosity
- Delamination probability from an energy release rate proxy `G ~ sigma^2 h / E` vs interface toughness

These are simplified heuristics to generate realistic trends, not validated models.

### Requirements
```bash
python3 -m pip install numpy pandas
```

