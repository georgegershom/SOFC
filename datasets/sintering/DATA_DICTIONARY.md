## Sintering Process–Microstructure Dataset (Synthetic)

This dataset links sintering process parameters (inputs) to resulting microstructure metrics (outputs). Values are synthetic but follow plausible correlations for pressureless/pressure-assisted sintering of ceramic-like powders.

- Source: Programmatically generated
- Rows: 300 (default)
- Files:
  - `sintering_process_microstructure.csv`
  - `sintering_process_microstructure.json`
  - `sintering_process_microstructure_meta.json`

### Columns

| Field | Role | Type | Unit | Description | Typical Range |
|---|---|---|---|---|---|
| `sample_id` | ID | string | - | Unique sample identifier | - |
| `peak_temp_c` | Input | float | °C | Peak sintering temperature | 1100–1600 |
| `ramp_rate_c_per_min` | Input | float | °C/min | Heating ramp rate to peak | 2–20 |
| `hold_time_min` | Input | int | min | Hold time at peak temperature | 0–180 |
| `cool_rate_c_per_min` | Input | float | °C/min | Cooling rate after hold | 2–20 |
| `pressure_mpa` | Input | float | MPa | Applied uniaxial pressure; 0 for pressureless | 0–50 |
| `atmosphere` | Input | category | - | Furnace atmosphere (`air`, `argon`, `nitrogen`, `vacuum`, `hydrogen`) | - |
| `green_rel_density` | Input | float | fraction | Relative density of the green body | 0.50–0.65 |
| `green_porosity_pct` | Input | float | % | Green porosity (100 × (1 − density)) | 35–50 |
| `powder_particle_size_d50_um` | Input | float | µm | Powder particle median size (D50) | 0.4–6 |
| `green_pore_size_d50_um` | Input | float | µm | Green pore median size (D50) | 0.3–10 |
| `grain_size_d10_um` | Output | float | µm | Grain size 10th percentile (SEM-derived) | 0.3–20 |
| `grain_size_d50_um` | Output | float | µm | Grain size median (D50) | 0.5–30 |
| `grain_size_d90_um` | Output | float | µm | Grain size 90th percentile | 1–60 |
| `pore_size_d10_um` | Output | float | µm | Pore size 10th percentile (3D/SEM) | 0.03–10 |
| `pore_size_d50_um` | Output | float | µm | Pore size median (D50) | 0.05–15 |
| `pore_size_d90_um` | Output | float | µm | Pore size 90th percentile | 0.1–25 |
| `gb_high_angle_fraction` | Output | float | fraction | Fraction of high-angle grain boundaries | 0.30–0.90 |
| `gb_mean_misorientation_deg` | Output | float | ° | Mean grain boundary misorientation | 10–60 |
| `relative_density_final` | Output | float | fraction | Final relative density after sintering | 0.80–0.99 |
| `final_porosity_pct` | Output | float | % | Final porosity (100 × (1 − density)) | 1–20 |

### Intended Correlations (Heuristic)

- Peak temperature, longer hold, higher initial density, vacuum/hydrogen atmospheres, and applied pressure all increase densification (higher `relative_density_final`, lower `final_porosity_pct`).
- Higher peak temperature, longer hold, slower ramp/cool, and H₂/vacuum tend to increase grain growth (`grain_size_*`). Pressure modestly suppresses growth/spread.
- Pore sizes shrink with densification; distributions narrow as density increases.
- High-angle boundary fraction increases with temperature/hold and decreases with porosity; hydrogen/vacuum modestly increase it.

### Measurement Mapping (How you'd collect this experimentally)

- SEM: Grain size distribution (`grain_size_d10_um`, `grain_size_d50_um`, `grain_size_d90_um`).
- Archimedes: Bulk/relative density (`relative_density_final`, `final_porosity_pct`).
- X-ray CT (or FIB-SEM): Pore size distribution (`pore_size_*`).

### Notes

- Synthetic dataset; not tied to a specific material system. Use for prototyping models and pipelines only. Calibrate against real data for decision-making.

