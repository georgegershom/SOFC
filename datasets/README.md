# SOFC PINN Datasets

## ⚠️ SYNTHETIC DATA DISCLAIMER
**All data in this repository are synthetically generated** for research-scaffolding
and demonstration purposes. They are NOT real experimental or simulation measurements.
Physical trends follow literature-informed relationships with added Gaussian noise
(see generation script for details). Do not cite this data as experimental evidence.

---

## Research Context
"Physics-informed 3D deep learning for thermo-mechanical lifetime prediction of
short-stack reversible solid oxide cells cycled at 600–850 °C in nuclear battery service."

---

## Directory Structure

```
datasets/
├── 1_half_cell/          # Thermo-mechanical constitutive data (CTE, creep, Weibull, …)
├── 2_full_cell/          # Button-cell electrochemical & degradation data
├── 3_short_stack/        # 5-cell stack time series, TC arrays, periodic EIS
├── 4_simulation_sweep/   # Parametric multiphysics sweep inputs/outputs, 3D field snapshot
├── 5_system_level/       # HTGR power, grid price, SOC schedule, Li-ion benchmark
├── 6_literature/         # Attention-mechanism / PINN paper table
├── 7_pinn_training/      # PINN supervised, sparse experimental, inverse, micro-CT validation
├── 8_metadata/           # Dataset inventory & PINN architecture config
├── figures/              # 14 publication-quality PNG figures (300 dpi)
└── SOFC_PINN_datasets.zip  # All CSVs + figures in a single downloadable archive
```

---

## Physical Relationships Used

| Dataset | Physical law / model |
|---------|----------------------|
| CTE | Linear fit: α(T) = α₀ + k·(T–25) |
| Chemical expansion | Logarithmic: ε = A·log₁₀(pO₂) + B |
| Elastic moduli | Linear softening: E(T) = E₀·(1 – c·(T–25)) |
| Creep | Norton power law: ε̇ = A·σⁿ·exp(–Q/RT) |
| Fracture strength | Weibull distribution: m, σ₀ temperature-dependent |
| I-V curves | Butler-Volmer activation + ohmic ASR; Arrhenius ASR(T) |
| EIS | Randles-type circuit: R_ohm + R_pol/(1+jωτ) |
| Stack degradation | Power-law creep accumulation |
| HTGR power | Sinusoidal annual variation + noise |
| Grid price | Diurnal + weekly sinusoids + noise |
| Li-ion cycle life | Empirical DoD and T scaling from literature |
| PINN collocation | Arrhenius ASR, Norton damage, linear CTE for T/σ/D fields |
| Surface IR/DIC | Sinusoidal spatial distribution + cycle-dependent growth |
| Inverse fracture | CTE mismatch crack opening: exponential COD near defect |
| Micro-CT damage | Interface-concentrated damage (Gaussian decay) + edge effects |

---

## How to Regenerate

```bash
pip install -r requirements.txt
python scripts/generate_datasets.py
```

All CSVs, figures, and the zip file will be recreated deterministically (seed 42).

---

## Units

| Quantity | Unit |
|----------|------|
| Temperature | °C (unless stated) |
| CTE | ppm/K = 10⁻⁶ K⁻¹ |
| Stress | MPa |
| Strain | dimensionless |
| Current density | A/cm² |
| Voltage | V |
| Impedance | Ω·cm² |
| Fracture energy | J/m² |
| Frequency | Hz |
| Thermal power | MWth |
| Electricity price | USD/MWh |

---

## Citation (if used)
Please note this is synthetic data. If used in a publication, cite as:
> "Synthetic dataset generated for SOFC PINN research scaffolding, 2026.
>  github.com/georgegershom/SOFC. ⚠️ Not real experimental data."
