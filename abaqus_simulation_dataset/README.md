# Abaqus Simulation Dataset: YSZ/GDC/LSCF Mixed-Mode Fracture Analysis

**Topic:** Calibration and Mesh Objectivity in Mixed-Mode Fracture of YSZ/GDC/LSCF Interfaces: Bridging Implicit UEL and UMAT Frameworks

## Overview

This dataset provides all material properties, geometric parameters, and validation data needed for coupled phase-field fracture / cohesive zone modeling of SOFC (Solid Oxide Fuel Cell) tri-layer half-cells. It is structured to feed directly into the governing energy functional and Abaqus UMAT/UEL subroutines.

### Governing Energy Functional

```
Π(u,d) = ∫_Ω [g(d)Ψ₀⁺(εᵉ) + Ψ₀⁻(εᵉ)] dΩ + ∫_Ω G_{c,b} γ(d,∇d) dΩ + ∫_{Γ_c} ϕ(Δ_n, Δ_t) dΓ
```

where `εᵉ = ε_total - ε_th(α,ΔT) - ε_ch(β,Δδ)`

## Dataset Structure

```
abaqus_simulation_dataset/
├── csv/                          # 21 CSV data files
│   ├── 01_*.csv                  # Section 1: Geometric & Microstructural
│   ├── 02_*.csv                  # Section 2: Thermo-Elastic (UMAT)
│   ├── 03_*.csv                  # Section 3: Chemical Expansion
│   ├── 04_*.csv                  # Section 4: Fracture & Cohesive (UEL)
│   ├── 05_*.csv                  # Section 5: Validation Data
│   └── 06_*.csv                  # Section 6: Parametric Sweeps
├── figures/                      # 20 publication-quality figures (PNG + PDF)
├── abaqus_input/                 # Generated Abaqus .inp blocks
│   ├── abaqus_material_input.inp # Complete UMAT/UEL input
│   └── material_dict.py          # Python dict for scripting
├── scripts/                      # Generation & verification scripts
│   ├── generate_dataset.py       # CSV generator
│   ├── generate_figures.py       # Figure generator (300 DPI)
│   ├── generate_material_dict.py # Abaqus .inp generator
│   ├── qa_verification.py        # Pre-simulation QA (69 checks)
│   └── package_dataset.py        # ZIP packager
├── abaqus_ysz_gdc_lscf_dataset.zip  # Download-ready ZIP
└── README.md
```

## CSV Files Summary

### Section 1: Geometric & Microstructural Data (Mesh Objectivity & RVE Fidelity)
| File | Description | Rows |
|------|-------------|------|
| `01_layer_thicknesses.csv` | YSZ/GDC/LSCF layer dimensions | 3 |
| `01_interface_roughness.csv` | Ra and wavelength for both interfaces | 4 |
| `01_porosity_data.csv` | FIB-SEM porosity (bulk, near-interface) | 5 |
| `01_mesh_sensitivity.csv` | Mesh objectivity study (6 mesh sizes) | 6 |
| `01_rve_convergence.csv` | RVE size convergence for porous LSCF | 8 |

### Section 2: Thermo-Elastic Continuum Data (UMAT Input)
| File | Description | Rows |
|------|-------------|------|
| `02_youngs_modulus.csv` | E(T) for all materials, 25-800°C | 9 |
| `02_poissons_ratio.csv` | ν(T) for all materials | 9 |
| `02_thermal_expansion.csv` | α(T) + CTE mismatch data | 9 |
| `02_umat_full_input.csv` | Complete UMAT input (E, ν, α, λ, μ, K) | 36 |

### Section 3: Defect-Chemical Expansion Data
| File | Description | Rows |
|------|-------------|------|
| `03_gdc_chemical_expansion.csv` | GDC: δ and ε_ch vs pO₂ at 600-800°C | 27 |
| `03_lscf_chemical_expansion.csv` | LSCF: anisotropic β₁₁, β₃₃ | 16 |
| `03_nonstoichiometry_profile.csv` | Δδ(z) gradient across LSCF cathode | 63 |

### Section 4: Fracture & Cohesive Zone Data (UEL Input)
| File | Description | Rows |
|------|-------------|------|
| `04_bulk_fracture_toughness.csv` | G_c(T) and K_Ic(T) for bulk materials | 9 |
| `04_cohesive_zone_parameters.csv` | Full CZM parameters (Φ_n, Φ_t, T_max, η) | 4 |
| `04_interface_toughness_vs_T.csv` | Interface G_c,I and G_c,II vs temperature | 18 |
| `04_phase_field_parameters.csv` | AT2 model parameters, l₀, degradation | 4 |
| `04_weibull_statistics.csv` | Weibull modulus m and σ₀ for all materials | 8 |

### Section 5: Experimental Validation Data
| File | Description | Rows |
|------|-------------|------|
| `05_curvature_evolution.csv` | κ(T) model vs DIC measurement | 32 |
| `05_delamination_onset.csv` | Critical T, pO₂, cycle for delamination | 6 |
| `05_crack_path_morphology.csv` | Crack path + phase-field damage overlay | 200 |

### Section 6: Parametric Sweep Configuration
| File | Description | Rows |
|------|-------------|------|
| `06_parametric_sweep_config.csv` | 6 sweep types, 32 total configurations | 32 |

## Figures (20 Total)

| Figure | Description |
|--------|-------------|
| `fig01` | Layer geometry schematic |
| `fig02` | Mesh sensitivity (energy release convergence) |
| `fig03` | RVE convergence (E_eff, α_eff) |
| `fig04` | Young's modulus E(T) |
| `fig05` | Poisson's ratio ν(T) |
| `fig06` | CTE α(T) + interface mismatch Δα |
| `fig07` | GDC chemical expansion vs pO₂ |
| `fig08` | LSCF anisotropic chemical expansion |
| `fig09` | Nonstoichiometry Δδ(z) profile |
| `fig10` | Bulk fracture toughness G_c(T) |
| `fig11` | Interface toughness G_c,I and G_c,II vs T |
| `fig12` | BK mixed-mode failure envelope |
| `fig13` | Weibull strength distributions |
| `fig14` | Curvature evolution κ(T) during cooldown |
| `fig15` | Crack path morphology with phase-field overlay |
| `fig16` | UMAT input heatmaps (E, λ, μ) |
| `fig17` | Parametric sweep computational cost |
| `fig18` | Energy functional → dataset mapping schematic |
| `fig19` | Traction-separation laws (Mode I & II) |
| `fig20` | Complete 9-panel summary dashboard |

## QA Verification

The `qa_verification.py` script performs **69 automated checks**:
- Physical bounds (E > 0, -1 < ν < 0.5, G_c > 0, α > 0)
- Thermodynamic consistency (G_c,II ≥ G_c,I, η_BK > 0)
- Positive definiteness of stiffness tensor C(T)
- Temperature monotonicity
- Mesh objectivity (l₀/h ≥ 2)
- Chemical expansion validity
- Weibull parameter ranges
- Porosity bounds
- Interface roughness consistency
- Data completeness

**Result: 68/69 passed, 1 warning** (marginal l₀/h at h=1.0 µm)

## Quick Start

```bash
# 1. Generate CSV datasets
python3 scripts/generate_dataset.py

# 2. Generate figures
python3 scripts/generate_figures.py

# 3. Generate Abaqus .inp file
python3 scripts/generate_material_dict.py

# 4. Run QA checks
python3 scripts/qa_verification.py

# 5. Package into ZIP
python3 scripts/package_dataset.py
```

## Material Systems

| Material | Composition | Role | Morphology |
|----------|------------|------|------------|
| **YSZ** | 8 mol% Y₂O₃-ZrO₂ | Electrolyte | Dense |
| **GDC** | Ce₀.₉Gd₀.₁O₂₋δ | Interlayer | Dense, nanoscale |
| **LSCF** | La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃₋δ | Cathode | Porous (φ=0.35) |

## Literature Sources

All property values are sourced from peer-reviewed Q1 journals:
- Selcuk & Atkinson, J. Eur. Ceram. Soc. (1997)
- Atkinson & Selcuk, Solid State Ionics (2000)
- Radovic & Lara-Curzio, Acta Materialia (2004)
- Bishop et al., Acta Materialia (2009)
- Zhao et al., J. Am. Ceram. Soc. (2011)
- Qu et al., Acta Materialia (2012)
- Kuhn et al., Solid State Ionics (2013)
- Wang et al., Solid State Ionics (2015)
- Chen et al., Chemistry of Materials (2015)
- Miehe et al., CMAME (2010) - Phase-field model

## Requirements

```
python >= 3.8
numpy >= 1.20
pandas >= 1.3
matplotlib >= 3.5
scipy >= 1.7
```
