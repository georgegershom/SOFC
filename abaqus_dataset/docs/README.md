# Abaqus SOFC Simulation Dataset: YSZ/GDC/LSCF Interface Fracture

## Overview

This dataset is designed to support phase-field and cohesive zone modeling of mixed-mode fracture in Solid Oxide Fuel Cell (SOFC) tri-layer systems consisting of:
- **8YSZ** (8 mol% Yttria-Stabilized Zirconia) - Dense electrolyte
- **GDC10** (Gd₀.₁Ce₀.₉O₁.₉₅) - Barrier interlayer
- **LSCF** (La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃₋δ) - Porous cathode

## Dataset Structure

```
abaqus_dataset/
├── data/
│   ├── master_material_database.csv          # Comprehensive material database
│   ├── geometric_parameters.csv               # Layer thicknesses, roughness, porosity
│   ├── thermoelastic_properties.csv           # E, ν, α vs. Temperature
│   ├── chemical_expansion_data.csv            # β, Δδ vs. pO₂
│   ├── fracture_cohesive_properties.csv       # Gc, Tmax, η parameters
│   └── experimental_validation_data.csv       # Curvature, delamination measurements
├── scripts/
│   ├── generate_abaqus_input.py               # CSV → Abaqus .inp converter
│   ├── verify_parameters.py                   # QA checks for physical bounds
│   └── visualize_properties.py                # Generate property vs. T plots
├── figures/
│   └── [Generated PNG files]
└── docs/
    └── README.md                               # This file
```

## Governing Physics

The dataset feeds directly into the energy functional:

**Π(u,d) = ∫_Ω [g(d)Ψ₀⁺(εᵉ) + Ψ₀⁻(εᵉ)]dΩ + ∫_Ω Gc,b γ(d,∇d)dΩ + ∫_Γc φ(Δn,Δt)dΓ**

Where:
- **εᵉ = ε_total - εᵗʰ(α, ΔT) - εᶜʰ(β, Δδ)** (elastic strain)
- **g(d) = (1-d)²** (quadratic degradation for AT2 model)
- **φ(Δn,Δt)** is the Xu-Needleman or BK mixed-mode cohesive potential

## Data Categories

### 1. Geometric & Microstructural Data
**Purpose**: Construct explicit 2D/3D meshes and study mesh objectivity.

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| YSZ Thickness | t_YSZ | 10 μm | Dense electrolyte |
| GDC Thickness | t_GDC | 5 μm | Barrier layer |
| LSCF Thickness | t_LSCF | 30 μm | Porous cathode |
| YSZ/GDC Roughness | Ra₁ | 0.15 μm | RMS from AFM |
| GDC/LSCF Roughness | Ra₂ | 0.35 μm | Higher due to porosity |
| LSCF Porosity | φ_pore | 0.35 | Volume fraction |

### 2. Thermo-Elastic Continuum Data
**Purpose**: Define C(T) and εᵗʰ for thermal cooldown (800°C → RT).

| Material | E (RT) | E (800°C) | ν | α (ppm/K) |
|----------|--------|-----------|---|-----------|
| 8YSZ | 210 GPa | 190 GPa | 0.31 | 10.5 |
| GDC10 | 160 GPa | 145 GPa | 0.30 | 12.5 |
| LSCF | 80 GPa | 65 GPa | 0.28 | 15.8 |

**CTE Mismatch**: Δα = 5.3 ppm/K (LSCF-YSZ) drives thermal stress.

### 3. Defect-Chemical Expansion Data
**Purpose**: Define εᶜʰ, the operational eigenstrain that shifts mode mixity.

| Material | β_iso (GDC) | β₁₁ (LSCF) | β₃₃ (LSCF) | Δδ |
|----------|-------------|------------|------------|-----|
| GDC10 | 0.025 | - | - | 0.05 |
| LSCF | - | 0.08 | 0.12 | 0.15 |

**Key Insight**: Anisotropic expansion in LSCF (β₃₃ > β₁₁) couples with texture to create shear at GDC/LSCF interface.

### 4. Fracture & Cohesive Zone Data
**Purpose**: Feed phase-field (Gc,b) and interface laws (φn, φt).

| Property | YSZ/GDC | GDC/LSCF | Notes |
|----------|---------|----------|-------|
| Gc,I | 12 J/m² | 8 J/m² | Mode I toughness |
| Gc,II | 18 J/m² | 14 J/m² | Mode II toughness |
| T_max,n | 150 MPa | 100 MPa | Normal cohesive strength |
| T_max,t | 180 MPa | 130 MPa | Shear cohesive strength |
| η_BK | 2.1 | 2.1 | Mixed-mode exponent |

**Failure Hierarchy**: GDC/LSCF (weakest) < YSZ/GDC < Bulk phases

### 5. Experimental Validation Data
**Purpose**: Target functions for model verification.

| Metric | Value | Method |
|--------|-------|--------|
| κ(800°C) | 0.0015 mm⁻¹ | DIC |
| κ(RT) | 0.045 mm⁻¹ | DIC |
| T_delam | 350°C | In-situ SEM |
| L_delam | 2.5 mm | Post-mortem |

**Validation Strategy**: 
1. Global curvature evolution → validates ε^th + ε^ch fields
2. Crack path morphology → validates phase-field damage contour
3. Delamination onset → validates mixed-mode failure criterion

## Data Quality Flags

| Flag | Meaning | Recommended Action |
|------|---------|-------------------|
| **HIGH** | Q1 literature / Direct measurement | Use as-is |
| **MEDIUM** | Reasonable estimate / Indirect measurement | Acceptable with uncertainty quantification |
| **LOW** | Rough estimate / Derived quantity | Requires sensitivity analysis |
| **FLAG** | Assumed value / Needs experimental validation | Priority for future testing |

## Usage Workflow

### Step 1: Populate Missing Data
```bash
# Review master_material_database.csv
# Update "MISSING" entries with measured/literature values
# Update "Quality_Flag" based on source reliability
```

### Step 2: Generate Abaqus Input
```bash
python scripts/generate_abaqus_input.py --input data/master_material_database.csv --output sofc_model.inp
```

This creates:
- `*MATERIAL` blocks for UMAT (Elastic, Expansion, Depvar)
- `*UEL PROPERTY` blocks for interface UEL (cohesive parameters)
- `*INITIAL CONDITIONS` for temperature field

### Step 3: Verify Physical Bounds
```bash
python scripts/verify_parameters.py --input data/master_material_database.csv
```

Checks:
- -1 < ν < 0.5 (thermodynamic stability)
- Gc,II ≥ Gc,I (typical for ceramics)
- α_LSCF > α_GDC > α_YSZ (expected trend)
- Positive definiteness of elastic tensor

### Step 4: Visualize Properties
```bash
python scripts/visualize_properties.py --input data/ --output figures/
```

Generates:
- E vs. T for all materials
- CTE comparison bar chart
- Fracture toughness hierarchy
- Chemical expansion strain fields

## Mesh Objectivity Requirements

To achieve mesh-objective fracture predictions:

1. **Phase-Field Resolution**: h ≤ l_pf/2 (at least 2 elements in the regularization zone)
   - YSZ: h ≤ 0.25 μm
   - GDC: h ≤ 0.20 μm
   - LSCF: h ≤ 0.30 μm

2. **Cohesive Zone Resolution**: h ≤ lc/3 (at least 3 elements in cohesive zone)
   - YSZ/GDC: h ≤ 0.027 μm
   - GDC/LSCF: h ≤ 0.020 μm

3. **Through-Thickness Elements**:
   - YSZ: 10 μm / 0.25 μm = 40 elements
   - GDC: 5 μm / 0.20 μm = 25 elements
   - LSCF: 30 μm / 0.30 μm = 100 elements

**Recommendation**: Adaptive mesh refinement near interfaces with 3:1 grading ratio.

## Parameter Sensitivity Hierarchy

Based on dimensional analysis and prior studies:

1. **Critical (>20% impact on Gc,eff)**:
   - Interface toughness ratio Gc,II/Gc,I
   - CTE mismatch Δα
   - Chemical expansion anisotropy β₃₃/β₁₁

2. **Important (10-20% impact)**:
   - Young's modulus mismatch
   - Layer thickness ratio t_LSCF/t_YSZ
   - Oxygen non-stoichiometry Δδ

3. **Secondary (<10% impact)**:
   - Poisson's ratio
   - Interface roughness wavelength
   - Weibull modulus

**Recommendation**: Focus experimental resources on Critical parameters first.

## Literature Sources

Key references for parameter values:

1. **Thermo-Elastic**: 
   - YSZ: Kilo et al., Acta Mater. 2008, doi:10.1016/j.actamat.2008.01.029
   - GDC: Hong & Virkar, SSI 2006, doi:10.1016/j.ssi.2006.05.030
   - LSCF: Jiang et al., JES 2005, doi:10.1149/1.1861177

2. **Chemical Expansion**:
   - Bishop et al., Acta Mater. 2013, doi:10.1016/j.actamat.2013.04.021
   - Marrocchelli et al., Adv. Funct. Mater. 2012, doi:10.1002/adfm.201102648

3. **Fracture**:
   - Malzbender et al., Fuel Cells 2009, doi:10.1002/fuce.200800110
   - Yakabe et al., J. Power Sources 2001, doi:10.1016/S0378-7753(00)00633-6

## Citation

If you use this dataset, please cite:

```
@dataset{sofc_fracture_dataset_2026,
  title = {Calibration Dataset for Mixed-Mode Fracture of YSZ/GDC/LSCF Interfaces},
  author = {Your Research Group},
  year = {2026},
  publisher = {Zenodo/GitHub/Institutional Repository},
  doi = {10.XXXX/XXXXXX}
}
```

## Contact & Support

For questions or data requests:
- **Email**: your.email@institution.edu
- **GitHub Issues**: [link to repository]
- **ORCID**: 0000-0000-0000-0000

## License

This dataset is released under **CC BY 4.0** license.  
You are free to share and adapt with proper attribution.

---

**Last Updated**: February 13, 2026  
**Version**: 1.0.0  
**Format**: CSV (UTF-8)  
**Total Records**: 60 parameters across 5 categories
