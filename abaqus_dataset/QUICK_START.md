# Quick Start Guide: SOFC Fracture Dataset

## Installation

1. **Extract the dataset:**
```bash
unzip abaqus_dataset.zip
cd abaqus_dataset/
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Verify Parameter Database
Check physical consistency of material parameters:
```bash
python scripts/verify_parameters.py --input data/master_material_database.csv
```

This will check:
- Thermodynamic bounds (Poisson's ratio)
- Material property trends (E vs. T)
- Fracture toughness hierarchy (Gc,II ≥ Gc,I)
- CTE hierarchy (α_LSCF > α_GDC > α_YSZ)
- Mesh resolution requirements

### 2. Generate Abaqus Input File
Convert CSV database to Abaqus .inp format:
```bash
python scripts/generate_abaqus_input.py \
    --input data/master_material_database.csv \
    --output sofc_model.inp \
    --summary
```

This generates:
- `*MATERIAL` blocks for UMAT
- `*UEL PROPERTY` blocks for cohesive interfaces
- `*AMPLITUDE` for thermal loading
- Templates for boundary conditions

### 3. Visualize Material Properties
Generate publication-quality figures:
```bash
python scripts/visualize_properties.py \
    --input data/ \
    --output figures/
```

Creates:
- `fig1_elastic_modulus_vs_temperature.png`
- `fig2_cte_comparison.png`
- `fig3_fracture_toughness.png`
- `fig4_chemical_expansion.png`
- `fig5_validation_data.png`

## Data Files

### CSV Files in `data/`
| File | Description | Records |
|------|-------------|---------|
| `master_material_database.csv` | Complete parameter database | 60+ |
| `geometric_parameters.csv` | Layer thicknesses, roughness, porosity | 13 |
| `thermoelastic_properties.csv` | E, ν, α vs. Temperature | 24 |
| `chemical_expansion_data.csv` | β, Δδ vs. pO₂ | 13 |
| `fracture_cohesive_properties.csv` | Gc, T_max, η parameters | 24 |
| `experimental_validation_data.csv` | Curvature, delamination measurements | 16 |

### Key Parameters

**Materials:**
- **8YSZ**: E = 210 GPa (RT), α = 10.5 ppm/K, Gc = 25 J/m²
- **GDC10**: E = 160 GPa (RT), α = 12.5 ppm/K, Gc = 20 J/m²
- **LSCF**: E = 80 GPa (RT), α = 15.8 ppm/K, Gc = 15 J/m²

**Interfaces:**
- **YSZ/GDC**: Gc,I = 12 J/m², Gc,II = 18 J/m²
- **GDC/LSCF**: Gc,I = 8 J/m², Gc,II = 14 J/m² (weakest link)

**Chemical Expansion:**
- **GDC**: β_iso = 0.025 strain/Δδ, Δδ = 0.05
- **LSCF**: β₁₁ = 0.08, β₃₃ = 0.12, Δδ = 0.15

## Running Abaqus Simulation

### Prerequisites
1. Abaqus/Standard 2021 or later
2. Intel Fortran Compiler (for UMAT/UEL)
3. Custom UMAT/UEL subroutines (not included - contact authors)

### Workflow
```bash
# 1. Generate input file
python scripts/generate_abaqus_input.py \
    --input data/master_material_database.csv \
    --output sofc_model.inp

# 2. Add mesh and element definitions to sofc_model.inp
# (Use Abaqus/CAE or Python scripting)

# 3. Compile and run with UMAT/UEL
abaqus job=sofc_model user=umat_uel.f cpus=8 interactive

# 4. Post-process results
abaqus viewer odb=sofc_model.odb
```

### Expected Simulation Time
- **2D Plane Strain**: ~2-4 hours (100k elements, 8 CPUs)
- **3D with Roughness**: ~24-48 hours (1M elements, 32 CPUs)

## Mesh Requirements

For mesh-objective results:

### Phase Field Resolution
```
h ≤ l_pf / 2

YSZ: h ≤ 0.25 μm
GDC: h ≤ 0.20 μm  
LSCF: h ≤ 0.30 μm
```

### Cohesive Zone Resolution
```
h ≤ lc / 3

YSZ/GDC: h ≤ 0.027 μm
GDC/LSCF: h ≤ 0.020 μm
```

### Through-Thickness
```
At least 5 elements per layer:

YSZ (10 μm): 40 elements
GDC (5 μm): 25 elements
LSCF (30 μm): 100 elements
```

## Validation Metrics

Compare simulation results to experimental data:

1. **Global Curvature**: κ(T) from DIC
   - Target: κ(800°C) = 0.0015 mm⁻¹, κ(RT) = 0.045 mm⁻¹

2. **Delamination Onset**: T_delam from in-situ SEM
   - Target: 350°C during cooling

3. **Crack Path**: Compare phase-field damage contour to SEM images
   - Primary failure: GDC/LSCF interface

## Troubleshooting

### Issue: "Module pandas not found"
```bash
pip install pandas numpy matplotlib
```

### Issue: Abaqus convergence problems
- Reduce initial time increment (try 0.001)
- Enable automatic stabilization
- Check mesh distortion near interfaces

### Issue: Phase field not converging
- Increase phase field length scale l_pf
- Use staggered solution scheme
- Refine mesh in crack region

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_fracture_dataset_2026,
  title = {Calibration Dataset for Mixed-Mode Fracture of YSZ/GDC/LSCF Interfaces},
  author = {Your Research Group},
  year = {2026},
  doi = {10.XXXX/XXXXXX}
}
```

## Support

For questions or issues:
- **Documentation**: See `docs/README.md`
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Email**: your.email@institution.edu

## License

CC BY 4.0 - Free to use with attribution

---

**Version**: 1.0.0  
**Last Updated**: February 13, 2026  
**Dataset DOI**: 10.XXXX/XXXXXX
