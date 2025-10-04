# Atomic-Scale Simulation Dataset Summary

## Dataset Overview
**Generated**: 2025-10-04  
**Purpose**: Quantum-enhanced inputs for multiscale material modeling  
**Focus**: High-temperature creep mechanisms in metallic alloys  

## Quick Statistics

### DFT Calculations (205 total)
| Calculation Type | Count | Key Result |
|-----------------|-------|------------|
| Vacancy Formation | 50 | 1.39 ± 0.18 eV |
| Dislocation Energies | 30 | 125-150 mJ/m² (stacking fault) |
| Grain Boundary Energies | 40 | 1.57 ± 1.24 J/m² |
| Activation Barriers | 60 | 0.99 ± 0.74 eV |
| Surface Energies | 25 | 1.95 ± 0.31 J/m² |

### MD Simulations (80 total)
| Simulation Type | Count | Key Metric |
|----------------|-------|------------|
| GB Sliding | 20 | Stress exponent n ≈ 3 |
| Dislocation Mobility | 25 | Type-dependent: edge > mixed > screw |
| Cross-Slip Events | 15 | Activation: 0.8-1.5 eV |
| Dislocation Interactions | 20 | Junction formation strength |

### Temperature & Stress Ranges
- **Temperature**: 600-1200 K
- **Applied Stress**: 50-1000 MPa
- **System Sizes**: 50,000-200,000 atoms (MD)

## File Structure
```
📁 atomic_simulation_dataset/
├── 📁 dft_calculations/      # 205 DFT outputs
├── 📁 md_simulations/         # 80 MD trajectories
├── 📁 processed_data/         # Analysis results & plots
├── 📁 scripts/               # Generation & analysis tools
├── 📁 docs/                  # Full documentation
└── 📄 README.md              # Quick start guide
```

## Key Parameters for Models

### For Phase-Field Simulations
- Vacancy formation energy: **1.39 eV**
- Vacancy migration barrier: **1.22 eV**
- GB energy: **1.57 J/m²**
- Surface energy: **1.95 J/m²**

### For Crystal Plasticity
- Edge dislocation mobility: **9.7×10⁻⁵ Å·ps⁻¹·MPa⁻¹**
- Screw dislocation mobility: **1.7×10⁻⁶ Å·ps⁻¹·MPa⁻¹**
- Peierls stress: **50-200 MPa**
- Cross-slip activation: **0.8-1.5 eV**

### For Continuum Creep Models
- Power-law exponent: **n ≈ 3**
- Activation energy: **2.5 eV** (GB sliding)
- Temperature sensitivity: Well-captured 600-1200 K

## Data Formats
- **Primary**: JSON with hierarchical structure
- **Exports**: CSV tables for tabular data
- **Units**: eV (energy), Å (length), ps (time), MPa (stress)

## Usage Quick Start
```python
import json

# Load DFT data
with open('dft_calculations/defect_energies/vacancy_formation.json') as f:
    vacancy_data = json.load(f)

# Load MD data  
with open('md_simulations/grain_boundary/gb_sliding.json') as f:
    gb_data = json.load(f)

# Access pre-analyzed results
with open('processed_data/summary_report.json') as f:
    summary = json.load(f)
```

## Computational Cost Estimate
- **DFT**: ~37,000 CPU-hours
- **MD**: ~2,000 GPU-hours
- **Total Storage**: ~100 MB

## Applications
✅ Train surrogate models for expensive calculations  
✅ Parameterize phase-field models  
✅ Inform crystal plasticity constitutive laws  
✅ Validate continuum creep models  
✅ Machine learning training data  

## Quality Metrics
- **Energy convergence**: 10⁻⁶ eV
- **Force convergence**: 0.01 eV/Å
- **Statistical sampling**: 50-200 configurations per property
- **Temperature coverage**: Full operational range

## Contact
For questions or collaborations regarding this dataset, please refer to the full documentation in `docs/DOCUMENTATION.md`

---
*Dataset Version 1.0 | Generated for quantum-enhanced multiscale modeling*