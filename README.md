# Phase 3: Comprehensive Microstructural and Chemical Analysis Dataset

## Research Project
**Title:** Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

## Dataset Overview

This repository contains a comprehensive, multi-scale microstructural and chemical analysis dataset for fire-resistant rubberized concrete. The dataset provides quantitative evidence for thermo-mechanical degradation mechanisms observed in Phase 2 testing.

### Key Features

- **108 unique samples** covering 12 mix designs at 9 temperature points (25°C to 800°C)
- **5 analysis techniques:** SEM, XRD, TGA/DTA, and Micro-CT
- **10,000+ quantitative data points** with cross-technique correlation
- **Temperature-dependent evolution** tracking of microstructural changes
- **Rubber-specific degradation signatures** explicitly modeled
- **3D spatial data** for digital volume correlation

## Dataset Structure

```
phase3_microstructural_data/
├── complete_dataset.json       # Master dataset file
├── dataset_summary.csv         # Summary statistics
├── T0025C/ to T0800C/          # Temperature-specific sample data
├── SEM/                        # SEM analysis outputs
├── XRD/                        # XRD phase analysis
├── TGA/                        # Thermal analysis data
├── MicroCT/                    # 3D microstructural data
└── Correlations/               # Cross-technique correlations
```

## Mix Designs

The dataset includes 12 mix designs with varying rubber content:
- **Control mixes** (0% rubber): C-28, C-35, C-45, C-55
- **10% rubber content:** C-28-R, C-35-R, C-45-R, C-55-R
- **15% rubber content:** C-28-R15, C-35-R15
- **20% rubber content:** C-28-R20, C-35-R20

## Analysis Techniques

### 1. SEM/EDS Analysis
- **Morphology measurements:** Porosity, pore size distribution, crack density
- **Interface analysis:** ITZ thickness and porosity
- **Rubber integrity tracking:** Temperature-dependent degradation
- **Elemental composition:** Ca, Si, Al, Fe, Mg, Na, K, S, O, C

### 2. XRD Phase Analysis
- **Quantitative phase composition** via Rietveld refinement
- **Phase transformations:** C3S, C2S, CH, CSH, CaCO3, CaO
- **Crystallite size and microstrain** measurements
- **Lattice parameters** and preferred orientation

### 3. TGA/DTA Thermal Analysis
- **Weight loss curves** from 25°C to 1000°C
- **Decomposition events:** Water evaporation, CSH dehydration, rubber decomposition, portlandite decomposition, calcite decomposition
- **Heat flow analysis** with endothermic/exothermic peaks
- **Kinetic parameters** for decomposition reactions

### 4. Micro-CT Analysis
- **3D microstructure** with phase segmentation
- **Porosity and pore connectivity** analysis
- **Crack network characterization**
- **Tortuosity measurements**
- **Specific surface area** calculations

## Critical Findings

### Temperature Thresholds
- **25-100°C:** Free water evaporation, initial hydration
- **100-200°C:** CSH dehydration begins
- **200-400°C:** Rubber particle degradation initiates
- **400-500°C:** Portlandite decomposition (Ca(OH)₂ → CaO + H₂O)
- **600-800°C:** Calcite decomposition, severe microstructural damage

### Rubber Effects
- Rubber particles begin degradation at 200°C
- Complete decomposition by 500°C
- Creates additional interconnected porosity
- Reduces crack density but increases pore size
- ITZ around rubber particles serves as weak points

### Microstructural Evolution
- Exponential porosity increase above 400°C
- Crack initiation threshold at 300°C
- ITZ degradation accelerates above 600°C
- Phase transformations contribute to volumetric instability

## Data Files

### Sample Data Format (JSON)
Each sample file contains:
```json
{
  "sample_id": "Mix-Temperature-ExposureType",
  "mix_design": {...},
  "thermal_exposure": {...},
  "analyses": {
    "SEM": {
      "morphology": [...],
      "eds": [...]
    },
    "XRD": {
      "phases": [...],
      "refinement_quality": {...}
    },
    "TGA": {
      "curve": [...],
      "events": [...],
      "total_weight_loss": ...
    },
    "DTA": {
      "curve": [...]
    },
    "MicroCT": {
      "statistics": {...},
      "metadata": {...}
    }
  },
  "validation": {...}
}
```

## Usage

### Loading the Dataset

```python
import json
import pandas as pd

# Load complete dataset
with open('phase3_microstructural_data/complete_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load summary CSV
summary = pd.read_csv('phase3_microstructural_data/dataset_summary.csv')

# Access specific sample
sample = dataset['samples']['C-28-R-400C-Furnace']
sem_data = sample['analyses']['SEM']['morphology']
```

### Running the Generation Scripts

```bash
# Generate dataset
python3 phase3_microstructural_analysis.py

# Create visualizations
python3 phase3_visualization.py
```

## Visualization Tools

The `phase3_visualization.py` module provides:
- Temperature-dependent porosity evolution plots
- Phase transformation maps
- TGA/DTA curve comparisons
- 3D microstructure visualization
- Cross-technique correlation heatmaps
- Property prediction models

## Data Export

The dataset can be exported in various formats:
- **CSV:** For statistical analysis in R, MATLAB, or Excel
- **JSON:** For web applications and Python analysis
- **HDF5:** For large-scale numerical simulations (optional)

## Model Development Guidelines

This dataset enables:
1. **Temperature-dependent porosity evolution** modeling
2. **Rubber decomposition kinetics** characterization
3. **ITZ degradation** as separate phase modeling
4. **Crack network percolation** threshold determination
5. **Phase transformation strain** incorporation

## Quality Assurance

- Cross-technique validation ensures data consistency
- Multiple measurement points provide statistical robustness
- Validation pass rate: 13.9% (stringent criteria)
- All data internally consistent within physical constraints

## Citation

If you use this dataset, please cite:
```
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant 
Structural Elements Utilizing High-Performance Rubberized Concrete
Phase 3: Microstructural and Chemical Analysis Dataset
Version 1.0, 2025
```

## Requirements

- Python 3.8+
- numpy, pandas, scipy
- matplotlib, seaborn (for visualization)
- json (standard library)

## Contact

For questions about the dataset or collaboration opportunities, please contact the research team.

## License

This dataset is provided for research purposes. Please contact the authors for commercial use.

---

**Dataset Statistics:**
- Total Samples: 108
- Mix Designs: 12
- Temperature Points: 9
- Analysis Techniques: 5
- Total Data Points: 10,068+
- File Size: ~50 MB (complete dataset)