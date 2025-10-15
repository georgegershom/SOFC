# Multi-Fidelity SOFC Degradation Dataset

## PhD Thesis: Multi-Fidelity Digital Twin for SOFCs
**Multi-Scale Modeling and Deep Learning for Predicting Thermo-Mechanical Degradation**

This repository contains a comprehensive synthetic dataset generator for Solid Oxide Fuel Cell (SOFC) multi-physics modeling, specifically designed for training multi-fidelity deep learning models to predict thermo-mechanical degradation.

## 🎯 Purpose

This dataset supports the development of a Multi-Fidelity Digital Twin that:
- Predicts SOFC degradation across multiple scales (micro to system level)
- Combines data from different fidelity levels (computational cost vs. accuracy trade-off)
- Enables real-time degradation prediction and remaining useful life estimation
- Supports uncertainty quantification in predictions

## 📊 Dataset Structure

### Fidelity Levels

1. **Low-Fidelity (LF)**: 100,000+ samples
   - 0D/1D lumped parameter models
   - ~0.1 seconds per sample
   - Global performance metrics and volume-averaged stresses

2. **Mid-Fidelity (MF)**: 5,000+ samples
   - 2D/3D coarse grid (20×20×10)
   - ~10 seconds per sample
   - Spatial fields of temperature, current, and stress

3. **High-Fidelity (HF)**: 200+ samples
   - 3D fine grid (100×100×50)
   - ~1 hour per sample
   - Detailed multi-physics coupling with microstructure evolution

4. **Experimental**: 20+ samples
   - Synthetic experimental data with realistic noise
   - I-V curves, EIS, IR thermography, SEM characterization
   - Includes measurement uncertainties

### Input Variables

- **Operating Conditions**: Temperature (650-900°C), Current density (0-1 A/cm²), Fuel/Air utilization, Pressure
- **Fuel Composition**: H₂, H₂O, CO, CO₂, CH₄, N₂ mixtures
- **Geometry**: Layer thicknesses with variations
- **Material Properties**: Porosity, tortuosity variations
- **Degradation History**: Operating hours, thermal cycles, redox cycles

### Output Variables

#### Thermo-Chemo-Electrical:
- Temperature fields (steady-state and transient)
- Current density distributions
- Species concentration fields (H₂, H₂O, O₂, etc.)
- Overpotentials (activation, ohmic, concentration)

#### Mechanical/Degradation:
- Stress tensor fields (6 components)
- Strain fields (elastic, plastic, creep, thermal)
- Damage indicators (crack density, delamination, Ni coarsening)
- Voltage degradation rates
- Time-to-failure predictions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sofc_multifidelity_dataset

# Install dependencies
pip install -r requirements.txt
```

### Generate Dataset

```bash
# Generate complete dataset (all fidelity levels)
python generate_dataset.py --all --parallel

# Generate specific fidelity level
python generate_dataset.py --fidelity low --samples 1000

# Quick test (10 samples each)
python generate_dataset.py --quick-test
```

### Visualize Data

```bash
# Generate all visualizations
python visualize_data.py --data ./data --output ./results/figures

# Specific visualizations
python visualize_data.py --plots iv temperature stress --samples 5
```

## 📁 Repository Structure

```
sofc_multifidelity_dataset/
│
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── generate_dataset.py         # Main dataset generation script
├── visualize_data.py          # Visualization tools
│
├── src/
│   ├── generators/            # Fidelity-specific generators
│   │   ├── low_fidelity_generator.py
│   │   ├── mid_fidelity_generator.py
│   │   ├── high_fidelity_generator.py
│   │   └── experimental_generator.py
│   │
│   └── utils/                 # Utility modules
│       ├── physics_utils.py   # SOFC physics models
│       ├── data_utils.py      # Data management
│       └── mesh_utils.py      # Mesh generation
│
├── data/                      # Generated datasets (HDF5)
│   ├── low_fidelity/
│   ├── mid_fidelity/
│   ├── high_fidelity/
│   └── experimental/
│
└── results/                   # Analysis results
    ├── figures/               # Visualizations
    └── analysis/              # Statistical analysis

```

## 🔬 Physics Models

### Electrochemistry
- Butler-Volmer kinetics with microstructure-dependent exchange currents
- Concentration overpotentials with dusty-gas model
- Water-gas shift reaction for syngas operation

### Thermal
- 3D heat conduction with volumetric heat generation
- Entropic heat effects
- Convective cooling at gas channels

### Mechanics
- Thermal stress from CTE mismatch
- Creep strain (Norton's law)
- Interface delamination (cohesive zone)
- Phase-field fracture modeling (HF only)

### Degradation
- Ni particle coarsening (Ostwald ripening)
- Chromium poisoning at cathode
- Redox cycling damage
- Thermal cycling fatigue (Coffin-Manson)

## 💻 Usage Examples

### Loading Data

```python
import h5py
import numpy as np

# Load low-fidelity data
with h5py.File('data/low_fidelity/low_fidelity_data.h5', 'r') as f:
    # Read inputs
    temperature = f['inputs/temperature'][:]
    current_density = f['inputs/current_density'][:]
    
    # Read outputs  
    voltage = f['outputs/voltage'][:]
    degradation_rate = f['outputs/degradation_rate'][:]
```

### Multi-Fidelity Training

```python
from src.utils.data_utils import DatasetManager

manager = DatasetManager('./data')

# Load multi-fidelity data
lf_data = manager.read_dataset('low_fidelity', indices=range(1000))
mf_data = manager.read_dataset('mid_fidelity', indices=range(100))
hf_data = manager.read_dataset('high_fidelity', indices=range(10))

# Use for multi-fidelity learning
# ... your ML model here ...
```

### Custom Configuration

Edit `config.yaml` to customize:
- Operating condition ranges
- Material properties
- Degradation parameters
- Dataset sizes
- Mesh resolutions

## 📈 Dataset Statistics

| Fidelity | Samples | Spatial Resolution | Time/Sample | File Size |
|----------|---------|-------------------|-------------|-----------|
| Low | 100,000 | 0D/1D | 0.1 sec | ~100 MB |
| Mid | 5,000 | 20×20×10 | 10 sec | ~5 GB |
| High | 200 | 100×100×50 | 1 hour | ~20 GB |
| Experimental | 20 | 64×64 (IR) | - | ~50 MB |

## 🎓 Applications

This dataset is designed for:

1. **Multi-Fidelity Machine Learning**
   - Transfer learning across fidelity levels
   - Gaussian process multi-fidelity regression
   - Deep neural network architectures (CNN, GNN, Transformers)

2. **Physics-Informed Neural Networks (PINNs)**
   - Incorporating governing equations
   - Enforcing conservation laws
   - Uncertainty quantification

3. **Digital Twin Development**
   - Real-time performance prediction
   - Remaining useful life estimation
   - Optimal operating strategy

4. **Uncertainty Quantification**
   - Bayesian deep learning
   - Ensemble methods
   - Confidence interval prediction

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_multifidelity_2024,
  title={Multi-Fidelity Dataset for SOFC Thermo-Mechanical Degradation},
  author={Your Name},
  year={2024},
  institution={Your University},
  note={PhD Thesis Dataset}
}
```

## ⚠️ Important Notes

1. **Synthetic Data**: This is synthetic data generated from physics-based models, not real experimental measurements
2. **Computational Requirements**: High-fidelity generation is computationally intensive
3. **Storage**: Full dataset requires ~25 GB of disk space
4. **Memory**: Loading full high-fidelity samples requires significant RAM (>16 GB recommended)

## 🔧 Customization

To add new physics or modify existing models:

1. Edit physics models in `src/utils/physics_utils.py`
2. Modify generators in `src/generators/`
3. Update configuration in `config.yaml`
4. Regenerate dataset

## 📊 Validation

The dataset includes:
- Physical consistency checks
- Conservation law validation
- Statistical distribution analysis
- Cross-fidelity correlation analysis

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional degradation mechanisms
- More sophisticated microstructure models
- Advanced uncertainty quantification
- Real experimental data integration

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- SOFC modeling community
- Multi-fidelity learning researchers
- Open-source scientific Python community

## 📧 Contact

For questions or collaborations:
- Email: your.email@university.edu
- GitHub Issues: [Create an issue](https://github.com/yourusername/repo/issues)

---

**Note**: This dataset generator creates synthetic data based on established SOFC physics models. While the physics is representative, real experimental validation is essential for production use.