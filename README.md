# Multi-Fidelity SOFC Digital Twin Dataset

## 🔬 Overview

This repository contains a comprehensive multi-fidelity dataset for Solid Oxide Fuel Cell (SOFC) digital twin modeling, specifically designed for PhD research on "Multi-Fidelity Digital Twin for SOFCs: Multi-Scale Modeling and Deep Learning for Predicting Thermo-Mechanical Degradation."

The dataset encompasses multiple scales and fidelity levels to support various modeling approaches from system-level performance prediction to detailed microstructural analysis.

## 📊 Dataset Structure

### Fidelity Levels

- **Low Fidelity (LF)**: 1,000 samples - System-level operating conditions
- **Medium Fidelity (MF)**: 500 samples - System + geometry + material properties  
- **High Fidelity (HF)**: 200 samples - Complete multi-scale parameters + microstructural data

### Scale Categories

| Scale | Parameters | Fidelity Levels | Description |
|-------|------------|----------------|-------------|
| **System** | Operating conditions, fuel composition, flow rates | LF, MF, HF | Temperature, pressure, utilization factors |
| **Cell/Stack** | Geometric dimensions, layer thicknesses | MF, HF | Physical cell architecture |
| **Material** | Component properties (Ni-YSZ, LSCF, YSZ, Crofer 22APU) | MF, HF | Conductivities, mechanical properties |
| **Microstructural** | 3D reconstructed data, phase fractions, connectivity | HF | Experimental tomography data |

## 🗂️ Files Generated

```
sofc_dataset/
├── sofc_parameters_lf_fidelity.csv      # Low fidelity parameters (1,000 samples)
├── sofc_parameters_mf_fidelity.csv      # Medium fidelity parameters (500 samples)  
├── sofc_parameters_hf_fidelity.csv      # High fidelity parameters (200 samples)
├── sofc_parameters_combined.csv         # All fidelity levels combined (1,700 samples)
├── sofc_microstructural_data.json       # High-fidelity microstructural data (200 samples)
├── sofc_transient_profiles.json         # Transient cycle profiles (300 profiles)
├── dataset_metadata.json                # Dataset metadata and specifications
├── dataset_summary_report.md            # Comprehensive summary report
└── [visualization plots]                # Generated analysis plots
```

## 🚀 Quick Start

### 1. Generate Dataset

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the complete dataset
python3 sofc_dataset_generator.py
```

### 2. Analyze Dataset

```bash
# Run comprehensive analysis and generate visualizations
python3 dataset_analyzer.py
```

### 3. Load Data in Python

```python
import pandas as pd
import json

# Load parameter datasets
lf_data = pd.read_csv('sofc_dataset/sofc_parameters_lf_fidelity.csv')
mf_data = pd.read_csv('sofc_dataset/sofc_parameters_mf_fidelity.csv')
hf_data = pd.read_csv('sofc_dataset/sofc_parameters_hf_fidelity.csv')

# Load microstructural data
with open('sofc_dataset/sofc_microstructural_data.json', 'r') as f:
    microstructural_data = json.load(f)

# Load transient profiles
with open('sofc_dataset/sofc_transient_profiles.json', 'r') as f:
    transient_profiles = json.load(f)
```

## 📋 Parameter Details

### System Level Parameters

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `fuel_utilization` | 0.6 - 0.95 | - | Fraction of fuel consumed |
| `oxidant_utilization` | 0.15 - 0.4 | - | Fraction of oxidant consumed |
| `current_density` | 0.1 - 1.5 | A/cm² | Operating current density |
| `temperature` | 973 - 1273 | K | Operating temperature |
| `pressure` | 1.0 - 10.0 | atm | System pressure |
| `h2_fraction` | 0.3 - 0.97 | - | Hydrogen fraction in fuel |

### Material Properties (Examples)

#### Anode (Ni-YSZ)
- `ni_volume_fraction`: 0.3 - 0.6
- `porosity`: 0.25 - 0.45  
- `tortuosity`: 2.0 - 8.0
- `ionic_conductivity`: 0.01 - 0.1 S/m

#### Electrolyte (YSZ)  
- `ionic_conductivity`: 0.01 - 0.5 S/m
- `youngs_modulus`: 150 - 250 GPa
- `thermal_expansion_coeff`: 9e-6 - 12e-6 K⁻¹

### Microstructural Data

Each sample includes:
- 3D voxel reconstruction parameters
- Phase fractions (Ni, YSZ, Pore)
- Connectivity metrics
- Specific surface area
- Particle size distributions

### Transient Profiles

Three types of degradation-relevant profiles:
- **Startup**: Temperature and current ramps (1 hour duration)
- **Shutdown**: Controlled cooling profiles (30 min duration)  
- **Load Following**: Dynamic load variations (2 hour cycles)

## 🔧 Design of Experiments

The dataset uses **Latin Hypercube Sampling (LHS)** for efficient exploration of the multi-dimensional parameter space:

- Ensures uniform coverage across parameter ranges
- Minimizes correlation between parameters
- Optimizes sample efficiency for machine learning applications
- Reproducible with fixed random seeds

## 🎯 Applications

This dataset supports various research applications:

### 1. Multi-Fidelity Modeling
- Train surrogate models at different fidelity levels
- Develop fidelity bridging techniques
- Optimize computational resource allocation

### 2. Degradation Prediction
- Correlate operating conditions with degradation mechanisms
- Develop predictive models for lifetime estimation
- Analyze thermo-mechanical stress evolution

### 3. Digital Twin Development
- Real-time parameter estimation
- Uncertainty quantification
- Model validation and calibration

### 4. Machine Learning Applications
- Deep learning for performance prediction
- Physics-informed neural networks
- Multi-task learning across scales

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 1,700 |
| **Parameters (LF)** | 42 |
| **Parameters (HF)** | 156 |
| **Microstructural Samples** | 200 |
| **Transient Profiles** | 300 |
| **Total Size** | ~50 MB |

## 🔬 Experimental Validation

The synthetic microstructural data is designed to match characteristics from:
- **FIB-SEM** tomography (50% of samples)
- **X-Ray CT** reconstruction (50% of samples)
- Realistic phase fractions and connectivity values
- Literature-validated parameter ranges

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_multifidelity_2025,
  title={Multi-Fidelity SOFC Digital Twin Dataset: Multi-Scale Modeling Parameters},
  author={[Your Name]},
  year={2025},
  publisher={PhD Thesis Dataset},
  note={Multi-Scale Modeling and Deep Learning for Predicting Thermo-Mechanical Degradation}
}
```

## 🤝 Contributing

This dataset is part of ongoing PhD research. For questions, suggestions, or collaboration opportunities, please contact the research team.

## 📄 License

This dataset is provided for academic and research purposes. Please refer to your institution's data sharing policies.

## 🔗 Related Work

- SOFC modeling and simulation frameworks
- Multi-fidelity optimization techniques  
- Digital twin methodologies for energy systems
- Machine learning for materials science

---

**Generated on:** 2025-10-15  
**Version:** 1.0  
**Maintainer:** PhD Research Team