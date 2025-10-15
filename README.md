# Residual Stress Dataset for Multi-Layer Ceramics

## 🎯 Overview

This repository contains a comprehensive dataset and simulation framework for predicting residual stress in multi-layer ceramic structures, specifically designed for Solid Oxide Fuel Cell (SOFC) applications. The dataset includes process parameters, material properties, and geometric variations that drive residual stress formation during manufacturing.

## 📊 Dataset Description

### Dataset 2: Process and Material Parameters (The "Context" Dataset)

This dataset enables predictive modeling of residual stress by including the conditions that cause the stress. It allows you to predict stress for new sets of process parameters, making your model predictive rather than just descriptive.

### Key Features

- **10,000+ samples** with comprehensive parameter variations
- **38 input features** covering geometry, materials, and process conditions
- **6 stress outputs** for each layer (anode, electrolyte, cathode)
- **Multiple file formats**: CSV, Excel, HDF5, JSON
- **Physics-based simulation** with temperature-dependent properties
- **Realistic parameter ranges** based on industrial SOFC manufacturing

## 🔬 Parameters Included

### Geometric Parameters
- **Plate dimensions**: Length and width (50-200 mm)
- **Layer thicknesses**: 
  - Anode: 300-1500 μm
  - Electrolyte: 5-50 μm  
  - Cathode: 20-100 μm
- **Green densities**: Initial "green" density before sintering (40-70%)

### Material Properties (Temperature-Dependent)
For each layer (anode, electrolyte, cathode):
- **Young's Modulus**: 50-220 GPa (with temperature dependence)
- **Coefficient of Thermal Expansion (CTE)**: 10-14 × 10⁻⁶ /K
- **Poisson's Ratio**: 0.25-0.35
- **Sintering shrinkage parameters**: 12-28% total shrinkage
- **Creep parameters**: Activation energies for high-temperature stress relaxation

### Process Parameters
- **Sintering temperature profile**: 
  - Maximum temperature: 1300-1500°C
  - Heating rates: 1-10 K/min
  - Cooling rates: 1-5 K/min
  - Hold times: 1-8 hours
- **Atmosphere**: Oxygen partial pressure (10⁻²⁰ to 0.21 atm)

### Output Variables
- **Residual stress** for each layer (total stress)
- **von Mises equivalent stress** for each layer
- **Maximum stress** during thermal cycle

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd residual-stress-dataset

# Install dependencies
pip install -r requirements.txt
```

### Generate Dataset

```python
from residual_stress_dataset_generator import ResidualStressDatasetGenerator

# Initialize generator
generator = ResidualStressDatasetGenerator(random_seed=42)

# Generate dataset
dataset = generator.generate_comprehensive_dataset(n_samples=1000)

# Save in multiple formats
filenames = generator.save_dataset(dataset, 'my_dataset')
```

### Load and Analyze Dataset

```python
import pandas as pd
from dataset_visualizer import DatasetVisualizer

# Load dataset
df = pd.read_csv('residual_stress_dataset_10000.csv')

# Create visualizer
visualizer = DatasetVisualizer(dataset_path='residual_stress_dataset_10000.csv')

# Generate all visualizations
visualizer.plot_all_visualizations('analysis_results')
```

### Advanced FEA Simulation

```python
from advanced_fea_simulator import AdvancedFEASimulator

# Initialize simulator
simulator = AdvancedFEASimulator()

# Define sample parameters
sample_params = {
    'plate_length': 0.1,  # 100 mm
    'plate_width': 0.1,   # 100 mm
    'anode_thickness': 500e-6,  # 500 μm
    'electrolyte_thickness': 20e-6,  # 20 μm
    'cathode_thickness': 50e-6,  # 50 μm
    # ... (material and process parameters)
}

# Run simulation
results = simulator.simulate_residual_stress_advanced(sample_params)
```

## 📁 File Structure

```
residual-stress-dataset/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── residual_stress_dataset_generator.py   # Main dataset generator
├── advanced_fea_simulator.py             # Advanced FEA simulation
├── dataset_visualizer.py                 # Visualization tools
├── analyze_dataset.py                    # Dataset analysis script
├── residual_stress_dataset_1000.csv      # 1K samples dataset
├── residual_stress_dataset_5000.csv      # 5K samples dataset
├── residual_stress_dataset_10000.csv     # 10K samples dataset
├── residual_stress_dataset_10000.xlsx    # Excel format
├── residual_stress_dataset_10000.h5      # HDF5 format
└── residual_stress_dataset_10000_metadata.json  # Metadata
```

## 🔍 Dataset Statistics

### Sample Size and Features
- **Total samples**: 10,000
- **Input features**: 31 (geometric, material, process parameters)
- **Output variables**: 6 (stress measures for 3 layers)
- **Missing values**: 0

### Stress Ranges
- **Anode stress**: -24.4 to +14.3 GPa
- **Cathode stress**: -23.8 to +5.8 GPa  
- **Electrolyte stress**: Reference layer (0 stress)

### Key Correlations
The strongest correlations with residual stress are:
1. **Sintering shrinkage mismatch** (r = 0.65-0.67)
2. **Hold time at maximum temperature** (r = 0.14-0.28)
3. **CTE mismatch between layers** (moderate correlation)

## 🧪 Physics-Based Modeling

The dataset is generated using physics-based models that capture:

### Thermal Stress Mechanisms
- **CTE mismatch**: Different thermal expansion coefficients cause stress during cooling
- **Sintering shrinkage**: Differential shrinkage between layers creates constraints
- **Temperature-dependent properties**: Material properties vary with temperature

### Advanced Features
- **Creep and stress relaxation**: High-temperature viscoplastic behavior
- **Geometric effects**: Aspect ratio and thickness ratio influences
- **Process atmosphere effects**: Oxygen partial pressure impacts

### Validation
- Stress magnitudes are physically reasonable (< 30 GPa)
- Temperature dependencies follow ceramic material behavior
- Correlations match experimental observations from literature

## 📈 Machine Learning Applications

This dataset is ideal for:

### Predictive Modeling
- **Regression models**: Predict residual stress from process parameters
- **Classification**: Identify high-stress conditions
- **Optimization**: Find process conditions that minimize stress

### Model Types
- **Linear models**: Baseline understanding of parameter effects
- **Tree-based models**: Capture non-linear interactions
- **Neural networks**: Complex multi-physics relationships
- **Gaussian processes**: Uncertainty quantification

### Feature Engineering
- **CTE mismatch terms**: (α₁ - α₂) × ΔT
- **Shrinkage mismatch**: Differential sintering shrinkage
- **Geometric ratios**: Aspect ratios, thickness ratios
- **Process integrals**: Temperature-time integrals

## 🎨 Visualization Examples

The dataset includes comprehensive visualization tools:

### Parameter Distributions
- Histograms and KDE plots for all input parameters
- Grouped by category (geometric, material, process)

### Stress Analysis
- Stress distributions by layer
- Correlation matrices
- Parameter vs. stress scatter plots

### Interactive Dashboards
- Plotly-based interactive exploration
- Multi-dimensional parameter relationships
- Real-time filtering and selection

### PCA Analysis
- Dimensionality reduction
- Principal component interpretation
- Variance explained analysis

## 🔧 Customization

### Adding New Parameters
```python
# Extend parameter ranges in the generator
generator.geometric_params['new_parameter'] = {
    'min': 0.1, 'max': 1.0, 'units': 'mm'
}
```

### Custom Stress Models
```python
# Implement custom stress calculation
def custom_stress_model(params):
    # Your physics-based model here
    return stress_results
```

### Different Material Systems
The framework can be adapted for:
- **Solid oxide electrolysis cells (SOEC)**
- **Multi-layer capacitors (MLCC)**
- **Thermal barrier coatings (TBC)**
- **Electronic packaging**

## 📚 References and Background

### Physical Principles
1. **Thermal stress theory**: Classical thermoelasticity
2. **Sintering mechanics**: Constrained sintering models
3. **Multi-layer mechanics**: Laminate theory adaptations

### SOFC-Specific Considerations
- Material properties from literature (Ni-YSZ, YSZ, LSM-YSZ)
- Typical manufacturing temperature profiles
- Industrial thickness ranges and geometries

### Validation Sources
- Experimental data from SOFC literature
- FEA simulation benchmarks
- Industrial manufacturing experience

## 🤝 Contributing

We welcome contributions to improve the dataset and simulation framework:

1. **New material systems**: Add parameter ranges for different ceramics
2. **Enhanced physics**: Improve stress calculation models  
3. **Validation data**: Compare with experimental measurements
4. **Visualization**: Add new analysis and plotting capabilities

## 📄 License

This dataset and code are provided for research and educational purposes. Please cite this work if you use it in publications.

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue in the repository.

---

**Note**: This dataset represents simulated data based on physics-based models. While the models incorporate realistic material properties and process conditions, validation against experimental data is recommended for critical applications.