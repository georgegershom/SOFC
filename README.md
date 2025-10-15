# SOFC Residual Stress Prediction Dataset Generator

A comprehensive system for generating large-scale datasets for residual stress prediction in Solid Oxide Fuel Cells (SOFCs) using a combination of analytical models, FEA simulations, and data augmentation techniques.

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt

# For FEA simulations (optional)
# Install FEniCS (see https://fenicsproject.org/download/)
```

### Generate Dataset

```bash
# Quick generation (5-10 minutes, 1000 samples)
python run_dataset_generation.py --quick

# Standard generation (30-60 minutes, 5000 samples)
python run_dataset_generation.py --config standard

# Comprehensive generation (2-4 hours, 10000 samples)
python run_dataset_generation.py --config comprehensive

# Research-grade generation (6-12 hours, 20000 samples)
python run_dataset_generation.py --config research
```

## 📊 Dataset Overview

The generated dataset includes:

### Input Parameters (Features)
- **Geometric Parameters**: Plate dimensions, layer thicknesses, green densities
- **Material Properties**: Young's modulus, CTE, Poisson's ratio for all layers
- **Process Parameters**: Sintering temperature profiles, heating/cooling rates
- **Sintering Kinetics**: Shrinkage rates, onset temperatures, creep parameters

### Output Variables (Targets)
- **Stress Components**: Max principal stress, Von Mises stress, shear stress
- **Safety Metrics**: Safety factors, fracture risk assessments
- **Constitutive Models**: Both elastic and viscoelastic predictions
- **Validation Data**: FEA simulation results for model validation

### Data Sources
1. **Analytical Dataset**: Large-scale analytical calculations (fast)
2. **FEA Validation**: High-fidelity FEA simulations (accurate)
3. **Sintering Simulation**: Physics-based sintering process modeling
4. **Augmented Data**: Synthetically generated samples with physics constraints

## 🏗️ Architecture

```
SOFC Dataset Generator
├── Analytical Generator (sofc_dataset_generator.py)
│   ├── Design of Experiments (DOE) sampling
│   ├── Analytical stress calculations
│   └── Parameter space exploration
├── FEA Simulator (fenics_simulation.py)
│   ├── 3D thermal-mechanical coupling
│   ├── Viscoelastic creep modeling
│   └── Fracture risk assessment
├── Sintering Simulator (sintering_simulation.py)
│   ├── Density evolution modeling
│   ├── Multi-layer sintering kinetics
│   └── Residual stress generation
├── Data Augmenter (data_augmentation.py)
│   ├── Physics-informed synthesis
│   ├── Noise injection and interpolation
│   └── Domain adaptation techniques
└── Complete Generator (generate_complete_dataset.py)
    ├── Pipeline orchestration
    ├── ML model training
    └── Comprehensive reporting
```

## 📁 Generated Files

After running the dataset generation, you'll find:

```
output_directory/
├── complete_sofc_dataset.csv          # Main dataset (CSV format)
├── complete_sofc_dataset.h5           # Main dataset (HDF5 format)
├── comprehensive_report.md            # Detailed report
├── comprehensive_report.json          # Machine-readable report
├── visualizations/                    # Generated plots and charts
│   ├── dataset_overview.png
│   ├── parameter_correlation.png
│   ├── stress_analysis.png
│   └── model_performance.png
├── analytical/                        # Analytical dataset
├── fenics_validation_dataset.csv      # FEA validation data
├── sintering_dataset.csv              # Sintering simulation data
├── augmented_dataset.csv              # Augmented data
└── model_*.pkl                        # Trained ML models
```

## 🔧 Usage Examples

### Load Dataset

```python
import pandas as pd
import h5py

# Load CSV dataset
df = pd.read_csv('complete_sofc_dataset/complete_sofc_dataset.csv')

# Load HDF5 dataset (more efficient for large datasets)
with h5py.File('complete_sofc_dataset/complete_sofc_dataset.h5', 'r') as f:
    # Access specific arrays
    stress_data = f['max_principal_stress_elastic'][:]
    feature_data = f['electrolyte_thickness'][:]
```

### Use Trained Models

```python
import joblib
import numpy as np

# Load trained model
model_data = joblib.load('complete_sofc_dataset/model_max_principal_stress_elastic.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']

# Prepare new data
new_data = np.array([[100, 100, 0.15, 200, 10.5e-6, 1350, 2.0, 2.0, 120]])  # Example
new_data_scaled = scaler.transform(new_data)

# Make prediction
predicted_stress = model.predict(new_data_scaled)
print(f"Predicted stress: {predicted_stress[0]:.1f} MPa")
```

### Analyze Dataset

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('complete_sofc_dataset/complete_sofc_dataset.csv')

# Plot stress vs thickness
plt.figure(figsize=(10, 6))
plt.scatter(df['electrolyte_thickness'], df['max_principal_stress_elastic'], alpha=0.6)
plt.xlabel('Electrolyte Thickness (mm)')
plt.ylabel('Max Principal Stress (MPa)')
plt.title('Stress vs Thickness Relationship')
plt.grid(True, alpha=0.3)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df[['electrolyte_thickness', 'electrolyte_E_25C', 
                        'electrolyte_CTE_25C', 'max_temperature', 
                        'max_principal_stress_elastic']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Parameter Correlation Matrix')
plt.show()
```

## 🧪 Advanced Usage

### Custom Parameter Ranges

```python
from sofc_dataset_generator import SOFCDatasetGenerator

# Initialize generator
generator = SOFCDatasetGenerator()

# Modify parameter ranges
generator.param_ranges['electrolyte_thickness'] = (0.05, 0.25)  # mm
generator.param_ranges['max_temperature'] = (1200, 1400)       # °C

# Generate dataset with custom ranges
dataset = generator.generate_dataset(n_samples=1000)
```

### Custom Augmentation

```python
from data_augmentation import SOFCDataAugmenter

# Load dataset
df = pd.read_csv('complete_sofc_dataset/complete_sofc_dataset.csv')

# Initialize augmenter
augmenter = SOFCDataAugmenter(df)

# Custom augmentation
noisy_data = augmenter.augment_with_noise(noise_level=0.1, n_samples=1000)
interpolated_data = augmenter.augment_with_interpolation(n_samples=1000)
physics_data = augmenter.augment_with_physics_informed_synthesis(n_samples=1000)
```

### FEA Simulation

```python
from fenics_simulation import SOFCFEASimulator

# Initialize simulator
simulator = SOFCFEASimulator()

# Define dimensions
dimensions = {
    'plate_length': 100.0,    # mm
    'plate_width': 100.0,     # mm
    'anode_thickness': 0.3,   # mm
    'electrolyte_thickness': 0.15,  # mm
    'cathode_thickness': 0.05,      # mm
    'interconnect_thickness': 2.0   # mm
}

# Create mesh
mesh_data = simulator.create_sofc_mesh(dimensions)

# Set material properties
material_properties = {
    'anode': {'youngs_modulus': 55e9, 'poisson_ratio': 0.29, 'cte': 12.5e-6},
    'electrolyte': {'youngs_modulus': 200e9, 'poisson_ratio': 0.23, 'cte': 10.0e-6},
    # ... other layers
}

simulator.set_material_properties(material_properties)

# Run simulation
temperature_field = simulator.solve_thermal_analysis({'temperature': 800})
displacement, stress = simulator.solve_mechanical_analysis(temperature_field)
```

## 📈 Performance Metrics

The generated dataset includes comprehensive performance metrics:

- **R² Score**: Coefficient of determination for model accuracy
- **RMSE**: Root mean square error for prediction quality
- **MAE**: Mean absolute error for average prediction error
- **Safety Factor**: Ratio of material strength to predicted stress
- **Fracture Risk**: Probability of failure based on stress levels

## 🔬 Research Applications

This dataset is designed for:

1. **Machine Learning Research**: Training and validation of ML models for stress prediction
2. **Design Optimization**: Finding optimal SOFC designs with minimal residual stress
3. **Sensitivity Analysis**: Understanding which parameters most affect stress levels
4. **Process Optimization**: Optimizing sintering parameters for reduced stress
5. **Material Development**: Evaluating new materials and their stress behavior

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@software{sofc_dataset_generator,
  title={SOFC Residual Stress Prediction Dataset Generator},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-repo/sofc-dataset-generator}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or support, please:

1. Check the documentation in this README
2. Look at the example code in the repository
3. Open an issue on GitHub
4. Contact the maintainers

## 🔄 Updates

- **v1.0.0**: Initial release with basic analytical and FEA capabilities
- **v1.1.0**: Added sintering simulation and data augmentation
- **v1.2.0**: Added comprehensive ML model training and visualization
- **v1.3.0**: Added interactive dashboard and advanced reporting

---

**Happy Dataset Generation! 🎉**