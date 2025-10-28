# 🧠 Machine Learning Training Dataset for Sintering Process Analysis

This repository contains a comprehensive machine learning training dataset generator for analyzing sintering processes, thermal stress, and failure prediction in ceramic materials. The dataset is specifically designed for training Artificial Neural Networks (ANN) and Physics-Informed Neural Networks (PINN) models.

## 📋 Dataset Overview

### Generated Data Specifications
- **10,000+ simulated samples** with varying process parameters
- **Sintering temperatures**: 1200–1500°C
- **Cooling rates**: 1–10°C/min  
- **TEC mismatch**: Δα = 2.3×10⁻⁶ K⁻¹
- **Variable porosity levels**: 1-25% volume fraction
- **Spatial resolution**: 50×50 grids for field data

### Input Features
- Sintering temperature
- Cooling rate
- Porosity distribution (average, standard deviation)
- Grain size
- Sample thickness
- Temperature gradients
- Dwell time
- Stress field statistics

### Output Labels
- **Stress hotspots**: Location and intensity of high-stress regions
- **Crack initiation risk**: Probability based on Griffith criterion
- **Delamination probability**: Interface failure risk assessment
- **Failure risk score**: Combined risk metric

### Validation Data
- Synthetic experimental DIC (Digital Image Correlation) measurements
- Synthetic XRD (X-Ray Diffraction) stress measurements
- Realistic measurement uncertainties and noise

## 🚀 Quick Start

### Installation
```bash
# Install required packages
pip install -r requirements.txt
```

### Generate Dataset
```bash
# Run the complete pipeline
python run_dataset_generation.py
```

This will:
1. Generate 10,000 training samples
2. Create spatial field data
3. Generate validation data
4. Perform comprehensive analysis
5. Create train/validation/test splits
6. Generate visualization plots

### Load Generated Data
```python
import pandas as pd
import numpy as np

# Load main dataset
dataset = pd.read_csv('ml_sintering_dataset/ml_training_dataset.csv')

# Load pre-split data
train_data = pd.read_csv('ml_sintering_dataset/train_split.csv')
val_data = pd.read_csv('ml_sintering_dataset/val_split.csv')
test_data = pd.read_csv('ml_sintering_dataset/test_split.csv')

# Load spatial field data
import h5py
with h5py.File('ml_sintering_dataset/spatial_fields_data.h5', 'r') as f:
    sample_data = f['sample_0']
    temperature_field = sample_data['temperature_field'][:]
    stress_field = sample_data['stress_field'][:]
```

## 📁 Generated Files

After running the generator, you'll find these files in the `ml_sintering_dataset/` directory:

### Core Dataset Files
- `ml_training_dataset.csv` - Main tabular dataset (10,000 samples)
- `experimental_validation_data.csv` - Synthetic experimental data
- `spatial_fields_data.h5` - 2D spatial field data (HDF5 format)
- `ml_dataset_arrays.npz` - NumPy arrays for quick loading
- `dataset_metadata.json` - Dataset metadata and parameters

### Data Splits
- `train_split.csv` - Training set (70% of data)
- `val_split.csv` - Validation set (10% of data)  
- `test_split.csv` - Test set (20% of data)

### Analysis and Visualization
- `feature_distributions.png` - Input feature histograms
- `label_distributions.png` - Output label histograms
- `feature_correlations.png` - Feature correlation matrix
- `feature_label_correlations.png` - Feature-label correlations
- `spatial_patterns.png` - Example spatial field visualizations
- `pca_analysis.png` - Principal component analysis
- `validation_data_analysis.png` - Validation data statistics
- `data_quality_report.json` - Comprehensive quality metrics

## 🔬 Physics-Based Modeling

The dataset incorporates realistic physics-based models:

### Thermal Stress Calculation
- **Thermal expansion**: σ = E·α·ΔT/(1-ν)
- **Temperature gradients**: Spatial variations in thermal fields
- **Cooling rate effects**: Rate-dependent stress development

### Material Property Effects
- **Porosity influence**: Stress concentration around pores
- **Grain size effects**: Microstructural impact on crack initiation
- **Elastic modulus**: Temperature and porosity dependent

### Failure Mechanisms
- **Griffith criterion**: Crack initiation based on fracture mechanics
- **Interface delamination**: Thermal mismatch-induced failure
- **Stress hotspots**: Statistical identification of high-risk regions

## 🤖 Machine Learning Applications

### Recommended Model Architectures

#### 1. Artificial Neural Networks (ANN)
```python
# Example ANN for tabular data
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(n_outputs, activation='sigmoid')
])
```

#### 2. Physics-Informed Neural Networks (PINN)
- Incorporate spatial field data as additional constraints
- Use physics equations as loss function components
- Enforce conservation laws and boundary conditions

#### 3. Convolutional Neural Networks (CNN)
```python
# For spatial field analysis
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(50, 50, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_outputs, activation='sigmoid')
])
```

### Training Strategies
- **Multi-task learning**: Predict multiple failure modes simultaneously
- **Transfer learning**: Pre-train on simulated data, fine-tune on experimental
- **Ensemble methods**: Combine predictions from multiple models
- **Cross-validation**: Use experimental validation data for model selection

## 📊 Dataset Statistics

### Input Feature Ranges
- **Sintering Temperature**: 1200-1500°C
- **Cooling Rate**: 1-10°C/min
- **Porosity**: 1-25% volume fraction
- **Grain Size**: 1-50 μm
- **Thickness**: 0.5-5 mm
- **Temperature Gradient**: 5-50°C/mm

### Output Label Distributions
- **Crack Risk**: 0-1 probability scale
- **Delamination Probability**: 0-1 probability scale
- **Failure Risk Score**: Combined metric (0-1)
- **Stress Hotspots**: Count and intensity measures

## 🔧 Customization

### Modify Process Parameters
Edit the `param_ranges` dictionary in `SinteringDatasetGenerator`:

```python
self.param_ranges = {
    'sintering_temp': (1100, 1600),  # Extend temperature range
    'cooling_rate': (0.5, 15),       # Modify cooling rates
    'porosity': (0.005, 0.30),       # Adjust porosity range
    # ... add more parameters
}
```

### Add New Features
Extend the `generate_spatial_fields` method to include additional physics:

```python
def calculate_additional_physics(self, params):
    # Add creep effects
    # Include oxidation kinetics  
    # Model grain growth
    # etc.
```

### Custom Output Labels
Modify the label calculation methods to include domain-specific failure modes:

```python
def calculate_custom_failure_mode(self, stress_field, temp_field):
    # Implement custom failure criterion
    # Return probability or risk score
```

## 📚 References and Background

### Physics Models
- Thermal stress theory in ceramics
- Griffith fracture mechanics
- Weibull statistics for material strength
- Sintering kinetics and microstructure evolution

### Machine Learning Applications
- Physics-informed neural networks for materials
- Multi-scale modeling approaches
- Uncertainty quantification in ML predictions
- Transfer learning for experimental validation

## 🤝 Contributing

Feel free to extend this dataset generator for your specific applications:

1. **Add new material systems**: Modify material properties and constants
2. **Include additional physics**: Extend the physics-based models
3. **Improve validation data**: Add more realistic experimental noise models
4. **Optimize performance**: Parallelize generation for larger datasets

## 📄 License

This dataset generator is provided for research and educational purposes. Please cite appropriately if used in publications.

## 🆘 Support

For questions or issues:
1. Check the generated `data_quality_report.json` for dataset statistics
2. Review the analysis plots for data validation
3. Examine the metadata files for parameter details

---

**Happy modeling! 🚀**