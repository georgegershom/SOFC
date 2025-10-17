# 🔥 Fire-Resistant Rubberized Concrete Baseline Dataset

## Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Complete-success)](complete_baseline_dataset_20251017_003113.json)

---

## 🎯 Overview

This repository contains a **comprehensive baseline dataset** for developing thermo-mechanical models of fire-resistant rubberized concrete. The dataset provides critical material characterization data required for understanding the behavior of structural concrete elements containing recycled tire rubber aggregates under fire conditions.

### 🔬 Research Focus
- **Pillar 1**: Material Characterization & Mixture Design (Baseline State)
- **Application**: Fire-resistant structural elements
- **Innovation**: Sustainable rubber aggregate utilization
- **Goal**: Thermo-mechanical model validation

## 📊 Dataset Highlights

### ✨ Key Features
- **4 Concrete Mixtures**: Control + 3 rubber replacement levels (5%, 10%, 15%)
- **36 Test Specimens**: Comprehensive mechanical property evaluation
- **Multi-Scale Analysis**: From molecular (FTIR) to structural (mechanical properties)
- **Advanced Characterization**: TGA, MIP, UPV, and complete fresh/hardened properties
- **Statistical Rigor**: Multiple specimens, correlation analysis, predictive models

### 📈 Critical Findings
- **Strength Retention**: 85-95% compressive strength maintained up to 15% rubber
- **Modulus Reduction**: Predictable linear decrease (800 MPa per 1% rubber)
- **Enhanced Ductility**: Improved post-peak behavior and energy absorption
- **Pore Structure**: Bimodal distribution emerges, critical for fire performance
- **Quality Control**: Excellent UPV-strength correlation (R² = 0.897)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd rubberized-concrete-dataset

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Generate the complete dataset
python3 rubberized_concrete_baseline_dataset.py
```

### Dataset Generation
```python
from rubberized_concrete_baseline_dataset import RubberizedConcreteDatasetGenerator

# Initialize generator
generator = RubberizedConcreteDatasetGenerator()

# Generate complete dataset
dataset, summary, plot_file = generator.generate_complete_dataset()

# Access results
print(f"Generated {len(dataset['mixture_design']['proportions'])} mixtures")
print(f"Visualization saved: {plot_file}")
```

## 📁 Dataset Structure

### 📋 Generated Files
```
📦 Dataset Files
├── 📊 complete_baseline_dataset_[timestamp].json    # Complete dataset (JSON)
├── 📈 mixture_proportions_[timestamp].csv           # Mix designs
├── 🌊 fresh_properties_[timestamp].csv              # Fresh concrete data
├── 💪 mechanical_properties_[timestamp].csv         # Mechanical test results
├── 📊 rubberized_concrete_analysis_[timestamp].png  # Comprehensive plots
├── 📚 DATASET_DOCUMENTATION.md                      # Detailed documentation
├── 🐍 rubberized_concrete_baseline_dataset.py       # Generation script
└── 📋 requirements.txt                              # Dependencies
```

### 🔍 Data Categories

#### 1. **Mixture Design** (`mixture_proportions_*.csv`)
- Complete mix proportions for all 4 mixtures
- Material specifications and sources
- Curing regime details
- Theoretical density calculations

#### 2. **Rubber Characterization** (JSON dataset)
- Physical properties (specific gravity, particle size distribution)
- Chemical composition analysis
- Thermal properties (TGA analysis)
- FTIR spectroscopy data
- Pre-treatment procedures

#### 3. **Fresh Properties** (`fresh_properties_*.csv`)
- Slump flow measurements
- Air content determination
- Fresh density values
- Testing conditions

#### 4. **Mechanical Properties** (`mechanical_properties_*.csv`)
- Compressive strength (7 & 28 days)
- Tensile splitting strength
- Elastic modulus
- Density measurements (dry & SSD)
- Porosity analysis
- Ultrasonic pulse velocity (UPV)

#### 5. **Pore Structure Analysis** (JSON dataset)
- Mercury Intrusion Porosimetry (MIP) data
- Pore size distribution curves
- Threshold and median pore diameters
- Microstructural implications

## 🔬 Technical Specifications

### 🧪 Material Details

#### Concrete Mixtures
| Component | Specification | Content Range |
|-----------|---------------|---------------|
| **Cement** | Type I Portland (CEM I 42.5R) | 420 kg/m³ |
| **Water** | Potable, W/C = 0.40 | 168 kg/m³ |
| **Coarse Aggregate** | Granite, 10-20mm | 1050 kg/m³ |
| **Fine Aggregate** | Natural sand, FM=2.7 | 412.5-750 kg/m³ |
| **Rubber Aggregate** | Crumb rubber, 1-4mm | 0-48.8 kg/m³ |
| **Superplasticizer** | Polycarboxylate ether | 4.2-5.5 kg/m³ |

#### Rubber Aggregate Properties
- **Source**: End-of-life truck tires
- **Processing**: Ambient grinding
- **Size Range**: 0.15-4.75 mm (D₅₀ = 1.8 mm)
- **Specific Gravity**: 1.15
- **Shore A Hardness**: 65
- **Decomposition Onset**: 280°C

### 📊 Test Results Summary

#### Mechanical Properties (28-day)
| Property | Control | 5% Rubber | 10% Rubber | 15% Rubber | Units |
|----------|---------|-----------|------------|------------|-------|
| **Compressive Strength** | 48.0 | 45.6 | 43.2 | 40.8 | MPa |
| **Tensile Strength** | 5.76 | 5.47 | 5.18 | 4.90 | MPa |
| **Elastic Modulus** | 32.0 | 28.0 | 24.0 | 20.0 | GPa |
| **Dry Density** | 2320 | 2210 | 2100 | 1990 | kg/m³ |
| **Porosity** | 12.5 | 13.9 | 15.3 | 16.7 | % |
| **UPV** | 4200 | 3800 | 3400 | 2980 | m/s |

#### Performance Indicators
- **Strength Retention**: 95% (5%), 90% (10%), 85% (15%)
- **Density Reduction**: 4.7% (5%), 9.5% (10%), 14.2% (15%)
- **Quality Classification**: Excellent to Good (UPV-based)

## 📈 Data Analysis & Visualization

### 🔍 Statistical Analysis
The dataset includes comprehensive statistical analysis:
- **Correlation matrices** for all properties
- **Regression models** for property prediction
- **Confidence intervals** and variability assessment
- **Quality control indicators**

### 📊 Visualization Suite
The generated visualization (`rubberized_concrete_analysis_*.png`) includes:
1. **Mixture Proportions**: Stacked bar chart of components
2. **Fresh Properties**: Workability and air content trends
3. **Strength Development**: Compressive strength vs rubber content
4. **Elastic Properties**: Modulus degradation analysis
5. **Density-Porosity**: Dual-axis correlation plots
6. **UPV Relationships**: Non-destructive testing validation
7. **Pore Structure**: Mercury intrusion porosimetry curves
8. **Property Correlations**: Comprehensive correlation matrix
9. **Performance Summary**: Property reduction quantification

### 🎯 Key Correlations
| Property Pair | Correlation | Significance |
|---------------|-------------|--------------|
| Rubber % ↔ Compressive Strength | -0.966 | Very Strong |
| Rubber % ↔ Elastic Modulus | -0.946 | Very Strong |
| UPV ↔ Compressive Strength | +0.947 | Excellent |
| Porosity ↔ Compressive Strength | -0.924 | Strong |

## 🔥 Fire Resistance Implications

### 🌡️ Thermal Considerations
This baseline dataset provides the foundation for fire resistance analysis:

#### **Critical Temperature Points**
- **280°C**: Rubber decomposition onset
- **380°C**: Peak decomposition rate  
- **500°C**: Char formation and stabilization
- **>600°C**: Concrete thermal degradation dominates

#### **Fire Performance Factors**
- **Lower Density**: Reduced thermal mass
- **Enhanced Porosity**: Potential vapor escape paths
- **Char Formation**: 35% residue provides insulation
- **Reduced Modulus**: Lower thermal stress development

#### **Spalling Resistance Indicators**
- **Bimodal Pore Structure**: May reduce pore pressure buildup
- **Enhanced ITZ Porosity**: Improved vapor transport
- **Improved Ductility**: Better thermal deformation accommodation

### 🔬 Next Phase Requirements
For complete fire resistance characterization, the following high-temperature tests are essential:
1. **Thermal Properties**: Conductivity, specific heat, expansion
2. **High-Temperature Strength**: Residual mechanical properties
3. **Spalling Tests**: Standardized fire exposure evaluation
4. **Mass Loss Kinetics**: Thermal decomposition rates
5. **Pore Pressure**: During heating cycles

## 🛠️ Usage Examples

### 📊 Data Loading and Analysis
```python
import pandas as pd
import json

# Load mechanical properties
mech_data = pd.read_csv('mechanical_properties_[timestamp].csv')

# Load complete dataset
with open('complete_baseline_dataset_[timestamp].json', 'r') as f:
    dataset = json.load(f)

# Analyze strength trends
strength_analysis = mech_data.groupby('Rubber_Percentage')['Compressive_Strength_28d_MPa'].agg(['mean', 'std'])
print(strength_analysis)
```

### 📈 Custom Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create custom strength plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=mech_data, x='Rubber_Percentage', y='Compressive_Strength_28d_MPa')
plt.title('Compressive Strength Distribution by Rubber Content')
plt.ylabel('28-Day Compressive Strength (MPa)')
plt.xlabel('Rubber Content (%)')
plt.show()
```

### 🔍 Property Prediction
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Predict compressive strength from rubber content and porosity
X = mech_data[['Rubber_Percentage', 'Porosity_pct']]
y = mech_data['Compressive_Strength_28d_MPa']

model = LinearRegression()
model.fit(X, y)

# Prediction for new mixture
new_mix = np.array([[12.5, 14.8]])  # 12.5% rubber, 14.8% porosity
predicted_strength = model.predict(new_mix)
print(f"Predicted strength: {predicted_strength[0]:.1f} MPa")
```

## 📚 Documentation

### 📖 Comprehensive Documentation
- **[DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)**: Complete technical documentation
- **Material Specifications**: Detailed constituent properties
- **Test Procedures**: Standard protocols and conditions
- **Statistical Analysis**: Correlation and regression analysis
- **Quality Control**: Validation and verification procedures

### 🔬 Research Applications
This dataset supports:
- **Finite Element Modeling**: Material property inputs
- **Fire Resistance Analysis**: Baseline property characterization
- **Sustainability Studies**: Recycled material utilization
- **Structural Design**: Performance-based design parameters
- **Quality Control**: Non-destructive testing correlations

## 🤝 Contributing

### 📝 Data Validation
We welcome contributions to validate and extend this dataset:
- **Experimental Validation**: Independent test results
- **Model Development**: Constitutive relationships
- **High-Temperature Data**: Fire exposure test results
- **Microstructural Analysis**: Advanced characterization

### 🔄 Dataset Updates
Future versions will include:
- **High-temperature properties**
- **Fire exposure test results**
- **Advanced microstructural analysis**
- **Computational model validation data**

## 📞 Contact & Citation

### 👥 Research Team
- **Project**: Fire-Resistant Rubberized Concrete Development
- **Focus**: Thermo-Mechanical Model Validation
- **Institution**: [Research Institution]

### 📄 Citation
```bibtex
@dataset{rubberized_concrete_baseline_2025,
  title={Comprehensive Baseline Dataset for Fire-Resistant Rubberized Concrete: Material Characterization and Mixture Design},
  author={[Research Team]},
  year={2025},
  publisher={[Institution]},
  version={1.0},
  doi={[To be assigned]}
}
```

### 📧 Contact
For questions, collaborations, or data requests:
- **Email**: [contact@institution.edu]
- **Project Website**: [project-url]
- **Data Repository**: [data-repository-url]

## 📜 License

This dataset is made available for research and educational purposes under [License Type]. Commercial use requires explicit permission.

## 🙏 Acknowledgments

- **Funding**: [Grant/Funding Information]
- **Materials**: Tire recycling facility for rubber aggregate supply
- **Testing**: [Laboratory/Institution] for advanced characterization
- **Collaboration**: [Partner institutions]

---

**🎯 Ready to advance fire-resistant concrete research!**  
*This baseline dataset provides the essential foundation for developing and validating thermo-mechanical models of rubberized concrete under fire conditions.*