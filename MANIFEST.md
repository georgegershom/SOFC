# Context Dataset for Residual Stress Prediction - Complete Package

**Generated:** 2025-10-15  
**Version:** 1.0  
**Purpose:** Predictive modeling of residual stress in multi-layer ceramic structures

---

## 📦 Package Contents

### 🎯 Primary Datasets (4 sizes)

Each size includes 4 files:

#### **Small Dataset (100 samples)**
- `context_dataset_small.csv` - Main dataset (149 KB)
- `context_dataset_small.xlsx` - Excel with multiple sheets (101 KB)
- `context_dataset_small_metadata.json` - Detailed metadata (4.9 KB)
- `context_dataset_small_data_dictionary.txt` - Human-readable documentation (7.7 KB)

#### **Medium Dataset (1,000 samples)** ⭐ Recommended for development
- `context_dataset_medium.csv` - Main dataset (1.5 MB)
- `context_dataset_medium.xlsx` - Excel with multiple sheets (870 KB)
- `context_dataset_medium_metadata.json` - Detailed metadata (4.9 KB)
- `context_dataset_medium_data_dictionary.txt` - Human-readable documentation (7.7 KB)

#### **Large Dataset (5,000 samples)**
- `context_dataset_large.csv` - Main dataset (7.2 MB)
- `context_dataset_large.xlsx` - Excel with multiple sheets (4.3 MB)
- `context_dataset_large_metadata.json` - Detailed metadata (4.9 KB)
- `context_dataset_large_data_dictionary.txt` - Human-readable documentation (7.7 KB)

#### **XLarge Dataset (10,000 samples)** ⭐ Recommended for production
- `context_dataset_xlarge.csv` - Main dataset (15 MB)
- `context_dataset_xlarge.xlsx` - Excel with multiple sheets (8.5 MB)
- `context_dataset_xlarge_metadata.json` - Detailed metadata (4.9 KB)
- `context_dataset_xlarge_data_dictionary.txt` - Human-readable documentation (7.7 KB)

---

### 📚 Documentation

- **`DATASET_README.md`** - Comprehensive documentation (13 KB)
  - Dataset overview and features
  - Parameter descriptions and ranges
  - Usage examples
  - Integration with FEA
  - Physical insights

- **`MANIFEST.md`** - This file
  - Package contents
  - Quick start guide
  - File descriptions

---

### 🔧 Code & Scripts

- **`generate_context_dataset.py`** (23 KB)
  - Main dataset generation script
  - Latin Hypercube Sampling implementation
  - Temperature profile generation
  - Derived parameter calculation
  - Quality flag assignment

- **`sample_dataset_analysis.py`** (21 KB)
  - Exploratory data analysis
  - Visualization generation
  - Statistical analysis
  - ML preparation utilities

- **`requirements.txt`** (354 B)
  - Python package dependencies
  - Version specifications

---

### 📊 Generated Visualizations

Results from running `sample_dataset_analysis.py`:

- **`analysis_cte_mismatch.png`** (706 KB)
  - CTE mismatch distributions
  - Interface relationships
  - Temperature dependence

- **`analysis_material_properties.png`** (2.0 MB)
  - Young's modulus distributions
  - Stiffness ratios
  - Sintering parameters
  - Creep properties

- **`analysis_process_parameters.png`** (1.3 MB)
  - Sintering temperature profiles
  - Heating/cooling rates
  - Process time distributions

- **`analysis_geometric_parameters.png`** (1.5 MB)
  - Layer thickness distributions
  - Thickness ratios
  - Plate dimensions

- **`analysis_temperature_profiles.png`** (1.7 MB)
  - Sample thermal cycles
  - Parameter space coverage

- **`analysis_correlation_heatmap.png`** (482 KB)
  - Feature correlations
  - Key relationships

---

### 📋 Analysis Outputs

- **`dataset_analysis_report.txt`** (1.5 KB)
  - Summary statistics
  - Key findings
  - Quality metrics

- **`ml_feature_names.txt`** (1.8 KB)
  - List of all features for ML
  - Feature engineering reference

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Datasets (Already Done!)

The datasets have been pre-generated. To regenerate:

```bash
python3 generate_context_dataset.py
```

### 3. Explore the Data

```bash
# Run comprehensive analysis
python3 sample_dataset_analysis.py
```

This generates:
- 6 visualization plots
- Statistical summary report
- ML-ready feature list

### 4. Load in Python

```python
import pandas as pd
import json

# Load dataset
df = pd.read_csv('context_dataset_medium.csv')

# Load metadata
with open('context_dataset_medium_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
```

### 5. Prepare for ML

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('context_dataset_xlarge.csv')

# Select features (exclude non-feature columns)
exclude = ['sample_id', 'generation_timestamp', 'temperature_profile_json']
feature_cols = [col for col in df.columns if col not in exclude]

# Separate numeric and categorical
X = df[feature_cols].copy()

# One-hot encode atmosphere
X = pd.get_dummies(X, columns=['atmosphere_type'], drop_first=True)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Ready for ML! Shape: {X_scaled.shape}")
```

### 6. Pair with FEA Results

```python
# Your workflow:
# 1. Use dataset as DOE inputs for FEA
# 2. Run simulations for each sample_id
# 3. Extract stress/strain outputs
# 4. Merge results

# Example merge
fea_results = pd.read_csv('your_fea_results.csv')
full_dataset = df.merge(fea_results, on='sample_id')

# Now you have: X (context) -> y (stress)
```

---

## 📊 Dataset Specifications

### Features: 65 Total

| Category | Count | Description |
|----------|-------|-------------|
| Geometric | 7 | Dimensions, thicknesses, green state |
| Anode Properties | 11 | Ni-YSZ material parameters |
| Electrolyte Properties | 11 | YSZ/GDC material parameters |
| Cathode Properties | 11 | LSM/LSCF material parameters |
| Process Parameters | 10 | Sintering profiles, atmosphere |
| Derived Parameters | 11 | CTE mismatches, ratios |
| Quality Flags | 4 | Risk indicators |

### Sampling Method

**Latin Hypercube Sampling (LHS)**
- Space-filling design
- Better than random for metamodeling
- Uniform parameter space coverage
- Reproducible (seed=42)

---

## 🎯 Key Features

### 1. Temperature-Dependent Properties
- Young's Modulus at 25°C and 1000°C
- CTE at 25°C and 1000°C
- Enables accurate thermal stress modeling

### 2. Complete Sintering Profiles
- JSON-encoded temperature-time curves
- Heating, hold, cooling segments
- Optional intermediate holds
- Multiple atmosphere types

### 3. Derived Physical Parameters
- CTE mismatches (primary stress driver)
- Stiffness ratios (load distribution)
- Thickness ratios (bending vs membrane)
- Process time totals

### 4. Quality Flags
- Extreme CTE mismatch (>3.5 ppm/K)
- Thin electrolyte (<10 μm)
- Fast cooling (>6 °C/min)
- Asymmetric structure

---

## 📈 Dataset Statistics (XLarge)

```
Samples: 10,000
Features: 65
Memory: ~10 MB

CTE Mismatch: 0.02 - 4.14 ppm/K
Sintering Temp: 1300 - 1500 °C
Total Thickness: 332.9 - 1134.1 μm

Atmosphere Distribution:
  Air: 48.5%
  Reducing (H₂/N₂): 19.8%
  Argon: 15.6%
  Nitrogen: 10.6%
  Vacuum: 5.5%
```

---

## 🔬 Physical Basis

### Material Systems

**Anode: Ni-YSZ (Nickel - Yttria Stabilized Zirconia)**
- Young's Modulus: 40-80 GPa (25°C)
- CTE: 10.5-13.5 ppm/K (25°C)
- Sintering Onset: 1100-1250°C

**Electrolyte: 8-YSZ (8 mol% Y₂O₃ stabilized ZrO₂)**
- Young's Modulus: 180-220 GPa (25°C)
- CTE: 9.5-11.5 ppm/K (25°C)
- Sintering Onset: 1200-1350°C

**Cathode: LSM/LSCF (Perovskite oxides)**
- Young's Modulus: 50-100 GPa (25°C)
- CTE: 11.0-14.0 ppm/K (25°C)
- Sintering Onset: 1000-1200°C

### Residual Stress Mechanisms

1. **CTE Mismatch** (Primary)
   - Differential thermal contraction
   - Stress ∝ ΔT × ΔCTE × E

2. **Sintering Shrinkage**
   - Differential densification
   - Constraint effects

3. **Creep Relaxation**
   - High-temperature stress relief
   - Time and temperature dependent

4. **Geometry Effects**
   - Bending vs membrane stress
   - Edge effects

---

## 💡 Use Cases

### 1. Predictive Modeling
Train ML models to predict residual stress from process parameters:
```
X (context) → Model → y (stress field)
```

### 2. Design Optimization
Use trained models for:
- Minimize warpage
- Reduce peak stress
- Optimize layer thicknesses
- Design thermal cycles

### 3. Sensitivity Analysis
Identify critical parameters:
- Which parameter most affects stress?
- Interaction effects?
- Design margins?

### 4. Process Window Definition
Find acceptable operating ranges:
- Maximum cooling rate?
- Minimum electrolyte thickness?
- Optimal temperature profile?

---

## ⚠️ Important Notes

1. **Synthetic Data**: Based on realistic material property ranges from literature
2. **No FEA Results**: You must generate stress outputs via simulation
3. **Material Combinations**: Some randomly generated combinations may be unrealistic
4. **Temperature Interpolation**: Linear interpolation between 25°C and 1000°C recommended

---

## 📚 Recommended Workflow

```
1. Load Context Dataset
   ↓
2. Run FEA Simulations
   - Use parameters as inputs
   - Extract stress/strain/warpage
   ↓
3. Merge Results
   - Match by sample_id
   - Create X → y pairs
   ↓
4. Train ML Models
   - Regression (stress prediction)
   - Classification (failure prediction)
   - Multi-output (full field)
   ↓
5. Optimize Design
   - Use model for "what-if" analysis
   - Identify optimal parameters
   - Validate with FEA
```

---

## 🔧 Technical Details

### Dataset Generation
- **Language**: Python 3.13
- **Sampling**: scipy.stats.qmc.LatinHypercube
- **Random Seed**: 42 (reproducible)
- **Generation Time**: ~30 seconds for all 4 datasets

### Dependencies
- numpy >= 1.23
- pandas >= 2.0
- scipy >= 1.10
- openpyxl >= 3.1

### File Formats
- **CSV**: Universal compatibility
- **Excel**: Multiple sheets (data + metadata)
- **JSON**: Machine-readable metadata
- **TXT**: Human-readable documentation

---

## 📧 Support

For questions or issues:
1. Read `DATASET_README.md` for detailed documentation
2. Check `*_data_dictionary.txt` for parameter descriptions
3. Review `metadata.json` for technical specifications
4. Run `sample_dataset_analysis.py` for exploration

---

## 📄 Citation

If using this dataset in research, please cite:

```
Context Dataset for Residual Stress Prediction in Multi-layer Ceramic Structures
Version 1.0, Generated 2025-10-15
Sampling Method: Latin Hypercube Sampling
```

---

## ✅ Verification Checklist

- [x] 4 dataset sizes generated (100, 1K, 5K, 10K samples)
- [x] 65 features per sample
- [x] Latin Hypercube Sampling implemented
- [x] Temperature-dependent properties included
- [x] Complete sintering profiles (JSON)
- [x] Derived parameters calculated
- [x] Quality flags assigned
- [x] Multiple file formats (CSV, Excel, JSON)
- [x] Comprehensive documentation
- [x] Sample analysis script
- [x] Visualization generation
- [x] ML preparation utilities
- [x] Requirements file
- [x] No missing values
- [x] Realistic parameter ranges

---

**🎉 Dataset Complete and Ready for Use!**

**Total Package Size:** ~60 MB (all datasets + docs + visualizations)

**Recommended Next Step:** Run FEA simulations using these parameters to generate stress outputs, then train predictive ML models.

---

*Generated with ❤️ for residual stress prediction in advanced ceramic structures*
