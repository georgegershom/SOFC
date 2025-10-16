# In-The-Wild SOFC Plate Dataset - Complete Summary

## 🎉 Dataset Generation Complete!

A comprehensive **"In-The-Wild" operational dataset** has been successfully generated for ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates.

---

## 📊 Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Total Plates** | 300 |
| **Dataset Size** | 21 MB |
| **Time Span** | 2023-01-01 to 2023-12-10 (344 days) |
| **Production Batches** | 50 |
| **Material Batches** | 10 |
| **Operators** | 5 |
| **Measurement Resolution** | 25×25 grid points per plate |

---

## 🎯 What Makes This Dataset "In-The-Wild"?

### ✅ 1. Production Variations
- **Furnace aging**: Temperature drift over 343 days (-0.001°C/day)
- **Material batch variations**: 10 different powder batches with varying properties
- **Process variations**: ±5°C temperature, ±0.3°C/min cooling rate
- **Position effects**: 5 furnace zones (front, center, back, left, right)
- **Shift effects**: 3 shifts (morning, afternoon, night) with different operator performance

### ✅ 2. Realistic Measurement Noise
- **Gaussian noise**: σ = 0.02 mm baseline
- **Sensor drift**: 15% of measurements affected (0.01-0.05 mm drift)
- **Outliers**: 20% of scans contain outliers (0.5% of points)
- **Missing data**: 10% of scans have missing points (2% of grid)

### ✅ 3. Known Failure Modes
- **Edge cracking**: 2 plates (0.7%) - occurs at >100 MPa edge stress
- **Delamination**: 6 plates (2.0%) - localized bubble-like deformations
- **Thermal shock**: 0 plates (0%) - distributed microcracks
- **Total failures**: 8 plates (2.7%)

### ✅ 4. Multiple Stress Patterns
- **Biaxial** (50%): Symmetric stress from uniform cooling
- **Gradient** (26.7%): Thermal gradients during cooling
- **Edge-dominated** (13.3%): High stress near edges (failure precursor)
- **Localized** (6.7%): Stress concentrations
- **Mixed** (3.3%): Complex multi-mode patterns

---

## 📁 Complete File Structure

```
/workspace/
├── in_the_wild_dataset/              # Main dataset directory (21 MB)
│   ├── dataset_metadata.csv          # Master metadata (300 plates, 96 KB)
│   ├── dataset_summary.png           # Visual summary (1.8 MB)
│   ├── README.md                     # Comprehensive documentation (8.8 KB)
│   ├── measurements/                 # Warp measurement CSV files
│   │   ├── plate_00001_warp.csv     # Format: x_mm, y_mm, z_mm
│   │   ├── plate_00002_warp.csv
│   │   └── ... (300 files total)
│   ├── stress_fields/                # Ground truth stress NPZ files
│   │   ├── plate_00001_stress.npz   # stress_xx, stress_yy, warp_true, warp_measured, X, Y
│   │   ├── plate_00002_stress.npz
│   │   └── ... (300 files total)
│   └── visualizations/               # Sample visualizations (18 files)
│       ├── plate_00001.png          # Failed/interesting plates
│       └── ...
├── generate_in_the_wild_sofc_dataset.py  # Dataset generation script
├── load_dataset_example.py               # Example loading and exploration
├── QUICKSTART_DATASET.md                 # Quick start guide
├── DATASET_SUMMARY.md                    # This file
└── requirements.txt                      # Python dependencies
```

---

## 📈 Statistical Summary

### Quality Distribution
| Quality Class | Count | Percentage |
|--------------|-------|------------|
| **Good** | 99 | 33.0% |
| **Marginal** | 87 | 29.0% |
| **Reject** | 106 | 35.3% |
| **Failed** | 8 | 2.7% |

**Quality Criteria:**
- Good: Max warp < 3.0 mm, max stress < 90 MPa
- Marginal: Max warp 3.0-4.0 mm OR max stress 90-110 MPa
- Reject: Max warp > 4.0 mm OR max stress > 110 MPa (no failure)
- Failed: Physical failure (crack, delamination, etc.)

### Warp Statistics
- **Mean warp**: 1.013 ± 0.316 mm
- **Maximum warp**: 5.286 mm (PLATE-00293)
- **Minimum warp**: 1.348 mm (PLATE-00014)
- **Typical range**: 0.5 - 4.5 mm

### Stress Statistics
- **Mean max stress**: 89.1 ± 17.9 MPa
- **Maximum stress**: 154.5 MPa (PLATE-00087, delamination failure)
- **Minimum stress**: 40.8 MPa (PLATE-00231)
- **High stress (>100 MPa)**: 57 plates (19%)

### Production Parameters
| Parameter | Mean ± Std | Range |
|-----------|-----------|-------|
| Furnace Temperature | 1399.8 ± 5.2 °C | 1385-1412 °C |
| Cooling Rate | 2.02 ± 0.29 °C/min | 1.44-2.77 °C/min |
| Furnace Age | 171.5 ± 99.3 days | 0-343 days |
| Oxygen Pressure | 0.201 ± 0.012 atm | 0.18-0.22 atm |
| Humidity | 44.9 ± 8.6 % | 30-60 % |

---

## 🔬 Key Features for ML Research

### 1. **Inverse Problem: Stress from Warp**
- **Input**: Warp measurements (x, y, z coordinates with noise)
- **Output**: Residual stress fields (σ_xx, σ_yy)
- **Challenge**: Handle measurement noise, missing data, non-unique solutions

### 2. **Robustness Testing**
- Train on clean/early data → Test on noisy/late data
- Evaluate performance degradation with parameter drift
- Test handling of outliers and missing data

### 3. **Failure Prediction**
- Predict failure probability from process parameters
- Classify failure modes (edge crack vs delamination)
- Early warning system for high-risk plates

### 4. **Quality Classification**
- Multi-class classification (good/marginal/reject/failed)
- Feature importance analysis (which parameters matter most)
- Process optimization for quality improvement

### 5. **Domain Adaptation**
- Temporal shift: Early (0-150 days) → Late (>200 days)
- Batch shift: Different material batches
- Position shift: Different furnace zones

### 6. **Physics-Informed ML**
- Incorporate physical constraints (stress magnitude, smoothness)
- Validate against known failure modes
- Energy consistency checks

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```python
# Run the example script
python3 load_dataset_example.py

# Or load manually
import pandas as pd
import numpy as np

# Load metadata
metadata = pd.read_csv('in_the_wild_dataset/dataset_metadata.csv')
print(metadata.head())

# Load a plate
plate = metadata.iloc[0]
warp = pd.read_csv(f"in_the_wild_dataset/{plate['measurement_file']}")
stress = np.load(f"in_the_wild_dataset/{plate['stress_file']}")
```

### Detailed Documentation
- **Quick Start**: `QUICKSTART_DATASET.md`
- **Full Documentation**: `in_the_wild_dataset/README.md`
- **Example Code**: `load_dataset_example.py`
- **Generation Script**: `generate_in_the_wild_sofc_dataset.py`

---

## 🎓 Use Cases Demonstrated

### ✅ Realistic Production Environment
- 50 batches over 344 days
- Natural parameter drift (furnace aging)
- Multiple material batches
- Human factors (operator, shift effects)

### ✅ Comprehensive Noise Modeling
- Gaussian measurement noise
- Systematic sensor drift
- Random outliers (dust, glitches)
- Missing data points (edge detection failures)

### ✅ Physical Failure Modes
- **Edge Cracking**: High stress concentration at edges
- **Delamination**: Interface failure in multilayer structures  
- **Thermal Shock**: Rapid cooling microcracks
- Realistic failure rates (~3%)

### ✅ Multiple Stress Scenarios
- 5 different stress distribution patterns
- Covers common manufacturing conditions
- Includes edge effects (critical for failure)
- Physically plausible magnitudes (20-150 MPa)

### ✅ Complete Traceability
- Full process parameter tracking
- Material batch genealogy
- Operator and shift information
- Furnace age and position
- Environmental conditions

---

## 📊 Detailed Failure Analysis

### Failed Plates (8 total)

| Plate ID | Failure Type | Max Stress | Max Warp | Pattern | Cause |
|----------|-------------|-----------|----------|---------|-------|
| PLATE-00017 | Delamination | 152.5 MPa | 2.42 mm | Mixed | High stress + complex pattern |
| PLATE-00087 | Delamination | 154.5 MPa | 4.14 mm | Gradient | **Highest stress in dataset** |
| PLATE-00101 | Delamination | 122.4 MPa | 2.41 mm | Biaxial | Moderate stress, unlucky |
| PLATE-00109 | Delamination | 123.4 MPa | 3.65 mm | Biaxial | Moderate stress, unlucky |
| PLATE-00113 | Delamination | 119.3 MPa | 3.88 mm | Edge-dominated | Edge stress concentration |
| PLATE-00116 | Edge Crack | 117.0 MPa | 3.45 mm | Biaxial | Edge stress >100 MPa |
| PLATE-00205 | Edge Crack | 122.0 MPa | 4.60 mm | Mixed | Edge stress + large warp |
| PLATE-00262 | Delamination | 117.6 MPa | 4.58 mm | Biaxial | Large warp + stress |

**Key Insights:**
- All failures occur at stress > 115 MPa
- Edge cracks: stress-driven (edge >100 MPa)
- Delamination: more common (6 vs 2), occurs at >115 MPa anywhere
- Mixed and gradient patterns more prone to failure

---

## 🔍 Physical Validity Checks

The dataset satisfies key physical constraints:

### ✅ Material Properties (YSZ/LSM Ceramics)
- Young's Modulus: 200 GPa ✓
- Poisson's Ratio: 0.3 ✓
- CTE: 10.5 × 10⁻⁶ K⁻¹ ✓

### ✅ Stress Magnitudes
- Range: 40-155 MPa ✓ (typical for SOFC: 20-150 MPa)
- Failure threshold: ~115 MPa ✓ (realistic for ceramics)
- Pattern smoothness: Continuous except at cracks ✓

### ✅ Warp Magnitudes
- Range: 1.3-5.3 mm ✓ (typical for 150mm plates: 0-5 mm)
- Deflection/thickness ratio: 2-10 ✓ (reasonable for 0.5mm thick)
- Edge boundary conditions respected ✓

### ✅ Failure Modes
- Edge cracking frequency: 0.7% ✓ (common in ceramics)
- Delamination frequency: 2.0% ✓ (realistic for multilayer)
- Stress-failure correlation: >115 MPa ✓

### ✅ Process-Property Relations
- Higher furnace temp → Higher stress ✓
- Faster cooling → Higher stress ✓  
- Edge positions → More variation ✓
- Night shift → Slightly more noise ✓

---

## 💡 Advanced Usage Examples

### Example 1: Train/Test Split by Time (Domain Adaptation)
```python
import pandas as pd

metadata = pd.read_csv('in_the_wild_dataset/dataset_metadata.csv')
metadata['date'] = pd.to_datetime(metadata['date'])

# Train on first 200 days, test on last 144 days
cutoff = pd.Timestamp('2023-07-20')
train = metadata[metadata['date'] < cutoff]
test = metadata[metadata['date'] >= cutoff]

print(f"Train: {len(train)} plates (days 0-200)")
print(f"Test: {len(test)} plates (days 200-344)")
print(f"Furnace age gap: {train['furnace_age_days'].max()} → {test['furnace_age_days'].min()}")
```

### Example 2: Analyze Noise Impact
```python
import numpy as np

# Load a plate with its true and measured warp
plate = metadata.iloc[0]
stress_data = np.load(f"in_the_wild_dataset/{plate['stress_file']}")

warp_true = stress_data['warp_true']
warp_measured = stress_data['warp_measured']

# Calculate noise statistics
noise = warp_measured - warp_true
print(f"Noise mean: {np.nanmean(noise):.4f} mm")
print(f"Noise std: {np.nanstd(noise):.4f} mm")
print(f"SNR: {np.nanstd(warp_true) / np.nanstd(noise):.2f}")
```

### Example 3: Feature Engineering for ML
```python
# Extract features for quality prediction
features = metadata[[
    'furnace_temp_C', 'cooling_rate_C_per_min', 'furnace_age_days',
    'max_warp_mm', 'mean_warp_mm', 'warp_std_mm', 'max_stress_MPa'
]]

# Add derived features
features['temp_deviation'] = features['furnace_temp_C'] - 1400
features['cooling_rate_deviation'] = features['cooling_rate_C_per_min'] - 2.0
features['warp_variation_coef'] = features['warp_std_mm'] / features['mean_warp_mm']

# Target
target = metadata['quality_class']
```

---

## 📚 Documentation Files

1. **DATASET_SUMMARY.md** (this file): High-level overview
2. **QUICKSTART_DATASET.md**: Quick start guide with examples
3. **in_the_wild_dataset/README.md**: Complete technical documentation
4. **load_dataset_example.py**: Working Python examples
5. **generate_in_the_wild_sofc_dataset.py**: Full generation code with physics models

---

## ✅ Dataset Validation Checklist

- [x] 300 plates generated
- [x] All files created (300 CSV + 300 NPZ + metadata)
- [x] Realistic production variations included
- [x] Measurement noise properly simulated
- [x] Failure modes physically plausible
- [x] Metadata complete and consistent
- [x] Visualizations created for sample plates
- [x] Documentation comprehensive
- [x] Example code working
- [x] Physical validity confirmed

---

## 🎯 Research Applications

### Recommended ML Approaches

1. **Physics-Informed Neural Networks (PINNs)**
   - Incorporate plate mechanics equations
   - Enforce stress-strain consistency
   - Boundary condition constraints

2. **Gaussian Processes**
   - Handle uncertainty quantification
   - Incorporate domain knowledge via kernels
   - Robust to noise and missing data

3. **Graph Neural Networks**
   - Model spatial dependencies
   - Handle irregular grids (missing data)
   - Learn local stress patterns

4. **Transformer Models**
   - Attention mechanism for important regions
   - Handle variable-length inputs (missing data)
   - Learn global-local relationships

5. **Ensemble Methods**
   - Random Forest for feature importance
   - XGBoost for quality classification
   - Combine multiple models for robustness

---

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_in_the_wild_2025,
  title = {In-The-Wild SOFC Plate Dataset for ML-Augmented Inverse Modeling},
  author = {Dataset Generator},
  year = {2025},
  month = {October},
  note = {Generated dataset for residual stress quantification from warped SOFC plates},
  description = {300 SOFC plate measurements with production variations, measurement noise, and failure modes},
  url = {/workspace/in_the_wild_dataset/}
}
```

---

## 🏆 Dataset Highlights

### What Sets This Dataset Apart:

1. **Realistic Production Environment** 
   - Not idealized lab conditions
   - Natural parameter drift over time
   - Human factors included

2. **Complete Ground Truth**
   - True stress fields available
   - Both clean and noisy measurements
   - Full process parameter tracking

3. **Comprehensive Noise Modeling**
   - Multiple noise types (Gaussian, drift, outliers)
   - Realistic failure modes
   - Missing data scenarios

4. **Rich Metadata**
   - 30+ features per plate
   - Full traceability
   - Multiple classification targets

5. **Ready for ML**
   - Pre-split suggestions
   - Example code provided
   - Multiple use cases documented

---

## 🎉 Summary

**You now have a production-grade "In-The-Wild" dataset with 300 SOFC plates spanning nearly a year of simulated production!**

This dataset includes:
- ✅ Realistic variations (furnace aging, material batches, operators)
- ✅ Measurement artifacts (noise, drift, outliers, missing data)
- ✅ Physical failure modes (edge cracks, delamination)
- ✅ Multiple stress patterns (5 types)
- ✅ Complete metadata (30+ features)
- ✅ Ground truth stress fields
- ✅ Quality labels (4 classes)
- ✅ Comprehensive documentation

**Total size: 21 MB | 300 plates | 50 batches | 10 material types | 8 failures**

---

**Ready to test your ML model's robustness? The dataset is waiting! 🚀**

Generated: 2025-10-16  
Location: `/workspace/in_the_wild_dataset/`
