# SOFC Experimental Validation Dataset

## Dataset Overview

**Title:** ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates - Experimental Validation Dataset

**Version:** 1.0

**Purpose:** This dataset provides comprehensive experimental validation data for validating machine learning models that predict residual stress fields in Solid Oxide Fuel Cell (SOFC) plates from warp measurements.

**Dataset Type:** Dataset 3 - "The Reality Check"

---

## 🎯 Key Features

- **35 fabricated SOFC samples** with varied parameters (20-50 recommended range)
- **Multi-technique stress measurements** (5 different methods)
- **High-resolution 3D warp data** (128×128 point clouds)
- **Realistic experimental uncertainties** and noise
- **Complete fabrication history** for each sample
- **Publication-ready visualizations**

---

## 📊 Dataset Components

### 1. Fabrication Parameters (`fabrication_parameters.csv`)

**35 samples** covering extremes and center of parameter space using Latin Hypercube Sampling.

#### Parameters:

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| `anode_thickness_um` | 300-800 | μm | NiO-YSZ anode layer thickness |
| `electrolyte_thickness_um` | 5-20 | μm | YSZ electrolyte thickness |
| `cathode_thickness_um` | 20-80 | μm | LSM cathode thickness |
| `anode_sinter_temp_C` | 1350-1450 | °C | Anode sintering temperature |
| `anode_sinter_time_h` | 2-6 | hours | Anode sintering duration |
| `electrolyte_sinter_temp_C` | 1400-1500 | °C | Electrolyte sintering temperature |
| `electrolyte_sinter_time_h` | 2-5 | hours | Electrolyte sintering duration |
| `cathode_sinter_temp_C` | 1100-1200 | °C | Cathode sintering temperature |
| `cathode_sinter_time_h` | 1-4 | hours | Cathode sintering duration |
| `cooling_rate_C_per_min` | 2-10 | °C/min | Cooling rate after sintering |

#### Derived Parameters:
- `total_thickness_um`: Sum of all layer thicknesses
- `electrolyte_anode_ratio`: Thickness ratio
- `cathode_anode_ratio`: Thickness ratio
- `sample_id`: Unique identifier (e.g., SOFC-EXP-001)
- `batch_id`: Fabrication batch identifier
- `fabrication_date`: Sample fabrication date

---

### 2. Warp Measurements (`warp_measurements_3d.h5`)

**Technique:** White Light Interferometry / Laser Scanning Confocal Microscopy

**All 35 samples** measured with high-resolution 3D surface topology.

#### Data Structure (HDF5):
```
/SOFC-EXP-001/
  ├── X (128×128 array)       # X coordinates (m)
  ├── Y (128×128 array)       # Y coordinates (m)
  ├── Z (128×128 array)       # Z displacement (μm)
  └── Attributes:
      ├── max_warp_um         # Peak-to-valley warp
      ├── rms_warp_um         # RMS warp
      ├── technique           # Measurement method
      ├── resolution_um       # Spatial resolution
      └── noise_level_um      # Measurement noise (≈0.05 μm)
```

#### Specifications:
- **Sample size:** 50 mm × 50 mm
- **Resolution:** 128 × 128 points (≈0.4 mm spacing)
- **Measurement noise:** ≈0.05 μm RMS (realistic for interferometry)
- **Edge uncertainty:** Higher near sample edges
- **Warp range:** Typically 5-50 μm (depends on thermal mismatch)

---

### 3. Curvature-Based Stress (`curvature_stress_measurements.csv`)

**Technique:** Modified Stoney's Formula (Curvature-Based Inverse Method)

**35 samples** - Non-destructive, through-thickness average stress.

#### Measurements:

| Column | Unit | Description |
|--------|------|-------------|
| `anode_stress_GPa` | GPa | Average anode stress |
| `electrolyte_stress_GPa` | GPa | Average electrolyte stress |
| `cathode_stress_GPa` | GPa | Average cathode stress |
| `curvature_x_1_per_m` | 1/m | Curvature in X direction |
| `curvature_y_1_per_m` | 1/m | Curvature in Y direction |
| `uncertainty_percent` | % | Measurement uncertainty (≈12%) |

#### Method:
Based on bilayer bending theory. Curvature measured from warp field is used to calculate through-thickness average stress in each layer.

**Advantages:**
- Non-destructive
- Fast measurement
- Established technique

**Limitations:**
- Only through-thickness average (no gradient)
- Assumes uniform stress in each layer
- ±10-15% uncertainty typical

---

### 4. Layer Removal Stress Profiles (`layer_removal_stress_profiles.h5`)

**Technique:** Sequential Layer Milling + Inverse Analysis

**15 samples** (subset) - Destructive, provides through-thickness stress gradient.

#### Data Structure:
```
/SOFC-EXP-001/
  ├── z_positions_um          # Through-thickness positions
  ├── stress_profile_GPa      # Stress at each position
  ├── removal_depths_um       # Milling depths
  ├── warp_after_removal_um   # Warp change after each step
  └── Attributes:
      ├── technique
      ├── n_steps             # Number of removal steps
      └── uncertainty_GPa     # ±0.05 GPa typical
```

#### Method:
1. Measure initial warp
2. Mill away layer incrementally (typically 5 steps)
3. Measure warp after each removal
4. Use inverse FEA to back-calculate original stress profile

**Advantages:**
- Provides through-thickness stress gradient
- Direct measurement of residual stress relief

**Limitations:**
- Destructive (sample destroyed)
- Time-consuming (hours per sample)
- Requires careful milling control

---

### 5. XRD Stress Measurements (`xrd_stress_measurements.csv`)

**Technique:** X-Ray Diffraction (sin²ψ method)

**20 samples**, **~25 points per sample** = **~500 measurement points**

#### Measurements:

| Column | Unit | Description |
|--------|------|-------------|
| `x_position_mm` | mm | X coordinate on surface |
| `y_position_mm` | mm | Y coordinate on surface |
| `stress_xx_GPa` | GPa | In-plane stress (X direction) |
| `stress_yy_GPa` | GPa | In-plane stress (Y direction) |
| `stress_xy_GPa` | GPa | In-plane shear stress |
| `peak_used` | - | XRD peak (e.g., YSZ_311) |
| `uncertainty_GPa` | GPa | ±0.03 GPa typical |
| `measurement_time_min` | min | 15-30 min per point |

#### Method:
Measures lattice strain in crystalline materials. Strain converted to stress using elastic constants.

**Advantages:**
- Direct stress measurement
- Point-wise measurements
- Non-destructive (surface only)

**Limitations:**
- Only surface measurements (~5-10 μm depth)
- Requires crystalline materials
- Time-consuming
- Localized measurements only

---

### 6. Raman Spectroscopy Stress (`raman_stress_measurements.csv`)

**Technique:** Raman Peak Shift Method

**15 samples**, **~50 points per sample** = **~750 measurement points**

#### Measurements:

| Column | Unit | Description |
|--------|------|-------------|
| `x_position_mm` | mm | X coordinate |
| `y_position_mm` | mm | Y coordinate |
| `peak_wavenumber_cm_inv` | cm⁻¹ | Raman peak position |
| `peak_width_cm_inv` | cm⁻¹ | Peak width (stress-sensitive) |
| `stress_estimate_GPa` | GPa | Estimated stress |
| `stress_coefficient_cm_inv_per_GPa` | cm⁻¹/GPa | Calibration constant (≈5.2) |
| `uncertainty_GPa` | GPa | ±0.08 GPa typical |
| `laser_wavelength_nm` | nm | 532 nm (green laser) |
| `spot_size_um` | μm | 1 μm (high resolution) |

#### Method:
Stress causes shift in Raman-active vibrational modes. Peak shift correlates with stress state.

**Advantages:**
- Higher spatial resolution than XRD (1 μm spot)
- No special sample preparation
- Can measure multiple layers

**Limitations:**
- Requires Raman-active materials
- Indirect stress measurement (requires calibration)
- Surface measurements only
- Higher uncertainty than XRD

---

## 🔬 Experimental Techniques Comparison

| Technique | Samples | Coverage | Resolution | Uncertainty | Time/Sample | Destructive? |
|-----------|---------|----------|------------|-------------|-------------|--------------|
| **Warp (WLI)** | 35 | Full surface | 0.4 mm | 0.05 μm | 5 min | No |
| **Curvature** | 35 | Through-thickness avg | - | 12% | 10 min | No |
| **Layer Removal** | 15 | Through-thickness profile | - | 0.05 GPa | 3-5 hours | Yes |
| **XRD** | 20 | Surface points | 1 mm spot | 0.03 GPa | 20 min/point | No |
| **Raman** | 15 | Surface mapping | 1 μm spot | 0.08 GPa | 10 min/point | No |

---

## 📁 File Structure

```
experimental_validation_dataset/
│
├── fabrication_parameters.csv              # Fabrication history
├── curvature_stress_measurements.csv       # Stoney's formula results
├── xrd_stress_measurements.csv             # XRD point measurements
├── raman_stress_measurements.csv           # Raman spectroscopy data
│
├── warp_measurements_3d.h5                 # 3D warp point clouds (HDF5)
├── layer_removal_stress_profiles.h5        # Through-thickness profiles (HDF5)
│
├── dataset_metadata.json                   # Dataset information
├── dataset_summary.json                    # Quick statistics
└── validation_report.json                  # Quality assurance results
│
└── visualizations/                         # Auto-generated plots
    ├── parameter_space_coverage.png
    ├── warp_measurements_3d.png
    ├── warp_contour_maps.png
    ├── stress_measurements_comparison.png
    ├── layer_removal_stress_profiles.png
    ├── measurement_uncertainty_analysis.png
    └── correlation_matrix.png
```

---

## 🚀 Quick Start

### 1. Generate Dataset

```bash
python generate_experimental_validation_dataset.py
```

This creates all dataset files with 35 samples (takes ~30 seconds).

### 2. Visualize Dataset

```bash
python visualize_experimental_dataset.py
```

Generates comprehensive publication-ready visualizations.

### 3. Validate Dataset

```bash
python validate_dataset.py
```

Runs quality assurance checks on the dataset.

### 4. Load Data (Python)

```python
import pandas as pd
import h5py

# Load fabrication parameters
fab_params = pd.read_csv('experimental_validation_dataset/fabrication_parameters.csv')

# Load warp data
with h5py.File('experimental_validation_dataset/warp_measurements_3d.h5', 'r') as f:
    sample = f['SOFC-EXP-001']
    X = sample['X'][:]
    Y = sample['Y'][:]
    Z = sample['Z'][:]  # Warp field
    
# Load stress measurements
stress = pd.read_csv('experimental_validation_dataset/curvature_stress_measurements.csv')
```

---

## 🎓 Usage in ML-Augmented Inverse Modeling Pipeline

### Step 1: Train ML Model on FEA Data
Use Dataset 1 (High-Fidelity FEA) and Dataset 2 (Synthetic Diverse) to train your ML model:
- **Input:** Warp field (3D displacement)
- **Output:** Residual stress field (3D tensor)

### Step 2: Validate on Experimental Data (This Dataset)
Use the **warp measurements** from this dataset as input to your trained model:

```python
# Load experimental warp
warp_experimental = load_warp_data('SOFC-EXP-001')

# Predict stress using ML model
stress_predicted = ml_model.predict(warp_experimental)
```

### Step 3: Compare with Experimental Stress Measurements

Compare ML predictions against:
1. **Curvature-based stress** (quick validation, all samples)
2. **Layer removal profiles** (through-thickness gradient validation)
3. **XRD measurements** (localized validation)
4. **Raman measurements** (high-resolution spatial validation)

### Step 4: Quantify Model Performance

```python
# Example validation metrics
rmse = calculate_rmse(stress_predicted, stress_experimental)
relative_error = calculate_relative_error(stress_predicted, stress_experimental)
correlation = calculate_correlation(stress_predicted, stress_experimental)
```

### Step 5: Identify Model Shortcomings

Discrepancies between ML predictions and experimental measurements indicate:
- **Physics missing from FEA model** (e.g., grain boundary effects, microstructure)
- **ML model generalization issues** (training-test domain gap)
- **Measurement uncertainties** (must account for experimental noise)

---

## 📊 Dataset Statistics

### Parameter Coverage:
- **Anode thickness:** 300-800 μm (uniform sampling)
- **Electrolyte thickness:** 5-20 μm
- **Cathode thickness:** 20-80 μm
- **Sintering temperatures:** 1100-1500°C
- **Cooling rates:** 2-10°C/min

### Measurement Summary:
- **Total samples:** 35
- **Warp measurements:** 35 (all samples)
- **Curvature stress:** 35 (all samples)
- **Layer removal profiles:** 15 (subset)
- **XRD points:** ~500 (20 samples)
- **Raman points:** ~750 (15 samples)

### Typical Ranges:
- **Warp magnitude:** 5-50 μm (peak-to-valley)
- **Electrolyte stress:** -0.5 to -1.2 GPa (compressive)
- **Cathode stress:** 0.2 to 0.5 GPa (tensile)
- **Anode stress:** -0.2 to 0.3 GPa (variable)

---

## ⚠️ Important Notes

### 1. Synthetic Nature
This dataset is **synthetically generated** to simulate realistic experimental measurements. It is designed for:
- Educational purposes
- Algorithm development
- Methodology validation
- Publication demonstrations

For actual research, you should:
- Fabricate real SOFC samples
- Perform physical measurements
- Use this dataset as a template/reference

### 2. Measurement Uncertainties
All measurement techniques include realistic uncertainties:
- **Interferometry:** ±0.05 μm
- **Curvature method:** ±12%
- **XRD:** ±0.03 GPa
- **Raman:** ±0.08 GPa
- **Layer removal:** ±0.05 GPa

These must be accounted for in validation metrics.

### 3. Physical Constraints
The data satisfies key physical constraints:
- Force balance (∑Fi ≈ 0)
- Moment balance
- Positive thicknesses
- Realistic material properties
- Thermal history consistency

### 4. Multi-Technique Validation
Different techniques measure different aspects:
- **Curvature:** Through-thickness average
- **Layer removal:** Through-thickness gradient
- **XRD/Raman:** Surface, localized

A complete validation requires comparing against multiple techniques.

---

## 🔧 Customization

### Change Number of Samples

Edit `generate_experimental_validation_dataset.py`:

```python
generator = SOFCExperimentalDataGenerator(
    n_samples=50,  # Change from default 35
    output_dir='experimental_validation_dataset'
)
```

### Modify Parameter Ranges

Edit the `param_ranges` dictionary in `generate_fabrication_parameters()`:

```python
param_ranges = {
    'anode_thickness_um': (200, 1000),  # Expanded range
    # ... other parameters
}
```

### Add More Measurement Points

Modify `n_points_per_sample` in XRD/Raman generation functions:

```python
self.generate_xrd_stress_measurements(n_points_per_sample=50)  # More points
```

---

## 📚 References

### Experimental Techniques:

1. **Curvature Method:**
   - Stoney, G. G. (1909). "The tension of metallic films deposited by electrolysis." Proc. R. Soc. Lond. A 82: 172–175.
   - Freund, L. B., & Suresh, S. (2004). "Thin Film Materials: Stress, Defect Formation and Surface Evolution."

2. **Layer Removal:**
   - Prime, M. B. (2001). "Cross-sectional mapping of residual stresses by measuring the surface contour after a cut." J. Eng. Mater. Technol. 123(2): 162-168.

3. **XRD Stress Measurement:**
   - Noyan, I. C., & Cohen, J. B. (1987). "Residual Stress: Measurement by Diffraction and Interpretation."
   - Welzel, U., et al. (2005). "Stress analysis of polycrystalline thin films and surface regions by X-ray diffraction." J. Appl. Cryst. 38: 1-29.

4. **Raman Spectroscopy:**
   - Pezzotti, G. (2013). "Raman spectroscopy in cell biology and microbiology." J. Raman Spectrosc. 44(11): 1478-1496.
   - De Wolf, I. (1996). "Micro-Raman spectroscopy to study local mechanical stress in silicon integrated circuits." Semicond. Sci. Technol. 11: 139.

### SOFC Materials:
- Singhal, S. C., & Kendall, K. (2003). "High-temperature Solid Oxide Fuel Cells: Fundamentals, Design and Applications."
- Atkinson, A., et al. (2004). "Advanced anodes for high-temperature fuel cells." Nat. Mater. 3: 17-27.

---

## 📧 Support & Citation

### Questions?
For questions about this dataset or ML-augmented inverse modeling methodology, please refer to the research article or contact the authors.

### Citation:
If you use this dataset in your research, please cite:

```
@dataset{sofc_experimental_validation_2025,
  title={SOFC Experimental Validation Dataset for ML-Augmented Inverse Modeling},
  author={Your Name},
  year={2025},
  version={1.0},
  description={Synthetic experimental dataset for validating residual stress predictions from warp measurements in SOFC plates}
}
```

---

## ✅ Quality Assurance

This dataset has been validated for:
- ✓ File structure completeness
- ✓ Data consistency across files
- ✓ Physical constraint satisfaction
- ✓ Statistical properties (parameter space filling)
- ✓ Measurement uncertainty realism
- ✓ Force/moment balance

See `validation_report.json` for detailed QA results.

---

## 🎯 Next Steps

1. ✅ **Generated:** Experimental validation dataset
2. 📊 **Visualize:** Run visualization script
3. ✓ **Validate:** Run validation script
4. 🤖 **Train ML model:** Use FEA datasets (Dataset 1 & 2)
5. 🔬 **Validate ML model:** Use this dataset (Dataset 3)
6. 📈 **Analyze discrepancies:** Identify model improvements
7. 🔄 **Iterate:** Refine FEA/ML models based on validation

---

**Good luck with your ML-augmented inverse modeling research! 🚀**
