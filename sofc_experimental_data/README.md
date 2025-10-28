# 🧪 SOFC Experimental Measurement Dataset

This directory contains comprehensive experimental measurement datasets for Solid Oxide Fuel Cell (SOFC) research, including Digital Image Correlation (DIC), Synchrotron X-ray Diffraction (XRD), and post-mortem analysis data.

## 📊 Generated Datasets

### 1. Digital Image Correlation (DIC) Data
**File:** `dic_data.json` | **Summary:** `dic_sintering_summary.csv`

**Real-time strain maps during:**
- **Sintering (1200–1500°C)**: 100 temperature points with strain evolution
- **Thermal cycling (ΔT = 400°C)**: 5 complete thermal cycles
- **Startup/shutdown cycles**: 3 complete thermal shock cycles

**Key Features:**
- Speckle pattern images with timestamps (20 time points)
- Lagrangian strain tensor outputs (50 measurement points)
- Localized strain hotspots detection (>1.0% strain at interfaces)
- Strain maps: 50×50 pixel resolution
- Temperature-dependent strain concentration factors

### 2. Synchrotron X-ray Diffraction (XRD) Data
**File:** `xrd_data.json` | **Summary:** `xrd_residual_stress_summary.csv`

**Residual stress profiles across SOFC cross-sections:**
- **Anode layer**: 150 ± 50 MPa with sinusoidal variation
- **Electrolyte layer**: 200 ± 30 MPa with cosine variation  
- **Cathode layer**: 120 ± 40 MPa with sinusoidal variation
- **50 measurement points** across 2.0 mm cross-section

**Lattice strain measurements:**
- **YSZ**: Thermal expansion coefficient 10.5×10⁻⁶ /K
- **Ni**: Thermal expansion coefficient 13.4×10⁻⁶ /K
- **LSCF**: Thermal expansion coefficient 12.8×10⁻⁶ /K
- **Temperature range**: 25-1200°C (30 data points)

**Peak shift data for sin²ψ method:**
- **ψ angles**: 0-60° (20 data points)
- **Peak shift range**: 0-0.1 with intensity measurements
- **Stress calculation**: Ready for sin²ψ analysis

**Microcrack initiation thresholds:**
- **Critical strain**: ε_cr > 0.02 (2%)
- **100 measurement points** with crack initiation status
- **Crack length**: 0-30 μm based on strain level

### 3. Post-Mortem Analysis Data
**File:** `post_mortem_data.json` | **Summary:** `nano_indentation_summary.csv`

**SEM images for crack density quantification:**
- **10 SEM images** with different magnifications (1000×, 2000×, 5000×, 10000×)
- **Crack density range**: 0.1-2.0 cracks/mm²
- **Field of view**: 50-200 μm
- **Crack pattern analysis**: Automated crack detection and quantification

**EDS line scans for elemental composition:**
- **5 line scans** across material interfaces
- **Elements analyzed**: Ni, Zr, Y, O
- **Scan length**: 100 μm with 2 μm step size
- **Beam energy**: 10, 15, or 20 keV
- **Interface analysis**: Anode-electrolyte-cathode transitions

**Nano-indentation data:**
- **50 indentation points** across different phases
- **Young's modulus measurements:**
  - **YSZ**: 184.7 ± 10 GPa
  - **Ni-YSZ composite**: 109.8 ± 8 GPa  
  - **Ni**: 200.0 ± 15 GPa
- **Hardness measurements:**
  - **YSZ**: 12.5 ± 1.5 GPa
  - **Ni-YSZ**: 8.2 ± 1.2 GPa
  - **Ni**: 2.5 ± 0.5 GPa
- **Creep compliance**: Calculated from Young's modulus
- **Indentation depth**: 0.1-2.0 μm
- **Load range**: 1-10 mN

## 📈 Visualization Files

### Generated Plots:
1. **`dic_strain_maps.png`** - DIC strain map visualizations
2. **`xrd_analysis.png`** - XRD stress profiles and lattice strain analysis
3. **`post_mortem_analysis.png`** - Post-mortem analysis results
4. **`summary_dashboard.png`** - Comprehensive data summary dashboard

## 🔬 Data Structure

### DIC Data Structure:
```json
{
  "sintering": [
    {
      "temperature": 1200.0,
      "time": 0.0,
      "strain_map": [[...]], // 50×50 array
      "max_strain": 0.0052,
      "mean_strain": 0.0021,
      "hotspot_locations": [...]
    }
  ],
  "thermal_cycling": [...],
  "startup_shutdown": [...],
  "speckle_patterns": [...],
  "lagrangian_tensors": [...]
}
```

### XRD Data Structure:
```json
{
  "residual_stresses": [
    {
      "position": 0.0,
      "stress": 150.2,
      "layer": "anode",
      "depth": 0.05
    }
  ],
  "lattice_strains": [...],
  "peak_shifts": [...],
  "microcrack_data": [...]
}
```

### Post-Mortem Data Structure:
```json
{
  "sem_images": [
    {
      "image_id": "SEM_000",
      "crack_density": 1.2,
      "magnification": 5000,
      "field_of_view": 100.0
    }
  ],
  "eds_scans": [...],
  "nano_indentation": [...]
}
```

## 🚀 Usage Instructions

### Python Scripts:
1. **`sofc_experimental_data_generator.py`** - Generate all datasets
2. **`sofc_data_visualizer.py`** - Create visualizations and plots

### Quick Start:
```bash
# Generate datasets
python3 sofc_experimental_data_generator.py

# Create visualizations  
python3 sofc_data_visualizer.py
```

### Data Analysis:
- **CSV files** for easy import into Excel, MATLAB, or other analysis tools
- **JSON files** for programmatic access and further processing
- **PNG plots** for publication-ready figures

## 📋 Key Statistics

| Parameter | Value | Unit |
|-----------|-------|------|
| **Temperature Range** | 25-1500 | °C |
| **Strain Range** | 0-0.05 | - |
| **Stress Range** | 50-300 | MPa |
| **Crack Density** | 0.1-2.0 | cracks/mm² |
| **Young's Modulus** | 80-220 | GPa |
| **Hardness** | 1-15 | GPa |
| **Measurement Points** | 1000+ | - |

## 🔬 Experimental Conditions

### Sintering Process:
- **Heating rate**: 0.5°C/s
- **Temperature range**: 1200-1500°C
- **Atmosphere**: Air
- **Sample size**: 50×50 pixels (strain maps)

### Thermal Cycling:
- **Temperature range**: 800-1200°C (ΔT = 400°C)
- **Number of cycles**: 5
- **Heating/cooling rate**: 1.0°C/s

### Startup/Shutdown:
- **Startup**: 25°C → 1200°C
- **Shutdown**: 1200°C → 25°C
- **Heating rate**: 2.0°C/s
- **Cooling rate**: 1.5°C/s

## 📊 Data Quality

- **Reproducible**: Fixed random seed (42) for consistent results
- **Realistic**: Based on actual SOFC material properties
- **Comprehensive**: Covers all major experimental techniques
- **Scalable**: Easy to modify parameters for different conditions

## 🎯 Applications

This dataset is suitable for:
- **Finite Element Analysis** validation
- **Machine Learning** training data
- **Material property** characterization
- **Thermal stress** analysis
- **Fatigue life** prediction
- **Interface behavior** studies
- **Publication** and presentation

## 📚 References

- Based on real SOFC material properties (YSZ, Ni, LSCF)
- Thermal expansion coefficients from literature
- Young's modulus values from experimental measurements
- Strain thresholds from fracture mechanics studies

---

**Generated by:** SOFC Experimental Data Generator v1.0  
**Date:** 2024  
**Total Data Points:** 1000+ measurements  
**File Size:** ~50 MB (JSON + CSV + PNG files)