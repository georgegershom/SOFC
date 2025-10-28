# 🧪 SOFC Experimental Measurement Dataset

## Overview

This repository contains **synthetic experimental datasets** for Solid Oxide Fuel Cell (SOFC) research, specifically designed to support machine learning and computational modeling of SOFC degradation mechanisms. The data simulates realistic experimental measurements from three key characterization techniques:

1. **Digital Image Correlation (DIC)** - Real-time strain mapping
2. **Synchrotron X-ray Diffraction (XRD)** - Residual stress and lattice strain analysis
3. **Post-Mortem Analysis** - SEM, EDS, and nanoindentation characterization

## 📊 Dataset Structure

```
sofc_experimental_data/
│
├── dic_data/                    # Digital Image Correlation Data
│   ├── sintering/              # Sintering process (1200-1500°C)
│   ├── thermal_cycling/        # Thermal cycling (ΔT = 400°C)
│   └── startup_shutdown/       # Startup/shutdown cycles
│
├── xrd_data/                    # Synchrotron XRD Data
│   ├── sintering/              # High-temperature sintering
│   ├── thermal_cycling/        # Cyclic thermal loading
│   └── startup_shutdown/       # Rapid thermal transients
│
├── post_mortem/                 # Post-Mortem Analysis Data
│   ├── sintering/              # Post-sintering characterization
│   ├── thermal_cycling/        # After thermal cycling
│   └── startup_shutdown/       # After operational cycles
│
└── figures/                     # Generated visualizations
```

## 🔬 Data Components

### 1. Digital Image Correlation (DIC) Data

#### Features:
- **Strain Maps**: Full-field strain tensors (εxx, εyy, εxy, von Mises)
- **Temporal Evolution**: Time-resolved strain data with temperature profiles
- **Hotspot Detection**: Localized high-strain regions (>1.0% strain)
- **Vic-3D Compatible**: Output format compatible with commercial DIC software

#### Key Files:
- `dic_summary.csv`: Time-series strain statistics
- `strain_maps.json`: Full-field strain tensor data
- `vic3d_output.json`: Vic-3D software compatible format
- `speckle_metadata.json`: Speckle pattern and camera settings

#### Experimental Conditions:
- **Sintering**: 25°C → 1500°C → 25°C (with 50 min hold)
- **Thermal Cycling**: 10 cycles, 600°C ↔ 1000°C (ΔT = 400°C)
- **Startup/Shutdown**: 5 rapid cycles, 25°C ↔ 800°C

### 2. Synchrotron X-ray Diffraction (XRD) Data

#### Features:
- **Residual Stress Profiles**: Through-thickness stress distribution (σxx, σyy, σzz)
- **Lattice Strain**: Temperature-dependent lattice parameter changes
- **sin²ψ Analysis**: Stress calculation using tilting method
- **Microcrack Thresholds**: Critical strain for crack initiation (εcr > 0.02)

#### Key Files:
- `residual_stress_profile.csv`: Stress distribution across SOFC layers
- `lattice_strain_data.csv`: Material-specific lattice strain
- `sin2psi_raw_data.csv`: Raw sin²ψ method measurements
- `sin2psi_stress_results.csv`: Calculated stresses
- `microcrack_threshold_data.csv`: Crack initiation criteria
- `xrd_patterns.json`: Simulated diffraction patterns

#### Materials Characterized:
- **YSZ** (Yttria-Stabilized Zirconia): E = 184.7 GPa
- **Ni-YSZ** (Anode): E = 109.8 GPa
- **GDC** (Gadolinium-Doped Ceria): E = 175.0 GPa
- **LSM** (Cathode): E = 120.0 GPa

### 3. Post-Mortem Analysis Data

#### Features:
- **SEM Crack Analysis**: Crack density quantification (cracks/mm²)
- **EDS Line Scans**: Elemental composition profiles
- **Nanoindentation**: Mechanical property mapping
- **Porosity Analysis**: Pore size distribution and connectivity

#### Key Files:
- `sem_crack_analysis.csv`: Regional crack density statistics
- `crack_details_*.csv`: Individual crack measurements
- `eds_line_scan.csv`: Elemental composition across interfaces
- `eds_maps.json`: 2D elemental distribution maps
- `nanoindentation_grid.csv`: Mechanical property grid data
- `load_displacement_curves.json`: Force-displacement curves
- `porosity_analysis.csv`: Microstructural characterization

#### Measured Properties:
- **Young's Modulus**: 109.8-200 GPa (material-dependent)
- **Hardness**: 3.5-13.5 GPa
- **Crack Density**: 2.5-8.5 cracks/mm²
- **Porosity**: 2-35% (layer-dependent)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sofc_experimental_data

# Install dependencies
pip install -r requirements.txt
```

### Generate All Datasets

```bash
# Generate complete synthetic datasets
python generate_all_data.py
```

This will create all experimental datasets for three conditions:
- Sintering process
- Thermal cycling
- Startup/shutdown cycles

### Visualize Data

```bash
# Create comprehensive visualizations
python visualize_data.py
```

This generates publication-quality figures in the `figures/` directory.

### Generate Individual Datasets

```python
# Generate only DIC data
from generate_dic_data import DICDataGenerator
dic_gen = DICDataGenerator()
dic_gen.run_all()

# Generate only XRD data
from generate_xrd_data import XRDDataGenerator
xrd_gen = XRDDataGenerator()
xrd_gen.run_all()

# Generate only post-mortem data
from generate_postmortem_data import PostMortemDataGenerator
pm_gen = PostMortemDataGenerator()
pm_gen.run_all()
```

## 📈 Data Usage Examples

### Loading DIC Strain Data

```python
import pandas as pd
import json

# Load temporal strain data
dic_data = pd.read_csv('dic_data/thermal_cycling/dic_summary.csv')

# Load full strain maps
with open('dic_data/thermal_cycling/strain_maps.json', 'r') as f:
    strain_maps = json.load(f)

# Access von Mises strain field
von_mises = strain_maps[0]['strain_tensor']['von_mises_strain']
```

### Analyzing XRD Stress Profiles

```python
# Load residual stress data
stress_data = pd.read_csv('xrd_data/sintering/residual_stress_profile.csv')

# Filter by layer
electrolyte_stress = stress_data[stress_data['layer'] == 'Electrolyte']

# Calculate stress gradient
stress_gradient = np.gradient(stress_data['sigma_xx_MPa'], 
                             stress_data['position_um'])
```

### Processing Nanoindentation Data

```python
# Load mechanical properties
nano_data = pd.read_csv('post_mortem/thermal_cycling/nanoindentation_grid.csv')

# Calculate average modulus by material
modulus_by_material = nano_data.groupby('material')['youngs_modulus_GPa'].agg(['mean', 'std'])

# Load force-displacement curves
with open('post_mortem/thermal_cycling/load_displacement_curves.json', 'r') as f:
    curves = json.load(f)
```

## 🔧 Customization

### Modifying Material Properties

Edit material properties in the respective generator classes:

```python
# In generate_xrd_data.py
self.materials = {
    'YSZ': {
        'elastic_modulus': 210e9,  # Pa
        'poisson_ratio': 0.31,
        'thermal_expansion': 10.5e-6,  # /K
        # ... modify as needed
    }
}
```

### Adjusting Experimental Parameters

```python
# In generate_dic_data.py
# Modify temperature profiles
temps = np.linspace(25, 1600, 100)  # Increase max temperature

# Adjust strain levels
base_strain = 0.002  # Increase base strain
```

## 📊 Data Statistics

### Dataset Size
- **Total Files**: ~150 files
- **Total Size**: ~50 MB
- **Data Points**: >100,000 measurements
- **Time Points**: 250-600 per experiment

### Coverage
- **Temperature Range**: 25-1500°C
- **Strain Range**: 0-2%
- **Stress Range**: -150 to +50 MPa
- **Materials**: 5 SOFC components

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new experimental conditions
- Implement additional characterization techniques
- Improve data realism
- Add validation against real experimental data

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{sofc_experimental_2025,
  title={Synthetic SOFC Experimental Measurement Dataset},
  author={Your Research Group},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/sofc-experimental-data}}
}
```

## ⚠️ Disclaimer

This is a **synthetic dataset** generated for research and development purposes. While the data is designed to be physically realistic and follows expected trends from SOFC literature, it should not be used as a substitute for actual experimental measurements in critical applications.

## 📄 License

This dataset is provided under the MIT License. See LICENSE file for details.

## 📧 Contact

For questions or collaborations, please contact:
- Email: your.email@institution.edu
- GitHub Issues: [Create an issue](https://github.com/yourusername/repo/issues)

---

**Note**: This dataset was generated on October 3, 2025, using physics-based models and statistical distributions derived from SOFC literature.