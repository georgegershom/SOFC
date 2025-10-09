# SOFC Ground Truth Fracture Dataset

This dataset contains synthetic "ground truth" fracture data for training Physics-Informed Neural Networks (PINNs) and validating fracture prediction models in Solid Oxide Fuel Cells.

## Dataset Structure

```
fracture_dataset/
├── dataset_summary.json          # Dataset metadata and parameters
├── sample_000/                   # First sample
│   ├── phase_field_data.h5      # 4D crack evolution data
│   ├── sem_data.h5              # SEM/FIB post-mortem images
│   ├── metadata.json            # Sample-specific metadata
│   └── performance_data.json    # Degradation measurements
├── sample_001/                   # Second sample
│   └── ...
└── sample_099/                   # Last sample
    └── ...
```

## Data Types

### 1. In-situ Crack Evolution Data
- **File**: `phase_field_data.h5`
- **Format**: 4D array (x, y, z, t) representing crack phase field
- **Values**: 0 = intact material, 1 = fully cracked
- **Resolution**: 128×128×64 voxels, 1.17 μm/voxel
- **Temporal**: 50 time steps, 1 hour intervals

### 2. Ex-situ Post-mortem Analysis
- **File**: `sem_data.h5`
- **Format**: 2D SEM-like images (512×512 pixels)
- **Resolution**: 50 nm/pixel
- **Content**: Cross-sections through final crack state
- **Analysis**: Crack measurements, microstructure data

### 3. Macroscopic Performance Degradation
- **File**: `performance_data.json`
- **Metrics**: Voltage, ASR, power density, mechanical properties
- **Correlations**: Linked to delamination area evolution
- **Duration**: 5000 hours of simulated operation

## Usage

```python
import h5py
import json

# Load phase field data
with h5py.File('sample_000/phase_field_data.h5', 'r') as f:
    phase_field = f['phase_field'][:]
    time_array = f['physical_time'][:]

# Load performance data
with open('sample_000/performance_data.json', 'r') as f:
    performance = json.load(f)
```

## Physical Basis

The dataset is based on realistic material properties for 8YSZ electrolyte:
- Young's modulus: 170 GPa (at 800°C)
- Thermal expansion: 10.5×10⁻⁶ K⁻¹
- Fracture toughness: 3.0 MPa√m
- Operating temperature: 800°C

## Applications

- PINN training for fracture prediction
- Validation of phase-field models
- Correlation analysis between microstructure and performance
- Development of degradation prediction algorithms

## Citation

If you use this dataset, please cite:
"Synthetic Ground Truth Fracture Dataset for SOFC PINN Training and Validation"
Generated: 2025-10-09
