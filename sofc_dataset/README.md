# SOFC 3D Microstructural Dataset

## Dataset Information
- **Generated**: 2025-10-08 20:44:23
- **Dimensions**: 128 x 128 x 64 voxels
- **Voxel Size**: 0.5 µm
- **Physical Size**: 64.0 x 64.0 x 32.0 µm

## Phase Identification
| ID | Material | Volume Fraction |
|----|----------|-----------------|
| 0 | Phase 0 | 22.5% |
| 1 | Phase 1 | 7.9% |
| 2 | Phase 2 | 52.1% |
| 3 | Phase 3 | 12.5% |
| 4 | Phase 4 | 4.9% |
| 5 | Phase 5 | 0.0% |

## Key Properties
- **TPB Density**: 5.19e+08 m/m³
- **Anode Porosity**: ~35%
- **Electrolyte Density**: ~95%

## Files
- `microstructure.h5`: Complete dataset in HDF5 format
- `tiff_stack/`: TIFF image stack
- `metadata.json`: Complete metadata and properties
- `README.md`: This file

## Usage
### Loading in Python
```python
import h5py
import numpy as np

with h5py.File('microstructure.h5', 'r') as f:
    volume = f['microstructure'][:]
    voxel_size = f['microstructure'].attrs['voxel_size']
    # Access individual phases
    phase_ni = f['phases/phase_1'][:]
```

### Visualization
The data can be visualized using:
- ParaView (load .h5 or convert to .vtk)
- ImageJ/Fiji (load TIFF stack)
- Python (matplotlib, pyvista)

## Citation
If using this dataset, please acknowledge:
"SOFC 3D Microstructural Dataset - Synthetic data for electrode modeling"
