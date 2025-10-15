# Multi-Fidelity Digital Twin for SOFCs Dataset Generator

## Overview
This repository contains a comprehensive dataset generator for multi-scale modeling and deep learning applications in Solid Oxide Fuel Cells (SOFCs), focusing on thermo-mechanical degradation prediction.

## Dataset Structure

### Fidelity Levels
- **LF (Low-Fidelity)**: System-level lumped parameter models
- **MF (Medium-Fidelity)**: 1D/2D simplified physics models
- **HF (High-Fidelity)**: 3D detailed physics simulations with microstructural resolution

### Parameter Scales
1. **System Level**: Operating conditions and performance metrics
2. **Cell/Stack Level**: Geometry and configuration parameters
3. **Material Level**: Physical and chemical properties
4. **Microstructural Level**: 3D reconstruction data from FIB-SEM/X-Ray tomography

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Complete Dataset
```python
python generate_dataset.py --samples 10000 --fidelity all --output datasets/
```

### Generate Specific Fidelity Level
```python
python generate_dataset.py --samples 5000 --fidelity LF --output datasets/
```

### Visualize Dataset
```python
python visualize_data.py --dataset datasets/sofc_dataset.h5
```

## Dataset Contents

The generated dataset includes:
- Multi-scale input parameters
- Operating conditions history
- Degradation trajectories
- Performance metrics
- Microstructural evolution data (synthetic)
- Thermal stress distributions
- Voltage degradation curves

## File Formats
- HDF5 for structured numerical data
- CSV for tabular exports
- NPZ for NumPy arrays
- JSON for metadata

## Citation
If you use this dataset in your research, please cite:
```
Multi-Fidelity Digital Twin for SOFCs: Multi-Scale Modeling and Deep Learning 
for Predicting Thermo-Mechanical Degradation
```