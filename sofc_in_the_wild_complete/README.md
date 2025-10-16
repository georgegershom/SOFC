# SOFC "In-The-Wild" Dataset Package

Complete dataset package for ML-Augmented Inverse Modeling of Residual Stress Quantification from Warped SOFC Plates.

## Quick Start

1. Run quick analysis:
   ```bash
   python quick_start.py
   ```

2. Load ML-ready data:
   ```python
   import numpy as np
   data = np.load('sofc_ml_ready.npz')
   X = data['X']  # Features
   y = data['y_max_stress']  # Target stress
   ```

3. Explore visualizations:
   - Check `quick_viz/` directory for overview plots
   - See `dataset/analysis_figures/` for detailed analysis

## Contents

- `dataset/` - Complete dataset with all files
- `sofc_ml_ready.npz` - ML-ready features and targets
- `quick_viz/` - Quick visualization plots
- `quick_start.py` - Quick start analysis script

## Dataset Structure

- **500 SOFC plates** with realistic manufacturing variations
- **Manufacturing parameters**: Temperature, time, cooling rate, density, humidity
- **Measurements**: High-resolution displacement fields (31×31 grid)
- **Ground truth**: Stress fields for validation
- **Failure analysis**: Risk indicators for various failure modes
- **Temporal data**: Production timeline with parameter drift

## Key Features

✅ **Realistic Manufacturing Variations**: Natural parameter drift and batch effects
✅ **Measurement Noise**: Multiple noise sources and uncertainties
✅ **Known Failure Modes**: Edge cracking, delamination, thermal shock patterns
✅ **ML-Ready**: Pre-processed features and comprehensive analysis
✅ **Validated**: Extensive validation for physical plausibility

## Citation

If you use this dataset in your research, please cite:

```
SOFC "In-The-Wild" Operational Dataset for ML-Augmented Inverse Modeling 
of Residual Stress Quantification from Warped Plates
Generated: 2025-10-16
```

For detailed documentation, see `dataset/README.md`
