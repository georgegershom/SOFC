# SOFC "In-The-Wild" Operational Dataset

## Overview

This dataset contains warp measurement data from 500 SOFC (Solid Oxide Fuel Cell) plates produced under nominally identical conditions in a simulated industrial environment. The dataset is designed for ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates.

## Dataset Characteristics

### Realism Features
- **Manufacturing Variations**: Natural variations in sintering temperature, time, cooling rate, green density, and humidity
- **Parameter Drift**: Time-dependent drift in manufacturing parameters due to equipment aging and calibration drift
- **Measurement Noise**: Realistic measurement uncertainties including random noise, systematic errors, temperature drift, and vibration
- **Failure Modes**: Known failure signatures including edge cracking, delamination, and thermal shock susceptibility
- **Production Schedule**: Realistic production timeline with shifts, batches, maintenance downtime, and seasonal variations

### Physical Basis
- **Stress-Displacement Relationship**: Based on thin plate theory and elasticity principles
- **Material Properties**: Realistic SOFC material properties (YSZ-based ceramics)
- **Thermal Effects**: Temperature-dependent stress generation during cooling
- **Microstructural Effects**: Grain boundary effects and stress concentrations

## File Structure

```
sofc_in_the_wild_dataset/
├── plates/                          # Individual plate data (JSON)
│   ├── SOFC_0001.json
│   ├── SOFC_0002.json
│   └── ...
├── manufacturing_parameters.csv      # Manufacturing conditions summary
├── quality_analysis.csv            # Failure analysis and quality metrics
├── measurement_summary.csv          # Displacement measurement statistics
├── dataset_statistics.json         # Overall dataset statistics
├── analysis_figures/               # Visualization and analysis plots
│   ├── manufacturing_parameter_drift.png
│   ├── failure_analysis.png
│   └── example_displacement_fields.png
└── README.md                       # This documentation

```

## Data Format

### Individual Plate Data (JSON)
Each plate file contains:

- **metadata**: Plate ID, production date, shift, batch ID, measurement date
- **manufacturing_params**: Sintering temperature, time, cooling rate, green density, humidity, furnace position
- **coordinates**: X,Y measurement grid coordinates (mm)
- **measurements**: Measured displacement field (μm) with uncertainties
- **ground_truth**: True displacement and stress fields (for validation)
- **failure_analysis**: Risk indicators for various failure modes
- **quality_metrics**: SNR, stress uniformity, edge quality

### Summary Files (CSV)
- **manufacturing_parameters.csv**: All manufacturing conditions for easy analysis
- **quality_analysis.csv**: Failure risks and quality metrics
- **measurement_summary.csv**: Statistical summary of displacement measurements

## Usage Examples

### Loading Data in Python

```python
import json
import pandas as pd
import numpy as np

# Load manufacturing parameters
manufacturing_df = pd.read_csv('manufacturing_parameters.csv')

# Load individual plate
with open('plates/SOFC_0001.json', 'r') as f:
    plate_data = json.load(f)

# Extract displacement field
displacement = np.array(plate_data['measurements']['displacement_um'])
x_coords = np.array(plate_data['coordinates']['x_mm'])
y_coords = np.array(plate_data['coordinates']['y_mm'])
```

### Analysis Workflow

1. **Exploratory Analysis**: Use summary CSV files to understand parameter distributions and correlations
2. **Quality Assessment**: Analyze failure risks and measurement quality metrics
3. **Model Training**: Use measured displacement as input, stress field as target
4. **Validation**: Compare predictions against ground truth stress fields
5. **Robustness Testing**: Evaluate model performance across different manufacturing conditions

## Key Research Applications

### Inverse Modeling
- Train ML models to predict stress fields from displacement measurements
- Validate against ground truth stress data
- Test robustness across manufacturing variations

### Failure Prediction
- Correlate stress patterns with failure risk indicators
- Develop early warning systems for quality control
- Optimize manufacturing parameters to reduce failure risk

### Uncertainty Quantification
- Account for measurement uncertainties in predictions
- Propagate manufacturing parameter uncertainties
- Develop confidence intervals for stress predictions

## Dataset Statistics

- **Total Plates**: 500
- **Measurement Grid**: 31 × 31 points
- **Plate Dimensions**: 150 × 150 mm
- **Production Timespan**: ~20 days
- **Measurement Precision**: 2.0 μm

## Manufacturing Parameter Ranges

| Parameter | Nominal | Std Dev | Units |
|-----------|---------|---------|-------|
| Sintering Temperature | 1400 | 15 | °C |
| Sintering Time | 4.0 | 0.2 | hours |
| Cooling Rate | 2.0 | 0.3 | °C/min |
| Green Density | 0.55 | 0.02 | - |
| Humidity | 45 | 8 | % |

## Citation

If you use this dataset in your research, please cite:

```
SOFC "In-The-Wild" Operational Dataset for ML-Augmented Inverse Modeling 
of Residual Stress Quantification from Warped Plates
Generated: 2025-10-16
```

## Contact

For questions about this dataset or to report issues, please contact the dataset maintainer.

---
*This dataset was generated using physics-based modeling with realistic manufacturing variations and measurement uncertainties to support research in ML-augmented inverse modeling for SOFC applications.*
