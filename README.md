# Welding Inverse Design Dataset

## Overview

This repository contains a comprehensive multi-tiered dataset for welding inverse design research, specifically focused on extreme-temperature performance prediction for battery pack applications. The dataset was generated using a combination of experimental data simulation, computational modeling, and literature curation.

## Dataset Structure

### Multi-Tier Architecture

The dataset follows a three-tier approach as recommended for PhD-level research:

1. **Tier 1: High-Fidelity Experimental Data** (400 samples)
   - Generated using Latin Hypercube Sampling (LHS) for efficient parameter space exploration
   - Physics-based models with high fidelity (5% noise)
   - Represents ground truth experimental data

2. **Tier 2: Computational Simulation Data** (10,000 samples)
   - FEM-based simulation models for laser welding processes
   - Medium fidelity (15% noise) to represent computational predictions
   - Dense coverage of parameter space

3. **Tier 3: Literature & Legacy Data** (500 samples)
   - Curated from published literature and industry reports
   - Lower fidelity (25% noise) but broader context
   - Conservative parameter ranges typical in literature

### Input Parameters (X)

#### Energy Input Parameters
- `laser_power`: 500-3000 W
- `welding_speed`: 10-200 mm/s
- `pulse_frequency`: 1-1000 Hz
- `pulse_duration`: 0.1-20 ms

#### Beam Characteristics
- `beam_focus_position`: -2 to 2 mm
- `beam_spot_size`: 50-500 µm

#### Material & Setup
- `clamping_pressure`: 0.1-2.0 MPa
- `shield_gas_flow`: 5-30 L/min
- `material_thickness`: 0.1-3.0 mm
- `overlap_distance`: 0.5-5.0 mm
- `material_combination`: 0=Cu-Al, 1=Al-Al, 2=Cu-Steel, 3=Al-Steel

### Output Properties (Y)

#### Weld Morphology & Quality
- `nugget_width`: 0.5-3.0 mm
- `penetration_depth`: 0.1-2.0 mm
- `haz_width`: 0.2-1.5 mm
- `crack_presence`: Binary (0/1)
- `porosity_percentage`: 0-15%
- `undercut_depth`: 0-0.3 mm

#### Mechanical & Electrical Properties
- `tensile_shear_strength`: 500-5000 N
- `peel_strength`: 50-800 N
- `contact_resistance`: 5-100 µΩ

#### Extreme-Temperature Performance
- `thermal_cycles_to_failure`: 100-2000 cycles
- `strength_degradation_pct`: 0-50%
- `resistance_increase_pct`: 0-200%
- `imc_thickness`: 0.1-10 µm
- `creep_time_to_failure`: 1-1000 hours

## Files Description

### Core Dataset Files
- `welding_master_dataset.csv`: Complete dataset with all tiers combined
- `welding_dataset_train.csv`: Training split (70%)
- `welding_dataset_validation.csv`: Validation split (10%)
- `welding_dataset_test.csv`: Test split (20%)
- `dataset_metadata.json`: Dataset metadata and feature descriptions

### Code Files
- `welding_inverse_design_dataset_fixed.py`: Main dataset generator
- `fem_simulation.py`: FEM simulation module for computational data
- `data_visualization.py`: Comprehensive visualization tools
- `requirements.txt`: Python dependencies

### Visualization Outputs
- `visualization_outputs/`: Directory containing all generated visualizations
  - `parameter_distributions.png`: Input parameter distributions by data source
  - `output_distributions.png`: Output property distributions
  - `correlation_heatmap.png`: Feature correlation matrix
  - `performance_metrics.png`: Key performance metrics by data source
  - `material_analysis.png`: Analysis by material combination
  - `energy_density_analysis.png`: Energy density vs performance relationships
  - `extreme_temperature_analysis.png`: Extreme temperature performance analysis
  - `pca_analysis.png`: Principal component analysis
  - `interactive_dashboard.html`: Interactive Plotly dashboard

## Usage

### Basic Usage

```python
import pandas as pd
import numpy as np

# Load the master dataset
dataset = pd.read_csv('welding_master_dataset.csv')

# Load specific splits
train_data = pd.read_csv('welding_dataset_train.csv')
val_data = pd.read_csv('welding_dataset_validation.csv')
test_data = pd.read_csv('welding_dataset_test.csv')

# Access metadata
import json
with open('dataset_metadata.json', 'r') as f:
    metadata = json.load(f)
```

### For Inverse Design ML Models

```python
# Separate input and output features
input_features = metadata['input_features']
output_features = metadata['output_features']

X = dataset[input_features]
y = dataset[output_features]

# For inverse design, you want to predict inputs from desired outputs
# X_desired -> Y_target (forward model)
# Y_desired -> X_predicted (inverse model)
```

### Data Quality Assessment

```python
# Check data source distribution
print(dataset['data_source'].value_counts())

# Check extreme temperature testing samples
extreme_samples = dataset[dataset.get('thermal_cycling_applied', False) == True]
print(f"Extreme temperature tested samples: {len(extreme_samples)}")

# Check material combinations
print(dataset['material_combination'].value_counts())
```

## Key Features

### Physics-Based Generation
- All outputs are generated using physics-based empirical models
- Energy density calculations drive most relationships
- Material-dependent properties for different metal combinations

### Extreme Temperature Focus
- 100 samples subjected to thermal cycling testing
- Temperature ranges: -40°C to +125°C
- Degradation modeling for strength and electrical properties
- IMC (Intermetallic Compound) growth prediction

### Multi-Fidelity Approach
- Different noise levels for different data sources
- Quality scores assigned to each data point
- Stratified sampling for balanced representation

### ML-Ready Format
- Pre-split into train/validation/test sets
- Standardized scaling available
- Feature importance analysis included
- Correlation matrices provided

## Research Applications

This dataset is specifically designed for:

1. **Inverse Design Optimization**: Finding optimal welding parameters for desired performance
2. **Extreme Temperature Performance Prediction**: Understanding thermal degradation
3. **Multi-Physics Modeling**: Coupling thermal, mechanical, and electrical properties
4. **Material Compatibility Analysis**: Comparing different metal combinations
5. **Process Optimization**: Improving weld quality and reliability

## Citation

If you use this dataset in your research, please cite:

```
Welding Inverse Design Dataset for Extreme-Temperature Performance Prediction
AI Assistant, 2024
Multi-tier dataset for laser welding process optimization
```

## Technical Details

### Generation Method
- **Tier 1**: Latin Hypercube Sampling + Physics-based models
- **Tier 2**: Monte Carlo sampling + FEM simulation models
- **Tier 3**: Literature-based parameter ranges + empirical models

### Validation
- Physics-based relationships validated against literature
- Energy density calculations follow established formulas
- Material properties based on standard reference values

### Limitations
- Simulated data - not from actual experiments
- Simplified physics models for computational efficiency
- Limited to laser welding processes
- Focus on battery pack applications

## Future Work

- Integration with actual experimental data
- More sophisticated FEM models
- Additional material combinations
- Real-time process monitoring integration
- Industry validation studies

## Contact

For questions about this dataset or collaboration opportunities, please refer to the research documentation or contact the development team.

---

**Note**: This dataset is generated for research purposes. While based on established physics principles and literature data, it should be validated against actual experimental results before use in production applications.