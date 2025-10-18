# DATASET DOWNLOAD AND USAGE INSTRUCTIONS

## Quick Start

1. **Extract the archive**:
   ```bash
   # For ZIP files
   unzip rubberized_concrete_dataset_YYYYMMDD_HHMMSS.zip
   
   # For TAR.GZ files  
   tar -xzf rubberized_concrete_dataset_YYYYMMDD_HHMMSS.tar.gz
   ```

2. **Navigate to the dataset directory**:
   ```bash
   cd rubberized_concrete_dataset
   ```

3. **Read the documentation**:
   ```bash
   # Main documentation
   cat README.md
   
   # Generation report
   cat GENERATION_REPORT.md
   
   # Data dictionary
   cat data_dictionary.json
   ```

4. **Test the dataset**:
   ```bash
   python3 usage_examples.py
   ```

## Dataset Contents

### Material Properties (Model Input)
- `thermal_properties.csv/json` - Temperature-dependent thermal properties
- `mechanical_properties.csv/json` - Temperature-dependent mechanical properties  
- `deformation_properties.csv/json` - Thermal expansion and transient strain
- `poromechanical_properties.csv/json` - Porosity, permeability, and damage

### Validation Data (Experimental)
- `temperature_evolution_validation.csv/json` - Thermocouple measurements
- `deformation_strain_validation.csv/json` - Strain evolution under loading
- `spalling_failure_summary.csv` - Spalling occurrence and failure analysis
- `spalling_failure_detailed.csv` - Time-series spalling data

### Visualizations
- `visualizations/` directory contains all plots:
  - `thermal_properties.png` - Thermal property plots
  - `mechanical_properties.png` - Mechanical property plots
  - `temperature_evolution.png` - Temperature validation plots
  - `strain_evolution.png` - Strain validation plots
  - `spalling_analysis.png` - Spalling analysis plots
  - `summary_dashboard.png` - Comprehensive overview

### Documentation
- `README.md` - Complete documentation and usage guide
- `data_dictionary.json` - Field descriptions and units
- `metadata.json` - Dataset metadata and specifications
- `usage_examples.py` - Code examples and templates
- `MANIFEST.json` - Complete file listing and checksums

## System Requirements

- Python 3.7+
- Required packages: numpy, pandas, matplotlib, seaborn, scipy
- Install with: `pip install numpy pandas matplotlib seaborn scipy`

## Data Loading Examples

### Python (Pandas)
```python
import pandas as pd

# Load thermal properties
thermal_df = pd.read_csv('thermal_properties.csv')

# Filter for specific rubber content
rubber_10 = thermal_df[thermal_df['rubber_content_pct'] == 10]

# Plot thermal conductivity vs temperature
import matplotlib.pyplot as plt
plt.plot(rubber_10['temperature_C'], rubber_10['thermal_conductivity_W_m_K'])
plt.xlabel('Temperature (°C)')
plt.ylabel('Thermal Conductivity (W/m·K)')
plt.show()
```

### Python (JSON)
```python
import json

# Load hierarchical data
with open('thermal_properties.json', 'r') as f:
    thermal_data = json.load(f)

# Access specific rubber content data
rubber_10_data = thermal_data['rubber_10pct']
temperatures = rubber_10_data['temperature']
conductivity = rubber_10_data['thermal_conductivity']
```

### R
```r
# Load CSV data
thermal_data <- read.csv('thermal_properties.csv')

# Filter and plot
library(ggplot2)
rubber_10 <- subset(thermal_data, rubber_content_pct == 10)
ggplot(rubber_10, aes(x=temperature_C, y=thermal_conductivity_W_m_K)) +
  geom_line() +
  labs(x='Temperature (°C)', y='Thermal Conductivity (W/m·K)')
```

## Applications

1. **Finite Element Modeling**: Use material properties for FE model calibration
2. **Fire Safety Design**: Validate fire resistance predictions
3. **Material Optimization**: Compare different rubber contents
4. **Research**: Basis for further experimental or numerical studies
5. **Education**: Teaching fire engineering and material science

## Citation

If you use this dataset in your research, please cite:

```
Fire-Resistant Rubberized Concrete Dataset
Topic: Development and Validation of a Thermo-Mechanical Model for 
       Fire-Resistant Structural Elements Utilizing High-Performance 
       Rubberized Concrete
Generated: 2025-10-18
Version: 1.0
```

## Support

For questions about the dataset:
1. Check the README.md file
2. Review the usage_examples.py script
3. Examine the data_dictionary.json for field descriptions
4. Refer to the visualizations for data overview

## License

This dataset is provided for research and educational purposes.
Please acknowledge the source when using in publications or presentations.
