# SOFC Material Properties Dataset

## Overview

This dataset contains comprehensive thermo-physical, mechanical, and electrochemical properties for Solid Oxide Fuel Cell (SOFC) components. The data is based on typical values found in literature for commonly used SOFC materials.

## Materials Included

### 1. Ni-YSZ Anode (anode_ni_ysz)
- **Composition**: 40% Ni, 60% 8YSZ
- **Porosity**: 35%
- **Typical operating temperature**: 600-1000°C

### 2. 8YSZ Electrolyte (electrolyte_8ysz)
- **Composition**: 8 mol% Y₂O₃ - ZrO₂
- **Porosity**: 2%
- **Typical operating temperature**: 600-1000°C

### 3. LSM Cathode (cathode_lsm)
- **Composition**: La₀.₈Sr₀.₂MnO₃
- **Porosity**: 30%
- **Typical operating temperature**: 600-1000°C

### 4. LSCF Cathode (cathode_lscf)
- **Composition**: La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃-δ
- **Porosity**: 25%
- **Typical operating temperature**: 600-1000°C

### 5. Crofer 22 APU Interconnect (interconnect_crofer22)
- **Composition**: Fe-22Cr-0.5Mn-0.3Ti-0.1La
- **Porosity**: 0%
- **Typical operating temperature**: 600-800°C

## Property Categories

### Thermo-Physical Properties
- **Thermal Expansion Coefficient (TEC)**: Temperature-dependent linear expansion
- **Thermal Conductivity**: Heat transfer capability
- **Specific Heat Capacity**: Energy storage per unit mass
- **Density**: Mass per unit volume

### Mechanical Properties
- **Young's Modulus**: Elastic stiffness
- **Poisson's Ratio**: Lateral strain response
- **Creep Parameters**: Norton-Bailey law parameters (B, n, Q)
- **Plasticity Parameters**: Johnson-Cook model parameters
- **Yield Strength**: Plastic deformation threshold

### Electrochemical Properties
- **Electronic Conductivity**: Electron transport capability
- **Ionic Conductivity**: Ion transport capability
- **Exchange Current Density**: Electrochemical reaction rate
- **Activation Energy**: Temperature dependence of conductivity
- **Activation Overpotential**: Electrode polarization parameters

## Data Formats

### 1. JSON Format (`sofc_material_properties.json`)
- Hierarchical structure with nested properties
- Temperature-dependent data as key-value pairs
- Includes metadata and references
- Best for programmatic access

### 2. CSV Format (`sofc_material_properties.csv`)
- Flat table structure
- Easy to import into Excel or other analysis tools
- Temperature columns for interpolation
- Best for data analysis

### 3. Creep Parameters (`sofc_creep_parameters.csv`)
- Specialized table for creep modeling
- Norton-Bailey and Johnson-Cook parameters
- Activation energies for different processes

## Usage Examples

### Python Script (`sofc_data_processor.py`)
The included Python script provides utilities for:
- Loading and accessing material properties
- Temperature interpolation
- Property comparison across materials
- Thermal stress calculations
- Creep strain rate calculations
- Data export to pandas DataFrames

### Basic Usage
```python
from sofc_data_processor import SOFCMaterialDatabase

# Initialize database
db = SOFCMaterialDatabase()

# Get property at specific temperature
thermal_cond = db.get_material_property('anode_ni_ysz', 'thermal_conductivity', 600)

# Get all properties for a material
props = db.get_all_properties('electrolyte_8ysz', 800)

# Calculate thermal stress
stress = db.calculate_thermal_stress('anode_ni_ysz', 800)
```

## Property Validation

### Typical Ranges (at 800°C)
- **Thermal Expansion**: 10-15 × 10⁻⁶ 1/K
- **Thermal Conductivity**: 1-25 W/m·K
- **Young's Modulus**: 35-210 GPa
- **Electronic Conductivity**: 10³-10⁶ S/m
- **Ionic Conductivity**: 10⁻³-1 S/m

### Temperature Dependencies
- Most properties show linear or exponential temperature dependence
- Thermal expansion generally increases with temperature
- Electrical conductivity typically follows Arrhenius behavior
- Mechanical properties generally decrease with temperature

## Applications

This dataset is suitable for:
- **Finite Element Analysis (FEA)**: Material property inputs
- **Thermal Modeling**: Heat transfer simulations
- **Stress Analysis**: Thermal and mechanical stress calculations
- **Electrochemical Modeling**: Fuel cell performance simulations
- **Creep Analysis**: Long-term deformation predictions
- **Multi-physics Simulations**: Coupled thermal-electrical-mechanical models

## Limitations and Notes

1. **Temperature Range**: Data is valid for 25-1000°C range
2. **Material Variations**: Properties may vary with processing conditions
3. **Interpolation**: Linear interpolation used between data points
4. **Literature Values**: Based on typical values from published literature
5. **Porosity Effects**: Properties are for bulk materials; porosity effects included where applicable

## References

The property values are based on typical ranges found in:
- SOFC material characterization studies
- Fuel cell component testing literature
- Material property databases
- Industry standard specifications

## File Structure

```
/workspace/
├── sofc_material_properties.json      # Main dataset (JSON format)
├── sofc_material_properties.csv       # Main dataset (CSV format)
├── sofc_creep_parameters.csv          # Creep parameters
├── sofc_data_processor.py             # Python utilities
└── README_SOFC_Dataset.md             # This documentation
```

## Dependencies

For the Python script:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `json` (built-in)

Install with:
```bash
pip install numpy pandas scipy matplotlib
```

## Contact

For questions about this dataset or to report issues, please refer to the material property literature or contact the dataset maintainer.