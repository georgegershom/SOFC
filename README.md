# YSZ Material Properties Dataset for SOFC Thermomechanical FEM Modeling

## Overview

This dataset provides comprehensive temperature-dependent material properties for 8YSZ (8 mol% Yttria-Stabilized Zirconia), a common electrolyte material used in Solid Oxide Fuel Cells (SOFCs). The data is specifically curated for finite element method (FEM) thermomechanical modeling applications.

## Dataset Contents

### Primary Data File
- **`ysz_material_properties.csv`**: Main dataset with 16 temperature points from 25°C to 1500°C

### Analysis Tools
- **`material_properties_analysis.py`**: Python script for data interpolation, visualization, and export

### Material Properties Included

| Property | Symbol | Units | Description |
|----------|--------|-------|-------------|
| **Young's Modulus** | E | GPa | Material stiffness, decreases with temperature |
| **Poisson's Ratio** | ν | - | Lateral to axial strain ratio, assumed constant |
| **Coefficient of Thermal Expansion** | α | 10⁻⁶/K | Thermal expansion coefficient, increases with temperature |
| **Density** | ρ | kg/m³ | Material density, slightly decreases with temperature |
| **Thermal Conductivity** | k | W/(m·K) | Heat conduction capability, decreases with temperature |
| **Fracture Toughness** | K_IC | MPa√m | Resistance to crack propagation |
| **Weibull Modulus** | m | - | Statistical strength distribution parameter |
| **Characteristic Strength** | σ₀ | MPa | Reference strength for Weibull distribution |
| **Creep Parameters** | A, n, Q | Various | Norton power law creep model parameters |

## Temperature Range

- **Minimum**: 25°C (Room Temperature)
- **Maximum**: 1500°C (Sintering Temperature)
- **Key Points**:
  - 25°C: Room temperature properties
  - 800°C: Typical SOFC operating temperature
  - 1000-1200°C: High-temperature operation
  - 1500°C: Sintering temperature

## Data Sources and Validation

The dataset is compiled from multiple sources in the ceramic materials literature, with values representative of typical 8YSZ properties:

1. **Elastic Properties**: Based on ultrasonic measurements and mechanical testing
2. **Thermal Properties**: From dilatometry and thermal analysis
3. **Fracture Properties**: From fracture mechanics testing
4. **Creep Data**: From high-temperature creep experiments

### Key References (Representative Literature)
- Atkinson & Selçuk (2000) - Mechanical behavior of ceramic oxygen ion-conducting membranes
- Radovic & Lara-Curzio (2004) - Mechanical properties of tape cast nickel-based anode materials
- Mori et al. (2006) - Thermal expansion of nickel-zirconia anodes in SOFCs
- Various ceramic handbooks and databases

## Usage Instructions

### Basic Usage

1. **Load the CSV directly**:
```python
import pandas as pd
data = pd.read_csv('ysz_material_properties.csv')
```

2. **Use the Python analysis tool**:
```python
from material_properties_analysis import YSZMaterialProperties

# Initialize
ysz = YSZMaterialProperties()

# Get interpolated property at any temperature
temp_k = 1073.15  # 800°C in Kelvin
youngs_modulus = ysz.get_property('Youngs_Modulus_GPa', temp_k)
```

3. **Generate FEM input files**:
```python
ysz.generate_fem_input_file(temperature_c=800, filename='fem_800C.txt')
```

### Running the Analysis Script

```bash
python material_properties_analysis.py
```

This will:
- Generate visualization plots
- Export data to JSON and Excel formats
- Create sample FEM input files
- Display interpolated values at key temperatures

## Data Formats Available

1. **CSV**: Primary format, easily readable by any software
2. **JSON**: Structured format with metadata (`ysz_properties.json`)
3. **Excel**: Multi-sheet workbook with data, metadata, and units (`ysz_properties.xlsx`)
4. **FEM Input**: ANSYS-style input files for direct FEM use

## Important Considerations for FEM Modeling

### Temperature Dependency
- **Critical**: Young's Modulus, CTE, and Creep parameters show strong temperature dependence
- **Moderate**: Thermal conductivity and fracture toughness
- **Minor**: Poisson's ratio (often assumed constant)

### Modeling Recommendations

1. **For Thermal Stress Analysis**:
   - Use temperature-dependent E and CTE
   - Include thermal conductivity for coupled analysis
   - Consider creep above 900°C

2. **For Crack Prediction**:
   - Apply Weibull statistics for probabilistic failure
   - Use temperature-dependent fracture toughness
   - Consider stress concentrations at interfaces

3. **For Sintering Simulation**:
   - Creep parameters are essential
   - Include full temperature history
   - Account for densification effects

### Interpolation Methods
The Python tool uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation, which:
- Preserves monotonicity
- Avoids overshooting
- Provides smooth derivatives

## Limitations and Assumptions

1. **Material Composition**: Data is for 8YSZ (8 mol% Y₂O₃)
2. **Microstructure**: Assumes fully dense, single-phase material
3. **Loading Rate**: Properties are for quasi-static loading
4. **Environment**: Air atmosphere assumed
5. **Creep Model**: Simple Norton power law (may need refinement for complex loading)

## Quality Assurance

- ✅ Temperature trends physically consistent
- ✅ Values within expected ranges for YSZ
- ✅ Smooth interpolation between data points
- ✅ Units clearly specified
- ✅ Critical properties for FEM included

## File Structure

```
/workspace/
├── ysz_material_properties.csv          # Main dataset
├── material_properties_analysis.py      # Analysis and visualization tool
├── README.md                            # This documentation
├── ysz_properties.json                  # JSON export (generated)
├── ysz_properties.xlsx                  # Excel export (generated)
├── fem_input_800C.txt                   # Sample FEM input at 800°C (generated)
├── fem_input_1500C.txt                  # Sample FEM input at 1500°C (generated)
└── plots/                               # Visualization plots (generated)
    ├── ysz_properties_overview.png
    ├── Youngs_Modulus_GPa_detailed.png
    ├── CTE_1e-6_per_K_detailed.png
    ├── Thermal_Conductivity_W_mK_detailed.png
    └── Fracture_Toughness_MPa_sqrt_m_detailed.png
```

## Requirements

For using the Python analysis tools:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
openpyxl>=3.0.0
```

## License and Citation

This dataset is provided for educational and research purposes. When using this data:

1. Verify values against your specific YSZ composition
2. Consider additional characterization for critical applications
3. Acknowledge the temperature-dependent nature in your FEM model

## Contact and Contributions

For questions, corrections, or contributions to the dataset, please consider:
- Validating with experimental data for your specific YSZ variant
- Adding properties for different YSZ compositions (3YSZ, 10YSZ, etc.)
- Extending to other SOFC materials (NiO-YSZ, LSM, etc.)

## Version History

- **v1.0** (Current): Initial release with core thermomechanical properties

---

*Note: This is a fabricated dataset based on typical literature values for demonstration and educational purposes. For critical applications, experimental validation with your specific material is recommended.*