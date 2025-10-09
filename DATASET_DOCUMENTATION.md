# SOFC Material Property Dataset Documentation

## Overview

This comprehensive dataset contains material property data for Solid Oxide Fuel Cell (SOFC) components, including elastic properties, fracture properties, interface characteristics, thermal expansion coefficients, and chemical expansion behavior. The dataset is designed to support computational modeling, machine learning applications, and materials optimization for SOFC systems.

## Dataset Structure

### Files Generated

1. **JSON Master Database**: `material_property_dataset.json`
   - Comprehensive hierarchical data structure with all base properties
   - Mean values, standard deviations, and temperature dependencies
   - Processing effects and measurement methods
   - Literature references

2. **CSV Datasets** (with timestamps):
   - `sofc_material_elastic_properties_*.csv` - 1000 samples
   - `sofc_material_fracture_properties_*.csv` - 1000 samples
   - `sofc_material_interface_properties_*.csv` - 498 samples
   - `sofc_material_thermal_properties_*.csv` - 1000 samples
   - `sofc_material_chemical_expansion_*.csv` - 498 samples
   - `sofc_material_summary_statistics.csv` - Statistical summary

3. **Visualizations**: `material_property_visualizations/`
   - Analysis plots for each property category
   - Correlation matrices
   - Distribution histograms
   - Temperature and processing dependencies

## Materials Included

### Electrolyte Materials
- **YSZ-8mol**: 8 mol% Yttria-Stabilized Zirconia (cubic phase)
- **YSZ-3mol**: 3 mol% Yttria-Stabilized Zirconia (tetragonal phase)
- **GDC**: Gadolinium Doped Ceria (Ce₀.₉Gd₀.₁O₁.₉₅)

### Anode Materials
- **Ni**: Metallic Nickel (FCC)
- **NiO**: Nickel Oxide (rocksalt structure)
- **Ni-YSZ**: Nickel-YSZ Cermet (various compositions)

### Cathode Materials
- **LSM**: La₀.₈Sr₀.₂MnO₃ (perovskite)
- **LSCF**: La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃ (perovskite)
- **LSC**: La₀.₆Sr₀.₄CoO₃ (perovskite)

### Interconnect Materials
- **Crofer22APU**: Chromium-Iron Alloy
- **SUS430**: Ferritic Stainless Steel

## Property Categories

### 1. Elastic Properties
- **Young's Modulus (E)**: GPa
- **Poisson's Ratio (ν)**: dimensionless
- **Shear Modulus (G)**: GPa (calculated)
- **Bulk Modulus (K)**: GPa (calculated)

**Key Dependencies**:
- Temperature (25-1000°C)
- Porosity (0-50%)
- Grain size (0.1-10 μm)
- Processing method (sintered, tape cast, screen printed, plasma sprayed)

### 2. Fracture Properties
- **Fracture Toughness (K_IC)**: MPa·m^0.5
- **Critical Energy Release Rate (G_c)**: J/m²
- **Weibull Modulus**: dimensionless (brittle materials)
- **Characteristic Strength**: MPa
- **J-Integral (J_IC)**: kJ/m² (ductile materials)

**Key Features**:
- R-curve behavior for transformation-toughened materials
- Mode mixity effects
- Environmental sensitivity (humidity, H₂ atmosphere)
- Loading rate dependence

### 3. Interface Properties
- **Interface Toughness**: MPa·m^0.5
- **Adhesion Energy**: J/m²
- **Residual Stress**: MPa
- **Debonding Characteristics**: mode mixity, critical angles

**Interfaces Studied**:
- Ni/YSZ
- Ni-YSZ/YSZ (anode/electrolyte)
- LSM/YSZ, LSCF/GDC, LSC/YSZ (cathode/electrolyte)
- GDC/YSZ (barrier layer)

**Degradation Factors**:
- Thermal cycling (up to 1000 cycles)
- Redox cycling (up to 100 cycles for Ni-containing)
- Processing method effects

### 4. Thermal Properties
- **Coefficient of Thermal Expansion (CTE)**: 10⁻⁶/K
- **Thermal Conductivity**: W/m·K
- **Specific Heat Capacity**: J/g·K

**Temperature Ranges**: 25-1000°C
**Anisotropy**: Included for non-cubic materials

### 5. Chemical Expansion
- **Linear Strain**: dimensionless
- **Volume Change**: fraction
- **Oxygen Nonstoichiometry Coefficient**: for perovskites and ceria
- **Redox Strain**: for Ni/NiO transformation

**Key Mechanisms**:
- Ni ↔ NiO oxidation/reduction (69.5% volume change)
- Oxygen vacancy formation in perovskites
- Ce⁴⁺ → Ce³⁺ reduction in GDC
- Redox cycling damage accumulation

## Data Generation Methodology

### Base Values
- Literature-derived mean values and standard deviations
- Temperature-dependent properties from experimental data
- Processing-structure-property relationships

### Synthetic Variations
- **Gaussian noise**: 5% of standard deviation
- **Outliers**: 2% of data points
- **Physical constraints**: Applied to ensure realistic bounds
- **Correlations**: Preserved between related properties

### Models Used
1. **Porosity Effect on Modulus**: E = E₀(1 - 1.9P + 0.9P²)
2. **Temperature Dependence**: Linear/exponential models
3. **Chemical Expansion Kinetics**: 1 - exp(-t/τ) model
4. **Interface Degradation**: Exponential decay with cycles

## Usage Guidelines

### For Computational Modeling
- Use mean values from JSON file for baseline simulations
- Apply temperature-dependent properties for thermal stress analysis
- Consider interface properties for delamination studies
- Include chemical expansion for redox cycling simulations

### For Machine Learning
- CSV files provide training data with realistic variations
- Consider data normalization due to property range differences
- Use correlation matrices to identify feature relationships
- Split by material type for material-specific models

### For Uncertainty Quantification
- Standard deviations provided for Monte Carlo simulations
- Processing variations capture manufacturing uncertainty
- Environmental conditions represent operational variability

## Quality Assurance

### Data Validation
- Physical bounds enforced (e.g., 0 < ν < 0.5)
- Cross-property consistency checked
- Temperature trends verified against physics

### Known Limitations
1. Interface properties are challenging to measure experimentally
2. High-temperature data (>800°C) has higher uncertainty
3. Chemical expansion kinetics simplified to first-order models
4. Microstructure effects simplified to grain size and porosity

## References

### Key Literature Sources
1. **YSZ Properties**: Selçuk & Atkinson (2000), Radovic & Lara-Curzio (2004)
2. **Ni-YSZ Cermets**: Pihlatie et al. (2009), Atkinson & Selçuk (2000)
3. **Interface Properties**: Laurencin et al. (2012), Faes et al. (2009)
4. **Perovskite Properties**: Bishop et al. (2014), Chen et al. (2005)
5. **Chemical Expansion**: Kuhn et al. (2011), Adler (2004)

### Measurement Techniques Referenced
- **Elastic Properties**: Nanoindentation, resonant ultrasound spectroscopy
- **Fracture Properties**: SENB, DCB, chevron notch, indentation
- **Thermal Properties**: Dilatometry, TMA, high-temperature XRD
- **Chemical Expansion**: In-situ XRD, optical dilatometry

## Data Access and Updates

### Current Version
- Version: 1.0.0
- Generated: 2025-10-09
- Total Samples: ~4000 across all properties

### File Formats
- **JSON**: Hierarchical structure with metadata
- **CSV**: Tabular format for analysis tools
- **PNG**: Visualization outputs

### Future Enhancements
- Additional materials (BSCF, SDC, metallic interconnects)
- Creep and fatigue properties
- Electrical/ionic conductivity data
- Microstructure-property correlations
- Time-dependent degradation models

## Citation

If you use this dataset in your research, please acknowledge:

```
SOFC Material Property Dataset v1.0
Generated using literature-based models and synthetic variations
Created: October 2025
```

## Contact and Contributions

This dataset represents a synthesis of published experimental data with physically-based models for property variations. Users are encouraged to validate against their specific experimental conditions and contribute improvements or additional data.

---

*Note: This is a synthetic dataset generated for research and development purposes. While based on literature values, individual measurements should be validated experimentally for critical applications.*