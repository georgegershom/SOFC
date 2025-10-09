# SOFC Material Property Dataset

## Comprehensive Material Property Database for Solid Oxide Fuel Cells

This repository contains a comprehensive, fabricated dataset of material properties for Solid Oxide Fuel Cell (SOFC) materials, including elastic properties, fracture properties, thermo-physical properties, and chemical expansion coefficients.

### 🎯 Dataset Overview

The dataset includes properties for:
- **YSZ (8 mol% Yttria-Stabilized Zirconia)** - Electrolyte material
- **Ni (Nickel)** - Anode current collector
- **Ni-YSZ Composites** - Anode functional layer (various volume fractions)
- **Critical Interfaces** - Anode/electrolyte and Ni/YSZ interfaces

### 📊 Material Properties Included

#### 1. Elastic Properties
- **Young's Modulus (E)** - GPa
- **Poisson's Ratio (ν)** - dimensionless
- **Shear Modulus (G)** - GPa
- **Bulk Modulus (K)** - GPa

#### 2. Fracture Properties
- **Critical Energy Release Rate (G_c)** - J/m²
- **Fracture Toughness (K_Ic)** - MPa√m
- **Crack Growth Exponent** - dimensionless

#### 3. Thermo-Physical Properties
- **Coefficient of Thermal Expansion (CTE)** - K⁻¹
- **Thermal Conductivity** - W/m·K
- **Specific Heat** - J/kg·K

#### 4. Chemical Expansion Properties
- **Chemical Expansion Coefficient** - dimensionless
- **Oxidation State Range** - dimensionless
- **Activation Energy** - eV

### 🏗️ Repository Structure

```
├── material_property_dataset.py          # Main database with all material properties
├── experimental_data_generator.py        # Synthetic experimental data generation
├── data_validation_and_analysis.py       # Data validation and uncertainty quantification
├── run_material_dataset_generation.py    # Main execution script
├── README.md                            # This documentation
└── Generated Files:
    ├── sofc_material_properties_complete.csv    # Complete dataset (CSV)
    ├── sofc_material_properties_complete.json   # Complete dataset (JSON)
    ├── sofc_validation_report.json             # Validation report
    ├── sofc_materials_summary.csv              # Summary table
    ├── sofc_interfaces_summary.csv             # Interface properties
    └── sofc_experimental_*.csv                 # Experimental datasets
```

### 🚀 Quick Start

1. **Generate Complete Dataset:**
   ```bash
   python run_material_dataset_generation.py
   ```

2. **Load Database in Python:**
   ```python
   from material_property_dataset import SOFCMaterialDatabase
   
   # Initialize database
   db = SOFCMaterialDatabase()
   
   # Get YSZ properties at 800°C
   ysz_props = db.get_material_properties('YSZ')
   print(f"YSZ Young's Modulus: {ysz_props['elastic_1073K'].youngs_modulus_GPa.value} GPa")
   
   # Get interface properties
   interface_props = db.get_interface_properties('anode_electrolyte')
   print(f"Interface Toughness: {interface_props['fracture'].fracture_toughness_MPa_sqrt_m.value} MPa√m")
   ```

3. **Generate Experimental Data:**
   ```python
   from experimental_data_generator import generate_comprehensive_experimental_dataset
   
   experimental_data = generate_comprehensive_experimental_dataset(db)
   ```

### 📈 Key Material Properties Summary

| Material | E (GPa) | ν | K_Ic (MPa√m) | CTE (×10⁻⁶/K) |
|----------|---------|---|--------------|----------------|
| YSZ | 165.0 ± 12.0 | 0.330 ± 0.025 | 2.2 ± 0.3 | 10.8 ± 0.5 |
| Ni | 155.0 ± 8.0 | 0.330 ± 0.015 | 85.0 ± 10.0 | 16.8 ± 0.3 |
| Ni-YSZ (40%) | 159.8 ± 2.5 | 0.330 ± 0.000 | 5.4 ± 1.4 | 13.2 ± 1.2 |
| Ni-YSZ (50%) | 159.7 ± 2.5 | 0.330 ± 0.000 | 6.2 ± 1.6 | 13.8 ± 1.5 |

### 🔗 Critical Interface Properties

| Interface | G_c (J/m²) | K_Ic (MPa√m) | Criticality |
|-----------|------------|--------------|-------------|
| Anode/Electrolyte | 12.0 ± 4.0 | 1.1 ± 0.4 | HIGH |
| Ni/YSZ | 8.5 ± 2.5 | 0.8 ± 0.3 | MEDIUM |

### 🌡️ Temperature Dependencies

All properties include temperature dependencies from 25°C to 1000°C:
- **Young's Modulus**: Decreases ~20% from RT to 800°C
- **Thermal Expansion**: Nonlinear temperature dependence
- **Interface Properties**: Most temperature-sensitive

### 🧪 Experimental Data Features

The synthetic experimental datasets include:
- **Nanoindentation**: Load-displacement curves with Oliver-Pharr analysis
- **Fracture Testing**: Compact tension and interface fracture specimens
- **Thermal Analysis**: Dilatometry and chemical expansion measurements
- **Realistic Artifacts**: Measurement noise, systematic errors, outliers

### 📊 Data Validation

Comprehensive validation includes:
- **Statistical Tests**: Normality, outlier detection, confidence intervals
- **Physical Constraints**: Range validation, coefficient of variation limits
- **Uncertainty Quantification**: Type A/B uncertainties, Monte Carlo propagation
- **Quality Scoring**: 0-100 scale based on precision, accuracy, reliability

### 🔍 Key Insights

1. **Interface Criticality**: Anode/electrolyte interface is the weakest link (K_Ic = 1.1 MPa√m)
2. **CTE Mismatch**: Primary driver of thermal stress (6×10⁻⁶/K difference)
3. **Chemical Expansion**: Ni oxidation causes 21% volume expansion
4. **Composite Optimization**: 40-50% Ni volume fraction recommended
5. **Temperature Effects**: Properties degrade significantly at operating temperature

### 📋 Applications

This dataset is designed for:
- **Finite Element Modeling**: Complete property sets with uncertainties
- **Failure Analysis**: Critical interface properties for delamination prediction
- **Design Optimization**: Trade-offs between performance and durability
- **Reliability Assessment**: Statistical distributions for Monte Carlo analysis
- **Material Development**: Baseline properties for new material evaluation

### 🔬 Data Sources and Methods

Properties derived from:
- **Literature Review**: Peer-reviewed publications (2000-2023)
- **Nanoindentation**: Oliver-Pharr method for elastic properties
- **Fracture Testing**: Compact tension, double cantilever beam
- **Thermal Analysis**: Dilatometry, DSC, TGA
- **Atomistic Simulations**: DFT calculations for interface properties
- **Micromechanical Models**: Voigt-Reuss-Hill averaging for composites

### ⚠️ Important Notes

1. **Fabricated Data**: This is a synthetic dataset for research and educational purposes
2. **Realistic Uncertainties**: All properties include realistic measurement uncertainties
3. **Temperature Dependence**: Properties valid for 25-1000°C range
4. **Interface Focus**: Special emphasis on critical interface properties
5. **Validation Required**: Experimental validation recommended for critical applications

### 📚 References

Key literature sources include:
- Atkinson & Selçuk (2003) - YSZ mechanical properties
- Evans et al. (2001) - SOFC failure mechanisms
- Malzbender & Steinbrech (2007) - Interface fracture
- Pihlatie et al. (2009) - Ni chemical expansion
- Nakajo et al. (2012) - Thermal cycling effects

### 🤝 Contributing

To extend or improve this dataset:
1. Add new materials in `material_property_dataset.py`
2. Implement new experimental methods in `experimental_data_generator.py`
3. Enhance validation criteria in `data_validation_and_analysis.py`
4. Update documentation and examples

### 📄 License

This dataset is provided for research and educational purposes. Please cite appropriately if used in publications.

### 📞 Contact

For questions or suggestions regarding this dataset, please open an issue in the repository.

---

**Generated for SOFC Research - 2025-10-09**