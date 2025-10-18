# Thermo-Mechanical Modeling Dataset - Complete Summary

## 📁 Project Overview

This comprehensive dataset generator has been successfully created for the research:
**"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete"**

## ✅ Completed Components

### 1. Core Dataset Generation System
- ✓ Main dataset generator class (`ThermoMechanicalDataset`)
- ✓ Multi-physics property coupling
- ✓ Temperature-dependent functions (20°C to 800°C)
- ✓ Stochastic variation implementation
- ✓ Calibration/validation data splitting

### 2. Property Generators

#### Thermal Properties (`thermal_properties.py`)
- Thermal conductivity with degradation
- Specific heat capacity with phase changes
- Thermal diffusivity
- Density evolution
- Thermal expansion coefficients
- Surface emissivity
- Phase change temperatures and latent heats

#### Mechanical Properties (`mechanical_properties.py`)
- **Elastic Properties**: E, ν, G, K with temperature degradation
- **Strength Properties**: fc, ft, flexural strength, fracture energy
- **Plastic Properties**: Yield stress, hardening, dilation angle
- **Damage Evolution**: Initiation, evolution laws, stiffness degradation
- **Creep Properties**: Compliance, activation energy, power law

#### Transport Properties (`transport_properties.py`)
- **Porosity Evolution**: Total, connected, capillary, gel, crack-induced
- **Permeability**: Intrinsic, gas, liquid, relative permeabilities
- **Moisture Transport**: Content, diffusivity, sorption isotherms
- **Pore Pressure**: Vapor, capillary, gas buildup
- **Spalling Risk**: Risk index, critical pressure, moisture clog

### 3. FEA Software Exporters (`fea_exporters.py`)

#### ABAQUS Exporter
- Material property .inp files
- Amplitude curves (ISO 834, ASTM E119, Hydrocarbon)
- Field variable dependencies
- Concrete Damaged Plasticity parameters

#### ANSYS Exporter
- APDL macro files
- Table arrays for temperature dependencies
- TB commands for concrete material
- Master analysis macro

#### COMSOL Exporter
- Java API files
- Interpolation functions
- Material function definitions
- Multiphysics coupling setup

### 4. Data Management & Visualization
- ✓ JSON and CSV export formats
- ✓ Comprehensive validation checks
- ✓ Property evolution plots
- ✓ Rubber effect analysis
- ✓ Statistical summaries

## 🎯 Key Features Implemented

### Physical Consistency
- **Temperature Degradation**: Follows established models (ISO 834, Eurocode 2)
- **Multi-Stage Decomposition**: Moisture (20-200°C), Dehydration (200-400°C), Decomposition (>400°C)
- **Rubber Effects**: Properly accounts for rubber content on all properties

### Multi-Physics Coupling
- **Thermal-Mechanical**: Thermal expansion, temperature-dependent stiffness
- **Poro-Mechanical**: Biot coefficients, effective stress
- **Thermal-Transport**: Moisture diffusivity, vapor pressure
- **Fully Coupled**: All properties maintain physical interdependence

### Model-Ready Formatting
- Direct import into commercial FEA software
- Proper units and temperature increments
- Constitutive model parameters included
- Mesh sensitivity considerations

## 📊 Dataset Specifications

### Mix Designs
| Mix | Rubber % | Size | Properties Generated |
|-----|----------|------|---------------------|
| C | 0% | - | Complete set |
| R5S | 5% | Small (2mm) | Complete set |
| R10S | 10% | Small (2mm) | Complete set |
| R15S | 15% | Small (2mm) | Complete set |
| R20S | 20% | Small (2mm) | Complete set |
| R10L | 10% | Large (8mm) | Complete set |

### Temperature Range
- **Min**: 20°C (ambient)
- **Max**: 800°C (severe fire)
- **Points**: 40-100 per property
- **Critical**: 100, 200, 300, 400, 500, 600, 700°C

### Data Organization
```
Total Properties per Mix:
- Thermal: 7 primary + degradation functions
- Mechanical: 20+ including damage evolution
- Transport: 15+ including pore pressure
- Coupling: 8+ parameters
- Total data points: ~2000 per mix
```

## 💻 Usage Instructions

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Generate with defaults
python3 generate_dataset.py

# Custom parameters
python3 generate_dataset.py \
    --output-dir ./my_data \
    --uncertainty 0.1 \
    --n-validation-sets 10 \
    --visualize \
    --validate
```

### Python API
```python
from thermo_mechanical_dataset import ThermoMechanicalDataset

# Initialize
dataset = ThermoMechanicalDataset(
    output_dir='./output',
    seed=42,
    uncertainty_level=0.05
)

# Generate
cal_data, val_data = dataset.generate_complete_dataset()
```

## 📈 Validation Features

1. **Physical Consistency Checks**
   - Monotonic property degradation
   - Thermodynamic constraints
   - Material bounds

2. **Data Completeness**
   - All required fields present
   - Temperature range coverage
   - Coupling parameters

3. **Statistical Validation**
   - Stochastic variation applied
   - Uncertainty bounds maintained
   - Distribution checks

## 🔬 Scientific Basis

The dataset incorporates established models:
- **Thermal**: Bazant & Kaplan (1996) degradation models
- **Mechanical**: Concrete Damaged Plasticity (Lee & Fenves, 1998)
- **Transport**: Darcy-Fick laws with Kozeny-Carman permeability
- **Fire Curves**: ISO 834, ASTM E119, Eurocode 2

## 📦 Output Files

```
output/
├── calibration/           # Training data
├── validation/            # Testing data (multiple sets)
├── fea_inputs/           # Software-specific formats
│   ├── abaqus/          # .inp files
│   ├── ansys/           # .mac files
│   └── comsol/          # .java files
├── visualizations/       # Generated plots
├── dataset_summary.json  # Complete statistics
└── validation_report.txt # Quality checks
```

## 🚀 Next Steps for Users

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Generate dataset**: `python3 generate_dataset.py`
3. **Review output**: Check `output/dataset_summary.md`
4. **Import to FEA**: Use files in `output/fea_inputs/`
5. **Customize**: Modify property generators as needed

## ⚠️ Important Notes

- Requires Python 3.8+ with NumPy, Pandas, SciPy, Matplotlib
- Generated data maintains physical consistency
- Stochastic variations included for uncertainty quantification
- All temperature dependencies are continuous functions
- Microstructure parameters directly inform macro-properties

## 📚 References

The dataset generator is based on:
- ISO 834-1: Fire resistance tests
- ASTM E119: Fire test methods
- Eurocode 2: Concrete fire design
- fib Bulletin 38: Fire design guidelines
- Extensive literature on rubberized concrete

---

**Dataset Generator Version**: 1.0.0  
**Status**: ✅ Complete and Ready for Use  
**Last Updated**: 2024