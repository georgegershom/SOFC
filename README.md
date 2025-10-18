# Thermo-Mechanical Modeling Dataset Generator

## Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

This comprehensive dataset generator creates physically consistent, model-ready numerical data for finite element analysis of fire-resistant rubberized concrete structures.

## 🎯 Key Features

- **Multi-Physics Coupling**: Generates interdependent thermal, mechanical, and transport properties
- **Temperature-Dependent Functions**: Continuous property evolution from 20°C to 800°C
- **FEA Software Ready**: Direct export to ABAQUS, ANSYS, and COMSOL formats
- **Calibration-Validation Split**: Separate datasets for model development and validation
- **Stochastic Bounds**: Statistical variations for probabilistic modeling
- **Multi-Scale Linking**: Microstructural parameters inform macro-scale properties

## 📋 Requirements

- Python 3.8+
- NumPy, Pandas, SciPy, Matplotlib, Seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```bash
# Generate complete dataset with default parameters
python generate_dataset.py

# Generate with custom parameters
python generate_dataset.py --output-dir ./my_data --uncertainty 0.1 --n-validation-sets 10

# Generate with visualizations and validation
python generate_dataset.py --visualize --validate
```

### Command Line Options

- `--output-dir`: Output directory for datasets (default: ./output)
- `--seed`: Random seed for reproducibility (default: 42)
- `--uncertainty`: Uncertainty level as coefficient of variation (default: 0.05)
- `--calibration-ratio`: Fraction of data for calibration (default: 0.7)
- `--n-validation-sets`: Number of independent validation datasets (default: 5)
- `--visualize`: Generate visualization plots
- `--validate`: Run validation checks on generated data

## 📊 Dataset Structure

```
output/
├── calibration/              # Calibration dataset
│   ├── complete_dataset.json # Complete JSON format
│   ├── C/                    # Control mix
│   ├── R5S/                  # 5% small rubber
│   ├── R10S/                 # 10% small rubber
│   ├── R15S/                 # 15% small rubber
│   ├── R20S/                 # 20% small rubber
│   └── R10L/                 # 10% large rubber
├── validation/               # Validation datasets
│   ├── Set_1/
│   ├── Set_2/
│   └── ...
├── fea_inputs/              # FEA software inputs
│   ├── abaqus/              # ABAQUS .inp files
│   ├── ansys/               # ANSYS APDL macros
│   └── comsol/              # COMSOL Java API
├── visualizations/          # Generated plots
└── documentation/           # Reports and summaries
```

## 🏗️ Mix Identifications

| Mix ID | Description | Rubber Content | Particle Size |
|--------|-------------|----------------|---------------|
| C | Control | 0% | - |
| R5S | 5% small rubber | 5% | 2±0.5 mm |
| R10S | 10% small rubber | 10% | 2±0.5 mm |
| R15S | 15% small rubber | 15% | 2±0.5 mm |
| R20S | 20% small rubber | 20% | 2±0.5 mm |
| R10L | 10% large rubber | 10% | 8±2.0 mm |

## 📈 Properties Generated

### Thermal Properties
- Thermal conductivity (temperature-dependent)
- Specific heat capacity
- Thermal diffusivity
- Density evolution
- Thermal expansion coefficient
- Surface emissivity
- Phase change parameters

### Mechanical Properties
- **Elastic**: Modulus, Poisson's ratio, shear/bulk moduli
- **Strength**: Compressive, tensile, flexural strengths
- **Plastic**: Yield stress, hardening, dilation angle
- **Damage**: Initiation, evolution, stiffness degradation
- **Creep**: Compliance, activation energy, power law parameters

### Transport Properties
- **Porosity**: Total, connected, capillary, gel, crack-induced
- **Permeability**: Intrinsic, gas, liquid, relative
- **Moisture**: Content, diffusivity, sorption isotherms
- **Pore Pressure**: Vapor, capillary, gas buildup
- **Spalling**: Risk index, critical pressure, moisture clog

### Multi-Physics Coupling
- Thermal-mechanical coupling coefficients
- Biot poro-mechanical parameters
- Thermal-transport coupling factors
- Moisture-mechanical interactions

## 🔧 FEA Software Integration

### ABAQUS
```abaqus
*Include, input=R10S/R10S_material.inp
*Material, name=CONCRETE_R10S
*Concrete Damaged Plasticity
*Concrete Compression Hardening
*Concrete Tension Stiffening
```

### ANSYS APDL
```apdl
/PREP7
*USE,R10S/R10S_material.mac
TB,CONCR,MAT_R10S,,,,MISO
MP,EX,MAT_R10S,%EX_R10S%
```

### COMSOL
```java
model.func().create("k_R10S", "Interpolation");
model.component("comp1").material().create("mat_R10S", "Common");
```

## 📐 Physical Models Implemented

### Temperature Degradation
- ISO 834 fire curve compatibility
- ASTM E119 compliance
- Eurocode 2 degradation models
- Multi-stage decomposition kinetics

### Constitutive Models
- Concrete Damaged Plasticity (CDP)
- Temperature-dependent elasticity
- Drucker-Prager plasticity
- Exponential damage evolution

### Transport Models
- Darcy flow for porous media
- Fick's law for diffusion
- Kozeny-Carman permeability
- BET sorption isotherms

## 🔬 Validation

The generator includes comprehensive validation checks:

1. **Physical Consistency**: Ensures monotonic property degradation
2. **Data Completeness**: Verifies all required fields
3. **Temperature Coverage**: Checks full range (20-800°C)
4. **Stochastic Variation**: Confirms uncertainty application
5. **Coupling Parameters**: Validates multi-physics links

Run validation:
```bash
python generate_dataset.py --validate
```

## 📊 Visualization

Generate comprehensive plots of all properties:

```bash
python generate_dataset.py --visualize
```

Outputs include:
- Thermal property evolution
- Mechanical property degradation
- Transport property changes
- Rubber content effect analysis

## 🎯 Example Python Usage

```python
from thermo_mechanical_dataset import ThermoMechanicalDataset

# Initialize generator
dataset = ThermoMechanicalDataset(
    output_dir='./my_output',
    seed=42,
    uncertainty_level=0.05
)

# Generate complete dataset
calibration_data, validation_data = dataset.generate_complete_dataset(
    calibration_ratio=0.7,
    n_validation_sets=5
)

# Access specific property
thermal_conductivity = calibration_data['R10S']['thermal']['thermal_conductivity']
temperatures = calibration_data['R10S']['thermal']['temperature']

# Plot results
import matplotlib.pyplot as plt
plt.plot(temperatures, thermal_conductivity)
plt.xlabel('Temperature (°C)')
plt.ylabel('Thermal Conductivity (W/m·K)')
plt.show()
```

## 📚 References

- ISO 834-1: Fire-resistance tests — Elements of building construction
- ASTM E119: Standard Test Methods for Fire Tests
- Eurocode 2: Design of concrete structures - Part 1-2: Fire design
- fib Bulletin 38: Fire design of concrete structures
- Bažant, Z. P., & Kaplan, M. F. (1996). Concrete at high temperatures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided for research purposes. Please cite appropriately when using this dataset in publications.

## 📧 Contact

For questions or support regarding this dataset generator, please contact the research team.

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Developed for:** Thermo-Mechanical Modeling Research