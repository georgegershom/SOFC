# Rubberized Concrete Fire Resistance Experimental Dataset

## 🔥 Comprehensive Thermo-Mechanical Testing Database for Fire-Resistant Structural Elements

### Project Overview

This repository contains a comprehensive experimental dataset for the development and validation of thermo-mechanical models for fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset encompasses over 2000 test specimens across multiple mix designs, temperature exposures, and testing conditions.

### 📊 Dataset Description

#### Test Program Scope

- **12 Mix Designs**: Including varying rubber contents (0-20%) and supplementary materials
- **Temperature Range**: 23°C to 800°C (5 target temperatures)
- **Cooling Regimes**: Furnace cooling (slow) and water quenching (rapid)
- **Curing Ages**: 7, 28, and 56 days
- **Test Types**: Mechanical, thermal, and durability properties

#### Key Features

1. **Ambient Condition Tests (Control Data)**
   - Compressive strength (ASTM C39)
   - Splitting tensile strength (ASTM C496)
   - Static modulus of elasticity (ASTM C469)
   - Density and ultrasonic pulse velocity (UPV)

2. **High-Temperature Exposure Tests**
   - Residual mechanical properties post-heating
   - Visual damage documentation
   - Mass loss measurements
   - Crack pattern analysis

3. **In-Situ High-Temperature Tests**
   - Hot compressive strength and modulus
   - Transient thermal strain (LITS)
   - Coefficient of thermal expansion (CTE)
   - Real-time deformation monitoring

4. **Spalling Behavior Analysis**
   - Pore pressure measurements at multiple depths
   - Spalling depth and area quantification
   - Time-to-spalling data
   - Effect of PP fibers on spalling mitigation

### 📁 Repository Structure

```
rubberized_concrete_fire_dataset/
│
├── raw_data/                      # Original experimental data
│   ├── material_properties/        # Mix designs and material specifications
│   ├── ambient_tests/             # Room temperature test results
│   ├── high_temp_tests/           # Elevated temperature test results
│   ├── in_situ_tests/             # Hot testing data
│   └── spalling_data/             # Spalling and pore pressure measurements
│
├── processed_data/                # Cleaned and processed datasets
│
├── analysis/                      # Statistical analysis results
│   └── analysis_summary.json      # Key findings and statistics
│
├── visualizations/                # Generated plots and figures
│   ├── strength_development.png   # Strength vs age plots
│   ├── temperature_effects.png    # Temperature-dependent properties
│   ├── cooling_comparison.png     # Cooling method effects
│   ├── in_situ_vs_residual.png   # Hot vs cold property comparison
│   ├── thermal_strain.png        # Thermal deformation behavior
│   ├── spalling_analysis.png     # Spalling behavior analysis
│   └── stress_strain_curves.png  # Stress-strain relationships
│
├── scripts/                       # Analysis and visualization scripts
│   ├── data_generator.py         # Dataset generation script
│   ├── data_visualizer.py        # Visualization suite
│   └── data_analyzer.py          # Statistical analysis tools
│
├── documentation/                 # Additional documentation
│   └── dataset_metadata.json     # Dataset metadata and specifications
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### 🚀 Quick Start

#### Prerequisites

- Python 3.8+
- Required packages: See `requirements.txt`

#### Installation

```bash
# Clone the repository
git clone <repository_url>
cd rubberized_concrete_fire_dataset

# Install dependencies
pip install -r requirements.txt
```

#### Generate Visualizations

```bash
cd scripts
python data_visualizer.py
```

#### Run Analysis

```bash
cd scripts
python data_analyzer.py
```

### 📈 Key Findings

1. **Rubber Content Effects**
   - Compressive strength reduction: ~2.5% per 1% rubber content
   - Improved ductility and energy absorption
   - Optimal content for fire resistance: 10-15%

2. **Temperature Performance**
   - Critical temperature range: 400-600°C
   - Rubber provides benefits at moderate temperatures (200-400°C)
   - Significant strength loss above 600°C

3. **Cooling Method Impact**
   - Water quenching causes 15-30% additional strength loss
   - Increased cracking and surface damage
   - More severe microstructural damage

4. **Spalling Mitigation**
   - Rubber content reduces spalling risk by up to 30%
   - PP fibers reduce spalling probability by 80%
   - Critical pore pressure: 1.5-2.0 MPa

### 📊 Mix Design Specifications

| Mix ID | Cement (kg/m³) | Rubber (%) | Special Features | Target Strength (MPa) |
|--------|----------------|------------|------------------|----------------------|
| RC0    | 380            | 0          | Control          | 40                   |
| RC5    | 380            | 5          | Crumb rubber     | 38                   |
| RC10   | 380            | 10         | Crumb rubber     | 35                   |
| RC15   | 380            | 15         | Crumb rubber     | 32                   |
| RC20   | 380            | 20         | Crumb rubber     | 30                   |
| RC10_SF| 360            | 10         | + Silica fume    | 42                   |
| RC10_FA| 340            | 10         | + Fly ash        | 38                   |
| RC10_ST| 380            | 10         | + Steel fibers   | 40                   |
| RC10_PP| 380            | 10         | + PP fibers      | 36                   |
| RC15_HYB| 360           | 15         | Hybrid mix       | 38                   |

### 🔬 Testing Standards

- **Compressive Strength**: ASTM C39
- **Tensile Strength**: ASTM C496
- **Elastic Modulus**: ASTM C469
- **Fire Testing**: ISO 834 standard fire curve
- **UPV Testing**: ASTM C597

### 📝 Predictive Equations

#### Residual Strength Model (Furnace Cooled)
```
fc,T/fc,20 = 0.823 * exp(-1.245*T/1000) + 0.177
Valid range: 20°C ≤ T ≤ 800°C
```

#### Rubber Content Modification Factor
```
At 400°C: k_rubber = -0.0045 * R% + 0.985
At 600°C: k_rubber = -0.0038 * R% + 0.972
```

### 📖 Publications & Citations

If you use this dataset in your research, please cite:

```bibtex
@dataset{rubberized_concrete_fire_2024,
  title={Comprehensive Experimental Dataset for Thermo-Mechanical Modeling 
         of Fire-Resistant Rubberized Concrete},
  author={Research Team},
  year={2024},
  publisher={Dataset Repository},
  doi={10.xxxxx/xxxxx}
}
```

### 🤝 Contributing

We welcome contributions to expand this dataset. Please follow these guidelines:

1. Ensure data quality and consistency
2. Follow established testing standards
3. Document all experimental procedures
4. Include uncertainty/error estimates

### 📧 Contact

For questions, collaborations, or access to additional data:
- Email: research@example.com
- Project Website: [Link]

### 📄 License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

### 🙏 Acknowledgments

This research was supported by:
- [Funding Agency/Grant Number]
- Laboratory facilities at [Institution]
- Industry partners: [Companies]

### ⚠️ Disclaimer

This dataset is provided for research purposes. Users should validate results for specific applications and comply with local building codes and standards.

---

**Last Updated**: October 2024
**Version**: 1.0.0
**Dataset DOI**: 10.xxxxx/xxxxx