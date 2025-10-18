# Fire-Resistant Rubberized Concrete Dataset Generator

## Overview

This project provides a comprehensive dataset generation system for **Fire-Resistant High-Performance Rubberized Concrete** research. The system generates realistic synthetic datasets covering all aspects of material characterization, mix design, fresh properties, mechanical testing, thermal analysis, and fire resistance testing.

## Research Focus

**Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete**

## Features

### 🔬 **Comprehensive Material Characterization**
- **Cement Properties**: Chemical composition (XRF), Bogue composition, thermal properties
- **Aggregate Properties**: Coarse & fine aggregates with sieve analysis and thermal characteristics
- **Crumb Rubber**: Detailed particle size distribution, chemical composition (TGA, FTIR), morphology (SEM)
- **Water Quality**: pH, impurity content, temperature
- **Chemical Admixtures**: Superplasticizers, air-entraining agents, fire-resistant additives

### 🏗️ **Advanced Mix Design Matrix**
- **Control Mixes**: Baseline concrete without rubber
- **Rubber Content Variation**: 5%, 10%, 15%, 20%, 25%, 30% by volume replacement
- **Rubber Size Variation**: Multiple size ranges (0.075-1mm, 1-2mm, 2-4mm, 4-8mm, 8-12mm)
- **Cement Content Variation**: 300, 400, 500 kg/m³
- **Water-Cement Ratio**: 0.35, 0.40, 0.45, 0.50, 0.55
- **Fire-Resistant Additives**: Fiber content, fire-retardant chemicals

### 🧪 **Fresh Concrete Properties**
- **Workability**: Slump, flow table spread, V-funnel time, L-box height ratio
- **Air Content**: Pressure method and gravimetric method
- **Unit Weight**: Fresh concrete density
- **Rheological Properties**: Yield stress, plastic viscosity, thixotropy index

### 🔥 **Thermo-Mechanical Testing**
- **Mechanical Properties**: Compressive, tensile, flexural strength, elastic modulus, Poisson's ratio
- **Temperature Range**: 20°C to 800°C
- **Loading Rates**: 0.25, 0.5, 1.0 MPa/s
- **Heating Rates**: 5, 10, 20°C/min
- **Age at Testing**: 7, 14, 28, 56, 90, 180 days

### 🌡️ **Thermal Properties**
- **Thermal Conductivity**: Hot wire method, temperature-dependent
- **Specific Heat Capacity**: DSC analysis, temperature-dependent
- **Thermal Expansion**: Dilatometry, temperature-dependent
- **Thermal Diffusivity**: Laser flash method, temperature-dependent

### 🚨 **Fire Resistance Testing**
- **Fire Resistance Rating**: ASTM E119, 30-180 minutes
- **Spalling Resistance**: Explosive spalling test
- **Smoke Production**: Smoke density chamber analysis
- **Heat Release Rate**: Cone calorimeter analysis

## Quick Start

### 1. **Installation**
```bash
# Clone or download the project
cd fire-resistant-rubberized-concrete-dataset

# Install dependencies
pip install -r data_generation_scripts/requirements.txt
```

### 2. **Generate Datasets**
```bash
# Run the complete dataset generation
python run_data_generation.py
```

### 3. **Access Generated Data**
The system will create a `generated_datasets/` directory containing:
- `material_properties_data.json` - Constituent materials data
- `mix_design_data.json` - Mix designs and fresh properties
- `thermo_mechanical_testing_data.json` - Mechanical and thermal testing data
- `analysis/` - Comprehensive analysis and visualizations
- `README.md` - Usage instructions
- `summary_report.json` - Detailed summary statistics

## File Structure

```
fire-resistant-rubberized-concrete-dataset/
├── enhanced_rubberized_concrete_dataset_prompt.md  # Enhanced research prompt
├── data_generation_scripts/
│   ├── material_properties_generator.py           # Material characterization
│   ├── mix_design_generator.py                    # Mix design matrix
│   ├── thermo_mechanical_testing_generator.py     # Testing protocols
│   ├── master_data_generator.py                   # Orchestrates all generators
│   └── requirements.txt                           # Python dependencies
├── run_data_generation.py                         # Main execution script
└── README.md                                      # This file
```

## Generated Datasets

### 1. **Material Properties Data**
```json
{
  "cement": {
    "chemical_composition": {...},
    "bogue_composition": {...},
    "physical_properties": {...},
    "thermal_properties": {...}
  },
  "aggregates": {
    "coarse": {...},
    "fine": {...},
    "thermal_properties": {...}
  },
  "rubber": {
    "physical_properties": {...},
    "chemical_composition": {...},
    "thermal_analysis": {...},
    "fire_resistance": {...}
  }
}
```

### 2. **Mix Design Data**
```json
{
  "mix_designs": {
    "1": {
      "mix_id": 1,
      "mix_type": "control",
      "cement_content": 400,
      "water_content": 180,
      "w_c_ratio": 0.45,
      "rubber_fine_content": 0,
      "fiber_content": 0,
      "fire_retardant_content": 0
    }
  },
  "fresh_properties": {
    "1": {
      "slump": 150,
      "flow_spread": 500,
      "air_content_pressure": 4.5,
      "unit_weight": 2400,
      "yield_stress": 50,
      "plastic_viscosity": 2.5
    }
  }
}
```

### 3. **Testing Data**
```json
{
  "mechanical_properties": {
    "1": {
      "compressive_strength": {
        "age_28_temp_20_loading_0.5_heating_10": {
          "values": [38.5, 39.2, 37.8, 40.1, 38.9, 39.5],
          "mean": 39.0,
          "std": 0.8,
          "cov": 0.02
        }
      }
    }
  },
  "thermal_properties": {
    "1": {
      "thermal_conductivity": {
        "temperature": [20, 100, 200, ...],
        "conductivity": [2.0, 1.8, 1.6, ...]
      }
    }
  }
}
```

## Usage Examples

### Python Analysis
```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
with open('generated_datasets/mix_design_data.json', 'r') as f:
    mix_data = json.load(f)

# Convert to DataFrame for analysis
mix_df = pd.DataFrame(mix_data['mix_designs']).T

# Analyze rubber content vs. strength
rubber_content = mix_df['rubber_fine_content']
slump = mix_df['slump']

plt.figure(figsize=(10, 6))
plt.scatter(rubber_content, slump, alpha=0.6)
plt.xlabel('Rubber Content (%)')
plt.ylabel('Slump (mm)')
plt.title('Rubber Content vs. Workability')
plt.show()
```

### R Analysis
```r
library(jsonlite)
library(ggplot2)
library(dplyr)

# Load the datasets
mix_data <- fromJSON("generated_datasets/mix_design_data.json")

# Convert to data frame
mix_df <- as.data.frame(mix_data$mix_designs)

# Create visualization
ggplot(mix_df, aes(x = rubber_fine_content, y = slump)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm") +
  labs(x = "Rubber Content (%)", 
       y = "Slump (mm)",
       title = "Rubber Content vs. Workability")
```

## Key Features for Research

### 🔬 **Material Characterization**
- **Complete XRF Analysis**: All major oxides and trace elements
- **Bogue Composition**: C₃S, C₂S, C₃A, C₄AF percentages
- **Thermal Properties**: Temperature-dependent conductivity, specific heat, expansion
- **Rubber Analysis**: TGA curves, FTIR spectra, SEM morphology

### 🏗️ **Mix Design Optimization**
- **Comprehensive Matrix**: 1000+ mix combinations
- **Rubber Integration**: Multiple size ranges and replacement levels
- **Fire Resistance**: Optimized for high-temperature performance
- **Workability Control**: Balanced fresh properties

### 🧪 **Testing Protocols**
- **Standard Compliance**: ASTM, ISO, EN standards
- **Temperature Range**: Ambient to 800°C
- **Multiple Conditions**: Various loading rates, heating rates, ages
- **Statistical Rigor**: 6 specimens per condition, proper COV

### 📊 **Data Quality**
- **Realistic Parameters**: Based on literature and industry standards
- **Statistical Validation**: Proper distributions and correlations
- **Completeness**: 100% data coverage
- **Consistency**: Cross-validated relationships

## Research Applications

### 1. **Model Development**
- **Thermo-Mechanical Models**: Temperature-dependent material properties
- **Finite Element Analysis**: Input parameters for FEA software
- **Machine Learning**: Training data for ML models
- **Optimization**: Multi-objective optimization algorithms

### 2. **Validation Studies**
- **Experimental Validation**: Compare with laboratory data
- **Literature Comparison**: Benchmark against published studies
- **Sensitivity Analysis**: Parameter importance assessment
- **Uncertainty Quantification**: Statistical analysis of results

### 3. **Design Applications**
- **Structural Design**: Fire-resistant structural elements
- **Mix Optimization**: Rubber content and size optimization
- **Performance Prediction**: Service life and durability
- **Code Development**: Building code recommendations

## Advanced Features

### 🔄 **Data Generation**
- **Synthetic Data**: Realistic parameters based on literature
- **Statistical Models**: Normal, log-normal, Weibull distributions
- **Correlation Matrices**: Material property relationships
- **Temperature Dependencies**: Polynomial and exponential functions

### 📈 **Analysis Tools**
- **Comprehensive Statistics**: Mean, std, COV, min, max
- **Correlation Analysis**: Property relationships
- **Visualization**: Matplotlib and Seaborn plots
- **Export Options**: JSON, CSV, Excel formats

### 🎯 **Customization**
- **Parameter Ranges**: Adjustable material property ranges
- **Test Conditions**: Customizable testing protocols
- **Mix Designs**: Flexible mix design matrix
- **Output Formats**: Multiple data export options

## Quality Assurance

### ✅ **Data Validation**
- **Range Checks**: Physical property bounds
- **Consistency Checks**: Property relationships
- **Outlier Detection**: Statistical methods
- **Cross-Validation**: Multiple testing methods

### 📋 **Standards Compliance**
- **ASTM Standards**: C39, C78, C496, C469, E119, E228
- **ISO Standards**: 834, 8990, 11357
- **EN Standards**: 12390-3, 12390-5, 12390-6

### 🔍 **Documentation**
- **Complete Metadata**: Test conditions, equipment, operators
- **Quality Control**: Calibration records, repeatability
- **Traceability**: Full data lineage and processing history

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required packages using `pip install -r requirements.txt`
2. **Memory Issues**: Reduce dataset size or use chunked processing
3. **File Not Found**: Ensure you're running from the correct directory
4. **Permission Errors**: Check file write permissions

### Support

For technical support or questions:
- Check the generated `README.md` in the output directory
- Review the `summary_report.json` for detailed statistics
- Examine the analysis plots in the `analysis/` directory

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{fire_resistant_rubberized_concrete_2024,
  title={Fire-Resistant Rubberized Concrete Dataset for Thermo-Mechanical Modeling},
  author={Dataset Generator},
  year={2024},
  url={https://github.com/your-repo/fire-resistant-rubberized-concrete-dataset}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on comprehensive literature review of rubberized concrete research
- Incorporates industry best practices for material testing
- Follows international standards for concrete testing
- Designed for fire-resistant structural applications

---

**Generated by Fire-Resistant Rubberized Concrete Dataset Generator**  
*For the development and validation of thermo-mechanical models for fire-resistant structural elements*