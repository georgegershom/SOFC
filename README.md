# Synthetic Dataset for Fire-Resistant Structural Elements Using High-Performance Rubberized Concrete

## Overview

This repository contains a comprehensive synthetic dataset generated for the research project: **"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete."**

The dataset is scientifically plausible, internally consistent, and formatted for immediate use in analysis, visualization, and as input for subsequent phases of the research.

## Dataset Structure

### Core Files

1. **`constituent_materials_cement.csv`** - Cement characterization data (OPC 52.5N)
2. **`constituent_materials_aggregates.csv`** - Coarse and fine aggregate properties
3. **`constituent_materials_crumb_rubber.csv`** - Crumb rubber properties (two particle sizes)
4. **`constituent_materials_other.csv`** - Water and superplasticizer properties
5. **`mix_proportions_fresh_properties.csv`** - Mix designs and fresh concrete properties
6. **`mechanical_properties.csv`** - Mechanical properties at various ages (1-90 days)
7. **`thermal_properties.csv`** - Thermal properties at various temperatures (20-600°C)
8. **`specimen_preparation_testing.csv`** - Specimen preparation and testing conditions
9. **`data_validation_qa.csv`** - Data validation and quality assurance metrics
10. **`complete_dataset.json`** - Complete dataset in JSON format

### Analysis Tools

- **`data_analysis_script.py`** - Python script for comprehensive data analysis and visualization
- **`synthetic_dataset_analysis.png`** - Generated visualization plots
- **`analysis_results.json`** - Summary statistics and analysis results

## Mix Designations

| Mix ID | Description | Rubber Content | Rubber Size |
|--------|-------------|----------------|-------------|
| C | Control | 0% | - |
| R5S | 5% Rubber Small | 5% | 1-4 mm |
| R10S | 10% Rubber Small | 10% | 1-4 mm |
| R15S | 15% Rubber Small | 15% | 1-4 mm |
| R20S | 20% Rubber Small | 20% | 1-4 mm |
| R10L | 10% Rubber Large | 10% | 4-8 mm |
| R15L | 15% Rubber Large | 15% | 4-8 mm |
| R20L | 20% Rubber Large | 20% | 4-8 mm |
| R5M | 5% Rubber Mixed | 5% | Mixed sizes |
| R10M | 10% Rubber Mixed | 10% | Mixed sizes |
| R15M | 15% Rubber Mixed | 15% | Mixed sizes |
| R20M | 20% Rubber Mixed | 20% | Mixed sizes |

## Key Features

### Scientific Plausibility
- All data points fall within scientifically accepted ranges for concrete technology
- Relationships between variables are logical (e.g., increasing rubber content decreases density and strength)
- Chemical compositions align with declared cement type

### Data Consistency
- Mix proportions sum to 100% for all mixes
- Water-cement ratio maintained at 0.40 across all mixes
- Internal consistency maintained across all parameters

### Controlled Variation
- Random noise (±5-10%) applied to simulated measured values
- Realistic experimental error simulation
- Statistical validity with sufficient replicates

### Completeness
- Exhaustive parameter coverage for quality assurance
- Full traceability with specimen IDs and operator tracking
- Complete testing conditions and environmental data

## Usage Instructions

### 1. Data Analysis
```bash
python data_analysis_script.py
```

This will generate:
- Summary statistics
- Rubber content effect analysis
- Comprehensive visualizations
- Analysis results in JSON format

### 2. Data Import
```python
import pandas as pd

# Load individual datasets
mix_proportions = pd.read_csv('mix_proportions_fresh_properties.csv')
mechanical_props = pd.read_csv('mechanical_properties.csv')
thermal_props = pd.read_csv('thermal_properties.csv')

# Or load complete JSON dataset
import json
with open('complete_dataset.json', 'r') as f:
    complete_data = json.load(f)
```

### 3. Visualization
The analysis script generates six key visualizations:
1. Compressive strength development over time
2. Fresh density vs rubber content
3. Thermal conductivity vs temperature
4. Fire resistance rating vs rubber content
5. Modulus of elasticity vs rubber content
6. Workability (slump) vs rubber content

## Data Quality Assurance

All data has been validated for:
- ✅ Mix proportion consistency (sums to 100%)
- ✅ Water-cement ratio consistency (0.40)
- ✅ Density relationships (decreases with rubber content)
- ✅ Strength relationships (decreases with rubber content)
- ✅ Thermal property relationships (improved insulation with rubber)
- ✅ Age progression (strength increases with age)
- ✅ Temperature effects (realistic thermal behavior)
- ✅ Specimen dimensions (standard 150×300mm cylinders)
- ✅ Curing conditions (consistent 20°C, 95% RH)
- ✅ Testing standards compliance (ASTM/EN standards)
- ✅ Data completeness (all required parameters)
- ✅ Statistical validity (sufficient replicates)
- ✅ Traceability (full specimen tracking)
- ✅ Measurement uncertainty (realistic ±5-10% variation)

## Research Applications

This dataset is designed for:
- **Phase 1**: Material characterization and specimen preparation
- **Phase 2**: Thermo-mechanical modeling development
- **Phase 3**: Model validation and optimization
- **Phase 4**: Fire resistance performance evaluation

## Technical Specifications

- **Total Mixes**: 12 (1 control + 11 rubberized)
- **Total Specimens**: 72 (6 per mix)
- **Testing Ages**: 1, 3, 7, 28, 56, 90 days
- **Temperature Range**: 20-600°C
- **Rubber Content Range**: 0-20%
- **Particle Sizes**: 1-4 mm, 4-8 mm, Mixed
- **Standards**: ASTM C127, C128, C136, C188, C29, D792, EN 197-1

## File Formats

- **CSV**: Individual datasets for easy import into analysis software
- **JSON**: Complete dataset for programmatic access
- **PNG**: Visualization plots for presentations and reports

## Contact Information

For questions about this dataset or the research project, please refer to the research documentation or contact the research team.

---

**Note**: This is a synthetic dataset generated for research purposes. All data has been fabricated to be scientifically plausible and internally consistent, but should not be used as actual experimental data without proper validation.