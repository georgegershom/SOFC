# Phase 1: Material Characterization & Specimen Preparation

## Fire-Resistant Rubberized Concrete Project

### Project Overview
This repository contains comprehensive material characterization data and analysis for the development of fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset includes detailed properties of all constituent materials, mix design matrices, and fresh concrete properties.

### Directory Structure
```
phase1_material_characterization/
│
├── data/                     # Raw JSON data files
│   ├── cement_properties.json
│   ├── aggregates_properties.json
│   ├── crumb_rubber_properties.json
│   ├── water_and_admixtures.json
│   ├── mix_designs_matrix.json
│   └── fresh_concrete_properties.json
│
├── scripts/                  # Analysis and visualization scripts
│   ├── visualize_data.py
│   └── generate_report.py
│
├── visualizations/          # Generated plots and figures
│
├── reports/                 # Generated PDF reports
│
└── requirements.txt         # Python dependencies
```

### Dataset Contents

#### 1. Cement Properties
- Type I Portland Cement (OPC)
- Chemical composition (XRF analysis)
- Bogue composition
- Mineralogical phases (XRD)
- Physical and mechanical properties
- Thermal properties

#### 2. Aggregates Characterization
- **Coarse Aggregate**: Crushed granite, 19mm nominal
- **Fine Aggregate**: Natural river sand, Zone II
- Complete sieve analysis
- Physical properties (specific gravity, absorption)
- Chemical properties
- Thermal expansion coefficients

#### 3. Crumb Rubber Analysis
- Source: End-of-life tires (ELT)
- Two size ranges: Fine (1-4mm) and Coarse (4-8mm)
- Particle size distribution
- Chemical composition (TGA, FTIR)
- Surface properties (BET, contact angle)
- Morphology (SEM analysis)
- Environmental leaching tests

#### 4. Water and Admixtures
- Potable water analysis
- Superplasticizer (PCE-based)
- Air entraining agent
- Set retarder
- Silica fume properties

#### 5. Mix Design Matrix
- 12 unique mix designs
- Rubber replacement: 0%, 5%, 10%, 15%, 20%
- Fine and coarse rubber variations
- Silica fume enhanced mixes
- Target strength: 40 MPa at 28 days

#### 6. Fresh Concrete Properties
- Slump and slump retention
- Air content
- Unit weight
- Setting time
- Bleeding characteristics
- Rheological properties

### Installation and Usage

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation
```bash
# Clone or navigate to the project directory
cd phase1_material_characterization

# Install required packages
pip install -r requirements.txt
```

#### Running the Analysis

1. **Generate Visualizations**:
```bash
cd scripts
python visualize_data.py
```
This will create various plots in the `visualizations/` directory:
- Cement characterization plots
- Aggregate gradation curves
- Rubber property analysis
- Mix design comparisons
- Fresh property trends

2. **Generate Comprehensive Report**:
```bash
python generate_report.py
```
This will create a detailed PDF report in the `reports/` directory.

### Key Findings

#### Material Performance
- Cement shows typical Type I characteristics with 58.2% C3S
- Aggregates meet ASTM gradation requirements
- Crumb rubber exhibits hydrophobic nature (118° contact angle)
- Rubber particles show rough, porous surface morphology

#### Mix Design Trends
- Workability decreases with rubber content (180mm → 115mm slump)
- Air content increases from 2.1% to 5.2% with rubber addition
- Unit weight reduces by 7.7% at 20% rubber replacement
- Superplasticizer demand increases exponentially with rubber

#### Optimal Range
- 10-15% rubber replacement shows best balance of properties
- Fine rubber (1-4mm) provides better workability
- Silica fume addition enhances performance

### Data Format
All data files are in JSON format for easy parsing and analysis. Each file contains:
- Metadata (testing dates, standards, batch IDs)
- Numerical data with units
- Quality compliance information
- Testing conditions and methods

### Testing Standards
All tests conducted according to:
- ASTM C150 (Cement)
- ASTM C136 (Sieve Analysis)
- ASTM C143 (Slump)
- ASTM C231 (Air Content)
- ASTM C138 (Unit Weight)
- ASTM C403 (Setting Time)
- ASTM D5603 (Rubber Properties)

### Applications
This dataset is suitable for:
- Concrete mix optimization
- Fire resistance modeling
- Sustainability assessments
- Machine learning predictions
- Thermal-mechanical simulations
- Durability studies

### Citation
If you use this dataset, please cite:
```
Development and Validation of a Thermo-Mechanical Model for 
Fire-Resistant Structural Elements Utilizing High-Performance 
Rubberized Concrete - Phase 1: Material Characterization
Advanced Concrete Technology Lab, 2024
```

### Contact
For questions or additional data requests, contact:
- Principal Investigator: Dr. Sarah Chen
- Laboratory: Advanced Concrete Technology Lab

### License
This dataset is provided for research and educational purposes.

---
*Generated: October 2024*