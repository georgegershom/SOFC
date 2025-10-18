# Fire-Resistant Rubberized Concrete Dataset
## Phase 1: Material Characterization & Specimen Preparation

### Overview
This comprehensive dataset supports research on **"Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete"**. The dataset provides detailed material characterization, mix design optimization, and fresh concrete properties for rubberized concrete incorporating crumb rubber from recycled tires.

### Dataset Structure

```
dataset/
├── 01_constituent_materials/     # Raw material characterization data
│   ├── cement_data.json         # OPC Type I characterization
│   ├── aggregates_data.json     # Coarse & fine aggregate properties
│   ├── crumb_rubber_data.json   # Tire-derived rubber characterization
│   └── water_admixtures_data.json # Water quality & chemical admixtures
├── 02_mix_designs/              # Mix proportion matrices
│   └── mix_design_matrix.json   # Complete mix design database
├── 03_fresh_properties/         # Fresh concrete test results
│   └── fresh_concrete_data.json # Workability & fresh state properties
├── 04_analysis_scripts/         # Data analysis & visualization tools
│   ├── data_analysis.py         # Comprehensive statistical analysis
│   ├── advanced_visualization.py # Interactive & publication plots
│   ├── requirements.txt         # Python dependencies
│   └── [generated outputs]     # Analysis results & visualizations
└── README.md                    # This documentation file
```

### Key Features

#### 🔬 **Constituent Materials**
- **Cement**: OPC Type I with complete XRF analysis, Bogue composition, and thermal properties
- **Aggregates**: Crushed granite (coarse) and river sand (fine) with full sieve analysis
- **Crumb Rubber**: Two size ranges (1-4mm, 4-8mm) from passenger car tires
- **Admixtures**: PCE superplasticizer, air-entraining agent, silica fume, PP fibers

#### 🧪 **Mix Design Matrix**
- **12 distinct mix designs** with rubber replacement levels: 0%, 5%, 10%, 15%, 20%, 25%, 30%
- **Variable parameters**: Rubber content, particle size, W/CM ratio
- **High-strength series** included for advanced applications
- **Complete material proportions** for each mix design

#### 📊 **Fresh Properties Database**
- **Slump/Flow measurements** (ASTM C143)
- **Air content** via pressure method (ASTM C231)
- **Unit weight** determination (ASTM C138)
- **Fresh temperature** monitoring (ASTM C1064)
- **Workability assessment** and segregation observations
- **Triplicate testing** with statistical analysis

### Technical Specifications

#### Material Characterization Methods
| Test | Standard | Equipment/Method |
|------|----------|------------------|
| Chemical Composition | XRF Analysis | X-ray Fluorescence Spectrometry |
| Particle Size Distribution | ASTM C136 | Mechanical Sieving |
| Specific Gravity | ASTM C127/C128 | Pycnometer Method |
| Thermal Analysis | TGA/DSC | Thermogravimetric Analysis |
| Morphology | SEM | Scanning Electron Microscopy |
| Chemical Identification | FTIR | Fourier Transform Infrared |

#### Mix Design Parameters
- **Target Strength**: 40 MPa (standard), 60 MPa (high-strength)
- **W/CM Ratio**: 0.35 - 0.45
- **Cement Content**: 380-420 kg/m³
- **Rubber Replacement**: By volume of fine aggregate
- **Admixture Optimization**: Based on workability requirements

#### Quality Control
- **Batch Consistency**: Triplicate testing for all properties
- **Statistical Analysis**: Mean, standard deviation, correlation analysis
- **Control Limits**: ±2σ quality control boundaries
- **Workability Rating**: Qualitative assessment scale

### Research Applications

#### 🔥 **Fire Resistance Studies**
- Thermal expansion coefficient data for all materials
- High-temperature decomposition characteristics
- Limiting oxygen index for rubber components
- Heat release rate measurements

#### 🏗️ **Structural Applications**
- Fresh property optimization for placement
- Workability assessment for different rubber contents
- Unit weight reduction quantification
- Air entrainment effects on durability

#### 📈 **Modeling & Simulation**
- Material property database for FEM modeling
- Statistical relationships for predictive models
- Thermal property inputs for fire simulation
- Fresh state behavior for construction modeling

### Data Analysis Tools

#### **data_analysis.py**
Comprehensive analysis script providing:
- Constituent material characterization summary
- Mix design optimization analysis
- Fresh property statistical evaluation
- Correlation analysis and regression modeling
- Quality control limit establishment
- Export to CSV for further analysis

#### **advanced_visualization.py**
Publication-ready visualization suite:
- Interactive Plotly dashboards
- Publication-quality matplotlib figures
- 3D property space visualization
- Animated property evolution plots
- Statistical correlation heatmaps

### Usage Instructions

#### 1. **Environment Setup**
```bash
pip install -r 04_analysis_scripts/requirements.txt
```

#### 2. **Run Complete Analysis**
```bash
cd 04_analysis_scripts/
python data_analysis.py
```

#### 3. **Generate Advanced Visualizations**
```bash
python advanced_visualization.py
```

#### 4. **Access Interactive Plots**
Open generated HTML files in web browser:
- `interactive_dashboard.html`
- `3d_visualization.html`
- `animated_plot.html`

### Key Findings & Trends

#### **Fresh Property Relationships**
| Property | Correlation with Rubber Content | R² Value |
|----------|--------------------------------|----------|
| Slump | Strong Positive (+) | 0.892 |
| Air Content | Strong Positive (+) | 0.945 |
| Unit Weight | Strong Negative (-) | 0.987 |
| Workability | Decreases at >20% rubber | - |

#### **Optimal Mix Ranges**
- **Workability**: Best at 5-15% rubber replacement
- **Unit Weight Reduction**: 3-8% decrease with rubber addition
- **Air Entrainment**: Increases 0.12% per 1% rubber content
- **Segregation Threshold**: >20% rubber shows floating tendency

#### **Material Efficiency**
- **Superplasticizer Demand**: Increases 0.15 kg/m³ per 1% rubber
- **Mixing Time**: Extended by 0.5-1.0 minutes for rubber mixes
- **Fresh Temperature**: Minimal impact (<1°C variation)

### Data Quality & Validation

#### **Measurement Precision**
- **Slump**: ±5 mm repeatability
- **Air Content**: ±0.2% precision
- **Unit Weight**: ±10 kg/m³ accuracy
- **Temperature**: ±0.5°C precision

#### **Statistical Validation**
- **Sample Size**: Minimum 3 batches per mix design
- **Outlier Detection**: Grubbs test at 95% confidence
- **Normality Testing**: Shapiro-Wilk test for distributions
- **Correlation Significance**: p-value < 0.05 threshold

### Future Extensions

#### **Phase 2 Integration**
This dataset serves as the foundation for:
- Hardened concrete property testing
- Mechanical strength development
- Thermal property measurement
- Fire resistance testing
- Durability assessment

#### **Model Development**
Data supports development of:
- Thermo-mechanical constitutive models
- Fire resistance prediction algorithms
- Mix design optimization tools
- Quality control systems

### Citation & Usage

When using this dataset, please cite:
```
Fire-Resistant Rubberized Concrete Dataset - Phase 1: Material Characterization & Specimen Preparation
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements 
Utilizing High-Performance Rubberized Concrete
[Year] - [Institution/Research Group]
```

### Contact & Support

For questions, clarifications, or collaboration opportunities:
- **Technical Issues**: Check analysis scripts and documentation
- **Data Interpretation**: Refer to statistical analysis outputs
- **Research Collaboration**: Contact research team

### License & Terms

This dataset is provided for research and educational purposes. Commercial use requires permission. Please acknowledge the source in any publications or presentations using this data.

---

**Dataset Version**: 1.0  
**Last Updated**: 2024  
**Total Data Points**: 12 mix designs × 3 batches × 4 properties = 144 measurements  
**File Size**: ~2.5 MB (JSON format)  
**Quality Level**: Research Grade - Peer Review Ready