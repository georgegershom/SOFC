# Material Properties & Calibration Dataset for Multi-Physics Models

## Overview
This comprehensive dataset provides fundamental material properties essential for building and calibrating both traditional and AI-enhanced multi-physics models. The synthetic data is based on realistic values for nickel-based superalloys (similar to Inconel 718) and covers a wide temperature range suitable for high-temperature applications.

## Dataset Contents

### 1. Mechanical Properties (`mechanical/`)
Temperature-dependent mechanical properties from 20°C to 1000°C:
- **Young's Modulus (GPa)**: Elastic stiffness decreasing with temperature
- **Tensile Strength (MPa)**: Ultimate strength showing rapid decline above 600°C
- **Yield Strength (MPa)**: Approximately 85% of tensile strength
- **Poisson's Ratio**: Slight temperature dependence (0.29-0.31)
- **Elongation at Break (%)**: Ductility increasing with temperature

**Key Files:**
- `mechanical_properties.csv`: Main dataset (50 temperature points)
- `mechanical_properties_metadata.json`: Test conditions and material specifications
- `mechanical_properties_plot.png`: Visualization of all properties

### 2. Creep Properties (`creep/`)
Comprehensive creep strain vs. time curves at various conditions:
- **Temperature Range**: 600-1000°C (8 levels)
- **Stress Range**: 100-400 MPa (7 levels)
- **Time Duration**: 0.01 to 3,162 hours (log scale)
- **Test Conditions**: 56 unique temperature-stress combinations
- **Data Points**: Over 5,600 measurements

**Key Files:**
- `creep_curves_full.csv`: Complete creep data with strain evolution
- `creep_summary.csv`: Summary statistics for each test condition
- `creep_curves_plot.png`: Selected creep curves visualization
- `creep_rate_map.png`: Minimum creep rate contour map

### 3. Thermo-Physical Properties (`thermophysical/`)
Temperature-dependent thermal properties from 20°C to 1200°C:
- **Coefficient of Thermal Expansion (CTE)**: 11.6-17.1 ppm/K
- **Thermal Conductivity**: 11.3-28.8 W/m·K
- **Specific Heat Capacity**: 420-586 J/kg·K
- **Thermal Diffusivity**: Calculated from k/(ρ·Cp)
- **Linear Thermal Expansion**: Cumulative expansion percentage

**Key Files:**
- `thermophysical_properties.csv`: Main dataset (119 temperature points)
- `thermophysical_validation.csv`: Validation measurements at key temperatures
- `thermophysical_properties_plot.png`: All properties visualization

### 4. Electrochemical Properties (`electrochemical/`)

#### Conductivity Data
Temperature and atmosphere-dependent transport properties:
- **Electronic Conductivity**: 8.6 to 5,170 S/cm
- **Ionic Conductivity**: 1e-12 to 4.33 S/cm
- **Mixed Conductivity**: Geometric mean of electronic and ionic
- **Transference Numbers**: Relative contribution of each transport mode
- **Oxygen Partial Pressures**: 8 environments from 1e-20 to 1.0 atm
- **Temperature Range**: 200-1200°C

#### Corrosion Data
Electrochemical corrosion in various electrolytes:
- **Electrolytes**: 3.5% NaCl, 0.1M H₂SO₄, 0.1M NaOH, Simulated Seawater
- **Temperature Range**: 25-80°C
- **Measurements**: Corrosion potential, current density, Tafel slopes
- **Corrosion Rates**: 0.0000-0.0061 mm/year

**Key Files:**
- `electrochemical_conductivity.csv`: Transport properties dataset
- `electrochemical_corrosion.csv`: Corrosion test results
- `conductivity_arrhenius.png`: Arrhenius plots for conductivity
- `corrosion_properties.png`: Corrosion behavior visualizations
- `transference_numbers.png`: Ionic vs electronic transport

## Analysis Results (`docs/`)

### Correlation Analysis
- `correlation_matrix.png`: Cross-correlation between mechanical and thermophysical properties
- `correlation_data.csv`: Merged dataset for correlation studies

### Empirical Models
- `empirical_models.json`: Fitted mathematical models for property prediction
  - Young's Modulus: 3rd-order polynomial
  - Tensile Strength: Piecewise (linear/exponential)
  - CTE: Logarithmic fit

### Summary Report
- `summary_report.json`: Comprehensive statistics for all datasets
- `master_visualization.png`: Combined overview of all properties

## Data Generation & Usage

### Installation
```bash
pip3 install -r requirements.txt
```

### Generate All Datasets
```bash
cd scripts
python3 generate_all_datasets.py
```

### Run Analysis
```bash
python3 analyze_datasets.py
```

### Individual Dataset Generation
```bash
python3 generate_mechanical_properties.py
python3 generate_creep_properties.py
python3 generate_thermophysical_properties.py
python3 generate_electrochemical_properties.py
```

## Material Model Calibration

### Applicable Models
The dataset supports calibration of various constitutive models:

1. **Mechanical Models**
   - Ramberg-Osgood (nonlinear elasticity)
   - Johnson-Cook (temperature-dependent plasticity)
   - Power-law creep models

2. **Creep Models**
   - Norton-Bailey
   - Theta Projection
   - Wilshire equations
   - Larson-Miller Parameter

3. **Thermal Models**
   - Fourier heat conduction
   - Thermal stress analysis
   - Coupled thermo-mechanical models

4. **Multi-Physics Models**
   - Thermo-mechanical-chemical coupling
   - Oxidation-induced property degradation
   - Stress-assisted diffusion

## Data Format & Structure

### CSV Format
All CSV files use standard comma-separated format with headers:
- Numerical data with appropriate precision
- Consistent units across datasets
- Missing values handled as NULL or specified defaults

### JSON Metadata
Each dataset includes JSON metadata with:
- Material specifications
- Test conditions and standards
- Measurement methods
- Uncertainties and validation information

## Physical Units

| Property | Unit |
|----------|------|
| Temperature | °C (Celsius) |
| Stress/Strength | MPa |
| Young's Modulus | GPa |
| Strain | % (percent) |
| Time | hours |
| CTE | ppm/K |
| Thermal Conductivity | W/m·K |
| Specific Heat | J/kg·K |
| Conductivity | S/cm |
| Corrosion Rate | mm/year |

## Applications

This dataset is suitable for:
- **Multi-physics finite element analysis (FEA)**
- **Machine learning model training for property prediction**
- **Constitutive model parameter identification**
- **Life prediction and reliability analysis**
- **Digital twin development for high-temperature components**
- **Optimization of thermal barrier coatings**
- **SOFC/SOEC component modeling**

## Important Notes

1. **Synthetic Data**: This is fabricated/synthetic data based on literature values for demonstration and testing purposes
2. **Material Class**: Properties representative of nickel-based superalloys
3. **Temperature Range**: Covers service conditions up to 1200°C
4. **Scatter**: Includes realistic experimental scatter (1-5% depending on property)
5. **Validation**: Includes separate validation measurements for model verification

## Quality Assurance

- **Physical Bounds**: All properties constrained to physically realistic ranges
- **Consistency Checks**: Cross-property relationships maintained
- **Temperature Continuity**: Smooth property variations with temperature
- **Units Verification**: Consistent SI units throughout

## Citation

If using this dataset for research or development:
```
Material Properties Dataset for Multi-Physics Models
Generated: [Current Date]
Type: Synthetic/Fabricated Data
Base Material: Nickel-based Superalloy (Inconel 718-like)
```

## Future Extensions

Potential additions to enhance the dataset:
- Fatigue properties (S-N curves, crack growth)
- Fracture toughness vs temperature
- Microstructural evolution data
- In-situ measurement simulations
- Uncertainty quantification bands

## Contact & Support

For questions about the dataset structure or generation scripts, please refer to the individual Python scripts in the `scripts/` directory, which contain detailed documentation and comments.

---

*Dataset generated using physics-based models and literature correlations to provide realistic training data for multi-physics model development.*