# Synthetic Dataset: High-Performance Rubberized Concrete for Fire-Resistant Structural Elements

## Project Overview
This dataset was generated for the research project: "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete."

## Dataset Description
This is a comprehensive synthetic dataset designed for the initial phase of material characterization and specimen preparation. The data is scientifically plausible, internally consistent, and formatted for immediate use in analysis, visualization, and as input for subsequent research phases.

## Key Features
- **Realism & Plausibility**: All data points fall within scientifically accepted ranges for concrete technology
- **Consistency**: Data for each mix ID is self-consistent with proper material balance
- **Controlled Variation**: Random noise (±5-10%) added to simulate experimental error
- **Completeness**: Exhaustive dataset including all necessary parameters for quality assurance and analysis
- **Machine-Readable**: Structured CSV and JSON formats with clear, descriptive headers

## Mix Designs
The dataset includes 6 different mix designs:

| Mix ID | Description | Rubber Content | Rubber Size | Batches |
|--------|-------------|----------------|-------------|---------|
| C | Control (No rubber) | 0% | - | 3 |
| R5S | 5% Rubber, Small particles | 5% | 1-4mm | 3 |
| R10S | 10% Rubber, Small particles | 10% | 1-4mm | 3 |
| R15S | 15% Rubber, Small particles | 15% | 1-4mm | 3 |
| R20S | 20% Rubber, Small particles | 20% | 1-4mm | 3 |
| R10L | 10% Rubber, Large particles | 10% | 4-8mm | 3 |

## File Structure

### Constituent Materials
- `constituent_materials_cement.csv` - Cement characterization (OPC 52.5N)
- `constituent_materials_aggregates.csv` - Coarse and fine aggregate properties
- `constituent_materials_crumb_rubber.csv` - Crumb rubber properties for two particle sizes
- `constituent_materials_other.csv` - Water and superplasticizer properties

### Core Datasets
- `mix_proportions_fresh_properties.csv` - Mix proportions and fresh concrete properties
- `mechanical_properties.csv` - Mechanical properties at 7, 28, and 90 days
- `thermal_properties.csv` - Thermal properties and fire resistance data
- `specimen_preparation_testing.csv` - Specimen preparation and testing conditions

### Documentation
- `dataset_summary.json` - Comprehensive dataset metadata and structure
- `README.md` - This documentation file

## Testing Ages
- 7 days
- 28 days
- 90 days

## Specimen Types
- Cylinders (150×300 mm) - Compressive strength testing
- Beams (100×100×400 mm) - Flexural strength testing
- Prisms (100×100×100 mm) - Thermal properties testing

## Properties Measured

### Fresh Properties
- Slump (mm)
- Air Content (%)
- Fresh Density (kg/m³)
- Workability Class

### Mechanical Properties
- Compressive Strength (MPa)
- Flexural Strength (MPa)
- Split Tensile Strength (MPa)
- Static Modulus (GPa)
- Poisson Ratio
- Hardened Density (kg/m³)
- Water Absorption (%)

### Thermal Properties
- Thermal Conductivity (W/m·K)
- Specific Heat (J/kg·K)
- Thermal Diffusivity (m²/s)
- Linear Thermal Expansion (1/K)
- Fire Resistance Rating (minutes)
- Residual Strength after Elevated Temperatures (%)

## Data Quality Assurance
- All mix proportions sum to 100%
- Chemical composition aligns with declared cement type
- Relationships between variables are logical (e.g., increasing rubber content decreases density and strength)
- Standard deviations and coefficients of variation are realistic
- Full traceability through batch and specimen IDs

## Usage Notes
- All measurements include standard deviations and coefficients of variation
- Testing conditions (temperature, humidity) are recorded for each test
- Equipment and operator information is included for traceability
- Data is ready for statistical analysis, machine learning, and visualization

## Standards Referenced
- ASTM C39 - Compressive Strength
- ASTM C127/C128 - Aggregate Properties
- ASTM C188 - Specific Gravity
- ASTM C136 - Fineness Modulus
- ASTM C29 - Bulk Density
- ASTM D792 - Rubber Specific Gravity
- ASTM E119 - Fire Resistance Testing

## Contact
For questions about this dataset or the research project, please refer to the project documentation or contact the research team.