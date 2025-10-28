# Sintering Process & Microstructure Dataset

## Overview
This comprehensive dataset simulates experimental data for sintering process optimization and microstructure analysis. The dataset contains 50 samples with varying sintering parameters and their resulting microstructural characteristics.

## Dataset Files

### 1. `sintering_parameters.csv`
**Input Parameters for Sintering Optimization**

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| Sample_ID | Unique identifier for each sample | - | S001-S050 |
| Green_Density_g_cm3 | Initial density of green body | g/cm³ | 2.12-2.21 |
| Green_Porosity_pct | Initial porosity percentage | % | 12.8-16.3 |
| Initial_Grain_Size_um | Starting grain size | μm | 0.7-1.1 |
| Sintering_Temp_C | Maximum sintering temperature | °C | 1200-1400 |
| Hold_Time_min | Time at maximum temperature | min | 30-120 |
| Cooling_Rate_C_min | Cooling rate | °C/min | 0.5-5 |
| Applied_Pressure_MPa | Applied pressure during sintering | MPa | 0-25 |
| Atmosphere | Sintering atmosphere | - | Air, Nitrogen, Argon, Vacuum |
| Sintering_Time_total_min | Total sintering time | min | 45-150 |

### 2. `microstructure_outputs.csv`
**Resulting Microstructural Characteristics**

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| Sample_ID | Unique identifier | - | S001-S050 |
| Final_Density_g_cm3 | Final sintered density | g/cm³ | 3.41-3.85 |
| Relative_Density_pct | Relative density percentage | % | 89.2-99.9 |
| Total_Porosity_pct | Total porosity | % | 0.1-10.8 |
| Open_Porosity_pct | Open porosity | % | 0.1-8.5 |
| Closed_Porosity_pct | Closed porosity | % | 0.0-2.7 |
| Mean_Grain_Size_um | Average grain size | μm | 1.9-4.0 |
| Grain_Size_StdDev_um | Grain size standard deviation | μm | 0.7-1.7 |
| Min_Grain_Size_um | Minimum grain size | μm | 0.4-1.0 |
| Max_Grain_Size_um | Maximum grain size | μm | 3.8-8.0 |
| Grain_Size_D10_um | 10th percentile grain size | μm | 1.1-2.4 |
| Grain_Size_D50_um | 50th percentile grain size | μm | 1.7-3.6 |
| Grain_Size_D90_um | 90th percentile grain size | μm | 2.9-5.8 |
| Grain_Boundary_Density_per_um2 | Grain boundary density | /μm² | 0.06-0.18 |
| Intergranular_Porosity_pct | Porosity at grain boundaries | % | 0.1-6.5 |
| Transgranular_Porosity_pct | Porosity within grains | % | 0.0-4.6 |

### 3. `experimental_characterization.csv`
**Experimental Measurement Results**

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| Sample_ID | Unique identifier | - | S001-S050 |
| SEM_Grain_Count | Number of grains counted in SEM | count | 395-1456 |
| SEM_Mean_Grain_Size_um | SEM-measured grain size | μm | 1.95-4.15 |
| SEM_Grain_Size_StdDev_um | SEM grain size std dev | μm | 0.70-1.70 |
| SEM_Porosity_Area_pct | Porosity from SEM area analysis | % | 0.1-10.5 |
| SEM_Pore_Count | Number of pores counted | count | 1-89 |
| SEM_Mean_Pore_Size_um | Average pore size from SEM | μm | 0.1-1.4 |
| SEM_Pore_Size_StdDev_um | Pore size standard deviation | μm | 0.0-0.7 |
| Archimedes_Density_g_cm3 | Density from Archimedes method | g/cm³ | 3.41-3.85 |
| Archimedes_Open_Porosity_pct | Open porosity from Archimedes | % | 0.1-8.3 |
| Archimedes_Closed_Porosity_pct | Closed porosity from Archimedes | % | 0.0-2.6 |
| CT_Total_Porosity_pct | Total porosity from CT scan | % | 0.1-10.8 |
| CT_Mean_Pore_Size_um | Mean pore size from CT | μm | 0.25-1.25 |
| CT_Pore_Connectivity_pct | Pore connectivity percentage | % | 1.2-28.1 |
| CT_Tortuosity | Pore tortuosity factor | - | 0.6-2.1 |
| CT_Specific_Surface_Area_m2_g | Specific surface area | m²/g | 0.14-0.52 |

### 4. `grain_size_distribution.csv`
**Detailed Grain Size Distribution Data**

| Column | Description | Units |
|--------|-------------|-------|
| Sample_ID | Unique identifier | - |
| Size_Range_um | Grain size range | μm |
| Count_in_Range | Number of grains in range | count |
| Percentage_in_Range | Percentage of grains in range | % |
| Cumulative_Percentage | Cumulative percentage | % |
| Size_Range_Center_um | Center of size range | μm |

### 5. `pore_size_distribution.csv`
**Detailed Pore Size Distribution Data**

| Column | Description | Units |
|--------|-------------|-------|
| Sample_ID | Unique identifier | - |
| Pore_Size_Range_um | Pore size range | μm |
| Count_in_Range | Number of pores in range | count |
| Percentage_in_Range | Percentage of pores in range | % |
| Cumulative_Percentage | Cumulative percentage | % |
| Pore_Size_Center_um | Center of pore size range | μm |
| Pore_Type | Type of porosity | Open/Closed |

### 6. `temperature_profiles.csv`
**Sintering Temperature Profiles**

| Column | Description | Units |
|--------|-------------|-------|
| Sample_ID | Unique identifier | - |
| Time_min | Time point | min |
| Temperature_C | Temperature at time point | °C |
| Heating_Rate_C_min | Heating rate | °C/min |
| Cooling_Rate_C_min | Cooling rate | °C/min |
| Pressure_MPa | Applied pressure | MPa |
| Atmosphere | Sintering atmosphere | - |

## Key Relationships in the Data

### 1. Temperature Effects
- **Higher sintering temperatures** (1300-1400°C) → **Larger grain sizes** (3.2-4.0 μm)
- **Lower sintering temperatures** (1200-1250°C) → **Smaller grain sizes** (1.9-2.8 μm)
- **Longer hold times** → **Increased grain growth**

### 2. Pressure Effects
- **Applied pressure** (5-25 MPa) → **Higher final densities** (95.8-99.9%)
- **Pressure-assisted sintering** → **Reduced porosity** (0.1-4.7%)
- **Higher pressures** → **Faster densification**

### 3. Atmosphere Effects
- **Vacuum sintering** → **Highest densities** (96.6-99.9%)
- **Air atmosphere** → **Lower densities** (89.2-99.2%)
- **Inert atmospheres** (N₂, Ar) → **Intermediate results**

### 4. Microstructure Correlations
- **Grain size** ↔ **Porosity**: Larger grains often correlate with lower porosity
- **Density** ↔ **Mechanical properties**: Higher density → better properties
- **Pore connectivity** ↔ **Transport properties**: Higher connectivity → better permeability

## Usage for Optimization

### 1. Process Optimization
Use `sintering_parameters.csv` as input variables and `microstructure_outputs.csv` as target variables to:
- Predict final density from sintering conditions
- Optimize temperature profiles for desired grain size
- Minimize porosity while maintaining grain size control

### 2. Microstructure Modeling
Use the detailed distribution data (`grain_size_distribution.csv`, `pore_size_distribution.csv`) to:
- Calibrate grain growth models
- Validate pore evolution simulations
- Develop structure-property relationships

### 3. Experimental Validation
Compare simulation results with experimental data (`experimental_characterization.csv`) to:
- Validate model predictions
- Identify model limitations
- Guide further experimental work

## Data Quality Notes

- All data is synthetic but based on realistic sintering behavior
- Grain size distributions follow log-normal distributions
- Pore size distributions are typically bimodal
- Temperature profiles include realistic heating/cooling rates
- Experimental measurements include typical measurement uncertainties

## Recommended Analysis Workflow

1. **Exploratory Data Analysis**: Visualize relationships between inputs and outputs
2. **Correlation Analysis**: Identify key process parameters affecting microstructure
3. **Machine Learning**: Train models to predict microstructure from process parameters
4. **Optimization**: Use models to find optimal sintering conditions
5. **Validation**: Compare predictions with experimental data

This dataset provides a comprehensive foundation for sintering process optimization and microstructure modeling studies.