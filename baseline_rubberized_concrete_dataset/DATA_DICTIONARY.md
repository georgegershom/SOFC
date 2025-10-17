# Data Dictionary
## Baseline Rubberized Concrete Dataset

This document provides detailed descriptions of all variables, units, abbreviations, and data formats used in the dataset.

---

## General Conventions

### Mix Identification System
- **Mix_ID:** Alphanumeric code (e.g., RC-00, RC-05, RC-10, RC-15)
  - RC = Rubberized Concrete
  - Number = Rubber replacement percentage (00, 05, 10, 15)
- **Mix_Name:** Descriptive name including rubber content
- **Rubber_Content_Percent:** Rubber replacement level (0, 5, 10, 15) by volume of fine aggregate

### Specimen Identification System
- **Specimen_ID Format:** `MixID-Age-TestType-Number`
  - Example: `RC05-28D-C3` = RC-05 mix, 28 days, Compressive test, specimen #3
  - Age codes: 7D (7 days), 28D (28 days)
  - Test type codes: C (Compressive), T (Tensile), E (Elastic modulus), D (Density), U (UPV)

### Units and Abbreviations

#### Length/Distance
- `mm` = millimeters
- `μm` = micrometers (microns)
- `nm` = nanometers
- `m` = meters
- `km` = kilometers

#### Mass/Density
- `kg` = kilograms
- `g` = grams
- `mg` = milligrams
- `kg/m³` or `kg_m3` = kilograms per cubic meter
- `kg/m²` or `kg_m2` = kilograms per square meter

#### Stress/Strength
- `MPa` = Megapascals (N/mm²)
- `GPa` = Gigapascals (1000 MPa)
- `Pa` = Pascals

#### Temperature
- `C` or `°C` = Degrees Celsius
- `K` = Kelvin

#### Time
- `s` or `sec` = seconds
- `min` = minutes
- `h` = hours

#### Energy/Thermal
- `W/(m·K)` or `W_mK` = Watts per meter-Kelvin (thermal conductivity)
- `J/(kg·K)` or `J_kgK` = Joules per kilogram-Kelvin (specific heat capacity)
- `mm²/s` = square millimeters per second (thermal diffusivity)
- `mW` = milliwatts

#### Permeability
- `m/s` = meters per second (coefficient of permeability)
- `m²/s` = square meters per second (diffusion coefficient)
- `m²` = square meters (intrinsic permeability)

#### Other
- `percent` or `%` = percentage
- `ml/g` or `ml_g` = milliliters per gram
- `cm⁻¹` or `cm_minus1` = reciprocal centimeters (wavenumber for spectroscopy)
- `microstrain` or `με` = microstrain (10⁻⁶ strain)
- `per_K` = per Kelvin (thermal expansion coefficient)

---

## File-Specific Variable Definitions

### 1_mixture_proportions.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture identification code | - | RC-00, RC-05, RC-10, RC-15 |
| Mix_Name | Descriptive mixture name | - | Text description |
| Rubber_Replacement_Level_Percent | Rubber replacement percentage | % | By volume of fine aggregate |
| Rubber_Replacement_Type | Type of replacement | - | "Volume replacement of fine aggregate" |
| Cement_Type | Type of cement used | - | Portland Cement |
| Cement_Grade | Cement strength grade | - | Type I (52.5N) per ASTM C150 |
| Cement_kg_m3 | Cement content | kg/m³ | Constant at 420 across all mixes |
| Cement_Source | Supplier/origin | - | LafargeHolcim Portland Plant |
| Coarse_Aggregate_Type | Type of coarse aggregate | - | Crushed Granite |
| Coarse_Aggregate_kg_m3 | Coarse aggregate content | kg/m³ | Constant at 1050 |
| Coarse_Aggregate_Max_Size_mm | Maximum aggregate size | mm | 20mm nominal maximum size |
| Coarse_Aggregate_Specific_Gravity_SSD | Specific gravity (saturated-surface-dry) | - | Dimensionless ratio |
| Coarse_Aggregate_Water_Absorption_Percent | Water absorption capacity | % | After 24h immersion |
| Coarse_Aggregate_Source | Supplier/origin | - | Regional Quarry Site A |
| Fine_Aggregate_Type | Type of fine aggregate | - | Natural River Sand |
| Fine_Aggregate_kg_m3 | Fine aggregate content | kg/m³ | Decreases with rubber addition |
| Fine_Aggregate_Specific_Gravity_SSD | Specific gravity (SSD) | - | Dimensionless |
| Fine_Aggregate_Water_Absorption_Percent | Water absorption | % | After 24h |
| Fine_Aggregate_Fineness_Modulus | Fineness modulus (grading) | - | Per ASTM C136 |
| Fine_Aggregate_Source | Supplier/origin | - | Regional River Site B |
| Rubber_Aggregate_kg_m3 | Rubber aggregate content | kg/m³ | Increases with replacement level |
| Rubber_Aggregate_Volume_m3 | Rubber volume per m³ concrete | m³ | Calculated from density |
| Water_kg_m3 | Water content | kg/m³ | Constant at 189 (w/c = 0.45) |
| Water_Cement_Ratio | Water to cement mass ratio | - | 0.45 for all mixes |
| Superplasticizer_Type | Type of admixture | - | Polycarboxylate Ether (PCE) |
| Superplasticizer_Percent_by_Cement_Weight | SP dosage | % | Increases with rubber to maintain workability |
| Superplasticizer_kg_m3 | SP content | kg/m³ | Calculated from cement × percentage |
| Total_Aggregate_Volume_Ratio | Volume ratio of aggregates | - | 0.700 (70% of concrete volume) |
| Curing_Method | Curing procedure | - | Lime-saturated water immersion |
| Curing_Temperature_C | Curing temperature | °C | 23°C constant |
| Curing_Duration_Days | Curing period | days | 28 days standard |
| Curing_Humidity_Percent | Relative humidity | % | 100% (fully immersed) |

### 2_aggregate_grading.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Sieve_Size_mm | Sieve opening size | mm | Standard sieve series |
| Coarse_Aggregate_Cumulative_Passing_Percent | Cumulative passing | % | Percentage passing this sieve |
| Fine_Aggregate_Cumulative_Passing_Percent | Cumulative passing | % | Well-graded natural sand |
| Rubber_Aggregate_Cumulative_Passing_Percent | Cumulative passing | % | Narrow grading (1-4mm) |
| ASTM_C33_Coarse_Min | ASTM C33 lower limit | % | For coarse aggregate |
| ASTM_C33_Coarse_Max | ASTM C33 upper limit | % | For coarse aggregate |
| ASTM_C33_Fine_Min | ASTM C33 lower limit | % | For fine aggregate |
| ASTM_C33_Fine_Max | ASTM C33 upper limit | % | For fine aggregate |

### 3_rubber_characterization.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Property | Name of property measured | - | Descriptive text |
| Value | Measured value | Variable | See Unit column |
| Unit | Unit of measurement | - | Various units |
| Test_Method | Standard or method used | - | ASTM designation or description |
| Notes | Additional information | - | Context and interpretation |

**Key Properties Listed:**
- Source, Processing_Method, Particle_Size_Range
- Specific_Gravity, Bulk_Density (loose and compacted)
- Water_Absorption (24h and 30min)
- Hardness_Shore_A, Tensile_Strength_Rubber, Elongation_at_Break
- Surface characteristics, Treatment details
- Moisture_Content_Delivered, Surface_Area_BET, Oil_Content

### 4_rubber_chemical_composition.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Component | Chemical component name | - | Main constituents |
| Content_Percent | Mass percentage | % | By weight |
| Method | Analytical method | - | TGA, XRF, FTIR, GC-MS, etc. |
| Description | Function/significance | - | Role in rubber compound |

**Components Include:**
- Polymers (Natural_Rubber_SBR_Blend)
- Fillers (Carbon_Black)
- Vulcanization agents (Zinc_Oxide, Sulfur_Compounds)
- Additives (Processing_Oils, Antioxidants, Accelerators)
- Contaminants (Textile_Fiber_Residue, Steel_Fiber_Residue)

### 5_rubber_thermal_analysis.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Temperature_C | Temperature | °C | Heating rate: 10°C/min |
| Weight_Loss_Percent | Cumulative mass loss | % | From initial mass |
| DTG_Peak_Percent_per_C | Derivative mass loss rate | %/°C | DTG = derivative thermogravimetry |
| Heat_Flow_mW | Heat flow (DSC) | mW | Negative = endothermic |
| Phase_Transition | Description of transition | - | Thermal event occurring |
| Degradation_Product | Main products released | - | Identified by TGA-MS |
| Significance_for_Fire_Resistance | Interpretation | - | Relevance to fire behavior |

### 6_rubber_ftir_peaks.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Wavenumber_cm_minus1 | Infrared wavenumber | cm⁻¹ | Position of absorption band |
| Peak_Intensity | Relative intensity | - | weak, medium, strong |
| Band_Assignment | Vibrational mode | - | Type of molecular vibration |
| Functional_Group | Chemical group | - | Molecular structure responsible |
| Significance | Interpretation | - | What this tells us about rubber |

### 7_fresh_state_properties.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | RC-00, RC-05, RC-10, RC-15 |
| Slump_Flow_mm | Slump flow diameter | mm | ASTM C143 |
| Slump_Flow_T50_sec | Time to 500mm spread | s | Flow table test |
| Slump_Retention_30min_mm | Slump after 30 min | mm | Workability retention |
| Slump_Retention_60min_mm | Slump after 60 min | mm | Extended retention |
| Air_Content_Percent | Entrapped air | % | ASTM C231 pressure method |
| Air_Void_Spacing_Factor_mm | Air void spacing | mm | ASTM C457 (hardened air void analysis) |
| Fresh_Density_kg_m3 | Density of fresh concrete | kg/m³ | Before setting |
| Temperature_at_Mixing_C | Concrete temperature | °C | During mixing |
| Mixing_Time_min | Duration of mixing | min | Drum mixer |
| Setting_Time_Initial_min | Initial set | min | ASTM C403 (penetration resistance) |
| Setting_Time_Final_min | Final set | min | Complete hardening onset |
| Bleeding_ml_per_cm2 | Bleed water volume | ml/cm² | ASTM C232 |
| Segregation_Index | Segregation resistance | - | 0 = no segregation |
| Workability_Rating | Qualitative assessment | - | Excellent/Very Good/Good/Acceptable |
| Visual_Appearance | Description | - | Color, texture, homogeneity |

**Additional Variables in Lower Section:**
- Flow_Table_Spread_mm, Compacting_Factor, Vebe_Time_sec
- Cohesiveness_Visual, Finishability, Pumpability_Rating
- Plastic_Viscosity_Relative, Yield_Stress_Relative (normalized to RC-00)

### 8_compressive_strength.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Mix_Name | Mixture name | - | - |
| Rubber_Content_Percent | Rubber level | % | - |
| Specimen_Type | Specimen geometry | - | Cube or Cylinder |
| Specimen_Dimension_mm | Size | mm | 150×150×150 for cubes |
| Age_Days | Curing age at test | days | 7 or 28 days |
| Test_Standard | Test method | - | BS EN 12390-3 |
| Specimen_ID | Individual specimen code | - | Unique identifier |
| Compressive_Strength_MPa | Individual strength | MPa | Peak load / area |
| Average_Strength_MPa | Mean of 5 specimens | MPa | - |
| Std_Dev_MPa | Standard deviation | MPa | - |
| COV_Percent | Coefficient of variation | % | (Std Dev / Mean) × 100 |
| Failure_Mode | Description of failure | - | Visual observation |
| Notes | Additional observations | - | - |

### 9_tensile_splitting_strength.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Specimen_Type | Specimen geometry | - | Cylinder |
| Diameter_mm | Cylinder diameter | mm | 150mm |
| Length_mm | Cylinder height | mm | 300mm |
| Age_Days | Curing age | days | 28 days |
| Test_Standard | Test method | - | ASTM C496 |
| Specimen_ID | Individual specimen code | - | - |
| Tensile_Splitting_Strength_MPa | Split tensile strength | MPa | f_t = 2P/(πLD) |
| Average_Strength_MPa | Mean of 5 specimens | MPa | - |
| Std_Dev_MPa | Standard deviation | MPa | - |
| COV_Percent | Coefficient of variation | % | - |
| Failure_Pattern | Crack pattern description | - | Visual observation |
| Energy_Absorption_Relative | Toughness relative to RC-00 | - | Normalized to 1.00 for control |
| Crack_Width_at_Peak_mm | Crack opening at peak load | mm | Measured with LVDT |
| Post_Peak_Behavior | Description of ductility | - | Brittle vs. ductile |

### 10_modulus_of_elasticity.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Specimen_Type | Specimen geometry | - | Cylinder |
| Diameter_mm | Cylinder diameter | mm | 150mm |
| Length_mm | Cylinder height | mm | 300mm |
| Age_Days | Curing age | days | 28 days |
| Test_Standard | Test method | - | ASTM C469 |
| Specimen_ID | Individual specimen code | - | - |
| Static_Modulus_GPa | Elastic modulus (tangent) | GPa | From initial linear portion |
| Average_Modulus_GPa | Mean of 5 specimens | GPa | - |
| Std_Dev_GPa | Standard deviation | GPa | - |
| COV_Percent | Coefficient of variation | % | - |
| Secant_Modulus_0_40_Percent_GPa | Secant modulus | GPa | From 0 to 40% of peak stress |
| Poisson_Ratio | Lateral strain ratio | - | ν = -ε_lateral / ε_axial |
| Strain_at_40_Percent_Peak_microstrain | Strain at 40% f'c | με | - |
| Ultimate_Strain_microstrain | Strain at failure | με | - |

### 11_density_porosity.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Age_Days | Curing age | days | 28 days |
| Specimen_ID | Individual specimen code | - | - |
| Oven_Dry_Density_kg_m3 | Density after oven drying | kg/m³ | Dried at 105°C to constant mass |
| SSD_Density_kg_m3 | Saturated-surface-dry density | kg/m³ | After saturation, surface dried |
| Apparent_Density_kg_m3 | Density excluding accessible voids | kg/m³ | Based on displaced water volume |
| Water_Absorption_Percent | Water absorption capacity | % | (m_SSD - m_dry) / m_dry × 100 |
| Total_Porosity_Percent | Total void content | % | Calculated from densities |
| Capillary_Porosity_Percent | Capillary pore fraction | % | Pores >10nm accessible to water |
| Gel_Porosity_Percent | Gel pore fraction | % | C-S-H interlayer porosity |
| Air_Void_Content_Percent | Entrapped/entrained air | % | Large voids |
| Permeable_Voids_Percent | Interconnected porosity | % | Accessible to water/gas |
| Average_Pore_Diameter_nm | Mean pore size | nm | From MIP (volume-weighted) |

### 12_pore_size_distribution_MIP.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Pore_Diameter_nm | Pore entrance diameter | nm | Calculated from Washburn equation |
| RC00_Cumulative_Intrusion_ml_g | Cumulative Hg intrusion (RC-00) | ml/g | Volume intruded up to this diameter |
| RC00_Incremental_Intrusion_ml_g | Incremental intrusion (RC-00) | ml/g | Volume in this size class |
| (Same for RC05, RC10, RC15) | | | - |
| Pore_Classification | Pore type category | - | Macro, capillary, or gel pores |

**Pore Classifications:**
- **Macro-pores:** >50,000 nm (large cracks, ITZ voids, rubber-cement interface)
- **Capillary pores:** 10-50,000 nm (subdivided into large, medium, small)
- **Gel pores:** <10 nm (C-S-H interlayer spaces)

### 13_ultrasonic_pulse_velocity.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Age_Days | Curing age | days | 28 days |
| Specimen_Type | Specimen geometry | - | Cylinder |
| Path_Length_mm | Transmission distance | mm | 300mm (cylinder height) |
| Specimen_ID | Individual specimen code | - | - |
| Direct_Transmission_UPV_km_s | Direct UPV | km/s | Through-transmission mode |
| Indirect_UPV_km_s | Indirect UPV | km/s | Surface measurement (less reliable) |
| Surface_UPV_km_s | Surface wave velocity | km/s | Rayleigh wave |
| Average_Direct_UPV_km_s | Mean direct UPV | km/s | Average of 5 specimens |
| Dynamic_Modulus_GPa | Dynamic elastic modulus | GPa | E_dyn = ρ × V² × (1+ν)(1-2ν)/(1-ν) |
| Concrete_Quality_Rating | Quality assessment | - | Based on UPV ranges |
| Homogeneity_Index | Uniformity measure | - | Ratio of indirect/direct UPV |
| Notes | Observations | - | - |

**UPV Quality Ratings (per BS 8110):**
- Excellent: >4.5 km/s
- Good: 4.0-4.5 km/s
- Medium: 3.0-4.0 km/s
- Doubtful: <3.0 km/s

### 14_microstructural_analysis.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Analysis_Method | Technique used | - | SEM, XRD, etc. |
| Parameter | Property measured | - | Descriptive name |
| Value | Measured value | Variable | See Unit column |
| Unit | Unit of measurement | - | Various |
| Location | Where measured | - | Bulk, ITZ, matrix, etc. |
| Notes | Interpretation | - | Significance of finding |

**Key Parameters:**
- **ITZ_Thickness_Average:** Width of interfacial transition zone
- **ITZ_Porosity:** Void content in ITZ
- **ITZ_Gap_Width:** Physical separation between aggregate and cement
- **Crack_Density_Matrix:** Number of microcracks per unit area
- **CH_Crystal_Size:** Calcium hydroxide crystal dimensions
- **Portlandite_CH (XRD):** Calcium hydroxide content from XRD
- **CSH_Gel (XRD):** Calcium silicate hydrate content

### 15_property_correlations.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Rubber_Content_Percent | Rubber level | % | - |
| Compressive_Strength_28d_MPa | 28-day compressive strength | MPa | Average value |
| Tensile_Strength_28d_MPa | 28-day split tensile strength | MPa | Average value |
| Modulus_Elasticity_GPa | Elastic modulus | GPa | Average value |
| UPV_km_s | Ultrasonic pulse velocity | km/s | Average direct transmission |
| Density_kg_m3 | Oven-dry density | kg/m³ | Average value |
| Porosity_Percent | Total porosity | % | Average value |
| Strength_Density_Ratio | Specific strength | - | f'c / density |
| Tensile_Compressive_Ratio | Strength ratio | - | f_t / f'c |
| Ductility_Index | Relative ductility | - | Normalized to RC-00 = 1.00 |
| Toughness_Index | Relative energy absorption | - | Area under stress-strain curve, normalized |

### 16_thermal_properties_ambient.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Thermal_Conductivity_W_mK | Heat conduction coefficient | W/(m·K) | Steady-state method |
| Specific_Heat_Capacity_J_kgK | Heat capacity | J/(kg·K) | DSC measurement |
| Thermal_Diffusivity_mm2_s | Rate of temperature change | mm²/s | α = k / (ρ × c_p) |
| Thermal_Expansion_Coefficient_per_K | Linear expansion | K⁻¹ | Coefficient of thermal expansion |
| Test_Temperature_C | Test temperature | °C | Ambient condition (23°C) |
| Test_Method | Standards used | - | ASTM C518, DSC |
| Notes | Observations | - | - |

### 17_permeability_durability.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Water_Permeability_Coefficient_m_s | Hydraulic conductivity | m/s | Darcy's law, pressure head |
| Gas_Permeability_m2 | Intrinsic permeability | m² | Oxygen permeability |
| Chloride_Penetration_Depth_mm | Chloride ingress | mm | After 90 days ponding, AgNO₃ spray |
| Chloride_Migration_Coefficient_m2_s | Chloride diffusion | m²/s | NT BUILD 492 rapid test |
| Carbonation_Depth_28d_mm | Carbonation ingress | mm | After 28 days accelerated exposure |
| Sorptivity_mm_sqrt_h | Capillary absorption rate | mm/√h | ASTM C1585 |
| Freeze_Thaw_Resistance_Mass_Loss_Percent | F-T damage | % | After 300 cycles ASTM C666 |
| Scaling_Resistance_kg_m2 | Surface scaling | kg/m² | ASTM C672 (salt scaling) |

### 18_stress_strain_curves.csv

| Variable | Description | Unit | Notes |
|----------|-------------|------|-------|
| Mix_ID | Mixture code | - | - |
| Strain_Microstrain | Axial strain | με | Measured with extensometer |
| Stress_MPa | Axial stress | MPa | Load / area |
| Loading_State | Phase of loading | - | Loading, Peak, Post-peak, or Failure |
| Specimen_Type | Specimen geometry | - | Cylinder 150×300mm |
| Notes | Observation | - | Description of behavior |

**Loading States:**
- **Loading:** Pre-peak ascending branch
- **Peak:** Maximum stress point
- **Post-peak:** Descending branch (softening)
- **Failure:** Final data point (end of test or specimen collapse)

---

## Abbreviations and Acronyms

### General
- **RC:** Rubberized Concrete
- **SSD:** Saturated-Surface-Dry
- **ITZ:** Interfacial Transition Zone
- **COV:** Coefficient of Variation
- **w/c:** Water-to-cement ratio

### Materials
- **SBR:** Styrene-Butadiene Rubber
- **PCE:** Polycarboxylate Ether (superplasticizer)
- **C-S-H:** Calcium Silicate Hydrate (main binding phase in concrete)
- **CH:** Calcium Hydroxide (Portlandite)
- **ZnO:** Zinc Oxide
- **CaCO₃:** Calcium Carbonate (Calcite)

### Testing Methods
- **TGA:** Thermogravimetric Analysis
- **DTG:** Derivative Thermogravimetry
- **DSC:** Differential Scanning Calorimetry
- **FTIR:** Fourier Transform Infrared Spectroscopy
- **ATR:** Attenuated Total Reflectance (FTIR mode)
- **SEM:** Scanning Electron Microscopy
- **XRD:** X-Ray Diffraction
- **XRF:** X-Ray Fluorescence
- **MIP:** Mercury Intrusion Porosimetry
- **UPV:** Ultrasonic Pulse Velocity
- **BET:** Brunauer-Emmett-Teller (surface area measurement)
- **GC-MS:** Gas Chromatography-Mass Spectrometry
- **HPLC:** High-Performance Liquid Chromatography

### Test Standards
- **ASTM:** American Society for Testing and Materials
- **BS EN:** British Standard European Norm
- **ISO:** International Organization for Standardization

---

## Data Types and Formats

### Numeric Data
- **Integer:** Whole numbers (e.g., Age_Days, Specimen counts)
- **Float:** Decimal numbers (most measurement data)
- **Precision:** Typically 1-2 decimal places for mechanical properties, up to 3-4 for small values

### Text Data
- **Mix_ID:** Alphanumeric code (e.g., "RC-05")
- **Specimen_ID:** Alphanumeric with hyphens (e.g., "RC05-28D-C3")
- **Descriptive fields:** Free text (e.g., Notes, Visual_Appearance, Failure_Mode)

### Categorical Data
- **Loading_State:** {Loading, Peak, Post-peak, Failure}
- **Concrete_Quality_Rating:** {Excellent, Good, Medium, Doubtful}
- **Peak_Intensity:** {weak, medium, strong}
- **Workability_Rating:** {Excellent, Very Good, Good, Acceptable, Fair, Poor}

### Missing Data
- Not applicable in this dataset (all planned tests completed)
- If present in future versions, indicated by: `NaN`, `N/A`, or blank cells

---

## Calculation Formulas

### Derived Properties

**1. Coefficient of Variation (COV):**
```
COV (%) = (Standard Deviation / Mean) × 100
```

**2. Tensile Splitting Strength:**
```
f_t = 2P / (π × L × D)
where:
  P = peak load (N)
  L = cylinder length (mm)
  D = cylinder diameter (mm)
```

**3. Total Porosity:**
```
Porosity (%) = [(ρ_apparent - ρ_dry) / ρ_apparent] × 100
where:
  ρ_apparent = apparent density (kg/m³)
  ρ_dry = oven-dry density (kg/m³)
```

**4. Water Absorption:**
```
Absorption (%) = [(m_SSD - m_dry) / m_dry] × 100
where:
  m_SSD = saturated-surface-dry mass
  m_dry = oven-dry mass
```

**5. Dynamic Modulus from UPV:**
```
E_dyn = ρ × V² × [(1+ν)(1-2ν)] / (1-ν)
where:
  ρ = density (kg/m³)
  V = ultrasonic pulse velocity (m/s)
  ν = Poisson's ratio
```

**6. Thermal Diffusivity:**
```
α = k / (ρ × c_p)
where:
  α = thermal diffusivity (m²/s)
  k = thermal conductivity (W/(m·K))
  ρ = density (kg/m³)
  c_p = specific heat capacity (J/(kg·K))
```

**7. Strength-to-Density Ratio:**
```
Specific Strength = f'c / ρ
where:
  f'c = compressive strength (MPa)
  ρ = density (kg/m³)
```

---

## Quality Indicators

### Reliability Flags
- **COV < 5%:** Excellent repeatability
- **COV 5-10%:** Good repeatability (typical for concrete)
- **COV 10-15%:** Acceptable (expected for rubberized concrete)
- **COV > 15%:** High variability (noted and explained in dataset)

### Sample Sizes
- **Mechanical Properties:** n = 5 specimens per mix per test age
- **Microstructural Analysis:** n = 3 locations per specimen, 2 specimens per mix
- **Chemical Analysis:** n = 3 replicate measurements

---

## File Format Details

### CSV Files
- **Delimiter:** Comma (`,`)
- **Text Qualifier:** None (fields with commas are avoided)
- **Encoding:** UTF-8
- **Line Endings:** Unix style (LF)
- **Header Row:** Yes (first row contains column names)
- **Decimal Separator:** Period (`.`)

### Column Naming Convention
- Use underscores (`_`) instead of spaces
- Units may be appended with underscores (e.g., `_MPa`, `_kg_m3`)
- Avoid special characters except underscore and minus sign
- Use descriptive but concise names

---

## Usage Notes

1. **Unit Consistency:** All measurements within a column use the same unit. Unit conversions have been applied where necessary.

2. **Significant Figures:** Reported precision reflects measurement accuracy:
   - Strength: 0.1 MPa
   - Density: 1 kg/m³
   - Modulus: 0.1 GPa
   - UPV: 0.01 km/s
   - Porosity: 0.1%

3. **Reference Conditions:** Unless otherwise stated:
   - Mechanical tests: 23°C, 50% RH
   - Density measurements: Oven-dried to constant mass at 105°C
   - Age: 28 days after casting

4. **Normalization:** Some properties are normalized relative to the control mix (RC-00 = 1.00):
   - Ductility_Index
   - Toughness_Index
   - Energy_Absorption_Relative
   - Plastic_Viscosity_Relative
   - Yield_Stress_Relative

---

## Data Validation

All data have been checked for:
- ✅ Unit consistency
- ✅ Physical plausibility (e.g., density > 0, porosity < 100%)
- ✅ Internal consistency (e.g., total porosity ≥ capillary porosity)
- ✅ Trend consistency (e.g., strength decreases with rubber content)
- ✅ Mass balance (mixture proportions sum correctly)
- ✅ Statistical validity (n ≥ 3 for all averaged values)

---

**Data Dictionary Version:** 1.0  
**Last Updated:** 2025-10-17  
**Maintained By:** Rubberized Concrete Fire Resistance Research Project

---

For questions or clarifications about any variable definitions, please refer to the main README.md or consult the relevant ASTM/BS standards cited in the Test_Method columns.
