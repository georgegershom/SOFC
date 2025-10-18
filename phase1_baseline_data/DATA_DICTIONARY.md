# Phase 1 Baseline Data - Data Dictionary
## Fire-Resistant Rubberized Concrete Research Project

**Project Title:** Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Phase:** 1 - Material Characterization & Specimen Preparation  
**Data Collection Period:** January 15, 2024 - February 17, 2024  
**Principal Investigator:** [To be filled]  
**Institution:** [To be filled]  

---

## 1. CEMENT CHARACTERIZATION DATA
**File:** `cement_characterization.csv`

### Physical Properties
- **Material_ID**: Unique identifier for cement batch (e.g., CEM-001)
- **Cement_Type**: Portland cement classification (e.g., OPC Type I per ASTM C150)
- **Specific_Gravity**: Ratio of cement density to water density (dimensionless, typically 3.10-3.25)
- **Blaine_Fineness_m2_kg**: Surface area per unit mass (m²/kg, typically 300-400)
- **Initial_Setting_Time_min**: Time from mixing to initial set (minutes, ASTM C191)
- **Final_Setting_Time_min**: Time from mixing to final set (minutes, ASTM C191)
- **Compressive_Strength_Xday_MPa**: Mortar cube strength at X days (MPa, ASTM C109)

### Chemical Composition (XRF Analysis)
- **CaO**: Calcium oxide (%, primary cement component)
- **SiO2**: Silicon dioxide (%, forms strength-giving CSH)
- **Al2O3**: Aluminum oxide (%, rapid setting component)
- **Fe2O3**: Iron oxide (%, affects color and early hydration)
- **MgO**: Magnesium oxide (%, soundness indicator, limit <6%)
- **SO3**: Sulfur trioxide (%, controls setting time, limit <3.5%)
- **Na2O, K2O**: Alkalis (%, can cause ASR issues)
- **LOI**: Loss on ignition (%, indicates carbonation/prehydration)

### Bogue Composition (Calculated Phases)
- **C3S**: Tricalcium silicate (%, 3CaO·SiO2, early strength)
- **C2S**: Dicalcium silicate (%, 2CaO·SiO2, long-term strength)
- **C3A**: Tricalcium aluminate (%, 3CaO·Al2O3, rapid hydration)
- **C4AF**: Tetracalcium aluminoferrite (%, 4CaO·Al2O3·Fe2O3)

---

## 2. FINE AGGREGATE CHARACTERIZATION
**File:** `fine_aggregate_characterization.csv`

### Basic Properties
- **Material_ID**: Unique identifier (e.g., FA-001)
- **Aggregate_Type**: Natural or manufactured sand
- **Specific_Gravity_SSD**: Saturated surface-dry specific gravity (ASTM C128)
- **Specific_Gravity_OD**: Oven-dry specific gravity (ASTM C128)
- **Water_Absorption_percent**: % mass increase when saturated (ASTM C128)
- **Bulk_Density_kg_m3**: Compacted dry bulk density (ASTM C29)
- **Fineness_Modulus**: Sum of cumulative % retained divided by 100 (ASTM C136)

### Sieve Analysis
- **Sieve_Size_mm**: Opening size of sieve
- **Mass_Retained_g**: Mass retained on each sieve
- **Percent_Retained**: Mass retained as % of total
- **Cumulative_Percent_Retained**: Running sum of % retained
- **Percent_Passing**: 100 - cumulative % retained
- **Specification_ASTM_C33_Min/Max**: ASTM C33 grading limits

### Deleterious Substances
- **Soundness_MgSO4**: Mass loss after magnesium sulfate test (%, <10% limit)
- **Organic_Impurities**: Color plate comparison (ASTM C40)
- **Clay_Lumps_Friable**: % friable particles (<1% limit)
- **Material_Finer_than_75um**: Silt/clay content (%, <3% typical)

---

## 3. COARSE AGGREGATE CHARACTERIZATION
**File:** `coarse_aggregate_characterization.csv`

### Basic Properties
- Similar to fine aggregate, plus:
- **Nominal_Max_Size_mm**: 95% passing this sieve size
- **LA_Abrasion_Loss_percent**: Los Angeles abrasion resistance (ASTM C131, <40%)

### Additional Mechanical Properties
- **Crushing_Value**: Resistance to crushing (%, BS 812, <30% good)
- **Impact_Value**: Resistance to impact loading (%, BS 812, <30% good)
- **Flat_Elongated_Particles**: % particles with length/width >3 (ASTM D4791)

---

## 4. CRUMB RUBBER CHARACTERIZATION
**File:** `crumb_rubber_characterization.csv`

### Physical Properties
- **Material_ID**: CR-001 (fine, 1-4mm), CR-002 (coarse, 4-8mm)
- **Rubber_Type**: Crumb rubber from recycled tires
- **Tire_Type**: Source tire type (passenger, truck, etc.)
- **Processing_Method**: Ambient or cryogenic grinding
- **Specific_Gravity**: Typically 1.10-1.20 (much lower than aggregate ~2.65)
- **Apparent_Density_kg_m3**: Particle density
- **Bulk_Density_kg_m3**: Loose bulk density (important for volume calculations)
- **Mohs_Hardness**: Surface hardness (2-3, soft)
- **Shore_A_Hardness**: Rubber hardness scale (60-70 typical)

### Chemical Composition (TGA/FTIR)
- **Natural_Rubber_SBR**: Polymer content (40-50%)
- **Carbon_Black**: Reinforcing filler (25-30%)
- **Zinc_Oxide**: Vulcanization agent (1-2%)
- **Sulfur_Compounds**: Cross-linking agents (1-2%)
- **Oils_Plasticizers**: Processing aids (3-5%)
- **Ash_Inorganics**: Mineral fillers and residues (10-20%)

### Thermal Properties
- **Glass_Transition_Temp_Tg**: Temperature at which rubber becomes brittle (typically -40 to -50°C)
- **Thermal_Conductivity**: Heat transfer coefficient (0.2-0.25 W/m·K)
- **Specific_Heat_Capacity**: Heat storage capacity (1.3-1.5 kJ/kg·K)
- **Decomposition_Onset_Temp**: Start of thermal degradation (~280-300°C)
- **CTE**: Coefficient of thermal expansion (~180-200 × 10⁻⁶/°C)

---

## 5. WATER AND ADMIXTURE CHARACTERIZATION
**File:** `water_admixture_characterization.csv`

### Water Quality (ASTM D1293 and related)
- **pH**: Acidity/alkalinity (6.0-8.0 acceptable)
- **Total_Dissolved_Solids**: Dissolved mineral content (<2000 mg/L limit)
- **Chloride_Cl**: Corrosion concern (<500 mg/L for reinforced concrete)
- **Sulfate_SO4**: Durability concern (<3000 mg/L)

### Chemical Admixtures
#### ADM-001: Superplasticizer (High-Range Water Reducer)
- **Type**: Polycarboxylate ether (PCE) based
- **Function**: Increases workability without adding water
- **Solid_Content**: 38-42% by mass
- **Dosage_Range**: 0.2-1.5% by cement mass
- **Water_Reduction**: 20-35% typical

#### ADM-002: Air Entraining Agent (AEA)
- **Type**: Vinsol resin based
- **Function**: Improves freeze-thaw resistance
- **Target_Air_Content**: 4-7% for durability
- **Dosage_Range**: 0.01-0.05% by cement mass

#### ADM-003: Viscosity Modifying Agent (VMA)
- **Type**: Polysaccharide based
- **Function**: Prevents segregation, improves cohesion
- **Dosage_Range**: 0.05-0.3% by cement mass

---

## 6. MIX DESIGN MATRIX
**File:** `mix_design_matrix.csv`

### Mix Identification
- **Mix_ID**: Unique code (M-00 = control, M-05F = 5% fine rubber, etc.)
- **Mix_Name**: Descriptive name
- **Rubber_Replacement_Level_percent**: Volume % of fine aggregate replaced (0, 5, 10, 15, 20)
- **Rubber_Size_Range_mm**: 1-4mm (fine), 4-8mm (coarse), or 1-8mm (mixed)

### Mix Proportions (Absolute Volume Method)
All quantities are per cubic meter of concrete:
- **Cement_kg**: Always 420 kg/m³ (typical for 40 MPa concrete)
- **Water_kg**: Varies 176-202 kg (w/c ratio increases with rubber content)
- **Fine_Aggregate_kg**: Decreases as rubber replaces volume
- **Coarse_Aggregate_kg**: Constant at 1050 kg/m³
- **Crumb_Rubber_kg**: Calculated to replace specified volume of sand

### Volume Balance
Total volume must equal 1.000 m³:
- Cement volume = mass / (specific gravity × 1000)
- Water volume = mass (liters, since density ≈ 1 kg/L)
- Aggregate volume = mass / (specific gravity × 1000)
- Rubber volume = mass / (specific gravity × 1000)
- Air volume = target air content
- Admixture volume (negligible but included)

### Design Philosophy
- **Water-Cement Ratio**: Increases slightly with rubber content (0.42 to 0.48) to maintain workability
- **Superplasticizer**: Dosage increases with rubber content (1.0% to 1.4% by cement mass)
- **VMA**: Added at 10%+ rubber to prevent segregation
- **Target Strength**: All mixes designed for 40 MPa at 28 days (control)

---

## 7. FRESH PROPERTIES DATA
**File:** `fresh_properties_data.csv`

### Batch Information
- **Batch_ID**: Unique identifier for each concrete batch (B001-B033)
- **Casting_Date**: Date of mixing
- **Batch_Time**: Time of mixing
- **Ambient_Temp_C**: Lab temperature (affects hydration rate)
- **Relative_Humidity_percent**: Lab humidity
- **Concrete_Temp_C**: Fresh concrete temperature
- **Mixing_Time_min**: Duration of mixing (increases with rubber content)

### Workability Tests
- **Slump_mm**: ASTM C143 slump cone test (target 75 mm ± 25 mm)
  - Control: 72-78 mm
  - 20% rubber: 98-112 mm (higher due to rubber resilience)
- **VSI_Rating**: Visual Stability Index (0=stable, 3=segregated)
- **Segregation_Index_percent**: Measured separation

### Air Content and Density
- **Air_Content_Pressure_Method_percent**: ASTM C231 Type B meter
- **Air_Content_Volumetric_Method_percent**: ASTM C173 (verification)
- **Unit_Weight_kg_m3**: Fresh concrete density (decreases with rubber content)
  - Control: ~2325 kg/m³
  - 20% rubber: ~2240 kg/m³ (4% reduction)
- **Yield_m3**: Actual volume produced (should be ≈1.000)

### Setting Time
- **Initial_Setting_Time_min**: Penetration resistance 3.5 MPa (ASTM C403)
  - Control: ~180 min
  - 20% rubber: ~255 min (delayed due to rubber surface effects)
- **Final_Setting_Time_min**: Penetration resistance 27.6 MPa
  - Control: ~300 min
  - 20% rubber: ~420 min

### Bleeding
- **Bleeding_Rate_mL_cm2_hr**: Water accumulation rate
- **Total_Bleeding_percent**: Total bleed water as % of mix water
  - Increases with rubber content (0.8% to 3.9%)
  - Higher bleeding = weaker ITZ (interfacial transition zone)

### Rheological Properties
Measured with rotational rheometer (Bingham model):
- **Yield_Stress_Pa**: Stress required to initiate flow
  - Increases significantly with rubber content (42 to 158 Pa)
- **Plastic_Viscosity_Pa_s**: Flow resistance once moving
  - Increases with rubber content (18 to 52 Pa·s)
- **Thixotropy_Pa_s**: Time-dependent viscosity change
- **Hysteresis_Area_Pa_s**: Energy loss in loading-unloading cycle

**Note:** Higher rubber content = less workable, more viscous, requires more admixtures

---

## 8. SEM MORPHOLOGY DATA
**File:** `sem_morphology_data.csv`

### Imaging Parameters
- **Image_ID**: Unique identifier for each SEM image
- **Magnification**: 100X to 5000X
- **Accelerating_Voltage_kV**: 15-20 kV typical
- **Working_Distance_mm**: Distance from sample to detector
- **Coating_Info**: Gold-palladium sputter coating (10 nm) for conductivity

### Morphological Observations
- **Average_Particle_Diameter**: CR-001 ≈ 2.5 mm, CR-002 ≈ 5.8 mm
- **Particle_Shape_Factor**: 0.68 (angular, not spherical like aggregate)
- **Surface_Texture**: Rough, irregular with cracks
- **Porosity**: 12.5% visible porosity
- **Carbon_Black_Visible**: Dense aggregates of 30-50 nm particles

### Quantitative Analysis (ImageJ)
- **Circularity**: 0.49-0.58 (1.0 = perfect circle, rubber is irregular)
- **Aspect_Ratio**: 1.82-2.28 (elongated particles)
- **Solidity**: 0.84-0.89 (ratio of area to convex hull area)

### EDX Elemental Analysis
- **Carbon**: 81-85% (rubber polymer + carbon black)
- **Oxygen**: 7-8% (oxidized surface)
- **Zinc**: 3-4% (zinc oxide vulcanization agent)
- **Sulfur**: 1.7-2.2% (vulcanization cross-links)

---

## 9. THERMAL ANALYSIS DATA
**File:** `thermal_analysis_data.csv`

### TGA (Thermogravimetric Analysis)
Measures mass loss vs. temperature:
- **Heating_Rate**: 10°C/min in nitrogen atmosphere
- **Temperature_Range**: 25-800°C

#### Mass Loss Stages:
1. **25-150°C (1.2%)**: Moisture and volatiles
2. **150-350°C (4.5%)**: Oils and plasticizers
3. **350-500°C (48.5%)**: Main polymer decomposition (peak at 425°C)
4. **500-700°C (28.2%)**: Carbon black oxidation (in air) or stable (in N₂)
5. **700-800°C (1.8%)**: Zinc oxide decomposition
6. **Residue (15.8%)**: Ash and inorganics

**DTG_Peak_Rate**: Derivative TG shows decomposition rate (%/min)
- Main polymer peak: 8.5 %/min at 425°C (rapid decomposition)

### DSC (Differential Scanning Calorimetry)
Measures heat flow vs. temperature:
- **Glass_Transition_Temp**: -45°C (rubber becomes brittle below this)
- **Endothermic_Peaks**: 
  - 125°C: Plasticizer melting
  - 185°C: Decomposition onset
- **Exothermic_Peak**: 255°C: Oxidative cross-linking

### Temperature-Dependent Properties
Critical for fire modeling:

**Specific Heat (kJ/kg·K):**
- Rubber: 1.38 at 25°C → 1.85 at 300°C (increases ~35%)
- Concrete: 0.88 at 25°C → 1.08 at 300°C

**Thermal Conductivity (W/m·K):**
- Rubber: 0.23 at 25°C → 0.19 at 200°C (decreases)
- Control concrete: 1.75
- 20% rubber concrete: 1.23 (30% reduction = better insulation)

**Thermal Expansion (×10⁻⁶/°C):**
- Rubber: 185 (very high)
- Concrete: 10.5 (low)
- Mismatch can cause internal stresses at high temperatures

---

## DATA QUALITY AND RELIABILITY

### Replication
- Each mix design tested in triplicate (3 batches)
- Provides measure of variability and experimental error
- Standard deviation should be <5% for most properties

### Test Standards
All tests follow ASTM, BS, or ISO standards:
- ASTM C150: Cement specification
- ASTM C33: Aggregate specification
- ASTM C494: Chemical admixtures
- ASTM C143: Slump test
- ASTM C231: Air content
- ASTM C403: Setting time

### Calibration
- All instruments calibrated per manufacturer specifications
- SEM: Certified calibration samples
- TGA/DSC: Indium and zinc standards
- Load cells: NIST-traceable weights

### Environmental Control
- Lab temperature: 22 ± 2°C
- Relative humidity: 40-50%
- Curing: Moist room at 23 ± 2°C, 100% RH

---

## UNITS AND CONVENTIONS

### SI Units Used
- Length: mm, m
- Mass: g, kg
- Temperature: °C (K for absolute)
- Pressure/Stress: Pa, MPa (1 MPa = 1 N/mm²)
- Density: kg/m³
- Energy: J, kJ
- Power: W

### Abbreviations
- **OPC**: Ordinary Portland Cement
- **SSD**: Saturated Surface Dry
- **OD**: Oven Dry
- **w/c**: Water-cement ratio
- **ITZ**: Interfacial Transition Zone
- **CSH**: Calcium Silicate Hydrate (cement gel)
- **SBR**: Styrene-Butadiene Rubber
- **PCE**: Polycarboxylate Ether
- **HRWR**: High-Range Water Reducer
- **AEA**: Air Entraining Agent
- **VMA**: Viscosity Modifying Agent

---

## NEXT PHASES

### Phase 2: Mechanical Property Testing
- Compressive strength (3, 7, 28, 90 days)
- Tensile strength (split cylinder)
- Flexural strength (beam test)
- Modulus of elasticity
- Poisson's ratio
- Fracture energy

### Phase 3: High-Temperature Testing
- Residual strength after heating (200-800°C)
- Mass loss at temperature
- Spalling behavior
- Thermal strain
- Explosive spalling risk

### Phase 4: Thermo-Mechanical Model Development
- Constitutive model development
- Finite element implementation
- Model calibration with Phase 3 data
- Validation testing

### Phase 5: Full-Scale Fire Testing
- Structural element fabrication
- Standard fire curve testing (ISO 834, ASTM E119)
- Temperature profiles through thickness
- Structural response (deflection, failure mode)
- Model validation

---

## CITATIONS AND REFERENCES

### Key ASTM Standards
- ASTM C150: Standard Specification for Portland Cement
- ASTM C136: Sieve Analysis of Fine and Coarse Aggregates
- ASTM C143: Slump of Hydraulic-Cement Concrete
- ASTM C231: Air Content of Freshly Mixed Concrete by Pressure Method
- ASTM C494: Chemical Admixtures for Concrete
- ASTM D1293: pH of Water

### Related Research
1. Topçu, İ. B., & Bilir, T. (2009). Experimental investigation of some fresh and hardened properties of rubberized self-compacting concrete. *Materials & Design*, 30(8), 3056-3065.
2. Güneyisi, E., Gesoglu, M., & Özturan, T. (2004). Properties of rubberized concretes containing silica fume. *Cement and Concrete Research*, 34(12), 2309-2317.
3. Thomas, B. S., & Gupta, R. C. (2016). Properties of high strength concrete containing scrap tire rubber. *Journal of Cleaner Production*, 113, 86-92.
4. Holmes, N., Browne, A., & Montague, C. (2014). Acoustic properties of concrete panels with crumb rubber as a fine aggregate replacement. *Construction and Building Materials*, 73, 195-204.

---

## CONTACT INFORMATION

**Data Generated By:** Research Team  
**Date Range:** January 15 - February 17, 2024  
**Version:** 1.0  
**Last Updated:** 2024-02-20  

For questions about this dataset, contact: [To be filled]

---

## FILE STRUCTURE

```
phase1_baseline_data/
├── cement_characterization.csv
├── fine_aggregate_characterization.csv
├── coarse_aggregate_characterization.csv
├── crumb_rubber_characterization.csv
├── water_admixture_characterization.csv
├── mix_design_matrix.csv
├── fresh_properties_data.csv
├── sem_morphology_data.csv
├── thermal_analysis_data.csv
├── DATA_DICTIONARY.md
├── analysis_scripts/
│   └── data_analysis.py
└── README.md
```

---

**END OF DATA DICTIONARY**
