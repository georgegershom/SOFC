# Phase 1 Baseline Data: Fire-Resistant Rubberized Concrete

## Project Overview

**Title:** Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Phase:** 1 - Material Characterization & Specimen Preparation  
**Status:** Complete  
**Data Collection Period:** January 15 - February 17, 2024

## Executive Summary

This dataset contains comprehensive baseline characterization data for the development of fire-resistant rubberized concrete. The dataset includes:

- **11 mix designs** (1 control + 10 rubberized variations)
- **33 concrete batches** (3 replicates per mix)
- **5 constituent materials** fully characterized
- **Temperature range:** 25-800°C thermal analysis
- **Multiple analysis techniques:** XRF, TGA, DSC, FTIR, SEM-EDX

### Key Findings Preview

1. **Rubber Content:** 0-20% volume replacement of fine aggregate
2. **Density Reduction:** Up to 8% lighter concrete with 20% rubber
3. **Workability:** Increased slump but requires higher admixture dosage
4. **Thermal Insulation:** Up to 30% reduction in thermal conductivity
5. **Fire Resistance:** Expected improvement in fire ratings (to be confirmed in Phase 3)

## Dataset Structure

```
phase1_baseline_data/
├── cement_characterization.csv          # OPC Type I characterization
├── fine_aggregate_characterization.csv  # Natural sand properties
├── coarse_aggregate_characterization.csv # Crushed limestone properties
├── crumb_rubber_characterization.csv    # Tire-derived rubber (2 sizes)
├── water_admixture_characterization.csv # Water quality + 3 admixtures
├── mix_design_matrix.csv                # 11 mix proportions
├── fresh_properties_data.csv            # 33 batches, fresh state tests
├── sem_morphology_data.csv              # SEM imaging and EDX data
├── thermal_analysis_data.csv            # TGA, DSC, thermal properties
├── DATA_DICTIONARY.md                   # Complete data documentation
├── README.md                            # This file
└── analysis_scripts/
    └── data_analysis.py                 # Python visualization scripts
```

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy matplotlib seaborn scipy
```

### Loading the Data

```python
import pandas as pd

# Load main datasets
cement = pd.read_csv('cement_characterization.csv', comment='#')
rubber = pd.read_csv('crumb_rubber_characterization.csv', comment='#')
mix_designs = pd.read_csv('mix_design_matrix.csv', comment='#')
fresh_props = pd.read_csv('fresh_properties_data.csv', comment='#')

# View mix design summary
print(mix_designs[['Mix_ID', 'Mix_Name', 'Rubber_Replacement_Level_percent', 
                   'Water_Cement_Ratio', 'Theoretical_Density_kg_m3']])
```

### Running Analysis Scripts

```bash
cd analysis_scripts
python data_analysis.py
```

This will generate:
- Mix design comparison plots
- Fresh property trends
- Thermal analysis curves
- Statistical summaries

## Mix Design Matrix

| Mix ID | Description | Rubber % | Rubber Size | W/C Ratio | Target Strength |
|--------|-------------|----------|-------------|-----------|-----------------|
| M-00 | Control | 0% | N/A | 0.42 | 40 MPa |
| M-05F | Fine Rubber | 5% | 1-4 mm | 0.42 | 36 MPa |
| M-10F | Fine Rubber | 10% | 1-4 mm | 0.44 | 32 MPa |
| M-15F | Fine Rubber | 15% | 1-4 mm | 0.46 | 28 MPa |
| M-20F | Fine Rubber | 20% | 1-4 mm | 0.48 | 24 MPa |
| M-05C | Coarse Rubber | 5% | 4-8 mm | 0.42 | 35 MPa |
| M-10C | Coarse Rubber | 10% | 4-8 mm | 0.44 | 30 MPa |
| M-15C | Coarse Rubber | 15% | 4-8 mm | 0.46 | 26 MPa |
| M-20C | Coarse Rubber | 20% | 4-8 mm | 0.48 | 22 MPa |
| M-10M | Mixed Rubber | 10% | 1-8 mm | 0.44 | 31 MPa |
| M-15M | Mixed Rubber | 15% | 1-8 mm | 0.46 | 27 MPa |

## Key Material Properties

### Cement (CEM-001)
- **Type:** OPC Type I
- **Specific Gravity:** 3.15
- **Blaine Fineness:** 385 m²/kg
- **28-day Strength:** 48.7 MPa
- **C3S Content:** 55.2% (high early strength)

### Fine Aggregate (FA-001)
- **Type:** Natural river sand
- **Specific Gravity:** 2.65 (SSD)
- **Water Absorption:** 2.71%
- **Fineness Modulus:** 2.85

### Coarse Aggregate (CA-001)
- **Type:** Crushed limestone
- **Nominal Max Size:** 19 mm
- **Specific Gravity:** 2.68 (SSD)
- **LA Abrasion:** 22.5% (good quality)

### Crumb Rubber (CR-001 & CR-002)
- **Source:** Passenger car tires
- **Specific Gravity:** 1.14-1.15 (57% lighter than aggregate!)
- **Composition:** 48.5% polymer, 28.2% carbon black
- **Thermal Conductivity:** 0.23 W/m·K (vs 1.75 for concrete)
- **Glass Transition:** -45°C
- **Decomposition Onset:** 285°C

## Critical Observations

### 1. Fresh State Behavior

**Workability:**
- Slump increases with rubber content (78 mm → 108 mm)
- BUT: Higher viscosity and yield stress
- Requires 43% more superplasticizer at 20% rubber

**Air Content:**
- Increases from 2.0% (control) to 4.3% (20% rubber)
- Due to rubber surface characteristics and trapped air
- May affect strength but improves freeze-thaw resistance

**Setting Time:**
- Delayed by ~40% at 20% rubber replacement
- Initial set: 180 min → 255 min
- Due to rubber surface chemistry affecting cement hydration

### 2. Thermal Properties

**Insulation Effect:**
- Thermal conductivity decreases linearly with rubber content
- 20% rubber: 30% reduction (1.75 → 1.23 W/m·K)
- **Implication:** Better fire insulation, slower heat penetration

**Decomposition:**
- Rubber stable to ~285°C
- Main decomposition: 350-500°C (48.5% mass loss)
- Carbon black residue provides some post-fire structure

**Thermal Expansion Mismatch:**
- Rubber CTE: 185 × 10⁻⁶/°C
- Concrete CTE: 10.5 × 10⁻⁶/°C
- **Risk:** Internal stresses at high temperatures

### 3. Morphology (SEM Analysis)

**Rubber Particles:**
- Highly irregular, angular shape (circularity 0.49-0.58)
- Very rough surface texture with micro-cracks
- Embedded carbon black aggregates (30-50 nm)
- 12.5% porosity

**ITZ Concerns:**
- Weak interfacial transition zone expected
- Rubber surface chemistry incompatible with cement paste
- May benefit from surface treatment (future work)

## Data Quality Metrics

### Replication
- **3 batches per mix** = 33 total batches
- **Coefficient of variation:**
  - Slump: 3-5% (excellent)
  - Air content: 2-4% (excellent)
  - Unit weight: 0.5-1.5% (excellent)

### Calibration
- All instruments NIST-traceable
- SEM: Au/Pd standard
- TGA/DSC: Indium/zinc calibration
- Balances: ±0.01 g accuracy

### Standards Compliance
- ASTM C150 (cement)
- ASTM C33 (aggregates)
- ASTM C494 (admixtures)
- All test methods per ASTM/ISO standards

## Research Questions Addressed

✅ **Q1:** What is the exact composition of tire-derived crumb rubber?  
**A:** 48.5% polymer (NR/SBR), 28.2% carbon black, 15.8% ash/inorganics

✅ **Q2:** How does rubber content affect fresh concrete workability?  
**A:** Increases slump but also viscosity; requires higher admixture dosage (up to +43%)

✅ **Q3:** What is the thermal insulation potential?  
**A:** Linear decrease in thermal conductivity: 30% reduction at 20% rubber

✅ **Q4:** When does rubber decompose under fire conditions?  
**A:** Decomposition onset at 285°C, major decomposition 350-500°C

✅ **Q5:** What is the optimal rubber particle size?  
**A:** Fine (1-4 mm) shows better dispersion; coarse (4-8 mm) shows slightly lower strength loss (to be confirmed in Phase 2)

## Next Steps (Future Phases)

### Phase 2: Mechanical Testing (Upcoming)
- [ ] Compressive strength (3, 7, 28, 90 days)
- [ ] Tensile strength (split cylinder, flexural)
- [ ] Elastic modulus and Poisson's ratio
- [ ] Fracture energy and toughness

### Phase 3: High-Temperature Testing (Months 3-6)
- [ ] Residual strength after heating (200-800°C)
- [ ] Mass loss and spalling behavior
- [ ] Thermal strain measurements
- [ ] Explosive spalling risk assessment

### Phase 4: Model Development (Months 6-9)
- [ ] Constitutive model development
- [ ] FEM implementation (ABAQUS/ANSYS)
- [ ] Parameter calibration
- [ ] Sensitivity analysis

### Phase 5: Validation Testing (Months 9-12)
- [ ] Full-scale structural elements
- [ ] Standard fire curve testing (ISO 834)
- [ ] Temperature profiles through thickness
- [ ] Model validation and refinement

## Potential Applications

1. **Fire-Resistant Walls:** Improved thermal insulation delays structural collapse
2. **Parking Structures:** Reduced weight + fire resistance + uses recycled materials
3. **Blast-Resistant Barriers:** Enhanced energy absorption
4. **Seismic Applications:** Improved damping and ductility
5. **Acoustic Insulation:** Sound absorption properties
6. **Sustainable Construction:** 100-200 kg of recycled tires per 10 m³ concrete

## Limitations and Considerations

⚠️ **Strength Reduction:** Expected 10-40% reduction in compressive strength  
⚠️ **Long-Term Durability:** Rubber aging and degradation needs investigation  
⚠️ **ITZ Weakness:** Weak interface between rubber and cement paste  
⚠️ **Fire Aftermath:** Post-fire structural integrity uncertain (Phase 3)  
⚠️ **Bleeding:** Increased bleeding may affect surface quality  
⚠️ **Cost:** Premium admixtures required, rubber processing costs  

## Data Citation

If you use this dataset, please cite:

```
[Research Team]. (2024). Phase 1 Baseline Data: Fire-Resistant Rubberized Concrete. 
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural 
Elements Utilizing High-Performance Rubberized Concrete. [Institution Name].
```

## License and Usage

This dataset is provided for research and educational purposes.

**Permitted Uses:**
- Academic research and publications
- Educational purposes
- Model development and validation
- Non-commercial applications

**Restrictions:**
- Commercial use requires permission
- Proper attribution required
- No warranty provided

## Contact and Support

**Principal Investigator:** [Name]  
**Email:** [Email]  
**Institution:** [Institution]  
**Project Website:** [URL]

For questions about:
- **Data interpretation:** Contact [Name]
- **Experimental methods:** Contact [Name]
- **Analysis scripts:** Contact [Name]
- **Collaborations:** Contact [Name]

## Acknowledgments

This research was supported by:
- [Funding Agency]
- [Equipment Grants]
- [Industry Partners]

Special thanks to:
- Materials characterization lab staff
- Tire recycling facility for rubber samples
- Chemical admixture manufacturers for technical support

## Version History

- **v1.0** (2024-02-20): Initial release
  - Complete Phase 1 characterization data
  - 11 mix designs, 33 batches
  - All constituent materials characterized
  - Thermal analysis complete

## Related Publications

[To be updated as research progresses]

---

**Last Updated:** 2024-02-20  
**Dataset Version:** 1.0  
**Status:** Complete - Ready for Phase 2
