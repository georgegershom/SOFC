# Changelog
## Baseline Rubberized Concrete Dataset

All notable changes to this dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-10-17

### Added - Initial Release

#### Dataset Files (18 CSV files)
1. `1_mixture_proportions.csv` - Complete mix designs for 4 concrete mixtures
2. `2_aggregate_grading.csv` - Particle size distribution for all aggregates
3. `3_rubber_characterization.csv` - Physical properties of crumb rubber
4. `4_rubber_chemical_composition.csv` - Chemical composition analysis
5. `5_rubber_thermal_analysis.csv` - TGA/DSC data (25-800°C)
6. `6_rubber_ftir_peaks.csv` - FTIR spectroscopy results
7. `7_fresh_state_properties.csv` - Fresh concrete workability data
8. `8_compressive_strength.csv` - Compressive strength @ 7 & 28 days (100 specimens)
9. `9_tensile_splitting_strength.csv` - Split tensile strength @ 28 days (20 specimens)
10. `10_modulus_of_elasticity.csv` - Static elastic modulus @ 28 days (20 specimens)
11. `11_density_porosity.csv` - Density and porosity measurements (20 specimens)
12. `12_pore_size_distribution_MIP.csv` - Mercury intrusion porosimetry results
13. `13_ultrasonic_pulse_velocity.csv` - UPV measurements (20 specimens)
14. `14_microstructural_analysis.csv` - SEM and XRD characterization
15. `15_property_correlations.csv` - Summary correlations table
16. `16_thermal_properties_ambient.csv` - Baseline thermal properties @ 23°C
17. `17_permeability_durability.csv` - Transport and durability properties
18. `18_stress_strain_curves.csv` - Complete stress-strain behavior

#### Documentation
- `README.md` - Comprehensive dataset documentation (50+ pages)
- `DATA_DICTIONARY.md` - Complete variable definitions and units
- `CHANGELOG.md` - Version history (this file)

#### Analysis Tools
- `analysis_script.py` - Python script for data analysis and visualization
- `requirements.txt` - Python package dependencies

### Features

#### Material Characterization
- **4 mix designs:** 0%, 5%, 10%, 15% rubber replacement (by volume of fine aggregate)
- **Cement:** Type I Portland, 420 kg/m³, w/c = 0.45 (constant)
- **Rubber:** Crumb rubber from truck tires, 1-4mm, alkaline pre-treatment
- **Curing:** 28 days in lime-saturated water @ 23°C

#### Test Coverage
- **Mechanical Properties:** Compressive strength, tensile strength, elastic modulus, stress-strain curves
- **Physical Properties:** Density, porosity, pore size distribution, water absorption
- **Non-Destructive Testing:** Ultrasonic pulse velocity (direct, indirect, surface)
- **Microstructure:** SEM imaging, XRD phase analysis, ITZ characterization
- **Thermal Properties:** Conductivity, specific heat, diffusivity, thermal expansion
- **Durability:** Permeability, chloride penetration, carbonation, freeze-thaw, scaling
- **Fresh State:** Slump flow, air content, setting times, bleeding, workability

#### Statistical Rigor
- **Sample size:** n = 5 specimens per test per mix (total >100 specimens)
- **Quality metrics:** Mean, standard deviation, coefficient of variation for all properties
- **Data validation:** All data checked for consistency, plausibility, and trends

#### Scientific Insights
- Systematic strength reduction: -10.5% to -37.9% compressive strength with rubber
- Dramatic ductility enhancement: 2.4× increase at 15% rubber
- Weak rubber-cement interface: ITZ thickness 95-165 μm (vs. 45 μm normal)
- Increased porosity: 12.8% → 19.5% total porosity
- Enhanced thermal insulation: 1.82 → 1.12 W/(m·K) thermal conductivity
- Improved freeze-thaw resistance despite increased permeability

### Data Quality

#### Reliability
- **COV:** 2.1% to 11.3% across all tests (excellent to acceptable)
- **Standards compliance:** All tests per ASTM/BS/ISO standards
- **Environmental control:** 23±2°C, 50±5% RH maintained
- **Calibration:** All equipment calibrated per standards

#### Consistency
- ✅ Unit consistency across all files
- ✅ Physical plausibility validated
- ✅ Internal consistency (e.g., porosity ≥ capillary porosity)
- ✅ Trend consistency (systematic variation with rubber content)
- ✅ Mass balance (mixture proportions verified)

### Usage

#### For Researchers
- Baseline data for thermo-mechanical model development
- Validation dataset for ambient temperature predictions
- Microstructural data for multi-scale modeling
- TGA data for thermal decomposition kinetics

#### For Engineers
- Mix design guidance for rubberized concrete
- Property trade-off analysis (strength vs. ductility)
- Durability assessment data
- Specification development support

#### For Students
- Comprehensive dataset for concrete technology education
- Example of systematic experimental design
- Statistical analysis practice
- Material characterization methodology

### Known Limitations
- Ambient temperature only (high-temperature data not included)
- Single w/c ratio (0.45) - other ratios not explored
- One rubber source (truck tires) - variability not captured
- Laboratory-scale specimens - field-scale behavior may differ

### Future Work (Not in v1.0)
- [ ] High-temperature mechanical properties (100-800°C)
- [ ] Fire exposure testing (ASTM E119, ISO 834)
- [ ] Residual strength after fire
- [ ] Temperature-dependent thermal properties
- [ ] Microstructure after fire exposure
- [ ] Long-term durability (>90 days)
- [ ] Additional w/c ratios and rubber sources
- [ ] Full-scale structural element testing

---

## Version Numbering

**Format:** MAJOR.MINOR.PATCH

- **MAJOR:** Incompatible dataset structure changes
- **MINOR:** New data files or properties added (backwards compatible)
- **PATCH:** Data corrections, documentation updates

---

## Data Versioning Policy

1. **Corrections:** If errors are found in data, they will be corrected in patch releases with full documentation of changes.

2. **Additions:** New test data (e.g., high-temperature properties) will be added as minor version increments.

3. **Deprecation:** Deprecated data files will be marked in README and retained for at least one major version.

4. **Reproducibility:** Each release is tagged and archived to ensure reproducibility of analyses.

---

## How to Cite Different Versions

**Current Version (1.0.0):**
> Baseline Rubberized Concrete Dataset v1.0.0 (2025-10-17). Pillar 1: Material Characterization & Mixture Design. Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete.

**General Citation (version-independent):**
> Baseline Rubberized Concrete Dataset. Pillar 1: Material Characterization & Mixture Design. Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete. https://github.com/[repository]

---

## Contact for Dataset Issues

For questions, corrections, or suggestions regarding this dataset:
- Open an issue in the repository
- Email: [research contact - to be added]
- Include dataset version number and specific file/variable in question

---

**Maintained by:** Rubberized Concrete Fire Resistance Research Project  
**Last Updated:** 2025-10-17  
**Status:** ✅ Active Development
