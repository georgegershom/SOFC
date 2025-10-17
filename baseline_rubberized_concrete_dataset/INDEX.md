# Dataset Index
## Baseline Rubberized Concrete Dataset v1.0.0

**Quick Navigation Guide**

---

## 🎯 Start Here Based on Your Role

### 👨‍🔬 Researcher / Scientist
1. **EXECUTIVE_SUMMARY.md** - High-level findings (15 min read)
2. **README.md** - Complete documentation (1-2 hour read)
3. **15_property_correlations.csv** - Summary data table
4. **analysis_script.py** - Generate visualizations

### 👷 Structural Engineer
1. **QUICK_START.md** - Fast introduction (5 min read)
2. **15_property_correlations.csv** - Key properties summary
3. **8_compressive_strength.csv** - Design strength values
4. **10_modulus_of_elasticity.csv** - Elastic properties

### 👨‍💻 Computational Modeler
1. **DATA_DICTIONARY.md** - All variable definitions
2. **1_mixture_proportions.csv** - Material compositions
3. **18_stress_strain_curves.csv** - Constitutive behavior
4. **16_thermal_properties_ambient.csv** - Thermal parameters

### 🎓 Student / Learner
1. **QUICK_START.md** - Introduction to dataset
2. **EXECUTIVE_SUMMARY.md** - Key findings explained
3. **analysis_script.py** - Example analysis code
4. **README.md** - Deep dive into methodology

---

## 📂 Complete File Listing

### 📊 Data Files (18 CSV Files)

#### Mixture Design & Materials (Files 1-6)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **1_mixture_proportions.csv** | Complete mix designs for 4 mixtures | 5 | Cement, aggregates, rubber, water, w/c ratio |
| **2_aggregate_grading.csv** | Particle size distributions | 11 | Sieve sizes, cumulative passing % |
| **3_rubber_characterization.csv** | Physical properties of rubber | 21 | Density, absorption, hardness, treatment |
| **4_rubber_chemical_composition.csv** | Chemical composition | 13 | Polymer, carbon black, additives |
| **5_rubber_thermal_analysis.csv** | TGA/DSC data (25-800°C) | 15 | Temperature, mass loss, heat flow |
| **6_rubber_ftir_peaks.csv** | FTIR spectroscopy | 17 | Wavenumber, functional groups |

#### Fresh State & Workability (File 7)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **7_fresh_state_properties.csv** | Fresh concrete properties | 8 | Slump, air content, density, setting time |

#### Mechanical Properties (Files 8-10, 18)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **8_compressive_strength.csv** | Compressive strength @ 7 & 28d | 41 | Strength (MPa), failure mode, COV |
| **9_tensile_splitting_strength.csv** | Tensile strength @ 28d | 21 | Split tensile strength, ductility |
| **10_modulus_of_elasticity.csv** | Elastic modulus @ 28d | 21 | Modulus (GPa), Poisson's ratio, strain |
| **18_stress_strain_curves.csv** | Complete σ-ε curves | 107 | Strain (με), stress (MPa), loading state |

#### Physical Properties (Files 11-13)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **11_density_porosity.csv** | Density and porosity | 21 | Dry/SSD density, porosity (%), pore diameter |
| **12_pore_size_distribution_MIP.csv** | Mercury intrusion porosimetry | 13 | Pore diameter (nm), intrusion volume |
| **13_ultrasonic_pulse_velocity.csv** | UPV measurements | 21 | UPV (km/s), dynamic modulus, quality rating |

#### Microstructure (File 14)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **14_microstructural_analysis.csv** | SEM & XRD analysis | 32 | ITZ thickness, porosity, crack density, phases |

#### Summary & Correlations (File 15)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **15_property_correlations.csv** | **📌 START HERE!** Summary table | 5 | All key properties in one table |

#### Thermal & Durability (Files 16-17)
| File | Description | Rows | Key Variables |
|------|-------------|------|---------------|
| **16_thermal_properties_ambient.csv** | Thermal properties @ 23°C | 5 | Conductivity, specific heat, diffusivity |
| **17_permeability_durability.csv** | Transport & durability | 5 | Permeability, chloride, carbonation, F-T |

---

### 📖 Documentation Files (7 Files)

| File | Purpose | Length | Read Time |
|------|---------|--------|-----------|
| **INDEX.md** | 📌 This file - Navigation guide | 5 pages | 5 min |
| **QUICK_START.md** | Fast introduction for new users | 12 pages | 5-10 min |
| **EXECUTIVE_SUMMARY.md** | High-level findings & applications | 18 pages | 15-20 min |
| **README.md** | Comprehensive documentation | 50+ pages | 1-2 hours |
| **DATA_DICTIONARY.md** | Variable definitions & units | 30 pages | Reference |
| **CHANGELOG.md** | Version history | 8 pages | 5 min |
| **CITATION.cff** | Citation format (machine-readable) | 1 page | 2 min |

---

### 🔧 Analysis Tools (2 Files)

| File | Purpose | Language | Lines of Code |
|------|---------|----------|---------------|
| **analysis_script.py** | Data analysis & visualization | Python 3 | ~400 |
| **requirements.txt** | Python dependencies | - | 6 packages |

**Generates:**
- `mechanical_properties.png` - Property trends
- `stress_strain_curves.png` - Stress-strain behavior
- `rubber_tga.png` - Thermal decomposition
- `pore_distribution.png` - Pore structure
- `analysis_summary.txt` - Text report

---

## 🗺️ Reading Pathways

### Fast Track (30 minutes)
1. **INDEX.md** (this file) - 5 min
2. **EXECUTIVE_SUMMARY.md** - 15 min
3. **15_property_correlations.csv** - Open in Excel - 5 min
4. **Browse 2-3 other CSV files** of interest - 5 min

### Standard Track (2 hours)
1. **QUICK_START.md** - 10 min
2. **EXECUTIVE_SUMMARY.md** - 20 min
3. **README.md** - Full read - 60 min
4. **Run analysis_script.py** - 10 min
5. **Explore CSV files** based on research needs - 20 min

### Deep Dive (1 day)
1. Read all documentation (QUICK_START → EXECUTIVE_SUMMARY → README → DATA_DICTIONARY)
2. Run and modify analysis_script.py
3. Systematically review all 18 CSV files
4. Cross-reference data between files
5. Conduct custom analyses for your application

---

## 📋 Data File Categories

### By Property Type

**🔨 Mechanical Properties**
- 8_compressive_strength.csv
- 9_tensile_splitting_strength.csv
- 10_modulus_of_elasticity.csv
- 18_stress_strain_curves.csv

**⚙️ Physical Properties**
- 11_density_porosity.csv
- 12_pore_size_distribution_MIP.csv
- 13_ultrasonic_pulse_velocity.csv

**🔬 Microstructure**
- 14_microstructural_analysis.csv

**🌡️ Thermal Properties**
- 5_rubber_thermal_analysis.csv (rubber only)
- 6_rubber_ftir_peaks.csv (rubber only)
- 16_thermal_properties_ambient.csv (concrete)

**💧 Durability**
- 17_permeability_durability.csv

**🏗️ Mix Design & Fresh State**
- 1_mixture_proportions.csv
- 2_aggregate_grading.csv
- 7_fresh_state_properties.csv

**♻️ Rubber Characterization**
- 3_rubber_characterization.csv
- 4_rubber_chemical_composition.csv
- 5_rubber_thermal_analysis.csv
- 6_rubber_ftir_peaks.csv

---

## 🎯 Use Case → File Mapping

### For Structural Design
**Primary:**
- 8_compressive_strength.csv (design strength)
- 10_modulus_of_elasticity.csv (stiffness, Poisson's ratio)
- 11_density_porosity.csv (self-weight)

**Secondary:**
- 9_tensile_splitting_strength.csv (cracking)
- 18_stress_strain_curves.csv (ductility)

### For Fire Resistance Modeling
**Primary:**
- 5_rubber_thermal_analysis.csv (decomposition)
- 16_thermal_properties_ambient.csv (baseline thermal)
- 11_density_porosity.csv (initial porosity)
- 12_pore_size_distribution_MIP.csv (pore structure)

**Secondary:**
- 18_stress_strain_curves.csv (constitutive model)
- 14_microstructural_analysis.csv (ITZ effects)

### For Durability Assessment
**Primary:**
- 17_permeability_durability.csv (all durability indicators)
- 11_density_porosity.csv (porosity)
- 12_pore_size_distribution_MIP.csv (pore structure)

**Secondary:**
- 14_microstructural_analysis.csv (microstructure)

### For Mix Design Optimization
**Primary:**
- 1_mixture_proportions.csv (current designs)
- 7_fresh_state_properties.csv (workability)
- 15_property_correlations.csv (performance summary)

**Secondary:**
- 3_rubber_characterization.csv (rubber properties)
- 8_compressive_strength.csv (strength results)

### For Thermo-Mechanical FEA
**Required Inputs:**
- 1_mixture_proportions.csv (material composition)
- 10_modulus_of_elasticity.csv (E, ν)
- 11_density_porosity.csv (ρ)
- 16_thermal_properties_ambient.csv (k, c_p, α)
- 18_stress_strain_curves.csv (σ-ε relationship)

**For Validation:**
- 8_compressive_strength.csv (target strength)
- 13_ultrasonic_pulse_velocity.csv (UPV for E_dynamic check)

### For Microstructure Research
**Primary:**
- 14_microstructural_analysis.csv (SEM/XRD results)
- 12_pore_size_distribution_MIP.csv (pore structure)

**Secondary:**
- 3_rubber_characterization.csv (rubber properties)
- 6_rubber_ftir_peaks.csv (chemical composition)

---

## 📊 Dataset Statistics

**Total Files:** 27 (18 CSV + 7 Docs + 2 Tools)  
**Total Data Points:** >2,000 individual measurements  
**Total Lines of Data:** 2,673 (all text files combined)  
**Total Specimens Tested:** >100 specimens  
**Test Standards Used:** 15+ ASTM/BS/ISO standards  
**Curing Duration:** 28 days (all mechanical tests)  
**Temperature Range:** 23°C (ambient baseline)  

---

## 🔍 Finding Specific Information

### Looking for...

**Mixture compositions?**
→ `1_mixture_proportions.csv`

**Compressive strength values?**
→ `8_compressive_strength.csv` or `15_property_correlations.csv`

**Rubber thermal decomposition?**
→ `5_rubber_thermal_analysis.csv`

**Stress-strain curves?**
→ `18_stress_strain_curves.csv`

**Porosity and pore structure?**
→ `11_density_porosity.csv` + `12_pore_size_distribution_MIP.csv`

**Microstructure details?**
→ `14_microstructural_analysis.csv`

**Quick summary of all properties?**
→ `15_property_correlations.csv` (best starting point!)

**Variable definitions and units?**
→ `DATA_DICTIONARY.md`

**Key research findings?**
→ `EXECUTIVE_SUMMARY.md`

**Test methods and standards?**
→ `README.md` (Section: Testing Standards & Methods)

**How to cite the dataset?**
→ `CITATION.cff` or `README.md` (Citation section)

---

## 🆘 Troubleshooting

### "I don't know where to start!"
→ Read this INDEX.md, then QUICK_START.md

### "I need a quick overview of results"
→ Open `15_property_correlations.csv` in Excel

### "I don't understand the variables"
→ Check `DATA_DICTIONARY.md` for all definitions

### "What units are used?"
→ All units listed in `DATA_DICTIONARY.md` or column headers

### "How do I run the analysis script?"
→ See QUICK_START.md, Step 3

### "Which files do I need for my application?"
→ See "Use Case → File Mapping" section above

### "I found an error in the data"
→ Check CHANGELOG.md for known issues, then report via [contact method]

### "Can I use this data commercially?"
→ See LICENSE information in README.md

---

## 📞 Support

**Documentation Issues:**
- Check README.md first (most comprehensive)
- Review DATA_DICTIONARY.md for variable questions
- See CHANGELOG.md for version updates

**Data Analysis Questions:**
- Review analysis_script.py with detailed comments
- Check EXECUTIVE_SUMMARY.md for interpretation guidance

**Research Collaboration:**
- Contact information in README.md

---

## ✅ Quick Checklist for New Users

Before diving into the data, make sure you've:

- [ ] Read this INDEX.md for navigation guidance
- [ ] Reviewed QUICK_START.md or EXECUTIVE_SUMMARY.md
- [ ] Opened `15_property_correlations.csv` for quick overview
- [ ] Identified which files you need (see Use Case mapping)
- [ ] Consulted DATA_DICTIONARY.md when encountering unknown variables
- [ ] Installed Python dependencies if using analysis script
- [ ] Noted the dataset version (1.0.0) for citation

---

## 🔗 File Dependencies

Some analyses require multiple files:

**Strength-Density Correlation:**
- 8_compressive_strength.csv
- 11_density_porosity.csv

**UPV-Modulus Validation:**
- 13_ultrasonic_pulse_velocity.csv
- 10_modulus_of_elasticity.csv

**Porosity-Permeability Relationship:**
- 11_density_porosity.csv
- 12_pore_size_distribution_MIP.csv
- 17_permeability_durability.csv

**Microstructure-Strength Correlation:**
- 14_microstructural_analysis.csv
- 8_compressive_strength.csv

---

## 📈 Data Visualization Files

**Generated by analysis_script.py:**

1. **mechanical_properties.png**
   - 6 subplots showing property trends vs. rubber content
   - Source: 15_property_correlations.csv

2. **stress_strain_curves.png**
   - Complete stress-strain behavior for all 4 mixes
   - Source: 18_stress_strain_curves.csv

3. **rubber_tga.png**
   - TGA and DSC curves showing thermal decomposition
   - Source: 5_rubber_thermal_analysis.csv

4. **pore_distribution.png**
   - Cumulative and incremental pore size distributions
   - Source: 12_pore_size_distribution_MIP.csv

5. **analysis_summary.txt**
   - Text-based summary report
   - Sources: Multiple CSV files

---

## 🎓 Educational Resources

This dataset is ideal for:

- **Concrete Technology Courses:** Material characterization methodology
- **Structural Engineering:** Property relationships and trade-offs
- **Fire Safety Engineering:** Baseline data for fire modeling
- **Sustainable Construction:** Waste material utilization
- **Data Science:** Real-world dataset for statistical analysis
- **Research Methods:** Systematic experimental design example

**Suggested Exercises:**
1. Correlation analysis between properties
2. Statistical analysis of test variability (COV)
3. Development of predictive equations
4. Visualization of trends and relationships
5. Critical analysis of strength vs. ductility trade-offs

---

## 🌟 Key Highlights

**Most Important Files (Start Here):**
1. **15_property_correlations.csv** - Summary of all properties
2. **EXECUTIVE_SUMMARY.md** - Key findings explained
3. **README.md** - Complete documentation

**Most Comprehensive Data:**
1. **8_compressive_strength.csv** - 40 specimens, 2 ages
2. **18_stress_strain_curves.csv** - Full stress-strain behavior
3. **12_pore_size_distribution_MIP.csv** - Detailed pore structure

**Most Critical for Fire Modeling:**
1. **5_rubber_thermal_analysis.csv** - Rubber decomposition
2. **16_thermal_properties_ambient.csv** - Baseline thermal
3. **12_pore_size_distribution_MIP.csv** - Pore structure evolution

---

**Last Updated:** 2025-10-17  
**Dataset Version:** 1.0.0  
**Total Files:** 27  
**Total Data Points:** >2,000  

---

*Navigate with confidence! This comprehensive index will help you find exactly what you need in the Baseline Rubberized Concrete Dataset.*
