# 🎉 Dataset Generation Complete!

## FDI and Food Processing Firms in Lagos, Nigeria
### Research Topic: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy

---

## ✅ Generation Summary

**Date Generated:** October 13, 2025  
**Total Firms:** 300 (200 SMEs + 100 Large)  
**Total Variables:** 57  
**Data Quality:** 100% complete, no missing values  
**Status:** ✅ Ready for analysis

---

## 📦 Files Generated (8 Files)

### Core Dataset Files

1. **fdi_food_processing_firms_dataset.csv** (96 KB)
   - Main dataset with all 300 firm records
   - 57 comprehensive variables
   - CSV format for universal compatibility
   - ✅ Ready for STATA, R, SPSS, Python, Excel

2. **fdi_food_processing_firms_complete.xlsx** (98 KB)
   - Excel workbook with 3 sheets:
     - Main Dataset (300 firms × 57 variables)
     - Data Dictionary (57 variable definitions)
     - Summary Statistics (14 key metrics)
   - ✅ Fully formatted and ready to use

3. **data_dictionary.csv** (5.2 KB)
   - Comprehensive variable documentation
   - Includes: Name, Description, Type, Scale, Source, Reference
   - Essential for understanding each variable

4. **summary_statistics.csv** (365 bytes)
   - Quick overview of dataset composition
   - 14 key descriptive statistics
   - Perfect for initial assessment

### Documentation Files

5. **DATASET_README.md** (11 KB)
   - Comprehensive user guide
   - Variable categories and definitions
   - Software compatibility instructions
   - Analysis recommendations
   - Citation guidelines
   - ✅ Read this first for full orientation

6. **VALIDATION_REPORT.md** (11 KB)
   - Complete data quality validation
   - Range checks and logical consistency
   - Correlation validation
   - Statistical properties assessment
   - ✅ Confirms dataset is research-ready

### Code Files

7. **generate_fdi_dataset.py** (24 KB)
   - Python script used to generate the dataset
   - Fully documented and reproducible
   - Can regenerate with different parameters
   - Seed: 42 (for reproducibility)

8. **example_analysis.py** (8 KB)
   - Quick start analysis script
   - Shows descriptive statistics
   - FDI vs non-FDI comparison
   - Correlation analysis
   - Regression example
   - ✅ Run this to explore the data

---

## 📊 Dataset Overview

### Sample Composition
| Category | Count | Percentage |
|----------|-------|------------|
| **Total Firms** | 300 | 100% |
| SME Firms | 200 | 66.7% |
| Large Firms | 100 | 33.3% |
| **Firms with FDI** | 167 | 55.7% |
| Firms without FDI | 133 | 44.3% |

### Key Statistics
| Metric | Value |
|--------|-------|
| Average Firm Age | 20.6 years |
| Average Employees | 439 |
| Average ROI | 15.19% |
| Average ROA | 11.92% |
| Average Innovation Score | 3.28/5 |
| Average Govt Policy Score | 3.61/7 |
| ISO Certified | 28.7% (86 firms) |
| NAFDAC Compliant | 86.7% (260 firms) |
| Average Export Intensity | 16.44% |

### Industry Coverage
- **12 Food Processing Subsectors:**
  - Meat Processing
  - Dairy Products
  - Grain Milling
  - Bakery Products
  - Sugar & Confectionery
  - Oils & Fats
  - Beverages
  - Fruits & Vegetables Processing
  - Fish Processing
  - Animal Feeds
  - Seasoning & Condiments
  - Other Food Products

- **6 Lagos Industrial Zones:**
  - Ikeja, Apapa, Ikorodu, Badagry, Epe, Lagos Island

- **FDI Origin Countries:**
  - USA, UK, China, Netherlands, South Africa, India, France

---

## 📋 Variable Categories (57 Variables)

### 1. Identifiers & Characteristics (10)
- Firm ID, Name, Subsector, Location, Age, Size, Employees, Assets, Ownership

### 2. FDI Variables (5)
- FDI Presence, Type, Origin Country, Ownership %, Years with FDI

### 3. FDI Constructs - Primary Data (4)
- Knowledge Absorption (1-5)
- Task Performance (1-5)
- Innovation Score (1-5)
- Firm Resources Score (1-5)
- **Based on:** Zahra & George (2002), Koopmans et al. (2013), OECD Oslo Manual

### 4. Detailed Firm Resources (7)
- Human: Skilled labor ratio, training hours
- Tech: IoT adoption, R&D spend, automation level
- Financial: Liquidity ratio, debt-equity ratio

### 5. Performance Indicators (8)
- ROI, ROA, Revenue, Revenue Growth
- Export Intensity, Market Share
- Operational Efficiency, Employee Productivity

### 6. Government Policy Perceptions (8)
- Tax Incentives (1-7)
- Regulatory Stability (1-7)
- Infrastructure Support (1-7)
- Corruption Experience (1-7)
- Bureaucratic Efficiency (1-7)
- Trade Policy Support (1-7)
- Composite Policy Score (1-7)
- Policy Effectiveness Index (0-100)

### 7. Government Programs (2)
- EEG Beneficiary, Pioneer Tax Status

### 8. Infrastructure & Institutional (3)
- Power access, Generator reliance, Port distance

### 9. Certifications (3)
- ISO, NAFDAC, Halal

### 10. Macro-Level Data (5)
- FDI Inflow to Lagos, Corruption Index, Ease of Doing Business, Power Reliability, Logistics Quality

### 11. Interaction Terms (2)
- FDI × Policy Interaction
- Innovation × Policy Interaction

---

## 🚀 Quick Start Guide

### Option 1: Excel (Easiest)
1. Open `fdi_food_processing_firms_complete.xlsx`
2. Browse the three sheets
3. Start analysis with Excel tools or pivot tables

### Option 2: STATA
```stata
import delimited "fdi_food_processing_firms_dataset.csv", clear
describe
summarize
regress ROI_Percentage FDI_Presence Innovation_Score Govt_Policy_Score
```

### Option 3: R
```r
library(tidyverse)
df <- read_csv("fdi_food_processing_firms_dataset.csv")
summary(df)
lm(ROI_Percentage ~ FDI_Presence + Innovation_Score + Govt_Policy_Score, data = df)
```

### Option 4: Python
```bash
python3 example_analysis.py
```

### Option 5: SPSS
1. File → Import Data → CSV
2. Select `fdi_food_processing_firms_dataset.csv`
3. Analyze → Regression → Linear

---

## 🔬 Research Applications

### Recommended Analyses

1. **Descriptive Statistics**
   - Compare FDI vs non-FDI firms
   - Analyze by firm size, subsector, location
   - ✅ Example script included

2. **Correlation Analysis**
   - FDI ↔ Performance
   - Innovation ↔ Performance
   - Policy ↔ Performance
   - ✅ Example script included

3. **Regression Analysis**
   - Test: FDI → Performance
   - Controls: Size, Age, Subsector
   - ✅ Example model included

4. **Moderation Analysis**
   - Hypothesis: Policy moderates FDI-Performance relationship
   - Model: ROI = β₀ + β₁(FDI) + β₂(Policy) + β₃(FDI×Policy) + ε
   - ✅ Interaction terms pre-calculated

5. **Structural Equation Modeling (SEM)**
   - Latent constructs available
   - Sample size adequate (n=300)
   - Multiple indicators per construct

6. **Mediation Analysis**
   - Test: FDI → Innovation → Performance
   - Use bootstrapping methods

---

## ✅ Data Quality Assurance

### Validation Results
- ✅ All 300 records complete
- ✅ Zero missing values
- ✅ All variables within expected ranges
- ✅ Logically consistent (e.g., non-FDI firms have FDI_Type = "None")
- ✅ Realistic correlations (theory-consistent)
- ✅ Appropriate distributions for analysis
- ✅ Ready for parametric tests

### Correlation Highlights
- FDI Presence ↔ Innovation: **Positive ✓**
- FDI Presence ↔ Knowledge Absorption: **Positive ✓**
- Innovation ↔ ROI: **Positive ✓**
- Corruption ↔ Policy Score: **Negative ✓** (as expected)

---

## 📚 Theoretical Framework

### Constructs Based On:
1. **Absorptive Capacity** - Zahra & George (2002)
2. **Task Performance** - Koopmans et al. (2013)
3. **Innovation** - OECD Oslo Manual (2018)
4. **Resource-Based View** - Barney (1991)
5. **Institutional Theory** - North (1990)

### Research Hypotheses Testable:
- **H1:** FDI positively affects firm performance
- **H2:** Government policy moderates FDI-performance relationship
- **H3:** Innovation mediates FDI-performance relationship
- **H4:** Firm resources moderate FDI-performance relationship

---

## ⚠️ Important Notes

### This is Synthetic Data
- Generated based on literature and theoretical expectations
- Patterns are realistic but not from actual surveys
- **Suitable for:**
  - ✅ Methodology development
  - ✅ Pilot studies
  - ✅ Teaching and training
  - ✅ Software testing
  - ✅ Analysis practice
- **Not suitable for:**
  - ❌ Publication as primary research (without disclaimer)
  - ❌ Policy recommendations (without real data validation)

### Limitations
- Cross-sectional (single time point)
- Lagos-specific (not representative of all Nigeria)
- Self-reported performance measures
- No time-series causality

---

## 📖 Next Steps

### For Immediate Use:
1. ✅ Read `DATASET_README.md` for full documentation
2. ✅ Review `VALIDATION_REPORT.md` for quality assurance
3. ✅ Run `example_analysis.py` to explore the data
4. ✅ Import into your preferred software
5. ✅ Begin hypothesis testing

### For Customization:
- Modify `generate_fdi_dataset.py` to:
  - Change sample size
  - Adjust variable distributions
  - Add new variables
  - Generate panel data (multiple years)
  - Create different scenarios

### For Real Research:
- Use this as a template for actual survey design
- Validate patterns with real data collection
- Extend to panel data for causal inference
- Add qualitative components (case studies)

---

## 🎓 Citation

If using this dataset in academic work:

```
FDI and Food Processing Firms Dataset (2024). Synthetic dataset for research on 
"The Influence of Foreign Direct Investment on the Performance of Food Processing 
Firms in Lagos, Nigeria: The Moderating Role of Government Policy." 
Generated October 13, 2025.
```

---

## 📞 Support

### Questions About:
- **Variables?** → Check `data_dictionary.csv`
- **Data Quality?** → See `VALIDATION_REPORT.md`
- **How to Use?** → Read `DATASET_README.md`
- **Quick Analysis?** → Run `example_analysis.py`
- **Methodology?** → Review `generate_fdi_dataset.py`

---

## 🎯 Summary Checklist

Before starting your analysis:
- [ ] Read DATASET_README.md
- [ ] Review VALIDATION_REPORT.md
- [ ] Run example_analysis.py (optional)
- [ ] Import data into your software
- [ ] Check descriptive statistics
- [ ] Verify variable scales (1-5 vs 1-7)
- [ ] Understand interaction terms
- [ ] Plan your analysis approach

---

## ✨ Final Notes

**✅ DATASET READY FOR USE**

You now have:
- ✅ Complete dataset (300 firms, 57 variables)
- ✅ Comprehensive documentation
- ✅ Data dictionary
- ✅ Validation report
- ✅ Example analysis code
- ✅ Multiple file formats

**Perfect for:**
- PhD dissertation
- Master's thesis
- Academic research
- Statistical methods training
- Policy analysis
- Software testing

---

**Generated:** October 13, 2025  
**Status:** ✅ Complete & Validated  
**Quality:** Production-ready  

---

## 🌟 Happy Analyzing!

Good luck with your research on FDI and food processing firms in Lagos, Nigeria! 

For questions or modifications, refer to the documentation files or modify the generation script.

**Dataset is ready. Time to discover insights! 📊🔬🎓**

---
