# FDI and Food Processing Firms Dataset - Lagos, Nigeria

## 📋 Overview

This dataset was generated for research on **"The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy"**.

**Dataset Size:** 300 firms (200 SMEs, 100 Large firms)  
**Industry:** Food Processing (NACE Code 10)  
**Location:** Lagos, Nigeria  
**Time Frame:** Cross-sectional data (2024)  
**Variables:** 57 comprehensive variables covering FDI, performance, resources, and policy perceptions

---

## 📁 Files Included

### 1. **fdi_food_processing_firms_dataset.csv**
   - Main dataset with all 300 firm records
   - 57 variables across multiple categories
   - Ready for statistical analysis

### 2. **fdi_food_processing_firms_complete.xlsx**
   - Excel workbook with three sheets:
     - **Main Dataset:** All firm records
     - **Data Dictionary:** Variable definitions and metadata
     - **Summary Statistics:** Key descriptive statistics

### 3. **data_dictionary.csv**
   - Comprehensive variable documentation
   - Includes: Variable name, description, type, scale, source, and reference
   - Essential for understanding measurement methods

### 4. **summary_statistics.csv**
   - Quick overview of key metrics
   - Sample composition and averages

### 5. **generate_fdi_dataset.py**
   - Python script used to generate the dataset
   - Reproducible with random seed (42)
   - Can be modified to generate alternative scenarios

---

## 🎯 Variable Categories

### A. **Identifiers & Firm Characteristics** (10 variables)
- FirmID, Firm Name, Subsector, Location
- Firm Size, Age, Employees, Total Assets
- Ownership Type, Year Established

### B. **FDI Variables** (5 variables)
- FDI Presence (binary)
- FDI Type (Greenfield, M&A, Joint Venture)
- FDI Origin Country
- FDI Ownership Percentage
- Years with FDI

### C. **FDI Constructs - Primary Survey Data** (4 variables)
Based on established scales:
- **Knowledge Absorption** (1-5 Likert) - Zahra & George (2002)
- **Task Performance** (1-5 Likert) - Koopmans et al. (2013)
- **Innovation Score** (1-5 Likert) - OECD Oslo Manual
- **Firm Resources Score** (1-5 Likert) - Composite measure

### D. **Detailed Firm Resources** (7 variables)
- **Human Capital:** Skilled labor ratio, training hours
- **Technological:** IoT adoption, R&D spend ratio, automation level
- **Financial:** Liquidity ratio, debt-equity ratio

### E. **Firm Performance Indicators** (8 variables)
- ROI and ROA (%)
- Revenue (USD) and growth rate
- Export intensity
- Market share
- Operational efficiency (1-10 scale)
- Employee productivity

### F. **Government Policy Perception** (8 variables)
Survey-based, 1-7 Likert scales:
- Tax incentives effectiveness
- Regulatory stability
- Infrastructure support
- Corruption experience
- Bureaucratic efficiency
- Trade policy support
- **Composite Government Policy Score**
- **Policy Effectiveness Index** (0-100)

### G. **Government Programs & Support** (2 variables)
- EEG (Employment and Empowerment Grant) beneficiary
- Pioneer tax status

### H. **Infrastructure & Institutional Factors** (3 variables)
- Power access hours (daily)
- Generator reliance
- Distance to port (km)

### I. **Certifications & Compliance** (3 variables)
- ISO certification
- NAFDAC compliance
- Halal certification

### J. **Macro-Level Secondary Data** (5 variables)
Consistent across firms:
- FDI Inflow to Lagos (Billion USD)
- Corruption Perception Index
- Ease of Doing Business Rank
- Average Power Reliability
- Logistics Quality Index

### K. **Interaction Terms** (2 variables)
For moderation analysis:
- FDI × Policy Interaction
- Innovation × Policy Interaction

---

## 📊 Sample Composition

| Metric | Value |
|--------|-------|
| **Total Firms** | 300 |
| SME Firms | 200 (66.7%) |
| Large Firms | 100 (33.3%) |
| **Firms with FDI** | ~167 (55.7%) |
| **Subsectors** | 12 food processing categories |
| **Lagos Zones** | 6 zones covered |

### Key Averages
- Average Firm Age: ~21 years
- Average Employees: ~439
- Average ROI: 15.2%
- Average ROA: 11.9%
- Average Innovation Score: 3.28/5
- Average Government Policy Score: 3.61/7
- Average Export Intensity: 16.4%

---

## 🔬 Recommended Analysis Methods

### 1. **Structural Equation Modeling (SEM)**
   - Test mediation and moderation effects
   - Examine relationships between FDI constructs and performance
   - Assess government policy as moderator

### 2. **Hierarchical Regression**
   - Control for firm size, age, subsector
   - Add FDI variables
   - Test policy moderation interaction terms

### 3. **Difference-in-Differences (if panel extension)**
   - Compare FDI vs. non-FDI firms
   - Before/after policy changes

### 4. **Propensity Score Matching**
   - Match FDI and non-FDI firms on observables
   - Reduce selection bias

---

## 🛠️ Software Compatibility

### STATA
```stata
import delimited "fdi_food_processing_firms_dataset.csv", clear
describe
summarize
```

### R
```r
library(tidyverse)
df <- read_csv("fdi_food_processing_firms_dataset.csv")
summary(df)
```

### SPSS
1. File → Import Data → CSV
2. Select `fdi_food_processing_firms_dataset.csv`
3. Run: `DESCRIPTIVES VARIABLES=ALL.`

### Python
```python
import pandas as pd
df = pd.read_csv('fdi_food_processing_firms_dataset.csv')
df.info()
df.describe()
```

---

## 📐 Variable Scaling & Interpretation

### Likert Scales
- **1-5 Scale:** FDI constructs (knowledge, task performance, innovation, resources)
  - 1 = Strongly Disagree/Very Low
  - 5 = Strongly Agree/Very High

- **1-7 Scale:** Government policy perceptions
  - 1 = Very Ineffective/Very Poor
  - 7 = Very Effective/Excellent

### Binary Variables (0/1)
- FDI Presence, IoT Adoption, Certifications, Government Programs
- 0 = No/Absent, 1 = Yes/Present

### Continuous Variables
- Financial ratios, percentages, absolute values (revenue, assets, employees)

---

## 🔍 Data Quality Features

### Realistic Correlations
- FDI presence positively correlated with:
  - Innovation scores
  - Knowledge absorption
  - Task performance
  - Export intensity
  - Employee productivity

### Logical Constraints
- SME firms: 10-250 employees
- Large firms: 250-2,000 employees
- Performance metrics aligned with firm size
- Policy scores show natural variation

### Contextual Accuracy
- Subsectors match Nigerian food processing industry
- Lagos zones reflect actual industrial areas
- FDI origin countries reflect actual investment patterns
- Infrastructure challenges (power, logistics) incorporated

---

## 📖 Theoretical Foundations

### FDI Constructs Based On:
1. **Zahra & George (2002)** - Absorptive Capacity Theory
2. **Koopmans et al. (2013)** - Task Performance Framework
3. **OECD Oslo Manual** - Innovation Measurement Guidelines
4. **Resource-Based View (RBV)** - Firm resources and capabilities

### Policy Moderation Framework:
- Institutional Theory
- Triple Helix Model (Government-Industry-Innovation)
- Nigerian Industrial Policy context

---

## ⚠️ Ethical Considerations

### Data Anonymization
- Firm IDs are randomly generated (FIRM_001 to FIRM_300)
- Firm names are generic placeholders
- No personally identifiable information included

### Consent & IRB
This is **synthetic data** generated for research purposes:
- No actual firms were surveyed
- Data patterns based on literature and secondary sources
- Suitable for methodological development and pilot analysis

### Limitations
- **Cross-sectional:** Cannot establish causality without panel extension
- **Synthetic:** Patterns are modeled, not observed
- **Lagos-specific:** Results may not generalize to other Nigerian states

---

## 🚀 Getting Started

### Quick Start (5 minutes)
1. Open `fdi_food_processing_firms_complete.xlsx`
2. Review "Summary Statistics" sheet
3. Explore "Main Dataset" sheet
4. Check "Data Dictionary" for variable definitions

### For Analysis (15 minutes)
1. Import `fdi_food_processing_firms_dataset.csv` into your preferred software
2. Run descriptive statistics
3. Check for outliers and distributions
4. Test correlations between key variables
5. Begin regression or SEM modeling

---

## 📚 Citation & Usage

### Suggested Citation:
```
FDI and Food Processing Firms Dataset (2024). Synthetic dataset for research on 
"The Influence of Foreign Direct Investment on the Performance of Food Processing 
Firms in Lagos, Nigeria." Generated October 2024.
```

### License
This dataset is provided for academic and research purposes. Feel free to:
- Use in thesis, dissertations, or academic papers
- Modify variables or sample size using the Python script
- Share with collaborators

---

## 🆘 Support & Modifications

### Need to Modify the Dataset?
Edit `generate_fdi_dataset.py` to:
- Change sample size (default: 300 firms)
- Adjust SME/Large ratio (default: 200/100)
- Modify variable distributions
- Add new variables
- Change correlation patterns
- Generate panel data (multiple years)

### Re-generate Dataset
```bash
python3 generate_fdi_dataset.py
```

---

## 📞 Questions?

For questions about:
- **Variable definitions:** See `data_dictionary.csv`
- **Methodological approach:** See theoretical foundations section above
- **Data generation logic:** Review `generate_fdi_dataset.py` script
- **Analysis recommendations:** See software compatibility section

---

## ✅ Quality Checklist

Before using this dataset for analysis, verify:
- [ ] All 300 records loaded successfully
- [ ] 57 variables present
- [ ] No missing values in key variables
- [ ] Distributions appear reasonable (check histograms)
- [ ] Correlations make theoretical sense
- [ ] Interaction terms calculated correctly
- [ ] Scale interpretations understood (1-5 vs 1-7)

---

**Generated:** October 13, 2025  
**Version:** 1.0  
**Format:** CSV & Excel  
**Encoding:** UTF-8

---

## 🎓 Research Applications

This dataset is suitable for:
- ✅ Master's thesis (quantitative methods)
- ✅ PhD dissertation chapters
- ✅ Course assignments (econometrics, international business)
- ✅ Methodological papers
- ✅ Policy analysis simulations
- ✅ Teaching statistical methods
- ✅ SEM and regression workshops

**Good luck with your research! 🚀**
