# FDI Impact Study Dataset: Lagos Food Processing Firms

## 🎯 Project Overview

This repository contains a comprehensive dataset and analysis tools for studying **"The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy"**.

The dataset includes fabricated but realistic firm-level survey data from 300 food processing companies in Lagos, designed for academic research and statistical analysis.

## 📊 Dataset Contents

### Quick Stats
- **Sample Size**: 300 food processing firms
- **Variables**: 48 comprehensive variables
- **Geographic Coverage**: Lagos State, Nigeria
- **Industry Focus**: Food processing sector
- **Data Type**: Cross-sectional survey data (2024)

### Key Variable Categories
1. **FDI Constructs** (Likert 1-5)
   - Knowledge absorption capacity
   - Task performance measures
   - Innovation indicators

2. **Firm Performance Metrics**
   - ROI, ROA, export intensity
   - Market share, operational efficiency

3. **Firm Resources**
   - Human capital indicators
   - Technological resources
   - Financial resources

4. **Government Policy Perception** (Likert 1-7)
   - Tax incentives effectiveness
   - Regulatory stability
   - Infrastructure support
   - Corruption experience

5. **Control Variables**
   - Firm size, age, subsector
   - Ownership type, location
   - Export behavior

## 📁 Files in This Repository

| File | Description |
|------|-------------|
| `lagos_food_processing_fdi_dataset.csv` | Main dataset (300 firms × 48 variables) |
| `generate_fdi_dataset.py` | Python script to generate the dataset |
| `data_dictionary.md` | Comprehensive variable definitions and methodology |
| `README.md` | This file - project overview and usage guide |

## 🚀 Quick Start

### 1. Download the Dataset
```bash
# Clone or download the repository
git clone [repository-url]
cd fdi-lagos-dataset
```

### 2. Load the Data

#### Python (pandas)
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('lagos_food_processing_fdi_dataset.csv')

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"Variables: {list(df.columns)}")
print(df.head())
```

#### R
```r
# Load the dataset
df <- read.csv('lagos_food_processing_fdi_dataset.csv')

# Basic exploration
dim(df)
names(df)
head(df)
```

#### STATA
```stata
import delimited "lagos_food_processing_fdi_dataset.csv", clear
describe
summarize
```

### 3. Regenerate Dataset (Optional)
```bash
# Install dependencies
pip install pandas numpy

# Run generation script
python3 generate_fdi_dataset.py
```

## 📈 Sample Analysis Examples

### Basic Descriptive Statistics
```python
# Performance by ownership type
performance_by_ownership = df.groupby('ownership_type')[['roi_percent', 'roa_percent']].mean()
print(performance_by_ownership)

# FDI impact analysis
fdi_impact = df.groupby('has_fdi')[['roi_percent', 'operational_efficiency']].mean()
print(fdi_impact)
```

### Correlation Analysis
```python
# Key correlations
correlations = df[['roi_percent', 'knowledge_acquisition', 'policy_effectiveness_index', 'rd_spend_percent']].corr()
print(correlations)
```

### Regression Analysis Setup
```python
# Prepare variables for regression
X = df[['has_fdi', 'firm_age_years', 'employees', 'rd_spend_percent', 'policy_effectiveness_index']]
y = df['roi_percent']

# Add interaction term for moderation analysis
df['fdi_policy_interaction'] = df['has_fdi'] * df['policy_effectiveness_index']
```

## 🔬 Research Applications

This dataset is designed to support research on:

### Primary Research Questions
- **Main Effect**: How does FDI influence food processing firm performance?
- **Moderation Effect**: How do government policies moderate the FDI-performance relationship?
- **Mediating Mechanisms**: What role do absorptive capacity and innovation play?

### Suitable Analysis Methods
- **Descriptive Analysis**: Cross-tabulations, means comparison
- **Correlation Analysis**: Pearson/Spearman correlations
- **Regression Analysis**: OLS, hierarchical regression
- **Moderation Analysis**: Interaction effects testing
- **Mediation Analysis**: Path analysis, structural equation modeling

### Sample Hypotheses to Test
1. H1: FDI positively influences firm performance (ROI, ROA)
2. H2: Government policy effectiveness moderates the FDI-performance relationship
3. H3: Knowledge absorption mediates the FDI-performance relationship
4. H4: Firm resources moderate the FDI-innovation relationship

## 📊 Data Quality & Characteristics

### Strengths
✅ **Realistic Distributions**: Based on actual economic patterns  
✅ **Comprehensive Coverage**: All major FDI and performance constructs  
✅ **Appropriate Sample Size**: n=300 suitable for multivariate analysis  
✅ **No Missing Data**: Complete dataset ready for analysis  
✅ **Validated Scales**: Based on established academic instruments  

### Considerations
⚠️ **Fabricated Data**: Generated for research/educational purposes  
⚠️ **Cross-sectional**: No longitudinal/panel data  
⚠️ **Self-reported**: Performance measures based on survey responses  
⚠️ **Geographic Scope**: Limited to Lagos context  

## 🛠️ Technical Requirements

### Minimum Requirements
- **Software**: Any statistical software (R, Python, STATA, SPSS)
- **Memory**: Standard requirements for 300×48 dataset
- **Skills**: Basic statistical analysis knowledge

### Recommended Setup
```python
# Python environment
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
statsmodels >= 0.12.0
```

## 📚 Theoretical Framework

### Key Literature References
- **Absorptive Capacity**: Zahra & George (2002)
- **Task Performance**: Koopmans et al. (2013)
- **Innovation Measurement**: OECD Oslo Manual
- **FDI Theory**: Dunning's OLI Paradigm
- **Institutional Theory**: North (1990)

### Variable Operationalization
All variables follow established academic scales and measurement approaches. See `data_dictionary.md` for detailed operationalization and sources.

## 🤝 Contributing

### Reporting Issues
If you find any issues with the dataset or documentation:
1. Check existing documentation
2. Review the generation script
3. Open an issue with detailed description

### Enhancements
Suggestions for dataset improvements:
- Additional variables
- Alternative distributions
- Extended sample size
- Longitudinal extensions

## 📄 License & Usage

### Academic Use
✅ **Permitted**: Research, education, thesis work, academic publications  
✅ **Attribution**: Please cite this dataset in your research  
✅ **Modification**: Feel free to adapt for your research needs  

### Commercial Use
⚠️ **Restricted**: Contact for commercial applications  
⚠️ **Liability**: No warranty for business decisions based on this data  

### Citation Format
```
Lagos Food Processing FDI Dataset (2024). Generated dataset for research on 
"The Influence of Foreign Direct Investment on the Performance of Food Processing 
Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy". 
[Version 1.0]
```

## 📞 Support & Contact

### Documentation
- **Data Dictionary**: See `data_dictionary.md` for complete variable definitions
- **Generation Script**: Review `generate_fdi_dataset.py` for methodology
- **This README**: Comprehensive usage guide

### Getting Help
1. **First**: Check the data dictionary and generation script
2. **Second**: Review similar academic datasets and methodologies  
3. **Third**: Consult statistical analysis resources for your software

---

## 🎉 Happy Analyzing!

This dataset provides a solid foundation for exploring FDI impacts on firm performance in the Nigerian context. Whether you're conducting academic research, learning statistical methods, or exploring policy implications, this comprehensive dataset offers rich opportunities for analysis.

**Remember**: While this is fabricated data, it's designed to reflect realistic patterns and relationships that you might find in actual firm-level data from Lagos food processing companies.

---

*Generated: October 2024 | Version: 1.0 | Format: CSV*