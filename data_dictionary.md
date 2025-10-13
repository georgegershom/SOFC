# Data Dictionary: Lagos Food Processing Firms FDI Study

## Dataset Overview
**Title**: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy

**Description**: This dataset contains fabricated but realistic firm-level survey data from 300 food processing companies in Lagos, Nigeria. The data is designed for research on FDI impact on firm performance with government policy as a moderating factor.

**Sample Size**: 300 firms  
**Variables**: 48 variables  
**Data Collection Period**: Simulated 2024 survey  
**Geographic Coverage**: Lagos State, Nigeria  

---

## Variable Categories and Definitions

### 1. Firm Identification
| Variable | Type | Description | Range/Values |
|----------|------|-------------|--------------|
| `firm_id` | String | Unique firm identifier | FPF0001 - FPF0300 |

### 2. FDI Constructs (Likert Scale 1-5)
*Based on Zahra & George (2002), Koopmans et al. (2013), OECD Oslo Manual*

#### Knowledge Absorption Capacity
| Variable | Description | Scale |
|----------|-------------|-------|
| `knowledge_acquisition` | Firm's ability to identify and acquire external knowledge | 1=Very Poor, 5=Excellent |
| `knowledge_assimilation` | Firm's ability to analyze and understand acquired knowledge | 1=Very Poor, 5=Excellent |
| `knowledge_transformation` | Firm's ability to develop and refine knowledge | 1=Very Poor, 5=Excellent |
| `knowledge_exploitation` | Firm's ability to implement and commercialize knowledge | 1=Very Poor, 5=Excellent |

#### Task Performance (Koopmans et al., 2013)
| Variable | Description | Scale |
|----------|-------------|-------|
| `task_proficiency` | Quality of core job tasks execution | 1=Very Poor, 5=Excellent |
| `task_adaptability` | Ability to adapt to changes in work environment | 1=Very Poor, 5=Excellent |
| `task_proactivity` | Initiative in improving work processes | 1=Very Poor, 5=Excellent |
| `contextual_performance` | Supporting organizational environment | 1=Very Poor, 5=Excellent |
| `counterproductive_behavior` | Frequency of behaviors that harm organization | 1=Never, 5=Very Frequently (Reverse scored) |

#### Innovation Capacity (OECD Oslo Manual)
| Variable | Description | Scale |
|----------|-------------|-------|
| `product_innovation` | Introduction of new or improved products | 1=Very Low, 5=Very High |
| `process_innovation` | Implementation of new production methods | 1=Very Low, 5=Very High |
| `organizational_innovation` | New organizational methods implementation | 1=Very Low, 5=Very High |
| `marketing_innovation` | New marketing methods adoption | 1=Very Low, 5=Very High |

### 3. Firm Performance Metrics

| Variable | Type | Description | Unit | Range |
|----------|------|-------------|------|-------|
| `roi_percent` | Continuous | Return on Investment | Percentage | -15% to 45% |
| `roa_percent` | Continuous | Return on Assets | Percentage | -10% to 25% |
| `export_intensity_percent` | Continuous | Revenue from exports as % of total revenue | Percentage | 0% to 80% |
| `market_share_percent` | Continuous | Market share in subsector | Percentage | 0.1% to 35% |
| `operational_efficiency` | Ordinal | Overall operational efficiency rating | 1-5 Scale | 1=Very Poor, 5=Excellent |

### 4. Firm Resources

#### Human Capital
| Variable | Type | Description | Unit/Scale |
|----------|------|-------------|-----------|
| `skilled_labor_ratio` | Continuous | Percentage of skilled workers | 0-100% |
| `training_investment` | Ordinal | Investment in employee training | 1=Very Low, 5=Very High |
| `management_quality` | Ordinal | Quality of management practices | 1=Very Poor, 5=Excellent |

#### Technological Resources
| Variable | Type | Description | Unit/Scale |
|----------|------|-------------|-----------|
| `iot_adoption` | Binary | Internet of Things technology adoption | 0=No, 1=Yes |
| `rd_spend_percent` | Continuous | R&D expenditure as % of revenue | 0-15% |
| `technology_sophistication` | Ordinal | Level of technology sophistication | 1=Very Basic, 5=Very Advanced |
| `digital_infrastructure` | Ordinal | Quality of digital infrastructure | 1=Very Poor, 5=Excellent |

#### Financial Resources
| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| `liquidity_ratio` | Continuous | Current assets / Current liabilities | Ratio (0.5-5.0) |
| `debt_equity_ratio` | Continuous | Total debt / Total equity | Ratio (0.1-3.0) |
| `financial_flexibility` | Ordinal | Access to financial resources | 1=Very Limited, 5=Very Flexible |

### 5. Government Policy Perception (Likert Scale 1-7)

#### Tax Incentives
| Variable | Description | Scale |
|----------|-------------|-------|
| `tax_incentive_effectiveness` | Effectiveness of government tax incentives | 1=Very Ineffective, 7=Very Effective |
| `tax_incentive_accessibility` | Ease of accessing tax incentives | 1=Very Difficult, 7=Very Easy |

#### Regulatory Environment
| Variable | Description | Scale |
|----------|-------------|-------|
| `regulatory_predictability` | Predictability of regulatory changes | 1=Very Unpredictable, 7=Very Predictable |
| `policy_consistency` | Consistency in policy implementation | 1=Very Inconsistent, 7=Very Consistent |
| `bureaucratic_efficiency` | Efficiency of bureaucratic processes | 1=Very Inefficient, 7=Very Efficient |

#### Infrastructure Support
| Variable | Description | Scale |
|----------|-------------|-------|
| `transport_infrastructure` | Quality of transport infrastructure | 1=Very Poor, 7=Excellent |
| `power_supply_reliability` | Reliability of electricity supply | 1=Very Unreliable, 7=Very Reliable |
| `telecommunications` | Quality of telecommunications | 1=Very Poor, 7=Excellent |

#### Corruption Experience
| Variable | Description | Scale |
|----------|-------------|-------|
| `corruption_frequency` | Frequency of corruption encounters | 1=Never, 7=Very Frequently |
| `corruption_impact` | Impact of corruption on business | 1=No Impact, 7=Severe Impact |

#### Composite Index
| Variable | Description | Calculation |
|----------|-------------|------------|
| `policy_effectiveness_index` | Overall policy effectiveness | Mean of tax effectiveness, regulatory predictability, transport infrastructure, and reverse corruption frequency |

### 6. Control Variables

#### Firm Characteristics
| Variable | Type | Description | Unit/Values |
|----------|------|-------------|-------------|
| `employees` | Integer | Number of employees | 10-2000 |
| `total_assets_million_naira` | Integer | Total assets value | 50-50,000 million Naira |
| `firm_age_years` | Integer | Years since establishment | 4-39 years |
| `subsector` | Categorical | Food processing subsector | See subsector list below |
| `ownership_type` | Categorical | Type of ownership | Domestic, Foreign, Joint Venture |
| `has_fdi` | Binary | Presence of foreign investment | 0=No FDI, 1=Has FDI |
| `location_lagos` | Categorical | Location within Lagos | See location list below |

#### Export Behavior
| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `export_category` | Categorical | Export intensity category | Non-exporter, Low exporter (<25%), Medium exporter (25-50%), High exporter (>50%) |

---

## Categorical Variable Values

### Subsectors
- Grain Processing
- Dairy Products  
- Meat Processing
- Beverage Manufacturing
- Snack Foods
- Bakery Products
- Canned Foods
- Spice Processing
- Oil & Fats
- Confectionery

### Lagos Locations
- Victoria Island
- Ikeja
- Apapa
- Ikorodu
- Agege
- Alimosho
- Mushin

---

## Data Generation Methodology

### Realistic Correlations Implemented
1. **FDI Impact**: Firms with FDI show higher ROI (+3%), ROA (+2%), and operational efficiency
2. **Size Effects**: Larger firms have higher R&D spending and technology sophistication
3. **Age Effects**: Older firms show lower innovation but more stable performance
4. **Geographic Clustering**: Different Lagos areas have varying infrastructure quality

### Distribution Characteristics
- **Ownership**: 60% Domestic, 25% Foreign, 15% Joint Venture
- **Export Behavior**: 40% non-exporters, 60% exporters with varying intensities
- **Firm Size**: Log-normal distribution reflecting realistic size heterogeneity
- **Performance**: Normal distributions with realistic means and standard deviations

### Data Quality Features
- **Missing Values**: None (complete dataset)
- **Outliers**: Realistic outliers within plausible ranges
- **Consistency**: Cross-variable consistency maintained
- **Realism**: Based on Nigerian economic context and food processing industry characteristics

---

## Usage Notes

### Research Applications
This dataset is suitable for:
- FDI impact analysis on firm performance
- Government policy moderation studies  
- Absorptive capacity research
- Innovation and performance relationships
- Cross-sectional econometric analysis

### Statistical Considerations
- Sample size (n=300) appropriate for multivariate analysis
- Mix of continuous, ordinal, and categorical variables
- Sufficient variation in key variables for meaningful analysis
- Realistic correlations for hypothesis testing

### Limitations
- Fabricated data for research/educational purposes
- Self-reported performance measures (as typical in surveys)
- Cross-sectional design (no time series)
- Lagos-specific context may limit generalizability

---

## File Information
**Filename**: `lagos_food_processing_fdi_dataset.csv`  
**Format**: CSV (Comma-separated values)  
**Encoding**: UTF-8  
**Generated**: October 2024  
**Generator**: Python script with pandas/numpy  

## Contact
For questions about this dataset or methodology, please refer to the generation script: `generate_fdi_dataset.py`