# Data Dictionary: Food Processing Firms FDI Research Dataset

## Research Title
**The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy**

## Dataset Overview
- **Total Observations**: 300 food processing firms
- **Geographic Scope**: Lagos State, Nigeria
- **Time Period**: 2022-2024 (cross-sectional with time-varying macro data)
- **Data Collection Period**: January - March 2024
- **Sample Method**: Stratified random sampling (200 SMEs, 100 Large firms)

---

## 1. IDENTIFICATION VARIABLES

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `firm_id` | String | Unique firm identifier | FIRM_001 to FIRM_300 |
| `survey_year` | Integer | Year survey was conducted | 2024 |
| `reporting_year` | Integer | Year for which firm data is reported | 2022, 2023, 2024 |
| `macro_year` | Integer | Year of macro-economic data matched to firm | 2022, 2023, 2024 |

---

## 2. FIRM CHARACTERISTICS

### 2.1 Basic Firm Information

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `firm_size` | Categorical | Firm size classification | SME, Large |
| `firm_age` | Integer | Years since firm establishment | 1-50 years |
| `employees` | Integer | Total number of employees | 10-2000 |
| `total_assets_million_naira` | Float | Total assets in millions of Naira | 50-50,000 |
| `subsector` | Categorical | Food processing subsector | See subsector list below |
| `ownership_type` | Categorical | Type of ownership | Local Private, Foreign, Joint Venture, Government |

**Subsector Categories:**
- Meat Processing
- Dairy Products  
- Grain Milling
- Bakery Products
- Beverages
- Fruits & Vegetables
- Fish Processing
- Oil & Fats
- Sugar & Confectionery

### 2.2 Derived Firm Variables

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `firm_age_category` | Categorical | Firm age grouping | Young (≤5), Growing (6-10), Mature (11-20), Established (>20) |
| `export_oriented` | Binary | Export-oriented firm indicator | 0=No, 1=Yes (>10% export intensity) |
| `technology_adoption_level` | Categorical | Technology adoption classification | Very Low, Low, Medium, High |

---

## 3. FDI CONSTRUCTS (Likert Scales 1-5)

### 3.1 Knowledge Absorption (Zahra & George, 2002)

| Variable Name | Type | Description | Scale |
|---------------|------|-------------|-------|
| `ka_acquire_external_knowledge` | Float | Ability to acquire external knowledge | 1-5 Likert |
| `ka_assimilate_new_info` | Float | Ability to assimilate new information | 1-5 Likert |
| `ka_transform_knowledge` | Float | Ability to transform knowledge | 1-5 Likert |
| `ka_exploit_knowledge_commercially` | Float | Ability to exploit knowledge commercially | 1-5 Likert |
| `knowledge_absorption_score` | Float | Composite knowledge absorption score | 1-5 (average) |

### 3.2 Task Performance (Koopmans et al., 2013)

| Variable Name | Type | Description | Scale |
|---------------|------|-------------|-------|
| `tp_work_quality_standards` | Float | Meeting work quality standards | 1-5 Likert |
| `tp_efficient_task_completion` | Float | Efficient task completion | 1-5 Likert |
| `tp_productivity_levels` | Float | Productivity levels | 1-5 Likert |
| `tp_goal_achievement` | Float | Goal achievement | 1-5 Likert |
| `task_performance_score` | Float | Composite task performance score | 1-5 (average) |

### 3.3 Innovation (OECD Oslo Manual)

| Variable Name | Type | Description | Scale |
|---------------|------|-------------|-------|
| `inn_product_innovation` | Float | Product innovation capability | 1-5 Likert |
| `inn_process_innovation` | Float | Process innovation capability | 1-5 Likert |
| `inn_marketing_innovation` | Float | Marketing innovation capability | 1-5 Likert |
| `inn_organizational_innovation` | Float | Organizational innovation capability | 1-5 Likert |
| `innovation_score` | Float | Composite innovation score | 1-5 (average) |

### 3.4 Firm Resources

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `fr_skilled_labor_ratio` | Float | Percentage of skilled labor | 5-80% |
| `fr_rd_spend_ratio` | Float | R&D spending as % of revenue | 0-15% |
| `fr_iot_adoption` | Binary | IoT technology adoption | 0=No, 1=Yes |
| `fr_liquidity_ratio` | Float | Current liquidity ratio | 0.5-5.0 |
| `firm_resources_score` | Float | Composite firm resources score | 0-5 (normalized) |

---

## 4. FIRM PERFORMANCE METRICS

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `roi_percent` | Float | Return on Investment (%) | -5% to 35% |
| `roa_percent` | Float | Return on Assets (%) | -3% to 25% |
| `export_intensity_percent` | Float | Export revenue as % of total revenue | 0-60% |
| `market_share_percent` | Float | Market share in subsector (%) | 0.1-25% |
| `operational_efficiency_score` | Float | Operational efficiency score | 1-10 scale |
| `overall_performance_index` | Float | Composite performance index | 0-10 (normalized) |

---

## 5. FDI VARIABLES

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `fdi_presence` | Binary | FDI presence in firm | 0=No, 1=Yes |
| `fdi_type` | Categorical | Type of FDI investment | None, Greenfield, M&A, Joint Venture |
| `fdi_amount_million_usd` | Float | FDI amount in millions USD | 0-500+ |
| `fdi_origin_country` | Categorical | Origin country of FDI | See country list below |
| `fdi_intensity` | Float | FDI amount relative to firm assets | 0-100+ |

**FDI Origin Countries:**
- South Africa, UK, USA, Netherlands, China, India, Germany, France, UAE, Other

---

## 6. GOVERNMENT POLICY PERCEPTION (Likert Scales 1-7)

| Variable Name | Type | Description | Scale |
|---------------|------|-------------|-------|
| `gp_tax_incentives_effectiveness` | Float | Tax incentives effectiveness | 1-7 Likert |
| `gp_regulatory_stability` | Float | Regulatory stability perception | 1-7 Likert |
| `gp_infrastructure_support` | Float | Infrastructure support quality | 1-7 Likert |
| `gp_ease_of_doing_business` | Float | Ease of doing business perception | 1-7 Likert |
| `gp_corruption_experience` | Float | Corruption experience level | 1-7 Likert (higher=more corruption) |
| `gp_policy_consistency` | Float | Policy consistency perception | 1-7 Likert |
| `gp_government_support_programs` | Float | Government support programs effectiveness | 1-7 Likert |
| `policy_effectiveness_index` | Float | Composite policy effectiveness index | 1-7 (average, corruption reversed) |

---

## 7. MACROECONOMIC DATA (Annual, 2019-2024)

### 7.1 FDI Inflows

| Variable Name | Type | Description | Source |
|---------------|------|-------------|--------|
| `macro_nigeria_total_fdi_billion_usd` | Float | Nigeria total FDI inflows (billion USD) | CBN, UNCTAD |
| `macro_lagos_total_fdi_billion_usd` | Float | Lagos State total FDI inflows (billion USD) | CBN, NBS |
| `macro_lagos_food_processing_fdi_million_usd` | Float | Lagos food processing FDI (million USD) | CBN, NIPC |
| `macro_greenfield_fdi_ratio` | Float | Greenfield FDI as ratio of total | UNCTAD |
| `macro_ma_fdi_ratio` | Float | M&A FDI as ratio of total | UNCTAD |
| `macro_jv_fdi_ratio` | Float | Joint Venture FDI as ratio of total | UNCTAD |

### 7.2 Economic Indicators

| Variable Name | Type | Description | Source |
|---------------|------|-------------|--------|
| `macro_gdp_growth_rate` | Float | GDP growth rate (%) | NBS |
| `macro_inflation_rate` | Float | Inflation rate (%) | NBS |
| `macro_exchange_rate_naira_usd` | Float | Exchange rate (Naira per USD) | CBN |
| `macro_interest_rate` | Float | Monetary policy rate (%) | CBN |
| `macro_unemployment_rate` | Float | Unemployment rate (%) | NBS |

### 7.3 Governance & Policy Indicators

| Variable Name | Type | Description | Source |
|---------------|------|-------------|--------|
| `macro_ease_of_doing_business_rank` | Integer | Ease of Doing Business rank | World Bank |
| `macro_ease_of_doing_business_score` | Float | Ease of Doing Business score | World Bank |
| `macro_corruption_perception_index` | Float | Corruption Perception Index (0-100) | Transparency International |
| `macro_regulatory_quality_index` | Float | Regulatory Quality Index (-2.5 to 2.5) | World Bank |
| `macro_government_effectiveness_index` | Float | Government Effectiveness Index (-2.5 to 2.5) | World Bank |
| `macro_corporate_tax_rate` | Float | Corporate tax rate (%) | FIRS |
| `macro_infrastructure_quality_score` | Float | Infrastructure quality score (1-7) | WEF |
| `macro_pioneer_status_grants` | Integer | Number of pioneer status grants | NIPC |

### 7.4 Sectoral Performance

| Variable Name | Type | Description | Source |
|---------------|------|-------------|--------|
| `macro_sector_gdp_billion_naira` | Float | Food processing sector GDP (billion Naira) | NBS |
| `macro_sector_employment_thousands` | Float | Sector employment (thousands) | NBS |
| `macro_sector_exports_million_usd` | Float | Sector exports (million USD) | NBS |
| `macro_number_of_firms` | Integer | Number of registered firms in sector | Lagos State |
| `macro_capacity_utilization_percent` | Float | Sector capacity utilization (%) | CBN |
| `macro_average_firm_size_employees` | Integer | Average firm size in sector | NBS |

### 7.5 Infrastructure & Financial Development

| Variable Name | Type | Description | Source |
|---------------|------|-------------|--------|
| `macro_power_supply_hours_per_day` | Float | Average power supply (hours/day) | NERC |
| `macro_logistics_performance_index` | Float | Logistics Performance Index (1-5) | World Bank |
| `macro_internet_penetration_percent` | Float | Internet penetration rate (%) | NCC |
| `macro_port_efficiency_score` | Float | Port efficiency score (1-7) | WEF |
| `macro_road_quality_index` | Float | Road quality index (1-7) | WEF |
| `macro_financial_inclusion_rate` | Float | Financial inclusion rate (%) | CBN |
| `macro_credit_to_private_sector_percent_gdp` | Float | Credit to private sector (% of GDP) | CBN |

### 7.6 Sectoral Financial Performance

| Variable Name | Type | Description | Source |
|---------------|------|-------------|--------|
| `macro_sector_average_roa` | Float | Sector average ROA (%) | CBN |
| `macro_sector_average_roi` | Float | Sector average ROI (%) | CBN |
| `macro_total_sector_revenue_billion_naira` | Float | Total sector revenue (billion Naira) | NBS |
| `macro_total_sector_assets_billion_naira` | Float | Total sector assets (billion Naira) | CBN |
| `macro_average_debt_equity_ratio` | Float | Average debt-to-equity ratio | CBN |
| `macro_sector_export_revenue_million_usd` | Float | Sector export revenue (million USD) | NBS |

---

## 8. INTERACTION TERMS (Moderation Analysis)

| Variable Name | Type | Description | Calculation |
|---------------|------|-------------|-------------|
| `fdi_x_policy_effectiveness` | Float | FDI presence × Policy effectiveness | fdi_presence × policy_effectiveness_index |
| `fdi_intensity_x_policy_effectiveness` | Float | FDI intensity × Policy effectiveness | fdi_intensity × policy_effectiveness_index |
| `fdi_x_gdp_growth` | Float | FDI presence × GDP growth | fdi_presence × macro_gdp_growth_rate |
| `fdi_x_ease_of_business` | Float | FDI presence × Ease of business | fdi_presence × macro_ease_of_doing_business_score |
| `knowledge_absorption_x_policy` | Float | Knowledge absorption × Policy | knowledge_absorption_score × policy_effectiveness_index |

---

## 9. SURVEY METADATA

| Variable Name | Type | Description | Values/Range |
|---------------|------|-------------|--------------|
| `survey_date` | Date | Date survey was completed | 2024-01-15 to 2024-03-30 |
| `respondent_position` | Categorical | Position of survey respondent | CEO, Operations Manager, General Manager, etc. |
| `response_completeness` | Float | Survey completion rate | 0.85-1.00 |
| `response_time_minutes` | Integer | Time taken to complete survey | 5-60 minutes |
| `data_collection_date` | Date | Date secondary data was collected | 2024-03-15 |
| `last_updated` | Date | Last update timestamp | Current date |

---

## 10. DATA QUALITY NOTES

### Missing Data
- No systematic missing data in the fabricated dataset
- All firms have complete survey responses (response_completeness ≥ 85%)
- Secondary data is complete for all years 2019-2024

### Data Validation
- Likert scale responses are bounded within specified ranges
- Financial ratios are within realistic ranges for Nigerian food processing sector
- FDI amounts are consistent with firm size and type
- Cross-validation between primary and secondary data sources

### Limitations
- This is a fabricated dataset for research demonstration purposes
- Actual data collection would require IRB approval and firm consent
- Real-world data may have more missing values and measurement errors
- Temporal matching assumes firms report data for the year specified

---

## 11. RECOMMENDED ANALYSIS APPROACHES

### Dependent Variables (Performance)
- `overall_performance_index` (primary)
- `roi_percent`, `roa_percent` (financial performance)
- `export_intensity_percent` (internationalization)
- `operational_efficiency_score` (operational performance)

### Independent Variables (FDI)
- `fdi_presence` (binary treatment)
- `fdi_intensity` (continuous treatment)
- `knowledge_absorption_score`, `task_performance_score`, `innovation_score` (mechanisms)

### Moderating Variables (Government Policy)
- `policy_effectiveness_index` (primary moderator)
- Individual policy dimensions (gp_* variables)
- Macro-level policy indicators (macro_ease_of_doing_business_score, etc.)

### Control Variables
- Firm characteristics: `firm_size`, `firm_age`, `subsector`, `ownership_type`
- Macro controls: `macro_gdp_growth_rate`, `macro_inflation_rate`, etc.
- Time controls: `reporting_year` fixed effects

### Suggested Models
1. **OLS Regression**: Basic FDI-performance relationship
2. **Moderated Regression**: Include interaction terms for policy moderation
3. **Structural Equation Modeling**: Test mediation through knowledge absorption/innovation
4. **Propensity Score Matching**: Address FDI selection bias
5. **Multilevel Models**: Account for year and subsector clustering