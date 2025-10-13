* SPSS Syntax for FDI Survey Data Import
* Generated: 2025-10-13

* Import CSV data
GET DATA /TYPE=TXT
  /FILE='fdi_survey_data.csv'
  /DELCASE=LINE
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
    firm_code A20
    firm_size A20
    survey_date A20
    industrial_zone A20
    response_time_minutes F8.2
    years_operation F8.2
    employees_category F8.2
    revenue_category F8.2
    ownership_type F8.2
    has_fdi F8.2
    fdi_equity F8.2
    fdi_joint_venture F8.2
    fdi_tech_transfer F8.2
    fdi_mgmt_contract F8.2
    years_fdi_partnership F8.2
    ka_technical_manuals F8.2
    ka_staff_training F8.2
    ka_tech_adaptation F8.2
    ka_knowledge_commercialization F8.2
    ka_average F8.2
    tp_production_efficiency F8.2
    tp_quality_control F8.2
    tp_order_fulfillment F8.2
    tp_employee_productivity F8.2
    tp_average F8.2
    rd_spending_percent F8.2
    new_products_3years F8.2
    process_inn_iot F8.2
    process_inn_automation F8.2
    process_inn_quality F8.2
    process_inn_other F8.2
    innovation_score F8.2
    hr_skilled_workforce_percent F8.2
    hr_training_hours F8.2
    tech_modern_equipment F8.2
    tech_machinery_age F8.2
    fin_access_credit F8.2
    fin_reinvestment_rate F8.2
    firm_resources_score F8.2
    gp_tax_incentives F8.2
    gp_regulatory_stability F8.2
    gp_infrastructure_support F8.2
    gp_ease_permits F8.2
    gp_average F8.2
    perf_roi F8.2
    perf_roa F8.2
    perf_export_intensity F8.2
    perf_capacity_utilization F8.2
    perf_market_share F8.2
    performance_composite F8.2
.

* Variable Labels
VARIABLE LABELS firm_code 'Unique firm identifier'.
VARIABLE LABELS firm_size 'Firm size category (SME/Large)'.
VARIABLE LABELS survey_date 'Date of survey completion'.
VARIABLE LABELS industrial_zone 'Lagos industrial zone location'.
VARIABLE LABELS response_time_minutes 'Time taken to complete survey (minutes)'.
VARIABLE LABELS years_operation 'Years in operation'.
VARIABLE LABELS employees_category 'Number of employees (1=1-50, 2=51-250, 3=251-500, 4=500+)'.
VARIABLE LABELS revenue_category 'Annual revenue (1=<50M, 2=50M-500M, 3=500M-5B, 4=>5B)'.
VARIABLE LABELS ownership_type 'Ownership type (1=Local, 2=Foreign-owned, 3=Joint venture)'.
VARIABLE LABELS has_fdi 'Has foreign direct investment (0=No, 1=Yes)'.
VARIABLE LABELS fdi_equity 'FDI type: Equity investment (0=No, 1=Yes)'.
VARIABLE LABELS fdi_joint_venture 'FDI type: Joint venture (0=No, 1=Yes)'.
VARIABLE LABELS fdi_tech_transfer 'FDI type: Technology transfer (0=No, 1=Yes)'.
VARIABLE LABELS fdi_mgmt_contract 'FDI type: Management contract (0=No, 1=Yes)'.
VARIABLE LABELS years_fdi_partnership 'Years with FDI partnership'.
VARIABLE LABELS ka_technical_manuals 'KA: Acquire technical manuals from FDI partners'.
VARIABLE LABELS ka_staff_training 'KA: Staff receive training from foreign partners'.
VARIABLE LABELS ka_tech_adaptation 'KA: Adapt foreign technology to local needs'.
VARIABLE LABELS ka_knowledge_commercialization 'KA: Commercialize knowledge from FDI'.
VARIABLE LABELS ka_average 'Knowledge absorption average score'.
VARIABLE LABELS tp_production_efficiency 'TP: Production efficiency'.
VARIABLE LABELS tp_quality_control 'TP: Quality control'.
VARIABLE LABELS tp_order_fulfillment 'TP: Order fulfillment time'.
VARIABLE LABELS tp_employee_productivity 'TP: Employee productivity'.
VARIABLE LABELS tp_average 'Task performance average score'.
VARIABLE LABELS rd_spending_percent 'R&D spending as % of revenue'.
VARIABLE LABELS new_products_3years 'New products launched (past 3 years)'.
VARIABLE LABELS process_inn_iot 'Process innovation: IoT systems (0=No, 1=Yes)'.
VARIABLE LABELS process_inn_automation 'Process innovation: Automation (0=No, 1=Yes)'.
VARIABLE LABELS process_inn_quality 'Process innovation: Quality management (0=No, 1=Yes)'.
VARIABLE LABELS process_inn_other 'Process innovation: Other (0=No, 1=Yes)'.
VARIABLE LABELS innovation_score 'Innovation composite score'.
VARIABLE LABELS hr_skilled_workforce_percent '% of skilled workforce'.
VARIABLE LABELS hr_training_hours 'Annual training hours per employee'.
VARIABLE LABELS tech_modern_equipment 'Use of modern equipment (0=No, 1=Yes)'.
VARIABLE LABELS tech_machinery_age 'Age of primary machinery (years)'.
VARIABLE LABELS fin_access_credit 'Access to credit (1=Easy, 2=Moderate, 3=Difficult)'.
VARIABLE LABELS fin_reinvestment_rate 'Reinvestment rate (%)'.
VARIABLE LABELS firm_resources_score 'Firm resources composite score'.
VARIABLE LABELS gp_tax_incentives 'GP: Tax incentives effectiveness'.
VARIABLE LABELS gp_regulatory_stability 'GP: Regulatory stability'.
VARIABLE LABELS gp_infrastructure_support 'GP: Infrastructure support'.
VARIABLE LABELS gp_ease_permits 'GP: Ease of obtaining permits'.
VARIABLE LABELS gp_average 'Government policy average score'.
VARIABLE LABELS perf_roi 'Return on Investment (%)'.
VARIABLE LABELS perf_roa 'Return on Assets (%)'.
VARIABLE LABELS perf_export_intensity 'Export intensity (%)'.
VARIABLE LABELS perf_capacity_utilization 'Production capacity utilization (%)'.
VARIABLE LABELS perf_market_share 'Market share in Lagos (%)'.
VARIABLE LABELS performance_composite 'Overall performance composite score'.

* Value Labels

VALUE LABELS employees_category
    1 '1-50 employees'
    2 '51-250 employees'
    3 '251-500 employees'
    4 '500+ employees'.

VALUE LABELS revenue_category
    1 'Less than 50M Naira'
    2 '50M-500M Naira'
    3 '500M-5B Naira'
    4 'More than 5B Naira'.

VALUE LABELS ownership_type
    1 'Local'
    2 'Foreign-owned'
    3 'Joint venture'.

VALUE LABELS fin_access_credit
    1 'Easy'
    2 'Moderate'
    3 'Difficult'.

* Save as SPSS file
SAVE OUTFILE='fdi_survey_data.sav'.
EXECUTE.
