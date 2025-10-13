#!/usr/bin/env python3
"""
SYNTHETIC FDI SURVEY DATA GENERATOR
PhD Research: The Influence of Foreign Direct Investment on the Performance 
of Food Processing Firms in Lagos, Nigeria
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(12345)
random.seed(12345)

# =============================================================================
# 1. SAMPLE ALLOCATION (Neyman Allocation)
# =============================================================================
N_sme = 300
N_large = 150
N_total = 450
n_sme = 200
n_large = 100
n_total = 300

print("=== GENERATING FDI SURVEY DATA ===")
print(f"Sample size: {n_total} firms")
print(f"SMEs: {n_sme}, Large firms: {n_large}")

# =============================================================================
# 2. FIRM CHARACTERISTICS GENERATION
# =============================================================================

# Generate firm IDs
firm_ids = [f"FIRM_{i:03d}" for i in range(1, n_total + 1)]

# Firm size distribution (SME vs Large)
firm_size = ['SME'] * n_sme + ['Large'] * n_large

# Years of operation (SMEs: 5-25 years, Large: 10-40 years)
years_operation = (list(np.random.randint(5, 26, n_sme)) + 
                  list(np.random.randint(10, 41, n_large)))

# Number of employees (correlated with firm size)
employees = []
# SMEs: 70% have 1-50 employees, 30% have 51-250 employees
for i in range(n_sme):
    if np.random.random() < 0.7:
        employees.append(np.random.randint(1, 51))
    else:
        employees.append(np.random.randint(51, 251))

# Large firms: 60% have 251-500 employees, 40% have 501-1000 employees
for i in range(n_large):
    if np.random.random() < 0.6:
        employees.append(np.random.randint(251, 501))
    else:
        employees.append(np.random.randint(501, 1001))

# Annual revenue in Naira (correlated with firm size and employees)
revenue_categories = (list(np.random.choice(['<50M', '50M-500M'], n_sme, p=[0.6, 0.4])) +
                     list(np.random.choice(['50M-500M', '500M-5B', '>5B'], n_large, p=[0.3, 0.5, 0.2])))

# Ownership type (higher FDI probability for large firms)
ownership = (list(np.random.choice(['Local', 'Foreign-owned', 'Joint venture'], n_sme, p=[0.7, 0.1, 0.2])) +
            list(np.random.choice(['Local', 'Foreign-owned', 'Joint venture'], n_large, p=[0.4, 0.3, 0.3])))

# =============================================================================
# 3. FDI ENGAGEMENT GENERATION
# =============================================================================

# FDI partnerships (higher probability for large firms and foreign/joint ownership)
fdi_prob = []
for i in range(n_total):
    if i < n_sme:  # SME
        if ownership[i] in ['Foreign-owned', 'Joint venture']:
            fdi_prob.append(0.8)
        else:
            fdi_prob.append(0.3)
    else:  # Large firm
        if ownership[i] in ['Foreign-owned', 'Joint venture']:
            fdi_prob.append(0.9)
        else:
            fdi_prob.append(0.5)

fdi_partnership = np.random.binomial(1, fdi_prob, n_total)

# Type of FDI (only for firms with FDI)
fdi_type = [''] * n_total
fdi_firms = np.where(fdi_partnership == 1)[0]
fdi_types = ['Equity', 'Joint venture', 'Technology transfer', 'Management contract']
fdi_type_probs = [0.3, 0.3, 0.25, 0.15]

for firm_idx in fdi_firms:
    fdi_type[firm_idx] = np.random.choice(fdi_types, p=fdi_type_probs)

# Years with FDI partnership
years_fdi = np.zeros(n_total)
for firm_idx in fdi_firms:
    max_years = min(years_operation[firm_idx], 20)  # Cap at 20 years
    years_fdi[firm_idx] = np.random.randint(1, max_years + 1)

# =============================================================================
# 4. KNOWLEDGE ABSORPTION SCALE (Zahra & George)
# =============================================================================

# Generate correlated knowledge absorption items (1-5 scale)
# Higher scores for firms with FDI and larger firms
fdi_effect = np.where(fdi_partnership == 1, 0.8, 0)
size_effect = np.where(np.array(firm_size) == 'Large', 0.5, 0)
base_ka = 2.5 + fdi_effect + size_effect + np.random.normal(0, 0.5, n_total)
base_ka = np.clip(base_ka, 1, 5)

# Generate correlated items
ka_cor = 0.7  # High correlation between KA items
ka_cov = np.full((4, 4), ka_cor)
np.fill_diagonal(ka_cov, 1)

ka_items = np.random.multivariate_normal([0, 0, 0, 0], ka_cov, n_total)
ka_items = ka_items + base_ka.reshape(-1, 1)
ka_items = np.clip(ka_items, 1, 5)

ka1 = np.round(ka_items[:, 0], 1)  # Technical manuals acquisition
ka2 = np.round(ka_items[:, 1], 1)  # Staff training from partners
ka3 = np.round(ka_items[:, 2], 1)  # Technology adaptation
ka4 = np.round(ka_items[:, 3], 1)  # Knowledge commercialization

# =============================================================================
# 5. TASK PERFORMANCE SCALE (Koopmans et al.)
# =============================================================================

# Generate correlated task performance items (1-5 scale)
# Performance influenced by FDI, knowledge absorption, and firm size
fdi_perf_effect = np.where(fdi_partnership == 1, 0.6, 0)
ka_perf_effect = (np.mean(ka_items, axis=1) - 3) * 0.4
size_perf_effect = np.where(np.array(firm_size) == 'Large', 0.3, 0)
base_tp = 2.8 + fdi_perf_effect + ka_perf_effect + size_perf_effect + np.random.normal(0, 0.4, n_total)
base_tp = np.clip(base_tp, 1, 5)

# Generate correlated items
tp_cor = 0.6
tp_cov = np.full((4, 4), tp_cor)
np.fill_diagonal(tp_cov, 1)

tp_items = np.random.multivariate_normal([0, 0, 0, 0], tp_cov, n_total)
tp_items = tp_items + base_tp.reshape(-1, 1)
tp_items = np.clip(tp_items, 1, 5)

tp1 = np.round(tp_items[:, 0], 1)  # Production efficiency
tp2 = np.round(tp_items[:, 1], 1)  # Quality control
tp3 = np.round(tp_items[:, 2], 1)  # Order fulfillment time
tp4 = np.round(tp_items[:, 3], 1)  # Employee productivity

# =============================================================================
# 6. INNOVATION METRICS (OECD Oslo Manual)
# =============================================================================

# R&D spending as % of revenue (higher for FDI firms and large firms)
rd_base = (2 + np.where(fdi_partnership == 1, 1.5, 0) + 
          np.where(np.array(firm_size) == 'Large', 1, 0) + 
          np.random.normal(0, 1, n_total))
rd_spending = np.clip(rd_base, 0, 15)

# New products launched (past 3 years)
new_products = np.random.poisson(2 + np.where(fdi_partnership == 1, 2, 0) + 
                                np.where(np.array(firm_size) == 'Large', 1, 0), n_total)

# Process innovations (binary indicators)
iot_adoption = np.random.binomial(1, 0.3 + np.where(fdi_partnership == 1, 0.2, 0) + 
                                 np.where(np.array(firm_size) == 'Large', 0.2, 0), n_total)
automation = np.random.binomial(1, 0.4 + np.where(fdi_partnership == 1, 0.3, 0) + 
                               np.where(np.array(firm_size) == 'Large', 0.2, 0), n_total)
quality_mgmt = np.random.binomial(1, 0.6 + np.where(fdi_partnership == 1, 0.2, 0), n_total)

# =============================================================================
# 7. FIRM RESOURCES
# =============================================================================

# Human Resources
skilled_workforce = (40 + np.where(fdi_partnership == 1, 15, 0) + 
                    np.where(np.array(firm_size) == 'Large', 10, 0) + 
                    np.random.normal(0, 8, n_total))
skilled_workforce = np.clip(skilled_workforce, 10, 90)

training_hours = (20 + np.where(fdi_partnership == 1, 15, 0) + 
                 np.where(np.array(firm_size) == 'Large', 10, 0) + 
                 np.random.normal(0, 5, n_total))
training_hours = np.clip(training_hours, 5, 80)

# Technological Resources
modern_equipment = np.random.binomial(1, 0.5 + np.where(fdi_partnership == 1, 0.3, 0) + 
                                     np.where(np.array(firm_size) == 'Large', 0.2, 0), n_total)
machinery_age = (8 + np.where(modern_equipment == 1, -3, 2) + 
                np.random.normal(0, 2, n_total))
machinery_age = np.clip(machinery_age, 1, 20)

# Financial Resources
credit_access = []
for i in range(n_total):
    if firm_size[i] == 'Large':
        probs = [0.5, 0.4, 0.1]  # Easy, Moderate, Difficult
    else:
        probs = [0.3, 0.5, 0.2]
    credit_access.append(np.random.choice(['Easy', 'Moderate', 'Difficult'], p=probs))

reinvestment_rate = (15 + np.where(fdi_partnership == 1, 5, 0) + 
                    np.where(np.array(firm_size) == 'Large', 3, 0) + 
                    np.random.normal(0, 5, n_total))
reinvestment_rate = np.clip(reinvestment_rate, 5, 50)

# =============================================================================
# 8. GOVERNMENT POLICY PERCEPTION (1-7 scale)
# =============================================================================

# Generate correlated government policy items
# Lower scores for larger firms (more critical), higher for FDI firms (beneficiaries)
gp_base = (4 + np.where(fdi_partnership == 1, 0.5, 0) - 
          np.where(np.array(firm_size) == 'Large', 0.3, 0) + 
          np.random.normal(0, 0.8, n_total))
gp_base = np.clip(gp_base, 1, 7)

gp_cor = 0.5
gp_cov = np.full((4, 4), gp_cor)
np.fill_diagonal(gp_cov, 1)

gp_items = np.random.multivariate_normal([0, 0, 0, 0], gp_cov, n_total)
gp_items = gp_items + gp_base.reshape(-1, 1)
gp_items = np.clip(gp_items, 1, 7)

gp1 = np.round(gp_items[:, 0], 1)  # Tax incentives effectiveness
gp2 = np.round(gp_items[:, 1], 1)  # Regulatory stability
gp3 = np.round(gp_items[:, 2], 1)  # Infrastructure support
gp4 = np.round(gp_items[:, 3], 1)  # Ease of obtaining permits

# =============================================================================
# 9. PERFORMANCE METRICS
# =============================================================================

# Financial Performance (influenced by FDI, KA, Innovation, and firm size)
fdi_fin_effect = np.where(fdi_partnership == 1, 3, 0)
ka_fin_effect = (np.mean(ka_items, axis=1) - 3) * 2
innovation_effect = (rd_spending / 10) + (new_products * 0.5)
size_fin_effect = np.where(np.array(firm_size) == 'Large', 2, 0)

# ROI (Return on Investment)
roi_base = (8 + fdi_fin_effect + ka_fin_effect + innovation_effect + 
           size_fin_effect + np.random.normal(0, 3, n_total))
roi = np.clip(roi_base, -5, 35)

# ROA (Return on Assets)
roa_base = (6 + fdi_fin_effect + ka_fin_effect + innovation_effect + 
           size_fin_effect + np.random.normal(0, 2.5, n_total))
roa = np.clip(roa_base, -3, 25)

# Export intensity
export_intensity = (15 + np.where(fdi_partnership == 1, 20, 0) + 
                   np.where(np.array(firm_size) == 'Large', 10, 0) + 
                   np.random.normal(0, 8, n_total))
export_intensity = np.clip(export_intensity, 0, 80)

# Operational Performance
capacity_utilization = (70 + np.where(fdi_partnership == 1, 10, 0) + 
                       np.where(np.array(firm_size) == 'Large', 5, 0) + 
                       np.random.normal(0, 8, n_total))
capacity_utilization = np.clip(capacity_utilization, 30, 100)

market_share = (5 + np.where(fdi_partnership == 1, 3, 0) + 
               np.where(np.array(firm_size) == 'Large', 4, 0) + 
               np.random.normal(0, 3, n_total))
market_share = np.clip(market_share, 0.5, 25)

# =============================================================================
# 10. CREATE MAIN DATASET
# =============================================================================

# Generate random survey dates
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 3, 31)
date_range = (end_date - start_date).days
survey_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_total)]

# Create the main dataset
fdi_data = pd.DataFrame({
    # Firm identification
    'firm_id': firm_ids,
    'firm_size': firm_size,
    'date_survey': survey_dates,
    
    # Section A: Firm Background
    'years_operation': years_operation,
    'employees': employees,
    'revenue_category': revenue_categories,
    'ownership_type': ownership,
    
    # FDI Engagement
    'fdi_partnership': fdi_partnership,
    'fdi_type': fdi_type,
    'years_fdi': years_fdi,
    
    # Section B: Knowledge Absorption (1-5 scale)
    'ka1_technical_manuals': ka1,
    'ka2_staff_training': ka2,
    'ka3_technology_adaptation': ka3,
    'ka4_knowledge_commercialization': ka4,
    
    # Section C: Task Performance (1-5 scale)
    'tp1_production_efficiency': tp1,
    'tp2_quality_control': tp2,
    'tp3_order_fulfillment': tp3,
    'tp4_employee_productivity': tp4,
    
    # Section D: Innovation
    'rd_spending_pct': np.round(rd_spending, 1),
    'new_products_3years': new_products,
    'iot_adoption': iot_adoption,
    'automation_adoption': automation,
    'quality_management': quality_mgmt,
    
    # Section E: Firm Resources
    'skilled_workforce_pct': np.round(skilled_workforce, 1),
    'training_hours_annual': np.round(training_hours, 1),
    'modern_equipment': modern_equipment,
    'machinery_age': np.round(machinery_age, 1),
    'credit_access': credit_access,
    'reinvestment_rate_pct': np.round(reinvestment_rate, 1),
    
    # Section F: Government Policy (1-7 scale)
    'gp1_tax_incentives': gp1,
    'gp2_regulatory_stability': gp2,
    'gp3_infrastructure_support': gp3,
    'gp4_permits_ease': gp4,
    
    # Section G: Performance Metrics
    'roi_pct': np.round(roi, 1),
    'roa_pct': np.round(roa, 1),
    'export_intensity_pct': np.round(export_intensity, 1),
    'capacity_utilization_pct': np.round(capacity_utilization, 1),
    'market_share_pct': np.round(market_share, 1)
})

# =============================================================================
# 11. CREATE COMPOSITE SCORES FOR SEM ANALYSIS
# =============================================================================

# Knowledge Absorption composite (average of 4 items)
fdi_data['knowledge_absorption'] = np.round(
    fdi_data[['ka1_technical_manuals', 'ka2_staff_training', 
              'ka3_technology_adaptation', 'ka4_knowledge_commercialization']].mean(axis=1), 2)

# Task Performance composite (average of 4 items)
fdi_data['task_performance'] = np.round(
    fdi_data[['tp1_production_efficiency', 'tp2_quality_control', 
              'tp3_order_fulfillment', 'tp4_employee_productivity']].mean(axis=1), 2)

# Innovation composite (standardized combination of R&D, new products, process innovations)
innovation_vars = ['rd_spending_pct', 'new_products_3years', 'iot_adoption', 'automation_adoption', 'quality_management']
innovation_std = fdi_data[innovation_vars].apply(lambda x: (x - x.mean()) / x.std())
fdi_data['innovation'] = np.round(innovation_std.sum(axis=1), 2)

# Government Policy composite (average of 4 items)
fdi_data['government_policy'] = np.round(
    fdi_data[['gp1_tax_incentives', 'gp2_regulatory_stability', 
              'gp3_infrastructure_support', 'gp4_permits_ease']].mean(axis=1), 2)

# Firm Performance composite (standardized combination of financial and operational metrics)
performance_vars = ['roi_pct', 'roa_pct', 'export_intensity_pct', 'capacity_utilization_pct', 'market_share_pct']
performance_std = fdi_data[performance_vars].apply(lambda x: (x - x.mean()) / x.std())
fdi_data['firm_performance'] = np.round(performance_std.sum(axis=1), 2)

# =============================================================================
# 12. ADD MODERATION INTERACTION TERMS
# =============================================================================

# FDI × Government Policy interaction
fdi_data['fdi_gov_interaction'] = fdi_data['fdi_partnership'] * fdi_data['government_policy']

# Knowledge Absorption × Government Policy interaction
fdi_data['ka_gov_interaction'] = fdi_data['knowledge_absorption'] * fdi_data['government_policy']

# Task Performance × Government Policy interaction
fdi_data['tp_gov_interaction'] = fdi_data['task_performance'] * fdi_data['government_policy']

# Innovation × Government Policy interaction
fdi_data['innovation_gov_interaction'] = fdi_data['innovation'] * fdi_data['government_policy']

# =============================================================================
# 13. EXPORT DATASETS
# =============================================================================

# Export main dataset as CSV
fdi_data.to_csv('fdi_survey_data.csv', index=False)

# Export as Excel with multiple sheets
with pd.ExcelWriter('fdi_survey_data.xlsx', engine='openpyxl') as writer:
    fdi_data.to_excel(writer, sheet_name='Survey Data', index=False)
    
    # Summary statistics sheet
    summary_stats = pd.DataFrame({
        'Variable': fdi_data.columns,
        'Mean': fdi_data.select_dtypes(include=[np.number]).mean(),
        'SD': fdi_data.select_dtypes(include=[np.number]).std(),
        'Min': fdi_data.select_dtypes(include=[np.number]).min(),
        'Max': fdi_data.select_dtypes(include=[np.number]).max(),
        'N': fdi_data.count()
    })
    summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)

# =============================================================================
# 14. GENERATE DATA DOCUMENTATION
# =============================================================================

# Create codebook
codebook = pd.DataFrame({
    'Variable': fdi_data.columns,
    'Description': [
        'Unique firm identifier',
        'Firm size category (SME or Large)',
        'Date of survey completion',
        'Years of firm operation',
        'Number of employees',
        'Annual revenue category in Naira',
        'Ownership type',
        'Has FDI partnership (1=Yes, 0=No)',
        'Type of FDI partnership',
        'Years with FDI partnership',
        'Technical manuals acquisition (1-5 scale)',
        'Staff training from partners (1-5 scale)',
        'Technology adaptation (1-5 scale)',
        'Knowledge commercialization (1-5 scale)',
        'Production efficiency (1-5 scale)',
        'Quality control (1-5 scale)',
        'Order fulfillment time (1-5 scale)',
        'Employee productivity (1-5 scale)',
        'R&D spending as % of revenue',
        'New products launched in past 3 years',
        'IoT systems adoption (1=Yes, 0=No)',
        'Automation adoption (1=Yes, 0=No)',
        'Quality management adoption (1=Yes, 0=No)',
        'Percentage of skilled workforce',
        'Annual training hours per employee',
        'Uses modern equipment (1=Yes, 0=No)',
        'Age of primary machinery in years',
        'Access to credit rating',
        'Reinvestment rate as % of profit',
        'Tax incentives effectiveness (1-7 scale)',
        'Regulatory stability (1-7 scale)',
        'Infrastructure support (1-7 scale)',
        'Ease of obtaining permits (1-7 scale)',
        'Return on Investment (%)',
        'Return on Assets (%)',
        'Export intensity (%)',
        'Production capacity utilization (%)',
        'Market share in Lagos (%)',
        'Knowledge Absorption composite score',
        'Task Performance composite score',
        'Innovation composite score',
        'Government Policy composite score',
        'Firm Performance composite score',
        'FDI × Government Policy interaction',
        'Knowledge Absorption × Government Policy interaction',
        'Task Performance × Government Policy interaction',
        'Innovation × Government Policy interaction'
    ],
    'Scale': [
        'Nominal', 'Nominal', 'Date', 'Ratio', 'Ratio', 'Ordinal', 'Nominal',
        'Binary', 'Nominal', 'Ratio', 'Interval', 'Interval', 'Interval', 'Interval',
        'Interval', 'Interval', 'Interval', 'Interval', 'Ratio', 'Count',
        'Binary', 'Binary', 'Binary', 'Ratio', 'Ratio', 'Binary', 'Ratio',
        'Ordinal', 'Ratio', 'Interval', 'Interval', 'Interval', 'Interval',
        'Ratio', 'Ratio', 'Ratio', 'Ratio', 'Ratio', 'Interval', 'Interval',
        'Interval', 'Interval', 'Interval', 'Interval', 'Interval', 'Interval', 'Interval'
    ]
})

codebook.to_csv('fdi_survey_codebook.csv', index=False)

# =============================================================================
# 15. GENERATE CORRELATION MATRIX FOR VALIDATION
# =============================================================================

# Select numeric variables for correlation
numeric_vars = fdi_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_vars.corr()

# Save correlation matrix
correlation_matrix.to_csv('fdi_correlation_matrix.csv')

# =============================================================================
# 16. PRINT SUMMARY STATISTICS
# =============================================================================

print("\n=== FDI SURVEY DATA GENERATION COMPLETE ===")
print(f"Total firms generated: {n_total}")
print(f"SMEs: {n_sme}")
print(f"Large firms: {n_large}")
print(f"Firms with FDI: {sum(fdi_partnership)}")
print(f"Response rate: {sum(fdi_partnership)/n_total*100:.1f}%")

print("\n=== KEY STATISTICS ===")
print(f"Knowledge Absorption (mean ± sd): {fdi_data['knowledge_absorption'].mean():.2f} ± {fdi_data['knowledge_absorption'].std():.2f}")
print(f"Task Performance (mean ± sd): {fdi_data['task_performance'].mean():.2f} ± {fdi_data['task_performance'].std():.2f}")
print(f"Innovation (mean ± sd): {fdi_data['innovation'].mean():.2f} ± {fdi_data['innovation'].std():.2f}")
print(f"Government Policy (mean ± sd): {fdi_data['government_policy'].mean():.2f} ± {fdi_data['government_policy'].std():.2f}")
print(f"Firm Performance (mean ± sd): {fdi_data['firm_performance'].mean():.2f} ± {fdi_data['firm_performance'].std():.2f}")

print("\n=== FILES GENERATED ===")
print("1. fdi_survey_data.csv - Main dataset")
print("2. fdi_survey_data.xlsx - Excel format with summary")
print("3. fdi_survey_codebook.csv - Variable documentation")
print("4. fdi_correlation_matrix.csv - Correlation matrix")

print("\n=== READY FOR SEM ANALYSIS ===")
print("The dataset includes all variables needed for your structural equation model:")
print("- Latent variables: Knowledge Absorption, Innovation, Government Policy, Performance")
print("- Interaction terms for moderation analysis")
print("- Composite scores for SEM modeling")
print("- Realistic correlations between variables")

# Display first few rows
print("\n=== SAMPLE DATA (First 5 rows) ===")
print(fdi_data.head())

print("\n=== DATA GENERATION COMPLETE ===")