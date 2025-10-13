#!/usr/bin/env python3
"""
Generate Synthetic Dataset for FDI Study
Topic: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms 
       in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy
"""

import numpy as np
import pandas as pd
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Number of firms to survey
N_FIRMS = 250

print(f"Generating dataset for {N_FIRMS} food processing firms in Lagos, Nigeria...")

# ============================================================================
# 1. CONTROL VARIABLES (Generate first as they influence other variables)
# ============================================================================

# Firm ID
firm_ids = [f"FIRM_{str(i).zfill(4)}" for i in range(1, N_FIRMS + 1)]

# Firm Age (years) - mixture of young and established firms
firm_age = np.random.choice(
    [np.random.randint(1, 5), np.random.randint(5, 15), np.random.randint(15, 40)],
    size=N_FIRMS,
    p=[0.3, 0.5, 0.2]
)

# Firm Size (number of employees)
# Small (< 50), Medium (50-250), Large (> 250)
firm_size_category = np.random.choice(['Small', 'Medium', 'Large'], 
                                      size=N_FIRMS, 
                                      p=[0.4, 0.45, 0.15])
employees = []
for cat in firm_size_category:
    if cat == 'Small':
        employees.append(np.random.randint(10, 50))
    elif cat == 'Medium':
        employees.append(np.random.randint(50, 250))
    else:
        employees.append(np.random.randint(250, 1000))
employees = np.array(employees)

# Total Assets (in Million Naira)
assets_million_ngn = employees * np.random.uniform(0.5, 2.0, N_FIRMS) * \
                     (1 + firm_age * 0.05) * np.random.uniform(0.8, 1.2, N_FIRMS)

# Subsector
subsectors = np.random.choice(
    ['Dairy Products', 'Bakery & Confectionery', 'Meat Processing', 
     'Fruit & Vegetable Processing', 'Beverages', 'Oil & Fats', 
     'Grain Milling', 'Other Food Products'],
    size=N_FIRMS,
    p=[0.15, 0.18, 0.12, 0.15, 0.15, 0.10, 0.08, 0.07]
)

# Ownership Type
ownership_type = np.random.choice(
    ['Fully Local', 'Joint Venture', 'Foreign-Owned', 'Family Business'],
    size=N_FIRMS,
    p=[0.45, 0.25, 0.20, 0.10]
)

# FDI Status (derived from ownership)
has_fdi = np.where(
    (ownership_type == 'Joint Venture') | (ownership_type == 'Foreign-Owned'),
    1, 0
)
# Add some noise - some local firms might have received FDI
has_fdi = np.where(
    (ownership_type == 'Fully Local') & (np.random.random(N_FIRMS) < 0.15),
    1, has_fdi
)

# ============================================================================
# 2. FDI CONSTRUCTS (Likert 1-5)
# Based on Zahra & George (2002), Koopmans et al. (2013), OECD Oslo Manual
# ============================================================================

# Base scores influenced by FDI status, firm size, and age
base_fdi_effect = has_fdi * 0.6 + (firm_age / 40) * 0.3 + \
                  (employees / 1000) * 0.4

# Knowledge Absorption Capacity (4 items averaged)
# Acquisition, Assimilation, Transformation, Exploitation
knowledge_absorption_items = []
for i in range(4):
    item = 2.5 + base_fdi_effect * 1.5 + np.random.normal(0, 0.7, N_FIRMS)
    item = np.clip(item, 1, 5)
    knowledge_absorption_items.append(item)
knowledge_absorption = np.mean(knowledge_absorption_items, axis=0)

# Task Performance (5 items averaged)
task_performance_items = []
for i in range(5):
    item = 2.8 + base_fdi_effect * 1.2 + np.random.normal(0, 0.6, N_FIRMS)
    item = np.clip(item, 1, 5)
    task_performance_items.append(item)
task_performance = np.mean(task_performance_items, axis=0)

# Innovation Capability (5 items averaged)
# Product, Process, Organizational, Marketing innovation
innovation_items = []
for i in range(5):
    item = 2.3 + base_fdi_effect * 1.4 + \
           (knowledge_absorption - 3) * 0.3 + np.random.normal(0, 0.8, N_FIRMS)
    item = np.clip(item, 1, 5)
    innovation_items.append(item)
innovation_capability = np.mean(innovation_items, axis=0)

# ============================================================================
# 3. FIRM RESOURCES
# ============================================================================

# Human Capital - Skilled Labor Ratio (%)
skilled_labor_ratio = 20 + base_fdi_effect * 25 + \
                      (firm_age / 40) * 15 + np.random.normal(0, 10, N_FIRMS)
skilled_labor_ratio = np.clip(skilled_labor_ratio, 5, 90)

# Technological Resources
# IoT/Automation Use (1-5 scale)
iot_usage = 1.5 + base_fdi_effect * 2 + \
            (employees / 1000) * 1.5 + np.random.normal(0, 0.6, N_FIRMS)
iot_usage = np.clip(iot_usage, 1, 5)

# R&D Expenditure (% of revenue)
rd_spend_pct = has_fdi * 2.5 + (innovation_capability - 3) * 1.2 + \
               np.random.exponential(1.5, N_FIRMS)
rd_spend_pct = np.clip(rd_spend_pct, 0, 15)

# Financial Resources - Liquidity Ratio
liquidity_ratio = 1.2 + base_fdi_effect * 0.5 + \
                  (firm_age / 40) * 0.3 + np.random.normal(0, 0.4, N_FIRMS)
liquidity_ratio = np.clip(liquidity_ratio, 0.3, 4.0)

# ============================================================================
# 4. GOVERNMENT POLICY PERCEPTION (Likert 1-7)
# ============================================================================

# Overall policy environment (affects all policy variables)
policy_environment = np.random.normal(4, 1.2, N_FIRMS)

# Tax Incentives Effectiveness
tax_incentives = policy_environment + np.random.normal(0, 1, N_FIRMS)
tax_incentives = np.clip(tax_incentives, 1, 7)

# Regulatory Stability
regulatory_stability = policy_environment - 0.3 + np.random.normal(0, 1.2, N_FIRMS)
regulatory_stability = np.clip(regulatory_stability, 1, 7)

# Infrastructure Support
infrastructure_support = policy_environment - 0.5 + np.random.normal(0, 1, N_FIRMS)
infrastructure_support = np.clip(infrastructure_support, 1, 7)

# Corruption Experience (reverse coded - higher = more corruption)
corruption_experience = 8 - policy_environment + np.random.normal(0, 1.3, N_FIRMS)
corruption_experience = np.clip(corruption_experience, 1, 7)

# Policy Effectiveness Index (composite)
policy_effectiveness_index = (
    tax_incentives * 0.3 + 
    regulatory_stability * 0.3 + 
    infrastructure_support * 0.25 + 
    (8 - corruption_experience) * 0.15
)

# ============================================================================
# 5. FIRM PERFORMANCE INDICATORS
# Performance is influenced by FDI, resources, and moderated by policy
# ============================================================================

# Calculate composite FDI effect
fdi_composite = (knowledge_absorption * 0.3 + task_performance * 0.3 + 
                 innovation_capability * 0.4) / 5  # Normalize to 0-1

# Calculate resource composite
resource_composite = (skilled_labor_ratio / 100 * 0.3 + 
                     iot_usage / 5 * 0.3 + 
                     rd_spend_pct / 15 * 0.2 +
                     liquidity_ratio / 4 * 0.2)

# Policy moderating effect (normalized)
policy_moderator = policy_effectiveness_index / 7

# Return on Investment (ROI %)
roi_base = 5 + fdi_composite * 15 + resource_composite * 10
roi_moderated = roi_base * (0.7 + policy_moderator * 0.6)
roi = roi_moderated + np.random.normal(0, 4, N_FIRMS)
roi = np.clip(roi, -5, 40)

# Return on Assets (ROA %)
roa_base = 3 + fdi_composite * 12 + resource_composite * 8
roa_moderated = roa_base * (0.7 + policy_moderator * 0.6)
roa = roa_moderated + np.random.normal(0, 3, N_FIRMS)
roa = np.clip(roa, -3, 35)

# Export Intensity (% of revenue from exports)
export_base = has_fdi * 15 + fdi_composite * 20 + (employees / 1000) * 10
export_moderated = export_base * (0.6 + policy_moderator * 0.8)
export_intensity = export_moderated + np.random.exponential(5, N_FIRMS)
export_intensity = np.clip(export_intensity, 0, 85)

# Market Share (% in primary market segment)
market_share_base = (employees / 1000) * 15 + fdi_composite * 10 + firm_age * 0.3
market_share = market_share_base + np.random.normal(0, 5, N_FIRMS)
market_share = np.clip(market_share, 0.5, 45)

# Operational Efficiency (Likert 1-5)
efficiency_base = 2.5 + fdi_composite * 1.5 + resource_composite * 1.0
efficiency_moderated = efficiency_base * (0.8 + policy_moderator * 0.4)
operational_efficiency = efficiency_moderated + np.random.normal(0, 0.5, N_FIRMS)
operational_efficiency = np.clip(operational_efficiency, 1, 5)

# ============================================================================
# 6. ADDITIONAL SURVEY METADATA
# ============================================================================

# Location within Lagos
locations = np.random.choice(
    ['Ikeja Industrial Estate', 'Apapa', 'Ilupeju', 'Isolo', 'Oshodi', 
     'Agbara', 'Ikorodu', 'Alimosho', 'Other'],
    size=N_FIRMS,
    p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05]
)

# Years of FDI involvement (0 if no FDI)
years_fdi = np.where(has_fdi == 1, 
                     np.random.randint(1, min(20, firm_age.max()), N_FIRMS),
                     0)
years_fdi = np.minimum(years_fdi, firm_age)

# Survey completion date
survey_dates = pd.date_range('2024-01-15', '2024-03-30', periods=N_FIRMS)

# ============================================================================
# 7. CREATE DATAFRAME
# ============================================================================

data = pd.DataFrame({
    # Identifiers
    'firm_id': firm_ids,
    'survey_date': survey_dates,
    'location': locations,
    
    # Control Variables
    'firm_age_years': firm_age,
    'firm_size_category': firm_size_category,
    'num_employees': employees,
    'total_assets_million_ngn': np.round(assets_million_ngn, 2),
    'subsector': subsectors,
    'ownership_type': ownership_type,
    'has_fdi': has_fdi,
    'years_fdi_involvement': years_fdi,
    
    # FDI Constructs (1-5 Likert)
    'knowledge_absorption': np.round(knowledge_absorption, 2),
    'task_performance': np.round(task_performance, 2),
    'innovation_capability': np.round(innovation_capability, 2),
    
    # Firm Resources
    'skilled_labor_ratio_pct': np.round(skilled_labor_ratio, 1),
    'iot_automation_use': np.round(iot_usage, 2),
    'rd_expenditure_pct_revenue': np.round(rd_spend_pct, 2),
    'liquidity_ratio': np.round(liquidity_ratio, 2),
    
    # Government Policy Perception (1-7 Likert)
    'tax_incentives_effectiveness': np.round(tax_incentives, 2),
    'regulatory_stability': np.round(regulatory_stability, 2),
    'infrastructure_support': np.round(infrastructure_support, 2),
    'corruption_experience': np.round(corruption_experience, 2),
    'policy_effectiveness_index': np.round(policy_effectiveness_index, 2),
    
    # Firm Performance
    'roi_percent': np.round(roi, 2),
    'roa_percent': np.round(roa, 2),
    'export_intensity_pct': np.round(export_intensity, 2),
    'market_share_pct': np.round(market_share, 2),
    'operational_efficiency': np.round(operational_efficiency, 2),
})

# ============================================================================
# 8. SAVE DATASET
# ============================================================================

# Save as CSV
csv_filename = 'fdi_food_processing_lagos_dataset.csv'
data.to_csv(csv_filename, index=False)
print(f"\n✅ Dataset saved as: {csv_filename}")
print(f"   Total firms: {len(data)}")
print(f"   Total variables: {len(data.columns)}")
print(f"   Firms with FDI: {data['has_fdi'].sum()} ({data['has_fdi'].mean()*100:.1f}%)")

# Save as Excel with multiple sheets
excel_filename = 'fdi_food_processing_lagos_dataset.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    data.to_excel(writer, sheet_name='Survey Data', index=False)
    
    # Summary statistics sheet
    summary = data.describe(include='all').T
    summary.to_excel(writer, sheet_name='Summary Statistics')
    
print(f"✅ Excel file saved as: {excel_filename}")

# ============================================================================
# 9. DISPLAY SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

print("\n📊 Sample Size by Category:")
print(f"  - Firm Size: {data['firm_size_category'].value_counts().to_dict()}")
print(f"  - Ownership Type: {data['ownership_type'].value_counts().to_dict()}")
print(f"  - FDI Status: No FDI={sum(data['has_fdi']==0)}, Has FDI={sum(data['has_fdi']==1)}")

print("\n📈 Key Performance Indicators (Mean ± SD):")
print(f"  - ROI: {data['roi_percent'].mean():.2f}% ± {data['roi_percent'].std():.2f}")
print(f"  - ROA: {data['roa_percent'].mean():.2f}% ± {data['roa_percent'].std():.2f}")
print(f"  - Export Intensity: {data['export_intensity_pct'].mean():.2f}% ± {data['export_intensity_pct'].std():.2f}")
print(f"  - Market Share: {data['market_share_pct'].mean():.2f}% ± {data['market_share_pct'].std():.2f}")

print("\n🔬 FDI Constructs (Mean ± SD, 1-5 scale):")
print(f"  - Knowledge Absorption: {data['knowledge_absorption'].mean():.2f} ± {data['knowledge_absorption'].std():.2f}")
print(f"  - Task Performance: {data['task_performance'].mean():.2f} ± {data['task_performance'].std():.2f}")
print(f"  - Innovation Capability: {data['innovation_capability'].mean():.2f} ± {data['innovation_capability'].std():.2f}")

print("\n🏛️ Policy Perception (Mean ± SD, 1-7 scale):")
print(f"  - Tax Incentives: {data['tax_incentives_effectiveness'].mean():.2f} ± {data['tax_incentives_effectiveness'].std():.2f}")
print(f"  - Regulatory Stability: {data['regulatory_stability'].mean():.2f} ± {data['regulatory_stability'].std():.2f}")
print(f"  - Infrastructure Support: {data['infrastructure_support'].mean():.2f} ± {data['infrastructure_support'].std():.2f}")
print(f"  - Corruption Experience: {data['corruption_experience'].mean():.2f} ± {data['corruption_experience'].std():.2f}")

print("\n💡 FDI vs Non-FDI Performance Comparison:")
fdi_firms = data[data['has_fdi'] == 1]
non_fdi_firms = data[data['has_fdi'] == 0]
print(f"  - Mean ROI: FDI={fdi_firms['roi_percent'].mean():.2f}%, Non-FDI={non_fdi_firms['roi_percent'].mean():.2f}%")
print(f"  - Mean ROA: FDI={fdi_firms['roa_percent'].mean():.2f}%, Non-FDI={non_fdi_firms['roa_percent'].mean():.2f}%")
print(f"  - Mean Export: FDI={fdi_firms['export_intensity_pct'].mean():.2f}%, Non-FDI={non_fdi_firms['export_intensity_pct'].mean():.2f}%")
print(f"  - Mean Innovation: FDI={fdi_firms['innovation_capability'].mean():.2f}, Non-FDI={non_fdi_firms['innovation_capability'].mean():.2f}")

print("\n" + "="*80)
print("Dataset generation complete! 🎉")
print("="*80)
