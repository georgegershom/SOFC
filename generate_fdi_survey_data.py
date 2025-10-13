#!/usr/bin/env python3
"""
Generate Synthetic Survey Data for FDI Performance Study
Topic: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Sample size (70% response rate of 300 target)
N_SME = 140  # 70% of 200
N_LARGE = 70  # 70% of 100
N_TOTAL = N_SME + N_LARGE

print(f"Generating synthetic survey data for {N_TOTAL} firms...")
print(f"  - SMEs: {N_SME}")
print(f"  - Large firms: {N_LARGE}")

# Initialize data dictionary
data = {}

# ============================================================================
# SECTION A: FIRM BACKGROUND
# ============================================================================

# Firm codes
data['firm_code'] = [f"FIRM{str(i).zfill(3)}" for i in range(1, N_TOTAL + 1)]

# Survey dates (spread over 3 months: July-September 2024)
start_date = datetime(2024, 7, 1)
end_date = datetime(2024, 9, 30)
date_range = (end_date - start_date).days
data['survey_date'] = [(start_date + timedelta(days=random.randint(0, date_range))).strftime('%d/%m/%Y') 
                        for _ in range(N_TOTAL)]

# Firm size (SME vs Large)
data['firm_size'] = ['SME'] * N_SME + ['Large'] * N_LARGE

# Years of operation (5-40 years, skewed toward older firms)
data['years_operation'] = np.concatenate([
    np.random.gamma(3, 5, N_SME).astype(int) + 5,  # SMEs: 5-30 years typically
    np.random.gamma(4, 6, N_LARGE).astype(int) + 8  # Large: 8-40 years typically
])
data['years_operation'] = np.clip(data['years_operation'], 5, 40)

# Number of employees (categorical)
emp_categories = {
    'SME': ['1-50', '51-250'],
    'Large': ['251-500', '500+']
}
data['num_employees'] = (
    [random.choice(emp_categories['SME']) for _ in range(N_SME)] +
    [random.choice(emp_categories['Large']) for _ in range(N_LARGE)]
)

# Annual revenue (categorical, correlated with size)
revenue_dist = {
    'SME': ['<50M', '50M-500M'],
    'Large': ['500M-5B', '>5B']
}
data['annual_revenue'] = (
    [random.choices(revenue_dist['SME'], weights=[0.3, 0.7])[0] for _ in range(N_SME)] +
    [random.choices(revenue_dist['Large'], weights=[0.6, 0.4])[0] for _ in range(N_LARGE)]
)

# Ownership type
ownership_weights = {
    'SME': [0.65, 0.20, 0.15],  # More local ownership in SMEs
    'Large': [0.30, 0.45, 0.25]  # More FDI in large firms
}
data['ownership_type'] = (
    [random.choices(['Local', 'Foreign-owned', 'Joint venture'], 
                    weights=ownership_weights['SME'])[0] for _ in range(N_SME)] +
    [random.choices(['Local', 'Foreign-owned', 'Joint venture'], 
                    weights=ownership_weights['Large'])[0] for _ in range(N_LARGE)]
)

# FDI Engagement (Yes if Foreign-owned or Joint venture)
data['has_fdi'] = ['Yes' if ot in ['Foreign-owned', 'Joint venture'] else 
                   random.choices(['Yes', 'No'], weights=[0.2, 0.8])[0] 
                   for ot in data['ownership_type']]

# Type of FDI (only for firms with FDI)
fdi_types = ['Equity', 'Joint venture', 'Technology transfer', 'Management contract']
data['fdi_type'] = [random.choice(fdi_types) if fdi == 'Yes' else '' 
                    for fdi in data['has_fdi']]

# Years with FDI partnership (0-15 years, less than years of operation)
data['years_fdi'] = [
    random.randint(1, min(15, data['years_operation'][i]-2)) if data['has_fdi'][i] == 'Yes' else 0
    for i in range(N_TOTAL)
]

# ============================================================================
# SECTION B: KNOWLEDGE ABSORPTION (Zahra & George Scale)
# 1=Strongly Disagree to 5=Strongly Agree
# Higher scores for firms with FDI
# ============================================================================

def generate_likert_5(n, has_fdi_list, base_mean=3.0, fdi_boost=0.8):
    """Generate 5-point Likert scale with FDI effect"""
    scores = []
    for has_fdi in has_fdi_list:
        mean = base_mean + (fdi_boost if has_fdi == 'Yes' else 0)
        score = np.random.normal(mean, 0.8)
        score = int(np.clip(np.round(score), 1, 5))
        scores.append(score)
    return scores

data['ka1_technical_manuals'] = generate_likert_5(N_TOTAL, data['has_fdi'], 2.8, 1.0)
data['ka2_staff_training'] = generate_likert_5(N_TOTAL, data['has_fdi'], 2.9, 1.1)
data['ka3_adapt_technology'] = generate_likert_5(N_TOTAL, data['has_fdi'], 3.1, 0.9)
data['ka4_commercialize_knowledge'] = generate_likert_5(N_TOTAL, data['has_fdi'], 2.7, 1.0)

# ============================================================================
# SECTION C: TASK PERFORMANCE (Koopmans et al. Scale)
# 1=Very Poor to 5=Excellent
# Correlated with knowledge absorption and FDI
# ============================================================================

data['tp1_production_efficiency'] = generate_likert_5(N_TOTAL, data['has_fdi'], 3.2, 0.7)
data['tp2_quality_control'] = generate_likert_5(N_TOTAL, data['has_fdi'], 3.3, 0.6)
data['tp3_order_fulfillment'] = generate_likert_5(N_TOTAL, data['has_fdi'], 3.1, 0.7)
data['tp4_employee_productivity'] = generate_likert_5(N_TOTAL, data['has_fdi'], 3.0, 0.8)

# ============================================================================
# SECTION D: INNOVATION (OECD Oslo Manual)
# ============================================================================

# R&D spending as % of revenue (0-8%, higher for firms with FDI)
data['rd_spending_pct'] = [
    round(np.random.gamma(2, 0.8) if fdi == 'Yes' else np.random.gamma(1.5, 0.4), 2)
    for fdi in data['has_fdi']
]
data['rd_spending_pct'] = np.clip(data['rd_spending_pct'], 0, 8)

# New products launched (past 3 years): 0-15 products
data['new_products_3yrs'] = [
    int(np.random.poisson(5)) if fdi == 'Yes' else int(np.random.poisson(2))
    for fdi in data['has_fdi']
]
data['new_products_3yrs'] = np.clip(data['new_products_3yrs'], 0, 15)

# Process innovations (binary for each type)
data['innov_iot'] = [random.choices([0, 1], weights=[0.7, 0.3] if fdi == 'Yes' else [0.85, 0.15])[0] 
                     for fdi in data['has_fdi']]
data['innov_automation'] = [random.choices([0, 1], weights=[0.5, 0.5] if fdi == 'Yes' else [0.75, 0.25])[0] 
                            for fdi in data['has_fdi']]
data['innov_quality_mgmt'] = [random.choices([0, 1], weights=[0.3, 0.7] if fdi == 'Yes' else [0.6, 0.4])[0] 
                              for fdi in data['has_fdi']]
data['innov_other'] = [random.choices([0, 1], weights=[0.8, 0.2])[0] for _ in range(N_TOTAL)]

# ============================================================================
# SECTION E: FIRM RESOURCES
# ============================================================================

# Human Resources
# % of skilled workforce (20-90%)
data['skilled_workforce_pct'] = [
    round(np.random.normal(65, 12) if size == 'Large' else np.random.normal(52, 15), 1)
    for size in data['firm_size']
]
data['skilled_workforce_pct'] = np.clip(data['skilled_workforce_pct'], 20, 90)

# Annual training hours per employee (10-200 hours)
data['training_hours_per_emp'] = [
    int(np.random.gamma(8, 8)) if fdi == 'Yes' else int(np.random.gamma(5, 6))
    for fdi in data['has_fdi']
]
data['training_hours_per_emp'] = np.clip(data['training_hours_per_emp'], 10, 200)

# Technological Resources
# Use of modern equipment
data['modern_equipment'] = [
    random.choices(['Yes', 'No'], weights=[0.8, 0.2] if fdi == 'Yes' else [0.5, 0.5])[0]
    for fdi in data['has_fdi']
]

# Age of primary machinery (1-25 years)
data['machinery_age_years'] = [
    int(np.random.gamma(2, 2)) if equip == 'Yes' else int(np.random.gamma(4, 3))
    for equip in data['modern_equipment']
]
data['machinery_age_years'] = np.clip(data['machinery_age_years'], 1, 25)

# Financial Resources
# Access to credit
credit_weights = {
    'Large': [0.4, 0.45, 0.15],
    'SME': [0.15, 0.50, 0.35]
}
data['access_credit'] = [
    random.choices(['Easy', 'Moderate', 'Difficult'], 
                   weights=credit_weights[size])[0]
    for size in data['firm_size']
]

# Reinvestment rate (5-40%)
data['reinvestment_rate_pct'] = [
    round(np.random.normal(25, 8) if size == 'Large' else np.random.normal(18, 7), 1)
    for size in data['firm_size']
]
data['reinvestment_rate_pct'] = np.clip(data['reinvestment_rate_pct'], 5, 40)

# ============================================================================
# SECTION F: GOVERNMENT POLICY PERCEPTION
# 1=Very Ineffective to 7=Very Effective
# ============================================================================

def generate_likert_7(n, base_mean=4.0, std=1.2):
    """Generate 7-point Likert scale"""
    scores = np.random.normal(base_mean, std, n)
    return np.clip(np.round(scores).astype(int), 1, 7).tolist()

# Government policy perceptions (generally moderate with variation)
data['gp1_tax_incentives'] = generate_likert_7(N_TOTAL, 4.2, 1.3)
data['gp2_regulatory_stability'] = generate_likert_7(N_TOTAL, 3.8, 1.4)
data['gp3_infrastructure'] = generate_likert_7(N_TOTAL, 3.5, 1.3)
data['gp4_permits_ease'] = generate_likert_7(N_TOTAL, 3.6, 1.4)

# ============================================================================
# SECTION G: PERFORMANCE METRICS
# ============================================================================

# Financial Performance (past 3 years)
# ROI (Return on Investment): -5% to 45%
base_roi = []
for i in range(N_TOTAL):
    # Base ROI affected by FDI, size, and knowledge absorption
    roi_base = 12 if data['has_fdi'][i] == 'Yes' else 8
    roi_base += 3 if data['firm_size'][i] == 'Large' else 0
    ka_avg = (data['ka1_technical_manuals'][i] + data['ka2_staff_training'][i] + 
              data['ka3_adapt_technology'][i] + data['ka4_commercialize_knowledge'][i]) / 4
    roi_base += (ka_avg - 3) * 4  # Knowledge absorption effect
    roi = round(np.random.normal(roi_base, 6), 2)
    base_roi.append(roi)
data['avg_roi_pct'] = np.clip(base_roi, -5, 45).tolist()

# ROA (Return on Assets): -3% to 35%
data['avg_roa_pct'] = [round(roi * 0.7 + np.random.normal(0, 2), 2) for roi in data['avg_roi_pct']]
data['avg_roa_pct'] = np.clip(data['avg_roa_pct'], -3, 35).tolist()

# Export intensity: 0-60%
data['export_intensity_pct'] = [
    round(np.random.gamma(3, 5) if fdi == 'Yes' else np.random.gamma(1.5, 3), 1)
    for fdi in data['has_fdi']
]
data['export_intensity_pct'] = np.clip(data['export_intensity_pct'], 0, 60)

# Operational Performance
# Production capacity utilization: 40-95%
data['capacity_utilization_pct'] = [
    round(np.random.normal(78, 10) if size == 'Large' else np.random.normal(68, 12), 1)
    for size in data['firm_size']
]
data['capacity_utilization_pct'] = np.clip(data['capacity_utilization_pct'], 40, 95)

# Market share in Lagos: 0.5-25%
market_share = []
for size in data['firm_size']:
    if size == 'Large':
        ms = np.random.gamma(4, 2)
    else:
        ms = np.random.gamma(2, 1)
    market_share.append(round(ms, 2))
data['market_share_pct'] = np.clip(market_share, 0.5, 25).tolist()

# ============================================================================
# CREATE DATAFRAME AND SAVE
# ============================================================================

df = pd.DataFrame(data)

# Reorder columns logically
column_order = [
    # Identifiers
    'firm_code', 'survey_date', 'firm_size',
    # Section A: Firm Background
    'years_operation', 'num_employees', 'annual_revenue', 'ownership_type',
    'has_fdi', 'fdi_type', 'years_fdi',
    # Section B: Knowledge Absorption
    'ka1_technical_manuals', 'ka2_staff_training', 'ka3_adapt_technology', 
    'ka4_commercialize_knowledge',
    # Section C: Task Performance
    'tp1_production_efficiency', 'tp2_quality_control', 'tp3_order_fulfillment', 
    'tp4_employee_productivity',
    # Section D: Innovation
    'rd_spending_pct', 'new_products_3yrs', 'innov_iot', 'innov_automation', 
    'innov_quality_mgmt', 'innov_other',
    # Section E: Firm Resources
    'skilled_workforce_pct', 'training_hours_per_emp', 'modern_equipment', 
    'machinery_age_years', 'access_credit', 'reinvestment_rate_pct',
    # Section F: Government Policy
    'gp1_tax_incentives', 'gp2_regulatory_stability', 'gp3_infrastructure', 
    'gp4_permits_ease',
    # Section G: Performance
    'avg_roi_pct', 'avg_roa_pct', 'export_intensity_pct', 
    'capacity_utilization_pct', 'market_share_pct'
]

df = df[column_order]

# Save to CSV
output_file = 'fdi_lagos_survey_data.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Dataset saved to: {output_file}")
print(f"   Total responses: {len(df)}")

# Generate summary statistics
print("\n" + "="*80)
print("DATASET SUMMARY STATISTICS")
print("="*80)

print(f"\n1. SAMPLE COMPOSITION:")
print(f"   Total firms: {len(df)}")
print(f"   SMEs: {sum(df['firm_size'] == 'SME')} ({sum(df['firm_size'] == 'SME')/len(df)*100:.1f}%)")
print(f"   Large firms: {sum(df['firm_size'] == 'Large')} ({sum(df['firm_size'] == 'Large')/len(df)*100:.1f}%)")

print(f"\n2. FDI ENGAGEMENT:")
print(f"   Firms with FDI: {sum(df['has_fdi'] == 'Yes')} ({sum(df['has_fdi'] == 'Yes')/len(df)*100:.1f}%)")
print(f"   Firms without FDI: {sum(df['has_fdi'] == 'No')} ({sum(df['has_fdi'] == 'No')/len(df)*100:.1f}%)")

print(f"\n3. OWNERSHIP DISTRIBUTION:")
for ot in df['ownership_type'].unique():
    count = sum(df['ownership_type'] == ot)
    print(f"   {ot}: {count} ({count/len(df)*100:.1f}%)")

print(f"\n4. KEY PERFORMANCE INDICATORS (Mean ± SD):")
print(f"   ROI: {df['avg_roi_pct'].mean():.2f}% ± {df['avg_roi_pct'].std():.2f}%")
print(f"   ROA: {df['avg_roa_pct'].mean():.2f}% ± {df['avg_roa_pct'].std():.2f}%")
print(f"   Capacity Utilization: {df['capacity_utilization_pct'].mean():.1f}% ± {df['capacity_utilization_pct'].std():.1f}%")
print(f"   Export Intensity: {df['export_intensity_pct'].mean():.1f}% ± {df['export_intensity_pct'].std():.1f}%")

print(f"\n5. INNOVATION METRICS (Mean ± SD):")
print(f"   R&D Spending: {df['rd_spending_pct'].mean():.2f}% ± {df['rd_spending_pct'].std():.2f}%")
print(f"   New Products (3 yrs): {df['new_products_3yrs'].mean():.1f} ± {df['new_products_3yrs'].std():.1f}")
print(f"   IoT Adoption: {sum(df['innov_iot'])} firms ({sum(df['innov_iot'])/len(df)*100:.1f}%)")
print(f"   Automation: {sum(df['innov_automation'])} firms ({sum(df['innov_automation'])/len(df)*100:.1f}%)")

print(f"\n6. KNOWLEDGE ABSORPTION (Mean scores, 1-5 scale):")
ka_cols = ['ka1_technical_manuals', 'ka2_staff_training', 'ka3_adapt_technology', 'ka4_commercialize_knowledge']
for col in ka_cols:
    print(f"   {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")

print(f"\n7. GOVERNMENT POLICY PERCEPTION (Mean scores, 1-7 scale):")
gp_cols = ['gp1_tax_incentives', 'gp2_regulatory_stability', 'gp3_infrastructure', 'gp4_permits_ease']
for col in gp_cols:
    print(f"   {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")

print("\n" + "="*80)
print("Dataset generation complete!")
print("="*80)
