#!/usr/bin/env python3
"""
Generate Comprehensive Descriptive Statistics Report
FDI Performance Study - Lagos Food Processing Firms
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load dataset
df = pd.read_csv('fdi_lagos_survey_data.csv')

# Create output report
report = []

def add_section(title, level=1):
    """Add section header"""
    if level == 1:
        report.append(f"\n{'='*80}\n{title}\n{'='*80}\n")
    else:
        report.append(f"\n{'-'*80}\n{title}\n{'-'*80}\n")

def add_text(text):
    """Add text to report"""
    report.append(text + "\n")

# ============================================================================
# REPORT HEADER
# ============================================================================

add_text("="*80)
add_text("DESCRIPTIVE STATISTICS REPORT")
add_text("The Influence of Foreign Direct Investment on the Performance of")
add_text("Food Processing Firms in Lagos, Nigeria")
add_text("="*80)
add_text(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
add_text(f"Dataset: fdi_lagos_survey_data.csv")
add_text(f"Sample Size: {len(df)} firms")
add_text("="*80)

# ============================================================================
# 1. SAMPLE CHARACTERISTICS
# ============================================================================

add_section("1. SAMPLE CHARACTERISTICS")

# Firm size distribution
add_text("\n1.1 FIRM SIZE DISTRIBUTION")
size_dist = df['firm_size'].value_counts()
for size, count in size_dist.items():
    add_text(f"  {size}: {count} firms ({count/len(df)*100:.1f}%)")

# FDI engagement
add_text("\n1.2 FDI ENGAGEMENT")
fdi_dist = df['has_fdi'].value_counts()
for status, count in fdi_dist.items():
    add_text(f"  {status}: {count} firms ({count/len(df)*100:.1f}%)")

# Ownership structure
add_text("\n1.3 OWNERSHIP STRUCTURE")
own_dist = df['ownership_type'].value_counts()
for own_type, count in own_dist.items():
    add_text(f"  {own_type}: {count} firms ({count/len(df)*100:.1f}%)")

# Years of operation
add_text("\n1.4 YEARS OF OPERATION")
add_text(f"  Mean: {df['years_operation'].mean():.1f} years")
add_text(f"  Median: {df['years_operation'].median():.1f} years")
add_text(f"  Range: {df['years_operation'].min()}-{df['years_operation'].max()} years")
add_text(f"  Std Dev: {df['years_operation'].std():.1f} years")

# ============================================================================
# 2. FDI CHARACTERISTICS
# ============================================================================

add_section("2. FDI CHARACTERISTICS")

fdi_firms = df[df['has_fdi'] == 'Yes']

add_text(f"\nTotal firms with FDI: {len(fdi_firms)} ({len(fdi_firms)/len(df)*100:.1f}%)")

add_text("\n2.1 FDI TYPE DISTRIBUTION")
fdi_type_dist = fdi_firms['fdi_type'].value_counts()
for ftype, count in fdi_type_dist.items():
    add_text(f"  {ftype}: {count} firms ({count/len(fdi_firms)*100:.1f}%)")

add_text("\n2.2 YEARS WITH FDI PARTNERSHIP")
add_text(f"  Mean: {fdi_firms['years_fdi'].mean():.1f} years")
add_text(f"  Median: {fdi_firms['years_fdi'].median():.1f} years")
add_text(f"  Range: {fdi_firms['years_fdi'].min()}-{fdi_firms['years_fdi'].max()} years")

# ============================================================================
# 3. KNOWLEDGE ABSORPTION
# ============================================================================

add_section("3. KNOWLEDGE ABSORPTION (5-point Likert Scale)")

ka_vars = ['ka1_technical_manuals', 'ka2_staff_training', 
           'ka3_adapt_technology', 'ka4_commercialize_knowledge']

ka_labels = {
    'ka1_technical_manuals': 'KA1: Acquire technical manuals',
    'ka2_staff_training': 'KA2: Staff training from partners',
    'ka3_adapt_technology': 'KA3: Adapt foreign technology',
    'ka4_commercialize_knowledge': 'KA4: Commercialize knowledge'
}

for var in ka_vars:
    add_text(f"\n{ka_labels[var]}:")
    add_text(f"  Overall Mean: {df[var].mean():.2f} (SD: {df[var].std():.2f})")
    add_text(f"  FDI firms: {df[df['has_fdi']=='Yes'][var].mean():.2f}")
    add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No'][var].mean():.2f}")
    
    # Distribution
    dist = df[var].value_counts().sort_index()
    add_text(f"  Distribution: {dict(dist)}")

# Composite score
df['KA_score'] = df[ka_vars].mean(axis=1)
add_text(f"\nKNOWLEDGE ABSORPTION COMPOSITE SCORE:")
add_text(f"  Overall Mean: {df['KA_score'].mean():.2f} (SD: {df['KA_score'].std():.2f})")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['KA_score'].mean():.2f}")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['KA_score'].mean():.2f}")

# ============================================================================
# 4. TASK PERFORMANCE
# ============================================================================

add_section("4. TASK PERFORMANCE (5-point Likert Scale)")

tp_vars = ['tp1_production_efficiency', 'tp2_quality_control', 
           'tp3_order_fulfillment', 'tp4_employee_productivity']

tp_labels = {
    'tp1_production_efficiency': 'TP1: Production efficiency',
    'tp2_quality_control': 'TP2: Quality control',
    'tp3_order_fulfillment': 'TP3: Order fulfillment',
    'tp4_employee_productivity': 'TP4: Employee productivity'
}

for var in tp_vars:
    add_text(f"\n{tp_labels[var]}:")
    add_text(f"  Overall Mean: {df[var].mean():.2f} (SD: {df[var].std():.2f})")
    add_text(f"  FDI firms: {df[df['has_fdi']=='Yes'][var].mean():.2f}")
    add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No'][var].mean():.2f}")

# Composite score
df['TP_score'] = df[tp_vars].mean(axis=1)
add_text(f"\nTASK PERFORMANCE COMPOSITE SCORE:")
add_text(f"  Overall Mean: {df['TP_score'].mean():.2f} (SD: {df['TP_score'].std():.2f})")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['TP_score'].mean():.2f}")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['TP_score'].mean():.2f}")

# ============================================================================
# 5. INNOVATION METRICS
# ============================================================================

add_section("5. INNOVATION METRICS")

add_text("\n5.1 R&D SPENDING (% of Revenue)")
add_text(f"  Overall Mean: {df['rd_spending_pct'].mean():.2f}% (SD: {df['rd_spending_pct'].std():.2f}%)")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['rd_spending_pct'].mean():.2f}%")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['rd_spending_pct'].mean():.2f}%")
add_text(f"  Range: {df['rd_spending_pct'].min():.2f}% - {df['rd_spending_pct'].max():.2f}%")

add_text("\n5.2 NEW PRODUCTS LAUNCHED (Past 3 years)")
add_text(f"  Overall Mean: {df['new_products_3yrs'].mean():.1f} (SD: {df['new_products_3yrs'].std():.1f})")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['new_products_3yrs'].mean():.1f}")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['new_products_3yrs'].mean():.1f}")
add_text(f"  Range: {df['new_products_3yrs'].min()} - {df['new_products_3yrs'].max()} products")

add_text("\n5.3 PROCESS INNOVATIONS ADOPTED")
innovations = ['innov_iot', 'innov_automation', 'innov_quality_mgmt', 'innov_other']
innov_labels = {
    'innov_iot': 'IoT Systems',
    'innov_automation': 'Automation',
    'innov_quality_mgmt': 'Quality Management',
    'innov_other': 'Other innovations'
}

for innov in innovations:
    count = df[innov].sum()
    fdi_count = df[df['has_fdi']=='Yes'][innov].sum()
    add_text(f"  {innov_labels[innov]}: {count} firms ({count/len(df)*100:.1f}%)")
    add_text(f"    FDI firms: {fdi_count}/{len(fdi_firms)} ({fdi_count/len(fdi_firms)*100:.1f}%)")

# ============================================================================
# 6. FIRM RESOURCES
# ============================================================================

add_section("6. FIRM RESOURCES")

add_text("\n6.1 HUMAN RESOURCES")
add_text(f"  Skilled Workforce (%):")
add_text(f"    Overall: {df['skilled_workforce_pct'].mean():.1f}% (SD: {df['skilled_workforce_pct'].std():.1f}%)")
add_text(f"    SMEs: {df[df['firm_size']=='SME']['skilled_workforce_pct'].mean():.1f}%")
add_text(f"    Large firms: {df[df['firm_size']=='Large']['skilled_workforce_pct'].mean():.1f}%")

add_text(f"\n  Training Hours per Employee (annually):")
add_text(f"    Overall: {df['training_hours_per_emp'].mean():.1f} hours (SD: {df['training_hours_per_emp'].std():.1f})")
add_text(f"    FDI firms: {df[df['has_fdi']=='Yes']['training_hours_per_emp'].mean():.1f} hours")
add_text(f"    Non-FDI firms: {df[df['has_fdi']=='No']['training_hours_per_emp'].mean():.1f} hours")

add_text("\n6.2 TECHNOLOGICAL RESOURCES")
modern_eq = df['modern_equipment'].value_counts()
add_text(f"  Modern Equipment Usage:")
for eq, count in modern_eq.items():
    add_text(f"    {eq}: {count} firms ({count/len(df)*100:.1f}%)")

add_text(f"\n  Machinery Age:")
add_text(f"    Mean: {df['machinery_age_years'].mean():.1f} years (SD: {df['machinery_age_years'].std():.1f})")
add_text(f"    Modern equipment users: {df[df['modern_equipment']=='Yes']['machinery_age_years'].mean():.1f} years")
add_text(f"    Others: {df[df['modern_equipment']=='No']['machinery_age_years'].mean():.1f} years")

add_text("\n6.3 FINANCIAL RESOURCES")
credit_access = df['access_credit'].value_counts()
add_text(f"  Access to Credit:")
for level, count in credit_access.items():
    add_text(f"    {level}: {count} firms ({count/len(df)*100:.1f}%)")

add_text(f"\n  Reinvestment Rate:")
add_text(f"    Overall: {df['reinvestment_rate_pct'].mean():.1f}% (SD: {df['reinvestment_rate_pct'].std():.1f}%)")
add_text(f"    SMEs: {df[df['firm_size']=='SME']['reinvestment_rate_pct'].mean():.1f}%")
add_text(f"    Large firms: {df[df['firm_size']=='Large']['reinvestment_rate_pct'].mean():.1f}%")

# ============================================================================
# 7. GOVERNMENT POLICY PERCEPTION
# ============================================================================

add_section("7. GOVERNMENT POLICY PERCEPTION (7-point Likert Scale)")

gp_vars = ['gp1_tax_incentives', 'gp2_regulatory_stability', 
           'gp3_infrastructure', 'gp4_permits_ease']

gp_labels = {
    'gp1_tax_incentives': 'GP1: Tax incentives effectiveness',
    'gp2_regulatory_stability': 'GP2: Regulatory stability',
    'gp3_infrastructure': 'GP3: Infrastructure support',
    'gp4_permits_ease': 'GP4: Ease of obtaining permits'
}

for var in gp_vars:
    add_text(f"\n{gp_labels[var]}:")
    add_text(f"  Mean: {df[var].mean():.2f} (SD: {df[var].std():.2f})")
    add_text(f"  Median: {df[var].median():.0f}")
    add_text(f"  Range: {df[var].min()}-{df[var].max()}")

# Composite score
df['GP_score'] = df[gp_vars].mean(axis=1)
add_text(f"\nGOVERNMENT POLICY COMPOSITE SCORE:")
add_text(f"  Mean: {df['GP_score'].mean():.2f} (SD: {df['GP_score'].std():.2f})")

# ============================================================================
# 8. PERFORMANCE METRICS
# ============================================================================

add_section("8. PERFORMANCE METRICS")

add_text("\n8.1 FINANCIAL PERFORMANCE")

add_text(f"\nReturn on Investment (ROI):")
add_text(f"  Overall: {df['avg_roi_pct'].mean():.2f}% (SD: {df['avg_roi_pct'].std():.2f}%)")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['avg_roi_pct'].mean():.2f}%")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['avg_roi_pct'].mean():.2f}%")
add_text(f"  SMEs: {df[df['firm_size']=='SME']['avg_roi_pct'].mean():.2f}%")
add_text(f"  Large firms: {df[df['firm_size']=='Large']['avg_roi_pct'].mean():.2f}%")
add_text(f"  Range: {df['avg_roi_pct'].min():.2f}% to {df['avg_roi_pct'].max():.2f}%")

add_text(f"\nReturn on Assets (ROA):")
add_text(f"  Overall: {df['avg_roa_pct'].mean():.2f}% (SD: {df['avg_roa_pct'].std():.2f}%)")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['avg_roa_pct'].mean():.2f}%")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['avg_roa_pct'].mean():.2f}%")
add_text(f"  Range: {df['avg_roa_pct'].min():.2f}% to {df['avg_roa_pct'].max():.2f}%")

add_text(f"\nExport Intensity:")
add_text(f"  Overall: {df['export_intensity_pct'].mean():.1f}% (SD: {df['export_intensity_pct'].std():.1f}%)")
add_text(f"  FDI firms: {df[df['has_fdi']=='Yes']['export_intensity_pct'].mean():.1f}%")
add_text(f"  Non-FDI firms: {df[df['has_fdi']=='No']['export_intensity_pct'].mean():.1f}%")

add_text("\n8.2 OPERATIONAL PERFORMANCE")

add_text(f"\nCapacity Utilization:")
add_text(f"  Overall: {df['capacity_utilization_pct'].mean():.1f}% (SD: {df['capacity_utilization_pct'].std():.1f}%)")
add_text(f"  SMEs: {df[df['firm_size']=='SME']['capacity_utilization_pct'].mean():.1f}%")
add_text(f"  Large firms: {df[df['firm_size']=='Large']['capacity_utilization_pct'].mean():.1f}%")

add_text(f"\nMarket Share in Lagos:")
add_text(f"  Overall: {df['market_share_pct'].mean():.2f}% (SD: {df['market_share_pct'].std():.2f}%)")
add_text(f"  SMEs: {df[df['firm_size']=='SME']['market_share_pct'].mean():.2f}%")
add_text(f"  Large firms: {df[df['firm_size']=='Large']['market_share_pct'].mean():.2f}%")

# ============================================================================
# 9. COMPARATIVE ANALYSIS
# ============================================================================

add_section("9. COMPARATIVE ANALYSIS: FDI vs NON-FDI FIRMS")

comparison_vars = [
    ('avg_roi_pct', 'ROI (%)'),
    ('avg_roa_pct', 'ROA (%)'),
    ('capacity_utilization_pct', 'Capacity Utilization (%)'),
    ('export_intensity_pct', 'Export Intensity (%)'),
    ('rd_spending_pct', 'R&D Spending (%)'),
    ('new_products_3yrs', 'New Products (count)'),
    ('KA_score', 'Knowledge Absorption'),
    ('TP_score', 'Task Performance'),
]

add_text(f"\n{'Variable':<30} {'FDI Firms':<15} {'Non-FDI':<15} {'Difference':<15}")
add_text("-" * 80)

for var, label in comparison_vars:
    fdi_mean = df[df['has_fdi']=='Yes'][var].mean()
    non_fdi_mean = df[df['has_fdi']=='No'][var].mean()
    diff = fdi_mean - non_fdi_mean
    add_text(f"{label:<30} {fdi_mean:>12.2f}   {non_fdi_mean:>12.2f}   {diff:>12.2f}")

# ============================================================================
# 10. CORRELATION MATRIX
# ============================================================================

add_section("10. CORRELATION MATRIX (Key Variables)")

# Create binary FDI variable
df['fdi_binary'] = (df['has_fdi'] == 'Yes').astype(int)

corr_vars = ['fdi_binary', 'KA_score', 'TP_score', 'rd_spending_pct', 
             'avg_roi_pct', 'avg_roa_pct', 'capacity_utilization_pct', 'GP_score']

corr_matrix = df[corr_vars].corr()

add_text("\n" + corr_matrix.to_string())

# ============================================================================
# SAVE REPORT
# ============================================================================

report_text = ''.join(report)
with open('descriptive_statistics_report.txt', 'w') as f:
    f.write(report_text)

print("✅ Descriptive statistics report generated!")
print("   Saved to: descriptive_statistics_report.txt")
print(f"   Report length: {len(report_text)} characters")

# Also save as CSV for easy analysis
summary_stats = pd.DataFrame({
    'Variable': ['Sample Size', 'SMEs', 'Large Firms', 'FDI Firms', 'Non-FDI Firms',
                 'Mean ROI (%)', 'Mean ROA (%)', 'Mean Capacity Util (%)',
                 'Mean KA Score', 'Mean TP Score', 'Mean R&D Spending (%)'],
    'Value': [len(df), 
              sum(df['firm_size']=='SME'),
              sum(df['firm_size']=='Large'),
              sum(df['has_fdi']=='Yes'),
              sum(df['has_fdi']=='No'),
              df['avg_roi_pct'].mean(),
              df['avg_roa_pct'].mean(),
              df['capacity_utilization_pct'].mean(),
              df['KA_score'].mean(),
              df['TP_score'].mean(),
              df['rd_spending_pct'].mean()]
})

summary_stats.to_csv('summary_statistics.csv', index=False)
print("   Summary statistics saved to: summary_statistics.csv")
