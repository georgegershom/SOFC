#!/usr/bin/env python3
"""
Synthetic Data Generation for FDI Study
The Influence of Foreign Direct Investment on Food Processing Firms in Lagos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=============================================================================")
print("SYNTHETIC DATA GENERATION FOR FDI STUDY")
print("The Influence of Foreign Direct Investment on Food Processing Firms in Lagos")
print("=============================================================================\n")

# =============================================================================
# SAMPLE PARAMETERS
# =============================================================================

n_sme = 200      # SME firms
n_large = 100    # Large firms
n_total = 300    # Total sample

print(f"Sample Parameters:")
print(f"- SME firms: {n_sme}")
print(f"- Large firms: {n_large}")
print(f"- Total sample: {n_total}\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_likert(n, mean_val, sd_val, min_val=1, max_val=5):
    """Generate Likert scale responses"""
    values = np.random.normal(mean_val, sd_val, n)
    values = np.clip(np.round(values), min_val, max_val).astype(int)
    return values

def generate_7point(n, mean_val, sd_val, min_val=1, max_val=7):
    """Generate 7-point scale responses"""
    values = np.random.normal(mean_val, sd_val, n)
    values = np.clip(np.round(values), min_val, max_val).astype(int)
    return values

def generate_percentage(n, mean_val, sd_val, min_val=0, max_val=100):
    """Generate realistic percentages"""
    values = np.random.normal(mean_val, sd_val, n)
    values = np.clip(values, min_val, max_val)
    return np.round(values, 1)

def generate_correlated_likert(n, means, correlation_matrix, min_val=1, max_val=5):
    """Generate correlated Likert scale responses"""
    # Generate correlated normal variables
    mvn = np.random.multivariate_normal(means, correlation_matrix, n)
    # Convert to Likert scale
    likert_data = np.clip(np.round(mvn), min_val, max_val).astype(int)
    return likert_data

# =============================================================================
# GENERATE FIRM CHARACTERISTICS
# =============================================================================

print("Generating firm characteristics...")

# Create firm IDs and basic structure
firm_ids = [f"SME{i:03d}" for i in range(1, n_sme + 1)] + [f"LRG{i:03d}" for i in range(1, n_large + 1)]
firm_size = ["SME"] * n_sme + ["Large"] * n_large

# Survey dates
start_date = datetime(2024, 1, 15)
end_date = datetime(2024, 6, 30)
date_range = (end_date - start_date).days
survey_dates = [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(n_total)]

# Years of operation (correlated with firm size)
years_operation = []
for size in firm_size:
    if size == "SME":
        probs = np.exp(-0.1 * (np.array(range(3, 26)) - 8))
        probs = probs / probs.sum()  # Normalize probabilities
        years = np.random.choice(range(3, 26), p=probs)
    else:
        probs = np.exp(-0.05 * (np.array(range(8, 46)) - 15))
        probs = probs / probs.sum()  # Normalize probabilities
        years = np.random.choice(range(8, 46), p=probs)
    years_operation.append(years)

# Number of employees (categorical)
employees_cat = []
for size in firm_size:
    if size == "SME":
        emp = np.random.choice(["1-50", "51-250"], p=[0.6, 0.4])
    else:
        emp = np.random.choice(["251-500", "500+"], p=[0.7, 0.3])
    employees_cat.append(emp)

# Annual revenue (categorical, correlated with size)
revenue_cat = []
for size in firm_size:
    if size == "SME":
        rev = np.random.choice(["<50M", "50M-500M", "500M-5B", ">5B"], p=[0.4, 0.45, 0.13, 0.02])
    else:
        rev = np.random.choice(["<50M", "50M-500M", "500M-5B", ">5B"], p=[0.05, 0.25, 0.55, 0.15])
    revenue_cat.append(rev)

# Ownership type
ownership_type = np.random.choice(["Local", "Foreign-owned", "Joint venture"], 
                                 n_total, p=[0.55, 0.25, 0.20])

# =============================================================================
# FDI ENGAGEMENT
# =============================================================================

print("Generating FDI engagement data...")

# FDI partnership probability
fdi_prob = []
for i in range(n_total):
    if ownership_type[i] == "Local":
        prob = 0.3
    elif ownership_type[i] == "Foreign-owned":
        prob = 0.85
    else:  # Joint venture
        prob = 0.75
    
    if firm_size[i] == "Large":
        prob *= 1.2
    
    prob = min(prob, 0.95)  # Cap at 95%
    fdi_prob.append(prob)

has_fdi = np.random.binomial(1, fdi_prob)

# Type of FDI (for firms with FDI)
fdi_type = []
years_fdi = []

for i in range(n_total):
    if has_fdi[i] == 1:
        fdi_type.append(np.random.choice(["Equity", "Joint venture", "Technology transfer", "Management contract"],
                                        p=[0.35, 0.30, 0.25, 0.10]))
        probs = np.exp(-0.2 * (np.array(range(1, 16)) - 3))
        probs = probs / probs.sum()  # Normalize probabilities
        years_fdi.append(min(np.random.choice(range(1, 16), p=probs),
                            years_operation[i] - 1))
    else:
        fdi_type.append(None)
        years_fdi.append(None)

# =============================================================================
# KNOWLEDGE ABSORPTION (Section B)
# =============================================================================

print("Generating knowledge absorption data...")

# Base means for Knowledge Absorption
ka_base_mean = []
for i in range(n_total):
    if has_fdi[i] == 1:
        mean_val = 3.8
        if fdi_type[i] in ["Technology transfer", "Joint venture"]:
            mean_val += 0.5
    else:
        mean_val = 2.5
    ka_base_mean.append(mean_val)

# Correlation matrix for KA items
ka_correlation = np.array([
    [0.64, 0.56, 0.48, 0.40],
    [0.56, 0.64, 0.64, 0.48],
    [0.48, 0.64, 0.64, 0.56],
    [0.40, 0.48, 0.56, 0.64]
])

# Generate correlated KA responses
ka_data = []
for i in range(n_total):
    means = [ka_base_mean[i]] * 4
    mvn = np.random.multivariate_normal(means, ka_correlation)
    ka_responses = np.clip(np.round(mvn), 1, 5).astype(int)
    ka_data.append(ka_responses)

ka_data = np.array(ka_data)

# =============================================================================
# TASK PERFORMANCE (Section C)
# =============================================================================

print("Generating task performance data...")

# Task Performance base means
tp_base_mean = []
for i in range(n_total):
    mean_val = 3.2 + (has_fdi[i] * 0.6) + (1 if firm_size[i] == "Large" else 0) * 0.3
    tp_base_mean.append(mean_val)

# Correlation matrix for TP items
tp_correlation = np.array([
    [0.49, 0.39, 0.34, 0.37],
    [0.39, 0.49, 0.29, 0.34],
    [0.34, 0.29, 0.49, 0.32],
    [0.37, 0.34, 0.32, 0.49]
])

# Generate correlated TP responses
tp_data = []
for i in range(n_total):
    means = [tp_base_mean[i]] * 4
    mvn = np.random.multivariate_normal(means, tp_correlation)
    tp_responses = np.clip(np.round(mvn), 1, 5).astype(int)
    tp_data.append(tp_responses)

tp_data = np.array(tp_data)

# =============================================================================
# INNOVATION (Section D)
# =============================================================================

print("Generating innovation data...")

# R&D spending as % of revenue
rd_mean = []
for i in range(n_total):
    if firm_size[i] == "Large":
        mean_val = 2.5
    else:
        mean_val = 1.2
    mean_val += has_fdi[i] * 0.8
    rd_mean.append(mean_val)

rd_spending_pct = generate_percentage(n_total, rd_mean, [1.2] * n_total, 0, 8)

# New products launched (past 3 years)
new_products_mean = []
for i in range(n_total):
    if firm_size[i] == "Large":
        mean_val = 4
    else:
        mean_val = 2
    mean_val += has_fdi[i] * 1.5
    new_products_mean.append(mean_val)

new_products_3yr = np.maximum(0, np.round(np.random.normal(new_products_mean, 2))).astype(int)

# Process innovations
innovation_prob = []
for i in range(n_total):
    prob = 0.3 + (has_fdi[i] * 0.3) + (1 if firm_size[i] == "Large" else 0) * 0.2
    innovation_prob.append(prob)

innovation_iot = np.random.binomial(1, [p * 0.6 for p in innovation_prob])
innovation_automation = np.random.binomial(1, [p * 0.8 for p in innovation_prob])
innovation_quality_mgmt = np.random.binomial(1, [p * 0.9 for p in innovation_prob])
innovation_other = np.random.binomial(1, [p * 0.4 for p in innovation_prob])

# =============================================================================
# FIRM RESOURCES (Section E)
# =============================================================================

print("Generating firm resources data...")

# Skilled workforce percentage
skilled_workforce_mean = []
for i in range(n_total):
    if firm_size[i] == "Large":
        mean_val = 65
    else:
        mean_val = 45
    mean_val += has_fdi[i] * 10
    skilled_workforce_mean.append(mean_val)

skilled_workforce_pct = generate_percentage(n_total, skilled_workforce_mean, [15] * n_total, 10, 95)

# Training hours
training_hours_mean = []
for i in range(n_total):
    if firm_size[i] == "Large":
        mean_val = 35
    else:
        mean_val = 20
    mean_val += has_fdi[i] * 15
    training_hours_mean.append(mean_val)

training_hours_annual = np.maximum(5, np.round(np.random.normal(training_hours_mean, 12))).astype(int)

# Modern equipment
modern_equipment_prob = []
for i in range(n_total):
    prob = 0.6 + (has_fdi[i] * 0.25) + (1 if firm_size[i] == "Large" else 0) * 0.15
    modern_equipment_prob.append(prob)

modern_equipment = np.random.binomial(1, modern_equipment_prob)

# Machinery age
machinery_age_mean = []
for i in range(n_total):
    if modern_equipment[i] == 1:
        mean_val = 6
    else:
        mean_val = 12
    mean_val -= has_fdi[i] * 2
    machinery_age_mean.append(mean_val)

machinery_age_years = np.maximum(1, np.round(np.random.normal(machinery_age_mean, 4))).astype(int)

# Credit access
credit_access = []
for i in range(n_total):
    if firm_size[i] == "SME":
        if has_fdi[i] == 1:
            probs = [0.6, 0.35, 0.05]  # Easy, Moderate, Difficult
        else:
            probs = [0.4, 0.45, 0.15]
    else:  # Large
        if has_fdi[i] == 1:
            probs = [0.75, 0.23, 0.02]
        else:
            probs = [0.5, 0.4, 0.1]
    
    credit_access.append(np.random.choice(["Easy", "Moderate", "Difficult"], p=probs))

# Reinvestment rate
reinvestment_mean = []
for i in range(n_total):
    if firm_size[i] == "Large":
        mean_val = 18
    else:
        mean_val = 12
    mean_val += has_fdi[i] * 5
    reinvestment_mean.append(mean_val)

reinvestment_rate_pct = generate_percentage(n_total, reinvestment_mean, [8] * n_total, 2, 40)

# =============================================================================
# GOVERNMENT POLICY PERCEPTION (Section F)
# =============================================================================

print("Generating government policy data...")

# Government policy ratings (7-point scale)
gp_base_mean = 3.8
gp_fdi_bonus = 0.4
gp_size_bonus = 0.2

# Correlation matrix for GP items
gp_correlation = np.array([
    [1.44, 0.86, 0.72, 0.58],
    [0.86, 1.44, 1.01, 0.86],
    [0.72, 1.01, 1.44, 0.72],
    [0.58, 0.86, 0.72, 1.44]
])

# Generate correlated GP responses
gp_data = []
for i in range(n_total):
    gp_mean_adj = gp_base_mean + (has_fdi[i] * gp_fdi_bonus) + (1 if firm_size[i] == "Large" else 0) * gp_size_bonus
    means = [gp_mean_adj] * 4
    mvn = np.random.multivariate_normal(means, gp_correlation)
    gp_responses = np.clip(np.round(mvn), 1, 7).astype(int)
    gp_data.append(gp_responses)

gp_data = np.array(gp_data)

# =============================================================================
# PERFORMANCE METRICS (Section G)
# =============================================================================

print("Generating performance metrics...")

# ROI (Return on Investment)
roi_base = [12 if size == "Large" else 8 for size in firm_size]
roi_fdi_effect = has_fdi * 4
roi_innovation_effect = (new_products_3yr / 5) * 2
roi_mean = np.array(roi_base) + roi_fdi_effect + roi_innovation_effect

avg_roi_pct = generate_percentage(n_total, roi_mean, [6] * n_total, -5, 35)

# ROA (Return on Assets)
roa_base = [8 if size == "Large" else 5 for size in firm_size]
roa_fdi_effect = has_fdi * 3
roa_efficiency_effect = (np.mean(tp_data, axis=1) - 3) * 2
roa_mean = np.array(roa_base) + roa_fdi_effect + roa_efficiency_effect

avg_roa_pct = generate_percentage(n_total, roa_mean, [4] * n_total, -3, 25)

# Export intensity
export_base = []
for i in range(n_total):
    if ownership_type[i] == "Local":
        base = 5
    elif ownership_type[i] == "Foreign-owned":
        base = 25
    else:  # Joint venture
        base = 15
    export_base.append(base)

export_fdi_effect = has_fdi * 8
export_mean = np.array(export_base) + export_fdi_effect

export_intensity_pct = generate_percentage(n_total, export_mean, [12] * n_total, 0, 80)

# Production capacity utilization
capacity_base = 72
capacity_fdi_effect = has_fdi * 8
capacity_efficiency_effect = (tp_data[:, 0] - 3) * 4  # Using production efficiency
capacity_mean = capacity_base + capacity_fdi_effect + capacity_efficiency_effect

capacity_utilization_pct = generate_percentage(n_total, capacity_mean, [12] * n_total, 30, 98)

# Market share in Lagos
market_share_base = [3.5 if size == "Large" else 1.2 for size in firm_size]
market_share_fdi_effect = has_fdi * 1.5
market_share_performance_effect = (np.mean(tp_data, axis=1) - 3) * 0.8
market_share_mean = np.array(market_share_base) + market_share_fdi_effect + market_share_performance_effect

market_share_lagos_pct = generate_percentage(n_total, market_share_mean, [2] * n_total, 0.1, 15)

# =============================================================================
# CREATE COMPOSITE VARIABLES
# =============================================================================

print("Creating composite variables...")

# Knowledge Absorption composite
knowledge_absorption = np.mean(ka_data, axis=1)

# Task Performance composite
task_performance = np.mean(tp_data, axis=1)

# Innovation composite (standardized)
innovation_score = (stats.zscore(rd_spending_pct) * 0.4 + 
                   stats.zscore(new_products_3yr) * 0.4 +
                   stats.zscore(innovation_iot + innovation_automation + 
                               innovation_quality_mgmt + innovation_other) * 0.2)

# Government Policy composite
government_policy = np.mean(gp_data, axis=1)

# Firm Resources composite (standardized)
credit_numeric = [3 if c == "Easy" else 2 if c == "Moderate" else 1 for c in credit_access]
firm_resources = (stats.zscore(skilled_workforce_pct) * 0.3 +
                 stats.zscore(training_hours_annual) * 0.2 +
                 stats.zscore(modern_equipment) * 0.2 +
                 stats.zscore(20 - machinery_age_years) * 0.15 +  # Reverse coded
                 stats.zscore(credit_numeric) * 0.15)

# Overall Performance composite (standardized)
overall_performance = (stats.zscore(avg_roi_pct) * 0.25 +
                      stats.zscore(avg_roa_pct) * 0.25 +
                      stats.zscore(export_intensity_pct) * 0.2 +
                      stats.zscore(capacity_utilization_pct) * 0.15 +
                      stats.zscore(market_share_lagos_pct) * 0.15)

# =============================================================================
# CREATE FINAL DATASET
# =============================================================================

print("Creating final dataset...")

# Create the main dataset
data = pd.DataFrame({
    'firm_id': firm_ids,
    'firm_size': firm_size,
    'survey_date': survey_dates,
    'years_operation': years_operation,
    'employees_cat': employees_cat,
    'revenue_cat': revenue_cat,
    'ownership_type': ownership_type,
    'has_fdi': has_fdi,
    'fdi_type': fdi_type,
    'years_fdi': years_fdi,
    
    # Knowledge Absorption
    'ka1_technical_manuals': ka_data[:, 0],
    'ka2_staff_training': ka_data[:, 1],
    'ka3_adapt_technology': ka_data[:, 2],
    'ka4_commercialize_knowledge': ka_data[:, 3],
    
    # Task Performance
    'tp1_production_efficiency': tp_data[:, 0],
    'tp2_quality_control': tp_data[:, 1],
    'tp3_order_fulfillment': tp_data[:, 2],
    'tp4_employee_productivity': tp_data[:, 3],
    
    # Innovation
    'rd_spending_pct': rd_spending_pct,
    'new_products_3yr': new_products_3yr,
    'innovation_iot': innovation_iot,
    'innovation_automation': innovation_automation,
    'innovation_quality_mgmt': innovation_quality_mgmt,
    'innovation_other': innovation_other,
    
    # Firm Resources
    'skilled_workforce_pct': skilled_workforce_pct,
    'training_hours_annual': training_hours_annual,
    'modern_equipment': modern_equipment,
    'machinery_age_years': machinery_age_years,
    'credit_access': credit_access,
    'reinvestment_rate_pct': reinvestment_rate_pct,
    
    # Government Policy
    'gp1_tax_incentives': gp_data[:, 0],
    'gp2_regulatory_stability': gp_data[:, 1],
    'gp3_infrastructure_support': gp_data[:, 2],
    'gp4_permit_ease': gp_data[:, 3],
    
    # Performance Metrics
    'avg_roi_pct': avg_roi_pct,
    'avg_roa_pct': avg_roa_pct,
    'export_intensity_pct': export_intensity_pct,
    'capacity_utilization_pct': capacity_utilization_pct,
    'market_share_lagos_pct': market_share_lagos_pct,
    
    # Composite Variables
    'knowledge_absorption': knowledge_absorption,
    'task_performance': task_performance,
    'innovation_score': innovation_score,
    'government_policy': government_policy,
    'firm_resources': firm_resources,
    'overall_performance': overall_performance
})

# =============================================================================
# DATA QUALITY SUMMARY
# =============================================================================

print("\n=== DATA QUALITY SUMMARY ===")
print(f"Total observations: {len(data)}")
print(f"Firms with FDI: {data['has_fdi'].sum()}")
print(f"FDI percentage: {data['has_fdi'].mean() * 100:.1f}%")

# Check for missing values
missing_summary = data.isnull().sum()
print(f"\nMissing values: {missing_summary.sum()}")

# Basic statistics for key variables
key_vars = ['knowledge_absorption', 'task_performance', 'innovation_score', 
           'government_policy', 'firm_resources', 'overall_performance']

print(f"\n=== KEY VARIABLES SUMMARY ===")
print(data[key_vars].describe().round(3))

# Correlation matrix
print(f"\n=== CORRELATION MATRIX (Key Variables) ===")
corr_matrix = data[key_vars].corr()
print(corr_matrix.round(3))

# =============================================================================
# EXPORT DATA
# =============================================================================

print("\n=== EXPORTING DATA ===")

# Export full dataset
data.to_csv('fdi_synthetic_dataset.csv', index=False)
print("✓ fdi_synthetic_dataset.csv")

# Export SEM dataset (key variables only)
sem_vars = ['firm_id', 'firm_size', 'has_fdi', 'years_fdi', 'fdi_type',
           'ka1_technical_manuals', 'ka2_staff_training', 'ka3_adapt_technology', 'ka4_commercialize_knowledge',
           'tp1_production_efficiency', 'tp2_quality_control', 'tp3_order_fulfillment', 'tp4_employee_productivity',
           'rd_spending_pct', 'new_products_3yr', 'innovation_iot', 'innovation_automation', 'innovation_quality_mgmt',
           'gp1_tax_incentives', 'gp2_regulatory_stability', 'gp3_infrastructure_support', 'gp4_permit_ease',
           'skilled_workforce_pct', 'training_hours_annual', 'modern_equipment', 'reinvestment_rate_pct',
           'avg_roi_pct', 'avg_roa_pct', 'export_intensity_pct', 'capacity_utilization_pct', 'market_share_lagos_pct',
           'knowledge_absorption', 'task_performance', 'innovation_score', 'government_policy', 'firm_resources', 'overall_performance']

sem_data = data[sem_vars]
sem_data.to_csv('fdi_sem_dataset.csv', index=False)
print("✓ fdi_sem_dataset.csv")

# Export summary statistics
summary_stats = data.groupby(['firm_size', 'has_fdi']).agg({
    'overall_performance': 'mean',
    'knowledge_absorption': 'mean',
    'innovation_score': 'mean',
    'avg_roi_pct': 'mean',
    'avg_roa_pct': 'mean'
}).round(3).reset_index()

summary_stats['count'] = data.groupby(['firm_size', 'has_fdi']).size().values
summary_stats.to_csv('summary_statistics.csv', index=False)
print("✓ summary_statistics.csv")

# Export descriptive statistics
desc_stats = data.describe().round(3)
desc_stats.to_csv('descriptive_statistics.csv')
print("✓ descriptive_statistics.csv")

# =============================================================================
# CREATE VISUALIZATIONS
# =============================================================================

print("\n=== CREATING VISUALIZATIONS ===")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# 1. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix - Key Variables')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ correlation_heatmap.png")

# 2. Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, var in enumerate(key_vars):
    axes[i].hist(data[var], bins=20, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Distribution of {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ distribution_plots.png")

# 3. FDI comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ROI comparison
data.boxplot(column='avg_roi_pct', by='has_fdi', ax=axes[0,0])
axes[0,0].set_title('ROI by FDI Status')
axes[0,0].set_xlabel('Has FDI')
axes[0,0].set_ylabel('Average ROI (%)')

# ROA comparison
data.boxplot(column='avg_roa_pct', by='has_fdi', ax=axes[0,1])
axes[0,1].set_title('ROA by FDI Status')
axes[0,1].set_xlabel('Has FDI')
axes[0,1].set_ylabel('Average ROA (%)')

# Knowledge Absorption comparison
data.boxplot(column='knowledge_absorption', by='has_fdi', ax=axes[1,0])
axes[1,0].set_title('Knowledge Absorption by FDI Status')
axes[1,0].set_xlabel('Has FDI')
axes[1,0].set_ylabel('Knowledge Absorption Score')

# Overall Performance comparison
data.boxplot(column='overall_performance', by='has_fdi', ax=axes[1,1])
axes[1,1].set_title('Overall Performance by FDI Status')
axes[1,1].set_xlabel('Has FDI')
axes[1,1].set_ylabel('Overall Performance Score')

plt.suptitle('Performance Comparisons: FDI vs Non-FDI Firms')
plt.tight_layout()
plt.savefig('fdi_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ fdi_comparison_plots.png")

# 4. Sample distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Firm size distribution
size_counts = data['firm_size'].value_counts()
ax1.pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Sample Distribution by Firm Size')

# FDI distribution by firm size
fdi_cross = pd.crosstab(data['firm_size'], data['has_fdi'])
fdi_cross.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
ax2.set_title('FDI Distribution by Firm Size')
ax2.set_xlabel('Firm Size')
ax2.set_ylabel('Count')
ax2.legend(['No FDI', 'Has FDI'])
ax2.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('sample_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ sample_distribution.png")

print("\n=== DATA GENERATION COMPLETED ===")
print(f"✓ Generated synthetic dataset with {n_total} firms")
print(f"✓ {data['has_fdi'].sum()} firms with FDI ({data['has_fdi'].mean()*100:.1f}%)")
print(f"✓ {n_sme} SMEs and {n_large} large firms")
print(f"✓ All files exported successfully")
print("✓ Ready for SEM analysis")

print("\n=== FILES CREATED ===")
print("Datasets:")
print("- fdi_synthetic_dataset.csv (complete dataset)")
print("- fdi_sem_dataset.csv (SEM analysis ready)")
print("- summary_statistics.csv (group summaries)")
print("- descriptive_statistics.csv (variable statistics)")
print("\nVisualizations:")
print("- correlation_heatmap.png")
print("- distribution_plots.png") 
print("- fdi_comparison_plots.png")
print("- sample_distribution.png")