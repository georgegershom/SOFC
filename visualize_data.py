#!/usr/bin/env python3
"""
Generate Quick Visualizations for FDI Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv('fdi_lagos_survey_data.csv')

# Create composite scores
df['KA_score'] = df[['ka1_technical_manuals', 'ka2_staff_training', 
                      'ka3_adapt_technology', 'ka4_commercialize_knowledge']].mean(axis=1)
df['TP_score'] = df[['tp1_production_efficiency', 'tp2_quality_control', 
                      'tp3_order_fulfillment', 'tp4_employee_productivity']].mean(axis=1)

print("Generating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Sample Composition
ax1 = plt.subplot(3, 3, 1)
fdi_counts = df['has_fdi'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.pie(fdi_counts.values, labels=fdi_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('FDI Engagement Distribution', fontsize=12, fontweight='bold')

# 2. Ownership Type
ax2 = plt.subplot(3, 3, 2)
own_counts = df['ownership_type'].value_counts()
ax2.bar(range(len(own_counts)), own_counts.values, color='steelblue')
ax2.set_xticks(range(len(own_counts)))
ax2.set_xticklabels(own_counts.index, rotation=45, ha='right')
ax2.set_ylabel('Number of Firms')
ax2.set_title('Ownership Structure', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Firm Size Distribution
ax3 = plt.subplot(3, 3, 3)
size_counts = df['firm_size'].value_counts()
ax3.bar(size_counts.index, size_counts.values, color=['#3498db', '#9b59b6'])
ax3.set_ylabel('Number of Firms')
ax3.set_title('Firm Size Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. ROI Comparison: FDI vs Non-FDI
ax4 = plt.subplot(3, 3, 4)
fdi_roi = df[df['has_fdi']=='Yes']['avg_roi_pct']
non_fdi_roi = df[df['has_fdi']=='No']['avg_roi_pct']
bp = ax4.boxplot([fdi_roi, non_fdi_roi], labels=['FDI Firms', 'Non-FDI Firms'],
                   patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax4.set_ylabel('ROI (%)')
ax4.set_title('ROI Comparison', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Knowledge Absorption Comparison
ax5 = plt.subplot(3, 3, 5)
ka_comparison = df.groupby('has_fdi')['KA_score'].mean()
bars = ax5.bar(ka_comparison.index, ka_comparison.values, color=['#e74c3c', '#2ecc71'])
ax5.set_ylabel('Knowledge Absorption Score (1-5)')
ax5.set_title('Knowledge Absorption: FDI vs Non-FDI', fontsize=12, fontweight='bold')
ax5.set_ylim(0, 5)
ax5.grid(axis='y', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')

# 6. R&D Spending Distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist([df[df['has_fdi']=='Yes']['rd_spending_pct'],
          df[df['has_fdi']=='No']['rd_spending_pct']], 
         bins=15, label=['FDI Firms', 'Non-FDI Firms'], 
         color=['#2ecc71', '#e74c3c'], alpha=0.7)
ax6.set_xlabel('R&D Spending (% of Revenue)')
ax6.set_ylabel('Number of Firms')
ax6.set_title('R&D Spending Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. Innovation Adoption Rates
ax7 = plt.subplot(3, 3, 7)
innovations = ['IoT', 'Automation', 'Quality Mgmt', 'Other']
fdi_innovations = [
    df[df['has_fdi']=='Yes']['innov_iot'].mean() * 100,
    df[df['has_fdi']=='Yes']['innov_automation'].mean() * 100,
    df[df['has_fdi']=='Yes']['innov_quality_mgmt'].mean() * 100,
    df[df['has_fdi']=='Yes']['innov_other'].mean() * 100
]
non_fdi_innovations = [
    df[df['has_fdi']=='No']['innov_iot'].mean() * 100,
    df[df['has_fdi']=='No']['innov_automation'].mean() * 100,
    df[df['has_fdi']=='No']['innov_quality_mgmt'].mean() * 100,
    df[df['has_fdi']=='No']['innov_other'].mean() * 100
]
x = np.arange(len(innovations))
width = 0.35
ax7.bar(x - width/2, fdi_innovations, width, label='FDI Firms', color='#2ecc71')
ax7.bar(x + width/2, non_fdi_innovations, width, label='Non-FDI Firms', color='#e74c3c')
ax7.set_ylabel('Adoption Rate (%)')
ax7.set_title('Innovation Adoption Rates', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(innovations, rotation=45, ha='right')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# 8. Performance Scatter: KA vs ROI
ax8 = plt.subplot(3, 3, 8)
fdi_firms = df[df['has_fdi']=='Yes']
non_fdi_firms = df[df['has_fdi']=='No']
ax8.scatter(fdi_firms['KA_score'], fdi_firms['avg_roi_pct'], 
           alpha=0.6, s=50, c='#2ecc71', label='FDI Firms')
ax8.scatter(non_fdi_firms['KA_score'], non_fdi_firms['avg_roi_pct'], 
           alpha=0.6, s=50, c='#e74c3c', label='Non-FDI Firms')
ax8.set_xlabel('Knowledge Absorption Score')
ax8.set_ylabel('ROI (%)')
ax8.set_title('Knowledge Absorption vs Performance', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Government Policy Perception
ax9 = plt.subplot(3, 3, 9)
gp_vars = ['gp1_tax_incentives', 'gp2_regulatory_stability', 
           'gp3_infrastructure', 'gp4_permits_ease']
gp_labels = ['Tax\nIncentives', 'Regulatory\nStability', 
             'Infrastructure', 'Permits\nEase']
gp_means = [df[var].mean() for var in gp_vars]
bars = ax9.bar(range(len(gp_means)), gp_means, color='#3498db')
ax9.set_xticks(range(len(gp_means)))
ax9.set_xticklabels(gp_labels, fontsize=9)
ax9.set_ylabel('Mean Score (1-7)')
ax9.set_title('Government Policy Perception', fontsize=12, fontweight='bold')
ax9.set_ylim(0, 7)
ax9.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Neutral (4.0)')
ax9.grid(axis='y', alpha=0.3)
ax9.legend()

plt.tight_layout()
plt.savefig('fdi_dataset_visualizations.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations saved to: fdi_dataset_visualizations.png")

# Create correlation heatmap
fig2, ax = plt.subplots(figsize=(12, 10))

# Select key variables for correlation
df['fdi_binary'] = (df['has_fdi'] == 'Yes').astype(int)
corr_vars = ['fdi_binary', 'KA_score', 'TP_score', 'rd_spending_pct', 
             'new_products_3yrs', 'avg_roi_pct', 'avg_roa_pct', 
             'capacity_utilization_pct', 'export_intensity_pct']
corr_labels = ['FDI', 'Knowledge\nAbsorption', 'Task\nPerformance', 
               'R&D\nSpending', 'New\nProducts', 'ROI', 'ROA', 
               'Capacity\nUtil.', 'Export\nIntensity']

corr_matrix = df[corr_vars].corr()

# Plot heatmap
im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right')
ax.set_yticklabels(corr_labels)

# Add correlation values
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Correlation Matrix - Key Variables', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Correlation Coefficient')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Correlation heatmap saved to: correlation_heatmap.png")

print("\n" + "="*60)
print("VISUALIZATION SUMMARY")
print("="*60)
print("\nGenerated Files:")
print("1. fdi_dataset_visualizations.png - 9-panel overview")
print("2. correlation_heatmap.png - Correlation matrix")
print("\nKey Insights from Visualizations:")
print(f"- FDI firms show {fdi_roi.mean() - non_fdi_roi.mean():.1f}% higher ROI")
print(f"- Knowledge absorption is {ka_comparison['Yes'] - ka_comparison['No']:.2f} points higher in FDI firms")
print(f"- FDI firms spend {df[df['has_fdi']=='Yes']['rd_spending_pct'].mean():.2f}% vs {df[df['has_fdi']=='No']['rd_spending_pct'].mean():.2f}% on R&D")
print(f"- Automation adoption: {fdi_innovations[1]:.1f}% (FDI) vs {non_fdi_innovations[1]:.1f}% (Non-FDI)")
print("\n" + "="*60)
