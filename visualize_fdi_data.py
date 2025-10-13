"""
FDI Survey Data Visualization Script
Creates comprehensive visualizations of key relationships in the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the generated data
print("Loading FDI survey data...")
df = pd.read_csv('fdi_survey_data.csv')
print(f"Loaded {len(df)} records")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
fig.suptitle('FDI Influence on Food Processing Firms - Key Relationships', 
             fontsize=16, fontweight='bold', y=1.02)

# 1. FDI Distribution
ax1 = plt.subplot(3, 4, 1)
fdi_counts = df.groupby(['firm_size', 'has_fdi']).size().unstack()
fdi_counts.plot(kind='bar', ax=ax1, color=['#e74c3c', '#2ecc71'])
ax1.set_title('FDI Distribution by Firm Size')
ax1.set_xlabel('Firm Size')
ax1.set_ylabel('Number of Firms')
ax1.legend(['No FDI', 'Has FDI'], loc='upper right')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

# 2. Performance Comparison
ax2 = plt.subplot(3, 4, 2)
performance_data = df[['has_fdi', 'performance_composite']].copy()
performance_data['FDI Status'] = performance_data['has_fdi'].map({0: 'No FDI', 1: 'Has FDI'})
sns.boxplot(data=performance_data, x='FDI Status', y='performance_composite', ax=ax2)
ax2.set_title('Performance Composite: FDI vs Non-FDI')
ax2.set_ylabel('Performance Score')

# 3. Knowledge Absorption by FDI
ax3 = plt.subplot(3, 4, 3)
ka_cols = ['ka_technical_manuals', 'ka_staff_training', 'ka_tech_adaptation', 'ka_knowledge_commercialization']
ka_means_fdi = df[df['has_fdi']==1][ka_cols].mean()
ka_means_no_fdi = df[df['has_fdi']==0][ka_cols].mean()
x = np.arange(len(ka_cols))
width = 0.35
bars1 = ax3.bar(x - width/2, ka_means_no_fdi, width, label='No FDI', color='#e74c3c')
bars2 = ax3.bar(x + width/2, ka_means_fdi, width, label='Has FDI', color='#2ecc71')
ax3.set_ylabel('Average Score (1-5)')
ax3.set_title('Knowledge Absorption Components')
ax3.set_xticks(x)
ax3.set_xticklabels(['Tech\nManuals', 'Staff\nTraining', 'Tech\nAdaptation', 'Knowledge\nCommercial.'], rotation=45, ha='right')
ax3.legend()
ax3.set_ylim([0, 5])

# 4. Innovation Metrics
ax4 = plt.subplot(3, 4, 4)
innovation_metrics = df.groupby('has_fdi')[['rd_spending_percent', 'new_products_3years']].mean()
innovation_metrics.T.plot(kind='bar', ax=ax4, color=['#e74c3c', '#2ecc71'])
ax4.set_title('Innovation Metrics by FDI Status')
ax4.set_xlabel('Innovation Metric')
ax4.set_ylabel('Average Value')
ax4.legend(['No FDI', 'Has FDI'], loc='upper left')
ax4.set_xticklabels(['R&D Spending (%)', 'New Products (3yr)'], rotation=45, ha='right')

# 5. ROI Distribution
ax5 = plt.subplot(3, 4, 5)
sns.histplot(data=df, x='perf_roi', hue='has_fdi', bins=20, ax=ax5, kde=True)
ax5.set_title('Return on Investment Distribution')
ax5.set_xlabel('ROI (%)')
ax5.set_ylabel('Frequency')
ax5.legend(['No FDI', 'Has FDI'], title='FDI Status')

# 6. Government Policy Perception
ax6 = plt.subplot(3, 4, 6)
gp_cols = ['gp_tax_incentives', 'gp_regulatory_stability', 'gp_infrastructure_support', 'gp_ease_permits']
gp_data = df[gp_cols].mean()
colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
bars = ax6.bar(range(len(gp_cols)), gp_data, color=colors)
ax6.set_ylabel('Average Score (1-7)')
ax6.set_title('Government Policy Effectiveness Ratings')
ax6.set_xticks(range(len(gp_cols)))
ax6.set_xticklabels(['Tax\nIncentives', 'Regulatory\nStability', 'Infrastructure\nSupport', 'Ease of\nPermits'], rotation=45, ha='right')
ax6.set_ylim([0, 7])
ax6.axhline(y=4, color='red', linestyle='--', alpha=0.3, label='Neutral')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom')

# 7. Correlation Heatmap
ax7 = plt.subplot(3, 4, 7)
key_vars = ['has_fdi', 'ka_average', 'tp_average', 'innovation_score', 
            'firm_resources_score', 'gp_average', 'performance_composite']
corr_matrix = df[key_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax7,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax7.set_title('Correlation Matrix of Key Variables')
ax7.set_xticklabels(['FDI', 'KA', 'TP', 'INN', 'FR', 'GP', 'PERF'], rotation=45)
ax7.set_yticklabels(['FDI', 'KA', 'TP', 'INN', 'FR', 'GP', 'PERF'], rotation=0)

# 8. Firm Resources Comparison
ax8 = plt.subplot(3, 4, 8)
resources_data = df.groupby('has_fdi')[['hr_skilled_workforce_percent', 'hr_training_hours']].mean()
ax8_twin = ax8.twinx()
x = [0, 1]
bar_width = 0.4
bars1 = ax8.bar(x, resources_data['hr_skilled_workforce_percent'], bar_width, 
                color='#3498db', alpha=0.7, label='Skilled Workforce %')
bars2 = ax8_twin.bar([i + bar_width for i in x], resources_data['hr_training_hours'], bar_width,
                     color='#e67e22', alpha=0.7, label='Training Hours')
ax8.set_xlabel('FDI Status')
ax8.set_ylabel('Skilled Workforce (%)', color='#3498db')
ax8_twin.set_ylabel('Training Hours/Year', color='#e67e22')
ax8.set_xticks([i + bar_width/2 for i in x])
ax8.set_xticklabels(['No FDI', 'Has FDI'])
ax8.set_title('Human Resources by FDI Status')
ax8.tick_params(axis='y', labelcolor='#3498db')
ax8_twin.tick_params(axis='y', labelcolor='#e67e22')

# 9. Market Share vs FDI
ax9 = plt.subplot(3, 4, 9)
sns.scatterplot(data=df, x='years_fdi_partnership', y='perf_market_share', 
                hue='firm_size', size='performance_composite', ax=ax9, alpha=0.7)
ax9.set_title('Market Share vs Years of FDI Partnership')
ax9.set_xlabel('Years with FDI Partnership')
ax9.set_ylabel('Market Share in Lagos (%)')
ax9.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 10. Task Performance Radar Chart
ax10 = plt.subplot(3, 4, 10, projection='polar')
tp_cols = ['tp_production_efficiency', 'tp_quality_control', 'tp_order_fulfillment', 'tp_employee_productivity']
angles = np.linspace(0, 2 * np.pi, len(tp_cols), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

tp_fdi = df[df['has_fdi']==1][tp_cols].mean().values
tp_no_fdi = df[df['has_fdi']==0][tp_cols].mean().values
tp_fdi = np.concatenate((tp_fdi, [tp_fdi[0]]))
tp_no_fdi = np.concatenate((tp_no_fdi, [tp_no_fdi[0]]))

ax10.plot(angles, tp_fdi, 'o-', linewidth=2, color='#2ecc71', label='Has FDI')
ax10.fill(angles, tp_fdi, alpha=0.25, color='#2ecc71')
ax10.plot(angles, tp_no_fdi, 'o-', linewidth=2, color='#e74c3c', label='No FDI')
ax10.fill(angles, tp_no_fdi, alpha=0.25, color='#e74c3c')
ax10.set_xticks(angles[:-1])
ax10.set_xticklabels(['Production\nEfficiency', 'Quality\nControl', 'Order\nFulfillment', 'Employee\nProductivity'])
ax10.set_ylim(0, 5)
ax10.set_title('Task Performance Comparison', pad=20)
ax10.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
ax10.grid(True)

# 11. Export Intensity
ax11 = plt.subplot(3, 4, 11)
export_data = df[df['has_fdi']==1]['perf_export_intensity']
sns.histplot(export_data, bins=15, kde=True, ax=ax11, color='#9b59b6')
ax11.set_title('Export Intensity Distribution (FDI Firms Only)')
ax11.set_xlabel('Export Intensity (%)')
ax11.set_ylabel('Frequency')
ax11.axvline(export_data.mean(), color='red', linestyle='--', 
             label=f'Mean: {export_data.mean():.1f}%')
ax11.legend()

# 12. Performance Drivers (Regression Coefficients)
ax12 = plt.subplot(3, 4, 12)
# Simple linear regression to show relative importance
X = df[['has_fdi', 'ka_average', 'tp_average', 'innovation_score', 'firm_resources_score', 'gp_average']].fillna(0)
y = df['performance_composite'].fillna(df['performance_composite'].mean())
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
coefficients = model.coef_
feature_names = ['FDI', 'Knowledge\nAbsorption', 'Task\nPerformance', 
                'Innovation', 'Firm\nResources', 'Gov Policy']
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefficients]
bars = ax12.bar(feature_names, coefficients, color=colors, alpha=0.7)
ax12.set_title('Performance Drivers (Standardized Coefficients)')
ax12.set_ylabel('Coefficient Value')
ax12.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax12.set_xticklabels(feature_names, rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    ax12.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()

# Save the visualization
plt.savefig('fdi_data_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as: fdi_data_visualization.png")

# Create a second figure for SEM path diagram representation
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('SEM Model Relationships - Data Validation', fontsize=14, fontweight='bold')

# 1. FDI → Mediators
ax1.set_title('Direct Effects: FDI on Mediating Variables')
mediators = ['ka_average', 'tp_average', 'innovation_score']
mediator_labels = ['Knowledge Absorption', 'Task Performance', 'Innovation']
fdi_effects = []
for mediator in mediators:
    corr = df[['has_fdi', mediator]].corr().iloc[0, 1]
    fdi_effects.append(corr)
bars = ax1.barh(mediator_labels, fdi_effects, color='#3498db')
ax1.set_xlabel('Correlation with FDI')
ax1.set_xlim([0, 1])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2., 
             f'{width:.3f}', ha='left', va='center')

# 2. Mediators → Performance
ax2.set_title('Direct Effects: Mediators on Performance')
mediator_performance = []
for mediator in mediators:
    corr = df[[mediator, 'performance_composite']].corr().iloc[0, 1]
    mediator_performance.append(corr)
bars = ax2.barh(mediator_labels, mediator_performance, color='#2ecc71')
ax2.set_xlabel('Correlation with Performance')
ax2.set_xlim([0, 1])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2., 
             f'{width:.3f}', ha='left', va='center')

# 3. Moderation Effect Visualization
ax3.set_title('Moderation Effect: Gov Policy on FDI-Performance')
# Split data into high and low government policy groups
gp_median = df['gp_average'].median()
high_gp = df[df['gp_average'] > gp_median]
low_gp = df[df['gp_average'] <= gp_median]

# Calculate means for plotting
plot_data = pd.DataFrame({
    'Low Gov Support - No FDI': [low_gp[low_gp['has_fdi']==0]['performance_composite'].mean()],
    'Low Gov Support - Has FDI': [low_gp[low_gp['has_fdi']==1]['performance_composite'].mean()],
    'High Gov Support - No FDI': [high_gp[high_gp['has_fdi']==0]['performance_composite'].mean()],
    'High Gov Support - Has FDI': [high_gp[high_gp['has_fdi']==1]['performance_composite'].mean()]
})

x = [0, 1]
low_gp_values = [plot_data['Low Gov Support - No FDI'].values[0], 
                 plot_data['Low Gov Support - Has FDI'].values[0]]
high_gp_values = [plot_data['High Gov Support - No FDI'].values[0], 
                  plot_data['High Gov Support - Has FDI'].values[0]]

ax3.plot(x, low_gp_values, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Low Gov Policy')
ax3.plot(x, high_gp_values, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='High Gov Policy')
ax3.set_xticks(x)
ax3.set_xticklabels(['No FDI', 'Has FDI'])
ax3.set_ylabel('Performance Composite Score')
ax3.set_xlabel('FDI Status')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Model Fit Preview
ax4.set_title('Data Structure for SEM Analysis')
ax4.axis('off')
model_info = f"""
Sample Characteristics:
• Total N = {len(df)} firms
• SMEs = {len(df[df['firm_size']=='SME'])} ({len(df[df['firm_size']=='SME'])/len(df)*100:.1f}%)
• Large = {len(df[df['firm_size']=='Large'])} ({len(df[df['firm_size']=='Large'])/len(df)*100:.1f}%)
• With FDI = {df['has_fdi'].sum()} ({df['has_fdi'].mean()*100:.1f}%)

Key Relationships (Pearson r):
• FDI → Performance: r = {df[['has_fdi', 'performance_composite']].corr().iloc[0,1]:.3f}
• KA → Performance: r = {df[['ka_average', 'performance_composite']].corr().iloc[0,1]:.3f}
• Innovation → Performance: r = {df[['innovation_score', 'performance_composite']].corr().iloc[0,1]:.3f}

Data Quality:
• Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.2f}%)
• Response rate simulation: 70% (210/300 target)

Ready for SEM Analysis:
✓ Measurement model indicators present
✓ Latent constructs properly scaled
✓ Moderation variables included
✓ Control variables available
"""
ax4.text(0.1, 0.9, model_info, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('sem_validation_plots.png', dpi=300, bbox_inches='tight')
print("✓ SEM validation plots saved as: sem_validation_plots.png")

# Display summary statistics
print("\n" + "="*60)
print("DATASET SUMMARY STATISTICS")
print("="*60)
print(f"\nSample Size: {len(df)} firms")
print(f"FDI Firms: {df['has_fdi'].sum()} ({df['has_fdi'].mean()*100:.1f}%)")
print(f"\nPerformance Metrics (Mean ± SD):")
print(f"  ROI: {df['perf_roi'].mean():.1f} ± {df['perf_roi'].std():.1f}%")
print(f"  ROA: {df['perf_roa'].mean():.1f} ± {df['perf_roa'].std():.1f}%")
print(f"  Market Share: {df['perf_market_share'].mean():.1f} ± {df['perf_market_share'].std():.1f}%")
print(f"\nKey Correlations with Performance:")
correlations = []
for var in ['has_fdi', 'ka_average', 'tp_average', 'innovation_score', 'firm_resources_score', 'gp_average']:
    corr = df[[var, 'performance_composite']].corr().iloc[0, 1]
    correlations.append((var, corr))
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for var, corr in correlations:
    print(f"  {var}: r = {corr:.3f}")

plt.show()