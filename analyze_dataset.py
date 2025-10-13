"""
Dataset Analysis and Visualization Script
==========================================
Generates summary statistics and visualizations for the FDI dataset
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

def load_and_examine_data(filename='lagos_food_processing_fdi_dataset.csv'):
    """Load dataset and perform initial examination"""
    
    print("=" * 70)
    print("DATASET ANALYSIS REPORT")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(filename)
    print(f"\n📊 Dataset loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    
    # Basic info
    print("\n📋 DATASET OVERVIEW")
    print("-" * 40)
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%)")
    
    # Data types
    print("\n📊 Variable Types:")
    print(f"- Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"- Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    return df


def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Key performance indicators
    print("\n🎯 KEY PERFORMANCE INDICATORS")
    print("-" * 40)
    kpis = ['roi_pct', 'roa_pct', 'revenue_growth_pct', 'market_share_pct', 
            'productivity_index', 'operational_efficiency']
    
    for kpi in kpis:
        if kpi in df.columns:
            print(f"\n{kpi}:")
            print(f"  Mean: {df[kpi].mean():.2f}")
            print(f"  Std Dev: {df[kpi].std():.2f}")
            print(f"  Median: {df[kpi].median():.2f}")
            print(f"  25th Percentile: {df[kpi].quantile(0.25):.2f}")
            print(f"  75th Percentile: {df[kpi].quantile(0.75):.2f}")
    
    # FDI exposure
    print("\n🌍 FOREIGN DIRECT INVESTMENT")
    print("-" * 40)
    fdi_vars = ['fdi_ownership_pct', 'knowledge_absorption_avg', 'innovation_avg']
    
    for var in fdi_vars:
        if var in df.columns:
            print(f"\n{var}:")
            print(f"  Mean: {df[var].mean():.2f}")
            print(f"  Std Dev: {df[var].std():.2f}")
    
    # Government policy
    print("\n🏛️ GOVERNMENT POLICY PERCEPTION")
    print("-" * 40)
    if 'policy_effectiveness_index' in df.columns:
        print(f"Policy Effectiveness Index:")
        print(f"  Mean: {df['policy_effectiveness_index'].mean():.2f}")
        print(f"  Std Dev: {df['policy_effectiveness_index'].std():.2f}")
        print(f"  Range: [{df['policy_effectiveness_index'].min():.2f}, {df['policy_effectiveness_index'].max():.2f}]")
    
    # Firm characteristics
    print("\n🏭 FIRM CHARACTERISTICS")
    print("-" * 40)
    
    if 'ownership_type' in df.columns:
        print("\nOwnership Distribution:")
        ownership_dist = df['ownership_type'].value_counts()
        for owner_type, count in ownership_dist.items():
            print(f"  {owner_type}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'firm_size' in df.columns:
        print("\nFirm Size Distribution:")
        size_dist = df['firm_size'].value_counts()
        for size, count in size_dist.items():
            print(f"  {size}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'subsector' in df.columns:
        print("\nTop 5 Subsectors:")
        subsector_dist = df['subsector'].value_counts().head(5)
        for subsector, count in subsector_dist.items():
            print(f"  {subsector}: {count} ({count/len(df)*100:.1f}%)")


def create_visualizations(df):
    """Create comprehensive visualizations"""
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. FDI Ownership Distribution by Firm Type
    ax1 = plt.subplot(3, 3, 1)
    if 'ownership_type' in df.columns and 'fdi_ownership_pct' in df.columns:
        df.boxplot(column='fdi_ownership_pct', by='ownership_type', ax=ax1)
        ax1.set_title('FDI Ownership % by Firm Type')
        ax1.set_xlabel('Ownership Type')
        ax1.set_ylabel('FDI Ownership %')
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')
    
    # 2. Performance Metrics Distribution
    ax2 = plt.subplot(3, 3, 2)
    if 'roi_pct' in df.columns and 'roa_pct' in df.columns:
        ax2.scatter(df['roi_pct'], df['roa_pct'], alpha=0.5)
        ax2.set_xlabel('ROI (%)')
        ax2.set_ylabel('ROA (%)')
        ax2.set_title('ROI vs ROA Relationship')
        
        # Add trend line
        z = np.polyfit(df['roi_pct'].dropna(), df['roa_pct'].dropna(), 1)
        p = np.poly1d(z)
        ax2.plot(df['roi_pct'].sort_values(), p(df['roi_pct'].sort_values()), 
                "r-", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax2.legend()
    
    # 3. Knowledge Absorption by Firm Size
    ax3 = plt.subplot(3, 3, 3)
    if 'firm_size' in df.columns and 'knowledge_absorption_avg' in df.columns:
        size_order = ['Micro', 'Small', 'Medium', 'Large']
        existing_sizes = [s for s in size_order if s in df['firm_size'].values]
        sns.violinplot(data=df, x='firm_size', y='knowledge_absorption_avg', 
                      order=existing_sizes, ax=ax3)
        ax3.set_title('Knowledge Absorption by Firm Size')
        ax3.set_xlabel('Firm Size')
        ax3.set_ylabel('Knowledge Absorption (1-5)')
    
    # 4. Policy Effectiveness vs Performance
    ax4 = plt.subplot(3, 3, 4)
    if 'policy_effectiveness_index' in df.columns and 'performance_composite' in df.columns:
        ax4.scatter(df['policy_effectiveness_index'], df['performance_composite'], 
                   alpha=0.5, c=df['fdi_composite'] if 'fdi_composite' in df.columns else 'blue')
        ax4.set_xlabel('Policy Effectiveness Index')
        ax4.set_ylabel('Performance Composite')
        ax4.set_title('Policy Effectiveness vs Performance\n(Color = FDI Intensity)')
        
        if 'fdi_composite' in df.columns:
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('FDI Composite')
    
    # 5. Innovation Levels Distribution
    ax5 = plt.subplot(3, 3, 5)
    innovation_cols = ['product_innovation', 'process_innovation', 
                      'marketing_innovation', 'organizational_innovation']
    existing_innovation = [col for col in innovation_cols if col in df.columns]
    if existing_innovation:
        innovation_means = [df[col].mean() for col in existing_innovation]
        ax5.bar(range(len(existing_innovation)), innovation_means)
        ax5.set_xticks(range(len(existing_innovation)))
        ax5.set_xticklabels([col.replace('_innovation', '') for col in existing_innovation], 
                           rotation=45, ha='right')
        ax5.set_ylabel('Average Score (1-5)')
        ax5.set_title('Innovation Types - Average Scores')
        ax5.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Neutral')
        ax5.legend()
    
    # 6. Resource Utilization
    ax6 = plt.subplot(3, 3, 6)
    resource_vars = ['skilled_labor_ratio', 'rd_spend_pct', 'liquidity_ratio']
    existing_resources = [var for var in resource_vars if var in df.columns]
    if len(existing_resources) >= 2:
        correlation_matrix = df[existing_resources].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax6)
        ax6.set_title('Resource Variables Correlation')
    
    # 7. Subsector Performance
    ax7 = plt.subplot(3, 3, 7)
    if 'subsector' in df.columns and 'roi_pct' in df.columns:
        subsector_performance = df.groupby('subsector')['roi_pct'].mean().sort_values()
        subsector_performance.plot(kind='barh', ax=ax7)
        ax7.set_xlabel('Average ROI (%)')
        ax7.set_title('Average ROI by Subsector')
    
    # 8. FDI Moderation Effect
    ax8 = plt.subplot(3, 3, 8)
    if all(col in df.columns for col in ['fdi_composite', 'performance_composite', 'policy_effectiveness_index']):
        # Split data by policy effectiveness (high vs low)
        median_policy = df['policy_effectiveness_index'].median()
        high_policy = df[df['policy_effectiveness_index'] >= median_policy]
        low_policy = df[df['policy_effectiveness_index'] < median_policy]
        
        ax8.scatter(low_policy['fdi_composite'], low_policy['performance_composite'], 
                   alpha=0.5, label='Low Policy Support', color='red')
        ax8.scatter(high_policy['fdi_composite'], high_policy['performance_composite'], 
                   alpha=0.5, label='High Policy Support', color='green')
        
        # Add trend lines
        for data, color, label in [(low_policy, 'red', 'Low'), (high_policy, 'green', 'High')]:
            if len(data) > 1:
                z = np.polyfit(data['fdi_composite'].dropna(), 
                             data['performance_composite'].dropna(), 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data['fdi_composite'].min(), 
                                    data['fdi_composite'].max(), 100)
                ax8.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.8)
        
        ax8.set_xlabel('FDI Composite Score')
        ax8.set_ylabel('Performance Composite')
        ax8.set_title('Moderation Effect of Government Policy')
        ax8.legend()
    
    # 9. Export Intensity Distribution
    ax9 = plt.subplot(3, 3, 9)
    if 'export_intensity' in df.columns and 'ownership_type' in df.columns:
        ownership_types = df['ownership_type'].unique()
        export_by_ownership = [df[df['ownership_type']==ot]['export_intensity'].dropna() 
                              for ot in ownership_types]
        ax9.violinplot(export_by_ownership, positions=range(len(ownership_types)))
        ax9.set_xticks(range(len(ownership_types)))
        ax9.set_xticklabels(ownership_types, rotation=45, ha='right')
        ax9.set_ylabel('Export Intensity (%)')
        ax9.set_title('Export Intensity by Ownership Type')
    
    plt.suptitle('Food Processing Firms FDI Study - Lagos, Nigeria', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('fdi_dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: fdi_dataset_analysis.png")
    
    # Create correlation heatmap
    fig2, ax = plt.subplots(figsize=(14, 10))
    
    # Select key variables for correlation
    key_vars = ['fdi_ownership_pct', 'knowledge_absorption_avg', 'innovation_avg',
                'roi_pct', 'roa_pct', 'operational_efficiency', 'productivity_index',
                'policy_effectiveness_index', 'skilled_labor_ratio', 'rd_spend_pct']
    
    existing_vars = [var for var in key_vars if var in df.columns]
    
    if len(existing_vars) > 1:
        corr_matrix = df[existing_vars].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix - Key Variables', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: correlation_matrix.png")


def test_moderation_hypothesis(df):
    """Test the moderation hypothesis"""
    
    print("\n" + "=" * 70)
    print("MODERATION ANALYSIS")
    print("=" * 70)
    
    required_cols = ['fdi_composite', 'performance_composite', 
                    'policy_effectiveness_index', 'fdi_x_policy']
    
    if all(col in df.columns for col in required_cols):
        from scipy import stats
        
        # Clean data
        analysis_df = df[required_cols].dropna()
        
        print("\n📊 Hierarchical Regression Analysis")
        print("-" * 40)
        
        # Step 1: FDI only
        corr1 = stats.pearsonr(analysis_df['fdi_composite'], 
                              analysis_df['performance_composite'])
        print(f"\nStep 1 - FDI → Performance:")
        print(f"  Correlation: r = {corr1[0]:.3f}, p = {corr1[1]:.4f}")
        
        # Step 2: FDI + Policy
        corr2 = stats.pearsonr(analysis_df['policy_effectiveness_index'], 
                              analysis_df['performance_composite'])
        print(f"\nStep 2 - Policy → Performance:")
        print(f"  Correlation: r = {corr2[0]:.3f}, p = {corr2[1]:.4f}")
        
        # Step 3: Interaction
        corr3 = stats.pearsonr(analysis_df['fdi_x_policy'], 
                              analysis_df['performance_composite'])
        print(f"\nStep 3 - FDI × Policy → Performance:")
        print(f"  Correlation: r = {corr3[0]:.3f}, p = {corr3[1]:.4f}")
        
        # Test moderation strength
        print("\n🎯 Moderation Effect Interpretation:")
        if corr3[1] < 0.05:
            print("  ✓ Significant moderation effect detected (p < 0.05)")
            print("  → Government policy significantly moderates the FDI-performance relationship")
        else:
            print("  ✗ No significant moderation effect (p ≥ 0.05)")
        
        # Compare high vs low policy groups
        median_policy = analysis_df['policy_effectiveness_index'].median()
        high_policy = analysis_df[analysis_df['policy_effectiveness_index'] >= median_policy]
        low_policy = analysis_df[analysis_df['policy_effectiveness_index'] < median_policy]
        
        corr_high = stats.pearsonr(high_policy['fdi_composite'], 
                                  high_policy['performance_composite'])
        corr_low = stats.pearsonr(low_policy['fdi_composite'], 
                                 low_policy['performance_composite'])
        
        print("\n📈 Subgroup Analysis:")
        print(f"  High Policy Support: r = {corr_high[0]:.3f}")
        print(f"  Low Policy Support:  r = {corr_low[0]:.3f}")
        print(f"  Difference: {abs(corr_high[0] - corr_low[0]):.3f}")
        
        if abs(corr_high[0] - corr_low[0]) > 0.1:
            print("  → Substantial difference in FDI effectiveness based on policy support")


def generate_report(df):
    """Generate comprehensive analysis report"""
    
    # Create report
    report = f"""
================================================================================
FOOD PROCESSING FIRMS FDI DATASET - ANALYSIS REPORT
================================================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
---------------
• Total Firms: {len(df)}
• Variables: {len(df.columns)}
• Missing Data: {df.isnull().sum().sum()} values ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%)
• Date Range: {df['survey_date'].min() if 'survey_date' in df.columns else 'N/A'} to {df['survey_date'].max() if 'survey_date' in df.columns else 'N/A'}

KEY FINDINGS
------------
"""
    
    if 'roi_pct' in df.columns:
        report += f"• Average ROI: {df['roi_pct'].mean():.2f}% (SD: {df['roi_pct'].std():.2f}%)\n"
    
    if 'fdi_ownership_pct' in df.columns:
        foreign_firms = len(df[df['fdi_ownership_pct'] > 0])
        report += f"• Firms with FDI: {foreign_firms} ({foreign_firms/len(df)*100:.1f}%)\n"
        report += f"• Average FDI Ownership: {df[df['fdi_ownership_pct'] > 0]['fdi_ownership_pct'].mean():.1f}%\n"
    
    if 'policy_effectiveness_index' in df.columns:
        report += f"• Policy Effectiveness: {df['policy_effectiveness_index'].mean():.2f}/7.00\n"
    
    if 'innovation_avg' in df.columns:
        report += f"• Innovation Score: {df['innovation_avg'].mean():.2f}/5.00\n"
    
    if all(col in df.columns for col in ['fdi_composite', 'performance_composite']):
        corr = df[['fdi_composite', 'performance_composite']].corr().iloc[0, 1]
        report += f"• FDI-Performance Correlation: r = {corr:.3f}\n"
    
    report += """

RECOMMENDATIONS FOR ANALYSIS
-----------------------------
1. Test for multicollinearity using VIF before regression
2. Check assumptions of linear regression (normality, homoscedasticity)
3. Consider multilevel modeling for subsector effects
4. Use bootstrapping for robust confidence intervals
5. Perform sensitivity analysis on moderation effects

DATA FILES GENERATED
--------------------
✓ lagos_food_processing_fdi_dataset.csv - Main dataset
✓ lagos_food_processing_fdi_dataset.xlsx - Excel format with multiple sheets
✓ data_dictionary.json - Variable definitions
✓ CODEBOOK.txt - Detailed codebook
✓ variable_list.csv - Variable summary
✓ fdi_dataset_analysis.png - Comprehensive visualizations
✓ correlation_matrix.png - Key variables correlation
✓ analysis_report.txt - This report

================================================================================
"""
    
    # Save report
    with open('analysis_report.txt', 'w') as f:
        f.write(report)
    print("✓ Saved analysis_report.txt")
    
    return report


def main():
    """Main analysis execution"""
    
    try:
        # Load data
        df = load_and_examine_data()
        
        # Generate statistics
        generate_summary_statistics(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Test moderation hypothesis
        test_moderation_hypothesis(df)
        
        # Generate report
        report = generate_report(df)
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nAll analysis files have been generated successfully!")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Dataset file not found!")
        print("Please run 'generate_fdi_dataset.py' first to create the dataset.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check the data and try again.")


if __name__ == "__main__":
    main()