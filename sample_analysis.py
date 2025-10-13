#!/usr/bin/env python3
"""
Sample Analysis Script for Lagos Food Processing FDI Dataset
Demonstrates basic analysis capabilities and research applications
"""

import pandas as pd
import numpy as np

def load_and_explore_data():
    """Load dataset and perform basic exploration"""
    print("=" * 60)
    print("LAGOS FOOD PROCESSING FDI DATASET - SAMPLE ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('lagos_food_processing_fdi_dataset.csv')
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]} firms × {df.shape[1]} variables")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    return df

def descriptive_statistics(df):
    """Generate descriptive statistics"""
    print(f"\n📈 DESCRIPTIVE STATISTICS")
    print("-" * 40)
    
    # Ownership distribution
    print("\n🏢 Ownership Distribution:")
    ownership_dist = df['ownership_type'].value_counts()
    for ownership, count in ownership_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {ownership}: {count} firms ({pct:.1f}%)")
    
    # Subsector distribution
    print("\n🏭 Top 5 Subsectors:")
    subsector_dist = df['subsector'].value_counts().head()
    for subsector, count in subsector_dist.items():
        print(f"  {subsector}: {count} firms")
    
    # Performance metrics by ownership
    print("\n💰 Performance by Ownership Type:")
    perf_by_ownership = df.groupby('ownership_type')[['roi_percent', 'roa_percent', 'operational_efficiency']].mean()
    print(perf_by_ownership.round(2))

def fdi_impact_analysis(df):
    """Analyze FDI impact on firm performance"""
    print(f"\n🌍 FDI IMPACT ANALYSIS")
    print("-" * 40)
    
    # FDI vs Non-FDI comparison
    fdi_comparison = df.groupby('has_fdi')[['roi_percent', 'roa_percent', 'export_intensity_percent', 'rd_spend_percent']].agg(['mean', 'std'])
    
    print("\n📊 FDI vs Non-FDI Firms:")
    print("Performance Metrics (Mean ± Std):")
    
    metrics = ['roi_percent', 'roa_percent', 'export_intensity_percent', 'rd_spend_percent']
    metric_names = ['ROI (%)', 'ROA (%)', 'Export Intensity (%)', 'R&D Spend (%)']
    
    for metric, name in zip(metrics, metric_names):
        non_fdi_mean = fdi_comparison.loc[0, (metric, 'mean')]
        non_fdi_std = fdi_comparison.loc[0, (metric, 'std')]
        fdi_mean = fdi_comparison.loc[1, (metric, 'mean')]
        fdi_std = fdi_comparison.loc[1, (metric, 'std')]
        
        print(f"  {name}:")
        print(f"    Non-FDI: {non_fdi_mean:.1f} ± {non_fdi_std:.1f}")
        print(f"    FDI:     {fdi_mean:.1f} ± {fdi_std:.1f}")
        print(f"    Difference: {fdi_mean - non_fdi_mean:+.1f}")

def correlation_analysis(df):
    """Perform correlation analysis"""
    print(f"\n🔗 CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Key variables for correlation
    key_vars = [
        'roi_percent', 'roa_percent', 'has_fdi', 
        'knowledge_acquisition', 'product_innovation', 
        'policy_effectiveness_index', 'rd_spend_percent',
        'firm_age_years', 'employees'
    ]
    
    corr_matrix = df[key_vars].corr()
    
    print("\n🎯 Key Correlations with ROI:")
    roi_corrs = corr_matrix['roi_percent'].drop('roi_percent').sort_values(key=abs, ascending=False)
    for var, corr in roi_corrs.head(6).items():
        print(f"  {var}: {corr:+.3f}")
    
    print("\n🎯 Key Correlations with FDI:")
    fdi_corrs = corr_matrix['has_fdi'].drop('has_fdi').sort_values(key=abs, ascending=False)
    for var, corr in fdi_corrs.head(6).items():
        print(f"  {var}: {corr:+.3f}")

def policy_analysis(df):
    """Analyze government policy perceptions"""
    print(f"\n🏛️ GOVERNMENT POLICY ANALYSIS")
    print("-" * 40)
    
    policy_vars = [
        'tax_incentive_effectiveness', 'regulatory_predictability',
        'transport_infrastructure', 'power_supply_reliability',
        'corruption_frequency', 'policy_effectiveness_index'
    ]
    
    print("\n📋 Policy Perception Scores (1-7 scale, except index):")
    policy_stats = df[policy_vars].describe()
    
    for var in policy_vars:
        mean_val = policy_stats.loc['mean', var]
        std_val = policy_stats.loc['std', var]
        if 'index' in var:
            print(f"  {var}: {mean_val:.2f} ± {std_val:.2f}")
        else:
            print(f"  {var}: {mean_val:.1f} ± {std_val:.1f}")
    
    # Policy effectiveness by ownership type
    print("\n🏢 Policy Effectiveness by Ownership:")
    policy_by_ownership = df.groupby('ownership_type')['policy_effectiveness_index'].mean().sort_values(ascending=False)
    for ownership, score in policy_by_ownership.items():
        print(f"  {ownership}: {score:.2f}")

def research_insights(df):
    """Generate research insights and recommendations"""
    print(f"\n🔍 RESEARCH INSIGHTS")
    print("-" * 40)
    
    # Calculate key statistics
    fdi_firms = df[df['has_fdi'] == 1]
    non_fdi_firms = df[df['has_fdi'] == 0]
    
    roi_diff = fdi_firms['roi_percent'].mean() - non_fdi_firms['roi_percent'].mean()
    innovation_diff = fdi_firms['product_innovation'].mean() - non_fdi_firms['product_innovation'].mean()
    
    print("\n💡 Key Findings:")
    print(f"  • FDI firms show {roi_diff:+.1f}% higher ROI on average")
    print(f"  • FDI firms have {innovation_diff:+.2f} higher product innovation scores")
    print(f"  • {len(fdi_firms)} firms ({len(fdi_firms)/len(df)*100:.1f}%) have FDI")
    print(f"  • Average policy effectiveness index: {df['policy_effectiveness_index'].mean():.2f}/7")
    
    # Correlations for moderation analysis
    fdi_policy_corr = np.corrcoef(df['has_fdi'], df['policy_effectiveness_index'])[0,1]
    roi_policy_corr = np.corrcoef(df['roi_percent'], df['policy_effectiveness_index'])[0,1]
    
    print(f"\n🎯 Moderation Analysis Setup:")
    print(f"  • FDI-Policy correlation: {fdi_policy_corr:+.3f}")
    print(f"  • ROI-Policy correlation: {roi_policy_corr:+.3f}")
    print(f"  • Suitable for interaction effects testing")
    
    print(f"\n📚 Recommended Analyses:")
    print("  1. Hierarchical regression: ROI ~ FDI + Policy + FDI×Policy")
    print("  2. Mediation analysis: FDI → Knowledge Absorption → Performance")
    print("  3. Subsector analysis: Performance differences across food sectors")
    print("  4. Size effects: Moderation by firm size (employees/assets)")

def main():
    """Main analysis function"""
    # Load and explore data
    df = load_and_explore_data()
    
    # Run analyses
    descriptive_statistics(df)
    fdi_impact_analysis(df)
    correlation_analysis(df)
    policy_analysis(df)
    research_insights(df)
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("📄 See data_dictionary.md for variable definitions")
    print("📖 See README.md for detailed usage instructions")
    print("🔬 Dataset ready for advanced statistical analysis!")

if __name__ == "__main__":
    main()