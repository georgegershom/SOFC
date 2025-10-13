#!/usr/bin/env python3
"""
Sample Analysis Code for Food Processing Firms FDI Research Dataset
Demonstrates key analytical approaches for the research questions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the integrated dataset for analysis"""
    
    print("Loading integrated dataset...")
    df = pd.read_csv('/workspace/integrated_food_processing_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def descriptive_analysis(df):
    """Perform comprehensive descriptive analysis"""
    
    print("\n" + "="*60)
    print("DESCRIPTIVE ANALYSIS")
    print("="*60)
    
    # Basic firm characteristics
    print("\n1. FIRM CHARACTERISTICS")
    print("-" * 30)
    
    print(f"Firm Size Distribution:")
    print(df['firm_size'].value_counts())
    print(f"\nFDI Presence: {df['fdi_presence'].sum()} firms ({df['fdi_presence'].mean()*100:.1f}%)")
    
    print(f"\nSubsector Distribution:")
    print(df['subsector'].value_counts())
    
    # Performance metrics summary
    print(f"\n2. PERFORMANCE METRICS")
    print("-" * 30)
    
    performance_vars = ['roi_percent', 'roa_percent', 'export_intensity_percent', 
                       'market_share_percent', 'operational_efficiency_score', 
                       'overall_performance_index']
    
    perf_summary = df[performance_vars].describe()
    print(perf_summary.round(2))
    
    # FDI constructs summary
    print(f"\n3. FDI CONSTRUCTS")
    print("-" * 30)
    
    fdi_constructs = ['knowledge_absorption_score', 'task_performance_score', 
                     'innovation_score', 'firm_resources_score']
    
    fdi_summary = df[fdi_constructs].describe()
    print(fdi_summary.round(2))
    
    # Government policy perception
    print(f"\n4. GOVERNMENT POLICY PERCEPTION")
    print("-" * 30)
    
    policy_vars = ['gp_tax_incentives_effectiveness', 'gp_regulatory_stability',
                  'gp_infrastructure_support', 'gp_ease_of_doing_business',
                  'gp_corruption_experience', 'policy_effectiveness_index']
    
    policy_summary = df[policy_vars].describe()
    print(policy_summary.round(2))
    
    return df

def correlation_analysis(df):
    """Analyze correlations between key variables"""
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Key variables for correlation analysis
    key_vars = [
        'fdi_presence', 'fdi_intensity', 'knowledge_absorption_score',
        'task_performance_score', 'innovation_score', 'firm_resources_score',
        'overall_performance_index', 'roi_percent', 'roa_percent',
        'policy_effectiveness_index', 'macro_gdp_growth_rate',
        'macro_ease_of_doing_business_score'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[key_vars].corr()
    
    print("\nCorrelation Matrix (Key Variables):")
    print(corr_matrix.round(3))
    
    # Focus on performance correlations
    performance_corrs = corr_matrix['overall_performance_index'].sort_values(ascending=False)
    print(f"\nCorrelations with Overall Performance Index:")
    for var, corr in performance_corrs.items():
        if var != 'overall_performance_index':
            print(f"{var:35}: {corr:6.3f}")
    
    return corr_matrix

def fdi_performance_analysis(df):
    """Analyze FDI-Performance relationship"""
    
    print("\n" + "="*60)
    print("FDI-PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Compare performance between FDI and non-FDI firms
    print("\n1. PERFORMANCE COMPARISON: FDI vs NON-FDI FIRMS")
    print("-" * 50)
    
    performance_vars = ['overall_performance_index', 'roi_percent', 'roa_percent', 
                       'export_intensity_percent', 'operational_efficiency_score']
    
    for var in performance_vars:
        fdi_firms = df[df['fdi_presence'] == 1][var]
        non_fdi_firms = df[df['fdi_presence'] == 0][var]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(fdi_firms, non_fdi_firms)
        
        print(f"\n{var}:")
        print(f"  FDI Firms (n={len(fdi_firms)}):     Mean={fdi_firms.mean():.2f}, SD={fdi_firms.std():.2f}")
        print(f"  Non-FDI Firms (n={len(non_fdi_firms)}): Mean={non_fdi_firms.mean():.2f}, SD={non_fdi_firms.std():.2f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.3f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    
    # FDI intensity analysis (for FDI firms only)
    print(f"\n2. FDI INTENSITY ANALYSIS")
    print("-" * 30)
    
    fdi_firms = df[df['fdi_presence'] == 1].copy()
    
    # Correlations between FDI intensity and performance
    intensity_corrs = fdi_firms[['fdi_intensity'] + performance_vars].corr()['fdi_intensity']
    
    print(f"\nFDI Intensity Correlations with Performance (FDI firms only, n={len(fdi_firms)}):")
    for var, corr in intensity_corrs.items():
        if var != 'fdi_intensity':
            print(f"{var:35}: {corr:6.3f}")
    
    return fdi_firms

def mechanism_analysis(df):
    """Analyze mechanisms through which FDI affects performance"""
    
    print("\n" + "="*60)
    print("MECHANISM ANALYSIS")
    print("="*60)
    
    # Compare FDI constructs between FDI and non-FDI firms
    print("\n1. FDI CONSTRUCTS: FDI vs NON-FDI FIRMS")
    print("-" * 45)
    
    mechanism_vars = ['knowledge_absorption_score', 'task_performance_score', 
                     'innovation_score', 'firm_resources_score']
    
    for var in mechanism_vars:
        fdi_firms = df[df['fdi_presence'] == 1][var]
        non_fdi_firms = df[df['fdi_presence'] == 0][var]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(fdi_firms, non_fdi_firms)
        
        print(f"\n{var}:")
        print(f"  FDI Firms:     Mean={fdi_firms.mean():.2f}, SD={fdi_firms.std():.2f}")
        print(f"  Non-FDI Firms: Mean={non_fdi_firms.mean():.2f}, SD={non_fdi_firms.std():.2f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.3f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    
    # Mediation analysis (simplified)
    print(f"\n2. MEDIATION ANALYSIS (Correlations)")
    print("-" * 40)
    
    print(f"\nCorrelations between FDI, Mechanisms, and Performance:")
    mediation_vars = ['fdi_presence'] + mechanism_vars + ['overall_performance_index']
    mediation_corr = df[mediation_vars].corr()
    
    # Show key mediation paths
    print(f"\nFDI → Mechanisms:")
    for mech in mechanism_vars:
        corr = mediation_corr.loc['fdi_presence', mech]
        print(f"  FDI → {mech:25}: {corr:6.3f}")
    
    print(f"\nMechanisms → Performance:")
    for mech in mechanism_vars:
        corr = mediation_corr.loc[mech, 'overall_performance_index']
        print(f"  {mech:30} → Performance: {corr:6.3f}")
    
    return mediation_corr

def policy_moderation_analysis(df):
    """Analyze government policy moderation effects"""
    
    print("\n" + "="*60)
    print("POLICY MODERATION ANALYSIS")
    print("="*60)
    
    # Simple moderation analysis using correlation
    print("\n1. POLICY EFFECTIVENESS AND FDI-PERFORMANCE RELATIONSHIP")
    print("-" * 60)
    
    # Split sample by policy effectiveness (median split)
    median_policy = df['policy_effectiveness_index'].median()
    high_policy = df[df['policy_effectiveness_index'] > median_policy]
    low_policy = df[df['policy_effectiveness_index'] <= median_policy]
    
    print(f"Median Policy Effectiveness: {median_policy:.2f}")
    print(f"High Policy Environment: n={len(high_policy)} firms (Policy Index > {median_policy:.2f})")
    print(f"Low Policy Environment:  n={len(low_policy)} firms (Policy Index ≤ {median_policy:.2f})")
    
    # FDI-Performance correlation in each group
    high_policy_corr = high_policy['fdi_presence'].corr(high_policy['overall_performance_index'])
    low_policy_corr = low_policy['fdi_presence'].corr(low_policy['overall_performance_index'])
    
    print(f"\nFDI-Performance Correlations:")
    print(f"  High Policy Environment: {high_policy_corr:.3f}")
    print(f"  Low Policy Environment:  {low_policy_corr:.3f}")
    print(f"  Difference: {high_policy_corr - low_policy_corr:.3f}")
    
    # Interaction term analysis
    print(f"\n2. INTERACTION TERM ANALYSIS")
    print("-" * 35)
    
    interaction_vars = ['fdi_x_policy_effectiveness', 'fdi_intensity_x_policy_effectiveness',
                       'knowledge_absorption_x_policy']
    
    for var in interaction_vars:
        if var in df.columns:
            corr = df[var].corr(df['overall_performance_index'])
            print(f"{var:35}: {corr:6.3f}")
    
    return high_policy, low_policy

def regression_analysis(df):
    """Perform regression analysis for main hypotheses"""
    
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)
    
    # Prepare variables for regression
    # Dependent variable
    y = df['overall_performance_index']
    
    # Independent variables
    X_vars = [
        'fdi_presence', 'fdi_intensity', 'knowledge_absorption_score',
        'task_performance_score', 'innovation_score', 'firm_resources_score',
        'policy_effectiveness_index', 'firm_age', 'employees'
    ]
    
    # Add firm size dummy
    df_reg = df.copy()
    df_reg['large_firm'] = (df_reg['firm_size'] == 'Large').astype(int)
    X_vars.append('large_firm')
    
    # Add macro controls
    macro_controls = ['macro_gdp_growth_rate', 'macro_inflation_rate', 
                     'macro_ease_of_doing_business_score']
    X_vars.extend(macro_controls)
    
    # Prepare data
    X = df_reg[X_vars].fillna(df_reg[X_vars].mean())  # Simple imputation
    
    print(f"\nRegression Setup:")
    print(f"  Dependent Variable: {y.name}")
    print(f"  Independent Variables: {len(X_vars)}")
    print(f"  Sample Size: {len(X)}")
    
    # Model 1: Basic FDI effects
    print(f"\n1. MODEL 1: BASIC FDI EFFECTS")
    print("-" * 35)
    
    X1 = X[['fdi_presence', 'firm_age', 'employees', 'large_firm'] + macro_controls]
    
    # Standardize variables
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    
    # Fit model
    model1 = LinearRegression()
    model1.fit(X1_scaled, y)
    
    # Predictions and metrics
    y_pred1 = model1.predict(X1_scaled)
    r2_1 = r2_score(y, y_pred1)
    rmse_1 = np.sqrt(mean_squared_error(y, y_pred1))
    
    print(f"  R-squared: {r2_1:.3f}")
    print(f"  RMSE: {rmse_1:.3f}")
    
    # Coefficients
    print(f"\n  Coefficients:")
    for i, var in enumerate(X1.columns):
        print(f"    {var:30}: {model1.coef_[i]:7.3f}")
    
    # Model 2: FDI mechanisms
    print(f"\n2. MODEL 2: FDI MECHANISMS")
    print("-" * 30)
    
    mechanism_vars = ['knowledge_absorption_score', 'task_performance_score', 
                     'innovation_score', 'firm_resources_score']
    
    X2 = X[['fdi_presence'] + mechanism_vars + ['firm_age', 'employees', 'large_firm'] + macro_controls]
    
    # Standardize variables
    X2_scaled = scaler.fit_transform(X2)
    
    # Fit model
    model2 = LinearRegression()
    model2.fit(X2_scaled, y)
    
    # Predictions and metrics
    y_pred2 = model2.predict(X2_scaled)
    r2_2 = r2_score(y, y_pred2)
    rmse_2 = np.sqrt(mean_squared_error(y, y_pred2))
    
    print(f"  R-squared: {r2_2:.3f}")
    print(f"  RMSE: {rmse_2:.3f}")
    print(f"  R-squared improvement: {r2_2 - r2_1:.3f}")
    
    # Coefficients
    print(f"\n  Coefficients:")
    for i, var in enumerate(X2.columns):
        print(f"    {var:30}: {model2.coef_[i]:7.3f}")
    
    # Model 3: Policy moderation
    print(f"\n3. MODEL 3: POLICY MODERATION")
    print("-" * 35)
    
    # Add interaction term
    df_reg['fdi_x_policy'] = df_reg['fdi_presence'] * df_reg['policy_effectiveness_index']
    
    X3 = X[mechanism_vars + ['fdi_presence', 'policy_effectiveness_index'] + 
           ['firm_age', 'employees', 'large_firm'] + macro_controls]
    X3['fdi_x_policy'] = df_reg['fdi_x_policy']
    
    # Standardize variables
    X3_scaled = scaler.fit_transform(X3)
    
    # Fit model
    model3 = LinearRegression()
    model3.fit(X3_scaled, y)
    
    # Predictions and metrics
    y_pred3 = model3.predict(X3_scaled)
    r2_3 = r2_score(y, y_pred3)
    rmse_3 = np.sqrt(mean_squared_error(y, y_pred3))
    
    print(f"  R-squared: {r2_3:.3f}")
    print(f"  RMSE: {rmse_3:.3f}")
    print(f"  R-squared improvement: {r2_3 - r2_2:.3f}")
    
    # Coefficients
    print(f"\n  Coefficients:")
    for i, var in enumerate(X3.columns):
        print(f"    {var:30}: {model3.coef_[i]:7.3f}")
    
    return model1, model2, model3

def subsector_analysis(df):
    """Analyze differences across food processing subsectors"""
    
    print("\n" + "="*60)
    print("SUBSECTOR ANALYSIS")
    print("="*60)
    
    # Performance by subsector
    print("\n1. PERFORMANCE BY SUBSECTOR")
    print("-" * 35)
    
    subsector_performance = df.groupby('subsector')['overall_performance_index'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    print(subsector_performance)
    
    # FDI presence by subsector
    print(f"\n2. FDI PRESENCE BY SUBSECTOR")
    print("-" * 35)
    
    fdi_by_subsector = df.groupby('subsector').agg({
        'fdi_presence': ['count', 'sum', 'mean'],
        'fdi_amount_million_usd': 'mean'
    }).round(2)
    
    fdi_by_subsector.columns = ['Total_Firms', 'FDI_Firms', 'FDI_Rate', 'Avg_FDI_Amount']
    print(fdi_by_subsector)
    
    # Policy perception by subsector
    print(f"\n3. POLICY EFFECTIVENESS BY SUBSECTOR")
    print("-" * 40)
    
    policy_by_subsector = df.groupby('subsector')['policy_effectiveness_index'].agg([
        'mean', 'std'
    ]).round(2)
    
    print(policy_by_subsector)
    
    return subsector_performance, fdi_by_subsector

def create_summary_report(df):
    """Create executive summary of key findings"""
    
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY OF KEY FINDINGS")
    print("="*60)
    
    # Sample characteristics
    total_firms = len(df)
    fdi_firms = df['fdi_presence'].sum()
    fdi_rate = fdi_firms / total_firms * 100
    
    # Performance differences
    fdi_performance = df[df['fdi_presence'] == 1]['overall_performance_index'].mean()
    non_fdi_performance = df[df['fdi_presence'] == 0]['overall_performance_index'].mean()
    performance_diff = fdi_performance - non_fdi_performance
    
    # Policy effectiveness
    avg_policy_effectiveness = df['policy_effectiveness_index'].mean()
    
    # Correlations
    fdi_perf_corr = df['fdi_presence'].corr(df['overall_performance_index'])
    policy_perf_corr = df['policy_effectiveness_index'].corr(df['overall_performance_index'])
    
    print(f"\n1. SAMPLE CHARACTERISTICS")
    print(f"   • Total firms analyzed: {total_firms}")
    print(f"   • Firms with FDI presence: {fdi_firms} ({fdi_rate:.1f}%)")
    print(f"   • Large firms: {(df['firm_size'] == 'Large').sum()} ({(df['firm_size'] == 'Large').mean()*100:.1f}%)")
    print(f"   • Average firm age: {df['firm_age'].mean():.1f} years")
    
    print(f"\n2. KEY FINDINGS")
    print(f"   • FDI firms outperform non-FDI firms by {performance_diff:.2f} points")
    print(f"   • FDI-Performance correlation: {fdi_perf_corr:.3f}")
    print(f"   • Policy-Performance correlation: {policy_perf_corr:.3f}")
    print(f"   • Average policy effectiveness: {avg_policy_effectiveness:.2f}/7.0")
    
    print(f"\n3. MECHANISM INSIGHTS")
    mechanisms = ['knowledge_absorption_score', 'task_performance_score', 'innovation_score']
    for mech in mechanisms:
        fdi_mech = df[df['fdi_presence'] == 1][mech].mean()
        non_fdi_mech = df[df['fdi_presence'] == 0][mech].mean()
        diff = fdi_mech - non_fdi_mech
        print(f"   • {mech.replace('_score', '').replace('_', ' ').title()}: FDI firms higher by {diff:.2f}")
    
    print(f"\n4. POLICY IMPLICATIONS")
    # High vs low policy environment
    median_policy = df['policy_effectiveness_index'].median()
    high_policy_fdi_effect = df[df['policy_effectiveness_index'] > median_policy]['fdi_presence'].corr(
        df[df['policy_effectiveness_index'] > median_policy]['overall_performance_index'])
    low_policy_fdi_effect = df[df['policy_effectiveness_index'] <= median_policy]['fdi_presence'].corr(
        df[df['policy_effectiveness_index'] <= median_policy]['overall_performance_index'])
    
    print(f"   • FDI effect in high policy environment: {high_policy_fdi_effect:.3f}")
    print(f"   • FDI effect in low policy environment: {low_policy_fdi_effect:.3f}")
    print(f"   • Policy moderation effect: {high_policy_fdi_effect - low_policy_fdi_effect:.3f}")

def main():
    """Main analysis function"""
    
    print("FOOD PROCESSING FIRMS FDI RESEARCH - SAMPLE ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run analyses
    df = descriptive_analysis(df)
    corr_matrix = correlation_analysis(df)
    fdi_firms = fdi_performance_analysis(df)
    mediation_corr = mechanism_analysis(df)
    high_policy, low_policy = policy_moderation_analysis(df)
    model1, model2, model3 = regression_analysis(df)
    subsector_perf, subsector_fdi = subsector_analysis(df)
    
    # Create summary
    create_summary_report(df)
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nDataset: integrated_food_processing_dataset.csv")
    print(f"Documentation: data_dictionary.md, methodology_documentation.md")
    print(f"This analysis code: sample_analysis.py")
    
    return df

if __name__ == "__main__":
    # Run the complete analysis
    df = main()