#!/usr/bin/env python3
"""
Example Analysis Script for FDI Food Processing Dataset
Quick start guide for data analysis
"""

import pandas as pd
import numpy as np

def load_and_explore():
    """Load dataset and show basic exploration"""
    print("=" * 70)
    print("LOADING FDI FOOD PROCESSING DATASET")
    print("=" * 70)
    print()
    
    # Load data
    df = pd.read_csv('fdi_food_processing_firms_dataset.csv')
    
    print(f"✓ Dataset loaded: {len(df)} firms, {len(df.columns)} variables")
    print()
    
    # Basic info
    print("=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print()
    
    # Sample composition
    print("=" * 70)
    print("SAMPLE COMPOSITION")
    print("=" * 70)
    print("\nBy Firm Size:")
    print(df['Firm_Size'].value_counts())
    print("\nBy FDI Presence:")
    print(df['FDI_Presence'].value_counts())
    print("\nTop 5 Subsectors:")
    print(df['Subsector'].value_counts().head())
    print()
    
    return df


def descriptive_statistics(df):
    """Show key descriptive statistics"""
    print("=" * 70)
    print("KEY DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print()
    
    # Performance metrics
    print("Performance Metrics:")
    perf_vars = ['ROI_Percentage', 'ROA_Percentage', 'Revenue_USD', 'Export_Intensity']
    print(df[perf_vars].describe().round(2))
    print()
    
    # FDI Constructs
    print("FDI Constructs (1-5 Likert Scale):")
    fdi_vars = ['Knowledge_Absorption', 'Task_Performance', 'Innovation_Score', 'Firm_Resources_Score']
    print(df[fdi_vars].describe().round(2))
    print()
    
    # Government Policy Perceptions
    print("Government Policy Perceptions (1-7 Likert Scale):")
    policy_vars = ['Tax_Incentives_Score', 'Regulatory_Stability', 'Infrastructure_Support', 
                   'Corruption_Experience', 'Govt_Policy_Score']
    print(df[policy_vars].describe().round(2))
    print()


def compare_fdi_nonfdi(df):
    """Compare firms with and without FDI"""
    print("=" * 70)
    print("COMPARISON: FDI vs NON-FDI FIRMS")
    print("=" * 70)
    print()
    
    fdi_firms = df[df['FDI_Presence'] == 1]
    non_fdi_firms = df[df['FDI_Presence'] == 0]
    
    print(f"FDI Firms: {len(fdi_firms)}")
    print(f"Non-FDI Firms: {len(non_fdi_firms)}")
    print()
    
    comparison_vars = [
        'ROI_Percentage', 'ROA_Percentage', 'Innovation_Score',
        'Knowledge_Absorption', 'Export_Intensity', 'Operational_Efficiency'
    ]
    
    print("Average Values Comparison:")
    print("-" * 70)
    print(f"{'Variable':<30} {'FDI':<15} {'Non-FDI':<15} {'Difference':<15}")
    print("-" * 70)
    
    for var in comparison_vars:
        fdi_mean = fdi_firms[var].mean()
        non_fdi_mean = non_fdi_firms[var].mean()
        diff = fdi_mean - non_fdi_mean
        
        print(f"{var:<30} {fdi_mean:>10.2f}    {non_fdi_mean:>10.2f}    {diff:>+10.2f}")
    
    print()


def correlation_analysis(df):
    """Show key correlations"""
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Key variables for correlation
    key_vars = [
        'FDI_Presence', 'Knowledge_Absorption', 'Innovation_Score',
        'ROI_Percentage', 'ROA_Percentage', 'Govt_Policy_Score'
    ]
    
    corr_matrix = df[key_vars].corr()
    
    print("Correlation Matrix (Key Variables):")
    print(corr_matrix.round(3))
    print()
    
    # Highlight important correlations
    print("Key Findings:")
    print(f"  - FDI ↔ Innovation: {df['FDI_Presence'].corr(df['Innovation_Score']):.3f}")
    print(f"  - FDI ↔ Knowledge Absorption: {df['FDI_Presence'].corr(df['Knowledge_Absorption']):.3f}")
    print(f"  - Innovation ↔ ROI: {df['Innovation_Score'].corr(df['ROI_Percentage']):.3f}")
    print(f"  - Policy ↔ ROI: {df['Govt_Policy_Score'].corr(df['ROI_Percentage']):.3f}")
    print()


def regression_example(df):
    """Simple regression example (OLS)"""
    print("=" * 70)
    print("EXAMPLE: SIMPLE REGRESSION ANALYSIS")
    print("=" * 70)
    print()
    
    try:
        import statsmodels.api as sm
        
        # Prepare variables
        X = df[['FDI_Presence', 'Innovation_Score', 'Firm_Age', 'Govt_Policy_Score']]
        X = sm.add_constant(X)
        y = df['ROI_Percentage']
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        print("Dependent Variable: ROI_Percentage")
        print("Independent Variables: FDI_Presence, Innovation_Score, Firm_Age, Govt_Policy_Score")
        print()
        print(model.summary())
        print()
        
    except ImportError:
        print("Note: Install statsmodels for regression analysis:")
        print("  pip install statsmodels")
        print()
        print("Regression Model Specification:")
        print("  DV: ROI_Percentage")
        print("  IVs: FDI_Presence, Innovation_Score, Firm_Age, Govt_Policy_Score")
        print()


def moderation_setup(df):
    """Show moderation analysis setup"""
    print("=" * 70)
    print("MODERATION ANALYSIS SETUP")
    print("=" * 70)
    print()
    
    print("Hypothesis: Government policy moderates the FDI-Performance relationship")
    print()
    print("Model Specification:")
    print("  ROI = β0 + β1(FDI) + β2(Policy) + β3(FDI × Policy) + Controls + ε")
    print()
    print("Variables ready in dataset:")
    print("  ✓ FDI_Presence (binary)")
    print("  ✓ Govt_Policy_Score (continuous, 1-7)")
    print("  ✓ FDI_Policy_Interaction (pre-calculated)")
    print("  ✓ ROI_Percentage (dependent variable)")
    print()
    print("To test moderation:")
    print("  1. Run regression with main effects (FDI + Policy)")
    print("  2. Add interaction term (FDI_Policy_Interaction)")
    print("  3. Check if R² significantly increases")
    print("  4. Interpret coefficient of interaction term")
    print()
    
    # Show interaction term statistics
    print("Interaction Term Statistics:")
    print(df['FDI_Policy_Interaction'].describe().round(2))
    print()


def export_for_stata(df):
    """Export data for STATA analysis"""
    print("=" * 70)
    print("EXPORTING FOR STATA")
    print("=" * 70)
    print()
    
    try:
        df.to_stata('fdi_dataset_for_stata.dta', write_index=False)
        print("✓ STATA file created: fdi_dataset_for_stata.dta")
        print()
        print("STATA Commands to get started:")
        print("  use fdi_dataset_for_stata.dta, clear")
        print("  describe")
        print("  summarize")
        print("  regress ROI_Percentage FDI_Presence Innovation_Score Govt_Policy_Score")
        print()
    except Exception as e:
        print(f"Note: Could not create STATA file: {e}")
        print("Use CSV file with: import delimited 'fdi_food_processing_firms_dataset.csv'")
        print()


def main():
    """Run all example analyses"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FDI & Food Processing Firms - Example Analysis".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Load data
    df = load_and_explore()
    
    # Descriptive statistics
    descriptive_statistics(df)
    
    # Compare FDI vs non-FDI
    compare_fdi_nonfdi(df)
    
    # Correlations
    correlation_analysis(df)
    
    # Regression example
    regression_example(df)
    
    # Moderation setup
    moderation_setup(df)
    
    # Export for STATA
    export_for_stata(df)
    
    print("=" * 70)
    print("✅ EXAMPLE ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Review the outputs above")
    print("  2. Modify this script for your specific hypotheses")
    print("  3. Use your preferred software (STATA, R, SPSS)")
    print("  4. Refer to DATASET_README.md for detailed guidance")
    print()
    print("Happy analyzing! 📊")
    print()


if __name__ == "__main__":
    main()
