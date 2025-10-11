#!/usr/bin/env python3
"""
Data Exploration Script for SSA Macroeconomic Dataset
Provides quick analysis and visualizations
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the SSA macroeconomic dataset"""
    df = pd.read_csv('/workspace/data/ssa_macroeconomic_data.csv')
    print("=" * 80)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Records: {len(df):,}")
    print(f"Countries: {df['Country_Code'].nunique()}")
    print(f"Time Period: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Variables: {len(df.columns)}\n")
    return df

def show_countries(df):
    """Display all countries in the dataset"""
    print("=" * 80)
    print("COUNTRIES IN DATASET")
    print("=" * 80)
    countries = df[['Country_Code', 'Country_Name']].drop_duplicates().sort_values('Country_Name')
    for idx, row in countries.iterrows():
        print(f"  {row['Country_Code']}: {row['Country_Name']}")
    print()

def show_sample_data(df):
    """Show sample data from the dataset"""
    print("=" * 80)
    print("SAMPLE DATA (First 10 rows)")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
    print()

def show_variable_summary(df):
    """Show summary statistics for all numeric variables"""
    print("=" * 80)
    print("VARIABLE SUMMARY STATISTICS")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    print(summary.to_string())
    print()

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    print("=" * 80)
    print("MISSING DATA ANALYSIS")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Variable': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percent': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("✓ No missing data detected!")
    print()

def analyze_by_country(df, variable='GDP_Growth'):
    """Show statistics by country for a specific variable"""
    print("=" * 80)
    print(f"ANALYSIS BY COUNTRY: {variable}")
    print("=" * 80)
    
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in dataset.")
        return
    
    country_stats = df.groupby('Country_Name')[variable].agg([
        ('Mean', 'mean'),
        ('Std_Dev', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Latest_2024', lambda x: df[(df['Country_Name'] == x.name) & (df['Year'] == 2024)][variable].values[0] if len(df[(df['Country_Name'] == x.name) & (df['Year'] == 2024)]) > 0 else np.nan)
    ]).round(2)
    
    country_stats = country_stats.sort_values('Mean', ascending=False)
    print(country_stats.to_string())
    print()

def analyze_time_trends(df, country_code='KEN'):
    """Show time trends for a specific country"""
    print("=" * 80)
    print(f"TIME TRENDS FOR: {country_code}")
    print("=" * 80)
    
    country_data = df[df['Country_Code'] == country_code].sort_values('Year')
    
    if len(country_data) == 0:
        print(f"No data found for country code: {country_code}")
        return
    
    # Select key variables
    key_vars = ['Year', 'GDP_Growth', 'Inflation', 'Unemployment', 
                'Mobile_Subscriptions_per_100', 'Internet_Users_Percent']
    
    display_data = country_data[key_vars]
    print(display_data.to_string(index=False))
    print()

def analyze_correlations(df):
    """Analyze correlations between key variables"""
    print("=" * 80)
    print("CORRELATION ANALYSIS (Key Variables)")
    print("=" * 80)
    
    key_vars = ['GDP_Growth', 'Inflation', 'Unemployment', 'Exchange_Rate',
                'Interest_Rate', 'Debt_to_GDP', 'Mobile_Subscriptions_per_100',
                'Internet_Users_Percent']
    
    corr_matrix = df[key_vars].corr().round(3)
    print(corr_matrix.to_string())
    print()

def fintech_risk_indicators(df):
    """Calculate FinTech-specific risk indicators"""
    print("=" * 80)
    print("FINTECH RISK INDICATORS (2024)")
    print("=" * 80)
    
    # Filter for latest year
    latest = df[df['Year'] == 2024].copy()
    
    # Calculate risk scores (simple example)
    latest['Macro_Risk_Score'] = (
        (latest['GDP_Growth_Volatility'] / latest['GDP_Growth_Volatility'].max()) * 0.3 +
        (latest['Inflation'] / latest['Inflation'].max()) * 0.3 +
        (latest['Exchange_Rate_Volatility'] / latest['Exchange_Rate_Volatility'].max()) * 0.2 +
        (latest['Debt_to_GDP'] / latest['Debt_to_GDP'].max()) * 0.2
    ) * 100
    
    latest['Digital_Readiness_Score'] = (
        (latest['Mobile_Subscriptions_per_100'] / latest['Mobile_Subscriptions_per_100'].max()) * 0.4 +
        (latest['Internet_Users_Percent'] / latest['Internet_Users_Percent'].max()) * 0.4 +
        (latest['Secure_Servers_per_Million'] / latest['Secure_Servers_per_Million'].max()) * 0.2
    ) * 100
    
    risk_summary = latest[['Country_Name', 'Macro_Risk_Score', 'Digital_Readiness_Score']].round(2)
    risk_summary = risk_summary.sort_values('Macro_Risk_Score', ascending=True)
    
    print(risk_summary.to_string(index=False))
    print("\nNote: Lower Macro Risk Score = Lower Risk")
    print("      Higher Digital Readiness = Better Infrastructure")
    print()

def export_country_profiles(df):
    """Export individual country profiles"""
    print("=" * 80)
    print("EXPORTING COUNTRY PROFILES")
    print("=" * 80)
    
    import os
    output_dir = '/workspace/data/country_profiles'
    os.makedirs(output_dir, exist_ok=True)
    
    for country_code in df['Country_Code'].unique():
        country_data = df[df['Country_Code'] == country_code].sort_values('Year')
        filename = f"{output_dir}/{country_code}_profile.csv"
        country_data.to_csv(filename, index=False)
    
    print(f"✓ Exported {df['Country_Code'].nunique()} country profiles to: {output_dir}")
    print()

def main():
    """Main execution"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  SSA MACROECONOMIC DATA EXPLORER".center(78) + "║")
    print("║" + "  For FinTech Early Warning Model Research".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Load data
    df = load_data()
    
    # Run analyses
    show_countries(df)
    show_sample_data(df)
    show_variable_summary(df)
    analyze_missing_data(df)
    
    # Country-specific analyses
    analyze_by_country(df, 'GDP_Growth')
    analyze_by_country(df, 'Inflation')
    analyze_by_country(df, 'Internet_Users_Percent')
    
    # Time trends for Kenya (example)
    analyze_time_trends(df, 'KEN')
    
    # Correlation analysis
    analyze_correlations(df)
    
    # FinTech risk indicators
    fintech_risk_indicators(df)
    
    # Export country profiles
    export_country_profiles(df)
    
    print("=" * 80)
    print("EXPLORATION COMPLETE!")
    print("=" * 80)
    print("\nDataset is ready for FinTech Early Warning Model development.")
    print("Files available:")
    print("  - /workspace/data/ssa_macroeconomic_data.csv")
    print("  - /workspace/data/summary_statistics.csv")
    print("  - /workspace/data/data_dictionary.md")
    print("  - /workspace/data/country_profiles/*.csv")
    print()

if __name__ == "__main__":
    main()
