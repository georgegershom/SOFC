#!/usr/bin/env python3
"""
SSA FinTech Data Enhancer
Adds calculated indicators, volatilities, and additional features for the early warning model.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def calculate_volatility(df, value_col, country_col='country_name', window=3):
    """Calculate rolling volatility (standard deviation) for a given indicator."""
    result_data = []
    
    for country in df[country_col].unique():
        country_data = df[df[country_col] == country].copy()
        country_data = country_data.sort_values('year')
        
        # Calculate rolling standard deviation
        country_data[f'{value_col}_volatility'] = country_data[value_col].rolling(
            window=window, min_periods=2
        ).std()
        
        result_data.append(country_data)
    
    return pd.concat(result_data, ignore_index=True)

def calculate_growth_rates(df, value_col, country_col='country_name'):
    """Calculate year-over-year growth rates."""
    result_data = []
    
    for country in df[country_col].unique():
        country_data = df[df[country_col] == country].copy()
        country_data = country_data.sort_values('year')
        
        # Calculate percentage change
        country_data[f'{value_col}_growth'] = country_data[value_col].pct_change() * 100
        
        result_data.append(country_data)
    
    return pd.concat(result_data, ignore_index=True)

def add_fintech_risk_indicators(df):
    """Add specific indicators relevant for FinTech risk assessment."""
    
    # Digital divide indicator (gap between mobile and internet penetration)
    df['digital_divide'] = df['mobile_subs'] - df['internet_users']
    
    # Financial inclusion proxy (combination of mobile penetration and financial depth)
    df['financial_inclusion_proxy'] = (df['mobile_subs'] * df['financial_depth']) / 100
    
    # Economic stability index (inverse of GDP growth volatility + inflation volatility)
    if 'gdp_growth_volatility' in df.columns and 'inflation_volatility' in df.columns:
        df['economic_instability'] = df['gdp_growth_volatility'] + df['inflation_volatility']
    
    # Digital infrastructure readiness
    df['digital_infrastructure'] = (df['internet_users'] + df['secure_servers']/10) / 2
    
    # External vulnerability (combination of exchange rate volatility and trade openness)
    if 'exchange_rate_volatility' in df.columns:
        df['external_vulnerability'] = df['exchange_rate_volatility'] * (df['trade_openness'] / 100)
    
    return df

def create_risk_categories(df):
    """Create categorical risk indicators based on thresholds."""
    
    # GDP Growth Risk Categories
    df['gdp_growth_risk'] = pd.cut(
        df['gdp_growth'], 
        bins=[-np.inf, 0, 3, 6, np.inf], 
        labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
    )
    
    # Inflation Risk Categories
    df['inflation_risk'] = pd.cut(
        df['inflation'], 
        bins=[-np.inf, 0, 5, 10, np.inf], 
        labels=['Deflation Risk', 'Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Unemployment Risk Categories
    df['unemployment_risk'] = pd.cut(
        df['unemployment'], 
        bins=[-np.inf, 5, 10, 20, np.inf], 
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    )
    
    # Digital Infrastructure Categories
    df['digital_infrastructure_level'] = pd.cut(
        df['internet_users'], 
        bins=[-np.inf, 20, 40, 60, np.inf], 
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    return df

def add_regional_comparisons(df):
    """Add regional comparison indicators."""
    
    # For each year, calculate regional averages and percentiles
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != 'year':
            # Regional average by year
            df[f'{col}_regional_avg'] = df.groupby('year')[col].transform('mean')
            
            # Percentile rank within region by year
            df[f'{col}_regional_percentile'] = df.groupby('year')[col].rank(pct=True) * 100
    
    return df

def generate_summary_report(df):
    """Generate comprehensive summary statistics and insights."""
    
    report = {
        'dataset_overview': {
            'total_observations': len(df),
            'countries': df['country_name'].nunique(),
            'country_list': sorted(df['country_name'].unique().tolist()),
            'years_covered': sorted(df['year'].unique().tolist()),
            'indicators_count': len([col for col in df.columns if col not in ['country_code', 'country_name', 'year']])
        },
        'data_quality': {},
        'key_insights': {},
        'risk_indicators': {}
    }
    
    # Data completeness analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    completeness = {}
    for col in numeric_cols:
        total_possible = len(df)
        non_null = df[col].notna().sum()
        completeness[col] = {
            'non_null_count': int(non_null),
            'total_possible': int(total_possible),
            'completeness_rate': float(non_null / total_possible)
        }
    
    report['data_quality']['completeness'] = completeness
    
    # Key economic indicators summary
    key_indicators = ['gdp_growth', 'inflation', 'unemployment', 'internet_users', 'mobile_subs']
    for indicator in key_indicators:
        if indicator in df.columns:
            report['key_insights'][indicator] = {
                'mean': float(df[indicator].mean()),
                'std': float(df[indicator].std()),
                'min': float(df[indicator].min()),
                'max': float(df[indicator].max()),
                'latest_year_avg': float(df[df['year'] == df['year'].max()][indicator].mean())
            }
    
    # Risk indicator analysis
    if 'gdp_growth_risk' in df.columns:
        risk_dist = df['gdp_growth_risk'].value_counts(normalize=True) * 100
        report['risk_indicators']['gdp_growth_risk_distribution'] = risk_dist.to_dict()
    
    if 'inflation_risk' in df.columns:
        risk_dist = df['inflation_risk'].value_counts(normalize=True) * 100
        report['risk_indicators']['inflation_risk_distribution'] = risk_dist.to_dict()
    
    # Top performers and laggards
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    
    if 'gdp_growth' in latest_data.columns:
        top_gdp = latest_data.nlargest(5, 'gdp_growth')[['country_name', 'gdp_growth']].to_dict('records')
        bottom_gdp = latest_data.nsmallest(5, 'gdp_growth')[['country_name', 'gdp_growth']].to_dict('records')
        report['key_insights']['top_gdp_performers'] = top_gdp
        report['key_insights']['bottom_gdp_performers'] = bottom_gdp
    
    if 'internet_users' in latest_data.columns:
        top_digital = latest_data.nlargest(5, 'internet_users')[['country_name', 'internet_users']].to_dict('records')
        bottom_digital = latest_data.nsmallest(5, 'internet_users')[['country_name', 'internet_users']].to_dict('records')
        report['key_insights']['top_digital_adoption'] = top_digital
        report['key_insights']['bottom_digital_adoption'] = bottom_digital
    
    return report

def main():
    """Main processing function."""
    print("Starting data enhancement process...")
    
    # Load the basic dataset
    df = pd.read_csv('processed_data/ssa_macro_data_simple.csv')
    print(f"Loaded dataset with {len(df)} observations")
    
    # Calculate volatilities for key indicators
    print("Calculating volatilities...")
    volatility_indicators = ['gdp_growth', 'inflation', 'exchange_rate']
    
    for indicator in volatility_indicators:
        if indicator in df.columns:
            df = calculate_volatility(df, indicator)
            print(f"  Added {indicator}_volatility")
    
    # Calculate growth rates for level variables
    print("Calculating growth rates...")
    growth_indicators = ['mobile_subs', 'internet_users', 'financial_depth']
    
    for indicator in growth_indicators:
        if indicator in df.columns:
            df = calculate_growth_rates(df, indicator)
            print(f"  Added {indicator}_growth")
    
    # Add FinTech-specific risk indicators
    print("Adding FinTech risk indicators...")
    df = add_fintech_risk_indicators(df)
    
    # Create risk categories
    print("Creating risk categories...")
    df = create_risk_categories(df)
    
    # Add regional comparisons
    print("Adding regional comparisons...")
    df = add_regional_comparisons(df)
    
    # Generate summary report
    print("Generating summary report...")
    summary_report = generate_summary_report(df)
    
    # Save enhanced dataset
    print("Saving enhanced dataset...")
    df.to_csv('processed_data/ssa_macro_data_enhanced.csv', index=False)
    df.to_excel('processed_data/ssa_macro_data_enhanced.xlsx', index=False)
    
    # Save summary report
    with open('processed_data/enhanced_data_report.json', 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print(f"\nEnhancement completed!")
    print(f"Enhanced dataset shape: {df.shape}")
    print(f"New indicators added: {df.shape[1] - 16}")  # Original had 16 columns
    print(f"Files saved:")
    print(f"  - processed_data/ssa_macro_data_enhanced.csv")
    print(f"  - processed_data/ssa_macro_data_enhanced.xlsx")
    print(f"  - processed_data/enhanced_data_report.json")
    
    # Show some key statistics
    print(f"\nKey Statistics:")
    print(f"Countries with highest digital infrastructure readiness (latest year):")
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    if 'digital_infrastructure' in latest_data.columns:
        top_digital = latest_data.nlargest(5, 'digital_infrastructure')[['country_name', 'digital_infrastructure']]
        for _, row in top_digital.iterrows():
            print(f"  {row['country_name']}: {row['digital_infrastructure']:.1f}")
    
    print(f"\nCountries with highest economic instability (latest year):")
    if 'economic_instability' in latest_data.columns:
        top_unstable = latest_data.nlargest(5, 'economic_instability')[['country_name', 'economic_instability']]
        for _, row in top_unstable.iterrows():
            print(f"  {row['country_name']}: {row['economic_instability']:.2f}")

if __name__ == "__main__":
    main()