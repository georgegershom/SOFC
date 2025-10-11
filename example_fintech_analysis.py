#!/usr/bin/env python3
"""
Example: FinTech Early Warning Model Analysis
Demonstrates how to use the SSA Macroeconomic Dataset for FinTech risk assessment

This example shows:
1. Loading and preparing data
2. Creating risk indicators
3. Identifying high-risk periods
4. Basic early warning signals
"""

import pandas as pd
import numpy as np

def load_data():
    """Load the SSA macroeconomic dataset"""
    df = pd.read_csv('/workspace/data/ssa_macroeconomic_data.csv')
    return df

def create_risk_indicators(df):
    """
    Create comprehensive risk indicators for FinTech early warning
    
    Risk categories:
    1. Macroeconomic instability
    2. Financial stress
    3. Digital infrastructure gaps
    """
    
    df = df.copy()
    
    # 1. MACROECONOMIC INSTABILITY INDEX
    # Combines GDP volatility, inflation volatility, and negative growth
    df['Macro_Instability'] = (
        (df['GDP_Growth_Volatility'] / df['GDP_Growth_Volatility'].max()).fillna(0) * 0.3 +
        (df['Inflation_Volatility'] / df['Inflation_Volatility'].max()).fillna(0) * 0.3 +
        ((df['GDP_Growth'] < 0).astype(int)) * 0.4
    ) * 100
    
    # 2. FINANCIAL STRESS INDEX
    # High inflation, high interest rates, currency depreciation
    df['Financial_Stress'] = (
        (df['Inflation'] / df['Inflation'].max()) * 0.35 +
        (df['Exchange_Rate_Volatility'] / df['Exchange_Rate_Volatility'].max()).fillna(0) * 0.35 +
        (df['Debt_to_GDP'] / df['Debt_to_GDP'].max()) * 0.30
    ) * 100
    
    # 3. DIGITAL INFRASTRUCTURE GAP
    # Lower scores indicate infrastructure gaps
    df['Digital_Gap'] = (
        1 - (
            (df['Mobile_Subscriptions_per_100'] / df['Mobile_Subscriptions_per_100'].max()) * 0.4 +
            (df['Internet_Users_Percent'] / df['Internet_Users_Percent'].max()) * 0.4 +
            (df['Secure_Servers_per_Million'] / df['Secure_Servers_per_Million'].max()) * 0.2
        )
    ) * 100
    
    # 4. COMPOSITE FINTECH RISK SCORE
    # Higher scores = Higher risk
    df['FinTech_Risk_Score'] = (
        df['Macro_Instability'] * 0.35 +
        df['Financial_Stress'] * 0.40 +
        df['Digital_Gap'] * 0.25
    )
    
    # 5. RISK LEVEL CLASSIFICATION
    df['Risk_Level'] = pd.cut(
        df['FinTech_Risk_Score'],
        bins=[0, 30, 50, 70, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    return df

def identify_warning_signals(df):
    """
    Identify early warning signals for FinTech distress
    
    Warning triggers:
    - Sharp GDP decline (>3%)
    - High inflation (>15%)
    - Extreme exchange rate volatility
    - Deteriorating digital infrastructure
    """
    
    df = df.copy()
    
    # Create warning flags
    df['Warning_GDP_Decline'] = (df['GDP_Growth'] < -3.0).astype(int)
    df['Warning_High_Inflation'] = (df['Inflation'] > 15.0).astype(int)
    df['Warning_Currency_Volatility'] = (
        df['Exchange_Rate_Volatility'] > df['Exchange_Rate_Volatility'].quantile(0.90)
    ).astype(int)
    df['Warning_High_Risk'] = (df['FinTech_Risk_Score'] > 70).astype(int)
    
    # Count total warnings
    df['Total_Warnings'] = (
        df['Warning_GDP_Decline'] +
        df['Warning_High_Inflation'] +
        df['Warning_Currency_Volatility'] +
        df['Warning_High_Risk']
    )
    
    # Early warning signal (2+ warnings)
    df['Early_Warning_Signal'] = (df['Total_Warnings'] >= 2).astype(int)
    
    return df

def analyze_country_risk_trends(df, country_code='NGA'):
    """Analyze risk trends for a specific country"""
    
    country_data = df[df['Country_Code'] == country_code].sort_values('Year')
    
    print(f"\n{'=' * 80}")
    print(f"FINTECH RISK ANALYSIS: {country_data['Country_Name'].iloc[0]} ({country_code})")
    print(f"{'=' * 80}\n")
    
    # Latest year summary
    latest = country_data[country_data['Year'] == country_data['Year'].max()].iloc[0]
    
    print("LATEST YEAR SNAPSHOT (2024):")
    print(f"  FinTech Risk Score: {latest['FinTech_Risk_Score']:.2f}/100")
    print(f"  Risk Level: {latest['Risk_Level']}")
    print(f"  Macro Instability: {latest['Macro_Instability']:.2f}/100")
    print(f"  Financial Stress: {latest['Financial_Stress']:.2f}/100")
    print(f"  Digital Gap: {latest['Digital_Gap']:.2f}/100")
    print(f"  Early Warning Signal: {'YES' if latest['Early_Warning_Signal'] else 'NO'}")
    
    # Historical trends
    print(f"\n{'=' * 80}")
    print("RISK SCORE TRENDS (Last 5 Years):")
    print(f"{'=' * 80}\n")
    
    recent = country_data[country_data['Year'] >= 2020][
        ['Year', 'FinTech_Risk_Score', 'Risk_Level', 'Early_Warning_Signal']
    ]
    print(recent.to_string(index=False))
    
    # Warning periods
    warnings = country_data[country_data['Early_Warning_Signal'] == 1]
    if len(warnings) > 0:
        print(f"\n{'=' * 80}")
        print("EARLY WARNING PERIODS:")
        print(f"{'=' * 80}\n")
        print(f"Total Warning Years: {len(warnings)}")
        print(f"Warning Years: {', '.join(map(str, warnings['Year'].tolist()))}")
    else:
        print("\nNo early warning periods detected.")
    
    print()

def cross_country_risk_ranking(df, year=2024):
    """Rank countries by FinTech risk for a specific year"""
    
    print(f"\n{'=' * 80}")
    print(f"FINTECH RISK RANKING - YEAR {year}")
    print(f"{'=' * 80}\n")
    
    year_data = df[df['Year'] == year].sort_values('FinTech_Risk_Score', ascending=False)
    
    ranking = year_data[[
        'Country_Name', 'FinTech_Risk_Score', 'Risk_Level',
        'Macro_Instability', 'Financial_Stress', 'Digital_Gap'
    ]].round(2)
    
    ranking['Rank'] = range(1, len(ranking) + 1)
    ranking = ranking[['Rank', 'Country_Name', 'FinTech_Risk_Score', 'Risk_Level',
                       'Macro_Instability', 'Financial_Stress', 'Digital_Gap']]
    
    print(ranking.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("RISK DISTRIBUTION:")
    print(f"{'=' * 80}\n")
    
    risk_dist = year_data['Risk_Level'].value_counts().sort_index()
    for level, count in risk_dist.items():
        pct = (count / len(year_data)) * 100
        print(f"  {level}: {count} countries ({pct:.1f}%)")
    
    print()

def regional_analysis(df, year=2024):
    """Analyze risk by region"""
    
    # Define regions
    regions = {
        'West Africa': ['NGA', 'GHA', 'CIV', 'SEN', 'BEN', 'BFA', 'MLI'],
        'East Africa': ['KEN', 'ETH', 'TZA', 'UGA', 'RWA'],
        'Southern Africa': ['ZAF', 'ZMB', 'MOZ', 'BWA', 'NAM', 'ZWE'],
        'Central Africa': ['CMR', 'AGO']
    }
    
    print(f"\n{'=' * 80}")
    print(f"REGIONAL RISK ANALYSIS - YEAR {year}")
    print(f"{'=' * 80}\n")
    
    year_data = df[df['Year'] == year]
    
    regional_stats = []
    for region, countries in regions.items():
        region_data = year_data[year_data['Country_Code'].isin(countries)]
        
        stats = {
            'Region': region,
            'Countries': len(region_data),
            'Avg_Risk_Score': region_data['FinTech_Risk_Score'].mean(),
            'Avg_Macro_Instability': region_data['Macro_Instability'].mean(),
            'Avg_Financial_Stress': region_data['Financial_Stress'].mean(),
            'Avg_Digital_Gap': region_data['Digital_Gap'].mean(),
            'High_Risk_Countries': (region_data['FinTech_Risk_Score'] > 70).sum()
        }
        regional_stats.append(stats)
    
    regional_df = pd.DataFrame(regional_stats).round(2)
    regional_df = regional_df.sort_values('Avg_Risk_Score', ascending=False)
    
    print(regional_df.to_string(index=False))
    print()

def time_series_patterns(df):
    """Analyze time series patterns in risk"""
    
    print(f"\n{'=' * 80}")
    print("TIME SERIES PATTERNS: AVERAGE RISK BY YEAR")
    print(f"{'=' * 80}\n")
    
    yearly_avg = df.groupby('Year').agg({
        'FinTech_Risk_Score': 'mean',
        'Macro_Instability': 'mean',
        'Financial_Stress': 'mean',
        'Digital_Gap': 'mean',
        'Early_Warning_Signal': 'sum'
    }).round(2)
    
    yearly_avg.columns = ['Avg_Risk', 'Avg_Macro_Inst', 'Avg_Fin_Stress', 
                          'Avg_Digital_Gap', 'Countries_w_Warning']
    
    print(yearly_avg.to_string())
    
    # COVID-19 impact
    print(f"\n{'=' * 80}")
    print("COVID-19 IMPACT (2020 vs 2019):")
    print(f"{'=' * 80}\n")
    
    covid_impact = {
        'Metric': ['Risk Score', 'Macro Instability', 'Financial Stress', 'Countries with Warnings'],
        '2019': [
            yearly_avg.loc[2019, 'Avg_Risk'],
            yearly_avg.loc[2019, 'Avg_Macro_Inst'],
            yearly_avg.loc[2019, 'Avg_Fin_Stress'],
            yearly_avg.loc[2019, 'Countries_w_Warning']
        ],
        '2020': [
            yearly_avg.loc[2020, 'Avg_Risk'],
            yearly_avg.loc[2020, 'Avg_Macro_Inst'],
            yearly_avg.loc[2020, 'Avg_Fin_Stress'],
            yearly_avg.loc[2020, 'Countries_w_Warning']
        ]
    }
    
    covid_df = pd.DataFrame(covid_impact)
    covid_df['Change'] = covid_df['2020'] - covid_df['2019']
    covid_df['Change_Pct'] = ((covid_df['2020'] - covid_df['2019']) / covid_df['2019'] * 100).round(1)
    
    print(covid_df.to_string(index=False))
    print()

def save_risk_assessment(df):
    """Save the risk assessment dataset"""
    
    output_path = '/workspace/data/fintech_risk_assessment.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 80}")
    print("DATA SAVED")
    print(f"{'=' * 80}\n")
    print(f"✓ Risk assessment saved to: {output_path}")
    print(f"  Total records: {len(df):,}")
    print(f"  Variables: {len(df.columns)}")
    print()

def main():
    """Main analysis workflow"""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  FINTECH EARLY WARNING MODEL - EXAMPLE ANALYSIS".center(78) + "║")
    print("║" + "  Sub-Saharan Africa Economies".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"✓ Loaded {len(df):,} records for {df['Country_Code'].nunique()} countries")
    
    # Create risk indicators
    print("\nCalculating risk indicators...")
    df = create_risk_indicators(df)
    print("✓ Risk indicators calculated")
    
    # Identify warning signals
    print("\nIdentifying early warning signals...")
    df = identify_warning_signals(df)
    print("✓ Warning signals identified")
    
    # Run analyses
    cross_country_risk_ranking(df, year=2024)
    regional_analysis(df, year=2024)
    analyze_country_risk_trends(df, 'NGA')  # Nigeria
    analyze_country_risk_trends(df, 'KEN')  # Kenya
    analyze_country_risk_trends(df, 'ZAF')  # South Africa
    time_series_patterns(df)
    
    # Save results
    save_risk_assessment(df)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  1. Risk scores calculated for all countries (2010-2024)")
    print("  2. Early warning signals identified based on multiple indicators")
    print("  3. Regional patterns analyzed")
    print("  4. COVID-19 impact quantified")
    print("\nNext Steps:")
    print("  1. Integrate firm-level FinTech data")
    print("  2. Build predictive models (logistic regression, survival analysis)")
    print("  3. Validate against actual FinTech failures/distress events")
    print("  4. Refine risk thresholds based on empirical outcomes")
    print()

if __name__ == "__main__":
    main()
