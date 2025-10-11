#!/usr/bin/env python3
"""
Dataset Summary Generator
Creates a comprehensive overview of the financial system dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_dataset_summary():
    """Generate comprehensive dataset summary"""
    
    print("=" * 60)
    print("FINANCIAL SYSTEM & REGULATORY DATA SUMMARY")
    print("Sub-Saharan Africa FinTech Early Warning Model")
    print("=" * 60)
    
    # Load main dataset
    df = pd.read_csv('output/financial_system_regulatory_master_complete.csv')
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Total Observations: {len(df):,}")
    print(f"   Countries Covered: {df['country_name'].nunique()}")
    print(f"   Time Period: {df['year'].min()}-{df['year'].max()}")
    print(f"   Variables: {len(df.columns)}")
    
    print(f"\n🌍 COUNTRIES INCLUDED")
    countries = df['country_name'].unique()
    for i, country in enumerate(sorted(countries), 1):
        country_data = df[df['country_name'] == country]
        print(f"   {i:2d}. {country:<15} ({country_data['country_code'].iloc[0]}) - {len(country_data)} observations")
    
    print(f"\n📈 VARIABLE SUMMARY")
    variables = {
        'bank_npl_ratio': 'Bank Non-Performing Loans (%)',
        'bank_zscore': 'Bank Z-score (stability)',
        'bank_roa': 'Bank Return on Assets (%)',
        'domestic_credit_private': 'Domestic Credit to Private Sector (% GDP)',
        'regulatory_quality': 'Regulatory Quality Index',
        'digital_lending_regulation_dummy': 'Digital Lending Regulation',
        'open_banking_initiative_dummy': 'Open Banking Initiative',
        'fintech_regulatory_sandbox_dummy': 'FinTech Regulatory Sandbox'
    }
    
    for var, description in variables.items():
        if var in df.columns:
            series = df[var]
            missing_pct = (series.isna().sum() / len(series)) * 100
            
            if series.dtype in ['int64', 'float64']:
                print(f"   • {description:<40} | Mean: {series.mean():.2f} | Missing: {missing_pct:.1f}%")
            else:
                print(f"   • {description:<40} | Type: {series.dtype} | Missing: {missing_pct:.1f}%")
    
    print(f"\n🏦 BANKING SECTOR HEALTH (Latest Year)")
    latest_data = df.loc[df.groupby('country_name')['year'].idxmax()]
    
    print(f"   {'Country':<15} {'NPL %':<8} {'ROA %':<8} {'Z-Score':<10} {'Credit/GDP %':<12}")
    print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
    
    for _, row in latest_data.sort_values('bank_npl_ratio').iterrows():
        country = row['country_name'][:14]
        npl = f"{row['bank_npl_ratio']:.1f}" if not pd.isna(row['bank_npl_ratio']) else "N/A"
        roa = f"{row['bank_roa']:.1f}" if not pd.isna(row['bank_roa']) else "N/A"
        zscore = f"{row['bank_zscore']:.1f}" if not pd.isna(row['bank_zscore']) else "N/A"
        credit = f"{row['domestic_credit_private']:.1f}" if not pd.isna(row['domestic_credit_private']) else "N/A"
        
        print(f"   {country:<15} {npl:<8} {roa:<8} {zscore:<10} {credit:<12}")
    
    print(f"\n🏛️ REGULATORY LANDSCAPE")
    reg_summary = latest_data[['country_name', 'regulatory_quality', 
                              'digital_lending_regulation_dummy',
                              'open_banking_initiative_dummy', 
                              'fintech_regulatory_sandbox_dummy']].copy()
    
    reg_summary['total_regulations'] = (reg_summary['digital_lending_regulation_dummy'] + 
                                       reg_summary['open_banking_initiative_dummy'] + 
                                       reg_summary['fintech_regulatory_sandbox_dummy'])
    
    print(f"   {'Country':<15} {'Reg Quality':<12} {'Digital':<8} {'Open Bank':<10} {'Sandbox':<8} {'Total':<6}")
    print(f"   {'-'*15} {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    
    for _, row in reg_summary.sort_values('total_regulations', ascending=False).iterrows():
        country = row['country_name'][:14]
        reg_qual = f"{row['regulatory_quality']:.2f}" if not pd.isna(row['regulatory_quality']) else "N/A"
        digital = "✓" if row['digital_lending_regulation_dummy'] == 1 else "✗"
        open_bank = "✓" if row['open_banking_initiative_dummy'] == 1 else "✗"
        sandbox = "✓" if row['fintech_regulatory_sandbox_dummy'] == 1 else "✗"
        total = int(row['total_regulations'])
        
        print(f"   {country:<15} {reg_qual:<12} {digital:<8} {open_bank:<10} {sandbox:<8} {total:<6}")
    
    print(f"\n📁 FILES GENERATED")
    output_files = [
        ('financial_system_regulatory_master_complete.csv', 'Complete dataset with all variables'),
        ('bank_npl_ratio_raw.csv', 'Non-performing loans data'),
        ('bank_roa_raw.csv', 'Return on assets data'),
        ('bank_zscore_complete.csv', 'Bank stability scores'),
        ('domestic_credit_private_raw.csv', 'Credit to private sector data'),
        ('regulatory_quality_raw.csv', 'Governance indicators'),
        ('regulatory_dummies_raw.csv', 'Policy implementation indicators'),
        ('country_profiles.csv', 'Country-level statistics'),
        ('risk_assessment_report.csv', 'Latest risk assessment'),
        ('summary_statistics.csv', 'Descriptive statistics'),
        ('time_series_analysis.png', 'Time series visualizations'),
        ('correlation_heatmap.png', 'Correlation analysis'),
        ('country_radar_comparison.png', 'Country comparison chart'),
        ('regulatory_timeline.png', 'Regulatory implementation timeline')
    ]
    
    for filename, description in output_files:
        filepath = f'output/{filename}'
        if os.path.exists(filepath):
            if filename.endswith('.csv'):
                size = len(pd.read_csv(filepath))
                print(f"   ✓ {filename:<45} | {description} ({size} rows)")
            else:
                print(f"   ✓ {filename:<45} | {description}")
        else:
            print(f"   ✗ {filename:<45} | {description} (MISSING)")
    
    print(f"\n📊 DATA QUALITY ASSESSMENT")
    
    # Calculate completeness
    completeness = {}
    for col in ['bank_npl_ratio', 'bank_roa', 'bank_zscore', 'domestic_credit_private', 'regulatory_quality']:
        if col in df.columns:
            completeness[col] = (1 - df[col].isna().sum() / len(df)) * 100
    
    print(f"   Data Completeness:")
    for var, pct in completeness.items():
        status = "✓" if pct > 90 else "⚠" if pct > 70 else "✗"
        print(f"   {status} {var:<30}: {pct:.1f}%")
    
    # Time coverage
    print(f"\n   Time Coverage by Country:")
    for country in sorted(df['country_name'].unique()):
        country_data = df[df['country_name'] == country]
        years = sorted(country_data['year'].unique())
        coverage = len(years) / 14 * 100  # 14 years total (2010-2023)
        status = "✓" if coverage == 100 else "⚠"
        print(f"   {status} {country:<15}: {len(years):2d}/14 years ({coverage:.0f}%)")
    
    print(f"\n🎯 RESEARCH APPLICATIONS")
    print(f"   • FinTech Early Warning Models")
    print(f"   • Banking Sector Stability Analysis") 
    print(f"   • Regulatory Impact Assessment")
    print(f"   • Cross-Country Financial Development Studies")
    print(f"   • Policy Timeline Analysis")
    print(f"   • Risk Assessment and Stress Testing")
    
    print(f"\n📚 NEXT STEPS FOR RESEARCH")
    print(f"   1. Load 'financial_system_regulatory_master_complete.csv' as main dataset")
    print(f"   2. Use regulatory dummies for structural break analysis")
    print(f"   3. Combine with FinTech adoption data for early warning models")
    print(f"   4. Apply panel data techniques for policy impact assessment")
    print(f"   5. Use Z-scores and NPL ratios as dependent variables for stability models")
    
    print(f"\n" + "=" * 60)
    print(f"Dataset compilation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ready for FinTech Early Warning Model development!")
    print(f"=" * 60)

if __name__ == "__main__":
    generate_dataset_summary()