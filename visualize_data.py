"""
Data Visualization Script for Category 3: Financial System & Regulatory Data

Creates summary visualizations and statistical tables for the dataset.

Author: Generated for FinTech Research
Date: 2025-10-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the main dataset"""
    df = pd.read_csv('/workspace/category3_financial_regulatory_data.csv')
    return df

def create_summary_statistics():
    """Generate summary statistics tables"""
    df = load_data()
    
    print("="*80)
    print("CATEGORY 3: FINANCIAL SYSTEM & REGULATORY DATA")
    print("Summary Statistics Report")
    print("="*80)
    print()
    
    # Overall summary
    print("DATASET OVERVIEW")
    print("-" * 80)
    print(f"Total Records: {len(df)}")
    print(f"Countries: {df['country_code'].nunique()}")
    print(f"Time Period: {df['year'].min()} - {df['year'].max()}")
    print(f"Years Covered: {df['year'].nunique()}")
    print()
    
    # Banking indicators summary
    print("BANKING SECTOR HEALTH INDICATORS (2010-2023)")
    print("-" * 80)
    banking_vars = ['bank_npl', 'bank_zscore', 'bank_roa', 'domestic_credit']
    print(df[banking_vars].describe().round(2))
    print()
    
    # Regulatory quality by country (latest year)
    print("REGULATORY QUALITY BY COUNTRY (2023)")
    print("-" * 80)
    latest = df[df['year'] == df['year'].max()].sort_values('regulatory_quality', ascending=False)
    print(latest[['country_name', 'regulatory_quality', 'total_fintech_regulations']].to_string(index=False))
    print()
    
    # Regulatory adoption timeline
    print("FINTECH REGULATORY ADOPTION TIMELINE")
    print("-" * 80)
    reg_vars = [col for col in df.columns if col.startswith('reg_') and col != 'reg_mobile_money_regulation'][:4]
    
    for year in [2015, 2018, 2020, 2023]:
        year_data = df[df['year'] == year]
        print(f"\nYear {year}:")
        for var in reg_vars:
            if var in year_data.columns:
                adoption_rate = year_data[var].mean() * 100
                print(f"  {var.replace('reg_', '').replace('_', ' ').title()}: {adoption_rate:.0f}%")
    
    print()
    
    # Top and bottom performers
    print("BANKING SECTOR PERFORMANCE (2023)")
    print("-" * 80)
    latest = df[df['year'] == df['year'].max()]
    
    print("\nLowest NPL Ratios (Healthiest):")
    top_npl = latest.nsmallest(5, 'bank_npl')[['country_name', 'bank_npl']]
    print(top_npl.to_string(index=False))
    
    print("\nHighest Bank Z-scores (Most Stable):")
    top_zscore = latest.nlargest(5, 'bank_zscore')[['country_name', 'bank_zscore']]
    print(top_zscore.to_string(index=False))
    
    print("\nHighest Credit Penetration:")
    top_credit = latest.nlargest(5, 'domestic_credit')[['country_name', 'domestic_credit']]
    print(top_credit.to_string(index=False))
    
    print()
    print("="*80)
    print()

def create_visualizations():
    """Create visualization plots"""
    df = load_data()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. NPL Trends Over Time (Top 5 Countries)
    ax1 = plt.subplot(2, 3, 1)
    top_countries = ['KEN', 'NGA', 'ZAF', 'GHA', 'RWA']
    for country in top_countries:
        country_data = df[df['country_code'] == country].sort_values('year')
        country_name = country_data['country_name'].iloc[0]
        ax1.plot(country_data['year'], country_data['bank_npl'], marker='o', label=country_name)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('NPL Ratio (%)')
    ax1.set_title('Bank Non-Performing Loans Trends')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Bank Z-score Trends
    ax2 = plt.subplot(2, 3, 2)
    for country in top_countries:
        country_data = df[df['country_code'] == country].sort_values('year')
        country_name = country_data['country_name'].iloc[0]
        ax2.plot(country_data['year'], country_data['bank_zscore'], marker='o', label=country_name)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Z-score')
    ax2.set_title('Bank Stability (Z-score) Trends')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Regulatory Quality Distribution (2023)
    ax3 = plt.subplot(2, 3, 3)
    latest = df[df['year'] == df['year'].max()].sort_values('regulatory_quality')
    ax3.barh(range(len(latest)), latest['regulatory_quality'])
    ax3.set_yticks(range(len(latest)))
    ax3.set_yticklabels(latest['country_code'], fontsize=7)
    ax3.set_xlabel('Regulatory Quality Index')
    ax3.set_title('Regulatory Quality by Country (2023)')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Domestic Credit Growth
    ax4 = plt.subplot(2, 3, 4)
    for country in top_countries:
        country_data = df[df['country_code'] == country].sort_values('year')
        country_name = country_data['country_name'].iloc[0]
        ax4.plot(country_data['year'], country_data['domestic_credit'], marker='o', label=country_name)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Credit (% of GDP)')
    ax4.set_title('Domestic Credit to Private Sector')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Banking Sector ROA
    ax5 = plt.subplot(2, 3, 5)
    for country in top_countries:
        country_data = df[df['country_code'] == country].sort_values('year')
        country_name = country_data['country_name'].iloc[0]
        ax5.plot(country_data['year'], country_data['bank_roa'], marker='o', label=country_name)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('ROA (%)')
    ax5.set_title('Banking Sector Return on Assets')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. FinTech Regulations Adoption Over Time
    ax6 = plt.subplot(2, 3, 6)
    years = sorted(df['year'].unique())
    reg_adoption = []
    for year in years:
        year_data = df[df['year'] == year]
        avg_regs = year_data['total_fintech_regulations'].mean()
        reg_adoption.append(avg_regs)
    
    ax6.plot(years, reg_adoption, marker='o', linewidth=2, markersize=8)
    ax6.fill_between(years, reg_adoption, alpha=0.3)
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Average # of Regulations')
    ax6.set_title('FinTech Regulatory Adoption (Regional Average)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/category3_visualizations.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to: category3_visualizations.png")
    plt.close()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    correlation_vars = ['bank_npl', 'bank_zscore', 'bank_roa', 'domestic_credit', 
                       'regulatory_quality', 'total_fintech_regulations']
    
    corr_matrix = df[correlation_vars].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Matrix: Banking & Regulatory Indicators', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('/workspace/category3_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation matrix saved to: category3_correlation_matrix.png")
    plt.close()

def create_country_profiles():
    """Create detailed profiles for key countries"""
    df = load_data()
    
    key_countries = ['KEN', 'NGA', 'ZAF', 'GHA', 'RWA']
    
    print("\nDETAILED COUNTRY PROFILES (2023)")
    print("="*80)
    
    latest = df[df['year'] == df['year'].max()]
    
    for country_code in key_countries:
        country_data = latest[latest['country_code'] == country_code].iloc[0]
        
        print(f"\n{country_data['country_name']} ({country_code})")
        print("-" * 80)
        print(f"Banking Sector Health:")
        print(f"  • Non-Performing Loans: {country_data['bank_npl']:.2f}%")
        print(f"  • Bank Z-score (Stability): {country_data['bank_zscore']:.2f}")
        print(f"  • Return on Assets: {country_data['bank_roa']:.2f}%")
        print(f"  • Domestic Credit to Private Sector: {country_data['domestic_credit']:.2f}% of GDP")
        print(f"\nRegulatory Environment:")
        print(f"  • Regulatory Quality Index: {country_data['regulatory_quality']:.3f}")
        print(f"  • Total FinTech Regulations: {int(country_data['total_fintech_regulations'])}")
        
        reg_cols = [col for col in df.columns if col.startswith('reg_') and '_' in col][:4]
        print(f"  • Regulations in Effect:")
        for reg in reg_cols:
            if country_data[reg] == 1:
                reg_name = reg.replace('reg_', '').replace('_', ' ').title()
                print(f"    ✓ {reg_name}")
    
    print()
    print("="*80)

def export_summary_table():
    """Export a summary comparison table"""
    df = load_data()
    
    latest = df[df['year'] == df['year'].max()].copy()
    
    summary_cols = ['country_code', 'country_name', 'bank_npl', 'bank_zscore', 
                   'bank_roa', 'domestic_credit', 'regulatory_quality', 
                   'total_fintech_regulations']
    
    summary = latest[summary_cols].sort_values('regulatory_quality', ascending=False)
    
    # Round numerical columns
    summary['bank_npl'] = summary['bank_npl'].round(2)
    summary['bank_zscore'] = summary['bank_zscore'].round(2)
    summary['bank_roa'] = summary['bank_roa'].round(2)
    summary['domestic_credit'] = summary['domestic_credit'].round(1)
    summary['regulatory_quality'] = summary['regulatory_quality'].round(3)
    
    summary.to_csv('/workspace/summary_table_2023.csv', index=False)
    print("✓ Summary table saved to: summary_table_2023.csv")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CATEGORY 3 DATA ANALYSIS & VISUALIZATION")
    print("="*80 + "\n")
    
    try:
        # Generate summary statistics
        create_summary_statistics()
        
        # Create visualizations
        print("Generating visualizations...")
        create_visualizations()
        print()
        
        # Create country profiles
        create_country_profiles()
        
        # Export summary table
        export_summary_table()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  1. category3_visualizations.png - Main trend charts")
        print("  2. category3_correlation_matrix.png - Variable correlations")
        print("  3. summary_table_2023.csv - 2023 cross-country comparison")
        print()
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
