#!/usr/bin/env python3
"""
Summary Report Generator for SSA FinTech Dataset
Creates comprehensive summary statistics and basic visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def create_summary_report():
    """Generate comprehensive summary report."""
    
    print("Loading comprehensive dataset...")
    df = pd.read_csv('processed_data/ssa_comprehensive_dataset.csv')
    
    # Basic dataset info
    report = {
        'dataset_info': {
            'total_observations': len(df),
            'countries': df['country_name'].nunique(),
            'years_covered': sorted(df['year'].unique().tolist()),
            'total_indicators': df.shape[1] - 3,  # Minus country_code, country_name, year
            'generation_date': datetime.now().isoformat()
        },
        'country_coverage': sorted(df['country_name'].unique().tolist()),
        'data_completeness': {},
        'key_statistics': {},
        'regional_insights': {}
    }
    
    # Data completeness analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    completeness = {}
    
    for col in numeric_cols:
        if col != 'year':
            non_null = df[col].notna().sum()
            total = len(df)
            completeness[col] = {
                'complete_observations': int(non_null),
                'total_observations': int(total),
                'completeness_rate': float(non_null / total),
                'missing_observations': int(total - non_null)
            }
    
    # Sort by completeness rate
    completeness = dict(sorted(completeness.items(), 
                              key=lambda x: x[1]['completeness_rate'], 
                              reverse=True))
    
    report['data_completeness'] = completeness
    
    # Key statistics for important indicators
    key_indicators = [
        'gdp_growth', 'inflation', 'unemployment', 'internet_users', 
        'mobile_subs', 'fintech_adoption_rate', 'digital_payment_volume_gdp',
        'cybersecurity_incidents_per_100k', 'financial_exclusion_rate'
    ]
    
    for indicator in key_indicators:
        if indicator in df.columns:
            series = df[indicator].dropna()
            if len(series) > 0:
                report['key_statistics'][indicator] = {
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75))
                }
    
    # Regional insights for latest year
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # Top and bottom performers
    if 'gdp_growth' in latest_data.columns:
        top_gdp = latest_data.nlargest(5, 'gdp_growth')[['country_name', 'gdp_growth']].to_dict('records')
        bottom_gdp = latest_data.nsmallest(5, 'gdp_growth')[['country_name', 'gdp_growth']].to_dict('records')
        report['regional_insights']['gdp_growth'] = {
            'top_performers': top_gdp,
            'bottom_performers': bottom_gdp
        }
    
    if 'fintech_adoption_rate' in latest_data.columns:
        top_fintech = latest_data.nlargest(5, 'fintech_adoption_rate')[['country_name', 'fintech_adoption_rate']].to_dict('records')
        bottom_fintech = latest_data.nsmallest(5, 'fintech_adoption_rate')[['country_name', 'fintech_adoption_rate']].to_dict('records')
        report['regional_insights']['fintech_adoption'] = {
            'top_adopters': top_fintech,
            'bottom_adopters': bottom_fintech
        }
    
    if 'digital_infrastructure' in latest_data.columns:
        top_digital = latest_data.nlargest(5, 'digital_infrastructure')[['country_name', 'digital_infrastructure']].to_dict('records')
        bottom_digital = latest_data.nsmallest(5, 'digital_infrastructure')[['country_name', 'digital_infrastructure']].to_dict('records')
        report['regional_insights']['digital_infrastructure'] = {
            'most_developed': top_digital,
            'least_developed': bottom_digital
        }
    
    # Risk analysis
    if 'economic_instability' in latest_data.columns:
        high_risk = latest_data.nlargest(5, 'economic_instability')[['country_name', 'economic_instability']].to_dict('records')
        low_risk = latest_data.nsmallest(5, 'economic_instability')[['country_name', 'economic_instability']].to_dict('records')
        report['regional_insights']['economic_risk'] = {
            'highest_risk': high_risk,
            'lowest_risk': low_risk
        }
    
    return report, df

def create_basic_visualizations(df):
    """Create basic visualizations for the dataset."""
    
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. GDP Growth trends by country
    plt.subplot(4, 3, 1)
    countries_to_plot = ['Kenya', 'Nigeria', 'South Africa', 'Ghana', 'Rwanda']
    for country in countries_to_plot:
        country_data = df[df['country_name'] == country]
        if not country_data.empty and 'gdp_growth' in country_data.columns:
            plt.plot(country_data['year'], country_data['gdp_growth'], 
                    marker='o', label=country, linewidth=2)
    plt.title('GDP Growth Trends (Selected Countries)', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('GDP Growth (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Internet penetration growth
    plt.subplot(4, 3, 2)
    for country in countries_to_plot:
        country_data = df[df['country_name'] == country]
        if not country_data.empty and 'internet_users' in country_data.columns:
            plt.plot(country_data['year'], country_data['internet_users'], 
                    marker='s', label=country, linewidth=2)
    plt.title('Internet Users Growth (Selected Countries)', fontsize=12, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Internet Users (% of population)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. FinTech adoption distribution (latest year)
    plt.subplot(4, 3, 3)
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    if 'fintech_adoption_rate' in latest_data.columns:
        plt.hist(latest_data['fintech_adoption_rate'].dropna(), bins=15, 
                color='skyblue', alpha=0.7, edgecolor='black')
        plt.title(f'FinTech Adoption Distribution ({latest_year})', fontsize=12, fontweight='bold')
        plt.xlabel('FinTech Adoption Rate (%)')
        plt.ylabel('Number of Countries')
        plt.grid(True, alpha=0.3)
    
    # 4. Digital divide analysis
    plt.subplot(4, 3, 4)
    if 'digital_divide' in latest_data.columns:
        plt.scatter(latest_data['mobile_subs'], latest_data['internet_users'], 
                   s=100, alpha=0.7, c='coral')
        plt.plot([0, 150], [0, 150], 'k--', alpha=0.5, label='Equal penetration line')
        plt.title(f'Mobile vs Internet Penetration ({latest_year})', fontsize=12, fontweight='bold')
        plt.xlabel('Mobile Subscriptions (per 100 people)')
        plt.ylabel('Internet Users (% of population)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Inflation volatility
    plt.subplot(4, 3, 5)
    if 'inflation_volatility' in df.columns:
        volatility_data = df.groupby('country_name')['inflation_volatility'].mean().sort_values(ascending=False)
        top_10 = volatility_data.head(10)
        plt.barh(range(len(top_10)), top_10.values, color='lightcoral')
        plt.yticks(range(len(top_10)), top_10.index)
        plt.title('Average Inflation Volatility by Country', fontsize=12, fontweight='bold')
        plt.xlabel('Inflation Volatility (Standard Deviation)')
        plt.grid(True, alpha=0.3)
    
    # 6. Financial depth vs FinTech adoption
    plt.subplot(4, 3, 6)
    if 'financial_depth' in latest_data.columns and 'fintech_adoption_rate' in latest_data.columns:
        plt.scatter(latest_data['financial_depth'], latest_data['fintech_adoption_rate'], 
                   s=100, alpha=0.7, c='mediumseagreen')
        plt.title(f'Financial Depth vs FinTech Adoption ({latest_year})', fontsize=12, fontweight='bold')
        plt.xlabel('Financial Depth (% of GDP)')
        plt.ylabel('FinTech Adoption Rate (%)')
        plt.grid(True, alpha=0.3)
    
    # 7. Cybersecurity incidents
    plt.subplot(4, 3, 7)
    if 'cybersecurity_incidents_per_100k' in latest_data.columns:
        incidents_data = latest_data.groupby('country_name')['cybersecurity_incidents_per_100k'].mean().sort_values(ascending=False)
        top_10 = incidents_data.head(10)
        plt.bar(range(len(top_10)), top_10.values, color='orange', alpha=0.7)
        plt.xticks(range(len(top_10)), top_10.index, rotation=45, ha='right')
        plt.title('Cybersecurity Incidents by Country', fontsize=12, fontweight='bold')
        plt.ylabel('Incidents per 100k transactions')
        plt.grid(True, alpha=0.3)
    
    # 8. Economic risk distribution
    plt.subplot(4, 3, 8)
    if 'economic_instability' in latest_data.columns:
        plt.hist(latest_data['economic_instability'].dropna(), bins=12, 
                color='lightsteelblue', alpha=0.7, edgecolor='black')
        plt.title(f'Economic Instability Distribution ({latest_year})', fontsize=12, fontweight='bold')
        plt.xlabel('Economic Instability Index')
        plt.ylabel('Number of Countries')
        plt.grid(True, alpha=0.3)
    
    # 9. Mobile money penetration
    plt.subplot(4, 3, 9)
    if 'mobile_money_penetration' in latest_data.columns:
        penetration_data = latest_data.groupby('country_name')['mobile_money_penetration'].mean().sort_values(ascending=False)
        top_10 = penetration_data.head(10)
        plt.barh(range(len(top_10)), top_10.values, color='mediumorchid', alpha=0.7)
        plt.yticks(range(len(top_10)), top_10.index)
        plt.title('Mobile Money Penetration by Country', fontsize=12, fontweight='bold')
        plt.xlabel('Mobile Money Penetration (% of adults)')
        plt.grid(True, alpha=0.3)
    
    # 10. Regional comparison - GDP Growth
    plt.subplot(4, 3, 10)
    if 'gdp_growth' in df.columns:
        yearly_avg = df.groupby('year')['gdp_growth'].mean()
        plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=3, 
                color='darkblue', markersize=8)
        plt.title('Regional Average GDP Growth Over Time', fontsize=12, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('GDP Growth (%)')
        plt.grid(True, alpha=0.3)
    
    # 11. Data completeness heatmap
    plt.subplot(4, 3, 11)
    key_indicators = ['gdp_growth', 'inflation', 'unemployment', 'internet_users', 
                     'mobile_subs', 'exchange_rate', 'financial_depth']
    completeness_matrix = []
    countries = df['country_name'].unique()[:10]  # Top 10 countries
    
    for country in countries:
        country_data = df[df['country_name'] == country]
        completeness_row = []
        for indicator in key_indicators:
            if indicator in country_data.columns:
                completeness = country_data[indicator].notna().mean()
                completeness_row.append(completeness)
            else:
                completeness_row.append(0)
        completeness_matrix.append(completeness_row)
    
    plt.imshow(completeness_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Completeness Rate')
    plt.yticks(range(len(countries)), countries)
    plt.xticks(range(len(key_indicators)), key_indicators, rotation=45, ha='right')
    plt.title('Data Completeness by Country and Indicator', fontsize=12, fontweight='bold')
    
    # 12. FinTech investment trends
    plt.subplot(4, 3, 12)
    if 'fintech_investment_usd_millions' in df.columns:
        investment_trends = df.groupby('year')['fintech_investment_usd_millions'].sum()
        plt.bar(investment_trends.index, investment_trends.values, 
               color='gold', alpha=0.7, edgecolor='black')
        plt.title('Total FinTech Investment Over Time', fontsize=12, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Investment (USD Millions)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('processed_data/dataset_overview_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to: processed_data/dataset_overview_charts.png")

def main():
    """Main execution function."""
    print("Generating comprehensive summary report...")
    
    # Generate summary report
    report, df = create_summary_report()
    
    # Create visualizations
    create_basic_visualizations(df)
    
    # Save summary report
    with open('processed_data/comprehensive_summary_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nSummary Report Generated Successfully!")
    print(f"="*60)
    print(f"Dataset Overview:")
    print(f"  - Total observations: {report['dataset_info']['total_observations']}")
    print(f"  - Countries covered: {report['dataset_info']['countries']}")
    print(f"  - Years covered: {report['dataset_info']['years_covered'][0]}-{report['dataset_info']['years_covered'][-1]}")
    print(f"  - Total indicators: {report['dataset_info']['total_indicators']}")
    
    print(f"\nData Quality Summary:")
    completeness_items = list(report['data_completeness'].items())
    print(f"  - Highest completeness: {completeness_items[0][0]} ({completeness_items[0][1]['completeness_rate']:.1%})")
    print(f"  - Lowest completeness: {completeness_items[-1][0]} ({completeness_items[-1][1]['completeness_rate']:.1%})")
    
    print(f"\nKey Insights (Latest Year - {df['year'].max()}):")
    if 'gdp_growth' in report['regional_insights']:
        top_gdp = report['regional_insights']['gdp_growth']['top_performers'][0]
        print(f"  - Highest GDP growth: {top_gdp['country_name']} ({top_gdp['gdp_growth']:.1f}%)")
    
    if 'fintech_adoption' in report['regional_insights']:
        top_fintech = report['regional_insights']['fintech_adoption']['top_adopters'][0]
        print(f"  - Highest FinTech adoption: {top_fintech['country_name']} ({top_fintech['fintech_adoption_rate']:.1f}%)")
    
    print(f"\nFiles Generated:")
    print(f"  - processed_data/comprehensive_summary_report.json")
    print(f"  - processed_data/dataset_overview_charts.png")
    print(f"  - documentation/dataset_documentation.md")

if __name__ == "__main__":
    main()