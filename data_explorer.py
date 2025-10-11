#!/usr/bin/env python3
"""
Simple Data Explorer for FinTech Risk Nexus Dataset
Provides basic data exploration without visualization dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime

def explore_dataset():
    """Explore the FinTech Risk Nexus Dataset"""
    print("FinTech Risk Nexus Dataset Explorer")
    print("="*50)
    
    # Load datasets
    cyber_risk = pd.read_csv('/workspace/fintech_risk_nexus_cyber_risk.csv')
    consumer_sentiment = pd.read_csv('/workspace/fintech_risk_nexus_consumer_sentiment.csv')
    competitive_dynamics = pd.read_csv('/workspace/fintech_risk_nexus_competitive_dynamics.csv')
    macro_economic = pd.read_csv('/workspace/fintech_risk_nexus_macro_economic.csv')
    summary_stats = pd.read_csv('/workspace/fintech_risk_nexus_summary_statistics.csv')
    
    print(f"\nDataset Overview:")
    print(f"- Cyber Risk Records: {len(cyber_risk):,}")
    print(f"- Consumer Sentiment Records: {len(consumer_sentiment):,}")
    print(f"- Competitive Dynamics Records: {len(competitive_dynamics):,}")
    print(f"- Macro Economic Records: {len(macro_economic):,}")
    print(f"- Summary Statistics Records: {len(summary_stats):,}")
    print(f"- Total Records: {len(cyber_risk) + len(consumer_sentiment) + len(competitive_dynamics) + len(macro_economic) + len(summary_stats):,}")
    
    print(f"\nCountries Covered: {len(summary_stats)}")
    print("Countries:", ", ".join(summary_stats['country'].tolist()))
    
    print(f"\nTime Period: {cyber_risk['year'].min()}-{cyber_risk['year'].max()}")
    
    # Cyber Risk Analysis
    print(f"\n" + "="*50)
    print("CYBER RISK INSIGHTS")
    print("="*50)
    
    print(f"Average cyber incidents per quarter: {cyber_risk['cyber_incidents'].mean():.1f}")
    print(f"Maximum cyber incidents in a quarter: {cyber_risk['cyber_incidents'].max()}")
    print(f"Average fraud cases per quarter: {cyber_risk['mobile_money_fraud_cases'].mean():.1f}")
    print(f"Average data breach severity: {cyber_risk['data_breach_severity'].mean():.2f}")
    
    # Top 5 countries by cyber incidents
    top_cyber = cyber_risk.groupby('country')['cyber_incidents'].mean().sort_values(ascending=False).head(5)
    print(f"\nTop 5 Countries by Cyber Incidents:")
    for country, incidents in top_cyber.items():
        print(f"  {country}: {incidents:.1f} incidents/quarter")
    
    # Consumer Sentiment Analysis
    print(f"\n" + "="*50)
    print("CONSUMER SENTIMENT INSIGHTS")
    print("="*50)
    
    print(f"Average sentiment score: {consumer_sentiment['sentiment_score'].mean():.3f}")
    print(f"Average trust index: {consumer_sentiment['trust_index'].mean():.1f}")
    print(f"Average customer satisfaction: {consumer_sentiment['customer_satisfaction'].mean():.2f}")
    print(f"Average app store rating: {consumer_sentiment['app_store_rating'].mean():.2f}")
    
    # Top 5 brands by sentiment
    top_brands = consumer_sentiment.groupby('fintech_brand')['sentiment_score'].mean().sort_values(ascending=False).head(5)
    print(f"\nTop 5 FinTech Brands by Sentiment:")
    for brand, sentiment in top_brands.items():
        print(f"  {brand}: {sentiment:.3f}")
    
    # Competitive Dynamics Analysis
    print(f"\n" + "="*50)
    print("COMPETITIVE DYNAMICS INSIGHTS")
    print("="*50)
    
    print(f"Average HHI Index: {competitive_dynamics['hhi_index'].mean():.1f}")
    print(f"Total new licenses issued (5 years): {competitive_dynamics['new_fintech_licenses'].sum()}")
    print(f"Average FinTech investment: ${competitive_dynamics['fintech_investment_millions_usd'].mean():.1f}M")
    print(f"Average digital adoption rate: {competitive_dynamics['digital_adoption_rate'].mean():.1%}")
    
    # Market concentration distribution
    concentration_dist = competitive_dynamics['market_concentration'].value_counts()
    print(f"\nMarket Concentration Distribution:")
    for level, count in concentration_dist.items():
        print(f"  {level}: {count} countries")
    
    # Macro Economic Analysis
    print(f"\n" + "="*50)
    print("MACRO ECONOMIC INSIGHTS")
    print("="*50)
    
    print(f"Average GDP per capita: ${macro_economic['gdp_per_capita_usd'].mean():,.0f}")
    print(f"Average inflation rate: {macro_economic['inflation_rate_pct'].mean():.1f}%")
    print(f"Average unemployment rate: {macro_economic['unemployment_rate_pct'].mean():.1f}%")
    print(f"Average mobile penetration: {macro_economic['mobile_penetration_pct'].mean():.1f}%")
    print(f"Average internet penetration: {macro_economic['internet_penetration_pct'].mean():.1f}%")
    print(f"Average financial inclusion: {macro_economic['financial_inclusion_pct'].mean():.1f}%")
    
    # Risk Analysis
    print(f"\n" + "="*50)
    print("RISK ANALYSIS")
    print("="*50)
    
    risk_dist = summary_stats['risk_level'].value_counts()
    print(f"Risk Level Distribution:")
    for level, count in risk_dist.items():
        print(f"  {level}: {count} countries")
    
    # Countries with highest risk scores
    high_risk = summary_stats.nlargest(5, 'avg_cyber_incidents_per_quarter')[['country', 'avg_cyber_incidents_per_quarter', 'risk_level']]
    print(f"\nTop 5 Highest Risk Countries:")
    for _, row in high_risk.iterrows():
        print(f"  {row['country']}: {row['avg_cyber_incidents_per_quarter']:.1f} incidents/quarter ({row['risk_level']})")
    
    # Data Quality Check
    print(f"\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    datasets = {
        'Cyber Risk': cyber_risk,
        'Consumer Sentiment': consumer_sentiment,
        'Competitive Dynamics': competitive_dynamics,
        'Macro Economic': macro_economic,
        'Summary Statistics': summary_stats
    }
    
    for name, df in datasets.items():
        missing_data = df.isnull().sum().sum()
        print(f"{name}: {missing_data} missing values")
    
    print(f"\nDataset exploration complete!")
    print(f"All datasets are ready for analysis and modeling.")

if __name__ == "__main__":
    explore_dataset()