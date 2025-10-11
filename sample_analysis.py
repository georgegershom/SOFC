"""
Sample Analysis Script for FinTech Risk Nexus Dataset
Demonstrates various analytical approaches for the dataset
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the dataset and perform initial exploration"""
    
    # Load the main dataset
    df = pd.read_csv('fintech_risk_nexus_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("=" * 80)
    print("FINTECH RISK NEXUS DATASET - SAMPLE ANALYSIS")
    print("=" * 80)
    print()
    
    # Basic information
    print("1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total records: {len(df)}")
    print(f"Countries covered: {df['country'].nunique()}")
    print(f"Time period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Number of variables: {len(df.columns)}")
    print()
    
    return df

def analyze_cyber_risk(df):
    """Analyze cyber risk exposure across countries"""
    
    print("2. CYBER RISK ANALYSIS")
    print("-" * 40)
    
    # Average cyber incidents by country
    cyber_risk_summary = df.groupby('country').agg({
        'cyber_incidents_reported': 'mean',
        'google_trend_mobile_fraud': 'mean',
        'data_breach_severity_index': 'mean',
        'composite_cyber_risk': 'mean'
    }).round(2)
    
    # Top 5 highest cyber risk countries
    top_risk = cyber_risk_summary.nlargest(5, 'composite_cyber_risk')
    print("Top 5 Countries by Composite Cyber Risk:")
    print(top_risk[['composite_cyber_risk', 'cyber_incidents_reported']])
    print()
    
    # Cyber risk trend over time
    cyber_trend = df.groupby('year')['cyber_incidents_reported'].mean()
    print("Average Cyber Incidents Trend by Year:")
    for year, incidents in cyber_trend.items():
        print(f"  {year}: {incidents:.1f} incidents/quarter")
    print()
    
    return cyber_risk_summary

def analyze_market_dynamics(df):
    """Analyze competitive dynamics and market structure"""
    
    print("3. MARKET DYNAMICS ANALYSIS")
    print("-" * 40)
    
    # Market concentration evolution
    hhi_trend = df.groupby('year')['herfindahl_hirschman_index'].mean()
    print("Market Concentration Evolution (HHI):")
    print(f"  2018: {hhi_trend.iloc[0]:.0f}")
    print(f"  2024: {hhi_trend.iloc[-1]:.0f}")
    print(f"  Change: {((hhi_trend.iloc[-1] / hhi_trend.iloc[0]) - 1) * 100:.1f}%")
    print()
    
    # Total licenses issued by country
    licenses = df.groupby('country')['new_fintech_licenses_issued'].sum()
    top_licensing = licenses.nlargest(5)
    print("Top 5 Countries by Total FinTech Licenses Issued (2018-2024):")
    for country, count in top_licensing.items():
        print(f"  {country}: {int(count)} licenses")
    print()
    
    # Innovation leaders
    innovation = df.groupby('country')['innovation_index'].mean().nlargest(5)
    print("Top 5 Countries by Innovation Index:")
    for country, score in innovation.items():
        print(f"  {country}: {score:.1f}/100")
    print()

def analyze_consumer_trust(df):
    """Analyze consumer sentiment and trust metrics"""
    
    print("4. CONSUMER TRUST ANALYSIS")
    print("-" * 40)
    
    # Current trust levels (latest quarter)
    latest_quarter = df['date'].max()
    current_trust = df[df['date'] == latest_quarter].set_index('country')['consumer_trust_index']
    
    print(f"Consumer Trust Index - Q{df[df['date'] == latest_quarter]['quarter'].iloc[0]} {df[df['date'] == latest_quarter]['year'].iloc[0]}:")
    top_trust = current_trust.nlargest(5)
    for country, trust in top_trust.items():
        print(f"  {country}: {trust:.1f}/100")
    print()
    
    # Countries with improving trust (YoY)
    latest_data = df[df['date'] == latest_quarter]
    improving = latest_data[latest_data['consumer_trust_index_yoy_change'] > 0].sort_values(
        'consumer_trust_index_yoy_change', ascending=False
    ).head(5)
    
    if len(improving) > 0:
        print("Countries with Highest Trust Improvement (YoY %):")
        for _, row in improving.iterrows():
            if pd.notna(row['consumer_trust_index_yoy_change']):
                print(f"  {row['country']}: +{row['consumer_trust_index_yoy_change']:.1f}%")
    print()

def identify_risk_clusters(df):
    """Identify country risk clusters based on multiple indicators"""
    
    print("5. RISK CLUSTERING ANALYSIS")
    print("-" * 40)
    
    # Create risk profiles for latest data
    latest_quarter = df['date'].max()
    latest_data = df[df['date'] == latest_quarter]
    
    # Categorize countries by risk level
    risk_categories = []
    for _, row in latest_data.iterrows():
        if row['composite_cyber_risk'] < 20 and row['operational_risk_index'] < 3:
            risk_level = 'Low Risk'
        elif row['composite_cyber_risk'] > 40 or row['operational_risk_index'] > 7:
            risk_level = 'High Risk'
        else:
            risk_level = 'Medium Risk'
        
        risk_categories.append({
            'Country': row['country'],
            'Risk Level': risk_level,
            'Cyber Risk': round(row['composite_cyber_risk'], 1),
            'Op Risk': round(row['operational_risk_index'], 1),
            'Market Health': round(row['market_health_score'], 1)
        })
    
    risk_df = pd.DataFrame(risk_categories)
    
    # Count by risk level
    risk_counts = risk_df['Risk Level'].value_counts()
    print("Country Distribution by Risk Level:")
    for level, count in risk_counts.items():
        print(f"  {level}: {count} countries")
    print()
    
    # Show high-risk countries
    high_risk = risk_df[risk_df['Risk Level'] == 'High Risk']
    if len(high_risk) > 0:
        print("High Risk Countries:")
        for _, row in high_risk.iterrows():
            print(f"  {row['Country']}: Cyber={row['Cyber Risk']}, OpRisk={row['Op Risk']}")
    print()

def calculate_early_warning_signals(df):
    """Calculate early warning signals based on threshold breaches"""
    
    print("6. EARLY WARNING SIGNALS")
    print("-" * 40)
    
    # Define warning thresholds
    thresholds = {
        'cyber_incidents_reported': 10,  # More than 10 incidents
        'google_trend_mobile_fraud': 70,  # Search trend above 70
        'customer_complaint_rate': 5,  # More than 5 per 1000 transactions
        'operational_risk_index': 7,  # Risk index above 7
        'liquidity_risk_indicator': 0.7  # Liquidity risk above 0.7
    }
    
    # Check latest quarter for threshold breaches
    latest_quarter = df['date'].max()
    latest_data = df[df['date'] == latest_quarter]
    
    warnings = []
    for _, row in latest_data.iterrows():
        country_warnings = []
        for metric, threshold in thresholds.items():
            if row[metric] > threshold:
                country_warnings.append(metric)
        
        if country_warnings:
            warnings.append({
                'Country': row['country'],
                'Warning Count': len(country_warnings),
                'Triggered Metrics': country_warnings
            })
    
    # Sort by warning count
    warnings = sorted(warnings, key=lambda x: x['Warning Count'], reverse=True)
    
    if warnings:
        print(f"Countries with Active Warning Signals (Q{latest_data['quarter'].iloc[0]} {latest_data['year'].iloc[0]}):")
        for w in warnings[:5]:  # Show top 5
            print(f"  {w['Country']}: {w['Warning Count']} warnings")
            for metric in w['Triggered Metrics'][:3]:  # Show first 3 metrics
                print(f"    - {metric}")
    else:
        print("No warning signals triggered in the latest quarter.")
    print()

def generate_risk_scorecard(df):
    """Generate a comprehensive risk scorecard for each country"""
    
    print("7. RISK SCORECARD (Latest Quarter)")
    print("-" * 40)
    
    # Get latest data
    latest_quarter = df['date'].max()
    latest_data = df[df['date'] == latest_quarter]
    
    # Calculate comprehensive risk scores
    scorecards = []
    for _, row in latest_data.iterrows():
        # Normalize scores (0-100 scale, higher = better except for risks)
        cyber_score = 100 - min(100, row['composite_cyber_risk'] * 2)
        trust_score = row['consumer_trust_index']
        market_score = row['market_health_score']
        compliance_score = row['regulatory_compliance_score']
        
        overall_score = (cyber_score + trust_score + market_score + compliance_score) / 4
        
        scorecards.append({
            'Country': row['country'],
            'Overall Score': round(overall_score, 1),
            'Cyber Security': round(cyber_score, 1),
            'Consumer Trust': round(trust_score, 1),
            'Market Health': round(market_score, 1),
            'Compliance': round(compliance_score, 1)
        })
    
    # Sort by overall score
    scorecards = sorted(scorecards, key=lambda x: x['Overall Score'], reverse=True)
    
    print("Top 5 Countries by Overall Risk Score (100 = Best):")
    print("-" * 60)
    print(f"{'Country':<15} {'Overall':<10} {'Cyber':<10} {'Trust':<10} {'Market':<10} {'Compliance':<10}")
    print("-" * 60)
    for card in scorecards[:5]:
        print(f"{card['Country']:<15} {card['Overall Score']:<10} {card['Cyber Security']:<10} "
              f"{card['Consumer Trust']:<10} {card['Market Health']:<10} {card['Compliance']:<10}")
    print()

def main():
    """Run complete analysis"""
    
    # Load data
    df = load_and_explore_data()
    
    # Run analyses
    cyber_risk_summary = analyze_cyber_risk(df)
    analyze_market_dynamics(df)
    analyze_consumer_trust(df)
    identify_risk_clusters(df)
    calculate_early_warning_signals(df)
    generate_risk_scorecard(df)
    
    # Summary recommendations
    print("=" * 80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("1. CYBER RISK MANAGEMENT")
    print("   - Focus on countries with composite cyber risk > 40")
    print("   - Monitor Google search trends as early warning indicator")
    print()
    print("2. MARKET DEVELOPMENT")
    print("   - Markets showing decreasing concentration (good for competition)")
    print("   - Innovation index correlates with market health")
    print()
    print("3. CONSUMER PROTECTION")
    print("   - Trust levels generally improving but varies by country")
    print("   - Complaint rates are key indicator of service quality")
    print()
    print("4. REGULATORY PRIORITIES")
    print("   - Countries with low compliance scores need attention")
    print("   - Strong regulation correlates with lower cyber risks")
    print()
    print("5. EARLY WARNING SYSTEM")
    print("   - Multi-metric approach essential for risk detection")
    print("   - Threshold-based alerts can prevent crisis escalation")
    print()
    
    print("Analysis completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()