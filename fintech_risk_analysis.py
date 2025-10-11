#!/usr/bin/env python3
"""
FinTech Risk Nexus Analysis Script
Demonstrates how to analyze the generated dataset for early warning model development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinTechRiskAnalyzer:
    """Analyzes FinTech risk nexus dataset for early warning model development"""
    
    def __init__(self, data_dir='/workspace'):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """Load all dataset components"""
        print("Loading FinTech Risk Nexus Dataset...")
        
        self.cyber_risk = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_cyber_risk.csv")
        self.consumer_sentiment = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_consumer_sentiment.csv")
        self.competitive_dynamics = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_competitive_dynamics.csv")
        self.macro_economic = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_macro_economic.csv")
        self.summary_stats = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_summary_statistics.csv")
        
        print("Dataset loaded successfully!")
        print(f"Cyber Risk Records: {len(self.cyber_risk)}")
        print(f"Consumer Sentiment Records: {len(self.consumer_sentiment)}")
        print(f"Competitive Dynamics Records: {len(self.competitive_dynamics)}")
        print(f"Macro Economic Records: {len(self.macro_economic)}")
        print(f"Summary Statistics Records: {len(self.summary_stats)}")
    
    def analyze_cyber_risk_trends(self):
        """Analyze cyber risk trends across countries and time"""
        print("\n" + "="*60)
        print("CYBER RISK ANALYSIS")
        print("="*60)
        
        # Country-level cyber risk summary
        cyber_summary = self.cyber_risk.groupby('country').agg({
            'cyber_incidents': ['mean', 'std', 'max'],
            'mobile_money_fraud_cases': ['mean', 'std', 'max'],
            'data_breach_severity': ['mean', 'std', 'max'],
            'cyber_risk_score': 'mean'
        }).round(2)
        
        print("\nTop 10 Countries by Average Cyber Incidents:")
        top_cyber_risk = cyber_summary['cyber_incidents']['mean'].sort_values(ascending=False).head(10)
        print(top_cyber_risk)
        
        # Time series analysis
        print("\nCyber Risk Trends by Year:")
        yearly_cyber = self.cyber_risk.groupby('year').agg({
            'cyber_incidents': 'mean',
            'mobile_money_fraud_cases': 'mean',
            'data_breach_severity': 'mean'
        }).round(2)
        print(yearly_cyber)
        
        return cyber_summary, yearly_cyber
    
    def analyze_consumer_sentiment(self):
        """Analyze consumer sentiment and trust patterns"""
        print("\n" + "="*60)
        print("CONSUMER SENTIMENT ANALYSIS")
        print("="*60)
        
        # Brand-level sentiment analysis
        brand_sentiment = self.consumer_sentiment.groupby('fintech_brand').agg({
            'sentiment_score': ['mean', 'std'],
            'trust_index': ['mean', 'std'],
            'customer_satisfaction': ['mean', 'std'],
            'app_store_rating': ['mean', 'std']
        }).round(3)
        
        print("\nTop 10 FinTech Brands by Average Sentiment Score:")
        top_brands = brand_sentiment['sentiment_score']['mean'].sort_values(ascending=False).head(10)
        print(top_brands)
        
        # Country-level sentiment analysis
        country_sentiment = self.consumer_sentiment.groupby('country').agg({
            'sentiment_score': 'mean',
            'trust_index': 'mean',
            'customer_satisfaction': 'mean'
        }).round(3)
        
        print("\nTop 10 Countries by Average Trust Index:")
        top_trust = country_sentiment['trust_index'].sort_values(ascending=False).head(10)
        print(top_trust)
        
        return brand_sentiment, country_sentiment
    
    def analyze_competitive_dynamics(self):
        """Analyze competitive dynamics and market structure"""
        print("\n" + "="*60)
        print("COMPETITIVE DYNAMICS ANALYSIS")
        print("="*60)
        
        # Market concentration analysis
        concentration_analysis = self.competitive_dynamics.groupby('market_concentration').agg({
            'hhi_index': 'mean',
            'new_fintech_licenses': 'mean',
            'fintech_investment_millions_usd': 'mean'
        }).round(2)
        
        print("\nMarket Concentration Analysis:")
        print(concentration_analysis)
        
        # Country-level competitive dynamics
        country_competition = self.competitive_dynamics.groupby('country').agg({
            'hhi_index': 'mean',
            'new_fintech_licenses': 'sum',
            'fintech_investment_millions_usd': 'mean',
            'digital_adoption_rate': 'mean'
        }).round(2)
        
        print("\nTop 10 Countries by Total New Licenses (5 years):")
        top_licenses = country_competition['new_fintech_licenses'].sort_values(ascending=False).head(10)
        print(top_licenses)
        
        return concentration_analysis, country_competition
    
    def analyze_risk_correlations(self):
        """Analyze correlations between different risk factors"""
        print("\n" + "="*60)
        print("RISK CORRELATION ANALYSIS")
        print("="*60)
        
        # Merge data for correlation analysis
        merged_data = self.summary_stats.merge(
            self.cyber_risk.groupby('country').agg({
                'cyber_incidents': 'mean',
                'mobile_money_fraud_cases': 'mean',
                'data_breach_severity': 'mean'
            }),
            on='country'
        )
        
        merged_data = merged_data.merge(
            self.consumer_sentiment.groupby('country').agg({
                'sentiment_score': 'mean',
                'trust_index': 'mean'
            }),
            on='country'
        )
        
        merged_data = merged_data.merge(
            self.competitive_dynamics.groupby('country').agg({
                'hhi_index': 'mean',
                'digital_adoption_rate': 'mean'
            }),
            on='country'
        )
        
        # Calculate correlations
        correlation_vars = [
            'avg_cyber_incidents_per_quarter',
            'avg_fraud_cases_per_quarter',
            'avg_sentiment_score',
            'avg_trust_index',
            'avg_hhi_index',
            'avg_gdp_per_capita_usd',
            'avg_financial_inclusion_pct'
        ]
        
        correlation_matrix = merged_data[correlation_vars].corr()
        
        print("\nRisk Factor Correlations:")
        print(correlation_matrix.round(3))
        
        # Key correlations
        print("\nKey Correlations:")
        print(f"Cyber Incidents vs Trust Index: {correlation_matrix.loc['avg_cyber_incidents_per_quarter', 'avg_trust_index']:.3f}")
        print(f"Fraud Cases vs Sentiment Score: {correlation_matrix.loc['avg_fraud_cases_per_quarter', 'avg_sentiment_score']:.3f}")
        print(f"HHI vs Financial Inclusion: {correlation_matrix.loc['avg_hhi_index', 'avg_financial_inclusion_pct']:.3f}")
        
        return correlation_matrix
    
    def identify_high_risk_countries(self):
        """Identify countries with highest risk profiles"""
        print("\n" + "="*60)
        print("HIGH RISK COUNTRY IDENTIFICATION")
        print("="*60)
        
        # Create composite risk score
        risk_factors = self.summary_stats.copy()
        
        # Normalize risk factors (higher values = higher risk)
        risk_factors['cyber_risk_norm'] = (risk_factors['avg_cyber_incidents_per_quarter'] - 
                                         risk_factors['avg_cyber_incidents_per_quarter'].min()) / \
                                        (risk_factors['avg_cyber_incidents_per_quarter'].max() - 
                                         risk_factors['avg_cyber_incidents_per_quarter'].min())
        
        risk_factors['fraud_risk_norm'] = (risk_factors['avg_fraud_cases_per_quarter'] - 
                                         risk_factors['avg_fraud_cases_per_quarter'].min()) / \
                                        (risk_factors['avg_fraud_cases_per_quarter'].max() - 
                                         risk_factors['avg_fraud_cases_per_quarter'].min())
        
        # Lower sentiment and trust = higher risk
        risk_factors['sentiment_risk_norm'] = 1 - (risk_factors['avg_sentiment_score'] - 
                                                 risk_factors['avg_sentiment_score'].min()) / \
                                                (risk_factors['avg_sentiment_score'].max() - 
                                                 risk_factors['avg_sentiment_score'].min())
        
        risk_factors['trust_risk_norm'] = 1 - (risk_factors['avg_trust_index'] - 
                                             risk_factors['avg_trust_index'].min()) / \
                                            (risk_factors['avg_trust_index'].max() - 
                                             risk_factors['avg_trust_index'].min())
        
        # Composite risk score
        risk_factors['composite_risk_score'] = (
            risk_factors['cyber_risk_norm'] * 0.3 +
            risk_factors['fraud_risk_norm'] * 0.3 +
            risk_factors['sentiment_risk_norm'] * 0.2 +
            risk_factors['trust_risk_norm'] * 0.2
        )
        
        # Rank countries by risk
        risk_ranking = risk_factors[['country', 'composite_risk_score', 'risk_level']].sort_values(
            'composite_risk_score', ascending=False
        )
        
        print("\nTop 15 Highest Risk Countries:")
        print(risk_ranking.head(15).round(3))
        
        print("\nRisk Level Distribution:")
        print(risk_factors['risk_level'].value_counts())
        
        return risk_ranking
    
    def generate_early_warning_indicators(self):
        """Generate early warning indicators for FinTech risk"""
        print("\n" + "="*60)
        print("EARLY WARNING INDICATORS")
        print("="*60)
        
        # Calculate early warning indicators
        early_warning = self.summary_stats.copy()
        
        # Cyber risk early warning (incidents > 40 per quarter)
        early_warning['cyber_risk_alert'] = early_warning['avg_cyber_incidents_per_quarter'] > 40
        
        # Sentiment early warning (sentiment < 0.1)
        early_warning['sentiment_alert'] = early_warning['avg_sentiment_score'] < 0.1
        
        # Trust early warning (trust index < 30)
        early_warning['trust_alert'] = early_warning['avg_trust_index'] < 30
        
        # Market concentration early warning (HHI > 2500)
        early_warning['concentration_alert'] = early_warning['avg_hhi_index'] > 2500
        
        # Count total alerts per country
        early_warning['total_alerts'] = (
            early_warning['cyber_risk_alert'].astype(int) +
            early_warning['sentiment_alert'].astype(int) +
            early_warning['trust_alert'].astype(int) +
            early_warning['concentration_alert'].astype(int)
        )
        
        # Countries with multiple alerts
        multi_alert_countries = early_warning[early_warning['total_alerts'] >= 2][
            ['country', 'total_alerts', 'cyber_risk_alert', 'sentiment_alert', 
             'trust_alert', 'concentration_alert']
        ].sort_values('total_alerts', ascending=False)
        
        print("\nCountries with Multiple Early Warning Alerts:")
        print(multi_alert_countries)
        
        print(f"\nTotal countries with alerts: {len(early_warning[early_warning['total_alerts'] > 0])}")
        print(f"Countries with 3+ alerts: {len(early_warning[early_warning['total_alerts'] >= 3])}")
        
        return early_warning
    
    def run_complete_analysis(self):
        """Run complete analysis suite"""
        print("FinTech Risk Nexus Analysis - Sub-Saharan Africa")
        print("="*80)
        
        # Run all analyses
        cyber_analysis = self.analyze_cyber_risk_trends()
        sentiment_analysis = self.analyze_consumer_sentiment()
        competition_analysis = self.analyze_competitive_dynamics()
        correlations = self.analyze_risk_correlations()
        risk_ranking = self.identify_high_risk_countries()
        early_warning = self.generate_early_warning_indicators()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("Key Findings:")
        print(f"- Analyzed {len(self.summary_stats)} countries")
        print(f"- Identified {len(early_warning[early_warning['total_alerts'] > 0])} countries with early warning alerts")
        print(f"- Top risk country: {risk_ranking.iloc[0]['country']}")
        print(f"- Average cyber incidents per quarter: {self.summary_stats['avg_cyber_incidents_per_quarter'].mean():.1f}")
        print(f"- Average trust index: {self.summary_stats['avg_trust_index'].mean():.1f}")
        
        return {
            'cyber_analysis': cyber_analysis,
            'sentiment_analysis': sentiment_analysis,
            'competition_analysis': competition_analysis,
            'correlations': correlations,
            'risk_ranking': risk_ranking,
            'early_warning': early_warning
        }

def main():
    """Main function to run the analysis"""
    analyzer = FinTechRiskAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis results saved in memory. Use the analyzer object to access specific analyses.")
    print("Example: analyzer.cyber_risk.head() to see cyber risk data")

if __name__ == "__main__":
    main()