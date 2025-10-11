#!/usr/bin/env python3
"""
FinTech Risk Nexus Dataset Analysis and Validation
Comprehensive analysis of the generated dataset for thesis research

Author: Research Assistant
Date: 2025-10-11
Purpose: Validate and analyze the FinTech Early Warning Model dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinTechDataAnalyzer:
    """
    Comprehensive analyzer for FinTech Risk Nexus Dataset
    """
    
    def __init__(self):
        # Load the generated datasets
        self.cyber_df = pd.read_csv('cyber_risk_exposure_data.csv')
        self.sentiment_df = pd.read_csv('consumer_sentiment_trust_data.csv')
        self.competitive_df = pd.read_csv('competitive_dynamics_data.csv')
        
        # Convert date columns
        self.cyber_df['date'] = pd.to_datetime(self.cyber_df['date'])
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
        self.competitive_df['date'] = pd.to_datetime(self.competitive_df['date'])
        
        print("FinTech Data Analyzer initialized")
        print(f"Loaded datasets: Cyber Risk ({len(self.cyber_df)}), Sentiment ({len(self.sentiment_df)}), Competitive ({len(self.competitive_df)})")
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("="*80)
        
        datasets = {
            'Cyber Risk Exposure': self.cyber_df,
            'Consumer Sentiment & Trust': self.sentiment_df,
            'Competitive Dynamics': self.competitive_df
        }
        
        quality_report = {}
        
        for name, df in datasets.items():
            print(f"\n{name.upper()} DATASET QUALITY:")
            print("-" * 50)
            
            # Basic statistics
            print(f"Total Records: {len(df):,}")
            print(f"Columns: {len(df.columns)}")
            print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"Countries Covered: {df['country'].nunique()}")
            
            # Missing values
            missing_values = df.isnull().sum()
            missing_pct = (missing_values / len(df)) * 100
            
            if missing_values.sum() > 0:
                print(f"\nMissing Values:")
                for col in missing_values[missing_values > 0].index:
                    print(f"  {col}: {missing_values[col]} ({missing_pct[col]:.2f}%)")
            else:
                print("✓ No missing values detected")
            
            # Duplicates
            duplicates = df.duplicated().sum()
            print(f"Duplicate Records: {duplicates}")
            
            # Data types
            print(f"\nData Types:")
            for dtype in df.dtypes.value_counts().index:
                count = df.dtypes.value_counts()[dtype]
                print(f"  {dtype}: {count} columns")
            
            quality_report[name] = {
                'records': len(df),
                'columns': len(df.columns),
                'missing_values': missing_values.sum(),
                'duplicates': duplicates,
                'date_range_days': (df['date'].max() - df['date'].min()).days
            }
        
        return quality_report
    
    def analyze_cyber_risk_trends(self):
        """Analyze cyber risk exposure trends"""
        print("\n" + "="*80)
        print("CYBER RISK EXPOSURE ANALYSIS")
        print("="*80)
        
        # Aggregate by year and country
        cyber_yearly = self.cyber_df.groupby([self.cyber_df['date'].dt.year, 'country']).agg({
            'total_cyber_incidents': 'sum',
            'mobile_money_fraud_incidents': 'sum',
            'cyber_risk_index': 'mean',
            'avg_incident_severity_score': 'mean'
        }).reset_index()
        
        # Top countries by cyber incidents
        top_risk_countries = self.cyber_df.groupby('country')['total_cyber_incidents'].sum().sort_values(ascending=False)
        
        print(f"\nTOP 10 COUNTRIES BY TOTAL CYBER INCIDENTS:")
        print("-" * 50)
        for i, (country, incidents) in enumerate(top_risk_countries.head(10).items(), 1):
            print(f"{i:2d}. {country}: {incidents:,} incidents")
        
        # Incident type analysis
        incident_columns = [col for col in self.cyber_df.columns if col.endswith('_incidents') and col != 'total_cyber_incidents']
        incident_totals = self.cyber_df[incident_columns].sum().sort_values(ascending=False)
        
        print(f"\nINCIDENT TYPES BY FREQUENCY:")
        print("-" * 50)
        for i, (incident_type, count) in enumerate(incident_totals.items(), 1):
            clean_name = incident_type.replace('_incidents', '').replace('_', ' ').title()
            print(f"{i:2d}. {clean_name}: {count:,}")
        
        # Search trends analysis
        search_columns = [col for col in self.cyber_df.columns if col.startswith('search_')]
        search_trends = self.cyber_df[search_columns].mean().sort_values(ascending=False)
        
        print(f"\nAVERAGE SEARCH TREND VOLUMES:")
        print("-" * 50)
        for i, (search_term, volume) in enumerate(search_trends.items(), 1):
            clean_name = search_term.replace('search_', '').replace('_', ' ').title()
            print(f"{i:2d}. {clean_name}: {volume:.1f}")
        
        return {
            'top_risk_countries': top_risk_countries,
            'incident_types': incident_totals,
            'search_trends': search_trends,
            'yearly_trends': cyber_yearly
        }
    
    def analyze_sentiment_patterns(self):
        """Analyze consumer sentiment and trust patterns"""
        print("\n" + "="*80)
        print("CONSUMER SENTIMENT & TRUST ANALYSIS")
        print("="*80)
        
        # Brand sentiment analysis
        brand_sentiment = self.sentiment_df.groupby('fintech_brand').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'trust_index': 'mean',
            'total_mentions': 'sum',
            'positive_mentions': 'sum',
            'negative_mentions': 'sum'
        }).round(3)
        
        # Flatten column names
        brand_sentiment.columns = ['_'.join(col).strip() for col in brand_sentiment.columns]
        brand_sentiment = brand_sentiment.sort_values('sentiment_score_mean', ascending=False)
        
        print(f"\nTOP 10 FINTECH BRANDS BY SENTIMENT SCORE:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Brand':<20} {'Sentiment':<10} {'Trust Index':<12} {'Total Mentions':<15}")
        print("-" * 70)
        
        for i, (brand, row) in enumerate(brand_sentiment.head(10).iterrows(), 1):
            print(f"{i:<4} {brand:<20} {row['sentiment_score_mean']:<10.3f} {row['trust_index_mean']:<12.1f} {int(row['total_mentions_sum']):<15,}")
        
        # Country sentiment analysis
        country_sentiment = self.sentiment_df.groupby('country').agg({
            'sentiment_score': 'mean',
            'trust_index': 'mean',
            'security_perception_score': 'mean',
            'recommendation_likelihood': 'mean'
        }).round(2).sort_values('sentiment_score', ascending=False)
        
        print(f"\nCOUNTRY SENTIMENT RANKINGS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Country':<15} {'Sentiment':<10} {'Trust':<8} {'Security':<10} {'Recommendation':<15}")
        print("-" * 80)
        
        for i, (country, row) in enumerate(country_sentiment.head(10).iterrows(), 1):
            print(f"{i:<4} {country:<15} {row['sentiment_score']:<10.3f} {row['trust_index']:<8.1f} {row['security_perception_score']:<10.1f} {row['recommendation_likelihood']:<15.1f}")
        
        # Risk perception analysis
        risk_distribution = self.sentiment_df['risk_perception_level'].value_counts()
        
        print(f"\nRISK PERCEPTION DISTRIBUTION:")
        print("-" * 50)
        for risk_level, count in risk_distribution.items():
            percentage = (count / len(self.sentiment_df)) * 100
            print(f"{risk_level}: {count:,} ({percentage:.1f}%)")
        
        return {
            'brand_sentiment': brand_sentiment,
            'country_sentiment': country_sentiment,
            'risk_distribution': risk_distribution
        }
    
    def analyze_competitive_dynamics(self):
        """Analyze competitive dynamics and market structure"""
        print("\n" + "="*80)
        print("COMPETITIVE DYNAMICS ANALYSIS")
        print("="*80)
        
        # Market concentration analysis
        latest_quarter = self.competitive_df['date'].max()
        latest_data = self.competitive_df[self.competitive_df['date'] == latest_quarter]
        
        # HHI analysis
        hhi_stats = latest_data.groupby('country')['herfindahl_hirschman_index'].first().sort_values()
        
        print(f"\nMARKET CONCENTRATION (HHI) - LATEST QUARTER:")
        print("-" * 60)
        print(f"{'Country':<15} {'HHI':<8} {'Concentration Level':<25}")
        print("-" * 60)
        
        concentration_levels = latest_data.set_index('country')['market_concentration_level']
        
        for country, hhi in hhi_stats.items():
            level = concentration_levels.get(country, 'Unknown')
            print(f"{country:<15} {hhi:<8.0f} {level:<25}")
        
        # License issuance trends
        license_trends = self.competitive_df.groupby([self.competitive_df['date'].dt.year, 'country'])['new_fintech_licenses_issued'].sum().reset_index()
        yearly_licenses = license_trends.groupby('date')['new_fintech_licenses_issued'].sum()
        
        print(f"\nFINTECH LICENSES ISSUED BY YEAR:")
        print("-" * 40)
        for year, licenses in yearly_licenses.items():
            print(f"{year}: {licenses} licenses")
        
        # Market maturity analysis
        maturity_stats = latest_data.groupby('country').agg({
            'market_maturity_score': 'first',
            'innovation_index': 'first',
            'regulatory_clarity_score': 'first',
            'number_of_active_fintech_companies': 'first'
        }).sort_values('market_maturity_score', ascending=False)
        
        print(f"\nMARKET MATURITY RANKINGS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Country':<15} {'Maturity':<10} {'Innovation':<12} {'Regulatory':<12} {'Active Firms':<12}")
        print("-" * 80)
        
        for i, (country, row) in enumerate(maturity_stats.head(10).iterrows(), 1):
            print(f"{i:<4} {country:<15} {row['market_maturity_score']:<10.1f} {row['innovation_index']:<12.1f} {row['regulatory_clarity_score']:<12.1f} {int(row['number_of_active_fintech_companies']):<12}")
        
        return {
            'hhi_stats': hhi_stats,
            'license_trends': yearly_licenses,
            'maturity_stats': maturity_stats,
            'latest_data': latest_data
        }
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for all datasets"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80)
        
        # Set up the plotting environment
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Cyber Risk Trends Over Time
        plt.subplot(4, 2, 1)
        monthly_cyber = self.cyber_df.groupby(self.cyber_df['date'].dt.to_period('M'))['total_cyber_incidents'].sum()
        monthly_cyber.plot(kind='line', color='red', linewidth=2)
        plt.title('Total Cyber Incidents Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Top Countries by Cyber Risk
        plt.subplot(4, 2, 2)
        top_countries = self.cyber_df.groupby('country')['total_cyber_incidents'].sum().sort_values(ascending=True).tail(10)
        top_countries.plot(kind='barh', color='orange')
        plt.title('Top 10 Countries by Total Cyber Incidents', fontsize=14, fontweight='bold')
        plt.xlabel('Total Incidents')
        
        # 3. Sentiment Distribution by Brand Category
        plt.subplot(4, 2, 3)
        sentiment_by_category = self.sentiment_df.groupby('brand_perception_category')['sentiment_score'].mean().sort_values()
        sentiment_by_category.plot(kind='bar', color='green', alpha=0.7)
        plt.title('Average Sentiment by Brand Perception Category', fontsize=14, fontweight='bold')
        plt.xlabel('Brand Perception Category')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45)
        
        # 4. Trust Index vs Security Perception
        plt.subplot(4, 2, 4)
        sample_data = self.sentiment_df.sample(n=1000)  # Sample for better visualization
        plt.scatter(sample_data['trust_index'], sample_data['security_perception_score'], 
                   alpha=0.6, color='blue', s=30)
        plt.title('Trust Index vs Security Perception', fontsize=14, fontweight='bold')
        plt.xlabel('Trust Index')
        plt.ylabel('Security Perception Score')
        plt.grid(True, alpha=0.3)
        
        # 5. Market Concentration (HHI) Distribution
        plt.subplot(4, 2, 5)
        self.competitive_df['herfindahl_hirschman_index'].hist(bins=20, color='purple', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Market Concentration (HHI)', fontsize=14, fontweight='bold')
        plt.xlabel('Herfindahl-Hirschman Index')
        plt.ylabel('Frequency')
        plt.axvline(x=1500, color='red', linestyle='--', label='Competitive Threshold')
        plt.axvline(x=2500, color='orange', linestyle='--', label='Concentrated Threshold')
        plt.legend()
        
        # 6. Innovation Index by Country
        plt.subplot(4, 2, 6)
        latest_competitive = self.competitive_df[self.competitive_df['date'] == self.competitive_df['date'].max()]
        innovation_by_country = latest_competitive.groupby('country')['innovation_index'].first().sort_values(ascending=True).tail(10)
        innovation_by_country.plot(kind='barh', color='teal')
        plt.title('Top 10 Countries by Innovation Index', fontsize=14, fontweight='bold')
        plt.xlabel('Innovation Index')
        
        # 7. Correlation Heatmap - Cyber Risk Factors
        plt.subplot(4, 2, 7)
        cyber_corr_cols = ['total_cyber_incidents', 'mobile_money_fraud_incidents', 'cyber_risk_index', 
                          'search_mobile_money_fraud', 'avg_incident_severity_score']
        cyber_corr = self.cyber_df[cyber_corr_cols].corr()
        sns.heatmap(cyber_corr, annot=True, cmap='RdYlBu_r', center=0, square=True, fmt='.2f')
        plt.title('Cyber Risk Factors Correlation', fontsize=14, fontweight='bold')
        
        # 8. Time Series - New FinTech Licenses
        plt.subplot(4, 2, 8)
        quarterly_licenses = self.competitive_df.groupby(self.competitive_df['date'].dt.to_period('Q'))['new_fintech_licenses_issued'].sum()
        quarterly_licenses.plot(kind='line', color='brown', linewidth=2, marker='o')
        plt.title('New FinTech Licenses Issued Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Quarter')
        plt.ylabel('Number of Licenses')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fintech_risk_nexus_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Comprehensive visualization saved as 'fintech_risk_nexus_analysis.png'")
    
    def generate_executive_summary(self):
        """Generate executive summary of the dataset"""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY - FINTECH RISK NEXUS DATASET")
        print("="*80)
        
        # Key statistics
        total_records = len(self.cyber_df) + len(self.sentiment_df) + len(self.competitive_df)
        date_range = f"{self.cyber_df['date'].min().date()} to {self.cyber_df['date'].max().date()}"
        countries_covered = len(set(self.cyber_df['country'].unique()) | 
                               set(self.sentiment_df['country'].unique()) | 
                               set(self.competitive_df['country'].unique()))
        
        print(f"\nDATASET OVERVIEW:")
        print(f"• Total Records: {total_records:,}")
        print(f"• Date Coverage: {date_range}")
        print(f"• Countries Covered: {countries_covered}")
        print(f"• FinTech Brands Analyzed: {self.sentiment_df['fintech_brand'].nunique()}")
        
        # Key findings
        print(f"\nKEY FINDINGS:")
        
        # Cyber risk findings
        total_incidents = self.cyber_df['total_cyber_incidents'].sum()
        avg_monthly_incidents = self.cyber_df.groupby(self.cyber_df['date'].dt.to_period('M'))['total_cyber_incidents'].sum().mean()
        highest_risk_country = self.cyber_df.groupby('country')['total_cyber_incidents'].sum().idxmax()
        
        print(f"• Total cyber incidents recorded: {total_incidents:,}")
        print(f"• Average monthly incidents: {avg_monthly_incidents:.0f}")
        print(f"• Highest risk country: {highest_risk_country}")
        
        # Sentiment findings
        avg_sentiment = self.sentiment_df['sentiment_score'].mean()
        avg_trust = self.sentiment_df['trust_index'].mean()
        most_trusted_brand = self.sentiment_df.groupby('fintech_brand')['trust_index'].mean().idxmax()
        
        print(f"• Average sentiment score: {avg_sentiment:.3f} (scale: -1 to 1)")
        print(f"• Average trust index: {avg_trust:.1f} (scale: 0-100)")
        print(f"• Most trusted brand: {most_trusted_brand}")
        
        # Competitive findings
        avg_hhi = self.competitive_df['herfindahl_hirschman_index'].mean()
        total_licenses = self.competitive_df['new_fintech_licenses_issued'].sum()
        most_competitive_country = self.competitive_df.groupby('country')['herfindahl_hirschman_index'].mean().idxmin()
        
        print(f"• Average market concentration (HHI): {avg_hhi:.0f}")
        print(f"• Total new licenses issued: {total_licenses}")
        print(f"• Most competitive market: {most_competitive_country}")
        
        print(f"\nDATASET APPLICATIONS:")
        print("• Early warning system development for FinTech risks")
        print("• Market concentration and competition analysis")
        print("• Consumer sentiment and trust monitoring")
        print("• Cybersecurity risk assessment and prediction")
        print("• Regulatory policy impact evaluation")
        print("• Cross-country comparative studies")
        
        return {
            'total_records': total_records,
            'countries_covered': countries_covered,
            'key_metrics': {
                'total_incidents': total_incidents,
                'avg_sentiment': avg_sentiment,
                'avg_trust': avg_trust,
                'avg_hhi': avg_hhi,
                'total_licenses': total_licenses
            }
        }
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive analysis of FinTech Risk Nexus Dataset...")
        
        # Generate all analyses
        quality_report = self.generate_data_quality_report()
        cyber_analysis = self.analyze_cyber_risk_trends()
        sentiment_analysis = self.analyze_sentiment_patterns()
        competitive_analysis = self.analyze_competitive_dynamics()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary()
        
        print(f"\n✓ Complete analysis finished successfully!")
        
        return {
            'quality_report': quality_report,
            'cyber_analysis': cyber_analysis,
            'sentiment_analysis': sentiment_analysis,
            'competitive_analysis': competitive_analysis,
            'executive_summary': executive_summary
        }

def main():
    """Main function to run the analysis"""
    analyzer = FinTechDataAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - DATASET READY FOR THESIS RESEARCH")
    print("="*80)
    print("\nGenerated Files:")
    print("• cyber_risk_exposure_data.csv")
    print("• consumer_sentiment_trust_data.csv") 
    print("• competitive_dynamics_data.csv")
    print("• fintech_risk_nexus_analysis.png")
    print("\nThe dataset is now ready for your FinTech Early Warning Model research!")

if __name__ == "__main__":
    main()