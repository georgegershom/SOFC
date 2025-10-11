#!/usr/bin/env python3
"""
FinTech Dataset Analysis Script
Provides comprehensive analysis and insights into the generated FinTech distress dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinTechDatasetAnalyzer:
    def __init__(self, data_dir='fintech_dataset'):
        """Initialize the analyzer with the dataset directory"""
        self.data_dir = Path(data_dir)
        self.load_data()
    
    def load_data(self):
        """Load all dataset files"""
        try:
            self.companies = pd.read_csv(self.data_dir / 'companies.csv')
            self.financial = pd.read_csv(self.data_dir / 'financial_metrics.csv')
            self.operational = pd.read_csv(self.data_dir / 'operational_metrics.csv')
            self.funding = pd.read_csv(self.data_dir / 'funding_data.csv')
            self.regulatory = pd.read_csv(self.data_dir / 'regulatory_data.csv')
            self.distress = pd.read_csv(self.data_dir / 'distress_indicators.csv')
            print("✅ All datasets loaded successfully")
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            raise
    
    def basic_statistics(self):
        """Generate basic statistics about the dataset"""
        print("\n" + "="*60)
        print("📊 BASIC DATASET STATISTICS")
        print("="*60)
        
        print(f"📈 Total Companies: {len(self.companies)}")
        print(f"🌍 Countries Covered: {len(self.companies['country'].unique())}")
        print(f"🏢 FinTech Types: {len(self.companies['fintech_type'].unique())}")
        print(f"⚠️  Distressed Companies: {self.distress['is_distressed'].sum()}")
        print(f"📊 Distress Rate: {self.distress['is_distressed'].mean():.1%}")
        
        print(f"\n📅 Time Period: {len(self.financial['quarter'].unique())} quarters")
        print(f"💰 Total Financial Records: {len(self.financial)}")
        print(f"👥 Total Operational Records: {len(self.operational)}")
        print(f"💵 Total Funding Rounds: {len(self.funding)}")
        print(f"⚖️  Total Regulatory Events: {len(self.regulatory)}")
    
    def country_analysis(self):
        """Analyze FinTech distribution by country"""
        print("\n" + "="*60)
        print("🌍 COUNTRY ANALYSIS")
        print("="*60)
        
        # Merge companies with distress data
        companies_with_distress = self.companies.merge(self.distress[['company_id', 'is_distressed']], on='company_id')
        
        country_stats = companies_with_distress.groupby('country').agg({
            'company_id': 'count',
            'is_distressed': 'mean',
            'gdp_per_capita': 'first',
            'mobile_penetration': 'first',
            'fintech_maturity': 'first'
        }).round(3)
        
        country_stats.columns = ['Companies', 'Distress Rate', 'GDP per Capita', 'Mobile Penetration', 'FinTech Maturity']
        country_stats = country_stats.sort_values('Companies', ascending=False)
        
        print(country_stats)
        
        # Country vs FinTech type heatmap
        country_fintech = pd.crosstab(self.companies['country'], self.companies['fintech_type'])
        print(f"\n📊 FinTech Types by Country:")
        print(country_fintech)
    
    def fintech_type_analysis(self):
        """Analyze FinTech types and their characteristics"""
        print("\n" + "="*60)
        print("🏢 FINTECH TYPE ANALYSIS")
        print("="*60)
        
        # Merge companies with distress data
        companies_with_distress = self.companies.merge(self.distress[['company_id', 'is_distressed']], on='company_id')
        
        type_stats = companies_with_distress.groupby('fintech_type').agg({
            'company_id': 'count',
            'age_years': 'mean',
            'employees': 'mean',
            'is_distressed': 'mean'
        }).round(2)
        
        type_stats.columns = ['Count', 'Avg Age (years)', 'Avg Employees', 'Distress Rate']
        type_stats = type_stats.sort_values('Distress Rate', ascending=False)
        
        print(type_stats)
        
        # Size distribution by FinTech type
        size_dist = pd.crosstab(self.companies['fintech_type'], self.companies['company_size'])
        print(f"\n📊 Company Size Distribution by FinTech Type:")
        print(size_dist)
    
    def financial_analysis(self):
        """Analyze financial performance metrics"""
        print("\n" + "="*60)
        print("💰 FINANCIAL ANALYSIS")
        print("="*60)
        
        # Latest quarter financial data
        latest_financial = self.financial[self.financial['quarter'] == 'Q8'].copy()
        latest_financial = latest_financial.merge(self.distress[['company_id', 'is_distressed']], on='company_id')
        
        print("📈 Revenue Analysis (Latest Quarter):")
        revenue_stats = latest_financial.groupby('is_distressed')['revenue_usd'].agg(['count', 'mean', 'median', 'std']).round(2)
        revenue_stats.columns = ['Count', 'Mean', 'Median', 'Std Dev']
        print(revenue_stats)
        
        print(f"\n💵 Profit Margin Analysis:")
        margin_stats = latest_financial.groupby('is_distressed')['profit_margin'].agg(['mean', 'median', 'std']).round(3)
        margin_stats.columns = ['Mean', 'Median', 'Std Dev']
        print(margin_stats)
        
        # Revenue growth analysis
        print(f"\n📊 Revenue Growth Analysis:")
        growth_stats = latest_financial.groupby('is_distressed')['revenue_growth_rate'].agg(['mean', 'median', 'std']).round(3)
        growth_stats.columns = ['Mean', 'Median', 'Std Dev']
        print(growth_stats)
    
    def operational_analysis(self):
        """Analyze operational metrics"""
        print("\n" + "="*60)
        print("👥 OPERATIONAL ANALYSIS")
        print("="*60)
        
        # Latest quarter operational data
        latest_operational = self.operational[self.operational['quarter'] == 'Q8'].copy()
        latest_operational = latest_operational.merge(self.distress[['company_id', 'is_distressed']], on='company_id')
        
        print("👥 User Analysis (Latest Quarter):")
        user_stats = latest_operational.groupby('is_distressed')['active_users'].agg(['count', 'mean', 'median', 'std']).round(0)
        user_stats.columns = ['Count', 'Mean', 'Median', 'Std Dev']
        print(user_stats)
        
        print(f"\n💳 Transaction Analysis:")
        tx_stats = latest_operational.groupby('is_distressed')['transaction_volume_usd'].agg(['mean', 'median', 'std']).round(2)
        tx_stats.columns = ['Mean', 'Median', 'Std Dev']
        print(tx_stats)
        
        print(f"\n📉 Churn Rate Analysis:")
        churn_stats = latest_operational.groupby('is_distressed')['churn_rate'].agg(['mean', 'median', 'std']).round(3)
        churn_stats.columns = ['Mean', 'Median', 'Std Dev']
        print(churn_stats)
    
    def funding_analysis(self):
        """Analyze funding patterns"""
        print("\n" + "="*60)
        print("💵 FUNDING ANALYSIS")
        print("="*60)
        
        # Funding rounds by stage
        funding_by_stage = self.funding.groupby('round_type').agg({
            'amount_raised_usd': ['count', 'mean', 'median'],
            'valuation_usd': ['mean', 'median']
        }).round(2)
        
        funding_by_stage.columns = ['Count', 'Avg Amount', 'Median Amount', 'Avg Valuation', 'Median Valuation']
        print("💰 Funding by Round Type:")
        print(funding_by_stage)
        
        # Funding by country
        funding_country = self.funding.merge(self.companies[['company_id', 'country']], on='company_id')
        country_funding = funding_country.groupby('country')['amount_raised_usd'].agg(['count', 'sum', 'mean']).round(2)
        country_funding.columns = ['Rounds', 'Total Raised', 'Avg per Round']
        country_funding = country_funding.sort_values('Total Raised', ascending=False)
        print(f"\n🌍 Funding by Country:")
        print(country_funding.head(10))
    
    def regulatory_analysis(self):
        """Analyze regulatory events"""
        print("\n" + "="*60)
        print("⚖️ REGULATORY ANALYSIS")
        print("="*60)
        
        if len(self.regulatory) > 0:
            print("📊 Regulatory Events by Type:")
            event_types = self.regulatory['event_type'].value_counts()
            print(event_types)
            
            print(f"\n⚠️ Regulatory Events by Severity:")
            severity_counts = self.regulatory['severity'].value_counts()
            print(severity_counts)
            
            # Regulatory events by country
            reg_country = self.regulatory.merge(self.companies[['company_id', 'country']], on='company_id')
            country_events = reg_country.groupby('country')['event_type'].count().sort_values(ascending=False)
            print(f"\n🌍 Regulatory Events by Country:")
            print(country_events)
        else:
            print("No regulatory events in the dataset")
    
    def distress_analysis(self):
        """Analyze distress patterns and risk factors"""
        print("\n" + "="*60)
        print("⚠️ DISTRESS ANALYSIS")
        print("="*60)
        
        # Distress score distribution
        print("📊 Distress Score Distribution:")
        distress_stats = self.distress['distress_score'].describe()
        print(distress_stats.round(3))
        
        # Risk factors correlation
        risk_factors = ['revenue_decline_rate', 'user_decline_rate', 'has_regulatory_issues', 'regulatory_issues_count']
        risk_corr = self.distress[['distress_score'] + risk_factors].corr()['distress_score'].drop('distress_score')
        print(f"\n🔗 Risk Factor Correlations with Distress Score:")
        print(risk_corr.round(3))
        
        # Distress by FinTech type
        distress_by_type = self.distress.merge(self.companies[['company_id', 'fintech_type']], on='company_id')
        type_distress = distress_by_type.groupby('fintech_type')['is_distressed'].agg(['count', 'sum', 'mean']).round(3)
        type_distress.columns = ['Total', 'Distressed', 'Distress Rate']
        type_distress = type_distress.sort_values('Distress Rate', ascending=False)
        print(f"\n🏢 Distress Rate by FinTech Type:")
        print(type_distress)
    
    def generate_insights(self):
        """Generate key insights and recommendations"""
        print("\n" + "="*60)
        print("💡 KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Calculate key metrics
        total_companies = len(self.companies)
        distressed_companies = self.distress['is_distressed'].sum()
        distress_rate = distressed_companies / total_companies
        
        # Top risk factors
        risk_corr = self.distress[['distress_score', 'revenue_decline_rate', 'user_decline_rate', 'has_regulatory_issues']].corr()['distress_score'].drop('distress_score')
        top_risk_factor = risk_corr.idxmax()
        
        # Most distressed FinTech type
        distress_by_type = self.distress.merge(self.companies[['company_id', 'fintech_type']], on='company_id')
        most_distressed_type = distress_by_type.groupby('fintech_type')['is_distressed'].mean().idxmax()
        
        # Most active country
        most_active_country = self.companies['country'].value_counts().index[0]
        
        print(f"🎯 Key Findings:")
        print(f"   • Overall distress rate: {distress_rate:.1%}")
        print(f"   • Most significant risk factor: {top_risk_factor}")
        print(f"   • Most distressed FinTech type: {most_distressed_type}")
        print(f"   • Most active FinTech market: {most_active_country}")
        
        print(f"\n📋 Research Recommendations:")
        print(f"   1. Focus on {top_risk_factor} as primary early warning indicator")
        print(f"   2. Investigate {most_distressed_type} sector-specific risk factors")
        print(f"   3. Study regulatory impact on FinTech stability")
        print(f"   4. Analyze funding patterns and their relationship to distress")
        print(f"   5. Develop country-specific risk models for SSA markets")
    
    def run_complete_analysis(self):
        """Run all analysis methods"""
        print("🚀 Starting FinTech Dataset Analysis...")
        
        self.basic_statistics()
        self.country_analysis()
        self.fintech_type_analysis()
        self.financial_analysis()
        self.operational_analysis()
        self.funding_analysis()
        self.regulatory_analysis()
        self.distress_analysis()
        self.generate_insights()
        
        print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyzer = FinTechDatasetAnalyzer()
    analyzer.run_complete_analysis()