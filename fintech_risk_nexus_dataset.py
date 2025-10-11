#!/usr/bin/env python3
"""
FinTech Risk Nexus Dataset Generator
Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

This script generates Category 4: Nexus-Specific & Alternative Data for Sub-Saharan African economies.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FinTechRiskNexusDataGenerator:
    """Generates comprehensive FinTech risk nexus dataset for Sub-Saharan Africa"""
    
    def __init__(self):
        self.countries = [
            'Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Uganda', 'Tanzania',
            'Ethiopia', 'Angola', 'Mozambique', 'Zambia', 'Zimbabwe', 'Rwanda',
            'Senegal', 'Burkina Faso', 'Mali', 'Niger', 'Chad', 'Cameroon',
            'Côte d\'Ivoire', 'Madagascar', 'Malawi', 'Botswana', 'Namibia',
            'Mauritius', 'Seychelles', 'Eswatini', 'Lesotho'
        ]
        
        self.fintech_brands = [
            'M-Pesa', 'MTN Mobile Money', 'Airtel Money', 'Orange Money',
            'Tigo Pesa', 'Vodacom M-Pesa', 'Ecobank Mobile', 'GTBank',
            'Access Bank', 'First Bank', 'UBA', 'Zenith Bank', 'Fidelity Bank',
            'Sterling Bank', 'Kuda Bank', 'Opay', 'PalmPay', 'Carbon',
            'FairMoney', 'Branch', 'Tala', 'Jumo', 'MFS Africa', 'Flutterwave',
            'Paystack', 'Interswitch', 'Cellulant', 'Paga', 'Korapay'
        ]
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def generate_cyber_risk_data(self, years: int = 5) -> pd.DataFrame:
        """Generate cyber risk exposure data"""
        print("Generating cyber risk exposure data...")
        
        data = []
        start_date = datetime.now() - timedelta(days=years*365)
        
        for country in self.countries:
            # Base cyber risk level varies by country development
            base_risk = np.random.uniform(0.3, 0.9)
            
            for year in range(years):
                current_date = start_date + timedelta(days=year*365)
                
                # Cybersecurity incidents (quarterly data)
                for quarter in range(4):
                    quarter_date = current_date + timedelta(days=quarter*91)
                    
                    # Number of cybersecurity incidents
                    incidents = np.random.poisson(
                        np.random.uniform(5, 50) * (1 + base_risk)
                    )
                    
                    # Google search trends for mobile money fraud (0-100 scale)
                    search_trends = np.random.uniform(20, 90) * (1 + base_risk * 0.5)
                    
                    # Mobile money fraud cases
                    fraud_cases = np.random.poisson(
                        np.random.uniform(10, 200) * (1 + base_risk)
                    )
                    
                    # Data breach severity (1-10 scale)
                    breach_severity = np.random.uniform(1, 10) * base_risk
                    
                    # Digital payment volume (in millions USD)
                    payment_volume = np.random.uniform(100, 5000) * (1 + np.random.uniform(-0.3, 0.5))
                    
                    data.append({
                        'country': country,
                        'year': current_date.year,
                        'quarter': quarter + 1,
                        'date': quarter_date.strftime('%Y-%m-%d'),
                        'cyber_incidents': incidents,
                        'mobile_money_fraud_search_trends': search_trends,
                        'mobile_money_fraud_cases': fraud_cases,
                        'data_breach_severity': breach_severity,
                        'digital_payment_volume_millions_usd': payment_volume,
                        'cyber_risk_score': base_risk
                    })
        
        return pd.DataFrame(data)
    
    def generate_consumer_sentiment_data(self, years: int = 5) -> pd.DataFrame:
        """Generate consumer sentiment and trust data"""
        print("Generating consumer sentiment and trust data...")
        
        data = []
        start_date = datetime.now() - timedelta(days=years*365)
        
        for country in self.countries:
            # Base trust level varies by country
            base_trust = np.random.uniform(0.4, 0.9)
            
            for year in range(years):
                current_date = start_date + timedelta(days=year*365)
                
                for month in range(12):
                    month_date = current_date + timedelta(days=month*30)
                    
                    # Social media sentiment analysis (Twitter, Facebook, etc.)
                    for brand in random.sample(self.fintech_brands, 5):  # Sample 5 brands per country
                        
                        # Sentiment score (-1 to 1, where 1 is most positive)
                        sentiment = np.random.normal(0.2, 0.4) * base_trust
                        sentiment = max(-1, min(1, sentiment))  # Clamp to [-1, 1]
                        
                        # Trust index (0-100)
                        trust_index = (sentiment + 1) * 50 * base_trust
                        
                        # Social media mentions
                        mentions = np.random.poisson(np.random.uniform(50, 1000) * base_trust)
                        
                        # Customer satisfaction score (1-10)
                        satisfaction = np.random.uniform(3, 9) * base_trust
                        
                        # App store rating (1-5)
                        app_rating = np.random.uniform(2.5, 5) * base_trust
                        
                        # Customer complaints (monthly)
                        complaints = np.random.poisson(np.random.uniform(10, 200) * (1 - base_trust))
                        
                        data.append({
                            'country': country,
                            'year': current_date.year,
                            'month': month + 1,
                            'date': month_date.strftime('%Y-%m-%d'),
                            'fintech_brand': brand,
                            'sentiment_score': sentiment,
                            'trust_index': trust_index,
                            'social_media_mentions': mentions,
                            'customer_satisfaction': satisfaction,
                            'app_store_rating': app_rating,
                            'customer_complaints': complaints,
                            'base_trust_level': base_trust
                        })
        
        return pd.DataFrame(data)
    
    def generate_competitive_dynamics_data(self, years: int = 5) -> pd.DataFrame:
        """Generate competitive dynamics data"""
        print("Generating competitive dynamics data...")
        
        data = []
        start_date = datetime.now() - timedelta(days=years*365)
        
        for country in self.countries:
            # Base market development level
            market_dev = np.random.uniform(0.3, 0.9)
            
            for year in range(years):
                current_date = start_date + timedelta(days=year*365)
                
                # Number of FinTech licenses issued
                new_licenses = np.random.poisson(np.random.uniform(2, 15) * market_dev)
                
                # Total active FinTech companies
                total_companies = np.random.uniform(20, 200) * market_dev
                
                # Market share of top 5 companies (for HHI calculation)
                top5_market_share = np.random.uniform(0.6, 0.95) * market_dev
                
                # Calculate HHI (Herfindahl-Hirschman Index)
                # HHI = sum of squared market shares
                individual_shares = np.random.dirichlet(np.ones(5)) * top5_market_share
                hhi = np.sum(individual_shares ** 2) * 10000  # Scale to 0-10000
                
                # Market concentration level
                concentration = 'High' if hhi > 2500 else 'Moderate' if hhi > 1500 else 'Low'
                
                # Investment in FinTech (millions USD)
                investment = np.random.uniform(10, 500) * market_dev
                
                # Number of partnerships/mergers
                partnerships = np.random.poisson(np.random.uniform(1, 10) * market_dev)
                
                # Regulatory changes (binary: 0 or 1)
                regulatory_changes = np.random.binomial(1, 0.3)
                
                # Digital adoption rate
                digital_adoption = np.random.uniform(0.2, 0.8) * market_dev
                
                data.append({
                    'country': country,
                    'year': current_date.year,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'new_fintech_licenses': new_licenses,
                    'total_fintech_companies': int(total_companies),
                    'hhi_index': hhi,
                    'market_concentration': concentration,
                    'top5_market_share': top5_market_share,
                    'fintech_investment_millions_usd': investment,
                    'partnerships_mergers': partnerships,
                    'regulatory_changes': regulatory_changes,
                    'digital_adoption_rate': digital_adoption,
                    'market_development_level': market_dev
                })
        
        return pd.DataFrame(data)
    
    def generate_macro_economic_indicators(self, years: int = 5) -> pd.DataFrame:
        """Generate macro-economic indicators for context"""
        print("Generating macro-economic indicators...")
        
        data = []
        start_date = datetime.now() - timedelta(days=years*365)
        
        for country in self.countries:
            # Base economic indicators
            gdp_per_capita = np.random.uniform(500, 15000)
            inflation_rate = np.random.uniform(2, 25)
            unemployment_rate = np.random.uniform(5, 30)
            
            for year in range(years):
                current_date = start_date + timedelta(days=year*365)
                
                # GDP per capita (USD)
                gdp_growth = np.random.uniform(-5, 8)
                gdp_per_capita *= (1 + gdp_growth/100)
                
                # Inflation rate (%)
                inflation_rate += np.random.uniform(-2, 3)
                inflation_rate = max(0, inflation_rate)
                
                # Unemployment rate (%)
                unemployment_rate += np.random.uniform(-2, 2)
                unemployment_rate = max(0, min(50, unemployment_rate))
                
                # Mobile penetration rate (%)
                mobile_penetration = np.random.uniform(60, 120)
                
                # Internet penetration rate (%)
                internet_penetration = np.random.uniform(20, 80)
                
                # Financial inclusion rate (%)
                financial_inclusion = np.random.uniform(30, 90)
                
                data.append({
                    'country': country,
                    'year': current_date.year,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'gdp_per_capita_usd': gdp_per_capita,
                    'inflation_rate_pct': inflation_rate,
                    'unemployment_rate_pct': unemployment_rate,
                    'mobile_penetration_pct': mobile_penetration,
                    'internet_penetration_pct': internet_penetration,
                    'financial_inclusion_pct': financial_inclusion
                })
        
        return pd.DataFrame(data)
    
    def generate_complete_dataset(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Generate complete FinTech risk nexus dataset"""
        print(f"Generating complete FinTech risk nexus dataset for {years} years...")
        print(f"Covering {len(self.countries)} Sub-Saharan African countries")
        
        dataset = {}
        
        # Generate all data components
        dataset['cyber_risk'] = self.generate_cyber_risk_data(years)
        dataset['consumer_sentiment'] = self.generate_consumer_sentiment_data(years)
        dataset['competitive_dynamics'] = self.generate_competitive_dynamics_data(years)
        dataset['macro_economic'] = self.generate_macro_economic_indicators(years)
        
        # Create summary statistics
        summary_stats = self.create_summary_statistics(dataset)
        dataset['summary_statistics'] = summary_stats
        
        print("Dataset generation completed!")
        return dataset
    
    def create_summary_statistics(self, dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create summary statistics for the dataset"""
        summary_data = []
        
        for country in self.countries:
            # Cyber risk summary
            cyber_data = dataset['cyber_risk'][dataset['cyber_risk']['country'] == country]
            avg_cyber_incidents = cyber_data['cyber_incidents'].mean()
            avg_fraud_cases = cyber_data['mobile_money_fraud_cases'].mean()
            
            # Consumer sentiment summary
            sentiment_data = dataset['consumer_sentiment'][dataset['consumer_sentiment']['country'] == country]
            avg_sentiment = sentiment_data['sentiment_score'].mean()
            avg_trust = sentiment_data['trust_index'].mean()
            
            # Competitive dynamics summary
            comp_data = dataset['competitive_dynamics'][dataset['competitive_dynamics']['country'] == country]
            avg_hhi = comp_data['hhi_index'].mean()
            total_licenses = comp_data['new_fintech_licenses'].sum()
            
            # Macro economic summary
            macro_data = dataset['macro_economic'][dataset['macro_economic']['country'] == country]
            avg_gdp = macro_data['gdp_per_capita_usd'].mean()
            avg_financial_inclusion = macro_data['financial_inclusion_pct'].mean()
            
            summary_data.append({
                'country': country,
                'avg_cyber_incidents_per_quarter': avg_cyber_incidents,
                'avg_fraud_cases_per_quarter': avg_fraud_cases,
                'avg_sentiment_score': avg_sentiment,
                'avg_trust_index': avg_trust,
                'avg_hhi_index': avg_hhi,
                'total_new_licenses_5_years': total_licenses,
                'avg_gdp_per_capita_usd': avg_gdp,
                'avg_financial_inclusion_pct': avg_financial_inclusion,
                'risk_level': 'High' if avg_cyber_incidents > 30 else 'Medium' if avg_cyber_incidents > 15 else 'Low'
            })
        
        return pd.DataFrame(summary_data)
    
    def save_dataset(self, dataset: Dict[str, pd.DataFrame], output_dir: str = '/workspace'):
        """Save dataset to CSV files"""
        print(f"Saving dataset to {output_dir}...")
        
        for name, df in dataset.items():
            filename = f"{output_dir}/fintech_risk_nexus_{name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename} with {len(df)} records")
        
        # Save metadata
        metadata = {
            'dataset_name': 'FinTech Risk Nexus Dataset - Sub-Saharan Africa',
            'description': 'Category 4: Nexus-Specific & Alternative Data for FinTech Early Warning Model',
            'countries_covered': len(self.countries),
            'countries_list': self.countries,
            'fintech_brands_covered': len(self.fintech_brands),
            'fintech_brands_list': self.fintech_brands,
            'generated_on': datetime.now().isoformat(),
            'data_components': list(dataset.keys()),
            'total_records': sum(len(df) for df in dataset.values())
        }
        
        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved successfully!")
        print(f"Total records generated: {metadata['total_records']}")

def main():
    """Main function to generate and save the dataset"""
    generator = FinTechRiskNexusDataGenerator()
    
    # Generate 5 years of data
    dataset = generator.generate_complete_dataset(years=5)
    
    # Save dataset
    generator.save_dataset(dataset)
    
    # Display sample data
    print("\n" + "="*80)
    print("SAMPLE DATA PREVIEW")
    print("="*80)
    
    for name, df in dataset.items():
        if name != 'summary_statistics':
            print(f"\n{name.upper()} - First 3 records:")
            print(df.head(3).to_string(index=False))
            print(f"Total records: {len(df)}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(dataset['summary_statistics'].head(10).to_string(index=False))

if __name__ == "__main__":
    main()