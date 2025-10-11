#!/usr/bin/env python3
"""
FinTech Early Warning Model Dataset Generator
Category 4: Nexus-Specific & Alternative Data for Sub-Saharan Africa

This script generates comprehensive datasets for FinTech risk analysis focusing on:
1. Cyber Risk Exposure
2. Consumer Sentiment & Trust
3. Competitive Dynamics

Author: Research Assistant
Date: 2025-10-11
Purpose: Thesis Research on FinTech Early Warning Model in Sub-Sahara Africa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from faker import Faker
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

class FinTechRiskDataGenerator:
    """
    Comprehensive FinTech Risk Dataset Generator for Sub-Saharan Africa
    """
    
    def __init__(self):
        # Sub-Saharan African countries with significant FinTech presence
        self.countries = [
            'Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Uganda', 
            'Tanzania', 'Rwanda', 'Zambia', 'Botswana', 'Ethiopia',
            'Senegal', 'Ivory Coast', 'Mali', 'Burkina Faso', 'Cameroon'
        ]
        
        # Major FinTech brands in Sub-Saharan Africa
        self.fintech_brands = [
            'M-Pesa', 'Flutterwave', 'Paystack', 'Interswitch', 'Paga',
            'Tala', 'Branch', 'FairMoney', 'PalmPay', 'OPay', 'Kuda',
            'Carbon', 'Cowrywise', 'PiggyVest', 'MTN MoMo', 'Airtel Money',
            'Ecobank Xpress Account', 'Access Bank', 'Equity Bank', 'KCB'
        ]
        
        # Cybersecurity incident types
        self.incident_types = [
            'Mobile Money Fraud', 'SIM Swap Attack', 'Phishing Attack',
            'Data Breach', 'API Vulnerability', 'Social Engineering',
            'Malware Attack', 'DDoS Attack', 'Account Takeover',
            'Transaction Fraud', 'Identity Theft', 'Card Skimming'
        ]
        
        # Date range for the dataset (5 years of data)
        self.start_date = datetime(2019, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        print("FinTech Risk Data Generator initialized for Sub-Saharan Africa")
        print(f"Countries: {len(self.countries)}")
        print(f"FinTech Brands: {len(self.fintech_brands)}")
        print(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
    
    def generate_date_range(self, freq='M'):
        """Generate date range for the dataset"""
        return pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
    
    def generate_cyber_risk_data(self):
        """
        Generate Cyber Risk Exposure Data
        
        Variables:
        - Number of cybersecurity incidents reported in financial sector
        - Google search trends for mobile money fraud terms
        - Incident severity scores
        - Geographic distribution of incidents
        """
        print("\nGenerating Cyber Risk Exposure Data...")
        
        dates = self.generate_date_range('M')  # Monthly data
        cyber_data = []
        
        for date in dates:
            for country in self.countries:
                # Base incident rate varies by country development level
                country_risk_multiplier = self._get_country_risk_multiplier(country)
                
                # Seasonal trends (higher incidents during holiday seasons)
                seasonal_factor = 1.0
                if date.month in [11, 12, 1]:  # Holiday season
                    seasonal_factor = 1.4
                elif date.month in [6, 7, 8]:  # Mid-year
                    seasonal_factor = 1.2
                
                # COVID-19 impact (increased digital adoption = more cyber risk)
                covid_factor = 1.0
                if date.year >= 2020:
                    covid_factor = 1.6 if date.year == 2020 else 1.3
                
                # Generate incident data
                base_incidents = np.random.poisson(5) * country_risk_multiplier
                total_incidents = int(base_incidents * seasonal_factor * covid_factor)
                
                # Distribute incidents by type
                incident_distribution = np.random.multinomial(
                    total_incidents, 
                    [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]
                )
                
                # Google search trends (normalized 0-100)
                search_volume_base = np.random.normal(30, 10)
                search_volume = max(0, min(100, search_volume_base * seasonal_factor * covid_factor))
                
                # Generate search trends for different fraud-related terms
                fraud_terms = {
                    'mobile_money_fraud': max(0, min(100, np.random.normal(25, 8) * seasonal_factor)),
                    'sim_swap_fraud': max(0, min(100, np.random.normal(15, 5) * seasonal_factor)),
                    'fintech_scam': max(0, min(100, np.random.normal(20, 7) * seasonal_factor)),
                    'digital_payment_security': max(0, min(100, np.random.normal(35, 12) * seasonal_factor)),
                    'mobile_banking_safety': max(0, min(100, np.random.normal(40, 10) * seasonal_factor))
                }
                
                # Calculate risk scores
                incident_severity_score = np.random.normal(6.5, 1.5)  # 1-10 scale
                incident_severity_score = max(1, min(10, incident_severity_score))
                
                cyber_record = {
                    'date': date,
                    'country': country,
                    'total_cyber_incidents': total_incidents,
                    'mobile_money_fraud_incidents': incident_distribution[0],
                    'sim_swap_incidents': incident_distribution[1],
                    'phishing_incidents': incident_distribution[2],
                    'data_breach_incidents': incident_distribution[3],
                    'api_vulnerability_incidents': incident_distribution[4],
                    'social_engineering_incidents': incident_distribution[5],
                    'malware_incidents': incident_distribution[6],
                    'ddos_incidents': incident_distribution[7],
                    'account_takeover_incidents': incident_distribution[8],
                    'transaction_fraud_incidents': incident_distribution[9],
                    'identity_theft_incidents': incident_distribution[10],
                    'card_skimming_incidents': incident_distribution[11],
                    'search_mobile_money_fraud': fraud_terms['mobile_money_fraud'],
                    'search_sim_swap_fraud': fraud_terms['sim_swap_fraud'],
                    'search_fintech_scam': fraud_terms['fintech_scam'],
                    'search_digital_payment_security': fraud_terms['digital_payment_security'],
                    'search_mobile_banking_safety': fraud_terms['mobile_banking_safety'],
                    'avg_incident_severity_score': round(incident_severity_score, 2),
                    'cyber_risk_index': round((total_incidents * incident_severity_score) / 10, 2)
                }
                
                cyber_data.append(cyber_record)
        
        cyber_df = pd.DataFrame(cyber_data)
        print(f"Generated {len(cyber_df)} cyber risk records")
        return cyber_df
    
    def generate_sentiment_data(self):
        """
        Generate Consumer Sentiment & Trust Data
        
        Variables:
        - Social media sentiment analysis for major FinTech brands
        - Trust index scores
        - Consumer confidence metrics
        - Brand perception scores
        """
        print("\nGenerating Consumer Sentiment & Trust Data...")
        
        dates = self.generate_date_range('W')  # Weekly data for sentiment
        sentiment_data = []
        
        for date in dates:
            for country in self.countries:
                for brand in self.fintech_brands:
                    # Check if brand operates in this country (realistic distribution)
                    if not self._brand_operates_in_country(brand, country):
                        continue
                    
                    # Base sentiment varies by brand maturity and country
                    brand_maturity = self._get_brand_maturity_score(brand)
                    country_digital_literacy = self._get_country_digital_literacy(country)
                    
                    # Generate sentiment scores (-1 to 1, where -1 is very negative, 1 is very positive)
                    base_sentiment = np.random.normal(0.1, 0.3)  # Slightly positive overall
                    
                    # Adjust for brand and country factors
                    adjusted_sentiment = base_sentiment + (brand_maturity * 0.2) + (country_digital_literacy * 0.15)
                    sentiment_score = max(-1, min(1, adjusted_sentiment))
                    
                    # Generate social media metrics
                    mention_volume = max(0, int(np.random.exponential(50) * self._get_brand_popularity(brand, country)))
                    positive_mentions = int(mention_volume * max(0, (sentiment_score + 1) / 2))
                    negative_mentions = int(mention_volume * max(0, (1 - sentiment_score) / 2))
                    neutral_mentions = mention_volume - positive_mentions - negative_mentions
                    
                    # Trust metrics (0-100 scale)
                    trust_index = max(0, min(100, np.random.normal(65, 15) + (sentiment_score * 20)))
                    security_perception = max(0, min(100, np.random.normal(60, 12) + (sentiment_score * 15)))
                    ease_of_use_score = max(0, min(100, np.random.normal(70, 10) + (brand_maturity * 20)))
                    customer_support_score = max(0, min(100, np.random.normal(55, 18) + (sentiment_score * 25)))
                    
                    # Brand perception categories
                    brand_categories = self._categorize_brand_perception(sentiment_score, trust_index)
                    
                    sentiment_record = {
                        'date': date,
                        'country': country,
                        'fintech_brand': brand,
                        'sentiment_score': round(sentiment_score, 3),
                        'total_mentions': mention_volume,
                        'positive_mentions': positive_mentions,
                        'negative_mentions': negative_mentions,
                        'neutral_mentions': neutral_mentions,
                        'trust_index': round(trust_index, 2),
                        'security_perception_score': round(security_perception, 2),
                        'ease_of_use_score': round(ease_of_use_score, 2),
                        'customer_support_score': round(customer_support_score, 2),
                        'brand_perception_category': brand_categories['category'],
                        'risk_perception_level': brand_categories['risk_level'],
                        'recommendation_likelihood': round(max(0, min(100, trust_index + (sentiment_score * 20))), 2)
                    }
                    
                    sentiment_data.append(sentiment_record)
        
        sentiment_df = pd.DataFrame(sentiment_data)
        print(f"Generated {len(sentiment_df)} sentiment records")
        return sentiment_df
    
    def generate_competitive_dynamics_data(self):
        """
        Generate Competitive Dynamics Data
        
        Variables:
        - Herfindahl-Hirschman Index (HHI) for FinTech market
        - Number of new FinTech licenses issued per year
        - Market concentration metrics
        - Entry and exit rates
        """
        print("\nGenerating Competitive Dynamics Data...")
        
        dates = self.generate_date_range('Q')  # Quarterly data
        competitive_data = []
        
        for date in dates:
            for country in self.countries:
                # Market share distribution (varies by country maturity)
                market_maturity = self._get_market_maturity(country)
                
                # Generate market shares for active brands in this country
                active_brands = [b for b in self.fintech_brands if self._brand_operates_in_country(b, country)]
                
                if len(active_brands) == 0:
                    continue
                
                # Generate realistic market share distribution
                market_shares = self._generate_market_shares(active_brands, market_maturity)
                
                # Calculate HHI (sum of squared market shares * 10000)
                hhi = sum(share**2 for share in market_shares.values()) * 10000
                
                # Market concentration categories
                if hhi < 1500:
                    concentration_level = "Competitive"
                elif hhi < 2500:
                    concentration_level = "Moderately Concentrated"
                else:
                    concentration_level = "Highly Concentrated"
                
                # New licenses issued (varies by regulatory environment)
                regulatory_efficiency = self._get_regulatory_efficiency(country)
                base_licenses = np.random.poisson(2)
                new_licenses = max(0, int(base_licenses * regulatory_efficiency))
                
                # Market entry/exit dynamics
                market_entries = max(0, np.random.poisson(1) if market_maturity < 0.7 else np.random.poisson(0.5))
                market_exits = max(0, np.random.poisson(0.5) if hhi > 2000 else np.random.poisson(0.2))
                
                # Innovation metrics
                innovation_index = np.random.normal(50, 15) + (market_maturity * 30)
                innovation_index = max(0, min(100, innovation_index))
                
                # Regulatory metrics
                regulatory_clarity_score = np.random.normal(60, 20) + (regulatory_efficiency * 40)
                regulatory_clarity_score = max(0, min(100, regulatory_clarity_score))
                
                competitive_record = {
                    'date': date,
                    'country': country,
                    'herfindahl_hirschman_index': round(hhi, 2),
                    'market_concentration_level': concentration_level,
                    'number_of_active_fintech_companies': len(active_brands),
                    'new_fintech_licenses_issued': new_licenses,
                    'market_entries_quarter': market_entries,
                    'market_exits_quarter': market_exits,
                    'net_market_change': market_entries - market_exits,
                    'market_leader_share': round(max(market_shares.values()) * 100, 2),
                    'top_3_market_share': round(sum(sorted(market_shares.values(), reverse=True)[:3]) * 100, 2),
                    'innovation_index': round(innovation_index, 2),
                    'regulatory_clarity_score': round(regulatory_clarity_score, 2),
                    'market_maturity_score': round(market_maturity * 100, 2),
                    'competitive_intensity': round((1 - (hhi / 10000)) * 100, 2)
                }
                
                competitive_data.append(competitive_record)
        
        competitive_df = pd.DataFrame(competitive_data)
        print(f"Generated {len(competitive_df)} competitive dynamics records")
        return competitive_df
    
    def _get_country_risk_multiplier(self, country):
        """Get risk multiplier based on country characteristics"""
        risk_factors = {
            'Nigeria': 2.5, 'Kenya': 1.8, 'South Africa': 1.5, 'Ghana': 1.7,
            'Uganda': 1.9, 'Tanzania': 1.8, 'Rwanda': 1.3, 'Zambia': 1.6,
            'Botswana': 1.2, 'Ethiopia': 2.0, 'Senegal': 1.4, 'Ivory Coast': 1.6,
            'Mali': 1.8, 'Burkina Faso': 1.9, 'Cameroon': 1.7
        }
        return risk_factors.get(country, 1.5)
    
    def _brand_operates_in_country(self, brand, country):
        """Determine if a brand operates in a specific country"""
        # Simplified logic - some brands are regional, others are country-specific
        regional_brands = ['M-Pesa', 'MTN MoMo', 'Airtel Money', 'Ecobank Xpress Account']
        nigerian_brands = ['Flutterwave', 'Paystack', 'Interswitch', 'Paga', 'FairMoney', 'PalmPay', 'OPay', 'Kuda', 'Carbon', 'Cowrywise', 'PiggyVest']
        kenyan_brands = ['M-Pesa', 'Equity Bank', 'KCB']
        
        if brand in regional_brands:
            return True
        elif brand in nigerian_brands:
            return country in ['Nigeria', 'Ghana', 'Kenya', 'South Africa']
        elif brand in kenyan_brands:
            return country in ['Kenya', 'Uganda', 'Tanzania', 'Rwanda']
        else:
            return np.random.choice([True, False], p=[0.3, 0.7])  # 30% chance for other brands
    
    def _get_brand_maturity_score(self, brand):
        """Get brand maturity score (0-1)"""
        mature_brands = ['M-Pesa', 'Interswitch', 'Equity Bank', 'KCB', 'Access Bank']
        if brand in mature_brands:
            return np.random.uniform(0.7, 1.0)
        else:
            return np.random.uniform(0.3, 0.8)
    
    def _get_country_digital_literacy(self, country):
        """Get country digital literacy score (0-1)"""
        literacy_scores = {
            'South Africa': 0.8, 'Kenya': 0.7, 'Nigeria': 0.6, 'Ghana': 0.6,
            'Rwanda': 0.7, 'Botswana': 0.7, 'Uganda': 0.5, 'Tanzania': 0.5,
            'Zambia': 0.4, 'Ethiopia': 0.4, 'Senegal': 0.5, 'Ivory Coast': 0.5,
            'Mali': 0.3, 'Burkina Faso': 0.3, 'Cameroon': 0.5
        }
        return literacy_scores.get(country, 0.5)
    
    def _get_brand_popularity(self, brand, country):
        """Get brand popularity multiplier"""
        if brand == 'M-Pesa' and country == 'Kenya':
            return 3.0
        elif brand in ['Flutterwave', 'Paystack'] and country == 'Nigeria':
            return 2.5
        elif brand == 'MTN MoMo':
            return 2.0
        else:
            return np.random.uniform(0.5, 1.5)
    
    def _categorize_brand_perception(self, sentiment_score, trust_index):
        """Categorize brand perception"""
        if sentiment_score > 0.3 and trust_index > 70:
            return {'category': 'Highly Trusted', 'risk_level': 'Low'}
        elif sentiment_score > 0 and trust_index > 50:
            return {'category': 'Moderately Trusted', 'risk_level': 'Medium'}
        elif sentiment_score > -0.3 and trust_index > 30:
            return {'category': 'Neutral', 'risk_level': 'Medium'}
        else:
            return {'category': 'Distrusted', 'risk_level': 'High'}
    
    def _get_market_maturity(self, country):
        """Get market maturity score (0-1)"""
        maturity_scores = {
            'Kenya': 0.9, 'South Africa': 0.8, 'Nigeria': 0.7, 'Ghana': 0.6,
            'Rwanda': 0.6, 'Uganda': 0.5, 'Tanzania': 0.5, 'Botswana': 0.6,
            'Zambia': 0.4, 'Ethiopia': 0.4, 'Senegal': 0.4, 'Ivory Coast': 0.4,
            'Mali': 0.3, 'Burkina Faso': 0.3, 'Cameroon': 0.4
        }
        return maturity_scores.get(country, 0.4)
    
    def _generate_market_shares(self, brands, maturity):
        """Generate realistic market share distribution"""
        if maturity > 0.7:  # Mature market - more concentrated
            shares = np.random.dirichlet([3, 2, 1] + [0.5] * (len(brands) - 3))
        else:  # Emerging market - more fragmented
            shares = np.random.dirichlet([1] * len(brands))
        
        return dict(zip(brands, shares))
    
    def _get_regulatory_efficiency(self, country):
        """Get regulatory efficiency score (0-1)"""
        efficiency_scores = {
            'Rwanda': 0.9, 'Botswana': 0.8, 'South Africa': 0.7, 'Kenya': 0.7,
            'Ghana': 0.6, 'Nigeria': 0.5, 'Uganda': 0.5, 'Tanzania': 0.5,
            'Zambia': 0.4, 'Ethiopia': 0.4, 'Senegal': 0.5, 'Ivory Coast': 0.4,
            'Mali': 0.3, 'Burkina Faso': 0.3, 'Cameroon': 0.4
        }
        return efficiency_scores.get(country, 0.4)
    
    def generate_comprehensive_dataset(self):
        """Generate the complete dataset with all categories"""
        print("="*60)
        print("GENERATING COMPREHENSIVE FINTECH RISK NEXUS DATASET")
        print("="*60)
        
        # Generate all data categories
        cyber_df = self.generate_cyber_risk_data()
        sentiment_df = self.generate_sentiment_data()
        competitive_df = self.generate_competitive_dynamics_data()
        
        # Save individual datasets
        cyber_df.to_csv('cyber_risk_exposure_data.csv', index=False)
        sentiment_df.to_csv('consumer_sentiment_trust_data.csv', index=False)
        competitive_df.to_csv('competitive_dynamics_data.csv', index=False)
        
        print(f"\nDataset Generation Complete!")
        print(f"- Cyber Risk Records: {len(cyber_df):,}")
        print(f"- Sentiment Records: {len(sentiment_df):,}")
        print(f"- Competitive Records: {len(competitive_df):,}")
        print(f"- Total Records: {len(cyber_df) + len(sentiment_df) + len(competitive_df):,}")
        
        return {
            'cyber_risk': cyber_df,
            'sentiment_trust': sentiment_df,
            'competitive_dynamics': competitive_df
        }

def main():
    """Main function to generate the dataset"""
    generator = FinTechRiskDataGenerator()
    datasets = generator.generate_comprehensive_dataset()
    
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"Shape: {df.shape}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"Countries: {df['country'].nunique()}")
        print("Sample columns:", list(df.columns)[:5])

if __name__ == "__main__":
    main()