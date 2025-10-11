"""
FinTech Early Warning Model Dataset Generator
Category 4: Nexus-Specific & Alternative Data for Sub-Saharan Africa
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FinTechRiskDataGenerator:
    """Generate synthetic FinTech risk dataset for Sub-Saharan African economies"""
    
    def __init__(self):
        # Sub-Saharan African countries with significant FinTech presence
        self.countries = [
            'Kenya', 'Nigeria', 'South Africa', 'Ghana', 'Uganda',
            'Tanzania', 'Rwanda', 'Senegal', 'Ivory Coast', 'Ethiopia',
            'Zambia', 'Zimbabwe', 'Mozambique', 'Cameroon', 'Angola'
        ]
        
        # Time period: 2018-2024 (quarterly data)
        self.start_date = pd.Timestamp('2018-01-01')
        self.end_date = pd.Timestamp('2024-09-30')
        self.date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='Q'
        )
        
        # Country-specific characteristics affecting FinTech development
        self.country_profiles = {
            'Kenya': {'tech_maturity': 0.85, 'market_size': 0.8, 'regulatory_strength': 0.9},
            'Nigeria': {'tech_maturity': 0.75, 'market_size': 0.95, 'regulatory_strength': 0.7},
            'South Africa': {'tech_maturity': 0.9, 'market_size': 0.85, 'regulatory_strength': 0.85},
            'Ghana': {'tech_maturity': 0.7, 'market_size': 0.65, 'regulatory_strength': 0.75},
            'Uganda': {'tech_maturity': 0.65, 'market_size': 0.6, 'regulatory_strength': 0.65},
            'Tanzania': {'tech_maturity': 0.6, 'market_size': 0.7, 'regulatory_strength': 0.6},
            'Rwanda': {'tech_maturity': 0.75, 'market_size': 0.45, 'regulatory_strength': 0.8},
            'Senegal': {'tech_maturity': 0.55, 'market_size': 0.5, 'regulatory_strength': 0.65},
            'Ivory Coast': {'tech_maturity': 0.5, 'market_size': 0.55, 'regulatory_strength': 0.6},
            'Ethiopia': {'tech_maturity': 0.45, 'market_size': 0.75, 'regulatory_strength': 0.5},
            'Zambia': {'tech_maturity': 0.5, 'market_size': 0.45, 'regulatory_strength': 0.55},
            'Zimbabwe': {'tech_maturity': 0.4, 'market_size': 0.4, 'regulatory_strength': 0.45},
            'Mozambique': {'tech_maturity': 0.35, 'market_size': 0.5, 'regulatory_strength': 0.4},
            'Cameroon': {'tech_maturity': 0.45, 'market_size': 0.55, 'regulatory_strength': 0.5},
            'Angola': {'tech_maturity': 0.4, 'market_size': 0.5, 'regulatory_strength': 0.45}
        }
        
    def generate_cyber_risk_exposure(self, country: str, date: pd.Timestamp) -> Dict:
        """Generate cyber risk exposure metrics"""
        profile = self.country_profiles[country]
        
        # Base cyber incidents influenced by tech maturity and time
        time_factor = (date - self.start_date).days / 365.25 / 6  # 0 to 1 over 6 years
        
        # More mature markets have more incidents initially but better improvement
        base_incidents = 10 * profile['tech_maturity'] * profile['market_size']
        
        # Quarterly trend with seasonal variations
        quarter = date.quarter
        seasonal_factor = 1.1 if quarter in [1, 4] else 0.9  # Higher in Q1 and Q4
        
        # Generate number of incidents with some randomness
        incidents = max(1, int(
            base_incidents * (1 + time_factor * 0.5) * seasonal_factor * 
            np.random.poisson(1.2)
        ))
        
        # Google search trends for mobile money fraud (0-100 scale)
        # Inversely related to regulatory strength
        base_search_trend = 100 * (1 - profile['regulatory_strength'] * 0.6)
        search_trend = min(100, max(0, 
            base_search_trend + np.random.normal(0, 15) + 
            time_factor * 10 * (1 - profile['regulatory_strength'])
        ))
        
        # Data breach severity index (1-10 scale)
        breach_severity = min(10, max(1,
            5 + (1 - profile['regulatory_strength']) * 3 + 
            np.random.normal(0, 1.5)
        ))
        
        # Phishing attempt frequency (per 1000 users)
        phishing_frequency = max(0.1,
            2 * (1 - profile['tech_maturity']) + 
            np.random.exponential(0.5)
        )
        
        return {
            'cyber_incidents_reported': incidents,
            'google_trend_mobile_fraud': round(search_trend, 1),
            'data_breach_severity_index': round(breach_severity, 2),
            'phishing_attempts_per_1000_users': round(phishing_frequency, 2)
        }
    
    def generate_consumer_sentiment(self, country: str, date: pd.Timestamp) -> Dict:
        """Generate consumer sentiment and trust metrics"""
        profile = self.country_profiles[country]
        
        # Base sentiment influenced by tech maturity and regulatory strength
        base_sentiment = 50 + (profile['tech_maturity'] + profile['regulatory_strength']) * 20
        
        # Add time trend (generally improving)
        time_factor = (date - self.start_date).days / 365.25 / 6
        sentiment_trend = time_factor * 10 * profile['regulatory_strength']
        
        # Social media sentiment score (-100 to +100)
        social_sentiment = min(100, max(-100,
            base_sentiment + sentiment_trend + np.random.normal(0, 10) - 50
        ))
        
        # Trust index (0-100)
        trust_index = min(100, max(0,
            base_sentiment + sentiment_trend + np.random.normal(0, 8)
        ))
        
        # Net Promoter Score (-100 to +100)
        nps = min(100, max(-100,
            (trust_index - 50) * 1.5 + np.random.normal(0, 15)
        ))
        
        # Customer complaint rate (per 1000 transactions)
        complaint_rate = max(0.1,
            10 * (1 - profile['regulatory_strength']) * (1 - time_factor * 0.3) +
            np.random.exponential(1)
        )
        
        return {
            'social_media_sentiment_score': round(social_sentiment, 1),
            'consumer_trust_index': round(trust_index, 1),
            'net_promoter_score': round(nps, 1),
            'customer_complaint_rate': round(complaint_rate, 2)
        }
    
    def generate_competitive_dynamics(self, country: str, date: pd.Timestamp) -> Dict:
        """Generate competitive dynamics metrics"""
        profile = self.country_profiles[country]
        
        # Market concentration (HHI)
        # Less mature markets tend to be more concentrated
        base_hhi = 3000 * (1 - profile['tech_maturity'] * 0.6)
        
        # Time trend - markets become less concentrated over time
        time_factor = (date - self.start_date).days / 365.25 / 6
        hhi_trend = -500 * time_factor * profile['regulatory_strength']
        
        hhi = min(10000, max(100,
            base_hhi + hhi_trend + np.random.normal(0, 200)
        ))
        
        # New FinTech licenses
        # More in larger, more regulated markets
        base_licenses = 3 * profile['market_size'] * profile['regulatory_strength']
        
        # Growing over time
        licenses_trend = time_factor * 5 * profile['tech_maturity']
        
        # Quarterly variation
        quarter = date.quarter
        quarterly_factor = 1.2 if quarter == 1 else 1.0  # More in Q1
        
        new_licenses = max(0, int(
            (base_licenses + licenses_trend) * quarterly_factor * 
            np.random.poisson(1.1)
        ))
        
        # Market entry rate (new entrants as % of existing)
        entry_rate = max(0,
            5 * profile['tech_maturity'] * (1 + time_factor) + 
            np.random.normal(0, 2)
        )
        
        # Innovation index (0-100)
        innovation_index = min(100, max(0,
            50 * profile['tech_maturity'] + 
            30 * time_factor + 
            np.random.normal(0, 10)
        ))
        
        return {
            'herfindahl_hirschman_index': round(hhi, 0),
            'new_fintech_licenses_issued': new_licenses,
            'market_entry_rate': round(entry_rate, 2),
            'innovation_index': round(innovation_index, 1)
        }
    
    def generate_risk_indicators(self, country: str, date: pd.Timestamp) -> Dict:
        """Generate additional risk indicators"""
        profile = self.country_profiles[country]
        
        # System interconnectedness score (0-100)
        interconnectedness = min(100, max(0,
            60 * profile['tech_maturity'] + 
            np.random.normal(0, 10)
        ))
        
        # Regulatory compliance score (0-100)
        compliance_score = min(100, max(0,
            70 * profile['regulatory_strength'] + 
            np.random.normal(0, 8)
        ))
        
        # Operational risk index (1-10)
        operational_risk = min(10, max(1,
            5 * (1 - profile['tech_maturity']) + 
            np.random.normal(0, 1.5)
        ))
        
        # Liquidity risk indicator (0-1)
        liquidity_risk = min(1, max(0,
            0.3 * (1 - profile['regulatory_strength']) + 
            np.random.normal(0, 0.1)
        ))
        
        return {
            'system_interconnectedness_score': round(interconnectedness, 1),
            'regulatory_compliance_score': round(compliance_score, 1),
            'operational_risk_index': round(operational_risk, 2),
            'liquidity_risk_indicator': round(liquidity_risk, 3)
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset"""
        data = []
        
        for country in self.countries:
            for date in self.date_range:
                record = {
                    'country': country,
                    'date': date,
                    'year': date.year,
                    'quarter': date.quarter,
                    'year_quarter': f"{date.year}-Q{date.quarter}"
                }
                
                # Add all metric categories
                record.update(self.generate_cyber_risk_exposure(country, date))
                record.update(self.generate_consumer_sentiment(country, date))
                record.update(self.generate_competitive_dynamics(country, date))
                record.update(self.generate_risk_indicators(country, date))
                
                # Add country profile metrics
                profile = self.country_profiles[country]
                record['tech_maturity_score'] = round(profile['tech_maturity'] * 100, 1)
                record['market_size_score'] = round(profile['market_size'] * 100, 1)
                record['regulatory_strength_score'] = round(profile['regulatory_strength'] * 100, 1)
                
                data.append(record)
        
        return pd.DataFrame(data)
    
    def add_calculated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated and composite metrics"""
        # Composite Risk Score (weighted average of various risks)
        df['composite_cyber_risk'] = (
            df['cyber_incidents_reported'] * 2 + 
            df['google_trend_mobile_fraud'] / 10 + 
            df['data_breach_severity_index'] * 10 +
            df['phishing_attempts_per_1000_users'] * 20
        ) / 4
        
        # Market Health Score
        df['market_health_score'] = (
            df['consumer_trust_index'] + 
            (100 - df['herfindahl_hirschman_index'] / 100) + 
            df['innovation_index'] +
            df['regulatory_compliance_score']
        ) / 4
        
        # Risk-Adjusted Growth Potential
        df['risk_adjusted_growth_potential'] = (
            df['market_size_score'] * df['tech_maturity_score'] / 100 * 
            (1 - df['operational_risk_index'] / 10) * 
            (1 - df['liquidity_risk_indicator'])
        )
        
        # Year-over-year changes for key metrics
        for col in ['cyber_incidents_reported', 'consumer_trust_index', 
                    'herfindahl_hirschman_index', 'new_fintech_licenses_issued']:
            df[f'{col}_yoy_change'] = df.groupby('country')[col].pct_change(4) * 100
        
        return df
    
    def generate_metadata(self, df: pd.DataFrame) -> Dict:
        """Generate metadata for the dataset"""
        metadata = {
            'dataset_name': 'FinTech Risk Nexus Dataset - Sub-Saharan Africa',
            'category': 'Category 4: Nexus-Specific & Alternative Data',
            'description': 'Comprehensive dataset capturing interconnected FinTech risks in Sub-Saharan African economies',
            'time_period': f"{self.start_date.date()} to {self.end_date.date()}",
            'frequency': 'Quarterly',
            'countries': self.countries,
            'total_records': len(df),
            'generation_date': datetime.now().isoformat(),
            'variables': {
                'Cyber Risk Exposure': [
                    'cyber_incidents_reported',
                    'google_trend_mobile_fraud',
                    'data_breach_severity_index',
                    'phishing_attempts_per_1000_users'
                ],
                'Consumer Sentiment & Trust': [
                    'social_media_sentiment_score',
                    'consumer_trust_index',
                    'net_promoter_score',
                    'customer_complaint_rate'
                ],
                'Competitive Dynamics': [
                    'herfindahl_hirschman_index',
                    'new_fintech_licenses_issued',
                    'market_entry_rate',
                    'innovation_index'
                ],
                'Risk Indicators': [
                    'system_interconnectedness_score',
                    'regulatory_compliance_score',
                    'operational_risk_index',
                    'liquidity_risk_indicator'
                ],
                'Composite Metrics': [
                    'composite_cyber_risk',
                    'market_health_score',
                    'risk_adjusted_growth_potential'
                ]
            },
            'data_sources_note': 'This is synthetic data generated for research purposes',
            'variable_descriptions': {
                'cyber_incidents_reported': 'Number of cybersecurity incidents reported in the financial sector per quarter',
                'google_trend_mobile_fraud': 'Google search trend index (0-100) for "mobile money fraud" and related terms',
                'data_breach_severity_index': 'Severity index of data breaches (1-10 scale)',
                'phishing_attempts_per_1000_users': 'Frequency of phishing attempts per 1000 FinTech users',
                'social_media_sentiment_score': 'Aggregate sentiment from social media analysis (-100 to +100)',
                'consumer_trust_index': 'Consumer trust in FinTech services (0-100)',
                'net_promoter_score': 'NPS for major FinTech brands (-100 to +100)',
                'customer_complaint_rate': 'Customer complaints per 1000 transactions',
                'herfindahl_hirschman_index': 'Market concentration measure (0-10000, higher = more concentrated)',
                'new_fintech_licenses_issued': 'Number of new FinTech licenses issued per quarter',
                'market_entry_rate': 'New entrants as percentage of existing players',
                'innovation_index': 'FinTech innovation index (0-100)',
                'system_interconnectedness_score': 'Degree of system interconnection (0-100)',
                'regulatory_compliance_score': 'Regulatory compliance level (0-100)',
                'operational_risk_index': 'Operational risk level (1-10)',
                'liquidity_risk_indicator': 'Liquidity risk measure (0-1)',
                'composite_cyber_risk': 'Weighted composite of cyber risk factors',
                'market_health_score': 'Overall market health indicator',
                'risk_adjusted_growth_potential': 'Growth potential adjusted for risk factors'
            }
        }
        return metadata

def main():
    """Main function to generate and save the dataset"""
    print("=" * 80)
    print("FinTech Risk Nexus Dataset Generator")
    print("Category 4: Nexus-Specific & Alternative Data")
    print("Sub-Saharan Africa Economies")
    print("=" * 80)
    print()
    
    # Initialize generator
    generator = FinTechRiskDataGenerator()
    
    # Generate base dataset
    print("Generating base dataset...")
    df = generator.generate_dataset()
    
    # Add calculated metrics
    print("Adding calculated metrics...")
    df = generator.add_calculated_metrics(df)
    
    # Generate metadata
    print("Generating metadata...")
    metadata = generator.generate_metadata(df)
    
    # Save to CSV
    csv_file = 'fintech_risk_nexus_dataset.csv'
    print(f"Saving dataset to {csv_file}...")
    df.to_csv(csv_file, index=False)
    
    # Save to Excel with multiple sheets
    excel_file = 'fintech_risk_nexus_dataset.xlsx'
    print(f"Saving dataset to {excel_file}...")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Main data
        df.to_excel(writer, sheet_name='Main_Data', index=False)
        
        # Summary statistics by country
        summary = df.groupby('country').agg({
            'cyber_incidents_reported': ['mean', 'std', 'min', 'max'],
            'consumer_trust_index': ['mean', 'std', 'min', 'max'],
            'herfindahl_hirschman_index': ['mean', 'std', 'min', 'max'],
            'new_fintech_licenses_issued': ['sum', 'mean'],
            'composite_cyber_risk': ['mean', 'std'],
            'market_health_score': ['mean', 'std']
        }).round(2)
        summary.to_excel(writer, sheet_name='Country_Summary')
        
        # Time series summary
        time_summary = df.groupby('year_quarter').agg({
            'cyber_incidents_reported': 'mean',
            'consumer_trust_index': 'mean',
            'new_fintech_licenses_issued': 'sum',
            'market_health_score': 'mean'
        }).round(2)
        time_summary.to_excel(writer, sheet_name='Time_Series_Summary')
        
        # Metadata
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    # Save to JSON
    json_file = 'fintech_risk_nexus_dataset.json'
    print(f"Saving dataset to {json_file}...")
    df_json = df.copy()
    df_json['date'] = df_json['date'].astype(str)
    df_json.to_json(json_file, orient='records', indent=2)
    
    # Save metadata separately
    metadata_file = 'dataset_metadata.json'
    print(f"Saving metadata to {metadata_file}...")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary statistics
    print()
    print("=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print()
    print("Dataset Summary:")
    print(f"- Total records: {len(df)}")
    print(f"- Countries: {len(df['country'].unique())}")
    print(f"- Time period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"- Variables: {len(df.columns)} columns")
    print()
    print("Files created:")
    print(f"1. {csv_file} - Main dataset in CSV format")
    print(f"2. {excel_file} - Dataset with summary sheets in Excel format")
    print(f"3. {json_file} - Dataset in JSON format")
    print(f"4. {metadata_file} - Dataset metadata and documentation")
    print()
    print("Sample of the dataset:")
    print(df.head(10))
    print()
    print("Basic statistics for key metrics:")
    print(df[['cyber_incidents_reported', 'consumer_trust_index', 
              'herfindahl_hirschman_index', 'new_fintech_licenses_issued',
              'composite_cyber_risk', 'market_health_score']].describe().round(2))
    
    return df, metadata

if __name__ == "__main__":
    df, metadata = main()