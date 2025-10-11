#!/usr/bin/env python3
"""
Synthetic Data Generator for SSA FinTech Early Warning Model
Generates realistic synthetic data to complement the real World Bank data,
particularly for indicators with low data availability.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats

class SSASyntheticDataGenerator:
    def __init__(self):
        self.countries = {
            'KE': 'Kenya', 'NG': 'Nigeria', 'ZA': 'South Africa', 'GH': 'Ghana',
            'UG': 'Uganda', 'TZ': 'Tanzania', 'RW': 'Rwanda', 'SN': 'Senegal',
            'CI': 'Cote d\'Ivoire', 'ZM': 'Zambia', 'BW': 'Botswana', 'MW': 'Malawi',
            'MZ': 'Mozambique', 'ET': 'Ethiopia', 'ZW': 'Zimbabwe', 'CM': 'Cameroon',
            'BF': 'Burkina Faso', 'ML': 'Mali', 'BJ': 'Benin', 'TG': 'Togo'
        }
        
        self.years = list(range(2010, 2024))
        
        # Load real data for reference
        try:
            self.real_data = pd.read_csv('processed_data/ssa_macro_data_enhanced.csv')
        except:
            self.real_data = pd.read_csv('processed_data/ssa_macro_data_simple.csv')
    
    def generate_fintech_specific_indicators(self):
        """Generate FinTech-specific indicators not available in World Bank data."""
        
        synthetic_data = []
        
        for country_code, country_name in self.countries.items():
            # Get country's real data for correlation
            country_real = self.real_data[self.real_data['country_code'] == country_code]
            
            for year in self.years:
                year_real = country_real[country_real['year'] == year]
                
                # Base development level (affects all indicators)
                if not year_real.empty and 'internet_users' in year_real.columns:
                    base_development = year_real['internet_users'].iloc[0] / 100 if not pd.isna(year_real['internet_users'].iloc[0]) else 0.3
                else:
                    base_development = np.random.uniform(0.1, 0.8)
                
                # FinTech Adoption Rate (% of population using FinTech services)
                fintech_adoption = np.random.beta(2, 5) * base_development * 100
                fintech_adoption = np.clip(fintech_adoption, 0.5, 85)
                
                # Mobile Money Penetration (% of adults with mobile money accounts)
                mobile_money_penetration = np.random.beta(3, 2) * base_development * 120
                mobile_money_penetration = np.clip(mobile_money_penetration, 2, 95)
                
                # Digital Payment Volume (as % of GDP)
                digital_payment_volume = np.random.lognormal(np.log(base_development * 50), 0.5)
                digital_payment_volume = np.clip(digital_payment_volume, 1, 200)
                
                # FinTech Regulatory Score (0-100, higher is better)
                regulatory_score = np.random.beta(4, 3) * 100
                regulatory_score = np.clip(regulatory_score, 20, 95)
                
                # Cybersecurity Incidents (per 100,000 digital transactions)
                cyber_incidents = np.random.exponential(5) * (1 - base_development)
                cyber_incidents = np.clip(cyber_incidents, 0.1, 50)
                
                # Financial Inclusion Gap (% of adults without access to formal financial services)
                financial_exclusion = np.random.beta(5, 2) * (1 - base_development) * 100
                financial_exclusion = np.clip(financial_exclusion, 5, 85)
                
                # FinTech Investment (USD millions)
                fintech_investment = np.random.lognormal(np.log(base_development * 100 + 1), 1)
                fintech_investment = np.clip(fintech_investment, 0.1, 1000)
                
                # Digital Literacy Rate (% of population with basic digital skills)
                digital_literacy = base_development * 100 + np.random.normal(0, 10)
                digital_literacy = np.clip(digital_literacy, 10, 95)
                
                # Cross-border Payment Costs (% of transaction value)
                cross_border_costs = np.random.lognormal(np.log(8), 0.3) * (1.5 - base_development)
                cross_border_costs = np.clip(cross_border_costs, 2, 25)
                
                # Banking Sector Concentration (HHI index)
                banking_concentration = np.random.uniform(0.15, 0.8)
                
                synthetic_data.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'year': year,
                    'fintech_adoption_rate': round(fintech_adoption, 2),
                    'mobile_money_penetration': round(mobile_money_penetration, 2),
                    'digital_payment_volume_gdp': round(digital_payment_volume, 2),
                    'fintech_regulatory_score': round(regulatory_score, 1),
                    'cybersecurity_incidents_per_100k': round(cyber_incidents, 2),
                    'financial_exclusion_rate': round(financial_exclusion, 1),
                    'fintech_investment_usd_millions': round(fintech_investment, 1),
                    'digital_literacy_rate': round(digital_literacy, 1),
                    'cross_border_payment_costs_pct': round(cross_border_costs, 2),
                    'banking_sector_concentration_hhi': round(banking_concentration, 3)
                })
        
        return pd.DataFrame(synthetic_data)
    
    def generate_missing_real_indicators(self):
        """Generate synthetic data for real indicators with low availability."""
        
        synthetic_data = []
        
        for country_code, country_name in self.countries.items():
            country_real = self.real_data[self.real_data['country_code'] == country_code]
            
            for year in self.years:
                year_real = country_real[country_real['year'] == year]
                
                # Base economic conditions
                if not year_real.empty and 'gdp_growth' in year_real.columns:
                    base_gdp_growth = year_real['gdp_growth'].iloc[0] if not pd.isna(year_real['gdp_growth'].iloc[0]) else 4.0
                else:
                    base_gdp_growth = np.random.normal(4, 3)
                
                # Central Bank Policy Rate (if missing)
                if year_real.empty or 'interest_rate' not in year_real.columns or pd.isna(year_real.get('interest_rate', [np.nan]).iloc[0]):
                    policy_rate = max(0.5, np.random.normal(8, 3) + (5 - base_gdp_growth) * 0.5)
                    policy_rate = np.clip(policy_rate, 0.5, 25)
                else:
                    policy_rate = None
                
                # Government Debt to GDP (if missing)
                if year_real.empty or 'debt_gdp' not in year_real.columns or pd.isna(year_real.get('debt_gdp', [np.nan]).iloc[0]):
                    debt_gdp = np.random.lognormal(np.log(40), 0.4)
                    debt_gdp = np.clip(debt_gdp, 15, 120)
                else:
                    debt_gdp = None
                
                # Current Account Balance (% of GDP)
                current_account = np.random.normal(-3, 4)
                current_account = np.clip(current_account, -15, 10)
                
                # Foreign Exchange Reserves (months of imports)
                fx_reserves = np.random.lognormal(np.log(4), 0.5)
                fx_reserves = np.clip(fx_reserves, 1, 12)
                
                # Credit Rating Score (0-100)
                credit_rating = np.random.beta(3, 4) * 100
                credit_rating = np.clip(credit_rating, 20, 85)
                
                # Political Stability Index (-2.5 to 2.5)
                political_stability = np.random.normal(0, 0.8)
                political_stability = np.clip(political_stability, -2.5, 2.5)
                
                # Ease of Doing Business Rank (1-190)
                business_rank = np.random.uniform(50, 180)
                
                synthetic_data.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'year': year,
                    'central_bank_policy_rate': round(policy_rate, 2) if policy_rate else None,
                    'government_debt_gdp': round(debt_gdp, 1) if debt_gdp else None,
                    'current_account_balance_gdp': round(current_account, 2),
                    'fx_reserves_months_imports': round(fx_reserves, 1),
                    'credit_rating_score': round(credit_rating, 1),
                    'political_stability_index': round(political_stability, 2),
                    'ease_doing_business_rank': round(business_rank, 0)
                })
        
        return pd.DataFrame(synthetic_data)
    
    def create_comprehensive_dataset(self):
        """Combine real and synthetic data into a comprehensive dataset."""
        
        print("Generating FinTech-specific indicators...")
        fintech_data = self.generate_fintech_specific_indicators()
        
        print("Generating missing economic indicators...")
        missing_data = self.generate_missing_real_indicators()
        
        # Merge with real data
        print("Merging with real data...")
        
        # Start with real data
        comprehensive_data = self.real_data.copy()
        
        # Add synthetic FinTech indicators
        comprehensive_data = pd.merge(
            comprehensive_data,
            fintech_data,
            on=['country_code', 'country_name', 'year'],
            how='left'
        )
        
        # Add synthetic missing indicators
        comprehensive_data = pd.merge(
            comprehensive_data,
            missing_data,
            on=['country_code', 'country_name', 'year'],
            how='left'
        )
        
        return comprehensive_data
    
    def generate_crisis_scenarios(self):
        """Generate synthetic crisis scenarios for stress testing."""
        
        scenarios = {
            'global_financial_crisis': {
                'description': 'Global financial crisis scenario',
                'gdp_growth_shock': -5,
                'inflation_shock': 2,
                'exchange_rate_shock': 15,
                'fintech_adoption_impact': -10,
                'years': [2020, 2021]
            },
            'commodity_price_shock': {
                'description': 'Commodity price collapse',
                'gdp_growth_shock': -3,
                'inflation_shock': -2,
                'exchange_rate_shock': 20,
                'fintech_adoption_impact': 5,  # People turn to digital solutions
                'years': [2022]
            },
            'cyber_attack_scenario': {
                'description': 'Major cybersecurity incident',
                'gdp_growth_shock': -0.5,
                'inflation_shock': 0.5,
                'exchange_rate_shock': 5,
                'fintech_adoption_impact': -20,
                'cybersecurity_incidents_multiplier': 10,
                'years': [2023]
            },
            'regulatory_crackdown': {
                'description': 'Strict FinTech regulation implementation',
                'gdp_growth_shock': -1,
                'fintech_adoption_impact': -15,
                'fintech_regulatory_score_impact': -30,
                'years': [2021, 2022]
            }
        }
        
        return scenarios

def main():
    """Main execution function."""
    print("Starting synthetic data generation...")
    
    generator = SSASyntheticDataGenerator()
    
    # Generate comprehensive dataset
    comprehensive_data = generator.create_comprehensive_dataset()
    
    # Generate crisis scenarios
    crisis_scenarios = generator.generate_crisis_scenarios()
    
    # Save comprehensive dataset
    print("Saving comprehensive dataset...")
    comprehensive_data.to_csv('processed_data/ssa_comprehensive_dataset.csv', index=False)
    comprehensive_data.to_excel('processed_data/ssa_comprehensive_dataset.xlsx', index=False)
    
    # Save crisis scenarios
    with open('processed_data/crisis_scenarios.json', 'w') as f:
        json.dump(crisis_scenarios, f, indent=2)
    
    print(f"\nSynthetic data generation completed!")
    print(f"Comprehensive dataset shape: {comprehensive_data.shape}")
    print(f"Total indicators: {comprehensive_data.shape[1] - 3}")  # Minus country_code, country_name, year
    
    # Show summary of new indicators
    original_cols = set(generator.real_data.columns)
    new_cols = set(comprehensive_data.columns) - original_cols
    
    print(f"\nNew synthetic indicators added ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    # Data completeness summary
    print(f"\nData completeness for key FinTech indicators:")
    fintech_indicators = [col for col in comprehensive_data.columns if 'fintech' in col.lower()]
    for indicator in fintech_indicators[:5]:  # Show first 5
        completeness = comprehensive_data[indicator].notna().mean() * 100
        print(f"  {indicator}: {completeness:.1f}% complete")
    
    print(f"\nFiles saved:")
    print(f"  - processed_data/ssa_comprehensive_dataset.csv")
    print(f"  - processed_data/ssa_comprehensive_dataset.xlsx")
    print(f"  - processed_data/crisis_scenarios.json")

if __name__ == "__main__":
    main()