#!/usr/bin/env python3
"""
FinTech Early Warning Model Dataset Generator for Sub-Saharan Africa
Generates comprehensive dataset for FinTech distress prediction research
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FinTechDatasetGenerator:
    def __init__(self, seed=42):
        """Initialize the dataset generator with a random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
        # SSA Countries and their FinTech ecosystems
        self.countries = {
            'Nigeria': {'gdp_per_capita': 2028, 'mobile_penetration': 0.89, 'fintech_maturity': 'high'},
            'Kenya': {'gdp_per_capita': 1808, 'mobile_penetration': 0.91, 'fintech_maturity': 'high'},
            'South Africa': {'gdp_per_capita': 6994, 'mobile_penetration': 0.95, 'fintech_maturity': 'high'},
            'Ghana': {'gdp_per_capita': 2202, 'mobile_penetration': 0.87, 'fintech_maturity': 'medium'},
            'Uganda': {'gdp_per_capita': 884, 'mobile_penetration': 0.78, 'fintech_maturity': 'medium'},
            'Tanzania': {'gdp_per_capita': 1098, 'mobile_penetration': 0.82, 'fintech_maturity': 'medium'},
            'Rwanda': {'gdp_per_capita': 822, 'mobile_penetration': 0.85, 'fintech_maturity': 'medium'},
            'Senegal': {'gdp_per_capita': 1444, 'mobile_penetration': 0.88, 'fintech_maturity': 'medium'},
            'Ethiopia': {'gdp_per_capita': 925, 'mobile_penetration': 0.45, 'fintech_maturity': 'low'},
            'Zambia': {'gdp_per_capita': 1209, 'mobile_penetration': 0.79, 'fintech_maturity': 'low'}
        }
        
        # FinTech company types and their characteristics
        self.fintech_types = {
            'Mobile Money': {'default_risk': 0.15, 'growth_rate': 0.25, 'regulatory_risk': 0.20},
            'Digital Banking': {'default_risk': 0.12, 'growth_rate': 0.30, 'regulatory_risk': 0.25},
            'Payment Gateway': {'default_risk': 0.10, 'growth_rate': 0.35, 'regulatory_risk': 0.15},
            'Lending Platform': {'default_risk': 0.25, 'growth_rate': 0.40, 'regulatory_risk': 0.30},
            'Investment Platform': {'default_risk': 0.18, 'growth_rate': 0.28, 'regulatory_risk': 0.35},
            'Insurance Tech': {'default_risk': 0.20, 'growth_rate': 0.22, 'regulatory_risk': 0.40},
            'Crypto Exchange': {'default_risk': 0.35, 'growth_rate': 0.50, 'regulatory_risk': 0.45},
            'Remittance': {'default_risk': 0.08, 'growth_rate': 0.20, 'regulatory_risk': 0.10}
        }
        
        # Real FinTech companies in SSA (for realistic naming)
        self.real_companies = {
            'Nigeria': ['Flutterwave', 'Paystack', 'Kuda', 'Carbon', 'PiggyVest', 'Cowrywise', 'Bamboo'],
            'Kenya': ['M-Pesa', 'Tala', 'Branch', 'KCB M-Pesa', 'Equitel', 'M-Shwari', 'Kopo Kopo'],
            'South Africa': ['Yoco', 'Ozow', 'JUMO', 'Luno', 'EasyEquities', 'Mukuru', 'SpotMoney'],
            'Ghana': ['MTN Mobile Money', 'AirtelTigo Money', 'ExpressPay', 'Zeepay', 'Hubtel', 'Slydepay'],
            'Uganda': ['MTN Mobile Money', 'Airtel Money', 'Equity Bank', 'Centenary Bank', 'Stanbic Bank'],
            'Tanzania': ['M-Pesa', 'Tigo Pesa', 'Airtel Money', 'HaloPesa', 'NMB Mobile'],
            'Rwanda': ['MTN Mobile Money', 'Airtel Money', 'BK Mobile', 'Equity Bank'],
            'Senegal': ['Orange Money', 'Free Money', 'Wari', 'Joni Joni'],
            'Ethiopia': ['M-Birr', 'Kifiya', 'Chapa', 'Amole'],
            'Zambia': ['MTN Mobile Money', 'Airtel Money', 'Zoona', 'Mukuru']
        }

    def generate_company_data(self, num_companies=200) -> pd.DataFrame:
        """Generate comprehensive FinTech company data"""
        companies = []
        
        for i in range(num_companies):
            # Select country and company type
            country = np.random.choice(list(self.countries.keys()))
            fintech_type = np.random.choice(list(self.fintech_types.keys()))
            
            # Company characteristics
            company_name = np.random.choice(self.real_companies[country])
            if np.random.random() < 0.3:  # 30% chance of adding a suffix for variety
                suffixes = [' Ltd', ' Inc', ' Technologies', ' Solutions', ' Digital', ' Group']
                company_name += np.random.choice(suffixes)
            
            # Company age (years since founding)
            age = np.random.exponential(3) + 1  # Most companies are young
            age = min(age, 15)  # Cap at 15 years
            
            # Company size (employees)
            if age < 2:
                size = np.random.choice(['Startup', 'Small'], p=[0.7, 0.3])
                employees = np.random.randint(5, 50)
            elif age < 5:
                size = np.random.choice(['Small', 'Medium'], p=[0.6, 0.4])
                employees = np.random.randint(20, 200)
            else:
                size = np.random.choice(['Medium', 'Large'], p=[0.7, 0.3])
                employees = np.random.randint(100, 1000)
            
            # Regulatory status
            regulatory_status = np.random.choice(
                ['Licensed', 'Pending', 'Unlicensed', 'Suspended'],
                p=[0.6, 0.2, 0.15, 0.05]
            )
            
            # Funding stage
            funding_stages = ['Bootstrap', 'Seed', 'Series A', 'Series B', 'Series C', 'IPO', 'Acquired']
            funding_stage = np.random.choice(funding_stages, p=[0.2, 0.3, 0.25, 0.15, 0.05, 0.03, 0.02])
            
            companies.append({
                'company_id': f'FT{i+1:04d}',
                'company_name': company_name,
                'country': country,
                'fintech_type': fintech_type,
                'age_years': round(age, 1),
                'company_size': size,
                'employees': employees,
                'regulatory_status': regulatory_status,
                'funding_stage': funding_stage,
                'gdp_per_capita': self.countries[country]['gdp_per_capita'],
                'mobile_penetration': self.countries[country]['mobile_penetration'],
                'fintech_maturity': self.countries[country]['fintech_maturity']
            })
        
        return pd.DataFrame(companies)

    def generate_financial_metrics(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Generate financial performance metrics for each company"""
        financial_data = []
        
        for _, company in companies_df.iterrows():
            # Base financial metrics based on company characteristics
            base_revenue = self._calculate_base_revenue(company)
            base_profit_margin = self._calculate_base_profit_margin(company)
            
            # Generate quarterly data for the last 2 years (8 quarters)
            for quarter in range(8):
                quarter_date = datetime.now() - timedelta(days=90 * quarter)
                
                # Revenue with growth trends and seasonality
                growth_rate = self._calculate_growth_rate(company, quarter)
                revenue = base_revenue * (1 + growth_rate) ** (7 - quarter)
                revenue = max(revenue, 10000)  # Minimum revenue
                
                # Add some noise
                revenue *= np.random.normal(1, 0.1)
                
                # Profit margin with some variation
                profit_margin = base_profit_margin + np.random.normal(0, 0.05)
                profit_margin = max(profit_margin, -0.5)  # Cap losses at 50%
                
                net_income = revenue * profit_margin
                
                # Operating expenses
                operating_expenses = revenue * (1 - profit_margin)
                
                # Burn rate (for startups)
                burn_rate = 0
                if company['funding_stage'] in ['Bootstrap', 'Seed', 'Series A']:
                    burn_rate = operating_expenses * np.random.uniform(0.8, 1.2)
                
                financial_data.append({
                    'company_id': company['company_id'],
                    'quarter': f'Q{quarter + 1}',
                    'quarter_date': quarter_date.strftime('%Y-%m-%d'),
                    'revenue_usd': round(revenue, 2),
                    'revenue_growth_rate': round(growth_rate, 4),
                    'net_income_usd': round(net_income, 2),
                    'profit_margin': round(profit_margin, 4),
                    'operating_expenses_usd': round(operating_expenses, 2),
                    'burn_rate_usd': round(burn_rate, 2)
                })
        
        return pd.DataFrame(financial_data)

    def generate_operational_metrics(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Generate operational metrics (users, transactions, etc.)"""
        operational_data = []
        
        for _, company in companies_df.iterrows():
            # Base operational metrics
            base_users = self._calculate_base_users(company)
            base_transaction_volume = self._calculate_base_transaction_volume(company)
            
            for quarter in range(8):
                quarter_date = datetime.now() - timedelta(days=90 * quarter)
                
                # Active users with growth and churn
                user_growth_rate = np.random.normal(0.05, 0.1)  # 5% average growth
                churn_rate = np.random.uniform(0.02, 0.08)  # 2-8% churn
                
                if quarter == 0:
                    active_users = base_users
                else:
                    # Previous quarter users with growth and churn
                    prev_users = base_users * (1 + user_growth_rate) ** (7 - quarter + 1)
                    active_users = prev_users * (1 + user_growth_rate - churn_rate)
                
                active_users = max(active_users, 100)  # Minimum users
                
                # Transaction metrics
                avg_transaction_value = np.random.uniform(10, 500)  # USD
                transactions_per_user = np.random.uniform(2, 20)  # per quarter
                
                total_transactions = int(active_users * transactions_per_user)
                transaction_volume = total_transactions * avg_transaction_value
                
                # Customer acquisition cost
                cac = np.random.uniform(5, 50)  # USD per customer
                
                # Number of agents (for mobile money)
                agents = 0
                if company['fintech_type'] == 'Mobile Money':
                    agents = int(active_users / np.random.uniform(100, 500))
                
                operational_data.append({
                    'company_id': company['company_id'],
                    'quarter': f'Q{quarter + 1}',
                    'quarter_date': quarter_date.strftime('%Y-%m-%d'),
                    'active_users': int(active_users),
                    'user_growth_rate': round(user_growth_rate, 4),
                    'churn_rate': round(churn_rate, 4),
                    'total_transactions': total_transactions,
                    'transaction_volume_usd': round(transaction_volume, 2),
                    'avg_transaction_value_usd': round(avg_transaction_value, 2),
                    'customer_acquisition_cost_usd': round(cac, 2),
                    'agents_count': agents
                })
        
        return pd.DataFrame(operational_data)

    def generate_funding_data(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Generate funding rounds and investment data"""
        funding_data = []
        
        for _, company in companies_df.iterrows():
            # Number of funding rounds based on company age and stage
            num_rounds = self._calculate_num_funding_rounds(company)
            
            for round_num in range(num_rounds):
                # Funding round details
                round_date = datetime.now() - timedelta(days=np.random.randint(30, 365 * company['age_years']))
                
                # Round type and amount based on stage
                round_type, amount = self._calculate_funding_round(company, round_num)
                
                # Investors
                num_investors = np.random.randint(1, 5)
                investors = self._generate_investor_list(num_investors)
                
                # Valuation
                valuation = amount * np.random.uniform(3, 10)  # 3-10x multiple
                
                funding_data.append({
                    'company_id': company['company_id'],
                    'round_number': round_num + 1,
                    'round_date': round_date.strftime('%Y-%m-%d'),
                    'round_type': round_type,
                    'amount_raised_usd': amount,
                    'valuation_usd': round(valuation, 2),
                    'num_investors': num_investors,
                    'investors': ', '.join(investors),
                    'months_since_last_round': np.random.randint(6, 24) if round_num > 0 else 0
                })
        
        return pd.DataFrame(funding_data)

    def generate_regulatory_data(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regulatory sanctions and compliance data"""
        regulatory_data = []
        
        for _, company in companies_df.iterrows():
            # Probability of regulatory issues based on company characteristics
            sanction_prob = self.fintech_types[company['fintech_type']]['regulatory_risk']
            
            # Generate regulatory events
            num_events = np.random.poisson(sanction_prob * 2)  # Average events per company
            
            for event_num in range(num_events):
                event_date = datetime.now() - timedelta(days=np.random.randint(30, 365 * 2))
                
                # Event type and severity
                event_types = ['Warning', 'Fine', 'Suspension', 'License Revocation', 'Compliance Review']
                event_type = np.random.choice(event_types, p=[0.4, 0.3, 0.15, 0.1, 0.05])
                
                severity = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
                
                # Fine amount (if applicable)
                fine_amount = 0
                if event_type in ['Fine', 'Suspension']:
                    fine_amount = np.random.uniform(1000, 100000)
                
                # Resolution status
                resolution_status = np.random.choice(['Open', 'Resolved', 'Appealed'], p=[0.3, 0.6, 0.1])
                
                regulatory_data.append({
                    'company_id': company['company_id'],
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'event_type': event_type,
                    'severity': severity,
                    'fine_amount_usd': round(fine_amount, 2),
                    'resolution_status': resolution_status,
                    'description': f"{event_type} issued by {company['country']} Central Bank"
                })
        
        return pd.DataFrame(regulatory_data)

    def generate_distress_indicators(self, companies_df: pd.DataFrame, 
                                   financial_df: pd.DataFrame, 
                                   operational_df: pd.DataFrame) -> pd.DataFrame:
        """Generate distress indicators and dependent variables"""
        distress_data = []
        
        for _, company in companies_df.iterrows():
            # Get company's financial and operational data
            company_financial = financial_df[financial_df['company_id'] == company['company_id']]
            company_operational = operational_df[operational_df['company_id'] == company['company_id']]
            
            # Calculate distress indicators
            recent_revenue = company_financial['revenue_usd'].iloc[-1] if len(company_financial) > 0 else 0
            revenue_decline = 0
            if len(company_financial) >= 2:
                revenue_decline = (company_financial['revenue_usd'].iloc[-2] - recent_revenue) / company_financial['revenue_usd'].iloc[-2]
            
            recent_users = company_operational['active_users'].iloc[-1] if len(company_operational) > 0 else 0
            user_decline = 0
            if len(company_operational) >= 2:
                user_decline = (company_operational['active_users'].iloc[-2] - recent_users) / company_operational['active_users'].iloc[-2]
            
            # Calculate distress score (0-1, higher = more distressed)
            distress_score = 0
            
            # Revenue decline factor
            if revenue_decline > 0.2:  # 20% decline
                distress_score += 0.3
            elif revenue_decline > 0.1:  # 10% decline
                distress_score += 0.15
            
            # User decline factor
            if user_decline > 0.15:  # 15% user decline
                distress_score += 0.2
            elif user_decline > 0.05:  # 5% user decline
                distress_score += 0.1
            
            # Regulatory issues factor
            regulatory_issues = len(regulatory_df[regulatory_df['company_id'] == company['company_id']]) if 'regulatory_df' in globals() else 0
            if regulatory_issues > 2:
                distress_score += 0.2
            elif regulatory_issues > 0:
                distress_score += 0.1
            
            # Company age and size factors
            if company['age_years'] < 2:  # Young companies more at risk
                distress_score += 0.1
            
            if company['company_size'] == 'Startup':
                distress_score += 0.15
            
            # FinTech type risk
            type_risk = self.fintech_types[company['fintech_type']]['default_risk']
            distress_score += type_risk * 0.3
            
            # Add some randomness
            distress_score += np.random.normal(0, 0.05)
            distress_score = max(0, min(1, distress_score))  # Clamp between 0 and 1
            
            # Binary distress indicator (threshold at 0.4)
            is_distressed = 1 if distress_score > 0.4 else 0
            
            # Additional distress indicators
            has_revenue_decline = 1 if revenue_decline > 0.1 else 0
            has_user_decline = 1 if user_decline > 0.05 else 0
            has_regulatory_issues = 1 if regulatory_issues > 0 else 0
            
            distress_data.append({
                'company_id': company['company_id'],
                'distress_score': round(distress_score, 4),
                'is_distressed': is_distressed,
                'revenue_decline_rate': round(revenue_decline, 4),
                'user_decline_rate': round(user_decline, 4),
                'has_revenue_decline': has_revenue_decline,
                'has_user_decline': has_user_decline,
                'has_regulatory_issues': has_regulatory_issues,
                'regulatory_issues_count': regulatory_issues,
                'months_since_founded': int(company['age_years'] * 12)
            })
        
        return pd.DataFrame(distress_data)

    def _calculate_base_revenue(self, company: pd.Series) -> float:
        """Calculate base revenue based on company characteristics"""
        base = 100000  # Base revenue
        
        # Adjust by company size
        size_multipliers = {'Startup': 0.5, 'Small': 1, 'Medium': 3, 'Large': 10}
        base *= size_multipliers[company['company_size']]
        
        # Adjust by country GDP
        base *= (company['gdp_per_capita'] / 2000)  # Normalize to $2000 GDP
        
        # Adjust by FinTech type
        type_multipliers = {
            'Mobile Money': 2.0, 'Digital Banking': 1.5, 'Payment Gateway': 1.2,
            'Lending Platform': 0.8, 'Investment Platform': 0.6, 'Insurance Tech': 0.7,
            'Crypto Exchange': 1.0, 'Remittance': 1.3
        }
        base *= type_multipliers[company['fintech_type']]
        
        return base

    def _calculate_base_profit_margin(self, company: pd.Series) -> float:
        """Calculate base profit margin based on company characteristics"""
        base_margin = 0.1  # 10% base margin
        
        # Adjust by company age (older = more profitable)
        base_margin += company['age_years'] * 0.01
        
        # Adjust by company size
        size_adjustments = {'Startup': -0.15, 'Small': -0.05, 'Medium': 0.05, 'Large': 0.1}
        base_margin += size_adjustments[company['company_size']]
        
        # Adjust by FinTech type
        type_adjustments = {
            'Mobile Money': 0.15, 'Digital Banking': 0.1, 'Payment Gateway': 0.2,
            'Lending Platform': 0.05, 'Investment Platform': 0.08, 'Insurance Tech': 0.12,
            'Crypto Exchange': 0.0, 'Remittance': 0.18
        }
        base_margin += type_adjustments[company['fintech_type']]
        
        return max(base_margin, -0.3)  # Cap losses at 30%

    def _calculate_growth_rate(self, company: pd.Series, quarter: int) -> float:
        """Calculate growth rate for a specific quarter"""
        base_growth = 0.05  # 5% base growth
        
        # Adjust by company age (younger = higher growth)
        age_factor = max(0, (5 - company['age_years']) / 5)
        base_growth += age_factor * 0.1
        
        # Adjust by FinTech type
        type_growth = self.fintech_types[company['fintech_type']]['growth_rate']
        base_growth = (base_growth + type_growth) / 2
        
        # Add seasonality (Q4 typically higher)
        seasonality = 0.02 if quarter % 4 == 3 else 0
        
        # Add some randomness
        growth_rate = base_growth + seasonality + np.random.normal(0, 0.05)
        
        return max(growth_rate, -0.2)  # Cap decline at 20%

    def _calculate_base_users(self, company: pd.Series) -> int:
        """Calculate base number of users"""
        base = 1000  # Base users
        
        # Adjust by company size
        size_multipliers = {'Startup': 0.5, 'Small': 2, 'Medium': 10, 'Large': 50}
        base *= size_multipliers[company['company_size']]
        
        # Adjust by mobile penetration
        base *= company['mobile_penetration']
        
        # Adjust by FinTech type
        type_multipliers = {
            'Mobile Money': 5.0, 'Digital Banking': 2.0, 'Payment Gateway': 1.0,
            'Lending Platform': 0.5, 'Investment Platform': 0.3, 'Insurance Tech': 0.4,
            'Crypto Exchange': 0.8, 'Remittance': 1.5
        }
        base *= type_multipliers[company['fintech_type']]
        
        return int(base)

    def _calculate_base_transaction_volume(self, company: pd.Series) -> float:
        """Calculate base transaction volume"""
        base = 1000000  # $1M base volume
        
        # Adjust by company size
        size_multipliers = {'Startup': 0.3, 'Small': 1, 'Medium': 5, 'Large': 20}
        base *= size_multipliers[company['company_size']]
        
        # Adjust by country GDP
        base *= (company['gdp_per_capita'] / 2000)
        
        return base

    def _calculate_num_funding_rounds(self, company: pd.Series) -> int:
        """Calculate number of funding rounds based on company characteristics"""
        base_rounds = int(company['age_years'] / 2)  # One round every 2 years
        
        # Adjust by funding stage
        stage_adjustments = {
            'Bootstrap': 0, 'Seed': 1, 'Series A': 2, 'Series B': 3,
            'Series C': 4, 'IPO': 5, 'Acquired': 3
        }
        base_rounds = max(base_rounds, stage_adjustments[company['funding_stage']])
        
        # Add some randomness
        base_rounds += np.random.poisson(0.5)
        
        return min(base_rounds, 8)  # Cap at 8 rounds

    def _calculate_funding_round(self, company: pd.Series, round_num: int) -> Tuple[str, float]:
        """Calculate funding round type and amount"""
        round_types = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D', 'Bridge']
        round_amounts = [50000, 200000, 500000, 1000000, 2000000, 5000000]
        
        if round_num < len(round_types):
            round_type = round_types[round_num]
            base_amount = round_amounts[min(round_num, len(round_amounts) - 1)]
        else:
            round_type = 'Series D+'
            base_amount = 2000000
        
        # Adjust by company size and type
        size_multipliers = {'Startup': 0.5, 'Small': 1, 'Medium': 2, 'Large': 5}
        base_amount *= size_multipliers[company['company_size']]
        
        # Add some randomness
        amount = base_amount * np.random.uniform(0.5, 2.0)
        
        return round_type, round(amount, 2)

    def _generate_investor_list(self, num_investors: int) -> List[str]:
        """Generate list of investor names"""
        investors = [
            'Andela', 'TLcom Capital', 'Partech Partners', '4DX Ventures', 'CRE Venture Capital',
            'Novastar Ventures', 'TLcom Capital', 'Speedinvest', 'Global Founders Capital',
            'Y Combinator', '500 Startups', 'Accel Partners', 'Sequoia Capital', 'Andreessen Horowitz',
            'Local Angel Investors', 'Corporate VC', 'Government Fund', 'Development Bank'
        ]
        
        return random.sample(investors, min(num_investors, len(investors)))

    def generate_complete_dataset(self, num_companies=200) -> Dict[str, pd.DataFrame]:
        """Generate the complete FinTech distress dataset"""
        print("Generating FinTech companies...")
        companies_df = self.generate_company_data(num_companies)
        
        print("Generating financial metrics...")
        financial_df = self.generate_financial_metrics(companies_df)
        
        print("Generating operational metrics...")
        operational_df = self.generate_operational_metrics(companies_df)
        
        print("Generating funding data...")
        funding_df = self.generate_funding_data(companies_df)
        
        print("Generating regulatory data...")
        regulatory_df = self.generate_regulatory_data(companies_df)
        
        print("Generating distress indicators...")
        distress_df = self.generate_distress_indicators(companies_df, financial_df, operational_df)
        
        return {
            'companies': companies_df,
            'financial_metrics': financial_df,
            'operational_metrics': operational_df,
            'funding_data': funding_df,
            'regulatory_data': regulatory_df,
            'distress_indicators': distress_df
        }

    def save_dataset(self, dataset: Dict[str, pd.DataFrame], output_dir: str = 'fintech_dataset'):
        """Save the dataset to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in dataset.items():
            filepath = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"Saved {name} to {filepath}")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'num_companies': len(dataset['companies']),
            'countries': list(self.countries.keys()),
            'fintech_types': list(self.fintech_types.keys()),
            'description': 'FinTech Early Warning Model Dataset for Sub-Saharan Africa'
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}/")

if __name__ == "__main__":
    # Generate the dataset
    generator = FinTechDatasetGenerator(seed=42)
    dataset = generator.generate_complete_dataset(num_companies=200)
    
    # Save the dataset
    generator.save_dataset(dataset)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Number of companies: {len(dataset['companies'])}")
    print(f"Countries covered: {len(dataset['companies']['country'].unique())}")
    print(f"FinTech types: {len(dataset['companies']['fintech_type'].unique())}")
    print(f"Distressed companies: {dataset['distress_indicators']['is_distressed'].sum()}")
    print(f"Distress rate: {dataset['distress_indicators']['is_distressed'].mean():.2%}")
    
    print("\n=== Sample Data ===")
    print("\nCompanies:")
    print(dataset['companies'].head())
    
    print("\nDistress Indicators:")
    print(dataset['distress_indicators'].head())