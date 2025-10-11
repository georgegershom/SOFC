#!/usr/bin/env python3
"""
FinTech Early Warning Dataset Generator for Sub-Saharan Africa
Research Topic: FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

This script generates a comprehensive dataset for FinTech distress prediction research,
focusing on Category 1: FinTech-Specific Data (The Micro Foundation).
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FinTechDatasetGenerator:
    def __init__(self):
        self.ssa_countries = [
            'Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Uganda', 'Tanzania',
            'Rwanda', 'Zambia', 'Botswana', 'Senegal', 'Ivory Coast', 'Ethiopia',
            'Mali', 'Burkina Faso', 'Cameroon', 'Zimbabwe', 'Malawi', 'Mozambique'
        ]
        
        self.fintech_types = [
            'Mobile Money', 'Digital Banking', 'Payment Gateway', 'Lending Platform',
            'Investment Platform', 'Insurance Tech', 'Remittance Service', 'Crypto Exchange',
            'POS Solutions', 'Digital Wallet', 'Microfinance Tech', 'Crowdfunding Platform'
        ]
        
        self.company_names = [
            'M-Pesa', 'MTN Mobile Money', 'Airtel Money', 'Flutterwave', 'Paystack',
            'Interswitch', 'Paga', 'Kuda Bank', 'Carbon', 'FairMoney', 'PalmPay',
            'OPay', 'Chipper Cash', 'Wave', 'Tala', 'Branch', 'Jumo', 'Yoco',
            'PayFast', 'SnapScan', 'Ozow', 'Peach Payments', 'DPO Group',
            'Cellulant', 'Sendy', 'Twiga Foods', 'Sokowatch', 'MarketForce',
            'Lipa Later', 'Credpal', 'Renmoney', 'Lidya', 'Aella Credit',
            'Cowrywise', 'PiggyVest', 'Bamboo', 'Chaka', 'Trove Finance',
            'SunTrust Bank', 'Kopo Kopo', 'iPay', 'PesaLink', 'Equity Bank',
            'Tyme Bank', 'TymeBank', 'Discovery Bank', 'Bank Zero', 'Capitec',
            'Absa', 'Standard Bank', 'FirstBank', 'GTBank', 'Access Bank'
        ]
        
        self.funding_stages = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D', 'IPO']
        
        # Date range for the dataset (5 years of data)
        self.start_date = datetime(2019, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
    def generate_company_profiles(self, n_companies: int = 150) -> pd.DataFrame:
        """Generate realistic FinTech company profiles for SSA region."""
        companies = []
        
        for i in range(n_companies):
            # Create realistic company profile
            company_id = f"FT_{i+1:03d}"
            company_name = f"{random.choice(self.company_names)}_{random.randint(1, 999)}"
            
            # Ensure unique names
            while any(c['company_name'] == company_name for c in companies):
                company_name = f"{random.choice(self.company_names)}_{random.randint(1, 999)}"
            
            country = random.choice(self.ssa_countries)
            fintech_type = random.choice(self.fintech_types)
            
            # Company age affects many metrics
            founding_year = random.randint(2010, 2022)
            company_age = 2024 - founding_year
            
            # Market tier affects scale and resources
            market_tier = random.choices(['Tier 1', 'Tier 2', 'Tier 3'], 
                                       weights=[0.3, 0.5, 0.2])[0]
            
            companies.append({
                'company_id': company_id,
                'company_name': company_name,
                'country': country,
                'fintech_type': fintech_type,
                'founding_year': founding_year,
                'company_age': company_age,
                'market_tier': market_tier,
                'is_licensed': random.choices([True, False], weights=[0.8, 0.2])[0],
                'has_banking_partner': random.choices([True, False], weights=[0.7, 0.3])[0]
            })
        
        return pd.DataFrame(companies)
    
    def generate_time_series_data(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Generate time series data for each company across multiple quarters."""
        all_data = []
        
        # Generate quarterly data points
        current_date = self.start_date
        quarters = []
        while current_date <= self.end_date:
            quarters.append(current_date)
            # Move to next quarter
            if current_date.month == 10:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 3)
        
        for _, company in companies_df.iterrows():
            company_data = self._generate_company_time_series(company, quarters)
            all_data.extend(company_data)
        
        return pd.DataFrame(all_data)
    
    def _generate_company_time_series(self, company: pd.Series, quarters: List[datetime]) -> List[Dict]:
        """Generate time series data for a single company."""
        company_data = []
        
        # Initialize base metrics based on company characteristics
        base_metrics = self._initialize_base_metrics(company)
        
        # Determine if company will experience distress and when
        distress_info = self._determine_distress_pattern(company, quarters)
        
        for i, quarter in enumerate(quarters):
            # Skip quarters before company founding
            if quarter.year < company['founding_year']:
                continue
                
            # Calculate metrics for this quarter
            metrics = self._calculate_quarterly_metrics(
                company, quarter, i, base_metrics, distress_info
            )
            
            company_data.append(metrics)
        
        return company_data
    
    def _initialize_base_metrics(self, company: pd.Series) -> Dict:
        """Initialize base metrics based on company characteristics."""
        # Scale factors based on market tier and type
        scale_factors = {
            'Tier 1': {'users': 1000000, 'revenue': 10000000, 'funding': 50000000},
            'Tier 2': {'users': 100000, 'revenue': 1000000, 'funding': 5000000},
            'Tier 3': {'users': 10000, 'revenue': 100000, 'funding': 500000}
        }
        
        scale = scale_factors[company['market_tier']]
        
        # FinTech type affects metrics
        type_multipliers = {
            'Mobile Money': {'users': 2.0, 'revenue': 1.5, 'transactions': 3.0},
            'Digital Banking': {'users': 1.5, 'revenue': 2.0, 'transactions': 2.0},
            'Payment Gateway': {'users': 0.8, 'revenue': 1.8, 'transactions': 4.0},
            'Lending Platform': {'users': 0.6, 'revenue': 2.5, 'transactions': 1.0},
            'Investment Platform': {'users': 0.4, 'revenue': 1.2, 'transactions': 0.5},
            'Insurance Tech': {'users': 0.5, 'revenue': 1.3, 'transactions': 0.3},
            'Remittance Service': {'users': 0.7, 'revenue': 1.6, 'transactions': 1.5},
            'Crypto Exchange': {'users': 0.3, 'revenue': 2.2, 'transactions': 2.5},
            'POS Solutions': {'users': 1.2, 'revenue': 1.4, 'transactions': 3.5},
            'Digital Wallet': {'users': 1.8, 'revenue': 1.1, 'transactions': 2.8},
            'Microfinance Tech': {'users': 0.8, 'revenue': 1.7, 'transactions': 0.8},
            'Crowdfunding Platform': {'users': 0.2, 'revenue': 0.8, 'transactions': 0.2}
        }
        
        multiplier = type_multipliers.get(company['fintech_type'], {'users': 1.0, 'revenue': 1.0, 'transactions': 1.0})
        
        return {
            'base_users': int(scale['users'] * multiplier['users'] * random.uniform(0.5, 1.5)),
            'base_revenue': scale['revenue'] * multiplier['revenue'] * random.uniform(0.5, 1.5),
            'base_transactions': int(scale['users'] * multiplier['transactions'] * random.uniform(0.5, 2.0)),
            'base_funding': scale['funding'] * random.uniform(0.3, 2.0)
        }
    
    def _determine_distress_pattern(self, company: pd.Series, quarters: List[datetime]) -> Dict:
        """Determine if and when a company will experience distress."""
        # Probability of distress based on company characteristics
        distress_prob = 0.15  # Base 15% chance
        
        # Adjust based on characteristics
        if company['market_tier'] == 'Tier 3':
            distress_prob += 0.1
        if not company['is_licensed']:
            distress_prob += 0.05
        if not company['has_banking_partner']:
            distress_prob += 0.03
        if company['company_age'] < 3:
            distress_prob += 0.08
        
        will_have_distress = random.random() < distress_prob
        
        if will_have_distress:
            # Determine type of distress
            distress_types = ['closure', 'acquisition', 'regulatory_action', 'severe_downturn']
            distress_type = random.choice(distress_types)
            
            # Determine when distress occurs (usually in later periods)
            available_quarters = [q for q in quarters if q.year >= company['founding_year'] + 1]
            if available_quarters:
                distress_quarter_idx = random.randint(len(available_quarters)//2, len(available_quarters)-1)
                distress_quarter = available_quarters[distress_quarter_idx]
            else:
                distress_quarter = quarters[-1]
            
            return {
                'has_distress': True,
                'distress_type': distress_type,
                'distress_quarter': distress_quarter,
                'warning_quarters': 2 + random.randint(0, 2)  # 2-4 quarters of warning signs
            }
        
        return {'has_distress': False}
    
    def _calculate_quarterly_metrics(self, company: pd.Series, quarter: datetime, 
                                   quarter_idx: int, base_metrics: Dict, 
                                   distress_info: Dict) -> Dict:
        """Calculate all metrics for a specific quarter."""
        
        # Company age in quarters
        quarters_since_founding = max(0, (quarter.year - company['founding_year']) * 4 + (quarter.month - 1) // 3)
        
        # Growth patterns
        growth_phase = self._determine_growth_phase(quarters_since_founding)
        seasonal_factor = self._get_seasonal_factor(quarter.month)
        
        # Distress effects
        distress_factor = self._calculate_distress_factor(quarter, distress_info)
        
        # Calculate core metrics
        metrics = {
            'company_id': company['company_id'],
            'company_name': company['company_name'],
            'country': company['country'],
            'fintech_type': company['fintech_type'],
            'quarter': quarter,
            'year': quarter.year,
            'quarter_num': (quarter.month - 1) // 3 + 1,
            'quarters_since_founding': quarters_since_founding,
            'company_age_years': round(quarters_since_founding / 4, 1)
        }
        
        # Financial metrics
        metrics.update(self._calculate_financial_metrics(
            base_metrics, growth_phase, seasonal_factor, distress_factor, quarters_since_founding
        ))
        
        # Operational metrics
        metrics.update(self._calculate_operational_metrics(
            base_metrics, growth_phase, seasonal_factor, distress_factor, quarters_since_founding
        ))
        
        # Funding metrics
        metrics.update(self._calculate_funding_metrics(
            base_metrics, quarters_since_founding, distress_factor
        ))
        
        # Distress indicators (dependent variables)
        metrics.update(self._calculate_distress_indicators(quarter, distress_info))
        
        return metrics
    
    def _determine_growth_phase(self, quarters_since_founding: int) -> str:
        """Determine company growth phase based on age."""
        if quarters_since_founding <= 4:
            return 'startup'
        elif quarters_since_founding <= 12:
            return 'growth'
        elif quarters_since_founding <= 20:
            return 'expansion'
        else:
            return 'mature'
    
    def _get_seasonal_factor(self, month: int) -> float:
        """Get seasonal adjustment factor."""
        # Q4 typically stronger for financial services
        seasonal_factors = {1: 0.95, 2: 0.95, 3: 0.95,  # Q1
                          4: 1.0, 5: 1.0, 6: 1.0,      # Q2
                          7: 1.02, 8: 1.02, 9: 1.02,   # Q3
                          10: 1.08, 11: 1.08, 12: 1.08} # Q4
        return seasonal_factors.get(month, 1.0)
    
    def _calculate_distress_factor(self, quarter: datetime, distress_info: Dict) -> float:
        """Calculate distress impact factor."""
        if not distress_info.get('has_distress', False):
            return 1.0
        
        distress_quarter = distress_info['distress_quarter']
        warning_quarters = distress_info.get('warning_quarters', 2)
        
        # Calculate quarters until distress
        quarters_to_distress = ((distress_quarter.year - quarter.year) * 4 + 
                               (distress_quarter.month - quarter.month) // 3)
        
        if quarters_to_distress <= 0:
            # Post-distress or distress quarter
            if distress_info['distress_type'] == 'closure':
                return 0.1  # Severe decline
            else:
                return 0.3  # Significant decline
        elif quarters_to_distress <= warning_quarters:
            # Warning period - gradual decline
            decline_factor = 1.0 - (0.7 * (warning_quarters - quarters_to_distress) / warning_quarters)
            return max(0.3, decline_factor)
        else:
            # Normal operations
            return 1.0
    
    def _calculate_financial_metrics(self, base_metrics: Dict, growth_phase: str, 
                                   seasonal_factor: float, distress_factor: float,
                                   quarters_since_founding: int) -> Dict:
        """Calculate financial performance metrics."""
        
        # Growth rates by phase
        growth_rates = {
            'startup': random.uniform(0.8, 2.5),
            'growth': random.uniform(0.6, 1.8),
            'expansion': random.uniform(0.3, 1.2),
            'mature': random.uniform(0.05, 0.4)
        }
        
        base_growth = growth_rates[growth_phase]
        
        # Calculate revenue
        revenue_growth = base_growth * seasonal_factor * distress_factor
        quarterly_revenue = (base_metrics['base_revenue'] / 4) * (1 + revenue_growth) ** (quarters_since_founding / 4)
        quarterly_revenue *= random.uniform(0.8, 1.2)  # Add noise
        
        # Calculate costs and profitability
        # Startups typically have higher costs relative to revenue
        cost_ratio = {
            'startup': random.uniform(1.2, 2.0),
            'growth': random.uniform(0.9, 1.4),
            'expansion': random.uniform(0.7, 1.1),
            'mature': random.uniform(0.6, 0.9)
        }[growth_phase]
        
        quarterly_costs = quarterly_revenue * cost_ratio * random.uniform(0.9, 1.1)
        net_income = quarterly_revenue - quarterly_costs
        
        # Burn rate (relevant for startups and growth stage)
        if growth_phase in ['startup', 'growth']:
            burn_rate = abs(min(0, net_income)) + random.uniform(50000, 500000)
        else:
            burn_rate = max(0, abs(min(0, net_income)))
        
        return {
            'quarterly_revenue': max(0, quarterly_revenue),
            'quarterly_costs': max(0, quarterly_costs),
            'net_income': net_income,
            'burn_rate': burn_rate,
            'revenue_growth_rate': (revenue_growth - 1) * 100,
            'profit_margin': (net_income / quarterly_revenue * 100) if quarterly_revenue > 0 else -100
        }
    
    def _calculate_operational_metrics(self, base_metrics: Dict, growth_phase: str,
                                     seasonal_factor: float, distress_factor: float,
                                     quarters_since_founding: int) -> Dict:
        """Calculate operational metrics."""
        
        # User growth
        user_growth_rates = {
            'startup': random.uniform(0.5, 1.5),
            'growth': random.uniform(0.3, 1.0),
            'expansion': random.uniform(0.1, 0.5),
            'mature': random.uniform(0.02, 0.2)
        }
        
        user_growth = user_growth_rates[growth_phase] * distress_factor
        active_users = int(base_metrics['base_users'] * (1 + user_growth) ** (quarters_since_founding / 4))
        active_users = max(100, int(active_users * random.uniform(0.8, 1.2)))
        
        # Transaction metrics
        transactions_per_user = random.uniform(2, 15)  # Quarterly transactions per user
        transaction_count = int(active_users * transactions_per_user * seasonal_factor * distress_factor)
        
        # Average transaction value varies by FinTech type
        avg_transaction_values = {
            'Mobile Money': random.uniform(25, 150),
            'Digital Banking': random.uniform(200, 2000),
            'Payment Gateway': random.uniform(50, 500),
            'Lending Platform': random.uniform(500, 5000),
            'Investment Platform': random.uniform(1000, 10000),
            'Insurance Tech': random.uniform(100, 1000),
            'Remittance Service': random.uniform(100, 800),
            'Crypto Exchange': random.uniform(200, 2000),
            'POS Solutions': random.uniform(30, 200),
            'Digital Wallet': random.uniform(20, 300),
            'Microfinance Tech': random.uniform(100, 1000),
            'Crowdfunding Platform': random.uniform(50, 500)
        }
        
        # Get the company type from the first call (we need to pass it)
        avg_transaction_value = 100  # Default value, will be overridden
        
        transaction_volume = transaction_count * avg_transaction_value
        
        # Customer acquisition and churn
        if growth_phase == 'startup':
            cac = random.uniform(10, 50)
            churn_rate = random.uniform(15, 35)
        elif growth_phase == 'growth':
            cac = random.uniform(8, 30)
            churn_rate = random.uniform(10, 25)
        elif growth_phase == 'expansion':
            cac = random.uniform(5, 20)
            churn_rate = random.uniform(5, 15)
        else:  # mature
            cac = random.uniform(3, 15)
            churn_rate = random.uniform(3, 10)
        
        # Distress increases churn and CAC
        churn_rate *= (2 - distress_factor)
        cac *= (2 - distress_factor)
        
        # Number of agents (relevant for mobile money and POS)
        if quarters_since_founding > 0:
            agents = int(active_users / random.uniform(50, 500))
        else:
            agents = 0
        
        return {
            'active_users': active_users,
            'transaction_count': max(0, transaction_count),
            'transaction_volume': max(0, transaction_volume),
            'avg_transaction_value': avg_transaction_value,
            'customer_acquisition_cost': cac,
            'churn_rate': min(100, max(0, churn_rate)),
            'number_of_agents': max(0, agents),
            'user_growth_rate': user_growth * 100
        }
    
    def _calculate_funding_metrics(self, base_metrics: Dict, quarters_since_founding: int,
                                 distress_factor: float) -> Dict:
        """Calculate funding-related metrics."""
        
        # Determine if funding round occurs this quarter
        funding_probability = 0.05  # 5% chance per quarter
        
        # Adjust probability based on company age and distress
        if quarters_since_founding <= 8:  # First 2 years
            funding_probability = 0.15
        elif quarters_since_founding <= 16:  # Years 3-4
            funding_probability = 0.08
        
        funding_probability *= distress_factor  # Distressed companies less likely to get funding
        
        has_funding_round = random.random() < funding_probability
        
        if has_funding_round:
            # Determine funding stage based on company age
            if quarters_since_founding <= 4:
                stage = random.choice(['Pre-Seed', 'Seed'])
                amount = random.uniform(100000, 2000000)
            elif quarters_since_founding <= 12:
                stage = random.choice(['Seed', 'Series A'])
                amount = random.uniform(500000, 10000000)
            elif quarters_since_founding <= 20:
                stage = random.choice(['Series A', 'Series B'])
                amount = random.uniform(2000000, 25000000)
            else:
                stage = random.choice(['Series B', 'Series C', 'Series D'])
                amount = random.uniform(10000000, 100000000)
            
            return {
                'funding_round': True,
                'funding_stage': stage,
                'funding_amount': amount,
                'total_funding_to_date': amount  # This would be cumulative in real implementation
            }
        else:
            return {
                'funding_round': False,
                'funding_stage': None,
                'funding_amount': 0,
                'total_funding_to_date': 0
            }
    
    def _calculate_distress_indicators(self, quarter: datetime, distress_info: Dict) -> Dict:
        """Calculate dependent variables (distress indicators)."""
        
        if not distress_info.get('has_distress', False):
            return {
                'is_distressed': False,
                'distress_type': None,
                'regulatory_action': False,
                'closure_risk': 0.0,
                'acquisition_risk': 0.0
            }
        
        distress_quarter = distress_info['distress_quarter']
        distress_type = distress_info['distress_type']
        
        # Check if we're at or past the distress quarter
        is_distress_period = quarter >= distress_quarter
        
        # Calculate risk scores based on proximity to distress
        quarters_to_distress = ((distress_quarter.year - quarter.year) * 4 + 
                               (distress_quarter.month - quarter.month) // 3)
        
        if quarters_to_distress <= 0:
            # At or past distress point
            closure_risk = 0.9 if distress_type == 'closure' else 0.1
            acquisition_risk = 0.9 if distress_type == 'acquisition' else 0.1
            regulatory_action = distress_type == 'regulatory_action'
        elif quarters_to_distress <= 2:
            # High risk period
            closure_risk = 0.7 if distress_type == 'closure' else 0.05
            acquisition_risk = 0.7 if distress_type == 'acquisition' else 0.05
            regulatory_action = False
        elif quarters_to_distress <= 4:
            # Medium risk period
            closure_risk = 0.4 if distress_type == 'closure' else 0.02
            acquisition_risk = 0.4 if distress_type == 'acquisition' else 0.02
            regulatory_action = False
        else:
            # Low risk period
            closure_risk = 0.1 if distress_type == 'closure' else 0.01
            acquisition_risk = 0.1 if distress_type == 'acquisition' else 0.01
            regulatory_action = False
        
        return {
            'is_distressed': is_distress_period,
            'distress_type': distress_type if is_distress_period else None,
            'regulatory_action': regulatory_action,
            'closure_risk': min(1.0, closure_risk),
            'acquisition_risk': min(1.0, acquisition_risk)
        }
    
    def generate_dataset(self, n_companies: int = 150) -> Tuple[pd.DataFrame, Dict]:
        """Generate the complete dataset."""
        print("Generating company profiles...")
        companies_df = self.generate_company_profiles(n_companies)
        
        print("Generating time series data...")
        dataset = self.generate_time_series_data(companies_df)
        
        # Add some derived metrics
        dataset['revenue_per_user'] = dataset['quarterly_revenue'] / dataset['active_users'].replace(0, 1)
        dataset['transaction_value_per_user'] = dataset['transaction_volume'] / dataset['active_users'].replace(0, 1)
        dataset['cost_per_user'] = dataset['quarterly_costs'] / dataset['active_users'].replace(0, 1)
        
        # Create metadata
        metadata = {
            'dataset_info': {
                'title': 'FinTech Early Warning Dataset for Sub-Saharan Africa',
                'description': 'Comprehensive dataset for FinTech distress prediction research',
                'countries': self.ssa_countries,
                'fintech_types': self.fintech_types,
                'date_range': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
                'total_companies': n_companies,
                'total_observations': len(dataset),
                'distressed_companies': len(dataset[dataset['is_distressed'] == True]['company_id'].unique())
            },
            'variable_descriptions': self._get_variable_descriptions()
        }
        
        return dataset, metadata
    
    def _get_variable_descriptions(self) -> Dict:
        """Get descriptions of all variables in the dataset."""
        return {
            'company_identifiers': {
                'company_id': 'Unique identifier for each FinTech company',
                'company_name': 'Name of the FinTech company',
                'country': 'Sub-Saharan African country where company operates',
                'fintech_type': 'Type/category of FinTech service'
            },
            'temporal_variables': {
                'quarter': 'Date of the quarter (YYYY-MM-DD format)',
                'year': 'Year of observation',
                'quarter_num': 'Quarter number (1-4)',
                'quarters_since_founding': 'Number of quarters since company founding',
                'company_age_years': 'Company age in years'
            },
            'financial_metrics': {
                'quarterly_revenue': 'Total revenue for the quarter (USD)',
                'quarterly_costs': 'Total operational costs for the quarter (USD)',
                'net_income': 'Net income/loss for the quarter (USD)',
                'burn_rate': 'Cash burn rate per quarter (USD)',
                'revenue_growth_rate': 'Quarter-over-quarter revenue growth rate (%)',
                'profit_margin': 'Net profit margin (%)',
                'revenue_per_user': 'Revenue per active user (USD)'
            },
            'operational_metrics': {
                'active_users': 'Number of active users in the quarter',
                'transaction_count': 'Total number of transactions processed',
                'transaction_volume': 'Total value of transactions processed (USD)',
                'avg_transaction_value': 'Average transaction value (USD)',
                'customer_acquisition_cost': 'Cost to acquire a new customer (USD)',
                'churn_rate': 'Customer churn rate (%)',
                'number_of_agents': 'Number of agents/partners (for relevant FinTech types)',
                'user_growth_rate': 'User growth rate (%)',
                'transaction_value_per_user': 'Transaction volume per active user (USD)',
                'cost_per_user': 'Operational cost per active user (USD)'
            },
            'funding_metrics': {
                'funding_round': 'Whether company raised funding this quarter (Boolean)',
                'funding_stage': 'Stage of funding round (Pre-Seed, Seed, Series A, etc.)',
                'funding_amount': 'Amount raised in funding round (USD)',
                'total_funding_to_date': 'Cumulative funding raised (USD)'
            },
            'distress_indicators': {
                'is_distressed': 'Whether company is currently in distress (Boolean) - DEPENDENT VARIABLE',
                'distress_type': 'Type of distress (closure, acquisition, regulatory_action, severe_downturn)',
                'regulatory_action': 'Whether company faced regulatory sanctions (Boolean)',
                'closure_risk': 'Risk score for company closure (0-1)',
                'acquisition_risk': 'Risk score for distressed acquisition (0-1)'
            }
        }

def main():
    """Main function to generate and save the dataset."""
    print("=== FinTech Early Warning Dataset Generator ===")
    print("Research Topic: FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies")
    print()
    
    # Initialize generator
    generator = FinTechDatasetGenerator()
    
    # Generate dataset
    dataset, metadata = generator.generate_dataset(n_companies=150)
    
    # Save dataset
    print("Saving dataset...")
    dataset.to_csv('fintech_distress_dataset.csv', index=False)
    
    # Save metadata
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Generate summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total observations: {len(dataset):,}")
    print(f"Unique companies: {dataset['company_id'].nunique()}")
    print(f"Countries covered: {dataset['country'].nunique()}")
    print(f"FinTech types: {dataset['fintech_type'].nunique()}")
    print(f"Date range: {dataset['quarter'].min()} to {dataset['quarter'].max()}")
    print(f"Companies with distress events: {dataset[dataset['is_distressed'] == True]['company_id'].nunique()}")
    print(f"Total distress observations: {len(dataset[dataset['is_distressed'] == True])}")
    
    # Country distribution
    print("\n=== Country Distribution ===")
    country_dist = dataset['country'].value_counts()
    for country, count in country_dist.head(10).items():
        print(f"{country}: {count} observations")
    
    # FinTech type distribution
    print("\n=== FinTech Type Distribution ===")
    type_dist = dataset['fintech_type'].value_counts()
    for ftype, count in type_dist.head(10).items():
        print(f"{ftype}: {count} observations")
    
    # Distress type distribution
    print("\n=== Distress Type Distribution ===")
    distress_dist = dataset[dataset['is_distressed'] == True]['distress_type'].value_counts()
    for dtype, count in distress_dist.items():
        print(f"{dtype}: {count} observations")
    
    print(f"\nDataset saved as 'fintech_distress_dataset.csv'")
    print(f"Metadata saved as 'dataset_metadata.json'")
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    main()