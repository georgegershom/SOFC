"""
FinTech Early Warning Model Dataset Generator for Sub-Saharan Africa
This script generates synthetic but realistic FinTech company data for SSA economies
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FinTechSSADataGenerator:
    def __init__(self, n_companies=500, n_quarters=20):
        """
        Initialize the FinTech SSA Dataset Generator
        
        Parameters:
        n_companies: Number of FinTech companies to generate
        n_quarters: Number of quarterly observations per company (5 years = 20 quarters)
        """
        self.n_companies = n_companies
        self.n_quarters = n_quarters
        
        # SSA countries with FinTech presence
        self.countries = {
            'Nigeria': {'weight': 0.25, 'market_size': 'large', 'regulatory_strength': 'medium'},
            'Kenya': {'weight': 0.20, 'market_size': 'large', 'regulatory_strength': 'high'},
            'South Africa': {'weight': 0.15, 'market_size': 'large', 'regulatory_strength': 'high'},
            'Ghana': {'weight': 0.10, 'market_size': 'medium', 'regulatory_strength': 'medium'},
            'Uganda': {'weight': 0.08, 'market_size': 'medium', 'regulatory_strength': 'low'},
            'Tanzania': {'weight': 0.07, 'market_size': 'medium', 'regulatory_strength': 'low'},
            'Rwanda': {'weight': 0.05, 'market_size': 'small', 'regulatory_strength': 'medium'},
            'Senegal': {'weight': 0.04, 'market_size': 'small', 'regulatory_strength': 'low'},
            'Ivory Coast': {'weight': 0.03, 'market_size': 'small', 'regulatory_strength': 'low'},
            'Ethiopia': {'weight': 0.03, 'market_size': 'medium', 'regulatory_strength': 'low'}
        }
        
        # FinTech categories
        self.categories = {
            'Mobile Money': 0.35,
            'Digital Banking': 0.20,
            'Payment Processing': 0.15,
            'Lending': 0.12,
            'Insurance': 0.08,
            'Investment': 0.05,
            'Cryptocurrency': 0.05
        }
        
        # Company stages
        self.stages = {
            'Seed': {'prob': 0.25, 'avg_funding': 500000, 'failure_rate': 0.35},
            'Series A': {'prob': 0.30, 'avg_funding': 3000000, 'failure_rate': 0.25},
            'Series B': {'prob': 0.20, 'avg_funding': 15000000, 'failure_rate': 0.15},
            'Series C+': {'prob': 0.15, 'avg_funding': 50000000, 'failure_rate': 0.08},
            'Mature': {'prob': 0.10, 'avg_funding': 100000000, 'failure_rate': 0.05}
        }
        
    def generate_company_base_data(self):
        """Generate base company information"""
        companies = []
        
        for i in range(self.n_companies):
            # Select country based on weights
            country = np.random.choice(
                list(self.countries.keys()),
                p=list([c['weight'] for c in self.countries.values()])
            )
            
            # Select category
            category = np.random.choice(
                list(self.categories.keys()),
                p=list(self.categories.values())
            )
            
            # Select initial stage
            stage = np.random.choice(
                list(self.stages.keys()),
                p=[s['prob'] for s in self.stages.values()]
            )
            
            # Company age (in months)
            company_age = np.random.randint(6, 120)
            
            # Generate company ID and name
            company_id = f"FT_{country[:3].upper()}_{i:04d}"
            company_name = f"{category.replace(' ', '')}_{country[:3]}_{i:03d}"
            
            companies.append({
                'company_id': company_id,
                'company_name': company_name,
                'country': country,
                'category': category,
                'initial_stage': stage,
                'founding_date': datetime(2015, 1, 1) + timedelta(days=random.randint(0, 2555)),
                'company_age_months': company_age,
                'market_size': self.countries[country]['market_size'],
                'regulatory_env': self.countries[country]['regulatory_strength']
            })
        
        return pd.DataFrame(companies)
    
    def generate_financial_metrics(self, company_data):
        """Generate financial performance metrics over time"""
        financial_data = []
        
        for _, company in company_data.iterrows():
            # Get base parameters based on company characteristics
            stage_info = self.stages[company['initial_stage']]
            base_failure_rate = stage_info['failure_rate']
            
            # Determine if company will eventually fail
            will_fail = np.random.random() < base_failure_rate
            if will_fail:
                failure_quarter = np.random.randint(8, self.n_quarters)
            else:
                failure_quarter = None
            
            # Initialize starting values based on stage
            if company['initial_stage'] == 'Seed':
                initial_revenue = np.random.uniform(10000, 100000)
                initial_users = np.random.randint(100, 5000)
                initial_burn_rate = np.random.uniform(50000, 200000)
            elif company['initial_stage'] == 'Series A':
                initial_revenue = np.random.uniform(100000, 1000000)
                initial_users = np.random.randint(5000, 50000)
                initial_burn_rate = np.random.uniform(200000, 1000000)
            elif company['initial_stage'] == 'Series B':
                initial_revenue = np.random.uniform(1000000, 10000000)
                initial_users = np.random.randint(50000, 500000)
                initial_burn_rate = np.random.uniform(1000000, 5000000)
            elif company['initial_stage'] == 'Series C+':
                initial_revenue = np.random.uniform(10000000, 50000000)
                initial_users = np.random.randint(500000, 2000000)
                initial_burn_rate = np.random.uniform(5000000, 15000000)
            else:  # Mature
                initial_revenue = np.random.uniform(50000000, 200000000)
                initial_users = np.random.randint(2000000, 10000000)
                initial_burn_rate = np.random.uniform(1000000, 10000000)
            
            # Generate quarterly data
            for quarter in range(self.n_quarters):
                quarter_date = datetime(2019, 1, 1) + timedelta(days=quarter * 90)
                
                # Calculate growth trajectory
                if failure_quarter and quarter >= failure_quarter - 4:
                    # Company is approaching failure - declining metrics
                    growth_multiplier = 0.85 ** (quarter - (failure_quarter - 4))
                    distress_flag = 1 if quarter >= failure_quarter else 0
                else:
                    # Normal growth with some volatility
                    if company['category'] in ['Mobile Money', 'Digital Banking']:
                        base_growth = 1.08  # 8% quarterly growth
                    elif company['category'] in ['Payment Processing', 'Lending']:
                        base_growth = 1.10  # 10% quarterly growth
                    else:
                        base_growth = 1.06  # 6% quarterly growth
                    
                    # Add market-specific adjustments
                    if company['market_size'] == 'large':
                        base_growth *= 1.02
                    elif company['market_size'] == 'small':
                        base_growth *= 0.98
                    
                    growth_multiplier = base_growth ** quarter * np.random.uniform(0.9, 1.1)
                    distress_flag = 0
                
                # Calculate metrics for this quarter
                revenue = initial_revenue * growth_multiplier * np.random.uniform(0.95, 1.05)
                
                # Operating costs (higher for younger companies)
                operating_cost_ratio = 0.8 if company['initial_stage'] == 'Seed' else \
                                       0.7 if company['initial_stage'] == 'Series A' else \
                                       0.6 if company['initial_stage'] == 'Series B' else \
                                       0.5 if company['initial_stage'] == 'Series C+' else 0.4
                
                operating_costs = revenue * operating_cost_ratio * np.random.uniform(0.9, 1.1)
                
                # Net income
                net_income = revenue - operating_costs - (initial_burn_rate * np.random.uniform(0.8, 1.2))
                
                # Funding events (occasional)
                funding_amount = 0
                funding_round = None
                if quarter > 0 and quarter % 6 == 0 and np.random.random() < 0.3:
                    # Potential funding round every 1.5 years with 30% probability
                    if not (failure_quarter and quarter >= failure_quarter - 2):
                        funding_amount = stage_info['avg_funding'] * np.random.uniform(0.5, 2.0)
                        funding_round = company['initial_stage']
                
                # User metrics
                active_users = int(initial_users * growth_multiplier * np.random.uniform(0.95, 1.05))
                
                # Transaction metrics
                if company['category'] in ['Mobile Money', 'Payment Processing', 'Digital Banking']:
                    avg_transactions_per_user = np.random.uniform(5, 20)
                elif company['category'] == 'Lending':
                    avg_transactions_per_user = np.random.uniform(0.5, 2)
                else:
                    avg_transactions_per_user = np.random.uniform(1, 5)
                
                transaction_count = int(active_users * avg_transactions_per_user)
                avg_transaction_value = revenue / max(transaction_count, 1)
                transaction_volume = transaction_count * avg_transaction_value
                
                # Agent network (for applicable categories)
                if company['category'] in ['Mobile Money', 'Digital Banking']:
                    num_agents = int(active_users / np.random.uniform(100, 500))
                else:
                    num_agents = 0
                
                # Customer metrics
                cac = np.random.uniform(5, 50) if company['category'] != 'Mobile Money' else np.random.uniform(1, 10)
                
                # Churn rate (higher when distressed)
                base_churn = np.random.uniform(0.02, 0.08)
                if distress_flag == 1:
                    churn_rate = min(base_churn * 3, 0.25)
                else:
                    churn_rate = base_churn
                
                # Regulatory issues
                regulatory_fine = 0
                regulatory_sanction = 0
                if np.random.random() < 0.02:  # 2% chance of regulatory issue
                    regulatory_fine = np.random.uniform(10000, 1000000)
                    if np.random.random() < 0.3:  # 30% of regulatory issues lead to sanctions
                        regulatory_sanction = 1
                
                # Create record
                financial_data.append({
                    'company_id': company['company_id'],
                    'quarter_date': quarter_date,
                    'quarter': quarter + 1,
                    'revenue': revenue,
                    'revenue_growth_qoq': 0 if quarter == 0 else (revenue / initial_revenue - 1),
                    'operating_costs': operating_costs,
                    'net_income': net_income,
                    'profitability_ratio': net_income / revenue if revenue > 0 else -1,
                    'burn_rate': initial_burn_rate * growth_multiplier * 0.5,  # Burn rate decreases as company matures
                    'funding_amount': funding_amount,
                    'funding_round': funding_round,
                    'cumulative_funding': funding_amount if quarter == 0 else 0,  # Will be calculated later
                    'active_users': active_users,
                    'user_growth_qoq': 0 if quarter == 0 else (active_users / initial_users - 1),
                    'transaction_volume': transaction_volume,
                    'transaction_count': transaction_count,
                    'avg_transaction_value': avg_transaction_value,
                    'num_agents': num_agents,
                    'customer_acquisition_cost': cac,
                    'churn_rate': churn_rate,
                    'regulatory_fine': regulatory_fine,
                    'regulatory_sanction': regulatory_sanction,
                    'distress_flag': distress_flag,
                    'failure_imminent': 1 if failure_quarter and quarter >= failure_quarter - 2 else 0
                })
        
        return pd.DataFrame(financial_data)
    
    def calculate_derived_metrics(self, df):
        """Calculate additional derived metrics and indicators"""
        # Sort by company and quarter
        df = df.sort_values(['company_id', 'quarter'])
        
        # Calculate cumulative funding
        df['cumulative_funding'] = df.groupby('company_id')['funding_amount'].cumsum()
        
        # Calculate quarter-over-quarter growth rates
        for metric in ['revenue', 'active_users', 'transaction_volume']:
            df[f'{metric}_growth_qoq'] = df.groupby('company_id')[metric].pct_change()
        
        # Calculate moving averages
        for metric in ['revenue', 'active_users', 'churn_rate']:
            df[f'{metric}_ma3'] = df.groupby('company_id')[metric].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate volatility measures
        df['revenue_volatility'] = df.groupby('company_id')['revenue_growth_qoq'].rolling(window=4, min_periods=2).std().reset_index(0, drop=True)
        
        # Create composite risk score
        df['risk_score'] = (
            (df['churn_rate'] > 0.1).astype(int) * 0.3 +
            (df['profitability_ratio'] < -0.2).astype(int) * 0.3 +
            (df['revenue_growth_qoq'] < -0.1).astype(int) * 0.2 +
            (df['regulatory_sanction'] == 1).astype(int) * 0.2
        )
        
        # Early warning indicators
        df['early_warning_signal'] = ((df['risk_score'] > 0.5) | 
                                      (df['revenue_volatility'] > 0.3) |
                                      (df['churn_rate'] > 0.15)).astype(int)
        
        return df
    
    def generate_complete_dataset(self):
        """Generate the complete dataset"""
        print("Generating FinTech company base data...")
        company_data = self.generate_company_base_data()
        
        print("Generating financial and operational metrics...")
        financial_data = self.generate_financial_metrics(company_data)
        
        print("Calculating derived metrics...")
        financial_data = self.calculate_derived_metrics(financial_data)
        
        print("Merging datasets...")
        # Merge company and financial data
        complete_data = financial_data.merge(company_data, on='company_id', how='left')
        
        # Reorder columns for better readability
        column_order = [
            # Identifiers
            'company_id', 'company_name', 'country', 'category', 'quarter_date', 'quarter',
            
            # Company characteristics
            'initial_stage', 'founding_date', 'company_age_months', 'market_size', 'regulatory_env',
            
            # Financial metrics
            'revenue', 'revenue_growth_qoq', 'revenue_ma3', 'revenue_volatility',
            'operating_costs', 'net_income', 'profitability_ratio',
            'burn_rate', 'funding_amount', 'funding_round', 'cumulative_funding',
            
            # Operational metrics
            'active_users', 'user_growth_qoq', 'active_users_ma3',
            'transaction_volume', 'transaction_count', 'avg_transaction_value',
            'num_agents', 'customer_acquisition_cost', 'churn_rate', 'churn_rate_ma3',
            
            # Risk indicators
            'regulatory_fine', 'regulatory_sanction',
            'risk_score', 'early_warning_signal',
            
            # Dependent variables
            'distress_flag', 'failure_imminent'
        ]
        
        complete_data = complete_data[column_order]
        
        return complete_data
    
    def generate_data_dictionary(self):
        """Generate a data dictionary for the dataset"""
        data_dict = {
            'Variable Name': [
                'company_id', 'company_name', 'country', 'category', 'quarter_date', 'quarter',
                'initial_stage', 'founding_date', 'company_age_months', 'market_size', 'regulatory_env',
                'revenue', 'revenue_growth_qoq', 'revenue_ma3', 'revenue_volatility',
                'operating_costs', 'net_income', 'profitability_ratio',
                'burn_rate', 'funding_amount', 'funding_round', 'cumulative_funding',
                'active_users', 'user_growth_qoq', 'active_users_ma3',
                'transaction_volume', 'transaction_count', 'avg_transaction_value',
                'num_agents', 'customer_acquisition_cost', 'churn_rate', 'churn_rate_ma3',
                'regulatory_fine', 'regulatory_sanction',
                'risk_score', 'early_warning_signal',
                'distress_flag', 'failure_imminent'
            ],
            'Description': [
                'Unique identifier for each FinTech company',
                'Company name (synthetic)',
                'Country of operation in SSA',
                'FinTech category/vertical',
                'Quarter date for the observation',
                'Quarter number (1-20)',
                'Initial funding stage when company started',
                'Company founding date',
                'Age of company in months',
                'Market size classification (small/medium/large)',
                'Regulatory environment strength (low/medium/high)',
                'Quarterly revenue in USD',
                'Quarter-over-quarter revenue growth rate',
                '3-quarter moving average of revenue',
                'Revenue growth volatility (4-quarter rolling std)',
                'Quarterly operating costs in USD',
                'Quarterly net income in USD',
                'Net income to revenue ratio',
                'Monthly burn rate in USD',
                'Funding received this quarter in USD',
                'Type of funding round (if any)',
                'Total cumulative funding received in USD',
                'Number of active users',
                'Quarter-over-quarter user growth rate',
                '3-quarter moving average of active users',
                'Total transaction volume in USD',
                'Number of transactions',
                'Average transaction value in USD',
                'Number of agents (for applicable categories)',
                'Cost to acquire one customer in USD',
                'Customer churn rate (proportion)',
                '3-quarter moving average of churn rate',
                'Regulatory fine amount in USD (if any)',
                'Binary: 1 if regulatory sanction imposed',
                'Composite risk score (0-1)',
                'Binary: 1 if early warning triggered',
                'Binary: 1 if company in distress',
                'Binary: 1 if failure within 2 quarters'
            ],
            'Data Type': [
                'String', 'String', 'String', 'String', 'Date', 'Integer',
                'String', 'Date', 'Integer', 'String', 'String',
                'Float', 'Float', 'Float', 'Float',
                'Float', 'Float', 'Float',
                'Float', 'Float', 'String', 'Float',
                'Integer', 'Float', 'Float',
                'Float', 'Integer', 'Float',
                'Integer', 'Float', 'Float', 'Float',
                'Float', 'Binary',
                'Float', 'Binary',
                'Binary', 'Binary'
            ],
            'Unit': [
                'ID', 'Name', 'Country', 'Category', 'Date', 'Count',
                'Stage', 'Date', 'Months', 'Category', 'Category',
                'USD', 'Ratio', 'USD', 'Ratio',
                'USD', 'USD', 'Ratio',
                'USD/month', 'USD', 'Category', 'USD',
                'Count', 'Ratio', 'Count',
                'USD', 'Count', 'USD',
                'Count', 'USD', 'Proportion', 'Proportion',
                'USD', '0/1',
                '0-1', '0/1',
                '0/1', '0/1'
            ]
        }
        
        return pd.DataFrame(data_dict)

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("FinTech Early Warning Model Dataset Generator for SSA")
    print("=" * 80)
    
    # Initialize generator
    generator = FinTechSSADataGenerator(n_companies=500, n_quarters=20)
    
    # Generate complete dataset
    print("\nGenerating synthetic FinTech dataset...")
    dataset = generator.generate_complete_dataset()
    
    # Save main dataset
    dataset.to_csv('fintech_ssa_distress_dataset.csv', index=False)
    print(f"\nDataset saved to 'fintech_ssa_distress_dataset.csv'")
    print(f"Shape: {dataset.shape}")
    print(f"Number of companies: {dataset['company_id'].nunique()}")
    print(f"Time period: {dataset['quarter_date'].min()} to {dataset['quarter_date'].max()}")
    
    # Generate and save data dictionary
    data_dict = generator.generate_data_dictionary()
    data_dict.to_csv('fintech_ssa_data_dictionary.csv', index=False)
    print("\nData dictionary saved to 'fintech_ssa_data_dictionary.csv'")
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nCountry Distribution:")
    print(dataset.groupby('country')['company_id'].nunique().sort_values(ascending=False))
    
    print("\nCategory Distribution:")
    print(dataset.groupby('category')['company_id'].nunique().sort_values(ascending=False))
    
    print("\nDistress Statistics:")
    distress_companies = dataset[dataset['distress_flag'] == 1]['company_id'].nunique()
    total_companies = dataset['company_id'].nunique()
    print(f"Companies experiencing distress: {distress_companies} ({distress_companies/total_companies*100:.1f}%)")
    
    print("\nKey Financial Metrics (Latest Quarter):")
    latest_quarter = dataset[dataset['quarter'] == dataset['quarter'].max()]
    print(f"Average Revenue: ${latest_quarter['revenue'].mean():,.0f}")
    print(f"Average Active Users: {latest_quarter['active_users'].mean():,.0f}")
    print(f"Average Churn Rate: {latest_quarter['churn_rate'].mean():.2%}")
    print(f"Companies with Early Warning Signal: {latest_quarter['early_warning_signal'].sum()} ({latest_quarter['early_warning_signal'].mean()*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")