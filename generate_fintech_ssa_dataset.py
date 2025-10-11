"""
FinTech Early Warning Model Dataset Generator for Sub-Sahara Africa
Research Topic: FinTech Risk Early Warning System in SSA Economies

This script generates a comprehensive synthetic dataset for FinTech companies
operating in Sub-Sahara Africa, including dependent and independent variables
for early warning model development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_COMPANIES = 150  # Number of FinTech companies
NUM_QUARTERS = 16  # 4 years of quarterly data
START_DATE = datetime(2020, 1, 1)

# SSA Countries and their characteristics
SSA_COUNTRIES = {
    'Nigeria': {'market_size': 1.0, 'regulatory_strength': 0.7, 'economic_stability': 0.6},
    'Kenya': {'market_size': 0.4, 'regulatory_strength': 0.8, 'economic_stability': 0.7},
    'South Africa': {'market_size': 0.8, 'regulatory_strength': 0.9, 'economic_stability': 0.75},
    'Ghana': {'market_size': 0.3, 'regulatory_strength': 0.65, 'economic_stability': 0.6},
    'Uganda': {'market_size': 0.2, 'regulatory_strength': 0.6, 'economic_stability': 0.55},
    'Tanzania': {'market_size': 0.25, 'regulatory_strength': 0.6, 'economic_stability': 0.6},
    'Rwanda': {'market_size': 0.15, 'regulatory_strength': 0.75, 'economic_stability': 0.7},
    'Senegal': {'market_size': 0.18, 'regulatory_strength': 0.65, 'economic_stability': 0.65},
    'Zambia': {'market_size': 0.2, 'regulatory_strength': 0.6, 'economic_stability': 0.5},
    'Ethiopia': {'market_size': 0.5, 'regulatory_strength': 0.55, 'economic_stability': 0.5}
}

# FinTech Types
FINTECH_TYPES = [
    'Mobile Money',
    'Payment Gateway',
    'Digital Lending',
    'Remittance',
    'Insurance Tech',
    'Investment Platform',
    'Digital Banking',
    'Crypto Exchange'
]

# Company name prefixes and suffixes for realistic names
NAME_PREFIXES = ['M-', 'Digi', 'Quick', 'Easy', 'Smart', 'Pay', 'Cash', 'Instant', 'Safe', 
                 'Trust', 'Swift', 'Flash', 'Prime', 'Wave', 'Zap']
NAME_SUFFIXES = ['Pay', 'Cash', 'Wallet', 'Money', 'Bank', 'Finance', 'Tech', 'Connect', 
                 'Link', 'Hub', 'Express', 'Direct', 'Plus', 'Pro']

def generate_company_name():
    """Generate realistic FinTech company names"""
    if random.random() < 0.3:
        return random.choice(NAME_PREFIXES) + random.choice(NAME_SUFFIXES)
    else:
        return random.choice(NAME_PREFIXES + NAME_SUFFIXES) + random.choice(['', 'Tech', 'Africa', 'Digital'])

def calculate_company_health_score(company_profile):
    """Calculate a base health score for a company based on its profile"""
    score = 0.5  # Base score
    
    # Country factors
    score += company_profile['country_economic_stability'] * 0.2
    score += company_profile['country_regulatory_strength'] * 0.15
    
    # Company factors
    if company_profile['company_age'] > 5:
        score += 0.15
    elif company_profile['company_age'] > 3:
        score += 0.10
    elif company_profile['company_age'] > 1:
        score += 0.05
    
    # Type factors - some types are more stable
    stable_types = ['Mobile Money', 'Payment Gateway', 'Digital Banking']
    if company_profile['fintech_type'] in stable_types:
        score += 0.10
    
    # Market size
    score += company_profile['country_market_size'] * 0.1
    
    # Add some randomness
    score += np.random.normal(0, 0.1)
    
    return np.clip(score, 0.1, 0.95)

def generate_company_profiles(num_companies):
    """Generate profiles for FinTech companies"""
    companies = []
    
    for i in range(num_companies):
        country = random.choice(list(SSA_COUNTRIES.keys()))
        fintech_type = random.choice(FINTECH_TYPES)
        company_age = np.random.exponential(3) + 1  # Most companies are young
        
        profile = {
            'company_id': f'FT{i+1:03d}',
            'company_name': generate_company_name(),
            'country': country,
            'fintech_type': fintech_type,
            'company_age': round(company_age, 1),
            'country_market_size': SSA_COUNTRIES[country]['market_size'],
            'country_regulatory_strength': SSA_COUNTRIES[country]['regulatory_strength'],
            'country_economic_stability': SSA_COUNTRIES[country]['economic_stability'],
            'founding_year': 2020 - int(company_age),
            'initial_funding_stage': random.choice(['Seed', 'Seed', 'Series A', 'Series A', 'Series B', 'Bootstrapped', 'Bootstrapped'])
        }
        
        # Calculate base health score
        profile['base_health_score'] = calculate_company_health_score(profile)
        
        # Determine if company will fail during observation period
        # Companies with lower health scores more likely to fail
        failure_probability = (1 - profile['base_health_score']) * 0.4  # Max 40% failure rate
        profile['will_fail'] = np.random.random() < failure_probability
        
        if profile['will_fail']:
            # Determine when failure occurs (which quarter)
            profile['failure_quarter'] = random.randint(8, NUM_QUARTERS)
        else:
            profile['failure_quarter'] = None
            
        companies.append(profile)
    
    return companies

def generate_time_series_data(companies, num_quarters):
    """Generate quarterly time series data for all companies"""
    data = []
    
    for company in companies:
        base_health = company['base_health_score']
        failure_quarter = company['failure_quarter']
        
        # Initial scale based on country market size and company type
        base_users = company['country_market_size'] * 100000 * np.random.uniform(0.1, 2.0)
        base_transaction_value = base_users * np.random.uniform(50, 500)
        
        for quarter in range(1, num_quarters + 1):
            # Calculate quarter date
            quarter_date = START_DATE + timedelta(days=90 * (quarter - 1))
            
            # Determine if company has failed by this quarter
            has_failed = failure_quarter is not None and quarter >= failure_quarter
            
            # Health deteriorates for failing companies
            if failure_quarter and quarter >= failure_quarter - 4:
                # Start showing distress 4 quarters before failure
                quarters_to_failure = failure_quarter - quarter
                health_multiplier = max(0.2, 1 - (4 - quarters_to_failure) * 0.2)
            else:
                health_multiplier = 1.0
            
            # Growth trends
            if has_failed:
                # Failed companies show negative metrics
                growth_rate = -0.3
                revenue_multiplier = 0.3
            else:
                growth_rate = (base_health - 0.3) * 0.15  # Healthier companies grow faster
                revenue_multiplier = 1.0
            
            # Add seasonality and noise
            seasonality = 1 + 0.1 * np.sin(2 * np.pi * quarter / 4)
            noise = np.random.normal(1, 0.1)
            
            # Calculate metrics
            quarter_growth = (1 + growth_rate) ** quarter * seasonality * noise * health_multiplier
            
            # Active Users
            active_users = int(base_users * quarter_growth)
            active_users = max(1000, active_users)  # Minimum users
            
            # Transaction metrics
            transaction_volume = base_transaction_value * quarter_growth * revenue_multiplier
            transaction_count = int(active_users * np.random.uniform(2, 8) * seasonality)
            avg_transaction_value = transaction_volume / transaction_count if transaction_count > 0 else 0
            
            # Revenue (percentage of transaction volume)
            revenue_rate = np.random.uniform(0.01, 0.03)  # 1-3% of transaction volume
            revenue = transaction_volume * revenue_rate
            revenue_growth = (revenue / (base_transaction_value * revenue_rate * 4) - 1) * 100 if quarter > 1 else 0
            
            # Costs and profitability
            fixed_costs = revenue * np.random.uniform(0.3, 0.5)
            variable_costs = revenue * np.random.uniform(0.2, 0.4)
            net_income = revenue - fixed_costs - variable_costs
            
            # Burn rate (for loss-making companies)
            burn_rate = abs(net_income) if net_income < 0 else 0
            
            # Funding
            funding_probability = 0.05 if base_health > 0.6 and not has_failed else 0.01
            received_funding = np.random.random() < funding_probability
            funding_amount = np.random.uniform(500000, 10000000) if received_funding else 0
            
            # Determine funding stage
            if received_funding:
                if quarter < 4:
                    funding_stage = 'Seed'
                elif quarter < 8:
                    funding_stage = 'Series A'
                elif quarter < 12:
                    funding_stage = 'Series B'
                else:
                    funding_stage = 'Series C'
            else:
                funding_stage = company['initial_funding_stage']
            
            # Number of agents (for mobile money)
            if company['fintech_type'] == 'Mobile Money':
                num_agents = int(active_users / 100 * np.random.uniform(0.5, 2))
            else:
                num_agents = 0
            
            # Customer metrics
            cac = revenue * np.random.uniform(0.1, 0.3) / (active_users * 0.2)  # Cost to acquire 20% of users
            churn_rate = np.random.uniform(0.05, 0.15) * (2 - base_health)  # Higher churn for unhealthy companies
            if has_failed:
                churn_rate = np.random.uniform(0.3, 0.6)  # Very high churn for failed companies
            
            # Regulatory issues
            regulatory_sanction = 0
            if base_health < 0.4 and np.random.random() < 0.05:
                regulatory_sanction = 1
            if has_failed and np.random.random() < 0.3:
                regulatory_sanction = 1
            
            # Dependent Variables
            # Binary: 1 if failed, 0 otherwise
            fintech_failure = 1 if has_failed else 0
            
            # Distress indicator (1 if showing signs of distress)
            distress_indicators = 0
            if net_income < -revenue * 0.5:  # Large losses
                distress_indicators += 1
            if revenue_growth < -20 and quarter > 4:  # Significant revenue decline
                distress_indicators += 1
            if churn_rate > 0.25:  # High churn
                distress_indicators += 1
            if burn_rate > revenue:  # Burning more than earning
                distress_indicators += 1
                
            fintech_distress = 1 if distress_indicators >= 2 else 0
            
            # Create record
            record = {
                # Identifiers
                'company_id': company['company_id'],
                'company_name': company['company_name'],
                'quarter': quarter,
                'year': quarter_date.year,
                'quarter_of_year': ((quarter - 1) % 4) + 1,
                'date': quarter_date.strftime('%Y-%m-%d'),
                'country': company['country'],
                'fintech_type': company['fintech_type'],
                'company_age_years': round(company['company_age'] + (quarter - 1) / 4, 2),
                
                # DEPENDENT VARIABLES
                'fintech_failure': fintech_failure,  # Binary: 1 if failed/insolvent
                'fintech_distress': fintech_distress,  # Binary: 1 if showing distress signs
                'regulatory_sanction': regulatory_sanction,  # Binary: 1 if sanctioned
                
                # INDEPENDENT VARIABLES - Financial Performance
                'revenue_usd': round(revenue, 2),
                'revenue_growth_pct': round(revenue_growth, 2),
                'net_income_usd': round(net_income, 2),
                'profit_margin_pct': round((net_income / revenue * 100) if revenue > 0 else -100, 2),
                'burn_rate_usd': round(burn_rate, 2),
                'funding_amount_usd': round(funding_amount, 2),
                'funding_stage': funding_stage,
                'total_funding_to_date_usd': round(funding_amount, 2),  # Simplified
                
                # INDEPENDENT VARIABLES - Operational Metrics
                'active_users': active_users,
                'user_growth_pct': round((active_users / base_users - 1) * 100, 2),
                'transaction_volume_usd': round(transaction_volume, 2),
                'transaction_count': transaction_count,
                'avg_transaction_value_usd': round(avg_transaction_value, 2),
                'num_agents': num_agents,
                'customer_acquisition_cost_usd': round(cac, 2),
                'customer_churn_rate_pct': round(churn_rate * 100, 2),
                
                # Additional context variables
                'country_market_size_index': company['country_market_size'],
                'country_regulatory_strength_index': company['country_regulatory_strength'],
                'country_economic_stability_index': company['country_economic_stability'],
                'quarters_since_last_funding': quarter % 4,  # Simplified
            }
            
            data.append(record)
    
    return pd.DataFrame(data)

def generate_summary_statistics(df):
    """Generate summary statistics about the dataset"""
    summary = {
        'dataset_info': {
            'total_records': len(df),
            'num_companies': df['company_id'].nunique(),
            'num_quarters': df['quarter'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'countries': df['country'].unique().tolist(),
            'fintech_types': df['fintech_type'].unique().tolist()
        },
        'dependent_variables': {
            'total_failures': int(df['fintech_failure'].sum()),
            'failure_rate_pct': round(df.groupby('company_id')['fintech_failure'].max().mean() * 100, 2),
            'companies_in_distress': int(df['fintech_distress'].sum()),
            'regulatory_sanctions': int(df['regulatory_sanction'].sum())
        },
        'company_distribution': {
            'by_country': df.groupby('country')['company_id'].nunique().to_dict(),
            'by_type': df.groupby('fintech_type')['company_id'].nunique().to_dict()
        },
        'financial_metrics_summary': {
            'avg_revenue_usd': round(df['revenue_usd'].mean(), 2),
            'median_revenue_usd': round(df['revenue_usd'].median(), 2),
            'avg_active_users': int(df['active_users'].mean()),
            'total_transaction_volume_usd': round(df['transaction_volume_usd'].sum(), 2)
        }
    }
    
    return summary

def main():
    """Main function to generate the dataset"""
    print("=" * 80)
    print("FinTech Early Warning Model Dataset Generator")
    print("Sub-Sahara Africa Economies")
    print("=" * 80)
    print()
    
    # Generate company profiles
    print(f"Generating {NUM_COMPANIES} FinTech company profiles...")
    companies = generate_company_profiles(NUM_COMPANIES)
    print(f"✓ Generated {len(companies)} companies")
    print(f"  - Companies that will fail: {sum(1 for c in companies if c['will_fail'])}")
    print()
    
    # Generate time series data
    print(f"Generating {NUM_QUARTERS} quarters of time series data...")
    df = generate_time_series_data(companies, NUM_QUARTERS)
    print(f"✓ Generated {len(df)} records")
    print()
    
    # Generate summary statistics
    print("Generating summary statistics...")
    summary = generate_summary_statistics(df)
    print("✓ Summary statistics generated")
    print()
    
    # Save outputs
    print("Saving datasets...")
    
    # 1. Main dataset (CSV)
    df.to_csv('fintech_ssa_dataset.csv', index=False)
    print("✓ Saved: fintech_ssa_dataset.csv")
    
    # 2. Excel with multiple sheets
    with pd.ExcelWriter('fintech_ssa_dataset.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Full Dataset', index=False)
        
        # Summary by company
        company_summary = df.groupby('company_id').agg({
            'company_name': 'first',
            'country': 'first',
            'fintech_type': 'first',
            'fintech_failure': 'max',
            'fintech_distress': 'sum',
            'regulatory_sanction': 'sum',
            'revenue_usd': 'mean',
            'active_users': 'mean',
            'transaction_volume_usd': 'sum'
        }).reset_index()
        company_summary.to_excel(writer, sheet_name='Company Summary', index=False)
        
        # Failed companies
        failed_companies = df[df['fintech_failure'] == 1].groupby('company_id').first().reset_index()
        failed_companies.to_excel(writer, sheet_name='Failed Companies', index=False)
    
    print("✓ Saved: fintech_ssa_dataset.xlsx")
    
    # 3. JSON format
    df.to_json('fintech_ssa_dataset.json', orient='records', indent=2)
    print("✓ Saved: fintech_ssa_dataset.json")
    
    # 4. Company profiles
    company_df = pd.DataFrame(companies)
    company_df.to_csv('fintech_company_profiles.csv', index=False)
    print("✓ Saved: fintech_company_profiles.csv")
    
    # 5. Summary statistics
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved: dataset_summary.json")
    
    # 6. Data dictionary
    data_dictionary = generate_data_dictionary()
    with open('data_dictionary.md', 'w') as f:
        f.write(data_dictionary)
    print("✓ Saved: data_dictionary.md")
    
    print()
    print("=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Total Records: {summary['dataset_info']['total_records']:,}")
    print(f"  Companies: {summary['dataset_info']['num_companies']}")
    print(f"  Time Period: {summary['dataset_info']['date_range']}")
    print(f"  Failed Companies: {summary['dependent_variables']['total_failures']} ({summary['dependent_variables']['failure_rate_pct']}%)")
    print(f"  Companies with Distress: {summary['dependent_variables']['companies_in_distress']}")
    print(f"  Regulatory Sanctions: {summary['dependent_variables']['regulatory_sanctions']}")
    print()
    print("Files created:")
    print("  1. fintech_ssa_dataset.csv - Main dataset")
    print("  2. fintech_ssa_dataset.xlsx - Excel format with multiple sheets")
    print("  3. fintech_ssa_dataset.json - JSON format")
    print("  4. fintech_company_profiles.csv - Company profile information")
    print("  5. dataset_summary.json - Summary statistics")
    print("  6. data_dictionary.md - Variable descriptions")
    print()

def generate_data_dictionary():
    """Generate a comprehensive data dictionary"""
    dictionary = """# FinTech SSA Dataset - Data Dictionary

## Research Context
**Research Topic**: Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

This dataset contains quarterly time-series data for {num_companies} FinTech companies operating across 10 Sub-Sahara African countries from 2020-2023.

## Dataset Structure
- **Total Records**: {total_records}
- **Time Period**: Quarterly data for 16 quarters (Q1 2020 - Q4 2023)
- **Countries Covered**: Nigeria, Kenya, South Africa, Ghana, Uganda, Tanzania, Rwanda, Senegal, Zambia, Ethiopia

## Variable Definitions

### Identifier Variables

| Variable | Type | Description |
|----------|------|-------------|
| company_id | String | Unique identifier for each FinTech company (FT001-FT150) |
| company_name | String | Name of the FinTech company |
| quarter | Integer | Sequential quarter number (1-16) |
| year | Integer | Year of observation |
| quarter_of_year | Integer | Quarter within year (1-4) |
| date | Date | Date of the quarter (YYYY-MM-DD) |
| country | String | Country where FinTech operates |
| fintech_type | String | Type of FinTech service (Mobile Money, Payment Gateway, Digital Lending, etc.) |
| company_age_years | Float | Age of company in years at the time of observation |

### Dependent Variables (What we're predicting)

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| **fintech_failure** | Binary | 0-1 | Primary DV: 1 if company has failed/become insolvent/closed, 0 otherwise |
| **fintech_distress** | Binary | 0-1 | 1 if company shows signs of distress (multiple negative indicators), 0 otherwise |
| **regulatory_sanction** | Binary | 0-1 | 1 if company received regulatory sanction/fine/suspension in this quarter, 0 otherwise |

### Independent Variables - Financial Performance

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| revenue_usd | Float | USD | Quarterly revenue in US Dollars |
| revenue_growth_pct | Float | Percentage | Year-over-year revenue growth rate |
| net_income_usd | Float | USD | Quarterly net income (profit/loss) |
| profit_margin_pct | Float | Percentage | Net profit margin (net_income / revenue * 100) |
| burn_rate_usd | Float | USD | Cash burn rate for loss-making companies |
| funding_amount_usd | Float | USD | Amount of funding received in this quarter (0 if none) |
| funding_stage | String | Category | Current funding stage (Seed, Series A, B, C, Bootstrapped) |
| total_funding_to_date_usd | Float | USD | Cumulative funding received to date |

### Independent Variables - Operational Metrics

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| active_users | Integer | Count | Number of active users/customers in the quarter |
| user_growth_pct | Float | Percentage | User growth rate compared to baseline |
| transaction_volume_usd | Float | USD | Total value of transactions processed in the quarter |
| transaction_count | Integer | Count | Total number of transactions in the quarter |
| avg_transaction_value_usd | Float | USD | Average transaction size (volume/count) |
| num_agents | Integer | Count | Number of agents (for mobile money operators; 0 for other types) |
| customer_acquisition_cost_usd | Float | USD | Cost to acquire each new customer (CAC) |
| customer_churn_rate_pct | Float | Percentage | Customer churn rate (% of customers lost) |

### Contextual Variables (Country-level indicators)

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| country_market_size_index | Float | 0-1 | Relative market size indicator for the country |
| country_regulatory_strength_index | Float | 0-1 | Regulatory framework strength (higher = stronger) |
| country_economic_stability_index | Float | 0-1 | Economic stability indicator (higher = more stable) |
| quarters_since_last_funding | Integer | 0-4 | Number of quarters since last funding round |

## Data Notes

### Dependent Variable Details

1. **fintech_failure**: 
   - This is the primary outcome variable for building early warning models
   - A value of 1 indicates the company has permanently ceased operations
   - Once a company fails (1), it remains failed in subsequent quarters
   - Approximately {failure_rate}% of companies in the dataset experience failure

2. **fintech_distress**:
   - Leading indicator that may precede failure
   - Coded as 1 when company exhibits 2+ distress signals:
     * Large losses (net income < -50% of revenue)
     * Significant revenue decline (> 20% drop)
     * High customer churn (> 25%)
     * High burn rate (burning more than earning)

3. **regulatory_sanction**:
   - Indicates regulatory intervention
   - Can be a warning sign of operational or compliance issues
   - More common in companies with low health scores

### Data Quality & Limitations

1. **Synthetic Data**: This is a fabricated dataset designed for research and model development. While based on realistic parameters and relationships, it does not represent actual company data.

2. **Data Relationships**: The data includes realistic correlations:
   - Failed companies show deteriorating metrics in quarters leading up to failure
   - Healthier companies (better regulatory environment, larger markets) tend to perform better
   - Seasonal patterns are included in transaction volumes

3. **Missing Data**: This dataset has complete records (no missing values) to facilitate initial model development. Real-world data would have substantial missing values, especially for private company financial data.

## Suggested Usage

### For Early Warning Models:
- Use `fintech_failure` as the primary target variable
- Consider `fintech_distress` as an intermediate outcome
- Build models to predict failure 1-4 quarters in advance
- Use lagged variables (previous quarter metrics) as predictors

### Feature Engineering Suggestions:
- Create trend variables (3-quarter moving averages)
- Calculate quarter-over-quarter changes for key metrics
- Create interaction terms (e.g., burn_rate * funding_stage)
- Develop composite health scores from multiple indicators

### Model Development:
- Split data temporally (train on early periods, test on later)
- Consider company-level fixed effects
- Account for country-level clustering
- Handle class imbalance (failures are relatively rare)

## Data Sources Simulated

This synthetic dataset emulates data that would be collected from:
1. **Company Reports**: Financial statements, earnings reports
2. **Regulatory Filings**: Central Bank payment system statistics
3. **VC Databases**: Crunchbase, Disrupt Africa, Partech Africa
4. **Industry Reports**: GSMA Mobile Money reports, World Bank
5. **App Stores**: Download and usage statistics

## Citation

If using this dataset, please cite:
```
FinTech Early Warning Model Dataset for Sub-Sahara Africa
Generated: {generation_date}
Research Topic: FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies
```

## Contact & Support

For questions about variable definitions or data generation methodology, refer to the generation script: `generate_fintech_ssa_dataset.py`
"""
    
    return dictionary

if __name__ == "__main__":
    main()
