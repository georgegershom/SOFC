"""
Sample Analysis Script for FinTech SSA Distress Dataset
This script demonstrates various analyses that can be performed on the dataset
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the dataset for analysis"""
    print("Loading FinTech SSA dataset...")
    df = pd.read_csv('fintech_ssa_distress_dataset.csv')
    
    # Convert date columns
    df['quarter_date'] = pd.to_datetime(df['quarter_date'])
    df['founding_date'] = pd.to_datetime(df['founding_date'])
    
    print(f"Dataset loaded: {df.shape[0]:,} observations, {df['company_id'].nunique()} companies")
    return df

def distress_analysis(df):
    """Analyze distress patterns in the dataset"""
    print("\n" + "="*60)
    print("DISTRESS ANALYSIS")
    print("="*60)
    
    # Overall distress rate
    distress_rate = df.groupby('company_id')['distress_flag'].max().mean()
    print(f"\nOverall distress rate: {distress_rate:.1%}")
    
    # Distress by country
    print("\nDistress rates by country:")
    country_distress = df.groupby(['country', 'company_id'])['distress_flag'].max().reset_index()
    country_summary = country_distress.groupby('country')['distress_flag'].agg(['mean', 'sum', 'count'])
    country_summary.columns = ['Distress_Rate', 'Distressed_Companies', 'Total_Companies']
    country_summary['Distress_Rate'] = country_summary['Distress_Rate'].apply(lambda x: f"{x:.1%}")
    print(country_summary.sort_values('Distressed_Companies', ascending=False))
    
    # Distress by category
    print("\nDistress rates by FinTech category:")
    category_distress = df.groupby(['category', 'company_id'])['distress_flag'].max().reset_index()
    category_summary = category_distress.groupby('category')['distress_flag'].agg(['mean', 'sum', 'count'])
    category_summary.columns = ['Distress_Rate', 'Distressed_Companies', 'Total_Companies']
    category_summary['Distress_Rate'] = category_summary['Distress_Rate'].apply(lambda x: f"{x:.1%}")
    print(category_summary.sort_values('Distressed_Companies', ascending=False))
    
    # Distress by funding stage
    print("\nDistress rates by initial funding stage:")
    stage_distress = df.groupby(['initial_stage', 'company_id'])['distress_flag'].max().reset_index()
    stage_summary = stage_distress.groupby('initial_stage')['distress_flag'].agg(['mean', 'sum', 'count'])
    stage_summary.columns = ['Distress_Rate', 'Distressed_Companies', 'Total_Companies']
    stage_summary['Distress_Rate'] = stage_summary['Distress_Rate'].apply(lambda x: f"{x:.1%}")
    print(stage_summary.sort_values('Distress_Rate', ascending=False))

def early_warning_effectiveness(df):
    """Analyze the effectiveness of early warning signals"""
    print("\n" + "="*60)
    print("EARLY WARNING SIGNAL ANALYSIS")
    print("="*60)
    
    # Get companies that eventually fail
    failed_companies = df[df['distress_flag'] == 1]['company_id'].unique()
    
    # Check if early warning was triggered before failure
    early_warnings = []
    for company in failed_companies:
        company_data = df[df['company_id'] == company].sort_values('quarter')
        first_distress = company_data[company_data['distress_flag'] == 1]['quarter'].min()
        
        # Check if there was an early warning 1-2 quarters before
        if first_distress > 2:
            pre_distress = company_data[company_data['quarter'] < first_distress]
            had_warning = pre_distress['early_warning_signal'].max()
            early_warnings.append(had_warning)
    
    if early_warnings:
        warning_rate = np.mean(early_warnings)
        print(f"\nEarly warning effectiveness:")
        print(f"- Companies with early warning before distress: {warning_rate:.1%}")
        print(f"- Total distressed companies analyzed: {len(early_warnings)}")
    
    # False positive rate
    healthy_companies = df[~df['company_id'].isin(failed_companies)]
    false_positive_rate = healthy_companies['early_warning_signal'].mean()
    print(f"\nFalse positive rate: {false_positive_rate:.1%}")
    
    # Warning lead time analysis
    print("\nAverage quarters between first warning and distress:")
    lead_times = []
    for company in failed_companies:
        company_data = df[df['company_id'] == company].sort_values('quarter')
        first_warning = company_data[company_data['early_warning_signal'] == 1]['quarter'].min()
        first_distress = company_data[company_data['distress_flag'] == 1]['quarter'].min()
        
        if pd.notna(first_warning) and first_warning < first_distress:
            lead_times.append(first_distress - first_warning)
    
    if lead_times:
        print(f"- Average lead time: {np.mean(lead_times):.1f} quarters")
        print(f"- Median lead time: {np.median(lead_times):.0f} quarters")

def risk_factor_analysis(df):
    """Analyze key risk factors associated with distress"""
    print("\n" + "="*60)
    print("RISK FACTOR ANALYSIS")
    print("="*60)
    
    # Compare metrics between distressed and healthy companies
    distressed = df[df['distress_flag'] == 1]
    healthy = df[df['distress_flag'] == 0]
    
    print("\nKey metric comparison (Distressed vs Healthy):")
    metrics = ['churn_rate', 'profitability_ratio', 'revenue_growth_qoq', 
               'revenue_volatility', 'risk_score']
    
    comparison = pd.DataFrame()
    for metric in metrics:
        comparison.loc[metric, 'Distressed_Mean'] = distressed[metric].mean()
        comparison.loc[metric, 'Healthy_Mean'] = healthy[metric].mean()
        comparison.loc[metric, 'Difference'] = comparison.loc[metric, 'Distressed_Mean'] - comparison.loc[metric, 'Healthy_Mean']
        comparison.loc[metric, 'Ratio'] = comparison.loc[metric, 'Distressed_Mean'] / comparison.loc[metric, 'Healthy_Mean']
    
    print(comparison.round(3))
    
    # Regulatory impact
    print("\nRegulatory sanctions impact:")
    sanctioned = df[df['regulatory_sanction'] == 1]
    print(f"- Companies with sanctions: {sanctioned['company_id'].nunique()}")
    print(f"- Distress rate among sanctioned: {sanctioned['distress_flag'].mean():.1%}")
    print(f"- Overall distress rate: {df['distress_flag'].mean():.1%}")

def growth_trajectory_analysis(df):
    """Analyze growth trajectories of successful vs failed companies"""
    print("\n" + "="*60)
    print("GROWTH TRAJECTORY ANALYSIS")
    print("="*60)
    
    # Identify successful vs failed companies
    company_outcomes = df.groupby('company_id')['distress_flag'].max().reset_index()
    successful = company_outcomes[company_outcomes['distress_flag'] == 0]['company_id']
    failed = company_outcomes[company_outcomes['distress_flag'] == 1]['company_id']
    
    print(f"\nCompany outcomes:")
    print(f"- Successful companies: {len(successful)}")
    print(f"- Failed companies: {len(failed)}")
    
    # Average growth rates
    print("\nAverage quarterly growth rates:")
    
    successful_growth = df[df['company_id'].isin(successful)].groupby('quarter')[['revenue', 'active_users']].mean()
    failed_growth = df[df['company_id'].isin(failed)].groupby('quarter')[['revenue', 'active_users']].mean()
    
    # Calculate growth rates
    successful_revenue_growth = successful_growth['revenue'].pct_change().mean()
    failed_revenue_growth = failed_growth['revenue'].pct_change().mean()
    successful_user_growth = successful_growth['active_users'].pct_change().mean()
    failed_user_growth = failed_growth['active_users'].pct_change().mean()
    
    print(f"\nSuccessful companies:")
    print(f"- Revenue growth: {successful_revenue_growth:.1%} per quarter")
    print(f"- User growth: {successful_user_growth:.1%} per quarter")
    
    print(f"\nFailed companies:")
    print(f"- Revenue growth: {failed_revenue_growth:.1%} per quarter")
    print(f"- User growth: {failed_user_growth:.1%} per quarter")
    
    # Funding analysis
    print("\nFunding patterns:")
    successful_funding = df[df['company_id'].isin(successful)]['cumulative_funding'].max()
    failed_funding = df[df['company_id'].isin(failed)]['cumulative_funding'].max()
    
    print(f"- Avg total funding (successful): ${successful_funding/len(successful):,.0f}")
    print(f"- Avg total funding (failed): ${failed_funding/len(failed):,.0f}")

def market_dynamics_analysis(df):
    """Analyze market-specific dynamics"""
    print("\n" + "="*60)
    print("MARKET DYNAMICS ANALYSIS")
    print("="*60)
    
    # Market size impact
    print("\nPerformance by market size:")
    market_perf = df.groupby('market_size').agg({
        'revenue': 'mean',
        'active_users': 'mean',
        'distress_flag': 'mean',
        'company_id': 'nunique'
    })
    market_perf.columns = ['Avg_Revenue', 'Avg_Users', 'Distress_Rate', 'Num_Companies']
    market_perf['Avg_Revenue'] = market_perf['Avg_Revenue'].apply(lambda x: f"${x:,.0f}")
    market_perf['Avg_Users'] = market_perf['Avg_Users'].apply(lambda x: f"{x:,.0f}")
    market_perf['Distress_Rate'] = market_perf['Distress_Rate'].apply(lambda x: f"{x:.1%}")
    print(market_perf)
    
    # Regulatory environment impact
    print("\nPerformance by regulatory environment:")
    reg_perf = df.groupby('regulatory_env').agg({
        'revenue': 'mean',
        'distress_flag': 'mean',
        'regulatory_sanction': 'mean',
        'company_id': 'nunique'
    })
    reg_perf.columns = ['Avg_Revenue', 'Distress_Rate', 'Sanction_Rate', 'Num_Companies']
    reg_perf['Avg_Revenue'] = reg_perf['Avg_Revenue'].apply(lambda x: f"${x:,.0f}")
    reg_perf['Distress_Rate'] = reg_perf['Distress_Rate'].apply(lambda x: f"{x:.1%}")
    reg_perf['Sanction_Rate'] = reg_perf['Sanction_Rate'].apply(lambda x: f"{x:.2%}")
    print(reg_perf)

def create_ml_ready_dataset(df):
    """Create a dataset ready for machine learning"""
    print("\n" + "="*60)
    print("CREATING ML-READY DATASET")
    print("="*60)
    
    # Select features for ML
    feature_cols = [
        'revenue', 'revenue_growth_qoq', 'revenue_volatility',
        'profitability_ratio', 'burn_rate', 'cumulative_funding',
        'active_users', 'user_growth_qoq', 'transaction_volume',
        'churn_rate', 'customer_acquisition_cost', 'risk_score'
    ]
    
    # Create lagged features (using previous quarter's data to predict next quarter)
    ml_data = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company].sort_values('quarter')
        
        for i in range(1, len(company_data)):
            current = company_data.iloc[i]
            previous = company_data.iloc[i-1]
            
            record = {
                'company_id': company,
                'quarter': current['quarter'],
                'target_distress': current['distress_flag'],
                'target_failure_imminent': current['failure_imminent']
            }
            
            # Add lagged features
            for col in feature_cols:
                record[f'{col}_lag1'] = previous[col]
            
            # Add static features
            record['country'] = current['country']
            record['category'] = current['category']
            record['market_size'] = current['market_size']
            record['regulatory_env'] = current['regulatory_env']
            
            ml_data.append(record)
    
    ml_df = pd.DataFrame(ml_data)
    
    # Save ML-ready dataset
    ml_df.to_csv('fintech_ssa_ml_dataset.csv', index=False)
    print(f"\nML-ready dataset created: fintech_ssa_ml_dataset.csv")
    print(f"Shape: {ml_df.shape}")
    print(f"Features: {len([c for c in ml_df.columns if 'lag' in c])}")
    print(f"Target variable distribution:")
    print(f"- Distress cases: {ml_df['target_distress'].sum()} ({ml_df['target_distress'].mean():.1%})")
    print(f"- Failure imminent cases: {ml_df['target_failure_imminent'].sum()} ({ml_df['target_failure_imminent'].mean():.1%})")
    
    return ml_df

def main():
    """Run all analyses"""
    print("="*60)
    print("FINTECH SSA DISTRESS DATASET - COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run analyses
    distress_analysis(df)
    early_warning_effectiveness(df)
    risk_factor_analysis(df)
    growth_trajectory_analysis(df)
    market_dynamics_analysis(df)
    
    # Create ML-ready dataset
    ml_df = create_ml_ready_dataset(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("1. fintech_ssa_distress_dataset.csv - Main dataset")
    print("2. fintech_ssa_ml_dataset.csv - ML-ready dataset with lagged features")
    print("3. fintech_ssa_data_dictionary.csv - Variable descriptions")

if __name__ == "__main__":
    main()