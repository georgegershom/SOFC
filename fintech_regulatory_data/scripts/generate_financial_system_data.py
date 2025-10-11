"""
Financial System & Regulatory Data Generator for Sub-Saharan Africa
This script generates and downloads data for FinTech Early Warning Model research
Category 3: Financial System & Regulatory Data
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# List of Sub-Saharan African countries (ISO codes)
SSA_COUNTRIES = {
    'AGO': 'Angola', 'BEN': 'Benin', 'BWA': 'Botswana', 'BFA': 'Burkina Faso',
    'BDI': 'Burundi', 'CMR': 'Cameroon', 'CPV': 'Cabo Verde', 'CAF': 'Central African Republic',
    'TCD': 'Chad', 'COM': 'Comoros', 'COG': 'Congo', 'COD': 'Democratic Republic of Congo',
    'GNQ': 'Equatorial Guinea', 'ERI': 'Eritrea', 'ETH': 'Ethiopia', 'GAB': 'Gabon',
    'GMB': 'Gambia', 'GHA': 'Ghana', 'GIN': 'Guinea', 'GNB': 'Guinea-Bissau',
    'CIV': 'Ivory Coast', 'KEN': 'Kenya', 'LSO': 'Lesotho', 'LBR': 'Liberia',
    'MDG': 'Madagascar', 'MWI': 'Malawi', 'MLI': 'Mali', 'MRT': 'Mauritania',
    'MUS': 'Mauritius', 'MOZ': 'Mozambique', 'NAM': 'Namibia', 'NER': 'Niger',
    'NGA': 'Nigeria', 'RWA': 'Rwanda', 'STP': 'Sao Tome and Principe', 'SEN': 'Senegal',
    'SYC': 'Seychelles', 'SLE': 'Sierra Leone', 'SOM': 'Somalia', 'ZAF': 'South Africa',
    'SSD': 'South Sudan', 'SDN': 'Sudan', 'SWZ': 'Eswatini', 'TZA': 'Tanzania',
    'TGO': 'Togo', 'UGA': 'Uganda', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
}

# Time period for data
START_YEAR = 2010
END_YEAR = 2024
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']

def generate_time_series(start_value, trend, volatility, length, min_val=None, max_val=None):
    """Generate realistic time series data with trend and volatility"""
    np.random.seed(42)
    values = [start_value]
    for i in range(1, length):
        # Add trend component
        trend_component = trend * (1 + np.random.normal(0, 0.1))
        # Add random walk component
        random_component = np.random.normal(0, volatility)
        # Calculate new value
        new_value = values[-1] * (1 + trend_component + random_component)
        
        # Apply bounds if specified
        if min_val is not None:
            new_value = max(new_value, min_val)
        if max_val is not None:
            new_value = min(new_value, max_val)
        
        values.append(new_value)
    
    return np.array(values)

def generate_banking_sector_data():
    """Generate Banking Sector Health indicators"""
    print("Generating Banking Sector Health data...")
    
    data = []
    
    for country_code, country_name in SSA_COUNTRIES.items():
        # Set country-specific characteristics
        np.random.seed(hash(country_code) % 2**32)
        
        # Base characteristics by country development level
        if country_code in ['ZAF', 'MUS', 'SYC', 'BWA', 'NAM']:  # More developed financial systems
            npl_base = np.random.uniform(3, 6)
            z_score_base = np.random.uniform(15, 25)
            roa_base = np.random.uniform(1.5, 2.5)
            credit_base = np.random.uniform(60, 120)
        elif country_code in ['KEN', 'GHA', 'NGA', 'SEN', 'RWA', 'UGA']:  # Emerging markets
            npl_base = np.random.uniform(5, 10)
            z_score_base = np.random.uniform(10, 20)
            roa_base = np.random.uniform(0.8, 2.0)
            credit_base = np.random.uniform(20, 60)
        else:  # Less developed
            npl_base = np.random.uniform(8, 15)
            z_score_base = np.random.uniform(5, 15)
            roa_base = np.random.uniform(0.2, 1.5)
            credit_base = np.random.uniform(5, 30)
        
        # Generate time series for each indicator
        num_periods = (END_YEAR - START_YEAR + 1) * 4
        
        # Non-Performing Loans (should decrease over time in stable economies)
        npl_trend = -0.002 if country_code in ['ZAF', 'MUS', 'KEN', 'GHA', 'RWA'] else 0.001
        npl_values = generate_time_series(npl_base, npl_trend, 0.05, num_periods, 0.5, 30)
        
        # Bank Z-score (higher is better, should improve over time)
        z_trend = 0.003 if country_code in ['ZAF', 'MUS', 'KEN', 'GHA', 'RWA'] else 0.001
        z_values = generate_time_series(z_score_base, z_trend, 0.03, num_periods, 2, 50)
        
        # Return on Assets
        roa_trend = 0.002 if country_code in ['ZAF', 'MUS', 'KEN', 'GHA', 'RWA'] else -0.001
        roa_values = generate_time_series(roa_base, roa_trend, 0.04, num_periods, -2, 5)
        
        # Domestic Credit to Private Sector
        credit_trend = 0.005 if country_code in ['KEN', 'GHA', 'NGA', 'RWA', 'UGA'] else 0.002
        credit_values = generate_time_series(credit_base, credit_trend, 0.02, num_periods, 0, 200)
        
        # Add COVID-19 impact (Q1 2020 - Q4 2021)
        covid_start = (2020 - START_YEAR) * 4
        covid_end = covid_start + 8
        
        # NPLs increase during COVID
        npl_values[covid_start:covid_end] *= np.random.uniform(1.2, 1.5)
        # Z-score decreases
        z_values[covid_start:covid_end] *= np.random.uniform(0.7, 0.9)
        # ROA decreases
        roa_values[covid_start:covid_end] *= np.random.uniform(0.6, 0.8)
        
        # Create records
        idx = 0
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                if idx < num_periods:
                    data.append({
                        'Country_Code': country_code,
                        'Country_Name': country_name,
                        'Year': year,
                        'Quarter': quarter,
                        'Date': f"{year}-{quarter}",
                        'Bank_NPL_to_Total_Loans_%': round(npl_values[idx], 2),
                        'Bank_Z_Score': round(z_values[idx], 2),
                        'Bank_ROA_%': round(roa_values[idx], 3),
                        'Domestic_Credit_to_Private_Sector_%_GDP': round(credit_values[idx], 2)
                    })
                    idx += 1
    
    return pd.DataFrame(data)

def generate_regulatory_quality_data():
    """Generate Regulatory Quality indicators"""
    print("Generating Regulatory Quality data...")
    
    data = []
    
    for country_code, country_name in SSA_COUNTRIES.items():
        np.random.seed(hash(country_code) % 2**32)
        
        # Base regulatory quality scores (-2.5 to 2.5 scale)
        if country_code in ['MUS', 'ZAF', 'BWA', 'SYC', 'NAM', 'RWA']:
            reg_quality_base = np.random.uniform(0.2, 1.0)
            fintech_reg_score_base = np.random.uniform(60, 80)
        elif country_code in ['KEN', 'GHA', 'NGA', 'SEN', 'UGA', 'TZA']:
            reg_quality_base = np.random.uniform(-0.5, 0.5)
            fintech_reg_score_base = np.random.uniform(40, 60)
        else:
            reg_quality_base = np.random.uniform(-1.5, -0.2)
            fintech_reg_score_base = np.random.uniform(20, 40)
        
        num_periods = (END_YEAR - START_YEAR + 1) * 4
        
        # WGI Regulatory Quality (gradual improvement)
        reg_trend = 0.002 if country_code in ['RWA', 'KEN', 'GHA', 'MUS'] else 0.0005
        reg_quality_values = generate_time_series(reg_quality_base, reg_trend, 0.01, num_periods, -2.5, 2.5)
        
        # Financial Regulation Index (0-100 scale)
        fintech_trend = 0.005 if country_code in ['KEN', 'GHA', 'NGA', 'RWA', 'ZAF'] else 0.002
        fintech_reg_values = generate_time_series(fintech_reg_score_base, fintech_trend, 0.02, num_periods, 0, 100)
        
        # Digital lending guidelines introduction (dummy variable)
        # Countries introduce regulations at different times
        digital_lending_year = {
            'KEN': 2016, 'GHA': 2017, 'NGA': 2018, 'RWA': 2019,
            'ZAF': 2015, 'UGA': 2020, 'TZA': 2021, 'SEN': 2022
        }.get(country_code, None)
        
        # Mobile money regulations
        mobile_money_year = {
            'KEN': 2014, 'GHA': 2015, 'UGA': 2013, 'TZA': 2015,
            'RWA': 2016, 'ZAF': 2016, 'NGA': 2018, 'ZMB': 2019
        }.get(country_code, None)
        
        # Data protection regulations
        data_protection_year = {
            'ZAF': 2013, 'KEN': 2019, 'GHA': 2020, 'NGA': 2019,
            'RWA': 2021, 'MUS': 2017, 'SEN': 2020, 'UGA': 2019
        }.get(country_code, None)
        
        # Regulatory sandbox introduction
        sandbox_year = {
            'KEN': 2019, 'RWA': 2020, 'GHA': 2021, 'ZAF': 2020,
            'MUS': 2018, 'SLE': 2020, 'MOZ': 2021, 'NGA': 2021
        }.get(country_code, None)
        
        idx = 0
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                if idx < num_periods:
                    # Determine dummy variables based on year
                    digital_lending_reg = 1 if digital_lending_year and year >= digital_lending_year else 0
                    mobile_money_reg = 1 if mobile_money_year and year >= mobile_money_year else 0
                    data_protection_reg = 1 if data_protection_year and year >= data_protection_year else 0
                    sandbox_reg = 1 if sandbox_year and year >= sandbox_year else 0
                    
                    # Boost fintech regulation score after introducing key regulations
                    if digital_lending_reg or mobile_money_reg:
                        fintech_reg_values[idx] *= 1.1
                    if sandbox_reg:
                        fintech_reg_values[idx] *= 1.05
                    
                    data.append({
                        'Country_Code': country_code,
                        'Country_Name': country_name,
                        'Year': year,
                        'Quarter': quarter,
                        'Date': f"{year}-{quarter}",
                        'WGI_Regulatory_Quality': round(reg_quality_values[idx], 3),
                        'Financial_Regulation_Index': round(min(fintech_reg_values[idx], 100), 2),
                        'Digital_Lending_Regulation': digital_lending_reg,
                        'Mobile_Money_Regulation': mobile_money_reg,
                        'Data_Protection_Law': data_protection_reg,
                        'Regulatory_Sandbox': sandbox_reg,
                        'Total_FinTech_Regulations': digital_lending_reg + mobile_money_reg + 
                                                     data_protection_reg + sandbox_reg
                    })
                    idx += 1
    
    return pd.DataFrame(data)

def generate_financial_inclusion_metrics():
    """Generate additional financial inclusion and market structure metrics"""
    print("Generating Financial Inclusion metrics...")
    
    data = []
    
    for country_code, country_name in SSA_COUNTRIES.items():
        np.random.seed(hash(country_code + 'inclusion') % 2**32)
        
        # Base values for financial inclusion
        if country_code in ['KEN', 'ZAF', 'MUS', 'SYC', 'BWA']:
            account_ownership_base = np.random.uniform(70, 85)
            mobile_money_base = np.random.uniform(60, 80)
            bank_branches_base = np.random.uniform(5, 15)
            atm_base = np.random.uniform(10, 30)
        elif country_code in ['GHA', 'RWA', 'UGA', 'TZA', 'SEN', 'NGA']:
            account_ownership_base = np.random.uniform(40, 60)
            mobile_money_base = np.random.uniform(30, 60)
            bank_branches_base = np.random.uniform(2, 8)
            atm_base = np.random.uniform(5, 15)
        else:
            account_ownership_base = np.random.uniform(15, 40)
            mobile_money_base = np.random.uniform(5, 30)
            bank_branches_base = np.random.uniform(0.5, 3)
            atm_base = np.random.uniform(1, 8)
        
        num_periods = (END_YEAR - START_YEAR + 1) * 4
        
        # Generate time series with positive trends (financial inclusion generally improves)
        account_ownership = generate_time_series(account_ownership_base, 0.008, 0.02, num_periods, 0, 100)
        mobile_money_accounts = generate_time_series(mobile_money_base, 0.015, 0.03, num_periods, 0, 100)
        bank_branches = generate_time_series(bank_branches_base, 0.003, 0.01, num_periods, 0.1, 50)
        atms = generate_time_series(atm_base, 0.005, 0.02, num_periods, 0.1, 100)
        
        # Market concentration (HHI - Herfindahl-Hirschman Index)
        hhi_base = np.random.uniform(1500, 3500) if country_code in ['KEN', 'ZAF', 'NGA'] else np.random.uniform(2500, 5000)
        market_concentration = generate_time_series(hhi_base, -0.002, 0.02, num_periods, 1000, 10000)
        
        # Interest rate spread
        spread_base = np.random.uniform(5, 10) if country_code in ['ZAF', 'MUS', 'BWA'] else np.random.uniform(8, 15)
        interest_spread = generate_time_series(spread_base, -0.001, 0.03, num_periods, 2, 25)
        
        idx = 0
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                if idx < num_periods:
                    data.append({
                        'Country_Code': country_code,
                        'Country_Name': country_name,
                        'Year': year,
                        'Quarter': quarter,
                        'Date': f"{year}-{quarter}",
                        'Account_Ownership_%_Adults': round(min(account_ownership[idx], 100), 2),
                        'Mobile_Money_Account_%_Adults': round(min(mobile_money_accounts[idx], 100), 2),
                        'Bank_Branches_per_100k_Adults': round(bank_branches[idx], 2),
                        'ATMs_per_100k_Adults': round(atms[idx], 2),
                        'Banking_Market_HHI': round(market_concentration[idx], 0),
                        'Interest_Rate_Spread_%': round(interest_spread[idx], 2)
                    })
                    idx += 1
    
    return pd.DataFrame(data)

def generate_systemic_risk_indicators():
    """Generate systemic risk and financial stability indicators"""
    print("Generating Systemic Risk indicators...")
    
    data = []
    
    for country_code, country_name in SSA_COUNTRIES.items():
        np.random.seed(hash(country_code + 'risk') % 2**32)
        
        # Base values for systemic indicators
        if country_code in ['ZAF', 'MUS', 'BWA', 'NAM']:
            capital_adequacy_base = np.random.uniform(15, 20)
            liquid_assets_base = np.random.uniform(30, 45)
            forex_reserves_base = np.random.uniform(4, 8)
        elif country_code in ['KEN', 'GHA', 'NGA', 'RWA', 'UGA']:
            capital_adequacy_base = np.random.uniform(12, 18)
            liquid_assets_base = np.random.uniform(20, 35)
            forex_reserves_base = np.random.uniform(2, 5)
        else:
            capital_adequacy_base = np.random.uniform(8, 15)
            liquid_assets_base = np.random.uniform(15, 30)
            forex_reserves_base = np.random.uniform(0.5, 3)
        
        num_periods = (END_YEAR - START_YEAR + 1) * 4
        
        # Generate time series
        capital_adequacy = generate_time_series(capital_adequacy_base, 0.002, 0.02, num_periods, 8, 30)
        liquid_assets_ratio = generate_time_series(liquid_assets_base, 0.001, 0.03, num_periods, 10, 60)
        forex_reserves_months = generate_time_series(forex_reserves_base, 0.001, 0.05, num_periods, 0.1, 12)
        
        # Financial Stress Index (0-100, higher = more stress)
        stress_base = np.random.uniform(20, 40) if country_code in ['ZAF', 'MUS'] else np.random.uniform(30, 60)
        financial_stress = generate_time_series(stress_base, -0.001, 0.05, num_periods, 0, 100)
        
        # Add crisis periods
        covid_start = (2020 - START_YEAR) * 4
        covid_end = covid_start + 6
        financial_stress[covid_start:covid_end] *= np.random.uniform(1.5, 2.0)
        capital_adequacy[covid_start:covid_end] *= np.random.uniform(0.85, 0.95)
        
        idx = 0
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                if idx < num_periods:
                    data.append({
                        'Country_Code': country_code,
                        'Country_Name': country_name,
                        'Year': year,
                        'Quarter': quarter,
                        'Date': f"{year}-{quarter}",
                        'Capital_Adequacy_Ratio_%': round(capital_adequacy[idx], 2),
                        'Liquid_Assets_to_Deposits_%': round(liquid_assets_ratio[idx], 2),
                        'Forex_Reserves_Months_Imports': round(forex_reserves_months[idx], 2),
                        'Financial_Stress_Index': round(financial_stress[idx], 2),
                        'Systemic_Risk_Score': round(100 - (capital_adequacy[idx] + liquid_assets_ratio[idx])/2 + financial_stress[idx]/2, 2)
                    })
                    idx += 1
    
    return pd.DataFrame(data)

def combine_all_datasets():
    """Combine all datasets into a comprehensive financial system dataset"""
    print("\nCombining all datasets...")
    
    # Generate all component datasets
    banking_df = generate_banking_sector_data()
    regulatory_df = generate_regulatory_quality_data()
    inclusion_df = generate_financial_inclusion_metrics()
    systemic_df = generate_systemic_risk_indicators()
    
    # Merge all datasets on common keys
    merged_df = banking_df.merge(
        regulatory_df,
        on=['Country_Code', 'Country_Name', 'Year', 'Quarter', 'Date'],
        how='outer'
    ).merge(
        inclusion_df,
        on=['Country_Code', 'Country_Name', 'Year', 'Quarter', 'Date'],
        how='outer'
    ).merge(
        systemic_df,
        on=['Country_Code', 'Country_Name', 'Year', 'Quarter', 'Date'],
        how='outer'
    )
    
    # Add metadata columns
    merged_df['Data_Source'] = 'Synthetic/Estimated'
    merged_df['Last_Updated'] = datetime.now().strftime('%Y-%m-%d')
    merged_df['Region'] = 'Sub-Saharan Africa'
    
    # Calculate composite indices
    merged_df['Banking_Health_Index'] = (
        (100 - merged_df['Bank_NPL_to_Total_Loans_%']) * 0.25 +
        merged_df['Bank_Z_Score'] * 2 +
        merged_df['Bank_ROA_%'] * 10 +
        merged_df['Capital_Adequacy_Ratio_%'] * 2
    ) / 4
    
    merged_df['Regulatory_Strength_Index'] = (
        (merged_df['WGI_Regulatory_Quality'] + 2.5) * 20 +
        merged_df['Financial_Regulation_Index'] +
        merged_df['Total_FinTech_Regulations'] * 10
    ) / 3
    
    merged_df['Financial_Development_Index'] = (
        merged_df['Domestic_Credit_to_Private_Sector_%_GDP'] +
        merged_df['Account_Ownership_%_Adults'] +
        merged_df['Mobile_Money_Account_%_Adults'] +
        (merged_df['Bank_Branches_per_100k_Adults'] * 5) +
        (merged_df['ATMs_per_100k_Adults'] * 2)
    ) / 5
    
    # Sort by country and date
    merged_df = merged_df.sort_values(['Country_Code', 'Year', 'Quarter'])
    
    return merged_df

def generate_summary_statistics(df):
    """Generate summary statistics for the dataset"""
    print("\nGenerating summary statistics...")
    
    summary = {
        'Dataset Overview': {
            'Total Records': int(len(df)),
            'Countries': int(df['Country_Code'].nunique()),
            'Time Period': f"{int(df['Year'].min())} - {int(df['Year'].max())}",
            'Variables': int(len(df.columns))
        },
        'Banking Sector Averages (2024)': {
            'NPL Ratio': f"{df[df['Year']==2024]['Bank_NPL_to_Total_Loans_%'].mean():.2f}%",
            'Bank Z-Score': f"{df[df['Year']==2024]['Bank_Z_Score'].mean():.2f}",
            'ROA': f"{df[df['Year']==2024]['Bank_ROA_%'].mean():.2f}%",
            'Credit to GDP': f"{df[df['Year']==2024]['Domestic_Credit_to_Private_Sector_%_GDP'].mean():.2f}%"
        },
        'Regulatory Quality (2024)': {
            'Avg WGI Score': f"{df[df['Year']==2024]['WGI_Regulatory_Quality'].mean():.3f}",
            'Countries with Digital Lending Regs': int(df[df['Year']==2024]['Digital_Lending_Regulation'].sum()),
            'Countries with Sandbox': int(df[df['Year']==2024]['Regulatory_Sandbox'].sum()),
            'Avg FinTech Regulations': f"{df[df['Year']==2024]['Total_FinTech_Regulations'].mean():.1f}"
        },
        'Financial Inclusion (2024)': {
            'Account Ownership': f"{df[df['Year']==2024]['Account_Ownership_%_Adults'].mean():.1f}%",
            'Mobile Money Accounts': f"{df[df['Year']==2024]['Mobile_Money_Account_%_Adults'].mean():.1f}%",
            'Bank Branches per 100k': f"{df[df['Year']==2024]['Bank_Branches_per_100k_Adults'].mean():.2f}",
            'ATMs per 100k': f"{df[df['Year']==2024]['ATMs_per_100k_Adults'].mean():.2f}"
        }
    }
    
    return summary

def main():
    """Main execution function"""
    print("=" * 60)
    print("FINANCIAL SYSTEM & REGULATORY DATA GENERATOR")
    print("Sub-Saharan Africa FinTech Risk Early Warning Model")
    print("=" * 60)
    
    # Generate combined dataset
    financial_system_df = combine_all_datasets()
    
    # Generate summary statistics
    summary = generate_summary_statistics(financial_system_df)
    
    # Save to different formats
    print("\nSaving datasets...")
    
    # Save main dataset
    financial_system_df.to_csv('../data/financial_system_regulatory_data.csv', index=False)
    financial_system_df.to_excel('../data/financial_system_regulatory_data.xlsx', index=False, sheet_name='Financial_System_Data')
    
    # Save country-level aggregated data (annual)
    annual_df = financial_system_df.groupby(['Country_Code', 'Country_Name', 'Year']).agg({
        'Bank_NPL_to_Total_Loans_%': 'mean',
        'Bank_Z_Score': 'mean',
        'Bank_ROA_%': 'mean',
        'Domestic_Credit_to_Private_Sector_%_GDP': 'mean',
        'WGI_Regulatory_Quality': 'mean',
        'Financial_Regulation_Index': 'mean',
        'Digital_Lending_Regulation': 'max',
        'Mobile_Money_Regulation': 'max',
        'Data_Protection_Law': 'max',
        'Regulatory_Sandbox': 'max',
        'Account_Ownership_%_Adults': 'mean',
        'Mobile_Money_Account_%_Adults': 'mean',
        'Banking_Health_Index': 'mean',
        'Regulatory_Strength_Index': 'mean',
        'Financial_Development_Index': 'mean',
        'Systemic_Risk_Score': 'mean'
    }).round(2).reset_index()
    
    annual_df.to_csv('../data/financial_system_annual_summary.csv', index=False)
    
    # Save summary statistics
    with open('../data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    
    print("\nFiles created:")
    print("1. financial_system_regulatory_data.csv - Full quarterly dataset")
    print("2. financial_system_regulatory_data.xlsx - Excel version")
    print("3. financial_system_annual_summary.csv - Annual aggregated data")
    print("4. dataset_summary.json - Summary statistics")
    
    print("\nDataset Summary:")
    for category, stats in summary.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
    
    return financial_system_df, summary

if __name__ == "__main__":
    df, summary = main()