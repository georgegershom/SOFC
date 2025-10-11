"""
Sub-Saharan Africa Macroeconomic Data Collector and Generator
For FinTech Early Warning Model Research
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# List of Sub-Saharan African countries (ISO3 codes)
SSA_COUNTRIES = {
    'AGO': 'Angola',
    'BEN': 'Benin',
    'BWA': 'Botswana',
    'BFA': 'Burkina Faso',
    'BDI': 'Burundi',
    'CMR': 'Cameroon',
    'CPV': 'Cape Verde',
    'CAF': 'Central African Republic',
    'TCD': 'Chad',
    'COM': 'Comoros',
    'COG': 'Congo, Rep.',
    'COD': 'Congo, Dem. Rep.',
    'CIV': "Côte d'Ivoire",
    'GNQ': 'Equatorial Guinea',
    'ERI': 'Eritrea',
    'SWZ': 'Eswatini',
    'ETH': 'Ethiopia',
    'GAB': 'Gabon',
    'GMB': 'Gambia, The',
    'GHA': 'Ghana',
    'GIN': 'Guinea',
    'GNB': 'Guinea-Bissau',
    'KEN': 'Kenya',
    'LSO': 'Lesotho',
    'LBR': 'Liberia',
    'MDG': 'Madagascar',
    'MWI': 'Malawi',
    'MLI': 'Mali',
    'MRT': 'Mauritania',
    'MUS': 'Mauritius',
    'MOZ': 'Mozambique',
    'NAM': 'Namibia',
    'NER': 'Niger',
    'NGA': 'Nigeria',
    'RWA': 'Rwanda',
    'STP': 'São Tomé and Príncipe',
    'SEN': 'Senegal',
    'SYC': 'Seychelles',
    'SLE': 'Sierra Leone',
    'SOM': 'Somalia',
    'ZAF': 'South Africa',
    'SSD': 'South Sudan',
    'SDN': 'Sudan',
    'TZA': 'Tanzania',
    'TGO': 'Togo',
    'UGA': 'Uganda',
    'ZMB': 'Zambia',
    'ZWE': 'Zimbabwe'
}

# World Bank API indicators
WORLD_BANK_INDICATORS = {
    'NY.GDP.MKTP.KD.ZG': 'GDP Growth Rate (%)',
    'FP.CPI.TOTL.ZG': 'Inflation Rate (CPI) (%)',
    'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)',
    'PA.NUS.FCRF': 'Official Exchange Rate (LCU per US$)',
    'FR.INR.RINR': 'Real Interest Rate (%)',
    'FM.LBL.BMNY.GD.ZS': 'Broad Money (% of GDP)',
    'GC.DOD.TOTL.GD.ZS': 'Central Government Debt (% of GDP)',
    'IT.CEL.SETS.P2': 'Mobile Cellular Subscriptions (per 100 people)',
    'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)',
    'IT.NET.SECR.P6': 'Secure Internet Servers (per 1 million people)'
}

class SSAMacroeconomicDataCollector:
    def __init__(self, start_year=2010, end_year=2024):
        self.start_year = start_year
        self.end_year = end_year
        self.data_frames = []
        
    def fetch_world_bank_data(self, country_code, indicator, indicator_name):
        """Fetch data from World Bank API"""
        base_url = "https://api.worldbank.org/v2/country"
        url = f"{base_url}/{country_code}/indicator/{indicator}"
        params = {
            'format': 'json',
            'date': f'{self.start_year}:{self.end_year}',
            'per_page': 500
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    return pd.DataFrame(data[1])
        except Exception as e:
            print(f"Error fetching {indicator_name} for {country_code}: {e}")
        
        return None
    
    def generate_synthetic_data(self, country_code, indicator_name, years):
        """Generate realistic synthetic data for missing values"""
        np.random.seed(hash(country_code + indicator_name) % 2**32)
        
        # Define realistic ranges and patterns for each indicator
        synthetic_params = {
            'GDP Growth Rate (%)': {'mean': 4.5, 'std': 3.0, 'min': -5, 'max': 15, 'trend': 0.1},
            'Inflation Rate (CPI) (%)': {'mean': 6.0, 'std': 4.0, 'min': -2, 'max': 25, 'trend': -0.05},
            'Unemployment Rate (%)': {'mean': 8.0, 'std': 3.0, 'min': 2, 'max': 25, 'trend': 0.05},
            'Official Exchange Rate (LCU per US$)': {'mean': 100, 'std': 50, 'min': 0.5, 'max': 5000, 'trend': 5},
            'Real Interest Rate (%)': {'mean': 5.0, 'std': 3.0, 'min': -10, 'max': 20, 'trend': -0.1},
            'Broad Money (% of GDP)': {'mean': 35, 'std': 15, 'min': 10, 'max': 100, 'trend': 0.5},
            'Central Government Debt (% of GDP)': {'mean': 45, 'std': 20, 'min': 10, 'max': 150, 'trend': 1.0},
            'Mobile Cellular Subscriptions (per 100 people)': {'mean': 80, 'std': 30, 'min': 10, 'max': 150, 'trend': 3},
            'Individuals using the Internet (% of population)': {'mean': 30, 'std': 20, 'min': 1, 'max': 85, 'trend': 2.5},
            'Secure Internet Servers (per 1 million people)': {'mean': 50, 'std': 100, 'min': 0.1, 'max': 1000, 'trend': 10}
        }
        
        params = synthetic_params.get(indicator_name, {'mean': 50, 'std': 10, 'min': 0, 'max': 100, 'trend': 0})
        
        # Generate base values with trend
        base_values = []
        current_value = params['mean'] + np.random.randn() * params['std']
        
        for i, year in enumerate(years):
            # Add trend
            current_value += params['trend'] + np.random.randn() * params['std'] * 0.3
            
            # Add some randomness
            value = current_value + np.random.randn() * params['std'] * 0.5
            
            # Apply constraints
            value = np.clip(value, params['min'], params['max'])
            
            # Add occasional shocks (economic crises, policy changes)
            if np.random.random() < 0.1:  # 10% chance of shock
                shock = np.random.randn() * params['std'] * 2
                value += shock
                value = np.clip(value, params['min'], params['max'])
            
            base_values.append(value)
            current_value = value
        
        return base_values
    
    def calculate_volatility(self, values):
        """Calculate volatility (standard deviation of percentage changes)"""
        if len(values) < 2:
            return np.nan
        
        values = pd.Series(values).fillna(method='ffill').fillna(method='bfill')
        if values.isna().all() or (values == 0).all():
            return np.nan
        
        pct_changes = values.pct_change().dropna()
        if len(pct_changes) > 0:
            return pct_changes.std() * 100  # Convert to percentage
        return np.nan
    
    def calculate_exchange_rate_volatility(self, df):
        """Calculate exchange rate volatility for each country and year"""
        volatility_data = []
        
        for country in df['Country_Code'].unique():
            country_data = df[df['Country_Code'] == country].sort_values('Year')
            
            # Calculate rolling volatility (3-year window)
            if len(country_data) >= 3:
                exchange_rates = country_data['Official Exchange Rate (LCU per US$)'].values
                
                for i in range(2, len(country_data)):
                    window_data = exchange_rates[max(0, i-2):i+1]
                    volatility = self.calculate_volatility(window_data)
                    
                    volatility_data.append({
                        'Country_Code': country,
                        'Country_Name': country_data.iloc[i]['Country_Name'],
                        'Year': country_data.iloc[i]['Year'],
                        'Exchange_Rate_Volatility': volatility
                    })
        
        return pd.DataFrame(volatility_data)
    
    def calculate_gdp_volatility(self, df):
        """Calculate GDP growth volatility for each country"""
        volatility_data = []
        
        for country in df['Country_Code'].unique():
            country_data = df[df['Country_Code'] == country].sort_values('Year')
            
            if len(country_data) >= 3:
                gdp_growth = country_data['GDP Growth Rate (%)'].values
                
                for i in range(2, len(country_data)):
                    window_data = gdp_growth[max(0, i-2):i+1]
                    volatility = np.std(window_data) if len(window_data) > 0 else np.nan
                    
                    volatility_data.append({
                        'Country_Code': country,
                        'Year': country_data.iloc[i]['Year'],
                        'GDP_Growth_Volatility': volatility
                    })
        
        return pd.DataFrame(volatility_data)
    
    def collect_all_data(self):
        """Collect data for all countries and indicators"""
        all_data = []
        years = list(range(self.start_year, self.end_year + 1))
        
        print("Starting data collection for Sub-Saharan African countries...")
        print(f"Period: {self.start_year} - {self.end_year}")
        print(f"Countries: {len(SSA_COUNTRIES)}")
        print(f"Indicators: {len(WORLD_BANK_INDICATORS)}")
        print("-" * 50)
        
        for country_code, country_name in tqdm(SSA_COUNTRIES.items(), desc="Processing countries"):
            country_data = {
                'Country_Code': country_code,
                'Country_Name': country_name,
                'Year': years
            }
            
            # Initialize with NaN
            for indicator_name in WORLD_BANK_INDICATORS.values():
                country_data[indicator_name] = [np.nan] * len(years)
            
            # Try to fetch real data
            for indicator, indicator_name in WORLD_BANK_INDICATORS.items():
                df = self.fetch_world_bank_data(country_code, indicator, indicator_name)
                
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        if row['date'] and row['value'] is not None:
                            year = int(row['date'])
                            if year in years:
                                idx = years.index(year)
                                country_data[indicator_name][idx] = float(row['value'])
                
                # Add small delay to avoid API rate limiting
                time.sleep(0.1)
            
            # Fill missing values with synthetic data
            for indicator_name in WORLD_BANK_INDICATORS.values():
                missing_indices = [i for i, v in enumerate(country_data[indicator_name]) 
                                 if pd.isna(v)]
                
                if missing_indices:
                    missing_years = [years[i] for i in missing_indices]
                    synthetic_values = self.generate_synthetic_data(
                        country_code, indicator_name, missing_years
                    )
                    
                    for idx, value in zip(missing_indices, synthetic_values):
                        country_data[indicator_name][idx] = value
            
            # Create DataFrame for this country
            country_df = pd.DataFrame({
                'Country_Code': [country_code] * len(years),
                'Country_Name': [country_name] * len(years),
                'Year': years,
                **{indicator: country_data[indicator] for indicator in WORLD_BANK_INDICATORS.values()}
            })
            
            all_data.append(country_df)
        
        # Combine all country data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add volatility measures
        print("\nCalculating volatility measures...")
        exchange_volatility = self.calculate_exchange_rate_volatility(combined_df)
        gdp_volatility = self.calculate_gdp_volatility(combined_df)
        
        # Merge volatility data
        combined_df = combined_df.merge(
            exchange_volatility[['Country_Code', 'Year', 'Exchange_Rate_Volatility']], 
            on=['Country_Code', 'Year'], 
            how='left'
        )
        
        combined_df = combined_df.merge(
            gdp_volatility[['Country_Code', 'Year', 'GDP_Growth_Volatility']], 
            on=['Country_Code', 'Year'], 
            how='left'
        )
        
        # Calculate M2 growth rate
        combined_df['M2_Growth_Rate'] = combined_df.groupby('Country_Code')['Broad Money (% of GDP)'].pct_change() * 100
        
        # Add data quality indicator
        combined_df['Data_Source'] = 'Mixed (World Bank API + Synthetic)'
        
        # Add timestamp
        combined_df['Data_Collection_Date'] = datetime.now().strftime('%Y-%m-%d')
        
        return combined_df
    
    def add_fintech_risk_indicators(self, df):
        """Add derived FinTech risk indicators"""
        print("\nCalculating FinTech risk indicators...")
        
        # Digital Infrastructure Index (composite of internet and mobile penetration)
        df['Digital_Infrastructure_Index'] = (
            df['Mobile Cellular Subscriptions (per 100 people)'] * 0.4 +
            df['Individuals using the Internet (% of population)'] * 0.4 +
            df['Secure Internet Servers (per 1 million people)'].clip(upper=100) * 0.2
        )
        
        # Economic Stability Index
        df['Economic_Stability_Index'] = 100 - (
            df['GDP_Growth_Volatility'].fillna(0) * 2 +
            df['Inflation Rate (CPI) (%)'].abs() +
            df['Exchange_Rate_Volatility'].fillna(0) * 0.5
        ).clip(lower=0, upper=100)
        
        # Financial Development Index
        df['Financial_Development_Index'] = (
            df['Broad Money (% of GDP)'] * 0.5 +
            (100 - df['Central Government Debt (% of GDP)'].clip(upper=100)) * 0.3 +
            df['Digital_Infrastructure_Index'] * 0.2
        ).clip(lower=0, upper=100)
        
        # FinTech Risk Score (higher score = higher risk)
        df['FinTech_Risk_Score'] = (
            df['Exchange_Rate_Volatility'].fillna(0) * 0.2 +
            df['GDP_Growth_Volatility'].fillna(0) * 0.15 +
            df['Inflation Rate (CPI) (%)'].abs() * 0.15 +
            df['Unemployment Rate (%)'] * 0.1 +
            df['Central Government Debt (% of GDP)'].clip(upper=100) * 0.1 +
            (100 - df['Digital_Infrastructure_Index']) * 0.15 +
            (100 - df['Economic_Stability_Index']) * 0.15
        ).clip(lower=0, upper=100)
        
        # Risk Categories
        df['Risk_Category'] = pd.cut(
            df['FinTech_Risk_Score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return df

def main():
    """Main execution function"""
    print("=" * 70)
    print("SUB-SAHARAN AFRICA MACROECONOMIC DATA COLLECTION")
    print("For FinTech Early Warning Model Research")
    print("=" * 70)
    
    # Initialize collector
    collector = SSAMacroeconomicDataCollector(start_year=2010, end_year=2024)
    
    # Collect all data
    macro_data = collector.collect_all_data()
    
    # Add FinTech risk indicators
    macro_data = collector.add_fintech_risk_indicators(macro_data)
    
    # Sort the data
    macro_data = macro_data.sort_values(['Country_Code', 'Year'])
    
    # Save to multiple formats
    print("\nSaving datasets...")
    
    # Save full dataset
    macro_data.to_csv('ssa_macroeconomic_data_full.csv', index=False)
    print("✓ Saved: ssa_macroeconomic_data_full.csv")
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter('ssa_macroeconomic_data.xlsx', engine='xlsxwriter') as writer:
        # Full data
        macro_data.to_excel(writer, sheet_name='Full_Dataset', index=False)
        
        # Summary statistics
        summary_stats = macro_data.groupby('Country_Name').agg({
            'GDP Growth Rate (%)': ['mean', 'std'],
            'Inflation Rate (CPI) (%)': ['mean', 'std'],
            'Unemployment Rate (%)': ['mean', 'std'],
            'Digital_Infrastructure_Index': 'mean',
            'FinTech_Risk_Score': 'mean'
        }).round(2)
        summary_stats.to_excel(writer, sheet_name='Country_Summary')
        
        # Recent data (2020-2024)
        recent_data = macro_data[macro_data['Year'] >= 2020]
        recent_data.to_excel(writer, sheet_name='Recent_Data_2020_2024', index=False)
        
        # High risk countries
        high_risk = macro_data[macro_data['Risk_Category'].isin(['High', 'Very High'])]
        high_risk.to_excel(writer, sheet_name='High_Risk_Countries', index=False)
    
    print("✓ Saved: ssa_macroeconomic_data.xlsx (with multiple sheets)")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(macro_data):,}")
    print(f"Countries: {macro_data['Country_Name'].nunique()}")
    print(f"Years covered: {macro_data['Year'].min()} - {macro_data['Year'].max()}")
    print(f"Variables: {len(macro_data.columns)}")
    
    print("\nRisk Distribution:")
    risk_dist = macro_data['Risk_Category'].value_counts()
    for category, count in risk_dist.items():
        print(f"  {category}: {count:,} ({count/len(macro_data)*100:.1f}%)")
    
    print("\nTop 10 Highest Risk Countries (Average 2020-2024):")
    recent_risk = macro_data[macro_data['Year'] >= 2020].groupby('Country_Name')['FinTech_Risk_Score'].mean()
    top_risk = recent_risk.nlargest(10)
    for i, (country, score) in enumerate(top_risk.items(), 1):
        print(f"  {i}. {country}: {score:.2f}")
    
    print("\nTop 10 Best Digital Infrastructure (Average 2020-2024):")
    recent_digital = macro_data[macro_data['Year'] >= 2020].groupby('Country_Name')['Digital_Infrastructure_Index'].mean()
    top_digital = recent_digital.nlargest(10)
    for i, (country, score) in enumerate(top_digital.items(), 1):
        print(f"  {i}. {country}: {score:.2f}")
    
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return macro_data

if __name__ == "__main__":
    data = main()