#!/usr/bin/env python3
"""
Sub-Saharan Africa FinTech Early Warning Model - Macroeconomic Data Collector

This script downloads macroeconomic and country-level data for Sub-Saharan African countries
from various sources including World Bank, IMF, and other international databases.

Author: Research Assistant
Date: 2025-10-11
"""

import pandas as pd
import numpy as np
import requests
import wbdata
import json
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SSAMacroDataCollector:
    def __init__(self):
        """Initialize the data collector with SSA country codes and indicator mappings."""
        
        # Major Sub-Saharan African countries (focusing on those with significant FinTech activity)
        self.ssa_countries = {
            'KE': 'Kenya',
            'NG': 'Nigeria', 
            'ZA': 'South Africa',
            'GH': 'Ghana',
            'UG': 'Uganda',
            'TZ': 'Tanzania',
            'RW': 'Rwanda',
            'SN': 'Senegal',
            'CI': 'Cote d\'Ivoire',
            'ZM': 'Zambia',
            'BW': 'Botswana',
            'MW': 'Malawi',
            'MZ': 'Mozambique',
            'ET': 'Ethiopia',
            'ZW': 'Zimbabwe',
            'CM': 'Cameroon',
            'BF': 'Burkina Faso',
            'ML': 'Mali',
            'BJ': 'Benin',
            'TG': 'Togo'
        }
        
        # World Bank indicator codes for our variables
        self.wb_indicators = {
            # Economic Growth & Stability
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'inflation_cpi': 'FP.CPI.TOTL.ZG',  # Inflation, consumer prices (annual %)
            'unemployment': 'SL.UEM.TOTL.ZS',   # Unemployment, total (% of total labor force)
            'interest_rate': 'FR.INR.RINR',     # Real interest rate (%)
            'money_supply_m2': 'FM.LBL.BMNY.GD.ZS',  # Broad money (% of GDP)
            'debt_to_gdp': 'GC.DOD.TOTL.GD.ZS', # Central government debt, total (% of GDP)
            
            # Exchange Rate & External Sector
            'exchange_rate': 'PA.NUS.FCRF',     # Official exchange rate (LCU per US$, period average)
            'current_account': 'BN.CAB.XOKA.GD.ZS', # Current account balance (% of GDP)
            
            # Digital Infrastructure
            'mobile_subscriptions': 'IT.CEL.SETS.P2', # Mobile cellular subscriptions (per 100 people)
            'internet_users': 'IT.NET.USER.ZS',       # Individuals using the Internet (% of population)
            'secure_servers': 'IT.NET.SECR.P6',       # Secure Internet servers (per 1 million people)
            
            # Additional Economic Indicators
            'fdi_inflows': 'BX.KLT.DINV.WD.GD.ZS',   # Foreign direct investment, net inflows (% of GDP)
            'trade_openness': 'NE.TRD.GNFS.ZS',       # Trade (% of GDP)
            'financial_depth': 'FD.AST.PRVT.GD.ZS',   # Domestic credit to private sector (% of GDP)
        }
        
        # Time period for data collection (2010-2023)
        self.start_year = 2010
        self.end_year = 2023
        
        print(f"Initialized SSA Macro Data Collector")
        print(f"Target countries: {len(self.ssa_countries)}")
        print(f"Indicators: {len(self.wb_indicators)}")
        print(f"Time period: {self.start_year}-{self.end_year}")
    
    def download_wb_data(self, indicator_code, indicator_name):
        """Download data for a specific World Bank indicator."""
        print(f"\nDownloading {indicator_name} data...")
        
        try:
            # Download data for all SSA countries using the correct API format
            data = wbdata.get_dataframe(
                {indicator_code: indicator_name},
                country=list(self.ssa_countries.keys())
            )
            
            if data.empty:
                print(f"No data available for {indicator_name}")
                return None
            
            # Reset index to get country and date as columns
            data = data.reset_index()
            
            # Filter for our target years
            data['date'] = pd.to_datetime(data['date'])
            data = data[
                (data['date'].dt.year >= self.start_year) & 
                (data['date'].dt.year <= self.end_year)
            ]
            
            # Add country name mapping
            data['country_name'] = data['country'].map(self.ssa_countries)
            
            print(f"Downloaded {len(data)} observations for {indicator_name}")
            return data
            
        except Exception as e:
            print(f"Error downloading {indicator_name}: {str(e)}")
            # Try alternative approach using requests to World Bank API directly
            return self.download_wb_data_direct(indicator_code, indicator_name)
    
    def download_wb_data_direct(self, indicator_code, indicator_name):
        """Download data directly from World Bank API using requests."""
        print(f"Trying direct API for {indicator_name}...")
        
        try:
            all_data = []
            
            for country_code in self.ssa_countries.keys():
                url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
                params = {
                    'format': 'json',
                    'date': f'{self.start_year}:{self.end_year}',
                    'per_page': 1000
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:  # Check if data exists
                        for item in data[1]:
                            if item['value'] is not None:
                                all_data.append({
                                    'country': country_code,
                                    'country_name': self.ssa_countries[country_code],
                                    'date': pd.to_datetime(item['date']),
                                    indicator_name: float(item['value'])
                                })
                
                time.sleep(0.1)  # Small delay to be respectful
            
            if all_data:
                df = pd.DataFrame(all_data)
                print(f"Downloaded {len(df)} observations for {indicator_name} via direct API")
                return df
            else:
                print(f"No data found for {indicator_name}")
                return None
                
        except Exception as e:
            print(f"Direct API also failed for {indicator_name}: {str(e)}")
            return None
    
    def calculate_volatility(self, df, value_col, country_col='country', window=3):
        """Calculate rolling volatility for a given indicator."""
        volatility_data = []
        
        for country in df[country_col].unique():
            country_data = df[df[country_col] == country].copy()
            country_data = country_data.sort_values('date')
            
            # Calculate rolling standard deviation
            country_data[f'{value_col}_volatility'] = country_data[value_col].rolling(
                window=window, min_periods=2
            ).std()
            
            volatility_data.append(country_data)
        
        return pd.concat(volatility_data, ignore_index=True)
    
    def collect_all_data(self):
        """Collect all macroeconomic indicators."""
        all_datasets = {}
        
        print("="*60)
        print("STARTING DATA COLLECTION FOR SSA FINTECH EARLY WARNING MODEL")
        print("="*60)
        
        # Download each indicator
        for indicator_code, indicator_name in self.wb_indicators.items():
            dataset = self.download_wb_data(indicator_code, indicator_name)
            if dataset is not None:
                all_datasets[indicator_name] = dataset
            time.sleep(1)  # Be respectful to the API
        
        return all_datasets
    
    def merge_datasets(self, datasets):
        """Merge all datasets into a single comprehensive dataset."""
        print("\nMerging all datasets...")
        
        if not datasets:
            print("No datasets to merge!")
            return None
        
        # Start with the first dataset
        merged_data = None
        
        for indicator_name, dataset in datasets.items():
            if dataset is None or dataset.empty:
                continue
                
            # Prepare dataset for merging
            dataset_clean = dataset[['country', 'country_name', 'date', indicator_name]].copy()
            
            if merged_data is None:
                merged_data = dataset_clean
            else:
                merged_data = pd.merge(
                    merged_data, 
                    dataset_clean,
                    on=['country', 'country_name', 'date'],
                    how='outer'
                )
        
        if merged_data is not None:
            # Sort by country and date
            merged_data = merged_data.sort_values(['country_name', 'date'])
            
            # Add additional calculated fields
            merged_data = self.add_calculated_indicators(merged_data)
            
            print(f"Merged dataset shape: {merged_data.shape}")
            print(f"Countries: {merged_data['country_name'].nunique()}")
            print(f"Date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
        
        return merged_data
    
    def add_calculated_indicators(self, df):
        """Add calculated indicators like volatilities and growth rates."""
        print("Adding calculated indicators...")
        
        # Calculate GDP growth volatility
        if 'GDP growth (annual %)' in df.columns:
            df = self.calculate_volatility(df, 'GDP growth (annual %)', 'country_name')
        
        # Calculate exchange rate volatility if we have exchange rate data
        if 'Official exchange rate (LCU per US$, period average)' in df.columns:
            # First calculate exchange rate changes
            df = df.sort_values(['country_name', 'date'])
            df['exchange_rate_change'] = df.groupby('country_name')['Official exchange rate (LCU per US$, period average)'].pct_change() * 100
            df = self.calculate_volatility(df, 'exchange_rate_change', 'country_name')
        
        # Add year column for easier analysis
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        return df
    
    def generate_summary_stats(self, df):
        """Generate summary statistics for the dataset."""
        print("\nGenerating summary statistics...")
        
        # Basic info
        summary = {
            'dataset_info': {
                'total_observations': len(df),
                'countries': df['country_name'].nunique(),
                'country_list': sorted(df['country_name'].unique().tolist()),
                'years': sorted(df['year'].unique().tolist()),
                'indicators': [col for col in df.columns if col not in ['country', 'country_name', 'date', 'year']]
            }
        }
        
        # Descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary['descriptive_stats'] = df[numeric_cols].describe().to_dict()
        
        # Data completeness
        summary['data_completeness'] = {}
        for col in numeric_cols:
            total_possible = len(df)
            non_null = df[col].notna().sum()
            summary['data_completeness'][col] = {
                'non_null_count': int(non_null),
                'total_possible': int(total_possible),
                'completeness_rate': float(non_null / total_possible)
            }
        
        return summary
    
    def save_data(self, df, summary_stats, base_path="ssa_fintech_data"):
        """Save the collected data and summary statistics."""
        print(f"\nSaving data to {base_path}...")
        
        # Save main dataset
        df.to_csv(f"{base_path}/processed_data/ssa_macro_data.csv", index=False)
        df.to_excel(f"{base_path}/processed_data/ssa_macro_data.xlsx", index=False)
        
        # Save summary statistics
        with open(f"{base_path}/processed_data/summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        # Save country mapping
        with open(f"{base_path}/processed_data/country_mapping.json", 'w') as f:
            json.dump(self.ssa_countries, f, indent=2)
        
        print("Data saved successfully!")
        print(f"- Main dataset: {base_path}/processed_data/ssa_macro_data.csv")
        print(f"- Excel format: {base_path}/processed_data/ssa_macro_data.xlsx") 
        print(f"- Summary stats: {base_path}/processed_data/summary_statistics.json")
        print(f"- Country mapping: {base_path}/processed_data/country_mapping.json")

def main():
    """Main execution function."""
    collector = SSAMacroDataCollector()
    
    # Collect all data
    datasets = collector.collect_all_data()
    
    # Merge datasets
    merged_data = collector.merge_datasets(datasets)
    
    if merged_data is not None:
        # Generate summary statistics
        summary_stats = collector.generate_summary_stats(merged_data)
        
        # Save everything
        collector.save_data(merged_data, summary_stats)
        
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final dataset contains {len(merged_data)} observations")
        print(f"Covering {merged_data['country_name'].nunique()} countries")
        print(f"Time period: {merged_data['year'].min()}-{merged_data['year'].max()}")
        
    else:
        print("Failed to collect and merge data!")

if __name__ == "__main__":
    main()