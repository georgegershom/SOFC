#!/usr/bin/env python3
"""
Simple SSA Macroeconomic Data Collector
A more straightforward approach to collecting World Bank data for SSA countries.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime

def get_wb_data(country_code, indicator_code, start_year=2010, end_year=2023):
    """Get data from World Bank API for a specific country and indicator."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {
        'format': 'json',
        'date': f'{start_year}:{end_year}',
        'per_page': 1000
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                return data[1]
        return []
    except Exception as e:
        print(f"Error fetching data for {country_code}-{indicator_code}: {e}")
        return []

def main():
    # SSA Countries
    countries = {
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
    
    # Key indicators
    indicators = {
        'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
        'inflation': 'FP.CPI.TOTL.ZG',
        'unemployment': 'SL.UEM.TOTL.ZS',
        'exchange_rate': 'PA.NUS.FCRF',
        'interest_rate': 'FR.INR.RINR',
        'money_supply': 'FM.LBL.BMNY.GD.ZS',
        'debt_gdp': 'GC.DOD.TOTL.GD.ZS',
        'mobile_subs': 'IT.CEL.SETS.P2',
        'internet_users': 'IT.NET.USER.ZS',
        'secure_servers': 'IT.NET.SECR.P6',
        'fdi_inflows': 'BX.KLT.DINV.WD.GD.ZS',
        'trade_openness': 'NE.TRD.GNFS.ZS',
        'financial_depth': 'FD.AST.PRVT.GD.ZS'
    }
    
    print("Starting data collection...")
    all_data = []
    
    total_requests = len(countries) * len(indicators)
    current_request = 0
    
    for country_code, country_name in countries.items():
        print(f"\nProcessing {country_name} ({country_code})...")
        
        for indicator_name, indicator_code in indicators.items():
            current_request += 1
            print(f"  [{current_request}/{total_requests}] Fetching {indicator_name}...")
            
            data = get_wb_data(country_code, indicator_code)
            
            for item in data:
                if item['value'] is not None:
                    all_data.append({
                        'country_code': country_code,
                        'country_name': country_name,
                        'year': int(item['date']),
                        'indicator_name': indicator_name,
                        'indicator_code': indicator_code,
                        'value': float(item['value'])
                    })
            
            time.sleep(0.1)  # Be nice to the API
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        print(f"\nCollected {len(df)} data points")
        print(f"Countries: {df['country_name'].nunique()}")
        print(f"Indicators: {df['indicator_name'].nunique()}")
        print(f"Years: {sorted(df['year'].unique())}")
        
        # Pivot to wide format
        df_wide = df.pivot_table(
            index=['country_code', 'country_name', 'year'],
            columns='indicator_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Save data
        df_wide.to_csv('processed_data/ssa_macro_data_simple.csv', index=False)
        df.to_csv('processed_data/ssa_macro_data_long.csv', index=False)
        
        print(f"\nData saved:")
        print(f"- Wide format: processed_data/ssa_macro_data_simple.csv ({df_wide.shape})")
        print(f"- Long format: processed_data/ssa_macro_data_long.csv ({df.shape})")
        
        # Show sample
        print(f"\nSample of wide format data:")
        print(df_wide.head())
        
        # Data completeness
        print(f"\nData completeness by indicator:")
        completeness = df_wide.select_dtypes(include=[np.number]).isnull().mean().sort_values()
        for indicator, missing_rate in completeness.items():
            print(f"  {indicator}: {(1-missing_rate)*100:.1f}% complete")
    
    else:
        print("No data collected!")

if __name__ == "__main__":
    main()