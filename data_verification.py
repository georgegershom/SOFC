"""
Data Verification Script for FinTech Early Warning Model
Quick verification of data quality and structure
"""

import pandas as pd
import numpy as np

def verify_dataset():
    """Verify the generated dataset quality and structure"""
    print("FinTech Early Warning Model - Data Verification")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('fintech_macroeconomic_synthetic.csv')
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Countries: {df['Country'].nunique()}")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Variables: {len(df.columns)}")
    
    print("\nColumn Information:")
    print("-" * 30)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"{col:35} | {str(dtype):10} | {null_count:3} ({null_pct:5.1f}%)")
    
    print("\nData Quality Summary:")
    print("-" * 30)
    print(f"Total Observations: {len(df):,}")
    print(f"Missing Values: {df.isnull().sum().sum():,}")
    print(f"Missing Percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    print("\nVariable Ranges:")
    print("-" * 30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Year':
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"{col:35} | Min: {min_val:8.2f} | Max: {max_val:8.2f} | Mean: {mean_val:8.2f}")
    
    print("\nCountry Coverage:")
    print("-" * 30)
    country_counts = df['Country'].value_counts()
    print(f"Countries with complete data: {(country_counts == 14).sum()}")
    print(f"Countries with missing data: {(country_counts < 14).sum()}")
    
    print("\nYear Coverage:")
    print("-" * 30)
    year_counts = df['Year'].value_counts().sort_index()
    print(f"Years with complete data: {(year_counts == 49).sum()}")
    print(f"Years with missing data: {(year_counts < 49).sum()}")
    
    print("\nSample Data (First 5 rows):")
    print("-" * 30)
    print(df.head().to_string())
    
    print("\nSample Data (Last 5 rows):")
    print("-" * 30)
    print(df.tail().to_string())
    
    print("\nCorrelation Matrix (Top 5 correlations):")
    print("-" * 30)
    corr_matrix = df[numeric_cols].corr()
    
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    
    # Find top correlations
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                correlations.append({
                    'Var1': corr_matrix.columns[i],
                    'Var2': corr_matrix.columns[j],
                    'Correlation': abs(corr_val)
                })
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: x['Correlation'], reverse=True)
    
    print("Top 5 Correlations:")
    for i, corr in enumerate(correlations[:5]):
        print(f"{i+1}. {corr['Var1']} ↔ {corr['Var2']}: {corr['Correlation']:.3f}")
    
    print("\nRisk Assessment (2023 data):")
    print("-" * 30)
    latest_data = df[df['Year'] == 2023]
    
    # High inflation countries
    high_inflation = latest_data[latest_data['Inflation_Rate_CPI'] > 10]
    print(f"High Inflation Risk (>10%): {len(high_inflation)} countries")
    
    # High unemployment countries
    high_unemployment = latest_data[latest_data['Unemployment_Rate'] > 15]
    print(f"High Unemployment Risk (>15%): {len(high_unemployment)} countries")
    
    # High debt countries
    high_debt = latest_data[latest_data['Public_Debt_to_GDP_Ratio'] > 80]
    print(f"High Debt Risk (>80%): {len(high_debt)} countries")
    
    # High volatility countries
    high_volatility = latest_data[latest_data['GDP_Growth_Volatility'] > 5]
    print(f"High Volatility Risk (>5%): {len(high_volatility)} countries")
    
    print("\nData Verification Complete!")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset()