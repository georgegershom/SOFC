"""
Data Collection Script for Category 3: Financial System & Regulatory Data
Thesis: Research on FinTech Early Warning Model in Nexus of Fintech Risk in Sub-Sahara Africa Economies

This script downloads, generates, and fabricates financial system and regulatory data for Sub-Saharan African countries.

Data Categories:
1. Banking Sector Health Indicators
2. Regulatory Quality Measures
3. Dummy Variables for Regulatory Changes

Author: Generated for FinTech Research
Date: 2025-10-11
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sub-Saharan African Countries
SSA_COUNTRIES = {
    'KEN': 'Kenya',
    'NGA': 'Nigeria', 
    'ZAF': 'South Africa',
    'GHA': 'Ghana',
    'TZA': 'Tanzania',
    'UGA': 'Uganda',
    'RWA': 'Rwanda',
    'ETH': 'Ethiopia',
    'SEN': 'Senegal',
    'CIV': "Côte d'Ivoire",
    'BWA': 'Botswana',
    'ZMB': 'Zambia',
    'MOZ': 'Mozambique',
    'CMR': 'Cameroon',
    'AGO': 'Angola',
    'MWI': 'Malawi',
    'BEN': 'Benin',
    'BFA': 'Burkina Faso',
    'MLI': 'Mali',
    'NER': 'Niger'
}

# Time period for analysis
START_YEAR = 2010
END_YEAR = 2023
YEARS = list(range(START_YEAR, END_YEAR + 1))

class WorldBankDataCollector:
    """Collects data from World Bank API"""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    # World Bank Indicator Codes
    INDICATORS = {
        'bank_npl': 'FB.AST.NPER.ZS',  # Bank nonperforming loans to total gross loans (%)
        'domestic_credit': 'FS.AST.PRVT.GD.ZS',  # Domestic credit to private sector (% of GDP)
        'regulatory_quality': 'RQ.EST',  # Regulatory Quality Index
        'bank_capital': 'FB.BNK.CAPA.ZS',  # Bank capital to assets ratio (%)
        'bank_roa': 'FB.BNK.ROAA.ZS',  # Bank return on assets (%, after tax)
        'bank_zscore': 'GFDD.SI.01',  # Bank Z-score
    }
    
    def __init__(self):
        self.data_cache = {}
    
    def fetch_indicator(self, indicator_code, country_codes, start_year, end_year):
        """Fetch data for a specific indicator from World Bank API"""
        
        countries_str = ';'.join(country_codes)
        url = f"{self.BASE_URL}/country/{countries_str}/indicator/{indicator_code}"
        
        params = {
            'date': f'{start_year}:{end_year}',
            'format': 'json',
            'per_page': 5000
        }
        
        try:
            print(f"Fetching {indicator_code}...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    return data[1]
            
            print(f"  Warning: No data returned for {indicator_code}")
            return None
            
        except Exception as e:
            print(f"  Error fetching {indicator_code}: {str(e)}")
            return None
    
    def collect_all_indicators(self, country_codes, start_year, end_year):
        """Collect all World Bank indicators"""
        
        all_data = {}
        
        for ind_name, ind_code in self.INDICATORS.items():
            data = self.fetch_indicator(ind_code, country_codes, start_year, end_year)
            if data:
                all_data[ind_name] = data
            time.sleep(0.5)  # Rate limiting
        
        return all_data
    
    def parse_wb_data(self, wb_data):
        """Parse World Bank API response into DataFrame"""
        
        records = []
        
        for indicator_name, data_list in wb_data.items():
            if data_list:
                for entry in data_list:
                    records.append({
                        'country_code': entry['countryiso3code'],
                        'country_name': entry['country']['value'],
                        'year': int(entry['date']),
                        'indicator': indicator_name,
                        'value': entry['value']
                    })
        
        df = pd.DataFrame(records)
        
        # Pivot to wide format
        if not df.empty:
            df_wide = df.pivot_table(
                index=['country_code', 'country_name', 'year'],
                columns='indicator',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            return df_wide
        
        return pd.DataFrame()


class SyntheticDataGenerator:
    """Generates realistic synthetic/fabricated data for missing values"""
    
    @staticmethod
    def generate_bank_zscore(country_code, year, seed_base=42):
        """
        Generate Bank Z-score (higher values = more stable banking system)
        Typical range: 5-25, with higher values for more stable countries
        """
        np.random.seed(seed_base + hash(country_code + str(year)) % 1000)
        
        # Base Z-score by country (stability ranking)
        stable_countries = ['ZAF', 'BWA', 'KEN', 'RWA', 'GHA']
        moderate_countries = ['TZA', 'UGA', 'SEN', 'CIV', 'ETH']
        
        if country_code in stable_countries:
            base = np.random.uniform(15, 22)
        elif country_code in moderate_countries:
            base = np.random.uniform(10, 16)
        else:
            base = np.random.uniform(6, 12)
        
        # Add time trend (improving stability)
        time_trend = (year - START_YEAR) * 0.2
        
        # Add random variation
        noise = np.random.normal(0, 1.5)
        
        zscore = base + time_trend + noise
        return max(5.0, min(25.0, zscore))  # Bound between 5-25
    
    @staticmethod
    def generate_bank_roa(country_code, year, seed_base=42):
        """
        Generate Bank Return on Assets (%)
        Typical range: 0.5% - 3.5%
        """
        np.random.seed(seed_base + hash(country_code + str(year)) % 1000)
        
        # Base ROA by country
        high_profit = ['KEN', 'NGA', 'GHA', 'TZA']
        
        if country_code in high_profit:
            base = np.random.uniform(1.8, 3.0)
        else:
            base = np.random.uniform(0.8, 2.0)
        
        # Economic cycle effects
        cycle = 0.3 * np.sin(2 * np.pi * (year - START_YEAR) / 7)
        
        # Random variation
        noise = np.random.normal(0, 0.2)
        
        roa = base + cycle + noise
        return max(0.1, min(4.5, roa))
    
    @staticmethod
    def generate_bank_npl(country_code, year, seed_base=42):
        """
        Generate Bank Non-Performing Loans ratio (%)
        Typical range: 2% - 15%
        """
        np.random.seed(seed_base + hash(country_code + str(year)) % 1000)
        
        # Base NPL by country
        low_npl = ['BWA', 'RWA', 'KEN', 'ZAF']
        moderate_npl = ['GHA', 'TZA', 'UGA', 'SEN']
        
        if country_code in low_npl:
            base = np.random.uniform(3, 6)
        elif country_code in moderate_npl:
            base = np.random.uniform(6, 10)
        else:
            base = np.random.uniform(9, 14)
        
        # Crisis effects (2020 COVID)
        if year == 2020:
            crisis_shock = np.random.uniform(1.5, 3.0)
        elif year in [2021, 2022]:
            crisis_shock = np.random.uniform(0.5, 1.5)
        else:
            crisis_shock = 0
        
        # Random variation
        noise = np.random.normal(0, 0.5)
        
        npl = base + crisis_shock + noise
        return max(1.0, min(20.0, npl))
    
    @staticmethod
    def generate_domestic_credit(country_code, year, seed_base=42):
        """
        Generate Domestic Credit to Private Sector (% of GDP)
        Typical range: 10% - 180%
        """
        np.random.seed(seed_base + hash(country_code + str(year)) % 1000)
        
        # Base credit by country development
        high_credit = ['ZAF', 'BWA', 'KEN']
        moderate_credit = ['NGA', 'GHA', 'RWA', 'TZA']
        
        if country_code in high_credit:
            base = np.random.uniform(60, 90)
        elif country_code in moderate_credit:
            base = np.random.uniform(25, 50)
        else:
            base = np.random.uniform(12, 30)
        
        # Growing trend
        trend = (year - START_YEAR) * 1.2
        
        # Random variation
        noise = np.random.normal(0, 2)
        
        credit = base + trend + noise
        return max(5.0, min(200.0, credit))
    
    @staticmethod
    def generate_regulatory_quality(country_code, year, seed_base=42):
        """
        Generate Regulatory Quality Index
        Range: -2.5 to +2.5 (higher is better)
        """
        np.random.seed(seed_base + hash(country_code + str(year)) % 1000)
        
        # Base regulatory quality
        high_quality = ['BWA', 'ZAF', 'RWA', 'KEN']
        moderate_quality = ['GHA', 'SEN', 'TZA', 'UGA']
        
        if country_code in high_quality:
            base = np.random.uniform(0.2, 0.8)
        elif country_code in moderate_quality:
            base = np.random.uniform(-0.3, 0.3)
        else:
            base = np.random.uniform(-0.8, -0.2)
        
        # Improvement trend
        trend = (year - START_YEAR) * 0.02
        
        # Random variation
        noise = np.random.normal(0, 0.08)
        
        quality = base + trend + noise
        return max(-2.5, min(2.5, quality))


class RegulatoryDummyGenerator:
    """Generates dummy variables for regulatory changes"""
    
    # Key FinTech regulations by country and year
    FINTECH_REGULATIONS = {
        'KEN': {
            'mobile_money_regulation': 2013,
            'digital_lending_guidelines': 2020,
            'data_protection_act': 2019,
            'payment_services_act': 2014,
        },
        'NGA': {
            'mobile_money_regulation': 2015,
            'digital_lending_guidelines': 2021,
            'data_protection_regulation': 2019,
            'payment_services_banks': 2018,
        },
        'ZAF': {
            'mobile_money_regulation': 2011,
            'digital_lending_guidelines': 2019,
            'data_protection_act': 2013,
            'payment_services_act': 2012,
        },
        'GHA': {
            'mobile_money_regulation': 2015,
            'digital_lending_guidelines': 2020,
            'data_protection_act': 2012,
            'payment_services_directive': 2016,
        },
        'RWA': {
            'mobile_money_regulation': 2014,
            'digital_lending_guidelines': 2019,
            'data_protection_law': 2016,
            'payment_systems_law': 2017,
        },
        'TZA': {
            'mobile_money_regulation': 2015,
            'digital_lending_guidelines': 2021,
            'data_protection_act': 2022,
            'payment_systems_act': 2015,
        },
        'UGA': {
            'mobile_money_regulation': 2013,
            'digital_lending_guidelines': 2021,
            'data_protection_act': 2019,
            'payment_systems_act': 2020,
        },
    }
    
    @staticmethod
    def generate_regulatory_dummies(country_code, year):
        """Generate dummy variables for regulatory changes"""
        
        dummies = {}
        
        if country_code in RegulatoryDummyGenerator.FINTECH_REGULATIONS:
            regulations = RegulatoryDummyGenerator.FINTECH_REGULATIONS[country_code]
            
            for reg_name, reg_year in regulations.items():
                # Dummy = 1 if regulation is in effect, 0 otherwise
                dummies[f'reg_{reg_name}'] = 1 if year >= reg_year else 0
        else:
            # Default dummies for countries without specific regulations
            dummies = {
                'reg_mobile_money_regulation': 1 if year >= 2015 else 0,
                'reg_digital_lending_guidelines': 1 if year >= 2020 else 0,
                'reg_data_protection_act': 1 if year >= 2018 else 0,
                'reg_payment_services_act': 1 if year >= 2016 else 0,
            }
        
        # Count total regulations in effect
        dummies['total_fintech_regulations'] = sum([v for k, v in dummies.items() if k.startswith('reg_')])
        
        return dummies


def main():
    """Main function to collect and generate the complete dataset"""
    
    print("="*80)
    print("FINANCIAL SYSTEM & REGULATORY DATA COLLECTION")
    print("Category 3: FinTech Early Warning Model - Sub-Saharan Africa")
    print("="*80)
    print()
    
    # Initialize collectors
    wb_collector = WorldBankDataCollector()
    synthetic_gen = SyntheticDataGenerator()
    regulatory_gen = RegulatoryDummyGenerator()
    
    # Step 1: Collect World Bank Data
    print("STEP 1: Collecting data from World Bank API...")
    print(f"Countries: {len(SSA_COUNTRIES)}")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print()
    
    wb_data = wb_collector.collect_all_indicators(
        list(SSA_COUNTRIES.keys()),
        START_YEAR,
        END_YEAR
    )
    
    df_wb = wb_collector.parse_wb_data(wb_data)
    
    print(f"✓ World Bank data collected: {len(df_wb)} records")
    print()
    
    # Step 2: Generate complete dataset with all countries and years
    print("STEP 2: Generating complete dataset with synthetic data for missing values...")
    
    all_records = []
    
    for country_code, country_name in SSA_COUNTRIES.items():
        for year in YEARS:
            record = {
                'country_code': country_code,
                'country_name': country_name,
                'year': year,
            }
            
            # Try to get World Bank data first
            wb_row = df_wb[
                (df_wb['country_code'] == country_code) & 
                (df_wb['year'] == year)
            ]
            
            if not wb_row.empty:
                # Use World Bank data where available
                for col in wb_row.columns:
                    if col not in ['country_code', 'country_name', 'year']:
                        value = wb_row[col].iloc[0]
                        record[col] = value if pd.notna(value) else None
            
            # Generate/fill missing values with synthetic data
            if pd.isna(record.get('bank_zscore')) or record.get('bank_zscore') is None:
                record['bank_zscore'] = synthetic_gen.generate_bank_zscore(country_code, year)
                record['bank_zscore_source'] = 'synthetic'
            else:
                record['bank_zscore_source'] = 'worldbank'
            
            if pd.isna(record.get('bank_roa')) or record.get('bank_roa') is None:
                record['bank_roa'] = synthetic_gen.generate_bank_roa(country_code, year)
                record['bank_roa_source'] = 'synthetic'
            else:
                record['bank_roa_source'] = 'worldbank'
            
            if pd.isna(record.get('bank_npl')) or record.get('bank_npl') is None:
                record['bank_npl'] = synthetic_gen.generate_bank_npl(country_code, year)
                record['bank_npl_source'] = 'synthetic'
            else:
                record['bank_npl_source'] = 'worldbank'
            
            if pd.isna(record.get('domestic_credit')) or record.get('domestic_credit') is None:
                record['domestic_credit'] = synthetic_gen.generate_domestic_credit(country_code, year)
                record['domestic_credit_source'] = 'synthetic'
            else:
                record['domestic_credit_source'] = 'worldbank'
            
            if pd.isna(record.get('regulatory_quality')) or record.get('regulatory_quality') is None:
                record['regulatory_quality'] = synthetic_gen.generate_regulatory_quality(country_code, year)
                record['regulatory_quality_source'] = 'synthetic'
            else:
                record['regulatory_quality_source'] = 'worldbank'
            
            # Add regulatory dummies
            reg_dummies = regulatory_gen.generate_regulatory_dummies(country_code, year)
            record.update(reg_dummies)
            
            all_records.append(record)
    
    # Create final DataFrame
    df_final = pd.DataFrame(all_records)
    
    print(f"✓ Complete dataset generated: {len(df_final)} records")
    print(f"  Countries: {df_final['country_code'].nunique()}")
    print(f"  Years: {df_final['year'].min()} - {df_final['year'].max()}")
    print()
    
    # Step 3: Calculate additional banking indicators
    print("STEP 3: Calculating additional banking sector health indicators...")
    
    # Bank Capital Adequacy (if not available, generate)
    if 'bank_capital' not in df_final.columns or df_final['bank_capital'].isna().all():
        df_final['bank_capital'] = df_final.apply(
            lambda row: np.random.uniform(12, 20) if row['country_code'] in ['ZAF', 'BWA', 'KEN']
            else np.random.uniform(10, 16),
            axis=1
        )
        df_final['bank_capital_source'] = 'synthetic'
    
    print("✓ Additional indicators calculated")
    print()
    
    # Step 4: Data Quality Assessment
    print("STEP 4: Data Quality Assessment...")
    print()
    
    key_indicators = ['bank_npl', 'bank_zscore', 'bank_roa', 'domestic_credit', 'regulatory_quality']
    
    for indicator in key_indicators:
        total = len(df_final)
        source_col = f'{indicator}_source'
        
        if source_col in df_final.columns:
            wb_count = (df_final[source_col] == 'worldbank').sum()
            synthetic_count = (df_final[source_col] == 'synthetic').sum()
            
            print(f"{indicator}:")
            print(f"  World Bank: {wb_count} ({wb_count/total*100:.1f}%)")
            print(f"  Synthetic:  {synthetic_count} ({synthetic_count/total*100:.1f}%)")
        else:
            print(f"{indicator}: All synthetic")
    
    print()
    
    # Step 5: Summary Statistics
    print("STEP 5: Summary Statistics")
    print("="*80)
    print()
    
    print("Banking Sector Health Indicators:")
    print(df_final[['bank_npl', 'bank_zscore', 'bank_roa', 'bank_capital', 'domestic_credit']].describe())
    print()
    
    print("Regulatory Quality Index:")
    print(df_final['regulatory_quality'].describe())
    print()
    
    print("Regulatory Coverage (% of countries with regulation):")
    reg_cols = [col for col in df_final.columns if col.startswith('reg_') and col != 'reg_mobile_money_regulation']
    for col in reg_cols[:5]:  # Show first 5
        latest_year = df_final[df_final['year'] == END_YEAR]
        coverage = latest_year[col].mean() * 100
        print(f"  {col}: {coverage:.1f}%")
    print()
    
    # Step 6: Save datasets
    print("STEP 6: Saving datasets...")
    
    # Main dataset
    output_file = '/workspace/category3_financial_regulatory_data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"✓ Main dataset saved: {output_file}")
    
    # Create separate files for different aspects
    
    # Banking sector indicators only
    banking_cols = ['country_code', 'country_name', 'year', 'bank_npl', 'bank_zscore', 
                   'bank_roa', 'bank_capital', 'domestic_credit']
    df_banking = df_final[banking_cols]
    df_banking.to_csv('/workspace/banking_sector_health_data.csv', index=False)
    print(f"✓ Banking sector data saved: /workspace/banking_sector_health_data.csv")
    
    # Regulatory indicators only
    reg_cols_full = ['country_code', 'country_name', 'year', 'regulatory_quality'] + \
                    [col for col in df_final.columns if col.startswith('reg_')]
    df_regulatory = df_final[reg_cols_full]
    df_regulatory.to_csv('/workspace/regulatory_quality_data.csv', index=False)
    print(f"✓ Regulatory data saved: /workspace/regulatory_quality_data.csv")
    
    # Time series by country
    for country_code in ['KEN', 'NGA', 'ZAF', 'GHA', 'RWA']:
        df_country = df_final[df_final['country_code'] == country_code]
        country_file = f'/workspace/data_{country_code}_{SSA_COUNTRIES[country_code].replace(" ", "_")}.csv'
        df_country.to_csv(country_file, index=False)
    print(f"✓ Country-specific files saved for 5 major economies")
    
    print()
    print("="*80)
    print("DATA COLLECTION COMPLETE!")
    print("="*80)
    print()
    print(f"Total records: {len(df_final)}")
    print(f"Countries: {df_final['country_code'].nunique()}")
    print(f"Time period: {df_final['year'].min()}-{df_final['year'].max()}")
    print()
    print("Output files:")
    print("  1. category3_financial_regulatory_data.csv (Complete dataset)")
    print("  2. banking_sector_health_data.csv (Banking indicators)")
    print("  3. regulatory_quality_data.csv (Regulatory measures)")
    print("  4. data_[COUNTRY]_*.csv (Country-specific files)")
    print()
    print("Next steps:")
    print("  - Review data quality and coverage")
    print("  - Validate synthetic data against available sources")
    print("  - Integrate with Categories 1 & 2 for complete analysis")
    print("  - Proceed with FinTech early warning model development")
    print()
    
    return df_final


if __name__ == "__main__":
    df = main()
