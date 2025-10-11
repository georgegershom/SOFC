"""
Advanced Synthetic Data Generator for FinTech Early Warning Model
Generates realistic macroeconomic data for Sub-Saharan Africa countries
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SyntheticMacroeconomicDataGenerator:
    def __init__(self):
        self.countries = [
            'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CPV', 'CAF', 'TCD', 'COM',
            'COG', 'COD', 'CIV', 'DJI', 'GNQ', 'ERI', 'SWZ', 'ETH', 'GAB', 'GMB',
            'GHA', 'GIN', 'GNB', 'KEN', 'LSO', 'LBR', 'MDG', 'MWI', 'MLI', 'MRT',
            'MUS', 'MOZ', 'NAM', 'NER', 'NGA', 'RWA', 'STP', 'SEN', 'SYC', 'SLE',
            'SOM', 'ZAF', 'SSD', 'SDN', 'TZA', 'TGO', 'UGA', 'ZMB', 'ZWE'
        ]
        
        self.country_names = {
            'AGO': 'Angola', 'BEN': 'Benin', 'BWA': 'Botswana', 'BFA': 'Burkina Faso',
            'BDI': 'Burundi', 'CMR': 'Cameroon', 'CPV': 'Cape Verde', 'CAF': 'Central African Republic',
            'TCD': 'Chad', 'COM': 'Comoros', 'COG': 'Congo', 'COD': 'Congo, Dem. Rep.',
            'CIV': 'Cote d\'Ivoire', 'DJI': 'Djibouti', 'GNQ': 'Equatorial Guinea',
            'ERI': 'Eritrea', 'SWZ': 'Eswatini', 'ETH': 'Ethiopia', 'GAB': 'Gabon',
            'GMB': 'Gambia', 'GHA': 'Ghana', 'GIN': 'Guinea', 'GNB': 'Guinea-Bissau',
            'KEN': 'Kenya', 'LSO': 'Lesotho', 'LBR': 'Liberia', 'MDG': 'Madagascar',
            'MWI': 'Malawi', 'MLI': 'Mali', 'MRT': 'Mauritania', 'MUS': 'Mauritius',
            'MOZ': 'Mozambique', 'NAM': 'Namibia', 'NER': 'Niger', 'NGA': 'Nigeria',
            'RWA': 'Rwanda', 'STP': 'Sao Tome and Principe', 'SEN': 'Senegal',
            'SYC': 'Seychelles', 'SLE': 'Sierra Leone', 'SOM': 'Somalia',
            'ZAF': 'South Africa', 'SSD': 'South Sudan', 'SDN': 'Sudan',
            'TZA': 'Tanzania', 'TGO': 'Togo', 'UGA': 'Uganda', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
        }
        
        # Country characteristics for realistic data generation
        self.country_characteristics = self._define_country_characteristics()
        
    def _define_country_characteristics(self):
        """Define economic characteristics for each country"""
        characteristics = {}
        
        # Economic development levels and characteristics
        for country in self.countries:
            # Random but realistic economic characteristics
            characteristics[country] = {
                'development_level': random.choice(['Low', 'Lower-Middle', 'Upper-Middle']),
                'resource_dependency': random.uniform(0.1, 0.8),  # 0 = diversified, 1 = resource dependent
                'political_stability': random.uniform(0.3, 0.9),  # 0 = unstable, 1 = stable
                'institutional_quality': random.uniform(0.2, 0.8),  # 0 = poor, 1 = good
                'trade_openness': random.uniform(0.2, 0.7),  # 0 = closed, 1 = open
                'financial_development': random.uniform(0.1, 0.6),  # 0 = underdeveloped, 1 = developed
            }
        
        # Special cases for known characteristics
        characteristics['ZAF'] = {  # South Africa
            'development_level': 'Upper-Middle',
            'resource_dependency': 0.3,
            'political_stability': 0.7,
            'institutional_quality': 0.8,
            'trade_openness': 0.6,
            'financial_development': 0.7
        }
        
        characteristics['NGA'] = {  # Nigeria
            'development_level': 'Lower-Middle',
            'resource_dependency': 0.7,
            'political_stability': 0.5,
            'institutional_quality': 0.4,
            'trade_openness': 0.4,
            'financial_development': 0.3
        }
        
        characteristics['KEN'] = {  # Kenya
            'development_level': 'Lower-Middle',
            'resource_dependency': 0.2,
            'political_stability': 0.6,
            'institutional_quality': 0.6,
            'trade_openness': 0.5,
            'financial_development': 0.5
        }
        
        return characteristics
    
    def generate_gdp_growth_data(self, start_year=2010, end_year=2023):
        """Generate realistic GDP growth rate data"""
        print("Generating GDP Growth Rate data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base growth rate based on development level
            if char['development_level'] == 'Low':
                base_growth = random.uniform(2.0, 6.0)
            elif char['development_level'] == 'Lower-Middle':
                base_growth = random.uniform(1.5, 5.0)
            else:  # Upper-Middle
                base_growth = random.uniform(0.5, 3.5)
            
            # Add volatility based on political stability and resource dependency
            volatility = 0.5 + (1 - char['political_stability']) * 2 + char['resource_dependency'] * 1.5
            
            for year in range(start_year, end_year + 1):
                # Add cyclical patterns
                cycle = np.sin(2 * np.pi * (year - start_year) / 7) * 0.5
                
                # Add random shocks
                shock = np.random.normal(0, volatility)
                
                # Add trend based on institutional quality
                trend = (char['institutional_quality'] - 0.5) * 0.1 * (year - start_year)
                
                growth_rate = base_growth + cycle + shock + trend
                
                # Ensure realistic bounds
                growth_rate = np.clip(growth_rate, -10, 15)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'GDP_Growth_Rate': round(growth_rate, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_inflation_data(self, start_year=2010, end_year=2023):
        """Generate realistic inflation rate data"""
        print("Generating Inflation Rate data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base inflation based on development level and institutional quality
            dev_level_multiplier = 3 if char['development_level'] == 'Low' else 0
            base_inflation = 3.0 + (1 - char['institutional_quality']) * 5 + dev_level_multiplier
            
            for year in range(start_year, end_year + 1):
                # Add volatility
                volatility = 1.0 + (1 - char['political_stability']) * 3
                shock = np.random.normal(0, volatility)
                
                # Add trend
                trend = np.random.normal(0, 0.5)
                
                inflation_rate = base_inflation + shock + trend
                
                # Ensure realistic bounds
                inflation_rate = np.clip(inflation_rate, -2, 50)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Inflation_Rate_CPI': round(inflation_rate, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_unemployment_data(self, start_year=2010, end_year=2023):
        """Generate realistic unemployment rate data"""
        print("Generating Unemployment Rate data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base unemployment based on development level
            if char['development_level'] == 'Low':
                base_unemployment = random.uniform(8, 15)
            elif char['development_level'] == 'Lower-Middle':
                base_unemployment = random.uniform(5, 12)
            else:  # Upper-Middle
                base_unemployment = random.uniform(3, 8)
            
            for year in range(start_year, end_year + 1):
                # Add volatility
                volatility = 1.0 + (1 - char['political_stability']) * 2
                shock = np.random.normal(0, volatility)
                
                # Add trend
                trend = np.random.normal(0, 0.3)
                
                unemployment_rate = base_unemployment + shock + trend
                
                # Ensure realistic bounds
                unemployment_rate = np.clip(unemployment_rate, 1, 30)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Unemployment_Rate': round(unemployment_rate, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_exchange_rate_volatility_data(self, start_year=2010, end_year=2023):
        """Generate exchange rate volatility data"""
        print("Generating Exchange Rate Volatility data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base volatility based on institutional quality and trade openness
            base_volatility = 5.0 + (1 - char['institutional_quality']) * 10 + (1 - char['trade_openness']) * 5
            
            for year in range(start_year, end_year + 1):
                # Add volatility to volatility (meta-volatility)
                volatility_shock = np.random.normal(0, 2)
                
                # Add trend
                trend = np.random.normal(0, 1)
                
                exchange_volatility = base_volatility + volatility_shock + trend
                
                # Ensure realistic bounds
                exchange_volatility = np.clip(exchange_volatility, 1, 50)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Exchange_Rate_Volatility': round(exchange_volatility, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_interest_rate_data(self, start_year=2010, end_year=2023):
        """Generate central bank policy rate data"""
        print("Generating Interest Rate data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base interest rate based on development level and inflation
            dev_level_multiplier = 3 if char['development_level'] == 'Low' else 0
            base_rate = 5.0 + dev_level_multiplier + (1 - char['institutional_quality']) * 2
            
            for year in range(start_year, end_year + 1):
                # Add volatility
                volatility = 1.0 + (1 - char['political_stability']) * 2
                shock = np.random.normal(0, volatility)
                
                # Add trend
                trend = np.random.normal(0, 0.5)
                
                interest_rate = base_rate + shock + trend
                
                # Ensure realistic bounds
                interest_rate = np.clip(interest_rate, 0, 30)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Central_Bank_Policy_Rate': round(interest_rate, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_money_supply_data(self, start_year=2010, end_year=2023):
        """Generate broad money supply (M2) growth data"""
        print("Generating Money Supply Growth data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base growth based on development level and financial development
            dev_level_multiplier = 5 if char['development_level'] == 'Low' else 0
            base_growth = 8.0 + dev_level_multiplier + (1 - char['financial_development']) * 3
            
            for year in range(start_year, end_year + 1):
                # Add volatility
                volatility = 2.0 + (1 - char['political_stability']) * 3
                shock = np.random.normal(0, volatility)
                
                # Add trend
                trend = np.random.normal(0, 1)
                
                money_growth = base_growth + shock + trend
                
                # Ensure realistic bounds
                money_growth = np.clip(money_growth, -5, 50)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Broad_Money_Supply_M2_Growth': round(money_growth, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_debt_to_gdp_data(self, start_year=2010, end_year=2023):
        """Generate public debt-to-GDP ratio data"""
        print("Generating Public Debt-to-GDP Ratio data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Base debt ratio based on development level and institutional quality
            dev_level_multiplier = 20 if char['development_level'] == 'Low' else 0
            base_debt = 30.0 + dev_level_multiplier + (1 - char['institutional_quality']) * 30
            
            for year in range(start_year, end_year + 1):
                # Add volatility
                volatility = 5.0 + (1 - char['political_stability']) * 10
                shock = np.random.normal(0, volatility)
                
                # Add trend (debt tends to increase over time)
                trend = np.random.normal(1, 0.5) * (year - start_year)
                
                debt_ratio = base_debt + shock + trend
                
                # Ensure realistic bounds
                debt_ratio = np.clip(debt_ratio, 10, 150)
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Public_Debt_to_GDP_Ratio': round(debt_ratio, 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_digital_infrastructure_data(self, start_year=2010, end_year=2023):
        """Generate digital infrastructure data"""
        print("Generating Digital Infrastructure data...")
        
        data = []
        
        for country in self.countries:
            char = self.country_characteristics[country]
            
            # Mobile subscriptions (per 100 people)
            dev_level_mobile = 40 if char['development_level'] == 'Upper-Middle' else 0
            mobile_base = 20.0 + dev_level_mobile + char['institutional_quality'] * 20
            
            # Internet usage (% of population)
            dev_level_internet = 30 if char['development_level'] == 'Upper-Middle' else 0
            internet_base = 10.0 + dev_level_internet + char['institutional_quality'] * 15
            
            # Secure internet servers
            dev_level_servers = 5 if char['development_level'] == 'Upper-Middle' else 0
            servers_base = 1.0 + dev_level_servers + char['institutional_quality'] * 2
            
            for year in range(start_year, end_year + 1):
                # Mobile subscriptions
                mobile_growth = np.random.normal(5, 2)  # Annual growth
                mobile_penetration = min(100, mobile_base + mobile_growth * (year - start_year))
                
                # Internet usage
                internet_growth = np.random.normal(3, 1.5)
                internet_usage = min(100, internet_base + internet_growth * (year - start_year))
                
                # Secure servers
                servers_growth = np.random.normal(0.2, 0.1)
                servers_count = max(0, servers_base + servers_growth * (year - start_year))
                
                data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'Mobile_Cellular_Subscriptions_per_100': round(mobile_penetration, 2),
                    'Internet_Users_Percent': round(internet_usage, 2),
                    'Secure_Internet_Servers': round(servers_count, 2)
                })
        
        return pd.DataFrame(data)
    
    def calculate_gdp_volatility(self, gdp_data):
        """Calculate GDP growth volatility"""
        print("Calculating GDP Growth Volatility...")
        
        volatility_data = []
        
        for country in self.countries:
            country_data = gdp_data[gdp_data['Country'] == country].sort_values('Year')
            
            for i in range(len(country_data)):
                year = country_data.iloc[i]['Year']
                
                # Calculate rolling standard deviation over 3 years
                start_idx = max(0, i - 2)
                end_idx = i + 1
                window_data = country_data.iloc[start_idx:end_idx]['GDP_Growth_Rate']
                
                if len(window_data) >= 2:
                    volatility = window_data.std()
                else:
                    volatility = np.nan
                
                volatility_data.append({
                    'Country': country,
                    'Country_Name': self.country_names[country],
                    'Year': year,
                    'GDP_Growth_Volatility': round(volatility, 2) if not np.isnan(volatility) else np.nan
                })
        
        return pd.DataFrame(volatility_data)
    
    def generate_complete_dataset(self, start_year=2010, end_year=2023):
        """Generate complete macroeconomic dataset"""
        print("Generating complete macroeconomic dataset...")
        print("=" * 60)
        
        # Generate all indicators
        gdp_data = self.generate_gdp_growth_data(start_year, end_year)
        inflation_data = self.generate_inflation_data(start_year, end_year)
        unemployment_data = self.generate_unemployment_data(start_year, end_year)
        exchange_volatility_data = self.generate_exchange_rate_volatility_data(start_year, end_year)
        interest_rate_data = self.generate_interest_rate_data(start_year, end_year)
        money_supply_data = self.generate_money_supply_data(start_year, end_year)
        debt_data = self.generate_debt_to_gdp_data(start_year, end_year)
        digital_data = self.generate_digital_infrastructure_data(start_year, end_year)
        
        # Calculate GDP volatility
        gdp_volatility_data = self.calculate_gdp_volatility(gdp_data)
        
        # Combine all data
        print("\nCombining all data...")
        
        # Start with GDP data
        combined_data = gdp_data.copy()
        
        # Merge other datasets
        datasets = [
            (inflation_data, 'Inflation_Rate_CPI'),
            (unemployment_data, 'Unemployment_Rate'),
            (exchange_volatility_data, 'Exchange_Rate_Volatility'),
            (interest_rate_data, 'Central_Bank_Policy_Rate'),
            (money_supply_data, 'Broad_Money_Supply_M2_Growth'),
            (debt_data, 'Public_Debt_to_GDP_Ratio'),
            (gdp_volatility_data, 'GDP_Growth_Volatility'),
            (digital_data, ['Mobile_Cellular_Subscriptions_per_100', 'Internet_Users_Percent', 'Secure_Internet_Servers'])
        ]
        
        for dataset, columns in datasets:
            if isinstance(columns, list):
                # For digital data with multiple columns
                for col in columns:
                    combined_data = combined_data.merge(
                        dataset[['Country', 'Year', col]], 
                        on=['Country', 'Year'], 
                        how='left'
                    )
            else:
                # For single column data
                combined_data = combined_data.merge(
                    dataset[['Country', 'Year', columns]], 
                    on=['Country', 'Year'], 
                    how='left'
                )
        
        return combined_data
    
    def save_dataset(self, dataset, filename_prefix='fintech_macroeconomic_synthetic'):
        """Save the generated dataset"""
        print(f"\nSaving dataset as {filename_prefix}...")
        
        # Save as CSV
        dataset.to_csv(f'{filename_prefix}.csv', index=False)
        
        # Save as Excel
        with pd.ExcelWriter(f'{filename_prefix}.xlsx', engine='openpyxl') as writer:
            dataset.to_excel(writer, sheet_name='Macroeconomic_Data', index=False)
            
            # Create summary statistics
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            summary_stats = dataset[numeric_cols].describe()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Create country summary
            country_summary = dataset.groupby(['Country', 'Country_Name'])[numeric_cols].mean()
            country_summary.to_excel(writer, sheet_name='Country_Averages')
        
        # Save as JSON
        dataset.to_json(f'{filename_prefix}.json', orient='records', indent=2)
        
        print(f"Dataset saved as:")
        print(f"- {filename_prefix}.csv")
        print(f"- {filename_prefix}.xlsx")
        print(f"- {filename_prefix}.json")
        
        return dataset

def main():
    """Main execution function"""
    print("FinTech Early Warning Model - Synthetic Data Generator")
    print("Sub-Saharan Africa Macroeconomic Data")
    print("=" * 60)
    
    generator = SyntheticMacroeconomicDataGenerator()
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(start_year=2010, end_year=2023)
    
    # Save dataset
    generator.save_dataset(dataset)
    
    print("\n" + "=" * 60)
    print("Synthetic data generation completed successfully!")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Countries: {dataset['Country'].nunique()}")
    print(f"Years: {dataset['Year'].min()} - {dataset['Year'].max()}")
    print(f"Variables: {len(dataset.columns) - 3}")  # Excluding Country, Country_Name, Year
    
    # Display sample data
    print("\nSample data:")
    print(dataset.head(10))

if __name__ == "__main__":
    main()