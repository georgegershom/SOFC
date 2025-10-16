#!/usr/bin/env python3
"""
Energy Prices Data Generator
Generates comprehensive energy pricing data including historical trends, forecasts,
time-of-use rates, and demand charges for building retrofit analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class EnergyPricesGenerator:
    def __init__(self, location="New York", utility="ConEd"):
        self.location = location
        self.utility = utility
        
        # Base energy prices ($/kWh, $/therm, etc.) - 2023 baseline
        self.base_prices = {
            'electricity': {
                'residential': 0.25,
                'commercial': 0.18,
                'industrial': 0.12
            },
            'natural_gas': {
                'residential': 1.20,  # $/therm
                'commercial': 0.95,
                'industrial': 0.75
            },
            'fuel_oil': {
                'heating_oil': 3.85,  # $/gallon
                'diesel': 4.20
            },
            'district_heating': {
                'steam': 45.0,  # $/MWh
                'hot_water': 35.0
            }
        }
        
        # Time-of-use periods
        self.tou_periods = {
            'peak': {'hours': [16, 17, 18, 19, 20], 'multiplier': 1.8},
            'shoulder': {'hours': [7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22], 'multiplier': 1.2},
            'off_peak': {'hours': [23, 0, 1, 2, 3, 4, 5, 6], 'multiplier': 0.7}
        }
        
        # Seasonal multipliers
        self.seasonal_multipliers = {
            'winter': {'months': [12, 1, 2], 'electricity': 1.1, 'gas': 1.3},
            'spring': {'months': [3, 4, 5], 'electricity': 0.95, 'gas': 0.9},
            'summer': {'months': [6, 7, 8], 'electricity': 1.4, 'gas': 0.8},
            'fall': {'months': [9, 10, 11], 'electricity': 1.0, 'gas': 1.0}
        }
    
    def generate_historical_prices(self, start_year=2015, end_year=2023):
        """Generate historical energy price trends"""
        
        years = list(range(start_year, end_year + 1))
        historical_data = []
        
        for year in years:
            # Generate annual trends with realistic volatility
            np.random.seed(year)
            
            # Electricity price trend (generally increasing)
            elec_trend = 1 + (year - start_year) * 0.03  # 3% annual increase
            elec_volatility = np.random.normal(1, 0.08)  # 8% volatility
            
            # Natural gas price trend (more volatile)
            gas_trend = 1 + (year - start_year) * 0.02  # 2% annual increase
            gas_volatility = np.random.normal(1, 0.15)  # 15% volatility
            
            # Oil price trend (highly volatile)
            oil_trend = 1 + (year - start_year) * 0.025  # 2.5% annual increase
            oil_volatility = np.random.normal(1, 0.25)  # 25% volatility
            
            for customer_type in ['residential', 'commercial', 'industrial']:
                historical_data.append({
                    'year': year,
                    'customer_type': customer_type,
                    'electricity_price_kwh': self.base_prices['electricity'][customer_type] * elec_trend * elec_volatility,
                    'natural_gas_price_therm': self.base_prices['natural_gas'][customer_type] * gas_trend * gas_volatility,
                    'heating_oil_price_gallon': self.base_prices['fuel_oil']['heating_oil'] * oil_trend * oil_volatility,
                    'location': self.location,
                    'utility': self.utility
                })
        
        return pd.DataFrame(historical_data)
    
    def generate_forecast_prices(self, start_year=2024, end_year=2040):
        """Generate forecasted energy prices with uncertainty bands"""
        
        years = list(range(start_year, end_year + 1))
        forecast_data = []
        
        # Define forecast scenarios
        scenarios = {
            'low': {'elec_growth': 0.015, 'gas_growth': 0.01, 'oil_growth': 0.02},
            'base': {'elec_growth': 0.03, 'gas_growth': 0.025, 'oil_growth': 0.03},
            'high': {'elec_growth': 0.045, 'gas_growth': 0.04, 'oil_growth': 0.045}
        }
        
        for scenario_name, scenario_params in scenarios.items():
            for year in years:
                years_from_base = year - 2023
                
                # Calculate price escalation
                elec_multiplier = (1 + scenario_params['elec_growth']) ** years_from_base
                gas_multiplier = (1 + scenario_params['gas_growth']) ** years_from_base
                oil_multiplier = (1 + scenario_params['oil_growth']) ** years_from_base
                
                for customer_type in ['residential', 'commercial', 'industrial']:
                    forecast_data.append({
                        'year': year,
                        'scenario': scenario_name,
                        'customer_type': customer_type,
                        'electricity_price_kwh': self.base_prices['electricity'][customer_type] * elec_multiplier,
                        'natural_gas_price_therm': self.base_prices['natural_gas'][customer_type] * gas_multiplier,
                        'heating_oil_price_gallon': self.base_prices['fuel_oil']['heating_oil'] * oil_multiplier,
                        'location': self.location,
                        'utility': self.utility
                    })
        
        return pd.DataFrame(forecast_data)
    
    def generate_tou_rates(self, year=2023):
        """Generate detailed time-of-use electricity rates"""
        
        # Create hourly data for full year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
        
        tou_data = []
        
        for dt in date_range:
            hour = dt.hour
            month = dt.month
            
            # Determine TOU period
            period = 'off_peak'  # default
            for period_name, period_info in self.tou_periods.items():
                if hour in period_info['hours']:
                    period = period_name
                    break
            
            # Determine seasonal multiplier
            seasonal_mult = 1.0
            for season_name, season_info in self.seasonal_multipliers.items():
                if month in season_info['months']:
                    seasonal_mult = season_info['electricity']
                    break
            
            # Calculate rates for each customer type
            for customer_type in ['residential', 'commercial', 'industrial']:
                base_rate = self.base_prices['electricity'][customer_type]
                tou_multiplier = self.tou_periods[period]['multiplier']
                
                final_rate = base_rate * tou_multiplier * seasonal_mult
                
                tou_data.append({
                    'datetime': dt,
                    'year': dt.year,
                    'month': dt.month,
                    'day': dt.day,
                    'hour': dt.hour,
                    'weekday': dt.weekday(),
                    'customer_type': customer_type,
                    'tou_period': period,
                    'season': season_name,
                    'electricity_rate_kwh': final_rate,
                    'base_rate': base_rate,
                    'tou_multiplier': tou_multiplier,
                    'seasonal_multiplier': seasonal_mult,
                    'location': self.location,
                    'utility': self.utility
                })
        
        return pd.DataFrame(tou_data)
    
    def generate_demand_charges(self):
        """Generate demand charge structures"""
        
        demand_charges = {
            'residential': {
                'has_demand_charge': False,
                'demand_charge_kw': 0.0,
                'description': 'Residential customers typically do not have demand charges'
            },
            'commercial': {
                'has_demand_charge': True,
                'demand_charge_kw': 15.50,
                'peak_demand_charge_kw': 22.75,
                'time_periods': {
                    'on_peak': {'months': [6, 7, 8, 9], 'hours': [12, 13, 14, 15, 16, 17, 18, 19]},
                    'off_peak': 'all_other_times'
                },
                'description': 'Commercial demand charges based on peak kW usage'
            },
            'industrial': {
                'has_demand_charge': True,
                'demand_charge_kw': 12.25,
                'peak_demand_charge_kw': 18.90,
                'ratchet_percentage': 0.75,
                'time_periods': {
                    'on_peak': {'months': [6, 7, 8, 9], 'hours': [12, 13, 14, 15, 16, 17, 18, 19]},
                    'off_peak': 'all_other_times'
                },
                'description': 'Industrial demand charges with ratchet clause'
            }
        }
        
        return demand_charges
    
    def generate_renewable_energy_prices(self, year=2023):
        """Generate renewable energy pricing data"""
        
        renewable_data = []
        
        # Solar PPA prices (Power Purchase Agreement)
        solar_ppa_prices = {
            'utility_scale': 0.048,  # $/kWh
            'commercial_rooftop': 0.065,
            'residential_rooftop': 0.085
        }
        
        # Wind PPA prices
        wind_ppa_prices = {
            'onshore': 0.042,
            'offshore': 0.078
        }
        
        # Renewable Energy Certificate (REC) prices
        rec_prices = {
            'solar': 15.50,  # $/MWh
            'wind': 12.25,
            'hydro': 8.75,
            'biomass': 18.90
        }
        
        # Carbon offset prices
        carbon_offset_prices = {
            'voluntary': 8.50,  # $/tonne CO2e
            'compliance': 25.75,
            'california_cap_trade': 28.90
        }
        
        renewable_data.append({
            'year': year,
            'solar_ppa_prices': solar_ppa_prices,
            'wind_ppa_prices': wind_ppa_prices,
            'rec_prices': rec_prices,
            'carbon_offset_prices': carbon_offset_prices,
            'location': self.location
        })
        
        return renewable_data
    
    def generate_net_metering_rates(self):
        """Generate net metering compensation rates"""
        
        net_metering = {
            'policy_type': 'net_energy_metering',
            'compensation_method': 'retail_rate',
            'excess_credit_treatment': 'annual_rollover',
            'system_size_limit_kw': 2000,
            'rates_by_customer_type': {
                'residential': {
                    'export_rate_kwh': 0.22,  # Slightly below retail rate
                    'import_rate_kwh': 0.25,
                    'monthly_connection_fee': 12.50
                },
                'commercial': {
                    'export_rate_kwh': 0.16,
                    'import_rate_kwh': 0.18,
                    'monthly_connection_fee': 25.00,
                    'demand_charge_applies': True
                },
                'industrial': {
                    'export_rate_kwh': 0.10,
                    'import_rate_kwh': 0.12,
                    'monthly_connection_fee': 50.00,
                    'demand_charge_applies': True
                }
            },
            'location': self.location,
            'utility': self.utility,
            'last_updated': datetime.now().isoformat()
        }
        
        return net_metering

def main():
    """Generate energy pricing data for multiple locations"""
    
    locations = [
        {"name": "New York", "utility": "ConEd"},
        {"name": "California", "utility": "PG&E"},
        {"name": "Texas", "utility": "ERCOT"},
        {"name": "Florida", "utility": "FPL"},
        {"name": "Illinois", "utility": "ComEd"}
    ]
    
    for location_info in locations:
        print(f"Generating energy pricing data for {location_info['name']}...")
        
        generator = EnergyPricesGenerator(
            location=location_info['name'],
            utility=location_info['utility']
        )
        
        # Generate all pricing datasets
        historical_prices = generator.generate_historical_prices()
        forecast_prices = generator.generate_forecast_prices()
        tou_rates = generator.generate_tou_rates()
        demand_charges = generator.generate_demand_charges()
        renewable_prices = generator.generate_renewable_energy_prices()
        net_metering = generator.generate_net_metering_rates()
        
        # Create location directory
        location_dir = f"economic_market/energy_prices_{location_info['name'].lower()}"
        os.makedirs(location_dir, exist_ok=True)
        
        # Save datasets
        historical_prices.to_csv(f"{location_dir}/historical_prices_2015_2023.csv", index=False)
        forecast_prices.to_csv(f"{location_dir}/forecast_prices_2024_2040.csv", index=False)
        tou_rates.to_csv(f"{location_dir}/tou_rates_2023.csv", index=False)
        
        # Save JSON data
        with open(f"{location_dir}/demand_charges.json", 'w') as f:
            json.dump(demand_charges, f, indent=2)
        
        with open(f"{location_dir}/renewable_energy_prices.json", 'w') as f:
            json.dump(renewable_prices, f, indent=2, default=str)
        
        with open(f"{location_dir}/net_metering_rates.json", 'w') as f:
            json.dump(net_metering, f, indent=2)
        
        print(f"Saved energy pricing data for {location_info['name']}")

if __name__ == "__main__":
    main()