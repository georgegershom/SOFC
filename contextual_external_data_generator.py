#!/usr/bin/env python3
"""
Contextual & External Data Generator for Dynamic Digital Twin Framework
Multi-Objective Building Retrofit Optimization

This script generates comprehensive datasets for:
- Weather & Climate Data (TMY, Future Projections)
- Economic & Market Data (Energy prices, material costs, labor costs)
- Geospatial & Regulatory Data (Location, carbon intensity, building codes)
"""

import pandas as pd
import numpy as np
import json
import requests
import zipfile
import os
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ContextualDataGenerator:
    def __init__(self, base_path: str = "/workspace/data"):
        self.base_path = base_path
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories for data storage"""
        dirs = [
            f"{self.base_path}/weather_climate",
            f"{self.base_path}/economic_market", 
            f"{self.base_path}/geospatial_regulatory",
            f"{self.base_path}/integrated",
            f"{self.base_path}/raw"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_tmy_data(self, locations: List[Dict], years: int = 20) -> pd.DataFrame:
        """
        Generate Typical Meteorological Year (TMY) data for multiple locations
        Based on ASHRAE standards and real weather patterns
        """
        print("Generating TMY Weather Data...")
        
        tmy_data = []
        
        for location in locations:
            lat, lon, city, country = location['lat'], location['lon'], location['city'], location['country']
            
            # Generate realistic TMY data based on location
            base_temp = self._get_base_temperature(lat)
            seasonal_variation = self._get_seasonal_variation(lat)
            
            for year in range(2020, 2020 + years):
                for month in range(1, 13):
                    days_in_month = 31 if month in [1,3,5,7,8,10,12] else 30 if month in [4,6,9,11] else 28
                    
                    for day in range(1, days_in_month + 1):
                        for hour in range(24):
                            # Generate realistic hourly weather data
                            temp = self._generate_hourly_temperature(base_temp, month, day, hour, lat)
                            humidity = self._generate_humidity(temp, month, lat)
                            wind_speed = self._generate_wind_speed(month, hour)
                            wind_direction = random.uniform(0, 360)
                            pressure = self._generate_pressure(lat, temp)
                            solar_radiation = self._generate_solar_radiation(lat, lon, month, day, hour)
                            cloud_cover = self._generate_cloud_cover(month, hour)
                            
                            tmy_data.append({
                                'location_id': f"{city}_{country}",
                                'latitude': lat,
                                'longitude': lon,
                                'city': city,
                                'country': country,
                                'year': year,
                                'month': month,
                                'day': day,
                                'hour': hour,
                                'datetime': datetime(year, month, day, hour),
                                'dry_bulb_temperature_c': round(temp, 2),
                                'relative_humidity_pct': round(humidity, 1),
                                'wind_speed_mps': round(wind_speed, 2),
                                'wind_direction_deg': round(wind_direction, 1),
                                'atmospheric_pressure_pa': round(pressure, 0),
                                'global_horizontal_irradiance_whm2': round(solar_radiation, 1),
                                'cloud_cover_pct': round(cloud_cover, 1),
                                'precipitation_mm': round(self._generate_precipitation(month, lat), 2)
                            })
        
        df = pd.DataFrame(tmy_data)
        df.to_csv(f"{self.base_path}/weather_climate/tmy_data.csv", index=False)
        print(f"Generated TMY data for {len(locations)} locations, {len(df)} records")
        return df
    
    def generate_future_climate_projections(self, locations: List[Dict], scenarios: List[str] = None) -> pd.DataFrame:
        """
        Generate future climate projections based on IPCC scenarios
        """
        if scenarios is None:
            scenarios = ['RCP2.6', 'RCP4.5', 'RCP8.5']
        
        print("Generating Future Climate Projections...")
        
        climate_data = []
        
        for location in locations:
            lat, lon, city, country = location['lat'], location['lon'], location['city'], location['country']
            
            for scenario in scenarios:
                # Get baseline temperature for the location
                base_temp = self._get_base_temperature(lat)
                
                for year in range(2020, 2100, 10):  # Every 10 years
                    # Calculate temperature increase based on scenario and year
                    temp_increase = self._calculate_temperature_increase(scenario, year)
                    
                    for month in range(1, 13):
                        # Generate monthly climate projections
                        projected_temp = base_temp + temp_increase + self._get_seasonal_variation(lat) * (month - 6) / 6
                        
                        # Project other climate variables
                        humidity_change = self._project_humidity_change(scenario, year)
                        precipitation_change = self._project_precipitation_change(scenario, year, lat)
                        extreme_weather_frequency = self._project_extreme_weather(scenario, year)
                        
                        climate_data.append({
                            'location_id': f"{city}_{country}",
                            'latitude': lat,
                            'longitude': lon,
                            'city': city,
                            'country': country,
                            'scenario': scenario,
                            'year': year,
                            'month': month,
                            'projected_temperature_c': round(projected_temp, 2),
                            'temperature_increase_c': round(temp_increase, 2),
                            'humidity_change_pct': round(humidity_change, 1),
                            'precipitation_change_pct': round(precipitation_change, 1),
                            'extreme_weather_frequency': round(extreme_weather_frequency, 2),
                            'heat_wave_days': self._calculate_heat_wave_days(projected_temp, month),
                            'cooling_degree_days': self._calculate_cdd(projected_temp, 18.3),  # 65°F base
                            'heating_degree_days': self._calculate_hdd(projected_temp, 18.3)
                        })
        
        df = pd.DataFrame(climate_data)
        df.to_csv(f"{self.base_path}/weather_climate/future_climate_projections.csv", index=False)
        print(f"Generated climate projections for {len(locations)} locations, {len(scenarios)} scenarios, {len(df)} records")
        return df
    
    def generate_energy_prices_data(self, regions: List[Dict], years: int = 10) -> pd.DataFrame:
        """
        Generate comprehensive energy pricing data including TOU and demand charges
        """
        print("Generating Energy Prices Data...")
        
        energy_data = []
        
        for region in regions:
            region_name, country, currency = region['name'], region['country'], region['currency']
            
            # Base energy prices by region (realistic ranges)
            base_electricity_price = self._get_base_electricity_price(country)
            base_gas_price = self._get_base_gas_price(country)
            
            for year in range(2020, 2020 + years):
                for month in range(1, 13):
                    for day in range(1, 32):
                        if day > 28 and month == 2:
                            continue
                        if day > 30 and month in [4,6,9,11]:
                            continue
                            
                        for hour in range(24):
                            # Generate time-of-use pricing
                            tou_multiplier = self._get_tou_multiplier(hour, month, day)
                            
                            # Generate seasonal variations
                            seasonal_multiplier = self._get_seasonal_price_multiplier(month, country)
                            
                            # Generate volatility
                            volatility = self._get_price_volatility(year, month)
                            
                            # Calculate final prices
                            electricity_price = base_electricity_price * tou_multiplier * seasonal_multiplier * (1 + volatility)
                            gas_price = base_gas_price * seasonal_multiplier * (1 + volatility * 0.5)
                            
                            # Generate demand charges (for commercial/industrial)
                            demand_charge = self._generate_demand_charge(country, month, hour)
                            
                            # Generate renewable energy credits
                            rec_price = self._generate_rec_price(country, year)
                            
                            energy_data.append({
                                'region_id': region_name,
                                'country': country,
                                'currency': currency,
                                'year': year,
                                'month': month,
                                'day': day,
                                'hour': hour,
                                'datetime': datetime(year, month, day, hour),
                                'electricity_price_per_kwh': round(electricity_price, 4),
                                'natural_gas_price_per_m3': round(gas_price, 4),
                                'demand_charge_per_kw': round(demand_charge, 2),
                                'tou_period': self._get_tou_period(hour, month, day),
                                'renewable_energy_credit_per_kwh': round(rec_price, 4),
                                'carbon_tax_per_tonne_co2': self._get_carbon_tax(country, year),
                                'grid_emissions_factor_kgco2_per_kwh': self._get_grid_emissions_factor(country, hour, month)
                            })
        
        df = pd.DataFrame(energy_data)
        df.to_csv(f"{self.base_path}/economic_market/energy_prices.csv", index=False)
        print(f"Generated energy prices for {len(regions)} regions, {len(df)} records")
        return df
    
    def generate_material_technology_costs(self) -> pd.DataFrame:
        """
        Generate comprehensive material and technology cost data
        """
        print("Generating Material & Technology Costs Data...")
        
        # Define material categories and their cost ranges
        materials = {
            'insulation': {
                'fiberglass_batt': {'base_cost': 0.5, 'unit': 'per_sqft', 'efficiency_range': (0.3, 0.4)},
                'cellulose': {'base_cost': 0.7, 'unit': 'per_sqft', 'efficiency_range': (0.35, 0.45)},
                'spray_foam': {'base_cost': 1.2, 'unit': 'per_sqft', 'efficiency_range': (0.5, 0.7)},
                'rigid_foam': {'base_cost': 0.8, 'unit': 'per_sqft', 'efficiency_range': (0.4, 0.6)},
                'aerogel': {'base_cost': 3.0, 'unit': 'per_sqft', 'efficiency_range': (0.7, 0.9)}
            },
            'windows': {
                'single_pane': {'base_cost': 15, 'unit': 'per_sqft', 'u_value_range': (1.0, 1.2)},
                'double_pane': {'base_cost': 25, 'unit': 'per_sqft', 'u_value_range': (0.3, 0.5)},
                'triple_pane': {'base_cost': 45, 'unit': 'per_sqft', 'u_value_range': (0.15, 0.25)},
                'low_e_coating': {'base_cost': 35, 'unit': 'per_sqft', 'u_value_range': (0.2, 0.4)},
                'smart_windows': {'base_cost': 80, 'unit': 'per_sqft', 'u_value_range': (0.1, 0.3)}
            },
            'hvac': {
                'central_air_standard': {'base_cost': 3000, 'unit': 'per_ton', 'efficiency_range': (13, 16)},
                'central_air_high_efficiency': {'base_cost': 5000, 'unit': 'per_ton', 'efficiency_range': (18, 22)},
                'heat_pump_air_source': {'base_cost': 4000, 'unit': 'per_ton', 'efficiency_range': (3.0, 4.0)},
                'heat_pump_ground_source': {'base_cost': 8000, 'unit': 'per_ton', 'efficiency_range': (4.0, 5.5)},
                'vrf_system': {'base_cost': 6000, 'unit': 'per_ton', 'efficiency_range': (3.5, 4.5)}
            },
            'renewable_energy': {
                'solar_pv_standard': {'base_cost': 2.5, 'unit': 'per_watt', 'efficiency_range': (0.15, 0.18)},
                'solar_pv_high_efficiency': {'base_cost': 3.5, 'unit': 'per_watt', 'efficiency_range': (0.20, 0.22)},
                'wind_turbine_small': {'base_cost': 3000, 'unit': 'per_kw', 'efficiency_range': (0.25, 0.35)},
                'geothermal_heat_pump': {'base_cost': 15000, 'unit': 'per_ton', 'efficiency_range': (4.0, 5.5)},
                'battery_storage': {'base_cost': 500, 'unit': 'per_kwh', 'efficiency_range': (0.85, 0.95)}
            },
            'lighting': {
                'led_standard': {'base_cost': 5, 'unit': 'per_fixture', 'efficiency_range': (80, 100)},
                'led_high_efficiency': {'base_cost': 8, 'unit': 'per_fixture', 'efficiency_range': (100, 120)},
                'smart_lighting': {'base_cost': 15, 'unit': 'per_fixture', 'efficiency_range': (90, 110)},
                'daylighting_systems': {'base_cost': 50, 'unit': 'per_sqft', 'efficiency_range': (0.6, 0.8)}
            }
        }
        
        material_data = []
        
        for category, items in materials.items():
            for material, specs in items.items():
                for year in range(2020, 2030):
                    # Generate cost trends (generally decreasing for technology, stable for materials)
                    cost_trend = self._get_material_cost_trend(category, year)
                    current_cost = specs['base_cost'] * cost_trend
                    
                    # Generate efficiency improvements over time
                    efficiency_improvement = self._get_efficiency_improvement(category, year)
                    if 'efficiency_range' in specs:
                        current_efficiency = np.mean(specs['efficiency_range']) * (1 + efficiency_improvement)
                    else:
                        current_efficiency = 1.0 * (1 + efficiency_improvement)
                    
                    # Generate regional cost variations
                    for region in ['North America', 'Europe', 'Asia', 'Other']:
                        regional_multiplier = self._get_regional_cost_multiplier(region, category)
                        regional_cost = current_cost * regional_multiplier
                        
                        material_data.append({
                            'category': category,
                            'material_name': material,
                            'year': year,
                            'region': region,
                            'base_cost': round(current_cost, 2),
                            'regional_cost': round(regional_cost, 2),
                            'unit': specs['unit'],
                            'efficiency_rating': round(current_efficiency, 3),
                            'efficiency_unit': self._get_efficiency_unit(category),
                            'lifespan_years': self._get_material_lifespan(category, material),
                            'maintenance_cost_annual_pct': self._get_maintenance_cost(category),
                            'installation_complexity': self._get_installation_complexity(category, material),
                            'carbon_footprint_kgco2_per_unit': self._get_carbon_footprint(category, material)
                        })
        
        df = pd.DataFrame(material_data)
        df.to_csv(f"{self.base_path}/economic_market/material_technology_costs.csv", index=False)
        print(f"Generated material costs for {len(material_data)} material-region-year combinations")
        return df
    
    def generate_labor_costs_data(self, regions: List[Dict], years: int = 10) -> pd.DataFrame:
        """
        Generate labor cost data for various retrofit activities
        """
        print("Generating Labor Costs Data...")
        
        labor_activities = {
            'insulation_installation': {'base_hours_per_sqft': 0.5, 'skill_level': 'medium'},
            'window_replacement': {'base_hours_per_sqft': 2.0, 'skill_level': 'high'},
            'hvac_installation': {'base_hours_per_ton': 8.0, 'skill_level': 'high'},
            'solar_pv_installation': {'base_hours_per_kw': 4.0, 'skill_level': 'high'},
            'lighting_retrofit': {'base_hours_per_fixture': 0.5, 'skill_level': 'low'},
            'roofing_work': {'base_hours_per_sqft': 0.3, 'skill_level': 'medium'},
            'electrical_work': {'base_hours_per_outlet': 1.0, 'skill_level': 'high'},
            'plumbing_work': {'base_hours_per_fixture': 2.0, 'skill_level': 'high'},
            'general_construction': {'base_hours_per_sqft': 0.2, 'skill_level': 'medium'}
        }
        
        labor_data = []
        
        for region in regions:
            region_name, country = region['name'], region['country']
            base_wage = self._get_base_wage(country)
            
            for year in range(2020, 2020 + years):
                # Wage inflation
                wage_inflation = self._get_wage_inflation(country, year)
                current_wage = base_wage * (1 + wage_inflation)
                
                for activity, specs in labor_activities.items():
                    skill_multiplier = self._get_skill_multiplier(specs['skill_level'])
                    regional_multiplier = self._get_regional_labor_multiplier(region_name)
                    
                    hourly_rate = current_wage * skill_multiplier * regional_multiplier
                    
                    labor_data.append({
                        'region_id': region_name,
                        'country': country,
                        'year': year,
                        'activity': activity,
                        'skill_level': specs['skill_level'],
                        'base_hours_per_unit': specs.get('base_hours_per_sqft', specs.get('base_hours_per_ton', specs.get('base_hours_per_kw', specs.get('base_hours_per_fixture', specs.get('base_hours_per_outlet', 1.0))))),
                        'unit_type': 'sqft' if 'base_hours_per_sqft' in specs else 'ton' if 'base_hours_per_ton' in specs else 'kw' if 'base_hours_per_kw' in specs else 'fixture' if 'base_hours_per_fixture' in specs else 'outlet',
                        'hourly_rate_usd': round(hourly_rate, 2),
                        'overtime_multiplier': 1.5,
                        'weekend_multiplier': 1.25,
                        'holiday_multiplier': 2.0,
                        'productivity_factor': self._get_productivity_factor(activity, year),
                        'safety_requirements': self._get_safety_requirements(activity),
                        'certification_required': self._get_certification_required(activity)
                    })
        
        df = pd.DataFrame(labor_data)
        df.to_csv(f"{self.base_path}/economic_market/labor_costs.csv", index=False)
        print(f"Generated labor costs for {len(regions)} regions, {len(labor_activities)} activities, {len(df)} records")
        return df
    
    def generate_financial_parameters_data(self, countries: List[str], years: int = 20) -> pd.DataFrame:
        """
        Generate financial parameters including discount rates, inflation, incentives
        """
        print("Generating Financial Parameters Data...")
        
        financial_data = []
        
        for country in countries:
            for year in range(2020, 2020 + years):
                # Generate realistic financial parameters
                discount_rate = self._get_discount_rate(country, year)
                inflation_rate = self._get_inflation_rate(country, year)
                interest_rate = self._get_interest_rate(country, year)
                
                # Generate government incentives
                incentives = self._generate_government_incentives(country, year)
                
                financial_data.append({
                    'country': country,
                    'year': year,
                    'discount_rate_pct': round(discount_rate, 2),
                    'inflation_rate_pct': round(inflation_rate, 2),
                    'interest_rate_pct': round(interest_rate, 2),
                    'real_discount_rate_pct': round(discount_rate - inflation_rate, 2),
                    'energy_tax_credit_pct': round(incentives['energy_tax_credit'], 1),
                    'solar_rebate_per_kw': round(incentives['solar_rebate'], 0),
                    'efficiency_rebate_pct': round(incentives['efficiency_rebate'], 1),
                    'carbon_tax_per_tonne': round(incentives['carbon_tax'], 2),
                    'net_metering_rate': round(incentives['net_metering'], 4),
                    'feed_in_tariff_rate': round(incentives['feed_in_tariff'], 4),
                    'property_tax_exemption_years': incentives['property_tax_exemption'],
                    'accelerated_depreciation_years': incentives['accelerated_depreciation'],
                    'green_bond_availability': incentives['green_bond_availability'],
                    'esg_investment_multiplier': round(incentives['esg_multiplier'], 2)
                })
        
        df = pd.DataFrame(financial_data)
        df.to_csv(f"{self.base_path}/economic_market/financial_parameters.csv", index=False)
        print(f"Generated financial parameters for {len(countries)} countries, {len(df)} records")
        return df
    
    def generate_geospatial_data(self, locations: List[Dict]) -> pd.DataFrame:
        """
        Generate geospatial and location-specific data
        """
        print("Generating Geospatial Data...")
        
        geospatial_data = []
        
        for location in locations:
            lat, lon, city, country = location['lat'], location['lon'], location['city'], location['country']
            
            # Generate location-specific data
            altitude = self._get_altitude(lat, lon)
            urban_context = self._get_urban_context(city, country)
            shading_factor = self._calculate_shading_factor(urban_context)
            wind_exposure = self._get_wind_exposure(lat, lon, altitude)
            solar_potential = self._calculate_solar_potential(lat, lon, altitude)
            
            geospatial_data.append({
                'location_id': f"{city}_{country}",
                'latitude': lat,
                'longitude': lon,
                'city': city,
                'country': country,
                'altitude_m': altitude,
                'urban_context': urban_context,
                'building_density': self._get_building_density(urban_context),
                'shading_factor': round(shading_factor, 3),
                'wind_exposure_class': wind_exposure,
                'solar_potential_kwh_per_sqft': round(solar_potential, 2),
                'climate_zone': self._get_climate_zone(lat, lon),
                'seismic_zone': self._get_seismic_zone(lat, lon),
                'flood_risk': self._get_flood_risk(lat, lon),
                'air_quality_index': self._get_air_quality_index(city, country),
                'noise_level_db': self._get_noise_level(urban_context),
                'transportation_access': self._get_transportation_access(city, country),
                'utility_infrastructure': self._get_utility_infrastructure(city, country)
            })
        
        df = pd.DataFrame(geospatial_data)
        df.to_csv(f"{self.base_path}/geospatial_regulatory/geospatial_data.csv", index=False)
        print(f"Generated geospatial data for {len(locations)} locations")
        return df
    
    def generate_carbon_intensity_data(self, regions: List[Dict], years: int = 10) -> pd.DataFrame:
        """
        Generate carbon intensity factors for grid electricity
        """
        print("Generating Carbon Intensity Data...")
        
        carbon_data = []
        
        for region in regions:
            region_name, country = region['name'], region['country']
            base_emissions = self._get_base_grid_emissions(country)
            
            for year in range(2020, 2020 + years):
                # Generate hourly carbon intensity variations
                for month in range(1, 13):
                    for day in range(1, 32):
                        if day > 28 and month == 2:
                            continue
                        if day > 30 and month in [4,6,9,11]:
                            continue
                            
                        for hour in range(24):
                            # Generate realistic hourly carbon intensity
                            seasonal_factor = self._get_seasonal_carbon_factor(month, country)
                            hourly_factor = self._get_hourly_carbon_factor(hour, month)
                            renewable_factor = self._get_renewable_factor(country, year, month, hour)
                            
                            carbon_intensity = base_emissions * seasonal_factor * hourly_factor * (1 - renewable_factor)
                            
                            carbon_data.append({
                                'region_id': region_name,
                                'country': country,
                                'year': year,
                                'month': month,
                                'day': day,
                                'hour': hour,
                                'datetime': datetime(year, month, day, hour),
                                'carbon_intensity_kgco2_per_kwh': round(carbon_intensity, 4),
                                'renewable_percentage': round(renewable_factor * 100, 1),
                                'coal_percentage': self._get_coal_percentage(country, year),
                                'natural_gas_percentage': self._get_gas_percentage(country, year),
                                'nuclear_percentage': self._get_nuclear_percentage(country, year),
                                'solar_percentage': self._get_solar_percentage(country, year, month, hour),
                                'wind_percentage': self._get_wind_percentage(country, year, month, hour),
                                'hydro_percentage': self._get_hydro_percentage(country, year, month)
                            })
        
        df = pd.DataFrame(carbon_data)
        df.to_csv(f"{self.base_path}/geospatial_regulatory/carbon_intensity.csv", index=False)
        print(f"Generated carbon intensity data for {len(regions)} regions, {len(df)} records")
        return df
    
    def generate_building_codes_data(self, locations: List[Dict]) -> pd.DataFrame:
        """
        Generate building codes and standards data
        """
        print("Generating Building Codes & Standards Data...")
        
        codes_data = []
        
        for location in locations:
            city, country = location['city'], location['country']
            
            # Generate building code requirements
            energy_code = self._get_energy_code(country, city)
            emissions_targets = self._get_emissions_targets(country, city)
            performance_standards = self._get_performance_standards(country, city)
            
            codes_data.append({
                'location_id': f"{city}_{country}",
                'city': city,
                'country': country,
                'energy_code': energy_code['name'],
                'energy_code_version': energy_code['version'],
                'energy_code_year': energy_code['year'],
                'u_value_walls_max': energy_code['u_value_walls'],
                'u_value_roof_max': energy_code['u_value_roof'],
                'u_value_windows_max': energy_code['u_value_windows'],
                'air_tightness_max': energy_code['air_tightness'],
                'hvac_efficiency_min': energy_code['hvac_efficiency'],
                'lighting_power_density_max': energy_code['lighting_power_density'],
                'renewable_energy_requirement_pct': energy_code['renewable_requirement'],
                'emissions_target_2030_pct': emissions_targets['target_2030'],
                'emissions_target_2050_pct': emissions_targets['target_2050'],
                'net_zero_requirement_year': emissions_targets['net_zero_year'],
                'carbon_tax_rate': emissions_targets['carbon_tax'],
                'performance_rating_system': performance_standards['rating_system'],
                'certification_required': performance_standards['certification_required'],
                'inspection_frequency': performance_standards['inspection_frequency'],
                'penalty_rate': performance_standards['penalty_rate'],
                'incentive_programs': json.dumps(performance_standards['incentive_programs'])
            })
        
        df = pd.DataFrame(codes_data)
        df.to_csv(f"{self.base_path}/geospatial_regulatory/building_codes.csv", index=False)
        print(f"Generated building codes data for {len(locations)} locations")
        return df
    
    # Helper methods for data generation
    def _get_base_temperature(self, lat: float) -> float:
        """Get base temperature based on latitude"""
        return 20 - (abs(lat) * 0.5) + random.uniform(-2, 2)
    
    def _get_seasonal_variation(self, lat: float) -> float:
        """Get seasonal temperature variation based on latitude"""
        return 15 + (abs(lat) * 0.3) + random.uniform(-3, 3)
    
    def _generate_hourly_temperature(self, base_temp: float, month: int, day: int, hour: int, lat: float) -> float:
        """Generate realistic hourly temperature"""
        seasonal_temp = base_temp + 10 * np.sin(2 * np.pi * (month - 1) / 12)
        daily_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        random_variation = random.uniform(-2, 2)
        return seasonal_temp + daily_variation + random_variation
    
    def _generate_humidity(self, temp: float, month: int, lat: float) -> float:
        """Generate realistic humidity based on temperature and season"""
        base_humidity = 60 - (temp - 20) * 2
        seasonal_adjustment = 10 * np.sin(2 * np.pi * (month - 1) / 12)
        return max(20, min(90, base_humidity + seasonal_adjustment + random.uniform(-5, 5)))
    
    def _generate_wind_speed(self, month: int, hour: int) -> float:
        """Generate realistic wind speed"""
        base_speed = 3 + 2 * np.sin(2 * np.pi * (month - 1) / 12)
        daily_variation = 1 + np.sin(2 * np.pi * hour / 24)
        return max(0, base_speed + daily_variation + random.uniform(-1, 1))
    
    def _generate_pressure(self, lat: float, temp: float) -> float:
        """Generate atmospheric pressure"""
        altitude = self._get_altitude(lat, 0)  # Get altitude for the latitude
        base_pressure = 101325 - (altitude * 12)  # Simplified altitude effect
        temp_effect = (temp - 15) * 10
        return base_pressure + temp_effect + random.uniform(-100, 100)
    
    def _generate_solar_radiation(self, lat: float, lon: float, month: int, day: int, hour: int) -> float:
        """Generate solar radiation based on location and time"""
        # Simplified solar radiation calculation
        declination = 23.45 * np.sin(2 * np.pi * (284 + (month - 1) * 30 + day) / 365)
        hour_angle = 15 * (hour - 12)
        solar_altitude = np.arcsin(np.sin(np.radians(lat)) * np.sin(np.radians(declination)) + 
                                 np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle)))
        
        if solar_altitude > 0:
            radiation = 1000 * np.sin(solar_altitude) * (0.7 ** (1/np.sin(solar_altitude)))
            return max(0, radiation + random.uniform(-50, 50))
        return 0
    
    def _generate_cloud_cover(self, month: int, hour: int) -> float:
        """Generate cloud cover percentage"""
        base_cover = 40 + 20 * np.sin(2 * np.pi * (month - 1) / 12)
        daily_variation = 10 * np.sin(2 * np.pi * hour / 24)
        return max(0, min(100, base_cover + daily_variation + random.uniform(-10, 10)))
    
    def _generate_precipitation(self, month: int, lat: float) -> float:
        """Generate precipitation amount"""
        if random.random() < 0.3:  # 30% chance of precipitation
            base_precip = 5 + 3 * np.sin(2 * np.pi * (month - 1) / 12)
            return max(0, base_precip + random.uniform(-2, 2))
        return 0
    
    def _calculate_temperature_increase(self, scenario: str, year: int) -> float:
        """Calculate temperature increase based on climate scenario"""
        base_year = 2020
        years_since_base = year - base_year
        
        if scenario == 'RCP2.6':
            return years_since_base * 0.02
        elif scenario == 'RCP4.5':
            return years_since_base * 0.03
        elif scenario == 'RCP8.5':
            return years_since_base * 0.05
        return 0
    
    def _project_humidity_change(self, scenario: str, year: int) -> float:
        """Project humidity changes due to climate change"""
        base_year = 2020
        years_since_base = year - base_year
        
        if scenario == 'RCP2.6':
            return years_since_base * 0.5
        elif scenario == 'RCP4.5':
            return years_since_base * 1.0
        elif scenario == 'RCP8.5':
            return years_since_base * 2.0
        return 0
    
    def _project_precipitation_change(self, scenario: str, year: int, lat: float) -> float:
        """Project precipitation changes due to climate change"""
        base_year = 2020
        years_since_base = year - base_year
        
        # Higher latitudes see more precipitation increase
        lat_factor = 1 + abs(lat) / 90
        
        if scenario == 'RCP2.6':
            return years_since_base * 0.5 * lat_factor
        elif scenario == 'RCP4.5':
            return years_since_base * 1.0 * lat_factor
        elif scenario == 'RCP8.5':
            return years_since_base * 2.0 * lat_factor
        return 0
    
    def _project_extreme_weather(self, scenario: str, year: int) -> float:
        """Project extreme weather frequency"""
        base_year = 2020
        years_since_base = year - base_year
        
        if scenario == 'RCP2.6':
            return 1 + years_since_base * 0.02
        elif scenario == 'RCP4.5':
            return 1 + years_since_base * 0.05
        elif scenario == 'RCP8.5':
            return 1 + years_since_base * 0.1
        return 1
    
    def _calculate_heat_wave_days(self, temp: float, month: int) -> int:
        """Calculate heat wave days based on temperature"""
        if temp > 30 and month in [6, 7, 8]:  # Summer months
            return random.randint(0, 10)
        return 0
    
    def _calculate_cdd(self, temp: float, base_temp: float) -> float:
        """Calculate cooling degree days"""
        return max(0, temp - base_temp)
    
    def _calculate_hdd(self, temp: float, base_temp: float) -> float:
        """Calculate heating degree days"""
        return max(0, base_temp - temp)
    
    def _get_base_electricity_price(self, country: str) -> float:
        """Get base electricity price by country (USD per kWh)"""
        prices = {
            'USA': 0.12, 'Germany': 0.30, 'France': 0.18, 'UK': 0.25,
            'Japan': 0.20, 'China': 0.08, 'India': 0.07, 'Canada': 0.10,
            'Australia': 0.22, 'Brazil': 0.15
        }
        return prices.get(country, 0.15)
    
    def _get_base_gas_price(self, country: str) -> float:
        """Get base natural gas price by country (USD per m³)"""
        prices = {
            'USA': 0.25, 'Germany': 0.80, 'France': 0.60, 'UK': 0.70,
            'Japan': 1.20, 'China': 0.30, 'India': 0.40, 'Canada': 0.20,
            'Australia': 0.50, 'Brazil': 0.35
        }
        return prices.get(country, 0.50)
    
    def _get_tou_multiplier(self, hour: int, month: int, day: int) -> float:
        """Get time-of-use pricing multiplier"""
        # Peak hours: 6-9 AM and 6-9 PM on weekdays
        is_weekday = day % 7 not in [0, 6]  # Simplified weekday check
        is_peak_hour = (6 <= hour <= 9) or (18 <= hour <= 21)
        
        if is_weekday and is_peak_hour:
            return 1.5
        elif is_weekday and (9 <= hour <= 18):
            return 1.2
        else:
            return 0.8
    
    def _get_seasonal_price_multiplier(self, month: int, country: str) -> float:
        """Get seasonal price multiplier"""
        # Higher prices in summer for cooling, winter for heating
        if month in [6, 7, 8]:  # Summer
            return 1.2
        elif month in [12, 1, 2]:  # Winter
            return 1.1
        else:
            return 1.0
    
    def _get_price_volatility(self, year: int, month: int) -> float:
        """Get price volatility factor"""
        # Simulate market volatility
        base_volatility = 0.05
        seasonal_volatility = 0.02 * np.sin(2 * np.pi * month / 12)
        return base_volatility + seasonal_volatility + random.uniform(-0.02, 0.02)
    
    def _generate_demand_charge(self, country: str, month: int, hour: int) -> float:
        """Generate demand charges for commercial/industrial customers"""
        base_charge = 15 if country in ['USA', 'Canada'] else 20
        seasonal_multiplier = 1.2 if month in [6, 7, 8, 12, 1, 2] else 1.0
        return base_charge * seasonal_multiplier
    
    def _generate_rec_price(self, country: str, year: int) -> float:
        """Generate renewable energy credit prices"""
        base_price = 0.02
        year_increase = (year - 2020) * 0.001
        return base_price + year_increase + random.uniform(-0.005, 0.005)
    
    def _get_carbon_tax(self, country: str, year: int) -> float:
        """Get carbon tax rate by country and year"""
        base_rates = {
            'USA': 0, 'Germany': 25, 'France': 45, 'UK': 30,
            'Japan': 20, 'China': 5, 'India': 0, 'Canada': 20,
            'Australia': 15, 'Brazil': 10
        }
        base_rate = base_rates.get(country, 10)
        year_increase = (year - 2020) * 2
        return base_rate + year_increase
    
    def _get_grid_emissions_factor(self, country: str, hour: int, month: int) -> float:
        """Get grid emissions factor (kg CO2 per kWh)"""
        base_emissions = {
            'USA': 0.4, 'Germany': 0.3, 'France': 0.05, 'UK': 0.2,
            'Japan': 0.5, 'China': 0.6, 'India': 0.8, 'Canada': 0.1,
            'Australia': 0.7, 'Brazil': 0.1
        }
        base = base_emissions.get(country, 0.4)
        
        # Lower emissions during peak solar hours
        solar_hours = [10, 11, 12, 13, 14, 15]
        if hour in solar_hours:
            return base * 0.8
        return base
    
    def _get_tou_period(self, hour: int, month: int, day: int) -> str:
        """Get time-of-use period classification"""
        is_weekday = day % 7 not in [0, 6]
        
        if is_weekday and (6 <= hour <= 9):
            return 'peak_morning'
        elif is_weekday and (18 <= hour <= 21):
            return 'peak_evening'
        elif is_weekday and (9 <= hour <= 18):
            return 'mid_peak'
        else:
            return 'off_peak'
    
    def _get_material_cost_trend(self, category: str, year: int) -> float:
        """Get material cost trend over time"""
        if category in ['renewable_energy', 'lighting']:
            # Technology costs generally decrease
            return 1 - (year - 2020) * 0.05
        else:
            # Material costs generally increase with inflation
            return 1 + (year - 2020) * 0.02
    
    def _get_efficiency_improvement(self, category: str, year: int) -> float:
        """Get efficiency improvement over time"""
        if category in ['renewable_energy', 'hvac', 'lighting']:
            return (year - 2020) * 0.02
        return 0
    
    def _get_regional_cost_multiplier(self, region: str, category: str) -> float:
        """Get regional cost multiplier"""
        multipliers = {
            'North America': {'insulation': 1.0, 'windows': 1.0, 'hvac': 1.0, 'renewable_energy': 1.0, 'lighting': 1.0},
            'Europe': {'insulation': 1.2, 'windows': 1.3, 'hvac': 1.1, 'renewable_energy': 0.9, 'lighting': 1.1},
            'Asia': {'insulation': 0.7, 'windows': 0.8, 'hvac': 0.8, 'renewable_energy': 0.8, 'lighting': 0.7},
            'Other': {'insulation': 0.9, 'windows': 1.0, 'hvac': 0.9, 'renewable_energy': 1.1, 'lighting': 0.9}
        }
        return multipliers.get(region, {}).get(category, 1.0)
    
    def _get_efficiency_unit(self, category: str) -> str:
        """Get efficiency unit for category"""
        units = {
            'insulation': 'R-value',
            'windows': 'U-value',
            'hvac': 'SEER/COP',
            'renewable_energy': 'efficiency_pct',
            'lighting': 'lumens_per_watt'
        }
        return units.get(category, 'efficiency_ratio')
    
    def _get_material_lifespan(self, category: str, material: str) -> int:
        """Get material lifespan in years"""
        lifespans = {
            'insulation': {'fiberglass_batt': 50, 'cellulose': 30, 'spray_foam': 40, 'rigid_foam': 50, 'aerogel': 30},
            'windows': {'single_pane': 20, 'double_pane': 30, 'triple_pane': 40, 'low_e_coating': 35, 'smart_windows': 25},
            'hvac': {'central_air_standard': 15, 'central_air_high_efficiency': 20, 'heat_pump_air_source': 15, 'heat_pump_ground_source': 25, 'vrf_system': 20},
            'renewable_energy': {'solar_pv_standard': 25, 'solar_pv_high_efficiency': 30, 'wind_turbine_small': 20, 'geothermal_heat_pump': 25, 'battery_storage': 10},
            'lighting': {'led_standard': 15, 'led_high_efficiency': 20, 'smart_lighting': 15, 'daylighting_systems': 30}
        }
        return lifespans.get(category, {}).get(material, 20)
    
    def _get_maintenance_cost(self, category: str) -> float:
        """Get annual maintenance cost as percentage of initial cost"""
        costs = {
            'insulation': 0.5, 'windows': 1.0, 'hvac': 3.0, 'renewable_energy': 2.0, 'lighting': 1.5
        }
        return costs.get(category, 2.0)
    
    def _get_installation_complexity(self, category: str, material: str) -> str:
        """Get installation complexity level"""
        if category == 'insulation':
            return 'low' if 'batt' in material else 'medium' if 'foam' in material else 'high'
        elif category == 'windows':
            return 'medium' if 'single' in material or 'double' in material else 'high'
        elif category == 'hvac':
            return 'high' if 'ground_source' in material else 'medium'
        elif category == 'renewable_energy':
            return 'high' if 'geothermal' in material else 'medium'
        else:
            return 'low'
    
    def _get_carbon_footprint(self, category: str, material: str) -> float:
        """Get carbon footprint per unit (kg CO2)"""
        # Simplified carbon footprint estimates
        footprints = {
            'insulation': {'fiberglass_batt': 0.5, 'cellulose': 0.3, 'spray_foam': 1.0, 'rigid_foam': 0.8, 'aerogel': 2.0},
            'windows': {'single_pane': 15, 'double_pane': 25, 'triple_pane': 40, 'low_e_coating': 30, 'smart_windows': 50},
            'hvac': {'central_air_standard': 200, 'central_air_high_efficiency': 300, 'heat_pump_air_source': 400, 'heat_pump_ground_source': 600, 'vrf_system': 500},
            'renewable_energy': {'solar_pv_standard': 50, 'solar_pv_high_efficiency': 60, 'wind_turbine_small': 100, 'geothermal_heat_pump': 200, 'battery_storage': 80},
            'lighting': {'led_standard': 2, 'led_high_efficiency': 3, 'smart_lighting': 5, 'daylighting_systems': 10}
        }
        return footprints.get(category, {}).get(material, 10)
    
    def _get_base_wage(self, country: str) -> float:
        """Get base hourly wage by country (USD)"""
        wages = {
            'USA': 25, 'Germany': 30, 'France': 28, 'UK': 22,
            'Japan': 20, 'China': 8, 'India': 5, 'Canada': 22,
            'Australia': 25, 'Brazil': 12
        }
        return wages.get(country, 15)
    
    def _get_wage_inflation(self, country: str, year: int) -> float:
        """Get wage inflation rate"""
        base_inflation = 0.02
        country_adjustment = 0.01 if country in ['USA', 'Germany', 'France'] else 0.0
        return (year - 2020) * (base_inflation + country_adjustment)
    
    def _get_skill_multiplier(self, skill_level: str) -> float:
        """Get skill level multiplier for wages"""
        multipliers = {'low': 1.0, 'medium': 1.3, 'high': 1.8}
        return multipliers.get(skill_level, 1.0)
    
    def _get_regional_labor_multiplier(self, region: str) -> float:
        """Get regional labor cost multiplier"""
        multipliers = {
            'North America': 1.0, 'Europe': 1.2, 'Asia': 0.6, 'Other': 0.8
        }
        return multipliers.get(region, 1.0)
    
    def _get_productivity_factor(self, activity: str, year: int) -> float:
        """Get productivity factor for activity"""
        base_productivity = 1.0
        improvement = (year - 2020) * 0.01  # 1% improvement per year
        return base_productivity + improvement
    
    def _get_safety_requirements(self, activity: str) -> str:
        """Get safety requirements level"""
        if activity in ['hvac_installation', 'electrical_work', 'roofing_work']:
            return 'high'
        elif activity in ['insulation_installation', 'window_replacement']:
            return 'medium'
        else:
            return 'low'
    
    def _get_certification_required(self, activity: str) -> bool:
        """Check if certification is required"""
        return activity in ['hvac_installation', 'electrical_work', 'solar_pv_installation']
    
    def _get_discount_rate(self, country: str, year: int) -> float:
        """Get discount rate by country and year"""
        base_rates = {
            'USA': 0.07, 'Germany': 0.05, 'France': 0.04, 'UK': 0.06,
            'Japan': 0.03, 'China': 0.08, 'India': 0.10, 'Canada': 0.06,
            'Australia': 0.07, 'Brazil': 0.12
        }
        base_rate = base_rates.get(country, 0.07)
        # Add some variation over time
        variation = random.uniform(-0.01, 0.01)
        return base_rate + variation
    
    def _get_inflation_rate(self, country: str, year: int) -> float:
        """Get inflation rate by country and year"""
        base_rates = {
            'USA': 0.025, 'Germany': 0.02, 'France': 0.02, 'UK': 0.03,
            'Japan': 0.01, 'China': 0.03, 'India': 0.05, 'Canada': 0.025,
            'Australia': 0.025, 'Brazil': 0.05
        }
        base_rate = base_rates.get(country, 0.03)
        # Add some variation over time
        variation = random.uniform(-0.005, 0.005)
        return base_rate + variation
    
    def _get_interest_rate(self, country: str, year: int) -> float:
        """Get interest rate by country and year"""
        base_rates = {
            'USA': 0.05, 'Germany': 0.03, 'France': 0.03, 'UK': 0.04,
            'Japan': 0.01, 'China': 0.04, 'India': 0.06, 'Canada': 0.05,
            'Australia': 0.05, 'Brazil': 0.08
        }
        base_rate = base_rates.get(country, 0.05)
        # Add some variation over time
        variation = random.uniform(-0.01, 0.01)
        return base_rate + variation
    
    def _generate_government_incentives(self, country: str, year: int) -> Dict:
        """Generate government incentive data"""
        incentives = {
            'energy_tax_credit': random.uniform(10, 30),
            'solar_rebate': random.uniform(500, 2000),
            'efficiency_rebate': random.uniform(5, 20),
            'carbon_tax': random.uniform(10, 50),
            'net_metering': random.uniform(0.05, 0.15),
            'feed_in_tariff': random.uniform(0.08, 0.20),
            'property_tax_exemption': random.randint(5, 15),
            'accelerated_depreciation': random.randint(3, 7),
            'green_bond_availability': random.choice([True, False]),
            'esg_multiplier': random.uniform(1.0, 1.5)
        }
        return incentives
    
    def _get_altitude(self, lat: float, lon: float) -> float:
        """Get altitude for location (simplified)"""
        # Simplified altitude calculation
        return random.uniform(0, 2000)
    
    def _get_urban_context(self, city: str, country: str) -> str:
        """Get urban context classification"""
        contexts = ['urban_core', 'urban_suburban', 'suburban', 'rural']
        return random.choice(contexts)
    
    def _calculate_shading_factor(self, urban_context: str) -> float:
        """Calculate shading factor based on urban context"""
        factors = {
            'urban_core': 0.7, 'urban_suburban': 0.8, 'suburban': 0.9, 'rural': 1.0
        }
        return factors.get(urban_context, 0.8)
    
    def _get_wind_exposure(self, lat: float, lon: float, altitude: float) -> str:
        """Get wind exposure classification"""
        if altitude > 1000:
            return 'high'
        elif altitude > 500:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_solar_potential(self, lat: float, lon: float, altitude: float) -> float:
        """Calculate solar potential (kWh per sqft per year)"""
        # Simplified solar potential calculation
        base_potential = 4.5 - (abs(lat) * 0.02)
        altitude_bonus = altitude * 0.0001
        return base_potential + altitude_bonus + random.uniform(-0.5, 0.5)
    
    def _get_building_density(self, urban_context: str) -> str:
        """Get building density classification"""
        densities = {
            'urban_core': 'high', 'urban_suburban': 'medium_high', 
            'suburban': 'medium', 'rural': 'low'
        }
        return densities.get(urban_context, 'medium')
    
    def _get_climate_zone(self, lat: float, lon: float) -> str:
        """Get climate zone classification"""
        if abs(lat) < 23.5:
            return 'tropical'
        elif abs(lat) < 35:
            return 'subtropical'
        elif abs(lat) < 50:
            return 'temperate'
        else:
            return 'cold'
    
    def _get_seismic_zone(self, lat: float, lon: float) -> str:
        """Get seismic zone classification"""
        # Simplified seismic zone calculation
        if abs(lat) < 30 and (120 < lon < 150 or -120 < lon < -60):
            return 'high'
        elif abs(lat) < 40 and (100 < lon < 160 or -130 < lon < -70):
            return 'medium'
        else:
            return 'low'
    
    def _get_flood_risk(self, lat: float, lon: float) -> str:
        """Get flood risk classification"""
        # Simplified flood risk calculation
        if abs(lat) < 30 and (lon < 80 or lon > 100):
            return 'high'
        elif abs(lat) < 40:
            return 'medium'
        else:
            return 'low'
    
    def _get_air_quality_index(self, city: str, country: str) -> int:
        """Get air quality index"""
        # Simplified AQI calculation
        base_aqi = 50
        if country in ['China', 'India']:
            base_aqi = 80
        elif country in ['USA', 'Germany', 'France']:
            base_aqi = 40
        return base_aqi + random.randint(-10, 20)
    
    def _get_noise_level(self, urban_context: str) -> float:
        """Get noise level in dB"""
        levels = {
            'urban_core': 70, 'urban_suburban': 60, 'suburban': 50, 'rural': 40
        }
        return levels.get(urban_context, 55) + random.uniform(-5, 5)
    
    def _get_transportation_access(self, city: str, country: str) -> str:
        """Get transportation access level"""
        if country in ['USA', 'Germany', 'France', 'UK']:
            return 'high'
        elif country in ['Japan', 'China']:
            return 'very_high'
        else:
            return 'medium'
    
    def _get_utility_infrastructure(self, city: str, country: str) -> str:
        """Get utility infrastructure quality"""
        if country in ['USA', 'Germany', 'France', 'UK', 'Japan']:
            return 'high'
        elif country in ['China', 'India']:
            return 'medium'
        else:
            return 'low'
    
    def _get_base_grid_emissions(self, country: str) -> float:
        """Get base grid emissions factor"""
        emissions = {
            'USA': 0.4, 'Germany': 0.3, 'France': 0.05, 'UK': 0.2,
            'Japan': 0.5, 'China': 0.6, 'India': 0.8, 'Canada': 0.1,
            'Australia': 0.7, 'Brazil': 0.1
        }
        return emissions.get(country, 0.4)
    
    def _get_seasonal_carbon_factor(self, month: int, country: str) -> float:
        """Get seasonal carbon intensity factor"""
        # Higher emissions in winter due to heating demand
        if month in [12, 1, 2]:
            return 1.2
        elif month in [6, 7, 8]:
            return 0.9
        else:
            return 1.0
    
    def _get_hourly_carbon_factor(self, hour: int, month: int) -> float:
        """Get hourly carbon intensity factor"""
        # Higher emissions during peak hours
        if 6 <= hour <= 9 or 18 <= hour <= 21:
            return 1.1
        else:
            return 0.9
    
    def _get_renewable_factor(self, country: str, year: int, month: int, hour: int) -> float:
        """Get renewable energy factor"""
        base_renewable = {
            'USA': 0.2, 'Germany': 0.4, 'France': 0.8, 'UK': 0.3,
            'Japan': 0.2, 'China': 0.3, 'India': 0.2, 'Canada': 0.6,
            'Australia': 0.2, 'Brazil': 0.8
        }
        base = base_renewable.get(country, 0.3)
        
        # Increase renewable share over time
        year_increase = (year - 2020) * 0.02
        current_renewable = min(0.9, base + year_increase)
        
        # Higher renewable during solar hours
        if 10 <= hour <= 16:
            return current_renewable * 1.2
        return current_renewable
    
    def _get_coal_percentage(self, country: str, year: int) -> float:
        """Get coal percentage in grid mix"""
        base_coal = {
            'USA': 0.2, 'Germany': 0.3, 'France': 0.05, 'UK': 0.1,
            'Japan': 0.3, 'China': 0.6, 'India': 0.7, 'Canada': 0.05,
            'Australia': 0.6, 'Brazil': 0.05
        }
        base = base_coal.get(country, 0.3)
        # Decrease coal over time
        year_decrease = (year - 2020) * 0.02
        return max(0, base - year_decrease)
    
    def _get_gas_percentage(self, country: str, year: int) -> float:
        """Get natural gas percentage in grid mix"""
        base_gas = {
            'USA': 0.4, 'Germany': 0.2, 'France': 0.1, 'UK': 0.4,
            'Japan': 0.4, 'China': 0.1, 'India': 0.1, 'Canada': 0.1,
            'Australia': 0.2, 'Brazil': 0.1
        }
        return base_gas.get(country, 0.2)
    
    def _get_nuclear_percentage(self, country: str, year: int) -> float:
        """Get nuclear percentage in grid mix"""
        base_nuclear = {
            'USA': 0.2, 'Germany': 0.1, 'France': 0.7, 'UK': 0.2,
            'Japan': 0.1, 'China': 0.05, 'India': 0.02, 'Canada': 0.15,
            'Australia': 0.0, 'Brazil': 0.02
        }
        return base_nuclear.get(country, 0.1)
    
    def _get_solar_percentage(self, country: str, year: int, month: int, hour: int) -> float:
        """Get solar percentage in grid mix"""
        base_solar = {
            'USA': 0.1, 'Germany': 0.15, 'France': 0.05, 'UK': 0.1,
            'Japan': 0.1, 'China': 0.1, 'India': 0.1, 'Canada': 0.05,
            'Australia': 0.1, 'Brazil': 0.05
        }
        base = base_solar.get(country, 0.1)
        
        # Increase solar over time
        year_increase = (year - 2020) * 0.01
        current_solar = min(0.3, base + year_increase)
        
        # Higher solar during peak hours
        if 10 <= hour <= 16:
            return current_solar * 2
        return current_solar
    
    def _get_wind_percentage(self, country: str, year: int, month: int, hour: int) -> float:
        """Get wind percentage in grid mix"""
        base_wind = {
            'USA': 0.1, 'Germany': 0.2, 'France': 0.05, 'UK': 0.2,
            'Japan': 0.05, 'China': 0.1, 'India': 0.1, 'Canada': 0.1,
            'Australia': 0.1, 'Brazil': 0.05
        }
        base = base_wind.get(country, 0.1)
        
        # Increase wind over time
        year_increase = (year - 2020) * 0.01
        current_wind = min(0.3, base + year_increase)
        
        # Wind varies by season and hour
        seasonal_factor = 1.2 if month in [10, 11, 12, 1, 2, 3] else 0.8
        return current_wind * seasonal_factor
    
    def _get_hydro_percentage(self, country: str, year: int, month: int) -> float:
        """Get hydro percentage in grid mix"""
        base_hydro = {
            'USA': 0.1, 'Germany': 0.05, 'France': 0.1, 'UK': 0.05,
            'Japan': 0.05, 'China': 0.15, 'India': 0.1, 'Canada': 0.6,
            'Australia': 0.05, 'Brazil': 0.6
        }
        base = base_hydro.get(country, 0.1)
        
        # Hydro varies by season
        seasonal_factor = 1.2 if month in [4, 5, 6, 7, 8, 9] else 0.8
        return base * seasonal_factor
    
    def _get_energy_code(self, country: str, city: str) -> Dict:
        """Get energy code requirements"""
        codes = {
            'USA': {'name': 'IECC', 'version': '2021', 'year': 2021, 'u_value_walls': 0.065, 'u_value_roof': 0.040, 'u_value_windows': 0.32, 'air_tightness': 3.0, 'hvac_efficiency': 14, 'lighting_power_density': 0.9, 'renewable_requirement': 0},
            'Germany': {'name': 'EnEV', 'version': '2016', 'year': 2016, 'u_value_walls': 0.24, 'u_value_roof': 0.20, 'u_value_windows': 1.3, 'air_tightness': 1.5, 'hvac_efficiency': 16, 'lighting_power_density': 0.8, 'renewable_requirement': 15},
            'France': {'name': 'RT2012', 'version': '2012', 'year': 2012, 'u_value_walls': 0.36, 'u_value_roof': 0.20, 'u_value_windows': 1.4, 'air_tightness': 0.6, 'hvac_efficiency': 15, 'lighting_power_density': 0.7, 'renewable_requirement': 20},
            'UK': {'name': 'Part L', 'version': '2021', 'year': 2021, 'u_value_walls': 0.18, 'u_value_roof': 0.13, 'u_value_windows': 1.4, 'air_tightness': 5.0, 'hvac_efficiency': 14, 'lighting_power_density': 0.8, 'renewable_requirement': 10}
        }
        return codes.get(country, codes['USA'])
    
    def _get_emissions_targets(self, country: str, city: str) -> Dict:
        """Get emissions targets"""
        targets = {
            'USA': {'target_2030': 50, 'target_2050': 100, 'net_zero_year': 2050, 'carbon_tax': 0},
            'Germany': {'target_2030': 65, 'target_2050': 100, 'net_zero_year': 2045, 'carbon_tax': 25},
            'France': {'target_2030': 40, 'target_2050': 100, 'net_zero_year': 2050, 'carbon_tax': 45},
            'UK': {'target_2030': 68, 'target_2050': 100, 'net_zero_year': 2050, 'carbon_tax': 30}
        }
        return targets.get(country, targets['USA'])
    
    def _get_performance_standards(self, country: str, city: str) -> Dict:
        """Get performance standards"""
        standards = {
            'USA': {'rating_system': 'LEED', 'certification_required': True, 'inspection_frequency': 'annual', 'penalty_rate': 0.1, 'incentive_programs': ['Energy Star', 'LEED', 'Green Globes']},
            'Germany': {'rating_system': 'DGNB', 'certification_required': True, 'inspection_frequency': 'biennial', 'penalty_rate': 0.15, 'incentive_programs': ['KfW', 'DGNB', 'BREEAM']},
            'France': {'rating_system': 'HQE', 'certification_required': True, 'inspection_frequency': 'biennial', 'penalty_rate': 0.12, 'incentive_programs': ['HQE', 'BBC', 'RT2012']},
            'UK': {'rating_system': 'BREEAM', 'certification_required': True, 'inspection_frequency': 'annual', 'penalty_rate': 0.08, 'incentive_programs': ['BREEAM', 'LEED', 'Passivhaus']}
        }
        return standards.get(country, standards['USA'])

def main():
    """Main function to generate all datasets"""
    print("Starting Contextual & External Data Generation for Digital Twin Framework")
    print("=" * 80)
    
    # Initialize generator
    generator = ContextualDataGenerator()
    
    # Define sample locations and regions
    locations = [
        {'lat': 40.7128, 'lon': -74.0060, 'city': 'New_York', 'country': 'USA'},
        {'lat': 51.5074, 'lon': -0.1278, 'city': 'London', 'country': 'UK'},
        {'lat': 48.8566, 'lon': 2.3522, 'city': 'Paris', 'country': 'France'},
        {'lat': 52.5200, 'lon': 13.4050, 'city': 'Berlin', 'country': 'Germany'},
        {'lat': 35.6762, 'lon': 139.6503, 'city': 'Tokyo', 'country': 'Japan'},
        {'lat': 39.9042, 'lon': 116.4074, 'city': 'Beijing', 'country': 'China'},
        {'lat': 19.0760, 'lon': 72.8777, 'city': 'Mumbai', 'country': 'India'},
        {'lat': 43.6532, 'lon': -79.3832, 'city': 'Toronto', 'country': 'Canada'},
        {'lat': -33.8688, 'lon': 151.2093, 'city': 'Sydney', 'country': 'Australia'},
        {'lat': -23.5505, 'lon': -46.6333, 'city': 'Sao_Paulo', 'country': 'Brazil'}
    ]
    
    regions = [
        {'name': 'North_America_East', 'country': 'USA', 'currency': 'USD'},
        {'name': 'North_America_West', 'country': 'USA', 'currency': 'USD'},
        {'name': 'Europe_West', 'country': 'Germany', 'currency': 'EUR'},
        {'name': 'Europe_North', 'country': 'UK', 'currency': 'GBP'},
        {'name': 'Asia_Pacific', 'country': 'Japan', 'currency': 'JPY'},
        {'name': 'Asia_East', 'country': 'China', 'currency': 'CNY'},
        {'name': 'Asia_South', 'country': 'India', 'currency': 'INR'},
        {'name': 'North_America_North', 'country': 'Canada', 'currency': 'CAD'},
        {'name': 'Oceania', 'country': 'Australia', 'currency': 'AUD'},
        {'name': 'South_America', 'country': 'Brazil', 'currency': 'BRL'}
    ]
    
    countries = ['USA', 'Germany', 'France', 'UK', 'Japan', 'China', 'India', 'Canada', 'Australia', 'Brazil']
    
    # Generate all datasets
    print("\n1. Generating Weather & Climate Data...")
    tmy_data = generator.generate_tmy_data(locations, years=5)
    climate_projections = generator.generate_future_climate_projections(locations)
    
    print("\n2. Generating Economic & Market Data...")
    energy_prices = generator.generate_energy_prices_data(regions, years=5)
    material_costs = generator.generate_material_technology_costs()
    labor_costs = generator.generate_labor_costs_data(regions, years=5)
    financial_params = generator.generate_financial_parameters_data(countries, years=10)
    
    print("\n3. Generating Geospatial & Regulatory Data...")
    geospatial_data = generator.generate_geospatial_data(locations)
    carbon_intensity = generator.generate_carbon_intensity_data(regions, years=5)
    building_codes = generator.generate_building_codes_data(locations)
    
    print("\n4. Creating Integrated Dataset...")
    # Create an integrated summary dataset
    integrated_summary = {
        'dataset_info': {
            'generation_date': datetime.now().isoformat(),
            'total_locations': len(locations),
            'total_regions': len(regions),
            'total_countries': len(countries),
            'tmy_records': len(tmy_data),
            'climate_projection_records': len(climate_projections),
            'energy_price_records': len(energy_prices),
            'material_cost_records': len(material_costs),
            'labor_cost_records': len(labor_costs),
            'financial_param_records': len(financial_params),
            'geospatial_records': len(geospatial_data),
            'carbon_intensity_records': len(carbon_intensity),
            'building_code_records': len(building_codes)
        },
        'data_sources': {
            'weather_climate': ['tmy_data.csv', 'future_climate_projections.csv'],
            'economic_market': ['energy_prices.csv', 'material_technology_costs.csv', 'labor_costs.csv', 'financial_parameters.csv'],
            'geospatial_regulatory': ['geospatial_data.csv', 'carbon_intensity.csv', 'building_codes.csv']
        }
    }
    
    with open(f"{generator.base_path}/integrated/dataset_summary.json", 'w') as f:
        json.dump(integrated_summary, f, indent=2)
    
    print("\n5. Generating Data Validation Scripts...")
    # Create validation script
    validation_script = """
import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_datasets():
    \"\"\"Validate the generated datasets\"\"\"
    base_path = Path("/workspace/data")
    
    # Load all datasets
    datasets = {}
    for category in ['weather_climate', 'economic_market', 'geospatial_regulatory']:
        category_path = base_path / category
        for file in category_path.glob('*.csv'):
            datasets[file.stem] = pd.read_csv(file)
    
    # Validation results
    validation_results = {}
    
    for name, df in datasets.items():
        results = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_ranges': {}
        }
        
        # Check numeric ranges
        for col in df.select_dtypes(include=[np.number]).columns:
            results['numeric_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
        
        validation_results[name] = results
    
    # Save validation results
    with open(base_path / 'integrated' / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print("Dataset validation completed!")
    return validation_results

if __name__ == "__main__":
    validate_datasets()
"""
    
    with open(f"{generator.base_path}/integrated/validate_datasets.py", 'w') as f:
        f.write(validation_script)
    
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print(f"All data saved to: {generator.base_path}")
    print("\nGenerated Files:")
    print("- Weather & Climate: TMY data, Future climate projections")
    print("- Economic & Market: Energy prices, Material costs, Labor costs, Financial parameters")
    print("- Geospatial & Regulatory: Location data, Carbon intensity, Building codes")
    print("- Integrated: Dataset summary and validation scripts")
    print("\nNext steps:")
    print("1. Run validation script: python /workspace/data/integrated/validate_datasets.py")
    print("2. Use datasets in your Digital Twin Framework")
    print("3. Integrate with real-time IoT data and LCA models")

if __name__ == "__main__":
    main()