#!/usr/bin/env python3
"""
Generate energy performance data for building retrofit research.
Creates historical energy consumption data, efficiency ratings,
and post-retrofit performance improvements.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

class EnergyPerformanceGenerator:
    def __init__(self):
        self.efficiency_ratings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.energy_star_ratings = list(range(1, 101))  # 1-100 scale
        self.retrofit_types = [
            'insulation_upgrade', 'window_replacement', 'hvac_upgrade', 
            'lighting_upgrade', 'roof_insulation', 'air_sealing',
            'renewable_energy', 'smart_controls', 'comprehensive'
        ]
        
    def generate_historical_consumption(self, building_id: str, building_type: str, 
                                      construction_year: int, floor_area: float) -> pd.DataFrame:
        """Generate historical energy consumption data."""
        
        # Base energy intensity by building type (kWh/m²/year)
        base_intensities = {
            'residential': {'min': 80, 'max': 200, 'typical': 120},
            'office': {'min': 150, 'max': 400, 'typical': 250},
            'retail': {'min': 200, 'max': 500, 'typical': 300},
            'educational': {'min': 100, 'max': 300, 'typical': 180},
            'healthcare': {'min': 200, 'max': 600, 'typical': 350},
            'industrial': {'min': 300, 'max': 800, 'typical': 500}
        }
        
        config = base_intensities.get(building_type, base_intensities['office'])
        
        # Age factor (older buildings typically consume more)
        age_factor = 1.0 + (2023 - construction_year) * 0.01
        
        # Generate consumption for each year (2018-2023)
        years = list(range(2018, 2024))
        consumption_data = []
        
        for year in years:
            # Base consumption with some year-to-year variation
            base_intensity = random.uniform(config['min'], config['max'])
            yearly_intensity = base_intensity * age_factor * random.uniform(0.9, 1.1)
            
            # Calculate total consumption
            total_consumption = yearly_intensity * floor_area
            
            # Breakdown by fuel type
            fuel_breakdown = self.generate_fuel_breakdown(building_type, total_consumption)
            
            # Monthly breakdown
            monthly_consumption = self.generate_monthly_breakdown(total_consumption, building_type)
            
            consumption_data.append({
                'building_id': building_id,
                'year': year,
                'total_consumption_kwh': round(total_consumption, 2),
                'energy_intensity_kwh_m2': round(yearly_intensity, 2),
                'electricity_kwh': round(fuel_breakdown['electricity'], 2),
                'natural_gas_kwh': round(fuel_breakdown['natural_gas'], 2),
                'oil_kwh': round(fuel_breakdown['oil'], 2),
                'district_heating_kwh': round(fuel_breakdown['district_heating'], 2),
                'renewable_kwh': round(fuel_breakdown['renewable'], 2),
                'jan_consumption': round(monthly_consumption[0], 2),
                'feb_consumption': round(monthly_consumption[1], 2),
                'mar_consumption': round(monthly_consumption[2], 2),
                'apr_consumption': round(monthly_consumption[3], 2),
                'may_consumption': round(monthly_consumption[4], 2),
                'jun_consumption': round(monthly_consumption[5], 2),
                'jul_consumption': round(monthly_consumption[6], 2),
                'aug_consumption': round(monthly_consumption[7], 2),
                'sep_consumption': round(monthly_consumption[8], 2),
                'oct_consumption': round(monthly_consumption[9], 2),
                'nov_consumption': round(monthly_consumption[10], 2),
                'dec_consumption': round(monthly_consumption[11], 2)
            })
        
        return pd.DataFrame(consumption_data)
    
    def generate_fuel_breakdown(self, building_type: str, total_consumption: float) -> Dict:
        """Generate fuel type breakdown based on building type."""
        
        # Fuel type preferences by building type
        fuel_preferences = {
            'residential': {'electricity': 0.4, 'natural_gas': 0.5, 'oil': 0.08, 'district_heating': 0.02, 'renewable': 0.0},
            'office': {'electricity': 0.7, 'natural_gas': 0.25, 'oil': 0.03, 'district_heating': 0.02, 'renewable': 0.0},
            'retail': {'electricity': 0.8, 'natural_gas': 0.15, 'oil': 0.03, 'district_heating': 0.02, 'renewable': 0.0},
            'educational': {'electricity': 0.6, 'natural_gas': 0.3, 'oil': 0.05, 'district_heating': 0.05, 'renewable': 0.0},
            'healthcare': {'electricity': 0.65, 'natural_gas': 0.25, 'oil': 0.05, 'district_heating': 0.05, 'renewable': 0.0},
            'industrial': {'electricity': 0.5, 'natural_gas': 0.3, 'oil': 0.15, 'district_heating': 0.05, 'renewable': 0.0}
        }
        
        preferences = fuel_preferences.get(building_type, fuel_preferences['office'])
        
        breakdown = {}
        for fuel, proportion in preferences.items():
            # Add some randomness
            actual_proportion = proportion * random.uniform(0.8, 1.2)
            breakdown[fuel] = total_consumption * actual_proportion
        
        # Normalize to ensure total adds up
        total_actual = sum(breakdown.values())
        for fuel in breakdown:
            breakdown[fuel] = breakdown[fuel] * total_consumption / total_actual
        
        return breakdown
    
    def generate_monthly_breakdown(self, total_consumption: float, building_type: str) -> List[float]:
        """Generate monthly consumption breakdown with seasonal patterns."""
        
        # Monthly factors based on building type and climate
        if building_type == 'residential':
            # Residential: higher in winter (heating) and summer (cooling)
            monthly_factors = [1.2, 1.1, 1.0, 0.8, 0.7, 0.8, 1.0, 1.1, 0.9, 0.8, 1.0, 1.2]
        elif building_type == 'office':
            # Office: more consistent, slight summer peak
            monthly_factors = [0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 0.9, 0.9]
        elif building_type == 'retail':
            # Retail: higher in winter (holiday season)
            monthly_factors = [1.3, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2]
        else:
            # Default pattern
            monthly_factors = [1.1, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.1, 1.0, 0.9, 1.0, 1.1]
        
        # Normalize factors
        total_factor = sum(monthly_factors)
        monthly_factors = [f / total_factor for f in monthly_factors]
        
        # Generate monthly consumption
        monthly_consumption = [total_consumption * factor * random.uniform(0.95, 1.05) 
                             for factor in monthly_factors]
        
        return monthly_consumption
    
    def generate_efficiency_ratings(self, building_id: str, building_type: str, 
                                   construction_year: int, energy_intensity: float) -> Dict:
        """Generate energy efficiency ratings."""
        
        # EU A-G rating based on energy intensity and building type
        rating_thresholds = {
            'residential': {'A': 50, 'B': 75, 'C': 100, 'D': 130, 'E': 160, 'F': 200},
            'office': {'A': 100, 'B': 150, 'C': 200, 'D': 250, 'E': 300, 'F': 400},
            'retail': {'A': 150, 'B': 200, 'C': 250, 'D': 300, 'E': 400, 'F': 500},
            'educational': {'A': 80, 'B': 120, 'C': 160, 'D': 200, 'E': 250, 'F': 300},
            'healthcare': {'A': 120, 'B': 180, 'C': 240, 'D': 300, 'E': 400, 'F': 500},
            'industrial': {'A': 200, 'B': 300, 'C': 400, 'D': 500, 'E': 600, 'F': 800}
        }
        
        thresholds = rating_thresholds.get(building_type, rating_thresholds['office'])
        
        # Determine EU rating
        eu_rating = 'G'  # Default
        for rating, threshold in thresholds.items():
            if energy_intensity <= threshold:
                eu_rating = rating
                break
        
        # ENERGY STAR score (1-100, higher is better)
        if energy_intensity <= thresholds['A']:
            energy_star_score = random.randint(85, 100)
        elif energy_intensity <= thresholds['B']:
            energy_star_score = random.randint(70, 84)
        elif energy_intensity <= thresholds['C']:
            energy_star_score = random.randint(55, 69)
        elif energy_intensity <= thresholds['D']:
            energy_star_score = random.randint(40, 54)
        elif energy_intensity <= thresholds['E']:
            energy_star_score = random.randint(25, 39)
        else:
            energy_star_score = random.randint(1, 24)
        
        # LEED certification probability (higher for better buildings)
        leed_probability = min(0.8, energy_star_score / 100)
        leed_certified = random.random() < leed_probability
        
        leed_levels = ['Certified', 'Silver', 'Gold', 'Platinum']
        leed_level = random.choice(leed_levels) if leed_certified else None
        
        return {
            'building_id': building_id,
            'eu_rating': eu_rating,
            'energy_star_score': energy_star_score,
            'leed_certified': leed_certified,
            'leed_level': leed_level,
            'energy_intensity_kwh_m2': round(energy_intensity, 2)
        }
    
    def generate_retrofit_data(self, building_id: str, building_type: str, 
                             construction_year: int, current_consumption: float) -> Dict:
        """Generate retrofit scenarios and performance data."""
        
        # Determine if building has been retrofitted
        retrofit_probability = min(0.7, (2023 - construction_year) / 50)  # Older buildings more likely
        has_retrofit = random.random() < retrofit_probability
        
        if not has_retrofit:
            return {
                'building_id': building_id,
                'has_retrofit': False,
                'retrofit_year': None,
                'retrofit_type': None,
                'retrofit_cost_eur': None,
                'energy_savings_percent': None,
                'payback_period_years': None,
                'co2_reduction_percent': None
            }
        
        # Generate retrofit details
        retrofit_year = random.randint(2015, 2023)
        retrofit_type = random.choice(self.retrofit_types)
        
        # Retrofit cost based on building type and retrofit type
        cost_per_m2 = {
            'insulation_upgrade': {'min': 50, 'max': 150},
            'window_replacement': {'min': 200, 'max': 400},
            'hvac_upgrade': {'min': 100, 'max': 300},
            'lighting_upgrade': {'min': 20, 'max': 80},
            'roof_insulation': {'min': 80, 'max': 200},
            'air_sealing': {'min': 10, 'max': 30},
            'renewable_energy': {'min': 500, 'max': 1500},
            'smart_controls': {'min': 30, 'max': 100},
            'comprehensive': {'min': 200, 'max': 600}
        }
        
        cost_range = cost_per_m2.get(retrofit_type, cost_per_m2['insulation_upgrade'])
        cost_per_m2_actual = random.uniform(cost_range['min'], cost_range['max'])
        
        # Assume average floor area for cost calculation
        avg_floor_area = 1000  # m²
        retrofit_cost = cost_per_m2_actual * avg_floor_area
        
        # Energy savings based on retrofit type
        savings_ranges = {
            'insulation_upgrade': {'min': 15, 'max': 30},
            'window_replacement': {'min': 10, 'max': 25},
            'hvac_upgrade': {'min': 20, 'max': 40},
            'lighting_upgrade': {'min': 5, 'max': 15},
            'roof_insulation': {'min': 10, 'max': 20},
            'air_sealing': {'min': 5, 'max': 15},
            'renewable_energy': {'min': 30, 'max': 60},
            'smart_controls': {'min': 5, 'max': 20},
            'comprehensive': {'min': 40, 'max': 70}
        }
        
        savings_range = savings_ranges.get(retrofit_type, savings_ranges['insulation_upgrade'])
        energy_savings = random.uniform(savings_range['min'], savings_range['max'])
        
        # Calculate payback period
        annual_savings = current_consumption * (energy_savings / 100) * 0.12  # €0.12/kWh
        payback_period = retrofit_cost / annual_savings if annual_savings > 0 else None
        
        # CO2 reduction (correlated with energy savings)
        co2_reduction = energy_savings * random.uniform(0.8, 1.2)
        
        return {
            'building_id': building_id,
            'has_retrofit': True,
            'retrofit_year': retrofit_year,
            'retrofit_type': retrofit_type,
            'retrofit_cost_eur': round(retrofit_cost, 2),
            'energy_savings_percent': round(energy_savings, 2),
            'payback_period_years': round(payback_period, 1) if payback_period else None,
            'co2_reduction_percent': round(co2_reduction, 2)
        }
    
    def generate_energy_performance_data(self, buildings: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete energy performance data for all buildings."""
        
        historical_data = []
        efficiency_data = []
        retrofit_data = []
        
        for building in buildings:
            building_id = building['id']
            building_type = building['type']
            construction_year = building['construction_year']
            floor_area = building['floor_area_m2']
            
            print(f"Generating energy performance for building {building_id} ({building_type})...")
            
            # Generate historical consumption
            hist_df = self.generate_historical_consumption(building_id, building_type, construction_year, floor_area)
            historical_data.append(hist_df)
            
            # Use latest year's data for efficiency ratings
            latest_consumption = hist_df[hist_df['year'] == 2023].iloc[0]
            energy_intensity = latest_consumption['energy_intensity_kwh_m2']
            
            # Generate efficiency ratings
            efficiency_dict = self.generate_efficiency_ratings(building_id, building_type, construction_year, energy_intensity)
            efficiency_data.append(efficiency_dict)
            
            # Generate retrofit data
            retrofit_dict = self.generate_retrofit_data(building_id, building_type, construction_year, latest_consumption['total_consumption_kwh'])
            retrofit_data.append(retrofit_dict)
        
        # Combine all data
        historical_df = pd.concat(historical_data, ignore_index=True)
        efficiency_df = pd.DataFrame(efficiency_data)
        retrofit_df = pd.DataFrame(retrofit_data)
        
        return historical_df, efficiency_df, retrofit_df

def main():
    """Generate energy performance data for the building retrofit dataset."""
    
    # Sample buildings (would normally load from building attributes)
    buildings = [
        {'id': 'B001', 'type': 'residential', 'construction_year': 1985, 'floor_area_m2': 120},
        {'id': 'B002', 'type': 'office', 'construction_year': 1995, 'floor_area_m2': 2500},
        {'id': 'B003', 'type': 'retail', 'construction_year': 2005, 'floor_area_m2': 800},
        {'id': 'B004', 'type': 'educational', 'construction_year': 1970, 'floor_area_m2': 3000},
        {'id': 'B005', 'type': 'residential', 'construction_year': 2010, 'floor_area_m2': 150},
        {'id': 'B006', 'type': 'office', 'construction_year': 1980, 'floor_area_m2': 1800},
        {'id': 'B007', 'type': 'retail', 'construction_year': 1990, 'floor_area_m2': 1200},
        {'id': 'B008', 'type': 'educational', 'construction_year': 2000, 'floor_area_m2': 4000},
        {'id': 'B009', 'type': 'residential', 'construction_year': 1965, 'floor_area_m2': 100},
        {'id': 'B010', 'type': 'office', 'construction_year': 2015, 'floor_area_m2': 3200},
        {'id': 'B011', 'type': 'healthcare', 'construction_year': 1998, 'floor_area_m2': 5000},
        {'id': 'B012', 'type': 'industrial', 'construction_year': 1988, 'floor_area_m2': 8000},
        {'id': 'B013', 'type': 'residential', 'construction_year': 2008, 'floor_area_m2': 180},
        {'id': 'B014', 'type': 'office', 'construction_year': 1975, 'floor_area_m2': 2200},
        {'id': 'B015', 'type': 'retail', 'construction_year': 2012, 'floor_area_m2': 1500}
    ]
    
    # Initialize generator
    generator = EnergyPerformanceGenerator()
    
    # Generate energy performance data
    print("Generating energy performance data...")
    historical_df, efficiency_df, retrofit_df = generator.generate_energy_performance_data(buildings)
    
    # Save data
    output_dir = '../raw_data/energy_performance'
    os.makedirs(output_dir, exist_ok=True)
    
    historical_df.to_csv(f"{output_dir}/historical_consumption.csv", index=False)
    efficiency_df.to_csv(f"{output_dir}/efficiency_ratings.csv", index=False)
    retrofit_df.to_csv(f"{output_dir}/retrofit_data.csv", index=False)
    
    print(f"Saved historical consumption data: {len(historical_df)} records")
    print(f"Saved efficiency ratings data: {len(efficiency_df)} records")
    print(f"Saved retrofit data: {len(retrofit_df)} records")
    
    # Create summary
    summary = {
        'total_buildings': len(buildings),
        'historical_years': sorted(historical_df['year'].unique().tolist()),
        'eu_ratings_distribution': efficiency_df['eu_rating'].value_counts().to_dict(),
        'retrofit_coverage': retrofit_df['has_retrofit'].value_counts().to_dict(),
        'retrofit_types': retrofit_df[retrofit_df['has_retrofit']]['retrofit_type'].value_counts().to_dict()
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEnergy performance generation complete!")
    print(f"Generated data for {len(buildings)} buildings")
    print(f"EU ratings distribution: {summary['eu_ratings_distribution']}")
    print(f"Retrofit coverage: {summary['retrofit_coverage']}")

if __name__ == "__main__":
    main()