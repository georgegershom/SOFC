"""
Energy Performance Data Generator
Generates historical energy consumption and efficiency data including:
- Historical energy consumption patterns
- Energy efficiency ratings over time
- Post-retrofit energy savings data
- Benchmarking metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class EnergyPerformanceGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_historical_consumption(self, building_id: str, 
                                      construction_year: int,
                                      floor_area: float,
                                      building_type: str,
                                      n_years: int = 10) -> pd.DataFrame:
        """Generate historical annual energy consumption data"""
        
        # Energy use intensity benchmarks (kWh/m²/year)
        eui_benchmarks = {
            'office': (150, 350),
            'residential': (100, 250),
            'retail': (200, 500),
            'educational': (120, 280),
            'healthcare': (300, 600),
            'industrial': (150, 400),
            'mixed_use': (150, 350)
        }
        
        eui_range = eui_benchmarks.get(building_type, (150, 350))
        base_eui = random.uniform(*eui_range)
        
        # Generate yearly data
        historical_data = []
        current_year = 2024
        
        for year_offset in range(n_years):
            year = current_year - year_offset
            
            # Add trend (slight increase in older buildings, decrease in newer ones)
            if construction_year < 2000:
                trend_factor = 1 + (year_offset * 0.01)  # Degradation
            else:
                trend_factor = 1 - (year_offset * 0.005)  # Improvements
            
            # Annual variation
            annual_variation = random.uniform(0.9, 1.1)
            
            annual_eui = base_eui * trend_factor * annual_variation
            annual_consumption = annual_eui * floor_area
            
            # Monthly breakdown with seasonal patterns
            monthly_data = self._generate_monthly_consumption(
                year, annual_consumption, building_type
            )
            
            for month_data in monthly_data:
                historical_data.append({
                    'building_id': building_id,
                    'year': year,
                    'month': month_data['month'],
                    'electricity_kwh': month_data['electricity'],
                    'gas_kwh': month_data['gas'],
                    'total_kwh': month_data['total'],
                    'electricity_cost_eur': month_data['electricity'] * random.uniform(0.15, 0.25),
                    'gas_cost_eur': month_data['gas'] * random.uniform(0.05, 0.10),
                    'eui_kwh_m2': month_data['total'] / floor_area,
                    'carbon_emissions_kg_co2': month_data['total'] * random.uniform(0.2, 0.4)
                })
        
        return pd.DataFrame(historical_data)
    
    def _generate_monthly_consumption(self, year: int, annual_total: float, 
                                     building_type: str) -> List[Dict]:
        """Generate monthly consumption breakdown"""
        
        # Seasonal patterns (relative monthly consumption)
        seasonal_patterns = {
            'office': [1.1, 1.0, 0.95, 0.85, 0.8, 0.85, 0.95, 0.95, 0.85, 0.9, 1.0, 1.1],
            'residential': [1.2, 1.15, 1.0, 0.85, 0.7, 0.65, 0.7, 0.7, 0.75, 0.9, 1.05, 1.15],
            'retail': [1.15, 1.1, 0.95, 0.85, 0.8, 0.85, 0.9, 0.9, 0.85, 0.9, 1.1, 1.25],
            'educational': [1.1, 1.0, 0.95, 0.9, 0.85, 0.6, 0.4, 0.4, 0.9, 1.0, 1.05, 1.1],
            'healthcare': [1.05, 1.0, 0.95, 0.95, 0.95, 1.0, 1.05, 1.05, 1.0, 1.0, 1.0, 1.05],
            'industrial': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'mixed_use': [1.1, 1.05, 0.95, 0.85, 0.8, 0.8, 0.85, 0.85, 0.85, 0.95, 1.05, 1.1]
        }
        
        pattern = seasonal_patterns.get(building_type, seasonal_patterns['office'])
        
        # Normalize pattern
        pattern = np.array(pattern)
        pattern = pattern / pattern.sum() * 12
        
        monthly_data = []
        for month, factor in enumerate(pattern, 1):
            monthly_total = annual_total * factor / 12
            
            # Split between electricity and gas (varies by season)
            if building_type == 'residential':
                # More gas in winter for heating
                gas_ratio = 0.6 if month in [1, 2, 11, 12] else 0.3
            else:
                gas_ratio = 0.4 if month in [1, 2, 11, 12] else 0.2
            
            electricity = monthly_total * (1 - gas_ratio) * random.uniform(0.9, 1.1)
            gas = monthly_total * gas_ratio * random.uniform(0.9, 1.1)
            
            monthly_data.append({
                'month': month,
                'electricity': electricity,
                'gas': gas,
                'total': electricity + gas
            })
        
        return monthly_data
    
    def generate_efficiency_ratings(self, building_id: str, 
                                   construction_year: int,
                                   current_rating: str,
                                   n_assessments: int = 5) -> pd.DataFrame:
        """Generate historical energy efficiency ratings"""
        
        rating_scale = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        current_index = rating_scale.index(current_rating)
        
        assessments = []
        current_year = 2024
        
        for i in range(n_assessments):
            assessment_year = current_year - (i * 2)  # Assessments every 2 years
            
            # Rating might have been worse in the past
            if i > 0:
                historical_index = min(6, current_index + random.randint(0, i // 2))
            else:
                historical_index = current_index
            
            historical_rating = rating_scale[historical_index]
            
            # Generate detailed metrics
            assessments.append({
                'building_id': building_id,
                'assessment_date': f"{assessment_year}-01-15",
                'energy_rating': historical_rating,
                'primary_energy_kwh_m2_yr': self._rating_to_primary_energy(historical_rating),
                'co2_emissions_kg_m2_yr': self._rating_to_co2(historical_rating),
                'renewable_energy_pct': random.uniform(0, 30) if assessment_year > 2010 else 0,
                'assessor_id': f"ASR_{random.randint(1000, 9999)}",
                'certificate_number': f"EPC{assessment_year}{random.randint(100000, 999999)}",
                'valid_until': f"{assessment_year + 10}-01-15",
                'recommendations': self._generate_recommendations(historical_rating)
            })
        
        return pd.DataFrame(assessments)
    
    def _rating_to_primary_energy(self, rating: str) -> float:
        """Convert rating to primary energy consumption"""
        energy_ranges = {
            'A': (0, 50),
            'B': (50, 90),
            'C': (90, 150),
            'D': (150, 230),
            'E': (230, 330),
            'F': (330, 450),
            'G': (450, 800)
        }
        return round(random.uniform(*energy_ranges.get(rating, (200, 400))), 1)
    
    def _rating_to_co2(self, rating: str) -> float:
        """Convert rating to CO2 emissions"""
        co2_ranges = {
            'A': (0, 10),
            'B': (10, 20),
            'C': (20, 35),
            'D': (35, 55),
            'E': (55, 80),
            'F': (80, 110),
            'G': (110, 200)
        }
        return round(random.uniform(*co2_ranges.get(rating, (40, 80))), 1)
    
    def _generate_recommendations(self, rating: str) -> str:
        """Generate efficiency improvement recommendations"""
        recommendations = {
            'A': ['Install solar panels', 'Smart building controls'],
            'B': ['Upgrade to heat pump', 'Install solar panels', 'Smart thermostats'],
            'C': ['Improve insulation', 'Upgrade windows', 'LED lighting'],
            'D': ['Wall insulation', 'Double glazing', 'Efficient boiler', 'LED lighting'],
            'E': ['Cavity wall insulation', 'Loft insulation', 'New boiler', 'Double glazing'],
            'F': ['Full insulation upgrade', 'Window replacement', 'Heating system upgrade'],
            'G': ['Complete thermal envelope retrofit', 'Full HVAC replacement', 'Insulation throughout']
        }
        
        return '; '.join(random.sample(recommendations.get(rating, ['General improvements']), 
                                       min(3, len(recommendations.get(rating, [])))))
    
    def generate_retrofit_savings(self, building_id: str,
                                 pre_retrofit_consumption: float,
                                 retrofit_measures: List[str],
                                 retrofit_date: str = '2023-01-01') -> pd.DataFrame:
        """Generate post-retrofit energy savings data"""
        
        # Savings potential by measure
        savings_potential = {
            'wall_insulation': (0.15, 0.25),
            'roof_insulation': (0.10, 0.20),
            'window_upgrade': (0.08, 0.15),
            'hvac_upgrade': (0.20, 0.35),
            'lighting_led': (0.05, 0.10),
            'solar_panels': (0.15, 0.30),
            'heat_pump': (0.25, 0.40),
            'building_controls': (0.10, 0.20),
            'air_sealing': (0.05, 0.12)
        }
        
        # Calculate total savings
        total_savings_pct = 0
        for measure in retrofit_measures:
            if measure in savings_potential:
                total_savings_pct += random.uniform(*savings_potential[measure])
        
        total_savings_pct = min(0.6, total_savings_pct)  # Cap at 60% savings
        
        # Generate monthly post-retrofit data
        post_retrofit_data = []
        retrofit_dt = pd.to_datetime(retrofit_date)
        
        for month_offset in range(24):  # 2 years of post-retrofit data
            current_date = retrofit_dt + pd.DateOffset(months=month_offset)
            
            # Add some seasonal variation and random noise
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * current_date.month / 12)
            monthly_consumption = pre_retrofit_consumption / 12 * (1 - total_savings_pct) * seasonal_factor
            monthly_consumption *= random.uniform(0.9, 1.1)
            
            monthly_savings = pre_retrofit_consumption / 12 - monthly_consumption
            
            post_retrofit_data.append({
                'building_id': building_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'post_retrofit_consumption_kwh': round(monthly_consumption, 2),
                'energy_savings_kwh': round(monthly_savings, 2),
                'savings_percentage': round(total_savings_pct * 100, 1),
                'cost_savings_eur': round(monthly_savings * random.uniform(0.15, 0.20), 2),
                'carbon_savings_kg_co2': round(monthly_savings * random.uniform(0.2, 0.4), 2),
                'retrofit_measures': ', '.join(retrofit_measures),
                'measurement_verified': random.choice([True, False]),
                'normalized_for_weather': True
            })
        
        return pd.DataFrame(post_retrofit_data)
    
    def generate_benchmarking_data(self, building_id: str, building_type: str,
                                  floor_area: float, energy_rating: str) -> Dict:
        """Generate benchmarking metrics against similar buildings"""
        
        # Generate peer group statistics
        peer_group_size = random.randint(50, 500)
        
        # Your building's performance
        your_eui = self._rating_to_primary_energy(energy_rating)
        
        # Peer group distribution
        peer_euis = np.random.normal(loc=200, scale=50, size=peer_group_size)
        peer_euis = np.clip(peer_euis, 50, 500)
        
        percentile = (peer_euis < your_eui).sum() / peer_group_size * 100
        
        return {
            'building_id': building_id,
            'peer_group': f"{building_type}_{int(floor_area/1000)}k_m2",
            'peer_group_size': peer_group_size,
            'your_eui_kwh_m2': your_eui,
            'peer_median_eui': round(np.median(peer_euis), 1),
            'peer_25th_percentile_eui': round(np.percentile(peer_euis, 25), 1),
            'peer_75th_percentile_eui': round(np.percentile(peer_euis, 75), 1),
            'your_percentile': round(percentile, 1),
            'performance_class': 'excellent' if percentile < 25 else 'good' if percentile < 50 else 'average' if percentile < 75 else 'poor',
            'potential_savings_pct': max(0, round((your_eui - np.percentile(peer_euis, 25)) / your_eui * 100, 1)),
            'benchmark_date': '2024-01-01'
        }