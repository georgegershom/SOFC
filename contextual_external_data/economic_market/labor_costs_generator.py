#!/usr/bin/env python3
"""
Labor Costs Data Generator
Generates comprehensive labor cost data for building retrofit installations
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class LaborCostsGenerator:
    def __init__(self, location="United States", currency="USD"):
        self.location = location
        self.currency = currency
        
        # Base labor rates ($/hour) - 2023 baseline
        self.trade_rates = {
            'general_contractor': {'base_rate': 85, 'skill_level': 'management', 'union_premium': 1.15},
            'electrician': {'base_rate': 75, 'skill_level': 'skilled', 'union_premium': 1.25},
            'plumber': {'base_rate': 72, 'skill_level': 'skilled', 'union_premium': 1.22},
            'hvac_technician': {'base_rate': 68, 'skill_level': 'skilled', 'union_premium': 1.20},
            'insulation_installer': {'base_rate': 45, 'skill_level': 'semi_skilled', 'union_premium': 1.10},
            'window_installer': {'base_rate': 52, 'skill_level': 'skilled', 'union_premium': 1.15},
            'roofer': {'base_rate': 58, 'skill_level': 'skilled', 'union_premium': 1.18},
            'solar_installer': {'base_rate': 62, 'skill_level': 'skilled', 'union_premium': 1.12},
            'building_automation_tech': {'base_rate': 78, 'skill_level': 'highly_skilled', 'union_premium': 1.08},
            'energy_auditor': {'base_rate': 65, 'skill_level': 'skilled', 'union_premium': 1.05},
            'project_manager': {'base_rate': 95, 'skill_level': 'management', 'union_premium': 1.10},
            'laborer': {'base_rate': 35, 'skill_level': 'unskilled', 'union_premium': 1.25}
        }
        
        # Installation time estimates (hours per unit)
        self.installation_times = {
            'insulation': {
                'fiberglass_batt': {'hours_per_sqft': 0.015, 'crew_size': 2},
                'spray_foam': {'hours_per_sqft': 0.025, 'crew_size': 2},
                'blown_cellulose': {'hours_per_sqft': 0.012, 'crew_size': 2}
            },
            'windows': {
                'standard_replacement': {'hours_per_sqft': 0.45, 'crew_size': 2},
                'new_construction': {'hours_per_sqft': 0.35, 'crew_size': 2},
                'historic_renovation': {'hours_per_sqft': 0.65, 'crew_size': 3}
            },
            'hvac': {
                'furnace_replacement': {'hours_per_unit': 8, 'crew_size': 2},
                'heat_pump_installation': {'hours_per_unit': 12, 'crew_size': 2},
                'ductwork_per_foot': {'hours_per_foot': 0.25, 'crew_size': 2},
                'mini_split_installation': {'hours_per_unit': 6, 'crew_size': 2}
            },
            'solar': {
                'residential_rooftop': {'hours_per_kw': 8, 'crew_size': 3},
                'commercial_rooftop': {'hours_per_kw': 6, 'crew_size': 4},
                'ground_mount': {'hours_per_kw': 10, 'crew_size': 4}
            },
            'building_automation': {
                'thermostat_installation': {'hours_per_unit': 2, 'crew_size': 1},
                'sensor_installation': {'hours_per_unit': 1.5, 'crew_size': 1},
                'bas_programming': {'hours_per_point': 0.5, 'crew_size': 1}
            }
        }
        
        # Regional wage multipliers
        self.regional_multipliers = {
            'Northeast': 1.25,
            'Southeast': 0.85,
            'Midwest': 0.90,
            'Southwest': 0.95,
            'West_Coast': 1.35,
            'Mountain': 1.00,
            'Alaska': 1.45,
            'Hawaii': 1.30
        }
        
        # Benefit and overhead rates
        self.overhead_rates = {
            'workers_compensation': 0.08,
            'general_liability': 0.02,
            'unemployment_insurance': 0.03,
            'social_security': 0.062,
            'medicare': 0.0145,
            'health_insurance': 0.15,
            'retirement_contribution': 0.06,
            'paid_time_off': 0.12,
            'training_certification': 0.03,
            'equipment_depreciation': 0.05,
            'general_overhead': 0.15,
            'profit_margin': 0.12
        }
    
    def generate_trade_labor_rates(self, year=2023, region="National"):
        """Generate detailed labor rates by trade"""
        
        labor_data = []
        
        # Apply regional multiplier
        regional_mult = self.regional_multipliers.get(region, 1.0)
        
        for trade, trade_info in self.trade_rates.items():
            base_rate = trade_info['base_rate'] * regional_mult
            
            # Calculate union vs non-union rates
            union_rate = base_rate * trade_info['union_premium']
            non_union_rate = base_rate
            
            # Calculate fully burdened rates (with benefits and overhead)
            total_overhead_rate = sum(self.overhead_rates.values())
            
            union_burdened = union_rate * (1 + total_overhead_rate)
            non_union_burdened = non_union_rate * (1 + total_overhead_rate)
            
            labor_data.append({
                'year': year,
                'region': region,
                'trade': trade,
                'skill_level': trade_info['skill_level'],
                'base_rate_union': union_rate,
                'base_rate_non_union': non_union_rate,
                'fully_burdened_union': union_burdened,
                'fully_burdened_non_union': non_union_burdened,
                'union_premium_factor': trade_info['union_premium'],
                'overhead_rate': total_overhead_rate,
                'currency': self.currency
            })
        
        return pd.DataFrame(labor_data)
    
    def generate_installation_cost_estimates(self, year=2023, region="National"):
        """Generate installation cost estimates for retrofit measures"""
        
        installation_costs = []
        
        # Get labor rates for the region
        labor_rates_df = self.generate_trade_labor_rates(year, region)
        
        for category, installations in self.installation_times.items():
            for installation_type, specs in installations.items():
                
                # Determine primary trade for this installation
                primary_trade = self._get_primary_trade(category, installation_type)
                
                # Get labor rate for primary trade
                trade_rate = labor_rates_df[
                    labor_rates_df['trade'] == primary_trade
                ]['fully_burdened_non_union'].iloc[0]
                
                # Calculate cost per unit
                if 'hours_per_sqft' in specs:
                    labor_cost_per_unit = specs['hours_per_sqft'] * trade_rate * specs['crew_size']
                    unit = 'sqft'
                    time_per_unit = specs['hours_per_sqft']
                elif 'hours_per_unit' in specs:
                    labor_cost_per_unit = specs['hours_per_unit'] * trade_rate
                    unit = 'unit'
                    time_per_unit = specs['hours_per_unit']
                elif 'hours_per_foot' in specs:
                    labor_cost_per_unit = specs['hours_per_foot'] * trade_rate * specs['crew_size']
                    unit = 'foot'
                    time_per_unit = specs['hours_per_foot']
                elif 'hours_per_kw' in specs:
                    labor_cost_per_unit = specs['hours_per_kw'] * trade_rate * specs['crew_size']
                    unit = 'kw'
                    time_per_unit = specs['hours_per_kw']
                else:  # hours_per_point
                    labor_cost_per_unit = specs['hours_per_point'] * trade_rate
                    unit = 'point'
                    time_per_unit = specs['hours_per_point']
                
                installation_costs.append({
                    'year': year,
                    'region': region,
                    'category': category,
                    'installation_type': installation_type,
                    'primary_trade': primary_trade,
                    'labor_cost_per_unit': labor_cost_per_unit,
                    'unit': unit,
                    'time_per_unit': time_per_unit,
                    'crew_size': specs['crew_size'],
                    'hourly_rate_used': trade_rate,
                    'currency': self.currency
                })
        
        return pd.DataFrame(installation_costs)
    
    def _get_primary_trade(self, category, installation_type):
        """Determine the primary trade for an installation type"""
        
        trade_mapping = {
            'insulation': 'insulation_installer',
            'windows': 'window_installer',
            'hvac': 'hvac_technician',
            'solar': 'solar_installer',
            'building_automation': 'building_automation_tech'
        }
        
        return trade_mapping.get(category, 'general_contractor')
    
    def generate_project_labor_estimates(self, project_specs):
        """Generate complete labor estimates for a retrofit project"""
        
        project_labor = []
        total_cost = 0
        total_hours = 0
        
        for spec in project_specs:
            category = spec['category']
            installation_type = spec['installation_type']
            quantity = spec['quantity']
            region = spec.get('region', 'National')
            
            # Get installation costs
            installation_df = self.generate_installation_cost_estimates(region=region)
            
            matching_installation = installation_df[
                (installation_df['category'] == category) &
                (installation_df['installation_type'] == installation_type)
            ]
            
            if not matching_installation.empty:
                cost_per_unit = matching_installation['labor_cost_per_unit'].iloc[0]
                time_per_unit = matching_installation['time_per_unit'].iloc[0]
                crew_size = matching_installation['crew_size'].iloc[0]
                
                line_cost = cost_per_unit * quantity
                line_hours = time_per_unit * quantity * crew_size
                
                total_cost += line_cost
                total_hours += line_hours
                
                project_labor.append({
                    'category': category,
                    'installation_type': installation_type,
                    'quantity': quantity,
                    'unit': matching_installation['unit'].iloc[0],
                    'cost_per_unit': cost_per_unit,
                    'total_cost': line_cost,
                    'hours_per_unit': time_per_unit,
                    'total_hours': line_hours,
                    'crew_size': crew_size,
                    'region': region
                })
        
        return {
            'line_items': project_labor,
            'project_totals': {
                'total_labor_cost': total_cost,
                'total_labor_hours': total_hours,
                'average_hourly_rate': total_cost / total_hours if total_hours > 0 else 0
            }
        }
    
    def generate_seasonal_adjustments(self):
        """Generate seasonal labor cost adjustments"""
        
        seasonal_adjustments = {
            'winter': {
                'months': [12, 1, 2],
                'adjustment_factor': 1.15,
                'reasons': ['Weather delays', 'Heating costs', 'Holiday premiums']
            },
            'spring': {
                'months': [3, 4, 5],
                'adjustment_factor': 1.05,
                'reasons': ['High demand season', 'Weather improvements']
            },
            'summer': {
                'months': [6, 7, 8],
                'adjustment_factor': 1.10,
                'reasons': ['Peak construction season', 'Heat-related slowdowns']
            },
            'fall': {
                'months': [9, 10, 11],
                'adjustment_factor': 1.00,
                'reasons': ['Optimal working conditions', 'Standard rates']
            }
        }
        
        return seasonal_adjustments
    
    def generate_labor_shortage_impacts(self, year=2023):
        """Generate labor shortage impact factors by trade"""
        
        # Simulate labor shortage impacts (higher in skilled trades)
        shortage_impacts = {}
        
        for trade, trade_info in self.trade_rates.items():
            if trade_info['skill_level'] == 'highly_skilled':
                shortage_multiplier = 1.25
            elif trade_info['skill_level'] == 'skilled':
                shortage_multiplier = 1.15
            elif trade_info['skill_level'] == 'semi_skilled':
                shortage_multiplier = 1.08
            else:  # unskilled
                shortage_multiplier = 1.02
            
            shortage_impacts[trade] = {
                'shortage_multiplier': shortage_multiplier,
                'skill_level': trade_info['skill_level'],
                'availability_rating': self._get_availability_rating(shortage_multiplier),
                'year': year
            }
        
        return shortage_impacts
    
    def _get_availability_rating(self, multiplier):
        """Convert shortage multiplier to availability rating"""
        if multiplier >= 1.20:
            return 'Critical Shortage'
        elif multiplier >= 1.10:
            return 'Moderate Shortage'
        elif multiplier >= 1.05:
            return 'Slight Shortage'
        else:
            return 'Adequate Supply'

def main():
    """Generate labor cost data for multiple regions"""
    
    regions = ['National', 'Northeast', 'Southeast', 'Midwest', 'Southwest', 'West_Coast']
    
    for region in regions:
        print(f"Generating labor cost data for {region}...")
        
        generator = LaborCostsGenerator()
        
        # Generate labor datasets
        trade_rates = generator.generate_trade_labor_rates(region=region)
        installation_costs = generator.generate_installation_cost_estimates(region=region)
        seasonal_adjustments = generator.generate_seasonal_adjustments()
        shortage_impacts = generator.generate_labor_shortage_impacts()
        
        # Create region directory
        region_dir = f"economic_market/labor_costs_{region.lower()}"
        os.makedirs(region_dir, exist_ok=True)
        
        # Save datasets
        trade_rates.to_csv(f"{region_dir}/trade_labor_rates_2023.csv", index=False)
        installation_costs.to_csv(f"{region_dir}/installation_cost_estimates_2023.csv", index=False)
        
        # Save JSON data
        with open(f"{region_dir}/seasonal_adjustments.json", 'w') as f:
            json.dump(seasonal_adjustments, f, indent=2)
        
        with open(f"{region_dir}/labor_shortage_impacts.json", 'w') as f:
            json.dump(shortage_impacts, f, indent=2)
        
        print(f"Saved labor cost data for {region}")
    
    # Generate sample project estimate
    sample_project = [
        {'category': 'insulation', 'installation_type': 'spray_foam', 'quantity': 2500, 'region': 'Northeast'},
        {'category': 'windows', 'installation_type': 'standard_replacement', 'quantity': 150, 'region': 'Northeast'},
        {'category': 'hvac', 'installation_type': 'heat_pump_installation', 'quantity': 2, 'region': 'Northeast'},
        {'category': 'solar', 'installation_type': 'residential_rooftop', 'quantity': 10, 'region': 'Northeast'}
    ]
    
    generator = LaborCostsGenerator()
    project_estimate = generator.generate_project_labor_estimates(sample_project)
    
    # Save sample project estimate
    with open("economic_market/sample_project_labor_estimate.json", 'w') as f:
        json.dump(project_estimate, f, indent=2, default=str)
    
    print("Labor cost data generation completed!")

if __name__ == "__main__":
    main()