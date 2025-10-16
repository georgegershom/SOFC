#!/usr/bin/env python3
"""
Material & Technology Costs Data Generator
Generates comprehensive cost data for building retrofit materials and technologies
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class MaterialTechnologyCostsGenerator:
    def __init__(self, location="United States", currency="USD"):
        self.location = location
        self.currency = currency
        
        # Base material costs (2023 baseline)
        self.insulation_costs = {
            'fiberglass_batt': {'cost_per_sqft': 1.25, 'r_value_per_inch': 3.2, 'lifespan_years': 30},
            'cellulose_blown': {'cost_per_sqft': 1.45, 'r_value_per_inch': 3.6, 'lifespan_years': 25},
            'spray_foam_closed': {'cost_per_sqft': 3.85, 'r_value_per_inch': 6.5, 'lifespan_years': 50},
            'spray_foam_open': {'cost_per_sqft': 2.25, 'r_value_per_inch': 3.8, 'lifespan_years': 30},
            'rigid_foam_xps': {'cost_per_sqft': 2.15, 'r_value_per_inch': 5.0, 'lifespan_years': 40},
            'rigid_foam_polyiso': {'cost_per_sqft': 1.95, 'r_value_per_inch': 6.0, 'lifespan_years': 35},
            'mineral_wool': {'cost_per_sqft': 1.85, 'r_value_per_inch': 4.0, 'lifespan_years': 50}
        }
        
        self.window_costs = {
            'single_pane': {'cost_per_sqft': 8.50, 'u_factor': 1.25, 'shgc': 0.75, 'lifespan_years': 20},
            'double_pane_clear': {'cost_per_sqft': 15.25, 'u_factor': 0.48, 'shgc': 0.65, 'lifespan_years': 25},
            'double_pane_low_e': {'cost_per_sqft': 18.75, 'u_factor': 0.32, 'shgc': 0.45, 'lifespan_years': 30},
            'triple_pane_low_e': {'cost_per_sqft': 28.50, 'u_factor': 0.22, 'shgc': 0.35, 'lifespan_years': 35},
            'triple_pane_high_performance': {'cost_per_sqft': 42.75, 'u_factor': 0.15, 'shgc': 0.25, 'lifespan_years': 40}
        }
        
        self.hvac_costs = {
            'gas_furnace_standard': {'cost_per_btu': 0.045, 'efficiency_afue': 0.80, 'lifespan_years': 15},
            'gas_furnace_high_efficiency': {'cost_per_btu': 0.065, 'efficiency_afue': 0.95, 'lifespan_years': 20},
            'heat_pump_air_source': {'cost_per_btu': 0.125, 'efficiency_hspf': 8.5, 'efficiency_seer': 16, 'lifespan_years': 15},
            'heat_pump_ground_source': {'cost_per_btu': 0.285, 'efficiency_cop': 4.2, 'lifespan_years': 25},
            'heat_pump_mini_split': {'cost_per_btu': 0.165, 'efficiency_hspf': 10.0, 'efficiency_seer': 20, 'lifespan_years': 18},
            'boiler_gas_standard': {'cost_per_btu': 0.055, 'efficiency_afue': 0.82, 'lifespan_years': 20},
            'boiler_gas_condensing': {'cost_per_btu': 0.085, 'efficiency_afue': 0.95, 'lifespan_years': 25},
            'chiller_air_cooled': {'cost_per_ton': 1250, 'efficiency_eer': 10.5, 'lifespan_years': 20},
            'chiller_water_cooled': {'cost_per_ton': 1850, 'efficiency_eer': 12.5, 'lifespan_years': 25}
        }
        
        self.solar_costs = {
            'pv_residential_rooftop': {'cost_per_watt': 3.25, 'efficiency': 0.20, 'lifespan_years': 25, 'degradation_rate': 0.005},
            'pv_commercial_rooftop': {'cost_per_watt': 2.85, 'efficiency': 0.21, 'lifespan_years': 25, 'degradation_rate': 0.005},
            'pv_utility_scale': {'cost_per_watt': 1.95, 'efficiency': 0.22, 'lifespan_years': 25, 'degradation_rate': 0.005},
            'solar_thermal_residential': {'cost_per_sqft': 15.50, 'efficiency': 0.65, 'lifespan_years': 20},
            'solar_thermal_commercial': {'cost_per_sqft': 12.25, 'efficiency': 0.70, 'lifespan_years': 25}
        }
        
        self.energy_storage_costs = {
            'lithium_ion_residential': {'cost_per_kwh': 485, 'efficiency': 0.95, 'lifespan_years': 15, 'cycles': 6000},
            'lithium_ion_commercial': {'cost_per_kwh': 425, 'efficiency': 0.96, 'lifespan_years': 15, 'cycles': 8000},
            'flow_battery': {'cost_per_kwh': 325, 'efficiency': 0.85, 'lifespan_years': 20, 'cycles': 12000},
            'compressed_air': {'cost_per_kwh': 125, 'efficiency': 0.75, 'lifespan_years': 30, 'cycles': 20000}
        }
        
        self.smart_building_costs = {
            'building_automation_system': {'cost_per_sqft': 2.85, 'lifespan_years': 15},
            'smart_thermostats': {'cost_per_unit': 285, 'lifespan_years': 10},
            'occupancy_sensors': {'cost_per_unit': 125, 'lifespan_years': 12},
            'daylight_sensors': {'cost_per_unit': 185, 'lifespan_years': 15},
            'smart_lighting_controls': {'cost_per_fixture': 65, 'lifespan_years': 20},
            'energy_monitoring_system': {'cost_per_sqft': 0.85, 'lifespan_years': 10},
            'demand_response_system': {'cost_per_sqft': 1.25, 'lifespan_years': 12}
        }
    
    def generate_historical_cost_trends(self, start_year=2015, end_year=2023):
        """Generate historical cost trends for all material categories"""
        
        years = list(range(start_year, end_year + 1))
        historical_data = []
        
        # Define cost trend factors for different categories
        cost_trends = {
            'insulation': {'annual_change': 0.025, 'volatility': 0.08},  # 2.5% annual increase
            'windows': {'annual_change': 0.015, 'volatility': 0.06},    # 1.5% annual increase
            'hvac': {'annual_change': 0.03, 'volatility': 0.10},        # 3% annual increase
            'solar': {'annual_change': -0.08, 'volatility': 0.15},      # 8% annual decrease (learning curve)
            'storage': {'annual_change': -0.12, 'volatility': 0.20},    # 12% annual decrease
            'smart_building': {'annual_change': -0.05, 'volatility': 0.12}  # 5% annual decrease
        }
        
        for year in years:
            np.random.seed(year)
            years_from_base = year - 2023
            
            # Calculate cost multipliers for each category
            for category, trend_data in cost_trends.items():
                trend_multiplier = (1 + trend_data['annual_change']) ** years_from_base
                volatility_factor = np.random.normal(1, trend_data['volatility'])
                final_multiplier = trend_multiplier * volatility_factor
                
                historical_data.append({
                    'year': year,
                    'category': category,
                    'cost_multiplier': final_multiplier,
                    'trend_factor': trend_multiplier,
                    'volatility_factor': volatility_factor,
                    'location': self.location
                })
        
        return pd.DataFrame(historical_data)
    
    def generate_detailed_material_costs(self, year=2023, cost_multiplier=1.0):
        """Generate detailed material costs with specifications"""
        
        all_materials = []
        
        # Process insulation materials
        for material_type, specs in self.insulation_costs.items():
            all_materials.append({
                'year': year,
                'category': 'insulation',
                'material_type': material_type,
                'cost_per_unit': specs['cost_per_sqft'] * cost_multiplier,
                'unit': 'sqft',
                'performance_metric': 'r_value_per_inch',
                'performance_value': specs['r_value_per_inch'],
                'lifespan_years': specs['lifespan_years'],
                'location': self.location,
                'currency': self.currency
            })
        
        # Process window materials
        for window_type, specs in self.window_costs.items():
            all_materials.append({
                'year': year,
                'category': 'windows',
                'material_type': window_type,
                'cost_per_unit': specs['cost_per_sqft'] * cost_multiplier,
                'unit': 'sqft',
                'performance_metric': 'u_factor',
                'performance_value': specs['u_factor'],
                'secondary_metric': 'shgc',
                'secondary_value': specs['shgc'],
                'lifespan_years': specs['lifespan_years'],
                'location': self.location,
                'currency': self.currency
            })
        
        # Process HVAC systems
        for hvac_type, specs in self.hvac_costs.items():
            cost_key = 'cost_per_btu' if 'cost_per_btu' in specs else 'cost_per_ton'
            unit = 'btu' if 'cost_per_btu' in specs else 'ton'
            
            all_materials.append({
                'year': year,
                'category': 'hvac',
                'material_type': hvac_type,
                'cost_per_unit': specs[cost_key] * cost_multiplier,
                'unit': unit,
                'performance_metric': list(specs.keys())[1],  # First efficiency metric
                'performance_value': list(specs.values())[1],
                'lifespan_years': specs['lifespan_years'],
                'location': self.location,
                'currency': self.currency
            })
        
        # Process solar systems
        for solar_type, specs in self.solar_costs.items():
            unit = 'watt' if 'cost_per_watt' in specs else 'sqft'
            cost_key = 'cost_per_watt' if 'cost_per_watt' in specs else 'cost_per_sqft'
            
            all_materials.append({
                'year': year,
                'category': 'solar',
                'material_type': solar_type,
                'cost_per_unit': specs[cost_key] * cost_multiplier * 0.92,  # Solar cost decline
                'unit': unit,
                'performance_metric': 'efficiency',
                'performance_value': specs['efficiency'],
                'lifespan_years': specs['lifespan_years'],
                'degradation_rate': specs.get('degradation_rate', 0),
                'location': self.location,
                'currency': self.currency
            })
        
        # Process energy storage
        for storage_type, specs in self.energy_storage_costs.items():
            all_materials.append({
                'year': year,
                'category': 'energy_storage',
                'material_type': storage_type,
                'cost_per_unit': specs['cost_per_kwh'] * cost_multiplier * 0.88,  # Storage cost decline
                'unit': 'kwh',
                'performance_metric': 'efficiency',
                'performance_value': specs['efficiency'],
                'lifespan_years': specs['lifespan_years'],
                'cycle_life': specs['cycles'],
                'location': self.location,
                'currency': self.currency
            })
        
        # Process smart building technologies
        for smart_type, specs in self.smart_building_costs.items():
            unit_key = [k for k in specs.keys() if 'cost_per' in k][0]
            unit = unit_key.replace('cost_per_', '')
            
            all_materials.append({
                'year': year,
                'category': 'smart_building',
                'material_type': smart_type,
                'cost_per_unit': specs[unit_key] * cost_multiplier * 0.95,  # Smart tech cost decline
                'unit': unit,
                'lifespan_years': specs['lifespan_years'],
                'location': self.location,
                'currency': self.currency
            })
        
        return pd.DataFrame(all_materials)
    
    def generate_cost_forecasts(self, start_year=2024, end_year=2040):
        """Generate cost forecasts with multiple scenarios"""
        
        years = list(range(start_year, end_year + 1))
        forecast_data = []
        
        # Define forecast scenarios
        scenarios = {
            'conservative': {
                'insulation': 0.02, 'windows': 0.01, 'hvac': 0.025,
                'solar': -0.04, 'storage': -0.08, 'smart_building': -0.03
            },
            'base_case': {
                'insulation': 0.025, 'windows': 0.015, 'hvac': 0.03,
                'solar': -0.06, 'storage': -0.10, 'smart_building': -0.05
            },
            'aggressive': {
                'insulation': 0.03, 'windows': 0.02, 'hvac': 0.035,
                'solar': -0.08, 'storage': -0.12, 'smart_building': -0.07
            }
        }
        
        for scenario_name, scenario_rates in scenarios.items():
            for year in years:
                years_from_base = year - 2023
                
                for category, annual_rate in scenario_rates.items():
                    cost_multiplier = (1 + annual_rate) ** years_from_base
                    
                    forecast_data.append({
                        'year': year,
                        'scenario': scenario_name,
                        'category': category,
                        'cost_multiplier': cost_multiplier,
                        'annual_change_rate': annual_rate,
                        'location': self.location
                    })
        
        return pd.DataFrame(forecast_data)
    
    def generate_regional_cost_variations(self):
        """Generate regional cost variation factors"""
        
        regional_factors = {
            'Northeast': {
                'labor_multiplier': 1.25,
                'material_multiplier': 1.15,
                'permitting_costs': 1500,
                'description': 'High labor and material costs, complex permitting'
            },
            'Southeast': {
                'labor_multiplier': 0.85,
                'material_multiplier': 0.95,
                'permitting_costs': 800,
                'description': 'Lower labor costs, moderate material costs'
            },
            'Midwest': {
                'labor_multiplier': 0.90,
                'material_multiplier': 1.00,
                'permitting_costs': 600,
                'description': 'Moderate costs across categories'
            },
            'Southwest': {
                'labor_multiplier': 0.95,
                'material_multiplier': 1.05,
                'permitting_costs': 900,
                'description': 'Growing market, increasing costs'
            },
            'West_Coast': {
                'labor_multiplier': 1.35,
                'material_multiplier': 1.20,
                'permitting_costs': 2200,
                'description': 'Highest costs, strict regulations'
            }
        }
        
        return regional_factors
    
    def generate_bulk_pricing_tiers(self):
        """Generate bulk pricing discounts for large projects"""
        
        bulk_tiers = {
            'small_project': {
                'sqft_range': [0, 5000],
                'discount_factor': 1.00,
                'description': 'Standard retail pricing'
            },
            'medium_project': {
                'sqft_range': [5001, 25000],
                'discount_factor': 0.92,
                'description': '8% discount for medium projects'
            },
            'large_project': {
                'sqft_range': [25001, 100000],
                'discount_factor': 0.85,
                'description': '15% discount for large projects'
            },
            'mega_project': {
                'sqft_range': [100001, float('inf')],
                'discount_factor': 0.78,
                'description': '22% discount for mega projects'
            }
        }
        
        return bulk_tiers

def main():
    """Generate material and technology cost data"""
    
    print("Generating material and technology cost data...")
    
    generator = MaterialTechnologyCostsGenerator()
    
    # Generate all cost datasets
    historical_trends = generator.generate_historical_cost_trends()
    current_costs = generator.generate_detailed_material_costs()
    cost_forecasts = generator.generate_cost_forecasts()
    regional_variations = generator.generate_regional_cost_variations()
    bulk_pricing = generator.generate_bulk_pricing_tiers()
    
    # Create output directory
    output_dir = "economic_market/material_technology_costs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    historical_trends.to_csv(f"{output_dir}/historical_cost_trends_2015_2023.csv", index=False)
    current_costs.to_csv(f"{output_dir}/detailed_material_costs_2023.csv", index=False)
    cost_forecasts.to_csv(f"{output_dir}/cost_forecasts_2024_2040.csv", index=False)
    
    # Save JSON data
    with open(f"{output_dir}/regional_cost_variations.json", 'w') as f:
        json.dump(regional_variations, f, indent=2)
    
    with open(f"{output_dir}/bulk_pricing_tiers.json", 'w') as f:
        json.dump(bulk_pricing, f, indent=2)
    
    # Generate cost summary report
    summary_report = {
        'generation_date': datetime.now().isoformat(),
        'location': generator.location,
        'currency': generator.currency,
        'data_coverage': {
            'historical_years': '2015-2023',
            'forecast_years': '2024-2040',
            'material_categories': ['insulation', 'windows', 'hvac', 'solar', 'energy_storage', 'smart_building'],
            'total_materials': len(current_costs)
        },
        'key_insights': {
            'fastest_declining_costs': 'Energy storage and solar PV',
            'fastest_increasing_costs': 'HVAC systems and labor',
            'highest_regional_variation': 'West Coast vs Southeast',
            'bulk_discount_range': '8% to 22%'
        }
    }
    
    with open(f"{output_dir}/cost_data_summary.json", 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print("Material and technology cost data generation completed!")

if __name__ == "__main__":
    main()