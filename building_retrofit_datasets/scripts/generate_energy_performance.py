"""
Energy Performance Dataset Generator for Building Retrofit Research
Generates historical energy consumption, efficiency ratings, and retrofit scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

np.random.seed(42)


class EnergyPerformanceGenerator:
    """Generate comprehensive energy performance and retrofit scenario data."""
    
    def __init__(self, num_buildings=50):
        self.num_buildings = num_buildings
        
        # Retrofit measures database
        self.retrofit_measures = {
            'wall_insulation': {
                'u_value_improvement': 0.7,  # reduction factor
                'cost_per_m2': 85,
                'lifetime_years': 40,
                'carbon_saving_percent': 25,
                'energy_saving_percent': 22
            },
            'roof_insulation': {
                'u_value_improvement': 0.65,
                'cost_per_m2': 75,
                'lifetime_years': 40,
                'carbon_saving_percent': 15,
                'energy_saving_percent': 18
            },
            'window_replacement': {
                'u_value_improvement': 0.5,
                'cost_per_m2': 450,
                'lifetime_years': 30,
                'carbon_saving_percent': 18,
                'energy_saving_percent': 15
            },
            'hvac_upgrade_heat_pump': {
                'u_value_improvement': 1.0,
                'cost_fixed': 12000,
                'lifetime_years': 15,
                'carbon_saving_percent': 45,
                'energy_saving_percent': 35
            },
            'led_lighting': {
                'u_value_improvement': 1.0,
                'cost_per_m2': 25,
                'lifetime_years': 15,
                'carbon_saving_percent': 8,
                'energy_saving_percent': 12
            },
            'solar_pv': {
                'u_value_improvement': 1.0,
                'cost_per_kwp': 1200,
                'lifetime_years': 25,
                'carbon_saving_percent': 30,
                'energy_saving_percent': 25
            },
            'smart_controls': {
                'u_value_improvement': 1.0,
                'cost_fixed': 3500,
                'lifetime_years': 10,
                'carbon_saving_percent': 12,
                'energy_saving_percent': 15
            },
            'ventilation_heat_recovery': {
                'u_value_improvement': 1.0,
                'cost_fixed': 8500,
                'lifetime_years': 20,
                'carbon_saving_percent': 10,
                'energy_saving_percent': 12
            }
        }
    
    def generate_historical_consumption(self, building_attrs):
        """Generate 5 years of historical annual energy consumption."""
        base_year = 2019
        years = 5
        
        # Base consumption estimation
        floor_area = building_attrs['total_floor_area_m2']
        building_type = building_attrs['building_type']
        epc_rating = building_attrs['epc_rating']
        
        # Typical consumption per m² by building type (kWh/m²/year)
        base_consumption_per_m2 = {
            'residential': 120,
            'commercial': 180,
            'industrial': 250,
            'educational': 140
        }.get(building_type, 150)
        
        # EPC rating multiplier
        epc_multipliers = {'A': 0.6, 'B': 0.75, 'C': 0.9, 'D': 1.0, 'E': 1.15, 'F': 1.3, 'G': 1.5}
        epc_factor = epc_multipliers.get(epc_rating, 1.0)
        
        annual_consumption = []
        for year in range(years):
            # Base consumption with slight degradation over time
            degradation = 1 + (year * 0.015)  # 1.5% degradation per year
            
            # Weather variation
            weather_factor = np.random.uniform(0.95, 1.05)
            
            # Calculate consumption
            consumption = floor_area * base_consumption_per_m2 * epc_factor * degradation * weather_factor
            
            # Breakdown by end-use
            heating_cooling_pct = np.random.uniform(0.45, 0.55)
            lighting_pct = np.random.uniform(0.15, 0.25)
            equipment_pct = np.random.uniform(0.15, 0.25)
            other_pct = 1.0 - heating_cooling_pct - lighting_pct - equipment_pct
            
            annual_consumption.append({
                'year': base_year + year,
                'total_consumption_kwh': round(consumption, 2),
                'heating_cooling_kwh': round(consumption * heating_cooling_pct, 2),
                'lighting_kwh': round(consumption * lighting_pct, 2),
                'equipment_kwh': round(consumption * equipment_pct, 2),
                'other_kwh': round(consumption * other_pct, 2),
                'cost_eur': round(consumption * 0.15, 2),  # Assuming 0.15 EUR/kWh
                'carbon_emissions_kgco2': round(consumption * 0.35, 2)  # 0.35 kg CO2/kWh
            })
        
        return annual_consumption
    
    def generate_monthly_consumption(self, annual_total, building_type):
        """Generate monthly consumption breakdown with seasonal variation."""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Seasonal patterns (heating dominated climate)
        if building_type in ['residential', 'educational']:
            seasonal_factors = [1.4, 1.3, 1.2, 1.0, 0.8, 0.7, 0.7, 0.7, 0.8, 1.0, 1.2, 1.4]
        else:  # Commercial/Industrial with more stable consumption
            seasonal_factors = [1.2, 1.15, 1.1, 0.95, 0.85, 0.85, 0.85, 0.85, 0.9, 1.0, 1.1, 1.2]
        
        # Normalize factors to sum to 12 (average = 1.0)
        factor_sum = sum(seasonal_factors)
        seasonal_factors = [f * 12 / factor_sum for f in seasonal_factors]
        
        monthly_data = []
        for month, factor in zip(months, seasonal_factors):
            monthly_consumption = (annual_total / 12) * factor
            monthly_data.append({
                'month': month,
                'consumption_kwh': round(monthly_consumption, 2)
            })
        
        return monthly_data
    
    def identify_retrofit_potential(self, building_attrs):
        """Identify applicable retrofit measures and calculate potential savings."""
        epc_rating = building_attrs['epc_rating']
        construction_year = building_attrs['construction_year']
        building_type = building_attrs['building_type']
        floor_area = building_attrs['total_floor_area_m2']
        wall_area = building_attrs['wall_area_m2']
        roof_area = building_attrs['rooftop_area_m2']
        window_area = building_attrs['window_area_m2']
        wall_u_value = building_attrs['wall_u_value']
        roof_u_value = building_attrs['roof_u_value']
        window_u_value = building_attrs['window_u_value']
        
        applicable_measures = []
        
        # Wall insulation (if U-value > 0.5)
        if wall_u_value > 0.5:
            applicable_measures.append('wall_insulation')
        
        # Roof insulation (if U-value > 0.3)
        if roof_u_value > 0.3:
            applicable_measures.append('roof_insulation')
        
        # Window replacement (if U-value > 2.5)
        if window_u_value > 2.5:
            applicable_measures.append('window_replacement')
        
        # HVAC upgrade (if building older than 15 years and not already heat pump)
        if construction_year < 2009 and building_attrs.get('hvac_system', '') != 'air_source_heat_pump':
            applicable_measures.append('hvac_upgrade_heat_pump')
        
        # LED lighting (always applicable for older buildings)
        if construction_year < 2015:
            applicable_measures.append('led_lighting')
        
        # Solar PV (if suitable roof area available)
        if roof_area > 100:
            applicable_measures.append('solar_pv')
        
        # Smart controls
        applicable_measures.append('smart_controls')
        
        # Ventilation heat recovery (for larger buildings)
        if floor_area > 1000:
            applicable_measures.append('ventilation_heat_recovery')
        
        return applicable_measures
    
    def calculate_retrofit_scenario(self, building_attrs, measures, baseline_consumption):
        """Calculate costs, savings, and ROI for a retrofit scenario."""
        floor_area = building_attrs['total_floor_area_m2']
        wall_area = building_attrs['wall_area_m2']
        roof_area = building_attrs['rooftop_area_m2']
        window_area = building_attrs['window_area_m2']
        
        total_cost = 0
        total_energy_saving_pct = 0
        total_carbon_saving_pct = 0
        measure_details = []
        
        for measure in measures:
            measure_data = self.retrofit_measures[measure]
            
            # Calculate cost
            if 'cost_per_m2' in measure_data:
                if measure == 'wall_insulation':
                    cost = wall_area * measure_data['cost_per_m2']
                elif measure == 'roof_insulation':
                    cost = roof_area * measure_data['cost_per_m2']
                elif measure == 'window_replacement':
                    cost = window_area * measure_data['cost_per_m2']
                elif measure == 'led_lighting':
                    cost = floor_area * measure_data['cost_per_m2']
                else:
                    cost = floor_area * measure_data['cost_per_m2']
            elif 'cost_per_kwp' in measure_data:
                # Solar PV sizing (10% of roof area, ~150W/m²)
                pv_capacity = (roof_area * 0.1) * 0.15
                cost = pv_capacity * measure_data['cost_per_kwp']
            else:
                cost = measure_data['cost_fixed']
            
            total_cost += cost
            
            # Savings are not simply additive - use diminishing returns
            energy_saving = measure_data['energy_saving_percent'] * (1 - total_energy_saving_pct / 100) * 0.8
            carbon_saving = measure_data['carbon_saving_percent'] * (1 - total_carbon_saving_pct / 100) * 0.8
            
            total_energy_saving_pct += energy_saving
            total_carbon_saving_pct += carbon_saving
            
            measure_details.append({
                'measure': measure,
                'cost_eur': round(cost, 2),
                'energy_saving_pct': round(energy_saving, 2),
                'carbon_saving_pct': round(carbon_saving, 2),
                'lifetime_years': measure_data['lifetime_years']
            })
        
        # Cap total savings at realistic levels
        total_energy_saving_pct = min(total_energy_saving_pct, 75)
        total_carbon_saving_pct = min(total_carbon_saving_pct, 80)
        
        # Calculate annual savings
        annual_energy_saving_kwh = baseline_consumption * (total_energy_saving_pct / 100)
        annual_cost_saving_eur = annual_energy_saving_kwh * 0.15
        annual_carbon_saving_kgco2 = baseline_consumption * (total_carbon_saving_pct / 100) * 0.35
        
        # Calculate payback period and ROI
        simple_payback_years = total_cost / annual_cost_saving_eur if annual_cost_saving_eur > 0 else 999
        
        # NPV calculation (20 year horizon, 3% discount rate)
        years = 20
        discount_rate = 0.03
        npv = -total_cost
        for year in range(1, years + 1):
            npv += annual_cost_saving_eur / ((1 + discount_rate) ** year)
        
        roi = (npv / total_cost * 100) if total_cost > 0 else 0
        
        # Estimate post-retrofit EPC rating
        current_epc = building_attrs['epc_rating']
        epc_ratings = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
        current_index = epc_ratings.index(current_epc)
        
        # Improve rating based on energy savings
        if total_energy_saving_pct > 50:
            improvement = 3
        elif total_energy_saving_pct > 35:
            improvement = 2
        elif total_energy_saving_pct > 20:
            improvement = 1
        else:
            improvement = 0
        
        new_index = min(current_index + improvement, len(epc_ratings) - 1)
        post_retrofit_epc = epc_ratings[new_index]
        
        return {
            'measures': measure_details,
            'total_cost_eur': round(total_cost, 2),
            'annual_energy_saving_kwh': round(annual_energy_saving_kwh, 2),
            'annual_cost_saving_eur': round(annual_cost_saving_eur, 2),
            'annual_carbon_saving_kgco2': round(annual_carbon_saving_kgco2, 2),
            'total_energy_saving_pct': round(total_energy_saving_pct, 2),
            'total_carbon_saving_pct': round(total_carbon_saving_pct, 2),
            'simple_payback_years': round(simple_payback_years, 2),
            'npv_20years_eur': round(npv, 2),
            'roi_percent': round(roi, 2),
            'pre_retrofit_epc': current_epc,
            'post_retrofit_epc': post_retrofit_epc
        }
    
    def generate_dataset(self, building_attributes_df):
        """Generate complete energy performance dataset."""
        print(f"Generating energy performance data for {len(building_attributes_df)} buildings...")
        
        all_historical = []
        all_retrofit_scenarios = []
        
        for idx, building in building_attributes_df.iterrows():
            building_id = building['building_id']
            print(f"Processing {building_id}...")
            
            # Historical consumption
            historical = self.generate_historical_consumption(building)
            for record in historical:
                record['building_id'] = building_id
                all_historical.append(record)
            
            # Get latest year consumption as baseline
            baseline_consumption = historical[-1]['total_consumption_kwh']
            
            # Identify retrofit potential
            applicable_measures = self.identify_retrofit_potential(building)
            
            # Generate multiple retrofit scenarios
            scenarios = [
                {
                    'scenario_name': 'Minimal',
                    'measures': applicable_measures[:2] if len(applicable_measures) >= 2 else applicable_measures
                },
                {
                    'scenario_name': 'Standard',
                    'measures': applicable_measures[:4] if len(applicable_measures) >= 4 else applicable_measures
                },
                {
                    'scenario_name': 'Deep Retrofit',
                    'measures': applicable_measures
                }
            ]
            
            for scenario in scenarios:
                if scenario['measures']:
                    retrofit_calc = self.calculate_retrofit_scenario(
                        building, 
                        scenario['measures'], 
                        baseline_consumption
                    )
                    
                    all_retrofit_scenarios.append({
                        'building_id': building_id,
                        'scenario_name': scenario['scenario_name'],
                        'baseline_consumption_kwh': baseline_consumption,
                        'num_measures': len(scenario['measures']),
                        'measure_list': ', '.join(scenario['measures']),
                        **retrofit_calc
                    })
        
        historical_df = pd.DataFrame(all_historical)
        retrofit_df = pd.DataFrame(all_retrofit_scenarios)
        
        return historical_df, retrofit_df


def main():
    """Main function to generate and save energy performance data."""
    print("=" * 80)
    print("ENERGY PERFORMANCE DATA GENERATOR FOR RETROFIT RESEARCH")
    print("=" * 80)
    
    # Load building attributes
    print("\nLoading building attributes...")
    building_attrs = pd.read_csv('../data/building_attributes.csv')
    
    # Generate data
    generator = EnergyPerformanceGenerator(num_buildings=len(building_attrs))
    historical_df, retrofit_df = generator.generate_dataset(building_attrs)
    
    # Save datasets
    print("\nSaving datasets...")
    historical_df.to_csv('../data/energy_performance_historical.csv', index=False)
    retrofit_df.to_csv('../data/retrofit_scenarios.csv', index=False)
    retrofit_df.to_excel('../data/retrofit_scenarios.xlsx', index=False)
    
    # Save retrofit measures database
    with open('../data/retrofit_measures_database.json', 'w') as f:
        json.dump(generator.retrofit_measures, f, indent=2)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Historical records: {len(historical_df):,}")
    print(f"Retrofit scenarios: {len(retrofit_df):,}")
    print(f"Buildings analyzed: {building_attrs['building_id'].nunique()}")
    
    print("\n" + "-" * 80)
    print("HISTORICAL CONSUMPTION STATISTICS:")
    print("-" * 80)
    print(historical_df.groupby('year')['total_consumption_kwh'].agg(['mean', 'min', 'max']))
    
    print("\n" + "-" * 80)
    print("RETROFIT SCENARIO STATISTICS:")
    print("-" * 80)
    print(retrofit_df.groupby('scenario_name')[['total_cost_eur', 'annual_energy_saving_kwh', 
                                                  'simple_payback_years', 'roi_percent']].mean())
    
    print("\n" + "-" * 80)
    print("ENERGY SAVINGS POTENTIAL:")
    print("-" * 80)
    print(f"Average energy savings (Deep Retrofit): {retrofit_df[retrofit_df['scenario_name']=='Deep Retrofit']['total_energy_saving_pct'].mean():.1f}%")
    print(f"Average carbon savings (Deep Retrofit): {retrofit_df[retrofit_df['scenario_name']=='Deep Retrofit']['total_carbon_saving_pct'].mean():.1f}%")
    print(f"Average payback period: {retrofit_df['simple_payback_years'].mean():.1f} years")
    
    print("\n✅ Energy performance data generation complete!")
    print(f"📁 Historical data: ../data/energy_performance_historical.csv")
    print(f"📁 Retrofit scenarios: ../data/retrofit_scenarios.csv")
    print(f"📁 Retrofit scenarios: ../data/retrofit_scenarios.xlsx")
    print(f"📁 Retrofit measures DB: ../data/retrofit_measures_database.json")


if __name__ == "__main__":
    main()
