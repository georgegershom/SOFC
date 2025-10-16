#!/usr/bin/env python3
"""
Carbon Intensity Factors Data Generator
Generates grid electricity carbon intensity data including hourly marginal emissions
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class CarbonIntensityGenerator:
    def __init__(self):
        # Regional grid carbon intensities (kg CO2e/kWh) - 2023 baseline
        self.regional_intensities = {
            'NERC_Regions': {
                'NPCC': {'name': 'Northeast Power Coordinating Council', 'intensity': 0.285},
                'RFC': {'name': 'ReliabilityFirst Corporation', 'intensity': 0.445},
                'SERC': {'name': 'SERC Reliability Corporation', 'intensity': 0.485},
                'MRO': {'name': 'Midwest Reliability Organization', 'intensity': 0.525},
                'TRE': {'name': 'Texas Regional Entity', 'intensity': 0.395},
                'WECC': {'name': 'Western Electricity Coordinating Council', 'intensity': 0.325},
                'FRCC': {'name': 'Florida Reliability Coordinating Council', 'intensity': 0.435},
                'SPP': {'name': 'Southwest Power Pool', 'intensity': 0.515}
            },
            'ISO_RTO_Regions': {
                'CAISO': {'name': 'California ISO', 'intensity': 0.245, 'renewable_penetration': 0.52},
                'ERCOT': {'name': 'Electric Reliability Council of Texas', 'intensity': 0.395, 'renewable_penetration': 0.31},
                'PJM': {'name': 'PJM Interconnection', 'intensity': 0.415, 'renewable_penetration': 0.18},
                'NYISO': {'name': 'New York ISO', 'intensity': 0.295, 'renewable_penetration': 0.28},
                'ISO_NE': {'name': 'ISO New England', 'intensity': 0.315, 'renewable_penetration': 0.22},
                'MISO': {'name': 'Midcontinent ISO', 'intensity': 0.485, 'renewable_penetration': 0.25},
                'SPP': {'name': 'Southwest Power Pool', 'intensity': 0.515, 'renewable_penetration': 0.35}
            }
        }
        
        # Fuel mix data for different regions
        self.fuel_mixes = {
            'CAISO': {
                'natural_gas': 0.42, 'renewables': 0.52, 'nuclear': 0.06, 'coal': 0.00, 'other': 0.00
            },
            'ERCOT': {
                'natural_gas': 0.47, 'renewables': 0.31, 'nuclear': 0.11, 'coal': 0.11, 'other': 0.00
            },
            'PJM': {
                'natural_gas': 0.35, 'renewables': 0.18, 'nuclear': 0.35, 'coal': 0.12, 'other': 0.00
            },
            'NYISO': {
                'natural_gas': 0.40, 'renewables': 0.28, 'nuclear': 0.30, 'coal': 0.02, 'other': 0.00
            },
            'ISO_NE': {
                'natural_gas': 0.50, 'renewables': 0.22, 'nuclear': 0.25, 'coal': 0.03, 'other': 0.00
            },
            'MISO': {
                'natural_gas': 0.35, 'renewables': 0.25, 'nuclear': 0.18, 'coal': 0.22, 'other': 0.00
            },
            'SPP': {
                'natural_gas': 0.35, 'renewables': 0.35, 'nuclear': 0.05, 'coal': 0.25, 'other': 0.00
            }
        }
        
        # Emission factors by fuel type (kg CO2e/kWh)
        self.emission_factors = {
            'coal': 0.95,
            'natural_gas': 0.49,
            'oil': 0.78,
            'nuclear': 0.012,
            'hydro': 0.024,
            'wind': 0.011,
            'solar': 0.041,
            'biomass': 0.23,
            'geothermal': 0.038
        }
    
    def generate_hourly_carbon_intensity(self, region='PJM', year=2023):
        """Generate hourly carbon intensity data for a full year"""
        
        # Create hourly datetime index
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
        
        hourly_data = []
        
        # Get base intensity for region
        base_intensity = self.regional_intensities['ISO_RTO_Regions'][region]['intensity']
        fuel_mix = self.fuel_mixes[region]
        
        for dt in date_range:
            # Calculate time-varying factors
            seasonal_factor = self._calculate_seasonal_factor(dt.month)
            daily_factor = self._calculate_daily_factor(dt.hour, dt.weekday())
            renewable_factor = self._calculate_renewable_availability(dt, region)
            
            # Calculate marginal intensity (typically higher than average)
            marginal_multiplier = self._calculate_marginal_multiplier(dt.hour, dt.month)
            
            # Base carbon intensity
            average_intensity = base_intensity * seasonal_factor * daily_factor
            
            # Marginal intensity (what matters for energy efficiency decisions)
            marginal_intensity = average_intensity * marginal_multiplier
            
            # Adjust for renewable availability
            adjusted_average = average_intensity * (1 - renewable_factor * 0.3)
            adjusted_marginal = marginal_intensity * (1 - renewable_factor * 0.2)
            
            # Calculate displaced emissions (for energy efficiency)
            displaced_emissions = self._calculate_displaced_emissions(
                adjusted_marginal, fuel_mix, dt.hour
            )
            
            hourly_data.append({
                'datetime': dt,
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'weekday': dt.weekday(),
                'region': region,
                
                # Carbon intensities (kg CO2e/kWh)
                'average_intensity': adjusted_average,
                'marginal_intensity': adjusted_marginal,
                'displaced_emissions': displaced_emissions,
                
                # Contributing factors
                'seasonal_factor': seasonal_factor,
                'daily_factor': daily_factor,
                'renewable_factor': renewable_factor,
                'marginal_multiplier': marginal_multiplier,
                
                # Fuel mix at this hour
                'coal_fraction': fuel_mix['coal'] * (1 + 0.3 * (1 - renewable_factor)),
                'gas_fraction': fuel_mix['natural_gas'] * (1 + 0.2 * (1 - renewable_factor)),
                'renewable_fraction': fuel_mix['renewables'] * (1 + renewable_factor),
                'nuclear_fraction': fuel_mix['nuclear'],
                
                'data_source': 'Synthetic Carbon Intensity Generator'
            })
        
        return pd.DataFrame(hourly_data)
    
    def _calculate_seasonal_factor(self, month):
        """Calculate seasonal variation in carbon intensity"""
        
        # Higher intensity in winter (more heating load, less renewables)
        # Lower intensity in spring/fall (moderate load, good renewables)
        seasonal_factors = {
            1: 1.15,   # January - high heating
            2: 1.12,   # February - high heating
            3: 1.05,   # March - moderate
            4: 0.95,   # April - low load, good renewables
            5: 0.92,   # May - low load, good renewables
            6: 1.08,   # June - AC load starts
            7: 1.18,   # July - peak AC load
            8: 1.16,   # August - peak AC load
            9: 1.02,   # September - moderate
            10: 0.98,  # October - good conditions
            11: 1.06,  # November - heating starts
            12: 1.12   # December - heating load
        }
        
        return seasonal_factors.get(month, 1.0)
    
    def _calculate_daily_factor(self, hour, weekday):
        """Calculate daily and weekly variation in carbon intensity"""
        
        # Weekday vs weekend factor
        weekend_factor = 0.92 if weekday >= 5 else 1.0
        
        # Hourly factors (higher during peak hours)
        if 6 <= hour <= 9:  # Morning peak
            hourly_factor = 1.15
        elif 17 <= hour <= 21:  # Evening peak
            hourly_factor = 1.25
        elif 22 <= hour <= 5:  # Overnight (low load, base generation)
            hourly_factor = 0.85
        else:  # Mid-day and other hours
            hourly_factor = 1.0
        
        return weekend_factor * hourly_factor
    
    def _calculate_renewable_availability(self, dt, region):
        """Calculate renewable energy availability factor"""
        
        # Seasonal renewable availability
        if region == 'CAISO':
            # High solar, some wind
            solar_factor = max(0, np.sin((dt.hour - 6) * np.pi / 12)) if 6 <= dt.hour <= 18 else 0
            wind_factor = 0.3 + 0.2 * np.sin(dt.hour * np.pi / 12)
            renewable_factor = 0.7 * solar_factor + 0.3 * wind_factor
        elif region == 'ERCOT':
            # High wind, moderate solar
            wind_factor = 0.4 + 0.3 * np.sin((dt.hour + 6) * np.pi / 12)  # Wind often peaks at night
            solar_factor = max(0, np.sin((dt.hour - 6) * np.pi / 12)) if 6 <= dt.hour <= 18 else 0
            renewable_factor = 0.6 * wind_factor + 0.4 * solar_factor
        else:
            # Moderate renewables
            renewable_factor = 0.3 + 0.2 * np.sin(dt.hour * np.pi / 12)
        
        # Seasonal adjustment
        month_factor = 1.0 + 0.2 * np.sin((dt.month - 3) * np.pi / 6)  # Peak in summer
        
        return min(1.0, renewable_factor * month_factor)
    
    def _calculate_marginal_multiplier(self, hour, month):
        """Calculate marginal emission factor multiplier"""
        
        # Marginal emissions are typically higher than average
        # because the marginal plant is often a peaker (gas turbine)
        
        base_multiplier = 1.3  # Marginal typically 30% higher than average
        
        # Higher during peak hours when peakers are running
        if 17 <= hour <= 21:  # Evening peak
            peak_multiplier = 1.6
        elif 6 <= hour <= 9:  # Morning peak
            peak_multiplier = 1.4
        else:
            peak_multiplier = 1.2
        
        # Summer months have higher marginal emissions due to AC load
        summer_multiplier = 1.1 if month in [6, 7, 8] else 1.0
        
        return base_multiplier * peak_multiplier * summer_multiplier
    
    def _calculate_displaced_emissions(self, marginal_intensity, fuel_mix, hour):
        """Calculate emissions displaced by energy efficiency measures"""
        
        # Energy efficiency typically displaces marginal generation
        # But the specific fuel depends on the time of day and fuel mix
        
        if 17 <= hour <= 21:  # Evening peak - likely gas peakers
            displaced_factor = 1.2
        elif 6 <= hour <= 9:  # Morning peak - mix of gas and coal
            displaced_factor = 1.1
        elif 22 <= hour <= 5:  # Overnight - base load (coal/nuclear)
            displaced_factor = 0.9
        else:  # Mid-day - mix including renewables
            displaced_factor = 1.0
        
        return marginal_intensity * displaced_factor
    
    def generate_regional_comparison(self, year=2023):
        """Generate carbon intensity comparison across regions"""
        
        regional_data = []
        
        for region in self.regional_intensities['ISO_RTO_Regions'].keys():
            # Generate summary statistics for the region
            hourly_data = self.generate_hourly_carbon_intensity(region, year)
            
            regional_summary = {
                'region': region,
                'region_name': self.regional_intensities['ISO_RTO_Regions'][region]['name'],
                'year': year,
                
                # Annual averages
                'annual_avg_intensity': hourly_data['average_intensity'].mean(),
                'annual_avg_marginal': hourly_data['marginal_intensity'].mean(),
                'annual_avg_displaced': hourly_data['displaced_emissions'].mean(),
                
                # Peak values
                'peak_intensity': hourly_data['average_intensity'].max(),
                'peak_marginal': hourly_data['marginal_intensity'].max(),
                'min_intensity': hourly_data['average_intensity'].min(),
                'min_marginal': hourly_data['marginal_intensity'].min(),
                
                # Seasonal variations
                'summer_avg_intensity': hourly_data[hourly_data['month'].isin([6, 7, 8])]['average_intensity'].mean(),
                'winter_avg_intensity': hourly_data[hourly_data['month'].isin([12, 1, 2])]['average_intensity'].mean(),
                
                # Peak hour analysis
                'peak_hour_avg_intensity': hourly_data[hourly_data['hour'].isin([17, 18, 19, 20])]['average_intensity'].mean(),
                'off_peak_avg_intensity': hourly_data[hourly_data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])]['average_intensity'].mean(),
                
                # Fuel mix
                'avg_coal_fraction': hourly_data['coal_fraction'].mean(),
                'avg_gas_fraction': hourly_data['gas_fraction'].mean(),
                'avg_renewable_fraction': hourly_data['renewable_fraction'].mean(),
                'avg_nuclear_fraction': hourly_data['nuclear_fraction'].mean(),
                
                # Renewable penetration
                'renewable_penetration': self.regional_intensities['ISO_RTO_Regions'][region].get('renewable_penetration', 0.25)
            }
            
            regional_data.append(regional_summary)
        
        return pd.DataFrame(regional_data)
    
    def generate_future_projections(self, start_year=2024, end_year=2040):
        """Generate carbon intensity projections with grid decarbonization"""
        
        projection_data = []
        
        # Decarbonization scenarios
        scenarios = {
            'business_as_usual': {'annual_reduction': 0.02},  # 2% per year
            'moderate_decarbonization': {'annual_reduction': 0.04},  # 4% per year
            'aggressive_decarbonization': {'annual_reduction': 0.06}  # 6% per year
        }
        
        for scenario_name, scenario_params in scenarios.items():
            for region in self.regional_intensities['ISO_RTO_Regions'].keys():
                base_intensity = self.regional_intensities['ISO_RTO_Regions'][region]['intensity']
                
                for year in range(start_year, end_year + 1):
                    years_from_base = year - 2023
                    
                    # Calculate decarbonization factor
                    reduction_factor = (1 - scenario_params['annual_reduction']) ** years_from_base
                    projected_intensity = base_intensity * reduction_factor
                    
                    # Add some variability
                    np.random.seed(year + hash(region + scenario_name))
                    variability = np.random.normal(1, 0.05)  # 5% standard deviation
                    final_intensity = max(0.05, projected_intensity * variability)  # Floor at 50g/kWh
                    
                    projection_data.append({
                        'year': year,
                        'region': region,
                        'scenario': scenario_name,
                        'projected_intensity': final_intensity,
                        'base_intensity': base_intensity,
                        'reduction_factor': reduction_factor,
                        'annual_reduction_rate': scenario_params['annual_reduction'],
                        'cumulative_reduction': 1 - reduction_factor
                    })
        
        return pd.DataFrame(projection_data)
    
    def generate_marginal_emission_rates(self, region='PJM', year=2023):
        """Generate detailed marginal emission rates for different times"""
        
        # Time-differentiated marginal rates
        time_periods = {
            'peak_summer': {'months': [6, 7, 8], 'hours': [14, 15, 16, 17, 18, 19, 20]},
            'peak_winter': {'months': [12, 1, 2], 'hours': [7, 8, 9, 17, 18, 19, 20]},
            'shoulder': {'months': [3, 4, 5, 9, 10, 11], 'hours': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
            'off_peak': {'months': list(range(1, 13)), 'hours': [22, 23, 0, 1, 2, 3, 4, 5, 6]}
        }
        
        base_intensity = self.regional_intensities['ISO_RTO_Regions'][region]['intensity']
        
        marginal_rates = []
        
        for period_name, period_def in time_periods.items():
            # Calculate marginal rate for this period
            if period_name == 'peak_summer':
                marginal_multiplier = 1.8  # Gas peakers during AC load
            elif period_name == 'peak_winter':
                marginal_multiplier = 1.6  # Gas and some oil during heating
            elif period_name == 'shoulder':
                marginal_multiplier = 1.3  # Mix of gas and renewables
            else:  # off_peak
                marginal_multiplier = 1.1  # Base load plants
            
            marginal_rate = base_intensity * marginal_multiplier
            
            marginal_rates.append({
                'region': region,
                'time_period': period_name,
                'months': period_def['months'],
                'hours': period_def['hours'],
                'marginal_emission_rate': marginal_rate,
                'base_intensity': base_intensity,
                'marginal_multiplier': marginal_multiplier,
                'year': year,
                'primary_marginal_fuel': self._get_marginal_fuel(period_name),
                'confidence_level': self._get_confidence_level(period_name)
            })
        
        return marginal_rates
    
    def _get_marginal_fuel(self, period_name):
        """Determine the likely marginal fuel for each time period"""
        
        marginal_fuels = {
            'peak_summer': 'natural_gas_peaker',
            'peak_winter': 'natural_gas_combined_cycle',
            'shoulder': 'natural_gas_combined_cycle',
            'off_peak': 'coal_or_nuclear'
        }
        
        return marginal_fuels.get(period_name, 'natural_gas')
    
    def _get_confidence_level(self, period_name):
        """Assign confidence level to marginal emission estimates"""
        
        confidence_levels = {
            'peak_summer': 'high',      # Clear marginal plant
            'peak_winter': 'high',      # Clear marginal plant
            'shoulder': 'medium',       # More variable
            'off_peak': 'low'          # Complex base load dispatch
        }
        
        return confidence_levels.get(period_name, 'medium')

def main():
    """Generate carbon intensity data for all regions"""
    
    print("Generating carbon intensity data...")
    
    generator = CarbonIntensityGenerator()
    
    # Generate datasets for key regions
    key_regions = ['PJM', 'CAISO', 'ERCOT', 'NYISO', 'ISO_NE']
    
    for region in key_regions:
        print(f"Generating hourly data for {region}...")
        
        # Generate hourly data
        hourly_data = generator.generate_hourly_carbon_intensity(region)
        
        # Create region directory
        region_dir = f"geospatial_regulatory/carbon_intensity/{region.lower()}"
        os.makedirs(region_dir, exist_ok=True)
        
        # Save hourly data
        hourly_data.to_csv(f"{region_dir}/hourly_carbon_intensity_2023.csv", index=False)
        
        # Generate marginal rates
        marginal_rates = generator.generate_marginal_emission_rates(region)
        
        with open(f"{region_dir}/marginal_emission_rates.json", 'w') as f:
            json.dump(marginal_rates, f, indent=2, default=str)
    
    # Generate regional comparison
    regional_comparison = generator.generate_regional_comparison()
    regional_comparison.to_csv("geospatial_regulatory/carbon_intensity/regional_comparison_2023.csv", index=False)
    
    # Generate future projections
    future_projections = generator.generate_future_projections()
    future_projections.to_csv("geospatial_regulatory/carbon_intensity/future_projections_2024_2040.csv", index=False)
    
    # Generate comprehensive summary
    carbon_summary = {
        'generation_date': datetime.now().isoformat(),
        'regions_covered': key_regions,
        'data_year': 2023,
        'projection_years': '2024-2040',
        'data_categories': [
            'hourly_average_intensity', 'hourly_marginal_intensity',
            'displaced_emissions', 'fuel_mix_hourly', 'seasonal_variations',
            'peak_vs_offpeak', 'regional_comparison', 'future_projections'
        ],
        'key_insights': {
            'cleanest_region': regional_comparison.loc[regional_comparison['annual_avg_intensity'].idxmin(), 'region'],
            'dirtiest_region': regional_comparison.loc[regional_comparison['annual_avg_intensity'].idxmax(), 'region'],
            'highest_renewable_penetration': regional_comparison.loc[regional_comparison['renewable_penetration'].idxmax(), 'region'],
            'peak_vs_offpeak_difference': 'Marginal emissions 30-80% higher during peak hours'
        },
        'methodology': {
            'marginal_calculation': 'Based on dispatch order and fuel mix',
            'seasonal_factors': 'Higher winter/summer, lower spring/fall',
            'renewable_integration': 'Time-varying based on solar/wind availability',
            'future_projections': 'Linear decarbonization scenarios'
        }
    }
    
    with open("geospatial_regulatory/carbon_intensity/carbon_intensity_summary.json", 'w') as f:
        json.dump(carbon_summary, f, indent=2)
    
    print("Carbon intensity data generation completed!")

if __name__ == "__main__":
    main()