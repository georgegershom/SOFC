#!/usr/bin/env python3
"""
Future Climate Projections Data Generator
Generates climate change scenarios based on IPCC projections for building retrofit planning
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class ClimateProjectionsGenerator:
    def __init__(self, location_name="New York City", latitude=40.7128, longitude=-74.0060):
        self.location_name = location_name
        self.latitude = latitude
        self.longitude = longitude
        
        # IPCC scenarios and their characteristics
        self.scenarios = {
            'SSP1-1.9': {
                'name': 'Very Low Emissions',
                'description': 'Strong mitigation, 1.5°C target',
                'temp_increase_2050': 1.4,
                'temp_increase_2080': 1.6,
                'precipitation_change_2050': 0.05,
                'precipitation_change_2080': 0.08
            },
            'SSP1-2.6': {
                'name': 'Low Emissions',
                'description': 'Strong mitigation, well below 2°C',
                'temp_increase_2050': 1.8,
                'temp_increase_2080': 2.0,
                'precipitation_change_2050': 0.08,
                'precipitation_change_2080': 0.12
            },
            'SSP2-4.5': {
                'name': 'Intermediate Emissions',
                'description': 'Middle of the road',
                'temp_increase_2050': 2.2,
                'temp_increase_2080': 2.8,
                'precipitation_change_2050': 0.10,
                'precipitation_change_2080': 0.15
            },
            'SSP3-7.0': {
                'name': 'High Emissions',
                'description': 'Regional rivalry',
                'temp_increase_2050': 2.8,
                'temp_increase_2080': 4.1,
                'precipitation_change_2050': 0.12,
                'precipitation_change_2080': 0.20
            },
            'SSP5-8.5': {
                'name': 'Very High Emissions',
                'description': 'Fossil-fueled development',
                'temp_increase_2050': 3.2,
                'temp_increase_2080': 4.8,
                'precipitation_change_2050': 0.15,
                'precipitation_change_2080': 0.25
            }
        }
    
    def generate_climate_projections(self, baseline_year=2023, projection_years=[2030, 2040, 2050, 2060, 2070, 2080]):
        """Generate climate projections for multiple scenarios and years"""
        
        projections = {}
        
        for scenario_id, scenario_data in self.scenarios.items():
            projections[scenario_id] = {}
            
            for year in projection_years:
                print(f"Generating projections for {scenario_id} - {year}")
                
                # Calculate climate changes based on year and scenario
                years_from_baseline = year - baseline_year
                
                # Linear interpolation between 2050 and 2080 targets
                if year <= 2050:
                    temp_factor = years_from_baseline / (2050 - baseline_year)
                    temp_increase = scenario_data['temp_increase_2050'] * temp_factor
                    precip_factor = years_from_baseline / (2050 - baseline_year)
                    precip_change = scenario_data['precipitation_change_2050'] * precip_factor
                else:
                    temp_factor = (years_from_baseline - (2050 - baseline_year)) / (2080 - 2050)
                    temp_increase = scenario_data['temp_increase_2050'] + \
                                  (scenario_data['temp_increase_2080'] - scenario_data['temp_increase_2050']) * temp_factor
                    precip_factor = (years_from_baseline - (2050 - baseline_year)) / (2080 - 2050)
                    precip_change = scenario_data['precipitation_change_2050'] + \
                                  (scenario_data['precipitation_change_2080'] - scenario_data['precipitation_change_2050']) * precip_factor
                
                # Generate modified weather data
                projection_data = self._generate_projected_weather(
                    year, temp_increase, precip_change, scenario_id
                )
                
                projections[scenario_id][year] = projection_data
        
        return projections
    
    def _generate_projected_weather(self, year, temp_increase, precip_change, scenario):
        """Generate weather data modified by climate change projections"""
        
        # Create hourly datetime index
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
        
        # Base weather generation (similar to TMY but modified)
        data = {
            'datetime': date_range,
            'year': date_range.year,
            'month': date_range.month,
            'day': date_range.day,
            'hour': date_range.hour,
            'scenario': scenario
        }
        
        # Generate modified weather parameters
        data.update(self._generate_modified_temperature(date_range, temp_increase))
        data.update(self._generate_modified_precipitation(date_range, precip_change))
        data.update(self._generate_modified_extremes(date_range, temp_increase, scenario))
        data.update(self._generate_modified_humidity(date_range, temp_increase))
        data.update(self._generate_modified_wind(date_range, scenario))
        
        df = pd.DataFrame(data)
        
        # Add climate change indicators
        df['heating_degree_days'] = np.maximum(0, 18.3 - df['daily_mean_temp'])
        df['cooling_degree_days'] = np.maximum(0, df['daily_mean_temp'] - 18.3)
        
        return df
    
    def _generate_modified_temperature(self, date_range, temp_increase):
        """Generate temperature data modified by climate change"""
        hours = len(date_range)
        
        # Base seasonal temperature
        day_of_year = date_range.dayofyear
        seasonal_temp = 15 + 15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
        
        # Apply climate change increase
        seasonal_temp += temp_increase
        
        # Enhanced diurnal variation (climate change can increase temperature swings)
        hour_of_day = date_range.hour
        diurnal_variation = 8.5 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
        
        # Increased variability
        np.random.seed(42 + int(temp_increase * 10))
        weather_noise = np.random.normal(0, 3.5, hours)
        
        dry_bulb_temp = seasonal_temp + diurnal_variation + weather_noise
        
        # Calculate daily means for degree days
        daily_temps = []
        for i in range(0, hours, 24):
            daily_mean = np.mean(dry_bulb_temp[i:i+24])
            daily_temps.extend([daily_mean] * min(24, hours - i))
        
        return {
            'dry_bulb_temp_c': dry_bulb_temp,
            'daily_mean_temp': daily_temps[:hours],
            'temp_increase_from_baseline': temp_increase
        }
    
    def _generate_modified_precipitation(self, date_range, precip_change):
        """Generate precipitation data modified by climate change"""
        hours = len(date_range)
        np.random.seed(45)
        
        # Modified precipitation probability and intensity
        base_prob = 0.1
        modified_prob = base_prob * (1 + precip_change)
        
        has_precip = np.random.random(hours) < modified_prob
        
        # Increased intensity when it rains (climate change pattern)
        base_intensity = 2.0
        modified_intensity = base_intensity * (1 + precip_change * 1.5)
        
        precip_amount = np.where(has_precip, np.random.exponential(modified_intensity, hours), 0)
        
        return {
            'precipitation_mm': precip_amount,
            'precipitation_change_factor': 1 + precip_change,
            'extreme_precipitation_events': (precip_amount > 10).astype(int)
        }
    
    def _generate_modified_extremes(self, date_range, temp_increase, scenario):
        """Generate extreme weather events modified by climate change"""
        hours = len(date_range)
        
        # Heat wave indicators
        heat_wave_threshold = 32 + temp_increase  # Adjusted threshold
        
        # Extreme heat events (more frequent with climate change)
        extreme_heat_multiplier = {
            'SSP1-1.9': 1.2,
            'SSP1-2.6': 1.4,
            'SSP2-4.5': 1.8,
            'SSP3-7.0': 2.2,
            'SSP5-8.5': 2.8
        }
        
        base_extreme_prob = 0.02
        extreme_prob = base_extreme_prob * extreme_heat_multiplier.get(scenario, 1.5)
        
        extreme_heat_events = np.random.random(hours) < extreme_prob
        
        return {
            'heat_wave_threshold_c': heat_wave_threshold,
            'extreme_heat_events': extreme_heat_events.astype(int),
            'extreme_weather_multiplier': extreme_heat_multiplier.get(scenario, 1.5)
        }
    
    def _generate_modified_humidity(self, date_range, temp_increase):
        """Generate humidity data accounting for warmer air holding more moisture"""
        hours = len(date_range)
        np.random.seed(43)
        
        # Base humidity
        day_of_year = date_range.dayofyear
        seasonal_humidity = 60 + 20 * np.cos(2 * np.pi * (day_of_year - 180) / 365)
        
        # Warmer air can hold more moisture (Clausius-Clapeyron relation)
        humidity_increase = temp_increase * 3  # Approximate 7% per degree, but moderated
        modified_humidity = seasonal_humidity + humidity_increase
        
        # Daily variation
        hour_of_day = date_range.hour
        diurnal_humidity = 15 * np.cos(2 * np.pi * (hour_of_day - 6) / 24)
        
        humidity_noise = np.random.normal(0, 10, hours)
        relative_humidity = np.clip(modified_humidity + diurnal_humidity + humidity_noise, 10, 95)
        
        return {
            'relative_humidity': relative_humidity,
            'humidity_increase_from_baseline': humidity_increase
        }
    
    def _generate_modified_wind(self, date_range, scenario):
        """Generate wind data with climate change modifications"""
        hours = len(date_range)
        np.random.seed(44)
        
        # Base wind patterns
        day_of_year = date_range.dayofyear
        seasonal_wind = 4 + 2 * np.cos(2 * np.pi * (day_of_year - 60) / 365)
        
        # Climate change can alter wind patterns
        wind_change_factors = {
            'SSP1-1.9': 1.02,
            'SSP1-2.6': 1.05,
            'SSP2-4.5': 1.08,
            'SSP3-7.0': 1.12,
            'SSP5-8.5': 1.15
        }
        
        wind_factor = wind_change_factors.get(scenario, 1.05)
        modified_wind = seasonal_wind * wind_factor
        
        wind_noise = np.random.exponential(2, hours)
        wind_speed = modified_wind + wind_noise
        
        return {
            'wind_speed_ms': wind_speed,
            'wind_change_factor': wind_factor
        }
    
    def save_projections(self, projections, output_dir="weather_climate"):
        """Save climate projections to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario_id, scenario_projections in projections.items():
            scenario_dir = os.path.join(output_dir, f"climate_projections_{scenario_id}")
            os.makedirs(scenario_dir, exist_ok=True)
            
            for year, data in scenario_projections.items():
                # Save CSV data
                filename = f"climate_projection_{scenario_id}_{year}.csv"
                filepath = os.path.join(scenario_dir, filename)
                data.to_csv(filepath, index=False)
                
                # Save summary statistics
                summary = {
                    'scenario': scenario_id,
                    'year': year,
                    'location': self.location_name,
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                    'statistics': {
                        'mean_temperature': float(data['dry_bulb_temp_c'].mean()),
                        'max_temperature': float(data['dry_bulb_temp_c'].max()),
                        'min_temperature': float(data['dry_bulb_temp_c'].min()),
                        'total_precipitation': float(data['precipitation_mm'].sum()),
                        'extreme_heat_days': int(data['extreme_heat_events'].sum()),
                        'heating_degree_days': float(data['heating_degree_days'].sum()),
                        'cooling_degree_days': float(data['cooling_degree_days'].sum())
                    }
                }
                
                summary_filename = f"summary_{scenario_id}_{year}.json"
                summary_filepath = os.path.join(scenario_dir, summary_filename)
                
                with open(summary_filepath, 'w') as f:
                    json.dump(summary, f, indent=2)

def main():
    """Generate climate projections for multiple locations"""
    
    locations = [
        {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
        {"name": "Seattle", "lat": 47.6062, "lon": -122.3321}
    ]
    
    projection_years = [2030, 2040, 2050, 2060, 2070, 2080]
    
    for location in locations:
        print(f"\nGenerating climate projections for {location['name']}...")
        
        generator = ClimateProjectionsGenerator(
            location_name=location['name'],
            latitude=location['lat'],
            longitude=location['lon']
        )
        
        projections = generator.generate_climate_projections(
            baseline_year=2023,
            projection_years=projection_years
        )
        
        # Create location-specific output directory
        location_dir = f"weather_climate/projections_{location['name'].lower().replace(' ', '_')}"
        generator.save_projections(projections, location_dir)
        
        print(f"Saved climate projections for {location['name']}")

if __name__ == "__main__":
    main()