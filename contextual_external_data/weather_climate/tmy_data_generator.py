#!/usr/bin/env python3
"""
TMY (Typical Meteorological Year) Data Generator
Generates synthetic but realistic TMY data for building energy simulation
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class TMYDataGenerator:
    def __init__(self, location_name="New York City", latitude=40.7128, longitude=-74.0060):
        self.location_name = location_name
        self.latitude = latitude
        self.longitude = longitude
        
    def generate_tmy_data(self, year=2023):
        """Generate comprehensive TMY data for a full year"""
        
        # Create hourly datetime index for full year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
        
        # Initialize data dictionary
        data = {
            'datetime': date_range,
            'year': date_range.year,
            'month': date_range.month,
            'day': date_range.day,
            'hour': date_range.hour,
            'day_of_year': date_range.dayofyear
        }
        
        # Generate weather parameters
        data.update(self._generate_temperature_data(date_range))
        data.update(self._generate_humidity_data(date_range))
        data.update(self._generate_solar_radiation_data(date_range))
        data.update(self._generate_wind_data(date_range))
        data.update(self._generate_precipitation_data(date_range))
        data.update(self._generate_pressure_data(date_range))
        data.update(self._generate_cloud_cover_data(date_range))
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add derived parameters
        df['heat_index'] = self._calculate_heat_index(df['dry_bulb_temp_c'], df['relative_humidity'])
        df['wind_chill'] = self._calculate_wind_chill(df['dry_bulb_temp_c'], df['wind_speed_ms'])
        df['dew_point_temp_c'] = self._calculate_dew_point(df['dry_bulb_temp_c'], df['relative_humidity'])
        
        return df
    
    def _generate_temperature_data(self, date_range):
        """Generate realistic temperature data with seasonal and diurnal variations"""
        hours = len(date_range)
        
        # Base seasonal temperature (NYC climate)
        day_of_year = date_range.dayofyear
        seasonal_temp = 15 + 15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
        
        # Diurnal variation
        hour_of_day = date_range.hour
        diurnal_variation = 8 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
        
        # Random weather variations
        np.random.seed(42)
        weather_noise = np.random.normal(0, 3, hours)
        
        # Combine all effects
        dry_bulb_temp = seasonal_temp + diurnal_variation + weather_noise
        
        # Wet bulb temperature (typically 2-5°C lower than dry bulb)
        wet_bulb_temp = dry_bulb_temp - np.random.uniform(2, 5, hours)
        
        return {
            'dry_bulb_temp_c': dry_bulb_temp,
            'wet_bulb_temp_c': wet_bulb_temp,
            'dry_bulb_temp_f': dry_bulb_temp * 9/5 + 32,
            'wet_bulb_temp_f': wet_bulb_temp * 9/5 + 32
        }
    
    def _generate_humidity_data(self, date_range):
        """Generate humidity data correlated with temperature"""
        hours = len(date_range)
        np.random.seed(43)
        
        # Base humidity with seasonal variation
        day_of_year = date_range.dayofyear
        seasonal_humidity = 60 + 20 * np.cos(2 * np.pi * (day_of_year - 180) / 365)
        
        # Daily variation (higher at night)
        hour_of_day = date_range.hour
        diurnal_humidity = 15 * np.cos(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Random variations
        humidity_noise = np.random.normal(0, 10, hours)
        
        relative_humidity = np.clip(seasonal_humidity + diurnal_humidity + humidity_noise, 10, 95)
        
        return {
            'relative_humidity': relative_humidity,
            'specific_humidity': relative_humidity / 100 * 0.02  # Simplified calculation
        }
    
    def _generate_solar_radiation_data(self, date_range):
        """Generate solar radiation data based on time of day and season"""
        hours = len(date_range)
        
        # Solar elevation angle approximation
        day_of_year = date_range.dayofyear
        hour_of_day = date_range.hour
        
        # Declination angle
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        
        # Hour angle
        hour_angle = 15 * (hour_of_day - 12)
        
        # Solar elevation (simplified)
        elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(self.latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(self.latitude)) * 
            np.cos(np.radians(hour_angle))
        )
        
        # Direct normal irradiance (DNI)
        dni = np.maximum(0, 900 * np.sin(elevation) * (1 - 0.3 * np.random.random(hours)))
        
        # Diffuse horizontal irradiance (DHI)
        dhi = np.maximum(0, 150 + 100 * np.sin(elevation) * np.random.random(hours))
        
        # Global horizontal irradiance (GHI)
        ghi = dni * np.sin(elevation) + dhi
        
        return {
            'global_horizontal_irradiance': np.maximum(0, ghi),
            'direct_normal_irradiance': np.maximum(0, dni),
            'diffuse_horizontal_irradiance': np.maximum(0, dhi),
            'solar_elevation_angle': np.degrees(elevation),
            'solar_azimuth_angle': 180 + np.degrees(np.arctan2(
                np.sin(np.radians(hour_angle)),
                np.cos(np.radians(hour_angle)) * np.sin(np.radians(self.latitude)) -
                np.tan(np.radians(declination)) * np.cos(np.radians(self.latitude))
            ))
        }
    
    def _generate_wind_data(self, date_range):
        """Generate wind speed and direction data"""
        hours = len(date_range)
        np.random.seed(44)
        
        # Base wind speed with seasonal variation
        day_of_year = date_range.dayofyear
        seasonal_wind = 4 + 2 * np.cos(2 * np.pi * (day_of_year - 60) / 365)
        
        # Random variations
        wind_noise = np.random.exponential(2, hours)
        wind_speed = seasonal_wind + wind_noise
        
        # Wind direction (prevailing westerly with variations)
        wind_direction = 270 + 60 * np.random.normal(0, 1, hours)
        wind_direction = wind_direction % 360
        
        return {
            'wind_speed_ms': wind_speed,
            'wind_speed_kmh': wind_speed * 3.6,
            'wind_direction_deg': wind_direction
        }
    
    def _generate_precipitation_data(self, date_range):
        """Generate precipitation data"""
        hours = len(date_range)
        np.random.seed(45)
        
        # Probability of precipitation
        precip_prob = 0.1  # 10% chance per hour
        has_precip = np.random.random(hours) < precip_prob
        
        # Precipitation amount when it occurs
        precip_amount = np.where(has_precip, np.random.exponential(2, hours), 0)
        
        return {
            'precipitation_mm': precip_amount,
            'precipitation_in': precip_amount / 25.4
        }
    
    def _generate_pressure_data(self, date_range):
        """Generate atmospheric pressure data"""
        hours = len(date_range)
        np.random.seed(46)
        
        # Base pressure with small variations
        base_pressure = 1013.25  # Standard atmospheric pressure in hPa
        pressure_variation = np.random.normal(0, 10, hours)
        
        atmospheric_pressure = base_pressure + pressure_variation
        
        return {
            'atmospheric_pressure_hpa': atmospheric_pressure,
            'atmospheric_pressure_inhg': atmospheric_pressure * 0.02953
        }
    
    def _generate_cloud_cover_data(self, date_range):
        """Generate cloud cover data"""
        hours = len(date_range)
        np.random.seed(47)
        
        # Cloud cover (0-10 oktas)
        cloud_cover = np.random.choice(range(11), hours, p=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
        
        return {
            'cloud_cover_oktas': cloud_cover,
            'cloud_cover_percent': cloud_cover * 10
        }
    
    def _calculate_heat_index(self, temp_c, humidity):
        """Calculate heat index"""
        temp_f = temp_c * 9/5 + 32
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        return (hi - 32) * 5/9  # Convert back to Celsius
    
    def _calculate_wind_chill(self, temp_c, wind_ms):
        """Calculate wind chill"""
        temp_f = temp_c * 9/5 + 32
        wind_mph = wind_ms * 2.237
        
        if temp_f <= 50 and wind_mph >= 3:
            wc = 35.74 + (0.6215 * temp_f) - (35.75 * (wind_mph ** 0.16)) + (0.4275 * temp_f * (wind_mph ** 0.16))
            return (wc - 32) * 5/9
        else:
            return temp_c
    
    def _calculate_dew_point(self, temp_c, humidity):
        """Calculate dew point temperature"""
        a = 17.27
        b = 237.7
        alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)

def main():
    """Generate TMY data for multiple locations"""
    
    locations = [
        {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
        {"name": "Seattle", "lat": 47.6062, "lon": -122.3321}
    ]
    
    for location in locations:
        print(f"Generating TMY data for {location['name']}...")
        
        generator = TMYDataGenerator(
            location_name=location['name'],
            latitude=location['lat'],
            longitude=location['lon']
        )
        
        tmy_data = generator.generate_tmy_data()
        
        # Save to CSV
        filename = f"tmy_data_{location['name'].lower().replace(' ', '_')}_2023.csv"
        filepath = os.path.join("weather_climate", filename)
        tmy_data.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            "location_name": location['name'],
            "latitude": location['lat'],
            "longitude": location['lon'],
            "year": 2023,
            "data_source": "Synthetic TMY Data Generator",
            "generation_date": datetime.now().isoformat(),
            "parameters": {
                "temporal_resolution": "hourly",
                "total_hours": len(tmy_data),
                "variables": list(tmy_data.columns)
            }
        }
        
        metadata_filename = f"tmy_metadata_{location['name'].lower().replace(' ', '_')}_2023.json"
        metadata_filepath = os.path.join("weather_climate", metadata_filename)
        
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {filename} and {metadata_filename}")

if __name__ == "__main__":
    main()