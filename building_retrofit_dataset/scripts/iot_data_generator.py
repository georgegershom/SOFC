#!/usr/bin/env python3
"""
IoT Data Generator for Building Retrofit Dataset

Generates realistic IoT sensor data including:
- Energy consumption (whole building & end-use)
- Indoor environmental parameters (CO₂, TVOC, PM2.5, temperature, humidity)
- Outdoor weather conditions
- Occupancy patterns

Author: AI Assistant
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import pytz

class IoTDataGenerator:
    """Generator for IoT sensor data with realistic patterns and correlations."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Time series parameters
        self.duration_months = config['scale']['time_series_duration_months']
        self.frequency_minutes = config['scale']['sensor_frequency_minutes']
        
        # Generate time index
        self.start_date = datetime(2021, 1, 1)
        self.end_date = self.start_date + timedelta(days=30 * self.duration_months)
        self.time_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f'{self.frequency_minutes}min'
        )
        
    def generate_sensor_data(self, building_registry: pd.DataFrame, 
                           building_attributes: pd.DataFrame,
                           energy_performance: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate complete IoT sensor dataset for all buildings."""
        
        self.logger.info(f"Generating IoT data for {len(building_registry)} buildings")
        
        # Initialize data containers
        energy_data_list = []
        environmental_data_list = []
        weather_data_list = []
        occupancy_data_list = []
        
        # Process buildings in batches to manage memory
        batch_size = 100
        num_batches = len(building_registry) // batch_size + 1
        
        for batch_idx in tqdm(range(num_batches), desc="Generating IoT data batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(building_registry))
            
            if start_idx >= len(building_registry):
                break
                
            batch_buildings = building_registry.iloc[start_idx:end_idx]
            batch_attributes = building_attributes[
                building_attributes['building_id'].isin(batch_buildings['building_id'])
            ]
            batch_energy_perf = energy_performance[
                energy_performance['building_id'].isin(batch_buildings['building_id'])
            ]
            
            # Generate data for this batch
            batch_energy = self._generate_energy_consumption_batch(
                batch_buildings, batch_attributes, batch_energy_perf
            )
            batch_environmental = self._generate_environmental_data_batch(
                batch_buildings, batch_attributes
            )
            batch_weather = self._generate_weather_data_batch(batch_buildings)
            batch_occupancy = self._generate_occupancy_data_batch(
                batch_buildings, batch_attributes
            )
            
            energy_data_list.append(batch_energy)
            environmental_data_list.append(batch_environmental)
            weather_data_list.append(batch_weather)
            occupancy_data_list.append(batch_occupancy)
        
        # Combine all batches
        energy_data = pd.concat(energy_data_list, ignore_index=True)
        environmental_data = pd.concat(environmental_data_list, ignore_index=True)
        weather_data = pd.concat(weather_data_list, ignore_index=True)
        occupancy_data = pd.concat(occupancy_data_list, ignore_index=True)
        
        return {
            'iot_energy_consumption': energy_data,
            'iot_environmental_parameters': environmental_data,
            'iot_weather_data': weather_data,
            'iot_occupancy_patterns': occupancy_data
        }
    
    def _generate_energy_consumption_batch(self, buildings: pd.DataFrame,
                                         attributes: pd.DataFrame,
                                         energy_perf: pd.DataFrame) -> pd.DataFrame:
        """Generate energy consumption data for a batch of buildings."""
        
        energy_records = []
        
        for _, building in buildings.iterrows():
            building_id = building['building_id']
            
            # Get building characteristics
            attr = attributes[attributes['building_id'] == building_id].iloc[0]
            perf = energy_perf[energy_perf['building_id'] == building_id].iloc[0]
            
            # Base consumption from energy performance data
            annual_electricity = perf['annual_electricity_kwh']
            annual_heating = perf['annual_heating_kwh']
            
            # Generate hourly patterns
            for timestamp in self.time_index:
                # Time-based factors
                hour = timestamp.hour
                month = timestamp.month
                day_of_week = timestamp.weekday()
                
                # Seasonal factors
                heating_season_factor = self._get_heating_season_factor(month)
                cooling_season_factor = self._get_cooling_season_factor(month)
                
                # Daily patterns
                occupancy_factor = self._get_occupancy_factor(hour, day_of_week, attr['building_type'])
                
                # Base loads (kW)
                base_electricity = (annual_electricity / 8760) * (0.3 + 0.7 * occupancy_factor)
                
                # Heating consumption
                heating_consumption = (annual_heating / 8760) * heating_season_factor * \
                                    (0.5 + 0.5 * occupancy_factor)
                
                # Cooling consumption (estimated from electricity)
                cooling_consumption = base_electricity * 0.3 * cooling_season_factor
                
                # Hot water consumption
                hot_water_consumption = (annual_electricity * 0.15 / 8760) * \
                                      (0.4 + 0.6 * occupancy_factor)
                
                # Lighting consumption
                lighting_factor = self._get_lighting_factor(hour, month)
                lighting_consumption = (annual_electricity * 0.2 / 8760) * lighting_factor
                
                # Appliance consumption
                appliance_consumption = (annual_electricity * 0.35 / 8760) * occupancy_factor
                
                # Add realistic noise
                noise_factor = np.random.normal(1.0, 0.05)
                
                # Whole building consumption
                whole_building_electricity = (base_electricity + cooling_consumption + 
                                            lighting_consumption + appliance_consumption) * noise_factor
                
                record = {
                    'building_id': building_id,
                    'timestamp': timestamp,
                    'whole_building_electricity_kw': max(0, whole_building_electricity),
                    'heating_consumption_kw': max(0, heating_consumption * noise_factor),
                    'cooling_consumption_kw': max(0, cooling_consumption * noise_factor),
                    'hot_water_consumption_kw': max(0, hot_water_consumption * noise_factor),
                    'lighting_consumption_kw': max(0, lighting_consumption * noise_factor),
                    'appliance_consumption_kw': max(0, appliance_consumption * noise_factor)
                }
                
                energy_records.append(record)
        
        return pd.DataFrame(energy_records)
    
    def _generate_environmental_data_batch(self, buildings: pd.DataFrame,
                                         attributes: pd.DataFrame) -> pd.DataFrame:
        """Generate indoor environmental data for a batch of buildings."""
        
        env_records = []
        
        for _, building in buildings.iterrows():
            building_id = building['building_id']
            attr = attributes[attributes['building_id'] == building_id].iloc[0]
            
            for timestamp in self.time_index:
                hour = timestamp.hour
                month = timestamp.month
                day_of_week = timestamp.weekday()
                
                # Base environmental conditions
                occupancy_factor = self._get_occupancy_factor(hour, day_of_week, attr['building_type'])
                
                # CO2 concentration (ppm)
                base_co2 = 400  # Outdoor baseline
                occupancy_co2 = occupancy_factor * 600  # Additional from occupants
                ventilation_effectiveness = 0.7 + 0.3 * np.random.random()
                co2_concentration = base_co2 + (occupancy_co2 / ventilation_effectiveness)
                
                # TVOC (ppb)
                base_tvoc = 50 + np.random.normal(0, 10)
                occupancy_tvoc = occupancy_factor * 200
                tvoc_concentration = max(0, base_tvoc + occupancy_tvoc)
                
                # PM2.5 (μg/m³)
                outdoor_pm25 = 15 + 10 * np.sin(2 * np.pi * (month - 1) / 12)  # Seasonal variation
                indoor_pm25 = outdoor_pm25 * 0.6 + occupancy_factor * 5
                pm25_concentration = max(0, indoor_pm25 + np.random.normal(0, 2))
                
                # Temperature (°C)
                outdoor_temp = self._get_outdoor_temperature(month, building['latitude'])
                hvac_setpoint = 21 if month in [11, 12, 1, 2, 3] else 24
                temp_variation = np.random.normal(0, 1)
                indoor_temperature = hvac_setpoint + temp_variation
                
                # Humidity (%)
                base_humidity = 45 + 10 * np.sin(2 * np.pi * (month - 1) / 12)
                occupancy_humidity = occupancy_factor * 10
                indoor_humidity = np.clip(base_humidity + occupancy_humidity + 
                                        np.random.normal(0, 3), 20, 80)
                
                # Noise level (dB)
                base_noise = 35  # Background noise
                occupancy_noise = occupancy_factor * 15
                noise_level = base_noise + occupancy_noise + np.random.normal(0, 2)
                
                record = {
                    'building_id': building_id,
                    'timestamp': timestamp,
                    'co2_concentration_ppm': max(300, co2_concentration),
                    'tvoc_concentration_ppb': tvoc_concentration,
                    'pm25_concentration_ugm3': pm25_concentration,
                    'indoor_temperature_celsius': indoor_temperature,
                    'indoor_humidity_percent': indoor_humidity,
                    'noise_level_db': max(30, noise_level)
                }
                
                env_records.append(record)
        
        return pd.DataFrame(env_records)
    
    def _generate_weather_data_batch(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """Generate outdoor weather data for a batch of buildings."""
        
        weather_records = []
        
        # Group buildings by location to avoid duplicate weather data
        location_groups = buildings.groupby(['city', 'latitude', 'longitude'])
        
        for (city, lat, lon), city_buildings in location_groups:
            for timestamp in self.time_index:
                hour = timestamp.hour
                month = timestamp.month
                day_of_year = timestamp.timetuple().tm_yday
                
                # Temperature with seasonal and daily variation
                annual_mean_temp = 10 + 5 * np.cos(np.radians(abs(lat) - 45))  # Latitude effect
                seasonal_temp = 15 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
                daily_temp_variation = 8 * np.cos(2 * np.pi * (hour - 14) / 24)
                outdoor_temperature = (annual_mean_temp + seasonal_temp + 
                                     daily_temp_variation + np.random.normal(0, 2))
                
                # Humidity
                base_humidity = 60 + 20 * np.sin(2 * np.pi * (month - 1) / 12)
                outdoor_humidity = np.clip(base_humidity + np.random.normal(0, 5), 20, 95)
                
                # Wind speed and direction
                wind_speed = np.random.gamma(2, 2)  # Realistic wind speed distribution
                wind_direction = np.random.uniform(0, 360)
                
                # Solar irradiance
                solar_elevation = self._calculate_solar_elevation(lat, day_of_year, hour)
                clear_sky_irradiance = max(0, 1000 * np.sin(np.radians(solar_elevation)))
                cloud_factor = np.random.beta(2, 2)  # Cloud coverage effect
                solar_irradiance = clear_sky_irradiance * cloud_factor
                
                # Precipitation
                precipitation_prob = 0.1 + 0.05 * np.sin(2 * np.pi * (month - 1) / 12)
                precipitation = np.random.exponential(2) if np.random.random() < precipitation_prob else 0
                
                # Atmospheric pressure
                base_pressure = 1013.25
                pressure_variation = np.random.normal(0, 10)
                atmospheric_pressure = base_pressure + pressure_variation
                
                # Create records for all buildings in this location
                for _, building in city_buildings.iterrows():
                    record = {
                        'building_id': building['building_id'],
                        'timestamp': timestamp,
                        'outdoor_temperature_celsius': outdoor_temperature,
                        'outdoor_humidity_percent': outdoor_humidity,
                        'wind_speed_ms': wind_speed,
                        'wind_direction_degrees': wind_direction,
                        'solar_irradiance_wm2': solar_irradiance,
                        'precipitation_mm': precipitation,
                        'atmospheric_pressure_hpa': atmospheric_pressure
                    }
                    weather_records.append(record)
        
        return pd.DataFrame(weather_records)
    
    def _generate_occupancy_data_batch(self, buildings: pd.DataFrame,
                                     attributes: pd.DataFrame) -> pd.DataFrame:
        """Generate occupancy data for a batch of buildings."""
        
        occupancy_records = []
        
        for _, building in buildings.iterrows():
            building_id = building['building_id']
            attr = attributes[attributes['building_id'] == building_id].iloc[0]
            
            # Determine maximum occupancy based on building type and size
            floor_area = attr['gross_floor_area_m2']
            if attr['building_type'] == 'residential':
                max_occupancy = max(1, int(floor_area / 40))  # ~40 m²/person
            elif attr['building_type'] == 'commercial':
                max_occupancy = max(1, int(floor_area / 10))  # ~10 m²/person
            elif attr['building_type'] == 'institutional':
                max_occupancy = max(1, int(floor_area / 5))   # ~5 m²/person
            else:  # mixed_use
                max_occupancy = max(1, int(floor_area / 15))  # ~15 m²/person
            
            for timestamp in self.time_index:
                hour = timestamp.hour
                day_of_week = timestamp.weekday()
                
                # Calculate occupancy factor
                occupancy_factor = self._get_occupancy_factor(hour, day_of_week, attr['building_type'])
                
                # Add random variation
                occupancy_variation = np.random.normal(1.0, 0.2)
                actual_occupancy_factor = np.clip(occupancy_factor * occupancy_variation, 0, 1)
                
                # Calculate actual occupant count
                occupant_count = int(max_occupancy * actual_occupancy_factor)
                
                # Determine activity level
                if hour >= 22 or hour <= 6:
                    activity_level = 'sleeping'
                elif hour >= 7 and hour <= 9:
                    activity_level = 'high'  # Morning activity
                elif hour >= 17 and hour <= 19:
                    activity_level = 'high'  # Evening activity
                else:
                    activity_level = 'medium' if occupant_count > 0 else 'none'
                
                # Occupancy pattern classification
                if attr['building_type'] == 'residential':
                    if day_of_week < 5:  # Weekday
                        pattern = 'weekday_residential'
                    else:
                        pattern = 'weekend_residential'
                elif attr['building_type'] == 'commercial':
                    if day_of_week < 5:
                        pattern = 'weekday_commercial'
                    else:
                        pattern = 'weekend_commercial'
                else:
                    pattern = 'mixed_use'
                
                record = {
                    'building_id': building_id,
                    'timestamp': timestamp,
                    'occupant_count': occupant_count,
                    'max_occupancy': max_occupancy,
                    'occupancy_rate': actual_occupancy_factor,
                    'occupancy_pattern': pattern,
                    'activity_level': activity_level
                }
                
                occupancy_records.append(record)
        
        return pd.DataFrame(occupancy_records)
    
    def _get_heating_season_factor(self, month: int) -> float:
        """Get heating season factor (0-1) based on month."""
        heating_months = {11: 0.3, 12: 0.8, 1: 1.0, 2: 1.0, 3: 0.8, 4: 0.3}
        return heating_months.get(month, 0.0)
    
    def _get_cooling_season_factor(self, month: int) -> float:
        """Get cooling season factor (0-1) based on month."""
        cooling_months = {5: 0.2, 6: 0.6, 7: 1.0, 8: 1.0, 9: 0.6, 10: 0.2}
        return cooling_months.get(month, 0.0)
    
    def _get_occupancy_factor(self, hour: int, day_of_week: int, building_type: str) -> float:
        """Get occupancy factor (0-1) based on time and building type."""
        
        if building_type == 'residential':
            if day_of_week < 5:  # Weekday
                if 7 <= hour <= 9 or 17 <= hour <= 23:
                    return 0.8
                elif 0 <= hour <= 6:
                    return 0.9  # Sleeping
                else:
                    return 0.3  # Away during day
            else:  # Weekend
                if 8 <= hour <= 23:
                    return 0.9
                else:
                    return 0.95  # Home and sleeping
                    
        elif building_type == 'commercial':
            if day_of_week < 5:  # Weekday
                if 8 <= hour <= 18:
                    return 0.8
                else:
                    return 0.1
            else:  # Weekend
                return 0.2
                
        elif building_type == 'institutional':
            if day_of_week < 5:  # Weekday
                if 7 <= hour <= 19:
                    return 0.7
                else:
                    return 0.3
            else:  # Weekend
                if 9 <= hour <= 17:
                    return 0.4
                else:
                    return 0.2
                    
        else:  # mixed_use
            return 0.6  # Constant moderate occupancy
    
    def _get_lighting_factor(self, hour: int, month: int) -> float:
        """Get lighting factor based on hour and season."""
        # Daylight hours vary by season
        if month in [11, 12, 1, 2]:  # Winter
            daylight_start, daylight_end = 8, 16
        elif month in [3, 4, 9, 10]:  # Spring/Fall
            daylight_start, daylight_end = 7, 18
        else:  # Summer
            daylight_start, daylight_end = 6, 20
        
        if daylight_start <= hour <= daylight_end:
            return 0.3  # Reduced lighting during daylight
        else:
            return 1.0  # Full lighting during dark hours
    
    def _get_outdoor_temperature(self, month: int, latitude: float) -> float:
        """Get typical outdoor temperature for month and latitude."""
        # Simplified temperature model
        annual_mean = 10 + 5 * np.cos(np.radians(abs(latitude) - 45))
        seasonal_variation = 15 * np.cos(2 * np.pi * (month - 7) / 12)
        return annual_mean + seasonal_variation
    
    def _calculate_solar_elevation(self, latitude: float, day_of_year: int, hour: int) -> float:
        """Calculate solar elevation angle."""
        # Simplified solar position calculation
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        hour_angle = 15 * (hour - 12)
        
        elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * 
            np.cos(np.radians(hour_angle))
        )
        
        return np.degrees(elevation)