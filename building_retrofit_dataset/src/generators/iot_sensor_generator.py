"""
IoT Sensor Data Generator
Generates realistic time-series data for building IoT sensors including:
- Energy consumption (whole building & end-use)
- Indoor environmental quality (IEQ) parameters
- Outdoor weather conditions
- Occupancy patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class IoTSensorDataGenerator:
    def __init__(self, building_id: str, start_date: datetime, end_date: datetime, 
                 sampling_rate_minutes: int = 15):
        self.building_id = building_id
        self.start_date = start_date
        self.end_date = end_date
        self.sampling_rate = sampling_rate_minutes
        self.timestamps = pd.date_range(start=start_date, end=end_date, 
                                       freq=f'{sampling_rate_minutes}min')
        
    def generate_energy_consumption(self, building_type: str = 'office', 
                                   floor_area: float = 5000) -> pd.DataFrame:
        """Generate energy consumption data with realistic patterns"""
        n_points = len(self.timestamps)
        
        # Base load varies by building type
        base_loads = {
            'office': 0.5,
            'residential': 0.3,
            'retail': 0.7,
            'educational': 0.4,
            'healthcare': 0.8
        }
        base_load = base_loads.get(building_type, 0.5) * floor_area / 100
        
        # Create daily and weekly patterns
        hourly_pattern = self._create_hourly_pattern(building_type)
        weekly_pattern = self._create_weekly_pattern(building_type)
        
        # Generate consumption data
        energy_data = []
        for ts in self.timestamps:
            hour_factor = hourly_pattern[ts.hour]
            day_factor = weekly_pattern[ts.dayofweek]
            seasonal_factor = self._seasonal_factor(ts.month)
            
            # Total consumption with noise
            total = base_load * hour_factor * day_factor * seasonal_factor
            noise = np.random.normal(0, total * 0.05)
            total_consumption = max(0, total + noise)
            
            # End-use breakdown (approximate percentages)
            hvac = total_consumption * (0.4 + np.random.uniform(-0.05, 0.05))
            lighting = total_consumption * (0.25 + np.random.uniform(-0.03, 0.03))
            equipment = total_consumption * (0.25 + np.random.uniform(-0.03, 0.03))
            other = total_consumption - hvac - lighting - equipment
            
            energy_data.append({
                'timestamp': ts,
                'building_id': self.building_id,
                'total_consumption_kw': round(total_consumption, 2),
                'hvac_consumption_kw': round(hvac, 2),
                'lighting_consumption_kw': round(lighting, 2),
                'equipment_consumption_kw': round(equipment, 2),
                'other_consumption_kw': round(other, 2)
            })
            
        return pd.DataFrame(energy_data)
    
    def generate_ieq_data(self) -> pd.DataFrame:
        """Generate Indoor Environmental Quality data"""
        ieq_data = []
        
        for ts in self.timestamps:
            # Temperature (Celsius) - varies with time of day and season
            base_temp = 22 + self._seasonal_factor(ts.month) * 2
            temp_variation = np.sin(2 * np.pi * ts.hour / 24) * 2
            temperature = base_temp + temp_variation + np.random.normal(0, 0.5)
            
            # Relative Humidity (%)
            base_humidity = 45 + self._seasonal_factor(ts.month) * 10
            humidity = base_humidity + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 80)
            
            # CO2 (ppm) - higher during occupied hours
            occupancy_factor = self._occupancy_pattern(ts.hour, ts.dayofweek)
            co2 = 400 + occupancy_factor * 600 + np.random.normal(0, 50)
            
            # TVOC (μg/m³) - Total Volatile Organic Compounds
            tvoc = 200 + occupancy_factor * 300 + np.random.normal(0, 30)
            
            # PM2.5 (μg/m³) - Particulate Matter
            pm25 = 10 + np.random.exponential(5) + occupancy_factor * 5
            
            ieq_data.append({
                'timestamp': ts,
                'building_id': self.building_id,
                'temperature_c': round(temperature, 1),
                'relative_humidity_pct': round(humidity, 1),
                'co2_ppm': round(co2),
                'tvoc_ugm3': round(tvoc),
                'pm25_ugm3': round(pm25, 1)
            })
            
        return pd.DataFrame(ieq_data)
    
    def generate_weather_data(self, location: str = 'temperate') -> pd.DataFrame:
        """Generate outdoor weather conditions"""
        weather_data = []
        
        # Climate presets
        climates = {
            'temperate': {'temp_range': (5, 25), 'humidity_range': (40, 70)},
            'tropical': {'temp_range': (20, 35), 'humidity_range': (60, 90)},
            'arid': {'temp_range': (10, 40), 'humidity_range': (10, 40)},
            'cold': {'temp_range': (-10, 15), 'humidity_range': (30, 60)}
        }
        
        climate = climates.get(location, climates['temperate'])
        
        for ts in self.timestamps:
            # Outdoor temperature
            seasonal_temp = self._seasonal_temperature(ts.month, climate['temp_range'])
            daily_variation = np.sin(2 * np.pi * (ts.hour - 6) / 24) * 5
            outdoor_temp = seasonal_temp + daily_variation + np.random.normal(0, 2)
            
            # Outdoor humidity
            outdoor_humidity = np.random.uniform(*climate['humidity_range'])
            
            # Solar radiation (W/m²)
            if 6 <= ts.hour <= 18:
                solar_rad = 800 * np.sin(np.pi * (ts.hour - 6) / 12) * \
                           (1 - np.random.uniform(0, 0.3))  # Cloud factor
            else:
                solar_rad = 0
                
            # Wind speed (m/s)
            wind_speed = np.random.gamma(2, 2)
            
            # Precipitation (mm/hr)
            if np.random.random() < 0.1:  # 10% chance of rain
                precipitation = np.random.exponential(2)
            else:
                precipitation = 0
                
            weather_data.append({
                'timestamp': ts,
                'location_id': f'{self.building_id}_outdoor',
                'outdoor_temperature_c': round(outdoor_temp, 1),
                'outdoor_humidity_pct': round(outdoor_humidity, 1),
                'solar_radiation_wm2': round(solar_rad, 1),
                'wind_speed_ms': round(wind_speed, 1),
                'precipitation_mm': round(precipitation, 2)
            })
            
        return pd.DataFrame(weather_data)
    
    def generate_occupancy_data(self, max_occupancy: int = 200) -> pd.DataFrame:
        """Generate occupancy patterns"""
        occupancy_data = []
        
        for ts in self.timestamps:
            occupancy_rate = self._occupancy_pattern(ts.hour, ts.dayofweek)
            current_occupancy = int(max_occupancy * occupancy_rate + 
                                   np.random.normal(0, max_occupancy * 0.05))
            current_occupancy = np.clip(current_occupancy, 0, max_occupancy)
            
            occupancy_data.append({
                'timestamp': ts,
                'building_id': self.building_id,
                'occupancy_count': current_occupancy,
                'occupancy_rate': round(current_occupancy / max_occupancy, 3),
                'max_occupancy': max_occupancy
            })
            
        return pd.DataFrame(occupancy_data)
    
    def _create_hourly_pattern(self, building_type: str) -> np.ndarray:
        """Create hourly energy consumption pattern"""
        patterns = {
            'office': [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0, 
                      1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4,
                      0.3, 0.3, 0.3, 0.3],
            'residential': [0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 0.6, 0.5,
                           0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, 0.9, 1.0, 1.0,
                           0.9, 0.7, 0.6, 0.5],
            'retail': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7,
                      0.5, 0.3, 0.2, 0.2]
        }
        return np.array(patterns.get(building_type, patterns['office']))
    
    def _create_weekly_pattern(self, building_type: str) -> np.ndarray:
        """Create weekly energy consumption pattern"""
        patterns = {
            'office': [1.0, 1.0, 1.0, 1.0, 0.9, 0.3, 0.2],  # Mon-Sun
            'residential': [0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0],
            'retail': [0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 0.9]
        }
        return np.array(patterns.get(building_type, patterns['office']))
    
    def _seasonal_factor(self, month: int) -> float:
        """Calculate seasonal adjustment factor"""
        # Peak in winter (1) and summer (7), low in spring/fall
        return 1.0 + 0.3 * np.cos(2 * np.pi * (month - 1) / 12)
    
    def _seasonal_temperature(self, month: int, temp_range: Tuple[float, float]) -> float:
        """Calculate seasonal temperature"""
        min_temp, max_temp = temp_range
        # Sinusoidal variation throughout the year
        seasonal_factor = (1 + np.sin(2 * np.pi * (month - 3) / 12)) / 2
        return min_temp + (max_temp - min_temp) * seasonal_factor
    
    def _occupancy_pattern(self, hour: int, day_of_week: int) -> float:
        """Calculate occupancy rate based on hour and day"""
        if day_of_week >= 5:  # Weekend
            return 0.1 if 8 <= hour <= 17 else 0.05
        else:  # Weekday
            if hour < 6 or hour > 20:
                return 0.05
            elif 6 <= hour < 8:
                return 0.2 + (hour - 6) * 0.3
            elif 8 <= hour < 12:
                return 0.8 + np.random.uniform(-0.1, 0.1)
            elif 12 <= hour < 13:
                return 0.6
            elif 13 <= hour < 17:
                return 0.8 + np.random.uniform(-0.1, 0.1)
            elif 17 <= hour < 19:
                return 0.8 - (hour - 17) * 0.3
            else:
                return 0.2
    
    def generate_all_sensor_data(self, building_type: str = 'office', 
                                floor_area: float = 5000,
                                max_occupancy: int = 200,
                                location: str = 'temperate') -> Dict[str, pd.DataFrame]:
        """Generate all IoT sensor data categories"""
        return {
            'energy': self.generate_energy_consumption(building_type, floor_area),
            'ieq': self.generate_ieq_data(),
            'weather': self.generate_weather_data(location),
            'occupancy': self.generate_occupancy_data(max_occupancy)
        }


def generate_iot_dataset_for_buildings(building_ids: List[str], 
                                      start_date: str = '2023-01-01',
                                      end_date: str = '2024-01-01',
                                      sampling_rate_minutes: int = 15) -> Dict[str, pd.DataFrame]:
    """Generate IoT sensor data for multiple buildings"""
    
    all_data = {
        'energy': [],
        'ieq': [],
        'weather': [],
        'occupancy': []
    }
    
    building_types = ['office', 'residential', 'retail', 'educational', 'healthcare']
    locations = ['temperate', 'tropical', 'arid', 'cold']
    
    for building_id in building_ids:
        print(f"Generating IoT data for building {building_id}...")
        
        # Random building characteristics
        building_type = random.choice(building_types)
        floor_area = random.uniform(1000, 20000)
        max_occupancy = int(floor_area / 25)  # Approximately 25 m² per person
        location = random.choice(locations)
        
        # Generate data
        generator = IoTSensorDataGenerator(
            building_id=building_id,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            sampling_rate_minutes=sampling_rate_minutes
        )
        
        sensor_data = generator.generate_all_sensor_data(
            building_type=building_type,
            floor_area=floor_area,
            max_occupancy=max_occupancy,
            location=location
        )
        
        # Append to combined datasets
        for key in all_data:
            all_data[key].append(sensor_data[key])
    
    # Concatenate all building data
    combined_data = {}
    for key in all_data:
        combined_data[key] = pd.concat(all_data[key], ignore_index=True)
    
    return combined_data