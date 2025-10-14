#!/usr/bin/env python3
"""
Generate synthetic IoT sensor data for building retrofit research.
Creates realistic time-series data for energy consumption, environmental parameters,
weather conditions, and occupancy patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import random

class IoTDataGenerator:
    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
    def generate_energy_consumption(self, building_id: str, building_type: str) -> pd.DataFrame:
        """Generate energy consumption data for whole building and end-uses."""
        
        # Base consumption patterns by building type (kWh)
        base_consumption = {
            'residential': {'base': 50, 'peak': 120, 'seasonal_var': 0.3},
            'office': {'base': 200, 'peak': 400, 'seasonal_var': 0.2},
            'retail': {'base': 150, 'peak': 350, 'seasonal_var': 0.4},
            'educational': {'base': 100, 'peak': 250, 'seasonal_var': 0.5}
        }
        
        config = base_consumption.get(building_type, base_consumption['office'])
        
        # Generate base consumption with seasonal and daily patterns
        consumption = []
        for timestamp in self.timestamps:
            # Seasonal variation
            seasonal_factor = 1 + config['seasonal_var'] * np.sin(2 * np.pi * timestamp.dayofyear / 365)
            
            # Daily pattern (higher during day, lower at night)
            hour_factor = 0.3 + 0.7 * (1 + np.sin(2 * np.pi * (timestamp.hour - 6) / 24)) / 2
            
            # Weekend effect
            weekend_factor = 0.6 if timestamp.weekday() >= 5 else 1.0
            
            # Random noise
            noise = np.random.normal(1, 0.1)
            
            base_consumption_hourly = config['base'] * seasonal_factor * hour_factor * weekend_factor * noise
            consumption.append(max(0, base_consumption_hourly))
        
        # Generate end-use breakdown
        end_uses = {
            'heating': 0.35,
            'cooling': 0.25,
            'lighting': 0.15,
            'equipment': 0.15,
            'hot_water': 0.10
        }
        
        data = {'timestamp': self.timestamps, 'total_consumption': consumption}
        
        for end_use, fraction in end_uses.items():
            data[f'{end_use}_consumption'] = [c * fraction * np.random.normal(1, 0.05) for c in consumption]
        
        df = pd.DataFrame(data)
        df['building_id'] = building_id
        df['building_type'] = building_type
        
        return df
    
    def generate_environmental_data(self, building_id: str) -> pd.DataFrame:
        """Generate indoor environmental parameters."""
        
        data = {'timestamp': self.timestamps}
        
        # CO2 levels (ppm) - higher during occupied hours
        co2_base = 400
        co2_occupied = 600
        co2_levels = []
        
        for timestamp in self.timestamps:
            if 8 <= timestamp.hour <= 18 and timestamp.weekday() < 5:  # Occupied hours
                co2 = co2_occupied + np.random.normal(0, 50)
            else:
                co2 = co2_base + np.random.normal(0, 20)
            co2_levels.append(max(300, min(2000, co2)))
        
        data['co2_ppm'] = co2_levels
        
        # TVOC levels (μg/m³)
        data['tvoc_ug_m3'] = [50 + np.random.exponential(20) for _ in self.timestamps]
        
        # PM2.5 levels (μg/m³)
        data['pm25_ug_m3'] = [10 + np.random.exponential(5) for _ in self.timestamps]
        
        # Temperature (°C) - seasonal variation
        temp_data = []
        for timestamp in self.timestamps:
            seasonal_temp = 20 + 10 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365)
            daily_variation = 2 * np.sin(2 * np.pi * (timestamp.hour - 6) / 24)
            temp = seasonal_temp + daily_variation + np.random.normal(0, 1)
            temp_data.append(temp)
        
        data['temperature_c'] = temp_data
        
        # Humidity (%)
        data['humidity_percent'] = [40 + 20 * np.sin(2 * np.pi * timestamp.dayofyear / 365) + 
                                  np.random.normal(0, 5) for timestamp in self.timestamps]
        
        df = pd.DataFrame(data)
        df['building_id'] = building_id
        
        return df
    
    def generate_weather_data(self, building_id: str) -> pd.DataFrame:
        """Generate outdoor weather conditions."""
        
        data = {'timestamp': self.timestamps}
        
        # Outdoor temperature (°C)
        temp_data = []
        for timestamp in self.timestamps:
            seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365)
            daily_variation = 5 * np.sin(2 * np.pi * (timestamp.hour - 6) / 24)
            temp = seasonal_temp + daily_variation + np.random.normal(0, 2)
            temp_data.append(temp)
        
        data['outdoor_temp_c'] = temp_data
        
        # Humidity (%)
        data['outdoor_humidity_percent'] = [60 + 30 * np.sin(2 * np.pi * timestamp.dayofyear / 365) + 
                                          np.random.normal(0, 10) for timestamp in self.timestamps]
        
        # Wind speed (m/s)
        data['wind_speed_ms'] = [3 + np.random.exponential(2) for _ in self.timestamps]
        
        # Solar radiation (W/m²)
        solar_data = []
        for timestamp in self.timestamps:
            if 6 <= timestamp.hour <= 18:  # Daylight hours
                solar = 200 * np.sin(np.pi * (timestamp.hour - 6) / 12) + np.random.normal(0, 20)
            else:
                solar = 0
            solar_data.append(max(0, solar))
        
        data['solar_radiation_w_m2'] = solar_data
        
        # Precipitation (mm/h)
        data['precipitation_mmh'] = [np.random.exponential(0.5) if np.random.random() < 0.1 else 0 
                                   for _ in self.timestamps]
        
        df = pd.DataFrame(data)
        df['building_id'] = building_id
        
        return df
    
    def generate_occupancy_data(self, building_id: str, building_type: str) -> pd.DataFrame:
        """Generate occupancy patterns."""
        
        data = {'timestamp': self.timestamps}
        
        # Occupancy patterns by building type
        if building_type == 'residential':
            # Residential: higher occupancy in evenings and weekends
            occupancy = []
            for timestamp in self.timestamps:
                if timestamp.weekday() >= 5:  # Weekend
                    base_occupancy = 0.8
                else:  # Weekday
                    if 18 <= timestamp.hour <= 23 or 6 <= timestamp.hour <= 8:
                        base_occupancy = 0.9
                    else:
                        base_occupancy = 0.3
                
                occupancy.append(max(0, min(1, base_occupancy + np.random.normal(0, 0.1))))
        
        elif building_type == 'office':
            # Office: higher occupancy during business hours
            occupancy = []
            for timestamp in self.timestamps:
                if timestamp.weekday() < 5 and 8 <= timestamp.hour <= 18:
                    base_occupancy = 0.8
                else:
                    base_occupancy = 0.1
                
                occupancy.append(max(0, min(1, base_occupancy + np.random.normal(0, 0.1))))
        
        else:  # Retail, educational, etc.
            # Mixed pattern
            occupancy = []
            for timestamp in self.timestamps:
                if 9 <= timestamp.hour <= 17:
                    base_occupancy = 0.6
                else:
                    base_occupancy = 0.2
                
                occupancy.append(max(0, min(1, base_occupancy + np.random.normal(0, 0.1))))
        
        data['occupancy_ratio'] = occupancy
        data['people_count'] = [int(occ * 100) for occ in occupancy]  # Assuming max 100 people
        
        df = pd.DataFrame(data)
        df['building_id'] = building_id
        df['building_type'] = building_type
        
        return df
    
    def generate_all_iot_data(self, buildings: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Generate all IoT data for multiple buildings."""
        
        all_data = {}
        
        for building in buildings:
            building_id = building['id']
            building_type = building['type']
            
            print(f"Generating IoT data for building {building_id} ({building_type})...")
            
            # Generate each type of IoT data
            energy_data = self.generate_energy_consumption(building_id, building_type)
            env_data = self.generate_environmental_data(building_id)
            weather_data = self.generate_weather_data(building_id)
            occupancy_data = self.generate_occupancy_data(building_id, building_type)
            
            # Store data
            all_data[f'{building_id}_energy'] = energy_data
            all_data[f'{building_id}_environmental'] = env_data
            all_data[f'{building_id}_weather'] = weather_data
            all_data[f'{building_id}_occupancy'] = occupancy_data
        
        return all_data

def main():
    """Generate IoT sensor data for the building retrofit dataset."""
    
    # Sample buildings
    buildings = [
        {'id': 'B001', 'type': 'residential'},
        {'id': 'B002', 'type': 'office'},
        {'id': 'B003', 'type': 'retail'},
        {'id': 'B004', 'type': 'educational'},
        {'id': 'B005', 'type': 'residential'},
        {'id': 'B006', 'type': 'office'},
        {'id': 'B007', 'type': 'retail'},
        {'id': 'B008', 'type': 'educational'},
        {'id': 'B009', 'type': 'residential'},
        {'id': 'B010', 'type': 'office'}
    ]
    
    # Initialize generator
    generator = IoTDataGenerator()
    
    # Generate all IoT data
    print("Generating IoT sensor data...")
    iot_data = generator.generate_all_iot_data(buildings)
    
    # Save data
    output_dir = '../raw_data/iot_sensors'
    os.makedirs(output_dir, exist_ok=True)
    
    for data_name, df in iot_data.items():
        filename = f"{output_dir}/{data_name}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(df)} records")
    
    # Create summary
    summary = {
        'total_buildings': len(buildings),
        'date_range': f"{generator.start_date.date()} to {generator.end_date.date()}",
        'total_hours': len(generator.timestamps),
        'data_files': list(iot_data.keys())
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nIoT data generation complete!")
    print(f"Generated data for {len(buildings)} buildings")
    print(f"Time range: {generator.start_date.date()} to {generator.end_date.date()}")
    print(f"Total records per building: {len(generator.timestamps)}")

if __name__ == "__main__":
    main()