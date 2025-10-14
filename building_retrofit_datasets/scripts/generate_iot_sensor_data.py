"""
IoT Sensor Data Generator for Building Retrofit Research
Generates realistic time-series data from IoT sensors in buildings.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

np.random.seed(42)


class IoTSensorDataGenerator:
    """Generate realistic IoT sensor data for multiple buildings."""
    
    def __init__(self, num_buildings=50, days=365):
        self.num_buildings = num_buildings
        self.days = days
        self.start_date = datetime(2023, 1, 1)
        
    def generate_base_patterns(self, building_type, hours):
        """Generate base consumption patterns based on building type."""
        patterns = {
            'residential': {
                'morning_peak': (6, 9),
                'evening_peak': (18, 23),
                'base_load': 0.3,
                'peak_multiplier': 2.5
            },
            'commercial': {
                'morning_peak': (8, 12),
                'evening_peak': (13, 18),
                'base_load': 0.5,
                'peak_multiplier': 3.0
            },
            'industrial': {
                'morning_peak': (7, 19),
                'evening_peak': (19, 23),
                'base_load': 0.7,
                'peak_multiplier': 1.5
            },
            'educational': {
                'morning_peak': (8, 12),
                'evening_peak': (13, 16),
                'base_load': 0.2,
                'peak_multiplier': 3.5
            }
        }
        
        pattern = patterns.get(building_type, patterns['residential'])
        hour_of_day = hours % 24
        
        # Base pattern
        consumption = pattern['base_load']
        
        # Morning peak
        if pattern['morning_peak'][0] <= hour_of_day < pattern['morning_peak'][1]:
            consumption *= pattern['peak_multiplier']
        # Evening peak
        elif pattern['evening_peak'][0] <= hour_of_day < pattern['evening_peak'][1]:
            consumption *= pattern['peak_multiplier'] * 0.8
        
        return consumption
    
    def generate_weather_data(self, timestamp):
        """Generate realistic outdoor weather conditions."""
        # Seasonal temperature variation
        day_of_year = timestamp.timetuple().tm_yday
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation
        hour_of_day = timestamp.hour
        daily_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        temperature = base_temp + daily_variation + np.random.normal(0, 2)
        
        # Humidity inversely correlated with temperature
        humidity = 70 - (temperature - 15) * 1.5 + np.random.normal(0, 5)
        humidity = np.clip(humidity, 20, 95)
        
        # Solar radiation
        if 6 <= hour_of_day <= 18:
            solar_radiation = 800 * np.sin(np.pi * (hour_of_day - 6) / 12) * (0.8 + 0.4 * np.random.random())
        else:
            solar_radiation = 0
        
        # Wind speed
        wind_speed = np.abs(np.random.normal(3, 2))
        
        return {
            'outdoor_temperature': round(temperature, 2),
            'outdoor_humidity': round(humidity, 2),
            'solar_radiation': round(solar_radiation, 2),
            'wind_speed': round(wind_speed, 2)
        }
    
    def generate_indoor_environmental_data(self, outdoor_temp, occupancy, hvac_efficiency):
        """Generate indoor environmental quality parameters."""
        # Indoor temperature affected by outdoor temp and HVAC
        target_temp = 21
        indoor_temp = target_temp + (outdoor_temp - target_temp) * (1 - hvac_efficiency) + np.random.normal(0, 0.5)
        
        # CO2 levels based on occupancy
        base_co2 = 400
        occupancy_co2 = occupancy * np.random.uniform(30, 50)
        co2 = base_co2 + occupancy_co2 + np.random.normal(0, 20)
        
        # TVOC (Total Volatile Organic Compounds) in µg/m³
        tvoc = np.random.uniform(50, 300) + occupancy * 10
        
        # PM2.5 in µg/m³
        pm25 = np.random.uniform(5, 35) + np.random.normal(0, 5)
        
        # Indoor humidity
        indoor_humidity = 45 + np.random.normal(0, 5) + occupancy * 0.5
        indoor_humidity = np.clip(indoor_humidity, 30, 70)
        
        return {
            'indoor_temperature': round(indoor_temp, 2),
            'indoor_humidity': round(indoor_humidity, 2),
            'co2_level': round(co2, 2),
            'tvoc': round(tvoc, 2),
            'pm25': round(pm25, 2)
        }
    
    def generate_occupancy_pattern(self, building_type, timestamp):
        """Generate occupancy patterns based on building type and time."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if building_type == 'residential':
            if day_of_week < 5:  # Weekday
                if 0 <= hour < 7:
                    return np.random.uniform(0.8, 1.0)
                elif 7 <= hour < 9:
                    return np.random.uniform(0.5, 0.8)
                elif 9 <= hour < 17:
                    return np.random.uniform(0.1, 0.3)
                elif 17 <= hour < 23:
                    return np.random.uniform(0.6, 0.9)
                else:
                    return np.random.uniform(0.8, 1.0)
            else:  # Weekend
                return np.random.uniform(0.7, 1.0)
        
        elif building_type in ['commercial', 'educational']:
            if day_of_week < 5:  # Weekday
                if 8 <= hour < 18:
                    return np.random.uniform(0.6, 1.0)
                else:
                    return np.random.uniform(0.0, 0.2)
            else:  # Weekend
                return np.random.uniform(0.0, 0.1)
        
        elif building_type == 'industrial':
            if day_of_week < 5:  # Weekday
                if 6 <= hour < 22:
                    return np.random.uniform(0.7, 1.0)
                else:
                    return np.random.uniform(0.3, 0.5)
            else:
                return np.random.uniform(0.2, 0.4)
        
        return np.random.uniform(0.1, 0.3)
    
    def generate_energy_consumption(self, building_attrs, timestamp, occupancy, weather):
        """Generate realistic energy consumption data."""
        building_type = building_attrs['type']
        floor_area = building_attrs['floor_area']
        efficiency_rating = building_attrs['efficiency_rating']
        
        # Base consumption per m² (kWh)
        base_consumption_per_m2 = {
            'residential': 0.015,
            'commercial': 0.025,
            'industrial': 0.035,
            'educational': 0.020
        }.get(building_type, 0.020)
        
        # Efficiency factor (A=0.7, B=0.8, C=0.9, D=1.0, E=1.1, F=1.2, G=1.3)
        efficiency_factors = {'A': 0.7, 'B': 0.8, 'C': 0.9, 'D': 1.0, 'E': 1.1, 'F': 1.2, 'G': 1.3}
        efficiency_factor = efficiency_factors.get(efficiency_rating, 1.0)
        
        # Base consumption
        hour_of_day = timestamp.hour
        pattern = self.generate_base_patterns(building_type, hour_of_day)
        base_consumption = floor_area * base_consumption_per_m2 * pattern * efficiency_factor
        
        # Occupancy effect
        occupancy_consumption = base_consumption * occupancy * 0.3
        
        # Weather effect (heating/cooling)
        outdoor_temp = weather['outdoor_temperature']
        temp_diff = abs(21 - outdoor_temp)  # Target comfort temperature
        hvac_consumption = temp_diff * floor_area * 0.002 * efficiency_factor
        
        # Lighting (based on solar radiation and occupancy)
        solar = weather['solar_radiation']
        lighting_need = max(0, 1 - solar / 800) * occupancy
        lighting_consumption = floor_area * 0.005 * lighting_need
        
        total_consumption = base_consumption + occupancy_consumption + hvac_consumption + lighting_consumption
        
        # Add some noise
        total_consumption *= (1 + np.random.normal(0, 0.05))
        
        return {
            'total_energy_consumption_kwh': round(total_consumption, 3),
            'hvac_consumption_kwh': round(hvac_consumption, 3),
            'lighting_consumption_kwh': round(lighting_consumption, 3),
            'equipment_consumption_kwh': round(base_consumption + occupancy_consumption, 3)
        }
    
    def generate_building_profiles(self):
        """Generate building profiles with attributes."""
        building_types = ['residential', 'commercial', 'industrial', 'educational']
        efficiency_ratings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        buildings = []
        for i in range(self.num_buildings):
            building = {
                'building_id': f'BLD_{i+1:03d}',
                'type': np.random.choice(building_types),
                'floor_area': np.random.uniform(500, 5000),
                'efficiency_rating': np.random.choice(efficiency_ratings, p=[0.05, 0.1, 0.15, 0.25, 0.25, 0.15, 0.05]),
                'hvac_efficiency': np.random.uniform(0.6, 0.95)
            }
            buildings.append(building)
        
        return buildings
    
    def generate_dataset(self):
        """Generate complete IoT sensor dataset."""
        print(f"Generating IoT sensor data for {self.num_buildings} buildings over {self.days} days...")
        
        buildings = self.generate_building_profiles()
        all_data = []
        
        # Generate hourly data for each building
        for building in buildings:
            building_id = building['building_id']
            print(f"Processing {building_id} ({building['type']})...")
            
            for day in range(self.days):
                for hour in range(24):
                    timestamp = self.start_date + timedelta(days=day, hours=hour)
                    
                    # Generate all sensor readings
                    weather = self.generate_weather_data(timestamp)
                    occupancy = self.generate_occupancy_pattern(building['type'], timestamp)
                    indoor_env = self.generate_indoor_environmental_data(
                        weather['outdoor_temperature'], 
                        occupancy, 
                        building['hvac_efficiency']
                    )
                    energy = self.generate_energy_consumption(building, timestamp, occupancy, weather)
                    
                    # Combine all data
                    data_point = {
                        'timestamp': timestamp,
                        'building_id': building_id,
                        'building_type': building['type'],
                        'occupancy_ratio': round(occupancy, 3),
                        **weather,
                        **indoor_env,
                        **energy
                    }
                    
                    all_data.append(data_point)
        
        df = pd.DataFrame(all_data)
        return df, buildings


def main():
    """Main function to generate and save IoT sensor data."""
    print("=" * 80)
    print("IoT SENSOR DATA GENERATOR FOR BUILDING RETROFIT RESEARCH")
    print("=" * 80)
    
    # Generate data for 50 buildings over 1 year (hourly resolution)
    generator = IoTSensorDataGenerator(num_buildings=50, days=365)
    sensor_df, building_profiles = generator.generate_dataset()
    
    # Save datasets
    print("\nSaving datasets...")
    sensor_df.to_csv('../data/iot_sensor_data.csv', index=False)
    sensor_df.to_parquet('../data/iot_sensor_data.parquet', index=False)
    
    # Save building profiles
    profiles_df = pd.DataFrame(building_profiles)
    profiles_df.to_csv('../data/building_profiles_iot.csv', index=False)
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total records generated: {len(sensor_df):,}")
    print(f"Number of buildings: {len(building_profiles)}")
    print(f"Time period: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")
    print(f"Temporal resolution: Hourly")
    print(f"\nDataset size: {sensor_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-" * 80)
    print("DATA COLUMNS:")
    print("-" * 80)
    for col in sensor_df.columns:
        print(f"  - {col}")
    
    print("\n" + "-" * 80)
    print("SAMPLE STATISTICS:")
    print("-" * 80)
    print(sensor_df.describe())
    
    print("\n" + "-" * 80)
    print("BUILDING TYPE DISTRIBUTION:")
    print("-" * 80)
    print(sensor_df['building_type'].value_counts())
    
    print("\n✅ IoT sensor data generation complete!")
    print(f"📁 Saved to: ../data/iot_sensor_data.csv")
    print(f"📁 Saved to: ../data/iot_sensor_data.parquet")
    print(f"📁 Building profiles: ../data/building_profiles_iot.csv")


if __name__ == "__main__":
    main()
