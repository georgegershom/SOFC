#!/usr/bin/env python3
"""
IoT Building Dataset Generator for Digital Twin Framework
Generates one year of realistic building IoT sensor data with seasonal patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Set random seed for reproducibility
np.random.seed(42)

class BuildingIoTDataGenerator:
    def __init__(self, start_date='2024-01-01', duration_days=365, interval_minutes=15):
        """
        Initialize the IoT data generator
        
        Parameters:
        - start_date: Starting date for data generation
        - duration_days: Number of days to generate data for
        - interval_minutes: Data sampling interval in minutes
        """
        self.start_date = pd.to_datetime(start_date)
        self.duration_days = duration_days
        self.interval_minutes = interval_minutes
        
        # Generate timestamp index
        self.timestamps = pd.date_range(
            start=self.start_date,
            periods=int(duration_days * 24 * 60 / interval_minutes),
            freq=f'{interval_minutes}min'
        )
        
        self.building_config = {
            'total_area_sqm': 5000,
            'num_floors': 5,
            'num_zones': 12,
            'max_occupancy': 250,
            'location': {'latitude': 40.7128, 'longitude': -74.0060, 'timezone': 'America/New_York'},
            'building_type': 'Office',
            'year_built': 1995
        }
        
    def get_day_of_year(self):
        """Get day of year for seasonal patterns"""
        return self.timestamps.dayofyear
    
    def get_hour_of_day(self):
        """Get hour of day for daily patterns"""
        return self.timestamps.hour + self.timestamps.minute / 60
    
    def is_business_hours(self):
        """Check if timestamp is during business hours (8am-6pm, weekdays)"""
        is_weekday = self.timestamps.dayofweek < 5
        is_work_hours = (self.timestamps.hour >= 8) & (self.timestamps.hour < 18)
        return is_weekday & is_work_hours
    
    def seasonal_pattern(self, amplitude=1.0, phase_shift=0):
        """Generate seasonal sinusoidal pattern"""
        day_of_year = self.get_day_of_year()
        return amplitude * np.sin(2 * np.pi * (day_of_year + phase_shift) / 365)
    
    def daily_pattern(self, amplitude=1.0, peak_hour=14):
        """Generate daily sinusoidal pattern"""
        hour = self.get_hour_of_day()
        return amplitude * np.sin(2 * np.pi * (hour - 6 + peak_hour) / 24)
    
    def add_noise(self, data, noise_level=0.05):
        """Add Gaussian noise to data"""
        data_array = np.array(data)
        return data + np.random.normal(0, noise_level * np.abs(data_array).mean(), len(data))
    
    def generate_energy_consumption(self):
        """Generate whole-building and sub-metered energy consumption data"""
        print("Generating energy consumption data...")
        
        # Base loads with seasonal variation
        base_electricity = 150  # kW base load
        base_gas = 80  # kW base load
        base_water = 0.5  # m³/15min base
        
        # Seasonal patterns (heating in winter, cooling in summer)
        seasonal_hvac = self.seasonal_pattern(amplitude=100, phase_shift=-90)
        seasonal_heating = np.maximum(0, -self.seasonal_pattern(amplitude=150, phase_shift=-90))
        
        # Daily occupancy pattern
        occupancy_pattern = np.where(
            self.is_business_hours(),
            0.7 + 0.3 * self.daily_pattern(amplitude=1, peak_hour=14),
            0.2
        )
        
        # Whole-building electricity (kW)
        electricity_total = (
            base_electricity + 
            seasonal_hvac * 0.8 +
            occupancy_pattern * 120 +
            self.add_noise(np.zeros(len(self.timestamps)), 15)
        )
        electricity_total = np.maximum(electricity_total, 50)  # Minimum load
        
        # Whole-building gas (kW thermal)
        gas_total = (
            base_gas +
            seasonal_heating +
            occupancy_pattern * 20 +
            self.add_noise(np.zeros(len(self.timestamps)), 10)
        )
        gas_total = np.maximum(gas_total, 10)
        
        # Water consumption (m³ per interval)
        water_total = (
            base_water +
            occupancy_pattern * 0.8 +
            self.add_noise(np.zeros(len(self.timestamps)), 0.1)
        )
        water_total = np.maximum(water_total, 0.1)
        
        # District heating/cooling (kW)
        district_heating = np.maximum(0, seasonal_heating * 0.6 + self.add_noise(np.zeros(len(self.timestamps)), 8))
        district_cooling = np.maximum(0, -seasonal_hvac * 0.5 + self.add_noise(np.zeros(len(self.timestamps)), 8))
        
        # Sub-metered data (breakdown of electricity)
        hvac_electricity = electricity_total * (0.40 + 0.15 * np.abs(self.seasonal_pattern(amplitude=1)))
        lighting_electricity = electricity_total * (0.25 * occupancy_pattern + 0.05)
        plug_loads = electricity_total * (0.20 * occupancy_pattern + 0.08)
        other_loads = electricity_total - (hvac_electricity + lighting_electricity + plug_loads)
        
        # Create DataFrames
        whole_building_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'electricity_kw': electricity_total,
            'gas_kw': gas_total,
            'water_m3': water_total,
            'district_heating_kw': district_heating,
            'district_cooling_kw': district_cooling
        })
        
        sub_metered_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'hvac_electricity_kw': hvac_electricity,
            'lighting_electricity_kw': lighting_electricity,
            'plug_loads_kw': plug_loads,
            'other_loads_kw': other_loads
        })
        
        return whole_building_df, sub_metered_df
    
    def generate_ieq_data(self):
        """Generate Indoor Environmental Quality data for multiple zones"""
        print("Generating IEQ data...")
        
        num_zones = self.building_config['num_zones']
        
        # External temperature (seasonal pattern)
        outdoor_temp = 15 + self.seasonal_pattern(amplitude=15, phase_shift=-90)
        
        # Solar gain effect
        solar_gain = np.maximum(0, self.daily_pattern(amplitude=5, peak_hour=14))
        
        ieq_data_list = []
        
        for zone in range(1, num_zones + 1):
            # Temperature varies by zone and floor (higher floors are warmer)
            zone_offset = (zone % 5) * 0.5  # Floor effect
            indoor_temp = (
                21 +  # Base setpoint
                0.3 * (outdoor_temp - 15) +  # Outdoor influence
                solar_gain * 0.3 * (1 if zone <= 6 else 0.5) +  # Solar gain (higher on lower floors with more windows)
                self.add_noise(np.zeros(len(self.timestamps)), 0.8) +
                zone_offset
            )
            
            # Humidity (inversely related to temperature)
            relative_humidity = (
                50 -
                (indoor_temp - 21) * 2 +
                self.seasonal_pattern(amplitude=10, phase_shift=-90) +
                self.add_noise(np.zeros(len(self.timestamps)), 3)
            )
            relative_humidity = np.clip(relative_humidity, 25, 75)
            
            # CO2 levels (occupancy-driven)
            occupancy_factor = np.where(self.is_business_hours(), 0.7 + 0.3 * np.random.random(len(self.timestamps)), 0.1)
            co2_ppm = (
                400 +  # Outdoor CO2
                occupancy_factor * 600 +
                self.add_noise(np.zeros(len(self.timestamps)), 50)
            )
            co2_ppm = np.maximum(co2_ppm, 400)
            
            # Particulate Matter
            pm25 = np.maximum(5 + self.add_noise(np.zeros(len(self.timestamps)), 3), 0)
            pm10 = pm25 * 1.8 + np.maximum(3 + self.add_noise(np.zeros(len(self.timestamps)), 2), 0)
            
            # TVOCs
            tvoc_ppb = (
                50 +
                occupancy_factor * 200 +
                self.add_noise(np.zeros(len(self.timestamps)), 30)
            )
            tvoc_ppb = np.maximum(tvoc_ppb, 30)
            
            # Illuminance (lighting levels)
            natural_light = np.maximum(0, solar_gain * 150 * (1 if zone <= 6 else 0.6))
            artificial_light = np.where(self.is_business_hours(), 400 + 100 * np.random.random(len(self.timestamps)), 50)
            illuminance_lux = natural_light + artificial_light
            
            # Noise levels
            noise_db = np.where(
                self.is_business_hours(),
                45 + occupancy_factor * 15 + self.add_noise(np.zeros(len(self.timestamps)), 3),
                35 + self.add_noise(np.zeros(len(self.timestamps)), 2)
            )
            
            zone_data = pd.DataFrame({
                'timestamp': self.timestamps,
                'zone_id': f'Zone_{zone:02d}',
                'temperature_c': indoor_temp,
                'relative_humidity_pct': relative_humidity,
                'co2_ppm': co2_ppm,
                'pm25_ugm3': pm25,
                'pm10_ugm3': pm10,
                'tvoc_ppb': tvoc_ppb,
                'illuminance_lux': illuminance_lux,
                'noise_db': noise_db
            })
            
            ieq_data_list.append(zone_data)
        
        return pd.concat(ieq_data_list, ignore_index=True)
    
    def generate_occupancy_data(self):
        """Generate occupancy and usage pattern data"""
        print("Generating occupancy data...")
        
        max_occupancy = self.building_config['max_occupancy']
        
        # Base occupancy pattern
        occupancy_base = np.where(
            self.is_business_hours(),
            max_occupancy * (0.6 + 0.3 * self.daily_pattern(amplitude=1, peak_hour=14)),
            max_occupancy * 0.05  # Security/cleaning staff
        )
        
        # Add weekly variation (lower on Mondays, Fridays)
        weekday_factor = np.ones(len(self.timestamps))
        weekday_factor[self.timestamps.dayofweek == 0] = 0.85  # Monday
        weekday_factor[self.timestamps.dayofweek == 4] = 0.80  # Friday
        
        occupant_count = np.round(occupancy_base * weekday_factor + self.add_noise(np.zeros(len(self.timestamps)), 5))
        occupant_count = np.clip(occupant_count, 0, max_occupancy)
        
        # Space utilization (percentage)
        desk_utilization = (occupant_count / max_occupancy) * 100 * np.random.uniform(0.85, 1.15, len(self.timestamps))
        desk_utilization = np.clip(desk_utilization, 0, 100)
        
        meeting_room_utilization = np.where(
            self.is_business_hours(),
            np.random.uniform(40, 85, len(self.timestamps)),
            np.random.uniform(0, 10, len(self.timestamps))
        )
        
        # Window operation (weather-dependent)
        outdoor_temp = 15 + self.seasonal_pattern(amplitude=15, phase_shift=-90)
        windows_open_pct = np.where(
            (outdoor_temp > 18) & (outdoor_temp < 24) & self.is_business_hours(),
            np.random.uniform(30, 70, len(self.timestamps)),
            np.random.uniform(0, 15, len(self.timestamps))
        )
        
        # Blind operation (solar-dependent)
        solar_gain = np.maximum(0, self.daily_pattern(amplitude=1, peak_hour=14))
        blinds_closed_pct = np.where(
            solar_gain > 0.5,
            np.random.uniform(60, 90, len(self.timestamps)),
            np.random.uniform(10, 40, len(self.timestamps))
        )
        
        occupancy_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'occupant_count': occupant_count.astype(int),
            'desk_utilization_pct': desk_utilization,
            'meeting_room_utilization_pct': meeting_room_utilization,
            'windows_open_pct': windows_open_pct,
            'blinds_closed_pct': blinds_closed_pct
        })
        
        return occupancy_df
    
    def generate_weather_data(self):
        """Generate on-site weather station data"""
        print("Generating weather data...")
        
        # Ambient temperature
        ambient_temp = 15 + self.seasonal_pattern(amplitude=15, phase_shift=-90) + self.daily_pattern(amplitude=5, peak_hour=15)
        ambient_temp = self.add_noise(ambient_temp, 1.5)
        
        # Solar irradiance (W/m²)
        hour = self.get_hour_of_day()
        daytime_mask = (hour >= 6) & (hour <= 20)
        solar_base = np.where(daytime_mask, np.maximum(0, np.sin(np.pi * (hour - 6) / 14) * 1000), 0)
        
        # Seasonal variation in solar irradiance
        seasonal_solar = 1 + 0.3 * self.seasonal_pattern(amplitude=1, phase_shift=-90)
        solar_irradiance = solar_base * seasonal_solar * np.random.uniform(0.7, 1.0, len(self.timestamps))
        solar_irradiance = np.maximum(solar_irradiance, 0)
        
        # Wind speed
        wind_speed = (
            5 +  # Base wind
            2 * self.seasonal_pattern(amplitude=1, phase_shift=0) +  # Seasonal variation
            self.add_noise(np.zeros(len(self.timestamps)), 1.5)
        )
        wind_speed = np.maximum(wind_speed, 0)
        
        # Wind direction (degrees)
        wind_direction = np.random.uniform(0, 360, len(self.timestamps))
        
        # Relative humidity (inversely related to temperature)
        humidity = (
            60 -
            (ambient_temp - 15) * 1.5 +
            self.seasonal_pattern(amplitude=15, phase_shift=-90) +
            self.add_noise(np.zeros(len(self.timestamps)), 5)
        )
        humidity = np.clip(humidity, 20, 95)
        
        # Rainfall (mm per interval) - more in spring/fall
        rainfall_probability = 0.15 + 0.1 * np.abs(self.seasonal_pattern(amplitude=1, phase_shift=45))
        rainfall = np.where(
            np.random.random(len(self.timestamps)) < rainfall_probability,
            np.random.exponential(2, len(self.timestamps)),
            0
        )
        
        weather_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'solar_irradiance_wm2': solar_irradiance,
            'wind_speed_ms': wind_speed,
            'wind_direction_deg': wind_direction,
            'ambient_temperature_c': ambient_temp,
            'relative_humidity_pct': humidity,
            'rainfall_mm': rainfall
        })
        
        return weather_df
    
    def generate_hvac_data(self):
        """Generate HVAC system operation data for multiple zones"""
        print("Generating HVAC system data...")
        
        num_zones = self.building_config['num_zones']
        outdoor_temp = 15 + self.seasonal_pattern(amplitude=15, phase_shift=-90)
        
        hvac_data_list = []
        
        for zone in range(1, num_zones + 1):
            # Supply air temperature (varies with outdoor conditions)
            supply_temp = np.where(
                self.is_business_hours(),
                16 + (outdoor_temp - 15) * 0.2 + self.add_noise(np.zeros(len(self.timestamps)), 1),
                18 + self.add_noise(np.zeros(len(self.timestamps)), 0.5)
            )
            
            # Return air temperature
            return_temp = supply_temp + 8 + self.add_noise(np.zeros(len(self.timestamps)), 1.5)
            
            # Damper position (0-100%)
            damper_position = np.where(
                self.is_business_hours(),
                50 + (outdoor_temp - 15) * 2 + self.add_noise(np.zeros(len(self.timestamps)), 10),
                20 + self.add_noise(np.zeros(len(self.timestamps)), 5)
            )
            damper_position = np.clip(damper_position, 0, 100)
            
            # Fan speed (0-100%)
            fan_speed = np.where(
                self.is_business_hours(),
                60 + self.add_noise(np.zeros(len(self.timestamps)), 15),
                30 + self.add_noise(np.zeros(len(self.timestamps)), 8)
            )
            fan_speed = np.clip(fan_speed, 0, 100)
            
            # Valve position (0-100%)
            heating_valve = np.maximum(0, -(outdoor_temp - 18) * 5 + self.add_noise(np.zeros(len(self.timestamps)), 10))
            heating_valve = np.clip(heating_valve, 0, 100)
            
            cooling_valve = np.maximum(0, (outdoor_temp - 22) * 5 + self.add_noise(np.zeros(len(self.timestamps)), 10))
            cooling_valve = np.clip(cooling_valve, 0, 100)
            
            # Chiller/Boiler status
            chiller_status = (outdoor_temp > 20) & self.is_business_hours()
            boiler_status = (outdoor_temp < 15) & self.is_business_hours()
            
            # Setpoints
            heating_setpoint = np.where(
                self.is_business_hours(),
                21.0 + self.add_noise(np.zeros(len(self.timestamps)), 0.3),
                18.0
            )
            
            cooling_setpoint = np.where(
                self.is_business_hours(),
                24.0 + self.add_noise(np.zeros(len(self.timestamps)), 0.3),
                26.0
            )
            
            zone_hvac = pd.DataFrame({
                'timestamp': self.timestamps,
                'zone_id': f'Zone_{zone:02d}',
                'supply_air_temp_c': supply_temp,
                'return_air_temp_c': return_temp,
                'damper_position_pct': damper_position,
                'fan_speed_pct': fan_speed,
                'heating_valve_pct': heating_valve,
                'cooling_valve_pct': cooling_valve,
                'chiller_status': chiller_status.astype(int),
                'boiler_status': boiler_status.astype(int),
                'heating_setpoint_c': heating_setpoint,
                'cooling_setpoint_c': cooling_setpoint
            })
            
            hvac_data_list.append(zone_hvac)
        
        return pd.concat(hvac_data_list, ignore_index=True)
    
    def generate_all(self, output_dir='.'):
        """Generate all datasets and save to files"""
        print(f"Starting IoT dataset generation for {self.duration_days} days...")
        print(f"Timestamp interval: {self.interval_minutes} minutes")
        print(f"Total records per sensor: {len(self.timestamps):,}")
        print("-" * 60)
        
        # Generate all datasets
        whole_building_energy, sub_metered_energy = self.generate_energy_consumption()
        ieq_data = self.generate_ieq_data()
        occupancy_data = self.generate_occupancy_data()
        weather_data = self.generate_weather_data()
        hvac_data = self.generate_hvac_data()
        
        # Save datasets
        print("\nSaving datasets...")
        
        whole_building_energy.to_csv(f'{output_dir}/energy/whole_building_energy.csv', index=False)
        sub_metered_energy.to_csv(f'{output_dir}/energy/sub_metered_energy.csv', index=False)
        ieq_data.to_csv(f'{output_dir}/ieq/indoor_environmental_quality.csv', index=False)
        occupancy_data.to_csv(f'{output_dir}/occupancy/occupancy_usage.csv', index=False)
        weather_data.to_csv(f'{output_dir}/weather/weather_station.csv', index=False)
        hvac_data.to_csv(f'{output_dir}/hvac_systems/hvac_operation.csv', index=False)
        
        # Save building configuration
        with open(f'{output_dir}/building_config.json', 'w') as f:
            json.dump(self.building_config, f, indent=2)
        
        # Generate metadata
        metadata = {
            'dataset_info': {
                'name': 'IoT Building Digital Twin Dataset',
                'version': '1.0',
                'generated_date': datetime.now().isoformat(),
                'start_date': self.start_date.isoformat(),
                'end_date': (self.start_date + timedelta(days=self.duration_days)).isoformat(),
                'duration_days': self.duration_days,
                'sampling_interval_minutes': self.interval_minutes,
                'total_records_per_stream': len(self.timestamps)
            },
            'data_streams': {
                'energy': {
                    'whole_building': {
                        'file': 'energy/whole_building_energy.csv',
                        'records': len(whole_building_energy),
                        'metrics': ['electricity_kw', 'gas_kw', 'water_m3', 'district_heating_kw', 'district_cooling_kw']
                    },
                    'sub_metered': {
                        'file': 'energy/sub_metered_energy.csv',
                        'records': len(sub_metered_energy),
                        'metrics': ['hvac_electricity_kw', 'lighting_electricity_kw', 'plug_loads_kw', 'other_loads_kw']
                    }
                },
                'ieq': {
                    'file': 'ieq/indoor_environmental_quality.csv',
                    'records': len(ieq_data),
                    'zones': self.building_config['num_zones'],
                    'metrics': ['temperature_c', 'relative_humidity_pct', 'co2_ppm', 'pm25_ugm3', 'pm10_ugm3', 'tvoc_ppb', 'illuminance_lux', 'noise_db']
                },
                'occupancy': {
                    'file': 'occupancy/occupancy_usage.csv',
                    'records': len(occupancy_data),
                    'metrics': ['occupant_count', 'desk_utilization_pct', 'meeting_room_utilization_pct', 'windows_open_pct', 'blinds_closed_pct']
                },
                'weather': {
                    'file': 'weather/weather_station.csv',
                    'records': len(weather_data),
                    'metrics': ['solar_irradiance_wm2', 'wind_speed_ms', 'wind_direction_deg', 'ambient_temperature_c', 'relative_humidity_pct', 'rainfall_mm']
                },
                'hvac': {
                    'file': 'hvac_systems/hvac_operation.csv',
                    'records': len(hvac_data),
                    'zones': self.building_config['num_zones'],
                    'metrics': ['supply_air_temp_c', 'return_air_temp_c', 'damper_position_pct', 'fan_speed_pct', 
                               'heating_valve_pct', 'cooling_valve_pct', 'chiller_status', 'boiler_status', 
                               'heating_setpoint_c', 'cooling_setpoint_c']
                }
            },
            'building_config': self.building_config
        }
        
        with open(f'{output_dir}/dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nDataset generation completed!")
        print(f"\nDataset Summary:")
        print(f"  - Whole Building Energy: {len(whole_building_energy):,} records")
        print(f"  - Sub-Metered Energy: {len(sub_metered_energy):,} records")
        print(f"  - IEQ Data: {len(ieq_data):,} records ({self.building_config['num_zones']} zones)")
        print(f"  - Occupancy Data: {len(occupancy_data):,} records")
        print(f"  - Weather Data: {len(weather_data):,} records")
        print(f"  - HVAC Data: {len(hvac_data):,} records ({self.building_config['num_zones']} zones)")
        print(f"\nTotal dataset size: ~{(len(whole_building_energy) + len(sub_metered_energy) + len(ieq_data) + len(occupancy_data) + len(weather_data) + len(hvac_data)):,} records")
        print(f"\nAll files saved to: {output_dir}/")
        
        return metadata

if __name__ == "__main__":
    # Generate 1 year of data at 15-minute intervals
    generator = BuildingIoTDataGenerator(
        start_date='2024-01-01',
        duration_days=365,
        interval_minutes=15
    )
    
    metadata = generator.generate_all()
