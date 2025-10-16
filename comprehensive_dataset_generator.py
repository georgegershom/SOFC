#!/usr/bin/env python3
"""
Comprehensive IoT Building Dataset Generator
==========================================

Complete system for generating realistic IoT and real-time monitoring data
for building digital twin applications with enhanced correlations and
seasonal variations.

This is the main orchestrator that combines all generators and implements
realistic cross-system correlations and dependencies.

Author: AI Assistant
Date: 2025-10-16
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import json
import os
import h5py
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings

# Import our custom generators
from iot_building_dataset_generator import BuildingConfig, WeatherGenerator, EnergyConsumptionGenerator
from ieq_occupancy_generators import OccupancyPatternGenerator, IndoorEnvironmentalQualityGenerator
from building_systems_generator import BuildingSystemsGenerator

warnings.filterwarnings('ignore')

class ComprehensiveDatasetGenerator:
    """Main orchestrator for generating complete building IoT dataset."""
    
    def __init__(self, config: BuildingConfig):
        self.config = config
        self.metadata = {}
        
    def generate_complete_dataset(self, start_date: str, end_date: str, 
                                freq: str = '15T') -> Dict[str, pd.DataFrame]:
        """
        Generate complete IoT dataset with all subsystems and correlations.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD HH:MM:SS' format
            end_date: End date in 'YYYY-MM-DD HH:MM:SS' format
            freq: Data frequency (default: 15-minute intervals)
            
        Returns:
            Dictionary containing all dataset components
        """
        
        print("🏢 Comprehensive IoT Building Dataset Generation")
        print("=" * 60)
        print(f"Building: {self.config.building_type}")
        print(f"Location: {self.config.latitude:.2f}°N, {self.config.longitude:.2f}°W")
        print(f"Period: {start_date} to {end_date}")
        print(f"Frequency: {freq}")
        print()
        
        # Initialize generators
        weather_gen = WeatherGenerator(self.config)
        occupancy_gen = OccupancyPatternGenerator(self.config)
        energy_gen = EnergyConsumptionGenerator(self.config)
        
        # Step 1: Generate weather data (foundation)
        print("🌤️  Step 1/6: Generating weather and external conditions...")
        weather_df = weather_gen.generate_weather_data(start_date, end_date, freq)
        print(f"   ✓ Generated {len(weather_df):,} weather data points")
        
        # Step 2: Generate occupancy patterns
        print("👥 Step 2/6: Generating occupancy and usage patterns...")
        occupancy_df = occupancy_gen.generate_occupancy_data(weather_df)
        print(f"   ✓ Generated occupancy data for {self.config.num_zones} zones")
        
        # Step 3: Generate energy consumption (depends on weather + occupancy)
        print("⚡ Step 3/6: Generating energy consumption data...")
        energy_df = energy_gen.generate_energy_data(weather_df, occupancy_df)
        print(f"   ✓ Generated energy data with {len([c for c in energy_df.columns if 'kw' in c.lower()])} energy streams")
        
        # Step 4: Generate IEQ data (depends on weather + energy + occupancy)
        print("🌡️  Step 4/6: Generating indoor environmental quality data...")
        ieq_gen = IndoorEnvironmentalQualityGenerator(self.config, weather_df, energy_df)
        ieq_df = ieq_gen.generate_ieq_data(occupancy_df)
        print(f"   ✓ Generated IEQ data for {self.config.num_zones} zones")
        
        # Step 5: Generate building systems data (depends on all previous)
        print("🔧 Step 5/6: Generating building systems operation data...")
        systems_gen = BuildingSystemsGenerator(self.config, weather_df, energy_df, occupancy_df, ieq_df)
        systems_df = systems_gen.generate_systems_data()
        print(f"   ✓ Generated systems data with {len([c for c in systems_df.columns if 'setpoint' in c.lower()])} setpoints")
        
        # Step 6: Apply cross-system correlations and enhancements
        print("🔗 Step 6/6: Applying cross-system correlations...")
        enhanced_data = self._apply_cross_system_correlations({
            'weather': weather_df,
            'occupancy': occupancy_df,
            'energy': energy_df,
            'ieq': ieq_df,
            'systems': systems_df
        })
        print("   ✓ Applied realistic correlations and dependencies")
        
        # Generate metadata
        self._generate_metadata(enhanced_data, start_date, end_date, freq)
        
        print("\n✅ Dataset generation complete!")
        self._print_dataset_summary(enhanced_data)
        
        return enhanced_data
    
    def _apply_cross_system_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply realistic correlations between different building systems."""
        
        # Create enhanced copies
        enhanced_data = {key: df.copy() for key, df in data.items()}
        
        # Apply correlations
        enhanced_data = self._apply_weather_energy_correlations(enhanced_data)
        enhanced_data = self._apply_occupancy_ieq_correlations(enhanced_data)
        enhanced_data = self._apply_systems_performance_correlations(enhanced_data)
        enhanced_data = self._apply_seasonal_variations(enhanced_data)
        enhanced_data = self._add_realistic_noise_and_faults(enhanced_data)
        
        return enhanced_data
    
    def _apply_weather_energy_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply weather-energy correlations."""
        
        weather_df = data['weather']
        energy_df = data['energy']
        
        # Solar generation correlation with irradiance (already implemented but enhance)
        solar_irradiance = weather_df['global_horizontal_irradiance_w_m2'].values
        
        # Enhanced solar generation with shading and soiling effects
        shading_factor = self._calculate_shading_effects(weather_df)
        soiling_factor = self._calculate_soiling_effects(weather_df)
        
        enhanced_solar = energy_df['solar_pv_generation_kw'].values * shading_factor * soiling_factor
        energy_df['solar_pv_generation_kw'] = np.maximum(0, enhanced_solar)
        
        # Wind cooling effect on HVAC load
        wind_speed = weather_df['wind_speed_m_s'].values
        wind_cooling_factor = 1 - np.clip(wind_speed / 20, 0, 0.1)  # Up to 10% reduction
        
        energy_df['hvac_cooling_kw'] *= wind_cooling_factor
        
        # Humidity effect on dehumidification load
        humidity = weather_df['relative_humidity_pct'].values
        dehumid_load = np.maximum(0, (humidity - 60) / 40) * 5  # Additional load when humid
        energy_df['hvac_cooling_kw'] += dehumid_load
        
        data['energy'] = energy_df
        return data
    
    def _apply_occupancy_ieq_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply occupancy-IEQ correlations."""
        
        occupancy_df = data['occupancy']
        ieq_df = data['ieq']
        
        # CO2 correlation with occupancy (enhance existing correlation)
        total_occupancy = occupancy_df['total_occupancy'].values
        
        for zone in range(1, min(6, self.config.num_zones + 1)):
            zone_occ = occupancy_df[f'zone_{zone}_occupancy'].values
            
            # Enhanced CO2 model with metabolic rate variations
            metabolic_factor = 1 + 0.2 * np.sin(2 * np.pi * np.arange(len(zone_occ)) / (24 * 4))  # Daily cycle
            co2_enhancement = zone_occ * metabolic_factor * 15  # Additional CO2
            
            ieq_df[f'zone_{zone}_co2_ppm'] += co2_enhancement
            
            # Temperature correlation with occupancy (body heat)
            temp_increase = zone_occ * 0.15  # 0.15°C per person
            ieq_df[f'zone_{zone}_air_temp_c'] += temp_increase
            
            # Humidity correlation with occupancy (respiration, perspiration)
            humidity_increase = zone_occ * 0.8  # 0.8% RH per person
            ieq_df[f'zone_{zone}_relative_humidity_pct'] += humidity_increase
        
        data['ieq'] = ieq_df
        return data
    
    def _apply_systems_performance_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply systems performance correlations."""
        
        systems_df = data['systems']
        energy_df = data['energy']
        weather_df = data['weather']
        
        # Chiller efficiency correlation with outdoor temperature
        outdoor_temp = weather_df['ambient_temperature_c'].values
        
        # Chiller COP decreases with higher outdoor temperature
        temp_effect = 1 - (outdoor_temp - 25) * 0.02  # 2% per degree above 25°C
        temp_effect = np.clip(temp_effect, 0.6, 1.2)
        
        if 'chiller_cop' in systems_df.columns:
            systems_df['chiller_cop'] *= temp_effect
        
        # Boiler efficiency correlation with load
        if 'boiler_capacity_pct' in systems_df.columns:
            load_factor = systems_df['boiler_capacity_pct'].values / 100
            efficiency_curve = 0.7 + 0.25 * load_factor - 0.1 * load_factor**2  # Efficiency curve
            systems_df['boiler_efficiency_pct'] *= efficiency_curve
        
        # Fan energy correlation with airflow
        if 'supply_fan_speed_pct' in systems_df.columns:
            fan_speed = systems_df['supply_fan_speed_pct'].values / 100
            # Fan power follows cube law
            fan_power_factor = fan_speed**3
            energy_df['hvac_fans_kw'] *= (0.3 + 0.7 * fan_power_factor)
        
        data['systems'] = systems_df
        data['energy'] = energy_df
        return data
    
    def _apply_seasonal_variations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply realistic seasonal variations to all systems."""
        
        # Get time variables
        timestamps = data['weather']['timestamp']
        month = timestamps.dt.month.values
        day_of_year = timestamps.dt.dayofyear.values
        
        # Seasonal equipment performance variations
        systems_df = data['systems']
        
        # Chiller performance seasonal variation (better in winter)
        seasonal_chiller_factor = 1 + 0.1 * np.cos(2 * np.pi * (day_of_year - 180) / 365.25)
        if 'chiller_cop' in systems_df.columns:
            systems_df['chiller_cop'] *= seasonal_chiller_factor
        
        # Boiler performance seasonal variation (better in summer when not working hard)
        seasonal_boiler_factor = 1 + 0.05 * np.cos(2 * np.pi * (day_of_year - 60) / 365.25)
        if 'boiler_efficiency_pct' in systems_df.columns:
            systems_df['boiler_efficiency_pct'] *= seasonal_boiler_factor
        
        # Seasonal occupancy patterns (already partially implemented but enhance)
        occupancy_df = data['occupancy']
        
        # Holiday effects
        holiday_reduction = np.ones(len(timestamps))
        
        # Major holidays (simplified)
        christmas_period = (month == 12) & (timestamps.dt.day >= 20)
        thanksgiving_week = (month == 11) & (timestamps.dt.day >= 22) & (timestamps.dt.day <= 28)
        summer_vacation = (month == 7) | (month == 8)
        
        holiday_reduction[christmas_period] = 0.3
        holiday_reduction[thanksgiving_week] = 0.6
        holiday_reduction[summer_vacation] = 0.8
        
        occupancy_df['total_occupancy'] *= holiday_reduction
        
        # Seasonal air quality variations
        ieq_df = data['ieq']
        
        # Pollen season effects on air quality
        pollen_season = ((month >= 4) & (month <= 6)) | ((month >= 9) & (month <= 10))
        pollen_effect = np.where(pollen_season, 1.3, 1.0)
        
        for zone in range(1, min(6, self.config.num_zones + 1)):
            if f'zone_{zone}_pm25_ug_m3' in ieq_df.columns:
                ieq_df[f'zone_{zone}_pm25_ug_m3'] *= pollen_effect
            if f'zone_{zone}_pm10_ug_m3' in ieq_df.columns:
                ieq_df[f'zone_{zone}_pm10_ug_m3'] *= pollen_effect
        
        data['systems'] = systems_df
        data['occupancy'] = occupancy_df
        data['ieq'] = ieq_df
        
        return data
    
    def _add_realistic_noise_and_faults(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add realistic sensor noise and occasional equipment faults."""
        
        n_points = len(data['weather'])
        
        # Sensor noise characteristics
        noise_levels = {
            'temperature': 0.2,  # ±0.2°C
            'humidity': 2.0,     # ±2% RH
            'pressure': 10.0,    # ±10 Pa
            'flow': 0.05,        # ±5%
            'power': 0.02,       # ±2%
            'co2': 25.0,         # ±25 ppm
            'pm': 0.1            # ±10%
        }
        
        # Apply sensor noise to all dataframes
        for df_name, df in data.items():
            for col in df.columns:
                if col == 'timestamp':
                    continue
                    
                # Determine noise type based on column name
                noise_type = 'flow'  # Default
                if 'temp' in col.lower():
                    noise_type = 'temperature'
                elif 'humidity' in col.lower() or '_rh_' in col.lower():
                    noise_type = 'humidity'
                elif 'pressure' in col.lower():
                    noise_type = 'pressure'
                elif 'kw' in col.lower() or 'power' in col.lower():
                    noise_type = 'power'
                elif 'co2' in col.lower():
                    noise_type = 'co2'
                elif 'pm' in col.lower():
                    noise_type = 'pm'
                
                # Apply noise
                noise_std = noise_levels[noise_type]
                if noise_type == 'pm':
                    # Multiplicative noise for PM
                    noise = np.random.lognormal(0, noise_std, len(df))
                    df[col] *= noise
                else:
                    # Additive noise for others
                    noise = np.random.normal(0, noise_std, len(df))
                    df[col] += noise
        
        # Add occasional equipment faults
        data = self._add_equipment_faults(data)
        
        return data
    
    def _add_equipment_faults(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add realistic equipment faults and maintenance events."""
        
        systems_df = data['systems']
        energy_df = data['energy']
        n_points = len(systems_df)
        
        # Chiller fault simulation (rare events)
        if 'chiller_status' in systems_df.columns:
            fault_probability = 0.0001  # 0.01% per time step
            chiller_faults = np.random.random(n_points) < fault_probability
            
            for i in np.where(chiller_faults)[0]:
                # Fault duration: 4-24 hours
                fault_duration = np.random.randint(16, 96)  # 15-min intervals
                end_idx = min(i + fault_duration, n_points)
                
                # During fault: chiller off, backup cooling
                systems_df.loc[i:end_idx, 'chiller_status'] = 0
                systems_df.loc[i:end_idx, 'chiller_capacity_pct'] = 0
                
                # Increased energy consumption (backup systems)
                energy_df.loc[i:end_idx, 'hvac_cooling_kw'] *= 1.5
        
        # Filter loading simulation
        if 'ahu_1_filter_pressure_drop_pa' in systems_df.columns:
            # Gradual filter loading over time
            days_elapsed = (systems_df.index / (24 * 4))  # Days since start
            filter_loading = 1 + (days_elapsed % 90) / 90 * 0.8  # 80% increase over 90 days
            
            for ahu in range(1, self.config.num_ahu + 1):
                col_name = f'ahu_{ahu}_filter_pressure_drop_pa'
                if col_name in systems_df.columns:
                    systems_df[col_name] *= filter_loading
        
        # Sensor drift simulation
        if 'zone_1_air_temp_c' in data['ieq'].columns:
            # Gradual sensor drift over time
            drift_rate = 0.001  # °C per day
            time_days = np.arange(n_points) / (24 * 4)
            
            for zone in range(1, min(6, self.config.num_zones + 1)):
                col_name = f'zone_{zone}_air_temp_c'
                if col_name in data['ieq'].columns:
                    sensor_drift = time_days * drift_rate * np.random.uniform(-1, 1)
                    data['ieq'][col_name] += sensor_drift
        
        data['systems'] = systems_df
        data['energy'] = energy_df
        
        return data
    
    def _calculate_shading_effects(self, weather_df: pd.DataFrame) -> np.ndarray:
        """Calculate shading effects on solar panels."""
        
        solar_elevation = weather_df['solar_elevation_deg'].values
        solar_azimuth = weather_df['solar_azimuth_deg'].values
        
        # Simplified shading model
        # Assume some shading from nearby buildings/trees
        shading_factor = np.ones_like(solar_elevation)
        
        # Morning shading (east side)
        morning_shading = (solar_azimuth < 120) & (solar_elevation < 30)
        shading_factor[morning_shading] *= 0.7
        
        # Evening shading (west side)
        evening_shading = (solar_azimuth > 240) & (solar_elevation < 30)
        shading_factor[evening_shading] *= 0.8
        
        return shading_factor
    
    def _calculate_soiling_effects(self, weather_df: pd.DataFrame) -> np.ndarray:
        """Calculate soiling effects on solar panels."""
        
        rainfall = weather_df['rainfall_mm_h'].values
        n_points = len(rainfall)
        
        # Soiling accumulates over time, rain cleans panels
        soiling_factor = np.ones(n_points)
        current_soiling = 1.0
        
        daily_soiling_rate = 0.002  # 0.2% per day
        cleaning_threshold = 2.0  # mm/h rain cleans panels
        
        for i in range(n_points):
            # Accumulate soiling
            current_soiling -= daily_soiling_rate / (24 * 4)  # Per 15-min interval
            
            # Rain cleaning
            if rainfall[i] > cleaning_threshold:
                current_soiling = min(1.0, current_soiling + 0.1)  # Partial cleaning
            
            # Limit soiling effect
            current_soiling = max(0.7, current_soiling)  # Maximum 30% loss
            soiling_factor[i] = current_soiling
        
        return soiling_factor
    
    def _generate_metadata(self, data: Dict[str, pd.DataFrame], 
                          start_date: str, end_date: str, freq: str):
        """Generate comprehensive metadata for the dataset."""
        
        self.metadata = {
            'dataset_info': {
                'title': 'IoT Building Dataset for Digital Twin Framework',
                'description': 'Comprehensive IoT and real-time monitoring data for building retrofit optimization',
                'version': '1.0.0',
                'created_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'frequency': freq,
                'total_records': len(data['weather']),
                'duration_days': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            },
            'building_info': asdict(self.config),
            'data_sources': {
                'weather': {
                    'description': 'On-site weather station data',
                    'parameters': list(data['weather'].columns),
                    'units': {
                        'ambient_temperature_c': '°C',
                        'relative_humidity_pct': '%',
                        'global_horizontal_irradiance_w_m2': 'W/m²',
                        'wind_speed_m_s': 'm/s',
                        'rainfall_mm_h': 'mm/h',
                        'atmospheric_pressure_pa': 'Pa'
                    }
                },
                'energy': {
                    'description': 'Whole-building and sub-metered energy consumption',
                    'parameters': [col for col in data['energy'].columns if col != 'timestamp'],
                    'units': {col: 'kW' if 'kw' in col.lower() else 'L/min' if 'l_min' in col.lower() else 'kWh/m²' 
                             for col in data['energy'].columns if col != 'timestamp'}
                },
                'occupancy': {
                    'description': 'Occupancy patterns and space utilization',
                    'parameters': [col for col in data['occupancy'].columns if col != 'timestamp'],
                    'units': {col: 'people' if 'occupancy' in col and 'pct' not in col else 
                             '%' if 'pct' in col else 'count' if 'count' in col else 'devices'
                             for col in data['occupancy'].columns if col != 'timestamp'}
                },
                'ieq': {
                    'description': 'Indoor Environmental Quality measurements',
                    'parameters': [col for col in data['ieq'].columns if col != 'timestamp'],
                    'units': {col: '°C' if '_temp_c' in col else 
                             '%' if '_humidity_pct' in col or '_factor' in col else
                             'ppm' if '_co2_ppm' in col else
                             'μg/m³' if '_ug_m3' in col else
                             'ppb' if '_ppb' in col else
                             'lux' if '_lux' in col else
                             'dB(A)' if '_db' in col else 'dimensionless'
                             for col in data['ieq'].columns if col != 'timestamp'}
                },
                'systems': {
                    'description': 'Building systems operation and control',
                    'parameters': [col for col in data['systems'].columns if col != 'timestamp'],
                    'units': {col: '°C' if '_temp_' in col or 'setpoint_c' in col else
                             '%' if '_pct' in col else
                             'Pa' if '_pa' in col else
                             'Hz' if '_hz' in col else
                             'kW' if '_kw' in col else
                             'kW/m²' if '_kw_m2' in col else 'dimensionless'
                             for col in data['systems'].columns if col != 'timestamp'}
                }
            },
            'data_quality': {
                'completeness': {df_name: f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%" 
                               for df_name, df in data.items()},
                'sensor_accuracy': {
                    'temperature': '±0.2°C',
                    'humidity': '±2% RH',
                    'pressure': '±10 Pa',
                    'energy': '±2%',
                    'co2': '±25 ppm',
                    'particulate_matter': '±10%'
                }
            },
            'correlations_applied': [
                'Weather-energy correlations (solar, wind, humidity effects)',
                'Occupancy-IEQ correlations (CO2, temperature, humidity)',
                'Systems performance correlations (efficiency curves)',
                'Seasonal variations (equipment performance, occupancy patterns)',
                'Realistic sensor noise and equipment faults'
            ]
        }
    
    def _print_dataset_summary(self, data: Dict[str, pd.DataFrame]):
        """Print a comprehensive summary of the generated dataset."""
        
        print("\n📊 DATASET SUMMARY")
        print("=" * 60)
        
        total_points = len(data['weather'])
        total_parameters = sum(len(df.columns) - 1 for df in data.values())  # -1 for timestamp
        
        print(f"📈 Total Data Points: {total_points:,}")
        print(f"📊 Total Parameters: {total_parameters:,}")
        print(f"💾 Estimated Size: {(total_points * total_parameters * 8 / 1024 / 1024):.1f} MB")
        print()
        
        for df_name, df in data.items():
            print(f"{df_name.upper()}: {len(df.columns)-1} parameters, {len(df):,} records")
        
        print("\n🔍 KEY METRICS:")
        
        # Weather summary
        weather_df = data['weather']
        print(f"🌡️  Temperature Range: {weather_df['ambient_temperature_c'].min():.1f}°C to {weather_df['ambient_temperature_c'].max():.1f}°C")
        print(f"☀️  Peak Solar Irradiance: {weather_df['global_horizontal_irradiance_w_m2'].max():.0f} W/m²")
        
        # Energy summary
        energy_df = data['energy']
        print(f"⚡ Peak Electricity Demand: {energy_df['total_electricity_kw'].max():.1f} kW")
        print(f"🔥 Peak Gas Consumption: {energy_df['total_gas_kw'].max():.1f} kW")
        if 'solar_pv_generation_kw' in energy_df.columns:
            print(f"☀️  Peak Solar Generation: {energy_df['solar_pv_generation_kw'].max():.1f} kW")
        
        # Occupancy summary
        occupancy_df = data['occupancy']
        print(f"👥 Peak Occupancy: {occupancy_df['total_occupancy'].max()} people")
        print(f"🏢 Average Utilization: {occupancy_df['building_utilization_pct'].mean():.1f}%")
        
        # IEQ summary
        ieq_df = data['ieq']
        print(f"🌡️  Indoor Temp Range: {ieq_df['building_avg_temp_c'].min():.1f}°C to {ieq_df['building_avg_temp_c'].max():.1f}°C")
        print(f"💨 CO2 Range: {ieq_df['building_avg_co2_ppm'].min():.0f} to {ieq_df['building_avg_co2_ppm'].max():.0f} ppm")

def main():
    """Main function to generate the complete IoT building dataset."""
    
    # Configuration
    config = BuildingConfig(
        building_type="Commercial Office Building",
        floor_area=5000.0,  # m²
        num_floors=5,
        num_zones=20,
        occupancy_capacity=250,
        latitude=40.7128,   # New York City
        longitude=-74.0060,
        timezone="America/New_York",
        has_solar_panels=True,
        solar_capacity=100.0,  # kW
        has_energy_storage=True,
        battery_capacity=200.0  # kWh
    )
    
    # Initialize generator
    generator = ComprehensiveDatasetGenerator(config)
    
    # Generate one year of data
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-12-31 23:45:00"
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(start_date, end_date)
    
    return dataset, generator

if __name__ == "__main__":
    complete_dataset, generator = main()