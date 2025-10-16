#!/usr/bin/env python3
"""
IoT Building Dataset Generator for Digital Twin Framework
========================================================

A comprehensive system for generating realistic IoT and real-time monitoring data
for building retrofit optimization using Deep Reinforcement Learning.

This generator creates a minimum one-year dataset capturing seasonal variations
across all critical building systems and environmental parameters.

Author: AI Assistant
Date: 2025-10-16
Topic: Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import signal
from datetime import datetime, timedelta
import pytz
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import h5py

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class BuildingConfig:
    """Configuration parameters for the building being simulated."""
    
    # Building characteristics
    building_type: str = "Commercial Office"
    floor_area: float = 5000.0  # m²
    num_floors: int = 5
    num_zones: int = 20
    occupancy_capacity: int = 250
    
    # Geographic location (Default: New York City)
    latitude: float = 40.7128
    longitude: float = -74.0060
    timezone: str = "America/New_York"
    elevation: float = 10.0  # meters
    
    # Building envelope
    window_to_wall_ratio: float = 0.4
    building_orientation: float = 0.0  # degrees from north
    
    # HVAC system characteristics
    hvac_type: str = "VAV with Reheat"
    chiller_capacity: float = 500.0  # kW
    boiler_capacity: float = 300.0  # kW
    num_ahu: int = 4
    
    # Energy systems
    has_solar_panels: bool = True
    solar_capacity: float = 100.0  # kW
    has_energy_storage: bool = True
    battery_capacity: float = 200.0  # kWh

class WeatherGenerator:
    """Generates realistic weather data with seasonal variations."""
    
    def __init__(self, config: BuildingConfig):
        self.config = config
        self.tz = pytz.timezone(config.timezone)
        
    def generate_weather_data(self, start_date: str, end_date: str, 
                            freq: str = '15T') -> pd.DataFrame:
        """
        Generate comprehensive weather data including:
        - Solar irradiance (global horizontal, direct normal, diffuse)
        - Wind speed and direction
        - Ambient temperature and humidity
        - Rainfall
        - Atmospheric pressure
        """
        
        # Create datetime index
        date_range = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=freq, 
            tz=self.tz
        )
        
        n_points = len(date_range)
        
        # Day of year for seasonal calculations
        day_of_year = date_range.dayofyear
        hour_of_day = date_range.hour + date_range.minute / 60.0
        
        # Generate base temperature with seasonal variation
        temp_annual_mean = 15.0  # °C
        temp_annual_amplitude = 20.0  # °C seasonal swing
        temp_daily_amplitude = 8.0  # °C daily swing
        
        # Seasonal temperature pattern
        seasonal_temp = temp_annual_mean + temp_annual_amplitude * np.cos(
            2 * np.pi * (day_of_year - 15) / 365.25
        )
        
        # Daily temperature pattern
        daily_temp = temp_daily_amplitude * np.cos(
            2 * np.pi * (hour_of_day - 14) / 24
        )
        
        # Add weather noise and heat waves/cold snaps
        weather_noise = np.random.normal(0, 2, n_points)
        extreme_events = self._generate_extreme_weather_events(date_range)
        
        ambient_temp = seasonal_temp + daily_temp + weather_noise + extreme_events
        
        # Generate relative humidity (inversely correlated with temperature)
        base_humidity = 60.0  # %
        humidity_temp_correlation = -0.5 * (ambient_temp - temp_annual_mean)
        humidity_noise = np.random.normal(0, 10, n_points)
        relative_humidity = np.clip(
            base_humidity + humidity_temp_correlation + humidity_noise,
            20, 95
        )
        
        # Generate solar irradiance
        solar_data = self._generate_solar_irradiance(date_range, ambient_temp)
        
        # Generate wind data
        wind_speed, wind_direction = self._generate_wind_data(date_range, ambient_temp)
        
        # Generate precipitation
        rainfall = self._generate_rainfall(date_range)
        
        # Generate atmospheric pressure
        pressure = self._generate_atmospheric_pressure(date_range, ambient_temp)
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({
            'timestamp': date_range,
            'ambient_temperature_c': ambient_temp,
            'relative_humidity_pct': relative_humidity,
            'global_horizontal_irradiance_w_m2': solar_data['ghi'],
            'direct_normal_irradiance_w_m2': solar_data['dni'],
            'diffuse_horizontal_irradiance_w_m2': solar_data['dhi'],
            'wind_speed_m_s': wind_speed,
            'wind_direction_deg': wind_direction,
            'rainfall_mm_h': rainfall,
            'atmospheric_pressure_pa': pressure,
            'dew_point_c': self._calculate_dew_point(ambient_temp, relative_humidity),
            'wet_bulb_temp_c': self._calculate_wet_bulb_temp(ambient_temp, relative_humidity),
            'solar_elevation_deg': solar_data['elevation'],
            'solar_azimuth_deg': solar_data['azimuth']
        })
        
        return weather_df
    
    def _generate_extreme_weather_events(self, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Generate extreme weather events like heat waves and cold snaps."""
        n_points = len(date_range)
        extreme_events = np.zeros(n_points)
        
        # Heat waves (summer)
        summer_mask = (date_range.month >= 6) & (date_range.month <= 8)
        heat_wave_prob = 0.001  # 0.1% chance per time step
        heat_wave_locations = np.random.random(n_points) < heat_wave_prob
        heat_wave_locations &= summer_mask
        
        for i in np.where(heat_wave_locations)[0]:
            duration = np.random.randint(24, 120)  # 1-5 days
            end_idx = min(i + duration, n_points)
            intensity = np.random.uniform(5, 15)  # 5-15°C above normal
            extreme_events[i:end_idx] += intensity * np.exp(-np.arange(end_idx - i) / 48)
        
        # Cold snaps (winter)
        winter_mask = (date_range.month <= 2) | (date_range.month >= 11)
        cold_snap_prob = 0.001
        cold_snap_locations = np.random.random(n_points) < cold_snap_prob
        cold_snap_locations &= winter_mask
        
        for i in np.where(cold_snap_locations)[0]:
            duration = np.random.randint(24, 96)  # 1-4 days
            end_idx = min(i + duration, n_points)
            intensity = np.random.uniform(-10, -20)  # 10-20°C below normal
            extreme_events[i:end_idx] += intensity * np.exp(-np.arange(end_idx - i) / 36)
        
        return extreme_events
    
    def _generate_solar_irradiance(self, date_range: pd.DatetimeIndex, 
                                 temp: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate realistic solar irradiance data."""
        n_points = len(date_range)
        
        # Calculate solar position
        solar_elevation = np.zeros(n_points)
        solar_azimuth = np.zeros(n_points)
        
        for i, dt in enumerate(date_range):
            # Simplified solar position calculation
            day_of_year = dt.dayofyear
            hour_angle = 15 * (dt.hour + dt.minute/60 - 12)  # degrees
            
            # Solar declination
            declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
            
            # Solar elevation
            elevation = np.arcsin(
                np.sin(np.radians(declination)) * np.sin(np.radians(self.config.latitude)) +
                np.cos(np.radians(declination)) * np.cos(np.radians(self.config.latitude)) *
                np.cos(np.radians(hour_angle))
            )
            solar_elevation[i] = np.degrees(elevation)
            
            # Solar azimuth (simplified)
            azimuth = np.degrees(np.arctan2(
                np.sin(np.radians(hour_angle)),
                np.cos(np.radians(hour_angle)) * np.sin(np.radians(self.config.latitude)) -
                np.tan(np.radians(declination)) * np.cos(np.radians(self.config.latitude))
            ))
            solar_azimuth[i] = azimuth % 360
        
        # Generate clear sky irradiance
        clear_sky_ghi = np.maximum(0, 1000 * np.sin(np.radians(solar_elevation)))
        
        # Add cloud effects
        cloud_cover = self._generate_cloud_cover(date_range, temp)
        cloud_factor = 1 - 0.75 * cloud_cover
        
        # Global horizontal irradiance
        ghi = clear_sky_ghi * cloud_factor
        
        # Direct normal irradiance (higher reduction from clouds)
        dni = ghi * (1 + np.sin(np.radians(solar_elevation))) * (1 - 0.9 * cloud_cover)
        dni = np.maximum(0, dni)
        
        # Diffuse horizontal irradiance
        dhi = ghi - dni * np.sin(np.radians(np.maximum(0, solar_elevation)))
        dhi = np.maximum(0, dhi)
        
        return {
            'ghi': ghi,
            'dni': dni,
            'dhi': dhi,
            'elevation': solar_elevation,
            'azimuth': solar_azimuth
        }
    
    def _generate_cloud_cover(self, date_range: pd.DatetimeIndex, 
                            temp: np.ndarray) -> np.ndarray:
        """Generate realistic cloud cover patterns."""
        n_points = len(date_range)
        
        # Base cloud cover with seasonal variation
        day_of_year = date_range.dayofyear
        seasonal_clouds = 0.4 + 0.2 * np.cos(2 * np.pi * (day_of_year - 60) / 365.25)
        
        # Add weather system effects
        weather_systems = np.zeros(n_points)
        system_prob = 0.002  # Probability of weather system
        
        for i in range(n_points):
            if np.random.random() < system_prob:
                duration = np.random.randint(12, 72)  # 3-18 hours
                end_idx = min(i + duration, n_points)
                intensity = np.random.uniform(0.3, 0.8)
                weather_systems[i:end_idx] = intensity
        
        # Smooth transitions
        weather_systems = signal.savgol_filter(weather_systems, 5, 2)
        
        # Combine effects
        cloud_cover = np.clip(seasonal_clouds + weather_systems, 0, 1)
        
        return cloud_cover
    
    def _generate_wind_data(self, date_range: pd.DatetimeIndex, 
                          temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate wind speed and direction data."""
        n_points = len(date_range)
        
        # Base wind speed (higher in winter, lower in summer)
        day_of_year = date_range.dayofyear
        seasonal_wind = 4.0 + 2.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
        
        # Daily variation (higher during day)
        hour_of_day = date_range.hour + date_range.minute / 60.0
        daily_wind = 1.0 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Add turbulence
        wind_noise = np.random.lognormal(0, 0.3, n_points)
        
        wind_speed = seasonal_wind * daily_wind * wind_noise
        wind_speed = np.clip(wind_speed, 0, 25)  # Reasonable limits
        
        # Wind direction (prevailing westerly with variation)
        prevailing_direction = 270  # degrees (west)
        direction_variation = np.random.normal(0, 45, n_points)  # ±45° variation
        wind_direction = (prevailing_direction + direction_variation) % 360
        
        return wind_speed, wind_direction
    
    def _generate_rainfall(self, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Generate rainfall data with realistic patterns."""
        n_points = len(date_range)
        
        # Seasonal rainfall pattern
        day_of_year = date_range.dayofyear
        seasonal_rain_prob = 0.05 + 0.03 * np.cos(2 * np.pi * (day_of_year - 120) / 365.25)
        
        # Generate rain events
        rain_events = np.random.random(n_points) < seasonal_rain_prob
        
        # Rain intensity when it occurs
        rainfall = np.zeros(n_points)
        rain_intensity = np.random.exponential(2.0, n_points)  # mm/h
        rainfall[rain_events] = rain_intensity[rain_events]
        
        # Limit extreme values
        rainfall = np.clip(rainfall, 0, 50)
        
        return rainfall
    
    def _generate_atmospheric_pressure(self, date_range: pd.DatetimeIndex, 
                                     temp: np.ndarray) -> np.ndarray:
        """Generate atmospheric pressure data."""
        n_points = len(date_range)
        
        # Standard atmospheric pressure at sea level
        base_pressure = 101325  # Pa
        
        # Adjust for elevation
        altitude_adjustment = -self.config.elevation * 12  # Pa per meter
        
        # Weather system variations
        weather_variation = np.random.normal(0, 1000, n_points)
        
        # Smooth the variations
        weather_variation = signal.savgol_filter(weather_variation, 25, 3)
        
        pressure = base_pressure + altitude_adjustment + weather_variation
        
        return pressure
    
    def _calculate_dew_point(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """Calculate dew point temperature."""
        # Magnus formula approximation
        a, b = 17.27, 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        return dew_point
    
    def _calculate_wet_bulb_temp(self, temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """Calculate wet bulb temperature (simplified approximation)."""
        # Stull approximation
        wet_bulb = temp * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + \
                  np.arctan(temp + rh) - np.arctan(rh - 1.676331) + \
                  0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh) - 4.686035
        return wet_bulb

class EnergyConsumptionGenerator:
    """Generates realistic energy consumption data for building systems."""
    
    def __init__(self, config: BuildingConfig):
        self.config = config
        
    def generate_energy_data(self, weather_df: pd.DataFrame, 
                           occupancy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive energy consumption data including:
        - Whole building electricity, gas, water consumption
        - Sub-metered data for HVAC, lighting, plug loads
        - Renewable energy generation
        """
        
        timestamps = weather_df['timestamp']
        n_points = len(timestamps)
        
        # Base load calculations
        base_loads = self._calculate_base_loads(weather_df, occupancy_df)
        
        # HVAC energy consumption
        hvac_data = self._generate_hvac_consumption(weather_df, occupancy_df)
        
        # Lighting energy consumption
        lighting_data = self._generate_lighting_consumption(weather_df, occupancy_df)
        
        # Plug loads and equipment
        plug_loads = self._generate_plug_loads(weather_df, occupancy_df)
        
        # Renewable energy generation
        renewable_data = self._generate_renewable_energy(weather_df)
        
        # Water consumption
        water_data = self._generate_water_consumption(occupancy_df, weather_df)
        
        # Combine all energy data
        energy_df = pd.DataFrame({
            'timestamp': timestamps,
            
            # Whole building consumption
            'total_electricity_kw': (hvac_data['total_hvac_kw'] + 
                                   lighting_data['total_lighting_kw'] + 
                                   plug_loads['total_plug_loads_kw']),
            'total_gas_kw': hvac_data['gas_consumption_kw'],
            'total_water_l_min': water_data['total_water_l_min'],
            
            # Sub-metered HVAC
            'hvac_cooling_kw': hvac_data['cooling_kw'],
            'hvac_heating_kw': hvac_data['heating_kw'],
            'hvac_fans_kw': hvac_data['fans_kw'],
            'hvac_pumps_kw': hvac_data['pumps_kw'],
            'chiller_kw': hvac_data['chiller_kw'],
            'boiler_kw': hvac_data['boiler_kw'],
            'ahu_1_kw': hvac_data['ahu_1_kw'],
            'ahu_2_kw': hvac_data['ahu_2_kw'],
            'ahu_3_kw': hvac_data['ahu_3_kw'],
            'ahu_4_kw': hvac_data['ahu_4_kw'],
            
            # Lighting circuits
            'lighting_zone_1_kw': lighting_data['zone_1_kw'],
            'lighting_zone_2_kw': lighting_data['zone_2_kw'],
            'lighting_zone_3_kw': lighting_data['zone_3_kw'],
            'lighting_zone_4_kw': lighting_data['zone_4_kw'],
            'lighting_zone_5_kw': lighting_data['zone_5_kw'],
            'emergency_lighting_kw': lighting_data['emergency_kw'],
            'exterior_lighting_kw': lighting_data['exterior_kw'],
            
            # Plug loads and equipment
            'office_equipment_kw': plug_loads['office_equipment_kw'],
            'server_room_kw': plug_loads['server_room_kw'],
            'kitchen_equipment_kw': plug_loads['kitchen_equipment_kw'],
            'elevators_kw': plug_loads['elevators_kw'],
            'misc_equipment_kw': plug_loads['misc_equipment_kw'],
            
            # Renewable generation
            'solar_pv_generation_kw': renewable_data['solar_generation_kw'],
            'battery_charge_kw': renewable_data['battery_charge_kw'],
            'battery_discharge_kw': renewable_data['battery_discharge_kw'],
            'battery_soc_pct': renewable_data['battery_soc_pct'],
            
            # Water systems
            'domestic_hot_water_l_min': water_data['dhw_l_min'],
            'cooling_tower_water_l_min': water_data['cooling_tower_l_min'],
            'irrigation_water_l_min': water_data['irrigation_l_min'],
            
            # Energy efficiency metrics
            'total_site_eui_kwh_m2': None,  # Will be calculated
            'hvac_eui_kwh_m2': None,
            'lighting_eui_kwh_m2': None,
            'plug_loads_eui_kwh_m2': None
        })
        
        # Calculate energy use intensity (EUI) metrics
        energy_df = self._calculate_eui_metrics(energy_df)
        
        return energy_df
    
    def _calculate_base_loads(self, weather_df: pd.DataFrame, 
                            occupancy_df: pd.DataFrame) -> Dict:
        """Calculate base building loads."""
        # This would contain detailed load calculations
        # For now, return placeholder structure
        return {
            'base_electrical_kw': 50.0,
            'base_thermal_kw': 30.0
        }
    
    def _generate_hvac_consumption(self, weather_df: pd.DataFrame, 
                                 occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate HVAC energy consumption with realistic patterns."""
        n_points = len(weather_df)
        
        # Extract key variables
        outdoor_temp = weather_df['ambient_temperature_c'].values
        occupancy = occupancy_df['total_occupancy'].values
        hour_of_day = weather_df['timestamp'].dt.hour.values
        day_of_week = weather_df['timestamp'].dt.dayofweek.values
        
        # Cooling load calculation
        cooling_setpoint = 24.0  # °C
        cooling_need = np.maximum(0, outdoor_temp - cooling_setpoint)
        occupancy_heat_gain = occupancy * 0.1  # kW per person
        
        # Base cooling load
        base_cooling = cooling_need * self.config.floor_area * 0.02  # W/m²/°C
        cooling_kw = (base_cooling + occupancy_heat_gain) * 1.2  # COP factor
        
        # Heating load calculation  
        heating_setpoint = 20.0  # °C
        heating_need = np.maximum(0, heating_setpoint - outdoor_temp)
        base_heating = heating_need * self.config.floor_area * 0.025  # W/m²/°C
        heating_kw = base_heating * 0.85  # Boiler efficiency
        
        # Fan energy (proportional to HVAC operation)
        total_hvac_load = cooling_kw + heating_kw
        fans_kw = total_hvac_load * 0.15  # Fan energy ratio
        
        # Pump energy
        pumps_kw = total_hvac_load * 0.08  # Pump energy ratio
        
        # Individual equipment
        chiller_kw = cooling_kw * 0.7  # Chiller portion of cooling
        boiler_kw = heating_kw * 0.8  # Boiler portion of heating
        
        # AHU distribution (4 units)
        total_ahu_kw = fans_kw + pumps_kw * 0.5
        ahu_1_kw = total_ahu_kw * 0.3
        ahu_2_kw = total_ahu_kw * 0.25
        ahu_3_kw = total_ahu_kw * 0.25
        ahu_4_kw = total_ahu_kw * 0.2
        
        # Gas consumption for heating
        gas_consumption_kw = heating_kw / 0.85  # Account for boiler efficiency
        
        # Add operational schedules and variations
        schedule_factor = self._get_hvac_schedule_factor(hour_of_day, day_of_week)
        
        return {
            'cooling_kw': cooling_kw * schedule_factor,
            'heating_kw': heating_kw * schedule_factor,
            'fans_kw': fans_kw * schedule_factor,
            'pumps_kw': pumps_kw * schedule_factor,
            'chiller_kw': chiller_kw * schedule_factor,
            'boiler_kw': boiler_kw * schedule_factor,
            'ahu_1_kw': ahu_1_kw * schedule_factor,
            'ahu_2_kw': ahu_2_kw * schedule_factor,
            'ahu_3_kw': ahu_3_kw * schedule_factor,
            'ahu_4_kw': ahu_4_kw * schedule_factor,
            'total_hvac_kw': (cooling_kw + heating_kw + fans_kw + pumps_kw) * schedule_factor,
            'gas_consumption_kw': gas_consumption_kw * schedule_factor
        }
    
    def _generate_lighting_consumption(self, weather_df: pd.DataFrame, 
                                     occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate lighting energy consumption."""
        n_points = len(weather_df)
        
        # Extract variables
        solar_irradiance = weather_df['global_horizontal_irradiance_w_m2'].values
        occupancy = occupancy_df['total_occupancy'].values
        hour_of_day = weather_df['timestamp'].dt.hour.values
        day_of_week = weather_df['timestamp'].dt.dayofweek.values
        
        # Base lighting power density
        base_lpd = 8.0  # W/m² for LED lighting
        
        # Daylight responsive control
        daylight_factor = np.clip(1 - solar_irradiance / 500.0, 0.1, 1.0)
        
        # Occupancy control
        occupancy_factor = np.clip(occupancy / self.config.occupancy_capacity, 0.1, 1.0)
        
        # Schedule control
        schedule_factor = self._get_lighting_schedule_factor(hour_of_day, day_of_week)
        
        # Total lighting load
        total_lighting_kw = (base_lpd * self.config.floor_area * 
                           daylight_factor * occupancy_factor * schedule_factor) / 1000
        
        # Zone distribution
        zone_1_kw = total_lighting_kw * 0.25  # Main office areas
        zone_2_kw = total_lighting_kw * 0.20  # Conference rooms
        zone_3_kw = total_lighting_kw * 0.20  # Open office
        zone_4_kw = total_lighting_kw * 0.15  # Corridors
        zone_5_kw = total_lighting_kw * 0.10  # Support areas
        
        # Emergency lighting (always on, minimal)
        emergency_kw = np.full(n_points, 2.0)
        
        # Exterior lighting (time-based)
        exterior_schedule = ((hour_of_day < 7) | (hour_of_day > 18)).astype(float)
        exterior_kw = 15.0 * exterior_schedule
        
        return {
            'zone_1_kw': zone_1_kw,
            'zone_2_kw': zone_2_kw,
            'zone_3_kw': zone_3_kw,
            'zone_4_kw': zone_4_kw,
            'zone_5_kw': zone_5_kw,
            'emergency_kw': emergency_kw,
            'exterior_kw': exterior_kw,
            'total_lighting_kw': (zone_1_kw + zone_2_kw + zone_3_kw + 
                                zone_4_kw + zone_5_kw + emergency_kw + exterior_kw)
        }
    
    def _generate_plug_loads(self, weather_df: pd.DataFrame, 
                           occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate plug loads and miscellaneous equipment consumption."""
        n_points = len(weather_df)
        
        # Extract variables
        occupancy = occupancy_df['total_occupancy'].values
        hour_of_day = weather_df['timestamp'].dt.hour.values
        day_of_week = weather_df['timestamp'].dt.dayofweek.values
        
        # Office equipment (computers, printers, etc.)
        base_office_equipment = 12.0  # W/m²
        occupancy_factor = np.clip(occupancy / self.config.occupancy_capacity, 0.2, 1.0)
        schedule_factor = self._get_equipment_schedule_factor(hour_of_day, day_of_week)
        office_equipment_kw = (base_office_equipment * self.config.floor_area * 
                             occupancy_factor * schedule_factor) / 1000
        
        # Server room (constant base load with variations)
        base_server_load = 25.0  # kW
        server_variation = 1 + 0.1 * np.sin(2 * np.pi * hour_of_day / 24)
        server_room_kw = base_server_load * server_variation
        
        # Kitchen equipment
        kitchen_schedule = self._get_kitchen_schedule_factor(hour_of_day, day_of_week)
        kitchen_equipment_kw = 8.0 * kitchen_schedule
        
        # Elevators
        elevator_base = 5.0  # kW base
        elevator_usage = occupancy_factor * schedule_factor
        elevators_kw = elevator_base * (0.5 + 0.5 * elevator_usage)
        
        # Miscellaneous equipment
        misc_base = 3.0  # kW
        misc_variation = 1 + 0.2 * np.random.random(n_points)
        misc_equipment_kw = misc_base * misc_variation * schedule_factor
        
        total_plug_loads_kw = (office_equipment_kw + server_room_kw + 
                             kitchen_equipment_kw + elevators_kw + misc_equipment_kw)
        
        return {
            'office_equipment_kw': office_equipment_kw,
            'server_room_kw': server_room_kw,
            'kitchen_equipment_kw': kitchen_equipment_kw,
            'elevators_kw': elevators_kw,
            'misc_equipment_kw': misc_equipment_kw,
            'total_plug_loads_kw': total_plug_loads_kw
        }
    
    def _generate_renewable_energy(self, weather_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate renewable energy generation and storage data."""
        n_points = len(weather_df)
        
        # Solar PV generation
        solar_irradiance = weather_df['global_horizontal_irradiance_w_m2'].values
        ambient_temp = weather_df['ambient_temperature_c'].values
        
        # PV system parameters
        pv_efficiency = 0.20  # 20% efficiency
        temperature_coefficient = -0.004  # per °C
        reference_temp = 25.0  # °C
        
        # Temperature derating
        temp_factor = 1 + temperature_coefficient * (ambient_temp - reference_temp)
        
        # Solar generation
        solar_generation_kw = (self.config.solar_capacity * 
                             (solar_irradiance / 1000.0) * 
                             pv_efficiency * temp_factor)
        solar_generation_kw = np.maximum(0, solar_generation_kw)
        
        # Battery energy storage system
        battery_soc = np.zeros(n_points)  # State of charge
        battery_charge = np.zeros(n_points)
        battery_discharge = np.zeros(n_points)
        
        # Simple battery control logic
        initial_soc = 50.0  # Start at 50%
        current_soc = initial_soc
        
        for i in range(n_points):
            # Determine charging/discharging based on solar generation and time
            hour = weather_df.iloc[i]['timestamp'].hour
            
            if solar_generation_kw[i] > 20 and current_soc < 90:
                # Charge during high solar generation
                charge_power = min(25.0, solar_generation_kw[i] * 0.3)
                charge_energy = charge_power * 0.25  # 15-minute interval
                new_soc = min(100.0, current_soc + charge_energy / self.config.battery_capacity * 100)
                
                battery_charge[i] = charge_power
                battery_soc[i] = new_soc
                current_soc = new_soc
                
            elif (hour >= 17 and hour <= 21) and current_soc > 20:
                # Discharge during peak hours
                discharge_power = min(20.0, (current_soc - 20) / 100 * self.config.battery_capacity * 4)
                discharge_energy = discharge_power * 0.25
                new_soc = max(0.0, current_soc - discharge_energy / self.config.battery_capacity * 100)
                
                battery_discharge[i] = discharge_power
                battery_soc[i] = new_soc
                current_soc = new_soc
            else:
                battery_soc[i] = current_soc
        
        return {
            'solar_generation_kw': solar_generation_kw,
            'battery_charge_kw': battery_charge,
            'battery_discharge_kw': battery_discharge,
            'battery_soc_pct': battery_soc
        }
    
    def _generate_water_consumption(self, occupancy_df: pd.DataFrame, 
                                  weather_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate water consumption data."""
        n_points = len(occupancy_df)
        
        # Extract variables
        occupancy = occupancy_df['total_occupancy'].values
        outdoor_temp = weather_df['ambient_temperature_c'].values
        hour_of_day = weather_df['timestamp'].dt.hour.values
        day_of_week = weather_df['timestamp'].dt.dayofweek.values
        
        # Domestic hot water
        base_dhw_per_person = 2.0  # L/person/hour
        schedule_factor = self._get_water_schedule_factor(hour_of_day, day_of_week)
        dhw_l_min = (occupancy * base_dhw_per_person * schedule_factor) / 60
        
        # Cooling tower water (temperature dependent)
        cooling_load_factor = np.maximum(0, outdoor_temp - 20) / 30  # Normalized
        cooling_tower_base = 50.0  # L/min base
        cooling_tower_l_min = cooling_tower_base * cooling_load_factor
        
        # Irrigation water (seasonal and weather dependent)
        month = weather_df['timestamp'].dt.month.values
        irrigation_season = ((month >= 4) & (month <= 10)).astype(float)
        rainfall = weather_df['rainfall_mm_h'].values
        irrigation_factor = irrigation_season * np.maximum(0, 1 - rainfall / 5.0)
        irrigation_l_min = 20.0 * irrigation_factor
        
        # Total water consumption
        total_water_l_min = dhw_l_min + cooling_tower_l_min + irrigation_l_min
        
        return {
            'dhw_l_min': dhw_l_min,
            'cooling_tower_l_min': cooling_tower_l_min,
            'irrigation_l_min': irrigation_l_min,
            'total_water_l_min': total_water_l_min
        }
    
    def _get_hvac_schedule_factor(self, hour_of_day: np.ndarray, 
                                day_of_week: np.ndarray) -> np.ndarray:
        """Get HVAC schedule factor based on time."""
        schedule = np.ones_like(hour_of_day, dtype=float)
        
        # Weekday schedule
        weekday_mask = day_of_week < 5
        schedule[weekday_mask & (hour_of_day < 6)] = 0.3  # Night setback
        schedule[weekday_mask & (hour_of_day >= 22)] = 0.3  # Night setback
        
        # Weekend schedule
        weekend_mask = day_of_week >= 5
        schedule[weekend_mask & (hour_of_day < 8)] = 0.2
        schedule[weekend_mask & (hour_of_day >= 20)] = 0.2
        schedule[weekend_mask & ((hour_of_day >= 8) & (hour_of_day < 20))] = 0.6
        
        return schedule
    
    def _get_lighting_schedule_factor(self, hour_of_day: np.ndarray, 
                                    day_of_week: np.ndarray) -> np.ndarray:
        """Get lighting schedule factor."""
        schedule = np.ones_like(hour_of_day, dtype=float)
        
        # Weekday schedule
        weekday_mask = day_of_week < 5
        schedule[weekday_mask & (hour_of_day < 7)] = 0.1
        schedule[weekday_mask & (hour_of_day >= 19)] = 0.3
        
        # Weekend schedule
        weekend_mask = day_of_week >= 5
        schedule[weekend_mask] = 0.2
        
        return schedule
    
    def _get_equipment_schedule_factor(self, hour_of_day: np.ndarray, 
                                     day_of_week: np.ndarray) -> np.ndarray:
        """Get equipment schedule factor."""
        schedule = np.ones_like(hour_of_day, dtype=float)
        
        # Weekday schedule
        weekday_mask = day_of_week < 5
        schedule[weekday_mask & (hour_of_day < 7)] = 0.2
        schedule[weekday_mask & (hour_of_day >= 19)] = 0.4
        
        # Weekend schedule
        weekend_mask = day_of_week >= 5
        schedule[weekend_mask] = 0.3
        
        return schedule
    
    def _get_kitchen_schedule_factor(self, hour_of_day: np.ndarray, 
                                   day_of_week: np.ndarray) -> np.ndarray:
        """Get kitchen equipment schedule factor."""
        schedule = np.zeros_like(hour_of_day, dtype=float)
        
        # Meal times
        breakfast = (hour_of_day >= 7) & (hour_of_day <= 9)
        lunch = (hour_of_day >= 11) & (hour_of_day <= 14)
        dinner = (hour_of_day >= 17) & (hour_of_day <= 19)
        
        weekday_mask = day_of_week < 5
        schedule[weekday_mask & breakfast] = 0.6
        schedule[weekday_mask & lunch] = 1.0
        schedule[weekday_mask & dinner] = 0.4
        
        # Weekend - different pattern
        weekend_mask = day_of_week >= 5
        schedule[weekend_mask & ((hour_of_day >= 8) & (hour_of_day <= 10))] = 0.3
        schedule[weekend_mask & lunch] = 0.6
        
        return schedule
    
    def _get_water_schedule_factor(self, hour_of_day: np.ndarray, 
                                 day_of_week: np.ndarray) -> np.ndarray:
        """Get water usage schedule factor."""
        schedule = np.ones_like(hour_of_day, dtype=float)
        
        # Peak usage times
        morning_peak = (hour_of_day >= 8) & (hour_of_day <= 10)
        lunch_peak = (hour_of_day >= 12) & (hour_of_day <= 13)
        evening_peak = (hour_of_day >= 17) & (hour_of_day <= 18)
        
        schedule[morning_peak] = 1.5
        schedule[lunch_peak] = 1.3
        schedule[evening_peak] = 1.2
        
        # Low usage times
        schedule[(hour_of_day < 6) | (hour_of_day > 22)] = 0.2
        
        # Weekend adjustment
        weekend_mask = day_of_week >= 5
        schedule[weekend_mask] *= 0.4
        
        return schedule
    
    def _calculate_eui_metrics(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Energy Use Intensity metrics."""
        # Convert 15-minute data to hourly for EUI calculation
        time_factor = 4  # 15-minute intervals per hour
        area_factor = self.config.floor_area
        
        energy_df['total_site_eui_kwh_m2'] = (energy_df['total_electricity_kw'] + 
                                            energy_df['total_gas_kw']) / (time_factor * area_factor)
        
        energy_df['hvac_eui_kwh_m2'] = (energy_df['hvac_cooling_kw'] + 
                                      energy_df['hvac_heating_kw'] + 
                                      energy_df['hvac_fans_kw'] + 
                                      energy_df['hvac_pumps_kw']) / (time_factor * area_factor)
        
        total_lighting = (energy_df['lighting_zone_1_kw'] + energy_df['lighting_zone_2_kw'] + 
                         energy_df['lighting_zone_3_kw'] + energy_df['lighting_zone_4_kw'] + 
                         energy_df['lighting_zone_5_kw'])
        energy_df['lighting_eui_kwh_m2'] = total_lighting / (time_factor * area_factor)
        
        energy_df['plug_loads_eui_kwh_m2'] = energy_df['office_equipment_kw'] / (time_factor * area_factor)
        
        return energy_df

def main():
    """Main function to demonstrate the IoT dataset generator."""
    print("🏢 IoT Building Dataset Generator")
    print("=" * 50)
    
    # Initialize configuration
    config = BuildingConfig()
    print(f"Building Type: {config.building_type}")
    print(f"Floor Area: {config.floor_area:,.0f} m²")
    print(f"Location: {config.latitude:.2f}°N, {config.longitude:.2f}°W")
    
    # Initialize generators
    weather_gen = WeatherGenerator(config)
    energy_gen = EnergyConsumptionGenerator(config)
    
    # Generate one year of data
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-12-31 23:45:00"
    
    print(f"\n📅 Generating data from {start_date} to {end_date}")
    print("⏱️  Time resolution: 15-minute intervals")
    
    # Generate weather data
    print("\n🌤️  Generating weather data...")
    weather_df = weather_gen.generate_weather_data(start_date, end_date)
    print(f"   Generated {len(weather_df):,} weather data points")
    
    # For this demo, create simple occupancy data
    print("\n👥 Generating occupancy data...")
    # This would normally be generated by an OccupancyGenerator class
    occupancy_data = {
        'timestamp': weather_df['timestamp'],
        'total_occupancy': np.random.randint(50, 200, len(weather_df))  # Simplified
    }
    occupancy_df = pd.DataFrame(occupancy_data)
    
    # Generate energy data
    print("\n⚡ Generating energy consumption data...")
    energy_df = energy_gen.generate_energy_data(weather_df, occupancy_df)
    print(f"   Generated {len(energy_df):,} energy data points")
    
    # Display sample data
    print("\n📊 Sample Data Preview:")
    print("\nWeather Data:")
    print(weather_df.head())
    
    print("\nEnergy Data:")
    print(energy_df[['timestamp', 'total_electricity_kw', 'total_gas_kw', 
                    'hvac_cooling_kw', 'hvac_heating_kw']].head())
    
    # Basic statistics
    print("\n📈 Data Summary:")
    print(f"Temperature range: {weather_df['ambient_temperature_c'].min():.1f}°C to {weather_df['ambient_temperature_c'].max():.1f}°C")
    print(f"Peak electricity demand: {energy_df['total_electricity_kw'].max():.1f} kW")
    print(f"Annual electricity consumption: {energy_df['total_electricity_kw'].sum() * 0.25:.0f} kWh")  # 15-min to hourly
    print(f"Peak solar generation: {energy_df['solar_pv_generation_kw'].max():.1f} kW")
    
    return weather_df, energy_df, occupancy_df

if __name__ == "__main__":
    weather_data, energy_data, occupancy_data = main()