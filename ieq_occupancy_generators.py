#!/usr/bin/env python3
"""
Indoor Environmental Quality (IEQ) and Occupancy Generators
===========================================================

Advanced generators for creating realistic indoor environmental quality data
and occupancy patterns for building digital twin applications.

This module generates:
- Multi-zone thermal conditions (temperature, humidity)
- Air quality parameters (CO2, PM2.5, PM10, TVOCs)
- Lighting conditions (illuminance levels)
- Acoustic conditions (noise levels)
- Occupancy patterns with realistic behavioral models
- Space utilization data
- Window and blind operation patterns

Author: AI Assistant
Date: 2025-10-16
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import signal
from scipy.stats import norm, poisson, lognorm
import warnings

warnings.filterwarnings('ignore')

class IndoorEnvironmentalQualityGenerator:
    """Generates comprehensive IEQ data for multiple zones."""
    
    def __init__(self, config, weather_df: pd.DataFrame, energy_df: pd.DataFrame):
        self.config = config
        self.weather_df = weather_df
        self.energy_df = energy_df
        self.n_zones = config.num_zones
        
    def generate_ieq_data(self, occupancy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive IEQ data including:
        - Thermal conditions (temperature, humidity) by zone
        - Air quality (CO2, PM2.5, PM10, TVOCs) by zone
        - Lighting conditions (illuminance) by zone
        - Acoustic conditions (noise levels) by zone
        """
        
        timestamps = self.weather_df['timestamp']
        n_points = len(timestamps)
        
        # Generate thermal conditions
        thermal_data = self._generate_thermal_conditions(occupancy_df)
        
        # Generate air quality data
        air_quality_data = self._generate_air_quality(occupancy_df)
        
        # Generate lighting conditions
        lighting_data = self._generate_lighting_conditions(occupancy_df)
        
        # Generate acoustic conditions
        acoustic_data = self._generate_acoustic_conditions(occupancy_df)
        
        # Combine all IEQ data
        ieq_df = pd.DataFrame({'timestamp': timestamps})
        
        # Add thermal data for each zone
        for zone in range(1, self.n_zones + 1):
            ieq_df[f'zone_{zone}_air_temp_c'] = thermal_data[f'zone_{zone}_temp']
            ieq_df[f'zone_{zone}_relative_humidity_pct'] = thermal_data[f'zone_{zone}_rh']
            ieq_df[f'zone_{zone}_operative_temp_c'] = thermal_data[f'zone_{zone}_operative_temp']
        
        # Add air quality data for each zone
        for zone in range(1, self.n_zones + 1):
            ieq_df[f'zone_{zone}_co2_ppm'] = air_quality_data[f'zone_{zone}_co2']
            ieq_df[f'zone_{zone}_pm25_ug_m3'] = air_quality_data[f'zone_{zone}_pm25']
            ieq_df[f'zone_{zone}_pm10_ug_m3'] = air_quality_data[f'zone_{zone}_pm10']
            ieq_df[f'zone_{zone}_tvoc_ppb'] = air_quality_data[f'zone_{zone}_tvoc']
        
        # Add lighting data for each zone
        for zone in range(1, self.n_zones + 1):
            ieq_df[f'zone_{zone}_illuminance_lux'] = lighting_data[f'zone_{zone}_illuminance']
            ieq_df[f'zone_{zone}_daylight_factor'] = lighting_data[f'zone_{zone}_daylight_factor']
        
        # Add acoustic data for each zone
        for zone in range(1, self.n_zones + 1):
            ieq_df[f'zone_{zone}_noise_level_db'] = acoustic_data[f'zone_{zone}_noise']
        
        # Add building-wide averages
        temp_cols = [f'zone_{zone}_air_temp_c' for zone in range(1, self.n_zones + 1)]
        ieq_df['building_avg_temp_c'] = ieq_df[temp_cols].mean(axis=1)
        
        rh_cols = [f'zone_{zone}_relative_humidity_pct' for zone in range(1, self.n_zones + 1)]
        ieq_df['building_avg_rh_pct'] = ieq_df[rh_cols].mean(axis=1)
        
        co2_cols = [f'zone_{zone}_co2_ppm' for zone in range(1, self.n_zones + 1)]
        ieq_df['building_avg_co2_ppm'] = ieq_df[co2_cols].mean(axis=1)
        
        illuminance_cols = [f'zone_{zone}_illuminance_lux' for zone in range(1, self.n_zones + 1)]
        ieq_df['building_avg_illuminance_lux'] = ieq_df[illuminance_cols].mean(axis=1)
        
        return ieq_df
    
    def _generate_thermal_conditions(self, occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate thermal conditions for each zone."""
        n_points = len(self.weather_df)
        thermal_data = {}
        
        # Extract key variables
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        outdoor_rh = self.weather_df['relative_humidity_pct'].values
        solar_irradiance = self.weather_df['global_horizontal_irradiance_w_m2'].values
        hvac_cooling = self.energy_df['hvac_cooling_kw'].values
        hvac_heating = self.energy_df['hvac_heating_kw'].values
        
        # HVAC setpoints (will be detailed in building systems generator)
        cooling_setpoint = 24.0  # °C
        heating_setpoint = 20.0  # °C
        
        for zone in range(1, self.n_zones + 1):
            # Zone-specific characteristics
            zone_orientation = (zone - 1) * 360 / self.n_zones  # Distribute around building
            zone_solar_gain_factor = 0.5 + 0.3 * np.cos(np.radians(zone_orientation))
            zone_occupancy = occupancy_df[f'zone_{zone}_occupancy'].values if f'zone_{zone}_occupancy' in occupancy_df.columns else occupancy_df['total_occupancy'].values / self.n_zones
            
            # Calculate zone temperature
            zone_temp = self._calculate_zone_temperature(
                outdoor_temp, hvac_cooling, hvac_heating, solar_irradiance,
                zone_solar_gain_factor, zone_occupancy, cooling_setpoint, heating_setpoint
            )
            
            # Calculate zone humidity
            zone_rh = self._calculate_zone_humidity(
                outdoor_rh, zone_temp, outdoor_temp, zone_occupancy
            )
            
            # Calculate operative temperature (combination of air temp and radiant temp)
            radiant_temp_offset = solar_irradiance * zone_solar_gain_factor * 0.005  # Solar heating effect
            operative_temp = zone_temp + radiant_temp_offset * 0.5  # Simplified operative temp
            
            thermal_data[f'zone_{zone}_temp'] = zone_temp
            thermal_data[f'zone_{zone}_rh'] = zone_rh
            thermal_data[f'zone_{zone}_operative_temp'] = operative_temp
        
        return thermal_data
    
    def _calculate_zone_temperature(self, outdoor_temp: np.ndarray, 
                                  hvac_cooling: np.ndarray, hvac_heating: np.ndarray,
                                  solar_irradiance: np.ndarray, solar_gain_factor: float,
                                  occupancy: np.ndarray, cooling_sp: float, heating_sp: float) -> np.ndarray:
        """Calculate zone air temperature with HVAC control."""
        
        # Base temperature without HVAC (free-floating)
        thermal_mass_factor = 0.7  # Building thermal inertia
        solar_heat_gain = solar_irradiance * solar_gain_factor * 0.001  # W/m² to °C
        occupancy_heat_gain = occupancy * 0.1  # °C per person
        
        free_float_temp = (outdoor_temp * thermal_mass_factor + 
                          solar_heat_gain + occupancy_heat_gain)
        
        # HVAC control logic
        zone_temp = np.zeros_like(outdoor_temp)
        
        for i in range(len(outdoor_temp)):
            target_temp = free_float_temp[i]
            
            # Cooling control
            if hvac_cooling[i] > 0 and target_temp > cooling_sp:
                cooling_capacity = hvac_cooling[i] / self.config.num_zones  # Distributed cooling
                temp_reduction = min(target_temp - cooling_sp, cooling_capacity * 0.1)
                target_temp -= temp_reduction
            
            # Heating control
            if hvac_heating[i] > 0 and target_temp < heating_sp:
                heating_capacity = hvac_heating[i] / self.config.num_zones
                temp_increase = min(heating_sp - target_temp, heating_capacity * 0.1)
                target_temp += temp_increase
            
            zone_temp[i] = target_temp
        
        # Add control noise and sensor accuracy
        control_noise = np.random.normal(0, 0.5, len(zone_temp))
        zone_temp += control_noise
        
        return zone_temp
    
    def _calculate_zone_humidity(self, outdoor_rh: np.ndarray, zone_temp: np.ndarray,
                               outdoor_temp: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """Calculate zone relative humidity."""
        
        # Moisture sources
        occupancy_moisture = occupancy * 0.5  # % RH per person
        
        # HVAC moisture removal (simplified)
        hvac_dehumidification = np.maximum(0, zone_temp - 22) * 2  # Cooling coil condensation
        
        # Base humidity from outdoor conditions (with lag)
        humidity_lag_factor = 0.8
        base_humidity = outdoor_rh * humidity_lag_factor
        
        # Temperature effect on RH
        temp_effect = (outdoor_temp - zone_temp) * 2  # RH change per °C
        
        zone_rh = base_humidity + occupancy_moisture + temp_effect - hvac_dehumidification
        
        # Add noise and clamp to realistic range
        humidity_noise = np.random.normal(0, 3, len(zone_rh))
        zone_rh = np.clip(zone_rh + humidity_noise, 25, 80)
        
        return zone_rh
    
    def _generate_air_quality(self, occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate air quality parameters for each zone."""
        n_points = len(self.weather_df)
        air_quality_data = {}
        
        # Extract variables
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        wind_speed = self.weather_df['wind_speed_m_s'].values
        hvac_fans = self.energy_df['hvac_fans_kw'].values
        hour_of_day = self.weather_df['timestamp'].dt.hour.values
        day_of_week = self.weather_df['timestamp'].dt.dayofweek.values
        
        # Outdoor air quality baseline (varies by season and weather)
        outdoor_pm25 = self._generate_outdoor_pm25()
        outdoor_pm10 = outdoor_pm25 * 2.1  # Typical ratio
        
        for zone in range(1, self.n_zones + 1):
            zone_occupancy = occupancy_df[f'zone_{zone}_occupancy'].values if f'zone_{zone}_occupancy' in occupancy_df.columns else occupancy_df['total_occupancy'].values / self.n_zones
            
            # CO2 generation (primary indicator of ventilation effectiveness)
            co2_data = self._generate_co2_levels(zone_occupancy, hvac_fans, hour_of_day, day_of_week)
            
            # Particulate matter (PM2.5 and PM10)
            pm25_data = self._generate_pm_levels(outdoor_pm25, zone_occupancy, hvac_fans, 'pm25')
            pm10_data = self._generate_pm_levels(outdoor_pm10, zone_occupancy, hvac_fans, 'pm10')
            
            # Total Volatile Organic Compounds (TVOCs)
            tvoc_data = self._generate_tvoc_levels(zone_occupancy, outdoor_temp, hour_of_day)
            
            air_quality_data[f'zone_{zone}_co2'] = co2_data
            air_quality_data[f'zone_{zone}_pm25'] = pm25_data
            air_quality_data[f'zone_{zone}_pm10'] = pm10_data
            air_quality_data[f'zone_{zone}_tvoc'] = tvoc_data
        
        return air_quality_data
    
    def _generate_outdoor_pm25(self) -> np.ndarray:
        """Generate outdoor PM2.5 baseline with seasonal and weather effects."""
        n_points = len(self.weather_df)
        
        # Seasonal baseline (higher in winter due to heating)
        day_of_year = self.weather_df['timestamp'].dt.dayofyear.values
        seasonal_pm25 = 15 + 10 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
        
        # Weather effects
        wind_speed = self.weather_df['wind_speed_m_s'].values
        rainfall = self.weather_df['rainfall_mm_h'].values
        
        # Wind disperses pollution
        wind_factor = np.exp(-wind_speed / 5.0)
        
        # Rain washes out particles
        rain_factor = np.exp(-rainfall / 2.0)
        
        # Traffic and industrial patterns
        hour_of_day = self.weather_df['timestamp'].dt.hour.values
        traffic_factor = 1 + 0.3 * ((hour_of_day >= 7) & (hour_of_day <= 9)).astype(float)
        traffic_factor += 0.2 * ((hour_of_day >= 17) & (hour_of_day <= 19)).astype(float)
        
        outdoor_pm25 = seasonal_pm25 * wind_factor * rain_factor * traffic_factor
        
        # Add random events (wildfires, dust storms, etc.)
        extreme_events = np.random.exponential(1, n_points)
        extreme_mask = extreme_events > 5  # Rare events
        outdoor_pm25[extreme_mask] *= np.random.uniform(2, 5, np.sum(extreme_mask))
        
        # Add noise and ensure positive values
        noise = np.random.lognormal(0, 0.2, n_points)
        outdoor_pm25 = np.maximum(1, outdoor_pm25 * noise)
        
        return outdoor_pm25
    
    def _generate_co2_levels(self, occupancy: np.ndarray, hvac_fans: np.ndarray,
                           hour_of_day: np.ndarray, day_of_week: np.ndarray) -> np.ndarray:
        """Generate CO2 levels based on occupancy and ventilation."""
        
        # Outdoor CO2 baseline
        outdoor_co2 = 420  # ppm (current atmospheric level)
        
        # CO2 generation rate per person
        co2_generation_rate = 0.3  # L/min/person (at rest)
        activity_factor = 1 + 0.3 * ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(float)
        
        # Ventilation effectiveness (proportional to fan energy)
        ventilation_rate = hvac_fans * 10 + 2  # Base ventilation + fan-driven
        
        # CO2 accumulation model (simplified mass balance)
        co2_levels = np.zeros_like(occupancy)
        current_co2 = outdoor_co2
        
        for i in range(len(occupancy)):
            # CO2 generation
            co2_generation = occupancy[i] * co2_generation_rate * activity_factor[i]
            
            # CO2 removal by ventilation
            co2_removal = (current_co2 - outdoor_co2) * ventilation_rate[i] * 0.001
            
            # Update CO2 level (15-minute time step)
            time_step = 0.25  # hours
            current_co2 += (co2_generation - co2_removal) * time_step * 10  # ppm increase
            
            # Ensure minimum outdoor level
            current_co2 = max(outdoor_co2, current_co2)
            
            co2_levels[i] = current_co2
        
        # Add measurement noise
        noise = np.random.normal(0, 20, len(co2_levels))
        co2_levels = np.maximum(outdoor_co2, co2_levels + noise)
        
        return co2_levels
    
    def _generate_pm_levels(self, outdoor_pm: np.ndarray, occupancy: np.ndarray,
                          hvac_fans: np.ndarray, pm_type: str) -> np.ndarray:
        """Generate indoor PM levels."""
        
        # Infiltration factor (outdoor PM entering building)
        infiltration_factor = 0.6  # 60% of outdoor PM infiltrates
        
        # Indoor sources
        if pm_type == 'pm25':
            occupancy_pm = occupancy * 0.5  # μg/m³ per person (cooking, activities)
            equipment_pm = 2.0  # Base equipment emissions
        else:  # pm10
            occupancy_pm = occupancy * 0.8
            equipment_pm = 3.0
        
        # Filtration effectiveness (higher with more HVAC operation)
        filter_efficiency = np.clip(hvac_fans * 0.1, 0.1, 0.8)
        
        # Indoor PM calculation
        infiltrated_pm = outdoor_pm * infiltration_factor * (1 - filter_efficiency)
        indoor_sources = occupancy_pm + equipment_pm
        
        indoor_pm = infiltrated_pm + indoor_sources
        
        # Add noise
        noise = np.random.lognormal(0, 0.15, len(indoor_pm))
        indoor_pm = np.maximum(1, indoor_pm * noise)
        
        return indoor_pm
    
    def _generate_tvoc_levels(self, occupancy: np.ndarray, outdoor_temp: np.ndarray,
                            hour_of_day: np.ndarray) -> np.ndarray:
        """Generate Total Volatile Organic Compounds levels."""
        
        # Base TVOC from building materials and furnishings
        base_tvoc = 200  # ppb
        
        # Temperature effect (higher temps increase off-gassing)
        temp_factor = 1 + (outdoor_temp - 20) * 0.02
        
        # Occupancy effect (personal care products, activities)
        occupancy_tvoc = occupancy * 15  # ppb per person
        
        # Cleaning activities (higher during non-occupied hours)
        cleaning_factor = 1 + 0.5 * ((hour_of_day < 6) | (hour_of_day > 20)).astype(float)
        
        # Equipment and office activities
        equipment_tvoc = 50 * ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)
        
        tvoc_levels = (base_tvoc * temp_factor * cleaning_factor + 
                      occupancy_tvoc + equipment_tvoc)
        
        # Add noise and ensure positive values
        noise = np.random.lognormal(0, 0.2, len(tvoc_levels))
        tvoc_levels = np.maximum(50, tvoc_levels * noise)
        
        return tvoc_levels
    
    def _generate_lighting_conditions(self, occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate lighting conditions for each zone."""
        n_points = len(self.weather_df)
        lighting_data = {}
        
        # Extract variables
        solar_irradiance = self.weather_df['global_horizontal_irradiance_w_m2'].values
        solar_elevation = self.weather_df['solar_elevation_deg'].values
        hour_of_day = self.weather_df['timestamp'].dt.hour.values
        
        for zone in range(1, self.n_zones + 1):
            zone_occupancy = occupancy_df[f'zone_{zone}_occupancy'].values if f'zone_{zone}_occupancy' in occupancy_df.columns else occupancy_df['total_occupancy'].values / self.n_zones
            
            # Calculate daylight contribution
            daylight_illuminance = self._calculate_daylight_illuminance(
                solar_irradiance, solar_elevation, zone
            )
            
            # Calculate artificial lighting contribution
            artificial_illuminance = self._calculate_artificial_lighting(
                zone_occupancy, hour_of_day, daylight_illuminance
            )
            
            # Total illuminance
            total_illuminance = daylight_illuminance + artificial_illuminance
            
            # Daylight factor
            daylight_factor = np.where(total_illuminance > 0, 
                                     daylight_illuminance / total_illuminance, 0)
            
            lighting_data[f'zone_{zone}_illuminance'] = total_illuminance
            lighting_data[f'zone_{zone}_daylight_factor'] = daylight_factor
        
        return lighting_data
    
    def _calculate_daylight_illuminance(self, solar_irradiance: np.ndarray,
                                      solar_elevation: np.ndarray, zone: int) -> np.ndarray:
        """Calculate daylight illuminance in a zone."""
        
        # Zone orientation factor
        zone_orientation = (zone - 1) * 360 / self.n_zones
        orientation_factor = 0.5 + 0.4 * np.cos(np.radians(zone_orientation))
        
        # Window area and transmittance
        window_transmittance = 0.7
        window_area_factor = self.config.window_to_wall_ratio
        
        # Daylight factor (simplified)
        daylight_factor = 0.02 * window_area_factor * orientation_factor
        
        # Convert solar irradiance to illuminance (rough approximation)
        # 1 W/m² ≈ 120 lux for daylight
        daylight_illuminance = (solar_irradiance * 120 * daylight_factor * 
                              window_transmittance * np.maximum(0, np.sin(np.radians(solar_elevation))))
        
        return np.maximum(0, daylight_illuminance)
    
    def _calculate_artificial_lighting(self, occupancy: np.ndarray, 
                                     hour_of_day: np.ndarray,
                                     daylight_illuminance: np.ndarray) -> np.ndarray:
        """Calculate artificial lighting contribution."""
        
        # Target illuminance for office work
        target_illuminance = 500  # lux
        
        # Occupancy-based control
        occupancy_factor = np.clip(occupancy / 10, 0, 1)  # Normalize to zone capacity
        
        # Time-based control
        time_factor = ((hour_of_day >= 7) & (hour_of_day <= 19)).astype(float)
        
        # Daylight responsive control
        daylight_shortfall = np.maximum(0, target_illuminance - daylight_illuminance)
        
        # Artificial lighting output
        artificial_illuminance = (daylight_shortfall * occupancy_factor * time_factor)
        
        # Add dimming and control variations
        control_variation = np.random.uniform(0.8, 1.0, len(artificial_illuminance))
        artificial_illuminance *= control_variation
        
        return artificial_illuminance
    
    def _generate_acoustic_conditions(self, occupancy_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate acoustic conditions (noise levels) for each zone."""
        n_points = len(self.weather_df)
        acoustic_data = {}
        
        # Extract variables
        hour_of_day = self.weather_df['timestamp'].dt.hour.values
        day_of_week = self.weather_df['timestamp'].dt.dayofweek.values
        hvac_fans = self.energy_df['hvac_fans_kw'].values
        
        for zone in range(1, self.n_zones + 1):
            zone_occupancy = occupancy_df[f'zone_{zone}_occupancy'].values if f'zone_{zone}_occupancy' in occupancy_df.columns else occupancy_df['total_occupancy'].values / self.n_zones
            
            # Base noise level (building systems, ambient)
            base_noise = 35  # dB(A) - quiet office
            
            # HVAC noise
            hvac_noise = hvac_fans * 2 + 5  # dB(A) from HVAC operation
            
            # Occupancy noise
            occupancy_noise = zone_occupancy * 1.5  # dB(A) per person
            
            # Activity noise (higher during work hours)
            activity_factor = 1 + 0.3 * ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(float)
            
            # Equipment noise
            equipment_noise = 5 * ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)
            
            # Total noise level (logarithmic addition)
            total_noise = base_noise
            noise_sources = [hvac_noise, occupancy_noise * activity_factor, equipment_noise]
            
            for noise_source in noise_sources:
                total_noise = 10 * np.log10(10**(total_noise/10) + 10**(noise_source/10))
            
            # Add random variations
            noise_variation = np.random.normal(0, 2, len(total_noise))
            total_noise = np.maximum(30, total_noise + noise_variation)  # Minimum 30 dB(A)
            
            acoustic_data[f'zone_{zone}_noise'] = total_noise
        
        return acoustic_data

class OccupancyPatternGenerator:
    """Generates realistic occupancy and space utilization patterns."""
    
    def __init__(self, config):
        self.config = config
        
    def generate_occupancy_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive occupancy data including:
        - Total building occupancy
        - Zone-level occupancy
        - Space utilization patterns
        - Movement patterns
        - Window and blind operations
        """
        
        timestamps = weather_df['timestamp']
        n_points = len(timestamps)
        
        # Generate base occupancy patterns
        base_occupancy = self._generate_base_occupancy_patterns(weather_df)
        
        # Generate zone-level occupancy
        zone_occupancy = self._generate_zone_occupancy(base_occupancy, weather_df)
        
        # Generate space utilization
        space_utilization = self._generate_space_utilization(zone_occupancy, weather_df)
        
        # Generate window and blind operations
        window_operations = self._generate_window_blind_operations(weather_df, zone_occupancy)
        
        # Combine all occupancy data
        occupancy_df = pd.DataFrame({'timestamp': timestamps})
        
        # Add base occupancy data
        occupancy_df['total_occupancy'] = base_occupancy['total_occupancy']
        occupancy_df['arrival_count'] = base_occupancy['arrivals']
        occupancy_df['departure_count'] = base_occupancy['departures']
        occupancy_df['visitor_count'] = base_occupancy['visitors']
        
        # Add zone occupancy
        for zone in range(1, self.config.num_zones + 1):
            occupancy_df[f'zone_{zone}_occupancy'] = zone_occupancy[f'zone_{zone}']
            occupancy_df[f'zone_{zone}_utilization_pct'] = space_utilization[f'zone_{zone}_utilization']
        
        # Add movement and activity data
        occupancy_df['people_counter_entrance_in'] = base_occupancy['entrance_in']
        occupancy_df['people_counter_entrance_out'] = base_occupancy['entrance_out']
        occupancy_df['wifi_connected_devices'] = base_occupancy['wifi_devices']
        
        # Add window and blind operations
        for zone in range(1, self.config.num_zones + 1):
            occupancy_df[f'zone_{zone}_windows_open_pct'] = window_operations[f'zone_{zone}_windows']
            occupancy_df[f'zone_{zone}_blinds_closed_pct'] = window_operations[f'zone_{zone}_blinds']
        
        # Add derived metrics
        occupancy_df['occupancy_density_ppl_m2'] = (occupancy_df['total_occupancy'] / 
                                                   self.config.floor_area)
        occupancy_df['building_utilization_pct'] = ((occupancy_df['total_occupancy'] / 
                                                   self.config.occupancy_capacity) * 100)
        
        return occupancy_df
    
    def _generate_base_occupancy_patterns(self, weather_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate base building occupancy patterns."""
        n_points = len(weather_df)
        
        # Extract time variables
        hour_of_day = weather_df['timestamp'].dt.hour.values
        day_of_week = weather_df['timestamp'].dt.dayofweek.values
        month = weather_df['timestamp'].dt.month.values
        
        # Base occupancy schedule
        base_schedule = self._get_occupancy_schedule(hour_of_day, day_of_week)
        
        # Seasonal variations
        seasonal_factor = self._get_seasonal_occupancy_factor(month)
        
        # Weather impact on occupancy
        weather_factor = self._get_weather_occupancy_factor(weather_df)
        
        # Calculate total occupancy
        max_occupancy = self.config.occupancy_capacity
        total_occupancy = (base_schedule * seasonal_factor * weather_factor * 
                         max_occupancy).astype(int)
        
        # Generate arrival and departure patterns
        arrivals, departures = self._generate_arrival_departure_patterns(
            total_occupancy, hour_of_day, day_of_week
        )
        
        # Generate visitor patterns
        visitors = self._generate_visitor_patterns(hour_of_day, day_of_week, month)
        
        # Generate entrance counter data
        entrance_in, entrance_out = self._generate_entrance_counter_data(
            arrivals, departures, visitors
        )
        
        # Generate WiFi device counts (proxy for occupancy)
        wifi_devices = self._generate_wifi_device_counts(total_occupancy)
        
        return {
            'total_occupancy': total_occupancy,
            'arrivals': arrivals,
            'departures': departures,
            'visitors': visitors,
            'entrance_in': entrance_in,
            'entrance_out': entrance_out,
            'wifi_devices': wifi_devices
        }
    
    def _get_occupancy_schedule(self, hour_of_day: np.ndarray, 
                              day_of_week: np.ndarray) -> np.ndarray:
        """Get base occupancy schedule factor."""
        schedule = np.zeros_like(hour_of_day, dtype=float)
        
        # Weekday schedule
        weekday_mask = day_of_week < 5
        
        # Arrival period (7-9 AM)
        arrival_mask = weekday_mask & (hour_of_day >= 7) & (hour_of_day < 9)
        schedule[arrival_mask] = 0.3 + 0.4 * (hour_of_day[arrival_mask] - 7) / 2
        
        # Core hours (9 AM - 5 PM)
        core_mask = weekday_mask & (hour_of_day >= 9) & (hour_of_day < 17)
        schedule[core_mask] = 0.85 + 0.1 * np.sin(2 * np.pi * (hour_of_day[core_mask] - 9) / 8)
        
        # Departure period (5-7 PM)
        departure_mask = weekday_mask & (hour_of_day >= 17) & (hour_of_day < 19)
        schedule[departure_mask] = 0.7 - 0.6 * (hour_of_day[departure_mask] - 17) / 2
        
        # Evening/night (minimal occupancy)
        evening_mask = weekday_mask & ((hour_of_day >= 19) | (hour_of_day < 7))
        schedule[evening_mask] = 0.05
        
        # Weekend schedule (reduced occupancy)
        weekend_mask = day_of_week >= 5
        weekend_hours = (hour_of_day >= 9) & (hour_of_day < 15)
        schedule[weekend_mask & weekend_hours] = 0.15
        schedule[weekend_mask & ~weekend_hours] = 0.02
        
        return schedule
    
    def _get_seasonal_occupancy_factor(self, month: np.ndarray) -> np.ndarray:
        """Get seasonal occupancy variations."""
        seasonal_factor = np.ones_like(month, dtype=float)
        
        # Summer vacation period (reduced occupancy)
        summer_mask = (month >= 7) & (month <= 8)
        seasonal_factor[summer_mask] = 0.7
        
        # Holiday periods
        december_mask = month == 12
        seasonal_factor[december_mask] = 0.6
        
        # Spring/fall (normal occupancy)
        normal_mask = ((month >= 3) & (month <= 6)) | ((month >= 9) & (month <= 11))
        seasonal_factor[normal_mask] = 1.0
        
        # Winter (slightly reduced due to sick days, weather)
        winter_mask = (month <= 2)
        seasonal_factor[winter_mask] = 0.9
        
        return seasonal_factor
    
    def _get_weather_occupancy_factor(self, weather_df: pd.DataFrame) -> np.ndarray:
        """Get weather impact on occupancy."""
        n_points = len(weather_df)
        weather_factor = np.ones(n_points)
        
        # Extreme weather reduces occupancy
        temp = weather_df['ambient_temperature_c'].values
        rainfall = weather_df['rainfall_mm_h'].values
        wind_speed = weather_df['wind_speed_m_s'].values
        
        # Extreme cold
        cold_mask = temp < -10
        weather_factor[cold_mask] *= 0.8
        
        # Extreme heat
        heat_mask = temp > 35
        weather_factor[heat_mask] *= 0.9
        
        # Heavy rain
        rain_mask = rainfall > 10
        weather_factor[rain_mask] *= 0.85
        
        # High winds
        wind_mask = wind_speed > 15
        weather_factor[wind_mask] *= 0.95
        
        return weather_factor
    
    def _generate_arrival_departure_patterns(self, total_occupancy: np.ndarray,
                                           hour_of_day: np.ndarray, 
                                           day_of_week: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic arrival and departure patterns."""
        n_points = len(total_occupancy)
        arrivals = np.zeros(n_points)
        departures = np.zeros(n_points)
        
        # Calculate occupancy changes
        occupancy_changes = np.diff(total_occupancy, prepend=total_occupancy[0])
        
        # Positive changes are arrivals, negative are departures
        arrivals[1:] = np.maximum(0, occupancy_changes[1:])
        departures[1:] = np.maximum(0, -occupancy_changes[1:])
        
        # Add stochastic variations for people movements during steady periods
        weekday_mask = day_of_week < 5
        work_hours_mask = (hour_of_day >= 9) & (hour_of_day <= 17)
        
        # Random movements during work hours
        random_movements_mask = weekday_mask & work_hours_mask
        random_arrivals = np.random.poisson(0.5, n_points)
        random_departures = np.random.poisson(0.5, n_points)
        
        arrivals[random_movements_mask] += random_arrivals[random_movements_mask]
        departures[random_movements_mask] += random_departures[random_movements_mask]
        
        return arrivals.astype(int), departures.astype(int)
    
    def _generate_visitor_patterns(self, hour_of_day: np.ndarray, 
                                 day_of_week: np.ndarray, month: np.ndarray) -> np.ndarray:
        """Generate visitor patterns."""
        n_points = len(hour_of_day)
        
        # Base visitor rate
        base_visitor_rate = 2.0  # visitors per hour during business hours
        
        # Time-based patterns
        weekday_mask = day_of_week < 5
        business_hours_mask = (hour_of_day >= 9) & (hour_of_day <= 17)
        
        # Visitor schedule
        visitor_schedule = np.zeros(n_points)
        visitor_schedule[weekday_mask & business_hours_mask] = base_visitor_rate
        
        # Peak visitor times (10-11 AM, 2-3 PM)
        peak_morning = weekday_mask & (hour_of_day == 10)
        peak_afternoon = weekday_mask & (hour_of_day == 14)
        visitor_schedule[peak_morning | peak_afternoon] *= 2.0
        
        # Generate actual visitor counts
        visitors = np.random.poisson(visitor_schedule / 4)  # 15-minute intervals
        
        return visitors
    
    def _generate_entrance_counter_data(self, arrivals: np.ndarray, 
                                      departures: np.ndarray, 
                                      visitors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate entrance counter data with sensor noise."""
        
        # Base counts
        entrance_in = arrivals + visitors
        entrance_out = departures + visitors  # Visitors also leave
        
        # Add sensor noise and occasional miscounts
        noise_factor = 0.95  # 95% accuracy
        
        # Random sensor errors
        in_errors = np.random.binomial(entrance_in, 1 - noise_factor)
        out_errors = np.random.binomial(entrance_out, 1 - noise_factor)
        
        # Occasional false positives
        false_positives_in = np.random.poisson(0.1, len(entrance_in))
        false_positives_out = np.random.poisson(0.1, len(entrance_out))
        
        entrance_in = entrance_in - in_errors + false_positives_in
        entrance_out = entrance_out - out_errors + false_positives_out
        
        return np.maximum(0, entrance_in), np.maximum(0, entrance_out)
    
    def _generate_wifi_device_counts(self, total_occupancy: np.ndarray) -> np.ndarray:
        """Generate WiFi connected device counts as occupancy proxy."""
        
        # Assume each person has 1-3 devices (phone, laptop, tablet)
        devices_per_person = np.random.uniform(1.2, 2.5, len(total_occupancy))
        
        # Base device count
        base_devices = total_occupancy * devices_per_person
        
        # Add guest devices and IoT devices
        guest_devices = np.random.poisson(5, len(total_occupancy))  # Guest network
        iot_devices = 50  # Fixed IoT devices
        
        # Device connection variability
        connection_rate = np.random.uniform(0.8, 0.95, len(total_occupancy))
        
        total_devices = (base_devices * connection_rate + guest_devices + iot_devices)
        
        return total_devices.astype(int)
    
    def _generate_zone_occupancy(self, base_occupancy: Dict[str, np.ndarray], 
                               weather_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Distribute total occupancy across zones."""
        n_points = len(weather_df)
        zone_occupancy = {}
        
        total_occ = base_occupancy['total_occupancy']
        hour_of_day = weather_df['timestamp'].dt.hour.values
        
        # Zone characteristics (different types of spaces)
        zone_types = ['open_office', 'private_office', 'meeting_room', 'break_room', 
                     'lobby', 'conference', 'workspace', 'support']
        
        for zone in range(1, self.config.num_zones + 1):
            zone_type = zone_types[(zone - 1) % len(zone_types)]
            
            # Zone capacity factor
            capacity_factors = {
                'open_office': 0.15,
                'private_office': 0.08,
                'meeting_room': 0.12,
                'break_room': 0.05,
                'lobby': 0.03,
                'conference': 0.10,
                'workspace': 0.20,
                'support': 0.02
            }
            
            base_factor = capacity_factors[zone_type]
            
            # Time-based usage patterns
            usage_patterns = self._get_zone_usage_pattern(zone_type, hour_of_day)
            
            # Calculate zone occupancy
            zone_occ = (total_occ * base_factor * usage_patterns).astype(int)
            
            # Add stochastic variations
            noise = np.random.poisson(0.5, n_points)
            zone_occ = np.maximum(0, zone_occ + noise - 1)
            
            zone_occupancy[f'zone_{zone}'] = zone_occ
        
        # Normalize to ensure sum doesn't exceed total (approximately)
        total_distributed = sum(zone_occupancy.values())
        for zone in range(1, self.config.num_zones + 1):
            if np.sum(total_distributed) > 0:
                zone_occupancy[f'zone_{zone}'] = (zone_occupancy[f'zone_{zone}'] * 
                                                total_occ / total_distributed).astype(int)
        
        return zone_occupancy
    
    def _get_zone_usage_pattern(self, zone_type: str, hour_of_day: np.ndarray) -> np.ndarray:
        """Get usage pattern for different zone types."""
        n_points = len(hour_of_day)
        pattern = np.ones(n_points)
        
        if zone_type == 'meeting_room':
            # Peak usage during business hours with meeting patterns
            meeting_times = ((hour_of_day >= 9) & (hour_of_day <= 11)) | \
                          ((hour_of_day >= 14) & (hour_of_day <= 16))
            pattern[meeting_times] = 2.0
            pattern[~meeting_times] = 0.3
            
        elif zone_type == 'break_room':
            # Peak during lunch and break times
            lunch_time = (hour_of_day >= 12) & (hour_of_day <= 13)
            break_times = ((hour_of_day >= 10) & (hour_of_day <= 11)) | \
                         ((hour_of_day >= 15) & (hour_of_day <= 16))
            pattern[lunch_time] = 3.0
            pattern[break_times] = 2.0
            pattern[~(lunch_time | break_times)] = 0.2
            
        elif zone_type == 'lobby':
            # Higher usage during arrival/departure times
            arrival_departure = ((hour_of_day >= 7) & (hour_of_day <= 9)) | \
                              ((hour_of_day >= 17) & (hour_of_day <= 19))
            pattern[arrival_departure] = 2.5
            pattern[~arrival_departure] = 0.5
            
        elif zone_type in ['open_office', 'private_office', 'workspace']:
            # Steady usage during work hours
            work_hours = (hour_of_day >= 8) & (hour_of_day <= 18)
            pattern[work_hours] = 1.2
            pattern[~work_hours] = 0.1
        
        return pattern
    
    def _generate_space_utilization(self, zone_occupancy: Dict[str, np.ndarray], 
                                  weather_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate space utilization percentages."""
        space_utilization = {}
        
        # Zone capacities (people per zone)
        zone_capacity = self.config.occupancy_capacity / self.config.num_zones
        
        for zone in range(1, self.config.num_zones + 1):
            occupancy = zone_occupancy[f'zone_{zone}']
            
            # Base utilization
            base_utilization = (occupancy / zone_capacity) * 100
            
            # Account for space usage beyond just occupancy (equipment, storage, etc.)
            space_factor = np.random.uniform(1.1, 1.3, len(occupancy))
            
            utilization = np.clip(base_utilization * space_factor, 0, 100)
            
            space_utilization[f'zone_{zone}_utilization'] = utilization
        
        return space_utilization
    
    def _generate_window_blind_operations(self, weather_df: pd.DataFrame, 
                                        zone_occupancy: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate window and blind operation patterns."""
        n_points = len(weather_df)
        window_operations = {}
        
        # Extract weather variables
        outdoor_temp = weather_df['ambient_temperature_c'].values
        solar_irradiance = weather_df['global_horizontal_irradiance_w_m2'].values
        wind_speed = weather_df['wind_speed_m_s'].values
        hour_of_day = weather_df['timestamp'].dt.hour.values
        
        for zone in range(1, self.config.num_zones + 1):
            zone_occ = zone_occupancy[f'zone_{zone}']
            
            # Window operations
            windows_open = self._calculate_window_operations(
                outdoor_temp, wind_speed, zone_occ, hour_of_day
            )
            
            # Blind operations
            blinds_closed = self._calculate_blind_operations(
                solar_irradiance, outdoor_temp, zone_occ, hour_of_day, zone
            )
            
            window_operations[f'zone_{zone}_windows'] = windows_open
            window_operations[f'zone_{zone}_blinds'] = blinds_closed
        
        return window_operations
    
    def _calculate_window_operations(self, outdoor_temp: np.ndarray, 
                                   wind_speed: np.ndarray, occupancy: np.ndarray,
                                   hour_of_day: np.ndarray) -> np.ndarray:
        """Calculate percentage of windows open."""
        
        # Temperature comfort range for natural ventilation
        comfort_temp_min = 18  # °C
        comfort_temp_max = 26  # °C
        
        # Base probability of opening windows
        temp_factor = np.zeros_like(outdoor_temp)
        
        # Favorable temperature range
        favorable_mask = (outdoor_temp >= comfort_temp_min) & (outdoor_temp <= comfort_temp_max)
        temp_factor[favorable_mask] = 1.0
        
        # Less favorable but acceptable
        acceptable_mask = ((outdoor_temp >= 15) & (outdoor_temp < comfort_temp_min)) | \
                         ((outdoor_temp > comfort_temp_max) & (outdoor_temp <= 30))
        temp_factor[acceptable_mask] = 0.3
        
        # Wind factor (too windy discourages opening)
        wind_factor = np.clip(1 - (wind_speed - 5) / 10, 0, 1)
        
        # Occupancy factor (need people to operate windows)
        occupancy_factor = np.clip(occupancy / 5, 0, 1)
        
        # Time factor (more likely during work hours)
        time_factor = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float) * 0.8 + 0.2
        
        # Combine factors
        window_probability = temp_factor * wind_factor * occupancy_factor * time_factor
        
        # Add persistence (windows don't change state frequently)
        windows_open = np.zeros_like(window_probability)
        current_state = 0.2  # Start with 20% open
        
        for i in range(len(window_probability)):
            # Probability of changing state
            change_prob = 0.1  # 10% chance of change per 15-min period
            
            if np.random.random() < change_prob:
                current_state = window_probability[i]
            
            windows_open[i] = current_state
        
        # Convert to percentage
        return windows_open * 100
    
    def _calculate_blind_operations(self, solar_irradiance: np.ndarray, 
                                  outdoor_temp: np.ndarray, occupancy: np.ndarray,
                                  hour_of_day: np.ndarray, zone: int) -> np.ndarray:
        """Calculate percentage of blinds closed."""
        
        # Zone orientation affects solar exposure
        zone_orientation = (zone - 1) * 360 / self.config.num_zones
        solar_factor = 0.5 + 0.4 * np.cos(np.radians(zone_orientation))
        
        # Solar irradiance factor (close blinds when sunny)
        effective_solar = solar_irradiance * solar_factor
        solar_blind_factor = np.clip(effective_solar / 500, 0, 1)
        
        # Temperature factor (close blinds when hot)
        temp_blind_factor = np.clip((outdoor_temp - 25) / 10, 0, 1)
        
        # Occupancy factor (need people to operate blinds)
        occupancy_factor = np.clip(occupancy / 3, 0, 1)
        
        # Time factor (automatic systems or manual operation during work hours)
        time_factor = ((hour_of_day >= 7) & (hour_of_day <= 19)).astype(float) * 0.9 + 0.1
        
        # Glare avoidance (higher probability when sun is low)
        solar_elevation = 90 - np.abs(hour_of_day - 12) * 7.5  # Simplified
        glare_factor = np.clip((60 - solar_elevation) / 30, 0, 1)
        
        # Combine factors
        blind_close_probability = np.maximum(
            solar_blind_factor * 0.7,
            np.maximum(temp_blind_factor * 0.6, glare_factor * 0.5)
        ) * occupancy_factor * time_factor
        
        # Add persistence
        blinds_closed = np.zeros_like(blind_close_probability)
        current_state = 0.3  # Start with 30% closed
        
        for i in range(len(blind_close_probability)):
            # Probability of changing state
            change_prob = 0.05  # 5% chance of change per 15-min period
            
            if np.random.random() < change_prob:
                current_state = blind_close_probability[i]
            
            blinds_closed[i] = current_state
        
        # Convert to percentage
        return blinds_closed * 100

def main():
    """Demonstrate the IEQ and Occupancy generators."""
    print("🏢 IEQ and Occupancy Data Generators")
    print("=" * 50)
    
    # This would normally import the main config and weather data
    # For demo purposes, create simplified versions
    from iot_building_dataset_generator import BuildingConfig, WeatherGenerator, EnergyConsumptionGenerator
    
    config = BuildingConfig()
    weather_gen = WeatherGenerator(config)
    energy_gen = EnergyConsumptionGenerator(config)
    
    # Generate sample data
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-01-07 23:45:00"  # One week for demo
    
    print("📅 Generating sample data (1 week)...")
    weather_df = weather_gen.generate_weather_data(start_date, end_date)
    
    # Create simple occupancy for energy calculation
    simple_occupancy = pd.DataFrame({
        'timestamp': weather_df['timestamp'],
        'total_occupancy': np.random.randint(50, 200, len(weather_df))
    })
    
    energy_df = energy_gen.generate_energy_data(weather_df, simple_occupancy)
    
    # Generate detailed occupancy patterns
    print("👥 Generating occupancy patterns...")
    occupancy_gen = OccupancyPatternGenerator(config)
    occupancy_df = occupancy_gen.generate_occupancy_data(weather_df)
    
    # Generate IEQ data
    print("🌡️ Generating IEQ data...")
    ieq_gen = IndoorEnvironmentalQualityGenerator(config, weather_df, energy_df)
    ieq_df = ieq_gen.generate_ieq_data(occupancy_df)
    
    # Display sample data
    print("\n📊 Sample Occupancy Data:")
    print(occupancy_df[['timestamp', 'total_occupancy', 'zone_1_occupancy', 
                       'zone_1_utilization_pct', 'zone_1_windows_open_pct']].head())
    
    print("\n📊 Sample IEQ Data:")
    print(ieq_df[['timestamp', 'zone_1_air_temp_c', 'zone_1_co2_ppm', 
                 'zone_1_illuminance_lux', 'zone_1_noise_level_db']].head())
    
    # Statistics
    print("\n📈 Data Summary:")
    print(f"Peak occupancy: {occupancy_df['total_occupancy'].max()} people")
    print(f"Average building utilization: {occupancy_df['building_utilization_pct'].mean():.1f}%")
    print(f"Temperature range: {ieq_df['building_avg_temp_c'].min():.1f}°C to {ieq_df['building_avg_temp_c'].max():.1f}°C")
    print(f"CO2 range: {ieq_df['building_avg_co2_ppm'].min():.0f} to {ieq_df['building_avg_co2_ppm'].max():.0f} ppm")
    
    return occupancy_df, ieq_df

if __name__ == "__main__":
    occupancy_data, ieq_data = main()