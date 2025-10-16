#!/usr/bin/env python3
"""
Building Systems Operation Generator
===================================

Advanced generator for creating realistic building system operation data
including HVAC systems, setpoints, equipment status, and control sequences.

This module generates:
- HVAC system operational parameters
- Temperature and pressure setpoints with schedules
- Equipment status and performance data
- Control valve and damper positions
- Fan speeds and pump operations
- Chiller and boiler operational data
- Building automation system (BAS) data

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
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

class BuildingSystemsGenerator:
    """Generates comprehensive building systems operation data."""
    
    def __init__(self, config, weather_df: pd.DataFrame, energy_df: pd.DataFrame, 
                 occupancy_df: pd.DataFrame, ieq_df: pd.DataFrame):
        self.config = config
        self.weather_df = weather_df
        self.energy_df = energy_df
        self.occupancy_df = occupancy_df
        self.ieq_df = ieq_df
        
    def generate_systems_data(self) -> pd.DataFrame:
        """
        Generate comprehensive building systems data including:
        - HVAC system parameters and setpoints
        - Equipment operational status
        - Control system data (valves, dampers, etc.)
        - Performance metrics and efficiency indicators
        """
        
        timestamps = self.weather_df['timestamp']
        n_points = len(timestamps)
        
        # Generate HVAC setpoints and schedules
        hvac_setpoints = self._generate_hvac_setpoints()
        
        # Generate equipment operational data
        equipment_data = self._generate_equipment_operations()
        
        # Generate control system data
        control_data = self._generate_control_systems()
        
        # Generate performance metrics
        performance_data = self._generate_performance_metrics()
        
        # Combine all systems data
        systems_df = pd.DataFrame({'timestamp': timestamps})
        
        # Add HVAC setpoints
        for key, value in hvac_setpoints.items():
            systems_df[key] = value
        
        # Add equipment data
        for key, value in equipment_data.items():
            systems_df[key] = value
        
        # Add control data
        for key, value in control_data.items():
            systems_df[key] = value
        
        # Add performance data
        for key, value in performance_data.items():
            systems_df[key] = value
        
        return systems_df
    
    def _generate_hvac_setpoints(self) -> Dict[str, np.ndarray]:
        """Generate HVAC setpoints with realistic scheduling."""
        n_points = len(self.weather_df)
        
        # Extract time variables
        hour_of_day = self.weather_df['timestamp'].dt.hour.values
        day_of_week = self.weather_df['timestamp'].dt.dayofweek.values
        month = self.weather_df['timestamp'].dt.month.values
        
        # Extract environmental variables
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        occupancy = self.occupancy_df['total_occupancy'].values
        
        # Generate zone temperature setpoints
        zone_setpoints = self._generate_zone_temperature_setpoints(
            hour_of_day, day_of_week, month, occupancy
        )
        
        # Generate supply air temperature setpoints
        supply_air_setpoints = self._generate_supply_air_setpoints(
            outdoor_temp, zone_setpoints['cooling_setpoint']
        )
        
        # Generate pressure setpoints
        pressure_setpoints = self._generate_pressure_setpoints(occupancy)
        
        # Generate humidity setpoints
        humidity_setpoints = self._generate_humidity_setpoints(month, occupancy)
        
        # Combine all setpoints
        setpoints = {}
        setpoints.update(zone_setpoints)
        setpoints.update(supply_air_setpoints)
        setpoints.update(pressure_setpoints)
        setpoints.update(humidity_setpoints)
        
        return setpoints
    
    def _generate_zone_temperature_setpoints(self, hour_of_day: np.ndarray, 
                                           day_of_week: np.ndarray, month: np.ndarray,
                                           occupancy: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate zone temperature setpoints with scheduling."""
        n_points = len(hour_of_day)
        
        # Base setpoints
        base_cooling_sp = 24.0  # °C
        base_heating_sp = 20.0  # °C
        
        # Seasonal adjustments
        seasonal_cooling_adj = 1.0 + 0.5 * np.cos(2 * np.pi * (month - 7) / 12)
        seasonal_heating_adj = 1.0 - 0.5 * np.cos(2 * np.pi * (month - 1) / 12)
        
        # Occupancy-based scheduling
        occupied_mask = occupancy > (self.config.occupancy_capacity * 0.1)
        
        # Weekday schedule
        weekday_mask = day_of_week < 5
        
        # Work hours (occupied setpoints)
        work_hours_mask = weekday_mask & (hour_of_day >= 7) & (hour_of_day <= 18)
        
        # Initialize setpoints
        cooling_setpoint = np.full(n_points, base_cooling_sp)
        heating_setpoint = np.full(n_points, base_heating_sp)
        
        # Occupied periods - comfort setpoints
        occupied_periods = occupied_mask | work_hours_mask
        cooling_setpoint[occupied_periods] = base_cooling_sp * seasonal_cooling_adj[occupied_periods]
        heating_setpoint[occupied_periods] = base_heating_sp * seasonal_heating_adj[occupied_periods]
        
        # Unoccupied periods - setback
        unoccupied_periods = ~occupied_periods
        cooling_setpoint[unoccupied_periods] = (base_cooling_sp + 3) * seasonal_cooling_adj[unoccupied_periods]  # Setback +3°C
        heating_setpoint[unoccupied_periods] = (base_heating_sp - 3) * seasonal_heating_adj[unoccupied_periods]  # Setback -3°C
        
        # Weekend schedule (reduced hours)
        weekend_mask = day_of_week >= 5
        weekend_occupied = weekend_mask & (hour_of_day >= 9) & (hour_of_day <= 15)
        
        cooling_setpoint[weekend_occupied] = (base_cooling_sp + 1) * seasonal_cooling_adj[weekend_occupied]
        heating_setpoint[weekend_occupied] = (base_heating_sp - 1) * seasonal_heating_adj[weekend_occupied]
        
        # Add control deadband
        deadband = 2.0  # °C
        
        # Generate individual zone setpoints (with variations)
        zone_setpoints = {}
        
        for zone in range(1, min(6, self.config.num_zones + 1)):  # First 5 zones for detailed tracking
            zone_variation = np.random.uniform(-0.5, 0.5, n_points)
            
            zone_setpoints[f'zone_{zone}_cooling_setpoint_c'] = cooling_setpoint + zone_variation
            zone_setpoints[f'zone_{zone}_heating_setpoint_c'] = heating_setpoint + zone_variation
            zone_setpoints[f'zone_{zone}_deadband_c'] = np.full(n_points, deadband)
        
        # Building-wide setpoints
        zone_setpoints['building_cooling_setpoint_c'] = cooling_setpoint
        zone_setpoints['building_heating_setpoint_c'] = heating_setpoint
        zone_setpoints['cooling_setpoint'] = cooling_setpoint  # For backward compatibility
        zone_setpoints['heating_setpoint'] = heating_setpoint
        
        return zone_setpoints
    
    def _generate_supply_air_setpoints(self, outdoor_temp: np.ndarray, 
                                     cooling_setpoint: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate supply air temperature setpoints."""
        
        # Supply air temperature reset based on outdoor temperature
        base_supply_temp = 13.0  # °C
        reset_ratio = 0.3  # °C supply change per °C outdoor change
        
        # Calculate reset supply air temperature
        supply_air_temp_sp = base_supply_temp + (outdoor_temp - 20) * reset_ratio
        
        # Limit supply air temperature range
        supply_air_temp_sp = np.clip(supply_air_temp_sp, 10.0, 18.0)
        
        # Return air temperature setpoint (typically not controlled directly)
        return_air_temp_sp = cooling_setpoint - 8.0  # Typical temperature rise
        
        # Mixed air temperature setpoint (economizer control)
        mixed_air_temp_sp = np.minimum(supply_air_temp_sp + 2.0, outdoor_temp - 2.0)
        
        return {
            'supply_air_temp_setpoint_c': supply_air_temp_sp,
            'return_air_temp_setpoint_c': return_air_temp_sp,
            'mixed_air_temp_setpoint_c': mixed_air_temp_sp
        }
    
    def _generate_pressure_setpoints(self, occupancy: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate pressure setpoints for different zones."""
        n_points = len(occupancy)
        
        # Base pressure setpoints (Pa relative to outdoor)
        base_building_pressure = 12.5  # Pa (positive pressure)
        
        # Occupancy-based adjustment
        occupancy_factor = occupancy / self.config.occupancy_capacity
        pressure_adjustment = occupancy_factor * 5.0  # Up to 5 Pa additional
        
        building_pressure_sp = base_building_pressure + pressure_adjustment
        
        # Zone pressure setpoints (relative to building)
        zone_pressures = {
            'building_static_pressure_setpoint_pa': building_pressure_sp,
            'supply_duct_pressure_setpoint_pa': np.full(n_points, 250.0),  # Pa gauge
            'return_duct_pressure_setpoint_pa': np.full(n_points, -50.0),  # Pa gauge (negative)
        }
        
        # Individual zone pressure setpoints
        for zone in range(1, min(6, self.config.num_zones + 1)):
            zone_pressure_variation = np.random.uniform(-2.0, 2.0, n_points)
            zone_pressures[f'zone_{zone}_pressure_setpoint_pa'] = building_pressure_sp + zone_pressure_variation
        
        return zone_pressures
    
    def _generate_humidity_setpoints(self, month: np.ndarray, 
                                   occupancy: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate humidity setpoints."""
        n_points = len(month)
        
        # Seasonal humidity setpoints
        # Higher humidity allowed in winter, lower in summer
        base_humidity_sp = 45.0  # % RH
        seasonal_variation = 10.0 * np.cos(2 * np.pi * (month - 1) / 12)
        
        humidity_setpoint = base_humidity_sp + seasonal_variation
        
        # Humidity control limits
        max_humidity_sp = np.full(n_points, 60.0)  # % RH
        min_humidity_sp = np.full(n_points, 30.0)  # % RH
        
        # Dehumidification setpoint (summer)
        dehumid_setpoint = np.where(month >= 5, 55.0, 65.0)
        
        return {
            'humidity_setpoint_pct': np.clip(humidity_setpoint, 30, 60),
            'max_humidity_setpoint_pct': max_humidity_sp,
            'min_humidity_setpoint_pct': min_humidity_sp,
            'dehumidification_setpoint_pct': dehumid_setpoint
        }
    
    def _generate_equipment_operations(self) -> Dict[str, np.ndarray]:
        """Generate equipment operational status and parameters."""
        n_points = len(self.weather_df)
        
        # Extract key variables
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        hvac_cooling = self.energy_df['hvac_cooling_kw'].values
        hvac_heating = self.energy_df['hvac_heating_kw'].values
        
        # Chiller operations
        chiller_data = self._generate_chiller_operations(hvac_cooling, outdoor_temp)
        
        # Boiler operations
        boiler_data = self._generate_boiler_operations(hvac_heating, outdoor_temp)
        
        # Air handling unit operations
        ahu_data = self._generate_ahu_operations()
        
        # Pump operations
        pump_data = self._generate_pump_operations(hvac_cooling, hvac_heating)
        
        # Fan operations
        fan_data = self._generate_fan_operations()
        
        # Combine all equipment data
        equipment_data = {}
        equipment_data.update(chiller_data)
        equipment_data.update(boiler_data)
        equipment_data.update(ahu_data)
        equipment_data.update(pump_data)
        equipment_data.update(fan_data)
        
        return equipment_data
    
    def _generate_chiller_operations(self, cooling_load: np.ndarray, 
                                   outdoor_temp: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate chiller operational data."""
        n_points = len(cooling_load)
        
        # Chiller status (on when cooling load > threshold)
        chiller_threshold = 10.0  # kW
        chiller_status = (cooling_load > chiller_threshold).astype(int)
        
        # Chiller capacity utilization
        max_chiller_capacity = self.config.chiller_capacity
        chiller_capacity_pct = np.clip((cooling_load / max_chiller_capacity) * 100, 0, 100)
        
        # Chilled water supply temperature (varies with load and outdoor temp)
        base_chw_supply_temp = 6.0  # °C
        load_adjustment = (chiller_capacity_pct / 100) * 2.0  # Up to 2°C increase at full load
        outdoor_adjustment = (outdoor_temp - 25) * 0.1  # Slight adjustment for outdoor conditions
        
        chw_supply_temp = base_chw_supply_temp + load_adjustment + outdoor_adjustment
        chw_supply_temp = np.clip(chw_supply_temp, 4.0, 10.0)
        
        # Chilled water return temperature
        chw_return_temp = chw_supply_temp + 5.0 + (chiller_capacity_pct / 100) * 2.0
        
        # Condenser water temperatures
        base_cw_supply_temp = outdoor_temp + 3.0  # Cooling tower approach
        cw_supply_temp = np.clip(base_cw_supply_temp, 15.0, 35.0)
        cw_return_temp = cw_supply_temp + 5.0 + (chiller_capacity_pct / 100) * 3.0
        
        # Chiller efficiency (COP)
        base_cop = 5.5
        efficiency_curve = 1 - 0.3 * (1 - chiller_capacity_pct / 100)**2  # Efficiency drops at part load
        chiller_cop = base_cop * efficiency_curve * chiller_status
        
        # Add operational variations and noise
        noise_factor = 0.95 + 0.1 * np.random.random(n_points)
        
        return {
            'chiller_status': chiller_status,
            'chiller_capacity_pct': chiller_capacity_pct * noise_factor,
            'chw_supply_temp_c': chw_supply_temp * chiller_status,
            'chw_return_temp_c': chw_return_temp * chiller_status,
            'condenser_water_supply_temp_c': cw_supply_temp * chiller_status,
            'condenser_water_return_temp_c': cw_return_temp * chiller_status,
            'chiller_cop': chiller_cop,
            'chiller_power_kw': np.where(chiller_status, cooling_load / np.maximum(chiller_cop, 1), 0)
        }
    
    def _generate_boiler_operations(self, heating_load: np.ndarray, 
                                  outdoor_temp: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate boiler operational data."""
        n_points = len(heating_load)
        
        # Boiler status
        boiler_threshold = 5.0  # kW
        boiler_status = (heating_load > boiler_threshold).astype(int)
        
        # Boiler capacity utilization
        max_boiler_capacity = self.config.boiler_capacity
        boiler_capacity_pct = np.clip((heating_load / max_boiler_capacity) * 100, 0, 100)
        
        # Hot water supply temperature (varies with outdoor temp - reset schedule)
        base_hw_supply_temp = 80.0  # °C
        outdoor_reset = (20 - outdoor_temp) * 0.5  # Reset based on outdoor temp
        hw_supply_temp = base_hw_supply_temp + outdoor_reset
        hw_supply_temp = np.clip(hw_supply_temp, 60.0, 90.0)
        
        # Hot water return temperature
        hw_return_temp = hw_supply_temp - 15.0 - (boiler_capacity_pct / 100) * 5.0
        
        # Boiler efficiency
        base_efficiency = 0.85
        efficiency_curve = base_efficiency * (0.7 + 0.3 * boiler_capacity_pct / 100)  # Higher efficiency at higher loads
        boiler_efficiency = efficiency_curve * boiler_status
        
        # Gas consumption
        gas_consumption_kw = np.where(boiler_status, heating_load / np.maximum(boiler_efficiency, 0.1), 0)
        
        # Add operational variations
        noise_factor = 0.95 + 0.1 * np.random.random(n_points)
        
        return {
            'boiler_status': boiler_status,
            'boiler_capacity_pct': boiler_capacity_pct * noise_factor,
            'hw_supply_temp_c': hw_supply_temp * boiler_status,
            'hw_return_temp_c': hw_return_temp * boiler_status,
            'boiler_efficiency_pct': boiler_efficiency * 100,
            'boiler_gas_consumption_kw': gas_consumption_kw
        }
    
    def _generate_ahu_operations(self) -> Dict[str, np.ndarray]:
        """Generate Air Handling Unit operational data."""
        n_points = len(self.weather_df)
        
        # Extract variables
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        occupancy = self.occupancy_df['total_occupancy'].values
        hvac_fans_kw = self.energy_df['hvac_fans_kw'].values
        
        ahu_data = {}
        
        # Generate data for each AHU
        for ahu_num in range(1, self.config.num_ahu + 1):
            # AHU status based on schedule and demand
            ahu_status = self._get_ahu_status(ahu_num, occupancy)
            
            # Supply and return air temperatures
            supply_air_temp = self._calculate_ahu_supply_temp(ahu_num, outdoor_temp, ahu_status)
            return_air_temp = self._calculate_ahu_return_temp(ahu_num, supply_air_temp, ahu_status)
            mixed_air_temp = self._calculate_mixed_air_temp(outdoor_temp, return_air_temp, ahu_status)
            
            # Air flow rates
            supply_airflow, return_airflow = self._calculate_ahu_airflows(ahu_num, occupancy, ahu_status)
            
            # Fan speeds
            supply_fan_speed, return_fan_speed = self._calculate_fan_speeds(
                supply_airflow, return_airflow, ahu_status
            )
            
            # Filter status
            filter_pressure_drop = self._calculate_filter_pressure_drop(supply_airflow, ahu_num)
            
            ahu_data.update({
                f'ahu_{ahu_num}_status': ahu_status,
                f'ahu_{ahu_num}_supply_air_temp_c': supply_air_temp,
                f'ahu_{ahu_num}_return_air_temp_c': return_air_temp,
                f'ahu_{ahu_num}_mixed_air_temp_c': mixed_air_temp,
                f'ahu_{ahu_num}_supply_airflow_m3_s': supply_airflow,
                f'ahu_{ahu_num}_return_airflow_m3_s': return_airflow,
                f'ahu_{ahu_num}_supply_fan_speed_pct': supply_fan_speed,
                f'ahu_{ahu_num}_return_fan_speed_pct': return_fan_speed,
                f'ahu_{ahu_num}_filter_pressure_drop_pa': filter_pressure_drop
            })
        
        return ahu_data
    
    def _get_ahu_status(self, ahu_num: int, occupancy: np.ndarray) -> np.ndarray:
        """Get AHU operational status."""
        # Different AHUs serve different zones/schedules
        hour_of_day = self.weather_df['timestamp'].dt.hour.values
        day_of_week = self.weather_df['timestamp'].dt.dayofweek.values
        
        # Base schedule
        weekday_mask = day_of_week < 5
        work_hours = (hour_of_day >= 6) & (hour_of_day <= 19)
        
        # Occupancy-based operation
        occupancy_threshold = self.config.occupancy_capacity * 0.05  # 5% occupancy threshold
        occupancy_operation = occupancy > occupancy_threshold
        
        # AHU-specific variations
        if ahu_num <= 2:
            # Primary AHUs - follow main schedule
            status = (weekday_mask & work_hours) | occupancy_operation
        else:
            # Secondary AHUs - reduced schedule
            reduced_hours = (hour_of_day >= 8) & (hour_of_day <= 17)
            status = (weekday_mask & reduced_hours) | (occupancy > occupancy_threshold * 2)
        
        return status.astype(int)
    
    def _calculate_ahu_supply_temp(self, ahu_num: int, outdoor_temp: np.ndarray, 
                                 status: np.ndarray) -> np.ndarray:
        """Calculate AHU supply air temperature."""
        # Base supply temperature with reset
        base_supply_temp = 13.0 + (ahu_num - 1) * 0.5  # Slight variation between AHUs
        reset_factor = 0.2
        
        supply_temp = base_supply_temp + (outdoor_temp - 20) * reset_factor
        supply_temp = np.clip(supply_temp, 10.0, 18.0)
        
        return supply_temp * status
    
    def _calculate_ahu_return_temp(self, ahu_num: int, supply_temp: np.ndarray, 
                                 status: np.ndarray) -> np.ndarray:
        """Calculate AHU return air temperature."""
        # Return air temperature based on zone conditions
        zone_temp_rise = 8.0 + np.random.uniform(-1, 1, len(supply_temp))
        return_temp = supply_temp + zone_temp_rise
        
        return return_temp * status
    
    def _calculate_mixed_air_temp(self, outdoor_temp: np.ndarray, 
                                return_temp: np.ndarray, status: np.ndarray) -> np.ndarray:
        """Calculate mixed air temperature (economizer operation)."""
        # Economizer damper position (simplified)
        economizer_position = np.clip((25 - outdoor_temp) / 10, 0, 1)
        
        mixed_temp = (outdoor_temp * economizer_position + 
                     return_temp * (1 - economizer_position))
        
        return mixed_temp * status
    
    def _calculate_ahu_airflows(self, ahu_num: int, occupancy: np.ndarray, 
                              status: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate AHU supply and return airflows."""
        # Base airflow per AHU
        base_airflow = 2.0  # m³/s per AHU
        
        # Occupancy-based demand
        occupancy_factor = occupancy / self.config.occupancy_capacity
        demand_airflow = base_airflow * (0.3 + 0.7 * occupancy_factor)
        
        # AHU-specific capacity
        ahu_capacity_factor = 1.0 if ahu_num <= 2 else 0.8
        
        supply_airflow = demand_airflow * ahu_capacity_factor * status
        return_airflow = supply_airflow * 0.9  # 10% building pressurization
        
        return supply_airflow, return_airflow
    
    def _calculate_fan_speeds(self, supply_airflow: np.ndarray, 
                            return_airflow: np.ndarray, status: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate fan speeds based on airflow demand."""
        # Fan speed is proportional to airflow (simplified)
        max_airflow = 3.0  # m³/s
        
        supply_fan_speed = (supply_airflow / max_airflow) * 100 * status
        return_fan_speed = (return_airflow / max_airflow) * 100 * status
        
        # Minimum speed when running
        supply_fan_speed = np.where(status, np.maximum(supply_fan_speed, 30), 0)
        return_fan_speed = np.where(status, np.maximum(return_fan_speed, 25), 0)
        
        return supply_fan_speed, return_fan_speed
    
    def _calculate_filter_pressure_drop(self, airflow: np.ndarray, ahu_num: int) -> np.ndarray:
        """Calculate filter pressure drop."""
        # Base pressure drop increases with airflow and filter loading
        base_pressure_drop = 50.0  # Pa
        
        # Airflow effect (quadratic relationship)
        max_airflow = 3.0
        airflow_factor = (airflow / max_airflow) ** 2
        
        # Filter loading over time (simplified - increases over days)
        days_since_start = (self.weather_df['timestamp'] - self.weather_df['timestamp'].iloc[0]).dt.days
        loading_factor = 1 + (days_since_start % 90) / 90 * 0.5  # 50% increase over 90 days
        
        pressure_drop = base_pressure_drop * (1 + airflow_factor) * loading_factor
        
        return pressure_drop
    
    def _generate_pump_operations(self, cooling_load: np.ndarray, 
                                heating_load: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate pump operational data."""
        n_points = len(cooling_load)
        
        # Chilled water pumps
        chw_pump_status = (cooling_load > 5.0).astype(int)
        chw_pump_speed = np.clip((cooling_load / self.config.chiller_capacity) * 100, 0, 100) * chw_pump_status
        chw_pump_speed = np.where(chw_pump_status, np.maximum(chw_pump_speed, 30), 0)
        
        # Hot water pumps
        hw_pump_status = (heating_load > 3.0).astype(int)
        hw_pump_speed = np.clip((heating_load / self.config.boiler_capacity) * 100, 0, 100) * hw_pump_status
        hw_pump_speed = np.where(hw_pump_status, np.maximum(hw_pump_speed, 25), 0)
        
        # Condenser water pumps
        cw_pump_status = chw_pump_status  # Same as chiller operation
        cw_pump_speed = chw_pump_speed * 1.1  # Slightly higher flow rate
        
        return {
            'chw_pump_status': chw_pump_status,
            'chw_pump_speed_pct': chw_pump_speed,
            'hw_pump_status': hw_pump_status,
            'hw_pump_speed_pct': hw_pump_speed,
            'cw_pump_status': cw_pump_status,
            'cw_pump_speed_pct': np.clip(cw_pump_speed, 0, 100)
        }
    
    def _generate_fan_operations(self) -> Dict[str, np.ndarray]:
        """Generate fan operational data."""
        n_points = len(self.weather_df)
        
        # Extract HVAC fan energy
        total_fan_energy = self.energy_df['hvac_fans_kw'].values
        
        # Distribute among different fan types
        supply_fan_energy = total_fan_energy * 0.6  # 60% supply fans
        return_fan_energy = total_fan_energy * 0.25  # 25% return fans
        exhaust_fan_energy = total_fan_energy * 0.15  # 15% exhaust fans
        
        # Convert energy to operational parameters
        # Assuming 1 kW = approximately 50% fan speed
        supply_fan_speed = np.clip(supply_fan_energy * 50, 0, 100)
        return_fan_speed = np.clip(return_fan_energy * 50, 0, 100)
        exhaust_fan_speed = np.clip(exhaust_fan_energy * 50, 0, 100)
        
        # Fan status
        supply_fan_status = (supply_fan_speed > 5).astype(int)
        return_fan_status = (return_fan_speed > 5).astype(int)
        exhaust_fan_status = (exhaust_fan_speed > 5).astype(int)
        
        return {
            'supply_fan_status': supply_fan_status,
            'supply_fan_speed_pct': supply_fan_speed,
            'return_fan_status': return_fan_status,
            'return_fan_speed_pct': return_fan_speed,
            'exhaust_fan_status': exhaust_fan_status,
            'exhaust_fan_speed_pct': exhaust_fan_speed
        }
    
    def _generate_control_systems(self) -> Dict[str, np.ndarray]:
        """Generate control system data (valves, dampers, etc.)."""
        n_points = len(self.weather_df)
        
        # Extract key variables
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        cooling_load = self.energy_df['hvac_cooling_kw'].values
        heating_load = self.energy_df['hvac_heating_kw'].values
        
        # Cooling valve positions
        cooling_valve_data = self._generate_cooling_valve_positions(cooling_load)
        
        # Heating valve positions
        heating_valve_data = self._generate_heating_valve_positions(heating_load)
        
        # Damper positions
        damper_data = self._generate_damper_positions(outdoor_temp)
        
        # Variable frequency drives (VFDs)
        vfd_data = self._generate_vfd_data()
        
        # Combine all control data
        control_data = {}
        control_data.update(cooling_valve_data)
        control_data.update(heating_valve_data)
        control_data.update(damper_data)
        control_data.update(vfd_data)
        
        return control_data
    
    def _generate_cooling_valve_positions(self, cooling_load: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate cooling valve positions."""
        # Chilled water valve position based on cooling load
        max_cooling = self.config.chiller_capacity
        chw_valve_position = np.clip((cooling_load / max_cooling) * 100, 0, 100)
        
        # Add control variations and minimum position
        chw_valve_position = np.where(cooling_load > 5, 
                                    np.maximum(chw_valve_position, 10), 0)
        
        # Zone cooling valves (distribute load)
        zone_valves = {}
        for zone in range(1, min(6, self.config.num_zones + 1)):
            zone_variation = np.random.uniform(0.8, 1.2, len(cooling_load))
            zone_valve_pos = chw_valve_position * zone_variation / self.config.num_zones
            zone_valves[f'zone_{zone}_cooling_valve_pct'] = np.clip(zone_valve_pos, 0, 100)
        
        return {
            'chw_valve_position_pct': chw_valve_position,
            **zone_valves
        }
    
    def _generate_heating_valve_positions(self, heating_load: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate heating valve positions."""
        # Hot water valve position based on heating load
        max_heating = self.config.boiler_capacity
        hw_valve_position = np.clip((heating_load / max_heating) * 100, 0, 100)
        
        # Add control variations
        hw_valve_position = np.where(heating_load > 3, 
                                   np.maximum(hw_valve_position, 8), 0)
        
        # Zone heating valves
        zone_valves = {}
        for zone in range(1, min(6, self.config.num_zones + 1)):
            zone_variation = np.random.uniform(0.8, 1.2, len(heating_load))
            zone_valve_pos = hw_valve_position * zone_variation / self.config.num_zones
            zone_valves[f'zone_{zone}_heating_valve_pct'] = np.clip(zone_valve_pos, 0, 100)
        
        return {
            'hw_valve_position_pct': hw_valve_position,
            **zone_valves
        }
    
    def _generate_damper_positions(self, outdoor_temp: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate damper positions for economizer and ventilation control."""
        n_points = len(outdoor_temp)
        
        # Outdoor air damper (economizer control)
        # Open when outdoor temp is favorable (15-22°C)
        economizer_enable = (outdoor_temp >= 15) & (outdoor_temp <= 22)
        
        # Damper position based on outdoor temperature
        oa_damper_position = np.where(
            economizer_enable,
            np.clip(100 - (outdoor_temp - 15) * 10, 20, 90),  # Variable position
            20  # Minimum outdoor air
        )
        
        # Return air damper (opposite of outdoor air)
        ra_damper_position = 100 - oa_damper_position
        
        # Exhaust air damper
        exhaust_damper_position = oa_damper_position * 0.9  # Slightly less than OA
        
        # Zone dampers (VAV terminal units)
        zone_dampers = {}
        for zone in range(1, min(6, self.config.num_zones + 1)):
            # Zone damper based on zone load (simplified)
            zone_load_factor = np.random.uniform(0.3, 1.0, n_points)
            zone_damper_pos = zone_load_factor * 100
            zone_dampers[f'zone_{zone}_damper_position_pct'] = zone_damper_pos
        
        return {
            'outdoor_air_damper_pct': oa_damper_position,
            'return_air_damper_pct': ra_damper_position,
            'exhaust_air_damper_pct': exhaust_damper_position,
            **zone_dampers
        }
    
    def _generate_vfd_data(self) -> Dict[str, np.ndarray]:
        """Generate Variable Frequency Drive data."""
        n_points = len(self.weather_df)
        
        # Extract fan and pump speeds from previous calculations
        # VFD frequency is proportional to speed
        
        # Supply fan VFD
        supply_fan_energy = self.energy_df['hvac_fans_kw'].values * 0.6
        supply_fan_vfd_freq = np.clip(30 + supply_fan_energy * 10, 30, 60)  # 30-60 Hz
        
        # Chilled water pump VFD
        cooling_load = self.energy_df['hvac_cooling_kw'].values
        chw_pump_vfd_freq = np.clip(25 + (cooling_load / self.config.chiller_capacity) * 35, 25, 60)
        
        # Hot water pump VFD
        heating_load = self.energy_df['hvac_heating_kw'].values
        hw_pump_vfd_freq = np.clip(25 + (heating_load / self.config.boiler_capacity) * 35, 25, 60)
        
        return {
            'supply_fan_vfd_frequency_hz': supply_fan_vfd_freq,
            'chw_pump_vfd_frequency_hz': chw_pump_vfd_freq,
            'hw_pump_vfd_frequency_hz': hw_pump_vfd_freq
        }
    
    def _generate_performance_metrics(self) -> Dict[str, np.ndarray]:
        """Generate system performance metrics."""
        n_points = len(self.weather_df)
        
        # Extract key variables
        total_energy = self.energy_df['total_electricity_kw'].values
        hvac_energy = (self.energy_df['hvac_cooling_kw'].values + 
                      self.energy_df['hvac_heating_kw'].values + 
                      self.energy_df['hvac_fans_kw'].values + 
                      self.energy_df['hvac_pumps_kw'].values)
        
        outdoor_temp = self.weather_df['ambient_temperature_c'].values
        indoor_temp = self.ieq_df['building_avg_temp_c'].values
        
        # HVAC efficiency metrics
        hvac_efficiency = np.where(hvac_energy > 0, 
                                 (hvac_energy * 0.8) / hvac_energy * 100, 100)  # Simplified efficiency
        
        # Energy performance metrics
        energy_intensity = total_energy / self.config.floor_area  # kW/m²
        
        # Thermal performance
        temperature_deviation = np.abs(indoor_temp - 22.0)  # Deviation from ideal
        
        # System utilization
        system_utilization = np.clip(hvac_energy / (hvac_energy.max() + 1) * 100, 0, 100)
        
        # Demand response potential (simplified)
        demand_response_potential = np.clip(100 - system_utilization, 0, 50)  # Up to 50% DR potential
        
        return {
            'hvac_system_efficiency_pct': hvac_efficiency,
            'building_energy_intensity_kw_m2': energy_intensity,
            'thermal_comfort_deviation_c': temperature_deviation,
            'hvac_system_utilization_pct': system_utilization,
            'demand_response_potential_pct': demand_response_potential,
            'system_cop': np.where(hvac_energy > 0, hvac_energy * 0.8 / hvac_energy, 0)
        }

def main():
    """Demonstrate the Building Systems generator."""
    print("🏢 Building Systems Operation Generator")
    print("=" * 50)
    
    # Import required modules
    from iot_building_dataset_generator import BuildingConfig, WeatherGenerator, EnergyConsumptionGenerator
    from ieq_occupancy_generators import OccupancyPatternGenerator, IndoorEnvironmentalQualityGenerator
    
    config = BuildingConfig()
    
    # Generate sample data (1 week for demo)
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-01-07 23:45:00"
    
    print("📅 Generating prerequisite data...")
    
    # Weather data
    weather_gen = WeatherGenerator(config)
    weather_df = weather_gen.generate_weather_data(start_date, end_date)
    
    # Occupancy data
    occupancy_gen = OccupancyPatternGenerator(config)
    occupancy_df = occupancy_gen.generate_occupancy_data(weather_df)
    
    # Energy data
    energy_gen = EnergyConsumptionGenerator(config)
    energy_df = energy_gen.generate_energy_data(weather_df, occupancy_df)
    
    # IEQ data
    ieq_gen = IndoorEnvironmentalQualityGenerator(config, weather_df, energy_df)
    ieq_df = ieq_gen.generate_ieq_data(occupancy_df)
    
    # Building systems data
    print("🔧 Generating building systems data...")
    systems_gen = BuildingSystemsGenerator(config, weather_df, energy_df, occupancy_df, ieq_df)
    systems_df = systems_gen.generate_systems_data()
    
    # Display sample data
    print("\n📊 Sample Building Systems Data:")
    print(systems_df[['timestamp', 'building_cooling_setpoint_c', 'chiller_status', 
                     'ahu_1_supply_air_temp_c', 'chw_valve_position_pct']].head())
    
    # Statistics
    print("\n📈 Systems Data Summary:")
    print(f"Chiller operating hours: {systems_df['chiller_status'].sum() * 0.25:.1f} hours")
    print(f"Average cooling setpoint: {systems_df['building_cooling_setpoint_c'].mean():.1f}°C")
    print(f"Peak chiller capacity: {systems_df['chiller_capacity_pct'].max():.1f}%")
    print(f"Average HVAC efficiency: {systems_df['hvac_system_efficiency_pct'].mean():.1f}%")
    
    return systems_df

if __name__ == "__main__":
    systems_data = main()