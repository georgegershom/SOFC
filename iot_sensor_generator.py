#!/usr/bin/env python3
"""
IoT Sensor Data Generator
Generates realistic real-time IoT sensor data for building monitoring
and digital twin integration
"""

import numpy as np
import pandas as pd
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SensorReading:
    """Individual sensor reading with metadata"""
    sensor_id: str
    timestamp: str
    value: float
    unit: str
    quality: str  # good, fair, poor, bad
    confidence: float  # 0-1
    location: Tuple[float, float, float]  # x, y, z coordinates
    zone: str
    sensor_type: str
    calibration_date: str
    maintenance_due: str

@dataclass
class SensorConfiguration:
    """Sensor configuration and specifications"""
    sensor_id: str
    sensor_type: str
    location: Tuple[float, float, float]
    zone: str
    measurement_range: Tuple[float, float]
    accuracy: float
    resolution: float
    sampling_rate: float  # Hz
    calibration_interval: int  # days
    last_calibration: str
    next_maintenance: str
    manufacturer: str
    model: str
    installation_date: str
    status: str  # active, inactive, maintenance, error

@dataclass
class BuildingZone:
    """Building zone with environmental characteristics"""
    zone_id: str
    name: str
    level: int
    area: float  # m²
    volume: float  # m³
    occupancy: int
    zone_type: str  # office, residential, storage, mechanical
    setpoint_temperature: float  # °C
    setpoint_humidity: float  # %
    ventilation_rate: float  # ACH
    lighting_level: float  # lux
    equipment_load: float  # W/m²

class IoTSensorGenerator:
    """Generator for realistic IoT sensor data"""
    
    def __init__(self, seed: int = 42):
        """Initialize the IoT sensor generator"""
        np.random.seed(seed)
        random.seed(seed)
        self.sensor_types = self._initialize_sensor_types()
        self.manufacturers = ['Honeywell', 'Siemens', 'Johnson Controls', 'Schneider Electric', 
                            'Trane', 'Carrier', 'Daikin', 'Mitsubishi', 'LG', 'Samsung']
        
    def _initialize_sensor_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize sensor types and their characteristics"""
        return {
            'temperature': {
                'unit': '°C',
                'range': (-40, 60),
                'accuracy': 0.1,
                'resolution': 0.01,
                'sampling_rate': 0.1,  # 10 seconds
                'drift_rate': 0.001,  # °C per day
                'noise_std': 0.05,
                'calibration_interval': 365
            },
            'humidity': {
                'unit': '%',
                'range': (0, 100),
                'accuracy': 1.0,
                'resolution': 0.1,
                'sampling_rate': 0.1,
                'drift_rate': 0.01,  # % per day
                'noise_std': 0.5,
                'calibration_interval': 180
            },
            'co2': {
                'unit': 'ppm',
                'range': (300, 5000),
                'accuracy': 10,
                'resolution': 1,
                'sampling_rate': 0.1,
                'drift_rate': 0.1,  # ppm per day
                'noise_std': 5,
                'calibration_interval': 90
            },
            'pressure': {
                'unit': 'Pa',
                'range': (95000, 105000),
                'accuracy': 10,
                'resolution': 1,
                'sampling_rate': 0.1,
                'drift_rate': 0.01,  # Pa per day
                'noise_std': 2,
                'calibration_interval': 365
            },
            'air_velocity': {
                'unit': 'm/s',
                'range': (0, 10),
                'accuracy': 0.05,
                'resolution': 0.01,
                'sampling_rate': 0.2,
                'drift_rate': 0.001,  # m/s per day
                'noise_std': 0.02,
                'calibration_interval': 180
            },
            'light': {
                'unit': 'lux',
                'range': (0, 10000),
                'accuracy': 10,
                'resolution': 1,
                'sampling_rate': 0.1,
                'drift_rate': 0.1,  # lux per day
                'noise_std': 5,
                'calibration_interval': 365
            },
            'occupancy': {
                'unit': 'count',
                'range': (0, 100),
                'accuracy': 1,
                'resolution': 1,
                'sampling_rate': 1.0,  # 1 second
                'drift_rate': 0,
                'noise_std': 0,
                'calibration_interval': 30
            },
            'energy': {
                'unit': 'kWh',
                'range': (0, 1000),
                'accuracy': 0.1,
                'resolution': 0.01,
                'sampling_rate': 0.1,
                'drift_rate': 0.001,  # kWh per day
                'noise_std': 0.05,
                'calibration_interval': 365
            },
            'water_flow': {
                'unit': 'L/min',
                'range': (0, 100),
                'accuracy': 0.5,
                'resolution': 0.1,
                'sampling_rate': 0.1,
                'drift_rate': 0.01,  # L/min per day
                'noise_std': 0.1,
                'calibration_interval': 180
            },
            'vibration': {
                'unit': 'mm/s',
                'range': (0, 50),
                'accuracy': 0.1,
                'resolution': 0.01,
                'sampling_rate': 10,  # 10 Hz
                'drift_rate': 0.001,  # mm/s per day
                'noise_std': 0.05,
                'calibration_interval': 90
            }
        }
    
    def generate_building_zones(self, building_geometry: Dict[str, Any]) -> List[BuildingZone]:
        """Generate building zones based on geometry"""
        zones = []
        num_floors = building_geometry['number_of_floors']
        floor_areas = building_geometry['floor_areas']
        floor_heights = building_geometry['floor_heights']
        building_type = building_geometry['building_type']
        
        # Zone types by building type
        zone_types = {
            'residential': ['living', 'bedroom', 'kitchen', 'bathroom', 'storage'],
            'commercial': ['office', 'conference', 'lobby', 'storage', 'mechanical'],
            'office': ['open_office', 'private_office', 'conference', 'lobby', 'break_room', 'storage'],
            'industrial': ['production', 'warehouse', 'office', 'storage', 'mechanical']
        }
        
        available_zones = zone_types.get(building_type, zone_types['office'])
        
        for level in range(num_floors):
            floor_area = floor_areas[level]
            floor_height = floor_heights[level]
            
            # Divide floor into zones
            num_zones = np.random.randint(2, 6)
            zone_areas = np.random.dirichlet(np.ones(num_zones)) * floor_area
            
            for i, zone_area in enumerate(zone_areas):
                zone_type = np.random.choice(available_zones)
                
                # Generate zone properties based on type
                if zone_type in ['office', 'open_office', 'private_office']:
                    occupancy = int(zone_area / 10)  # 10 m² per person
                    setpoint_temp = np.random.uniform(22, 24)
                    setpoint_humidity = np.random.uniform(40, 60)
                    lighting_level = np.random.uniform(300, 500)
                    equipment_load = np.random.uniform(15, 25)
                elif zone_type in ['conference', 'meeting']:
                    occupancy = int(zone_area / 2)  # 2 m² per person
                    setpoint_temp = np.random.uniform(21, 23)
                    setpoint_humidity = np.random.uniform(45, 55)
                    lighting_level = np.random.uniform(400, 600)
                    equipment_load = np.random.uniform(20, 30)
                elif zone_type in ['storage', 'mechanical']:
                    occupancy = 0
                    setpoint_temp = np.random.uniform(18, 22)
                    setpoint_humidity = np.random.uniform(30, 50)
                    lighting_level = np.random.uniform(100, 200)
                    equipment_load = np.random.uniform(5, 15)
                else:  # residential zones
                    occupancy = np.random.randint(1, 4)
                    setpoint_temp = np.random.uniform(20, 25)
                    setpoint_humidity = np.random.uniform(40, 60)
                    lighting_level = np.random.uniform(200, 400)
                    equipment_load = np.random.uniform(10, 20)
                
                zone = BuildingZone(
                    zone_id=f'zone_{level}_{i}',
                    name=f'{zone_type.title()} {level}-{i}',
                    level=level,
                    area=zone_area,
                    volume=zone_area * floor_height,
                    occupancy=occupancy,
                    zone_type=zone_type,
                    setpoint_temperature=setpoint_temp,
                    setpoint_humidity=setpoint_humidity,
                    ventilation_rate=np.random.uniform(0.5, 2.0),
                    lighting_level=lighting_level,
                    equipment_load=equipment_load
                )
                zones.append(zone)
        
        return zones
    
    def generate_sensor_configurations(self, zones: List[BuildingZone]) -> List[SensorConfiguration]:
        """Generate sensor configurations for building zones"""
        configurations = []
        
        for zone in zones:
            # Determine sensor types for zone
            if zone.zone_type in ['office', 'open_office', 'private_office', 'conference']:
                sensor_types = ['temperature', 'humidity', 'co2', 'light', 'occupancy', 'energy']
            elif zone.zone_type in ['mechanical', 'storage']:
                sensor_types = ['temperature', 'humidity', 'vibration', 'energy']
            elif zone.zone_type in ['living', 'bedroom', 'kitchen', 'bathroom']:
                sensor_types = ['temperature', 'humidity', 'light', 'energy', 'water_flow']
            else:
                sensor_types = ['temperature', 'humidity', 'energy']
            
            # Generate sensors for zone
            num_sensors = np.random.randint(2, 8)
            for i in range(num_sensors):
                sensor_type = np.random.choice(sensor_types)
                sensor_specs = self.sensor_types[sensor_type]
                
                # Generate sensor location within zone
                x = np.random.uniform(-5, 5)  # Relative to zone center
                y = np.random.uniform(-5, 5)
                z = zone.level * 3.0 + np.random.uniform(0.5, 2.5)  # Height
                
                # Generate sensor properties
                measurement_range = sensor_specs['range']
                accuracy = sensor_specs['accuracy']
                resolution = sensor_specs['resolution']
                sampling_rate = sensor_specs['sampling_rate']
                calibration_interval = sensor_specs['calibration_interval']
                
                # Generate dates
                installation_date = (datetime.now() - timedelta(days=np.random.randint(30, 3650))).strftime('%Y-%m-%d')
                last_calibration = (datetime.now() - timedelta(days=np.random.randint(0, calibration_interval))).strftime('%Y-%m-%d')
                next_maintenance = (datetime.now() + timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d')
                
                # Generate status
                status_options = ['active', 'active', 'active', 'maintenance', 'error']  # Weighted
                status = np.random.choice(status_options)
                
                config = SensorConfiguration(
                    sensor_id=str(uuid.uuid4()),
                    sensor_type=sensor_type,
                    location=(x, y, z),
                    zone=zone.zone_id,
                    measurement_range=measurement_range,
                    accuracy=accuracy,
                    resolution=resolution,
                    sampling_rate=sampling_rate,
                    calibration_interval=calibration_interval,
                    last_calibration=last_calibration,
                    next_maintenance=next_maintenance,
                    manufacturer=np.random.choice(self.manufacturers),
                    model=f"{sensor_type.upper()}-{np.random.randint(100, 999)}",
                    installation_date=installation_date,
                    status=status
                )
                configurations.append(config)
        
        return configurations
    
    def generate_sensor_readings(self, configurations: List[SensorConfiguration], 
                               zones: List[BuildingZone], 
                               start_time: datetime, 
                               duration_hours: int = 24) -> List[SensorReading]:
        """Generate realistic sensor readings over time"""
        readings = []
        
        # Create zone lookup
        zone_lookup = {zone.zone_id: zone for zone in zones}
        
        for config in configurations:
            if config.status != 'active':
                continue
                
            zone = zone_lookup[config.zone]
            sensor_specs = self.sensor_types[config.sensor_type]
            
            # Calculate number of readings
            sampling_rate = config.sampling_rate
            total_seconds = duration_hours * 3600
            num_readings = int(total_seconds * sampling_rate)
            
            # Generate time series
            timestamps = [start_time + timedelta(seconds=i/sampling_rate) for i in range(num_readings)]
            
            # Generate base values based on zone and sensor type
            base_value = self._generate_base_value(config.sensor_type, zone)
            
            # Generate time series with realistic patterns
            values = self._generate_time_series(
                base_value, 
                sensor_specs, 
                timestamps, 
                zone, 
                config
            )
            
            # Generate readings
            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                # Calculate quality and confidence
                quality, confidence = self._calculate_quality_confidence(
                    config, timestamp, value, sensor_specs
                )
                
                reading = SensorReading(
                    sensor_id=config.sensor_id,
                    timestamp=timestamp.isoformat(),
                    value=value,
                    unit=sensor_specs['unit'],
                    quality=quality,
                    confidence=confidence,
                    location=config.location,
                    zone=config.zone,
                    sensor_type=config.sensor_type,
                    calibration_date=config.last_calibration,
                    maintenance_due=config.next_maintenance
                )
                readings.append(reading)
        
        return readings
    
    def _generate_base_value(self, sensor_type: str, zone: BuildingZone) -> float:
        """Generate base value for sensor type in zone"""
        if sensor_type == 'temperature':
            # Base temperature with seasonal variation
            base_temp = zone.setpoint_temperature
            seasonal_variation = 5 * np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
            return base_temp + seasonal_variation + np.random.normal(0, 1)
        
        elif sensor_type == 'humidity':
            return zone.setpoint_humidity + np.random.normal(0, 5)
        
        elif sensor_type == 'co2':
            # CO2 increases with occupancy
            base_co2 = 400 + zone.occupancy * 50
            return base_co2 + np.random.normal(0, 20)
        
        elif sensor_type == 'pressure':
            return 101325 + np.random.normal(0, 100)  # Atmospheric pressure
        
        elif sensor_type == 'air_velocity':
            return np.random.uniform(0.1, 0.5)  # Low air velocity
        
        elif sensor_type == 'light':
            # Light varies with time of day
            hour = datetime.now().hour
            if 6 <= hour <= 18:
                return zone.lighting_level + np.random.normal(0, 50)
            else:
                return np.random.uniform(0, 50)  # Night time
        
        elif sensor_type == 'occupancy':
            return zone.occupancy + np.random.randint(-1, 2)
        
        elif sensor_type == 'energy':
            return zone.equipment_load * zone.area / 1000 + np.random.normal(0, 0.1)
        
        elif sensor_type == 'water_flow':
            return np.random.uniform(0, 10)  # Random water flow
        
        elif sensor_type == 'vibration':
            return np.random.uniform(0, 2)  # Low vibration
        
        else:
            return np.random.uniform(0, 100)
    
    def _generate_time_series(self, base_value: float, sensor_specs: Dict[str, Any], 
                            timestamps: List[datetime], zone: BuildingZone, 
                            config: SensorConfiguration) -> List[float]:
        """Generate realistic time series for sensor"""
        values = []
        drift_rate = sensor_specs['drift_rate']
        noise_std = sensor_specs['noise_std']
        
        for i, timestamp in enumerate(timestamps):
            # Base value
            value = base_value
            
            # Add drift over time
            days_since_calibration = (timestamp - datetime.fromisoformat(config.last_calibration)).days
            value += days_since_calibration * drift_rate
            
            # Add daily patterns
            hour = timestamp.hour
            if config.sensor_type == 'temperature':
                # Temperature varies with time of day
                daily_variation = 2 * np.sin(2 * np.pi * (hour - 6) / 24)
                value += daily_variation
            elif config.sensor_type == 'occupancy':
                # Occupancy varies with time of day
                if 8 <= hour <= 18:
                    value *= np.random.uniform(0.8, 1.2)
                else:
                    value *= np.random.uniform(0.1, 0.3)
            elif config.sensor_type == 'light':
                # Light varies with time of day
                if 6 <= hour <= 18:
                    value *= np.random.uniform(0.8, 1.2)
                else:
                    value *= np.random.uniform(0.1, 0.3)
            elif config.sensor_type == 'energy':
                # Energy usage varies with time of day
                if 8 <= hour <= 18:
                    value *= np.random.uniform(0.8, 1.2)
                else:
                    value *= np.random.uniform(0.3, 0.7)
            
            # Add weekly patterns
            weekday = timestamp.weekday()
            if config.sensor_type in ['occupancy', 'energy']:
                if weekday >= 5:  # Weekend
                    value *= 0.3
            
            # Add noise
            noise = np.random.normal(0, noise_std)
            value += noise
            
            # Add occasional spikes or anomalies
            if np.random.random() < 0.01:  # 1% chance of anomaly
                value *= np.random.uniform(2, 5)
            
            # Ensure value is within sensor range
            min_val, max_val = config.measurement_range
            value = np.clip(value, min_val, max_val)
            
            values.append(value)
        
        return values
    
    def _calculate_quality_confidence(self, config: SensorConfiguration, 
                                    timestamp: datetime, value: float, 
                                    sensor_specs: Dict[str, Any]) -> Tuple[str, float]:
        """Calculate data quality and confidence"""
        # Check calibration age
        days_since_calibration = (timestamp - datetime.fromisoformat(config.last_calibration)).days
        calibration_factor = max(0.5, 1 - days_since_calibration / config.calibration_interval)
        
        # Check if maintenance is due
        days_to_maintenance = (datetime.fromisoformat(config.next_maintenance) - timestamp).days
        maintenance_factor = 1.0 if days_to_maintenance > 7 else max(0.3, days_to_maintenance / 7)
        
        # Check value range
        min_val, max_val = config.measurement_range
        range_factor = 1.0 if min_val <= value <= max_val else 0.1
        
        # Calculate overall confidence
        confidence = calibration_factor * maintenance_factor * range_factor
        
        # Determine quality
        if confidence > 0.8:
            quality = 'good'
        elif confidence > 0.6:
            quality = 'fair'
        elif confidence > 0.3:
            quality = 'poor'
        else:
            quality = 'bad'
        
        return quality, confidence
    
    def generate_energy_consumption_data(self, zones: List[BuildingZone], 
                                       start_time: datetime, 
                                       duration_hours: int = 24) -> pd.DataFrame:
        """Generate detailed energy consumption data"""
        data = []
        
        for zone in zones:
            # Base energy consumption
            base_consumption = zone.equipment_load * zone.area / 1000  # kW
            
            # Generate hourly data
            for hour in range(duration_hours):
                timestamp = start_time + timedelta(hours=hour)
                
                # Calculate consumption for each hour
                consumption = base_consumption
                
                # Add time-of-day variation
                if 8 <= timestamp.hour <= 18:
                    consumption *= np.random.uniform(0.8, 1.2)
                else:
                    consumption *= np.random.uniform(0.3, 0.7)
                
                # Add weekly variation
                if timestamp.weekday() >= 5:  # Weekend
                    consumption *= 0.3
                
                # Add random variation
                consumption *= np.random.uniform(0.9, 1.1)
                
                # Add equipment efficiency factor
                efficiency = np.random.uniform(0.7, 0.95)
                consumption /= efficiency
                
                data.append({
                    'timestamp': timestamp,
                    'zone_id': zone.zone_id,
                    'zone_name': zone.name,
                    'zone_type': zone.zone_type,
                    'level': zone.level,
                    'area': zone.area,
                    'occupancy': zone.occupancy,
                    'energy_consumption_kwh': consumption,
                    'efficiency': efficiency,
                    'equipment_load_w_m2': zone.equipment_load
                })
        
        return pd.DataFrame(data)
    
    def generate_occupancy_patterns(self, zones: List[BuildingZone], 
                                  start_time: datetime, 
                                  duration_hours: int = 24) -> pd.DataFrame:
        """Generate realistic occupancy patterns"""
        data = []
        
        for zone in zones:
            # Generate hourly occupancy
            for hour in range(duration_hours):
                timestamp = start_time + timedelta(hours=hour)
                
                # Base occupancy
                occupancy = zone.occupancy
                
                # Add time-of-day variation
                if 8 <= timestamp.hour <= 18:
                    occupancy *= np.random.uniform(0.8, 1.2)
                else:
                    occupancy *= np.random.uniform(0.1, 0.3)
                
                # Add weekly variation
                if timestamp.weekday() >= 5:  # Weekend
                    occupancy *= 0.2
                
                # Add random variation
                occupancy = max(0, int(occupancy * np.random.uniform(0.8, 1.2)))
                
                data.append({
                    'timestamp': timestamp,
                    'zone_id': zone.zone_id,
                    'zone_name': zone.name,
                    'zone_type': zone.zone_type,
                    'level': zone.level,
                    'area': zone.area,
                    'occupancy_count': occupancy,
                    'occupancy_density': occupancy / zone.area if zone.area > 0 else 0,
                    'occupancy_percentage': min(100, occupancy / max(1, zone.occupancy) * 100)
                })
        
        return pd.DataFrame(data)
    
    def export_sensor_data(self, readings: List[SensorReading], filename: str = None) -> str:
        """Export sensor readings to JSON"""
        if filename is None:
            filename = f"sensor_readings_{len(readings)}.json"
        
        data = {
            'readings': [asdict(reading) for reading in readings],
            'metadata': {
                'total_readings': len(readings),
                'generated_at': datetime.now().isoformat(),
                'sensor_types': list(set(r.sensor_type for r in readings)),
                'zones': list(set(r.zone for r in readings))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Sensor data exported to: {filename}")
        return filename
    
    def export_energy_data(self, energy_df: pd.DataFrame, filename: str = None) -> str:
        """Export energy consumption data to CSV"""
        if filename is None:
            filename = f"energy_consumption_{len(energy_df)}.csv"
        
        energy_df.to_csv(filename, index=False)
        print(f"Energy data exported to: {filename}")
        return filename
    
    def export_occupancy_data(self, occupancy_df: pd.DataFrame, filename: str = None) -> str:
        """Export occupancy data to CSV"""
        if filename is None:
            filename = f"occupancy_patterns_{len(occupancy_df)}.csv"
        
        occupancy_df.to_csv(filename, index=False)
        print(f"Occupancy data exported to: {filename}")
        return filename
    
    def visualize_sensor_data(self, readings: List[SensorReading], save_path: str = None):
        """Visualize sensor data over time"""
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in readings])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by sensor type
        sensor_types = df['sensor_type'].unique()
        
        fig = make_subplots(
            rows=len(sensor_types), cols=1,
            subplot_titles=[f'{st.title()} Readings' for st in sensor_types],
            vertical_spacing=0.05
        )
        
        for i, sensor_type in enumerate(sensor_types):
            sensor_data = df[df['sensor_type'] == sensor_type]
            
            fig.add_trace(
                go.Scatter(
                    x=sensor_data['timestamp'],
                    y=sensor_data['value'],
                    mode='lines',
                    name=sensor_type,
                    line=dict(width=1)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title='IoT Sensor Data Over Time',
            height=200 * len(sensor_types),
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Sensor visualization saved to: {save_path}")
        else:
            fig.show()
    
    def visualize_energy_consumption(self, energy_df: pd.DataFrame, save_path: str = None):
        """Visualize energy consumption patterns"""
        fig = px.line(
            energy_df, 
            x='timestamp', 
            y='energy_consumption_kwh',
            color='zone_name',
            title='Energy Consumption by Zone'
        )
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Energy Consumption (kWh)',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Energy visualization saved to: {save_path}")
        else:
            fig.show()

def main():
    """Main function to demonstrate IoT sensor data generation"""
    print("IoT Sensor Data Generator")
    print("=" * 30)
    
    # Initialize generator
    generator = IoTSensorGenerator(seed=42)
    
    # Create sample building geometry
    from building_dna_generator import BuildingDNAGenerator
    dna_generator = BuildingDNAGenerator(seed=42)
    building_geometry = dna_generator.generate_building_geometry('office')
    
    # Generate zones
    zones = generator.generate_building_zones(building_geometry)
    print(f"Generated {len(zones)} building zones")
    
    # Generate sensor configurations
    configurations = generator.generate_sensor_configurations(zones)
    print(f"Generated {len(configurations)} sensor configurations")
    
    # Generate sensor readings
    start_time = datetime.now() - timedelta(hours=24)
    readings = generator.generate_sensor_readings(configurations, zones, start_time, 24)
    print(f"Generated {len(readings)} sensor readings")
    
    # Generate energy and occupancy data
    energy_df = generator.generate_energy_consumption_data(zones, start_time, 24)
    occupancy_df = generator.generate_occupancy_patterns(zones, start_time, 24)
    
    # Export data
    generator.export_sensor_data(readings)
    generator.export_energy_data(energy_df)
    generator.export_occupancy_data(occupancy_df)
    
    # Generate visualizations
    generator.visualize_sensor_data(readings, 'sensor_data_visualization.html')
    generator.visualize_energy_consumption(energy_df, 'energy_consumption_visualization.html')
    
    print("\nIoT sensor data generation complete!")

if __name__ == "__main__":
    main()