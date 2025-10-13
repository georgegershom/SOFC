"""
Synthetic sensor data generator for SOFC digital twin.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .base_generator import BaseGenerator


class SensorGenerator(BaseGenerator):
    """
    Generator for synthetic sensor data including:
    - Thermocouple data
    - IR camera data
    - Strain gauge data
    - Voltage/current sensors
    - Flow sensors
    """
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        super().__init__(config_path)
        self.sensors = self.config['sensors']
        self.time_duration = self.config['generation']['time_duration']
        self.time_step = self.config['generation']['time_step']
        self.time_points = np.arange(0, self.time_duration, self.time_step)
        self.noise_level = self.config['generation']['noise_level']
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete synthetic sensor dataset."""
        print("Generating synthetic sensor data...")
        
        data = {
            'thermocouples': self._generate_thermocouple_data(),
            'ir_camera': self._generate_ir_camera_data(),
            'strain_gauges': self._generate_strain_gauge_data(),
            'voltage_current': self._generate_voltage_current_data(),
            'flow_sensors': self._generate_flow_sensor_data(),
            'pressure_sensors': self._generate_pressure_sensor_data(),
            'gas_analyzer': self._generate_gas_analyzer_data()
        }
        
        return data
    
    def _generate_thermocouple_data(self) -> Dict[str, Any]:
        """Generate thermocouple sensor data."""
        print("  Generating thermocouple data...")
        
        n_thermocouples = self.sensors['thermocouples']['count']
        accuracy = self.sensors['thermocouples']['accuracy']
        sampling_rate = self.sensors['thermocouples']['sampling_rate']
        
        # Generate time series for each thermocouple
        thermocouple_data = {}
        
        for i in range(n_thermocouples):
            # Generate temperature profile
            temperature = self._generate_temperature_profile(i)
            
            # Add sensor noise
            noise = np.random.normal(0, accuracy, len(temperature))
            temperature += noise
            
            # Add drift (sensor aging)
            drift = 0.1 * accuracy * self.time_points / 3600  # 0.1°C/hour drift
            temperature += drift
            
            # Add calibration error
            calibration_error = 0.5 * accuracy * np.random.normal(0, 1)
            temperature += calibration_error
            
            thermocouple_data[f'TC_{i+1:02d}'] = {
                'temperature': temperature,
                'accuracy': accuracy,
                'sampling_rate': sampling_rate,
                'location': self._get_thermocouple_location(i),
                'time_points': self.time_points
            }
        
        return thermocouple_data
    
    def _generate_temperature_profile(self, sensor_id: int) -> np.ndarray:
        """Generate temperature profile for a specific thermocouple."""
        # Base temperature
        base_temp = 750  # °C
        
        # Time-dependent variations
        time_variation = 20 * np.sin(2 * np.pi * self.time_points / 3600)  # 1 hour cycle
        
        # Spatial variations based on sensor location
        spatial_variation = 10 * np.sin(2 * np.pi * sensor_id / 8)  # Different for each sensor
        
        # Random fluctuations
        random_fluctuation = 5 * np.random.normal(0, 1, len(self.time_points))
        
        # Step changes (simulating load changes)
        step_changes = np.zeros_like(self.time_points)
        step_times = [1800, 3600, 5400]  # 30 min, 1 hour, 1.5 hours
        for step_time in step_times:
            if step_time < self.time_duration:
                step_mask = self.time_points >= step_time
                step_changes[step_mask] += 10 * np.random.normal(0, 1)
        
        # Combine all effects
        temperature = base_temp + time_variation + spatial_variation + random_fluctuation + step_changes
        
        return temperature
    
    def _get_thermocouple_location(self, sensor_id: int) -> Dict[str, float]:
        """Get thermocouple location coordinates."""
        # Define strategic locations
        locations = [
            {'x': 0.02, 'y': 0.02, 'z': 0.005, 'description': 'inlet_corner'},
            {'x': 0.08, 'y': 0.02, 'z': 0.005, 'description': 'outlet_corner'},
            {'x': 0.05, 'y': 0.05, 'z': 0.005, 'description': 'center'},
            {'x': 0.02, 'y': 0.08, 'z': 0.005, 'description': 'inlet_corner_2'},
            {'x': 0.08, 'y': 0.08, 'z': 0.005, 'description': 'outlet_corner_2'},
            {'x': 0.05, 'y': 0.02, 'z': 0.005, 'description': 'inlet_center'},
            {'x': 0.05, 'y': 0.08, 'z': 0.005, 'description': 'outlet_center'},
            {'x': 0.02, 'y': 0.05, 'z': 0.005, 'description': 'side_center'},
            {'x': 0.08, 'y': 0.05, 'z': 0.005, 'description': 'side_center_2'},
            {'x': 0.05, 'y': 0.05, 'z': 0.002, 'description': 'center_shallow'},
            {'x': 0.05, 'y': 0.05, 'z': 0.008, 'description': 'center_deep'},
            {'x': 0.03, 'y': 0.03, 'z': 0.005, 'description': 'quarter_1'},
            {'x': 0.07, 'y': 0.03, 'z': 0.005, 'description': 'quarter_2'},
            {'x': 0.03, 'y': 0.07, 'z': 0.005, 'description': 'quarter_3'},
            {'x': 0.07, 'y': 0.07, 'z': 0.005, 'description': 'quarter_4'},
            {'x': 0.05, 'y': 0.05, 'z': 0.005, 'description': 'center_duplicate'}
        ]
        
        return locations[sensor_id % len(locations)]
    
    def _generate_ir_camera_data(self) -> Dict[str, Any]:
        """Generate IR camera data."""
        print("  Generating IR camera data...")
        
        resolution = self.sensors['ir_camera']['resolution']
        temperature_range = self.sensors['ir_camera']['temperature_range']
        accuracy = self.sensors['ir_camera']['accuracy']
        sampling_rate = self.sensors['ir_camera']['sampling_rate']
        
        # Generate IR images at lower sampling rate
        ir_time_points = self.time_points[::int(1/sampling_rate)]
        n_images = len(ir_time_points)
        
        # Generate temperature field images
        temperature_images = np.zeros((n_images, resolution[0], resolution[1]))
        
        for i, t in enumerate(ir_time_points):
            # Generate 2D temperature field
            temp_field = self._generate_2d_temperature_field(t, resolution)
            
            # Add IR camera noise
            noise = np.random.normal(0, accuracy, temp_field.shape)
            temp_field += noise
            
            # Clamp to temperature range
            temp_field = np.clip(temp_field, temperature_range[0], temperature_range[1])
            
            temperature_images[i] = temp_field
        
        return {
            'temperature_images': temperature_images,
            'resolution': resolution,
            'temperature_range': temperature_range,
            'accuracy': accuracy,
            'sampling_rate': sampling_rate,
            'time_points': ir_time_points
        }
    
    def _generate_2d_temperature_field(self, time: float, resolution: Tuple[int, int]) -> np.ndarray:
        """Generate 2D temperature field at specific time."""
        nx, ny = resolution
        
        # Create coordinate grids
        x = np.linspace(0, 0.1, nx)  # 10 cm
        y = np.linspace(0, 0.1, ny)  # 10 cm
        X, Y = np.meshgrid(x, y)
        
        # Base temperature
        base_temp = 750 + 30 * np.sin(2 * np.pi * time / 3600)
        
        # Spatial temperature distribution
        # Hot spot near center
        hot_spot = 50 * np.exp(-((X - 0.05)**2 + (Y - 0.05)**2) / 0.01)
        
        # Inlet/outlet gradient
        inlet_gradient = 20 * (1 - X / 0.1)
        outlet_gradient = 15 * (X / 0.1)
        
        # Random thermal fluctuations
        noise = 5 * np.random.normal(0, 1, (nx, ny))
        
        # Combine all effects
        temperature_field = base_temp + hot_spot + inlet_gradient + outlet_gradient + noise
        
        return temperature_field
    
    def _generate_strain_gauge_data(self) -> Dict[str, Any]:
        """Generate strain gauge sensor data."""
        print("  Generating strain gauge data...")
        
        n_gauges = self.sensors['strain_gauges']['count']
        accuracy = self.sensors['strain_gauges']['accuracy']
        sampling_rate = self.sensors['strain_gauges']['sampling_rate']
        
        strain_gauge_data = {}
        
        for i in range(n_gauges):
            # Generate strain profile
            strain = self._generate_strain_profile(i)
            
            # Add sensor noise
            noise = np.random.normal(0, accuracy, len(strain))
            strain += noise
            
            # Add drift
            drift = 0.01 * accuracy * self.time_points / 3600  # 0.01 strain/hour drift
            strain += drift
            
            # Add calibration error
            calibration_error = 0.1 * accuracy * np.random.normal(0, 1)
            strain += calibration_error
            
            strain_gauge_data[f'strain_gauge_{i+1:02d}'] = {
                'strain': strain,
                'accuracy': accuracy,
                'sampling_rate': sampling_rate,
                'location': self._get_strain_gauge_location(i),
                'time_points': self.time_points
            }
        
        return strain_gauge_data
    
    def _generate_strain_profile(self, sensor_id: int) -> np.ndarray:
        """Generate strain profile for a specific strain gauge."""
        # Base strain
        base_strain = 100e-6  # 100 microstrain
        
        # Thermal expansion strain
        thermal_strain = 12e-6 * (self.time_points / 3600)  # Linear increase
        
        # Mechanical strain (sinusoidal)
        mechanical_strain = 50e-6 * np.sin(2 * np.pi * self.time_points / 1800)
        
        # Sensor-specific variations
        sensor_variation = 10e-6 * np.sin(2 * np.pi * sensor_id / 4)
        
        # Random fluctuations
        random_fluctuation = 5e-6 * np.random.normal(0, 1, len(self.time_points))
        
        # Combine all effects
        strain = base_strain + thermal_strain + mechanical_strain + sensor_variation + random_fluctuation
        
        return strain
    
    def _get_strain_gauge_location(self, sensor_id: int) -> Dict[str, float]:
        """Get strain gauge location coordinates."""
        # Define locations on interconnects
        locations = [
            {'x': 0.05, 'y': 0.05, 'z': 0.001, 'orientation': 'x', 'description': 'center_x'},
            {'x': 0.05, 'y': 0.05, 'z': 0.001, 'orientation': 'y', 'description': 'center_y'},
            {'x': 0.02, 'y': 0.05, 'z': 0.001, 'orientation': 'x', 'description': 'inlet_x'},
            {'x': 0.08, 'y': 0.05, 'z': 0.001, 'orientation': 'x', 'description': 'outlet_x'},
            {'x': 0.05, 'y': 0.02, 'z': 0.001, 'orientation': 'y', 'description': 'inlet_y'},
            {'x': 0.05, 'y': 0.08, 'z': 0.001, 'orientation': 'y', 'description': 'outlet_y'},
            {'x': 0.03, 'y': 0.03, 'z': 0.001, 'orientation': 'x', 'description': 'corner_1_x'},
            {'x': 0.07, 'y': 0.07, 'z': 0.001, 'orientation': 'y', 'description': 'corner_2_y'}
        ]
        
        return locations[sensor_id % len(locations)]
    
    def _generate_voltage_current_data(self) -> Dict[str, Any]:
        """Generate voltage and current sensor data."""
        print("  Generating voltage/current data...")
        
        # Generate current density profile
        current_density = self._generate_current_density_profile()
        
        # Generate voltage profile
        voltage = self._generate_voltage_profile(current_density)
        
        # Add sensor noise
        current_noise = 0.01 * np.random.normal(0, 1, len(current_density))
        voltage_noise = 0.001 * np.random.normal(0, 1, len(voltage))
        
        current_density += current_noise
        voltage += voltage_noise
        
        # Calculate power
        power = current_density * voltage
        
        return {
            'current_density': current_density,
            'voltage': voltage,
            'power': power,
            'time_points': self.time_points
        }
    
    def _generate_current_density_profile(self) -> np.ndarray:
        """Generate current density profile."""
        # Start at low current, ramp up, then vary
        current_range = [0.1, 1.0]  # A/cm²
        
        # Ramp up phase
        ramp_time = 0.2 * self.time_duration
        ramp_mask = self.time_points <= ramp_time
        current_density = np.zeros_like(self.time_points)
        current_density[ramp_mask] = (current_range[1] - current_range[0]) * self.time_points[ramp_mask] / ramp_time
        
        # Steady operation with variations
        steady_mask = self.time_points > ramp_time
        base_current = current_range[1] * 0.8
        variation = 0.1 * current_range[1] * np.sin(2 * np.pi * self.time_points[steady_mask] / 1800)
        noise = 0.05 * current_range[1] * np.random.normal(0, 1, np.sum(steady_mask))
        
        current_density[steady_mask] = base_current + variation + noise
        current_density = np.clip(current_density, current_range[0], current_range[1])
        
        return current_density
    
    def _generate_voltage_profile(self, current_density: np.ndarray) -> np.ndarray:
        """Generate voltage profile based on current density."""
        # Open circuit voltage
        E_ocv = 1.1  # V
        
        # Overpotentials
        # Activation overpotential
        i_0 = 0.1  # Exchange current density A/cm²
        eta_act = 0.026 * np.arcsinh(current_density / (2 * i_0))
        
        # Ohmic overpotential
        R_ohmic = 0.1  # Ohmic resistance Ω·cm²
        eta_ohmic = current_density * R_ohmic
        
        # Concentration overpotential
        i_L = 2.0  # Limiting current density A/cm²
        eta_conc = 0.026 * np.log(1 - current_density / i_L)
        
        # Total cell voltage
        voltage = E_ocv - eta_act - eta_ohmic - eta_conc
        
        return np.clip(voltage, 0.5, 1.2)
    
    def _generate_flow_sensor_data(self) -> Dict[str, Any]:
        """Generate flow sensor data."""
        print("  Generating flow sensor data...")
        
        # Fuel flow rate
        fuel_flow = self._generate_fuel_flow_profile()
        
        # Air flow rate
        air_flow = fuel_flow * 5.0  # 5:1 air to fuel ratio
        
        # Add sensor noise
        fuel_noise = 0.02 * np.random.normal(0, 1, len(fuel_flow))
        air_noise = 0.02 * np.random.normal(0, 1, len(air_flow))
        
        fuel_flow += fuel_noise
        air_flow += air_noise
        
        # Clamp to positive values
        fuel_flow = np.clip(fuel_flow, 0.05, 0.2)
        air_flow = np.clip(air_flow, 0.25, 1.0)
        
        return {
            'fuel_flow_rate': fuel_flow,
            'air_flow_rate': air_flow,
            'fuel_utilization': 0.8 * np.ones_like(fuel_flow),
            'time_points': self.time_points
        }
    
    def _generate_fuel_flow_profile(self) -> np.ndarray:
        """Generate fuel flow rate profile."""
        # Base flow rate
        base_flow = 0.1  # mol/s
        
        # Step changes
        step_changes = np.zeros_like(self.time_points)
        step_times = [1800, 3600, 5400]  # 30 min, 1 hour, 1.5 hours
        for step_time in step_times:
            if step_time < self.time_duration:
                step_mask = self.time_points >= step_time
                step_changes[step_mask] += 0.02 * np.random.normal(0, 1)
        
        # Random variations
        random_variation = 0.01 * np.random.normal(0, 1, len(self.time_points))
        
        # Combine all effects
        fuel_flow = base_flow + step_changes + random_variation
        
        return fuel_flow
    
    def _generate_pressure_sensor_data(self) -> Dict[str, Any]:
        """Generate pressure sensor data."""
        print("  Generating pressure sensor data...")
        
        # Fuel pressure
        fuel_pressure = 1e5 + 1000 * np.sin(2 * np.pi * self.time_points / 1800)  # 1 atm + variations
        
        # Air pressure
        air_pressure = 1e5 + 500 * np.sin(2 * np.pi * self.time_points / 1800)  # 1 atm + variations
        
        # Add sensor noise
        fuel_noise = 100 * np.random.normal(0, 1, len(fuel_pressure))
        air_noise = 100 * np.random.normal(0, 1, len(air_pressure))
        
        fuel_pressure += fuel_noise
        air_pressure += air_noise
        
        return {
            'fuel_pressure': fuel_pressure,
            'air_pressure': air_pressure,
            'pressure_difference': fuel_pressure - air_pressure,
            'time_points': self.time_points
        }
    
    def _generate_gas_analyzer_data(self) -> Dict[str, Any]:
        """Generate gas analyzer data."""
        print("  Generating gas analyzer data...")
        
        # Fuel composition
        fuel_composition = self._generate_fuel_composition_profile()
        
        # Exhaust gas composition
        exhaust_composition = self._generate_exhaust_composition_profile()
        
        return {
            'fuel_composition': fuel_composition,
            'exhaust_composition': exhaust_composition,
            'time_points': self.time_points
        }
    
    def _generate_fuel_composition_profile(self) -> Dict[str, np.ndarray]:
        """Generate fuel composition profile."""
        # Base composition
        base_comp = {'H2': 0.97, 'CO': 0.02, 'H2O': 0.01}
        
        # Add variations
        fuel_comp = {}
        for species, base_fraction in base_comp.items():
            variation = 0.05 * np.random.normal(0, 1, len(self.time_points))
            fuel_comp[species] = np.clip(base_fraction + variation, 0, 1)
        
        # Normalize to ensure fractions sum to 1
        total = sum(fuel_comp.values())
        for species in fuel_comp:
            fuel_comp[species] /= total
        
        return fuel_comp
    
    def _generate_exhaust_composition_profile(self) -> Dict[str, np.ndarray]:
        """Generate exhaust gas composition profile."""
        # Base exhaust composition
        base_comp = {'H2O': 0.4, 'CO2': 0.1, 'H2': 0.05, 'CO': 0.01, 'N2': 0.44}
        
        # Add variations
        exhaust_comp = {}
        for species, base_fraction in base_comp.items():
            variation = 0.02 * np.random.normal(0, 1, len(self.time_points))
            exhaust_comp[species] = np.clip(base_fraction + variation, 0, 1)
        
        # Normalize
        total = sum(exhaust_comp.values())
        for species in exhaust_comp:
            exhaust_comp[species] /= total
        
        return exhaust_comp
    
    def generate_sensor_fusion_data(self) -> Dict[str, Any]:
        """Generate sensor fusion data combining multiple sensors."""
        print("  Generating sensor fusion data...")
        
        # Get individual sensor data
        thermocouple_data = self._generate_thermocouple_data()
        strain_gauge_data = self._generate_strain_gauge_data()
        voltage_current_data = self._generate_voltage_current_data()
        
        # Create sensor fusion dataset
        sensor_fusion_data = {
            'temperature_measurements': [],
            'strain_measurements': [],
            'electrical_measurements': [],
            'time_points': self.time_points
        }
        
        # Combine temperature measurements
        for tc_id, tc_data in thermocouple_data.items():
            sensor_fusion_data['temperature_measurements'].append({
                'sensor_id': tc_id,
                'temperature': tc_data['temperature'],
                'location': tc_data['location']
            })
        
        # Combine strain measurements
        for sg_id, sg_data in strain_gauge_data.items():
            sensor_fusion_data['strain_measurements'].append({
                'sensor_id': sg_id,
                'strain': sg_data['strain'],
                'location': sg_data['location']
            })
        
        # Electrical measurements
        sensor_fusion_data['electrical_measurements'] = {
            'voltage': voltage_current_data['voltage'],
            'current_density': voltage_current_data['current_density'],
            'power': voltage_current_data['power']
        }
        
        return sensor_fusion_data
    
    def save_sensor_data(self, data: Dict[str, Any]) -> str:
        """Save sensor data to file."""
        return self.save_data(data, 'synthetic_sensors/sensor_data.h5')
    
    def save_sensor_fusion_data(self, data: Dict[str, Any]) -> str:
        """Save sensor fusion data to file."""
        return self.save_data(data, 'synthetic_sensors/sensor_fusion_data.h5')