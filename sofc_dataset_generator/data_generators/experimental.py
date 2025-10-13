"""
Experimental data generator for SOFC validation datasets.
Simulates realistic experimental measurements with noise and uncertainty.
"""

import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import signal
from scipy.stats import norm
import time

class ExperimentalDataGenerator:
    """
    Experimental data generator for SOFC validation datasets.
    
    Simulates:
    - Lab test rig measurements
    - Electrochemical Impedance Spectroscopy (EIS)
    - Thermal imaging data
    - Strain gauge measurements
    - Acoustic emission data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize experimental data generator.
        
        Args:
            config: Configuration dictionary with experimental parameters
        """
        self.config = config
        self.logger = logging.getLogger('ExperimentalDataGenerator')
        
        # Default experimental parameters
        self.noise_level = config.get('noise_level', 0.05)  # 5% noise
        self.sampling_frequency = config.get('sampling_frequency', 1.0)  # Hz
        self.measurement_uncertainty = config.get('measurement_uncertainty', 0.02)  # 2%
        
        # SOFC cell parameters
        self.cell_area = 100e-4  # m² (10 cm x 10 cm)
        self.operating_pressure = 1e5  # Pa
        
    def generate_dataset(
        self,
        test_duration_hours: float = 100.0,
        operating_profile: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate experimental validation dataset.
        
        Args:
            test_duration_hours: Duration of experimental test in hours
            operating_profile: Operating profile for the test
            
        Returns:
            Dictionary containing the experimental dataset
        """
        self.logger.info(f"Generating experimental data for {test_duration_hours} hours...")
        
        # Generate time series
        time_points = self._generate_time_series(test_duration_hours)
        
        # Generate operating profile
        if operating_profile is None:
            operating_profile = self._generate_default_operating_profile(time_points)
        
        # Generate global operational data
        global_data = self._generate_global_operational_data(time_points, operating_profile)
        
        # Generate EIS data
        eis_data = self._generate_eis_data(time_points, operating_profile)
        
        # Generate thermal imaging data
        thermal_data = self._generate_thermal_imaging_data(time_points, operating_profile)
        
        # Generate strain gauge data
        strain_data = self._generate_strain_gauge_data(time_points, operating_profile)
        
        # Generate acoustic emission data
        acoustic_data = self._generate_acoustic_emission_data(time_points, operating_profile)
        
        # Generate ex-situ post-mortem data
        post_mortem_data = self._generate_post_mortem_data(operating_profile)
        
        dataset = {
            'metadata': {
                'test_duration_hours': test_duration_hours,
                'sampling_frequency': self.sampling_frequency,
                'cell_area': self.cell_area,
                'operating_pressure': self.operating_pressure,
                'noise_level': self.noise_level,
                'measurement_uncertainty': self.measurement_uncertainty,
                'generation_time': time.time()
            },
            'time_points': time_points,
            'operating_profile': operating_profile,
            'global_operational': global_data,
            'eis_data': eis_data,
            'thermal_imaging': thermal_data,
            'strain_gauges': strain_data,
            'acoustic_emission': acoustic_data,
            'post_mortem': post_mortem_data
        }
        
        self.logger.info("Experimental dataset generation completed")
        return dataset
    
    def _generate_time_series(self, duration_hours: float) -> np.ndarray:
        """Generate time series for the test duration."""
        dt = 1.0 / self.sampling_frequency  # seconds
        n_points = int(duration_hours * 3600 * self.sampling_frequency)
        time_points = np.linspace(0, duration_hours * 3600, n_points)
        return time_points
    
    def _generate_default_operating_profile(self, time_points: np.ndarray) -> Dict:
        """Generate default operating profile for the test."""
        n_points = len(time_points)
        
        # Current density profile (A/cm²)
        current_density = 0.5 + 0.3 * np.sin(2 * np.pi * time_points / (24 * 3600))  # Daily cycle
        
        # Temperature profile (K)
        inlet_fuel_temp = 1073.15 + 50 * np.sin(2 * np.pi * time_points / (12 * 3600))  # 12-hour cycle
        inlet_air_temp = 1073.15 + 30 * np.sin(2 * np.pi * time_points / (12 * 3600))
        
        # Flow rates (mol/s)
        fuel_flow_rate = 0.1 + 0.02 * np.random.randn(n_points)
        air_flow_rate = 0.5 + 0.1 * np.random.randn(n_points)
        
        # Add some degradation over time
        degradation_factor = 1.0 - 0.001 * time_points / (24 * 3600)  # 0.1% per day
        
        return {
            'current_density': current_density * degradation_factor,
            'inlet_fuel_temp': inlet_fuel_temp,
            'inlet_air_temp': inlet_air_temp,
            'fuel_flow_rate': fuel_flow_rate,
            'air_flow_rate': air_flow_rate,
            'degradation_factor': degradation_factor
        }
    
    def _generate_global_operational_data(
        self, time_points: np.ndarray, operating_profile: Dict
    ) -> Dict[str, np.ndarray]:
        """Generate global operational measurements."""
        n_points = len(time_points)
        
        # Calculate cell voltage based on operating conditions
        current_density = operating_profile['current_density']
        temperature = operating_profile['inlet_fuel_temp']
        
        # Nernst potential (simplified)
        nernst_potential = 1.229 - 0.000845 * (temperature - 298.15)
        
        # Overpotentials (simplified)
        activation_overpotential = 0.1 * current_density
        ohmic_overpotential = 0.05 * current_density
        concentration_overpotential = 0.02 * current_density
        
        # Cell voltage
        cell_voltage = nernst_potential - activation_overpotential - ohmic_overpotential - concentration_overpotential
        
        # Add noise
        cell_voltage += self._add_measurement_noise(cell_voltage, self.noise_level)
        
        # Current and power
        current = current_density * self.cell_area * 10000  # Convert to A
        power = current * cell_voltage
        
        # Inlet/outlet temperatures
        inlet_fuel_temp = operating_profile['inlet_fuel_temp'] + self._add_measurement_noise(
            operating_profile['inlet_fuel_temp'], self.measurement_uncertainty
        )
        inlet_air_temp = operating_profile['inlet_air_temp'] + self._add_measurement_noise(
            operating_profile['inlet_air_temp'], self.measurement_uncertainty
        )
        
        # Outlet temperatures (simplified heat transfer)
        outlet_fuel_temp = inlet_fuel_temp + 50 + 10 * np.random.randn(n_points)
        outlet_air_temp = inlet_air_temp + 30 + 5 * np.random.randn(n_points)
        
        return {
            'current': current,
            'voltage': cell_voltage,
            'power': power,
            'inlet_fuel_temp': inlet_fuel_temp,
            'inlet_air_temp': inlet_air_temp,
            'outlet_fuel_temp': outlet_fuel_temp,
            'outlet_air_temp': outlet_air_temp,
            'fuel_flow_rate': operating_profile['fuel_flow_rate'],
            'air_flow_rate': operating_profile['air_flow_rate']
        }
    
    def _generate_eis_data(
        self, time_points: np.ndarray, operating_profile: Dict
    ) -> Dict[str, Any]:
        """Generate Electrochemical Impedance Spectroscopy data."""
        # EIS measurements every 24 hours
        eis_interval = 24 * 3600  # seconds
        eis_times = time_points[::int(eis_interval * self.sampling_frequency)]
        
        eis_data = {
            'measurement_times': eis_times,
            'frequencies': np.logspace(0, 5, 50),  # 1 Hz to 100 kHz
            'impedance_real': [],
            'impedance_imag': [],
            'phase_angle': [],
            'magnitude': []
        }
        
        for i, t in enumerate(eis_times):
            # Simulate EIS spectrum with degradation
            degradation_factor = operating_profile['degradation_factor'][i]
            
            # Equivalent circuit parameters (simplified)
            R_ohm = 0.1 * degradation_factor  # Ohmic resistance
            R_act = 0.2 * degradation_factor  # Activation resistance
            C_dl = 1e-3 / degradation_factor  # Double layer capacitance
            R_conc = 0.05 * degradation_factor  # Concentration resistance
            
            # Calculate impedance
            w = 2 * np.pi * eis_data['frequencies']
            Z_real = R_ohm + R_act / (1 + (w * R_act * C_dl)**2) + R_conc
            Z_imag = -w * R_act**2 * C_dl / (1 + (w * R_act * C_dl)**2)
            
            # Add noise
            Z_real += self._add_measurement_noise(Z_real, self.noise_level)
            Z_imag += self._add_measurement_noise(Z_imag, self.noise_level)
            
            eis_data['impedance_real'].append(Z_real)
            eis_data['impedance_imag'].append(Z_imag)
            eis_data['phase_angle'].append(np.angle(Z_real + 1j * Z_imag))
            eis_data['magnitude'].append(np.abs(Z_real + 1j * Z_imag))
        
        return eis_data
    
    def _generate_thermal_imaging_data(
        self, time_points: np.ndarray, operating_profile: Dict
    ) -> Dict[str, Any]:
        """Generate thermal imaging data."""
        # Thermal images every 6 hours
        thermal_interval = 6 * 3600  # seconds
        thermal_times = time_points[::int(thermal_interval * self.sampling_frequency)]
        
        # Image resolution
        nx, ny = 64, 64
        
        thermal_data = {
            'measurement_times': thermal_times,
            'image_resolution': (nx, ny),
            'temperature_images': [],
            'max_temperature': [],
            'min_temperature': [],
            'temperature_gradient': []
        }
        
        for i, t in enumerate(thermal_times):
            # Generate temperature field
            x = np.linspace(0, 0.1, nx)
            y = np.linspace(0, 0.1, ny)
            X, Y = np.meshgrid(x, y)
            
            # Base temperature with spatial variation
            base_temp = operating_profile['inlet_fuel_temp'][i]
            temp_field = base_temp + 50 * np.sin(2 * np.pi * X / 0.1) * np.cos(2 * np.pi * Y / 0.1)
            
            # Add hot spots (potential failure locations)
            if i > len(thermal_times) // 2:  # Add hot spots in second half
                hot_spot_x = 0.03 + 0.01 * np.random.randn()
                hot_spot_y = 0.07 + 0.01 * np.random.randn()
                hot_spot_temp = 100 + 50 * np.random.rand()
                
                dist = np.sqrt((X - hot_spot_x)**2 + (Y - hot_spot_y)**2)
                temp_field += hot_spot_temp * np.exp(-dist / 0.01)
            
            # Add noise
            temp_field += self._add_measurement_noise(temp_field, self.noise_level)
            
            thermal_data['temperature_images'].append(temp_field)
            thermal_data['max_temperature'].append(np.max(temp_field))
            thermal_data['min_temperature'].append(np.min(temp_field))
            thermal_data['temperature_gradient'].append(np.max(np.gradient(temp_field)))
        
        return thermal_data
    
    def _generate_strain_gauge_data(
        self, time_points: np.ndarray, operating_profile: Dict
    ) -> Dict[str, Any]:
        """Generate strain gauge measurements."""
        n_points = len(time_points)
        
        # Strain gauge locations (simplified)
        strain_locations = [
            {'name': 'anode_center', 'x': 0.05, 'y': 0.05, 'z': 0.001},
            {'name': 'cathode_center', 'x': 0.05, 'y': 0.05, 'z': 0.0015},
            {'name': 'interconnect_corner', 'x': 0.02, 'y': 0.02, 'z': 0.002},
        ]
        
        strain_data = {
            'measurement_times': time_points,
            'strain_locations': strain_locations,
            'strain_measurements': {}
        }
        
        for location in strain_locations:
            # Generate strain time series
            base_strain = 100e-6  # 100 microstrain
            
            # Thermal strain component
            temp_variation = operating_profile['inlet_fuel_temp'] - 1073.15
            thermal_strain = 12e-6 * temp_variation  # Thermal expansion coefficient
            
            # Mechanical strain component (simplified)
            current_strain = 50e-6 * operating_profile['current_density']
            
            # Total strain
            total_strain = base_strain + thermal_strain + current_strain
            
            # Add noise and drift
            noise = self._add_measurement_noise(total_strain, self.noise_level)
            drift = 1e-6 * time_points / (24 * 3600)  # 1 microstrain per day drift
            
            strain_measurement = total_strain + noise + drift
            
            strain_data['strain_measurements'][location['name']] = strain_measurement
        
        return strain_data
    
    def _generate_acoustic_emission_data(
        self, time_points: np.ndarray, operating_profile: Dict
    ) -> Dict[str, Any]:
        """Generate acoustic emission data."""
        n_points = len(time_points)
        
        # Acoustic emission events (rare but important)
        ae_events = []
        
        # Generate events based on stress and degradation
        stress_level = operating_profile['current_density'] * 1e6  # Simplified stress
        degradation = 1.0 - operating_profile['degradation_factor']
        
        # Event probability increases with stress and degradation
        event_probability = 0.001 * stress_level / np.max(stress_level) * degradation
        
        for i, t in enumerate(time_points):
            if np.random.random() < event_probability[i]:
                # Generate acoustic event
                event = {
                    'time': t,
                    'amplitude': np.random.exponential(100),  # dB
                    'frequency': np.random.uniform(100, 1000),  # Hz
                    'duration': np.random.exponential(0.1),  # seconds
                    'location': {
                        'x': np.random.uniform(0, 0.1),
                        'y': np.random.uniform(0, 0.1),
                        'z': np.random.uniform(0, 0.002)
                    }
                }
                ae_events.append(event)
        
        acoustic_data = {
            'measurement_times': time_points,
            'ae_events': ae_events,
            'total_events': len(ae_events),
            'event_rate': len(ae_events) / (time_points[-1] / 3600)  # events per hour
        }
        
        return acoustic_data
    
    def _generate_post_mortem_data(self, operating_profile: Dict) -> Dict[str, Any]:
        """Generate ex-situ post-mortem analysis data."""
        # Simulate post-mortem analysis results
        total_degradation = 1.0 - operating_profile['degradation_factor'][-1]
        
        post_mortem_data = {
            'crack_analysis': {
                'total_crack_length': 0.1 * total_degradation,  # mm
                'crack_density': 10 * total_degradation,  # cracks/cm²
                'max_crack_depth': 0.05 * total_degradation,  # mm
                'crack_locations': [
                    {'x': 0.03, 'y': 0.07, 'length': 0.02 * total_degradation},
                    {'x': 0.08, 'y': 0.02, 'length': 0.015 * total_degradation},
                ]
            },
            'microstructure_analysis': {
                'anode_porosity': 0.3 + 0.05 * total_degradation,
                'cathode_porosity': 0.25 + 0.03 * total_degradation,
                'electrolyte_density': 0.95 - 0.1 * total_degradation,
                'grain_size_anode': 2.0 + 0.5 * total_degradation,  # μm
                'grain_size_cathode': 1.5 + 0.3 * total_degradation,  # μm
            },
            'elemental_analysis': {
                'ni_diffusion': 0.1 * total_degradation,  # atomic %
                'cr_diffusion': 0.05 * total_degradation,  # atomic %
                'si_contamination': 0.02 * total_degradation,  # atomic %
            },
            'mechanical_properties': {
                'youngs_modulus_reduction': 0.1 * total_degradation,  # %
                'fracture_toughness_reduction': 0.15 * total_degradation,  # %
                'residual_stress': 50e6 * total_degradation,  # Pa
            }
        }
        
        return post_mortem_data
    
    def _add_measurement_noise(self, signal: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic measurement noise to signal."""
        if isinstance(signal, (int, float)):
            signal = np.array([signal])
        
        noise = np.random.normal(0, noise_level * np.abs(signal))
        return noise