"""
Real-time monitoring data generator for SOFC digital twin.
Generates high-frequency operational data and adaptive-scale monitoring data.
"""

import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from scipy import signal
from scipy.stats import norm

class MonitoringDataGenerator:
    """
    Real-time monitoring data generator for SOFC digital twin.
    
    Generates:
    - High-frequency operational data streams
    - Low-frequency high-value measurements
    - Data assimilation ready format
    - Adaptive-scale monitoring capabilities
    """
    
    def __init__(self, config: Dict):
        """
        Initialize monitoring data generator.
        
        Args:
            config: Configuration dictionary with monitoring parameters
        """
        self.config = config
        self.logger = logging.getLogger('MonitoringDataGenerator')
        
        # Monitoring parameters
        self.high_freq_sampling = config.get('high_freq_sampling', 1.0)  # Hz
        self.low_freq_sampling = config.get('low_freq_sampling', 1/3600)  # Hz (hourly)
        self.data_assimilation_window = config.get('data_assimilation_window', 3600)  # seconds
        
        # SOFC parameters
        self.cell_area = 100e-4  # m²
        self.operating_pressure = 1e5  # Pa
        
    def generate_dataset(
        self,
        duration_hours: float = 24.0,
        high_freq_sampling: float = 1.0,
        low_freq_sampling: float = 1/3600.0
    ) -> Dict[str, Any]:
        """
        Generate real-time monitoring dataset.
        
        Args:
            duration_hours: Duration of monitoring in hours
            high_freq_sampling: High-frequency sampling rate (Hz)
            low_freq_sampling: Low-frequency sampling rate (Hz)
            
        Returns:
            Dictionary containing the monitoring dataset
        """
        self.logger.info(f"Generating monitoring data for {duration_hours} hours...")
        
        # Update sampling rates
        self.high_freq_sampling = high_freq_sampling
        self.low_freq_sampling = low_freq_sampling
        
        # Generate time series
        high_freq_time = self._generate_high_freq_time_series(duration_hours)
        low_freq_time = self._generate_low_freq_time_series(duration_hours)
        
        # Generate operating conditions
        operating_conditions = self._generate_operating_conditions(high_freq_time)
        
        # Generate high-frequency data
        high_freq_data = self._generate_high_frequency_data(high_freq_time, operating_conditions)
        
        # Generate low-frequency data
        low_freq_data = self._generate_low_frequency_data(low_freq_time, operating_conditions)
        
        # Generate data assimilation data
        assimilation_data = self._generate_data_assimilation_data(
            high_freq_time, high_freq_data, low_freq_data
        )
        
        # Generate adaptive-scale triggers
        adaptive_triggers = self._generate_adaptive_triggers(high_freq_data, low_freq_data)
        
        dataset = {
            'metadata': {
                'duration_hours': duration_hours,
                'high_freq_sampling': high_freq_sampling,
                'low_freq_sampling': low_freq_sampling,
                'data_assimilation_window': self.data_assimilation_window,
                'generation_time': time.time()
            },
            'high_freq_time': high_freq_time,
            'low_freq_time': low_freq_time,
            'operating_conditions': operating_conditions,
            'high_frequency_data': high_freq_data,
            'low_frequency_data': low_freq_data,
            'data_assimilation': assimilation_data,
            'adaptive_triggers': adaptive_triggers
        }
        
        self.logger.info("Monitoring dataset generation completed")
        return dataset
    
    def _generate_high_freq_time_series(self, duration_hours: float) -> np.ndarray:
        """Generate high-frequency time series."""
        dt = 1.0 / self.high_freq_sampling
        n_points = int(duration_hours * 3600 * self.high_freq_sampling)
        time_points = np.linspace(0, duration_hours * 3600, n_points)
        return time_points
    
    def _generate_low_freq_time_series(self, duration_hours: float) -> np.ndarray:
        """Generate low-frequency time series."""
        dt = 1.0 / self.low_freq_sampling
        n_points = int(duration_hours * 3600 * self.low_freq_sampling)
        time_points = np.linspace(0, duration_hours * 3600, n_points)
        return time_points
    
    def _generate_operating_conditions(self, time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate realistic operating conditions with transients."""
        n_points = len(time_points)
        
        # Current density with realistic transients
        current_density = np.ones(n_points) * 0.5  # Base current density (A/cm²)
        
        # Add load changes
        load_changes = [
            {'time': 2*3600, 'duration': 1800, 'change': 0.2},  # 2h: +0.2 A/cm² for 30 min
            {'time': 6*3600, 'duration': 3600, 'change': -0.1},  # 6h: -0.1 A/cm² for 1h
            {'time': 12*3600, 'duration': 900, 'change': 0.3},   # 12h: +0.3 A/cm² for 15 min
        ]
        
        for change in load_changes:
            start_idx = int(change['time'] * self.high_freq_sampling)
            end_idx = int((change['time'] + change['duration']) * self.high_freq_sampling)
            if end_idx < n_points:
                # Smooth transition
                transition_length = min(300, end_idx - start_idx)  # 5 min transition
                transition = np.linspace(0, change['change'], transition_length)
                current_density[start_idx:start_idx+transition_length] += transition
                current_density[start_idx+transition_length:end_idx] += change['change']
                # Smooth transition out
                if end_idx + transition_length < n_points:
                    transition_out = np.linspace(change['change'], 0, transition_length)
                    current_density[end_idx:end_idx+transition_length] += transition_out
        
        # Temperature with thermal transients
        inlet_fuel_temp = 1073.15 + 20 * np.sin(2 * np.pi * time_points / (24 * 3600))  # Daily cycle
        inlet_air_temp = 1073.15 + 15 * np.sin(2 * np.pi * time_points / (24 * 3600))
        
        # Add thermal transients during load changes
        for change in load_changes:
            start_idx = int(change['time'] * self.high_freq_sampling)
            end_idx = int((change['time'] + change['duration']) * self.high_freq_sampling)
            if end_idx < n_points:
                temp_change = change['change'] * 50  # 50K per A/cm²
                inlet_fuel_temp[start_idx:end_idx] += temp_change
                inlet_air_temp[start_idx:end_idx] += temp_change * 0.8
        
        # Flow rates with realistic variations
        fuel_flow_rate = 0.1 + 0.01 * np.sin(2 * np.pi * time_points / (12 * 3600))
        air_flow_rate = 0.5 + 0.05 * np.sin(2 * np.pi * time_points / (12 * 3600))
        
        # Add flow rate changes during load changes
        for change in load_changes:
            start_idx = int(change['time'] * self.high_freq_sampling)
            end_idx = int((change['time'] + change['duration']) * self.high_freq_sampling)
            if end_idx < n_points:
                fuel_flow_change = change['change'] * 0.02  # 0.02 mol/s per A/cm²
                air_flow_change = change['change'] * 0.1    # 0.1 mol/s per A/cm²
                fuel_flow_rate[start_idx:end_idx] += fuel_flow_change
                air_flow_rate[start_idx:end_idx] += air_flow_change
        
        return {
            'current_density': current_density,
            'inlet_fuel_temp': inlet_fuel_temp,
            'inlet_air_temp': inlet_air_temp,
            'fuel_flow_rate': fuel_flow_rate,
            'air_flow_rate': air_flow_rate
        }
    
    def _generate_high_frequency_data(
        self, time_points: np.ndarray, operating_conditions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate high-frequency monitoring data."""
        n_points = len(time_points)
        
        # Calculate cell voltage with realistic dynamics
        current_density = operating_conditions['current_density']
        temperature = operating_conditions['inlet_fuel_temp']
        
        # Nernst potential
        nernst_potential = 1.229 - 0.000845 * (temperature - 298.15)
        
        # Overpotentials with dynamics
        activation_overpotential = 0.1 * current_density + 0.02 * np.gradient(current_density)
        ohmic_overpotential = 0.05 * current_density
        concentration_overpotential = 0.02 * current_density + 0.01 * np.gradient(current_density)
        
        # Cell voltage with noise
        cell_voltage = nernst_potential - activation_overpotential - ohmic_overpotential - concentration_overpotential
        cell_voltage += np.random.normal(0, 0.01, n_points)  # 10 mV noise
        
        # Current and power
        current = current_density * self.cell_area * 10000  # Convert to A
        power = current * cell_voltage
        
        # Temperature measurements with noise
        inlet_fuel_temp = operating_conditions['inlet_fuel_temp'] + np.random.normal(0, 1, n_points)
        inlet_air_temp = operating_conditions['inlet_air_temp'] + np.random.normal(0, 1, n_points)
        
        # Outlet temperatures (simplified heat transfer model)
        outlet_fuel_temp = inlet_fuel_temp + 50 + 5 * np.random.randn(n_points)
        outlet_air_temp = inlet_air_temp + 30 + 3 * np.random.randn(n_points)
        
        # Pressure measurements
        fuel_pressure = self.operating_pressure + 1000 * np.random.randn(n_points)
        air_pressure = self.operating_pressure + 1000 * np.random.randn(n_points)
        
        return {
            'current': current,
            'voltage': cell_voltage,
            'power': power,
            'inlet_fuel_temp': inlet_fuel_temp,
            'inlet_air_temp': inlet_air_temp,
            'outlet_fuel_temp': outlet_fuel_temp,
            'outlet_air_temp': outlet_air_temp,
            'fuel_pressure': fuel_pressure,
            'air_pressure': air_pressure,
            'fuel_flow_rate': operating_conditions['fuel_flow_rate'],
            'air_flow_rate': operating_conditions['air_flow_rate']
        }
    
    def _generate_low_frequency_data(
        self, time_points: np.ndarray, operating_conditions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Generate low-frequency high-value measurements."""
        n_points = len(time_points)
        
        # EIS measurements (every hour)
        eis_data = {
            'measurement_times': time_points,
            'frequencies': np.logspace(0, 5, 30),  # 1 Hz to 100 kHz
            'impedance_real': [],
            'impedance_imag': [],
            'phase_angle': [],
            'magnitude': []
        }
        
        for i, t in enumerate(time_points):
            # Simulate EIS with degradation
            degradation = 1.0 - 0.001 * t / (24 * 3600)  # 0.1% per day
            
            # Equivalent circuit parameters
            R_ohm = 0.1 * degradation
            R_act = 0.2 * degradation
            C_dl = 1e-3 / degradation
            R_conc = 0.05 * degradation
            
            # Calculate impedance
            w = 2 * np.pi * eis_data['frequencies']
            Z_real = R_ohm + R_act / (1 + (w * R_act * C_dl)**2) + R_conc
            Z_imag = -w * R_act**2 * C_dl / (1 + (w * R_act * C_dl)**2)
            
            # Add noise
            Z_real += np.random.normal(0, 0.01 * np.abs(Z_real))
            Z_imag += np.random.normal(0, 0.01 * np.abs(Z_imag))
            
            eis_data['impedance_real'].append(Z_real)
            eis_data['impedance_imag'].append(Z_imag)
            eis_data['phase_angle'].append(np.angle(Z_real + 1j * Z_imag))
            eis_data['magnitude'].append(np.abs(Z_real + 1j * Z_imag))
        
        # Thermal imaging data (every 6 hours)
        thermal_interval = 6 * 3600  # 6 hours
        thermal_times = time_points[::int(thermal_interval * self.low_freq_sampling)]
        
        thermal_data = {
            'measurement_times': thermal_times,
            'max_temperature': [],
            'min_temperature': [],
            'temperature_gradient': [],
            'hot_spots': []
        }
        
        for i, t in enumerate(thermal_times):
            # Simulate thermal imaging
            base_temp = 1073.15 + 50 * np.sin(2 * np.pi * t / (24 * 3600))
            max_temp = base_temp + 100 + 20 * np.random.randn()
            min_temp = base_temp - 50 + 10 * np.random.randn()
            temp_gradient = 50 + 10 * np.random.randn()
            
            thermal_data['max_temperature'].append(max_temp)
            thermal_data['min_temperature'].append(min_temp)
            thermal_data['temperature_gradient'].append(temp_gradient)
            
            # Hot spots (potential failure locations)
            if np.random.random() < 0.1:  # 10% chance of hot spot
                hot_spot = {
                    'time': t,
                    'x': np.random.uniform(0, 0.1),
                    'y': np.random.uniform(0, 0.1),
                    'temperature': max_temp + 50 + 20 * np.random.randn(),
                    'size': np.random.uniform(0.001, 0.01)
                }
                thermal_data['hot_spots'].append(hot_spot)
        
        # Strain gauge measurements
        strain_data = {
            'measurement_times': time_points,
            'strain_anode': [],
            'strain_cathode': [],
            'strain_interconnect': []
        }
        
        for i, t in enumerate(time_points):
            # Thermal strain
            temp_variation = operating_conditions['inlet_fuel_temp'][i] - 1073.15
            thermal_strain = 12e-6 * temp_variation
            
            # Mechanical strain
            current_strain = 50e-6 * operating_conditions['current_density'][i]
            
            # Total strain with noise
            base_strain = 100e-6
            total_strain = base_strain + thermal_strain + current_strain
            
            strain_anode = total_strain + np.random.normal(0, 1e-6)
            strain_cathode = total_strain * 0.8 + np.random.normal(0, 1e-6)
            strain_interconnect = total_strain * 1.2 + np.random.normal(0, 1e-6)
            
            strain_data['strain_anode'].append(strain_anode)
            strain_data['strain_cathode'].append(strain_cathode)
            strain_data['strain_interconnect'].append(strain_interconnect)
        
        return {
            'eis_data': eis_data,
            'thermal_data': thermal_data,
            'strain_data': strain_data
        }
    
    def _generate_data_assimilation_data(
        self, high_freq_time: np.ndarray, high_freq_data: Dict, low_freq_data: Dict
    ) -> Dict[str, Any]:
        """Generate data assimilation ready data."""
        # Data assimilation windows
        window_size = self.data_assimilation_window
        n_windows = int(high_freq_time[-1] / window_size)
        
        assimilation_data = {
            'window_size': window_size,
            'n_windows': n_windows,
            'windows': []
        }
        
        for i in range(n_windows):
            start_time = i * window_size
            end_time = (i + 1) * window_size
            
            # Find indices for this window
            start_idx = np.searchsorted(high_freq_time, start_time)
            end_idx = np.searchsorted(high_freq_time, end_time)
            
            if end_idx > start_idx:
                window_data = {
                    'window_id': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'high_freq_data': {
                        'time': high_freq_time[start_idx:end_idx],
                        'voltage': high_freq_data['voltage'][start_idx:end_idx],
                        'current': high_freq_data['current'][start_idx:end_idx],
                        'power': high_freq_data['power'][start_idx:end_idx],
                        'inlet_fuel_temp': high_freq_data['inlet_fuel_temp'][start_idx:end_idx],
                        'inlet_air_temp': high_freq_data['inlet_air_temp'][start_idx:end_idx]
                    },
                    'low_freq_data': self._get_low_freq_data_for_window(
                        start_time, end_time, low_freq_data
                    ),
                    'statistics': self._calculate_window_statistics(
                        high_freq_data, start_idx, end_idx
                    )
                }
                assimilation_data['windows'].append(window_data)
        
        return assimilation_data
    
    def _get_low_freq_data_for_window(
        self, start_time: float, end_time: float, low_freq_data: Dict
    ) -> Dict[str, Any]:
        """Get low-frequency data for a specific time window."""
        window_low_freq = {}
        
        # EIS data
        if 'eis_data' in low_freq_data:
            eis_times = low_freq_data['eis_data']['measurement_times']
            mask = (eis_times >= start_time) & (eis_times < end_time)
            if np.any(mask):
                window_low_freq['eis'] = {
                    'times': eis_times[mask],
                    'impedance_real': [low_freq_data['eis_data']['impedance_real'][i] 
                                     for i in np.where(mask)[0]],
                    'impedance_imag': [low_freq_data['eis_data']['impedance_imag'][i] 
                                     for i in np.where(mask)[0]]
                }
        
        # Thermal data
        if 'thermal_data' in low_freq_data:
            thermal_times = low_freq_data['thermal_data']['measurement_times']
            mask = (thermal_times >= start_time) & (thermal_times < end_time)
            if np.any(mask):
                window_low_freq['thermal'] = {
                    'times': thermal_times[mask],
                    'max_temperature': [low_freq_data['thermal_data']['max_temperature'][i] 
                                      for i in np.where(mask)[0]],
                    'min_temperature': [low_freq_data['thermal_data']['min_temperature'][i] 
                                      for i in np.where(mask)[0]]
                }
        
        return window_low_freq
    
    def _calculate_window_statistics(
        self, high_freq_data: Dict, start_idx: int, end_idx: int
    ) -> Dict[str, float]:
        """Calculate statistics for a time window."""
        window_data = {}
        
        for key, values in high_freq_data.items():
            if isinstance(values, np.ndarray) and len(values) > start_idx:
                window_values = values[start_idx:end_idx]
                window_data[key] = {
                    'mean': np.mean(window_values),
                    'std': np.std(window_values),
                    'min': np.min(window_values),
                    'max': np.max(window_values),
                    'trend': np.polyfit(range(len(window_values)), window_values, 1)[0]
                }
        
        return window_data
    
    def _generate_adaptive_triggers(
        self, high_freq_data: Dict, low_freq_data: Dict
    ) -> Dict[str, Any]:
        """Generate adaptive-scale monitoring triggers."""
        triggers = {
            'voltage_drop': [],
            'temperature_spike': [],
            'power_anomaly': [],
            'strain_threshold': [],
            'efficiency_degradation': []
        }
        
        # Voltage drop triggers
        voltage = high_freq_data['voltage']
        voltage_trend = np.gradient(voltage)
        voltage_drop_threshold = -0.01  # 10 mV/s
        
        drop_indices = np.where(voltage_trend < voltage_drop_threshold)[0]
        for idx in drop_indices:
            triggers['voltage_drop'].append({
                'time': idx / self.high_freq_sampling,
                'severity': abs(voltage_trend[idx]),
                'voltage': voltage[idx]
            })
        
        # Temperature spike triggers
        temp = high_freq_data['inlet_fuel_temp']
        temp_gradient = np.gradient(temp)
        temp_spike_threshold = 5.0  # 5 K/s
        
        spike_indices = np.where(temp_gradient > temp_spike_threshold)[0]
        for idx in spike_indices:
            triggers['temperature_spike'].append({
                'time': idx / self.high_freq_sampling,
                'severity': temp_gradient[idx],
                'temperature': temp[idx]
            })
        
        # Power anomaly triggers
        power = high_freq_data['power']
        power_mean = np.mean(power)
        power_std = np.std(power)
        power_anomaly_threshold = power_mean + 3 * power_std
        
        anomaly_indices = np.where(power > power_anomaly_threshold)[0]
        for idx in anomaly_indices:
            triggers['power_anomaly'].append({
                'time': idx / self.high_freq_sampling,
                'severity': (power[idx] - power_mean) / power_std,
                'power': power[idx]
            })
        
        # Strain threshold triggers
        if 'strain_data' in low_freq_data:
            strain_anode = low_freq_data['strain_data']['strain_anode']
            strain_threshold = 200e-6  # 200 microstrain
            
            for i, strain in enumerate(strain_anode):
                if strain > strain_threshold:
                    triggers['strain_threshold'].append({
                        'time': i / self.low_freq_sampling,
                        'severity': strain / strain_threshold,
                        'strain': strain
                    })
        
        return triggers