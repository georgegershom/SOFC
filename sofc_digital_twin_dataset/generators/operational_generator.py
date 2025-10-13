"""
Operational and electrochemical performance data generator for SOFC digital twin.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple, List
from .base_generator import BaseGenerator


class OperationalGenerator(BaseGenerator):
    """
    Generator for operational and electrochemical performance data including:
    - Controlled input parameters (fuel composition, flow rates, temperatures)
    - Electrochemical response data (voltage, current, EIS spectra)
    """
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        super().__init__(config_path)
        self.operating_conditions = self.config['operating_conditions']
        self.time_duration = self.config['generation']['time_duration']
        self.time_step = self.config['generation']['time_step']
        self.time_points = np.arange(0, self.time_duration, self.time_step)
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete operational and electrochemical dataset."""
        print("Generating operational and electrochemical data...")
        
        data = {
            'controlled_inputs': self._generate_controlled_inputs(),
            'electrochemical_response': self._generate_electrochemical_response(),
            'performance_metrics': self._generate_performance_metrics()
        }
        
        return data
    
    def _generate_controlled_inputs(self) -> Dict[str, Any]:
        """Generate controlled input parameters."""
        print("  Generating controlled input parameters...")
        
        # Fuel composition variations
        fuel_comp = self._generate_fuel_composition()
        
        # Flow rates
        flow_rates = self._generate_flow_rates()
        
        # Temperature profiles
        temperatures = self._generate_temperature_profiles()
        
        # Current density profiles
        current_density = self._generate_current_density()
        
        return {
            'fuel_composition': fuel_comp,
            'flow_rates': flow_rates,
            'temperatures': temperatures,
            'current_density': current_density,
            'time_points': self.time_points
        }
    
    def _generate_fuel_composition(self) -> Dict[str, np.ndarray]:
        """Generate time-varying fuel composition."""
        base_comp = self.operating_conditions['fuel_composition']
        
        # Add realistic variations
        fuel_comp = {}
        for species, base_fraction in base_comp.items():
            # Add small random variations
            variation = 0.05 * np.random.normal(0, 1, len(self.time_points))
            fuel_comp[species] = np.clip(base_fraction + variation, 0, 1)
        
        # Normalize to ensure fractions sum to 1
        total = sum(fuel_comp.values())
        for species in fuel_comp:
            fuel_comp[species] /= total
        
        return fuel_comp
    
    def _generate_flow_rates(self) -> Dict[str, np.ndarray]:
        """Generate flow rate profiles."""
        # Base flow rates (mol/s)
        base_fuel_flow = 0.1  # mol/s
        base_air_flow = 0.5   # mol/s
        
        # Add step changes and variations
        fuel_flow = np.ones_like(self.time_points) * base_fuel_flow
        
        # Step change at 1/3 of the time
        step_time = self.time_duration / 3
        step_mask = self.time_points > step_time
        fuel_flow[step_mask] *= 1.2
        
        # Add small random variations
        fuel_flow += 0.02 * np.random.normal(0, 1, len(self.time_points))
        fuel_flow = np.clip(fuel_flow, 0.05, 0.2)
        
        # Air flow rate (typically higher than fuel)
        air_flow = fuel_flow * 5.0  # 5:1 air to fuel ratio
        
        return {
            'fuel_flow_rate': fuel_flow,
            'air_flow_rate': air_flow,
            'fuel_utilization': self.operating_conditions['fuel_utilization'] * np.ones_like(self.time_points)
        }
    
    def _generate_temperature_profiles(self) -> Dict[str, np.ndarray]:
        """Generate temperature profiles for different locations."""
        temp_range = self.operating_conditions['temperature_range']
        base_temp = np.mean(temp_range)
        
        # Inlet temperature (controlled)
        inlet_temp = base_temp + 10 * np.sin(2 * np.pi * self.time_points / 3600)  # 1 hour cycle
        
        # Outlet temperature (depends on heat generation)
        outlet_temp = inlet_temp + 50 + 20 * np.random.normal(0, 1, len(self.time_points))
        
        # Center temperature (intermediate)
        center_temp = (inlet_temp + outlet_temp) / 2 + 10 * np.random.normal(0, 1, len(self.time_points))
        
        # Clamp to reasonable ranges
        inlet_temp = np.clip(inlet_temp, temp_range[0], temp_range[1])
        outlet_temp = np.clip(outlet_temp, temp_range[0], temp_range[1])
        center_temp = np.clip(center_temp, temp_range[0], temp_range[1])
        
        return {
            'inlet_temperature': inlet_temp,
            'outlet_temperature': outlet_temp,
            'center_temperature': center_temp,
            'ambient_temperature': 25 * np.ones_like(self.time_points)  # 25°C
        }
    
    def _generate_current_density(self) -> np.ndarray:
        """Generate current density profile."""
        current_range = self.operating_conditions['current_density_range']
        
        # Start at low current, ramp up, then vary
        current_density = np.zeros_like(self.time_points)
        
        # Ramp up phase (first 20% of time)
        ramp_time = 0.2 * self.time_duration
        ramp_mask = self.time_points <= ramp_time
        current_density[ramp_mask] = (current_range[1] - current_range[0]) * self.time_points[ramp_mask] / ramp_time
        
        # Steady operation with variations
        steady_mask = self.time_points > ramp_time
        base_current = current_range[1] * 0.8
        variation = 0.1 * current_range[1] * np.sin(2 * np.pi * self.time_points[steady_mask] / 1800)  # 30 min cycle
        noise = 0.05 * current_range[1] * np.random.normal(0, 1, np.sum(steady_mask))
        
        current_density[steady_mask] = base_current + variation + noise
        current_density = np.clip(current_density, current_range[0], current_range[1])
        
        return current_density
    
    def _generate_electrochemical_response(self) -> Dict[str, Any]:
        """Generate electrochemical response data."""
        print("  Generating electrochemical response data...")
        
        # Get current density
        current_density = self._generate_current_density()
        
        # Generate cell voltage based on current density and operating conditions
        cell_voltage = self._calculate_cell_voltage(current_density)
        
        # Generate EIS spectra at different operating points
        eis_data = self._generate_eis_spectra()
        
        # Generate power density
        power_density = current_density * cell_voltage
        
        return {
            'cell_voltage': cell_voltage,
            'stack_voltage': cell_voltage * 10,  # Assume 10 cells in stack
            'current_density': current_density,
            'power_density': power_density,
            'eis_spectra': eis_data,
            'time_points': self.time_points
        }
    
    def _calculate_cell_voltage(self, current_density: np.ndarray) -> np.ndarray:
        """Calculate cell voltage using simplified Butler-Volmer equation."""
        # Open circuit voltage (Nernst equation)
        T = 750 + 273.15  # Operating temperature in K
        R = 8.314  # J/mol/K
        F = 96485  # C/mol
        
        # Partial pressures (simplified)
        p_H2 = 0.97 * 1e5  # Pa
        p_H2O = 0.01 * 1e5  # Pa
        p_O2 = 0.21 * 1e5  # Pa
        
        E_ocv = (R * T / (2 * F)) * np.log(p_H2 * np.sqrt(p_O2) / p_H2O)
        
        # Overpotentials
        # Activation overpotential (simplified)
        i_0 = 0.1  # Exchange current density A/cm²
        eta_act = (R * T / F) * np.arcsinh(current_density / (2 * i_0))
        
        # Ohmic overpotential
        R_ohmic = 0.1  # Ohmic resistance Ω·cm²
        eta_ohmic = current_density * R_ohmic
        
        # Concentration overpotential (simplified)
        i_L = 2.0  # Limiting current density A/cm²
        eta_conc = (R * T / F) * np.log(1 - current_density / i_L)
        
        # Total cell voltage
        cell_voltage = E_ocv - eta_act - eta_ohmic - eta_conc
        
        return np.clip(cell_voltage, 0.5, 1.2)  # Reasonable voltage range
    
    def _generate_eis_spectra(self) -> Dict[str, Any]:
        """Generate EIS spectra at different operating points."""
        # Frequency range for EIS
        frequencies = np.logspace(0, 5, 50)  # 1 Hz to 100 kHz
        
        # Generate EIS at different current densities
        current_points = [0.1, 0.3, 0.5, 0.7, 0.9]  # A/cm²
        eis_spectra = {}
        
        for i, current in enumerate(current_points):
            # Generate impedance data (simplified Randles circuit)
            R_ohmic = 0.1 + 0.02 * i  # Increasing ohmic resistance
            R_ct = 0.2 + 0.1 * i      # Charge transfer resistance
            C_dl = 1e-3 / (1 + i)     # Double layer capacitance
            
            # Calculate impedance
            Z_real = R_ohmic + R_ct / (1 + (2 * np.pi * frequencies * R_ct * C_dl)**2)
            Z_imag = -R_ct * 2 * np.pi * frequencies * R_ct * C_dl / (1 + (2 * np.pi * frequencies * R_ct * C_dl)**2)
            
            eis_spectra[f'current_{current:.1f}A_cm2'] = {
                'frequencies': frequencies,
                'real_impedance': Z_real,
                'imaginary_impedance': Z_imag,
                'magnitude': np.sqrt(Z_real**2 + Z_imag**2),
                'phase': np.arctan2(Z_imag, Z_real)
            }
        
        return eis_spectra
    
    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics and efficiency data."""
        print("  Generating performance metrics...")
        
        # Get electrochemical data
        current_density = self._generate_current_density()
        cell_voltage = self._calculate_cell_voltage(current_density)
        
        # Calculate efficiency
        power_density = current_density * cell_voltage
        
        # Theoretical maximum power (simplified)
        max_power = 1.0  # W/cm²
        efficiency = power_density / max_power
        
        # Fuel utilization efficiency
        fuel_utilization = self.operating_conditions['fuel_utilization'] * np.ones_like(self.time_points)
        
        # Heat generation rate
        heat_generation = current_density * (1.2 - cell_voltage)  # Simplified
        
        return {
            'efficiency': efficiency,
            'fuel_utilization': fuel_utilization,
            'heat_generation_rate': heat_generation,
            'specific_power': power_density,
            'energy_density': np.cumsum(power_density) * self.time_step / 3600,  # Wh/cm²
            'time_points': self.time_points
        }
    
    def generate_operating_scenarios(self) -> Dict[str, Any]:
        """Generate different operating scenarios for testing."""
        scenarios = {}
        
        # Steady state operation
        scenarios['steady_state'] = {
            'current_density': 0.5 * np.ones(100),
            'temperature': 750 * np.ones(100),
            'fuel_flow': 0.1 * np.ones(100)
        }
        
        # Load following
        scenarios['load_following'] = {
            'current_density': 0.3 + 0.4 * np.sin(np.linspace(0, 4*np.pi, 100)),
            'temperature': 750 + 20 * np.sin(np.linspace(0, 2*np.pi, 100)),
            'fuel_flow': 0.1 + 0.02 * np.sin(np.linspace(0, 6*np.pi, 100))
        }
        
        # Thermal cycling
        scenarios['thermal_cycling'] = {
            'current_density': 0.5 * np.ones(100),
            'temperature': 600 + 200 * (np.sin(np.linspace(0, 2*np.pi, 100)) + 1) / 2,
            'fuel_flow': 0.1 * np.ones(100)
        }
        
        return scenarios
    
    def save_operational_data(self, data: Dict[str, Any]) -> str:
        """Save operational data to file."""
        return self.save_data(data, 'operational_electrochemical/operational_data.h5')
    
    def save_electrochemical_data(self, data: Dict[str, Any]) -> str:
        """Save electrochemical data to file."""
        return self.save_data(data, 'operational_electrochemical/electrochemical_data.h5')