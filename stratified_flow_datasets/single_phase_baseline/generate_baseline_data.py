"""
Generate Single-Phase Baseline Data for Acoustic Attenuation Studies
This script creates control datasets for water-only and air-only conditions
Based on theoretical models and empirical correlations from literature
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy import signal
from scipy.special import jv  # Bessel function

class SinglePhaseBaseline:
    def __init__(self):
        # Physical properties at 20°C, 1 atm
        self.water_properties = {
            'density': 998.2,  # kg/m³
            'sound_speed': 1482.3,  # m/s at 20°C
            'viscosity': 1.002e-3,  # Pa·s
            'bulk_modulus': 2.19e9,  # Pa
            'thermal_conductivity': 0.598,  # W/(m·K)
            'specific_heat': 4182,  # J/(kg·K)
            'thermal_expansion': 2.07e-4,  # 1/K
        }
        
        self.air_properties = {
            'density': 1.204,  # kg/m³
            'sound_speed': 343.2,  # m/s at 20°C
            'viscosity': 1.825e-5,  # Pa·s
            'bulk_modulus': 1.42e5,  # Pa
            'thermal_conductivity': 0.0257,  # W/(m·K)
            'specific_heat': 1005,  # J/(kg·K)
            'thermal_expansion': 3.43e-3,  # 1/K
            'adiabatic_index': 1.4
        }
        
        # Pipe configuration
        self.pipe_config = {
            'diameter': 0.05,  # m (50 mm)
            'length': 10.0,  # m
            'wall_thickness': 0.003,  # m (3 mm)
            'material': 'PVC',
            'roughness': 1.5e-6,  # m
        }
        
    def calculate_attenuation_coefficient(self, freq, medium='water'):
        """
        Calculate frequency-dependent attenuation coefficient
        Based on classical absorption (viscous + thermal) and scattering
        """
        if medium == 'water':
            props = self.water_properties
            # Classical absorption in water (Stokes-Kirchhoff)
            omega = 2 * np.pi * freq
            
            # Viscous attenuation
            alpha_visc = (2 * props['viscosity'] * omega**2) / (3 * props['density'] * props['sound_speed']**3)
            
            # Thermal attenuation
            gamma = props['specific_heat'] * props['thermal_expansion'] * props['sound_speed']**2 / props['thermal_conductivity']
            alpha_thermal = (omega**2 * props['thermal_conductivity'] * (gamma - 1)) / (2 * props['density'] * props['sound_speed']**3 * props['specific_heat'])
            
            # Additional frequency-dependent term (empirical)
            alpha_relax = 3.3e-4 * freq**2 / (1 + freq**2/1e6)  # Relaxation processes
            
            alpha_total = alpha_visc + alpha_thermal + alpha_relax
            
        else:  # air
            props = self.air_properties
            omega = 2 * np.pi * freq
            
            # Classical attenuation in air
            mu = props['viscosity']
            k = props['thermal_conductivity']
            cp = props['specific_heat']
            cv = cp / props['adiabatic_index']
            
            # Stokes-Kirchhoff formula
            alpha_classical = (omega**2 / (2 * props['density'] * props['sound_speed']**3)) * (
                (4/3) * mu + k * (props['adiabatic_index'] - 1) / cp
            )
            
            # Molecular relaxation (simplified)
            alpha_relax = 1.84e-11 * freq**2 * np.sqrt(293/293) * (0.01 + 100 * 0.05 / (0.05 + freq**2/400**2))
            
            alpha_total = alpha_classical + alpha_relax
            
        return alpha_total * 1e3  # Convert to dB/m
    
    def generate_frequency_sweep(self, medium='water'):
        """Generate frequency sweep data for attenuation and sound speed"""
        frequencies = np.logspace(1, 6, 100)  # 10 Hz to 1 MHz
        
        data = {
            'frequency_Hz': frequencies,
            'attenuation_dB_per_m': [],
            'sound_speed_m_per_s': [],
            'phase_velocity_m_per_s': [],
            'group_velocity_m_per_s': [],
            'wavelength_m': [],
            'acoustic_impedance_Pa_s_per_m': []
        }
        
        props = self.water_properties if medium == 'water' else self.air_properties
        base_speed = props['sound_speed']
        
        for freq in frequencies:
            # Attenuation
            alpha = self.calculate_attenuation_coefficient(freq, medium)
            data['attenuation_dB_per_m'].append(alpha)
            
            # Dispersion effects (small for single phase)
            k = 2 * np.pi * freq / base_speed
            # Small frequency-dependent correction
            dispersion = 1 + (1e-8 * freq if medium == 'water' else 1e-9 * freq)
            
            phase_vel = base_speed * dispersion
            group_vel = phase_vel * (1 - freq * 1e-11)  # Small dispersion
            
            data['sound_speed_m_per_s'].append(base_speed)
            data['phase_velocity_m_per_s'].append(phase_vel)
            data['group_velocity_m_per_s'].append(group_vel)
            data['wavelength_m'].append(base_speed / freq)
            data['acoustic_impedance_Pa_s_per_m'].append(props['density'] * base_speed)
        
        return pd.DataFrame(data)
    
    def generate_temperature_dependence(self, medium='water'):
        """Generate temperature-dependent acoustic properties"""
        temperatures = np.linspace(5, 40, 36)  # 5°C to 40°C
        frequencies = [100, 1000, 10000, 100000]  # Hz
        
        data_list = []
        
        for temp in temperatures:
            for freq in frequencies:
                if medium == 'water':
                    # Marczak equation for water sound speed
                    t = temp
                    c = 1402.385 + 5.03830*t - 5.81090e-2*t**2 + 3.34320e-4*t**3 - 1.47800e-6*t**4 + 3.14640e-9*t**5
                    
                    # Temperature-dependent density (IAPWS formulation simplified)
                    rho = 999.974950 * (1 - ((t-3.98)**2 * (t+283))/(503570*(t+67.26)))
                    
                    # Viscosity (Vogel equation)
                    mu = 1.002e-3 * np.exp(-1.94 - 4.8*(t-20)/20 + 6.74*(t-20)**2/400)
                    
                else:  # air
                    # Ideal gas approximation
                    T_kelvin = temp + 273.15
                    c = 331.3 * np.sqrt(T_kelvin / 273.15)
                    rho = 1.293 * (273.15 / T_kelvin) * (101325 / 101325)  # Assume constant pressure
                    mu = 1.716e-5 * (T_kelvin / 273.15)**0.7
                
                # Recalculate attenuation with temperature-dependent properties
                omega = 2 * np.pi * freq
                alpha = (2 * mu * omega**2) / (3 * rho * c**3)
                
                data_list.append({
                    'temperature_C': temp,
                    'frequency_Hz': freq,
                    'sound_speed_m_per_s': c,
                    'density_kg_per_m3': rho,
                    'viscosity_Pa_s': mu,
                    'attenuation_dB_per_m': alpha * 8.686 * 1e3,  # Np/m to dB/m
                    'acoustic_impedance_Pa_s_per_m': rho * c,
                    'medium': medium
                })
        
        return pd.DataFrame(data_list)
    
    def generate_pressure_dependence(self, medium='water'):
        """Generate pressure-dependent acoustic properties"""
        pressures = np.linspace(1, 10, 10)  # 1 to 10 bar
        frequencies = [100, 1000, 10000, 100000]  # Hz
        temp = 20  # °C
        
        data_list = []
        
        for pressure in pressures:
            for freq in frequencies:
                if medium == 'water':
                    # Pressure effects on water (simplified Tait equation)
                    P_Pa = pressure * 1e5
                    K0 = 2.19e9  # Bulk modulus at 1 atm
                    n = 7.15  # Tait parameter
                    
                    # Sound speed from bulk modulus
                    K = K0 * (1 + n * (P_Pa - 1e5) / K0)
                    rho = 998.2 * (1 + (P_Pa - 1e5) / K0)
                    c = np.sqrt(K / rho)
                    
                else:  # air
                    # Ideal gas
                    P_Pa = pressure * 1e5
                    rho = 1.204 * pressure  # Proportional to pressure
                    c = 343.2  # Approximately constant for ideal gas
                
                # Attenuation (pressure effect is small)
                alpha = self.calculate_attenuation_coefficient(freq, medium) * (pressure**0.1)
                
                data_list.append({
                    'pressure_bar': pressure,
                    'frequency_Hz': freq,
                    'sound_speed_m_per_s': c,
                    'density_kg_per_m3': rho,
                    'attenuation_dB_per_m': alpha,
                    'acoustic_impedance_Pa_s_per_m': rho * c,
                    'medium': medium
                })
        
        return pd.DataFrame(data_list)
    
    def generate_pipe_modes(self, medium='water'):
        """Calculate acoustic modes in cylindrical pipe"""
        R = self.pipe_config['diameter'] / 2
        props = self.water_properties if medium == 'water' else self.air_properties
        c = props['sound_speed']
        
        # Frequency range
        frequencies = np.linspace(100, 50000, 200)
        
        modes_data = []
        
        # Consider first few modes (n, m)
        for n in range(3):  # Circumferential modes
            for m in range(1, 4):  # Radial modes
                # Bessel function zeros for mode calculation
                if n == 0 and m == 1:
                    alpha_nm = 0  # Plane wave mode
                else:
                    # Approximate zeros of Bessel functions
                    if n == 0:
                        alpha_nm = [0, 3.83, 7.02][m-1] if m <= 2 else 3.83 + (m-1)*np.pi
                    elif n == 1:
                        alpha_nm = [1.84, 5.33, 8.54][m-1] if m <= 3 else 1.84 + (m-1)*np.pi
                    else:
                        alpha_nm = 3.05 + n*1.2 + (m-1)*np.pi
                
                for freq in frequencies:
                    k = 2 * np.pi * freq / c
                    
                    # Cut-off frequency
                    f_cutoff = alpha_nm * c / (2 * np.pi * R) if alpha_nm > 0 else 0
                    
                    if freq > f_cutoff:
                        # Propagating mode
                        k_z = np.sqrt(k**2 - (alpha_nm/R)**2)
                        phase_vel = 2 * np.pi * freq / k_z
                        group_vel = c**2 / phase_vel
                        attenuation = self.calculate_attenuation_coefficient(freq, medium) * (1 + 0.1*n + 0.05*m)
                    else:
                        # Evanescent mode
                        k_z = 0
                        phase_vel = np.inf
                        group_vel = 0
                        attenuation = 1000  # High attenuation for evanescent modes
                    
                    modes_data.append({
                        'frequency_Hz': freq,
                        'mode_n': n,
                        'mode_m': m,
                        'cutoff_frequency_Hz': f_cutoff,
                        'wavenumber_1_per_m': k_z,
                        'phase_velocity_m_per_s': phase_vel if phase_vel != np.inf else None,
                        'group_velocity_m_per_s': group_vel,
                        'attenuation_dB_per_m': attenuation,
                        'propagating': freq > f_cutoff,
                        'medium': medium
                    })
        
        return pd.DataFrame(modes_data)
    
    def save_all_datasets(self):
        """Generate and save all baseline datasets"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Water baseline data
        print("Generating water baseline data...")
        water_freq = self.generate_frequency_sweep('water')
        water_temp = self.generate_temperature_dependence('water')
        water_pressure = self.generate_pressure_dependence('water')
        water_modes = self.generate_pipe_modes('water')
        
        water_freq.to_csv('water_frequency_sweep.csv', index=False)
        water_temp.to_csv('water_temperature_dependence.csv', index=False)
        water_pressure.to_csv('water_pressure_dependence.csv', index=False)
        water_modes.to_csv('water_pipe_modes.csv', index=False)
        
        # Air baseline data
        print("Generating air baseline data...")
        air_freq = self.generate_frequency_sweep('air')
        air_temp = self.generate_temperature_dependence('air')
        air_pressure = self.generate_pressure_dependence('air')
        air_modes = self.generate_pipe_modes('air')
        
        air_freq.to_csv('air_frequency_sweep.csv', index=False)
        air_temp.to_csv('air_temperature_dependence.csv', index=False)
        air_pressure.to_csv('air_pressure_dependence.csv', index=False)
        air_modes.to_csv('air_pipe_modes.csv', index=False)
        
        # Save metadata
        metadata = {
            'generation_timestamp': timestamp,
            'water_properties': self.water_properties,
            'air_properties': self.air_properties,
            'pipe_configuration': self.pipe_config,
            'data_files': {
                'water': [
                    'water_frequency_sweep.csv',
                    'water_temperature_dependence.csv',
                    'water_pressure_dependence.csv',
                    'water_pipe_modes.csv'
                ],
                'air': [
                    'air_frequency_sweep.csv',
                    'air_temperature_dependence.csv',
                    'air_pressure_dependence.csv',
                    'air_pipe_modes.csv'
                ]
            },
            'notes': 'Single-phase baseline data for acoustic attenuation studies in stratified flows'
        }
        
        with open('metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Baseline datasets saved with timestamp: {timestamp}")
        return metadata

if __name__ == "__main__":
    generator = SinglePhaseBaseline()
    metadata = generator.save_all_datasets()
    print("\nGeneration complete!")
    print(f"Total files created: {len(metadata['data_files']['water']) + len(metadata['data_files']['air'])}")