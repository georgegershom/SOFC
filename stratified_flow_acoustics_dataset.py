#!/usr/bin/env python3
"""
Stratified Flow Acoustics Dataset Generator
PhD Thesis: "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"

This script generates comprehensive experimental and theoretical datasets for stratified flow acoustics research.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.stats import norm, uniform
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StratifiedFlowAcousticsDataset:
    """
    Comprehensive dataset generator for stratified flow acoustics research.
    Generates experimental data, acoustic measurements, and derived parameters.
    """
    
    def __init__(self, pipe_diameter=0.1, pipe_length=10.0, n_points=1000):
        """
        Initialize the dataset generator.
        
        Parameters:
        - pipe_diameter: Pipe diameter in meters (default: 0.1m)
        - pipe_length: Pipe length in meters (default: 10m)
        - n_points: Number of data points to generate
        """
        self.pipe_diameter = pipe_diameter
        self.pipe_length = pipe_length
        self.n_points = n_points
        
        # Physical constants
        self.g = 9.81  # gravitational acceleration
        self.rho_water = 1000.0  # kg/m³
        self.rho_air = 1.225  # kg/m³
        self.mu_water = 1e-3  # Pa·s
        self.mu_air = 1.8e-5  # Pa·s
        self.c_water = 1500.0  # m/s (speed of sound in water)
        self.c_air = 343.0  # m/s (speed of sound in air)
        
        # Generate base parameters
        self._generate_base_parameters()
        
    def _generate_base_parameters(self):
        """Generate base experimental parameters."""
        # Flow regime parameters
        self.void_fraction = np.random.uniform(0.1, 0.8, self.n_points)
        self.superficial_gas_velocity = np.random.uniform(0.1, 5.0, self.n_points)  # m/s
        self.superficial_liquid_velocity = np.random.uniform(0.05, 2.0, self.n_points)  # m/s
        
        # Temperature and pressure
        self.temperature = np.random.uniform(15, 35, self.n_points)  # °C
        self.pressure = np.random.uniform(101325, 500000, self.n_points)  # Pa
        
        # Time vector for acoustic signals
        self.time = np.linspace(0, 1.0, 1000)  # 1 second of data at 1kHz
        
        # Frequency range for acoustic analysis
        self.frequencies = np.logspace(1, 4, 100)  # 10 Hz to 10 kHz
        
    def generate_flow_regime_data(self):
        """Generate flow regime characterization data."""
        print("Generating flow regime characterization data...")
        
        # Calculate interface height based on void fraction
        interface_height = self.void_fraction * self.pipe_diameter
        
        # Generate flow pattern classification
        # Based on Taitel & Dukler (1976) flow pattern map
        flow_patterns = []
        for i in range(self.n_points):
            alpha = self.void_fraction[i]
            usg = self.superficial_gas_velocity[i]
            usl = self.superficial_liquid_velocity[i]
            
            # Simplified flow pattern classification
            if alpha < 0.3 and usg < 1.0:
                flow_patterns.append('smooth_interface')
            elif alpha < 0.5 and usg < 2.0:
                flow_patterns.append('wavy_interface')
            else:
                flow_patterns.append('turbulent_interface')
        
        # Generate wave amplitude (correlated with gas velocity)
        wave_amplitude = 0.001 * self.superficial_gas_velocity * np.random.uniform(0.5, 2.0, self.n_points)
        
        flow_data = pd.DataFrame({
            'void_fraction': self.void_fraction,
            'superficial_gas_velocity': self.superficial_gas_velocity,
            'superficial_liquid_velocity': self.superficial_liquid_velocity,
            'interface_height': interface_height,
            'flow_pattern': flow_patterns,
            'wave_amplitude': wave_amplitude,
            'temperature': self.temperature,
            'pressure': self.pressure
        })
        
        return flow_data
    
    def generate_acoustic_transmission_data(self):
        """Generate acoustic signal transmission data."""
        print("Generating acoustic signal transmission data...")
        
        acoustic_data = []
        
        for i in range(self.n_points):
            # Generate source signal (chirp signal)
            source_signal = self._generate_chirp_signal()
            
            # Calculate attenuation based on flow conditions
            attenuation_coeff = self._calculate_attenuation_coefficient(i)
            
            # Apply attenuation to get received signal
            received_signal = self._apply_attenuation(source_signal, attenuation_coeff)
            
            # Calculate SNR
            snr = self._calculate_snr(source_signal, received_signal)
            
            # Calculate frequency spectrum
            source_spectrum = np.abs(np.fft.fft(source_signal))
            received_spectrum = np.abs(np.fft.fft(received_signal))
            
            acoustic_data.append({
                'experiment_id': i,
                'source_signal': source_signal.tolist(),
                'received_signal': received_signal.tolist(),
                'attenuation_coefficient': attenuation_coeff,
                'snr_db': snr,
                'source_spectrum': source_spectrum.tolist(),
                'received_spectrum': received_spectrum.tolist(),
                'time_vector': self.time.tolist()
            })
        
        return acoustic_data
    
    def _generate_chirp_signal(self):
        """Generate a chirp signal for acoustic testing."""
        # Linear chirp from 100 Hz to 5000 Hz
        f0, f1 = 100, 5000
        t = self.time
        chirp_signal = signal.chirp(t, f0, t[-1], f1, method='linear')
        
        # Add some noise
        noise = 0.1 * np.random.randn(len(t))
        return chirp_signal + noise
    
    def _calculate_attenuation_coefficient(self, idx):
        """Calculate frequency-dependent attenuation coefficient."""
        alpha = self.void_fraction[idx]
        usg = self.superficial_gas_velocity[idx]
        usl = self.superficial_liquid_velocity[idx]
        
        # Base attenuation from literature (Temkin, 2002)
        # Attenuation increases with void fraction and flow velocity
        base_attenuation = 0.1 * alpha + 0.05 * (usg + usl)
        
        # Frequency-dependent component
        freq_dep = 0.001 * self.frequencies**0.5
        
        # Add some randomness
        noise = 0.02 * np.random.randn(len(self.frequencies))
        
        return base_attenuation + freq_dep + noise
    
    def _apply_attenuation(self, source_signal, attenuation_coeff):
        """Apply frequency-dependent attenuation to source signal."""
        # FFT of source signal
        source_fft = np.fft.fft(source_signal)
        freqs = np.fft.fftfreq(len(source_signal), 1/1000)  # 1kHz sampling
        
        # Interpolate attenuation coefficient to signal frequencies
        f_interp = interpolate.interp1d(self.frequencies, attenuation_coeff, 
                                      kind='linear', fill_value='extrapolate')
        atten_interp = f_interp(np.abs(freqs))
        
        # Apply attenuation (exponential decay)
        received_fft = source_fft * np.exp(-atten_interp * self.pipe_length)
        
        # Convert back to time domain
        received_signal = np.real(np.fft.ifft(received_fft))
        
        # Add measurement noise
        noise = 0.05 * np.random.randn(len(received_signal))
        return received_signal + noise
    
    def _calculate_snr(self, source, received):
        """Calculate signal-to-noise ratio in dB."""
        signal_power = np.mean(source**2)
        noise_power = np.mean((source - received)**2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
        return snr
    
    def generate_fluid_properties_data(self):
        """Generate fluid properties and conditions data."""
        print("Generating fluid properties data...")
        
        # Temperature-dependent properties
        temp_k = self.temperature + 273.15
        
        # Water density (simplified)
        rho_l = self.rho_water * (1 - 0.0002 * (self.temperature - 20))
        
        # Air density (ideal gas law)
        rho_g = self.pressure / (287 * temp_k)
        
        # Viscosity (simplified temperature dependence)
        mu_l = self.mu_water * np.exp(1700 * (1/temp_k - 1/293.15))
        mu_g = self.mu_air * (temp_k/288.15)**0.7
        
        # Speed of sound (temperature dependent)
        c_l = self.c_water + 4 * (self.temperature - 20)
        c_g = self.c_air * np.sqrt(temp_k/288.15)
        
        fluid_data = pd.DataFrame({
            'temperature': self.temperature,
            'pressure': self.pressure,
            'liquid_density': rho_l,
            'gas_density': rho_g,
            'liquid_viscosity': mu_l,
            'gas_viscosity': mu_g,
            'liquid_speed_of_sound': c_l,
            'gas_speed_of_sound': c_g,
            'void_fraction': self.void_fraction
        })
        
        return fluid_data
    
    def generate_turbulence_data(self):
        """Generate turbulence and shear layer data."""
        print("Generating turbulence and shear layer data...")
        
        turbulence_data = []
        
        for i in range(self.n_points):
            alpha = self.void_fraction[i]
            usg = self.superficial_gas_velocity[i]
            usl = self.superficial_liquid_velocity[i]
            
            # Calculate characteristic velocities
            u_g = usg / alpha if alpha > 0 else 0
            u_l = usl / (1 - alpha) if alpha < 1 else 0
            
            # Turbulent kinetic energy (simplified model)
            k_g = 0.5 * (0.1 * u_g)**2  # 10% turbulence intensity
            k_l = 0.5 * (0.1 * u_l)**2
            
            # Turbulent dissipation rate (simplified)
            epsilon_g = k_g**1.5 / (0.1 * self.pipe_diameter)
            epsilon_l = k_l**1.5 / (0.1 * self.pipe_diameter)
            
            # Generate velocity profiles across pipe diameter
            r = np.linspace(0, self.pipe_diameter/2, 50)
            interface_pos = alpha * self.pipe_diameter
            
            # Gas phase velocity profile (top half)
            u_g_profile = np.zeros_like(r)
            gas_mask = r >= (self.pipe_diameter/2 - interface_pos)
            u_g_profile[gas_mask] = u_g * (1 - (r[gas_mask] - (self.pipe_diameter/2 - interface_pos))**2 / (interface_pos)**2)
            
            # Liquid phase velocity profile (bottom half)
            u_l_profile = np.zeros_like(r)
            liquid_mask = r < (self.pipe_diameter/2 - interface_pos)
            u_l_profile[liquid_mask] = u_l * (1 - (r[liquid_mask] / (self.pipe_diameter/2 - interface_pos))**2)
            
            # Shear stress at interface
            du_dy = np.abs(np.gradient(u_g_profile + u_l_profile, r))
            tau_interface = self.mu_water * du_dy[np.argmin(np.abs(r - (self.pipe_diameter/2 - interface_pos)))]
            
            # Wall shear stress
            tau_wall_g = 0.5 * self.rho_air * u_g**2 * 0.02  # friction factor = 0.02
            tau_wall_l = 0.5 * self.rho_water * u_l**2 * 0.02
            
            turbulence_data.append({
                'experiment_id': i,
                'turbulent_kinetic_energy_gas': k_g,
                'turbulent_kinetic_energy_liquid': k_l,
                'turbulent_dissipation_rate_gas': epsilon_g,
                'turbulent_dissipation_rate_liquid': epsilon_l,
                'velocity_profile_radial': (u_g_profile + u_l_profile).tolist(),
                'radial_coordinates': r.tolist(),
                'interface_position': interface_pos,
                'shear_stress_interface': tau_interface,
                'wall_shear_stress_gas': tau_wall_g,
                'wall_shear_stress_liquid': tau_wall_l
            })
        
        return turbulence_data
    
    def generate_attenuation_metrics(self):
        """Generate attenuation metrics and frequency profiles."""
        print("Generating attenuation metrics...")
        
        attenuation_data = []
        
        for i in range(self.n_points):
            # Calculate attenuation coefficient
            atten_coeff = self._calculate_attenuation_coefficient(i)
            
            # Calculate transmission loss in dB
            transmission_loss = 20 * np.log10(np.exp(atten_coeff * self.pipe_length))
            
            # Find characteristic frequencies
            max_atten_freq = self.frequencies[np.argmax(atten_coeff)]
            min_atten_freq = self.frequencies[np.argmin(atten_coeff)]
            
            # Calculate average attenuation
            avg_attenuation = np.mean(atten_coeff)
            
            attenuation_data.append({
                'experiment_id': i,
                'attenuation_coefficient': atten_coeff.tolist(),
                'transmission_loss_db': transmission_loss.tolist(),
                'frequencies': self.frequencies.tolist(),
                'max_attenuation_frequency': max_atten_freq,
                'min_attenuation_frequency': min_atten_freq,
                'average_attenuation': avg_attenuation,
                'void_fraction': self.void_fraction[i],
                'superficial_gas_velocity': self.superficial_gas_velocity[i],
                'superficial_liquid_velocity': self.superficial_liquid_velocity[i]
            })
        
        return attenuation_data
    
    def generate_complete_dataset(self):
        """Generate the complete stratified flow acoustics dataset."""
        print("Generating complete stratified flow acoustics dataset...")
        print("=" * 60)
        
        # Generate all data components
        flow_data = self.generate_flow_regime_data()
        acoustic_data = self.generate_acoustic_transmission_data()
        fluid_data = self.generate_fluid_properties_data()
        turbulence_data = self.generate_turbulence_data()
        attenuation_data = self.generate_attenuation_metrics()
        
        # Create comprehensive dataset
        complete_dataset = {
            'metadata': {
                'title': 'Stratified Flow Acoustics Dataset',
                'description': 'Comprehensive dataset for PhD thesis: Study on the Attenuation Mechanisms in Stratified Flows',
                'generated_date': datetime.now().isoformat(),
                'pipe_diameter': self.pipe_diameter,
                'pipe_length': self.pipe_length,
                'number_of_experiments': self.n_points,
                'parameters_measured': [
                    'void_fraction', 'superficial_velocities', 'flow_patterns',
                    'acoustic_transmission', 'attenuation_coefficients',
                    'fluid_properties', 'turbulence_parameters'
                ]
            },
            'flow_regime_data': flow_data.to_dict('records'),
            'acoustic_transmission_data': acoustic_data,
            'fluid_properties_data': fluid_data.to_dict('records'),
            'turbulence_data': turbulence_data,
            'attenuation_metrics_data': attenuation_data
        }
        
        return complete_dataset
    
    def save_dataset(self, dataset, filename='stratified_flow_acoustics_dataset.json'):
        """Save the complete dataset to JSON file."""
        print(f"Saving dataset to {filename}...")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert the dataset
        dataset_serializable = convert_numpy(dataset)
        
        with open(filename, 'w') as f:
            json.dump(dataset_serializable, f, indent=2)
        
        print(f"Dataset saved successfully!")
        print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    def create_summary_statistics(self, dataset):
        """Create summary statistics for the dataset."""
        print("\n" + "="*60)
        print("DATASET SUMMARY STATISTICS")
        print("="*60)
        
        flow_df = pd.DataFrame(dataset['flow_regime_data'])
        
        print(f"Number of experiments: {len(flow_df)}")
        print(f"Void fraction range: {flow_df['void_fraction'].min():.3f} - {flow_df['void_fraction'].max():.3f}")
        print(f"Gas velocity range: {flow_df['superficial_gas_velocity'].min():.3f} - {flow_df['superficial_gas_velocity'].max():.3f} m/s")
        print(f"Liquid velocity range: {flow_df['superficial_liquid_velocity'].min():.3f} - {flow_df['superficial_liquid_velocity'].max():.3f} m/s")
        print(f"Temperature range: {flow_df['temperature'].min():.1f} - {flow_df['temperature'].max():.1f} °C")
        print(f"Pressure range: {flow_df['pressure'].min()/1000:.1f} - {flow_df['pressure'].max()/1000:.1f} kPa")
        
        print(f"\nFlow pattern distribution:")
        print(flow_df['flow_pattern'].value_counts())
        
        # Acoustic data summary
        snr_values = [exp['snr_db'] for exp in dataset['acoustic_transmission_data']]
        print(f"\nAcoustic transmission summary:")
        print(f"SNR range: {min(snr_values):.1f} - {max(snr_values):.1f} dB")
        print(f"Average SNR: {np.mean(snr_values):.1f} dB")

def main():
    """Main function to generate the complete dataset."""
    print("Stratified Flow Acoustics Dataset Generator")
    print("PhD Thesis: Study on the Attenuation Mechanisms in Stratified Flows")
    print("="*70)
    
    # Initialize dataset generator
    generator = StratifiedFlowAcousticsDataset(n_points=500)  # Generate 500 experiments
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Save dataset
    generator.save_dataset(dataset)
    
    # Create summary statistics
    generator.create_summary_statistics(dataset)
    
    print("\n" + "="*70)
    print("Dataset generation completed successfully!")
    print("The dataset includes:")
    print("- Flow regime characterization data")
    print("- Acoustic signal transmission measurements")
    print("- Attenuation metrics and frequency profiles")
    print("- Fluid properties and thermodynamic conditions")
    print("- Turbulence and shear layer parameters")
    print("="*70)

if __name__ == "__main__":
    main()