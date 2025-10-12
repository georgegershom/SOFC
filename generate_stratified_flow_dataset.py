#!/usr/bin/env python3
"""
Stratified Flow Acoustic Attenuation Dataset Generator

This script generates a comprehensive synthetic dataset for studying attenuation
mechanisms in stratified two-phase flows, based on established physical models
and empirical correlations from literature.

Author: Generated for PhD Research
Topic: Study on the Attenuation Mechanisms in Stratified Flows: 
       Beyond Single Phase Leakage Acoustics
Date: 2025-10-12
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Physical constants
G = 9.81  # gravitational acceleration (m/s^2)

class StratifiedFlowDataGenerator:
    """
    Generates realistic stratified flow and acoustic attenuation data
    based on physical models and empirical correlations.
    """
    
    def __init__(self, n_experiments=100, output_dir='stratified_flow_dataset'):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        n_experiments : int
            Number of experimental runs to simulate
        output_dir : str
            Directory to save generated datasets
        """
        self.n_experiments = n_experiments
        self.output_dir = output_dir
        self.pipe_diameter = 0.05  # 50 mm typical lab-scale pipe
        self.pipe_length = 5.0  # 5 meters
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_operating_conditions(self):
        """
        Generate realistic operating conditions for gas-liquid stratified flows.
        Returns different flow regimes: smooth stratified and wavy stratified.
        """
        data = []
        
        for exp_id in range(1, self.n_experiments + 1):
            # Generate superficial velocities
            # Stratified flow typically occurs at low to moderate gas and liquid velocities
            U_SG = np.random.uniform(0.5, 15.0)  # Gas superficial velocity (m/s)
            U_SL = np.random.uniform(0.01, 0.5)  # Liquid superficial velocity (m/s)
            
            # Determine flow pattern based on Taitel-Dukler map
            flow_pattern = self._classify_flow_pattern(U_SG, U_SL)
            
            # Void fraction using drift flux model
            # Simplified correlation for stratified flow
            void_fraction = self._calculate_void_fraction(U_SG, U_SL)
            
            # Interface height (non-dimensional, h/D)
            interface_height = 1.0 - void_fraction  # Approximation for stratified flow
            
            # Wave amplitude (for wavy stratified)
            if flow_pattern == 'wavy_stratified':
                wave_amplitude = np.random.uniform(0.001, 0.005)  # meters
                wave_frequency = np.random.uniform(2, 10)  # Hz
            else:
                wave_amplitude = np.random.uniform(0.0001, 0.001)  # minimal waves
                wave_frequency = np.random.uniform(0.5, 2)  # Hz
            
            # Fluid properties (Air-Water system at varying conditions)
            temperature = np.random.uniform(15, 30)  # Celsius
            pressure = np.random.uniform(1.0, 3.0)  # bar (absolute)
            
            # Water properties
            rho_L = 998.0 - 0.2 * (temperature - 20)  # kg/m^3
            mu_L = (1.002e-3) * np.exp(-0.025 * (temperature - 20))  # Pa.s
            
            # Air properties (ideal gas)
            rho_G = (pressure * 1e5) / (287.05 * (temperature + 273.15))  # kg/m^3
            mu_G = 1.81e-5 * (1 + 0.003 * (temperature - 15))  # Pa.s
            
            # Generate timestamp
            base_time = datetime(2024, 1, 1)
            timestamp = base_time + timedelta(hours=exp_id*2)
            
            data.append({
                'experiment_id': exp_id,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'U_SG': U_SG,
                'U_SL': U_SL,
                'void_fraction': void_fraction,
                'flow_pattern': flow_pattern,
                'interface_height': interface_height,
                'wave_amplitude': wave_amplitude,
                'wave_frequency': wave_frequency,
                'temperature': temperature,
                'pressure': pressure,
                'rho_gas': rho_G,
                'rho_liquid': rho_L,
                'mu_gas': mu_G,
                'mu_liquid': mu_L,
                'pipe_diameter': self.pipe_diameter,
                'pipe_length': self.pipe_length
            })
        
        return pd.DataFrame(data)
    
    def _classify_flow_pattern(self, U_SG, U_SL):
        """Classify flow pattern using simplified Taitel-Dukler criteria."""
        # Simplified classification
        if U_SG < 5 and U_SL < 0.2:
            return 'smooth_stratified'
        elif U_SG < 10:
            return 'wavy_stratified'
        else:
            return 'wavy_stratified'  # More likely to be wavy at high gas velocities
    
    def _calculate_void_fraction(self, U_SG, U_SL):
        """Calculate void fraction using homogeneous model with slip correction."""
        # Homogeneous void fraction
        alpha_h = U_SG / (U_SG + U_SL)
        
        # Apply slip correction for stratified flow
        # Slip ratio typically 1.5-3.0 for stratified flow
        slip_ratio = 1.5 + 0.5 * np.random.random()
        
        # Corrected void fraction
        alpha = alpha_h * slip_ratio / (1 + alpha_h * (slip_ratio - 1))
        
        # Ensure physical bounds
        alpha = np.clip(alpha, 0.05, 0.95)
        
        return alpha
    
    def generate_turbulence_data(self, operating_conditions):
        """
        Generate turbulence and shear layer data based on operating conditions.
        """
        data = []
        
        for _, row in operating_conditions.iterrows():
            exp_id = int(row['experiment_id'])
            U_SG = row['U_SG']
            U_SL = row['U_SL']
            rho_G = row['rho_gas']
            rho_L = row['rho_liquid']
            mu_G = row['mu_gas']
            mu_L = row['mu_liquid']
            alpha = row['void_fraction']
            D = row['pipe_diameter']
            
            # Reynolds numbers
            Re_G = rho_G * U_SG * D / mu_G
            Re_L = rho_L * U_SL * D / mu_L
            
            # Friction factors (using Blasius for turbulent, Hagen-Poiseuille for laminar)
            if Re_G > 2300:
                f_G = 0.316 * Re_G**(-0.25)
            else:
                f_G = 64 / Re_G
                
            if Re_L > 2300:
                f_L = 0.316 * Re_L**(-0.25)
            else:
                f_L = 64 / Re_L
            
            # Wall shear stress
            tau_wall_gas = 0.5 * rho_G * f_G * U_SG**2
            tau_wall_liquid = 0.5 * rho_L * f_L * U_SL**2
            
            # Interfacial shear stress (using empirical correlation)
            velocity_difference = abs(U_SG - U_SL)
            tau_interface = 0.5 * rho_G * 0.01 * velocity_difference**2  # Simplified
            
            # Turbulent kinetic energy (k)
            # Estimated from k ~ (u_tau)^2 / sqrt(C_mu), where u_tau is friction velocity
            u_tau_G = np.sqrt(tau_wall_gas / rho_G)
            u_tau_L = np.sqrt(tau_wall_liquid / rho_L)
            
            C_mu = 0.09  # Standard k-epsilon model constant
            k_gas = u_tau_G**2 / np.sqrt(C_mu)
            k_liquid = u_tau_L**2 / np.sqrt(C_mu)
            
            # Turbulent dissipation rate (epsilon)
            # epsilon ~ u_tau^3 / (kappa * y), using characteristic length
            kappa = 0.41  # von Karman constant
            y_char_G = alpha * D / 2
            y_char_L = (1 - alpha) * D / 2
            
            epsilon_gas = u_tau_G**3 / (kappa * max(y_char_G, 0.001))
            epsilon_liquid = u_tau_L**3 / (kappa * max(y_char_L, 0.001))
            
            # Mean velocity profiles (simplified power law)
            # Generate profiles at different radial positions
            n_positions = 10
            radial_positions = np.linspace(0, D/2, n_positions)
            
            data.append({
                'experiment_id': exp_id,
                'Re_gas': Re_G,
                'Re_liquid': Re_L,
                'friction_factor_gas': f_G,
                'friction_factor_liquid': f_L,
                'tau_wall_gas': tau_wall_gas,
                'tau_wall_liquid': tau_wall_liquid,
                'tau_interface': tau_interface,
                'TKE_gas': k_gas,
                'TKE_liquid': k_liquid,
                'dissipation_rate_gas': epsilon_gas,
                'dissipation_rate_liquid': epsilon_liquid,
                'friction_velocity_gas': u_tau_G,
                'friction_velocity_liquid': u_tau_L
            })
        
        return pd.DataFrame(data)
    
    def generate_acoustic_data(self, operating_conditions):
        """
        Generate acoustic signal transmission and attenuation data.
        """
        # Acoustic source parameters
        source_frequencies = [100, 500, 1000, 2000, 5000, 10000]  # Hz
        sampling_rate = 51200  # Hz
        signal_duration = 1.0  # seconds
        
        acoustic_data = []
        attenuation_data = []
        
        for _, row in operating_conditions.iterrows():
            exp_id = int(row['experiment_id'])
            alpha = row['void_fraction']
            rho_G = row['rho_gas']
            rho_L = row['rho_liquid']
            mu_G = row['mu_gas']
            mu_L = row['mu_liquid']
            U_SG = row['U_SG']
            U_SL = row['U_SL']
            temperature = row['temperature']
            pressure = row['pressure']
            flow_pattern = row['flow_pattern']
            
            # Speed of sound in each phase
            c_gas = self._speed_of_sound_air(temperature, pressure)
            c_liquid = self._speed_of_sound_water(temperature)
            
            # For each frequency, calculate attenuation
            for freq in source_frequencies:
                # Calculate attenuation mechanisms
                
                # 1. Viscous attenuation (classical absorption)
                alpha_visc_gas = self._viscous_attenuation(freq, rho_G, mu_G, c_gas)
                alpha_visc_liquid = self._viscous_attenuation(freq, rho_L, mu_L, c_liquid)
                
                # 2. Scattering attenuation (from interface waves and bubbles)
                if flow_pattern == 'wavy_stratified':
                    wave_amp = row['wave_amplitude']
                    wave_freq = row['wave_frequency']
                    alpha_scatter = self._scattering_attenuation(freq, wave_amp, wave_freq, c_liquid)
                else:
                    alpha_scatter = 0.01 * freq / 1000  # minimal scattering
                
                # 3. Turbulence-induced attenuation
                alpha_turb = self._turbulence_attenuation(freq, U_SG, U_SL, alpha)
                
                # 4. Two-phase mixture effects
                # Effective properties using Wood's equation for sound speed
                rho_mix = alpha * rho_G + (1 - alpha) * rho_L
                K_mix = 1 / (alpha / (rho_G * c_gas**2) + (1 - alpha) / (rho_L * c_liquid**2))
                c_mix = np.sqrt(K_mix / rho_mix)
                
                # Total attenuation coefficient (Np/m)
                alpha_total = (alpha * alpha_visc_gas + (1 - alpha) * alpha_visc_liquid + 
                              alpha_scatter + alpha_turb)
                
                # Add random variation (measurement uncertainty)
                alpha_total *= (1 + np.random.normal(0, 0.1))
                
                # Transmission loss (dB)
                transmission_loss = 20 * np.log10(np.exp(1)) * alpha_total * self.pipe_length
                
                # Signal amplitudes
                source_amplitude = 1.0  # Normalized
                received_amplitude = source_amplitude * np.exp(-alpha_total * self.pipe_length)
                
                # SNR calculation
                # SNR decreases with attenuation and distance
                base_SNR = 40  # dB
                SNR = base_SNR - transmission_loss + np.random.normal(0, 2)
                SNR = max(SNR, 5)  # Minimum SNR
                
                attenuation_data.append({
                    'experiment_id': exp_id,
                    'frequency': freq,
                    'attenuation_coefficient': alpha_total,
                    'transmission_loss_dB': transmission_loss,
                    'source_amplitude': source_amplitude,
                    'received_amplitude': received_amplitude,
                    'SNR_dB': SNR,
                    'sound_speed_gas': c_gas,
                    'sound_speed_liquid': c_liquid,
                    'sound_speed_mixture': c_mix,
                    'viscous_atten_contribution': alpha_visc_gas * alpha + alpha_visc_liquid * (1-alpha),
                    'scattering_atten_contribution': alpha_scatter,
                    'turbulence_atten_contribution': alpha_turb
                })
        
        # Generate time-series acoustic data for selected experiments
        # (Only for first 10 experiments to keep file size manageable)
        timeseries_data = []
        for exp_id in range(1, min(11, self.n_experiments + 1)):
            row = operating_conditions[operating_conditions['experiment_id'] == exp_id].iloc[0]
            alpha_void = row['void_fraction']
            
            # Generate acoustic signal at 1000 Hz for detailed analysis
            t = np.linspace(0, signal_duration, int(sampling_rate * signal_duration))
            
            # Source signal (chirp or multi-frequency)
            source_signal = np.sin(2 * np.pi * 1000 * t)
            
            # Add harmonics
            source_signal += 0.3 * np.sin(2 * np.pi * 2000 * t)
            source_signal += 0.2 * np.sin(2 * np.pi * 500 * t)
            
            # Get attenuation for this experiment at 1000 Hz
            atten_row = [a for a in attenuation_data if a['experiment_id'] == exp_id and a['frequency'] == 1000][0]
            alpha_total = atten_row['attenuation_coefficient']
            
            # Received signal (attenuated)
            received_signal = source_signal * np.exp(-alpha_total * self.pipe_length)
            
            # Add noise based on SNR
            SNR = atten_row['SNR_dB']
            noise_power = 10**(-SNR/10)
            noise = np.random.normal(0, np.sqrt(noise_power), len(received_signal))
            received_signal += noise
            
            # Add flow-induced noise
            if row['flow_pattern'] == 'wavy_stratified':
                flow_noise = 0.05 * np.random.randn(len(t)) * (1 + 0.5 * np.sin(2 * np.pi * row['wave_frequency'] * t))
                received_signal += flow_noise
            
            # Save time series data (subsample to reduce size)
            subsample = 100
            timeseries_data.append({
                'experiment_id': exp_id,
                'time': t[::subsample].tolist(),
                'source_signal': source_signal[::subsample].tolist(),
                'received_signal': received_signal[::subsample].tolist(),
                'sampling_rate': sampling_rate
            })
        
        return pd.DataFrame(attenuation_data), timeseries_data
    
    def _speed_of_sound_air(self, temperature, pressure):
        """Calculate speed of sound in air."""
        T_kelvin = temperature + 273.15
        c = 331.3 * np.sqrt(T_kelvin / 273.15)
        return c
    
    def _speed_of_sound_water(self, temperature):
        """Calculate speed of sound in water."""
        # Empirical formula
        c = 1402.7 + 5.04 * temperature - 0.058 * temperature**2 + 0.00033 * temperature**3
        return c
    
    def _viscous_attenuation(self, freq, rho, mu, c):
        """Calculate classical viscous attenuation."""
        omega = 2 * np.pi * freq
        alpha = (2 * omega**2 * mu) / (3 * rho * c**3)
        return alpha
    
    def _scattering_attenuation(self, freq, wave_amplitude, wave_freq, c):
        """Calculate scattering attenuation from interface waves."""
        # Simplified scattering model
        k = 2 * np.pi * freq / c  # wave number
        alpha_scatter = 0.5 * k**2 * wave_amplitude * (1 + wave_freq / freq)
        return alpha_scatter
    
    def _turbulence_attenuation(self, freq, U_SG, U_SL, alpha):
        """Calculate turbulence-induced attenuation."""
        # Empirical model based on velocity fluctuations
        velocity_diff = abs(U_SG - U_SL)
        alpha_turb = 0.001 * (freq / 1000) * velocity_diff * (alpha * (1 - alpha))
        return alpha_turb
    
    def generate_velocity_profiles(self, operating_conditions, turbulence_data):
        """
        Generate detailed velocity profiles across pipe diameter.
        """
        profiles = []
        
        for _, row in operating_conditions.iterrows():
            exp_id = int(row['experiment_id'])
            U_SG = row['U_SG']
            U_SL = row['U_SL']
            alpha = row['void_fraction']
            D = row['pipe_diameter']
            
            # Get turbulence data
            turb_row = turbulence_data[turbulence_data['experiment_id'] == exp_id].iloc[0]
            u_tau_G = turb_row['friction_velocity_gas']
            u_tau_L = turb_row['friction_velocity_liquid']
            
            # Generate profiles
            n_points = 20
            
            # Gas phase profile (in upper region)
            y_gas = np.linspace(0, alpha * D, n_points)
            # Power law profile
            n_turb = 7  # turbulent exponent
            U_profile_gas = U_SG * (y_gas / (alpha * D))**(1/n_turb)
            
            # Liquid phase profile (in lower region)
            y_liquid = np.linspace(0, (1 - alpha) * D, n_points)
            U_profile_liquid = U_SL * (y_liquid / ((1 - alpha) * D))**(1/n_turb)
            
            for i in range(n_points):
                profiles.append({
                    'experiment_id': exp_id,
                    'phase': 'gas',
                    'radial_position': y_gas[i],
                    'normalized_position': y_gas[i] / D,
                    'velocity': U_profile_gas[i],
                    'normalized_velocity': U_profile_gas[i] / U_SG if U_SG > 0 else 0
                })
                
                profiles.append({
                    'experiment_id': exp_id,
                    'phase': 'liquid',
                    'radial_position': y_liquid[i],
                    'normalized_position': y_liquid[i] / D,
                    'velocity': U_profile_liquid[i],
                    'normalized_velocity': U_profile_liquid[i] / U_SL if U_SL > 0 else 0
                })
        
        return pd.DataFrame(profiles)
    
    def save_datasets(self):
        """
        Generate all datasets and save to files.
        """
        print("Generating Stratified Flow Acoustic Attenuation Dataset...")
        print("=" * 70)
        
        # 1. Generate operating conditions and flow regime data
        print("\n[1/6] Generating flow regime characterization data...")
        operating_conditions = self.generate_operating_conditions()
        operating_conditions.to_csv(
            os.path.join(self.output_dir, 'flow_regime_characterization.csv'),
            index=False
        )
        print(f"   ✓ Saved {len(operating_conditions)} experimental conditions")
        
        # 2. Generate turbulence data
        print("\n[2/6] Generating turbulence and shear layer data...")
        turbulence_data = self.generate_turbulence_data(operating_conditions)
        turbulence_data.to_csv(
            os.path.join(self.output_dir, 'turbulence_shear_data.csv'),
            index=False
        )
        print(f"   ✓ Saved turbulence data for {len(turbulence_data)} experiments")
        
        # 3. Generate acoustic and attenuation data
        print("\n[3/6] Generating acoustic transmission and attenuation data...")
        attenuation_data, timeseries_data = self.generate_acoustic_data(operating_conditions)
        attenuation_data.to_csv(
            os.path.join(self.output_dir, 'acoustic_attenuation_data.csv'),
            index=False
        )
        print(f"   ✓ Saved attenuation data: {len(attenuation_data)} measurements")
        
        # Save time series data as JSON
        with open(os.path.join(self.output_dir, 'acoustic_timeseries_data.json'), 'w') as f:
            json.dump(timeseries_data, f, indent=2)
        print(f"   ✓ Saved time-series data for {len(timeseries_data)} experiments")
        
        # 4. Generate velocity profiles
        print("\n[4/6] Generating velocity profiles...")
        velocity_profiles = self.generate_velocity_profiles(operating_conditions, turbulence_data)
        velocity_profiles.to_csv(
            os.path.join(self.output_dir, 'velocity_profiles.csv'),
            index=False
        )
        print(f"   ✓ Saved velocity profiles: {len(velocity_profiles)} data points")
        
        # 5. Create summary statistics
        print("\n[5/6] Generating summary statistics...")
        self._generate_summary_statistics(
            operating_conditions, turbulence_data, attenuation_data
        )
        
        # 6. Generate metadata and documentation
        print("\n[6/6] Creating metadata and documentation...")
        self._generate_metadata(operating_conditions, attenuation_data)
        
        print("\n" + "=" * 70)
        print("Dataset generation complete!")
        print(f"All files saved in: {self.output_dir}/")
        print("=" * 70)
        
        return operating_conditions, turbulence_data, attenuation_data
    
    def _generate_summary_statistics(self, operating_conditions, turbulence_data, attenuation_data):
        """Generate summary statistics for the dataset."""
        summary = {
            'dataset_info': {
                'total_experiments': len(operating_conditions),
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pipe_diameter_m': self.pipe_diameter,
                'pipe_length_m': self.pipe_length
            },
            'flow_regime_statistics': {
                'flow_patterns': operating_conditions['flow_pattern'].value_counts().to_dict(),
                'void_fraction': {
                    'mean': float(operating_conditions['void_fraction'].mean()),
                    'std': float(operating_conditions['void_fraction'].std()),
                    'min': float(operating_conditions['void_fraction'].min()),
                    'max': float(operating_conditions['void_fraction'].max())
                },
                'superficial_gas_velocity': {
                    'mean': float(operating_conditions['U_SG'].mean()),
                    'std': float(operating_conditions['U_SG'].std()),
                    'min': float(operating_conditions['U_SG'].min()),
                    'max': float(operating_conditions['U_SG'].max()),
                    'unit': 'm/s'
                },
                'superficial_liquid_velocity': {
                    'mean': float(operating_conditions['U_SL'].mean()),
                    'std': float(operating_conditions['U_SL'].std()),
                    'min': float(operating_conditions['U_SL'].min()),
                    'max': float(operating_conditions['U_SL'].max()),
                    'unit': 'm/s'
                }
            },
            'acoustic_statistics': {
                'frequencies_tested_Hz': sorted(attenuation_data['frequency'].unique().tolist()),
                'attenuation_coefficient': {
                    'mean': float(attenuation_data['attenuation_coefficient'].mean()),
                    'std': float(attenuation_data['attenuation_coefficient'].std()),
                    'min': float(attenuation_data['attenuation_coefficient'].min()),
                    'max': float(attenuation_data['attenuation_coefficient'].max()),
                    'unit': 'Np/m'
                },
                'transmission_loss': {
                    'mean': float(attenuation_data['transmission_loss_dB'].mean()),
                    'std': float(attenuation_data['transmission_loss_dB'].std()),
                    'min': float(attenuation_data['transmission_loss_dB'].min()),
                    'max': float(attenuation_data['transmission_loss_dB'].max()),
                    'unit': 'dB'
                },
                'SNR': {
                    'mean': float(attenuation_data['SNR_dB'].mean()),
                    'std': float(attenuation_data['SNR_dB'].std()),
                    'min': float(attenuation_data['SNR_dB'].min()),
                    'max': float(attenuation_data['SNR_dB'].max()),
                    'unit': 'dB'
                }
            },
            'turbulence_statistics': {
                'Reynolds_number_gas': {
                    'mean': float(turbulence_data['Re_gas'].mean()),
                    'std': float(turbulence_data['Re_gas'].std()),
                    'min': float(turbulence_data['Re_gas'].min()),
                    'max': float(turbulence_data['Re_gas'].max())
                },
                'Reynolds_number_liquid': {
                    'mean': float(turbulence_data['Re_liquid'].mean()),
                    'std': float(turbulence_data['Re_liquid'].std()),
                    'min': float(turbulence_data['Re_liquid'].min()),
                    'max': float(turbulence_data['Re_liquid'].max())
                },
                'TKE_gas': {
                    'mean': float(turbulence_data['TKE_gas'].mean()),
                    'std': float(turbulence_data['TKE_gas'].std()),
                    'unit': 'm^2/s^2'
                },
                'interfacial_shear_stress': {
                    'mean': float(turbulence_data['tau_interface'].mean()),
                    'std': float(turbulence_data['tau_interface'].std()),
                    'unit': 'Pa'
                }
            }
        }
        
        with open(os.path.join(self.output_dir, 'dataset_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✓ Saved summary statistics")
    
    def _generate_metadata(self, operating_conditions, attenuation_data):
        """Generate comprehensive metadata file."""
        metadata = {
            'dataset_name': 'Stratified Flow Acoustic Attenuation Dataset',
            'version': '1.0',
            'creation_date': datetime.now().strftime('%Y-%m-%d'),
            'description': 'Synthetic dataset for studying attenuation mechanisms in gas-liquid stratified flows',
            'research_topic': 'Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics',
            
            'experimental_setup': {
                'pipe_diameter_m': self.pipe_diameter,
                'pipe_length_m': self.pipe_length,
                'flow_system': 'Air-Water',
                'flow_configuration': 'Horizontal stratified two-phase flow'
            },
            
            'data_files': {
                'flow_regime_characterization.csv': {
                    'description': 'Flow regime parameters including void fraction, velocities, and flow patterns',
                    'rows': len(operating_conditions),
                    'columns': list(operating_conditions.columns)
                },
                'acoustic_attenuation_data.csv': {
                    'description': 'Acoustic attenuation coefficients, transmission loss, and SNR data',
                    'rows': len(attenuation_data),
                    'columns': list(attenuation_data.columns)
                },
                'acoustic_timeseries_data.json': {
                    'description': 'Time-series acoustic pressure data for selected experiments',
                    'format': 'JSON with arrays of time, source, and received signals'
                },
                'turbulence_shear_data.csv': {
                    'description': 'Turbulence statistics and shear stress measurements'
                },
                'velocity_profiles.csv': {
                    'description': 'Velocity profiles across pipe diameter for both phases'
                }
            },
            
            'physical_models': {
                'void_fraction': 'Drift flux model with slip correction',
                'flow_pattern_classification': 'Taitel-Dukler flow map',
                'attenuation_mechanisms': [
                    'Classical viscous absorption',
                    'Scattering from interface waves',
                    'Turbulence-induced attenuation',
                    'Two-phase mixture effects'
                ],
                'turbulence_model': 'k-epsilon model with wall functions',
                'velocity_profiles': 'Power law distribution'
            },
            
            'measurement_ranges': {
                'void_fraction': [0.05, 0.95],
                'gas_velocity_m_s': [0.5, 15.0],
                'liquid_velocity_m_s': [0.01, 0.5],
                'temperature_celsius': [15, 30],
                'pressure_bar': [1.0, 3.0],
                'acoustic_frequency_Hz': [100, 10000]
            },
            
            'data_quality': {
                'synthetic_data': True,
                'physical_basis': 'Based on established correlations and models from literature',
                'uncertainty': 'Random variations added to simulate measurement uncertainty (~10%)',
                'validation': 'Recommended to validate against experimental data when available'
            },
            
            'usage_recommendations': {
                'suitable_for': [
                    'Algorithm development and testing',
                    'Preliminary analysis and hypothesis testing',
                    'Model training for machine learning applications',
                    'Educational purposes',
                    'Experimental planning and design'
                ],
                'limitations': [
                    'Synthetic data may not capture all physical complexities',
                    'Should be validated against real experimental data',
                    'Simplified models used for some phenomena'
                ]
            },
            
            'references': [
                'Taitel, Y., & Dukler, A. E. (1976). A model for predicting flow regime transitions',
                'Wood, A. B. (1930). A Textbook of Sound',
                'Brennen, C. E. (2005). Fundamentals of Multiphase Flow',
                'Prosperetti, A. (2015). Linear pressure waves in bubbly liquids'
            ]
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✓ Saved metadata file")


def main():
    """Main function to generate the complete dataset."""
    print("\n" + "=" * 70)
    print("STRATIFIED FLOW ACOUSTIC ATTENUATION DATASET GENERATOR")
    print("=" * 70)
    print("\nPhD Research Topic:")
    print("Study on the Attenuation Mechanisms in Stratified Flows:")
    print("Beyond Single Phase Leakage Acoustics")
    print("=" * 70)
    
    # Initialize generator with 100 experiments
    generator = StratifiedFlowDataGenerator(n_experiments=100)
    
    # Generate and save all datasets
    operating_conditions, turbulence_data, attenuation_data = generator.save_datasets()
    
    # Print some example data
    print("\n" + "=" * 70)
    print("SAMPLE DATA PREVIEW")
    print("=" * 70)
    
    print("\n1. Flow Regime Characterization (first 3 experiments):")
    print(operating_conditions.head(3).to_string())
    
    print("\n2. Acoustic Attenuation Data (sample):")
    sample_atten = attenuation_data[attenuation_data['experiment_id'] == 1]
    print(sample_atten.to_string())
    
    print("\n3. Turbulence Data (first 3 experiments):")
    print(turbulence_data.head(3).to_string())
    
    print("\n" + "=" * 70)
    print("Dataset ready for analysis!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
