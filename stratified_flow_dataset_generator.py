#!/usr/bin/env python3
"""
Stratified Flow Attenuation Dataset Generator
============================================

This script generates a comprehensive dataset for studying attenuation mechanisms 
in stratified flows, going beyond single-phase leakage acoustics.

Author: Research Dataset Generator
Date: 2025-10-12
Topic: Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.special import jv, hankel1
import warnings
warnings.filterwarnings('ignore')

class StratifiedFlowDataGenerator:
    """
    Comprehensive data generator for stratified flow attenuation mechanisms
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with reproducible random seed"""
        np.random.seed(seed)
        self.datasets = {}
        
        # Physical constants
        self.c_water = 1480  # Speed of sound in water (m/s)
        self.c_air = 343     # Speed of sound in air (m/s)
        self.rho_water = 1000  # Density of water (kg/m³)
        self.rho_air = 1.225   # Density of air (kg/m³)
        self.mu_water = 1e-3   # Dynamic viscosity of water (Pa·s)
        self.mu_air = 1.8e-5   # Dynamic viscosity of air (Pa·s)
        
    def generate_frequency_domain_data(self, n_samples=5000):
        """
        Generate frequency-domain attenuation data for various stratified flow configurations
        """
        print("Generating frequency-domain attenuation data...")
        
        # Frequency range from 10 Hz to 100 kHz (typical for acoustic measurements)
        frequencies = np.logspace(1, 5, n_samples)
        
        # Flow configurations
        flow_configs = ['horizontal_stratified', 'inclined_stratified', 'wavy_interface', 
                       'slug_flow', 'annular_flow', 'dispersed_flow']
        
        data = []
        
        for config in flow_configs:
            for i in range(n_samples // len(flow_configs)):
                freq = frequencies[i]
                
                # Generate realistic attenuation mechanisms
                if config == 'horizontal_stratified':
                    # Classical stratified flow with clear interface
                    interface_roughness = np.random.uniform(0.1, 2.0)  # mm
                    gas_fraction = np.random.uniform(0.3, 0.8)
                    liquid_fraction = 1 - gas_fraction
                    
                    # Attenuation due to interface scattering
                    ka = 2 * np.pi * freq * interface_roughness / 1000 / self.c_water
                    interface_attenuation = 20 * np.log10(np.exp(ka**2 / 4))
                    
                    # Viscous attenuation in liquid phase
                    viscous_attenuation_liquid = 2 * np.pi * freq**2 * self.mu_water / (self.rho_water * self.c_water**3)
                    
                    # Viscous attenuation in gas phase
                    viscous_attenuation_gas = 2 * np.pi * freq**2 * self.mu_air / (self.rho_air * self.c_air**3)
                    
                    # Combined attenuation
                    total_attenuation = (interface_attenuation + 
                                       liquid_fraction * viscous_attenuation_liquid * 1e6 +
                                       gas_fraction * viscous_attenuation_gas * 1e6)
                    
                elif config == 'wavy_interface':
                    # Wavy interface with enhanced scattering
                    wave_amplitude = np.random.uniform(0.5, 5.0)  # mm
                    wave_frequency = np.random.uniform(0.1, 10.0)  # Hz
                    gas_fraction = np.random.uniform(0.2, 0.9)
                    
                    # Enhanced scattering due to wavy interface
                    ka_wave = 2 * np.pi * freq * wave_amplitude / 1000 / self.c_water
                    wave_scattering = 30 * np.log10(np.exp(ka_wave**2 / 2))
                    
                    # Mode conversion losses
                    mode_conversion = 5 * np.log10(1 + (freq / 1000)**0.5)
                    
                    total_attenuation = wave_scattering + mode_conversion
                    
                elif config == 'slug_flow':
                    # Intermittent slug flow with high attenuation
                    slug_frequency = np.random.uniform(0.1, 5.0)  # Hz
                    gas_fraction = np.random.uniform(0.6, 0.95)
                    
                    # High attenuation due to multiple interfaces and turbulence
                    turbulent_scattering = 40 * np.log10(1 + (freq / 500)**0.3)
                    bubble_resonance = 15 * np.exp(-((freq - 2000) / 1000)**2)
                    
                    total_attenuation = turbulent_scattering + bubble_resonance
                    
                elif config == 'annular_flow':
                    # Annular flow with liquid film on walls
                    film_thickness = np.random.uniform(0.1, 2.0)  # mm
                    gas_fraction = np.random.uniform(0.8, 0.98)
                    
                    # Guided wave attenuation in thin liquid film
                    film_attenuation = 25 * np.log10(1 + freq / (film_thickness * 1000))
                    
                    # Gas core turbulence
                    gas_turbulence = 10 * np.log10(1 + (freq / 2000)**0.4)
                    
                    total_attenuation = film_attenuation + gas_turbulence
                    
                elif config == 'dispersed_flow':
                    # Dispersed droplets or bubbles
                    particle_size = np.random.uniform(0.01, 1.0)  # mm
                    volume_fraction = np.random.uniform(0.1, 0.6)
                    
                    # Rayleigh scattering for small particles
                    ka_particle = 2 * np.pi * freq * particle_size / 1000 / self.c_water
                    rayleigh_scattering = 20 * np.log10(np.exp(volume_fraction * ka_particle**4 / 3))
                    
                    # Thermal and viscous losses
                    thermal_losses = 8 * np.log10(1 + (freq / 10000)**0.6)
                    
                    total_attenuation = rayleigh_scattering + thermal_losses
                    
                else:  # inclined_stratified
                    # Inclined stratified flow
                    inclination_angle = np.random.uniform(0, 45)  # degrees
                    gas_fraction = np.random.uniform(0.4, 0.9)
                    
                    # Gravity wave effects
                    gravity_wave_freq = np.sqrt(9.81 * 2 * np.pi / 0.1)  # Approximate
                    gravity_effect = 12 * np.exp(-((freq - gravity_wave_freq) / 500)**2)
                    
                    # Inclination-induced mixing
                    mixing_attenuation = 8 * np.log10(1 + np.sin(np.radians(inclination_angle)))
                    
                    total_attenuation = gravity_effect + mixing_attenuation
                
                # Add measurement noise
                noise_level = np.random.normal(0, 0.5)
                total_attenuation += noise_level
                
                # Ensure physical bounds
                total_attenuation = np.clip(total_attenuation, 0.1, 200)
                
                data.append({
                    'frequency_hz': freq,
                    'flow_configuration': config,
                    'attenuation_db_per_m': total_attenuation,
                    'gas_fraction': gas_fraction if 'gas_fraction' in locals() else np.random.uniform(0.1, 0.9),
                    'liquid_fraction': 1 - gas_fraction if 'gas_fraction' in locals() else np.random.uniform(0.1, 0.9),
                    'interface_roughness_mm': interface_roughness if 'interface_roughness' in locals() else np.random.uniform(0.1, 3.0),
                    'reynolds_number_gas': np.random.uniform(1000, 50000),
                    'reynolds_number_liquid': np.random.uniform(100, 10000),
                    'weber_number': np.random.uniform(0.1, 100),
                    'froude_number': np.random.uniform(0.1, 10),
                    'temperature_c': np.random.uniform(10, 80),
                    'pressure_bar': np.random.uniform(1, 50)
                })
        
        self.datasets['frequency_domain'] = pd.DataFrame(data)
        return self.datasets['frequency_domain']
    
    def generate_time_domain_data(self, n_samples=2000, duration=10.0):
        """
        Generate time-domain acoustic signals for different stratified flow patterns
        """
        print("Generating time-domain acoustic signals...")
        
        sampling_rate = 44100  # Hz
        t = np.linspace(0, duration, int(sampling_rate * duration))
        
        data = []
        flow_patterns = ['smooth_stratified', 'wavy_stratified', 'slug_intermittent', 
                        'churn_turbulent', 'annular_dispersed']
        
        for pattern in flow_patterns:
            for sample_id in range(n_samples // len(flow_patterns)):
                
                if pattern == 'smooth_stratified':
                    # Low-frequency background with minimal fluctuations
                    base_freq = np.random.uniform(50, 200)
                    signal_amplitude = np.random.uniform(0.1, 0.5)
                    acoustic_signal = signal_amplitude * np.sin(2 * np.pi * base_freq * t)
                    
                    # Add low-level turbulence noise
                    noise = 0.05 * np.random.normal(0, 1, len(t))
                    acoustic_signal += noise
                    
                elif pattern == 'wavy_stratified':
                    # Modulated signal due to interface waves
                    carrier_freq = np.random.uniform(100, 500)
                    modulation_freq = np.random.uniform(0.5, 5.0)
                    signal_amplitude = np.random.uniform(0.2, 0.8)
                    
                    modulation = 1 + 0.3 * np.sin(2 * np.pi * modulation_freq * t)
                    acoustic_signal = signal_amplitude * modulation * np.sin(2 * np.pi * carrier_freq * t)
                    
                elif pattern == 'slug_intermittent':
                    # Intermittent high-amplitude bursts
                    slug_freq = np.random.uniform(0.1, 2.0)
                    burst_duration = np.random.uniform(0.5, 2.0)
                    
                    acoustic_signal = np.zeros_like(t)
                    slug_times = np.arange(0, duration, 1/slug_freq)
                    
                    for slug_time in slug_times:
                        if slug_time < duration - burst_duration:
                            burst_start = int(slug_time * sampling_rate)
                            burst_end = int((slug_time + burst_duration) * sampling_rate)
                            
                            # High-amplitude burst with multiple frequencies
                            burst_signal = (np.random.uniform(0.5, 2.0) * 
                                          (np.sin(2 * np.pi * 300 * t[burst_start:burst_end]) +
                                           0.5 * np.sin(2 * np.pi * 800 * t[burst_start:burst_end]) +
                                           0.3 * np.sin(2 * np.pi * 1200 * t[burst_start:burst_end])))
                            
                            acoustic_signal[burst_start:burst_end] = burst_signal
                    
                elif pattern == 'churn_turbulent':
                    # Broadband turbulent signal
                    # Generate colored noise (1/f spectrum)
                    freqs = np.fft.fftfreq(len(t), 1/sampling_rate)
                    freqs[0] = 1  # Avoid division by zero
                    
                    # 1/f^beta power spectrum
                    beta = np.random.uniform(0.5, 2.0)
                    power_spectrum = 1 / np.abs(freqs)**beta
                    power_spectrum[0] = 0
                    
                    # Generate random phases
                    phases = np.random.uniform(0, 2*np.pi, len(freqs))
                    
                    # Create complex spectrum
                    spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
                    
                    # Convert to time domain
                    acoustic_signal = np.real(np.fft.ifft(spectrum))
                    acoustic_signal = acoustic_signal / np.std(acoustic_signal) * np.random.uniform(0.3, 1.2)
                    
                else:  # annular_dispersed
                    # High-frequency content with droplet impacts
                    base_signal = 0.3 * np.random.normal(0, 1, len(t))
                    
                    # Add periodic droplet impact signatures
                    impact_freq = np.random.uniform(10, 100)
                    impact_times = np.arange(0, duration, 1/impact_freq)
                    
                    for impact_time in impact_times:
                        if impact_time < duration - 0.01:
                            impact_idx = int(impact_time * sampling_rate)
                            impact_duration = int(0.01 * sampling_rate)  # 10ms impact
                            
                            # Exponentially decaying impact signature
                            impact_t = np.linspace(0, 0.01, impact_duration)
                            impact_signal = (np.random.uniform(0.5, 1.5) * 
                                           np.exp(-impact_t * 200) * 
                                           np.sin(2 * np.pi * 2000 * impact_t))
                            
                            end_idx = min(impact_idx + impact_duration, len(base_signal))
                            base_signal[impact_idx:end_idx] += impact_signal[:end_idx-impact_idx]
                    
                    acoustic_signal = base_signal
                
                # Calculate signal statistics
                rms_amplitude = np.sqrt(np.mean(acoustic_signal**2))
                peak_amplitude = np.max(np.abs(acoustic_signal))
                zero_crossings = len(np.where(np.diff(np.sign(acoustic_signal)))[0])
                
                # Frequency domain analysis
                fft_signal = np.fft.fft(acoustic_signal)
                power_spectrum = np.abs(fft_signal)**2
                freqs = np.fft.fftfreq(len(acoustic_signal), 1/sampling_rate)
                
                # Find dominant frequency
                positive_freqs = freqs[:len(freqs)//2]
                positive_power = power_spectrum[:len(power_spectrum)//2]
                dominant_freq = positive_freqs[np.argmax(positive_power)]
                
                # Calculate spectral centroid
                spectral_centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
                
                data.append({
                    'sample_id': f"{pattern}_{sample_id:04d}",
                    'flow_pattern': pattern,
                    'duration_s': duration,
                    'sampling_rate_hz': sampling_rate,
                    'rms_amplitude': rms_amplitude,
                    'peak_amplitude': peak_amplitude,
                    'zero_crossings_per_sec': zero_crossings / duration,
                    'dominant_frequency_hz': dominant_freq,
                    'spectral_centroid_hz': spectral_centroid,
                    'signal_data': acoustic_signal.tolist()  # Store as list for JSON compatibility
                })
        
        self.datasets['time_domain'] = pd.DataFrame(data)
        return self.datasets['time_domain']
    
    def generate_experimental_conditions(self, n_experiments=1000):
        """
        Generate realistic experimental conditions and measurement parameters
        """
        print("Generating experimental conditions dataset...")
        
        data = []
        
        for exp_id in range(n_experiments):
            # Pipe geometry
            pipe_diameter = np.random.choice([0.025, 0.05, 0.1, 0.15, 0.2, 0.3])  # m
            pipe_length = np.random.uniform(1.0, 10.0)  # m
            pipe_roughness = np.random.uniform(1e-6, 1e-4)  # m
            inclination_angle = np.random.uniform(-10, 45)  # degrees
            
            # Flow conditions
            gas_superficial_velocity = np.random.uniform(0.1, 20.0)  # m/s
            liquid_superficial_velocity = np.random.uniform(0.01, 5.0)  # m/s
            
            # Fluid properties
            temperature = np.random.uniform(15, 80)  # °C
            pressure = np.random.uniform(1, 30)  # bar
            
            # Gas properties (air/nitrogen)
            gas_density = pressure * 100000 / (287 * (temperature + 273.15))  # kg/m³
            gas_viscosity = 1.8e-5 * ((temperature + 273.15) / 293.15)**0.7  # Pa·s
            
            # Liquid properties (water with additives)
            liquid_density = 1000 - 0.3 * (temperature - 20)  # kg/m³
            liquid_viscosity = 1e-3 * np.exp(-0.02 * (temperature - 20))  # Pa·s
            surface_tension = 0.072 - 0.0015 * (temperature - 20)  # N/m
            
            # Calculate dimensionless numbers
            reynolds_gas = gas_density * gas_superficial_velocity * pipe_diameter / gas_viscosity
            reynolds_liquid = liquid_density * liquid_superficial_velocity * pipe_diameter / liquid_viscosity
            
            weber_gas = gas_density * gas_superficial_velocity**2 * pipe_diameter / surface_tension
            froude_gas = gas_superficial_velocity / np.sqrt(9.81 * pipe_diameter)
            
            # Flow pattern prediction (simplified Mandhane map)
            x_coord = gas_superficial_velocity
            y_coord = liquid_superficial_velocity
            
            if y_coord < 0.1 and x_coord < 3:
                flow_pattern = 'stratified'
            elif y_coord < 0.3 and x_coord > 3:
                flow_pattern = 'wavy'
            elif y_coord > 0.3 and x_coord < 5:
                flow_pattern = 'slug'
            elif y_coord > 1.0:
                flow_pattern = 'dispersed'
            else:
                flow_pattern = 'annular'
            
            # Acoustic measurement setup
            transducer_frequency = np.random.choice([0.5, 1.0, 2.25, 5.0, 10.0]) * 1e6  # Hz
            measurement_distance = np.random.uniform(0.1, 2.0)  # m
            measurement_angle = np.random.uniform(0, 90)  # degrees
            
            # Environmental conditions
            ambient_temperature = np.random.uniform(18, 35)  # °C
            ambient_pressure = np.random.uniform(0.95, 1.05)  # bar
            humidity = np.random.uniform(30, 80)  # %
            
            data.append({
                'experiment_id': f"EXP_{exp_id:04d}",
                'pipe_diameter_m': pipe_diameter,
                'pipe_length_m': pipe_length,
                'pipe_roughness_m': pipe_roughness,
                'inclination_angle_deg': inclination_angle,
                'gas_superficial_velocity_ms': gas_superficial_velocity,
                'liquid_superficial_velocity_ms': liquid_superficial_velocity,
                'temperature_c': temperature,
                'pressure_bar': pressure,
                'gas_density_kgm3': gas_density,
                'gas_viscosity_pas': gas_viscosity,
                'liquid_density_kgm3': liquid_density,
                'liquid_viscosity_pas': liquid_viscosity,
                'surface_tension_nm': surface_tension,
                'reynolds_gas': reynolds_gas,
                'reynolds_liquid': reynolds_liquid,
                'weber_gas': weber_gas,
                'froude_gas': froude_gas,
                'predicted_flow_pattern': flow_pattern,
                'transducer_frequency_hz': transducer_frequency,
                'measurement_distance_m': measurement_distance,
                'measurement_angle_deg': measurement_angle,
                'ambient_temperature_c': ambient_temperature,
                'ambient_pressure_bar': ambient_pressure,
                'humidity_percent': humidity
            })
        
        self.datasets['experimental_conditions'] = pd.DataFrame(data)
        return self.datasets['experimental_conditions']
    
    def generate_attenuation_models(self, n_samples=3000):
        """
        Generate data based on various theoretical attenuation models
        """
        print("Generating theoretical attenuation models dataset...")
        
        data = []
        models = ['rayleigh_scattering', 'mie_scattering', 'viscous_losses', 
                 'thermal_losses', 'interface_scattering', 'mode_conversion']
        
        for model in models:
            for i in range(n_samples // len(models)):
                # Common parameters
                frequency = np.random.uniform(100, 100000)  # Hz
                temperature = np.random.uniform(10, 80)  # °C
                pressure = np.random.uniform(1, 20)  # bar
                
                if model == 'rayleigh_scattering':
                    # Small particle scattering (ka << 1)
                    particle_radius = np.random.uniform(1e-6, 1e-4)  # m
                    particle_concentration = np.random.uniform(1e6, 1e12)  # particles/m³
                    
                    # Rayleigh scattering coefficient
                    k = 2 * np.pi * frequency / self.c_water
                    ka = k * particle_radius
                    
                    scattering_cross_section = (8 * np.pi / 3) * (ka)**4 * particle_radius**2
                    attenuation = particle_concentration * scattering_cross_section * 8.686  # dB/m
                    
                elif model == 'mie_scattering':
                    # Large particle scattering (ka >= 1)
                    particle_radius = np.random.uniform(1e-4, 1e-2)  # m
                    particle_concentration = np.random.uniform(1e3, 1e9)  # particles/m³
                    
                    k = 2 * np.pi * frequency / self.c_water
                    ka = k * particle_radius
                    
                    # Simplified Mie scattering (approximation)
                    if ka < 1:
                        scattering_efficiency = (8/3) * ka**4
                    else:
                        scattering_efficiency = 2 * (1 - np.cos(ka))
                    
                    scattering_cross_section = scattering_efficiency * np.pi * particle_radius**2
                    attenuation = particle_concentration * scattering_cross_section * 8.686  # dB/m
                    
                elif model == 'viscous_losses':
                    # Viscous attenuation in fluid
                    fluid_type = np.random.choice(['water', 'oil', 'gas'])
                    
                    if fluid_type == 'water':
                        density = 1000
                        viscosity = 1e-3
                        sound_speed = 1480
                    elif fluid_type == 'oil':
                        density = 850
                        viscosity = np.random.uniform(1e-3, 1e-1)
                        sound_speed = 1200
                    else:  # gas
                        density = pressure * 100000 / (287 * (temperature + 273.15))
                        viscosity = 1.8e-5
                        sound_speed = 343
                    
                    # Classical viscous attenuation
                    attenuation = (2 * np.pi * frequency**2 * viscosity) / (density * sound_speed**3) * 8.686e6
                    
                elif model == 'thermal_losses':
                    # Thermal conduction losses
                    thermal_conductivity = np.random.uniform(0.1, 2.0)  # W/m·K
                    specific_heat = np.random.uniform(1000, 4000)  # J/kg·K
                    density = np.random.uniform(500, 2000)  # kg/m³
                    
                    thermal_diffusivity = thermal_conductivity / (density * specific_heat)
                    
                    # Thermal boundary layer thickness
                    delta_th = np.sqrt(2 * thermal_diffusivity / (2 * np.pi * frequency))
                    
                    # Thermal attenuation coefficient
                    attenuation = (np.pi * frequency) / (self.c_water * delta_th) * 8.686
                    
                elif model == 'interface_scattering':
                    # Scattering from rough interfaces
                    interface_roughness = np.random.uniform(1e-6, 1e-3)  # m
                    correlation_length = np.random.uniform(1e-4, 1e-2)  # m
                    
                    k = 2 * np.pi * frequency / self.c_water
                    
                    # Perturbation theory for rough surface scattering
                    roughness_parameter = k * interface_roughness
                    
                    if roughness_parameter < 1:
                        # Rayleigh roughness regime
                        attenuation = 4 * k * roughness_parameter**2 * 8.686
                    else:
                        # Kirchhoff approximation
                        attenuation = 2 * k * roughness_parameter * 8.686
                    
                else:  # mode_conversion
                    # Mode conversion at interfaces
                    impedance_contrast = np.random.uniform(0.1, 10.0)
                    interface_thickness = np.random.uniform(1e-5, 1e-3)  # m
                    
                    k = 2 * np.pi * frequency / self.c_water
                    
                    # Mode conversion efficiency
                    reflection_coeff = (impedance_contrast - 1) / (impedance_contrast + 1)
                    transmission_coeff = 2 / (impedance_contrast + 1)
                    
                    # Attenuation due to mode conversion
                    attenuation = -20 * np.log10(abs(transmission_coeff))
                
                # Add realistic bounds and noise
                attenuation = np.clip(attenuation, 0.001, 1000)
                attenuation *= np.random.uniform(0.8, 1.2)  # ±20% uncertainty
                
                data.append({
                    'model_type': model,
                    'frequency_hz': frequency,
                    'temperature_c': temperature,
                    'pressure_bar': pressure,
                    'predicted_attenuation_db_per_m': attenuation,
                    'particle_radius_m': particle_radius if 'particle_radius' in locals() else np.nan,
                    'particle_concentration_per_m3': particle_concentration if 'particle_concentration' in locals() else np.nan,
                    'interface_roughness_m': interface_roughness if 'interface_roughness' in locals() else np.nan,
                    'viscosity_pas': viscosity if 'viscosity' in locals() else np.nan,
                    'density_kgm3': density if 'density' in locals() else np.nan,
                    'sound_speed_ms': sound_speed if 'sound_speed' in locals() else np.nan
                })
        
        self.datasets['attenuation_models'] = pd.DataFrame(data)
        return self.datasets['attenuation_models']
    
    def generate_multiphase_flow_data(self, n_samples=2500):
        """
        Generate comprehensive multiphase flow characterization data
        """
        print("Generating multiphase flow characterization data...")
        
        data = []
        
        for i in range(n_samples):
            # Flow regime classification
            flow_regime = np.random.choice(['bubble', 'slug', 'churn', 'annular', 'stratified', 'wavy'])
            
            # Phase fractions
            if flow_regime == 'bubble':
                gas_fraction = np.random.uniform(0.01, 0.3)
            elif flow_regime == 'slug':
                gas_fraction = np.random.uniform(0.2, 0.8)
            elif flow_regime == 'churn':
                gas_fraction = np.random.uniform(0.6, 0.9)
            elif flow_regime == 'annular':
                gas_fraction = np.random.uniform(0.8, 0.99)
            else:  # stratified or wavy
                gas_fraction = np.random.uniform(0.3, 0.9)
            
            liquid_fraction = 1 - gas_fraction
            
            # Bubble/droplet characteristics
            if gas_fraction < 0.5:  # Bubble flow
                bubble_diameter = np.random.uniform(0.001, 0.01)  # m
                bubble_velocity = np.random.uniform(0.1, 2.0)  # m/s
                bubble_frequency = np.random.uniform(1, 100)  # Hz
                
                droplet_diameter = np.nan
                droplet_velocity = np.nan
            else:  # Droplet flow
                droplet_diameter = np.random.uniform(0.0001, 0.005)  # m
                droplet_velocity = np.random.uniform(0.5, 10.0)  # m/s
                
                bubble_diameter = np.nan
                bubble_velocity = np.nan
                bubble_frequency = np.nan
            
            # Interface characteristics
            interface_area_density = np.random.uniform(10, 10000)  # m²/m³
            interface_velocity = np.random.uniform(0.1, 5.0)  # m/s
            
            # Turbulence parameters
            turbulent_kinetic_energy = np.random.uniform(0.001, 1.0)  # m²/s²
            turbulent_dissipation_rate = np.random.uniform(0.01, 100)  # m²/s³
            
            # Acoustic properties
            mixture_density = gas_fraction * self.rho_air + liquid_fraction * self.rho_water
            
            # Wood's equation for mixture sound speed
            mixture_compressibility = (gas_fraction / (self.rho_air * self.c_air**2) + 
                                     liquid_fraction / (self.rho_water * self.c_water**2))
            mixture_sound_speed = 1 / np.sqrt(mixture_density * mixture_compressibility)
            
            # Attenuation mechanisms
            frequency_test = 1000  # Hz for attenuation calculation
            
            # Bubble/droplet scattering
            if not np.isnan(bubble_diameter):
                bubble_resonance_freq = 1 / (2 * np.pi * bubble_diameter) * np.sqrt(3 * 1.4 * 101325 / self.rho_water)
                scattering_attenuation = 20 * np.exp(-((frequency_test - bubble_resonance_freq) / 500)**2)
            else:
                scattering_attenuation = 5 * (frequency_test / 1000)**0.5
            
            # Viscous losses
            mixture_viscosity = gas_fraction * self.mu_air + liquid_fraction * self.mu_water
            viscous_attenuation = (2 * np.pi * frequency_test**2 * mixture_viscosity) / (mixture_density * mixture_sound_speed**3) * 1e6
            
            # Interface losses
            interface_attenuation = interface_area_density * 0.001 * (frequency_test / 1000)**0.3
            
            total_attenuation = scattering_attenuation + viscous_attenuation + interface_attenuation
            
            data.append({
                'sample_id': f"MF_{i:04d}",
                'flow_regime': flow_regime,
                'gas_fraction': gas_fraction,
                'liquid_fraction': liquid_fraction,
                'bubble_diameter_m': bubble_diameter,
                'bubble_velocity_ms': bubble_velocity,
                'bubble_frequency_hz': bubble_frequency,
                'droplet_diameter_m': droplet_diameter,
                'droplet_velocity_ms': droplet_velocity,
                'interface_area_density_m2m3': interface_area_density,
                'interface_velocity_ms': interface_velocity,
                'turbulent_kinetic_energy_m2s2': turbulent_kinetic_energy,
                'turbulent_dissipation_rate_m2s3': turbulent_dissipation_rate,
                'mixture_density_kgm3': mixture_density,
                'mixture_sound_speed_ms': mixture_sound_speed,
                'total_attenuation_db_per_m_at_1khz': total_attenuation,
                'scattering_component_db_per_m': scattering_attenuation,
                'viscous_component_db_per_m': viscous_attenuation,
                'interface_component_db_per_m': interface_attenuation
            })
        
        self.datasets['multiphase_flow'] = pd.DataFrame(data)
        return self.datasets['multiphase_flow']
    
    def save_datasets(self, output_dir='stratified_flow_datasets'):
        """
        Save all generated datasets to files
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\nSaving datasets to {output_dir}/...")
        
        for dataset_name, dataset in self.datasets.items():
            # Save as CSV
            csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
            dataset.to_csv(csv_path, index=False)
            print(f"Saved {dataset_name} dataset: {len(dataset)} samples -> {csv_path}")
            
            # Save as pickle for Python compatibility
            pickle_path = os.path.join(output_dir, f"{dataset_name}.pkl")
            dataset.to_pickle(pickle_path)
        
        # Create dataset summary
        summary_path = os.path.join(output_dir, "dataset_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Stratified Flow Attenuation Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write("Generated datasets for PhD research topic:\n")
            f.write("'Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics'\n\n")
            
            for dataset_name, dataset in self.datasets.items():
                f.write(f"{dataset_name.upper()} DATASET:\n")
                f.write(f"  - Samples: {len(dataset)}\n")
                f.write(f"  - Features: {len(dataset.columns)}\n")
                f.write(f"  - Columns: {', '.join(dataset.columns)}\n\n")
        
        print(f"Dataset summary saved to {summary_path}")
        
        return output_dir

def main():
    """
    Main function to generate all datasets
    """
    print("Stratified Flow Attenuation Dataset Generator")
    print("=" * 50)
    print("Generating comprehensive datasets for PhD research...")
    print()
    
    # Initialize generator
    generator = StratifiedFlowDataGenerator(seed=42)
    
    # Generate all datasets
    generator.generate_frequency_domain_data(n_samples=5000)
    generator.generate_time_domain_data(n_samples=2000, duration=10.0)
    generator.generate_experimental_conditions(n_experiments=1000)
    generator.generate_attenuation_models(n_samples=3000)
    generator.generate_multiphase_flow_data(n_samples=2500)
    
    # Save datasets
    output_dir = generator.save_datasets()
    
    print(f"\n✅ Dataset generation completed!")
    print(f"📁 All datasets saved to: {output_dir}")
    print(f"📊 Total samples generated: {sum(len(df) for df in generator.datasets.values())}")
    
    return generator

if __name__ == "__main__":
    generator = main()