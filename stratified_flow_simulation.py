#!/usr/bin/env python3
"""
Stratified Flow Simulation Data Generator
PhD Thesis: Study on the Attenuation Mechanisms in Stratified Flows

This module generates comprehensive simulation data for stratified flows including:
- CFD model outputs (velocity, pressure, VOF, turbulence)
- Mathematical model outputs (sound speed, attenuation)
- Acoustic propagation data
- Model validation data
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import vtk
import pandas as pd
from scipy import interpolate, signal
from scipy.spatial.distance import cdist
import os
from datetime import datetime
import json

class StratifiedFlowSimulator:
    """
    Main class for generating stratified flow simulation data
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the simulator with configuration parameters
        """
        self.config = self._load_config(config_file)
        self.setup_domain()
        
    def _load_config(self, config_file):
        """
        Load configuration parameters for the simulation
        """
        default_config = {
            # Domain parameters
            'domain': {
                'x_length': 10.0,  # meters
                'y_length': 2.0,   # meters
                'z_length': 1.0,   # meters
                'nx': 200,         # grid points in x
                'ny': 40,          # grid points in y
                'nz': 20           # grid points in z
            },
            # Fluid properties
            'fluids': {
                'density_1': 1000.0,    # kg/m³ (water)
                'density_2': 1.2,       # kg/m³ (air)
                'viscosity_1': 1e-3,    # Pa·s
                'viscosity_2': 1.8e-5,  # Pa·s
                'sound_speed_1': 1500.0, # m/s
                'sound_speed_2': 343.0   # m/s
            },
            # Flow conditions
            'flow': {
                'velocity_1': 0.1,      # m/s
                'velocity_2': 5.0,      # m/s
                'interface_height': 0.5, # m (from bottom)
                'turbulence_intensity': 0.05
            },
            # Acoustic parameters
            'acoustic': {
                'frequency_range': [100, 10000],  # Hz
                'source_position': [0.5, 1.0, 0.5],  # m
                'receiver_positions': [[5.0, 1.0, 0.5], [8.0, 1.0, 0.5]],
                'time_duration': 1.0,   # seconds
                'sampling_rate': 44100  # Hz
            },
            # Simulation parameters
            'simulation': {
                'time_step': 1e-4,      # seconds
                'total_time': 1.0,      # seconds
                'turbulence_model': 'k_epsilon',
                'multiphase_model': 'VOF'
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
        
        return default_config
    
    def setup_domain(self):
        """
        Set up the computational domain and grid
        """
        domain = self.config['domain']
        
        # Create coordinate arrays
        self.x = np.linspace(0, domain['x_length'], domain['nx'])
        self.y = np.linspace(0, domain['y_length'], domain['ny'])
        self.z = np.linspace(0, domain['z_length'], domain['nz'])
        
        # Create meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Initialize field arrays
        self.velocity_x = np.zeros_like(self.X)
        self.velocity_y = np.zeros_like(self.X)
        self.velocity_z = np.zeros_like(self.X)
        self.pressure = np.zeros_like(self.X)
        self.vof = np.zeros_like(self.X)
        self.turbulence_k = np.zeros_like(self.X)
        self.turbulence_epsilon = np.zeros_like(self.X)
        self.density = np.zeros_like(self.X)
        self.sound_speed = np.zeros_like(self.X)
        
        print(f"Domain setup complete: {domain['nx']}x{domain['ny']}x{domain['nz']} grid points")
    
    def generate_cfd_data(self):
        """
        Generate CFD simulation data including velocity, pressure, VOF, and turbulence
        """
        print("Generating CFD simulation data...")
        
        flow = self.config['flow']
        fluids = self.config['fluids']
        
        # Set up stratified flow conditions
        interface_y = flow['interface_height']
        
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                for k in range(self.X.shape[2]):
                    y_pos = self.Y[i, j, k]
                    
                    # Determine phase (VOF = 1 for fluid 1, 0 for fluid 2)
                    if y_pos < interface_y:
                        self.vof[i, j, k] = 1.0
                        self.density[i, j, k] = fluids['density_1']
                        self.sound_speed[i, j, k] = fluids['sound_speed_1']
                        self.velocity_x[i, j, k] = flow['velocity_1']
                    else:
                        self.vof[i, j, k] = 0.0
                        self.density[i, j, k] = fluids['density_2']
                        self.sound_speed[i, j, k] = fluids['sound_speed_2']
                        self.velocity_x[i, j, k] = flow['velocity_2']
                    
                    # Add turbulence and vertical velocity component
                    turbulence = flow['turbulence_intensity']
                    self.velocity_y[i, j, k] = np.random.normal(0, turbulence * abs(self.velocity_x[i, j, k]))
                    self.velocity_z[i, j, k] = np.random.normal(0, turbulence * abs(self.velocity_x[i, j, k]))
                    
                    # Pressure field (hydrostatic + dynamic)
                    hydrostatic_pressure = self.density[i, j, k] * 9.81 * (self.Y[i, j, k] - interface_y)
                    dynamic_pressure = 0.5 * self.density[i, j, k] * self.velocity_x[i, j, k]**2
                    self.pressure[i, j, k] = hydrostatic_pressure + dynamic_pressure
                    
                    # Turbulence parameters (k-epsilon model)
                    velocity_magnitude = np.sqrt(self.velocity_x[i, j, k]**2 + 
                                               self.velocity_y[i, j, k]**2 + 
                                               self.velocity_z[i, j, k]**2)
                    self.turbulence_k[i, j, k] = 1.5 * (turbulence * velocity_magnitude)**2
                    self.turbulence_epsilon[i, j, k] = self.turbulence_k[i, j, k]**1.5 / (0.1 * min(self.x[1]-self.x[0], self.y[1]-self.y[0]))
        
        print("CFD data generation complete")
    
    def generate_mathematical_model_data(self):
        """
        Generate mathematical model outputs for sound speed and attenuation
        """
        print("Generating mathematical model data...")
        
        # Calculate effective sound speed using Wood's equation
        self.effective_sound_speed = self._calculate_effective_sound_speed()
        
        # Calculate attenuation coefficients
        self.attenuation_coefficients = self._calculate_attenuation_coefficients()
        
        # Generate wave propagation patterns
        self.wave_patterns = self._generate_wave_propagation_patterns()
        
        # Calculate time delays
        self.time_delays = self._calculate_time_delays()
        
        print("Mathematical model data generation complete")
    
    def _calculate_effective_sound_speed(self):
        """
        Calculate effective sound speed using Wood's equation for stratified media
        """
        # Wood's equation: 1/c_eff² = α₁/c₁² + α₂/c₂²
        # where α is volume fraction and c is sound speed
        
        c1 = self.config['fluids']['sound_speed_1']
        c2 = self.config['fluids']['sound_speed_2']
        
        effective_sound_speed = np.zeros_like(self.X)
        
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                for k in range(self.X.shape[2]):
                    vof_val = self.vof[i, j, k]
                    c_eff_squared = 1.0 / (vof_val/c1**2 + (1-vof_val)/c2**2)
                    effective_sound_speed[i, j, k] = np.sqrt(c_eff_squared)
        
        return effective_sound_speed
    
    def _calculate_attenuation_coefficients(self):
        """
        Calculate attenuation coefficients for different frequencies
        """
        frequencies = np.logspace(2, 4, 50)  # 100 Hz to 10 kHz
        attenuation_data = {}
        
        for freq in frequencies:
            # Simplified attenuation model based on interface scattering
            # α = α_interface + α_turbulence + α_viscous
            
            alpha_interface = 0.1 * freq**0.5  # Interface scattering
            alpha_turbulence = 0.05 * freq**1.2  # Turbulence scattering
            alpha_viscous = 0.001 * freq**2  # Viscous attenuation
            
            total_alpha = alpha_interface + alpha_turbulence + alpha_viscous
            attenuation_data[freq] = total_alpha
        
        return attenuation_data
    
    def _generate_wave_propagation_patterns(self):
        """
        Generate wave propagation patterns including reflection and transmission
        """
        acoustic = self.config['acoustic']
        source_pos = acoustic['source_position']
        
        # Calculate distances from source
        distances = np.sqrt((self.X - source_pos[0])**2 + 
                           (self.Y - source_pos[1])**2 + 
                           (self.Z - source_pos[2])**2)
        
        # Generate wave patterns for different frequencies
        frequencies = [100, 500, 1000, 5000, 10000]  # Hz
        wave_patterns = {}
        
        for freq in frequencies:
            # Calculate wave number
            k = 2 * np.pi * freq / self.effective_sound_speed
            
            # Generate spherical wave with interface effects
            wave_amplitude = np.exp(-1j * k * distances) / (4 * np.pi * distances)
            
            # Add interface reflection/transmission effects
            interface_y = self.config['flow']['interface_height']
            reflection_coefficient = 0.3  # Simplified reflection coefficient
            
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    for k in range(self.X.shape[2]):
                        if self.Y[i, j, k] > interface_y:
                            # Above interface - transmitted wave
                            wave_amplitude[i, j, k] *= (1 - reflection_coefficient)
                        else:
                            # Below interface - incident + reflected wave
                            wave_amplitude[i, j, k] *= (1 + reflection_coefficient)
            
            wave_patterns[freq] = wave_amplitude
        
        return wave_patterns
    
    def _calculate_time_delays(self):
        """
        Calculate time delays for acoustic propagation
        """
        acoustic = self.config['acoustic']
        source_pos = acoustic['source_position']
        receiver_positions = acoustic['receiver_positions']
        
        time_delays = {}
        
        for i, receiver_pos in enumerate(receiver_positions):
            # Calculate direct path time delay
            direct_distance = np.sqrt(sum((np.array(source_pos) - np.array(receiver_pos))**2))
            direct_time = direct_distance / self.config['fluids']['sound_speed_1']
            
            # Calculate stratified path time delay (simplified)
            interface_y = self.config['flow']['interface_height']
            if receiver_pos[1] > interface_y:
                # Path through both media
                path1_distance = np.sqrt((source_pos[0] - 0)**2 + (source_pos[1] - interface_y)**2)
                path2_distance = np.sqrt((0 - receiver_pos[0])**2 + (interface_y - receiver_pos[1])**2)
                
                stratified_time = (path1_distance / self.config['fluids']['sound_speed_1'] + 
                                 path2_distance / self.config['fluids']['sound_speed_2'])
            else:
                stratified_time = direct_time
            
            time_delays[f'receiver_{i+1}'] = {
                'direct_time': direct_time,
                'stratified_time': stratified_time,
                'time_difference': stratified_time - direct_time
            }
        
        return time_delays
    
    def generate_acoustic_propagation_data(self):
        """
        Generate time-series acoustic pressure propagation data
        """
        print("Generating acoustic propagation data...")
        
        acoustic = self.config['acoustic']
        source_pos = acoustic['source_position']
        receiver_positions = acoustic['receiver_positions']
        duration = acoustic['time_duration']
        sampling_rate = acoustic['sampling_rate']
        
        time = np.linspace(0, duration, int(duration * sampling_rate))
        acoustic_data = {}
        
        for i, receiver_pos in enumerate(receiver_positions):
            # Generate acoustic signal at receiver
            signal_data = np.zeros_like(time)
            
            # Add multiple frequency components
            frequencies = [100, 500, 1000, 2000, 5000]
            amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2]
            
            for freq, amp in zip(frequencies, amplitudes):
                # Calculate time delay
                distance = np.sqrt(sum((np.array(source_pos) - np.array(receiver_pos))**2))
                time_delay = distance / self.config['fluids']['sound_speed_1']
                
                # Generate sinusoidal signal with attenuation
                signal = amp * np.sin(2 * np.pi * freq * (time - time_delay))
                signal *= np.exp(-0.1 * freq * time)  # Frequency-dependent attenuation
                
                signal_data += signal
            
            # Add noise
            noise = 0.05 * np.random.normal(0, 1, len(time))
            signal_data += noise
            
            acoustic_data[f'receiver_{i+1}'] = {
                'time': time,
                'pressure': signal_data,
                'position': receiver_pos
            }
        
        self.acoustic_data = acoustic_data
        print("Acoustic propagation data generation complete")
    
    def generate_validation_data(self):
        """
        Generate model validation data comparing simulated vs experimental results
        """
        print("Generating validation data...")
        
        # Create synthetic "experimental" data with added noise and uncertainties
        validation_data = {}
        
        # Sound speed validation
        experimental_sound_speeds = {
            'frequency': [100, 500, 1000, 2000, 5000, 10000],
            'measured': [1480, 1475, 1470, 1465, 1460, 1455],  # Synthetic experimental data
            'uncertainty': [5, 5, 5, 5, 5, 5]  # ±5 m/s uncertainty
        }
        
        # Calculate simulated sound speeds
        simulated_sound_speeds = []
        for freq in experimental_sound_speeds['frequency']:
            # Use effective sound speed at interface
            interface_idx = int(self.config['flow']['interface_height'] / self.y[1])
            sim_speed = np.mean(self.effective_sound_speed[:, interface_idx, :])
            simulated_sound_speeds.append(sim_speed)
        
        validation_data['sound_speed'] = {
            'experimental': experimental_sound_speeds,
            'simulated': simulated_sound_speeds,
            'frequency': experimental_sound_speeds['frequency']
        }
        
        # Attenuation validation
        experimental_attenuation = {
            'frequency': [100, 500, 1000, 2000, 5000, 10000],
            'measured': [0.1, 0.2, 0.3, 0.5, 0.8, 1.2],  # dB/m
            'uncertainty': [0.02, 0.02, 0.03, 0.05, 0.08, 0.1]
        }
        
        simulated_attenuation = []
        for freq in experimental_attenuation['frequency']:
            if freq in self.attenuation_coefficients:
                sim_att = self.attenuation_coefficients[freq] * 8.686  # Convert to dB/m
                simulated_attenuation.append(sim_att)
            else:
                simulated_attenuation.append(0.0)
        
        validation_data['attenuation'] = {
            'experimental': experimental_attenuation,
            'simulated': simulated_attenuation,
            'frequency': experimental_attenuation['frequency']
        }
        
        # Waveform validation
        validation_data['waveforms'] = {}
        for receiver in self.acoustic_data:
            exp_waveform = self.acoustic_data[receiver]['pressure'].copy()
            # Add experimental noise and filtering
            exp_waveform += 0.1 * np.random.normal(0, 1, len(exp_waveform))
            exp_waveform = signal.savgol_filter(exp_waveform, 21, 3)  # Smoothing filter
            
            validation_data['waveforms'][receiver] = {
                'experimental': exp_waveform,
                'simulated': self.acoustic_data[receiver]['pressure'],
                'time': self.acoustic_data[receiver]['time']
            }
        
        self.validation_data = validation_data
        print("Validation data generation complete")
    
    def export_data(self, output_dir='stratified_flow_data'):
        """
        Export all generated data in various formats
        """
        print(f"Exporting data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export CFD data as HDF5
        self._export_cfd_hdf5(output_dir)
        
        # Export VTK files for visualization
        self._export_vtk_files(output_dir)
        
        # Export CSV files for analysis
        self._export_csv_files(output_dir)
        
        # Export JSON metadata
        self._export_metadata(output_dir)
        
        print(f"Data export complete. Files saved to {output_dir}/")
    
    def _export_cfd_hdf5(self, output_dir):
        """
        Export CFD data as HDF5 files
        """
        with h5py.File(f'{output_dir}/cfd_data.h5', 'w') as f:
            # Create groups
            geometry = f.create_group('geometry')
            fields = f.create_group('fields')
            turbulence = f.create_group('turbulence')
            
            # Export coordinates
            geometry.create_dataset('x', data=self.x)
            geometry.create_dataset('y', data=self.y)
            geometry.create_dataset('z', data=self.z)
            
            # Export field data
            fields.create_dataset('velocity_x', data=self.velocity_x)
            fields.create_dataset('velocity_y', data=self.velocity_y)
            fields.create_dataset('velocity_z', data=self.velocity_z)
            fields.create_dataset('pressure', data=self.pressure)
            fields.create_dataset('vof', data=self.vof)
            fields.create_dataset('density', data=self.density)
            fields.create_dataset('sound_speed', data=self.sound_speed)
            fields.create_dataset('effective_sound_speed', data=self.effective_sound_speed)
            
            # Export turbulence data
            turbulence.create_dataset('k', data=self.turbulence_k)
            turbulence.create_dataset('epsilon', data=self.turbulence_epsilon)
    
    def _export_vtk_files(self, output_dir):
        """
        Export data as VTK files for visualization
        """
        try:
            import pyvista as pv
            
            # Create structured grid
            grid = pv.StructuredGrid()
            grid.dimensions = self.X.shape
            grid.points = np.column_stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()])
            
            # Add field data
            grid['velocity_x'] = self.velocity_x.ravel()
            grid['velocity_y'] = self.velocity_y.ravel()
            grid['velocity_z'] = self.velocity_z.ravel()
            grid['pressure'] = self.pressure.ravel()
            grid['vof'] = self.vof.ravel()
            grid['density'] = self.density.ravel()
            grid['sound_speed'] = self.sound_speed.ravel()
            grid['effective_sound_speed'] = self.effective_sound_speed.ravel()
            grid['turbulence_k'] = self.turbulence_k.ravel()
            grid['turbulence_epsilon'] = self.turbulence_epsilon.ravel()
            
            # Save VTK file
            grid.save(f'{output_dir}/stratified_flow_fields.vtk')
            
        except ImportError:
            print("PyVista not available. Skipping VTK export.")
    
    def _export_csv_files(self, output_dir):
        """
        Export data as CSV files for analysis
        """
        # Export attenuation data
        attenuation_df = pd.DataFrame({
            'frequency': list(self.attenuation_coefficients.keys()),
            'attenuation_coefficient': list(self.attenuation_coefficients.values())
        })
        attenuation_df.to_csv(f'{output_dir}/attenuation_coefficients.csv', index=False)
        
        # Export time delay data
        time_delay_data = []
        for receiver, data in self.time_delays.items():
            time_delay_data.append({
                'receiver': receiver,
                'direct_time': data['direct_time'],
                'stratified_time': data['stratified_time'],
                'time_difference': data['time_difference']
            })
        time_delay_df = pd.DataFrame(time_delay_data)
        time_delay_df.to_csv(f'{output_dir}/time_delays.csv', index=False)
        
        # Export validation data
        sound_speed_data = self.validation_data['sound_speed']
        validation_df = pd.DataFrame({
            'frequency': sound_speed_data['frequency'],
            'experimental': sound_speed_data['experimental']['measured'],
            'simulated': sound_speed_data['simulated'],
            'experimental_uncertainty': sound_speed_data['experimental']['uncertainty']
        })
        validation_df.to_csv(f'{output_dir}/sound_speed_validation.csv', index=False)
        
        attenuation_data = self.validation_data['attenuation']
        attenuation_validation_df = pd.DataFrame({
            'frequency': attenuation_data['frequency'],
            'experimental': attenuation_data['experimental']['measured'],
            'simulated': attenuation_data['simulated'],
            'experimental_uncertainty': attenuation_data['experimental']['uncertainty']
        })
        attenuation_validation_df.to_csv(f'{output_dir}/attenuation_validation.csv', index=False)
    
    def _export_metadata(self, output_dir):
        """
        Export simulation metadata and configuration
        """
        metadata = {
            'simulation_info': {
                'title': 'Stratified Flow Simulation Data',
                'description': 'CFD and mathematical model data for attenuation mechanisms in stratified flows',
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'configuration': self.config,
            'data_summary': {
                'grid_points': self.X.size,
                'domain_size': [self.config['domain']['x_length'], 
                               self.config['domain']['y_length'], 
                               self.config['domain']['z_length']],
                'frequencies_analyzed': len(self.attenuation_coefficients),
                'receivers': len(self.acoustic_data)
            }
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_complete_simulation(self, output_dir='stratified_flow_data'):
        """
        Run the complete simulation and generate all data
        """
        print("Starting complete stratified flow simulation...")
        
        # Generate all data
        self.generate_cfd_data()
        self.generate_mathematical_model_data()
        self.generate_acoustic_propagation_data()
        self.generate_validation_data()
        
        # Export all data
        self.export_data(output_dir)
        
        print("Complete simulation finished successfully!")
        return self

if __name__ == "__main__":
    # Create and run simulation
    simulator = StratifiedFlowSimulator()
    simulator.run_complete_simulation()