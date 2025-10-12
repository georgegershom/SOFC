#!/usr/bin/env python3
"""
Experimental Validation Data Generator
Generates synthetic experimental data for model validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import json

class ExperimentalDataGenerator:
    """
    Generates synthetic experimental data for validation purposes
    """
    
    def __init__(self, config):
        """
        Initialize with configuration parameters
        """
        self.config = config
        self.fluid_props = config['fluids']
        
    def generate_sound_speed_measurements(self, frequencies, volume_fractions):
        """
        Generate synthetic sound speed measurements with experimental uncertainties
        
        Parameters:
        -----------
        frequencies : array
            Frequency array
        volume_fractions : array
            Volume fraction array
            
        Returns:
        --------
        experimental_data : dict
            Experimental sound speed data with uncertainties
        """
        # Base sound speed using Wood's equation
        c1 = self.fluid_props['sound_speed_1']
        c2 = self.fluid_props['sound_speed_2']
        
        experimental_data = {
            'frequencies': frequencies,
            'volume_fractions': volume_fractions,
            'sound_speeds': [],
            'uncertainties': [],
            'measurement_conditions': []
        }
        
        for i, freq in enumerate(frequencies):
            freq_data = {
                'frequency': freq,
                'measurements': []
            }
            
            for j, vf in enumerate(volume_fractions):
                # Theoretical sound speed
                c_theory = 1.0 / np.sqrt(vf/c1**2 + (1-vf)/c2**2)
                
                # Add experimental uncertainties
                # Temperature effect (±2°C)
                temp_uncertainty = 0.002 * c_theory
                
                # Measurement uncertainty (±0.5%)
                measurement_uncertainty = 0.005 * c_theory
                
                # Interface effects (frequency dependent)
                interface_uncertainty = 0.001 * c_theory * (freq / 1000)**0.5
                
                # Total uncertainty
                total_uncertainty = np.sqrt(temp_uncertainty**2 + 
                                          measurement_uncertainty**2 + 
                                          interface_uncertainty**2)
                
                # Add random noise
                noise = np.random.normal(0, total_uncertainty)
                c_measured = c_theory + noise
                
                # Ensure physical bounds
                c_measured = np.clip(c_measured, c2, c1)
                
                measurement = {
                    'volume_fraction': vf,
                    'sound_speed': c_measured,
                    'uncertainty': total_uncertainty,
                    'temperature': 20.0 + np.random.normal(0, 1.0),  # °C
                    'pressure': 101325 + np.random.normal(0, 1000)   # Pa
                }
                
                freq_data['measurements'].append(measurement)
            
            experimental_data['sound_speeds'].append(freq_data)
        
        return experimental_data
    
    def generate_attenuation_measurements(self, frequencies, volume_fractions):
        """
        Generate synthetic attenuation measurements
        
        Parameters:
        -----------
        frequencies : array
            Frequency array
        volume_fractions : array
            Volume fraction array
            
        Returns:
        --------
        experimental_data : dict
            Experimental attenuation data
        """
        experimental_data = {
            'frequencies': frequencies,
            'volume_fractions': volume_fractions,
            'attenuation_data': []
        }
        
        for freq in frequencies:
            freq_data = {
                'frequency': freq,
                'measurements': []
            }
            
            for vf in volume_fractions:
                # Theoretical attenuation (simplified model)
                alpha_theory = 0.1 * (freq / 1000)**1.5 * (1 - vf)**2
                
                # Add experimental uncertainties
                # Measurement noise
                measurement_noise = 0.1 * alpha_theory
                
                # Frequency response uncertainty
                freq_uncertainty = 0.05 * alpha_theory * (freq / 1000)**0.5
                
                # Interface roughness effect
                roughness_uncertainty = 0.02 * alpha_theory * np.random.uniform(0.5, 1.5)
                
                # Total uncertainty
                total_uncertainty = np.sqrt(measurement_noise**2 + 
                                          freq_uncertainty**2 + 
                                          roughness_uncertainty**2)
                
                # Add random noise
                noise = np.random.normal(0, total_uncertainty)
                alpha_measured = alpha_theory + noise
                
                # Ensure positive values
                alpha_measured = max(0, alpha_measured)
                
                measurement = {
                    'volume_fraction': vf,
                    'attenuation': alpha_measured,
                    'uncertainty': total_uncertainty,
                    'measurement_method': 'pulse-echo' if freq < 1000 else 'transmission'
                }
                
                freq_data['measurements'].append(measurement)
            
            experimental_data['attenuation_data'].append(freq_data)
        
        return experimental_data
    
    def generate_acoustic_waveform_measurements(self, source_position, receiver_positions, 
                                              duration=1.0, sampling_rate=44100):
        """
        Generate synthetic acoustic waveform measurements
        
        Parameters:
        -----------
        source_position : array
            Position of acoustic source
        receiver_positions : list
            List of receiver positions
        duration : float
            Measurement duration in seconds
        sampling_rate : int
            Sampling rate in Hz
            
        Returns:
        --------
        experimental_data : dict
            Experimental waveform data
        """
        time = np.linspace(0, duration, int(duration * sampling_rate))
        
        experimental_data = {
            'time': time,
            'source_position': source_position,
            'receiver_data': []
        }
        
        for i, receiver_pos in enumerate(receiver_positions):
            # Calculate distance
            distance = np.linalg.norm(np.array(receiver_pos) - np.array(source_position))
            
            # Generate synthetic waveform
            waveform = self._generate_synthetic_waveform(time, distance, receiver_pos)
            
            # Add experimental noise and artifacts
            waveform_with_noise = self._add_experimental_artifacts(waveform, time)
            
            receiver_data = {
                'receiver_id': f'receiver_{i+1}',
                'position': receiver_pos,
                'distance': distance,
                'waveform': waveform_with_noise,
                'sampling_rate': sampling_rate,
                'measurement_conditions': {
                    'temperature': 20.0 + np.random.normal(0, 0.5),
                    'humidity': 50.0 + np.random.normal(0, 5.0),
                    'background_noise': np.random.uniform(30, 50)  # dB
                }
            }
            
            experimental_data['receiver_data'].append(receiver_data)
        
        return experimental_data
    
    def _generate_synthetic_waveform(self, time, distance, receiver_pos):
        """
        Generate synthetic acoustic waveform
        """
        # Multiple frequency components
        frequencies = [100, 500, 1000, 2000, 5000]
        amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        waveform = np.zeros_like(time)
        
        for freq, amp in zip(frequencies, amplitudes):
            # Time delay
            c = self.fluid_props['sound_speed_1']
            time_delay = distance / c
            
            # Attenuation
            attenuation = 0.1 * freq * time_delay / 1000
            
            # Generate signal
            signal = amp * np.sin(2 * np.pi * freq * (time - time_delay))
            signal *= np.exp(-attenuation)
            
            waveform += signal
        
        return waveform
    
    def _add_experimental_artifacts(self, waveform, time):
        """
        Add experimental artifacts and noise
        """
        # Add white noise
        noise_level = 0.05 * np.max(np.abs(waveform))
        white_noise = noise_level * np.random.normal(0, 1, len(waveform))
        
        # Add low-frequency drift
        drift = 0.01 * np.sin(2 * np.pi * 0.1 * time)
        
        # Add high-frequency artifacts
        artifacts = 0.02 * np.sin(2 * np.pi * 10000 * time) * np.exp(-time)
        
        # Add measurement system response (low-pass filter)
        b, a = signal.butter(4, 0.8, 'low')
        filtered_waveform = signal.filtfilt(b, a, waveform)
        
        # Combine all effects
        noisy_waveform = filtered_waveform + white_noise + drift + artifacts
        
        return noisy_waveform
    
    def generate_flow_visualization_data(self, velocity_field, vof_field, coordinates):
        """
        Generate flow visualization data (PIV-like measurements)
        
        Parameters:
        -----------
        velocity_field : tuple
            (u, v, w) velocity components
        vof_field : array
            Volume of fluid field
        coordinates : tuple
            (X, Y, Z) coordinate arrays
            
        Returns:
        --------
        experimental_data : dict
            PIV-like flow measurement data
        """
        u, v, w = velocity_field
        X, Y, Z = coordinates
        
        # Create measurement planes (2D slices)
        measurement_planes = []
        
        # XY plane at middle Z
        z_idx = u.shape[2] // 2
        xy_plane = {
            'plane': 'XY',
            'z_position': Z[0, 0, z_idx],
            'x_coords': X[:, :, z_idx],
            'y_coords': Y[:, :, z_idx],
            'u_velocity': u[:, :, z_idx],
            'v_velocity': v[:, :, z_idx],
            'vof': vof_field[:, :, z_idx]
        }
        measurement_planes.append(xy_plane)
        
        # XZ plane at middle Y
        y_idx = u.shape[1] // 2
        xz_plane = {
            'plane': 'XZ',
            'y_position': Y[0, y_idx, 0],
            'x_coords': X[:, y_idx, :],
            'z_coords': Z[:, y_idx, :],
            'u_velocity': u[:, y_idx, :],
            'w_velocity': w[:, y_idx, :],
            'vof': vof_field[:, y_idx, :]
        }
        measurement_planes.append(xz_plane)
        
        experimental_data = {
            'measurement_planes': measurement_planes,
            'measurement_info': {
                'technique': 'PIV (Particle Image Velocimetry)',
                'resolution': '0.1 mm',
                'measurement_uncertainty': '±2%',
                'temporal_resolution': '1000 Hz'
            }
        }
        
        return experimental_data
    
    def export_experimental_data(self, output_dir):
        """
        Export all experimental data
        """
        print("Generating experimental validation data...")
        
        # Generate data
        frequencies = np.logspace(2, 4, 50)  # 100 Hz to 10 kHz
        volume_fractions = np.linspace(0, 1, 20)
        
        # Sound speed measurements
        sound_speed_data = self.generate_sound_speed_measurements(frequencies, volume_fractions)
        
        # Attenuation measurements
        attenuation_data = self.generate_attenuation_measurements(frequencies, volume_fractions)
        
        # Acoustic waveforms
        source_pos = self.config['acoustic']['source_position']
        receiver_pos = self.config['acoustic']['receiver_positions']
        waveform_data = self.generate_acoustic_waveform_measurements(source_pos, receiver_pos)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Export data
        with open(f'{output_dir}/experimental_sound_speed.json', 'w') as f:
            json.dump(convert_numpy(sound_speed_data), f, indent=2)
        
        with open(f'{output_dir}/experimental_attenuation.json', 'w') as f:
            json.dump(convert_numpy(attenuation_data), f, indent=2)
        
        with open(f'{output_dir}/experimental_waveforms.json', 'w') as f:
            json.dump(convert_numpy(waveform_data), f, indent=2)
        
        # Create CSV files for easy analysis
        self._create_csv_files(output_dir, sound_speed_data, attenuation_data)
        
        print("Experimental validation data exported successfully!")
    
    def _create_csv_files(self, output_dir, sound_speed_data, attenuation_data):
        """
        Create CSV files for experimental data
        """
        # Sound speed CSV
        sound_speed_rows = []
        for freq_data in sound_speed_data['sound_speeds']:
            freq = freq_data['frequency']
            for measurement in freq_data['measurements']:
                sound_speed_rows.append({
                    'frequency': freq,
                    'volume_fraction': measurement['volume_fraction'],
                    'sound_speed': measurement['sound_speed'],
                    'uncertainty': measurement['uncertainty'],
                    'temperature': measurement['temperature'],
                    'pressure': measurement['pressure']
                })
        
        sound_speed_df = pd.DataFrame(sound_speed_rows)
        sound_speed_df.to_csv(f'{output_dir}/experimental_sound_speed.csv', index=False)
        
        # Attenuation CSV
        attenuation_rows = []
        for freq_data in attenuation_data['attenuation_data']:
            freq = freq_data['frequency']
            for measurement in freq_data['measurements']:
                attenuation_rows.append({
                    'frequency': freq,
                    'volume_fraction': measurement['volume_fraction'],
                    'attenuation': measurement['attenuation'],
                    'uncertainty': measurement['uncertainty'],
                    'measurement_method': measurement['measurement_method']
                })
        
        attenuation_df = pd.DataFrame(attenuation_rows)
        attenuation_df.to_csv(f'{output_dir}/experimental_attenuation.csv', index=False)

def main():
    """Main function to generate experimental validation data"""
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create experimental data generator
    exp_generator = ExperimentalDataGenerator(config)
    
    # Export experimental data
    exp_generator.export_experimental_data('experimental_validation_data')
    
    print("Experimental validation data generation complete!")

if __name__ == "__main__":
    main()