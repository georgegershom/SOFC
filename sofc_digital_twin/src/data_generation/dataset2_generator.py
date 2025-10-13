"""
Dataset 2 Generator: Experimental Data for Validation and Ground Truth
Generates synthetic experimental data that mimics real SOFC test rig measurements
"""

import numpy as np
import pandas as pd
import h5py
import yaml
import os
from typing import Dict, List, Tuple
import time
from datetime import datetime, timedelta
from tqdm import tqdm

class Dataset2Generator:
    """
    Generates Dataset 2: Experimental validation data
    
    This includes:
    - Global operational data (I-V curves, temperatures, flow rates)
    - Electrochemical Impedance Spectroscopy (EIS) data
    - Thermal imaging data
    - Strain gauge measurements
    - Acoustic emission data
    - Post-mortem analysis data
    """
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        self.output_dir = "datasets/dataset2_experimental"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Experimental parameters
        self.test_duration = 720  # hours (30 days)
        self.sampling_frequency = 1.0  # Hz for high-frequency data
        self.eis_frequency = 24  # hours between EIS measurements
        
    def generate_global_operational_data(self) -> pd.DataFrame:
        """
        Generate time-series operational data from SOFC test rig
        """
        print("Generating global operational data...")
        
        # Time vector
        total_seconds = int(self.test_duration * 3600)
        time_points = np.arange(0, total_seconds, 1/self.sampling_frequency)
        n_points = len(time_points)
        
        # Base operating conditions
        base_current = 0.8  # A/cm²
        base_voltage = 0.7  # V
        base_temp_fuel = 800  # °C
        base_temp_air = 750   # °C
        base_flow_fuel = 100  # sccm
        base_flow_air = 500   # sccm
        
        # Add realistic variations and degradation
        # Current density with load cycling
        current_density = base_current * (1 + 0.1 * np.sin(2*np.pi*time_points/(24*3600)) + 
                                        0.05 * np.random.normal(0, 1, n_points))
        
        # Voltage with degradation trend
        degradation_rate = -50e-6  # V/hour
        voltage = (base_voltage + degradation_rate * time_points/3600 + 
                  0.02 * np.random.normal(0, 1, n_points))
        
        # Power
        power = current_density * voltage
        
        # Temperatures with thermal cycling
        temp_fuel_inlet = (base_temp_fuel + 
                          20 * np.sin(2*np.pi*time_points/(12*3600)) +
                          5 * np.random.normal(0, 1, n_points))
        
        temp_air_inlet = (base_temp_air + 
                         15 * np.sin(2*np.pi*time_points/(12*3600)) +
                         3 * np.random.normal(0, 1, n_points))
        
        # Outlet temperatures (higher due to heat generation)
        temp_fuel_outlet = temp_fuel_inlet + 50 + 10 * np.random.normal(0, 1, n_points)
        temp_air_outlet = temp_air_inlet + 30 + 8 * np.random.normal(0, 1, n_points)
        
        # Flow rates with control variations
        flow_fuel = base_flow_fuel + 5 * np.random.normal(0, 1, n_points)
        flow_air = base_flow_air + 20 * np.random.normal(0, 1, n_points)
        
        # Create DataFrame
        data = {
            'timestamp': [datetime.now() + timedelta(seconds=t) for t in time_points],
            'time_hours': time_points / 3600,
            'current_density': current_density,
            'voltage': voltage,
            'power': power,
            'temp_fuel_inlet': temp_fuel_inlet,
            'temp_air_inlet': temp_air_inlet,
            'temp_fuel_outlet': temp_fuel_outlet,
            'temp_air_outlet': temp_air_outlet,
            'flow_fuel': flow_fuel,
            'flow_air': flow_air
        }
        
        df = pd.DataFrame(data)
        return df
    
    def generate_eis_data(self) -> List[Dict]:
        """
        Generate Electrochemical Impedance Spectroscopy data
        """
        print("Generating EIS data...")
        
        eis_measurements = []
        n_eis_points = int(self.test_duration / self.eis_frequency)
        
        # Frequency range for EIS (typical SOFC range)
        frequencies = np.logspace(-2, 5, 50)  # 0.01 Hz to 100 kHz
        
        for i in range(n_eis_points):
            test_time = i * self.eis_frequency  # hours
            
            # Base impedance parameters (evolve with degradation)
            R_ohmic = 0.15 + 0.001 * test_time  # Ohmic resistance increases
            R_act = 0.10 + 0.0005 * test_time   # Activation resistance increases
            R_conc = 0.05 + 0.0002 * test_time  # Concentration resistance increases
            
            # Capacitance values
            C_dl = 0.01  # Double layer capacitance
            C_chem = 0.1  # Chemical capacitance
            
            # Calculate impedance at each frequency
            Z_real = []
            Z_imag = []
            
            for f in frequencies:
                omega = 2 * np.pi * f
                
                # Equivalent circuit: R_ohmic + (R_act || C_dl) + (R_conc || C_chem)
                Z_act = R_act / (1 + 1j * omega * R_act * C_dl)
                Z_conc = R_conc / (1 + 1j * omega * R_conc * C_chem)
                Z_total = R_ohmic + Z_act + Z_conc
                
                # Add noise
                noise_real = 0.005 * np.random.normal()
                noise_imag = 0.005 * np.random.normal()
                
                Z_real.append(Z_total.real + noise_real)
                Z_imag.append(-Z_total.imag + noise_imag)  # Negative for capacitive
            
            eis_data = {
                'test_time_hours': test_time,
                'frequencies': frequencies.tolist(),
                'Z_real': Z_real,
                'Z_imag': Z_imag,
                'R_ohmic': R_ohmic,
                'R_activation': R_act,
                'R_concentration': R_conc
            }
            
            eis_measurements.append(eis_data)
        
        return eis_measurements
    
    def generate_thermal_imaging_data(self) -> List[Dict]:
        """
        Generate thermal imaging data (2D temperature maps)
        """
        print("Generating thermal imaging data...")
        
        thermal_images = []
        n_thermal_images = 100  # Images taken during thermal transients
        
        # Image dimensions
        img_width, img_height = 64, 48  # pixels
        
        for i in range(n_thermal_images):
            test_time = i * (self.test_duration / n_thermal_images)
            
            # Base temperature distribution
            x = np.linspace(0, 1, img_width)
            y = np.linspace(0, 1, img_height)
            X, Y = np.meshgrid(x, y)
            
            # Temperature pattern (hotspot in center, cooler edges)
            base_temp = 800  # °C
            temp_variation = 50
            
            temperature_map = (base_temp + 
                             temp_variation * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.1) +
                             10 * np.random.normal(0, 1, (img_height, img_width)))
            
            # Add degradation effects (hot spots develop over time)
            if test_time > 200:  # After 200 hours
                hotspot_intensity = (test_time - 200) / 100
                hotspot_x, hotspot_y = 0.3, 0.7
                hotspot = hotspot_intensity * 30 * np.exp(-((X-hotspot_x)**2 + (Y-hotspot_y)**2) / 0.05)
                temperature_map += hotspot
            
            thermal_data = {
                'test_time_hours': test_time,
                'temperature_map': temperature_map.tolist(),
                'max_temperature': float(np.max(temperature_map)),
                'min_temperature': float(np.min(temperature_map)),
                'mean_temperature': float(np.mean(temperature_map)),
                'image_dimensions': [img_width, img_height]
            }
            
            thermal_images.append(thermal_data)
        
        return thermal_images
    
    def generate_strain_gauge_data(self) -> pd.DataFrame:
        """
        Generate strain gauge measurements at critical locations
        """
        print("Generating strain gauge data...")
        
        # Time vector (lower frequency than operational data)
        time_points = np.arange(0, self.test_duration * 3600, 60)  # Every minute
        n_points = len(time_points)
        
        # Strain gauge locations
        locations = ['interconnect_center', 'seal_edge', 'current_collector', 'frame_corner']
        
        strain_data = {
            'timestamp': [datetime.now() + timedelta(seconds=t) for t in time_points],
            'time_hours': time_points / 3600
        }
        
        for location in locations:
            # Base strain with thermal cycling and creep
            thermal_strain = 200e-6 * np.sin(2*np.pi*time_points/(12*3600))  # Thermal cycling
            creep_strain = 50e-6 * (time_points / 3600) / 1000  # Creep accumulation
            mechanical_strain = 100e-6 * np.sin(2*np.pi*time_points/(24*3600))  # Load cycling
            noise = 10e-6 * np.random.normal(0, 1, n_points)
            
            total_strain = thermal_strain + creep_strain + mechanical_strain + noise
            strain_data[f'strain_{location}'] = total_strain
        
        return pd.DataFrame(strain_data)
    
    def generate_acoustic_emission_data(self) -> List[Dict]:
        """
        Generate acoustic emission events (crack formation/propagation)
        """
        print("Generating acoustic emission data...")
        
        ae_events = []
        
        # AE events become more frequent as degradation progresses
        for hour in range(int(self.test_duration)):
            # Event probability increases with time (degradation)
            event_probability = 0.001 + 0.00001 * hour  # Events per hour
            
            if np.random.random() < event_probability:
                # Generate AE event
                event_time = hour + np.random.random()  # Random time within hour
                
                # AE parameters
                amplitude = 40 + 20 * np.random.random()  # dB
                duration = 100 + 500 * np.random.random()  # microseconds
                energy = amplitude * duration / 1000  # Arbitrary units
                frequency_peak = 100 + 400 * np.random.random()  # kHz
                
                # Location (simplified 2D coordinates)
                x_location = np.random.random()
                y_location = np.random.random()
                
                ae_event = {
                    'test_time_hours': event_time,
                    'amplitude_db': amplitude,
                    'duration_us': duration,
                    'energy': energy,
                    'frequency_peak_khz': frequency_peak,
                    'x_location': x_location,
                    'y_location': y_location,
                    'event_type': 'crack_propagation' if amplitude > 50 else 'micro_crack'
                }
                
                ae_events.append(ae_event)
        
        return ae_events
    
    def generate_postmortem_data(self) -> Dict:
        """
        Generate post-mortem analysis data (SEM, X-ray tomography)
        """
        print("Generating post-mortem analysis data...")
        
        # SEM analysis data
        sem_data = {
            'crack_locations': [
                {'x': 0.3, 'y': 0.7, 'length_um': 150, 'width_um': 2.5},
                {'x': 0.6, 'y': 0.4, 'length_um': 80, 'width_um': 1.2},
                {'x': 0.8, 'y': 0.2, 'length_um': 200, 'width_um': 3.1}
            ],
            'delamination_areas': [
                {'x': 0.4, 'y': 0.6, 'area_mm2': 2.5, 'depth_um': 15},
                {'x': 0.7, 'y': 0.3, 'area_mm2': 1.8, 'depth_um': 12}
            ],
            'porosity_measurements': {
                'initial_porosity': 0.35,
                'final_porosity': 0.28,
                'degradation_percentage': 20.0
            },
            'elemental_analysis': {
                'chromium_migration': {'detected': True, 'penetration_depth_um': 25},
                'nickel_depletion': {'detected': True, 'depletion_percentage': 15},
                'sulfur_contamination': {'detected': False}
            }
        }
        
        # X-ray tomography data (3D microstructure)
        tomography_data = {
            'voxel_size_um': 0.5,
            'image_dimensions': [512, 512, 256],  # voxels
            'crack_volume_fraction': 0.003,
            'pore_size_distribution': {
                'mean_pore_size_um': 2.5,
                'std_pore_size_um': 1.2,
                'total_porosity': 0.28
            },
            'interconnectivity': {
                'connected_porosity': 0.85,
                'isolated_pores': 0.15
            }
        }
        
        postmortem_data = {
            'sem_analysis': sem_data,
            'xray_tomography': tomography_data,
            'test_duration_hours': self.test_duration,
            'final_performance': {
                'voltage_degradation_mv': 45,
                'power_degradation_percentage': 8.5,
                'resistance_increase_percentage': 12.3
            }
        }
        
        return postmortem_data
    
    def save_dataset2(self):
        """Save all Dataset 2 components"""
        print("Saving Dataset 2...")
        
        # Generate all data components
        operational_data = self.generate_global_operational_data()
        eis_data = self.generate_eis_data()
        thermal_data = self.generate_thermal_imaging_data()
        strain_data = self.generate_strain_gauge_data()
        ae_data = self.generate_acoustic_emission_data()
        postmortem_data = self.generate_postmortem_data()
        
        # Save operational data as CSV
        operational_data.to_csv(
            os.path.join(self.output_dir, 'operational_data.csv'), 
            index=False
        )
        
        # Save strain gauge data as CSV
        strain_data.to_csv(
            os.path.join(self.output_dir, 'strain_gauge_data.csv'),
            index=False
        )
        
        # Save other data as HDF5
        with h5py.File(os.path.join(self.output_dir, 'experimental_data.h5'), 'w') as f:
            # EIS data
            eis_group = f.create_group('eis_data')
            for i, eis in enumerate(eis_data):
                measurement_group = eis_group.create_group(f'measurement_{i:03d}')
                for key, value in eis.items():
                    measurement_group.create_dataset(key, data=value)
            
            # Thermal imaging data
            thermal_group = f.create_group('thermal_imaging')
            for i, thermal in enumerate(thermal_data):
                image_group = thermal_group.create_group(f'image_{i:03d}')
                for key, value in thermal.items():
                    image_group.create_dataset(key, data=value)
            
            # Acoustic emission data
            ae_group = f.create_group('acoustic_emission')
            for i, ae_event in enumerate(ae_data):
                event_group = ae_group.create_group(f'event_{i:04d}')
                for key, value in ae_event.items():
                    event_group.attrs[key] = value
            
            # Post-mortem data
            pm_group = f.create_group('postmortem_analysis')
            
            # SEM data
            sem_group = pm_group.create_group('sem_analysis')
            for key, value in postmortem_data['sem_analysis'].items():
                if isinstance(value, list):
                    list_group = sem_group.create_group(key)
                    for i, item in enumerate(value):
                        item_group = list_group.create_group(f'item_{i}')
                        for sub_key, sub_value in item.items():
                            item_group.attrs[sub_key] = sub_value
                elif isinstance(value, dict):
                    dict_group = sem_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            sub_dict_group = dict_group.create_group(sub_key)
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                sub_dict_group.attrs[sub_sub_key] = sub_sub_value
                        else:
                            dict_group.attrs[sub_key] = sub_value
                else:
                    sem_group.attrs[key] = value
            
            # Tomography data
            tomo_group = pm_group.create_group('xray_tomography')
            for key, value in postmortem_data['xray_tomography'].items():
                if isinstance(value, dict):
                    dict_group = tomo_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        dict_group.attrs[sub_key] = sub_value
                else:
                    tomo_group.attrs[key] = value
        
        # Save summary
        summary = {
            'dataset_info': {
                'name': 'Dataset 2 - Experimental Validation Data',
                'test_duration_hours': self.test_duration,
                'sampling_frequency_hz': self.sampling_frequency,
                'generation_timestamp': datetime.now().isoformat()
            },
            'data_components': {
                'operational_data_points': len(operational_data),
                'eis_measurements': len(eis_data),
                'thermal_images': len(thermal_data),
                'strain_measurements': len(strain_data),
                'acoustic_events': len(ae_data),
                'postmortem_analyses': 1
            },
            'file_locations': {
                'operational_data': 'operational_data.csv',
                'strain_data': 'strain_gauge_data.csv',
                'other_data': 'experimental_data.h5'
            }
        }
        
        with open(os.path.join(self.output_dir, 'dataset2_summary.yaml'), 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"Dataset 2 saved to: {self.output_dir}")
        return summary

def main():
    """Main function to generate Dataset 2"""
    config_path = "config/simulation_config.yaml"
    
    # Create generator
    generator = Dataset2Generator(config_path)
    
    # Generate and save dataset
    summary = generator.save_dataset2()
    
    print("\nDataset 2 generation completed!")
    print(f"Components generated: {summary['data_components']}")

if __name__ == "__main__":
    main()