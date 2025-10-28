"""
Digital Image Correlation (DIC) Data Generator for SOFC Experiments
Generates realistic strain maps and temporal data for sintering, thermal cycling, and startup/shutdown
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import os

class DICDataGenerator:
    def __init__(self, base_path="dic_data"):
        self.base_path = base_path
        np.random.seed(42)  # For reproducibility
        
    def generate_strain_tensor(self, base_strain, noise_level=0.05, hotspot_prob=0.1):
        """Generate Lagrangian strain tensor with realistic patterns"""
        # Create base strain field (100x100 grid points)
        grid_size = (100, 100)
        
        # Generate smooth base field using Gaussian filtering
        from scipy.ndimage import gaussian_filter
        
        # Ensure positive values for standard deviation
        std_dev = max(abs(base_strain * noise_level), 1e-8)
        
        # Initialize strain components
        exx = np.random.normal(base_strain, std_dev, grid_size)
        eyy = np.random.normal(base_strain * 0.8, std_dev, grid_size)
        exy = np.random.normal(0, std_dev * 0.5, grid_size)
        
        # Apply Gaussian smoothing for realistic spatial correlation
        exx = gaussian_filter(exx, sigma=3)
        eyy = gaussian_filter(eyy, sigma=3)
        exy = gaussian_filter(exy, sigma=3)
        
        # Add strain hotspots (localized high strain regions)
        n_hotspots = np.random.poisson(3)
        for _ in range(n_hotspots):
            if np.random.random() < hotspot_prob:
                x, y = np.random.randint(10, 90, 2)
                radius = np.random.randint(3, 8)
                Y, X = np.ogrid[:100, :100]
                mask = (X - x)**2 + (Y - y)**2 <= radius**2
                
                # Hotspots with >1.0% strain at interfaces
                hotspot_intensity = np.random.uniform(0.01, 0.015)  # 1.0-1.5% strain
                exx[mask] += hotspot_intensity
                eyy[mask] += hotspot_intensity * 0.7
        
        return {
            'exx': exx,
            'eyy': eyy,
            'exy': exy,
            'principal_strain_1': exx + eyy/2 + np.sqrt(((exx-eyy)/2)**2 + exy**2),
            'principal_strain_2': exx + eyy/2 - np.sqrt(((exx-eyy)/2)**2 + exy**2),
            'von_mises_strain': np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)
        }
    
    def generate_sintering_data(self):
        """Generate DIC data for sintering process (1200-1500°C)"""
        print("Generating sintering DIC data...")
        
        # Temperature profile
        temps = np.linspace(25, 1500, 100)  # Room temp to 1500°C
        hold_temps = np.ones(50) * 1500     # Hold at 1500°C
        cool_temps = np.linspace(1500, 25, 100)  # Cooling
        full_temp_profile = np.concatenate([temps, hold_temps, cool_temps])
        
        # Time points (minutes)
        time_points = np.arange(0, len(full_temp_profile)) * 2  # 2 min intervals
        
        data_collection = []
        strain_maps = []
        
        for i, (temp, time) in enumerate(zip(full_temp_profile, time_points)):
            # Base strain increases with temperature (thermal expansion + sintering shrinkage)
            if temp < 1200:
                base_strain = 0.0001 * (temp - 25) / 1175  # Linear thermal expansion
            else:
                # Sintering shrinkage kicks in above 1200°C
                sintering_strain = -0.002 * (1 - np.exp(-(temp - 1200)/200))
                thermal_strain = 0.0001 * (temp - 25) / 1175
                base_strain = thermal_strain + sintering_strain
            
            strain_tensor = self.generate_strain_tensor(base_strain, 
                                                       noise_level=0.03 if temp > 1200 else 0.01)
            
            # Store summary statistics
            data_collection.append({
                'time_min': float(time),
                'temperature_C': float(temp),
                'mean_exx': float(np.mean(strain_tensor['exx'])),
                'std_exx': float(np.std(strain_tensor['exx'])),
                'max_exx': float(np.max(strain_tensor['exx'])),
                'mean_eyy': float(np.mean(strain_tensor['eyy'])),
                'std_eyy': float(np.std(strain_tensor['eyy'])),
                'max_eyy': float(np.max(strain_tensor['eyy'])),
                'mean_von_mises': float(np.mean(strain_tensor['von_mises_strain'])),
                'max_von_mises': float(np.max(strain_tensor['von_mises_strain'])),
                'hotspot_count': int(np.sum(strain_tensor['von_mises_strain'] > 0.01)),
                'timestamp': (datetime.now() + timedelta(minutes=int(time))).isoformat()
            })
            
            # Save full strain maps at key points
            if i % 25 == 0:  # Every 25th frame
                strain_maps.append({
                    'frame': int(i),
                    'time_min': float(time),
                    'temperature_C': float(temp),
                    'strain_tensor': {k: v.tolist() for k, v in strain_tensor.items()}
                })
        
        # Save data
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, 'sintering', 'dic_summary.csv'), index=False)
        
        with open(os.path.join(self.base_path, 'sintering', 'strain_maps.json'), 'w') as f:
            json.dump(strain_maps, f, indent=2)
        
        # Generate Vic-3D compatible output format
        self.generate_vic3d_format(strain_maps[0], 'sintering')
        
        print(f"  - Generated {len(data_collection)} time points")
        print(f"  - Saved {len(strain_maps)} full strain maps")
        
    def generate_thermal_cycling_data(self):
        """Generate DIC data for thermal cycling (ΔT = 400°C)"""
        print("Generating thermal cycling DIC data...")
        
        n_cycles = 10
        data_collection = []
        strain_maps = []
        
        time_counter = 0
        for cycle in range(n_cycles):
            # Heating phase (600°C to 1000°C)
            heat_temps = np.linspace(600, 1000, 30)
            # Cooling phase (1000°C to 600°C)
            cool_temps = np.linspace(1000, 600, 30)
            cycle_temps = np.concatenate([heat_temps, cool_temps])
            
            for i, temp in enumerate(cycle_temps):
                # Strain accumulation with cycling (fatigue effect)
                base_strain = 0.00012 * (temp - 600) / 400  # Thermal strain
                fatigue_strain = 0.00001 * cycle * np.sin(2 * np.pi * i / len(cycle_temps))
                total_base_strain = base_strain + fatigue_strain
                
                # Increase noise and hotspot probability with cycles (degradation)
                noise_level = 0.02 + 0.005 * cycle
                hotspot_prob = 0.1 + 0.02 * cycle
                
                strain_tensor = self.generate_strain_tensor(total_base_strain, 
                                                           noise_level, hotspot_prob)
                
                data_collection.append({
                    'cycle': int(cycle + 1),
                    'time_min': int(time_counter),
                    'temperature_C': float(temp),
                    'phase': 'heating' if i < 30 else 'cooling',
                    'mean_exx': float(np.mean(strain_tensor['exx'])),
                    'std_exx': float(np.std(strain_tensor['exx'])),
                    'max_exx': float(np.max(strain_tensor['exx'])),
                    'mean_eyy': float(np.mean(strain_tensor['eyy'])),
                    'std_eyy': float(np.std(strain_tensor['eyy'])),
                    'max_eyy': float(np.max(strain_tensor['eyy'])),
                    'mean_von_mises': float(np.mean(strain_tensor['von_mises_strain'])),
                    'max_von_mises': float(np.max(strain_tensor['von_mises_strain'])),
                    'hotspot_count': int(np.sum(strain_tensor['von_mises_strain'] > 0.01)),
                    'timestamp': (datetime.now() + timedelta(minutes=time_counter)).isoformat()
                })
                
                # Save strain maps at cycle peaks and valleys
                if i == 0 or i == 29 or i == 59:
                    strain_maps.append({
                        'cycle': int(cycle + 1),
                        'frame': int(len(strain_maps)),
                        'time_min': int(time_counter),
                        'temperature_C': float(temp),
                        'phase': 'heating' if i < 30 else 'cooling',
                        'strain_tensor': {k: v.tolist() for k, v in strain_tensor.items()}
                    })
                
                time_counter += 2
        
        # Save data
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, 'thermal_cycling', 'dic_summary.csv'), index=False)
        
        with open(os.path.join(self.base_path, 'thermal_cycling', 'strain_maps.json'), 'w') as f:
            json.dump(strain_maps, f, indent=2)
        
        print(f"  - Generated {n_cycles} cycles with {len(data_collection)} time points")
        print(f"  - Saved {len(strain_maps)} full strain maps")
        
    def generate_startup_shutdown_data(self):
        """Generate DIC data for startup/shutdown cycles"""
        print("Generating startup/shutdown DIC data...")
        
        n_cycles = 5
        data_collection = []
        strain_maps = []
        
        time_counter = 0
        for cycle in range(n_cycles):
            # Startup: Room temp to operating temp (800°C) - rapid
            startup_temps = np.logspace(np.log10(25), np.log10(800), 20)
            # Operation hold
            operation_temps = np.ones(10) * 800
            # Shutdown: 800°C to room temp - controlled
            shutdown_temps = np.linspace(800, 25, 30)
            
            cycle_temps = np.concatenate([startup_temps, operation_temps, shutdown_temps])
            
            for i, temp in enumerate(cycle_temps):
                # Different strain behavior during startup vs shutdown
                if i < 20:  # Startup - rapid thermal shock
                    base_strain = 0.00015 * (temp - 25) / 775
                    shock_factor = 1.5  # Higher strain rate during rapid startup
                elif i < 30:  # Operation
                    base_strain = 0.00015
                    shock_factor = 1.0
                else:  # Shutdown
                    base_strain = 0.00015 * (temp - 25) / 775
                    shock_factor = 0.8
                
                # Cumulative damage effect
                damage_strain = 0.00002 * cycle
                total_base_strain = base_strain * shock_factor + damage_strain
                
                # Ensure positive base strain for tensor generation
                total_base_strain = max(total_base_strain, 1e-6)
                
                strain_tensor = self.generate_strain_tensor(total_base_strain,
                                                           noise_level=0.03 * shock_factor,
                                                           hotspot_prob=0.15 if i < 20 else 0.1)
                
                phase = 'startup' if i < 20 else ('operation' if i < 30 else 'shutdown')
                
                data_collection.append({
                    'cycle': int(cycle + 1),
                    'time_min': int(time_counter),
                    'temperature_C': float(temp),
                    'phase': phase,
                    'mean_exx': float(np.mean(strain_tensor['exx'])),
                    'std_exx': float(np.std(strain_tensor['exx'])),
                    'max_exx': float(np.max(strain_tensor['exx'])),
                    'mean_eyy': float(np.mean(strain_tensor['eyy'])),
                    'std_eyy': float(np.std(strain_tensor['eyy'])),
                    'max_eyy': float(np.max(strain_tensor['eyy'])),
                    'mean_von_mises': float(np.mean(strain_tensor['von_mises_strain'])),
                    'max_von_mises': float(np.max(strain_tensor['von_mises_strain'])),
                    'hotspot_count': int(np.sum(strain_tensor['von_mises_strain'] > 0.01)),
                    'timestamp': (datetime.now() + timedelta(minutes=time_counter)).isoformat()
                })
                
                # Save strain maps at critical points
                if i in [0, 19, 29, 59]:  # Start, end of startup, during operation, end
                    strain_maps.append({
                        'cycle': int(cycle + 1),
                        'frame': int(len(strain_maps)),
                        'time_min': int(time_counter),
                        'temperature_C': float(temp),
                        'phase': phase,
                        'strain_tensor': {k: v.tolist() for k, v in strain_tensor.items()}
                    })
                
                time_counter += 3
        
        # Save data
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, 'startup_shutdown', 'dic_summary.csv'), index=False)
        
        with open(os.path.join(self.base_path, 'startup_shutdown', 'strain_maps.json'), 'w') as f:
            json.dump(strain_maps, f, indent=2)
        
        print(f"  - Generated {n_cycles} cycles with {len(data_collection)} time points")
        print(f"  - Saved {len(strain_maps)} full strain maps")
        
    def generate_vic3d_format(self, strain_map_data, experiment_type):
        """Generate Vic-3D compatible output format"""
        # Vic-3D typically exports data in specific formats
        vic3d_output = {
            'header': {
                'software': 'Vic-3D 7',
                'date': datetime.now().isoformat(),
                'experiment': experiment_type,
                'calibration': {
                    'pixel_to_mm': 0.05,
                    'subset_size': 21,
                    'step_size': 5,
                    'strain_window': 15
                }
            },
            'data': {
                'coordinates': {
                    'x': np.linspace(0, 10, 100).tolist(),  # 10mm sample
                    'y': np.linspace(0, 10, 100).tolist()
                },
                'strain_data': strain_map_data['strain_tensor']
            }
        }
        
        with open(os.path.join(self.base_path, experiment_type, 'vic3d_output.json'), 'w') as f:
            json.dump(vic3d_output, f, indent=2)
    
    def generate_speckle_pattern_metadata(self):
        """Generate metadata for speckle pattern images"""
        print("Generating speckle pattern metadata...")
        
        for exp_type in ['sintering', 'thermal_cycling', 'startup_shutdown']:
            metadata = {
                'speckle_pattern': {
                    'type': 'spray_paint',
                    'speckle_size_mm': 0.1,
                    'coverage_percent': 50,
                    'contrast_ratio': 0.8
                },
                'camera_settings': {
                    'resolution': '2048x2048',
                    'frame_rate_fps': 1,
                    'exposure_ms': 10,
                    'aperture': 'f/8',
                    'working_distance_mm': 300
                },
                'illumination': {
                    'type': 'LED_array',
                    'intensity_lumens': 5000,
                    'angle_degrees': 45
                },
                'image_files': [
                    f'IMG_{i:04d}.tiff' for i in range(100)
                ]
            }
            
            with open(os.path.join(self.base_path, exp_type, 'speckle_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def run_all(self):
        """Generate all DIC datasets"""
        self.generate_sintering_data()
        self.generate_thermal_cycling_data()
        self.generate_startup_shutdown_data()
        self.generate_speckle_pattern_metadata()
        print("\nDIC data generation complete!")

if __name__ == "__main__":
    generator = DICDataGenerator()
    generator.run_all()