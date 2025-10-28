#!/usr/bin/env python3
"""
SOFC Experimental Measurement Dataset Generator
Generates realistic experimental data for SOFC research including:
- Digital Image Correlation (DIC) data
- Synchrotron X-ray Diffraction (XRD) data  
- Post-mortem analysis data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata
import json
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

class SOFCDataGenerator:
    def __init__(self):
        self.temperature_range = np.linspace(1200, 1500, 100)  # °C
        self.time_points = np.linspace(0, 3600, 100)  # seconds
        self.sample_size = (50, 50)  # pixels for strain maps
        
    def generate_dic_data(self):
        """Generate Digital Image Correlation data"""
        print("Generating DIC data...")
        
        # Strain maps during sintering (1200-1500°C)
        sintering_strain_data = []
        for temp in self.temperature_range:
            # Create realistic strain distribution with hotspots
            strain_map = self._generate_strain_map(temp)
            sintering_strain_data.append({
                'temperature': temp,
                'time': self._temp_to_time(temp),
                'strain_map': strain_map,
                'max_strain': np.max(strain_map),
                'mean_strain': np.mean(strain_map),
                'hotspot_locations': self._find_strain_hotspots(strain_map, threshold=0.01)
            })
        
        # Thermal cycling data (ΔT = 400°C)
        thermal_cycling_data = []
        for cycle in range(5):  # 5 thermal cycles
            for temp in np.linspace(800, 1200, 20):  # ΔT = 400°C
                strain_map = self._generate_thermal_cycling_strain(temp, cycle)
                thermal_cycling_data.append({
                    'cycle': cycle,
                    'temperature': temp,
                    'time': cycle * 3600 + (temp - 800) * 9,  # 9 seconds per degree
                    'strain_map': strain_map,
                    'max_strain': np.max(strain_map),
                    'mean_strain': np.mean(strain_map)
                })
        
        # Startup/shutdown cycles
        startup_shutdown_data = []
        for cycle in range(3):
            # Startup: 25°C to 1200°C
            startup_temps = np.linspace(25, 1200, 50)
            for temp in startup_temps:
                strain_map = self._generate_startup_strain(temp)
                startup_shutdown_data.append({
                    'cycle': cycle,
                    'phase': 'startup',
                    'temperature': temp,
                    'time': cycle * 7200 + (temp - 25) * 6,
                    'strain_map': strain_map,
                    'max_strain': np.max(strain_map)
                })
            
            # Shutdown: 1200°C to 25°C
            shutdown_temps = np.linspace(1200, 25, 50)
            for temp in shutdown_temps:
                strain_map = self._generate_shutdown_strain(temp)
                startup_shutdown_data.append({
                    'cycle': cycle,
                    'phase': 'shutdown',
                    'temperature': temp,
                    'time': cycle * 7200 + 3000 + (1200 - temp) * 6,
                    'strain_map': strain_map,
                    'max_strain': np.max(strain_map)
                })
        
        # Speckle pattern images with timestamps
        speckle_patterns = self._generate_speckle_patterns()
        
        # Lagrangian strain tensor outputs
        lagrangian_tensors = self._generate_lagrangian_tensors()
        
        return {
            'sintering': sintering_strain_data,
            'thermal_cycling': thermal_cycling_data,
            'startup_shutdown': startup_shutdown_data,
            'speckle_patterns': speckle_patterns,
            'lagrangian_tensors': lagrangian_tensors
        }
    
    def _generate_strain_map(self, temperature):
        """Generate realistic strain map for given temperature"""
        x, y = np.meshgrid(np.linspace(0, 1, self.sample_size[0]), 
                          np.linspace(0, 1, self.sample_size[1]))
        
        # Base thermal strain (increases with temperature)
        base_strain = (temperature - 1200) / 300 * 0.005
        
        # Add random noise and localized hotspots
        noise = np.random.normal(0, 0.001, self.sample_size)
        
        # Create interface hotspots (higher strain at material interfaces)
        interface_strain = np.zeros_like(x)
        interface_strain += 0.003 * np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / 0.1)  # Ni-YSZ interface
        interface_strain += 0.002 * np.exp(-((x - 0.7)**2 + (y - 0.3)**2) / 0.08)  # YSZ-electrolyte interface
        
        # Temperature-dependent strain concentration
        temp_factor = 1 + (temperature - 1200) / 300 * 0.5
        
        total_strain = base_strain + noise + interface_strain * temp_factor
        
        return total_strain
    
    def _generate_thermal_cycling_strain(self, temperature, cycle):
        """Generate strain map for thermal cycling"""
        x, y = np.meshgrid(np.linspace(0, 1, self.sample_size[0]), 
                          np.linspace(0, 1, self.sample_size[1]))
        
        # Cyclic strain accumulation
        cycle_factor = 1 + cycle * 0.1
        
        # Temperature-dependent strain
        temp_strain = (temperature - 1000) / 200 * 0.003 * cycle_factor
        
        # Add fatigue-induced microcracks
        fatigue_cracks = np.zeros_like(x)
        if cycle > 0:
            crack_density = cycle * 0.02
            for _ in range(int(crack_density * 100)):
                cx, cy = np.random.random(2)
                fatigue_cracks += 0.001 * np.exp(-((x - cx)**2 + (y - cy)**2) / 0.05)
        
        return temp_strain + fatigue_cracks + np.random.normal(0, 0.0005, self.sample_size)
    
    def _generate_startup_strain(self, temperature):
        """Generate strain map for startup phase"""
        x, y = np.meshgrid(np.linspace(0, 1, self.sample_size[0]), 
                          np.linspace(0, 1, self.sample_size[1]))
        
        # Rapid heating causes thermal shock
        heating_rate = 2.0  # °C/s
        thermal_shock = heating_rate * 0.0001 * (temperature / 1200)
        
        # Non-uniform heating creates stress gradients
        gradient_strain = 0.002 * (x - 0.5) * (temperature / 1200)
        
        return thermal_shock + gradient_strain + np.random.normal(0, 0.0003, self.sample_size)
    
    def _generate_shutdown_strain(self, temperature):
        """Generate strain map for shutdown phase"""
        x, y = np.meshgrid(np.linspace(0, 1, self.sample_size[0]), 
                          np.linspace(0, 1, self.sample_size[1]))
        
        # Rapid cooling causes tensile stress
        cooling_rate = 1.5  # °C/s
        cooling_stress = cooling_rate * 0.0002 * ((1200 - temperature) / 1200)
        
        # Shrinkage-induced cracking
        shrinkage_cracks = 0.001 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2) * ((1200 - temperature) / 1200)
        
        return cooling_stress + shrinkage_cracks + np.random.normal(0, 0.0004, self.sample_size)
    
    def _find_strain_hotspots(self, strain_map, threshold=0.01):
        """Find locations where strain exceeds threshold"""
        hotspots = []
        for i in range(strain_map.shape[0]):
            for j in range(strain_map.shape[1]):
                if strain_map[i, j] > threshold:
                    hotspots.append({
                        'x': j / strain_map.shape[1],
                        'y': i / strain_map.shape[0],
                        'strain': strain_map[i, j]
                    })
        return hotspots
    
    def _temp_to_time(self, temperature):
        """Convert temperature to time (heating rate: 0.5°C/s)"""
        return (temperature - 1200) / 0.5
    
    def _generate_speckle_patterns(self):
        """Generate speckle pattern images with timestamps"""
        patterns = []
        for i in range(20):  # 20 time points
            # Create random speckle pattern
            pattern = np.random.random((100, 100))
            # Add some structure to make it more realistic
            pattern += 0.3 * np.sin(10 * np.pi * np.linspace(0, 1, 100)[:, None])
            pattern += 0.3 * np.sin(10 * np.pi * np.linspace(0, 1, 100)[None, :])
            
            patterns.append({
                'timestamp': datetime.now() + timedelta(seconds=i * 180),  # Every 3 minutes
                'pattern': pattern,
                'image_id': f'SP_{i:03d}'
            })
        return patterns
    
    def _generate_lagrangian_tensors(self):
        """Generate Lagrangian strain tensor outputs"""
        tensors = []
        for i in range(50):  # 50 measurement points
            # Generate realistic strain tensor components
            exx = np.random.normal(0.002, 0.0005)
            eyy = np.random.normal(0.0015, 0.0004)
            ezz = np.random.normal(0.001, 0.0003)
            exy = np.random.normal(0, 0.0002)
            exz = np.random.normal(0, 0.0001)
            eyz = np.random.normal(0, 0.0001)
            
            tensor = np.array([[exx, exy, exz],
                             [exy, eyy, eyz],
                             [exz, eyz, ezz]])
            
            tensors.append({
                'point_id': i,
                'position': [np.random.random(), np.random.random(), np.random.random()],
                'strain_tensor': tensor.tolist(),
                'principal_strains': np.linalg.eigvals(tensor).tolist(),
                'von_mises_strain': np.sqrt(0.5 * ((exx-eyy)**2 + (eyy-ezz)**2 + (ezz-exx)**2 + 6*(exy**2 + exz**2 + eyz**2)))
            })
        return tensors
    
    def generate_xrd_data(self):
        """Generate Synchrotron X-ray Diffraction data"""
        print("Generating XRD data...")
        
        # Residual stress profiles across SOFC cross-sections
        cross_section_positions = np.linspace(0, 2.0, 50)  # mm across cross-section
        residual_stresses = []
        
        for pos in cross_section_positions:
            # Different stress profiles for different layers
            if pos < 0.5:  # Anode layer
                stress = 150 + 50 * np.sin(2 * np.pi * pos / 0.5) + np.random.normal(0, 10)
                layer = 'anode'
            elif pos < 1.0:  # Electrolyte layer
                stress = 200 + 30 * np.cos(2 * np.pi * (pos - 0.5) / 0.5) + np.random.normal(0, 8)
                layer = 'electrolyte'
            else:  # Cathode layer
                stress = 120 + 40 * np.sin(2 * np.pi * (pos - 1.0) / 1.0) + np.random.normal(0, 12)
                layer = 'cathode'
            
            residual_stresses.append({
                'position': pos,
                'stress': stress,
                'layer': layer,
                'depth': np.random.uniform(0, 0.1)  # mm
            })
        
        # Lattice strain measurements under thermal load
        lattice_strains = []
        temperatures = np.linspace(25, 1200, 30)
        
        for temp in temperatures:
            # Different materials have different thermal expansion
            for material in ['YSZ', 'Ni', 'LSCF']:
                if material == 'YSZ':
                    thermal_expansion = 10.5e-6  # /K
                    lattice_strain = thermal_expansion * (temp - 25)
                elif material == 'Ni':
                    thermal_expansion = 13.4e-6  # /K
                    lattice_strain = thermal_expansion * (temp - 25)
                else:  # LSCF
                    thermal_expansion = 12.8e-6  # /K
                    lattice_strain = thermal_expansion * (temp - 25)
                
                lattice_strains.append({
                    'temperature': temp,
                    'material': material,
                    'lattice_strain': lattice_strain,
                    'peak_position': 2 * np.arcsin(1.54 / (2 * (1 + lattice_strain) * 2.5))  # Bragg angle
                })
        
        # Peak shift data for sin²ψ method
        psi_angles = np.linspace(0, 60, 20)  # degrees
        peak_shifts = []
        
        for psi in psi_angles:
            # sin²ψ method for stress calculation
            sin2psi = np.sin(np.radians(psi))**2
            peak_shift = 0.1 * sin2psi + np.random.normal(0, 0.01)
            
            peak_shifts.append({
                'psi_angle': psi,
                'sin2psi': sin2psi,
                'peak_shift': peak_shift,
                'intensity': 1000 * np.exp(-0.1 * psi) + np.random.normal(0, 50)
            })
        
        # Microcrack initiation thresholds
        microcrack_data = []
        for i in range(100):
            strain = np.random.uniform(0, 0.05)
            if strain > 0.02:  # Critical strain threshold
                crack_initiated = True
                crack_length = (strain - 0.02) * 1000  # μm
            else:
                crack_initiated = False
                crack_length = 0
            
            microcrack_data.append({
                'strain': strain,
                'crack_initiated': crack_initiated,
                'crack_length': crack_length,
                'location': [np.random.random(), np.random.random()]
            })
        
        return {
            'residual_stresses': residual_stresses,
            'lattice_strains': lattice_strains,
            'peak_shifts': peak_shifts,
            'microcrack_data': microcrack_data
        }
    
    def generate_post_mortem_data(self):
        """Generate post-mortem analysis data"""
        print("Generating post-mortem analysis data...")
        
        # SEM images for crack density quantification
        sem_images = []
        for i in range(10):  # 10 SEM images
            # Generate crack pattern
            crack_density = np.random.uniform(0.1, 2.0)  # cracks/mm²
            image_size = (200, 200)  # pixels
            
            # Create crack pattern
            crack_pattern = np.zeros(image_size)
            num_cracks = int(crack_density * 0.01 * image_size[0] * image_size[1])  # Scale to pixels
            
            for _ in range(num_cracks):
                # Random crack orientation and position
                start_x, start_y = np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1])
                angle = np.random.uniform(0, 2*np.pi)
                length = np.random.uniform(10, 50)
                
                end_x = int(start_x + length * np.cos(angle))
                end_y = int(start_y + length * np.sin(angle))
                
                # Draw crack line
                if 0 <= end_x < image_size[0] and 0 <= end_y < image_size[1]:
                    crack_pattern[start_x:end_x, start_y:end_y] = 1
            
            sem_images.append({
                'image_id': f'SEM_{i:03d}',
                'crack_density': crack_density,
                'crack_pattern': crack_pattern,
                'magnification': np.random.choice([1000, 2000, 5000, 10000]),
                'field_of_view': np.random.uniform(50, 200)  # μm
            })
        
        # EDS line scans for elemental composition
        eds_scans = []
        for scan_id in range(5):
            positions = np.linspace(0, 100, 50)  # μm
            scan_data = []
            
            for pos in positions:
                # Different composition profiles across interfaces
                if pos < 20:  # Anode region
                    ni_content = 0.6 + 0.1 * np.sin(2 * np.pi * pos / 20)
                    zr_content = 0.2 + 0.05 * np.cos(2 * np.pi * pos / 20)
                    y_content = 0.1 + 0.02 * np.sin(2 * np.pi * pos / 20)
                elif pos < 40:  # Electrolyte region
                    ni_content = 0.05
                    zr_content = 0.4 + 0.1 * np.sin(2 * np.pi * (pos - 20) / 20)
                    y_content = 0.2 + 0.05 * np.cos(2 * np.pi * (pos - 20) / 20)
                else:  # Cathode region
                    ni_content = 0.02
                    zr_content = 0.1
                    y_content = 0.05
                
                # Add random noise
                ni_content += np.random.normal(0, 0.01)
                zr_content += np.random.normal(0, 0.01)
                y_content += np.random.normal(0, 0.005)
                
                scan_data.append({
                    'position': pos,
                    'Ni': max(0, min(1, ni_content)),
                    'Zr': max(0, min(1, zr_content)),
                    'Y': max(0, min(1, y_content)),
                    'O': 1 - (ni_content + zr_content + y_content)
                })
            
            eds_scans.append({
                'scan_id': scan_id,
                'scan_data': scan_data,
                'beam_energy': np.random.choice([10, 15, 20]),  # keV
                'step_size': 2.0  # μm
            })
        
        # Nano-indentation data
        nano_indentation = []
        for i in range(50):  # 50 indentation points
            # Different properties for different phases
            if np.random.random() < 0.3:  # YSZ phase
                youngs_modulus = np.random.normal(184.7, 10)  # GPa
                hardness = np.random.normal(12.5, 1.5)  # GPa
                phase = 'YSZ'
            elif np.random.random() < 0.6:  # Ni-YSZ composite
                youngs_modulus = np.random.normal(109.8, 8)  # GPa
                hardness = np.random.normal(8.2, 1.2)  # GPa
                phase = 'Ni-YSZ'
            else:  # Ni phase
                youngs_modulus = np.random.normal(200.0, 15)  # GPa
                hardness = np.random.normal(2.5, 0.5)  # GPa
                phase = 'Ni'
            
            # Creep compliance calculation
            creep_compliance = 1 / youngs_modulus * (1 + np.random.normal(0.1, 0.02))
            
            nano_indentation.append({
                'point_id': i,
                'position': [np.random.random(), np.random.random()],
                'youngs_modulus': youngs_modulus,
                'hardness': hardness,
                'creep_compliance': creep_compliance,
                'phase': phase,
                'indentation_depth': np.random.uniform(0.1, 2.0),  # μm
                'load': np.random.uniform(1, 10)  # mN
            })
        
        return {
            'sem_images': sem_images,
            'eds_scans': eds_scans,
            'nano_indentation': nano_indentation
        }
    
    def save_data(self, dic_data, xrd_data, post_mortem_data):
        """Save all generated data to files"""
        print("Saving data to files...")
        
        # Create output directory
        os.makedirs('/workspace/sofc_experimental_data', exist_ok=True)
        
        # Save DIC data
        with open('/workspace/sofc_experimental_data/dic_data.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            dic_data_serializable = self._convert_numpy_to_list(dic_data)
            json.dump(dic_data_serializable, f, indent=2, default=str)
        
        # Save XRD data
        with open('/workspace/sofc_experimental_data/xrd_data.json', 'w') as f:
            xrd_data_serializable = self._convert_numpy_to_list(xrd_data)
            json.dump(xrd_data_serializable, f, indent=2, default=str)
        
        # Save post-mortem data
        with open('/workspace/sofc_experimental_data/post_mortem_data.json', 'w') as f:
            post_mortem_data_serializable = self._convert_numpy_to_list(post_mortem_data)
            json.dump(post_mortem_data_serializable, f, indent=2, default=str)
        
        # Create summary CSV files
        self._create_summary_csvs(dic_data, xrd_data, post_mortem_data)
        
        print("Data saved to /workspace/sofc_experimental_data/")
    
    def _create_summary_csvs(self, dic_data, xrd_data, post_mortem_data):
        """Create summary CSV files for easy analysis"""
        
        # DIC summary
        dic_summary = []
        for data in dic_data['sintering']:
            dic_summary.append({
                'temperature': data['temperature'],
                'time': data['time'],
                'max_strain': data['max_strain'],
                'mean_strain': data['mean_strain'],
                'num_hotspots': len(data['hotspot_locations'])
            })
        
        pd.DataFrame(dic_summary).to_csv('/workspace/sofc_experimental_data/dic_sintering_summary.csv', index=False)
        
        # XRD summary
        xrd_summary = []
        for data in xrd_data['residual_stresses']:
            xrd_summary.append({
                'position': data['position'],
                'stress': data['stress'],
                'layer': data['layer'],
                'depth': data['depth']
            })
        
        pd.DataFrame(xrd_summary).to_csv('/workspace/sofc_experimental_data/xrd_residual_stress_summary.csv', index=False)
        
        # Post-mortem summary
        post_mortem_summary = []
        for data in post_mortem_data['nano_indentation']:
            post_mortem_summary.append({
                'point_id': data['point_id'],
                'youngs_modulus': data['youngs_modulus'],
                'hardness': data['hardness'],
                'creep_compliance': data['creep_compliance'],
                'phase': data['phase']
            })
        
        pd.DataFrame(post_mortem_summary).to_csv('/workspace/sofc_experimental_data/nano_indentation_summary.csv', index=False)
    
    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj

def main():
    """Main function to generate all experimental data"""
    print("🧪 SOFC Experimental Measurement Dataset Generator")
    print("=" * 60)
    
    generator = SOFCDataGenerator()
    
    # Generate all datasets
    print("\n1. Generating DIC data...")
    dic_data = generator.generate_dic_data()
    
    print("\n2. Generating XRD data...")
    xrd_data = generator.generate_xrd_data()
    
    print("\n3. Generating post-mortem analysis data...")
    post_mortem_data = generator.generate_post_mortem_data()
    
    # Save all data
    print("\n4. Saving data to files...")
    generator.save_data(dic_data, xrd_data, post_mortem_data)
    
    print("\n✅ All experimental datasets generated successfully!")
    print("\nGenerated datasets:")
    print("- Digital Image Correlation (DIC) data")
    print("- Synchrotron X-ray Diffraction (XRD) data")
    print("- Post-mortem analysis data (SEM, EDS, nano-indentation)")
    print("\nData saved to: /workspace/sofc_experimental_data/")

if __name__ == "__main__":
    main()