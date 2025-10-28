"""
SOFC Experimental Data Generator
Generates realistic synthetic data for:
1. Digital Image Correlation (DIC)
2. Synchrotron X-ray Diffraction (XRD)
3. Post-Mortem Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from scipy import ndimage
from scipy.interpolate import interp2d
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SOFCDataGenerator:
    def __init__(self, output_dir='sofc_experimental_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_dic_data(self):
        """Generate Digital Image Correlation data"""
        print("Generating DIC data...")
        dic_dir = os.path.join(self.output_dir, 'dic_data')
        os.makedirs(dic_dir, exist_ok=True)
        
        # 1. Sintering data (1200-1500°C)
        self._generate_sintering_dic(dic_dir)
        
        # 2. Thermal cycling data (ΔT = 400°C)
        self._generate_thermal_cycling_dic(dic_dir)
        
        # 3. Startup/shutdown cycles
        self._generate_startup_shutdown_dic(dic_dir)
        
        # 4. Speckle pattern images
        self._generate_speckle_patterns(dic_dir)
        
        # 5. Lagrangian strain tensor outputs
        self._generate_lagrangian_strain_tensors(dic_dir)
        
        # 6. Localized strain hotspots
        self._generate_strain_hotspots(dic_dir)
        
        print(f"DIC data saved to {dic_dir}")
        
    def _generate_sintering_dic(self, dic_dir):
        """Generate DIC data during sintering process"""
        temperatures = np.linspace(1200, 1500, 100)  # °C
        time_minutes = np.linspace(0, 180, 100)  # 3 hours
        
        # Create strain evolution during sintering
        data = []
        for i, (temp, time) in enumerate(zip(temperatures, time_minutes)):
            # Strain increases with temperature and time
            base_strain = (temp - 1200) / 300 * 0.015  # Up to 1.5% strain
            time_effect = np.exp(-time / 60) * 0.005  # Time-dependent relaxation
            
            # Different regions (anode, electrolyte, cathode)
            regions = ['anode', 'electrolyte', 'cathode', 'interface_ae', 'interface_ec']
            
            for region in regions:
                if 'interface' in region:
                    strain_multiplier = 1.5  # Interfaces have higher strain
                elif region == 'electrolyte':
                    strain_multiplier = 0.8  # YSZ is stiffer
                else:
                    strain_multiplier = 1.0
                
                exx = (base_strain + time_effect) * strain_multiplier + np.random.normal(0, 0.0002)
                eyy = (base_strain + time_effect) * strain_multiplier * 0.9 + np.random.normal(0, 0.0002)
                exy = np.random.normal(0, 0.0001)  # Shear strain
                
                data.append({
                    'timestamp': f"{int(time)}min_{int(temp)}C",
                    'time_min': time,
                    'temperature_C': temp,
                    'region': region,
                    'strain_xx': exx,
                    'strain_yy': eyy,
                    'strain_xy': exy,
                    'von_mises_strain': np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)
                })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(dic_dir, 'sintering_strain_data.csv'), index=False)
        
        # Create 2D strain map
        self._create_strain_map(df[df['time_min'] == 180], 
                                os.path.join(dic_dir, 'sintering_final_strain_map.png'))
        
    def _generate_thermal_cycling_dic(self, dic_dir):
        """Generate DIC data during thermal cycling"""
        n_cycles = 10
        points_per_cycle = 50
        total_points = n_cycles * points_per_cycle
        
        time_hours = np.linspace(0, 100, total_points)
        
        data = []
        for cycle in range(n_cycles):
            for i in range(points_per_cycle):
                idx = cycle * points_per_cycle + i
                time = time_hours[idx]
                
                # Temperature cycling: 800°C ± 400°C
                phase = 2 * np.pi * i / points_per_cycle
                temp = 800 + 400 * np.sin(phase)
                
                # Strain follows temperature with hysteresis
                thermal_strain = (temp - 400) / 1200 * 0.012
                
                # Accumulating damage with cycles
                damage_factor = 1 + cycle * 0.05
                
                regions = ['anode', 'electrolyte', 'cathode', 'interface_ae', 'interface_ec']
                
                for region in regions:
                    if 'interface' in region:
                        strain_multiplier = 1.8 * damage_factor
                    elif region == 'electrolyte':
                        strain_multiplier = 0.7
                    else:
                        strain_multiplier = 1.0 * damage_factor
                    
                    exx = thermal_strain * strain_multiplier + np.random.normal(0, 0.0003)
                    eyy = thermal_strain * strain_multiplier * 0.85 + np.random.normal(0, 0.0003)
                    exy = np.random.normal(0, 0.00015)
                    
                    data.append({
                        'cycle': cycle + 1,
                        'time_hours': time,
                        'temperature_C': temp,
                        'region': region,
                        'strain_xx': exx,
                        'strain_yy': eyy,
                        'strain_xy': exy,
                        'von_mises_strain': np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2),
                        'delta_T': 400
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(dic_dir, 'thermal_cycling_strain_data.csv'), index=False)
        
        # Plot strain evolution
        self._plot_thermal_cycling(df, dic_dir)
        
    def _generate_startup_shutdown_dic(self, dic_dir):
        """Generate DIC data for startup/shutdown cycles"""
        n_cycles = 5
        data = []
        
        for cycle in range(n_cycles):
            # Startup phase (room temp to 800°C in 2 hours)
            startup_time = np.linspace(0, 2, 50)
            startup_temp = 25 + (800 - 25) * (1 - np.exp(-startup_time / 0.5))
            
            for time, temp in zip(startup_time, startup_temp):
                self._add_cycle_data(data, cycle, 'startup', time, temp, cycle * 4)
            
            # Operation phase (800°C for 6 hours)
            operation_time = np.linspace(2, 8, 50)
            operation_temp = np.ones_like(operation_time) * 800
            
            for time, temp in zip(operation_time, operation_temp):
                self._add_cycle_data(data, cycle, 'operation', time, temp, cycle * 4 + 2)
            
            # Shutdown phase (800°C to room temp in 2 hours)
            shutdown_time = np.linspace(8, 10, 50)
            shutdown_temp = 800 - (800 - 25) * (1 - np.exp(-(shutdown_time - 8) / 0.5))
            
            for time, temp in zip(shutdown_time, shutdown_temp):
                self._add_cycle_data(data, cycle, 'shutdown', time, temp, cycle * 4 + 8)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(dic_dir, 'startup_shutdown_cycles.csv'), index=False)
        
    def _add_cycle_data(self, data, cycle, phase, time, temp, base_time):
        """Helper function to add cycle data"""
        regions = ['anode', 'electrolyte', 'cathode', 'interface_ae', 'interface_ec']
        
        # Calculate thermal strain
        thermal_strain = (temp - 25) / 775 * 0.010
        
        # Damage accumulation
        damage = 1 + cycle * 0.08
        
        for region in regions:
            if 'interface' in region:
                multiplier = 2.0 * damage
            elif region == 'electrolyte':
                multiplier = 0.65
            else:
                multiplier = 1.0 * damage
            
            exx = thermal_strain * multiplier + np.random.normal(0, 0.0002)
            eyy = thermal_strain * multiplier * 0.88 + np.random.normal(0, 0.0002)
            exy = np.random.normal(0, 0.0001)
            
            data.append({
                'cycle': cycle + 1,
                'phase': phase,
                'time_hours': base_time + time,
                'cycle_time_hours': time,
                'temperature_C': temp,
                'region': region,
                'strain_xx': exx,
                'strain_yy': eyy,
                'strain_xy': exy,
                'von_mises_strain': np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)
            })
    
    def _generate_speckle_patterns(self, dic_dir):
        """Generate synthetic speckle pattern images"""
        speckle_dir = os.path.join(dic_dir, 'speckle_patterns')
        os.makedirs(speckle_dir, exist_ok=True)
        
        # Generate reference and deformed speckle patterns
        timestamps = ['t0_25C', 't1_400C', 't2_800C', 't3_1200C', 't4_1500C']
        
        for i, timestamp in enumerate(timestamps):
            # Create speckle pattern
            img_size = (512, 512)
            speckle = self._create_speckle_image(img_size, deformation=i*0.002)
            
            # Save image
            plt.figure(figsize=(8, 8))
            plt.imshow(speckle, cmap='gray')
            plt.title(f'Speckle Pattern - {timestamp}')
            plt.colorbar(label='Intensity')
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            plt.tight_layout()
            plt.savefig(os.path.join(speckle_dir, f'speckle_{timestamp}.png'), dpi=150)
            plt.close()
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'image_size': img_size,
                'pixel_size_um': 2.5,
                'camera': 'Allied Vision Manta G-504B',
                'lens': 'Computar 50mm f/2.8',
                'exposure_ms': 15,
                'acquisition_rate_hz': 1
            }
            
            with open(os.path.join(speckle_dir, f'metadata_{timestamp}.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _create_speckle_image(self, img_size, deformation=0):
        """Create synthetic speckle pattern with optional deformation"""
        # Random speckle pattern
        speckle = np.random.rand(*img_size)
        
        # Apply Gaussian blur to create realistic speckles
        speckle = ndimage.gaussian_filter(speckle, sigma=2)
        
        # Create threshold pattern
        threshold = 0.7
        speckle = (speckle > threshold).astype(float)
        
        # Apply deformation field
        if deformation > 0:
            x, y = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            
            # Simulate strain-induced displacement
            u = deformation * x * 0.5  # Horizontal displacement
            v = deformation * y * 0.3  # Vertical displacement
            
            # Apply displacement
            coords = np.array([y + v, x + u])
            speckle = ndimage.map_coordinates(speckle, coords, order=1, mode='nearest')
        
        return speckle
    
    def _generate_lagrangian_strain_tensors(self, dic_dir):
        """Generate Lagrangian strain tensor outputs (Vic-3D format)"""
        tensor_dir = os.path.join(dic_dir, 'lagrangian_tensors')
        os.makedirs(tensor_dir, exist_ok=True)
        
        # Create spatial grid (mm)
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 8, 80)
        X, Y = np.meshgrid(x, y)
        
        # Generate strain field at different temperatures
        temperatures = [25, 400, 800, 1200, 1500]
        
        for temp in temperatures:
            # Calculate position-dependent strain
            # Higher strain near interfaces and edges
            
            # Base thermal strain
            thermal_strain = (temp - 25) / 1475 * 0.015
            
            # Create heterogeneous strain field
            # Simulate three layers: anode (0-3mm), electrolyte (3-5mm), cathode (5-8mm)
            exx = np.zeros_like(X)
            eyy = np.zeros_like(X)
            exy = np.zeros_like(X)
            
            for i in range(len(y)):
                y_pos = y[i]
                if y_pos < 3:  # Anode
                    base_strain = thermal_strain * 1.2
                elif y_pos < 5:  # Electrolyte
                    base_strain = thermal_strain * 0.8
                else:  # Cathode
                    base_strain = thermal_strain * 1.1
                
                # Add interface effects
                if 2.5 < y_pos < 3.5 or 4.5 < y_pos < 5.5:
                    interface_strain = 0.008  # High strain at interfaces
                else:
                    interface_strain = 0
                
                # Add edge effects
                edge_factor = 1 + 0.3 * (np.exp(-x/2) + np.exp(-(10-x)/2))
                
                exx[i, :] = (base_strain + interface_strain) * edge_factor + np.random.normal(0, 0.0001, len(x))
                eyy[i, :] = (base_strain + interface_strain) * edge_factor * 0.9 + np.random.normal(0, 0.0001, len(x))
                exy[i, :] = np.random.normal(0, 0.00005, len(x))
            
            # Save tensor components
            data = {
                'x_mm': X.flatten(),
                'y_mm': Y.flatten(),
                'E_xx': exx.flatten(),
                'E_yy': eyy.flatten(),
                'E_xy': exy.flatten(),
                'E_zz': (exx.flatten() + eyy.flatten()) * (-0.3),  # Poisson effect
                'von_mises': np.sqrt(exx.flatten()**2 + eyy.flatten()**2 - 
                                    exx.flatten()*eyy.flatten() + 3*exy.flatten()**2)
            }
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(tensor_dir, f'lagrangian_strain_{temp}C.csv'), index=False)
            
            # Create visualization
            self._plot_strain_tensor_field(X, Y, exx, eyy, exy, temp, tensor_dir)
    
    def _generate_strain_hotspots(self, dic_dir):
        """Identify and document localized strain hotspots (>1.0%)"""
        hotspot_dir = os.path.join(dic_dir, 'strain_hotspots')
        os.makedirs(hotspot_dir, exist_ok=True)
        
        # Load Lagrangian tensor data
        tensor_dir = os.path.join(dic_dir, 'lagrangian_tensors')
        
        hotspot_summary = []
        
        for temp in [25, 400, 800, 1200, 1500]:
            df = pd.read_csv(os.path.join(tensor_dir, f'lagrangian_strain_{temp}C.csv'))
            
            # Identify hotspots (von Mises strain > 1.0%)
            hotspots = df[df['von_mises'] > 0.010]
            
            if len(hotspots) > 0:
                # Cluster hotspots
                for idx, row in hotspots.iterrows():
                    # Determine region
                    y_pos = row['y_mm']
                    if y_pos < 3:
                        region = 'anode'
                    elif y_pos < 3.5:
                        region = 'anode-electrolyte interface'
                    elif y_pos < 5:
                        region = 'electrolyte'
                    elif y_pos < 5.5:
                        region = 'electrolyte-cathode interface'
                    else:
                        region = 'cathode'
                    
                    hotspot_summary.append({
                        'temperature_C': temp,
                        'x_mm': row['x_mm'],
                        'y_mm': row['y_mm'],
                        'region': region,
                        'von_mises_strain': row['von_mises'],
                        'E_xx': row['E_xx'],
                        'E_yy': row['E_yy'],
                        'E_xy': row['E_xy'],
                        'severity': 'critical' if row['von_mises'] > 0.015 else 'high'
                    })
        
        hotspot_df = pd.DataFrame(hotspot_summary)
        hotspot_df.to_csv(os.path.join(hotspot_dir, 'strain_hotspot_catalog.csv'), index=False)
        
        # Create visualization
        self._plot_hotspot_distribution(hotspot_df, hotspot_dir)
    
    def generate_xrd_data(self):
        """Generate Synchrotron X-ray Diffraction data"""
        print("Generating XRD data...")
        xrd_dir = os.path.join(self.output_dir, 'xrd_data')
        os.makedirs(xrd_dir, exist_ok=True)
        
        # 1. Residual stress profiles
        self._generate_residual_stress_profiles(xrd_dir)
        
        # 2. Lattice strain measurements
        self._generate_lattice_strain(xrd_dir)
        
        # 3. Peak shift data for sin²ψ method
        self._generate_sin2psi_data(xrd_dir)
        
        # 4. Microcrack initiation thresholds
        self._generate_microcrack_threshold_data(xrd_dir)
        
        print(f"XRD data saved to {xrd_dir}")
    
    def _generate_residual_stress_profiles(self, xrd_dir):
        """Generate residual stress profiles across SOFC cross-section"""
        # Scan across thickness (μm)
        depth = np.linspace(0, 800, 200)  # 0-800 μm total thickness
        
        data = []
        
        # Different conditions
        conditions = ['as_sintered', 'after_thermal_cycling', 'after_100h_operation']
        
        for condition in conditions:
            if condition == 'as_sintered':
                stress_factor = 1.0
            elif condition == 'after_thermal_cycling':
                stress_factor = 1.5
            else:
                stress_factor = 1.8
            
            for d in depth:
                # Determine layer
                if d < 300:  # Anode (Ni-YSZ)
                    base_stress = -150 * stress_factor  # Compressive (MPa)
                    modulus = 109.8  # GPa
                    phase = 'Ni-YSZ'
                elif d < 320:  # Interface
                    base_stress = 200 * stress_factor  # Tensile (MPa)
                    modulus = 147.25  # Average
                    phase = 'Interface'
                elif d < 500:  # Electrolyte (YSZ)
                    base_stress = 100 * stress_factor  # Tensile (MPa)
                    modulus = 184.7  # GPa
                    phase = 'YSZ'
                elif d < 520:  # Interface
                    base_stress = 180 * stress_factor  # Tensile (MPa)
                    modulus = 152.35  # Average
                    phase = 'Interface'
                else:  # Cathode (LSM or LSCF)
                    base_stress = -80 * stress_factor  # Compressive (MPa)
                    modulus = 120.0  # GPa
                    phase = 'LSM'
                
                # Add noise and gradients
                stress_gradient = np.random.normal(0, 10)
                sigma_11 = base_stress + stress_gradient
                sigma_22 = base_stress * 0.85 + stress_gradient * 0.7
                sigma_33 = base_stress * 0.6 + stress_gradient * 0.5
                
                # Calculate hydrostatic and von Mises stress
                hydrostatic = (sigma_11 + sigma_22 + sigma_33) / 3
                von_mises = np.sqrt(0.5 * ((sigma_11 - sigma_22)**2 + 
                                           (sigma_22 - sigma_33)**2 + 
                                           (sigma_33 - sigma_11)**2))
                
                data.append({
                    'condition': condition,
                    'depth_um': d,
                    'phase': phase,
                    'sigma_11_MPa': sigma_11,
                    'sigma_22_MPa': sigma_22,
                    'sigma_33_MPa': sigma_33,
                    'hydrostatic_stress_MPa': hydrostatic,
                    'von_mises_stress_MPa': von_mises,
                    'youngs_modulus_GPa': modulus
                })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(xrd_dir, 'residual_stress_profiles.csv'), index=False)
        
        # Plot profiles
        self._plot_stress_profiles(df, xrd_dir)
    
    def _generate_lattice_strain(self, xrd_dir):
        """Generate lattice strain measurements under thermal load"""
        temperatures = np.linspace(25, 1500, 50)
        
        data = []
        
        phases = {
            'YSZ': {'a0': 5.14, 'alpha': 10.5e-6},  # Å, thermal expansion
            'Ni': {'a0': 3.52, 'alpha': 13.4e-6},
            'LSM': {'a0': 5.50, 'alpha': 11.2e-6}
        }
        
        for temp in temperatures:
            for phase_name, props in phases.items():
                # Calculate thermal expansion
                delta_T = temp - 25
                thermal_strain = props['alpha'] * delta_T
                
                # Calculate lattice parameter
                a = props['a0'] * (1 + thermal_strain)
                
                # Add stress-induced lattice strain
                if phase_name == 'YSZ':
                    stress_strain = 100e-6 * (temp / 1500)  # Tensile
                elif phase_name == 'Ni':
                    stress_strain = -80e-6 * (temp / 1500)  # Compressive
                else:
                    stress_strain = -50e-6 * (temp / 1500)
                
                total_strain = thermal_strain + stress_strain
                a_stressed = props['a0'] * (1 + total_strain)
                
                # Add measurement noise
                a_measured = a_stressed + np.random.normal(0, 0.001)
                
                # Calculate elastic strain
                elastic_strain = (a_measured - a) / a
                
                data.append({
                    'temperature_C': temp,
                    'phase': phase_name,
                    'lattice_parameter_A': a_measured,
                    'thermal_strain': thermal_strain,
                    'elastic_strain': elastic_strain,
                    'total_strain': total_strain,
                    'reference_a0_A': props['a0']
                })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(xrd_dir, 'lattice_strain_vs_temperature.csv'), index=False)
        
        # Plot
        self._plot_lattice_strain(df, xrd_dir)
    
    def _generate_sin2psi_data(self, xrd_dir):
        """Generate peak shift data for sin²ψ method stress calculation"""
        # sin²ψ method: d_ψ = d_0 + (d_0 * σ * (1+ν) / E) * sin²ψ
        
        psi_angles = np.linspace(-45, 45, 15)  # degrees
        sin2psi = np.sin(np.deg2rad(psi_angles))**2
        
        data = []
        
        # Different measurement locations
        locations = [
            ('anode', 150, -150),  # depth (μm), stress (MPa)
            ('interface_ae', 310, 200),
            ('electrolyte', 410, 100),
            ('interface_ec', 510, 180),
            ('cathode', 650, -80)
        ]
        
        for location, depth, sigma in locations:
            # Material properties
            if 'anode' in location:
                E = 109.8e3  # MPa
                nu = 0.30
                d_0 = 2.410  # Å (Ni 111 peak)
                hkl = '(111)'
            elif 'electrolyte' in location or 'interface' in location:
                E = 184.7e3  # MPa
                nu = 0.25
                d_0 = 2.965  # Å (YSZ 111 peak)
                hkl = '(111)'
            else:  # cathode
                E = 120.0e3  # MPa
                nu = 0.28
                d_0 = 2.750  # Å (LSM peak)
                hkl = '(110)'
            
            for psi, s2p in zip(psi_angles, sin2psi):
                # Calculate d-spacing with stress
                d_psi = d_0 * (1 + (sigma * (1 + nu) / E) * s2p)
                
                # Add measurement noise
                d_measured = d_psi + np.random.normal(0, 0.0005)
                
                # Calculate 2θ (assuming Cu Kα, λ = 1.5406 Å)
                lambda_xray = 1.5406
                theta = np.arcsin(lambda_xray / (2 * d_measured))
                two_theta = 2 * np.rad2deg(theta)
                
                data.append({
                    'location': location,
                    'depth_um': depth,
                    'psi_deg': psi,
                    'sin2psi': s2p,
                    'd_spacing_A': d_measured,
                    '2theta_deg': two_theta,
                    'hkl': hkl,
                    'applied_stress_MPa': sigma,
                    'E_MPa': E,
                    'nu': nu
                })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(xrd_dir, 'sin2psi_stress_analysis.csv'), index=False)
        
        # Plot sin²ψ plots
        self._plot_sin2psi(df, xrd_dir)
    
    def _generate_microcrack_threshold_data(self, xrd_dir):
        """Generate microcrack initiation threshold data"""
        # Critical strain for microcrack initiation: ε_cr > 0.02
        
        data = []
        
        # Test specimens with increasing strain levels
        n_specimens = 30
        applied_strains = np.linspace(0.005, 0.035, n_specimens)
        
        for i, strain in enumerate(applied_strains):
            specimen_id = f'SOFC_{i+1:03d}'
            
            # Determine if cracking occurred
            # Probabilistic approach with threshold around 0.02
            crack_probability = 1 / (1 + np.exp(-100 * (strain - 0.020)))
            cracked = np.random.random() < crack_probability
            
            # If cracked, measure crack characteristics
            if cracked:
                crack_density = (strain - 0.020) * 50 + np.random.normal(0, 0.5)  # cracks/mm²
                crack_length = (strain - 0.020) * 100 + np.random.normal(5, 2)  # μm
                crack_opening = (strain - 0.020) * 10 + np.random.normal(0.5, 0.2)  # μm
            else:
                crack_density = 0
                crack_length = 0
                crack_opening = 0
            
            # XRD peak broadening indicates damage
            peak_fwhm = 0.15 + strain * 5 + np.random.normal(0, 0.02)  # degrees
            
            # Crystallite size decreases with damage (Scherrer equation)
            crystallite_size = 80 / (1 + strain * 20) + np.random.normal(0, 2)  # nm
            
            data.append({
                'specimen_id': specimen_id,
                'applied_strain': strain,
                'cracked': cracked,
                'crack_density_per_mm2': crack_density,
                'average_crack_length_um': crack_length,
                'crack_opening_um': crack_opening,
                'xrd_peak_fwhm_deg': peak_fwhm,
                'crystallite_size_nm': crystallite_size,
                'critical_threshold': strain > 0.020
            })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(xrd_dir, 'microcrack_threshold_data.csv'), index=False)
        
        # Plot threshold analysis
        self._plot_crack_threshold(df, xrd_dir)
    
    def generate_postmortem_data(self):
        """Generate post-mortem analysis data"""
        print("Generating post-mortem analysis data...")
        pm_dir = os.path.join(self.output_dir, 'postmortem_data')
        os.makedirs(pm_dir, exist_ok=True)
        
        # 1. SEM crack density data
        self._generate_sem_crack_data(pm_dir)
        
        # 2. EDS line scans
        self._generate_eds_line_scans(pm_dir)
        
        # 3. Nano-indentation data
        self._generate_nanoindentation_data(pm_dir)
        
        print(f"Post-mortem data saved to {pm_dir}")
    
    def _generate_sem_crack_data(self, pm_dir):
        """Generate SEM crack density quantification"""
        sem_dir = os.path.join(pm_dir, 'sem_analysis')
        os.makedirs(sem_dir, exist_ok=True)
        
        # Different specimens with varying operation hours
        specimens = [
            ('pristine', 0, 0.0),
            ('10h_operation', 10, 0.5),
            ('50h_operation', 50, 2.1),
            ('100h_operation', 100, 4.8),
            ('200h_operation', 200, 8.5),
            ('10_thermal_cycles', 40, 6.2)
        ]
        
        data = []
        
        for spec_name, hours, base_density in specimens:
            # Multiple ROIs (regions of interest)
            for roi in range(1, 11):
                # Different regions
                for region in ['anode', 'electrolyte', 'cathode', 'interface_ae', 'interface_ec']:
                    if 'interface' in region:
                        density_factor = 2.5
                    elif region == 'electrolyte':
                        density_factor = 0.7
                    else:
                        density_factor = 1.0
                    
                    crack_density = base_density * density_factor + np.random.normal(0, 0.3)
                    crack_density = max(0, crack_density)  # No negative cracks
                    
                    # Crack characteristics
                    if crack_density > 0:
                        avg_length = 10 + crack_density * 5 + np.random.normal(0, 2)
                        avg_width = 0.5 + crack_density * 0.2 + np.random.normal(0, 0.1)
                        max_length = avg_length * 1.8 + np.random.normal(0, 3)
                    else:
                        avg_length = 0
                        avg_width = 0
                        max_length = 0
                    
                    data.append({
                        'specimen': spec_name,
                        'operation_hours': hours,
                        'roi_id': roi,
                        'region': region,
                        'crack_density_per_mm2': crack_density,
                        'avg_crack_length_um': avg_length,
                        'avg_crack_width_um': avg_width,
                        'max_crack_length_um': max_length,
                        'magnification': '5000x',
                        'accelerating_voltage_kV': 15
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(sem_dir, 'crack_density_analysis.csv'), index=False)
        
        # Summary statistics
        summary = df.groupby(['specimen', 'region']).agg({
            'crack_density_per_mm2': ['mean', 'std'],
            'avg_crack_length_um': ['mean', 'std']
        }).round(2)
        summary.to_csv(os.path.join(sem_dir, 'crack_density_summary.csv'))
        
        # Create visualizations
        self._plot_crack_density(df, sem_dir)
    
    def _generate_eds_line_scans(self, pm_dir):
        """Generate EDS line scans for elemental composition"""
        eds_dir = os.path.join(pm_dir, 'eds_analysis')
        os.makedirs(eds_dir, exist_ok=True)
        
        # Line scan across SOFC cross-section
        distance = np.linspace(0, 800, 400)  # μm
        
        data = []
        
        for d in distance:
            # Define composition profiles
            if d < 280:  # Anode (Ni-YSZ)
                Ni = 40 + np.random.normal(0, 2)
                Zr = 35 + np.random.normal(0, 2)
                Y = 5 + np.random.normal(0, 0.5)
                O = 20 + np.random.normal(0, 1.5)
                La = 0
                Sr = 0
                Mn = 0
            elif d < 300:  # Interface region (mixing)
                transition = (d - 280) / 20
                Ni = 40 * (1 - transition) + np.random.normal(0, 3)
                Zr = 35 + np.random.normal(0, 2)
                Y = 5 + np.random.normal(0, 0.5)
                O = 20 + 40 * transition + np.random.normal(0, 2)
                La = 0
                Sr = 0
                Mn = 0
            elif d < 480:  # Electrolyte (YSZ)
                Ni = 0
                Zr = 53 + np.random.normal(0, 1.5)
                Y = 7 + np.random.normal(0, 0.3)
                O = 40 + np.random.normal(0, 1.5)
                La = 0
                Sr = 0
                Mn = 0
            elif d < 500:  # Interface region
                transition = (d - 480) / 20
                Ni = 0
                Zr = 53 * (1 - transition) + np.random.normal(0, 2)
                Y = 7 * (1 - transition) + np.random.normal(0, 0.5)
                O = 40 + np.random.normal(0, 2)
                La = 15 * transition + np.random.normal(0, 1)
                Sr = 3 * transition + np.random.normal(0, 0.3)
                Mn = 7 * transition + np.random.normal(0, 0.7)
            else:  # Cathode (LSM)
                Ni = 0
                Zr = 0
                Y = 0
                O = 40 + np.random.normal(0, 1.5)
                La = 15 + np.random.normal(0, 1)
                Sr = 3 + np.random.normal(0, 0.3)
                Mn = 7 + np.random.normal(0, 0.7)
            
            # Normalize to 100%
            total = Ni + Zr + Y + O + La + Sr + Mn
            
            data.append({
                'distance_um': d,
                'Ni_wt%': Ni / total * 100,
                'Zr_wt%': Zr / total * 100,
                'Y_wt%': Y / total * 100,
                'O_wt%': O / total * 100,
                'La_wt%': La / total * 100,
                'Sr_wt%': Sr / total * 100,
                'Mn_wt%': Mn / total * 100
            })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(eds_dir, 'eds_line_scan_cross_section.csv'), index=False)
        
        # Create EDS line scan plot
        self._plot_eds_line_scan(df, eds_dir)
        
        # Generate point analysis data
        self._generate_eds_point_analysis(eds_dir)
    
    def _generate_eds_point_analysis(self, eds_dir):
        """Generate EDS point analysis data"""
        points_data = []
        
        # Define point locations and expected compositions
        points = [
            ('Anode_center', 150, {'Ni': 40, 'Zr': 35, 'Y': 5, 'O': 20}),
            ('Anode_interface', 290, {'Ni': 38, 'Zr': 36, 'Y': 5, 'O': 21}),
            ('Electrolyte_center', 400, {'Zr': 53, 'Y': 7, 'O': 40}),
            ('Cathode_interface', 510, {'Zr': 5, 'Y': 1, 'O': 42, 'La': 15, 'Sr': 3, 'Mn': 7}),
            ('Cathode_center', 650, {'O': 40, 'La': 15, 'Sr': 3, 'Mn': 7})
        ]
        
        for location, depth, composition in points:
            for rep in range(1, 6):  # 5 replicate measurements
                point_data = {'location': location, 'depth_um': depth, 'replicate': rep}
                
                for element, nominal in composition.items():
                    measured = nominal + np.random.normal(0, nominal * 0.05)
                    point_data[f'{element}_wt%'] = measured
                
                points_data.append(point_data)
        
        df_points = pd.DataFrame(points_data)
        df_points.to_csv(os.path.join(eds_dir, 'eds_point_analysis.csv'), index=False)
    
    def _generate_nanoindentation_data(self, pm_dir):
        """Generate nano-indentation data"""
        nano_dir = os.path.join(pm_dir, 'nanoindentation')
        os.makedirs(nano_dir, exist_ok=True)
        
        # Spatial grid for indentation mapping
        x_positions = np.linspace(0, 10, 50)  # mm
        y_positions = np.linspace(0, 0.8, 40)  # mm (thickness)
        
        data = []
        
        for x in x_positions:
            for y in y_positions:
                # Determine layer
                y_um = y * 1000
                
                if y_um < 300:  # Anode (Ni-YSZ)
                    E_base = 109.8  # GPa
                    H_base = 5.8  # GPa
                    creep_base = 0.08
                elif y_um < 500:  # Electrolyte (YSZ)
                    E_base = 184.7  # GPa
                    H_base = 12.5  # GPa
                    creep_base = 0.02
                else:  # Cathode (LSM)
                    E_base = 120.0  # GPa
                    H_base = 6.5  # GPa
                    creep_base = 0.06
                
                # Add spatial variation and measurement noise
                E = E_base * (1 + 0.1 * np.sin(x * 2) * np.sin(y * 50)) + np.random.normal(0, E_base * 0.03)
                H = H_base * (1 + 0.08 * np.sin(x * 2.5) * np.sin(y * 45)) + np.random.normal(0, H_base * 0.04)
                creep = creep_base + np.random.normal(0, creep_base * 0.2)
                
                # Interface regions have reduced properties
                if 290 < y_um < 310 or 490 < y_um < 510:
                    E *= 0.85
                    H *= 0.80
                    creep *= 1.5
                
                data.append({
                    'x_mm': x,
                    'y_mm': y,
                    'depth_um': y_um,
                    'youngs_modulus_GPa': E,
                    'hardness_GPa': H,
                    'creep_compliance': creep,
                    'max_load_mN': 10,
                    'loading_rate_mN_s': 1,
                    'hold_time_s': 10
                })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(nano_dir, 'nanoindentation_map.csv'), index=False)
        
        # Generate load-displacement curves
        self._generate_load_displacement_curves(nano_dir)
        
        # Create property maps
        self._plot_nanoindentation_maps(df, nano_dir)
    
    def _generate_load_displacement_curves(self, nano_dir):
        """Generate individual load-displacement curves"""
        curves_dir = os.path.join(nano_dir, 'load_displacement_curves')
        os.makedirs(curves_dir, exist_ok=True)
        
        # Generate curves for representative locations
        locations = [
            ('anode', 109.8, 5.8),
            ('electrolyte', 184.7, 12.5),
            ('cathode', 120.0, 6.5),
            ('interface', 95.0, 4.5)
        ]
        
        for location, E, H in locations:
            # Loading phase
            load_time = np.linspace(0, 10, 100)  # seconds
            max_load = 10  # mN
            load = max_load * (load_time / 10)**2
            
            # Calculate displacement using Oliver-Pharr method
            # h = (P / (2*E*tan(α)))^(2/3) for Berkovich indenter
            displacement = (load / (2 * E * 0.142))**(2/3) * 1000  # nm
            
            # Hold phase
            hold_time = np.linspace(10, 20, 50)
            hold_load = np.ones_like(hold_time) * max_load
            
            # Creep during hold
            creep_disp = displacement[-1] + (hold_time - 10) * 5 * (1 - np.exp(-(hold_time - 10) / 3))
            
            # Unloading phase
            unload_time = np.linspace(20, 30, 100)
            unload_load = max_load * (1 - (unload_time - 20) / 10)**2
            
            # Unloading displacement (elastic recovery)
            unload_disp = displacement[-1] * (unload_load / max_load)**(3/2)
            
            # Combine phases
            time_all = np.concatenate([load_time, hold_time, unload_time])
            load_all = np.concatenate([load, hold_load, unload_load])
            disp_all = np.concatenate([displacement, creep_disp, unload_disp])
            
            # Add noise
            load_all += np.random.normal(0, 0.05, len(load_all))
            disp_all += np.random.normal(0, 2, len(disp_all))
            
            # Save curve
            curve_df = pd.DataFrame({
                'time_s': time_all,
                'load_mN': load_all,
                'displacement_nm': disp_all
            })
            curve_df.to_csv(os.path.join(curves_dir, f'curve_{location}.csv'), index=False)
            
            # Plot
            self._plot_load_displacement_curve(curve_df, location, curves_dir)
    
    # Plotting functions
    def _create_strain_map(self, df, output_path):
        """Create 2D strain map visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for region in df['region'].unique():
            region_data = df[df['region'] == region]
            ax.scatter(region_data.index, region_data['von_mises_strain'] * 100,
                      label=region, s=50, alpha=0.7)
        
        ax.set_xlabel('Spatial Position')
        ax.set_ylabel('Von Mises Strain (%)')
        ax.set_title('Final Strain Map after Sintering (1500°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def _plot_thermal_cycling(self, df, dic_dir):
        """Plot thermal cycling strain evolution"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Temperature profile
        cycle_1 = df[df['cycle'] == 1]
        temp_profile = cycle_1.groupby('time_hours')['temperature_C'].first()
        ax1.plot(temp_profile.index, temp_profile.values, 'r-', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Thermal Cycling Profile (ΔT = 400°C)')
        ax1.grid(True, alpha=0.3)
        
        # Strain evolution by region
        for region in ['anode', 'electrolyte', 'cathode', 'interface_ae']:
            region_data = df[df['region'] == region]
            ax2.plot(region_data['time_hours'], region_data['von_mises_strain'] * 100,
                    label=region, alpha=0.7)
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Von Mises Strain (%)')
        ax2.set_title('Strain Evolution during Thermal Cycling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dic_dir, 'thermal_cycling_analysis.png'), dpi=150)
        plt.close()
    
    def _plot_strain_tensor_field(self, X, Y, exx, eyy, exy, temp, tensor_dir):
        """Plot strain tensor field"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # E_xx
        im1 = axes[0, 0].contourf(X, Y, exx * 100, levels=20, cmap='RdYlBu_r')
        axes[0, 0].set_title(f'$E_{{xx}}$ at {temp}°C')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0, 0], label='Strain (%)')
        
        # E_yy
        im2 = axes[0, 1].contourf(X, Y, eyy * 100, levels=20, cmap='RdYlBu_r')
        axes[0, 1].set_title(f'$E_{{yy}}$ at {temp}°C')
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=axes[0, 1], label='Strain (%)')
        
        # E_xy
        im3 = axes[1, 0].contourf(X, Y, exy * 100, levels=20, cmap='RdYlBu_r')
        axes[1, 0].set_title(f'$E_{{xy}}$ (Shear) at {temp}°C')
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        plt.colorbar(im3, ax=axes[1, 0], label='Strain (%)')
        
        # Von Mises
        von_mises = np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)
        im4 = axes[1, 1].contourf(X, Y, von_mises * 100, levels=20, cmap='hot')
        axes[1, 1].set_title(f'Von Mises Strain at {temp}°C')
        axes[1, 1].set_xlabel('X (mm)')
        axes[1, 1].set_ylabel('Y (mm)')
        plt.colorbar(im4, ax=axes[1, 1], label='Strain (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(tensor_dir, f'strain_tensor_field_{temp}C.png'), dpi=150)
        plt.close()
    
    def _plot_hotspot_distribution(self, df, hotspot_dir):
        """Plot strain hotspot distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hotspot locations
        for temp in df['temperature_C'].unique():
            temp_data = df[df['temperature_C'] == temp]
            axes[0].scatter(temp_data['x_mm'], temp_data['y_mm'],
                          s=temp_data['von_mises_strain'] * 5000,
                          alpha=0.5, label=f'{int(temp)}°C')
        
        axes[0].set_xlabel('X Position (mm)')
        axes[0].set_ylabel('Y Position (mm)')
        axes[0].set_title('Strain Hotspot Locations (>1.0%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Hotspot count by region
        region_counts = df.groupby(['region', 'temperature_C']).size().unstack(fill_value=0)
        region_counts.plot(kind='bar', ax=axes[1])
        axes[1].set_xlabel('Region')
        axes[1].set_ylabel('Number of Hotspots')
        axes[1].set_title('Hotspot Distribution by Region')
        axes[1].legend(title='Temperature (°C)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(hotspot_dir, 'hotspot_analysis.png'), dpi=150)
        plt.close()
    
    def _plot_stress_profiles(self, df, xrd_dir):
        """Plot residual stress profiles"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            
            axes[0].plot(condition_data['depth_um'], condition_data['sigma_11_MPa'],
                        label=condition.replace('_', ' '), linewidth=2)
            axes[1].plot(condition_data['depth_um'], condition_data['hydrostatic_stress_MPa'],
                        label=condition.replace('_', ' '), linewidth=2)
            axes[2].plot(condition_data['depth_um'], condition_data['von_mises_stress_MPa'],
                        label=condition.replace('_', ' '), linewidth=2)
        
        axes[0].set_xlabel('Depth (μm)')
        axes[0].set_ylabel(r'$\sigma_{11}$ (MPa)')
        axes[0].set_title('Normal Stress Profile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        axes[1].set_xlabel('Depth (μm)')
        axes[1].set_ylabel('Hydrostatic Stress (MPa)')
        axes[1].set_title('Hydrostatic Stress Profile')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        axes[2].set_xlabel('Depth (μm)')
        axes[2].set_ylabel('Von Mises Stress (MPa)')
        axes[2].set_title('Von Mises Stress Profile')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(xrd_dir, 'residual_stress_profiles.png'), dpi=150)
        plt.close()
    
    def _plot_lattice_strain(self, df, xrd_dir):
        """Plot lattice strain vs temperature"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase]
            axes[0].plot(phase_data['temperature_C'], phase_data['lattice_parameter_A'],
                        'o-', label=phase, linewidth=2, markersize=4)
            axes[1].plot(phase_data['temperature_C'], phase_data['total_strain'] * 1e6,
                        'o-', label=phase, linewidth=2, markersize=4)
        
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Lattice Parameter (Å)')
        axes[0].set_title('Lattice Parameter vs Temperature')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].set_ylabel('Total Strain (με)')
        axes[1].set_title('Lattice Strain vs Temperature')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(xrd_dir, 'lattice_strain_analysis.png'), dpi=150)
        plt.close()
    
    def _plot_sin2psi(self, df, xrd_dir):
        """Plot sin²ψ stress analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, location in enumerate(df['location'].unique()):
            loc_data = df[df['location'] == location]
            
            axes[i].plot(loc_data['sin2psi'], loc_data['d_spacing_A'], 'o-', linewidth=2)
            
            # Linear fit
            coeffs = np.polyfit(loc_data['sin2psi'], loc_data['d_spacing_A'], 1)
            fit_line = np.polyval(coeffs, loc_data['sin2psi'])
            axes[i].plot(loc_data['sin2psi'], fit_line, 'r--', alpha=0.7, label='Linear fit')
            
            axes[i].set_xlabel('sin²ψ')
            axes[i].set_ylabel('d-spacing (Å)')
            axes[i].set_title(f'{location.replace("_", " ")}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Hide extra subplot
        if len(df['location'].unique()) < 6:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(xrd_dir, 'sin2psi_analysis.png'), dpi=150)
        plt.close()
    
    def _plot_crack_threshold(self, df, xrd_dir):
        """Plot microcrack threshold analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Crack occurrence vs strain
        axes[0].scatter(df['applied_strain'] * 100, df['cracked'].astype(int),
                       c=df['crack_density_per_mm2'], cmap='Reds', s=100, alpha=0.7)
        axes[0].axvline(x=2.0, color='r', linestyle='--', linewidth=2, label='ε_cr = 2.0%')
        axes[0].set_xlabel('Applied Strain (%)')
        axes[0].set_ylabel('Cracked (0=No, 1=Yes)')
        axes[0].set_title('Microcrack Initiation Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Crack density vs strain
        cracked_data = df[df['cracked']]
        axes[1].scatter(cracked_data['applied_strain'] * 100,
                       cracked_data['crack_density_per_mm2'],
                       s=100, alpha=0.7, c='red')
        axes[1].set_xlabel('Applied Strain (%)')
        axes[1].set_ylabel('Crack Density (cracks/mm²)')
        axes[1].set_title('Crack Density Evolution')
        axes[1].grid(True, alpha=0.3)
        
        # Peak broadening (damage indicator)
        axes[2].scatter(df['applied_strain'] * 100, df['xrd_peak_fwhm_deg'],
                       c=df['cracked'].astype(int), cmap='RdYlGn_r', s=100, alpha=0.7)
        axes[2].set_xlabel('Applied Strain (%)')
        axes[2].set_ylabel('XRD Peak FWHM (°)')
        axes[2].set_title('XRD Peak Broadening (Damage Indicator)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(xrd_dir, 'microcrack_threshold_analysis.png'), dpi=150)
        plt.close()
    
    def _plot_crack_density(self, df, sem_dir):
        """Plot SEM crack density analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Average crack density by specimen and region
        summary = df.groupby(['specimen', 'region'])['crack_density_per_mm2'].mean().unstack()
        summary.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_xlabel('Specimen')
        axes[0].set_ylabel('Crack Density (cracks/mm²)')
        axes[0].set_title('Average Crack Density by Region')
        axes[0].legend(title='Region', bbox_to_anchor=(1.05, 1))
        axes[0].tick_params(axis='x', rotation=45)
        
        # Crack density vs operation time
        time_summary = df.groupby(['operation_hours', 'region'])['crack_density_per_mm2'].mean().unstack()
        time_summary.plot(ax=axes[1], marker='o', linewidth=2)
        axes[1].set_xlabel('Operation Hours')
        axes[1].set_ylabel('Crack Density (cracks/mm²)')
        axes[1].set_title('Crack Density Evolution with Operation Time')
        axes[1].legend(title='Region')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sem_dir, 'crack_density_summary.png'), dpi=150)
        plt.close()
    
    def _plot_eds_line_scan(self, df, eds_dir):
        """Plot EDS line scan"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        elements = ['Ni_wt%', 'Zr_wt%', 'Y_wt%', 'O_wt%', 'La_wt%', 'Sr_wt%', 'Mn_wt%']
        colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        for element, color in zip(elements, colors):
            ax.plot(df['distance_um'], df[element], label=element.replace('_wt%', ''),
                   linewidth=2, color=color)
        
        # Add region labels
        ax.axvspan(0, 300, alpha=0.1, color='gray', label='Anode (Ni-YSZ)')
        ax.axvspan(300, 500, alpha=0.1, color='blue', label='Electrolyte (YSZ)')
        ax.axvspan(500, 800, alpha=0.1, color='red', label='Cathode (LSM)')
        
        ax.set_xlabel('Distance (μm)')
        ax.set_ylabel('Weight %')
        ax.set_title('EDS Line Scan across SOFC Cross-Section')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(eds_dir, 'eds_line_scan.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_nanoindentation_maps(self, df, nano_dir):
        """Plot nanoindentation property maps"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Reshape data for contour plots
        x_unique = sorted(df['x_mm'].unique())
        y_unique = sorted(df['y_mm'].unique())
        X, Y = np.meshgrid(x_unique, y_unique)
        
        E_grid = df.pivot(index='y_mm', columns='x_mm', values='youngs_modulus_GPa').values
        H_grid = df.pivot(index='y_mm', columns='x_mm', values='hardness_GPa').values
        C_grid = df.pivot(index='y_mm', columns='x_mm', values='creep_compliance').values
        
        # Young's modulus map
        im1 = axes[0].contourf(X, Y, E_grid, levels=20, cmap='viridis')
        axes[0].set_xlabel('X Position (mm)')
        axes[0].set_ylabel('Y Position (mm)')
        axes[0].set_title("Young's Modulus Map")
        plt.colorbar(im1, ax=axes[0], label='E (GPa)')
        
        # Hardness map
        im2 = axes[1].contourf(X, Y, H_grid, levels=20, cmap='plasma')
        axes[1].set_xlabel('X Position (mm)')
        axes[1].set_ylabel('Y Position (mm)')
        axes[1].set_title('Hardness Map')
        plt.colorbar(im2, ax=axes[1], label='H (GPa)')
        
        # Creep compliance map
        im3 = axes[2].contourf(X, Y, C_grid, levels=20, cmap='coolwarm')
        axes[2].set_xlabel('X Position (mm)')
        axes[2].set_ylabel('Y Position (mm)')
        axes[2].set_title('Creep Compliance Map')
        plt.colorbar(im3, ax=axes[2], label='Creep Compliance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(nano_dir, 'nanoindentation_property_maps.png'), dpi=150)
        plt.close()
    
    def _plot_load_displacement_curve(self, df, location, curves_dir):
        """Plot load-displacement curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(df['displacement_nm'], df['load_mN'], linewidth=2)
        
        ax.set_xlabel('Displacement (nm)')
        ax.set_ylabel('Load (mN)')
        ax.set_title(f'Load-Displacement Curve - {location.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Annotate max load
        max_idx = df['load_mN'].idxmax()
        ax.annotate(f'Max Load: {df.loc[max_idx, "load_mN"]:.2f} mN',
                   xy=(df.loc[max_idx, 'displacement_nm'], df.loc[max_idx, 'load_mN']),
                   xytext=(10, -30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f'load_disp_curve_{location}.png'), dpi=150)
        plt.close()
    
    def generate_all(self):
        """Generate all experimental datasets"""
        print("=" * 60)
        print("SOFC Experimental Data Generator")
        print("=" * 60)
        
        self.generate_dic_data()
        print()
        self.generate_xrd_data()
        print()
        self.generate_postmortem_data()
        
        print()
        print("=" * 60)
        print("Data generation complete!")
        print(f"All data saved to: {self.output_dir}")
        print("=" * 60)
        
        # Create summary document
        self._create_summary_document()
    
    def _create_summary_document(self):
        """Create summary document describing all generated data"""
        summary_path = os.path.join(self.output_dir, 'DATA_SUMMARY.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SOFC EXPERIMENTAL DATA SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("This synthetic dataset contains comprehensive experimental measurements\n")
            f.write("for Solid Oxide Fuel Cell (SOFC) characterization.\n\n")
            
            f.write("1. DIGITAL IMAGE CORRELATION (DIC) DATA\n")
            f.write("-" * 70 + "\n")
            f.write("   - Sintering strain data (1200-1500°C)\n")
            f.write("   - Thermal cycling data (ΔT = 400°C, 10 cycles)\n")
            f.write("   - Startup/shutdown cycle data (5 cycles)\n")
            f.write("   - Speckle pattern images with timestamps\n")
            f.write("   - Lagrangian strain tensor outputs (Vic-3D format)\n")
            f.write("   - Strain hotspot catalog (>1.0% strain)\n\n")
            
            f.write("2. SYNCHROTRON X-RAY DIFFRACTION (XRD) DATA\n")
            f.write("-" * 70 + "\n")
            f.write("   - Residual stress profiles across cross-sections\n")
            f.write("   - Lattice strain measurements (25-1500°C)\n")
            f.write("   - Sin²ψ method peak shift data\n")
            f.write("   - Microcrack initiation threshold data (ε_cr > 0.02)\n\n")
            
            f.write("3. POST-MORTEM ANALYSIS DATA\n")
            f.write("-" * 70 + "\n")
            f.write("   - SEM crack density analysis (cracks/mm²)\n")
            f.write("   - EDS line scans (Ni, Zr, Y, O, La, Sr, Mn)\n")
            f.write("   - Nano-indentation maps:\n")
            f.write("     * Young's modulus (YSZ: 184.7 GPa, Ni-YSZ: 109.8 GPa)\n")
            f.write("     * Hardness and creep compliance\n")
            f.write("     * Load-displacement curves\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("DATA STRUCTURE\n")
            f.write("=" * 70 + "\n\n")
            
            # List directory structure
            for root, dirs, files in os.walk(self.output_dir):
                level = root.replace(self.output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                f.write(f'{indent}{os.path.basename(root)}/\n')
                sub_indent = ' ' * 2 * (level + 1)
                for file in files:
                    if file != 'DATA_SUMMARY.txt':
                        f.write(f'{sub_indent}{file}\n')
        
        print(f"\nSummary document created: {summary_path}")


if __name__ == "__main__":
    # Initialize generator
    generator = SOFCDataGenerator(output_dir='sofc_experimental_data')
    
    # Generate all data
    generator.generate_all()
