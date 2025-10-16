"""
Experimental Validation Dataset Generator for SOFC Residual Stress Analysis
============================================================================

This script generates a comprehensive experimental validation dataset that simulates
realistic measurements from fabricated SOFC plates, including:
- Fabrication parameters (layer thicknesses, sintering conditions)
- Warp measurements (3D surface topology)
- Residual stress measurements (multiple techniques)
- Realistic experimental uncertainties

Dataset 3: Experimental Validation Dataset (The "Reality Check")
"""

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.spatial import distance_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class SOFCExperimentalDataGenerator:
    """
    Generates realistic experimental validation data for SOFC plates.
    """
    
    def __init__(self, n_samples=35, output_dir='experimental_validation_dataset'):
        """
        Initialize the dataset generator.
        
        Parameters:
        -----------
        n_samples : int
            Number of fabricated SOFC samples (20-50 recommended)
        output_dir : str
            Directory to save generated dataset
        """
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Physical constants
        self.youngs_modulus = {
            'anode': 90e9,      # Pa - NiO-YSZ
            'electrolyte': 200e9,  # Pa - YSZ
            'cathode': 50e9     # Pa - LSM
        }
        
        self.poisson_ratio = {
            'anode': 0.30,
            'electrolyte': 0.31,
            'cathode': 0.28
        }
        
        self.cte = {  # Coefficient of Thermal Expansion (1/K)
            'anode': 12.5e-6,
            'electrolyte': 10.5e-6,
            'cathode': 11.8e-6
        }
        
        # Sample dimensions (m)
        self.sample_width = 0.050  # 50 mm
        self.sample_length = 0.050  # 50 mm
        
        print(f"Initializing SOFC Experimental Data Generator")
        print(f"  Samples: {n_samples}")
        print(f"  Output directory: {output_dir}")
        
    def generate_fabrication_parameters(self):
        """
        Generate fabrication parameters covering extremes and center of parameter space.
        Uses Latin Hypercube Sampling for efficient space-filling design.
        """
        print("\nGenerating fabrication parameters...")
        
        # Parameter ranges (based on typical SOFC manufacturing)
        param_ranges = {
            'anode_thickness_um': (300, 800),      # μm
            'electrolyte_thickness_um': (5, 20),   # μm
            'cathode_thickness_um': (20, 80),      # μm
            'anode_sinter_temp_C': (1350, 1450),   # °C
            'anode_sinter_time_h': (2, 6),         # hours
            'electrolyte_sinter_temp_C': (1400, 1500),
            'electrolyte_sinter_time_h': (2, 5),
            'cathode_sinter_temp_C': (1100, 1200),
            'cathode_sinter_time_h': (1, 4),
            'cooling_rate_C_per_min': (2, 10),     # °C/min
        }
        
        # Latin Hypercube Sampling
        n_params = len(param_ranges)
        intervals = np.linspace(0, 1, self.n_samples + 1)
        
        samples = np.zeros((self.n_samples, n_params))
        for i in range(n_params):
            # Random permutation of intervals
            random_samples = np.random.uniform(intervals[:-1], intervals[1:])
            np.random.shuffle(random_samples)
            samples[:, i] = random_samples
        
        # Scale to actual parameter ranges
        param_data = {}
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_data[param_name] = samples[:, i] * (max_val - min_val) + min_val
        
        # Add derived parameters
        param_data['total_thickness_um'] = (
            param_data['anode_thickness_um'] + 
            param_data['electrolyte_thickness_um'] + 
            param_data['cathode_thickness_um']
        )
        
        # Thickness ratios
        param_data['electrolyte_anode_ratio'] = (
            param_data['electrolyte_thickness_um'] / param_data['anode_thickness_um']
        )
        param_data['cathode_anode_ratio'] = (
            param_data['cathode_thickness_um'] / param_data['anode_thickness_um']
        )
        
        # Sample IDs and fabrication dates
        base_date = datetime(2025, 1, 15)
        param_data['sample_id'] = [f'SOFC-EXP-{i+1:03d}' for i in range(self.n_samples)]
        param_data['fabrication_date'] = [
            (base_date + timedelta(days=i*2)).strftime('%Y-%m-%d') 
            for i in range(self.n_samples)
        ]
        
        # Batch information (multiple samples per batch)
        param_data['batch_id'] = [f'BATCH-{(i//5)+1:02d}' for i in range(self.n_samples)]
        
        self.fabrication_df = pd.DataFrame(param_data)
        
        print(f"  Generated {self.n_samples} fabrication parameter sets")
        print(f"  Parameter space coverage:")
        print(f"    Anode thickness: {param_data['anode_thickness_um'].min():.1f} - {param_data['anode_thickness_um'].max():.1f} μm")
        print(f"    Electrolyte thickness: {param_data['electrolyte_thickness_um'].min():.1f} - {param_data['electrolyte_thickness_um'].max():.1f} μm")
        print(f"    Cathode thickness: {param_data['cathode_thickness_um'].min():.1f} - {param_data['cathode_thickness_um'].max():.1f} μm")
        
        return self.fabrication_df
    
    def generate_warp_measurements(self):
        """
        Generate 3D warp measurements simulating laser scanning confocal microscopy
        or white light interferometry measurements.
        
        Returns realistic 3D point clouds for each sample.
        """
        print("\nGenerating warp measurements (3D point clouds)...")
        
        # Measurement grid parameters
        n_points_x = 128  # High resolution
        n_points_y = 128
        
        x = np.linspace(0, self.sample_width, n_points_x)
        y = np.linspace(0, self.sample_length, n_points_y)
        X, Y = np.meshgrid(x, y)
        
        warp_data = {}
        
        for idx, row in self.fabrication_df.iterrows():
            sample_id = row['sample_id']
            
            # Physics-based warp generation
            # Warp is influenced by thermal mismatch and thickness ratios
            cte_mismatch = (
                abs(self.cte['anode'] - self.cte['electrolyte']) + 
                abs(self.cte['cathode'] - self.cte['electrolyte'])
            )
            
            thickness_asymmetry = (
                row['cathode_thickness_um'] - row['anode_thickness_um']
            ) / row['total_thickness_um']
            
            # Maximum warp amplitude (μm)
            max_warp = (
                cte_mismatch * 1e6 * 
                (row['electrolyte_sinter_temp_C'] - 25) * 
                row['total_thickness_um'] * 
                (1 + abs(thickness_asymmetry) * 5)
            )
            
            # Add realistic cooling rate effect
            cooling_effect = 1.0 + (row['cooling_rate_C_per_min'] - 6) * 0.1
            max_warp *= cooling_effect
            
            # Generate realistic warp field using multiple modes
            Z = np.zeros_like(X)
            
            # Primary bending mode (bi-axial curvature)
            curvature_x = max_warp / (self.sample_width ** 2)
            curvature_y = max_warp / (self.sample_length ** 2) * 0.8  # Slight asymmetry
            
            Z += curvature_x * (X - self.sample_width/2)**2
            Z += curvature_y * (Y - self.sample_length/2)**2
            
            # Add twist mode (smaller amplitude)
            twist_amplitude = max_warp * 0.15
            Z += twist_amplitude * (X - self.sample_width/2) * (Y - self.sample_length/2) / (self.sample_width * self.sample_length)
            
            # Add higher-order modes (localized warping)
            for mode in range(3):
                freq_x = 2 * np.pi * (mode + 1) / self.sample_width
                freq_y = 2 * np.pi * (mode + 1) / self.sample_length
                amplitude = max_warp * 0.05 / (mode + 1)
                phase_x = np.random.uniform(0, 2*np.pi)
                phase_y = np.random.uniform(0, 2*np.pi)
                Z += amplitude * np.sin(freq_x * X + phase_x) * np.sin(freq_y * Y + phase_y)
            
            # Add measurement noise (realistic for interferometry: ~0.01 - 0.1 μm)
            noise_level = 0.05  # μm RMS
            Z += np.random.normal(0, noise_level, Z.shape)
            
            # Add edge effects (higher uncertainty near edges)
            edge_dist = np.minimum(
                np.minimum(X, self.sample_width - X),
                np.minimum(Y, self.sample_length - Y)
            )
            edge_uncertainty = np.exp(-edge_dist / (self.sample_width * 0.05))
            Z += np.random.normal(0, noise_level * edge_uncertainty, Z.shape)
            
            # Store data
            warp_data[sample_id] = {
                'X': X,
                'Y': Y,
                'Z': Z,
                'max_warp_um': np.max(Z) - np.min(Z),
                'rms_warp_um': np.sqrt(np.mean(Z**2)),
                'measurement_date': row['fabrication_date'],
                'technique': 'White_Light_Interferometry',
                'resolution_um': (self.sample_width / n_points_x) * 1e6,
                'noise_level_um': noise_level
            }
            
            if idx % 10 == 0:
                print(f"  Generated warp data for {idx+1}/{self.n_samples} samples")
        
        self.warp_data = warp_data
        
        print(f"  Warp range: {min(d['max_warp_um'] for d in warp_data.values()):.2f} - "
              f"{max(d['max_warp_um'] for d in warp_data.values()):.2f} μm")
        
        return warp_data
    
    def generate_curvature_based_stress(self):
        """
        Generate residual stress measurements using curvature-based inverse method
        (Stoney's formula approach). Provides through-thickness average stress.
        """
        print("\nGenerating curvature-based stress measurements...")
        
        curvature_stress_data = []
        
        for idx, row in self.fabrication_df.iterrows():
            sample_id = row['sample_id']
            warp = self.warp_data[sample_id]
            
            # Calculate curvature from warp field
            Z = warp['Z']
            dx = self.sample_width / Z.shape[1]
            dy = self.sample_length / Z.shape[0]
            
            # Second derivatives (curvature)
            d2z_dx2 = np.gradient(np.gradient(Z, dx, axis=1), dx, axis=1)
            d2z_dy2 = np.gradient(np.gradient(Z, dy, axis=0), dy, axis=0)
            
            # Average curvature
            kappa_x = np.mean(d2z_dx2)
            kappa_y = np.mean(d2z_dy2)
            
            # Modified Stoney's formula for multilayer
            # Stress in each layer (through-thickness average)
            
            # Substrate (anode) properties
            h_s = row['anode_thickness_um'] * 1e-6  # m
            E_s = self.youngs_modulus['anode']
            nu_s = self.poisson_ratio['anode']
            
            # Film (electrolyte) properties
            h_f_electrolyte = row['electrolyte_thickness_um'] * 1e-6
            E_f_electrolyte = self.youngs_modulus['electrolyte']
            nu_f_electrolyte = self.poisson_ratio['electrolyte']
            
            # Biaxial modulus
            M_s = E_s / (1 - nu_s)
            M_f_electrolyte = E_f_electrolyte / (1 - nu_f_electrolyte)
            
            # Electrolyte stress (GPa)
            sigma_electrolyte = (M_s * h_s**2 * kappa_x) / (6 * h_f_electrolyte)
            sigma_electrolyte /= 1e9  # Convert to GPa
            
            # Cathode stress calculation
            h_f_cathode = row['cathode_thickness_um'] * 1e-6
            E_f_cathode = self.youngs_modulus['cathode']
            nu_f_cathode = self.poisson_ratio['cathode']
            M_f_cathode = E_f_cathode / (1 - nu_f_cathode)
            
            sigma_cathode = (M_s * h_s**2 * kappa_y) / (6 * h_f_cathode)
            sigma_cathode /= 1e9
            
            # Anode stress (balancing constraint)
            total_force = (
                sigma_electrolyte * h_f_electrolyte * 1e9 + 
                sigma_cathode * h_f_cathode * 1e9
            )
            sigma_anode = -total_force / (h_s * 1e9)  # GPa
            
            # Add measurement uncertainty (±10-15% typical for curvature method)
            uncertainty = 0.12
            sigma_electrolyte += np.random.normal(0, abs(sigma_electrolyte) * uncertainty)
            sigma_cathode += np.random.normal(0, abs(sigma_cathode) * uncertainty)
            sigma_anode += np.random.normal(0, abs(sigma_anode) * uncertainty)
            
            curvature_stress_data.append({
                'sample_id': sample_id,
                'technique': 'Curvature_Stoney',
                'anode_stress_GPa': sigma_anode,
                'electrolyte_stress_GPa': sigma_electrolyte,
                'cathode_stress_GPa': sigma_cathode,
                'curvature_x_1_per_m': kappa_x,
                'curvature_y_1_per_m': kappa_y,
                'uncertainty_percent': uncertainty * 100,
                'measurement_date': row['fabrication_date']
            })
        
        self.curvature_stress_df = pd.DataFrame(curvature_stress_data)
        
        print(f"  Generated curvature-based stress for {len(curvature_stress_data)} samples")
        print(f"  Electrolyte stress range: {self.curvature_stress_df['electrolyte_stress_GPa'].min():.3f} - "
              f"{self.curvature_stress_df['electrolyte_stress_GPa'].max():.3f} GPa")
        
        return self.curvature_stress_df
    
    def generate_layer_removal_stress(self, n_removal_steps=5):
        """
        Generate stress profiles from destructive layer removal method.
        This provides through-thickness stress gradients.
        """
        print("\nGenerating layer removal stress profiles...")
        
        layer_removal_data = {}
        
        # Only perform this expensive technique on a subset of samples
        selected_samples = np.random.choice(
            range(self.n_samples), 
            size=min(15, self.n_samples),
            replace=False
        )
        
        for sample_idx in selected_samples:
            row = self.fabrication_df.iloc[sample_idx]
            sample_id = row['sample_id']
            
            # Through-thickness stress profile
            total_thickness = row['total_thickness_um']
            
            # Define layer boundaries (normalized position)
            anode_frac = row['anode_thickness_um'] / total_thickness
            electrolyte_frac = row['electrolyte_thickness_um'] / total_thickness
            cathode_frac = row['cathode_thickness_um'] / total_thickness
            
            # Thickness positions (μm from bottom)
            z_positions = np.linspace(0, total_thickness, n_removal_steps * 3)
            
            # Generate realistic stress profile with gradients
            stress_profile = np.zeros(len(z_positions))
            
            for i, z in enumerate(z_positions):
                z_norm = z / total_thickness
                
                if z_norm < anode_frac:  # Anode region
                    # Tensile stress with gradient
                    depth_factor = z_norm / anode_frac
                    base_stress = -0.15  # GPa (compressive at bottom)
                    gradient_stress = 0.25 * depth_factor  # Increasing tension
                    stress_profile[i] = base_stress + gradient_stress
                    
                elif z_norm < (anode_frac + electrolyte_frac):  # Electrolyte region
                    # High compressive stress
                    z_local = (z_norm - anode_frac) / electrolyte_frac
                    stress_profile[i] = -0.8 + 0.2 * z_local  # Strong compression
                    
                else:  # Cathode region
                    # Tensile stress
                    z_local = (z_norm - anode_frac - electrolyte_frac) / cathode_frac
                    stress_profile[i] = 0.3 - 0.15 * z_local
            
            # Add realistic variations based on sintering conditions
            temp_effect = (row['electrolyte_sinter_temp_C'] - 1450) / 100 * 0.1
            stress_profile *= (1 + temp_effect)
            
            # Add measurement noise (higher for this destructive technique)
            measurement_noise = np.random.normal(0, 0.05, len(stress_profile))
            stress_profile += measurement_noise
            
            # Simulate removal steps
            removal_depths = np.linspace(0, total_thickness * 0.8, n_removal_steps)
            warp_after_removal = []
            
            for removal_depth in removal_depths:
                # Warp changes as layers are removed
                remaining_thickness = total_thickness - removal_depth
                stress_relief = np.mean(stress_profile[z_positions <= removal_depth])
                
                # Simplified warp calculation
                warp_change = stress_relief * remaining_thickness * 0.5
                warp_after_removal.append(warp_change)
            
            layer_removal_data[sample_id] = {
                'z_positions_um': z_positions,
                'stress_profile_GPa': stress_profile,
                'removal_depths_um': removal_depths,
                'warp_after_removal_um': np.array(warp_after_removal),
                'technique': 'Layer_Removal_Milling',
                'n_steps': n_removal_steps,
                'uncertainty_GPa': 0.05
            }
            
        self.layer_removal_data = layer_removal_data
        
        print(f"  Generated layer removal data for {len(layer_removal_data)} samples")
        print(f"  Average stress gradient: {np.mean([np.std(d['stress_profile_GPa']) for d in layer_removal_data.values()]):.3f} GPa")
        
        return layer_removal_data
    
    def generate_xrd_stress_measurements(self, n_points_per_sample=25):
        """
        Generate XRD stress measurements at discrete surface points.
        Provides localized, direct stress measurements.
        """
        print("\nGenerating XRD stress measurements...")
        
        xrd_data = []
        
        # Select subset of samples for XRD (expensive technique)
        selected_samples = np.random.choice(
            range(self.n_samples),
            size=min(20, self.n_samples),
            replace=False
        )
        
        for sample_idx in selected_samples:
            row = self.fabrication_df.iloc[sample_idx]
            sample_id = row['sample_id']
            
            # Generate measurement points (regular grid on surface)
            grid_size = int(np.sqrt(n_points_per_sample))
            x_points = np.linspace(0.01, 0.04, grid_size)  # Avoid edges
            y_points = np.linspace(0.01, 0.04, grid_size)
            
            for x in x_points:
                for y in y_points:
                    # Surface stress depends on position
                    # Central region has more uniform stress
                    dist_from_center = np.sqrt(
                        (x - self.sample_width/2)**2 + 
                        (y - self.sample_length/2)**2
                    )
                    
                    # Base stress (cathode surface)
                    base_stress = 0.35  # GPa (tensile)
                    
                    # Add spatial variation
                    spatial_variation = 0.1 * np.sin(2 * np.pi * x / self.sample_width)
                    stress_xx = base_stress + spatial_variation
                    stress_yy = base_stress - spatial_variation * 0.5
                    
                    # Edge effects (higher stress gradients near edges)
                    if dist_from_center > 0.015:
                        edge_factor = 1 + (dist_from_center - 0.015) * 10
                        stress_xx *= edge_factor
                        stress_yy *= edge_factor
                    
                    # XRD measurement uncertainty (±0.03 GPa typical)
                    uncertainty = 0.03
                    stress_xx += np.random.normal(0, uncertainty)
                    stress_yy += np.random.normal(0, uncertainty)
                    
                    # XRD also gives shear stress component
                    stress_xy = np.random.normal(0, 0.02)  # Small shear
                    
                    xrd_data.append({
                        'sample_id': sample_id,
                        'technique': 'XRD',
                        'x_position_mm': x * 1000,
                        'y_position_mm': y * 1000,
                        'surface_layer': 'cathode',
                        'stress_xx_GPa': stress_xx,
                        'stress_yy_GPa': stress_yy,
                        'stress_xy_GPa': stress_xy,
                        'peak_used': 'YSZ_311',
                        'sin2psi_slope': stress_xx * 2.13e-6,  # Typical XRD constant
                        'uncertainty_GPa': uncertainty,
                        'measurement_time_min': np.random.uniform(15, 30)
                    })
        
        self.xrd_stress_df = pd.DataFrame(xrd_data)
        
        print(f"  Generated {len(xrd_data)} XRD measurement points")
        print(f"  Samples measured: {len(selected_samples)}")
        print(f"  Stress range: {self.xrd_stress_df['stress_xx_GPa'].min():.3f} - "
              f"{self.xrd_stress_df['stress_xx_GPa'].max():.3f} GPa")
        
        return self.xrd_stress_df
    
    def generate_raman_stress_measurements(self, n_points_per_sample=50):
        """
        Generate Raman spectroscopy stress measurements.
        Higher spatial resolution than XRD, but different material sensitivity.
        """
        print("\nGenerating Raman stress measurements...")
        
        raman_data = []
        
        # Select subset of samples
        selected_samples = np.random.choice(
            range(self.n_samples),
            size=min(15, self.n_samples),
            replace=False
        )
        
        for sample_idx in selected_samples:
            row = self.fabrication_df.iloc[sample_idx]
            sample_id = row['sample_id']
            
            # Random points for Raman mapping
            x_points = np.random.uniform(0.005, 0.045, n_points_per_sample)
            y_points = np.random.uniform(0.005, 0.045, n_points_per_sample)
            
            for x, y in zip(x_points, y_points):
                # Raman shift correlates with stress
                # Different for each layer material
                
                # Assume measuring cathode (LSM) - typical Raman active modes
                base_wavenumber = 640  # cm^-1 (LSM characteristic peak)
                
                # Stress causes peak shift
                # Typical: ~5 cm^-1/GPa shift rate
                stress_local = 0.3 + np.random.normal(0, 0.15)  # GPa
                peak_shift = stress_local * 5.2  # cm^-1
                
                measured_wavenumber = base_wavenumber + peak_shift
                
                # Peak width also sensitive to stress state
                peak_width = 25 + abs(stress_local) * 8  # cm^-1
                
                # Measurement uncertainty
                uncertainty = 0.08  # GPa
                stress_local += np.random.normal(0, uncertainty)
                
                raman_data.append({
                    'sample_id': sample_id,
                    'technique': 'Raman_Spectroscopy',
                    'x_position_mm': x * 1000,
                    'y_position_mm': y * 1000,
                    'surface_layer': 'cathode',
                    'peak_wavenumber_cm_inv': measured_wavenumber,
                    'peak_width_cm_inv': peak_width,
                    'stress_estimate_GPa': stress_local,
                    'stress_coefficient_cm_inv_per_GPa': 5.2,
                    'uncertainty_GPa': uncertainty,
                    'laser_wavelength_nm': 532,
                    'spot_size_um': 1.0,
                    'integration_time_s': np.random.uniform(5, 15)
                })
        
        self.raman_stress_df = pd.DataFrame(raman_data)
        
        print(f"  Generated {len(raman_data)} Raman measurement points")
        print(f"  Samples measured: {len(selected_samples)}")
        print(f"  Stress range: {self.raman_stress_df['stress_estimate_GPa'].min():.3f} - "
              f"{self.raman_stress_df['stress_estimate_GPa'].max():.3f} GPa")
        
        return self.raman_stress_df
    
    def save_dataset(self):
        """
        Save all generated data to multiple formats for easy access.
        """
        print("\nSaving dataset...")
        
        # 1. Save fabrication parameters (CSV)
        fab_file = self.output_dir / 'fabrication_parameters.csv'
        self.fabrication_df.to_csv(fab_file, index=False)
        print(f"  Saved: {fab_file}")
        
        # 2. Save curvature-based stress (CSV)
        curv_file = self.output_dir / 'curvature_stress_measurements.csv'
        self.curvature_stress_df.to_csv(curv_file, index=False)
        print(f"  Saved: {curv_file}")
        
        # 3. Save XRD measurements (CSV)
        xrd_file = self.output_dir / 'xrd_stress_measurements.csv'
        self.xrd_stress_df.to_csv(xrd_file, index=False)
        print(f"  Saved: {xrd_file}")
        
        # 4. Save Raman measurements (CSV)
        raman_file = self.output_dir / 'raman_stress_measurements.csv'
        self.raman_stress_df.to_csv(raman_file, index=False)
        print(f"  Saved: {raman_file}")
        
        # 5. Save warp data (HDF5 for 3D arrays)
        warp_file = self.output_dir / 'warp_measurements_3d.h5'
        with h5py.File(warp_file, 'w') as f:
            for sample_id, data in self.warp_data.items():
                grp = f.create_group(sample_id)
                grp.create_dataset('X', data=data['X'])
                grp.create_dataset('Y', data=data['Y'])
                grp.create_dataset('Z', data=data['Z'])
                grp.attrs['max_warp_um'] = data['max_warp_um']
                grp.attrs['rms_warp_um'] = data['rms_warp_um']
                grp.attrs['technique'] = data['technique']
                grp.attrs['resolution_um'] = data['resolution_um']
                grp.attrs['noise_level_um'] = data['noise_level_um']
        print(f"  Saved: {warp_file}")
        
        # 6. Save layer removal data (HDF5)
        if hasattr(self, 'layer_removal_data'):
            layer_file = self.output_dir / 'layer_removal_stress_profiles.h5'
            with h5py.File(layer_file, 'w') as f:
                for sample_id, data in self.layer_removal_data.items():
                    grp = f.create_group(sample_id)
                    grp.create_dataset('z_positions_um', data=data['z_positions_um'])
                    grp.create_dataset('stress_profile_GPa', data=data['stress_profile_GPa'])
                    grp.create_dataset('removal_depths_um', data=data['removal_depths_um'])
                    grp.create_dataset('warp_after_removal_um', data=data['warp_after_removal_um'])
                    grp.attrs['technique'] = data['technique']
                    grp.attrs['n_steps'] = data['n_steps']
                    grp.attrs['uncertainty_GPa'] = data['uncertainty_GPa']
            print(f"  Saved: {layer_file}")
        
        # 7. Save metadata
        metadata = {
            'dataset_name': 'SOFC Experimental Validation Dataset',
            'dataset_version': '1.0',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': self.n_samples,
            'sample_dimensions_mm': [self.sample_width * 1000, self.sample_length * 1000],
            'measurement_techniques': [
                'White Light Interferometry (Warp)',
                'Curvature-based Inverse (Stoney)',
                'Layer Removal Milling',
                'X-Ray Diffraction',
                'Raman Spectroscopy'
            ],
            'parameter_ranges': {
                'anode_thickness_um': [300, 800],
                'electrolyte_thickness_um': [5, 20],
                'cathode_thickness_um': [20, 80],
                'sintering_temp_C': [1100, 1500]
            },
            'notes': 'Synthetic dataset simulating realistic experimental measurements for ML validation'
        }
        
        metadata_file = self.output_dir / 'dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {metadata_file}")
        
        # 8. Create summary statistics
        summary = {
            'Total samples': self.n_samples,
            'Warp measurements': len(self.warp_data),
            'Curvature stress measurements': len(self.curvature_stress_df),
            'XRD measurement points': len(self.xrd_stress_df),
            'Raman measurement points': len(self.raman_stress_df),
            'Layer removal profiles': len(self.layer_removal_data) if hasattr(self, 'layer_removal_data') else 0,
            'Warp range (μm)': [
                f"{min(d['max_warp_um'] for d in self.warp_data.values()):.2f}",
                f"{max(d['max_warp_um'] for d in self.warp_data.values()):.2f}"
            ],
            'Electrolyte stress range (GPa)': [
                f"{self.curvature_stress_df['electrolyte_stress_GPa'].min():.3f}",
                f"{self.curvature_stress_df['electrolyte_stress_GPa'].max():.3f}"
            ]
        }
        
        summary_file = self.output_dir / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_file}")
        
        print(f"\nDataset saved successfully to: {self.output_dir.absolute()}")
        
        return self.output_dir
    
    def generate_complete_dataset(self):
        """
        Generate the complete experimental validation dataset.
        """
        print("="*80)
        print("SOFC EXPERIMENTAL VALIDATION DATASET GENERATION")
        print("="*80)
        
        # Step 1: Fabrication parameters
        self.generate_fabrication_parameters()
        
        # Step 2: Warp measurements
        self.generate_warp_measurements()
        
        # Step 3: Stress measurements (multiple techniques)
        self.generate_curvature_based_stress()
        self.generate_layer_removal_stress()
        self.generate_xrd_stress_measurements()
        self.generate_raman_stress_measurements()
        
        # Step 4: Save everything
        self.save_dataset()
        
        print("\n" + "="*80)
        print("DATASET GENERATION COMPLETE")
        print("="*80)
        
        return self.output_dir


def main():
    """
    Main execution function.
    """
    # Generate dataset with 35 samples (sweet spot between 20-50)
    generator = SOFCExperimentalDataGenerator(
        n_samples=35,
        output_dir='experimental_validation_dataset'
    )
    
    # Generate complete dataset
    output_path = generator.generate_complete_dataset()
    
    print(f"\n✓ Dataset ready for ML validation pipeline!")
    print(f"✓ Location: {output_path}")
    print(f"\n📊 Dataset can be used for:")
    print(f"   1. ML model validation (warp → predicted stress)")
    print(f"   2. FEA model calibration")
    print(f"   3. Uncertainty quantification")
    print(f"   4. Multi-technique comparison studies")


if __name__ == '__main__':
    main()
