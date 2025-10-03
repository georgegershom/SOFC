"""
Synchrotron X-ray Diffraction (XRD) Data Generator for SOFC Analysis
Generates residual stress profiles, lattice strain measurements, and microcrack data
"""

import numpy as np
import pandas as pd
import json
from scipy import signal
from datetime import datetime
import os

class XRDDataGenerator:
    def __init__(self, base_path="xrd_data"):
        self.base_path = base_path
        np.random.seed(43)  # Different seed for variety
        
        # Material parameters for SOFC components
        self.materials = {
            'YSZ': {  # Yttria-Stabilized Zirconia
                'd_spacing': 2.958,  # Angstroms for (111) peak
                'elastic_modulus': 210e9,  # Pa
                'poisson_ratio': 0.31,
                'thermal_expansion': 10.5e-6,  # /K
                'peaks': [(111, 30.2), (200, 34.9), (220, 50.2), (311, 59.8)]
            },
            'NiO': {  # Nickel Oxide (before reduction)
                'd_spacing': 2.412,
                'elastic_modulus': 200e9,
                'poisson_ratio': 0.31,
                'thermal_expansion': 14.1e-6,
                'peaks': [(111, 37.2), (200, 43.3), (220, 62.9)]
            },
            'Ni': {  # Metallic Nickel (after reduction)
                'd_spacing': 2.034,
                'elastic_modulus': 200e9,
                'poisson_ratio': 0.31,
                'thermal_expansion': 13.3e-6,
                'peaks': [(111, 44.5), (200, 51.8), (220, 76.4)]
            },
            'GDC': {  # Gadolinium-Doped Ceria
                'd_spacing': 3.124,
                'elastic_modulus': 180e9,
                'poisson_ratio': 0.33,
                'thermal_expansion': 12.5e-6,
                'peaks': [(111, 28.5), (200, 33.1), (220, 47.5), (311, 56.3)]
            }
        }
        
    def generate_residual_stress_profile(self, experiment_type):
        """Generate residual stress profiles across SOFC cross-section"""
        print(f"  Generating residual stress profiles for {experiment_type}...")
        
        # SOFC layer structure (in micrometers)
        layers = [
            {'name': 'Anode_Support', 'thickness': 500, 'material': 'Ni-YSZ', 
             'position': 0, 'stress_factor': 1.2},
            {'name': 'Anode_Functional', 'thickness': 20, 'material': 'Ni-YSZ',
             'position': 500, 'stress_factor': 1.5},
            {'name': 'Electrolyte', 'thickness': 10, 'material': 'YSZ',
             'position': 520, 'stress_factor': 2.0},
            {'name': 'Cathode_Functional', 'thickness': 20, 'material': 'GDC',
             'position': 530, 'stress_factor': 1.8},
            {'name': 'Cathode_Current_Collector', 'thickness': 50, 'material': 'LSM',
             'position': 550, 'stress_factor': 1.3}
        ]
        
        # Generate position array (micrometers)
        positions = np.linspace(0, 600, 200)
        
        # Initialize stress components
        sigma_xx = np.zeros_like(positions)
        sigma_yy = np.zeros_like(positions)
        sigma_zz = np.zeros_like(positions)
        
        # Base stress levels depend on experiment type
        if experiment_type == 'sintering':
            base_stress = -150e6  # Compressive stress from sintering (Pa)
            stress_variation = 50e6
        elif experiment_type == 'thermal_cycling':
            base_stress = -80e6  # Thermal mismatch stress
            stress_variation = 120e6  # Higher variation due to cycling
        else:  # startup_shutdown
            base_stress = -100e6
            stress_variation = 80e6
        
        # Generate stress profile for each layer
        for layer in layers:
            mask = (positions >= layer['position']) & \
                   (positions < layer['position'] + layer['thickness'])
            
            # Layer-specific stress with interface effects
            layer_stress = base_stress * layer['stress_factor']
            
            # Add interface stress concentration
            interface_distance = np.minimum(
                np.abs(positions - layer['position']),
                np.abs(positions - (layer['position'] + layer['thickness']))
            )
            interface_effect = np.exp(-interface_distance / 5) * stress_variation
            
            sigma_xx[mask] = layer_stress + interface_effect[mask] + \
                            np.random.normal(0, stress_variation * 0.1, np.sum(mask))
            sigma_yy[mask] = sigma_xx[mask] * 0.7  # Biaxial stress state
            sigma_zz[mask] = sigma_xx[mask] * 0.3  # Through-thickness component
        
        # Smooth the profiles
        from scipy.ndimage import gaussian_filter1d
        sigma_xx = gaussian_filter1d(sigma_xx, sigma=2)
        sigma_yy = gaussian_filter1d(sigma_yy, sigma=2)
        sigma_zz = gaussian_filter1d(sigma_zz, sigma=2)
        
        # Calculate von Mises stress
        von_mises = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + 
                                   (sigma_yy - sigma_zz)**2 + 
                                   (sigma_zz - sigma_xx)**2))
        
        # Create DataFrame
        stress_data = pd.DataFrame({
            'position_um': positions,
            'sigma_xx_MPa': sigma_xx / 1e6,
            'sigma_yy_MPa': sigma_yy / 1e6,
            'sigma_zz_MPa': sigma_zz / 1e6,
            'von_mises_MPa': von_mises / 1e6,
            'layer': [''] * len(positions)
        })
        
        # Assign layer names
        for layer in layers:
            mask = (stress_data['position_um'] >= layer['position']) & \
                   (stress_data['position_um'] < layer['position'] + layer['thickness'])
            stress_data.loc[mask, 'layer'] = layer['name']
        
        # Save data
        stress_data.to_csv(os.path.join(self.base_path, experiment_type, 
                                        'residual_stress_profile.csv'), index=False)
        
        return stress_data
    
    def generate_lattice_strain_data(self, experiment_type):
        """Generate lattice strain measurements under thermal load"""
        print(f"  Generating lattice strain data for {experiment_type}...")
        
        if experiment_type == 'thermal_cycling':
            temperatures = np.concatenate([
                np.linspace(600, 1000, 20),  # Heating
                np.linspace(1000, 600, 20)   # Cooling
            ])
            n_cycles = 5
        else:
            temperatures = np.linspace(25, 1000, 50)
            n_cycles = 1
        
        data_collection = []
        
        for cycle in range(n_cycles):
            for temp in temperatures:
                for material_name, material_props in self.materials.items():
                    # Calculate thermal strain
                    thermal_strain = material_props['thermal_expansion'] * (temp - 25)
                    
                    # Add mechanical strain component (stress-induced)
                    if experiment_type == 'sintering' and temp > 1200:
                        mechanical_strain = -0.002 * (1 - np.exp(-(temp - 1200)/200))
                    else:
                        mechanical_strain = np.random.normal(0, 0.0001)
                    
                    # Total lattice strain
                    total_strain = thermal_strain + mechanical_strain
                    
                    # Add measurement uncertainty
                    measured_strain = total_strain + np.random.normal(0, 1e-5)
                    
                    # Calculate d-spacing change
                    d0 = material_props['d_spacing']
                    d_measured = d0 * (1 + measured_strain)
                    
                    data_collection.append({
                        'cycle': cycle + 1,
                        'temperature_C': temp,
                        'material': material_name,
                        'd_spacing_A': d_measured,
                        'd0_spacing_A': d0,
                        'lattice_strain': measured_strain,
                        'thermal_strain': thermal_strain,
                        'mechanical_strain': mechanical_strain,
                        'measurement_error': measured_strain - total_strain
                    })
        
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, experiment_type, 
                               'lattice_strain_data.csv'), index=False)
        
        return df
    
    def generate_sin2psi_data(self, experiment_type):
        """Generate sin²ψ method stress calculation data"""
        print(f"  Generating sin²ψ stress analysis data for {experiment_type}...")
        
        # Tilt angles for sin²ψ method
        psi_angles = np.array([0, 15, 30, 45, 60, 75])
        sin2_psi = np.sin(np.radians(psi_angles))**2
        
        data_collection = []
        
        # Generate data for different positions in the sample
        positions = np.linspace(0, 600, 10)  # Across SOFC thickness
        
        for pos in positions:
            # Determine stress based on position and experiment
            if experiment_type == 'sintering':
                base_stress = -150e6 + pos * 100e3  # Stress gradient
            elif experiment_type == 'thermal_cycling':
                base_stress = -100e6 * np.sin(pos / 100)  # Oscillating stress
            else:
                base_stress = -80e6 + np.random.normal(0, 20e6)
            
            for material_name in ['YSZ', 'Ni']:
                if material_name not in self.materials:
                    continue
                    
                mat = self.materials[material_name]
                
                # Calculate strain for each tilt angle
                for psi, s2p in zip(psi_angles, sin2_psi):
                    # Strain calculation based on sin²ψ method
                    E = mat['elastic_modulus']
                    nu = mat['poisson_ratio']
                    
                    # Biaxial stress state
                    epsilon_psi = (1 + nu) / E * base_stress * s2p - \
                                 2 * nu / E * base_stress
                    
                    # Add measurement noise
                    epsilon_measured = epsilon_psi + np.random.normal(0, 1e-6)
                    
                    # Peak position shift
                    d0 = mat['d_spacing']
                    d_psi = d0 * (1 + epsilon_measured)
                    two_theta_0 = 2 * np.degrees(np.arcsin(1.5406 / (2 * d0)))  # Cu Kα
                    two_theta_psi = 2 * np.degrees(np.arcsin(1.5406 / (2 * d_psi)))
                    peak_shift = two_theta_psi - two_theta_0
                    
                    data_collection.append({
                        'position_um': pos,
                        'material': material_name,
                        'psi_deg': psi,
                        'sin2_psi': s2p,
                        'strain': epsilon_measured,
                        'd_spacing_A': d_psi,
                        'two_theta_deg': two_theta_psi,
                        'peak_shift_deg': peak_shift,
                        'calculated_stress_MPa': base_stress / 1e6,
                        'measurement_quality': np.random.choice(['good', 'excellent', 'fair'],
                                                               p=[0.7, 0.2, 0.1])
                    })
        
        df = pd.DataFrame(data_collection)
        
        # Perform linear regression for each position/material to get stress
        stress_results = []
        for pos in positions:
            for mat in ['YSZ', 'Ni']:
                subset = df[(df['position_um'] == pos) & (df['material'] == mat)]
                if len(subset) > 0:
                    # Linear fit of strain vs sin²ψ
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = \
                        stats.linregress(subset['sin2_psi'], subset['strain'])
                    
                    # Calculate stress from slope
                    if mat in self.materials:
                        E = self.materials[mat]['elastic_modulus']
                        nu = self.materials[mat]['poisson_ratio']
                        calculated_stress = slope * E / (1 + nu)
                        
                        stress_results.append({
                            'position_um': pos,
                            'material': mat,
                            'stress_MPa': calculated_stress / 1e6,
                            'fit_slope': slope,
                            'fit_intercept': intercept,
                            'fit_r2': r_value**2,
                            'fit_std_err': std_err
                        })
        
        # Save both raw data and stress results
        df.to_csv(os.path.join(self.base_path, experiment_type, 
                               'sin2psi_raw_data.csv'), index=False)
        
        stress_df = pd.DataFrame(stress_results)
        stress_df.to_csv(os.path.join(self.base_path, experiment_type,
                                      'sin2psi_stress_results.csv'), index=False)
        
        return df, stress_df
    
    def generate_microcrack_data(self, experiment_type):
        """Generate microcrack initiation threshold data"""
        print(f"  Generating microcrack threshold data for {experiment_type}...")
        
        # Critical strain thresholds for different materials
        crack_thresholds = {
            'YSZ': 0.002,  # 0.2% strain
            'Ni-YSZ': 0.0025,
            'GDC': 0.0018,
            'LSM': 0.0022
        }
        
        data_collection = []
        
        # Generate data for different loading conditions
        if experiment_type == 'thermal_cycling':
            n_cycles = np.arange(1, 51)
            for cycle in n_cycles:
                for material, threshold in crack_thresholds.items():
                    # Strain accumulation with cycling
                    accumulated_strain = threshold * (1 - np.exp(-cycle/10))
                    
                    # Add statistical variation
                    measured_strain = accumulated_strain + np.random.normal(0, threshold * 0.1)
                    
                    # Crack initiation probability
                    if measured_strain > threshold:
                        crack_prob = 1 - np.exp(-5 * (measured_strain - threshold) / threshold)
                    else:
                        crack_prob = 0
                    
                    # Actual crack detection (binary)
                    crack_detected = np.random.random() < crack_prob
                    
                    data_collection.append({
                        'cycle': cycle,
                        'material': material,
                        'critical_strain_threshold': threshold,
                        'measured_strain': measured_strain,
                        'strain_ratio': measured_strain / threshold,
                        'crack_probability': crack_prob,
                        'crack_detected': crack_detected,
                        'crack_length_um': np.random.exponential(10) if crack_detected else 0,
                        'crack_opening_um': np.random.exponential(0.5) if crack_detected else 0
                    })
        else:
            # Single loading case
            strain_levels = np.linspace(0, 0.004, 50)
            for strain in strain_levels:
                for material, threshold in crack_thresholds.items():
                    # Check if strain exceeds threshold
                    if strain > threshold:
                        crack_prob = 1 - np.exp(-5 * (strain - threshold) / threshold)
                    else:
                        crack_prob = 0
                    
                    crack_detected = np.random.random() < crack_prob
                    
                    data_collection.append({
                        'applied_strain': strain,
                        'material': material,
                        'critical_strain_threshold': threshold,
                        'strain_ratio': strain / threshold,
                        'crack_probability': crack_prob,
                        'crack_detected': crack_detected,
                        'crack_length_um': np.random.exponential(15) if crack_detected else 0,
                        'crack_opening_um': np.random.exponential(0.8) if crack_detected else 0,
                        'temperature_C': 800 if experiment_type == 'startup_shutdown' else 1000
                    })
        
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, experiment_type,
                               'microcrack_threshold_data.csv'), index=False)
        
        return df
    
    def generate_diffraction_patterns(self, experiment_type):
        """Generate simulated XRD diffraction patterns"""
        print(f"  Generating XRD diffraction patterns for {experiment_type}...")
        
        two_theta = np.linspace(20, 80, 3000)
        patterns = {}
        
        for material_name, material_props in self.materials.items():
            intensity = np.zeros_like(two_theta)
            
            # Add peaks for each reflection
            for hkl, peak_pos in material_props['peaks']:
                # Peak shape (Pseudo-Voigt)
                sigma = 0.1 + np.random.normal(0, 0.02)  # Peak width
                peak_intensity = 100 * np.random.uniform(0.5, 1.0)
                
                # Gaussian component
                gaussian = peak_intensity * np.exp(-(two_theta - peak_pos)**2 / (2 * sigma**2))
                
                # Lorentzian component
                gamma = sigma
                lorentzian = peak_intensity * (gamma**2 / ((two_theta - peak_pos)**2 + gamma**2))
                
                # Pseudo-Voigt (mix of Gaussian and Lorentzian)
                eta = 0.5  # Mixing parameter
                intensity += eta * lorentzian + (1 - eta) * gaussian
            
            # Add background
            background = 10 + 0.1 * two_theta + np.random.normal(0, 1, len(two_theta))
            intensity += background
            
            # Add noise
            intensity += np.random.normal(0, np.sqrt(intensity))
            intensity[intensity < 0] = 0
            
            patterns[material_name] = {
                'two_theta': two_theta.tolist(),
                'intensity': intensity.tolist(),
                'peaks': material_props['peaks']
            }
        
        # Save patterns
        with open(os.path.join(self.base_path, experiment_type,
                               'xrd_patterns.json'), 'w') as f:
            json.dump(patterns, f, indent=2)
        
        return patterns
    
    def run_all(self):
        """Generate all XRD datasets"""
        for exp_type in ['sintering', 'thermal_cycling', 'startup_shutdown']:
            print(f"\nGenerating XRD data for {exp_type}...")
            self.generate_residual_stress_profile(exp_type)
            self.generate_lattice_strain_data(exp_type)
            self.generate_sin2psi_data(exp_type)
            self.generate_microcrack_data(exp_type)
            self.generate_diffraction_patterns(exp_type)
        
        print("\nXRD data generation complete!")

if __name__ == "__main__":
    generator = XRDDataGenerator()
    generator.run_all()