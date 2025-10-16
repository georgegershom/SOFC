#!/usr/bin/env python3
"""
In-The-Wild SOFC Plate Dataset Generator
==========================================
Generates realistic operational data for residual stress quantification from warped SOFC plates.

This script simulates:
- Production batches with natural variations
- Parameter drift over time (furnace aging, material batch changes)
- Realistic measurement noise and sensor artifacts
- Known failure modes (edge cracking, delamination, thermal shock)
- Environmental variations (temperature, humidity effects)
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from scipy import interpolate, ndimage
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class SOFCPlateSimulator:
    """Simulates realistic SOFC plate warping with production variations."""
    
    def __init__(self, plate_length=150.0, plate_width=150.0, 
                 grid_resolution=25, thickness=0.5):
        """
        Initialize SOFC plate simulator.
        
        Parameters:
        -----------
        plate_length : float (mm)
        plate_width : float (mm)
        grid_resolution : int (measurement points per dimension)
        thickness : float (mm)
        """
        self.length = plate_length
        self.width = plate_width
        self.resolution = grid_resolution
        self.thickness = thickness
        
        # Create measurement grid
        x = np.linspace(0, plate_length, grid_resolution)
        y = np.linspace(0, plate_width, grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Material properties (typical SOFC ceramics)
        self.youngs_modulus = 200e3  # MPa
        self.poisson_ratio = 0.3
        self.cte = 10.5e-6  # Coefficient of thermal expansion (1/K)
        self.sintering_temp = 1400  # °C
        
    def generate_base_stress_field(self, pattern_type='biaxial'):
        """Generate base residual stress field."""
        if pattern_type == 'biaxial':
            # Symmetric biaxial stress (common in uniform cooling)
            stress_xx = 50 + 30 * np.sin(2*np.pi*self.X/self.length) * \
                              np.cos(2*np.pi*self.Y/self.width)
            stress_yy = 50 + 30 * np.cos(2*np.pi*self.X/self.length) * \
                              np.sin(2*np.pi*self.Y/self.width)
                              
        elif pattern_type == 'gradient':
            # Thermal gradient during cooling
            stress_xx = 80 * (1 - self.X/self.length)
            stress_yy = 80 * (1 - self.Y/self.width)
            
        elif pattern_type == 'localized':
            # Localized stress concentrations
            center_x, center_y = self.length/2, self.width/2
            r = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
            stress_xx = 100 * np.exp(-r**2 / (self.length/4)**2)
            stress_yy = stress_xx.copy()
            
        elif pattern_type == 'edge_dominated':
            # High stress near edges (common failure mode)
            edge_dist_x = np.minimum(self.X, self.length - self.X)
            edge_dist_y = np.minimum(self.Y, self.width - self.Y)
            edge_dist = np.minimum(edge_dist_x, edge_dist_y)
            stress_xx = 120 * np.exp(-edge_dist / 10)
            stress_yy = stress_xx.copy()
            
        else:  # mixed
            # Complex mixed pattern
            stress_xx = (50 * np.sin(3*np.pi*self.X/self.length) + 
                        40 * np.cos(2*np.pi*self.Y/self.width) + 60)
            stress_yy = (50 * np.cos(3*np.pi*self.Y/self.width) + 
                        40 * np.sin(2*np.pi*self.X/self.length) + 60)
        
        return stress_xx, stress_yy
    
    def stress_to_warp(self, stress_xx, stress_yy, stress_xy=None):
        """
        Convert residual stress field to plate warping using plate theory.
        Simplified Kirchhoff-Love plate theory.
        """
        if stress_xy is None:
            stress_xy = np.zeros_like(stress_xx)
        
        # Flexural rigidity
        D = (self.youngs_modulus * self.thickness**3) / (12 * (1 - self.poisson_ratio**2))
        
        # Moment resultants from stress field
        M_xx = stress_xx * self.thickness**2 / 6
        M_yy = stress_yy * self.thickness**2 / 6
        M_xy = stress_xy * self.thickness**2 / 6
        
        # Curvature from moments
        kappa_xx = M_xx / D
        kappa_yy = M_yy / D
        kappa_xy = M_xy / D
        
        # Integrate curvature to get deflection
        # Using simplified integration (physics-inspired but numerically stable)
        dx = self.length / (self.resolution - 1)
        dy = self.width / (self.resolution - 1)
        
        # Second order integration
        warp = np.zeros_like(self.X)
        for i in range(self.resolution):
            for j in range(self.resolution):
                # Distance-weighted contribution
                x_contrib = kappa_xx[i, j] * (self.X - self.X[i, j])**2
                y_contrib = kappa_yy[i, j] * (self.Y - self.Y[i, j])**2
                warp += 0.5 * (x_contrib + y_contrib) / (self.resolution**2)
        
        # Normalize to physically reasonable deflections (0-5mm typical)
        warp = warp - np.min(warp)  # Reference bottom surface
        warp = warp / np.max(warp) * np.random.uniform(1.5, 4.5)
        
        return warp
    
    def add_production_variations(self, warp, batch_params):
        """Add production-related variations to warp."""
        # Thickness variations (affects stiffness locally)
        thickness_var = np.random.normal(0, 0.05, warp.shape)
        thickness_effect = warp * (1 + thickness_var)
        
        # Green density variations (affects sintering shrinkage)
        density_var = batch_params['density_variation']
        density_pattern = np.random.normal(1.0, density_var, warp.shape)
        density_pattern = ndimage.gaussian_filter(density_pattern, sigma=2)
        
        # Furnace position effect (temperature gradient in furnace)
        furnace_pos = batch_params['furnace_position']
        if furnace_pos == 'front':
            temp_gradient = 1.0 + 0.1 * (self.X / self.length)
        elif furnace_pos == 'back':
            temp_gradient = 1.0 - 0.1 * (self.X / self.length)
        elif furnace_pos == 'left':
            temp_gradient = 1.0 + 0.1 * (self.Y / self.width)
        elif furnace_pos == 'right':
            temp_gradient = 1.0 - 0.1 * (self.Y / self.width)
        else:  # center
            temp_gradient = 1.0
        
        warp_modified = thickness_effect * density_pattern * temp_gradient
        
        return warp_modified
    
    def add_failure_modes(self, warp, stress_xx, stress_yy, failure_type='none'):
        """Simulate effects of various failure modes on warp measurements."""
        failed = False
        failure_location = None
        
        if failure_type == 'edge_crack':
            # Cracks typically occur at high-stress regions near edges
            edge_stress = np.maximum(stress_xx, stress_yy)
            edge_mask = ((self.X < 10) | (self.X > self.length - 10) | 
                        (self.Y < 10) | (self.Y > self.width - 10))
            
            max_edge_stress = np.max(edge_stress * edge_mask)
            if max_edge_stress > 100:  # Critical stress threshold
                failed = True
                crack_location = np.unravel_index(
                    np.argmax(edge_stress * edge_mask), edge_stress.shape
                )
                
                # Crack relieves stress locally, creating discontinuity
                crack_x, crack_y = self.X[crack_location], self.Y[crack_location]
                r = np.sqrt((self.X - crack_x)**2 + (self.Y - crack_y)**2)
                crack_effect = 1.5 * np.exp(-r / 20)
                warp = warp + crack_effect * np.random.uniform(0.5, 1.5)
                failure_location = f"edge_crack_at_({crack_x:.1f},{crack_y:.1f})"
        
        elif failure_type == 'delamination':
            # Delamination occurs in multilayer structures
            if np.max(stress_xx) > 90 or np.max(stress_yy) > 90:
                failed = True
                # Creates a bubble-like local deformation
                delam_x = np.random.uniform(30, self.length - 30)
                delam_y = np.random.uniform(30, self.width - 30)
                r = np.sqrt((self.X - delam_x)**2 + (self.Y - delam_y)**2)
                delam_size = np.random.uniform(15, 30)
                delam_effect = 2.0 * np.exp(-r**2 / delam_size**2)
                warp = warp + delam_effect
                failure_location = f"delamination_at_({delam_x:.1f},{delam_y:.1f})"
        
        elif failure_type == 'thermal_shock':
            # Rapid cooling creates microcracks throughout
            if np.mean(stress_xx) > 60:
                failed = True
                # Creates irregular surface with many small features
                shock_pattern = np.random.normal(0, 0.3, warp.shape)
                shock_pattern = ndimage.gaussian_filter(shock_pattern, sigma=1.5)
                warp = warp + shock_pattern
                failure_location = "distributed_thermal_shock"
        
        return warp, failed, failure_location
    
    def add_measurement_noise(self, warp, noise_params):
        """Add realistic measurement noise and sensor artifacts."""
        # Gaussian measurement noise
        gaussian_noise = np.random.normal(0, noise_params['gaussian_std'], warp.shape)
        
        # Systematic sensor bias (drift during scanning)
        if noise_params['sensor_drift']:
            drift_x = np.linspace(0, noise_params['drift_magnitude'], 
                                 self.resolution)
            drift_pattern = np.outer(drift_x, np.ones(self.resolution))
            gaussian_noise += drift_pattern
        
        # Outliers (dust particles, sensor glitches)
        if noise_params['outliers']:
            n_outliers = int(self.resolution**2 * noise_params['outlier_rate'])
            outlier_positions = np.random.choice(
                self.resolution**2, n_outliers, replace=False
            )
            outlier_values = np.random.uniform(-0.5, 0.5, n_outliers)
            flat_noise = gaussian_noise.flatten()
            flat_noise[outlier_positions] += outlier_values
            gaussian_noise = flat_noise.reshape(warp.shape)
        
        # Missing data points (edge detection failures)
        if noise_params['missing_data']:
            n_missing = int(self.resolution**2 * noise_params['missing_rate'])
            missing_positions = np.random.choice(
                self.resolution**2, n_missing, replace=False
            )
            warp_with_missing = warp.copy()
            flat_warp = warp_with_missing.flatten()
            flat_warp[missing_positions] = np.nan
            warp_with_missing = flat_warp.reshape(warp.shape)
        else:
            warp_with_missing = warp
        
        return warp_with_missing + gaussian_noise


class ProductionSimulator:
    """Simulates a production environment with parameter drift."""
    
    def __init__(self, start_date='2023-01-01', n_batches=50):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.n_batches = n_batches
        self.current_batch = 0
        
        # Furnace aging parameters
        self.furnace_age = 0  # days
        self.furnace_degradation_rate = 0.001  # per day
        
        # Material batch tracking
        self.material_batches = self._generate_material_batches()
        
    def _generate_material_batches(self):
        """Generate material batch characteristics."""
        batches = []
        for i in range(10):  # 10 different material batches
            batch = {
                'id': f'MAT-2023-{i+1:03d}',
                'powder_size_mean': np.random.uniform(0.8, 1.2),  # μm
                'powder_size_std': np.random.uniform(0.1, 0.3),
                'purity': np.random.uniform(99.5, 99.9),  # %
                'green_density': np.random.uniform(0.55, 0.65),
            }
            batches.append(batch)
        return batches
    
    def get_batch_params(self, batch_idx):
        """Get production parameters for a specific batch."""
        batch_date = self.start_date + timedelta(days=batch_idx * 7)
        self.furnace_age = (batch_date - self.start_date).days
        
        # Select material batch (changes every 5 production batches)
        material_batch = self.material_batches[batch_idx // 5 % len(self.material_batches)]
        
        # Furnace temperature drift due to aging
        nominal_temp = 1400  # °C
        temp_drift = -self.furnace_age * self.furnace_degradation_rate
        actual_temp = nominal_temp + temp_drift + np.random.normal(0, 5)
        
        # Cooling rate variations (affects stress)
        nominal_cooling = 2.0  # °C/min
        cooling_rate = nominal_cooling + np.random.normal(0, 0.3)
        
        # Atmospheric control variations
        oxygen_partial_pressure = np.random.uniform(0.18, 0.22)  # atm
        humidity = np.random.uniform(30, 60)  # %
        
        # Position in furnace (affects temperature uniformity)
        positions = ['front', 'center', 'back', 'left', 'right']
        furnace_position = np.random.choice(positions)
        
        # Operator and shift effects
        shifts = ['morning', 'afternoon', 'night']
        shift = shifts[batch_idx % 3]
        operator_id = f'OP-{np.random.randint(1, 6):02d}'
        
        params = {
            'batch_id': f'BATCH-{batch_idx+1:04d}',
            'date': batch_date.strftime('%Y-%m-%d'),
            'material_batch': material_batch['id'],
            'material_properties': material_batch,
            'furnace_temp': actual_temp,
            'cooling_rate': cooling_rate,
            'oxygen_pressure': oxygen_partial_pressure,
            'humidity': humidity,
            'furnace_position': furnace_position,
            'furnace_age_days': self.furnace_age,
            'shift': shift,
            'operator': operator_id,
            'density_variation': material_batch['powder_size_std'] / material_batch['powder_size_mean'],
        }
        
        return params


def generate_complete_dataset(n_plates=300, output_dir='in_the_wild_dataset'):
    """Generate complete in-the-wild dataset."""
    
    print("=" * 80)
    print("GENERATING IN-THE-WILD SOFC PLATE DATASET")
    print("=" * 80)
    print(f"Target: {n_plates} plates with production variations")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/measurements', exist_ok=True)
    os.makedirs(f'{output_dir}/stress_fields', exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
    
    # Initialize simulators
    plate_sim = SOFCPlateSimulator(
        plate_length=150, 
        plate_width=150, 
        grid_resolution=25,
        thickness=0.5
    )
    
    production_sim = ProductionSimulator(
        start_date='2023-01-01',
        n_batches=n_plates // 6  # 6 plates per batch average
    )
    
    # Stress pattern distribution (reflecting real production)
    pattern_types = ['biaxial'] * 150 + ['gradient'] * 80 + \
                   ['edge_dominated'] * 40 + ['localized'] * 20 + ['mixed'] * 10
    np.random.shuffle(pattern_types)
    
    # Failure mode probabilities (realistic failure rates)
    failure_modes = ['none'] * 260 + ['edge_crack'] * 25 + \
                    ['delamination'] * 10 + ['thermal_shock'] * 5
    np.random.shuffle(failure_modes)
    
    # Storage for dataset
    dataset_records = []
    
    # Generate plates
    for i in range(n_plates):
        print(f"Generating plate {i+1}/{n_plates}...", end='\r')
        
        # Get batch parameters with drift
        batch_idx = i // 6
        batch_params = production_sim.get_batch_params(batch_idx)
        
        # Generate base stress field
        pattern_type = pattern_types[i]
        stress_xx, stress_yy = plate_sim.generate_base_stress_field(pattern_type)
        
        # Add temperature-dependent variations
        temp_factor = (batch_params['furnace_temp'] - 1400) / 100
        stress_xx *= (1 + temp_factor * 0.1)
        stress_yy *= (1 + temp_factor * 0.1)
        
        # Add cooling rate effects
        cooling_factor = (batch_params['cooling_rate'] - 2.0) / 2.0
        stress_xx *= (1 + cooling_factor * 0.15)
        stress_yy *= (1 + cooling_factor * 0.15)
        
        # Convert stress to warp
        warp = plate_sim.stress_to_warp(stress_xx, stress_yy)
        
        # Add production variations
        warp = plate_sim.add_production_variations(warp, batch_params)
        
        # Add failure modes
        failure_type = failure_modes[i]
        warp, failed, failure_location = plate_sim.add_failure_modes(
            warp, stress_xx, stress_yy, failure_type
        )
        
        # Measurement noise parameters (varies by operator/shift)
        if batch_params['shift'] == 'night':
            noise_level = 1.5  # Night shift has slightly worse measurements
        else:
            noise_level = 1.0
        
        noise_params = {
            'gaussian_std': 0.02 * noise_level,
            'sensor_drift': np.random.random() < 0.15,  # 15% chance
            'drift_magnitude': np.random.uniform(0.01, 0.05),
            'outliers': np.random.random() < 0.20,  # 20% chance
            'outlier_rate': 0.005,
            'missing_data': np.random.random() < 0.10,  # 10% chance
            'missing_rate': 0.02,
        }
        
        # Add measurement noise
        warp_measured = plate_sim.add_measurement_noise(warp, noise_params)
        
        # Calculate quality metrics
        max_warp = np.nanmax(warp_measured)
        mean_warp = np.nanmean(warp_measured)
        warp_std = np.nanstd(warp_measured)
        max_stress = max(np.max(stress_xx), np.max(stress_yy))
        
        # Determine quality classification
        if failed:
            quality = 'failed'
        elif max_warp > 4.0 or max_stress > 110:
            quality = 'reject'
        elif max_warp > 3.0 or max_stress > 90:
            quality = 'marginal'
        else:
            quality = 'good'
        
        # Record metadata
        record = {
            'plate_id': f'PLATE-{i+1:05d}',
            'batch_id': batch_params['batch_id'],
            'date': batch_params['date'],
            'material_batch': batch_params['material_batch'],
            'furnace_temp_C': batch_params['furnace_temp'],
            'cooling_rate_C_per_min': batch_params['cooling_rate'],
            'oxygen_pressure_atm': batch_params['oxygen_pressure'],
            'humidity_percent': batch_params['humidity'],
            'furnace_position': batch_params['furnace_position'],
            'furnace_age_days': batch_params['furnace_age_days'],
            'shift': batch_params['shift'],
            'operator': batch_params['operator'],
            'stress_pattern': pattern_type,
            'max_warp_mm': float(max_warp),
            'mean_warp_mm': float(mean_warp),
            'warp_std_mm': float(warp_std),
            'max_stress_MPa': float(max_stress),
            'failed': failed,
            'failure_type': failure_type if failed else 'none',
            'failure_location': failure_location if failed else 'none',
            'quality_class': quality,
            'measurement_file': f'measurements/plate_{i+1:05d}_warp.csv',
            'stress_file': f'stress_fields/plate_{i+1:05d}_stress.npz',
        }
        dataset_records.append(record)
        
        # Save warp measurements
        warp_df = pd.DataFrame({
            'x_mm': plate_sim.X.flatten(),
            'y_mm': plate_sim.Y.flatten(),
            'z_mm': warp_measured.flatten(),
        })
        warp_df.to_csv(f'{output_dir}/measurements/plate_{i+1:05d}_warp.csv', 
                      index=False)
        
        # Save stress fields (ground truth)
        np.savez(f'{output_dir}/stress_fields/plate_{i+1:05d}_stress.npz',
                stress_xx=stress_xx,
                stress_yy=stress_yy,
                warp_true=warp,
                warp_measured=warp_measured,
                X=plate_sim.X,
                Y=plate_sim.Y)
        
        # Create visualization for sample plates
        if i % 30 == 0 or failed:  # Visualize every 30th plate and all failures
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Warp measurement
            im1 = axes[0, 0].contourf(plate_sim.X, plate_sim.Y, warp_measured, 
                                     levels=20, cmap='RdYlBu_r')
            axes[0, 0].set_title(f'Warp Measurement - {record["plate_id"]}')
            axes[0, 0].set_xlabel('X (mm)')
            axes[0, 0].set_ylabel('Y (mm)')
            plt.colorbar(im1, ax=axes[0, 0], label='Warp (mm)')
            
            # Stress XX
            im2 = axes[0, 1].contourf(plate_sim.X, plate_sim.Y, stress_xx, 
                                     levels=20, cmap='plasma')
            axes[0, 1].set_title('Residual Stress σ_xx')
            axes[0, 1].set_xlabel('X (mm)')
            axes[0, 1].set_ylabel('Y (mm)')
            plt.colorbar(im2, ax=axes[0, 1], label='Stress (MPa)')
            
            # Stress YY
            im3 = axes[1, 0].contourf(plate_sim.X, plate_sim.Y, stress_yy, 
                                     levels=20, cmap='plasma')
            axes[1, 0].set_title('Residual Stress σ_yy')
            axes[1, 0].set_xlabel('X (mm)')
            axes[1, 0].set_ylabel('Y (mm)')
            plt.colorbar(im3, ax=axes[1, 0], label='Stress (MPa)')
            
            # Info panel
            axes[1, 1].axis('off')
            info_text = f"""
Plate ID: {record['plate_id']}
Batch: {record['batch_id']}
Date: {record['date']}
Material: {record['material_batch']}

Process Parameters:
  Furnace Temp: {batch_params['furnace_temp']:.1f} °C
  Cooling Rate: {batch_params['cooling_rate']:.2f} °C/min
  Position: {batch_params['furnace_position']}
  Age: {batch_params['furnace_age_days']} days

Measurements:
  Max Warp: {max_warp:.3f} mm
  Mean Warp: {mean_warp:.3f} mm
  Max Stress: {max_stress:.1f} MPa

Quality: {quality.upper()}
Failed: {failed}
{f'Failure: {failure_location}' if failed else ''}
            """
            axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, 
                          verticalalignment='center', family='monospace')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/visualizations/plate_{i+1:05d}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"\nGenerated {n_plates} plates successfully!")
    
    # Save metadata
    metadata_df = pd.DataFrame(dataset_records)
    metadata_df.to_csv(f'{output_dir}/dataset_metadata.csv', index=False)
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal plates: {n_plates}")
    print(f"Date range: {metadata_df['date'].min()} to {metadata_df['date'].max()}")
    print(f"\nQuality Distribution:")
    print(metadata_df['quality_class'].value_counts())
    
    print(f"\nFailure Distribution:")
    print(metadata_df['failure_type'].value_counts())
    
    print(f"\nStress Pattern Distribution:")
    print(metadata_df['stress_pattern'].value_counts())
    
    print(f"\nWarp Statistics:")
    print(f"  Mean: {metadata_df['mean_warp_mm'].mean():.3f} ± {metadata_df['mean_warp_mm'].std():.3f} mm")
    print(f"  Max (across all): {metadata_df['max_warp_mm'].max():.3f} mm")
    print(f"  Min (across all): {metadata_df['max_warp_mm'].min():.3f} mm")
    
    print(f"\nStress Statistics:")
    print(f"  Mean Max Stress: {metadata_df['max_stress_MPa'].mean():.1f} ± {metadata_df['max_stress_MPa'].std():.1f} MPa")
    print(f"  Max (across all): {metadata_df['max_stress_MPa'].max():.1f} MPa")
    
    print(f"\nProduction Parameters:")
    print(f"  Furnace temperature: {metadata_df['furnace_temp_C'].mean():.1f} ± {metadata_df['furnace_temp_C'].std():.1f} °C")
    print(f"  Cooling rate: {metadata_df['cooling_rate_C_per_min'].mean():.2f} ± {metadata_df['cooling_rate_C_per_min'].std():.2f} °C/min")
    print(f"  Number of batches: {metadata_df['batch_id'].nunique()}")
    print(f"  Number of material batches: {metadata_df['material_batch'].nunique()}")
    print(f"  Number of operators: {metadata_df['operator'].nunique()}")
    
    # Generate summary visualizations
    generate_summary_plots(metadata_df, output_dir)
    
    # Generate detailed documentation
    generate_documentation(metadata_df, output_dir, plate_sim)
    
    print(f"\n{'=' * 80}")
    print(f"Dataset saved to: {output_dir}/")
    print(f"{'=' * 80}\n")
    
    return metadata_df


def generate_summary_plots(metadata_df, output_dir):
    """Generate summary visualizations of the dataset."""
    
    print("\nGenerating summary visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Warp distribution over time
    ax1 = plt.subplot(3, 3, 1)
    metadata_df['date'] = pd.to_datetime(metadata_df['date'])
    ax1.scatter(metadata_df['date'], metadata_df['max_warp_mm'], 
               c=metadata_df['furnace_age_days'], cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Production Date')
    ax1.set_ylabel('Max Warp (mm)')
    ax1.set_title('Warp Evolution Over Time')
    plt.colorbar(ax1.collections[0], ax=ax1, label='Furnace Age (days)')
    
    # 2. Stress vs Warp
    ax2 = plt.subplot(3, 3, 2)
    scatter = ax2.scatter(metadata_df['max_stress_MPa'], metadata_df['max_warp_mm'],
                         c=metadata_df['quality_class'].map({
                             'good': 0, 'marginal': 1, 'reject': 2, 'failed': 3
                         }), cmap='RdYlGn_r', alpha=0.6)
    ax2.set_xlabel('Max Stress (MPa)')
    ax2.set_ylabel('Max Warp (mm)')
    ax2.set_title('Stress-Warp Relationship')
    plt.colorbar(scatter, ax=ax2, label='Quality', 
                ticks=[0, 1, 2, 3], 
                format=plt.FuncFormatter(lambda x, p: ['Good', 'Marginal', 'Reject', 'Failed'][int(x)]))
    
    # 3. Quality distribution
    ax3 = plt.subplot(3, 3, 3)
    quality_counts = metadata_df['quality_class'].value_counts()
    colors = {'good': 'green', 'marginal': 'yellow', 'reject': 'orange', 'failed': 'red'}
    ax3.bar(quality_counts.index, quality_counts.values, 
           color=[colors[q] for q in quality_counts.index])
    ax3.set_ylabel('Count')
    ax3.set_title('Quality Distribution')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Furnace temperature drift
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(metadata_df['furnace_age_days'], metadata_df['furnace_temp_C'], 
               alpha=0.5)
    z = np.polyfit(metadata_df['furnace_age_days'], metadata_df['furnace_temp_C'], 1)
    p = np.poly1d(z)
    ax4.plot(metadata_df['furnace_age_days'], 
            p(metadata_df['furnace_age_days']), 
            "r--", linewidth=2, label=f'Trend: {z[0]:.4f}°C/day')
    ax4.set_xlabel('Furnace Age (days)')
    ax4.set_ylabel('Furnace Temperature (°C)')
    ax4.set_title('Furnace Temperature Drift')
    ax4.legend()
    
    # 5. Stress pattern distribution
    ax5 = plt.subplot(3, 3, 5)
    pattern_counts = metadata_df['stress_pattern'].value_counts()
    ax5.barh(pattern_counts.index, pattern_counts.values)
    ax5.set_xlabel('Count')
    ax5.set_title('Stress Pattern Distribution')
    
    # 6. Failure modes
    ax6 = plt.subplot(3, 3, 6)
    failure_counts = metadata_df['failure_type'].value_counts()
    ax6.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
           colors=['lightgreen', 'red', 'orange', 'yellow'])
    ax6.set_title('Failure Mode Distribution')
    
    # 7. Shift effect on quality
    ax7 = plt.subplot(3, 3, 7)
    shift_quality = pd.crosstab(metadata_df['shift'], metadata_df['quality_class'])
    shift_quality.plot(kind='bar', stacked=True, ax=ax7, 
                      color=['green', 'yellow', 'orange', 'red'])
    ax7.set_xlabel('Shift')
    ax7.set_ylabel('Count')
    ax7.set_title('Quality by Production Shift')
    ax7.legend(title='Quality', bbox_to_anchor=(1.05, 1))
    ax7.tick_params(axis='x', rotation=0)
    
    # 8. Furnace position effect
    ax8 = plt.subplot(3, 3, 8)
    position_warp = metadata_df.groupby('furnace_position')['max_warp_mm'].mean().sort_values()
    ax8.barh(position_warp.index, position_warp.values, color='steelblue')
    ax8.set_xlabel('Mean Max Warp (mm)')
    ax8.set_title('Furnace Position Effect on Warp')
    
    # 9. Cooling rate effect
    ax9 = plt.subplot(3, 3, 9)
    ax9.scatter(metadata_df['cooling_rate_C_per_min'], metadata_df['max_stress_MPa'],
               c=metadata_df['max_warp_mm'], cmap='coolwarm', alpha=0.6)
    ax9.set_xlabel('Cooling Rate (°C/min)')
    ax9.set_ylabel('Max Stress (MPa)')
    ax9.set_title('Cooling Rate Effect')
    plt.colorbar(ax9.collections[0], ax=ax9, label='Max Warp (mm)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Summary visualizations saved!")


def generate_documentation(metadata_df, output_dir, plate_sim):
    """Generate comprehensive documentation for the dataset."""
    
    doc = f"""
# In-The-Wild SOFC Plate Dataset Documentation

## Dataset Overview

This dataset contains {len(metadata_df)} SOFC (Solid Oxide Fuel Cell) plate measurements 
collected from simulated production operations spanning from {metadata_df['date'].min()} 
to {metadata_df['date'].max()}.

**Purpose**: Demonstrate ML model robustness on real-world operational data with natural 
variations, measurement noise, and parameter drift.

## Dataset Characteristics

### Physical Properties
- Plate dimensions: {plate_sim.length} mm × {plate_sim.width} mm
- Nominal thickness: {plate_sim.thickness} mm
- Material: Ceramic (typical SOFC materials: YSZ, LSM, NiO-YSZ)
- Young's Modulus: {plate_sim.youngs_modulus/1000:.0f} GPa
- Poisson's Ratio: {plate_sim.poisson_ratio}
- CTE: {plate_sim.cte*1e6:.1f} × 10⁻⁶ K⁻¹

### Measurement Grid
- Resolution: {plate_sim.resolution} × {plate_sim.resolution} points
- Measurement type: Surface profilometry (warp/deflection)
- Coordinate system: Cartesian (X, Y, Z) where Z is out-of-plane deflection

## Production Process Simulation

### Nominal Process Parameters
- Sintering temperature: 1400°C
- Cooling rate: 2.0 °C/min
- Oxygen partial pressure: 0.21 atm
- Ambient humidity: 30-60%

### Sources of Variation

1. **Furnace Aging Effect**
   - Temperature drift rate: -0.001 °C/day
   - Maximum age in dataset: {metadata_df['furnace_age_days'].max()} days
   - Total temperature drift: ~{metadata_df['furnace_age_days'].max() * 0.001:.1f} °C

2. **Material Batch Variations**
   - Number of material batches: {metadata_df['material_batch'].nunique()}
   - Powder size variations: 0.8-1.2 μm
   - Purity range: 99.5-99.9%
   - Green density range: 0.55-0.65

3. **Process Variations**
   - Temperature variations: ±5°C (random)
   - Cooling rate variations: ±0.3 °C/min
   - Furnace position effects: 5 zones (front, back, left, right, center)
   - Shift effects: 3 shifts (morning, afternoon, night)

4. **Measurement Noise**
   - Gaussian noise: σ = 0.02 mm (baseline)
   - Sensor drift: 15% of measurements affected
   - Outliers: 20% of measurements contain outliers (0.5% of points)
   - Missing data: 10% of measurements have missing points (2% of grid)

## Stress Patterns

The dataset includes 5 types of residual stress patterns:

1. **Biaxial** ({len(metadata_df[metadata_df['stress_pattern']=='biaxial'])} plates): 
   Symmetric stress from uniform cooling
   
2. **Gradient** ({len(metadata_df[metadata_df['stress_pattern']=='gradient'])} plates): 
   Thermal gradients during cooling
   
3. **Edge-Dominated** ({len(metadata_df[metadata_df['stress_pattern']=='edge_dominated'])} plates): 
   High stress near edges (common failure precursor)
   
4. **Localized** ({len(metadata_df[metadata_df['stress_pattern']=='localized'])} plates): 
   Localized stress concentrations
   
5. **Mixed** ({len(metadata_df[metadata_df['stress_pattern']=='mixed'])} plates): 
   Complex multi-mode patterns

## Failure Modes

The dataset includes realistic failure modes observed in SOFC production:

1. **Edge Cracking** ({len(metadata_df[metadata_df['failure_type']=='edge_crack'])} cases):
   - Occurs when edge stress exceeds 100 MPa
   - Creates local stress relief and warp discontinuities
   - Most common failure mode

2. **Delamination** ({len(metadata_df[metadata_df['failure_type']=='delamination'])} cases):
   - Occurs in multilayer structures under high stress
   - Creates localized bubble-like deformations
   - Serious structural failure

3. **Thermal Shock** ({len(metadata_df[metadata_df['failure_type']=='thermal_shock'])} cases):
   - Results from rapid cooling
   - Distributed microcracks throughout structure
   - Irregular surface features

4. **No Failure** ({len(metadata_df[metadata_df['failure_type']=='none'])} plates):
   - Normal production within acceptable limits

## Quality Classification

Plates are classified into 4 quality categories:

- **Good** ({len(metadata_df[metadata_df['quality_class']=='good'])} plates): 
  Max warp < 3.0 mm, max stress < 90 MPa, no failures
  
- **Marginal** ({len(metadata_df[metadata_df['quality_class']=='marginal'])} plates): 
  Max warp 3.0-4.0 mm or max stress 90-110 MPa
  
- **Reject** ({len(metadata_df[metadata_df['quality_class']=='reject'])} plates): 
  Max warp > 4.0 mm or max stress > 110 MPa, but not failed
  
- **Failed** ({len(metadata_df[metadata_df['quality_class']=='failed'])} plates): 
  Physical failure (cracks, delamination, etc.)

## File Structure

```
{output_dir}/
├── dataset_metadata.csv          # Complete metadata for all plates
├── dataset_summary.png            # Summary visualizations
├── README.md                      # This file
├── measurements/                  # Warp measurement data (CSV)
│   ├── plate_00001_warp.csv
│   ├── plate_00002_warp.csv
│   └── ...
├── stress_fields/                 # Ground truth stress fields (NPZ)
│   ├── plate_00001_stress.npz
│   ├── plate_00002_stress.npz
│   └── ...
└── visualizations/                # Sample visualizations
    ├── plate_00001.png
    ├── plate_00031.png
    └── ...
```

## Data Files

### 1. Measurement Files (CSV)
Each `plate_XXXXX_warp.csv` contains:
- `x_mm`: X coordinate (mm)
- `y_mm`: Y coordinate (mm)
- `z_mm`: Z deflection/warp (mm) [may contain NaN for missing data]

### 2. Stress Field Files (NPZ)
Each `plate_XXXXX_stress.npz` contains:
- `stress_xx`: Residual stress in X direction (MPa)
- `stress_yy`: Residual stress in Y direction (MPa)
- `warp_true`: True warp field without noise (mm)
- `warp_measured`: Measured warp with noise (mm)
- `X`: X coordinate meshgrid (mm)
- `Y`: Y coordinate meshgrid (mm)

### 3. Metadata File (CSV)
`dataset_metadata.csv` contains comprehensive information:
- Plate identification (plate_id, batch_id, date)
- Material tracking (material_batch)
- Process parameters (furnace_temp_C, cooling_rate_C_per_min, etc.)
- Production context (furnace_position, furnace_age_days, shift, operator)
- Stress characteristics (stress_pattern, max_stress_MPa)
- Warp measurements (max_warp_mm, mean_warp_mm, warp_std_mm)
- Quality assessment (quality_class, failed, failure_type, failure_location)
- File references (measurement_file, stress_file)

## Usage Examples

### Loading a Single Plate

```python
import pandas as pd
import numpy as np

# Load metadata
metadata = pd.read_csv('in_the_wild_dataset/dataset_metadata.csv')

# Select a plate
plate_info = metadata.iloc[0]

# Load warp measurements
warp_data = pd.read_csv(f"in_the_wild_dataset/{{plate_info['measurement_file']}}")

# Load ground truth stress
stress_data = np.load(f"in_the_wild_dataset/{{plate_info['stress_file']}}")
stress_xx = stress_data['stress_xx']
stress_yy = stress_data['stress_yy']
warp_true = stress_data['warp_true']
```

### Filtering by Quality

```python
# Get only good quality plates
good_plates = metadata[metadata['quality_class'] == 'good']

# Get failed plates  
failed_plates = metadata[metadata['failed'] == True]

# Get plates from specific batch
batch_plates = metadata[metadata['batch_id'] == 'BATCH-0001']
```

### Analyzing Parameter Drift

```python
import matplotlib.pyplot as plt

# Plot furnace temperature drift
plt.figure(figsize=(10, 6))
plt.scatter(metadata['furnace_age_days'], metadata['furnace_temp_C'])
plt.xlabel('Furnace Age (days)')
plt.ylabel('Temperature (°C)')
plt.title('Furnace Temperature Drift Over Time')
plt.show()

# Analyze correlation with quality
print(metadata.groupby('quality_class')['furnace_temp_C'].describe())
```

## Statistical Summary

### Warp Statistics
- Mean warp (across all plates): {metadata_df['mean_warp_mm'].mean():.3f} ± {metadata_df['mean_warp_mm'].std():.3f} mm
- Maximum warp observed: {metadata_df['max_warp_mm'].max():.3f} mm
- Minimum warp observed: {metadata_df['max_warp_mm'].min():.3f} mm

### Stress Statistics
- Mean maximum stress: {metadata_df['max_stress_MPa'].mean():.1f} ± {metadata_df['max_stress_MPa'].std():.1f} MPa
- Maximum stress observed: {metadata_df['max_stress_MPa'].max():.1f} MPa
- Minimum stress observed: {metadata_df['max_stress_MPa'].min():.1f} MPa

### Production Statistics
- Total production batches: {metadata_df['batch_id'].nunique()}
- Material batches used: {metadata_df['material_batch'].nunique()}
- Operators involved: {metadata_df['operator'].nunique()}
- Production span: {(metadata_df['date'].max() - metadata_df['date'].min()).days} days

## Use Cases for ML Model Testing

### 1. Robustness to Noise
Test model performance on noisy measurements with outliers and missing data.

### 2. Domain Adaptation
Train on early production data, test on later data with parameter drift.

### 3. Failure Prediction
Predict failure modes from warp measurements and process parameters.

### 4. Quality Classification
Classify plates into quality categories based on measurements.

### 5. Stress Reconstruction
Inverse problem: reconstruct stress fields from warp measurements.

### 6. Process Optimization
Identify optimal process parameters for minimal warp/stress.

### 7. Anomaly Detection
Detect unusual patterns that deviate from normal production.

## Physical Validity Checks

When validating ML predictions, ensure:

1. **Stress Magnitude**: Typical range 20-150 MPa for SOFC ceramics
2. **Warp Magnitude**: Typical range 0-5 mm for 150mm plates
3. **Edge Effects**: Higher stress near edges (boundary conditions)
4. **Symmetry**: Patterns should respect plate symmetry unless there's asymmetric loading
5. **Smoothness**: Stress fields should be continuous except at crack locations
6. **Energy Consistency**: Total strain energy should be physically reasonable

## Known Limitations

1. This is simulated data based on physics models, not actual experimental measurements
2. Some simplifications in plate theory (Kirchhoff-Love assumptions)
3. Failure modes are idealized representations
4. Material properties assumed homogeneous except for batch variations
5. No microstructure effects or grain boundary effects included

## Citation

If you use this dataset, please cite:
```
In-The-Wild SOFC Plate Dataset for ML-Augmented Inverse Modeling
Generated: {datetime.now().strftime('%Y-%m-%d')}
Purpose: Residual Stress Quantification from Warped SOFC Plates
```

## Contact & Support

This dataset was generated for research purposes in ML-augmented inverse modeling
for residual stress quantification in SOFC manufacturing.

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Version: 1.0
"""
    
    with open(f'{output_dir}/README.md', 'w') as f:
        f.write(doc)
    
    print("Documentation generated!")


if __name__ == '__main__':
    # Generate the dataset
    print("\n" + "🔬" * 40)
    print("IN-THE-WILD SOFC PLATE DATASET GENERATOR")
    print("🔬" * 40 + "\n")
    
    # Generate 300 plates (realistic production dataset size)
    dataset = generate_complete_dataset(
        n_plates=300,
        output_dir='in_the_wild_dataset'
    )
    
    print("\n✅ Dataset generation complete!")
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Explore dataset_metadata.csv for overview")
    print("2. Check dataset_summary.png for visual summary")
    print("3. Read README.md for detailed documentation")
    print("4. Load individual plate data from measurements/ and stress_fields/")
    print("5. Use this data to test ML model robustness!")
    print("=" * 80 + "\n")
