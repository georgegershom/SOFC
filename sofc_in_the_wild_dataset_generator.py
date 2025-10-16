#!/usr/bin/env python3
"""
SOFC "In-The-Wild" Operational Dataset Generator
==============================================

Generates realistic warp measurement data from SOFC plates produced in industrial settings.
This dataset simulates real production environments with:
- Natural manufacturing variations
- Measurement noise and uncertainties
- Parameter drift over time
- Known failure modes and stress patterns
- Environmental factors and aging effects

For: ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, ndimage
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class SOFCInTheWildDatasetGenerator:
    """
    Generates comprehensive "In-The-Wild" SOFC plate warp measurement dataset
    """
    
    def __init__(self, n_plates=500, seed=42):
        """
        Initialize the dataset generator
        
        Args:
            n_plates: Number of plates to generate (hundreds for realistic dataset)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_plates = n_plates
        self.plate_dimensions = (150, 150)  # mm, typical SOFC plate size
        self.measurement_grid = (31, 31)    # High-resolution measurement grid
        
        # Manufacturing parameters with realistic ranges
        self.manufacturing_params = {
            'sintering_temp': {'nominal': 1400, 'std': 15, 'drift_rate': 0.1},  # °C
            'sintering_time': {'nominal': 4.0, 'std': 0.2, 'drift_rate': 0.01}, # hours
            'cooling_rate': {'nominal': 2.0, 'std': 0.3, 'drift_rate': 0.02},   # °C/min
            'green_density': {'nominal': 0.55, 'std': 0.02, 'drift_rate': 0.0001},
            'humidity': {'nominal': 45, 'std': 8, 'drift_rate': 0.05},           # %
            'furnace_position': {'nominal': 0, 'std': 1, 'drift_rate': 0},      # categorical
        }
        
        # Material properties (temperature-dependent)
        self.material_props = {
            'youngs_modulus': 200e9,     # Pa
            'poisson_ratio': 0.3,
            'thermal_expansion': 12e-6,   # /K
            'density': 6000,             # kg/m³
            'thickness': 0.5e-3,         # m
        }
        
        # Measurement system parameters
        self.measurement_params = {
            'laser_precision': 2e-6,     # m (2 μm)
            'systematic_error': 5e-6,    # m (5 μm)
            'temperature_drift': 1e-6,   # m/°C
            'vibration_noise': 0.5e-6,   # m
        }
        
        # Initialize time-dependent parameters
        self.start_date = datetime(2023, 1, 1)
        self.production_schedule = self._generate_production_schedule()
        
    def _generate_production_schedule(self):
        """Generate realistic production schedule with shifts, maintenance, etc."""
        schedule = []
        current_date = self.start_date
        
        for i in range(self.n_plates):
            # Skip weekends and add maintenance downtime
            while current_date.weekday() >= 5:  # Weekend
                current_date += timedelta(days=1)
            
            # Add random production delays
            if np.random.random() < 0.05:  # 5% chance of maintenance
                current_date += timedelta(hours=np.random.exponential(8))
            
            schedule.append({
                'plate_id': f'SOFC_{i+1:04d}',
                'production_date': current_date,
                'shift': 'A' if current_date.hour < 8 else 'B' if current_date.hour < 16 else 'C',
                'batch_id': f'B{(i//20)+1:03d}',  # 20 plates per batch
            })
            
            # Normal production interval (30-90 minutes per plate)
            current_date += timedelta(minutes=np.random.uniform(30, 90))
            
        return schedule
    
    def _generate_base_stress_field(self, manufacturing_params):
        """
        Generate base residual stress field based on manufacturing parameters
        """
        x = np.linspace(0, self.plate_dimensions[0], self.measurement_grid[0])
        y = np.linspace(0, self.plate_dimensions[1], self.measurement_grid[1])
        X, Y = np.meshgrid(x, y)
        
        # Center coordinates
        cx, cy = self.plate_dimensions[0]/2, self.plate_dimensions[1]/2
        
        # Distance from center and edges
        r_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        r_edge = np.minimum(np.minimum(X, self.plate_dimensions[0] - X),
                           np.minimum(Y, self.plate_dimensions[1] - Y))
        
        # Base stress components influenced by manufacturing
        temp_factor = (manufacturing_params['sintering_temp'] - 1400) / 100
        time_factor = (manufacturing_params['sintering_time'] - 4.0) / 2.0
        cooling_factor = (manufacturing_params['cooling_rate'] - 2.0) / 1.0
        density_factor = (manufacturing_params['green_density'] - 0.55) / 0.1
        
        # Thermal stress pattern (higher at center, influenced by cooling rate)
        thermal_stress = (50e6 + temp_factor * 20e6) * (1 - r_center / (cx * 1.2))
        thermal_stress *= (1 + cooling_factor * 0.3)
        
        # Sintering stress (edge effects, influenced by time and density)
        sintering_stress = (30e6 + time_factor * 10e6) * np.exp(-r_edge / 20)
        sintering_stress *= (1 + density_factor * 0.4)
        
        # Furnace position effects (asymmetric heating)
        position_effect = manufacturing_params['furnace_position'] * 5e6
        furnace_gradient = position_effect * (X - cx) / cx
        
        # Combine stress components
        total_stress = thermal_stress + sintering_stress + furnace_gradient
        
        # Add realistic stress concentrations and patterns
        total_stress = self._add_stress_concentrations(total_stress, X, Y, manufacturing_params)
        
        return total_stress
    
    def _add_stress_concentrations(self, base_stress, X, Y, params):
        """Add realistic stress concentrations and microstructural effects"""
        stress = base_stress.copy()
        
        # Add random stress concentrations (defects, inclusions)
        n_concentrations = np.random.poisson(3)  # Average 3 concentrations per plate
        
        for _ in range(n_concentrations):
            # Random location
            conc_x = np.random.uniform(20, self.plate_dimensions[0] - 20)
            conc_y = np.random.uniform(20, self.plate_dimensions[1] - 20)
            
            # Concentration magnitude (influenced by manufacturing quality)
            quality_factor = 1 - (params['green_density'] - 0.50) / 0.10
            magnitude = np.random.uniform(10e6, 30e6) * quality_factor
            
            # Gaussian concentration
            r_conc = np.sqrt((X - conc_x)**2 + (Y - conc_y)**2)
            concentration = magnitude * np.exp(-(r_conc / 5)**2)
            
            stress += concentration
        
        # Add grain boundary effects (periodic variations)
        grain_size = np.random.uniform(50, 200)  # μm
        grain_effect = 2e6 * (np.sin(2*np.pi*X/grain_size) * np.cos(2*np.pi*Y/grain_size))
        stress += grain_effect
        
        return stress
    
    def _stress_to_displacement(self, stress_field, material_props):
        """
        Convert stress field to surface displacement using thin plate theory
        """
        # Simplified relationship: displacement proportional to stress/stiffness
        # In reality, this would involve solving the full elasticity equations
        
        stiffness = material_props['youngs_modulus'] / (1 - material_props['poisson_ratio']**2)
        thickness = material_props['thickness']
        
        # Base displacement from stress
        displacement = stress_field * thickness**2 / (12 * stiffness)
        
        # Add bending effects (edges tend to curl up)
        x = np.linspace(0, self.plate_dimensions[0], self.measurement_grid[0])
        y = np.linspace(0, self.plate_dimensions[1], self.measurement_grid[1])
        X, Y = np.meshgrid(x, y)
        
        # Edge curling effect
        edge_dist = np.minimum(np.minimum(X, self.plate_dimensions[0] - X),
                              np.minimum(Y, self.plate_dimensions[1] - Y))
        edge_effect = 50e-6 * np.exp(-edge_dist / 10)  # 50 μm max curl
        
        # Apply edge effect with stress-dependent magnitude
        stress_magnitude = np.abs(stress_field) / 50e6  # Normalize
        displacement += edge_effect * stress_magnitude
        
        # Smooth the displacement field
        displacement = ndimage.gaussian_filter(displacement, sigma=1.0)
        
        return displacement
    
    def _add_measurement_noise(self, true_displacement, measurement_date, plate_id):
        """
        Add realistic measurement noise and systematic errors
        """
        displacement = true_displacement.copy()
        
        # Random measurement noise
        noise_level = self.measurement_params['laser_precision']
        random_noise = np.random.normal(0, noise_level, displacement.shape)
        
        # Systematic error (calibration drift)
        days_since_start = (measurement_date - self.start_date).days
        calibration_drift = self.measurement_params['systematic_error'] * (days_since_start / 365)
        
        # Temperature-dependent drift (assume ±5°C lab temperature variation)
        lab_temp_variation = np.random.normal(0, 5)
        temp_drift = self.measurement_params['temperature_drift'] * lab_temp_variation
        
        # Vibration noise (correlated spatially)
        vibration = self._generate_correlated_noise(displacement.shape, 
                                                   self.measurement_params['vibration_noise'])
        
        # Measurement operator effects (edge effects, shadows)
        edge_uncertainty = self._add_edge_measurement_effects(displacement.shape)
        
        # Combine all noise sources
        total_noise = random_noise + calibration_drift + temp_drift + vibration + edge_uncertainty
        
        return displacement + total_noise
    
    def _generate_correlated_noise(self, shape, magnitude):
        """Generate spatially correlated noise (vibration, air currents)"""
        # Generate white noise
        white_noise = np.random.normal(0, magnitude, shape)
        
        # Apply spatial correlation (Gaussian filter)
        correlation_length = 3  # measurement points
        correlated_noise = ndimage.gaussian_filter(white_noise, sigma=correlation_length)
        
        return correlated_noise
    
    def _add_edge_measurement_effects(self, shape):
        """Add measurement uncertainties near plate edges"""
        x = np.linspace(0, self.plate_dimensions[0], shape[1])
        y = np.linspace(0, self.plate_dimensions[1], shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Distance from edges
        edge_dist = np.minimum(np.minimum(X, self.plate_dimensions[0] - X),
                              np.minimum(Y, self.plate_dimensions[1] - Y))
        
        # Increased uncertainty near edges (laser shadowing, edge detection issues)
        edge_factor = np.exp(-edge_dist / 5)  # Exponential increase near edges
        edge_noise = np.random.normal(0, 1e-6, shape) * edge_factor
        
        return edge_noise
    
    def _simulate_failure_modes(self, stress_field, manufacturing_params):
        """
        Simulate known failure modes and their signatures
        """
        failure_indicators = {}
        
        # Edge cracking probability
        max_edge_stress = self._calculate_edge_stress(stress_field)
        crack_threshold = 80e6  # Pa
        crack_probability = max(0, (max_edge_stress - crack_threshold) / (100e6 - crack_threshold))
        
        failure_indicators['edge_crack_risk'] = crack_probability
        failure_indicators['max_edge_stress'] = max_edge_stress
        
        # Delamination risk (high shear stress)
        shear_stress = self._calculate_shear_stress(stress_field)
        delamination_risk = np.max(shear_stress) / 60e6  # Normalized
        failure_indicators['delamination_risk'] = min(1.0, delamination_risk)
        
        # Thermal shock susceptibility
        temp_gradient = abs(manufacturing_params['cooling_rate'] - 2.0)
        thermal_shock_risk = temp_gradient / 3.0  # Normalized
        failure_indicators['thermal_shock_risk'] = min(1.0, thermal_shock_risk)
        
        # Overall failure probability
        failure_indicators['overall_failure_risk'] = (
            0.4 * crack_probability + 
            0.3 * delamination_risk + 
            0.3 * thermal_shock_risk
        )
        
        return failure_indicators
    
    def _calculate_edge_stress(self, stress_field):
        """Calculate maximum stress near plate edges"""
        # Extract edge regions (outer 10% of plate)
        h, w = stress_field.shape
        edge_width = max(3, int(0.1 * min(h, w)))
        
        edges = np.concatenate([
            stress_field[:edge_width, :].flatten(),      # Top edge
            stress_field[-edge_width:, :].flatten(),     # Bottom edge
            stress_field[:, :edge_width].flatten(),      # Left edge
            stress_field[:, -edge_width:].flatten(),     # Right edge
        ])
        
        return np.max(edges)
    
    def _calculate_shear_stress(self, stress_field):
        """Calculate shear stress from normal stress gradients"""
        # Approximate shear stress from gradients
        grad_x = np.gradient(stress_field, axis=1)
        grad_y = np.gradient(stress_field, axis=0)
        
        shear_stress = 0.5 * np.sqrt(grad_x**2 + grad_y**2)
        return shear_stress
    
    def _add_parameter_drift(self, base_params, plate_index, production_date):
        """
        Add realistic parameter drift over time (furnace aging, calibration drift)
        """
        drifted_params = base_params.copy()
        
        # Time-based drift
        days_elapsed = (production_date - self.start_date).days
        
        for param, config in self.manufacturing_params.items():
            if param == 'furnace_position':
                continue  # Categorical parameter
                
            # Linear drift over time
            drift = config['drift_rate'] * days_elapsed
            
            # Add seasonal variations (temperature, humidity)
            if param in ['sintering_temp', 'humidity']:
                seasonal_factor = np.sin(2 * np.pi * days_elapsed / 365.25)
                drift += seasonal_factor * config['std'] * 0.5
            
            # Add random walk component
            random_walk = np.random.normal(0, config['std'] * 0.1)
            
            drifted_params[param] += drift + random_walk
        
        # Furnace position (discrete changes during maintenance)
        if np.random.random() < 0.02:  # 2% chance of furnace recalibration
            drifted_params['furnace_position'] = np.random.choice([-1, 0, 1])
        
        return drifted_params
    
    def generate_single_plate(self, plate_index):
        """
        Generate complete data for a single SOFC plate
        """
        schedule_info = self.production_schedule[plate_index]
        
        # Generate manufacturing parameters with drift
        base_params = {}
        for param, config in self.manufacturing_params.items():
            if param == 'furnace_position':
                base_params[param] = np.random.choice([-1, 0, 1])
            else:
                base_params[param] = np.random.normal(config['nominal'], config['std'])
        
        # Add parameter drift
        manufacturing_params = self._add_parameter_drift(
            base_params, plate_index, schedule_info['production_date']
        )
        
        # Generate stress field
        stress_field = self._generate_base_stress_field(manufacturing_params)
        
        # Convert to displacement
        true_displacement = self._stress_to_displacement(stress_field, self.material_props)
        
        # Add measurement noise
        measured_displacement = self._add_measurement_noise(
            true_displacement, schedule_info['production_date'], schedule_info['plate_id']
        )
        
        # Simulate failure modes
        failure_indicators = self._simulate_failure_modes(stress_field, manufacturing_params)
        
        # Create coordinate grids
        x = np.linspace(0, self.plate_dimensions[0], self.measurement_grid[0])
        y = np.linspace(0, self.plate_dimensions[1], self.measurement_grid[1])
        X, Y = np.meshgrid(x, y)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        # Compile plate data
        plate_data = {
            'metadata': {
                'plate_id': schedule_info['plate_id'],
                'production_date': schedule_info['production_date'].isoformat(),
                'shift': schedule_info['shift'],
                'batch_id': schedule_info['batch_id'],
                'measurement_date': (schedule_info['production_date'] + 
                                   timedelta(hours=np.random.uniform(2, 48))).isoformat(),
            },
            'manufacturing_params': convert_numpy(manufacturing_params),
            'coordinates': {
                'x_mm': x.tolist(),
                'y_mm': y.tolist(),
            },
            'measurements': {
                'displacement_um': (measured_displacement * 1e6).tolist(),  # Convert to μm
                'measurement_uncertainty_um': np.full_like(measured_displacement, 
                                                         self.measurement_params['laser_precision'] * 1e6).tolist(),
            },
            'ground_truth': {
                'true_displacement_um': (true_displacement * 1e6).tolist(),
                'stress_field_MPa': (stress_field * 1e-6).tolist(),  # Convert to MPa
            },
            'failure_analysis': convert_numpy(failure_indicators),
            'quality_metrics': {
                'measurement_snr': float(np.mean(np.abs(true_displacement)) / self.measurement_params['laser_precision']),
                'stress_uniformity': float(1 - (np.std(stress_field) / np.mean(np.abs(stress_field)))),
                'edge_quality': float(1 - failure_indicators['edge_crack_risk']),
            }
        }
        
        return plate_data
    
    def generate_dataset(self, output_dir='sofc_in_the_wild_dataset'):
        """
        Generate the complete "In-The-Wild" dataset
        """
        print(f"Generating SOFC 'In-The-Wild' dataset with {self.n_plates} plates...")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plates
        all_plates = []
        summary_stats = {
            'manufacturing_params': {param: [] for param in self.manufacturing_params.keys()},
            'failure_risks': [],
            'measurement_quality': [],
        }
        
        for i in range(self.n_plates):
            if i % 50 == 0:
                print(f"Generated {i}/{self.n_plates} plates...")
            
            plate_data = self.generate_single_plate(i)
            all_plates.append(plate_data)
            
            # Collect summary statistics
            for param in self.manufacturing_params.keys():
                summary_stats['manufacturing_params'][param].append(
                    plate_data['manufacturing_params'][param]
                )
            summary_stats['failure_risks'].append(
                plate_data['failure_analysis']['overall_failure_risk']
            )
            summary_stats['measurement_quality'].append(
                plate_data['quality_metrics']['measurement_snr']
            )
        
        # Save individual plate data
        plates_dir = os.path.join(output_dir, 'plates')
        os.makedirs(plates_dir, exist_ok=True)
        
        for plate_data in all_plates:
            plate_id = plate_data['metadata']['plate_id']
            with open(os.path.join(plates_dir, f'{plate_id}.json'), 'w') as f:
                json.dump(plate_data, f, indent=2)
        
        # Create summary dataset
        self._create_summary_dataset(all_plates, output_dir)
        
        # Generate analysis and visualization
        self._generate_dataset_analysis(all_plates, summary_stats, output_dir)
        
        # Create documentation
        self._create_documentation(output_dir)
        
        print(f"\nDataset generation complete!")
        print(f"Generated {len(all_plates)} plates with realistic manufacturing variations")
        print(f"Dataset saved to: {output_dir}")
        
        return all_plates
    
    def _create_summary_dataset(self, all_plates, output_dir):
        """Create summary CSV files for easy analysis"""
        
        # Manufacturing parameters summary
        manufacturing_df = pd.DataFrame([
            {
                'plate_id': plate['metadata']['plate_id'],
                'production_date': plate['metadata']['production_date'],
                'batch_id': plate['metadata']['batch_id'],
                'shift': plate['metadata']['shift'],
                **plate['manufacturing_params']
            }
            for plate in all_plates
        ])
        manufacturing_df.to_csv(os.path.join(output_dir, 'manufacturing_parameters.csv'), index=False)
        
        # Quality and failure analysis summary
        quality_df = pd.DataFrame([
            {
                'plate_id': plate['metadata']['plate_id'],
                **plate['failure_analysis'],
                **plate['quality_metrics']
            }
            for plate in all_plates
        ])
        quality_df.to_csv(os.path.join(output_dir, 'quality_analysis.csv'), index=False)
        
        # Measurement summary statistics
        measurement_stats = []
        for plate in all_plates:
            displacement = np.array(plate['measurements']['displacement_um'])
            measurement_stats.append({
                'plate_id': plate['metadata']['plate_id'],
                'mean_displacement_um': np.mean(displacement),
                'std_displacement_um': np.std(displacement),
                'max_displacement_um': np.max(displacement),
                'min_displacement_um': np.min(displacement),
                'rms_displacement_um': np.sqrt(np.mean(displacement**2)),
            })
        
        measurement_df = pd.DataFrame(measurement_stats)
        measurement_df.to_csv(os.path.join(output_dir, 'measurement_summary.csv'), index=False)
    
    def _generate_dataset_analysis(self, all_plates, summary_stats, output_dir):
        """Generate comprehensive dataset analysis and visualizations"""
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig_dir = os.path.join(output_dir, 'analysis_figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # 1. Manufacturing parameter distributions and drift
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        param_names = list(self.manufacturing_params.keys())
        for i, param in enumerate(param_names[:6]):
            if param == 'furnace_position':
                continue
            values = summary_stats['manufacturing_params'][param]
            dates = [datetime.fromisoformat(plate['metadata']['production_date']) for plate in all_plates]
            
            axes[i].scatter(dates, values, alpha=0.6, s=20)
            axes[i].set_title(f'{param.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add trend line
            x_numeric = [(d - dates[0]).days for d in dates]
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            axes[i].plot(dates, p(x_numeric), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'manufacturing_parameter_drift.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Failure risk analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Failure risk distribution
        failure_risks = [plate['failure_analysis']['overall_failure_risk'] for plate in all_plates]
        axes[0,0].hist(failure_risks, bins=30, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Overall Failure Risk Distribution')
        axes[0,0].set_xlabel('Failure Risk')
        axes[0,0].set_ylabel('Count')
        
        # Edge crack vs delamination risk
        edge_risks = [plate['failure_analysis']['edge_crack_risk'] for plate in all_plates]
        delam_risks = [plate['failure_analysis']['delamination_risk'] for plate in all_plates]
        axes[0,1].scatter(edge_risks, delam_risks, alpha=0.6)
        axes[0,1].set_xlabel('Edge Crack Risk')
        axes[0,1].set_ylabel('Delamination Risk')
        axes[0,1].set_title('Failure Mode Correlation')
        
        # Failure risk vs manufacturing parameters
        sintering_temps = summary_stats['manufacturing_params']['sintering_temp']
        axes[1,0].scatter(sintering_temps, failure_risks, alpha=0.6)
        axes[1,0].set_xlabel('Sintering Temperature (°C)')
        axes[1,0].set_ylabel('Overall Failure Risk')
        axes[1,0].set_title('Temperature vs Failure Risk')
        
        # Quality metrics
        snr_values = summary_stats['measurement_quality']
        axes[1,1].hist(snr_values, bins=30, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Measurement Signal-to-Noise Ratio')
        axes[1,1].set_xlabel('SNR')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'failure_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Example displacement fields
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Select representative plates
        indices = [0, len(all_plates)//4, len(all_plates)//2, 
                  3*len(all_plates)//4, len(all_plates)-1, 
                  np.argmax(failure_risks)]  # Include highest risk plate
        
        titles = ['Early Production', 'Quarter Point', 'Mid Production', 
                 'Three-Quarter Point', 'Late Production', 'Highest Risk']
        
        for i, (idx, title) in enumerate(zip(indices, titles)):
            row, col = i // 3, i % 3
            
            displacement = np.array(all_plates[idx]['measurements']['displacement_um'])
            
            im = axes[row, col].imshow(displacement, cmap='RdBu_r', 
                                     extent=[0, self.plate_dimensions[0], 0, self.plate_dimensions[1]])
            axes[row, col].set_title(f'{title}\n{all_plates[idx]["metadata"]["plate_id"]}')
            axes[row, col].set_xlabel('X (mm)')
            axes[row, col].set_ylabel('Y (mm)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], label='Displacement (μm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'example_displacement_fields.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Statistical summary
        stats_summary = {
            'dataset_overview': {
                'total_plates': len(all_plates),
                'production_timespan_days': (datetime.fromisoformat(all_plates[-1]['metadata']['production_date']) - 
                                           datetime.fromisoformat(all_plates[0]['metadata']['production_date'])).days,
                'measurement_grid_size': self.measurement_grid,
                'plate_dimensions_mm': self.plate_dimensions,
            },
            'manufacturing_statistics': {
                param: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
                for param, values in summary_stats['manufacturing_params'].items()
                if param != 'furnace_position'
            },
            'quality_statistics': {
                'failure_risk': {
                    'mean': float(np.mean(failure_risks)),
                    'std': float(np.std(failure_risks)),
                    'high_risk_plates': int(np.sum(np.array(failure_risks) > 0.7)),
                },
                'measurement_snr': {
                    'mean': float(np.mean(snr_values)),
                    'std': float(np.std(snr_values)),
                    'min': float(np.min(snr_values)),
                    'max': float(np.max(snr_values)),
                }
            }
        }
        
        with open(os.path.join(output_dir, 'dataset_statistics.json'), 'w') as f:
            json.dump(stats_summary, f, indent=2)
    
    def _create_documentation(self, output_dir):
        """Create comprehensive documentation for the dataset"""
        
        doc_content = f"""# SOFC "In-The-Wild" Operational Dataset

## Overview

This dataset contains warp measurement data from {self.n_plates} SOFC (Solid Oxide Fuel Cell) plates produced under nominally identical conditions in a simulated industrial environment. The dataset is designed for ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates.

## Dataset Characteristics

### Realism Features
- **Manufacturing Variations**: Natural variations in sintering temperature, time, cooling rate, green density, and humidity
- **Parameter Drift**: Time-dependent drift in manufacturing parameters due to equipment aging and calibration drift
- **Measurement Noise**: Realistic measurement uncertainties including random noise, systematic errors, temperature drift, and vibration
- **Failure Modes**: Known failure signatures including edge cracking, delamination, and thermal shock susceptibility
- **Production Schedule**: Realistic production timeline with shifts, batches, maintenance downtime, and seasonal variations

### Physical Basis
- **Stress-Displacement Relationship**: Based on thin plate theory and elasticity principles
- **Material Properties**: Realistic SOFC material properties (YSZ-based ceramics)
- **Thermal Effects**: Temperature-dependent stress generation during cooling
- **Microstructural Effects**: Grain boundary effects and stress concentrations

## File Structure

```
sofc_in_the_wild_dataset/
├── plates/                          # Individual plate data (JSON)
│   ├── SOFC_0001.json
│   ├── SOFC_0002.json
│   └── ...
├── manufacturing_parameters.csv      # Manufacturing conditions summary
├── quality_analysis.csv            # Failure analysis and quality metrics
├── measurement_summary.csv          # Displacement measurement statistics
├── dataset_statistics.json         # Overall dataset statistics
├── analysis_figures/               # Visualization and analysis plots
│   ├── manufacturing_parameter_drift.png
│   ├── failure_analysis.png
│   └── example_displacement_fields.png
└── README.md                       # This documentation

```

## Data Format

### Individual Plate Data (JSON)
Each plate file contains:

- **metadata**: Plate ID, production date, shift, batch ID, measurement date
- **manufacturing_params**: Sintering temperature, time, cooling rate, green density, humidity, furnace position
- **coordinates**: X,Y measurement grid coordinates (mm)
- **measurements**: Measured displacement field (μm) with uncertainties
- **ground_truth**: True displacement and stress fields (for validation)
- **failure_analysis**: Risk indicators for various failure modes
- **quality_metrics**: SNR, stress uniformity, edge quality

### Summary Files (CSV)
- **manufacturing_parameters.csv**: All manufacturing conditions for easy analysis
- **quality_analysis.csv**: Failure risks and quality metrics
- **measurement_summary.csv**: Statistical summary of displacement measurements

## Usage Examples

### Loading Data in Python

```python
import json
import pandas as pd
import numpy as np

# Load manufacturing parameters
manufacturing_df = pd.read_csv('manufacturing_parameters.csv')

# Load individual plate
with open('plates/SOFC_0001.json', 'r') as f:
    plate_data = json.load(f)

# Extract displacement field
displacement = np.array(plate_data['measurements']['displacement_um'])
x_coords = np.array(plate_data['coordinates']['x_mm'])
y_coords = np.array(plate_data['coordinates']['y_mm'])
```

### Analysis Workflow

1. **Exploratory Analysis**: Use summary CSV files to understand parameter distributions and correlations
2. **Quality Assessment**: Analyze failure risks and measurement quality metrics
3. **Model Training**: Use measured displacement as input, stress field as target
4. **Validation**: Compare predictions against ground truth stress fields
5. **Robustness Testing**: Evaluate model performance across different manufacturing conditions

## Key Research Applications

### Inverse Modeling
- Train ML models to predict stress fields from displacement measurements
- Validate against ground truth stress data
- Test robustness across manufacturing variations

### Failure Prediction
- Correlate stress patterns with failure risk indicators
- Develop early warning systems for quality control
- Optimize manufacturing parameters to reduce failure risk

### Uncertainty Quantification
- Account for measurement uncertainties in predictions
- Propagate manufacturing parameter uncertainties
- Develop confidence intervals for stress predictions

## Dataset Statistics

- **Total Plates**: {self.n_plates}
- **Measurement Grid**: {self.measurement_grid[0]} × {self.measurement_grid[1]} points
- **Plate Dimensions**: {self.plate_dimensions[0]} × {self.plate_dimensions[1]} mm
- **Production Timespan**: ~{int(self.n_plates * 60 / (24 * 60))} days
- **Measurement Precision**: {self.measurement_params['laser_precision']*1e6:.1f} μm

## Manufacturing Parameter Ranges

| Parameter | Nominal | Std Dev | Units |
|-----------|---------|---------|-------|
| Sintering Temperature | {self.manufacturing_params['sintering_temp']['nominal']} | {self.manufacturing_params['sintering_temp']['std']} | °C |
| Sintering Time | {self.manufacturing_params['sintering_time']['nominal']} | {self.manufacturing_params['sintering_time']['std']} | hours |
| Cooling Rate | {self.manufacturing_params['cooling_rate']['nominal']} | {self.manufacturing_params['cooling_rate']['std']} | °C/min |
| Green Density | {self.manufacturing_params['green_density']['nominal']} | {self.manufacturing_params['green_density']['std']} | - |
| Humidity | {self.manufacturing_params['humidity']['nominal']} | {self.manufacturing_params['humidity']['std']} | % |

## Citation

If you use this dataset in your research, please cite:

```
SOFC "In-The-Wild" Operational Dataset for ML-Augmented Inverse Modeling 
of Residual Stress Quantification from Warped Plates
Generated: {datetime.now().strftime('%Y-%m-%d')}
```

## Contact

For questions about this dataset or to report issues, please contact the dataset maintainer.

---
*This dataset was generated using physics-based modeling with realistic manufacturing variations and measurement uncertainties to support research in ML-augmented inverse modeling for SOFC applications.*
"""

        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.write(doc_content)

def main():
    """
    Main function to generate the SOFC "In-The-Wild" dataset
    """
    print("SOFC 'In-The-Wild' Operational Dataset Generator")
    print("=" * 50)
    
    # Create generator with realistic number of plates
    generator = SOFCInTheWildDatasetGenerator(n_plates=500, seed=42)
    
    # Generate the complete dataset
    dataset = generator.generate_dataset()
    
    print("\nDataset generation completed successfully!")
    print(f"Generated {len(dataset)} plates with comprehensive manufacturing variations")
    print("\nKey features:")
    print("✓ Realistic manufacturing parameter drift over time")
    print("✓ Multiple noise sources and measurement uncertainties")
    print("✓ Known failure modes and stress concentration patterns")
    print("✓ Production schedule with shifts, batches, and maintenance")
    print("✓ Comprehensive documentation and analysis tools")
    
    return dataset

if __name__ == "__main__":
    dataset = main()