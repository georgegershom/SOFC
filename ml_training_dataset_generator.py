#!/usr/bin/env python3
"""
Machine Learning Training Dataset Generator for Sintering Process Analysis
Generates 10,000+ simulated datasets for ANN and PINN models

Features:
- Sintering temperatures (1200–1500°C)
- Cooling rates (1–10°C/min)
- TEC mismatch (Δα = 2.3×10⁻⁶ K⁻¹)
- Porosity levels
- Physics-based stress calculations
- Crack initiation risk modeling
- Delamination probability estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm, weibull_min
import h5py
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SinteringDatasetGenerator:
    """Generate comprehensive ML training datasets for sintering process analysis."""
    
    def __init__(self, random_seed=42):
        """Initialize the dataset generator with physics constants."""
        np.random.seed(random_seed)
        
        # Material properties and constants
        self.constants = {
            'young_modulus': 200e9,  # Pa (typical ceramic)
            'poisson_ratio': 0.25,
            'thermal_expansion_coeff': 8e-6,  # K^-1 (base material)
            'tec_mismatch': 2.3e-6,  # K^-1 (given constraint)
            'density': 3800,  # kg/m³
            'specific_heat': 800,  # J/(kg·K)
            'thermal_conductivity': 15,  # W/(m·K)
            'fracture_toughness': 3.5e6,  # Pa·m^0.5
            'critical_stress_factor': 0.7,  # Fraction of yield strength
            'reference_temp': 25,  # °C (room temperature)
        }
        
        # Process parameter ranges
        self.param_ranges = {
            'sintering_temp': (1200, 1500),  # °C
            'cooling_rate': (1, 10),  # °C/min
            'porosity': (0.01, 0.25),  # Volume fraction
            'grain_size': (1e-6, 50e-6),  # m
            'thickness': (0.5e-3, 5e-3),  # m
        }
        
        # Spatial discretization
        self.grid_size = 50  # 50x50 spatial grid
        
    def generate_process_parameters(self, n_samples):
        """Generate random process parameters within specified ranges."""
        params = {}
        
        # Core process parameters
        params['sintering_temp'] = np.random.uniform(
            self.param_ranges['sintering_temp'][0],
            self.param_ranges['sintering_temp'][1],
            n_samples
        )
        
        params['cooling_rate'] = np.random.uniform(
            self.param_ranges['cooling_rate'][0],
            self.param_ranges['cooling_rate'][1],
            n_samples
        )
        
        params['porosity'] = np.random.beta(2, 5, n_samples) * 0.24 + 0.01
        
        params['grain_size'] = np.random.lognormal(
            mean=np.log(10e-6), sigma=0.5, size=n_samples
        )
        
        params['thickness'] = np.random.uniform(
            self.param_ranges['thickness'][0],
            self.param_ranges['thickness'][1],
            n_samples
        )
        
        # Derived parameters
        params['max_temp_gradient'] = np.random.uniform(5, 50, n_samples)  # °C/mm
        params['dwell_time'] = np.random.uniform(0.5, 4, n_samples)  # hours
        
        return params
    
    def calculate_thermal_stress(self, temp_profile, cooling_rate, tec_mismatch):
        """Calculate thermal stress distribution based on temperature profile."""
        # Temperature gradient-induced stress
        temp_gradient = np.gradient(temp_profile)
        
        # Thermal stress calculation (simplified 2D)
        alpha_eff = self.constants['thermal_expansion_coeff'] + tec_mismatch
        E = self.constants['young_modulus']
        nu = self.constants['poisson_ratio']
        
        # Thermal stress components
        thermal_strain = alpha_eff * (temp_profile - self.constants['reference_temp'])
        
        # Stress calculation (plane stress assumption)
        stress_xx = E / (1 - nu**2) * thermal_strain
        stress_yy = E / (1 - nu**2) * thermal_strain
        stress_xy = np.zeros_like(stress_xx)
        
        # Add cooling rate effects
        cooling_stress_factor = np.log10(cooling_rate + 1) * 0.1
        stress_xx *= (1 + cooling_stress_factor)
        stress_yy *= (1 + cooling_stress_factor)
        
        return stress_xx, stress_yy, stress_xy
    
    def calculate_porosity_effects(self, porosity, stress_field):
        """Modify stress field based on porosity distribution."""
        # Porosity reduces effective modulus
        porosity_factor = (1 - porosity)**2.5  # Empirical relationship
        
        # Stress concentration around pores
        stress_concentration = 1 + 2 * porosity / (1 - porosity)
        
        modified_stress = stress_field * porosity_factor * stress_concentration
        
        return modified_stress
    
    def generate_spatial_fields(self, params_dict, sample_idx):
        """Generate 2D spatial fields for temperature, stress, and material properties."""
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Temperature field with realistic gradients
        temp_center = params_dict['sintering_temp'][sample_idx]
        temp_gradient = params_dict['max_temp_gradient'][sample_idx]
        
        # Create temperature distribution (higher at center, cooler at edges)
        r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        temp_field = temp_center - temp_gradient * r * 100
        
        # Add some noise for realism
        temp_field += np.random.normal(0, 5, temp_field.shape)
        
        # Porosity field (spatially correlated)
        base_porosity = params_dict['porosity'][sample_idx]
        porosity_noise = np.random.normal(0, base_porosity * 0.2, (self.grid_size, self.grid_size))
        porosity_field = np.clip(base_porosity + porosity_noise, 0.001, 0.5)
        
        # Calculate stress fields
        stress_xx, stress_yy, stress_xy = self.calculate_thermal_stress(
            temp_field, 
            params_dict['cooling_rate'][sample_idx],
            self.constants['tec_mismatch']
        )
        
        # Apply porosity effects
        stress_xx = self.calculate_porosity_effects(porosity_field, stress_xx)
        stress_yy = self.calculate_porosity_effects(porosity_field, stress_yy)
        
        # Calculate von Mises stress
        von_mises_stress = np.sqrt(
            stress_xx**2 + stress_yy**2 - stress_xx * stress_yy + 3 * stress_xy**2
        )
        
        return {
            'temperature': temp_field,
            'porosity': porosity_field,
            'stress_xx': stress_xx,
            'stress_yy': stress_yy,
            'stress_xy': stress_xy,
            'von_mises_stress': von_mises_stress,
            'X': X,
            'Y': Y
        }
    
    def calculate_stress_hotspots(self, stress_field, threshold_percentile=90):
        """Identify stress hotspot locations."""
        threshold = np.percentile(stress_field, threshold_percentile)
        hotspots = (stress_field > threshold).astype(float)
        
        # Calculate hotspot intensity
        hotspot_intensity = np.where(hotspots > 0, stress_field / threshold, 0)
        
        return hotspots, hotspot_intensity
    
    def calculate_crack_initiation_risk(self, stress_field, porosity_field, grain_size):
        """Calculate crack initiation risk based on Griffith criterion."""
        # Critical stress for crack initiation
        K_IC = self.constants['fracture_toughness']
        
        # Effective crack length (related to grain size and porosity)
        effective_crack_length = grain_size * (1 + 5 * porosity_field)
        
        # Critical stress
        critical_stress = K_IC / np.sqrt(np.pi * effective_crack_length)
        
        # Risk probability (sigmoid function)
        risk_factor = stress_field / critical_stress
        crack_risk = 1 / (1 + np.exp(-5 * (risk_factor - 1)))
        
        return crack_risk
    
    def calculate_delamination_probability(self, stress_field, temp_field, thickness):
        """Calculate delamination probability based on interface stress."""
        # Interface stress (simplified model)
        temp_gradient_magnitude = np.sqrt(
            np.gradient(temp_field, axis=0)**2 + np.gradient(temp_field, axis=1)**2
        )
        
        # Delamination stress (function of temperature gradient and thickness)
        delamination_stress = temp_gradient_magnitude * self.constants['thermal_expansion_coeff'] * \
                            self.constants['young_modulus'] * thickness
        
        # Critical delamination stress
        critical_delamination_stress = self.constants['young_modulus'] * \
                                     self.constants['critical_stress_factor']
        
        # Probability calculation
        delamination_prob = np.tanh(delamination_stress / critical_delamination_stress)
        
        return np.clip(delamination_prob, 0, 1)
    
    def generate_validation_data(self, n_samples=100):
        """Generate synthetic experimental validation data (DIC/XRD measurements)."""
        validation_data = []
        
        for i in range(n_samples):
            # Simulate experimental conditions
            temp = np.random.uniform(1250, 1450)
            cooling_rate = np.random.uniform(2, 8)
            
            # Simulate DIC strain measurements (with experimental noise)
            strain_xx = np.random.normal(0.001, 0.0002)
            strain_yy = np.random.normal(0.001, 0.0002)
            strain_xy = np.random.normal(0, 0.0001)
            
            # Simulate XRD stress measurements (with experimental uncertainty)
            residual_stress = np.random.normal(50e6, 10e6)  # Pa
            
            # Measurement uncertainty
            dic_uncertainty = 0.05  # 5% uncertainty
            xrd_uncertainty = 0.1   # 10% uncertainty
            
            validation_data.append({
                'sample_id': f'EXP_{i:03d}',
                'sintering_temp': temp,
                'cooling_rate': cooling_rate,
                'dic_strain_xx': strain_xx,
                'dic_strain_yy': strain_yy,
                'dic_strain_xy': strain_xy,
                'dic_uncertainty': dic_uncertainty,
                'xrd_residual_stress': residual_stress,
                'xrd_uncertainty': xrd_uncertainty,
                'measurement_date': datetime.now().isoformat(),
            })
        
        return pd.DataFrame(validation_data)
    
    def generate_complete_dataset(self, n_samples=10000, output_dir='ml_dataset'):
        """Generate the complete ML training dataset."""
        print(f"Generating {n_samples} samples for ML training dataset...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate process parameters
        params = self.generate_process_parameters(n_samples)
        
        # Initialize storage arrays
        input_features = []
        output_labels = []
        spatial_data = []
        
        print("Processing samples...")
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"  Processed {i}/{n_samples} samples")
            
            # Generate spatial fields for this sample
            fields = self.generate_spatial_fields(params, i)
            
            # Calculate output labels
            hotspots, hotspot_intensity = self.calculate_stress_hotspots(
                fields['von_mises_stress']
            )
            
            crack_risk = self.calculate_crack_initiation_risk(
                fields['von_mises_stress'],
                fields['porosity'],
                params['grain_size'][i]
            )
            
            delamination_prob = self.calculate_delamination_probability(
                fields['von_mises_stress'],
                fields['temperature'],
                params['thickness'][i]
            )
            
            # Aggregate features for this sample
            sample_features = {
                'sintering_temp': params['sintering_temp'][i],
                'cooling_rate': params['cooling_rate'][i],
                'porosity_avg': np.mean(fields['porosity']),
                'porosity_std': np.std(fields['porosity']),
                'grain_size': params['grain_size'][i],
                'thickness': params['thickness'][i],
                'max_temp_gradient': params['max_temp_gradient'][i],
                'dwell_time': params['dwell_time'][i],
                'temp_avg': np.mean(fields['temperature']),
                'temp_std': np.std(fields['temperature']),
                'stress_max': np.max(fields['von_mises_stress']),
                'stress_avg': np.mean(fields['von_mises_stress']),
                'stress_std': np.std(fields['von_mises_stress']),
            }
            
            sample_labels = {
                'stress_hotspot_count': np.sum(hotspots),
                'max_hotspot_intensity': np.max(hotspot_intensity),
                'avg_crack_risk': np.mean(crack_risk),
                'max_crack_risk': np.max(crack_risk),
                'avg_delamination_prob': np.mean(delamination_prob),
                'max_delamination_prob': np.max(delamination_prob),
                'failure_risk_score': np.mean(crack_risk) * 0.4 + np.mean(delamination_prob) * 0.6,
            }
            
            input_features.append(sample_features)
            output_labels.append(sample_labels)
            
            # Store spatial data for selected samples (every 100th sample to save space)
            if i % 100 == 0:
                spatial_sample = {
                    'sample_id': i,
                    'temperature_field': fields['temperature'],
                    'porosity_field': fields['porosity'],
                    'stress_field': fields['von_mises_stress'],
                    'hotspots': hotspots,
                    'crack_risk': crack_risk,
                    'delamination_prob': delamination_prob,
                }
                spatial_data.append(spatial_sample)
        
        # Convert to DataFrames
        features_df = pd.DataFrame(input_features)
        labels_df = pd.DataFrame(output_labels)
        
        # Combine features and labels
        complete_dataset = pd.concat([features_df, labels_df], axis=1)
        complete_dataset['sample_id'] = range(n_samples)
        
        # Generate validation data
        validation_df = self.generate_validation_data()
        
        # Save datasets
        self.save_datasets(complete_dataset, spatial_data, validation_df, output_path)
        
        print(f"\nDataset generation complete!")
        print(f"Total samples: {n_samples}")
        print(f"Features: {len(features_df.columns)}")
        print(f"Labels: {len(labels_df.columns)}")
        print(f"Spatial samples: {len(spatial_data)}")
        print(f"Validation samples: {len(validation_df)}")
        
        return complete_dataset, spatial_data, validation_df
    
    def save_datasets(self, complete_dataset, spatial_data, validation_df, output_path):
        """Save datasets in multiple formats."""
        
        # Save main dataset as CSV
        complete_dataset.to_csv(output_path / 'ml_training_dataset.csv', index=False)
        
        # Save validation data
        validation_df.to_csv(output_path / 'experimental_validation_data.csv', index=False)
        
        # Save spatial data as HDF5
        with h5py.File(output_path / 'spatial_fields_data.h5', 'w') as f:
            for i, sample in enumerate(spatial_data):
                grp = f.create_group(f'sample_{sample["sample_id"]}')
                for key, value in sample.items():
                    if key != 'sample_id':
                        grp.create_dataset(key, data=value)
        
        # Save as NumPy arrays for quick loading
        features = complete_dataset.iloc[:, :-7].values  # Exclude labels
        labels = complete_dataset.iloc[:, -7:].values    # Only labels
        
        np.savez_compressed(
            output_path / 'ml_dataset_arrays.npz',
            features=features,
            labels=labels,
            feature_names=complete_dataset.columns[:-7].tolist(),
            label_names=complete_dataset.columns[-7:].tolist()
        )
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'n_samples': len(complete_dataset),
            'n_features': len(complete_dataset.columns) - 7,
            'n_labels': 7,
            'parameter_ranges': self.param_ranges,
            'constants': self.constants,
            'grid_size': self.grid_size,
            'description': 'ML training dataset for sintering process analysis',
        }
        
        with open(output_path / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDatasets saved to: {output_path}")
        print("Files created:")
        print("  - ml_training_dataset.csv (main dataset)")
        print("  - experimental_validation_data.csv (validation data)")
        print("  - spatial_fields_data.h5 (spatial field data)")
        print("  - ml_dataset_arrays.npz (NumPy format)")
        print("  - dataset_metadata.json (metadata)")


def main():
    """Main function to generate the dataset."""
    generator = SinteringDatasetGenerator(random_seed=42)
    
    # Generate the complete dataset
    dataset, spatial_data, validation_data = generator.generate_complete_dataset(
        n_samples=10000,
        output_dir='ml_sintering_dataset'
    )
    
    # Display sample statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\nInput Features Statistics:")
    feature_cols = ['sintering_temp', 'cooling_rate', 'porosity_avg', 'stress_max']
    print(dataset[feature_cols].describe())
    
    print("\nOutput Labels Statistics:")
    label_cols = ['avg_crack_risk', 'max_delamination_prob', 'failure_risk_score']
    print(dataset[label_cols].describe())
    
    print("\nValidation Data Sample:")
    print(validation_data.head())


if __name__ == "__main__":
    main()