#!/usr/bin/env python3
"""
Machine Learning Training Dataset Generator for Sintering Process Analysis
Generates 10,000+ simulated datasets for ANN and PINN models
"""

import numpy as np
import pandas as pd
import h5py
import json
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SinteringDatasetGenerator:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        
        # Parameter ranges
        self.temp_range = (1200, 1500)  # °C
        self.cooling_rate_range = (1, 10)  # °C/min
        self.tec_mismatch = 2.3e-6  # K⁻¹
        self.porosity_range = (0.01, 0.15)  # 1-15%
        
        # Material properties
        self.materials = {
            'alumina': {'E': 370e9, 'nu': 0.22, 'alpha': 8.1e-6, 'density': 3950},
            'zirconia': {'E': 200e9, 'nu': 0.31, 'alpha': 10.5e-6, 'density': 6100},
            'silica': {'E': 70e9, 'nu': 0.17, 'alpha': 0.5e-6, 'density': 2200},
            'titanium': {'E': 110e9, 'nu': 0.34, 'alpha': 8.6e-6, 'density': 4500}
        }
    
    def generate_base_parameters(self):
        """Generate base sintering parameters"""
        np.random.seed(42)  # For reproducibility
        
        # Sintering temperatures (1200-1500°C)
        temperatures = np.random.uniform(*self.temp_range, self.n_samples)
        
        # Cooling rates (1-10°C/min)
        cooling_rates = np.random.uniform(*self.cooling_rate_range, self.n_samples)
        
        # Porosity levels (1-15%)
        porosities = np.random.uniform(*self.porosity_range, self.n_samples)
        
        # Material selection
        material_names = list(self.materials.keys())
        selected_materials = np.random.choice(material_names, self.n_samples)
        
        return temperatures, cooling_rates, porosities, selected_materials
    
    def calculate_thermal_stress(self, temp, cooling_rate, material_props):
        """Calculate thermal stress based on temperature and cooling rate"""
        # Simplified thermal stress calculation
        alpha = material_props['alpha']
        E = material_props['E']
        nu = material_props['nu']
        
        # Temperature gradient effect
        temp_gradient = cooling_rate * 0.1  # Simplified relationship
        
        # Thermal stress calculation
        thermal_stress = E * alpha * temp_gradient / (1 - nu)
        
        return thermal_stress
    
    def calculate_strain(self, stress, material_props):
        """Calculate strain from stress using Hooke's law"""
        E = material_props['E']
        strain = stress / E
        return strain
    
    def detect_stress_hotspots(self, stress_field, threshold_factor=1.5):
        """Detect stress hotspots in the material"""
        mean_stress = np.mean(stress_field)
        std_stress = np.std(stress_field)
        threshold = mean_stress + threshold_factor * std_stress
        
        hotspots = stress_field > threshold
        hotspot_density = np.sum(hotspots) / len(stress_field)
        
        return hotspots, hotspot_density
    
    def calculate_crack_risk(self, stress, strain, porosity, material_props):
        """Calculate crack initiation risk based on stress, strain, and porosity"""
        E = material_props['E']
        
        # Stress concentration factor due to porosity
        stress_concentration = 1 + 2 * porosity
        
        # Effective stress
        effective_stress = stress * stress_concentration
        
        # Critical stress for crack initiation (simplified)
        critical_stress = 0.1 * E  # Simplified fracture criterion
        
        # Crack risk probability
        crack_risk = min(1.0, effective_stress / critical_stress)
        
        return crack_risk
    
    def calculate_delamination_probability(self, stress, porosity, temp):
        """Calculate delamination probability"""
        # Temperature effect on delamination
        temp_factor = np.exp(-(temp - 1400) / 100) if temp < 1400 else 1.0
        
        # Porosity effect
        porosity_factor = 1 + 5 * porosity
        
        # Stress effect
        stress_factor = min(1.0, stress / 1e8)  # Normalized stress
        
        # Combined delamination probability
        delamination_prob = temp_factor * porosity_factor * stress_factor
        delamination_prob = min(1.0, delamination_prob)
        
        return delamination_prob
    
    def generate_stress_field(self, base_stress, n_points=100):
        """Generate spatial stress field with hotspots"""
        # Create spatial coordinates
        x = np.linspace(0, 1, int(np.sqrt(n_points)))
        y = np.linspace(0, 1, int(np.sqrt(n_points)))
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        
        # Generate stress field with spatial variation
        stress_field = np.zeros(n_points)
        
        # Add random hotspots
        n_hotspots = np.random.poisson(3)
        for _ in range(n_hotspots):
            hotspot_center = np.random.uniform(0, 1, 2)
            hotspot_strength = np.random.uniform(1.5, 3.0)
            hotspot_size = np.random.uniform(0.1, 0.3)
            
            distances = cdist(coords, [hotspot_center])
            hotspot_effect = hotspot_strength * np.exp(-distances.flatten() / hotspot_size)
            stress_field += hotspot_effect
        
        # Add base stress and noise
        stress_field = base_stress * (1 + stress_field + np.random.normal(0, 0.1, n_points))
        
        return stress_field, coords
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print(f"Generating {self.n_samples} samples...")
        
        # Generate base parameters
        temperatures, cooling_rates, porosities, materials = self.generate_base_parameters()
        
        # Initialize data storage
        data = {
            'sample_id': np.arange(self.n_samples),
            'temperature': temperatures,
            'cooling_rate': cooling_rates,
            'porosity': porosities,
            'material': materials,
            'tec_mismatch': np.full(self.n_samples, self.tec_mismatch),
            'stress': np.zeros(self.n_samples),
            'strain': np.zeros(self.n_samples),
            'stress_hotspot_density': np.zeros(self.n_samples),
            'crack_risk': np.zeros(self.n_samples),
            'delamination_probability': np.zeros(self.n_samples)
        }
        
        # Generate stress fields and calculate outputs
        stress_fields = []
        coordinates = []
        
        for i in range(self.n_samples):
            material_props = self.materials[materials[i]]
            
            # Calculate thermal stress
            stress = self.calculate_thermal_stress(temperatures[i], cooling_rates[i], material_props)
            data['stress'][i] = stress
            
            # Calculate strain
            strain = self.calculate_strain(stress, material_props)
            data['strain'][i] = strain
            
            # Generate spatial stress field
            stress_field, coords = self.generate_stress_field(stress)
            stress_fields.append(stress_field)
            coordinates.append(coords)
            
            # Detect stress hotspots
            hotspots, hotspot_density = self.detect_stress_hotspots(stress_field)
            data['stress_hotspot_density'][i] = hotspot_density
            
            # Calculate crack risk
            crack_risk = self.calculate_crack_risk(stress, strain, porosities[i], material_props)
            data['crack_risk'][i] = crack_risk
            
            # Calculate delamination probability
            delamination_prob = self.calculate_delamination_probability(stress, porosities[i], temperatures[i])
            data['delamination_probability'][i] = delamination_prob
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{self.n_samples} samples")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add material properties as separate columns
        for prop in ['E', 'nu', 'alpha', 'density']:
            df[f'material_{prop}'] = df['material'].map(lambda x: self.materials[x][prop])
        
        # Add derived features
        df['stress_strain_ratio'] = df['stress'] / (df['strain'] + 1e-10)
        df['thermal_gradient'] = df['cooling_rate'] * 0.1
        df['porosity_stress_factor'] = 1 + 2 * df['porosity']
        
        return df, stress_fields, coordinates
    
    def create_validation_data(self, n_validation=1000):
        """Create validation data simulating DIC/XRD measurements"""
        print(f"Generating {n_validation} validation samples...")
        
        # Generate validation parameters with more controlled ranges
        val_temps = np.random.normal(1350, 50, n_validation)  # Centered around 1350°C
        val_cooling = np.random.normal(5, 2, n_validation)    # Centered around 5°C/min
        val_porosity = np.random.normal(0.08, 0.03, n_validation)  # Centered around 8%
        
        # Clamp to valid ranges
        val_temps = np.clip(val_temps, *self.temp_range)
        val_cooling = np.clip(val_cooling, *self.cooling_rate_range)
        val_porosity = np.clip(val_porosity, *self.porosity_range)
        
        validation_data = {
            'sample_id': np.arange(n_validation),
            'temperature': val_temps,
            'cooling_rate': val_cooling,
            'porosity': val_porosity,
            'measurement_type': np.random.choice(['DIC', 'XRD'], n_validation),
            'measurement_uncertainty': np.random.uniform(0.02, 0.05, n_validation)
        }
        
        return pd.DataFrame(validation_data)
    
    def save_dataset(self, df, stress_fields, coordinates, validation_df, output_dir='/workspace'):
        """Save dataset in multiple formats"""
        print("Saving dataset...")
        
        # Save as CSV
        df.to_csv(f'{output_dir}/sintering_training_data.csv', index=False)
        validation_df.to_csv(f'{output_dir}/sintering_validation_data.csv', index=False)
        
        # Save as HDF5 for efficient storage
        with h5py.File(f'{output_dir}/sintering_dataset.h5', 'w') as f:
            # Main dataset
            main_group = f.create_group('training_data')
            for col in df.columns:
                main_group.create_dataset(col, data=df[col].values)
            
            # Stress fields
            stress_group = f.create_group('stress_fields')
            for i, stress_field in enumerate(stress_fields):
                stress_group.create_dataset(f'field_{i}', data=stress_field)
            
            # Coordinates
            coord_group = f.create_group('coordinates')
            for i, coord in enumerate(coordinates):
                coord_group.create_dataset(f'coords_{i}', data=coord)
            
            # Validation data
            val_group = f.create_group('validation_data')
            for col in validation_df.columns:
                val_group.create_dataset(col, data=validation_df[col].values)
        
        # Save metadata
        metadata = {
            'n_samples': self.n_samples,
            'parameter_ranges': {
                'temperature_range': self.temp_range,
                'cooling_rate_range': self.cooling_rate_range,
                'porosity_range': self.porosity_range,
                'tec_mismatch': self.tec_mismatch
            },
            'materials': self.materials,
            'features': list(df.columns),
            'outputs': ['stress_hotspot_density', 'crack_risk', 'delamination_probability']
        }
        
        with open(f'{output_dir}/dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}/")
        print(f"Files created:")
        print(f"  - sintering_training_data.csv")
        print(f"  - sintering_validation_data.csv") 
        print(f"  - sintering_dataset.h5")
        print(f"  - dataset_metadata.json")
    
    def generate_summary_statistics(self, df):
        """Generate summary statistics for the dataset"""
        print("\n=== DATASET SUMMARY ===")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns)}")
        
        print("\nParameter ranges:")
        print(f"Temperature: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
        print(f"Cooling rate: {df['cooling_rate'].min():.1f} - {df['cooling_rate'].max():.1f} °C/min")
        print(f"Porosity: {df['porosity'].min():.3f} - {df['porosity'].max():.3f}")
        
        print("\nOutput statistics:")
        print(f"Stress hotspot density: {df['stress_hotspot_density'].mean():.3f} ± {df['stress_hotspot_density'].std():.3f}")
        print(f"Crack risk: {df['crack_risk'].mean():.3f} ± {df['crack_risk'].std():.3f}")
        print(f"Delamination probability: {df['delamination_probability'].mean():.3f} ± {df['delamination_probability'].std():.3f}")
        
        print("\nMaterial distribution:")
        print(df['material'].value_counts())

def main():
    """Main function to generate the complete dataset"""
    print("🧠 Machine Learning Training Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = SinteringDatasetGenerator(n_samples=10000)
    
    # Generate dataset
    df, stress_fields, coordinates = generator.generate_dataset()
    
    # Generate validation data
    validation_df = generator.create_validation_data(n_validation=1000)
    
    # Save dataset
    generator.save_dataset(df, stress_fields, coordinates, validation_df)
    
    # Generate summary
    generator.generate_summary_statistics(df)
    
    print("\n✅ Dataset generation complete!")
    print("Ready for ANN and PINN model training.")

if __name__ == "__main__":
    main()