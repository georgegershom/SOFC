"""
Machine Learning Training Dataset Generator
For ANN and PINN Models - Sintering and Material Analysis

Generates 10,000+ simulated datasets with varying:
- Sintering temperatures (1200–1500°C)
- Cooling rates (1–10°C/min)
- TEC mismatch (Δα = 2.3×10⁻⁶ K⁻¹)
- Porosity levels

Input features: temperature, stress, strain, material properties
Output labels: stress hotspots, crack initiation risk, delamination probability
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SinteringDatasetGenerator:
    """Generate realistic sintering simulation datasets for ML training"""
    
    def __init__(self, num_samples: int = 10000, random_seed: int = 42):
        """
        Initialize the dataset generator
        
        Args:
            num_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Physical constants
        self.TEC_MISMATCH = 2.3e-6  # K^-1
        self.YOUNGS_MODULUS_BASE = 200e9  # Pa (typical ceramic)
        self.POISSON_RATIO = 0.25
        self.REFERENCE_TEMP = 25  # °C
        
    def generate_input_parameters(self) -> pd.DataFrame:
        """Generate varying input parameters for simulations"""
        
        # Sintering temperatures (1200-1500°C)
        sintering_temp = np.random.uniform(1200, 1500, self.num_samples)
        
        # Cooling rates (1-10°C/min)
        cooling_rate = np.random.uniform(1, 10, self.num_samples)
        
        # Porosity levels (0-30%)
        porosity = np.random.beta(2, 5, self.num_samples) * 30
        
        # Add some material property variations
        # Young's modulus varies with porosity and temperature
        youngs_modulus = self.YOUNGS_MODULUS_BASE * (1 - 1.9 * porosity/100 + 0.9 * (porosity/100)**2)
        
        # Density variations (g/cm³) - typical ceramics
        density = np.random.uniform(3.5, 6.0, self.num_samples) * (1 - porosity/100)
        
        # Thermal conductivity (W/m·K)
        thermal_conductivity = np.random.uniform(2, 15, self.num_samples) * (1 - porosity/150)
        
        # Grain size (micrometers) - affected by sintering temp
        grain_size = 0.5 + (sintering_temp - 1200) / 300 * 5 + np.random.normal(0, 0.5, self.num_samples)
        grain_size = np.clip(grain_size, 0.5, 10)
        
        # Sample location coordinates (normalized)
        x_coord = np.random.uniform(0, 1, self.num_samples)
        y_coord = np.random.uniform(0, 1, self.num_samples)
        z_coord = np.random.uniform(0, 1, self.num_samples)
        
        df = pd.DataFrame({
            'sintering_temperature_C': sintering_temp,
            'cooling_rate_C_per_min': cooling_rate,
            'porosity_percent': porosity,
            'youngs_modulus_Pa': youngs_modulus,
            'poisson_ratio': self.POISSON_RATIO + np.random.normal(0, 0.02, self.num_samples),
            'density_g_cm3': density,
            'thermal_conductivity_W_mK': thermal_conductivity,
            'grain_size_um': grain_size,
            'TEC_mismatch_K-1': self.TEC_MISMATCH + np.random.normal(0, 0.2e-6, self.num_samples),
            'x_coordinate': x_coord,
            'y_coordinate': y_coord,
            'z_coordinate': z_coord
        })
        
        return df
    
    def calculate_thermal_stress(self, inputs: pd.DataFrame) -> np.ndarray:
        """
        Calculate thermal stress based on temperature change and TEC mismatch
        
        σ = E * Δα * ΔT / (1 - ν)
        """
        delta_T = inputs['sintering_temperature_C'] - self.REFERENCE_TEMP
        
        stress = (inputs['youngs_modulus_Pa'] * 
                 inputs['TEC_mismatch_K-1'] * 
                 delta_T / 
                 (1 - inputs['poisson_ratio']))
        
        # Add cooling rate effect (faster cooling = higher stress)
        cooling_factor = 1 + (inputs['cooling_rate_C_per_min'] - 5.5) / 10
        stress *= cooling_factor
        
        # Add spatial variation (edge effects)
        edge_distance = np.minimum(
            np.minimum(inputs['x_coordinate'], 1 - inputs['x_coordinate']),
            np.minimum(inputs['y_coordinate'], 1 - inputs['y_coordinate'])
        )
        edge_factor = 1 + (1 - edge_distance) * 0.5  # Higher stress at edges
        stress *= edge_factor
        
        return stress
    
    def calculate_strain(self, stress: np.ndarray, inputs: pd.DataFrame) -> np.ndarray:
        """Calculate strain from stress using Hooke's law"""
        strain = stress / inputs['youngs_modulus_Pa']
        
        # Add plastic strain component for high stress
        plastic_threshold = 0.002  # 0.2% strain
        plastic_strain = np.maximum(0, strain - plastic_threshold) * 0.3
        total_strain = strain + plastic_strain
        
        return total_strain
    
    def calculate_stress_hotspots(self, stress: np.ndarray, inputs: pd.DataFrame) -> np.ndarray:
        """
        Identify stress hotspot intensity (0-1 scale)
        Based on local stress concentration
        """
        # Normalize stress
        stress_normalized = stress / np.max(stress)
        
        # Hotspots more likely at high porosity
        porosity_factor = inputs['porosity_percent'] / 30
        
        # Grain boundary effects
        grain_factor = np.exp(-inputs['grain_size_um'] / 5)
        
        hotspot_intensity = stress_normalized * (0.6 + 0.2 * porosity_factor + 0.2 * grain_factor)
        hotspot_intensity = np.clip(hotspot_intensity, 0, 1)
        
        # Add some noise
        hotspot_intensity += np.random.normal(0, 0.05, len(hotspot_intensity))
        hotspot_intensity = np.clip(hotspot_intensity, 0, 1)
        
        return hotspot_intensity
    
    def calculate_crack_initiation_risk(self, stress: np.ndarray, strain: np.ndarray, 
                                       inputs: pd.DataFrame) -> np.ndarray:
        """
        Calculate crack initiation risk (0-1 probability)
        Based on Griffith criterion and stress intensity
        """
        # Critical stress (varies with porosity)
        critical_stress = 100e6 * (1 - inputs['porosity_percent'] / 40)  # Pa
        
        # Stress ratio
        stress_ratio = stress / critical_stress
        
        # Strain energy density factor
        strain_energy = 0.5 * stress * strain
        critical_energy = 1e4  # J/m³
        energy_ratio = strain_energy / critical_energy
        
        # Flaw probability (higher with porosity)
        flaw_probability = inputs['porosity_percent'] / 30
        
        # Combine factors with sigmoid function
        risk_factor = (stress_ratio + energy_ratio) * (0.7 + 0.3 * flaw_probability)
        crack_risk = 1 / (1 + np.exp(-5 * (risk_factor - 1)))
        
        # Add some stochasticity
        crack_risk += np.random.normal(0, 0.05, len(crack_risk))
        crack_risk = np.clip(crack_risk, 0, 1)
        
        return crack_risk
    
    def calculate_delamination_probability(self, stress: np.ndarray, inputs: pd.DataFrame) -> np.ndarray:
        """
        Calculate delamination probability (0-1)
        Influenced by TEC mismatch, cooling rate, and interface properties
        """
        # Base delamination from TEC mismatch
        tec_factor = inputs['TEC_mismatch_K-1'] / self.TEC_MISMATCH
        
        # Cooling rate effect
        cooling_factor = inputs['cooling_rate_C_per_min'] / 5.5
        
        # Interface strength (inversely related to porosity)
        interface_strength = 1 - inputs['porosity_percent'] / 50
        
        # Temperature differential effect
        temp_factor = (inputs['sintering_temperature_C'] - 1350) / 300
        
        # Shear stress component (approximation)
        shear_stress = stress * 0.5 * (1 - inputs['poisson_ratio'])
        shear_normalized = shear_stress / np.max(shear_stress)
        
        # Combined delamination probability
        delam_prob = (0.25 * tec_factor + 
                     0.25 * cooling_factor + 
                     0.25 * (1 - interface_strength) + 
                     0.15 * temp_factor + 
                     0.10 * shear_normalized)
        
        delam_prob = 1 / (1 + np.exp(-8 * (delam_prob - 0.5)))
        
        # Add noise
        delam_prob += np.random.normal(0, 0.05, len(delam_prob))
        delam_prob = np.clip(delam_prob, 0, 1)
        
        return delam_prob
    
    def generate_training_dataset(self) -> pd.DataFrame:
        """Generate complete training dataset"""
        print(f"Generating {self.num_samples} training samples...")
        
        # Generate input features
        inputs = self.generate_input_parameters()
        
        # Calculate intermediate physics quantities
        stress = self.calculate_thermal_stress(inputs)
        strain = self.calculate_strain(stress, inputs)
        
        # Calculate output labels
        stress_hotspots = self.calculate_stress_hotspots(stress, inputs)
        crack_risk = self.calculate_crack_initiation_risk(stress, strain, inputs)
        delam_prob = self.calculate_delamination_probability(stress, inputs)
        
        # Combine into full dataset
        dataset = inputs.copy()
        dataset['thermal_stress_Pa'] = stress
        dataset['thermal_strain'] = strain
        dataset['stress_hotspot_intensity'] = stress_hotspots
        dataset['crack_initiation_risk'] = crack_risk
        dataset['delamination_probability'] = delam_prob
        
        # Add sample ID
        dataset.insert(0, 'sample_id', range(1, len(dataset) + 1))
        
        print(f"✓ Generated {len(dataset)} training samples")
        return dataset
    
    def generate_validation_dataset(self, num_validation: int = 1000) -> pd.DataFrame:
        """
        Generate validation dataset simulating experimental DIC/XRD measurements
        Adds measurement noise and uncertainty
        """
        print(f"\nGenerating {num_validation} validation samples (with experimental noise)...")
        
        # Create temporary generator for validation data
        temp_seed = self.random_seed + 999
        np.random.seed(temp_seed)
        
        # Generate base validation data
        original_num = self.num_samples
        self.num_samples = num_validation
        val_dataset = self.generate_training_dataset()
        self.num_samples = original_num
        
        # Add experimental measurement noise
        # DIC (Digital Image Correlation) - strain measurement noise
        dic_noise_level = 0.05  # 5% measurement uncertainty
        val_dataset['thermal_strain_measured'] = val_dataset['thermal_strain'] * (
            1 + np.random.normal(0, dic_noise_level, num_validation))
        
        # XRD (X-Ray Diffraction) - stress measurement noise
        xrd_noise_level = 0.08  # 8% measurement uncertainty
        val_dataset['thermal_stress_Pa_measured'] = val_dataset['thermal_stress_Pa'] * (
            1 + np.random.normal(0, xrd_noise_level, num_validation))
        
        # Add measurement confidence scores
        val_dataset['dic_measurement_confidence'] = np.random.beta(8, 2, num_validation)
        val_dataset['xrd_measurement_confidence'] = np.random.beta(7, 3, num_validation)
        
        # Add spatial resolution uncertainty
        val_dataset['spatial_resolution_um'] = np.random.uniform(50, 200, num_validation)
        
        # Some measurements might be invalid (equipment issues, bad samples, etc.)
        invalid_mask = np.random.random(num_validation) < 0.05  # 5% invalid
        val_dataset.loc[invalid_mask, ['dic_measurement_confidence', 
                                       'xrd_measurement_confidence']] *= 0.3
        
        # Reset random seed
        np.random.seed(self.random_seed)
        
        print(f"✓ Generated {len(val_dataset)} validation samples with experimental noise")
        return val_dataset


class DatasetStatistics:
    """Calculate and display dataset statistics"""
    
    @staticmethod
    def print_statistics(df: pd.DataFrame, dataset_name: str = "Dataset"):
        """Print comprehensive statistics"""
        print(f"\n{'='*70}")
        print(f"{dataset_name} Statistics")
        print(f"{'='*70}")
        print(f"Total samples: {len(df)}")
        print(f"\nInput Features Statistics:")
        print("-" * 70)
        
        input_features = [
            'sintering_temperature_C', 'cooling_rate_C_per_min', 'porosity_percent',
            'youngs_modulus_Pa', 'density_g_cm3', 'thermal_conductivity_W_mK',
            'grain_size_um', 'TEC_mismatch_K-1'
        ]
        
        for feature in input_features:
            if feature in df.columns:
                print(f"{feature:40s}: mean={df[feature].mean():.4e}, "
                      f"std={df[feature].std():.4e}, "
                      f"min={df[feature].min():.4e}, "
                      f"max={df[feature].max():.4e}")
        
        print(f"\nOutput Labels Statistics:")
        print("-" * 70)
        
        output_features = [
            'thermal_stress_Pa', 'thermal_strain', 'stress_hotspot_intensity',
            'crack_initiation_risk', 'delamination_probability'
        ]
        
        for feature in output_features:
            if feature in df.columns:
                print(f"{feature:40s}: mean={df[feature].mean():.4e}, "
                      f"std={df[feature].std():.4e}, "
                      f"min={df[feature].min():.4e}, "
                      f"max={df[feature].max():.4e}")
        
        # Risk distribution
        if 'crack_initiation_risk' in df.columns:
            print(f"\nRisk Distribution:")
            print("-" * 70)
            print(f"Low risk (< 0.3):        {(df['crack_initiation_risk'] < 0.3).sum()} samples "
                  f"({100*(df['crack_initiation_risk'] < 0.3).sum()/len(df):.1f}%)")
            print(f"Medium risk (0.3-0.7):   {((df['crack_initiation_risk'] >= 0.3) & (df['crack_initiation_risk'] < 0.7)).sum()} samples "
                  f"({100*((df['crack_initiation_risk'] >= 0.3) & (df['crack_initiation_risk'] < 0.7)).sum()/len(df):.1f}%)")
            print(f"High risk (>= 0.7):      {(df['crack_initiation_risk'] >= 0.7).sum()} samples "
                  f"({100*(df['crack_initiation_risk'] >= 0.7).sum()/len(df):.1f}%)")
    
    @staticmethod
    def save_metadata(output_dir: str, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Save dataset metadata"""
        metadata = {
            'generated_date': datetime.now().isoformat(),
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'total_samples': len(train_df) + len(val_df),
            'input_features': {
                'sintering_temperature_range': [1200, 1500],
                'sintering_temperature_unit': 'Celsius',
                'cooling_rate_range': [1, 10],
                'cooling_rate_unit': 'C/min',
                'TEC_mismatch': 2.3e-6,
                'TEC_mismatch_unit': 'K^-1',
                'porosity_range': [0, 30],
                'porosity_unit': 'percent'
            },
            'output_labels': [
                'stress_hotspot_intensity',
                'crack_initiation_risk',
                'delamination_probability'
            ],
            'validation_characteristics': {
                'DIC_noise_level': 0.05,
                'XRD_noise_level': 0.08,
                'invalid_measurement_rate': 0.05
            },
            'feature_columns': list(train_df.columns),
            'physics_based_simulation': True,
            'random_seed': 42
        }
        
        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved metadata to {metadata_path}")


def main():
    """Main function to generate all datasets"""
    print("="*70)
    print("ML Training Dataset Generator for Sintering Analysis")
    print("ANN and PINN Models")
    print("="*70)
    
    # Create output directory
    output_dir = '/workspace/ml_training_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SinteringDatasetGenerator(num_samples=10000, random_seed=42)
    
    # Generate training dataset
    print("\n[1/4] Generating Training Dataset...")
    train_dataset = generator.generate_training_dataset()
    
    # Generate validation dataset
    print("\n[2/4] Generating Validation Dataset...")
    val_dataset = generator.generate_validation_dataset(num_validation=1000)
    
    # Save datasets
    print("\n[3/4] Saving Datasets...")
    train_path = os.path.join(output_dir, 'training_dataset.csv')
    val_path = os.path.join(output_dir, 'validation_dataset.csv')
    
    train_dataset.to_csv(train_path, index=False)
    val_dataset.to_csv(val_path, index=False)
    
    print(f"✓ Saved training dataset to {train_path}")
    print(f"✓ Saved validation dataset to {val_path}")
    
    # Also save in parquet format for better performance
    train_parquet = os.path.join(output_dir, 'training_dataset.parquet')
    val_parquet = os.path.join(output_dir, 'validation_dataset.parquet')
    
    train_dataset.to_parquet(train_parquet, index=False)
    val_dataset.to_parquet(val_parquet, index=False)
    
    print(f"✓ Saved training dataset to {train_parquet}")
    print(f"✓ Saved validation dataset to {val_parquet}")
    
    # Print statistics
    print("\n[4/4] Computing Statistics...")
    stats = DatasetStatistics()
    stats.print_statistics(train_dataset, "Training Dataset")
    stats.print_statistics(val_dataset, "Validation Dataset")
    
    # Save metadata
    stats.save_metadata(output_dir, train_dataset, val_dataset)
    
    # Print summary
    print("\n" + "="*70)
    print("Dataset Generation Complete!")
    print("="*70)
    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Total samples:      {len(train_dataset) + len(val_dataset):,}")
    print(f"\nOutput directory:   {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - training_dataset.csv")
    print(f"  - training_dataset.parquet")
    print(f"  - validation_dataset.csv")
    print(f"  - validation_dataset.parquet")
    print(f"  - dataset_metadata.json")
    print("="*70)


if __name__ == "__main__":
    main()
