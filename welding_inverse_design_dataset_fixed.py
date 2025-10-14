#!/usr/bin/env python3
"""
Welding Inverse Design Dataset Generator
========================================

This module generates a comprehensive multi-tiered dataset for welding inverse design
research, focusing on extreme-temperature performance prediction.

The dataset includes:
- Tier 1: High-fidelity experimental data (100-500 samples)
- Tier 2: Computational FEM simulation data (10,000+ samples)  
- Tier 3: Literature and legacy data curation

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

class WeldingDatasetGenerator:
    """Main class for generating welding inverse design datasets."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define parameter ranges based on literature and industry standards
        self.parameter_ranges = {
            # Energy Input Parameters
            'laser_power': (500, 3000),  # W
            'welding_speed': (10, 200),  # mm/s
            'pulse_frequency': (1, 1000),  # Hz
            'pulse_duration': (0.1, 20),  # ms
            
            # Beam Characteristics
            'beam_focus_position': (-2, 2),  # mm
            'beam_spot_size': (50, 500),  # µm
            
            # Material & Setup
            'clamping_pressure': (0.1, 2.0),  # MPa
            'shield_gas_flow': (5, 30),  # L/min
            'material_thickness': (0.1, 3.0),  # mm
            'overlap_distance': (0.5, 5.0),  # mm
        }
        
        # Material combinations (encoded)
        self.material_combinations = {
            'Cu-Al': 0,
            'Al-Al': 1, 
            'Cu-Steel': 2,
            'Al-Steel': 3
        }
        
        # Define output property ranges
        self.output_ranges = {
            # Weld Morphology
            'nugget_width': (0.5, 3.0),  # mm
            'penetration_depth': (0.1, 2.0),  # mm
            'haz_width': (0.2, 1.5),  # mm
            
            # Defects (binary and continuous)
            'crack_presence': (0, 1),  # binary
            'porosity_percentage': (0, 15),  # %
            'undercut_depth': (0, 0.3),  # mm
            
            # Mechanical Properties
            'tensile_shear_strength': (500, 5000),  # N
            'peel_strength': (50, 800),  # N
            'contact_resistance': (5, 100),  # µΩ
            
            # Extreme Temperature Performance
            'thermal_cycles_to_failure': (100, 2000),  # cycles
            'strength_degradation_pct': (0, 50),  # %
            'resistance_increase_pct': (0, 200),  # %
            'imc_thickness': (0.1, 10),  # µm
            'creep_time_to_failure': (1, 1000),  # hours
        }
    
    def generate_tier1_experimental_data(self, n_samples=300):
        """
        Generate Tier 1 high-fidelity experimental data using Design of Experiments.
        Uses Latin Hypercube Sampling for efficient parameter space exploration.
        """
        print("Generating Tier 1 Experimental Data...")
        
        # Latin Hypercube Sampling for input parameters
        from scipy.stats import qmc
        
        # Define parameter bounds
        bounds = []
        param_names = []
        
        for param, (min_val, max_val) in self.parameter_ranges.items():
            bounds.append([min_val, max_val])
            param_names.append(param)
        
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=len(bounds), seed=self.random_state)
        lhs_samples = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        input_data = np.zeros((n_samples, len(param_names)))
        for i, (min_val, max_val) in enumerate(bounds):
            input_data[:, i] = lhs_samples[:, i] * (max_val - min_val) + min_val
        
        # Add material combinations
        material_combinations = np.random.choice(
            list(self.material_combinations.values()),
            size=n_samples
        )
        input_data = np.column_stack([input_data, material_combinations])
        param_names.append('material_combination')
        
        # Generate realistic output data using physics-based models
        output_data = self._generate_physics_based_outputs(input_data, param_names, fidelity='high')
        
        # Create DataFrame
        df = pd.DataFrame(input_data, columns=param_names)
        output_df = pd.DataFrame(output_data, columns=list(self.output_ranges.keys()))
        df = pd.concat([df, output_df], axis=1)
        
        # Add metadata
        df['data_source'] = 'Experimental'
        df['fidelity_level'] = 'High'
        df['weld_id'] = [f'W-{i+1:03d}' for i in range(n_samples)]
        
        return df
    
    def generate_tier2_simulation_data(self, n_samples=10000):
        """
        Generate Tier 2 computational data using FEM simulation models.
        """
        print("Generating Tier 2 Simulation Data...")
        
        # Generate more samples for simulation data
        input_data = np.random.uniform(
            low=[self.parameter_ranges[param][0] for param in self.parameter_ranges.keys()],
            high=[self.parameter_ranges[param][1] for param in self.parameter_ranges.keys()],
            size=(n_samples, len(self.parameter_ranges))
        )
        
        # Add material combinations
        material_combinations = np.random.choice(
            list(self.material_combinations.values()),
            size=n_samples
        )
        input_data = np.column_stack([input_data, material_combinations])
        
        param_names = list(self.parameter_ranges.keys()) + ['material_combination']
        
        # Generate simulation outputs (with more noise than experimental)
        output_data = self._generate_physics_based_outputs(input_data, param_names, fidelity='medium')
        
        # Create DataFrame
        df = pd.DataFrame(input_data, columns=param_names)
        output_df = pd.DataFrame(output_data, columns=list(self.output_ranges.keys()))
        df = pd.concat([df, output_df], axis=1)
        
        # Add metadata
        df['data_source'] = 'Simulation'
        df['fidelity_level'] = 'Medium'
        df['weld_id'] = [f'S-{i+1:05d}' for i in range(n_samples)]
        
        return df
    
    def generate_tier3_literature_data(self, n_samples=500):
        """
        Generate Tier 3 literature and legacy data.
        """
        print("Generating Tier 3 Literature Data...")
        
        # Literature data typically has more limited parameter ranges
        literature_ranges = {
            'laser_power': (800, 2500),
            'welding_speed': (20, 150),
            'pulse_frequency': (10, 500),
            'pulse_duration': (1, 15),
            'beam_focus_position': (-1, 1),
            'beam_spot_size': (100, 400),
            'clamping_pressure': (0.2, 1.5),
            'shield_gas_flow': (8, 25),
            'material_thickness': (0.2, 2.0),
            'overlap_distance': (1.0, 4.0),
        }
        
        # Generate literature-style data
        input_data = np.random.uniform(
            low=[literature_ranges[param][0] for param in literature_ranges.keys()],
            high=[literature_ranges[param][1] for param in literature_ranges.keys()],
            size=(n_samples, len(literature_ranges))
        )
        
        # Add material combinations (more conservative)
        material_combinations = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])  # Cu-Al, Al-Al
        input_data = np.column_stack([input_data, material_combinations])
        
        param_names = list(literature_ranges.keys()) + ['material_combination']
        
        # Generate outputs with literature-style characteristics
        output_data = self._generate_physics_based_outputs(input_data, param_names, fidelity='low')
        
        # Create DataFrame
        df = pd.DataFrame(input_data, columns=param_names)
        output_df = pd.DataFrame(output_data, columns=list(self.output_ranges.keys()))
        df = pd.concat([df, output_df], axis=1)
        
        # Add metadata
        df['data_source'] = 'Literature'
        df['fidelity_level'] = 'Low'
        df['weld_id'] = [f'L-{i+1:03d}' for i in range(n_samples)]
        
        return df
    
    def _generate_physics_based_outputs(self, input_data, param_names, fidelity='high'):
        """
        Generate physics-based output properties using empirical models.
        """
        n_samples = input_data.shape[0]
        output_data = np.zeros((n_samples, len(self.output_ranges)))
        output_names = list(self.output_ranges.keys())
        
        # Get parameter indices
        power_idx = param_names.index('laser_power')
        speed_idx = param_names.index('welding_speed')
        thickness_idx = param_names.index('material_thickness')
        material_idx = param_names.index('material_combination')
        
        for i in range(n_samples):
            power = input_data[i, power_idx]
            speed = input_data[i, speed_idx]
            thickness = input_data[i, thickness_idx]
            material = input_data[i, material_idx]
            
            # Calculate energy density
            energy_density = power / (speed * thickness)
            
            # Weld morphology (physics-based relationships)
            nugget_width = 0.8 + 0.4 * np.sqrt(energy_density/1000) + np.random.normal(0, 0.1)
            penetration_depth = 0.3 + 0.2 * np.sqrt(energy_density/1000) + np.random.normal(0, 0.05)
            haz_width = 0.4 + 0.3 * np.sqrt(energy_density/1000) + np.random.normal(0, 0.05)
            
            # Defects (inversely related to energy density and material compatibility)
            crack_prob = max(0, 0.3 - 0.2 * np.sqrt(energy_density/1000) + 0.1 * material)
            crack_presence = 1 if np.random.random() < crack_prob else 0
            
            porosity = max(0, 8 - 2 * np.sqrt(energy_density/1000) + np.random.normal(0, 2))
            undercut = max(0, 0.1 - 0.05 * np.sqrt(energy_density/1000) + np.random.normal(0, 0.02))
            
            # Mechanical properties (material-dependent)
            base_strength = 2000 if material == 0 else 1500  # Cu-Al vs Al-Al
            tensile_strength = base_strength + 500 * np.sqrt(energy_density/1000) + np.random.normal(0, 100)
            peel_strength = 200 + 100 * np.sqrt(energy_density/1000) + np.random.normal(0, 20)
            
            # Electrical properties
            contact_resistance = 20 + 30 * (1 - np.sqrt(energy_density/1000)) + np.random.normal(0, 5)
            
            # Extreme temperature performance
            cycles_to_failure = 500 + 300 * np.sqrt(energy_density/1000) - 100 * material + np.random.normal(0, 50)
            strength_degradation = 10 + 20 * (1 - np.sqrt(energy_density/1000)) + 5 * material + np.random.normal(0, 3)
            resistance_increase = 50 + 100 * (1 - np.sqrt(energy_density/1000)) + 20 * material + np.random.normal(0, 10)
            
            # IMC thickness (critical for dissimilar metal joints)
            imc_thickness = 1.0 + 2.0 * material + 0.5 * np.sqrt(energy_density/1000) + np.random.normal(0, 0.3)
            
            # Creep performance
            creep_time = 100 + 200 * np.sqrt(energy_density/1000) - 50 * material + np.random.normal(0, 20)
            
            # Apply fidelity-based noise
            noise_factor = {'high': 0.05, 'medium': 0.15, 'low': 0.25}[fidelity]
            
            outputs = [
                max(0.1, nugget_width + np.random.normal(0, abs(noise_factor * nugget_width))),
                max(0.05, penetration_depth + np.random.normal(0, abs(noise_factor * penetration_depth))),
                max(0.1, haz_width + np.random.normal(0, abs(noise_factor * haz_width))),
                crack_presence,
                max(0, min(15, porosity + np.random.normal(0, abs(noise_factor * porosity)))),
                max(0, undercut + np.random.normal(0, abs(noise_factor * undercut))),
                max(100, tensile_strength + np.random.normal(0, abs(noise_factor * tensile_strength))),
                max(10, peel_strength + np.random.normal(0, abs(noise_factor * peel_strength))),
                max(1, contact_resistance + np.random.normal(0, abs(noise_factor * contact_resistance))),
                max(50, cycles_to_failure + np.random.normal(0, abs(noise_factor * cycles_to_failure))),
                max(0, min(80, strength_degradation + np.random.normal(0, abs(noise_factor * strength_degradation)))),
                max(0, resistance_increase + np.random.normal(0, abs(noise_factor * resistance_increase))),
                max(0.1, imc_thickness + np.random.normal(0, abs(noise_factor * imc_thickness))),
                max(1, creep_time + np.random.normal(0, abs(noise_factor * creep_time)))
            ]
            
            output_data[i, :] = outputs
        
        return output_data
    
    def generate_extreme_temperature_testing_data(self, base_data, n_thermal_cycles=1000):
        """
        Generate extreme temperature testing data by subjecting welds to thermal cycling.
        """
        print("Generating Extreme Temperature Testing Data...")
        
        extreme_data = base_data.copy()
        
        # Simulate thermal cycling effects
        for idx, row in extreme_data.iterrows():
            # Calculate degradation based on material properties and initial conditions
            base_cycles = row['thermal_cycles_to_failure']
            material = row['material_combination']
            
            # Simulate different thermal cycling profiles
            temp_ranges = np.random.choice([85, 100, 125], p=[0.5, 0.3, 0.2])  # Temperature range in °C
            
            # Calculate degradation factors
            degradation_factor = 1 + (temp_ranges - 85) / 100  # Higher temp = more degradation
            material_factor = 1 + 0.3 * material  # Dissimilar metals degrade faster
            
            # Apply degradation to properties
            extreme_data.loc[idx, 'strength_degradation_pct'] *= degradation_factor * material_factor
            extreme_data.loc[idx, 'resistance_increase_pct'] *= degradation_factor * material_factor
            extreme_data.loc[idx, 'imc_thickness'] *= (1 + 0.2 * degradation_factor * material_factor)
            
            # Update cycles to failure based on thermal stress
            extreme_data.loc[idx, 'thermal_cycles_to_failure'] = max(50, 
                base_cycles / (degradation_factor * material_factor))
        
        extreme_data['thermal_cycling_applied'] = True
        extreme_data['max_temp_celsius'] = np.random.choice([85, 100, 125], size=len(extreme_data), p=[0.5, 0.3, 0.2])
        
        return extreme_data
    
    def create_master_dataset(self, n_exp=300, n_sim=10000, n_lit=500):
        """
        Create the master dataset by combining all three tiers.
        """
        print("Creating Master Dataset...")
        
        # Generate all three tiers
        tier1_data = self.generate_tier1_experimental_data(n_exp)
        tier2_data = self.generate_tier2_simulation_data(n_sim)
        tier3_data = self.generate_tier3_literature_data(n_lit)
        
        # Apply extreme temperature testing to a subset of experimental data
        extreme_subset = tier1_data.sample(n=min(100, len(tier1_data)), random_state=self.random_state)
        extreme_data = self.generate_extreme_temperature_testing_data(extreme_subset)
        
        # Combine all data
        master_dataset = pd.concat([tier1_data, tier2_data, tier3_data, extreme_data], 
                                 ignore_index=True)
        
        # Add comprehensive metadata
        master_dataset['dataset_version'] = '1.0'
        master_dataset['generation_date'] = pd.Timestamp.now()
        master_dataset['total_samples'] = len(master_dataset)
        
        # Create quality scores based on data source
        quality_scores = {
            'Experimental': 1.0,
            'Simulation': 0.7,
            'Literature': 0.5
        }
        master_dataset['quality_score'] = master_dataset['data_source'].map(quality_scores)
        
        return master_dataset
    
    def create_ml_ready_dataset(self, master_dataset, test_size=0.2, val_size=0.1):
        """
        Prepare dataset for ML training with proper splits and preprocessing.
        """
        print("Creating ML-Ready Dataset...")
        
        # Separate input and output features
        input_features = [col for col in master_dataset.columns 
                         if col not in self.output_ranges.keys() and 
                         col not in ['weld_id', 'data_source', 'fidelity_level', 
                                   'dataset_version', 'generation_date', 'total_samples',
                                   'quality_score', 'thermal_cycling_applied', 'max_temp_celsius']]
        
        output_features = list(self.output_ranges.keys())
        
        X = master_dataset[input_features]
        y = master_dataset[output_features]
        
        # Create stratified splits based on data source
        train_idx, temp_idx = train_test_split(
            range(len(master_dataset)), 
            test_size=test_size + val_size, 
            stratify=master_dataset['data_source'],
            random_state=self.random_state
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=test_size/(test_size + val_size),
            stratify=master_dataset.iloc[temp_idx]['data_source'],
            random_state=self.random_state
        )
        
        # Create splits
        splits = {
            'train': master_dataset.iloc[train_idx],
            'validation': master_dataset.iloc[val_idx], 
            'test': master_dataset.iloc[test_idx]
        }
        
        # Preprocessing
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Create feature importance analysis
        feature_importance = self._analyze_feature_importance(X, y)
        
        return {
            'splits': splits,
            'X_scaled': X_scaled,
            'y_scaled': y_scaled,
            'input_features': input_features,
            'output_features': output_features,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_importance': feature_importance
        }
    
    def _analyze_feature_importance(self, X, y):
        """
        Analyze feature importance using correlation analysis.
        """
        correlations = {}
        for output_col in y.columns:
            corr_matrix = X.corrwith(y[output_col]).abs().sort_values(ascending=False)
            correlations[output_col] = corr_matrix.to_dict()
        
        return correlations
    
    def generate_dataset_report(self, master_dataset, ml_data):
        """
        Generate comprehensive dataset report with statistics and visualizations.
        """
        print("Generating Dataset Report...")
        
        # Create comprehensive report
        numeric_cols = master_dataset.select_dtypes(include=[np.number]).columns
        report = {
            'dataset_summary': {
                'total_samples': len(master_dataset),
                'experimental_samples': len(master_dataset[master_dataset['data_source'] == 'Experimental']),
                'simulation_samples': len(master_dataset[master_dataset['data_source'] == 'Simulation']),
                'literature_samples': len(master_dataset[master_dataset['data_source'] == 'Literature']),
                'extreme_temp_samples': len(master_dataset[master_dataset.get('thermal_cycling_applied', False) == True])
            },
            'parameter_statistics': master_dataset.describe(),
            'correlation_analysis': master_dataset[numeric_cols].corr(),
            'feature_importance': ml_data['feature_importance']
        }
        
        return report

def main():
    """Main function to generate the complete welding inverse design dataset."""
    print("=" * 60)
    print("WELDING INVERSE DESIGN DATASET GENERATOR")
    print("=" * 60)
    
    # Initialize generator
    generator = WeldingDatasetGenerator(random_state=42)
    
    # Generate master dataset
    master_dataset = generator.create_master_dataset(
        n_exp=300,    # Experimental samples
        n_sim=10000,  # Simulation samples  
        n_lit=500     # Literature samples
    )
    
    # Create ML-ready dataset
    ml_data = generator.create_ml_ready_dataset(master_dataset)
    
    # Generate report
    report = generator.generate_dataset_report(master_dataset, ml_data)
    
    # Save datasets
    print("\nSaving datasets...")
    master_dataset.to_csv('welding_master_dataset.csv', index=False)
    
    # Save ML-ready splits
    for split_name, split_data in ml_data['splits'].items():
        split_data.to_csv(f'welding_dataset_{split_name}.csv', index=False)
    
    # Save metadata
    import json
    with open('dataset_metadata.json', 'w') as f:
        json.dump({
            'input_features': ml_data['input_features'],
            'output_features': ml_data['output_features'],
            'dataset_summary': report['dataset_summary']
        }, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(master_dataset)}")
    print(f"Experimental: {report['dataset_summary']['experimental_samples']}")
    print(f"Simulation: {report['dataset_summary']['simulation_samples']}")
    print(f"Literature: {report['dataset_summary']['literature_samples']}")
    print(f"Extreme temp tested: {report['dataset_summary']['extreme_temp_samples']}")
    
    return master_dataset, ml_data, report

if __name__ == "__main__":
    master_dataset, ml_data, report = main()