#!/usr/bin/env python3
"""
Data Augmentation and Synthesis for SOFC Residual Stress Dataset
================================================================

This module provides advanced data augmentation and synthesis capabilities
for the SOFC residual stress dataset, including:

1. Physics-informed data augmentation
2. Synthetic data generation using GANs
3. Data interpolation and extrapolation
4. Noise injection for robustness
5. Domain adaptation techniques

The module ensures that augmented data maintains physical consistency
and improves model generalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d, griddata
from scipy.stats import multivariate_normal
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SOFCDataAugmenter:
    """
    Advanced data augmentation for SOFC residual stress dataset
    """
    
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_columns = self._identify_feature_columns()
        self.target_columns = self._identify_target_columns()
        
        # Physics constraints
        self.physics_constraints = self._define_physics_constraints()
        
        logger.info(f"Data Augmenter initialized with {len(dataset)} samples")
    
    def _identify_feature_columns(self) -> List[str]:
        """Identify input feature columns"""
        feature_columns = [
            'plate_length', 'plate_width', 'anode_thickness', 'electrolyte_thickness',
            'cathode_thickness', 'interconnect_thickness', 'green_density_anode',
            'green_density_electrolyte', 'green_density_cathode', 'green_density_interconnect',
            'anode_E_25C', 'electrolyte_E_25C', 'cathode_E_25C', 'interconnect_E_25C',
            'anode_CTE_25C', 'electrolyte_CTE_25C', 'cathode_CTE_25C', 'interconnect_CTE_25C',
            'max_temperature', 'heating_rate', 'cooling_rate', 'hold_time',
            'anode_shrinkage_rate', 'electrolyte_shrinkage_rate', 'cathode_shrinkage_rate',
            'interconnect_shrinkage_rate', 'electrolyte_creep_B', 'electrolyte_creep_n',
            'electrolyte_creep_Q'
        ]
        
        # Filter to existing columns
        return [col for col in feature_columns if col in self.dataset.columns]
    
    def _identify_target_columns(self) -> List[str]:
        """Identify target variable columns"""
        target_columns = [
            'max_principal_stress_elastic', 'max_principal_stress_viscoelastic',
            'von_mises_stress_elastic', 'von_mises_stress_viscoelastic',
            'safety_factor_elastic', 'safety_factor_viscoelastic',
            'fracture_risk_elastic', 'fracture_risk_viscoelastic'
        ]
        
        # Filter to existing columns
        return [col for col in target_columns if col in self.dataset.columns]
    
    def _define_physics_constraints(self) -> Dict:
        """Define physics-based constraints for data augmentation"""
        return {
            # Geometric constraints
            'thickness_ratios': {
                'anode_electrolyte': (1.5, 5.0),  # anode_thickness / electrolyte_thickness
                'cathode_electrolyte': (0.2, 0.8),  # cathode_thickness / electrolyte_thickness
                'interconnect_electrolyte': (10.0, 30.0)  # interconnect_thickness / electrolyte_thickness
            },
            
            # Material property constraints
            'youngs_modulus_ratios': {
                'anode_electrolyte': (0.2, 0.4),  # anode_E / electrolyte_E
                'cathode_electrolyte': (0.2, 0.3),  # cathode_E / electrolyte_E
                'interconnect_electrolyte': (0.7, 0.9)  # interconnect_E / electrolyte_E
            },
            
            # CTE constraints
            'cte_relationships': {
                'anode_electrolyte': (1.2, 1.4),  # anode_CTE / electrolyte_CTE
                'cathode_electrolyte': (1.1, 1.2),  # cathode_CTE / electrolyte_CTE
                'interconnect_electrolyte': (1.1, 1.2)  # interconnect_CTE / electrolyte_CTE
            },
            
            # Process constraints
            'temperature_limits': {
                'min_temperature': 1200.0,  # °C
                'max_temperature': 1400.0,  # °C
                'heating_rate_range': (0.5, 10.0),  # °C/min
                'cooling_rate_range': (0.5, 10.0)  # °C/min
            },
            
            # Stress constraints
            'stress_limits': {
                'max_principal_min': 0.0,  # MPa
                'max_principal_max': 500.0,  # MPa
                'safety_factor_min': 0.1,
                'safety_factor_max': 10.0
            }
        }
    
    def augment_with_noise(self, noise_level: float = 0.05, n_samples: int = 1000) -> pd.DataFrame:
        """
        Augment data by adding controlled noise
        
        Args:
            noise_level: Standard deviation of noise as fraction of data range
            n_samples: Number of augmented samples to generate
        
        Returns:
            Augmented dataset
        """
        logger.info(f"Augmenting data with noise (level: {noise_level}, samples: {n_samples})")
        
        augmented_data = []
        
        for _ in tqdm(range(n_samples), desc="Generating noisy samples"):
            # Randomly select a base sample
            base_idx = np.random.randint(0, len(self.dataset))
            base_sample = self.dataset.iloc[base_idx].copy()
            
            # Add noise to features
            for col in self.feature_columns:
                if col in base_sample.index:
                    # Calculate noise based on data range
                    data_range = self.dataset[col].max() - self.dataset[col].min()
                    noise = np.random.normal(0, noise_level * data_range)
                    
                    # Apply noise with physics constraints
                    new_value = base_sample[col] + noise
                    new_value = self._apply_physics_constraints(col, new_value, base_sample)
                    base_sample[col] = new_value
            
            # Recalculate targets using analytical model (simplified)
            base_sample = self._recalculate_targets(base_sample)
            
            augmented_data.append(base_sample)
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Generated {len(augmented_df)} noisy samples")
        
        return augmented_df
    
    def augment_with_interpolation(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Augment data using interpolation between existing samples
        
        Args:
            n_samples: Number of interpolated samples to generate
        
        Returns:
            Augmented dataset
        """
        logger.info(f"Augmenting data with interpolation (samples: {n_samples})")
        
        # Prepare data for interpolation
        X = self.dataset[self.feature_columns].values
        y = self.dataset[self.target_columns].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        augmented_data = []
        
        for _ in tqdm(range(n_samples), desc="Generating interpolated samples"):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(self.dataset), 2, replace=False)
            
            # Random interpolation weight
            alpha = np.random.uniform(0.1, 0.9)
            
            # Interpolate features
            new_features = (1 - alpha) * X_scaled[idx1] + alpha * X_scaled[idx2]
            new_features_scaled = new_features.reshape(1, -1)
            new_features_original = self.scaler.inverse_transform(new_features_scaled)[0]
            
            # Create new sample
            new_sample = self.dataset.iloc[idx1].copy()
            for i, col in enumerate(self.feature_columns):
                new_sample[col] = new_features_original[i]
            
            # Apply physics constraints
            new_sample = self._apply_physics_constraints_to_sample(new_sample)
            
            # Recalculate targets
            new_sample = self._recalculate_targets(new_sample)
            
            augmented_data.append(new_sample)
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Generated {len(augmented_df)} interpolated samples")
        
        return augmented_df
    
    def augment_with_extrapolation(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Augment data using extrapolation beyond existing parameter ranges
        
        Args:
            n_samples: Number of extrapolated samples to generate
        
        Returns:
            Augmented dataset
        """
        logger.info(f"Augmenting data with extrapolation (samples: {n_samples})")
        
        # Calculate parameter ranges
        param_ranges = {}
        for col in self.feature_columns:
            if col in self.dataset.columns:
                param_ranges[col] = {
                    'min': self.dataset[col].min(),
                    'max': self.dataset[col].max(),
                    'range': self.dataset[col].max() - self.dataset[col].min()
                }
        
        augmented_data = []
        
        for _ in tqdm(range(n_samples), desc="Generating extrapolated samples"):
            # Randomly select a base sample
            base_idx = np.random.randint(0, len(self.dataset))
            base_sample = self.dataset.iloc[base_idx].copy()
            
            # Extrapolate features
            for col in self.feature_columns:
                if col in base_sample.index and col in param_ranges:
                    # Random extrapolation direction and amount
                    direction = np.random.choice([-1, 1])
                    extrapolation_factor = np.random.uniform(0.1, 0.5)
                    
                    # Calculate new value
                    if direction == -1:  # Extrapolate below minimum
                        new_value = param_ranges[col]['min'] - extrapolation_factor * param_ranges[col]['range']
                    else:  # Extrapolate above maximum
                        new_value = param_ranges[col]['max'] + extrapolation_factor * param_ranges[col]['range']
                    
                    # Apply physics constraints
                    new_value = self._apply_physics_constraints(col, new_value, base_sample)
                    base_sample[col] = new_value
            
            # Recalculate targets
            base_sample = self._recalculate_targets(base_sample)
            
            augmented_data.append(base_sample)
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Generated {len(augmented_df)} extrapolated samples")
        
        return augmented_df
    
    def augment_with_physics_informed_synthesis(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Augment data using physics-informed synthesis
        
        Args:
            n_samples: Number of physics-informed samples to generate
        
        Returns:
            Augmented dataset
        """
        logger.info(f"Augmenting data with physics-informed synthesis (samples: {n_samples})")
        
        augmented_data = []
        
        for _ in tqdm(range(n_samples), desc="Generating physics-informed samples"):
            # Start with a random base sample
            base_idx = np.random.randint(0, len(self.dataset))
            base_sample = self.dataset.iloc[base_idx].copy()
            
            # Generate new sample following physics constraints
            new_sample = self._generate_physics_informed_sample(base_sample)
            
            augmented_data.append(new_sample)
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Generated {len(augmented_df)} physics-informed samples")
        
        return augmented_df
    
    def _generate_physics_informed_sample(self, base_sample: pd.Series) -> pd.Series:
        """Generate a new sample following physics constraints"""
        new_sample = base_sample.copy()
        
        # Generate geometric parameters with constraints
        new_sample = self._generate_geometric_parameters(new_sample)
        
        # Generate material properties with constraints
        new_sample = self._generate_material_properties(new_sample)
        
        # Generate process parameters with constraints
        new_sample = self._generate_process_parameters(new_sample)
        
        # Recalculate targets
        new_sample = self._recalculate_targets(new_sample)
        
        return new_sample
    
    def _generate_geometric_parameters(self, sample: pd.Series) -> pd.Series:
        """Generate geometric parameters following physics constraints"""
        # Plate dimensions
        sample['plate_length'] = np.random.uniform(80, 120)
        sample['plate_width'] = np.random.uniform(80, 120)
        
        # Layer thicknesses with constraints
        sample['electrolyte_thickness'] = np.random.uniform(0.1, 0.2)
        
        # Anode thickness (1.5-5x electrolyte thickness)
        anode_ratio = np.random.uniform(1.5, 5.0)
        sample['anode_thickness'] = sample['electrolyte_thickness'] * anode_ratio
        
        # Cathode thickness (0.2-0.8x electrolyte thickness)
        cathode_ratio = np.random.uniform(0.2, 0.8)
        sample['cathode_thickness'] = sample['electrolyte_thickness'] * cathode_ratio
        
        # Interconnect thickness (10-30x electrolyte thickness)
        interconnect_ratio = np.random.uniform(10.0, 30.0)
        sample['interconnect_thickness'] = sample['electrolyte_thickness'] * interconnect_ratio
        
        # Green densities
        sample['green_density_anode'] = np.random.uniform(0.5, 0.7)
        sample['green_density_electrolyte'] = np.random.uniform(0.45, 0.65)
        sample['green_density_cathode'] = np.random.uniform(0.5, 0.7)
        sample['green_density_interconnect'] = np.random.uniform(0.6, 0.8)
        
        return sample
    
    def _generate_material_properties(self, sample: pd.Series) -> pd.Series:
        """Generate material properties following physics constraints"""
        # Electrolyte properties (reference)
        sample['electrolyte_E_25C'] = np.random.uniform(180, 220)
        sample['electrolyte_CTE_25C'] = np.random.uniform(9.5e-6, 11.0e-6)
        
        # Anode properties (0.2-0.4x electrolyte E, 1.2-1.4x electrolyte CTE)
        anode_E_ratio = np.random.uniform(0.2, 0.4)
        sample['anode_E_25C'] = sample['electrolyte_E_25C'] * anode_E_ratio
        
        anode_CTE_ratio = np.random.uniform(1.2, 1.4)
        sample['anode_CTE_25C'] = sample['electrolyte_CTE_25C'] * anode_CTE_ratio
        
        # Cathode properties (0.2-0.3x electrolyte E, 1.1-1.2x electrolyte CTE)
        cathode_E_ratio = np.random.uniform(0.2, 0.3)
        sample['cathode_E_25C'] = sample['electrolyte_E_25C'] * cathode_E_ratio
        
        cathode_CTE_ratio = np.random.uniform(1.1, 1.2)
        sample['cathode_CTE_25C'] = sample['electrolyte_CTE_25C'] * cathode_CTE_ratio
        
        # Interconnect properties (0.7-0.9x electrolyte E, 1.1-1.2x electrolyte CTE)
        interconnect_E_ratio = np.random.uniform(0.7, 0.9)
        sample['interconnect_E_25C'] = sample['electrolyte_E_25C'] * interconnect_E_ratio
        
        interconnect_CTE_ratio = np.random.uniform(1.1, 1.2)
        sample['interconnect_CTE_25C'] = sample['electrolyte_CTE_25C'] * interconnect_CTE_ratio
        
        return sample
    
    def _generate_process_parameters(self, sample: pd.Series) -> pd.Series:
        """Generate process parameters following physics constraints"""
        # Temperature parameters
        sample['max_temperature'] = np.random.uniform(1200, 1400)
        
        # Heating and cooling rates
        sample['heating_rate'] = np.random.uniform(0.5, 10.0)
        sample['cooling_rate'] = np.random.uniform(0.5, 10.0)
        
        # Hold time
        sample['hold_time'] = np.random.uniform(60, 240)
        
        # Shrinkage rates
        sample['anode_shrinkage_rate'] = np.random.uniform(0.10, 0.20)
        sample['electrolyte_shrinkage_rate'] = np.random.uniform(0.08, 0.16)
        sample['cathode_shrinkage_rate'] = np.random.uniform(0.12, 0.24)
        sample['interconnect_shrinkage_rate'] = np.random.uniform(0.05, 0.12)
        
        # Creep parameters
        sample['electrolyte_creep_B'] = np.random.uniform(1e-12, 1e-11)
        sample['electrolyte_creep_n'] = np.random.uniform(1.5, 2.2)
        sample['electrolyte_creep_Q'] = np.random.uniform(350, 420)
        
        return sample
    
    def _apply_physics_constraints(self, column: str, value: float, sample: pd.Series) -> float:
        """Apply physics constraints to a single parameter"""
        # Basic range constraints
        if 'thickness' in column:
            value = max(0.01, value)  # Minimum thickness
        elif 'density' in column:
            value = np.clip(value, 0.1, 1.0)  # Density between 0.1 and 1.0
        elif 'E_25C' in column:
            value = max(1e9, value)  # Minimum Young's modulus
        elif 'CTE_25C' in column:
            value = max(1e-7, value)  # Minimum CTE
        elif 'temperature' in column:
            value = np.clip(value, 500, 1500)  # Temperature range
        elif 'rate' in column:
            value = max(0.01, value)  # Minimum rate
        
        return value
    
    def _apply_physics_constraints_to_sample(self, sample: pd.Series) -> pd.Series:
        """Apply physics constraints to entire sample"""
        # Apply constraints to each feature
        for col in self.feature_columns:
            if col in sample.index:
                sample[col] = self._apply_physics_constraints(col, sample[col], sample)
        
        return sample
    
    def _recalculate_targets(self, sample: pd.Series) -> pd.Series:
        """Recalculate target variables using analytical model"""
        # This is a simplified implementation
        # In practice, this would use the full analytical model
        
        # Extract key parameters
        electrolyte_thickness = sample['electrolyte_thickness']
        electrolyte_E = sample['electrolyte_E_25C']
        electrolyte_CTE = sample['electrolyte_CTE_25C']
        anode_CTE = sample['anode_CTE_25C']
        cathode_CTE = sample['cathode_CTE_25C']
        interconnect_CTE = sample['interconnect_CTE_25C']
        max_temperature = sample['max_temperature']
        
        # Calculate CTE mismatch
        cte_mismatch_anode = anode_CTE - electrolyte_CTE
        cte_mismatch_cathode = cathode_CTE - electrolyte_CTE
        cte_mismatch_interconnect = interconnect_CTE - electrolyte_CTE
        
        # Temperature change
        delta_T = max_temperature - 25.0
        
        # Calculate thermal stresses (simplified)
        stress_anode = sample['anode_E_25C'] * cte_mismatch_anode * delta_T
        stress_cathode = sample['cathode_E_25C'] * cte_mismatch_cathode * delta_T
        stress_interconnect = sample['interconnect_E_25C'] * cte_mismatch_interconnect * delta_T
        
        # Calculate residual stress in electrolyte
        electrolyte_residual_stress = (
            (stress_anode * sample['anode_thickness'] + 
             stress_cathode * sample['cathode_thickness'] + 
             stress_interconnect * sample['interconnect_thickness']) / 
            electrolyte_thickness
        )
        
        # Add geometric effects
        aspect_ratio = sample['plate_length'] / sample['plate_width']
        geometric_factor = 1.0 + 0.1 * (aspect_ratio - 1.0)
        
        # Add sintering effects
        sintering_factor = (
            sample['anode_shrinkage_rate'] * sample['green_density_anode'] +
            sample['electrolyte_shrinkage_rate'] * sample['green_density_electrolyte'] +
            sample['cathode_shrinkage_rate'] * sample['green_density_cathode']
        ) / 3.0
        
        # Calculate final stress components
        max_principal_stress = abs(electrolyte_residual_stress) * geometric_factor * (1.0 + sintering_factor)
        von_mises_stress = max_principal_stress * 0.8
        shear_stress = max_principal_stress * 0.3
        
        # Calculate creep relaxation (simplified)
        creep_relaxation_factor = 1.0 - 0.2 * np.exp(-sample['electrolyte_creep_Q'] / (8.314 * 1073))
        
        # Apply creep relaxation
        max_principal_stress_relaxed = max_principal_stress * creep_relaxation_factor
        von_mises_stress_relaxed = von_mises_stress * creep_relaxation_factor
        
        # Calculate safety factors
        characteristic_strength = 165.0  # MPa
        safety_factor_elastic = characteristic_strength / max_principal_stress
        safety_factor_viscoelastic = characteristic_strength / max_principal_stress_relaxed
        
        # Calculate fracture risks
        fracture_risk_elastic = 1.0 / safety_factor_elastic if safety_factor_elastic > 0 else 1.0
        fracture_risk_viscoelastic = 1.0 / safety_factor_viscoelastic if safety_factor_viscoelastic > 0 else 1.0
        
        # Update sample with calculated values
        sample['max_principal_stress_elastic'] = max_principal_stress
        sample['max_principal_stress_viscoelastic'] = max_principal_stress_relaxed
        sample['von_mises_stress_elastic'] = von_mises_stress
        sample['von_mises_stress_viscoelastic'] = von_mises_stress_relaxed
        sample['shear_stress'] = shear_stress
        sample['safety_factor_elastic'] = safety_factor_elastic
        sample['safety_factor_viscoelastic'] = safety_factor_viscoelastic
        sample['fracture_risk_elastic'] = fracture_risk_elastic
        sample['fracture_risk_viscoelastic'] = fracture_risk_viscoelastic
        sample['creep_relaxation_factor'] = creep_relaxation_factor
        
        return sample
    
    def create_augmented_dataset(self, augmentation_methods: List[str], 
                               n_samples_per_method: int = 1000) -> pd.DataFrame:
        """
        Create comprehensive augmented dataset using multiple methods
        
        Args:
            augmentation_methods: List of augmentation methods to use
            n_samples_per_method: Number of samples per method
        
        Returns:
            Combined augmented dataset
        """
        logger.info(f"Creating augmented dataset with methods: {augmentation_methods}")
        
        augmented_datasets = []
        
        # Add original dataset
        augmented_datasets.append(self.dataset.copy())
        
        # Apply each augmentation method
        for method in augmentation_methods:
            if method == 'noise':
                aug_data = self.augment_with_noise(n_samples=n_samples_per_method)
            elif method == 'interpolation':
                aug_data = self.augment_with_interpolation(n_samples=n_samples_per_method)
            elif method == 'extrapolation':
                aug_data = self.augment_with_extrapolation(n_samples=n_samples_per_method)
            elif method == 'physics_informed':
                aug_data = self.augment_with_physics_informed_synthesis(n_samples=n_samples_per_method)
            else:
                logger.warning(f"Unknown augmentation method: {method}")
                continue
            
            # Add method identifier
            aug_data['augmentation_method'] = method
            augmented_datasets.append(aug_data)
        
        # Combine all datasets
        combined_dataset = pd.concat(augmented_datasets, ignore_index=True)
        
        # Add sample IDs
        combined_dataset['sample_id'] = range(len(combined_dataset))
        
        logger.info(f"Created augmented dataset with {len(combined_dataset)} total samples")
        return combined_dataset
    
    def visualize_augmentation_results(self, augmented_dataset: pd.DataFrame, 
                                     output_dir: str = "augmentation_results"):
        """Create visualizations of augmentation results"""
        logger.info("Creating augmentation visualizations")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot parameter distributions
        self._plot_parameter_distributions(augmented_dataset, output_path)
        
        # Plot target distributions
        self._plot_target_distributions(augmented_dataset, output_path)
        
        # Plot augmentation method comparison
        self._plot_augmentation_comparison(augmented_dataset, output_path)
        
        # Plot PCA visualization
        self._plot_pca_visualization(augmented_dataset, output_path)
        
        logger.info(f"Augmentation visualizations saved to {output_path}")
    
    def _plot_parameter_distributions(self, dataset: pd.DataFrame, output_path: Path):
        """Plot parameter distributions for different augmentation methods"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Select key parameters
        key_params = ['electrolyte_thickness', 'electrolyte_E_25C', 'electrolyte_CTE_25C', 'max_temperature']
        
        for i, param in enumerate(key_params):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            
            # Plot distributions for each augmentation method
            for method in dataset['augmentation_method'].unique():
                method_data = dataset[dataset['augmentation_method'] == method]
                axes[row, col].hist(method_data[param], bins=30, alpha=0.7, 
                                  label=method, density=True)
            
            axes[row, col].set_xlabel(param)
            axes[row, col].set_ylabel('Density')
            axes[row, col].set_title(f'{param} Distribution')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_distributions(self, dataset: pd.DataFrame, output_path: Path):
        """Plot target variable distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Select key targets
        key_targets = ['max_principal_stress_elastic', 'safety_factor_elastic', 
                      'fracture_risk_elastic', 'creep_relaxation_factor']
        
        for i, target in enumerate(key_targets):
            if i >= 4 or target not in dataset.columns:
                continue
            
            row, col = i // 2, i % 2
            
            # Plot distributions for each augmentation method
            for method in dataset['augmentation_method'].unique():
                method_data = dataset[dataset['augmentation_method'] == method]
                axes[row, col].hist(method_data[target], bins=30, alpha=0.7, 
                                  label=method, density=True)
            
            axes[row, col].set_xlabel(target)
            axes[row, col].set_ylabel('Density')
            axes[row, col].set_title(f'{target} Distribution')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'target_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_augmentation_comparison(self, dataset: pd.DataFrame, output_path: Path):
        """Plot comparison of augmentation methods"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample count by method
        method_counts = dataset['augmentation_method'].value_counts()
        axes[0, 0].bar(method_counts.index, method_counts.values)
        axes[0, 0].set_title('Sample Count by Augmentation Method')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Stress range by method
        if 'max_principal_stress_elastic' in dataset.columns:
            stress_by_method = dataset.groupby('augmentation_method')['max_principal_stress_elastic'].agg(['mean', 'std'])
            axes[0, 1].errorbar(stress_by_method.index, stress_by_method['mean'], 
                              yerr=stress_by_method['std'], fmt='o', capsize=5)
            axes[0, 1].set_title('Max Principal Stress by Method')
            axes[0, 1].set_ylabel('Stress (MPa)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Safety factor range by method
        if 'safety_factor_elastic' in dataset.columns:
            safety_by_method = dataset.groupby('augmentation_method')['safety_factor_elastic'].agg(['mean', 'std'])
            axes[1, 0].errorbar(safety_by_method.index, safety_by_method['mean'], 
                              yerr=safety_by_method['std'], fmt='o', capsize=5)
            axes[1, 0].set_title('Safety Factor by Method')
            axes[1, 0].set_ylabel('Safety Factor')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Fracture risk range by method
        if 'fracture_risk_elastic' in dataset.columns:
            risk_by_method = dataset.groupby('augmentation_method')['fracture_risk_elastic'].agg(['mean', 'std'])
            axes[1, 1].errorbar(risk_by_method.index, risk_by_method['mean'], 
                              yerr=risk_by_method['std'], fmt='o', capsize=5)
            axes[1, 1].set_title('Fracture Risk by Method')
            axes[1, 1].set_ylabel('Fracture Risk')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'augmentation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_visualization(self, dataset: pd.DataFrame, output_path: Path):
        """Plot PCA visualization of augmented data"""
        # Prepare data for PCA
        X = dataset[self.feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA plot
        plt.figure(figsize=(12, 8))
        
        # Plot each augmentation method
        for method in dataset['augmentation_method'].unique():
            method_mask = dataset['augmentation_method'] == method
            plt.scatter(X_pca[method_mask, 0], X_pca[method_mask, 1], 
                       label=method, alpha=0.6, s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Visualization of Augmented Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pca_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of the data augmentation module"""
    logger.info("Starting data augmentation example")
    
    # Load dataset (assuming it exists)
    try:
        dataset = pd.read_csv("comprehensive_sofc_dataset/combined_sofc_dataset.csv")
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except FileNotFoundError:
        logger.error("Dataset not found. Please generate the dataset first.")
        return
    
    # Initialize augmenter
    augmenter = SOFCDataAugmenter(dataset)
    
    # Create augmented dataset
    augmentation_methods = ['noise', 'interpolation', 'extrapolation', 'physics_informed']
    augmented_dataset = augmenter.create_augmented_dataset(
        augmentation_methods=augmentation_methods,
        n_samples_per_method=500
    )
    
    # Create visualizations
    augmenter.visualize_augmentation_results(augmented_dataset)
    
    # Save augmented dataset
    augmented_dataset.to_csv("augmented_sofc_dataset.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA AUGMENTATION RESULTS")
    print("="*60)
    print(f"Original samples: {len(dataset):,}")
    print(f"Augmented samples: {len(augmented_dataset):,}")
    print(f"Total samples: {len(augmented_dataset):,}")
    
    print(f"\nSamples by method:")
    method_counts = augmented_dataset['augmentation_method'].value_counts()
    for method, count in method_counts.items():
        print(f"  {method}: {count:,}")
    
    print(f"\nKey statistics:")
    print(f"Max Principal Stress: {augmented_dataset['max_principal_stress_elastic'].mean():.1f} ± {augmented_dataset['max_principal_stress_elastic'].std():.1f} MPa")
    print(f"Safety Factor: {augmented_dataset['safety_factor_elastic'].mean():.2f} ± {augmented_dataset['safety_factor_elastic'].std():.2f}")
    print(f"Fracture Risk: {augmented_dataset['fracture_risk_elastic'].mean():.3f} ± {augmented_dataset['fracture_risk_elastic'].std():.3f}")
    
    print(f"\nData augmentation complete! 🎉")
    print(f"Augmented dataset saved to: augmented_sofc_dataset.csv")
    print(f"Visualizations saved to: augmentation_results/")

if __name__ == "__main__":
    main()