"""
Design of Experiments (DoE) Sampling Methods for SOFC Parameters
"""

import numpy as np
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import List, Dict, Tuple, Optional
from parameters import SOFCParameters

class SOFCSampler:
    """Generate parameter samples using various DoE methods"""
    
    def __init__(self, parameters: SOFCParameters):
        self.parameters = parameters
        
    def latin_hypercube_sampling(self, 
                                n_samples: int, 
                                fidelity: str = 'HF',
                                seed: int = 42) -> pd.DataFrame:
        """
        Generate samples using Latin Hypercube Sampling
        
        Args:
            n_samples: Number of samples to generate
            fidelity: Fidelity level ('LF', 'MF', 'HF')
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with sampled parameters
        """
        # Get parameter names and bounds
        param_names = self.parameters.get_parameter_names(fidelity)
        lower_bounds, upper_bounds = self.parameters.get_parameter_bounds(fidelity)
        
        # Create Latin Hypercube sampler
        n_params = len(param_names)
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        
        # Generate samples in [0, 1]
        samples_unit = sampler.random(n=n_samples)
        
        # Scale to actual parameter ranges
        samples = qmc.scale(samples_unit, lower_bounds, upper_bounds)
        
        # Create DataFrame
        df = pd.DataFrame(samples, columns=param_names)
        df['fidelity'] = fidelity
        df['sample_id'] = np.arange(n_samples)
        
        return df
    
    def full_factorial_design(self, 
                             levels_per_param: int = 3,
                             fidelity: str = 'HF',
                             selected_params: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate full factorial design
        
        Args:
            levels_per_param: Number of levels for each parameter
            fidelity: Fidelity level
            selected_params: List of specific parameters to vary (None = all)
        
        Returns:
            DataFrame with factorial design
        """
        if selected_params is None:
            param_names = self.parameters.get_parameter_names(fidelity)
        else:
            param_names = selected_params
        
        # Create levels for each parameter
        param_levels = {}
        for param_name in param_names:
            # Find parameter bounds
            for category, params in self.parameters.parameters.items():
                for p_name, param in params.items():
                    if f"{category}.{p_name}" == param_name and fidelity in param.fidelity:
                        param_levels[param_name] = np.linspace(
                            param.min_val, 
                            param.max_val, 
                            levels_per_param
                        )
                        break
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_levels.values()))
        
        # Create DataFrame
        df = pd.DataFrame(combinations, columns=param_names)
        df['fidelity'] = fidelity
        df['sample_id'] = np.arange(len(df))
        
        return df
    
    def sobol_sequence_sampling(self,
                               n_samples: int,
                               fidelity: str = 'HF',
                               seed: int = 42) -> pd.DataFrame:
        """
        Generate samples using Sobol sequence (quasi-random)
        
        Args:
            n_samples: Number of samples
            fidelity: Fidelity level
            seed: Random seed
        
        Returns:
            DataFrame with Sobol sequence samples
        """
        # Get parameter names and bounds
        param_names = self.parameters.get_parameter_names(fidelity)
        lower_bounds, upper_bounds = self.parameters.get_parameter_bounds(fidelity)
        
        # Create Sobol sampler
        n_params = len(param_names)
        sampler = qmc.Sobol(d=n_params, seed=seed)
        
        # Generate samples
        samples_unit = sampler.random(n=n_samples)
        
        # Scale to actual parameter ranges
        samples = qmc.scale(samples_unit, lower_bounds, upper_bounds)
        
        # Create DataFrame
        df = pd.DataFrame(samples, columns=param_names)
        df['fidelity'] = fidelity
        df['sample_id'] = np.arange(n_samples)
        
        return df
    
    def orthogonal_array_sampling(self,
                                 n_samples: int,
                                 fidelity: str = 'HF',
                                 strength: int = 2) -> pd.DataFrame:
        """
        Generate orthogonal array design for parameter screening
        
        Args:
            n_samples: Approximate number of samples
            fidelity: Fidelity level
            strength: Orthogonal array strength
        
        Returns:
            DataFrame with orthogonal array samples
        """
        from pyDOE import lhs
        
        # Get parameter names and bounds
        param_names = self.parameters.get_parameter_names(fidelity)
        lower_bounds, upper_bounds = self.parameters.get_parameter_bounds(fidelity)
        
        n_params = len(param_names)
        
        # Generate orthogonal Latin hypercube
        samples_unit = lhs(n_params, samples=n_samples, criterion='maximin')
        
        # Scale to actual parameter ranges
        samples = np.zeros_like(samples_unit)
        for i in range(n_params):
            samples[:, i] = lower_bounds[i] + samples_unit[:, i] * (upper_bounds[i] - lower_bounds[i])
        
        # Create DataFrame
        df = pd.DataFrame(samples, columns=param_names)
        df['fidelity'] = fidelity
        df['sample_id'] = np.arange(len(df))
        
        return df
    
    def multi_fidelity_sampling(self,
                               n_lf: int = 10000,
                               n_mf: int = 1000,
                               n_hf: int = 100,
                               method: str = 'lhs',
                               seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Generate multi-fidelity dataset with nested sampling
        
        Args:
            n_lf: Number of low-fidelity samples
            n_mf: Number of medium-fidelity samples
            n_hf: Number of high-fidelity samples
            method: Sampling method ('lhs', 'sobol', 'orthogonal')
            seed: Random seed
        
        Returns:
            Dictionary with DataFrames for each fidelity level
        """
        datasets = {}
        
        # Generate high-fidelity samples first
        if method == 'lhs':
            datasets['HF'] = self.latin_hypercube_sampling(n_hf, 'HF', seed)
        elif method == 'sobol':
            datasets['HF'] = self.sobol_sequence_sampling(n_hf, 'HF', seed)
        else:
            datasets['HF'] = self.orthogonal_array_sampling(n_hf, 'HF')
        
        # Generate medium-fidelity samples (including HF points)
        if method == 'lhs':
            mf_additional = self.latin_hypercube_sampling(n_mf - n_hf, 'MF', seed + 1)
        elif method == 'sobol':
            mf_additional = self.sobol_sequence_sampling(n_mf - n_hf, 'MF', seed + 1)
        else:
            mf_additional = self.orthogonal_array_sampling(n_mf - n_hf, 'MF')
        
        # Project HF samples to MF space
        hf_to_mf = self._project_samples(datasets['HF'], 'HF', 'MF')
        datasets['MF'] = pd.concat([hf_to_mf, mf_additional], ignore_index=True)
        
        # Generate low-fidelity samples (including MF points)
        if method == 'lhs':
            lf_additional = self.latin_hypercube_sampling(n_lf - n_mf, 'LF', seed + 2)
        elif method == 'sobol':
            lf_additional = self.sobol_sequence_sampling(n_lf - n_mf, 'LF', seed + 2)
        else:
            lf_additional = self.orthogonal_array_sampling(n_lf - n_mf, 'LF')
        
        # Project MF samples to LF space
        mf_to_lf = self._project_samples(datasets['MF'], 'MF', 'LF')
        datasets['LF'] = pd.concat([mf_to_lf, lf_additional], ignore_index=True)
        
        return datasets
    
    def _project_samples(self, 
                        df: pd.DataFrame, 
                        from_fidelity: str, 
                        to_fidelity: str) -> pd.DataFrame:
        """
        Project samples from higher to lower fidelity
        
        Args:
            df: DataFrame with samples
            from_fidelity: Source fidelity level
            to_fidelity: Target fidelity level
        
        Returns:
            Projected DataFrame
        """
        # Get parameter names for target fidelity
        target_params = self.parameters.get_parameter_names(to_fidelity)
        
        # Select only parameters that exist in target fidelity
        available_params = [p for p in target_params if p in df.columns]
        
        # Create new DataFrame with projected samples
        projected_df = df[available_params + ['sample_id']].copy()
        
        # Add missing parameters with nominal values
        for param in target_params:
            if param not in projected_df.columns:
                # Find nominal value
                for category, params in self.parameters.parameters.items():
                    for p_name, param_obj in params.items():
                        if f"{category}.{p_name}" == param:
                            projected_df[param] = param_obj.nominal
                            break
        
        projected_df['fidelity'] = to_fidelity
        projected_df['parent_fidelity'] = from_fidelity
        projected_df['parent_sample_id'] = df['sample_id'].values
        
        return projected_df
    
    def add_noise_to_samples(self, 
                            df: pd.DataFrame, 
                            noise_level: float = 0.05,
                            seed: int = 42) -> pd.DataFrame:
        """
        Add realistic measurement noise to samples
        
        Args:
            df: DataFrame with samples
            noise_level: Relative noise level (0.05 = 5%)
            seed: Random seed
        
        Returns:
            DataFrame with noisy samples
        """
        np.random.seed(seed)
        df_noisy = df.copy()
        
        # Get numeric columns (exclude metadata)
        numeric_cols = [col for col in df.columns 
                       if col not in ['fidelity', 'sample_id', 'parent_fidelity', 'parent_sample_id']]
        
        for col in numeric_cols:
            # Add Gaussian noise proportional to parameter range
            param_range = df[col].max() - df[col].min()
            noise = np.random.normal(0, noise_level * param_range, len(df))
            df_noisy[col] = df[col] + noise
            
            # Ensure values stay within bounds
            for category, params in self.parameters.parameters.items():
                for p_name, param in params.items():
                    if f"{category}.{p_name}" == col:
                        df_noisy[col] = np.clip(df_noisy[col], param.min_val, param.max_val)
                        break
        
        return df_noisy
    
    def generate_correlated_parameters(self,
                                      n_samples: int,
                                      correlation_matrix: Optional[np.ndarray] = None,
                                      fidelity: str = 'HF',
                                      seed: int = 42) -> pd.DataFrame:
        """
        Generate samples with correlated parameters
        
        Args:
            n_samples: Number of samples
            correlation_matrix: Correlation matrix (None = use physical correlations)
            fidelity: Fidelity level
            seed: Random seed
        
        Returns:
            DataFrame with correlated samples
        """
        np.random.seed(seed)
        
        # Get parameter names and bounds
        param_names = self.parameters.get_parameter_names(fidelity)
        lower_bounds, upper_bounds = self.parameters.get_parameter_bounds(fidelity)
        n_params = len(param_names)
        
        # Define physical correlations if not provided
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_params)
            
            # Add some realistic correlations
            # Temperature affects conductivities
            temp_idx = None
            ionic_indices = []
            for i, name in enumerate(param_names):
                if 'temperature' in name:
                    temp_idx = i
                if 'ionic_conductivity' in name:
                    ionic_indices.append(i)
            
            if temp_idx is not None:
                for idx in ionic_indices:
                    correlation_matrix[temp_idx, idx] = 0.7
                    correlation_matrix[idx, temp_idx] = 0.7
            
            # Porosity affects tortuosity
            porosity_indices = []
            tortuosity_indices = []
            for i, name in enumerate(param_names):
                if 'porosity' in name:
                    porosity_indices.append(i)
                if 'tortuosity' in name:
                    tortuosity_indices.append(i)
            
            for p_idx in porosity_indices:
                for t_idx in tortuosity_indices:
                    if abs(p_idx - t_idx) < 5:  # Same component
                        correlation_matrix[p_idx, t_idx] = -0.6
                        correlation_matrix[t_idx, p_idx] = -0.6
        
        # Generate correlated normal samples
        mean = np.zeros(n_params)
        samples_normal = np.random.multivariate_normal(mean, correlation_matrix, n_samples)
        
        # Transform to uniform [0, 1] using CDF
        from scipy.stats import norm
        samples_uniform = norm.cdf(samples_normal)
        
        # Scale to actual parameter ranges
        samples = np.zeros_like(samples_uniform)
        for i in range(n_params):
            samples[:, i] = lower_bounds[i] + samples_uniform[:, i] * (upper_bounds[i] - lower_bounds[i])
        
        # Create DataFrame
        df = pd.DataFrame(samples, columns=param_names)
        df['fidelity'] = fidelity
        df['sample_id'] = np.arange(n_samples)
        
        return df