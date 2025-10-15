"""
Design of Experiments (DOE) Generator for SOFC Manufacturing

This module generates DOE matrices using various sampling strategies
to create comprehensive parameter combinations for FEA simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml
try:
    from pyDOE2 import lhs
    PYDOE_AVAILABLE = True
except ImportError:
    PYDOE_AVAILABLE = False
    print("Warning: pyDOE2 not available, using fallback LHS implementation")
from scipy.stats import qmc
import itertools
from dataclasses import dataclass

from .parameter_space import SOFCParameterSpace, Parameter, ParameterType


@dataclass
class DOEConfiguration:
    """Configuration for DOE generation."""
    n_samples: int = 1000
    sampling_method: str = "latin_hypercube"  # latin_hypercube, sobol, random, full_factorial
    seed: int = 42
    stratification: bool = True
    optimization: str = "maximin"  # For LHS: None, "center", "maximin", "centermaximin", "correlation"


class DOEGenerator:
    """Generates Design of Experiments matrices for SOFC manufacturing parameters."""
    
    def __init__(self, parameter_space: SOFCParameterSpace, config: Optional[DOEConfiguration] = None):
        """Initialize DOE generator with parameter space and configuration."""
        self.parameter_space = parameter_space
        self.config = config or DOEConfiguration()
        self.random_state = np.random.RandomState(self.config.seed)
        
        # Separate continuous and categorical parameters
        self.continuous_params = []
        self.categorical_params = []
        self.log_uniform_params = []
        
        for param in self.parameter_space.parameters.values():
            if param.param_type == ParameterType.CONTINUOUS:
                self.continuous_params.append(param)
            elif param.param_type == ParameterType.LOG_UNIFORM:
                self.log_uniform_params.append(param)
            elif param.param_type == ParameterType.CATEGORICAL:
                self.categorical_params.append(param)
    
    def generate_doe_matrix(self) -> pd.DataFrame:
        """Generate complete DOE matrix combining all parameter types."""
        
        if self.config.sampling_method == "latin_hypercube":
            return self._generate_lhs_matrix()
        elif self.config.sampling_method == "sobol":
            return self._generate_sobol_matrix()
        elif self.config.sampling_method == "random":
            return self._generate_random_matrix()
        elif self.config.sampling_method == "full_factorial":
            return self._generate_factorial_matrix()
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
    
    def _generate_lhs_matrix(self) -> pd.DataFrame:
        """Generate Latin Hypercube Sampling matrix."""
        n_continuous = len(self.continuous_params) + len(self.log_uniform_params)
        
        if n_continuous > 0:
            # Generate LHS for continuous parameters
            if PYDOE_AVAILABLE:
                lhs_samples = lhs(n_continuous, samples=self.config.n_samples, 
                                criterion=self.config.optimization, 
                                random_state=self.config.seed)
            else:
                # Fallback to random sampling
                lhs_samples = self.random_state.rand(self.config.n_samples, n_continuous)
            
            # Transform to actual parameter ranges
            continuous_data = {}
            param_idx = 0
            
            # Handle continuous parameters
            for param in self.continuous_params:
                samples = lhs_samples[:, param_idx]
                scaled_samples = param.min_val + samples * (param.max_val - param.min_val)
                continuous_data[param.name] = scaled_samples
                param_idx += 1
            
            # Handle log-uniform parameters
            for param in self.log_uniform_params:
                samples = lhs_samples[:, param_idx]
                log_min = np.log10(param.min_val)
                log_max = np.log10(param.max_val)
                log_samples = log_min + samples * (log_max - log_min)
                continuous_data[param.name] = 10 ** log_samples
                param_idx += 1
        else:
            continuous_data = {}
        
        # Handle categorical parameters
        categorical_data = {}
        for param in self.categorical_params:
            samples = self.random_state.choice(param.values, self.config.n_samples)
            categorical_data[param.name] = samples
        
        # Combine all data
        all_data = {**continuous_data, **categorical_data}
        return pd.DataFrame(all_data)
    
    def _generate_sobol_matrix(self) -> pd.DataFrame:
        """Generate Sobol sequence matrix."""
        n_continuous = len(self.continuous_params) + len(self.log_uniform_params)
        
        if n_continuous > 0:
            # Generate Sobol sequence
            sampler = qmc.Sobol(d=n_continuous, scramble=True, seed=self.config.seed)
            sobol_samples = sampler.random(n=self.config.n_samples)
            
            # Transform to actual parameter ranges
            continuous_data = {}
            param_idx = 0
            
            # Handle continuous parameters
            for param in self.continuous_params:
                samples = sobol_samples[:, param_idx]
                scaled_samples = param.min_val + samples * (param.max_val - param.min_val)
                continuous_data[param.name] = scaled_samples
                param_idx += 1
            
            # Handle log-uniform parameters
            for param in self.log_uniform_params:
                samples = sobol_samples[:, param_idx]
                log_min = np.log10(param.min_val)
                log_max = np.log10(param.max_val)
                log_samples = log_min + samples * (log_max - log_min)
                continuous_data[param.name] = 10 ** log_samples
                param_idx += 1
        else:
            continuous_data = {}
        
        # Handle categorical parameters
        categorical_data = {}
        for param in self.categorical_params:
            samples = self.random_state.choice(param.values, self.config.n_samples)
            categorical_data[param.name] = samples
        
        # Combine all data
        all_data = {**continuous_data, **categorical_data}
        return pd.DataFrame(all_data)
    
    def _generate_random_matrix(self) -> pd.DataFrame:
        """Generate random sampling matrix."""
        all_data = {}
        
        # Handle continuous parameters
        for param in self.continuous_params:
            samples = self.random_state.uniform(param.min_val, param.max_val, self.config.n_samples)
            all_data[param.name] = samples
        
        # Handle log-uniform parameters
        for param in self.log_uniform_params:
            log_min = np.log10(param.min_val)
            log_max = np.log10(param.max_val)
            log_samples = self.random_state.uniform(log_min, log_max, self.config.n_samples)
            all_data[param.name] = 10 ** log_samples
        
        # Handle categorical parameters
        for param in self.categorical_params:
            samples = self.random_state.choice(param.values, self.config.n_samples)
            all_data[param.name] = samples
        
        return pd.DataFrame(all_data)
    
    def _generate_factorial_matrix(self) -> pd.DataFrame:
        """Generate full factorial design (warning: can be very large!)."""
        # For continuous parameters, use discrete levels
        n_levels = int(np.ceil(self.config.n_samples ** (1.0 / len(self.continuous_params))))
        
        all_combinations = []
        param_names = []
        
        # Handle continuous parameters with discretization
        for param in self.continuous_params:
            levels = np.linspace(param.min_val, param.max_val, n_levels)
            all_combinations.append(levels)
            param_names.append(param.name)
        
        # Handle log-uniform parameters with discretization
        for param in self.log_uniform_params:
            log_levels = np.linspace(np.log10(param.min_val), np.log10(param.max_val), n_levels)
            levels = 10 ** log_levels
            all_combinations.append(levels)
            param_names.append(param.name)
        
        # Handle categorical parameters
        for param in self.categorical_params:
            all_combinations.append(param.values)
            param_names.append(param.name)
        
        # Generate all combinations
        factorial_combinations = list(itertools.product(*all_combinations))
        
        # Limit to requested sample size if needed
        if len(factorial_combinations) > self.config.n_samples:
            indices = self.random_state.choice(len(factorial_combinations), 
                                             self.config.n_samples, replace=False)
            factorial_combinations = [factorial_combinations[i] for i in indices]
        
        # Create DataFrame
        data = {name: [combo[i] for combo in factorial_combinations] 
                for i, name in enumerate(param_names)}
        
        return pd.DataFrame(data)
    
    def validate_doe_matrix(self, doe_matrix: pd.DataFrame) -> Dict[str, bool]:
        """Validate the generated DOE matrix."""
        validation_results = {}
        
        # Check sample count
        validation_results['correct_sample_count'] = len(doe_matrix) == self.config.n_samples
        
        # Check parameter coverage
        expected_params = set(self.parameter_space.get_parameter_names())
        actual_params = set(doe_matrix.columns)
        validation_results['all_parameters_present'] = expected_params == actual_params
        
        # Check parameter ranges
        for param_name, param in self.parameter_space.parameters.items():
            if param.param_type in [ParameterType.CONTINUOUS, ParameterType.LOG_UNIFORM]:
                values = doe_matrix[param_name]
                in_range = np.all((values >= param.min_val) & (values <= param.max_val))
                validation_results[f'{param_name}_in_range'] = in_range
            elif param.param_type == ParameterType.CATEGORICAL:
                values = doe_matrix[param_name]
                valid_values = np.all(np.isin(values, param.values))
                validation_results[f'{param_name}_valid_values'] = valid_values
        
        return validation_results
    
    def generate_stratified_doe(self, strata_params: List[str], n_strata: int = 4) -> pd.DataFrame:
        """Generate stratified DOE matrix for better coverage of important parameters."""
        if not self.config.stratification:
            return self.generate_doe_matrix()
        
        # Create strata based on specified parameters
        strata_combinations = []
        for param_name in strata_params:
            param = self.parameter_space.get_parameter(param_name)
            if param.param_type == ParameterType.CONTINUOUS:
                strata = np.linspace(param.min_val, param.max_val, n_strata + 1)
                strata_combinations.append([(strata[i], strata[i+1]) for i in range(n_strata)])
            elif param.param_type == ParameterType.CATEGORICAL:
                # For categorical, each value is its own stratum
                strata_combinations.append([(val,) for val in param.values])
        
        # Generate samples for each stratum
        all_strata = list(itertools.product(*strata_combinations))
        samples_per_stratum = self.config.n_samples // len(all_strata)
        
        stratified_samples = []
        for stratum in all_strata:
            # Temporarily modify parameter ranges for this stratum
            temp_config = DOEConfiguration(
                n_samples=samples_per_stratum,
                sampling_method=self.config.sampling_method,
                seed=self.random_state.randint(0, 10000)
            )
            
            # Generate samples within this stratum
            stratum_generator = DOEGenerator(self.parameter_space, temp_config)
            stratum_samples = stratum_generator.generate_doe_matrix()
            
            # Constrain to stratum bounds
            for i, param_name in enumerate(strata_params):
                if len(stratum[i]) == 2:  # Continuous parameter
                    min_val, max_val = stratum[i]
                    stratum_samples[param_name] = np.clip(stratum_samples[param_name], min_val, max_val)
                else:  # Categorical parameter
                    stratum_samples[param_name] = stratum[i][0]
            
            stratified_samples.append(stratum_samples)
        
        # Combine all strata
        return pd.concat(stratified_samples, ignore_index=True)
    
    def export_doe_matrix(self, doe_matrix: pd.DataFrame, filepath: Union[str, Path]):
        """Export DOE matrix to various formats."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            doe_matrix.to_csv(filepath, index=False)
        elif filepath.suffix == '.xlsx':
            doe_matrix.to_excel(filepath, index=False)
        elif filepath.suffix == '.hdf5' or filepath.suffix == '.h5':
            doe_matrix.to_hdf(filepath, key='doe_matrix', mode='w')
        elif filepath.suffix == '.parquet':
            doe_matrix.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


def create_doe_generator(config_path: Optional[str] = None, 
                        doe_config: Optional[DOEConfiguration] = None) -> DOEGenerator:
    """Factory function to create DOE generator."""
    param_space = SOFCParameterSpace(config_path)
    return DOEGenerator(param_space, doe_config)


if __name__ == "__main__":
    # Test DOE generation
    config = DOEConfiguration(n_samples=100, sampling_method="latin_hypercube")
    generator = create_doe_generator(doe_config=config)
    
    print("Generating DOE matrix...")
    doe_matrix = generator.generate_doe_matrix()
    
    print(f"Generated DOE matrix with shape: {doe_matrix.shape}")
    print(f"Columns: {list(doe_matrix.columns)}")
    
    # Validate the matrix
    validation = generator.validate_doe_matrix(doe_matrix)
    print(f"\nValidation results: {validation}")
    
    # Show sample of the data
    print(f"\nFirst 5 rows:")
    print(doe_matrix.head())