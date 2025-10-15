"""
Simple DOE Implementation

A simplified Design of Experiments implementation that doesn't rely on 
external libraries like pyDOE2, which may have compatibility issues.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .parameter_space import SOFCParameterSpace, Parameter, ParameterType


@dataclass
class SimpleDOEConfiguration:
    """Configuration for simple DOE generation."""
    n_samples: int = 1000
    sampling_method: str = "latin_hypercube"  # latin_hypercube, sobol, random
    seed: int = 42


class SimpleDOEGenerator:
    """Simple DOE generator without external dependencies."""
    
    def __init__(self, parameter_space: SOFCParameterSpace, config: Optional[SimpleDOEConfiguration] = None):
        """Initialize simple DOE generator."""
        self.parameter_space = parameter_space
        self.config = config or SimpleDOEConfiguration()
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
        """Generate DOE matrix using simple methods."""
        
        if self.config.sampling_method == "latin_hypercube":
            return self._generate_simple_lhs()
        elif self.config.sampling_method == "sobol":
            return self._generate_simple_sobol()
        elif self.config.sampling_method == "random":
            return self._generate_random_matrix()
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
    
    def _generate_simple_lhs(self) -> pd.DataFrame:
        """Generate simple Latin Hypercube Sampling matrix."""
        n_continuous = len(self.continuous_params) + len(self.log_uniform_params)
        
        if n_continuous > 0:
            # Simple LHS implementation
            lhs_samples = self._simple_lhs(n_continuous, self.config.n_samples)
            
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
    
    def _simple_lhs(self, n_dim: int, n_samples: int) -> np.ndarray:
        """Simple Latin Hypercube Sampling implementation."""
        
        # Create LHS matrix
        lhs_matrix = np.zeros((n_samples, n_dim))
        
        for i in range(n_dim):
            # Create equally spaced intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            
            # Random permutation of interval indices
            perm = self.random_state.permutation(n_samples)
            
            # Sample within each interval
            for j in range(n_samples):
                interval_idx = perm[j]
                lower = intervals[interval_idx]
                upper = intervals[interval_idx + 1]
                lhs_matrix[j, i] = self.random_state.uniform(lower, upper)
        
        return lhs_matrix
    
    def _generate_simple_sobol(self) -> pd.DataFrame:
        """Generate simple Sobol-like sequence."""
        # This is a simplified version - not a true Sobol sequence
        # For a true Sobol sequence, you'd need more complex implementation
        
        n_continuous = len(self.continuous_params) + len(self.log_uniform_params)
        
        if n_continuous > 0:
            # Generate van der Corput-like sequence (simplified Sobol)
            sobol_samples = self._van_der_corput_sequence(n_continuous, self.config.n_samples)
            
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
    
    def _van_der_corput_sequence(self, n_dim: int, n_samples: int) -> np.ndarray:
        """Generate van der Corput sequence (simplified Sobol-like)."""
        
        # Use different prime bases for each dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        
        if n_dim > len(primes):
            # Extend with more primes if needed
            primes.extend(range(73, 73 + 2 * (n_dim - len(primes)), 2))
        
        sequence = np.zeros((n_samples, n_dim))
        
        for dim in range(n_dim):
            base = primes[dim]
            for i in range(n_samples):
                sequence[i, dim] = self._van_der_corput_1d(i + 1, base)
        
        return sequence
    
    def _van_der_corput_1d(self, n: int, base: int) -> float:
        """Generate single van der Corput number."""
        result = 0.0
        f = 1.0 / base
        
        while n > 0:
            result += f * (n % base)
            n //= base
            f /= base
        
        return result
    
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


def create_simple_doe_generator(parameter_space: Optional[SOFCParameterSpace] = None, 
                              doe_config: Optional[SimpleDOEConfiguration] = None) -> SimpleDOEGenerator:
    """Factory function to create simple DOE generator."""
    if parameter_space is None:
        from .parameter_space import create_sofc_parameter_space
        parameter_space = create_sofc_parameter_space()
    
    return SimpleDOEGenerator(parameter_space, doe_config)


if __name__ == "__main__":
    # Test simple DOE generation
    from .parameter_space import create_sofc_parameter_space
    
    param_space = create_sofc_parameter_space()
    config = SimpleDOEConfiguration(n_samples=10, sampling_method="latin_hypercube")
    generator = create_simple_doe_generator(param_space, config)
    
    print("Generating simple DOE matrix...")
    doe_matrix = generator.generate_doe_matrix()
    
    print(f"Generated DOE matrix with shape: {doe_matrix.shape}")
    print(f"Columns: {list(doe_matrix.columns)}")
    
    # Validate the matrix
    validation = generator.validate_doe_matrix(doe_matrix)
    print(f"\nValidation results: {validation}")
    
    # Show sample of the data
    print(f"\nFirst 3 rows:")
    print(doe_matrix.head(3))