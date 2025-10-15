"""
Sampling Strategies for Design of Experiments

Implements various sampling methods for generating parameter combinations
in the SOFC manufacturing parameter space.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from scipy.stats import qmc
from .sofc_parameters import SOFCParameters, ParameterRange


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies"""
    
    @abstractmethod
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate samples from parameter ranges"""
        pass


class RandomSampling(SamplingStrategy):
    """Random uniform sampling"""
    
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate random samples"""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_params = len(parameter_ranges)
        samples = np.zeros((n_samples, n_params))
        
        for i, param_range in enumerate(parameter_ranges):
            samples[:, i] = param_range.sample(n_samples, random_state)
        
        return samples


class LatinHypercubeSampling(SamplingStrategy):
    """Latin Hypercube Sampling (LHS) for better space filling"""
    
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate LHS samples"""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_params = len(parameter_ranges)
        
        # Generate LHS samples in [0, 1]^n_params
        lhs_sampler = qmc.LatinHypercube(d=n_params, seed=random_state)
        lhs_samples = lhs_sampler.random(n=n_samples)
        
        # Transform to actual parameter ranges
        samples = np.zeros((n_samples, n_params))
        for i, param_range in enumerate(parameter_ranges):
            # Map [0, 1] to [min_value, max_value]
            samples[:, i] = (param_range.min_value + 
                           (param_range.max_value - param_range.min_value) * lhs_samples[:, i])
        
        return samples


class SobolSampling(SamplingStrategy):
    """Sobol quasi-random sampling for low-discrepancy sequences"""
    
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate Sobol samples"""
        n_params = len(parameter_ranges)
        
        # Generate Sobol samples in [0, 1]^n_params
        sobol_sampler = qmc.Sobol(d=n_params, seed=random_state)
        sobol_samples = sobol_sampler.random(n=n_samples)
        
        # Transform to actual parameter ranges
        samples = np.zeros((n_samples, n_params))
        for i, param_range in enumerate(parameter_ranges):
            # Map [0, 1] to [min_value, max_value]
            samples[:, i] = (param_range.min_value + 
                           (param_range.max_value - param_range.min_value) * sobol_samples[:, i])
        
        return samples


class HaltonSampling(SamplingStrategy):
    """Halton quasi-random sampling"""
    
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate Halton samples"""
        n_params = len(parameter_ranges)
        
        # Generate Halton samples in [0, 1]^n_params
        halton_sampler = qmc.Halton(d=n_params, seed=random_state)
        halton_samples = halton_sampler.random(n=n_samples)
        
        # Transform to actual parameter ranges
        samples = np.zeros((n_samples, n_params))
        for i, param_range in enumerate(parameter_ranges):
            # Map [0, 1] to [min_value, max_value]
            samples[:, i] = (param_range.min_value + 
                           (param_range.max_value - param_range.min_value) * halton_samples[:, i])
        
        return samples


class AdaptiveSampling(SamplingStrategy):
    """
    Adaptive sampling that focuses on regions of high parameter sensitivity
    """
    
    def __init__(self, base_strategy: SamplingStrategy = LatinHypercubeSampling()):
        self.base_strategy = base_strategy
    
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate adaptive samples"""
        # For now, just use the base strategy
        # In a full implementation, this would analyze previous results
        # and focus sampling on high-sensitivity regions
        return self.base_strategy.sample(parameter_ranges, n_samples, random_state)


class StratifiedSampling(SamplingStrategy):
    """
    Stratified sampling that ensures representation across parameter ranges
    """
    
    def sample(self, parameter_ranges: List[ParameterRange], n_samples: int, 
               random_state: Optional[int] = None) -> np.ndarray:
        """Generate stratified samples"""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_params = len(parameter_ranges)
        samples = np.zeros((n_samples, n_params))
        
        # Calculate number of strata per parameter
        n_strata = int(np.ceil(n_samples ** (1/n_params)))
        samples_per_stratum = n_samples // (n_strata ** n_params)
        
        for i, param_range in enumerate(parameter_ranges):
            # Create strata
            stratum_edges = np.linspace(param_range.min_value, param_range.max_value, n_strata + 1)
            
            # Sample within each stratum
            param_samples = []
            for j in range(n_strata):
                stratum_min = stratum_edges[j]
                stratum_max = stratum_edges[j + 1]
                stratum_samples = np.random.uniform(stratum_min, stratum_max, samples_per_stratum)
                param_samples.extend(stratum_samples)
            
            # Fill remaining samples randomly
            remaining = n_samples - len(param_samples)
            if remaining > 0:
                additional_samples = np.random.uniform(
                    param_range.min_value, param_range.max_value, remaining)
                param_samples.extend(additional_samples)
            
            samples[:, i] = param_samples[:n_samples]
        
        return samples


def create_sampling_strategy(strategy_name: str) -> SamplingStrategy:
    """Factory function to create sampling strategies"""
    strategies = {
        'random': RandomSampling(),
        'lhs': LatinHypercubeSampling(),
        'sobol': SobolSampling(),
        'halton': HaltonSampling(),
        'adaptive': AdaptiveSampling(),
        'stratified': StratifiedSampling()
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}. "
                        f"Available strategies: {list(strategies.keys())}")
    
    return strategies[strategy_name]


if __name__ == "__main__":
    # Example usage
    from .sofc_parameters import SOFCParameters
    
    sofc_params = SOFCParameters()
    parameter_ranges = sofc_params.get_all_parameters()
    
    n_samples = 100
    
    print("Sampling Strategy Comparison:")
    print("=" * 50)
    
    strategies = ['random', 'lhs', 'sobol', 'halton', 'stratified']
    
    for strategy_name in strategies:
        strategy = create_sampling_strategy(strategy_name)
        samples = strategy.sample(parameter_ranges, n_samples, random_state=42)
        
        print(f"\n{strategy_name.upper()} Sampling:")
        print(f"  Shape: {samples.shape}")
        print(f"  Mean: {np.mean(samples, axis=0)[:5]}...")  # First 5 parameters
        print(f"  Std:  {np.std(samples, axis=0)[:5]}...")   # First 5 parameters