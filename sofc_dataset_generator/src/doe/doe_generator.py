"""
Design of Experiments Generator for SOFC Dataset

Generates parameter combinations using various sampling strategies for creating
the synthetic SOFC dataset through FEA simulations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json
import os

from .sofc_parameters import SOFCParameters, ManufacturingParameters
from .sampling_strategies import SamplingStrategy, create_sampling_strategy


@dataclass
class ParameterSpace:
    """Container for parameter space definition"""
    parameter_ranges: SOFCParameters
    sampling_strategy: SamplingStrategy
    n_samples: int
    random_state: Optional[int] = None
    
    def generate_samples(self) -> np.ndarray:
        """Generate parameter samples"""
        param_list = self.parameter_ranges.get_all_parameters()
        return self.sampling_strategy.sample(param_list, self.n_samples, self.random_state)
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names in order"""
        return self.parameter_ranges.get_parameter_names()


class DOEGenerator:
    """Main DOE generator for SOFC parameter combinations"""
    
    def __init__(self, parameter_ranges: Optional[SOFCParameters] = None):
        self.parameter_ranges = parameter_ranges or SOFCParameters()
        self.samples = None
        self.sample_metadata = None
    
    def generate_doe(self, n_samples: int, 
                    strategy: str = 'lhs',
                    random_state: Optional[int] = None,
                    save_to_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate Design of Experiments matrix
        
        Args:
            n_samples: Number of parameter combinations to generate
            strategy: Sampling strategy ('random', 'lhs', 'sobol', 'halton', 'stratified')
            random_state: Random seed for reproducibility
            save_to_file: Optional file path to save DOE matrix
            
        Returns:
            DataFrame with parameter combinations
        """
        # Create sampling strategy
        sampling_strategy = create_sampling_strategy(strategy)
        
        # Create parameter space
        param_space = ParameterSpace(
            parameter_ranges=self.parameter_ranges,
            sampling_strategy=sampling_strategy,
            n_samples=n_samples,
            random_state=random_state
        )
        
        # Generate samples
        self.samples = param_space.generate_samples()
        param_names = param_space.get_parameter_names()
        
        # Create DataFrame
        doe_df = pd.DataFrame(self.samples, columns=param_names)
        
        # Add metadata
        self.sample_metadata = {
            'n_samples': n_samples,
            'strategy': strategy,
            'random_state': random_state,
            'parameter_ranges': {
                name: {
                    'min': getattr(self.parameter_ranges, name).min_value,
                    'max': getattr(self.parameter_ranges, name).max_value,
                    'unit': getattr(self.parameter_ranges, name).unit,
                    'distribution': getattr(self.parameter_ranges, name).distribution
                }
                for name in param_names
            }
        }
        
        # Save to file if requested
        if save_to_file:
            self.save_doe(doe_df, save_to_file)
        
        return doe_df
    
    def generate_manufacturing_parameters(self, doe_df: pd.DataFrame) -> List[ManufacturingParameters]:
        """Convert DOE DataFrame to ManufacturingParameters objects"""
        manufacturing_params = []
        
        for _, row in doe_df.iterrows():
            param_dict = row.to_dict()
            mfg_params = ManufacturingParameters(
                values=param_dict,
                parameter_ranges=self.parameter_ranges
            )
            manufacturing_params.append(mfg_params)
        
        return manufacturing_params
    
    def save_doe(self, doe_df: pd.DataFrame, filepath: str):
        """Save DOE matrix and metadata to files"""
        # Save DataFrame
        doe_df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata_file = filepath.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.sample_metadata, f, indent=2)
        
        print(f"DOE matrix saved to {filepath}")
        print(f"Metadata saved to {metadata_file}")
    
    def load_doe(self, filepath: str) -> pd.DataFrame:
        """Load DOE matrix from file"""
        doe_df = pd.read_csv(filepath)
        
        # Load metadata if available
        metadata_file = filepath.replace('.csv', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.sample_metadata = json.load(f)
        
        self.samples = doe_df.values
        return doe_df
    
    def analyze_doe(self, doe_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DOE matrix for coverage and distribution"""
        analysis = {}
        
        # Basic statistics
        analysis['n_samples'] = len(doe_df)
        analysis['n_parameters'] = len(doe_df.columns)
        
        # Parameter statistics
        analysis['parameter_stats'] = {}
        for col in doe_df.columns:
            analysis['parameter_stats'][col] = {
                'mean': float(doe_df[col].mean()),
                'std': float(doe_df[col].std()),
                'min': float(doe_df[col].min()),
                'max': float(doe_df[col].max()),
                'range': float(doe_df[col].max() - doe_df[col].min())
            }
        
        # Coverage analysis (for LHS and quasi-random methods)
        if self.sample_metadata and self.sample_metadata['strategy'] in ['lhs', 'sobol', 'halton']:
            analysis['coverage'] = self._analyze_coverage(doe_df)
        
        # Correlation analysis
        analysis['correlations'] = doe_df.corr().to_dict()
        
        return analysis
    
    def _analyze_coverage(self, doe_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze parameter space coverage"""
        coverage = {}
        
        for col in doe_df.columns:
            # Normalize to [0, 1]
            param_range = getattr(self.parameter_ranges, col)
            normalized = (doe_df[col] - param_range.min_value) / (param_range.max_value - param_range.min_value)
            
            # Calculate coverage metrics
            coverage[col] = {
                'min_gap': float(np.min(np.diff(np.sort(normalized)))),
                'max_gap': float(np.max(np.diff(np.sort(normalized)))),
                'mean_gap': float(np.mean(np.diff(np.sort(normalized)))),
                'coverage_ratio': float(len(np.unique(np.round(normalized, 3))) / len(normalized))
            }
        
        return coverage
    
    def visualize_doe(self, doe_df: pd.DataFrame, save_path: Optional[str] = None):
        """Create visualization of DOE matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plot
        n_params = len(doe_df.columns)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot parameter distributions
        for i, col in enumerate(doe_df.columns):
            row = i // n_cols
            col_idx = i % n_cols
            
            ax = axes[row, col_idx]
            ax.hist(doe_df[col], bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Add parameter range
            param_range = getattr(self.parameter_ranges, col)
            ax.axvline(param_range.min_value, color='red', linestyle='--', alpha=0.7, label='Min')
            ax.axvline(param_range.max_value, color='red', linestyle='--', alpha=0.7, label='Max')
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DOE visualization saved to {save_path}")
        
        plt.show()
    
    def generate_parameter_combinations(self, n_samples: int, 
                                      strategy: str = 'lhs',
                                      random_state: Optional[int] = None) -> List[ManufacturingParameters]:
        """Convenience method to generate parameter combinations directly"""
        doe_df = self.generate_doe(n_samples, strategy, random_state)
        return self.generate_manufacturing_parameters(doe_df)


if __name__ == "__main__":
    # Example usage
    doe_generator = DOEGenerator()
    
    # Generate DOE matrix
    print("Generating DOE matrix...")
    doe_df = doe_generator.generate_doe(
        n_samples=100,
        strategy='lhs',
        random_state=42,
        save_to_file='sofc_doe_matrix.csv'
    )
    
    print(f"Generated DOE matrix with shape: {doe_df.shape}")
    print(f"Parameter names: {list(doe_df.columns)}")
    
    # Analyze DOE
    print("\nAnalyzing DOE...")
    analysis = doe_generator.analyze_doe(doe_df)
    print(f"Number of samples: {analysis['n_samples']}")
    print(f"Number of parameters: {analysis['n_parameters']}")
    
    # Generate manufacturing parameters
    print("\nGenerating manufacturing parameters...")
    mfg_params = doe_generator.generate_parameter_combinations(10, strategy='lhs', random_state=42)
    print(f"Generated {len(mfg_params)} manufacturing parameter sets")
    
    # Show first parameter set
    print("\nFirst parameter set:")
    first_params = mfg_params[0]
    for name, value in first_params.get_geometric_values().items():
        print(f"  {name}: {value:.3f}")