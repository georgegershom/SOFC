"""
Parameter Space Definition for SOFC Manufacturing DOE

This module defines the parameter space for the Design of Experiments (DOE)
used to generate synthetic SOFC manufacturing scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path


class ParameterType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"


@dataclass
class Parameter:
    """Represents a single parameter in the DOE space."""
    name: str
    param_type: ParameterType
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    values: Optional[List[Any]] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate parameter definition."""
        if self.param_type in [ParameterType.CONTINUOUS, ParameterType.LOG_UNIFORM]:
            return self.min_val is not None and self.max_val is not None
        elif self.param_type == ParameterType.CATEGORICAL:
            return self.values is not None and len(self.values) > 0
        return True
    
    def sample(self, n_samples: int, random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample values from this parameter's distribution."""
        if random_state is None:
            random_state = np.random.RandomState()
            
        if self.param_type == ParameterType.CONTINUOUS:
            return random_state.uniform(self.min_val, self.max_val, n_samples)
        elif self.param_type == ParameterType.LOG_UNIFORM:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            log_samples = random_state.uniform(log_min, log_max, n_samples)
            return 10 ** log_samples
        elif self.param_type == ParameterType.CATEGORICAL:
            return random_state.choice(self.values, n_samples)
        else:
            raise NotImplementedError(f"Sampling not implemented for {self.param_type}")


class SOFCParameterSpace:
    """Defines the complete parameter space for SOFC manufacturing DOE."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize parameter space from configuration file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "sofc_parameters.yaml"
        
        self.config_path = Path(config_path)
        self.parameters = {}
        self._load_config()
    
    def _load_config(self):
        """Load parameter definitions from YAML configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Parse geometry parameters
        self._parse_parameter_group(config['geometry'], 'geometry')
        
        # Parse material properties
        for material, props in config['material_properties'].items():
            self._parse_parameter_group(props, f'material.{material}')
        
        # Parse thermal profile
        self._parse_parameter_group(config['thermal_profile'], 'thermal')
        
        # Parse manufacturing conditions
        self._parse_parameter_group(config['manufacturing_conditions'], 'manufacturing')
        
        # Parse defects (if enabled)
        if config['defects']['enable_defects']:
            defect_params = {k: v for k, v in config['defects'].items() if k != 'enable_defects'}
            self._parse_parameter_group(defect_params, 'defects')
    
    def _parse_parameter_group(self, group_config: Dict, prefix: str):
        """Parse a group of parameters from configuration."""
        for param_name, param_config in group_config.items():
            if isinstance(param_config, dict) and 'min' in param_config:
                # Continuous parameter
                full_name = f"{prefix}.{param_name}"
                param_type = ParameterType(param_config.get('type', 'continuous'))
                
                # Convert string numbers to float (handles scientific notation)
                min_val = float(param_config['min'])
                max_val = float(param_config['max'])
                
                self.parameters[full_name] = Parameter(
                    name=full_name,
                    param_type=param_type,
                    min_val=min_val,
                    max_val=max_val,
                    unit=param_config.get('unit'),
                    description=param_config.get('description')
                )
            elif isinstance(param_config, dict) and 'values' in param_config:
                # Categorical parameter
                full_name = f"{prefix}.{param_name}"
                self.parameters[full_name] = Parameter(
                    name=full_name,
                    param_type=ParameterType.CATEGORICAL,
                    values=param_config['values'],
                    description=param_config.get('description')
                )
            elif isinstance(param_config, dict):
                # Nested group
                self._parse_parameter_group(param_config, f"{prefix}.{param_name}")
    
    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return list(self.parameters.keys())
    
    def get_parameter(self, name: str) -> Parameter:
        """Get parameter by name."""
        return self.parameters[name]
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for continuous parameters."""
        continuous_params = [p for p in self.parameters.values() 
                           if p.param_type in [ParameterType.CONTINUOUS, ParameterType.LOG_UNIFORM]]
        
        lower_bounds = np.array([p.min_val for p in continuous_params])
        upper_bounds = np.array([p.max_val for p in continuous_params])
        
        return lower_bounds, upper_bounds
    
    def validate_parameters(self) -> bool:
        """Validate all parameter definitions."""
        return all(param.validate() for param in self.parameters.values())
    
    def summary(self) -> pd.DataFrame:
        """Generate a summary of all parameters."""
        data = []
        for param in self.parameters.values():
            row = {
                'name': param.name,
                'type': param.param_type.value,
                'min': param.min_val,
                'max': param.max_val,
                'values': param.values,
                'unit': param.unit,
                'description': param.description
            }
            data.append(row)
        
        return pd.DataFrame(data)


def create_sofc_parameter_space() -> SOFCParameterSpace:
    """Factory function to create SOFC parameter space."""
    return SOFCParameterSpace()


if __name__ == "__main__":
    # Test parameter space creation
    param_space = create_sofc_parameter_space()
    print(f"Created parameter space with {len(param_space.parameters)} parameters")
    print("\nParameter Summary:")
    print(param_space.summary())
    
    # Validate parameters
    if param_space.validate_parameters():
        print("\n✓ All parameters are valid")
    else:
        print("\n✗ Some parameters are invalid")