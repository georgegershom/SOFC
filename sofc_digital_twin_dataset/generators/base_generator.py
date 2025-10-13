"""
Base generator class for SOFC digital twin dataset generation.
"""

import os
import yaml
import h5py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class BaseGenerator(ABC):
    """
    Base class for all dataset generators.
    
    Provides common functionality for data generation, storage, and validation.
    """
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        """
        Initialize the generator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = "data"
        self._ensure_output_dirs()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/materials_geometry", exist_ok=True)
        os.makedirs(f"{self.output_dir}/operational_electrochemical", exist_ok=True)
        os.makedirs(f"{self.output_dir}/thermo_structural", exist_ok=True)
        os.makedirs(f"{self.output_dir}/degradation_failure", exist_ok=True)
        os.makedirs(f"{self.output_dir}/synthetic_sensors", exist_ok=True)
    
    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """
        Generate the dataset.
        
        Returns:
            Dictionary containing the generated data
        """
        pass
    
    def save_data(self, data: Dict[str, Any], filename: str, 
                  group: Optional[str] = None) -> str:
        """
        Save data to HDF5 file.
        
        Args:
            data: Dictionary containing the data to save
            filename: Name of the file to save to
            group: Optional HDF5 group name
            
        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, 'w') as f:
            if group:
                g = f.create_group(group)
            else:
                g = f
            
            self._save_dict_to_h5(data, g)
            
            # Add metadata
            g.attrs['generated_at'] = datetime.now().isoformat()
            g.attrs['generator_type'] = self.__class__.__name__
            g.attrs['config'] = str(self.config)
        
        return filepath
    
    def _save_dict_to_h5(self, data: Dict[str, Any], group: h5py.Group):
        """Recursively save dictionary to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_dict_to_h5(value, subgroup)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip')
            else:
                group.attrs[key] = value
    
    def load_data(self, filepath: str, group: Optional[str] = None) -> Dict[str, Any]:
        """
        Load data from HDF5 file.
        
        Args:
            filepath: Path to the HDF5 file
            group: Optional HDF5 group name
            
        Returns:
            Dictionary containing the loaded data
        """
        data = {}
        
        with h5py.File(filepath, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            
            self._load_dict_from_h5(g, data)
        
        return data
    
    def _load_dict_from_h5(self, group: h5py.Group, data: Dict[str, Any]):
        """Recursively load dictionary from HDF5 group."""
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = {}
                self._load_dict_from_h5(group[key], data[key])
            else:
                data[key] = group[key][:]
        
        # Load attributes
        for key, value in group.attrs.items():
            data[f"_{key}"] = value
    
    def add_noise(self, data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise to data.
        
        Args:
            data: Input data array
            noise_level: Standard deviation of noise as fraction of data range
            
        Returns:
            Data with added noise
        """
        data_range = np.ptp(data)
        noise_std = noise_level * data_range
        noise = np.random.normal(0, noise_std, data.shape)
        return data + noise
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated data.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Basic validation - check for NaN or infinite values
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    print(f"Warning: Invalid values found in {key}")
                    return False
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the generator."""
        return {
            'generator_type': self.__class__.__name__,
            'config_path': self.config_path,
            'output_dir': self.output_dir,
            'generated_at': datetime.now().isoformat()
        }