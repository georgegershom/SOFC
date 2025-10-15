"""Data handling and storage utilities for SOFC dataset."""

import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import yaml
from pathlib import Path
import hashlib
from datetime import datetime
from tqdm import tqdm
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


class DatasetManager:
    """Manage multi-fidelity SOFC dataset storage and retrieval."""
    
    def __init__(self, base_path: str = "./data"):
        """Initialize dataset manager."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Define fidelity levels
        self.fidelity_levels = ['low_fidelity', 'mid_fidelity', 
                                'high_fidelity', 'experimental']
        
        # Initialize file paths
        self.file_paths = {}
        for fidelity in self.fidelity_levels:
            fidelity_path = self.base_path / fidelity
            fidelity_path.mkdir(exist_ok=True)
            self.file_paths[fidelity] = fidelity_path / f"{fidelity}_data.h5"
    
    def create_hdf5_structure(self, fidelity: str, n_samples: int,
                              spatial_dims: Tuple[int, ...],
                              config: Dict) -> h5py.File:
        """
        Create HDF5 file structure for a fidelity level.
        
        Args:
            fidelity: Fidelity level name
            n_samples: Number of samples
            spatial_dims: Spatial dimensions for fields
            config: Configuration dictionary
            
        Returns:
            HDF5 file handle
        """
        file_path = self.file_paths[fidelity]
        f = h5py.File(file_path, 'w')
        
        # Add metadata
        f.attrs['fidelity'] = fidelity
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n_samples
        f.attrs['spatial_dims'] = spatial_dims
        f.attrs['config'] = json.dumps(config)
        
        # Create groups for different data types
        inputs = f.create_group('inputs')
        outputs = f.create_group('outputs')
        metadata = f.create_group('metadata')
        
        # Input datasets
        self._create_input_datasets(inputs, n_samples, config)
        
        # Output datasets
        self._create_output_datasets(outputs, n_samples, spatial_dims, fidelity)
        
        # Metadata datasets
        self._create_metadata_datasets(metadata, n_samples)
        
        return f
    
    def _create_input_datasets(self, group: h5py.Group, n_samples: int, 
                               config: Dict):
        """Create input variable datasets."""
        # Operating conditions
        group.create_dataset('temperature', shape=(n_samples,), dtype='f8')
        group.create_dataset('current_density', shape=(n_samples,), dtype='f8')
        group.create_dataset('fuel_utilization', shape=(n_samples,), dtype='f8')
        group.create_dataset('air_utilization', shape=(n_samples,), dtype='f8')
        group.create_dataset('pressure', shape=(n_samples,), dtype='f8')
        
        # Inlet compositions
        n_species = 6  # H2, H2O, O2, N2, CO, CO2
        group.create_dataset('fuel_composition', shape=(n_samples, n_species), dtype='f8')
        group.create_dataset('air_composition', shape=(n_samples, 2), dtype='f8')  # O2, N2
        
        # Geometry variations
        group.create_dataset('anode_thickness', shape=(n_samples,), dtype='f8')
        group.create_dataset('electrolyte_thickness', shape=(n_samples,), dtype='f8')
        group.create_dataset('cathode_thickness', shape=(n_samples,), dtype='f8')
        
        # Material properties variations
        group.create_dataset('anode_porosity', shape=(n_samples,), dtype='f8')
        group.create_dataset('cathode_porosity', shape=(n_samples,), dtype='f8')
        
        # Degradation state
        group.create_dataset('operating_time', shape=(n_samples,), dtype='f8')
        group.create_dataset('thermal_cycles', shape=(n_samples,), dtype='i4')
        group.create_dataset('redox_cycles', shape=(n_samples,), dtype='i4')
    
    def _create_output_datasets(self, group: h5py.Group, n_samples: int,
                                spatial_dims: Tuple[int, ...], fidelity: str):
        """Create output variable datasets based on fidelity."""
        
        if fidelity == 'low_fidelity':
            # 0D/1D averaged outputs
            group.create_dataset('voltage', shape=(n_samples,), dtype='f8')
            group.create_dataset('power_density', shape=(n_samples,), dtype='f8')
            group.create_dataset('avg_temperature', shape=(n_samples,), dtype='f8')
            group.create_dataset('avg_current_density', shape=(n_samples,), dtype='f8')
            group.create_dataset('avg_stress', shape=(n_samples,), dtype='f8')
            group.create_dataset('degradation_rate', shape=(n_samples,), dtype='f8')
            
        elif fidelity in ['mid_fidelity', 'high_fidelity']:
            # Spatial field outputs
            # Temperature field
            group.create_dataset('temperature_field', 
                               shape=(n_samples, *spatial_dims), 
                               dtype='f8', chunks=True, compression='gzip')
            
            # Current density field
            group.create_dataset('current_density_field',
                               shape=(n_samples, *spatial_dims),
                               dtype='f8', chunks=True, compression='gzip')
            
            # Species concentrations (6 species)
            group.create_dataset('species_concentration',
                               shape=(n_samples, 6, *spatial_dims),
                               dtype='f8', chunks=True, compression='gzip')
            
            # Overpotentials (3 types)
            group.create_dataset('overpotentials',
                               shape=(n_samples, 3, *spatial_dims),
                               dtype='f8', chunks=True, compression='gzip')
            
            # Stress tensor (6 components)
            group.create_dataset('stress_tensor',
                               shape=(n_samples, 6, *spatial_dims),
                               dtype='f8', chunks=True, compression='gzip')
            
            # Strain tensor (6 components x 4 types)
            group.create_dataset('strain_tensor',
                               shape=(n_samples, 24, *spatial_dims),
                               dtype='f8', chunks=True, compression='gzip')
            
            # Damage indicators
            group.create_dataset('damage_field',
                               shape=(n_samples, *spatial_dims),
                               dtype='f8', chunks=True, compression='gzip')
            
            # Performance metrics
            group.create_dataset('voltage', shape=(n_samples,), dtype='f8')
            group.create_dataset('power_density', shape=(n_samples,), dtype='f8')
            
        elif fidelity == 'experimental':
            # Experimental measurements with uncertainties
            group.create_dataset('iv_curve', shape=(n_samples, 20, 2), dtype='f8')
            group.create_dataset('iv_uncertainty', shape=(n_samples, 20, 2), dtype='f8')
            group.create_dataset('eis_data', shape=(n_samples, 50, 2), dtype='f8')
            group.create_dataset('eis_uncertainty', shape=(n_samples, 50, 2), dtype='f8')
            group.create_dataset('temperature_map', shape=(n_samples, 64, 64), dtype='f8')
            group.create_dataset('temperature_uncertainty', shape=(n_samples,), dtype='f8')
            group.create_dataset('microstructure_params', shape=(n_samples, 10), dtype='f8')
    
    def _create_metadata_datasets(self, group: h5py.Group, n_samples: int):
        """Create metadata datasets."""
        group.create_dataset('sample_id', shape=(n_samples,), dtype='S32')
        group.create_dataset('timestamp', shape=(n_samples,), dtype='f8')
        group.create_dataset('computation_time', shape=(n_samples,), dtype='f8')
        group.create_dataset('convergence_flag', shape=(n_samples,), dtype='bool')
        group.create_dataset('notes', shape=(n_samples,), dtype='S256')
    
    def write_sample(self, f: h5py.File, idx: int, data: Dict):
        """Write a single sample to HDF5 file."""
        # Write inputs
        for key, value in data['inputs'].items():
            if key in f['inputs']:
                f['inputs'][key][idx] = value
        
        # Write outputs
        for key, value in data['outputs'].items():
            if key in f['outputs']:
                f['outputs'][key][idx] = value
        
        # Write metadata
        for key, value in data.get('metadata', {}).items():
            if key in f['metadata']:
                if isinstance(value, str):
                    f['metadata'][key][idx] = value.encode()
                else:
                    f['metadata'][key][idx] = value
    
    def read_dataset(self, fidelity: str, indices: Optional[List[int]] = None) -> Dict:
        """Read dataset for a specific fidelity level."""
        file_path = self.file_paths[fidelity]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            data = {
                'inputs': {},
                'outputs': {},
                'metadata': {}
            }
            
            # Read all or selected indices
            if indices is None:
                indices = slice(None)
            else:
                indices = np.array(indices)
            
            # Read inputs
            for key in f['inputs'].keys():
                data['inputs'][key] = f['inputs'][key][indices]
            
            # Read outputs
            for key in f['outputs'].keys():
                data['outputs'][key] = f['outputs'][key][indices]
            
            # Read metadata
            for key in f['metadata'].keys():
                data['metadata'][key] = f['metadata'][key][indices]
            
            # Add attributes
            data['attrs'] = dict(f.attrs)
            
        return data
    
    def create_zarr_store(self, fidelity: str, n_samples: int,
                         spatial_dims: Tuple[int, ...]):
        """Create Zarr store for large-scale parallel writing."""
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is not installed. Install with: pip install zarr")
        store_path = self.base_path / fidelity / f"{fidelity}_data.zarr"
        store = zarr.open_group(str(store_path), mode='w')
        
        # Similar structure to HDF5 but optimized for parallel access
        store.attrs['fidelity'] = fidelity
        store.attrs['created'] = datetime.now().isoformat()
        store.attrs['n_samples'] = n_samples
        store.attrs['spatial_dims'] = spatial_dims
        
        return store
    
    def validate_dataset(self, fidelity: str) -> Dict[str, Any]:
        """Validate dataset integrity and statistics."""
        file_path = self.file_paths[fidelity]
        
        with h5py.File(file_path, 'r') as f:
            report = {
                'fidelity': fidelity,
                'n_samples': f.attrs['n_samples'],
                'file_size_mb': file_path.stat().st_size / 1e6,
                'datasets': {}
            }
            
            for group_name in ['inputs', 'outputs', 'metadata']:
                if group_name in f:
                    group = f[group_name]
                    report['datasets'][group_name] = {}
                    
                    for key in group.keys():
                        dataset = group[key]
                        stats = {
                            'shape': dataset.shape,
                            'dtype': str(dataset.dtype),
                            'size_mb': dataset.nbytes / 1e6
                        }
                        
                        # Compute statistics for numeric data
                        if dataset.dtype.kind in ['f', 'i']:
                            data_sample = dataset[:min(1000, len(dataset))]
                            if data_sample.ndim == 1:
                                stats.update({
                                    'min': float(np.min(data_sample)),
                                    'max': float(np.max(data_sample)),
                                    'mean': float(np.mean(data_sample)),
                                    'std': float(np.std(data_sample))
                                })
                        
                        report['datasets'][group_name][key] = stats
            
        return report


class DataAugmentation:
    """Data augmentation techniques for multi-fidelity learning."""
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01,
                  noise_type: str = 'gaussian') -> np.ndarray:
        """Add noise to data for robustness."""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, data.shape)
        elif noise_type == 'poisson':
            noise = np.random.poisson(noise_level * np.abs(data))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return data + noise
    
    @staticmethod
    def interpolate_fields(field_coarse: np.ndarray, 
                          target_shape: Tuple[int, ...]) -> np.ndarray:
        """Interpolate coarse fields to finer resolution."""
        from scipy.ndimage import zoom
        
        zoom_factors = [t / s for t, s in zip(target_shape, field_coarse.shape)]
        field_fine = zoom(field_coarse, zoom_factors, order=3)
        
        return field_fine
    
    @staticmethod
    def generate_synthetic_variations(base_data: Dict, n_variations: int = 10,
                                     variation_scale: float = 0.1) -> List[Dict]:
        """Generate synthetic variations of base data."""
        variations = []
        
        for i in range(n_variations):
            varied_data = {}
            for key, value in base_data.items():
                if isinstance(value, np.ndarray):
                    # Add random variations
                    variation = np.random.normal(0, variation_scale * np.std(value), 
                                               value.shape)
                    varied_data[key] = value + variation
                elif isinstance(value, (int, float)):
                    variation = np.random.normal(0, variation_scale * abs(value))
                    varied_data[key] = value + variation
                else:
                    varied_data[key] = value
            
            variations.append(varied_data)
        
        return variations


def create_sample_hash(data: Dict) -> str:
    """Create unique hash for a data sample."""
    # Convert data to string representation
    data_str = json.dumps(data, sort_keys=True, default=str)
    
    # Create hash
    hash_obj = hashlib.sha256(data_str.encode())
    return hash_obj.hexdigest()[:16]


def save_config(config: Dict, filepath: str):
    """Save configuration to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(filepath: str) -> Dict:
    """Load configuration from YAML file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config