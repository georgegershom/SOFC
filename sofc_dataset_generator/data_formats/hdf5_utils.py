"""
HDF5 utilities for SOFC dataset storage and retrieval.
Provides efficient storage and access to large multi-dimensional datasets.
"""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
from pathlib import Path

class HDF5DatasetManager:
    """
    HDF5 dataset manager for SOFC digital twin datasets.
    
    Provides:
    - Efficient storage of large multi-dimensional arrays
    - Compression and chunking for optimal performance
    - Metadata management
    - Data validation and integrity checks
    """
    
    def __init__(self, file_path: str, mode: str = 'a'):
        """
        Initialize HDF5 dataset manager.
        
        Args:
            file_path: Path to HDF5 file
            mode: File mode ('r', 'w', 'a')
        """
        self.file_path = file_path
        self.mode = mode
        self.logger = logging.getLogger('HDF5DatasetManager')
        
    def save_dataset(
        self,
        dataset: Dict[str, Any],
        group_name: str = 'dataset',
        compression: str = 'gzip',
        compression_opts: int = 9,
        chunk_size: int = 1000
    ) -> None:
        """
        Save dataset to HDF5 file.
        
        Args:
            dataset: Dataset dictionary to save
            group_name: Name of the group to create
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (0-9 for gzip)
            chunk_size: Chunk size for large arrays
        """
        self.logger.info(f"Saving dataset to {self.file_path}")
        
        with h5py.File(self.file_path, self.mode) as f:
            # Create or get group
            if group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            
            # Save metadata
            self._save_metadata(group, dataset.get('metadata', {}))
            
            # Save data arrays
            for key, value in dataset.items():
                if key == 'metadata':
                    continue
                elif isinstance(value, np.ndarray):
                    self._save_array(group, key, value, compression, compression_opts, chunk_size)
                elif isinstance(value, dict):
                    self._save_dict(group, key, value, compression, compression_opts, chunk_size)
                elif isinstance(value, (list, tuple)):
                    self._save_list(group, key, value, compression, compression_opts, chunk_size)
                else:
                    # Save scalar values as attributes
                    group.attrs[key] = value
            
            # Add file metadata
            group.attrs['file_version'] = '1.0.0'
            group.attrs['creation_time'] = np.datetime64('now').astype(str)
            group.attrs['data_types'] = self._get_data_types(dataset)
    
    def load_dataset(
        self,
        group_name: str = 'dataset',
        load_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Load dataset from HDF5 file.
        
        Args:
            group_name: Name of the group to load
            load_metadata: Whether to load metadata
            
        Returns:
            Loaded dataset dictionary
        """
        self.logger.info(f"Loading dataset from {self.file_path}")
        
        dataset = {}
        
        with h5py.File(self.file_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in file")
            
            group = f[group_name]
            
            # Load metadata
            if load_metadata:
                dataset['metadata'] = self._load_metadata(group)
            
            # Load data arrays
            for key in group.keys():
                if key == 'metadata':
                    continue
                
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    dataset[key] = item[:]
                elif isinstance(item, h5py.Group):
                    dataset[key] = self._load_group(item)
            
            # Load attributes
            for key, value in group.attrs.items():
                if key not in dataset:
                    dataset[key] = value
        
        return dataset
    
    def save_high_fidelity_data(
        self,
        dataset: Dict[str, Any],
        output_file: str,
        compression: str = 'gzip'
    ) -> None:
        """
        Save high-fidelity simulation data with optimized storage.
        
        Args:
            dataset: High-fidelity dataset
            output_file: Output file path
            compression: Compression algorithm
        """
        self.logger.info(f"Saving high-fidelity data to {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # Create main group
            main_group = f.create_group('high_fidelity_data')
            
            # Save metadata
            metadata = {
                'dataset_type': 'high_fidelity',
                'n_samples': len(dataset.get('parameters', [])),
                'spatial_resolution': dataset.get('spatial_resolution', []),
                'time_steps': dataset.get('time_steps', 0),
                'generation_time': np.datetime64('now').astype(str)
            }
            self._save_metadata(main_group, metadata)
            
            # Save parameters
            if 'parameters' in dataset:
                params_group = main_group.create_group('parameters')
                for i, param_dict in enumerate(dataset['parameters']):
                    param_dataset = params_group.create_group(f'sample_{i:06d}')
                    for key, value in param_dict.items():
                        param_dataset.attrs[key] = value
            
            # Save mesh
            if 'mesh' in dataset:
                mesh_data = dataset['mesh']
                main_group.create_dataset(
                    'mesh', data=mesh_data, compression=compression, chunks=True
                )
            
            # Save field data with chunking
            field_data_keys = [
                'potential_field', 'current_density_field', 'temperature_field',
                'displacement_field', 'stress_tensor', 'strain_tensor',
                'h2_concentration', 'h2o_concentration', 'o2_concentration'
            ]
            
            for key in field_data_keys:
                if key in dataset:
                    data = dataset[key]
                    if isinstance(data, np.ndarray):
                        # Calculate optimal chunk size
                        chunk_shape = self._calculate_chunk_shape(data.shape)
                        main_group.create_dataset(
                            key, data=data, compression=compression,
                            chunks=chunk_shape, shuffle=True
                        )
            
            # Save scalar data
            scalar_keys = [
                'cell_voltage', 'max_temperature', 'max_stress',
                'max_displacement', 'efficiency', 'power_density'
            ]
            
            for key in scalar_keys:
                if key in dataset:
                    data = dataset[key]
                    if isinstance(data, np.ndarray):
                        main_group.create_dataset(key, data=data, compression=compression)
    
    def save_experimental_data(
        self,
        dataset: Dict[str, Any],
        output_file: str,
        compression: str = 'gzip'
    ) -> None:
        """
        Save experimental data with time-series optimization.
        
        Args:
            dataset: Experimental dataset
            output_file: Output file path
            compression: Compression algorithm
        """
        self.logger.info(f"Saving experimental data to {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # Create main group
            main_group = f.create_group('experimental_data')
            
            # Save metadata
            metadata = dataset.get('metadata', {})
            self._save_metadata(main_group, metadata)
            
            # Save time series data
            if 'time_points' in dataset:
                main_group.create_dataset(
                    'time_points', data=dataset['time_points'], compression=compression
                )
            
            # Save global operational data
            if 'global_operational' in dataset:
                global_group = main_group.create_group('global_operational')
                for key, value in dataset['global_operational'].items():
                    if isinstance(value, np.ndarray):
                        global_group.create_dataset(
                            key, data=value, compression=compression, chunks=True
                        )
            
            # Save EIS data
            if 'eis_data' in dataset:
                eis_group = main_group.create_group('eis_data')
                eis_data = dataset['eis_data']
                
                if 'frequencies' in eis_data:
                    eis_group.create_dataset(
                        'frequencies', data=eis_data['frequencies'], compression=compression
                    )
                
                if 'impedance_real' in eis_data:
                    # Save as 2D array (time x frequency)
                    impedance_real = np.array(eis_data['impedance_real'])
                    eis_group.create_dataset(
                        'impedance_real', data=impedance_real, compression=compression, chunks=True
                    )
                
                if 'impedance_imag' in eis_data:
                    impedance_imag = np.array(eis_data['impedance_imag'])
                    eis_group.create_dataset(
                        'impedance_imag', data=impedance_imag, compression=compression, chunks=True
                    )
            
            # Save thermal imaging data
            if 'thermal_imaging' in dataset:
                thermal_group = main_group.create_group('thermal_imaging')
                thermal_data = dataset['thermal_imaging']
                
                if 'temperature_images' in thermal_data:
                    # Save as 3D array (time x height x width)
                    temp_images = np.array(thermal_data['temperature_images'])
                    thermal_group.create_dataset(
                        'temperature_images', data=temp_images, compression=compression, chunks=True
                    )
            
            # Save strain gauge data
            if 'strain_gauges' in dataset:
                strain_group = main_group.create_group('strain_gauges')
                strain_data = dataset['strain_gauges']
                
                for key, value in strain_data.items():
                    if isinstance(value, np.ndarray):
                        strain_group.create_dataset(
                            key, data=value, compression=compression
                        )
    
    def save_monitoring_data(
        self,
        dataset: Dict[str, Any],
        output_file: str,
        compression: str = 'gzip'
    ) -> None:
        """
        Save real-time monitoring data with streaming optimization.
        
        Args:
            dataset: Monitoring dataset
            output_file: Output file path
            compression: Compression algorithm
        """
        self.logger.info(f"Saving monitoring data to {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # Create main group
            main_group = f.create_group('monitoring_data')
            
            # Save metadata
            metadata = dataset.get('metadata', {})
            self._save_metadata(main_group, metadata)
            
            # Save high-frequency data
            if 'high_frequency_data' in dataset:
                hf_group = main_group.create_group('high_frequency_data')
                hf_data = dataset['high_frequency_data']
                
                for key, value in hf_data.items():
                    if isinstance(value, np.ndarray):
                        hf_group.create_dataset(
                            key, data=value, compression=compression, chunks=True
                        )
            
            # Save low-frequency data
            if 'low_frequency_data' in dataset:
                lf_group = main_group.create_group('low_frequency_data')
                lf_data = dataset['low_frequency_data']
                
                for key, value in lf_data.items():
                    if isinstance(value, dict):
                        sub_group = lf_group.create_group(key)
                        self._save_dict(sub_group, '', value, compression, 9, 1000)
            
            # Save data assimilation data
            if 'data_assimilation' in dataset:
                da_group = main_group.create_group('data_assimilation')
                da_data = dataset['data_assimilation']
                
                # Save window data
                if 'windows' in da_data:
                    windows_group = da_group.create_group('windows')
                    for i, window in enumerate(da_data['windows']):
                        window_group = windows_group.create_group(f'window_{i:06d}')
                        self._save_dict(window_group, '', window, compression, 9, 1000)
    
    def _save_metadata(self, group: h5py.Group, metadata: Dict[str, Any]) -> None:
        """Save metadata as JSON string in group attributes."""
        if metadata:
            group.attrs['metadata'] = json.dumps(metadata, default=str)
    
    def _load_metadata(self, group: h5py.Group) -> Dict[str, Any]:
        """Load metadata from group attributes."""
        if 'metadata' in group.attrs:
            return json.loads(group.attrs['metadata'])
        return {}
    
    def _save_array(
        self,
        group: h5py.Group,
        key: str,
        array: np.ndarray,
        compression: str,
        compression_opts: int,
        chunk_size: int
    ) -> None:
        """Save numpy array with compression and chunking."""
        if array.size == 0:
            return
        
        # Calculate chunk shape
        chunk_shape = self._calculate_chunk_shape(array.shape, chunk_size)
        
        # Create dataset
        group.create_dataset(
            key, data=array, compression=compression,
            compression_opts=compression_opts, chunks=chunk_shape,
            shuffle=True
        )
    
    def _save_dict(
        self,
        group: h5py.Group,
        key: str,
        data_dict: Dict[str, Any],
        compression: str,
        compression_opts: int,
        chunk_size: int
    ) -> None:
        """Save dictionary as subgroup."""
        if key:
            sub_group = group.create_group(key)
        else:
            sub_group = group
        
        for sub_key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                self._save_array(sub_group, sub_key, value, compression, compression_opts, chunk_size)
            elif isinstance(value, dict):
                self._save_dict(sub_group, sub_key, value, compression, compression_opts, chunk_size)
            elif isinstance(value, (list, tuple)):
                self._save_list(sub_group, sub_key, value, compression, compression_opts, chunk_size)
            else:
                sub_group.attrs[sub_key] = value
    
    def _save_list(
        self,
        group: h5py.Group,
        key: str,
        data_list: List[Any],
        compression: str,
        compression_opts: int,
        chunk_size: int
    ) -> None:
        """Save list as dataset or subgroup."""
        if not data_list:
            return
        
        # Check if all elements are the same type and can be converted to array
        try:
            array = np.array(data_list)
            self._save_array(group, key, array, compression, compression_opts, chunk_size)
        except (ValueError, TypeError):
            # Save as subgroup with indexed datasets
            sub_group = group.create_group(key)
            for i, item in enumerate(data_list):
                if isinstance(item, np.ndarray):
                    self._save_array(sub_group, f'item_{i:06d}', item, compression, compression_opts, chunk_size)
                elif isinstance(item, dict):
                    self._save_dict(sub_group, f'item_{i:06d}', item, compression, compression_opts, chunk_size)
                else:
                    sub_group.attrs[f'item_{i:06d}'] = item
    
    def _load_group(self, group: h5py.Group) -> Dict[str, Any]:
        """Load subgroup recursively."""
        data = {}
        
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[:]
            elif isinstance(item, h5py.Group):
                data[key] = self._load_group(item)
        
        # Load attributes
        for key, value in group.attrs.items():
            data[key] = value
        
        return data
    
    def _calculate_chunk_shape(self, array_shape: Tuple[int, ...], chunk_size: int = 1000) -> Tuple[int, ...]:
        """Calculate optimal chunk shape for array."""
        if len(array_shape) == 0:
            return ()
        
        # Start with chunk_size for first dimension
        chunk_shape = [min(chunk_size, array_shape[0])]
        
        # For remaining dimensions, use full size or reasonable chunk size
        for dim_size in array_shape[1:]:
            if dim_size <= 1000:
                chunk_shape.append(dim_size)
            else:
                chunk_shape.append(min(1000, dim_size))
        
        return tuple(chunk_shape)
    
    def _get_data_types(self, dataset: Dict[str, Any]) -> Dict[str, str]:
        """Get data types for all items in dataset."""
        data_types = {}
        
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                data_types[key] = f"ndarray{value.shape}"
            elif isinstance(value, dict):
                data_types[key] = "dict"
            elif isinstance(value, (list, tuple)):
                data_types[key] = f"{type(value).__name__}[{len(value)}]"
            else:
                data_types[key] = type(value).__name__
        
        return data_types
    
    def validate_dataset(self, group_name: str = 'dataset') -> Dict[str, Any]:
        """
        Validate dataset integrity and provide statistics.
        
        Args:
            group_name: Name of the group to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                if group_name not in f:
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Group '{group_name}' not found")
                    return validation_results
                
                group = f[group_name]
                
                # Check file size
                file_size = Path(self.file_path).stat().st_size
                validation_results['statistics']['file_size_mb'] = file_size / (1024 * 1024)
                
                # Check group structure
                validation_results['statistics']['n_datasets'] = len([k for k in group.keys() if isinstance(group[k], h5py.Dataset)])
                validation_results['statistics']['n_groups'] = len([k for k in group.keys() if isinstance(group[k], h5py.Group)])
                
                # Check for required metadata
                if 'metadata' not in group.attrs:
                    validation_results['warnings'].append("No metadata found")
                
                # Check dataset integrity
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        try:
                            # Try to read a small portion
                            _ = item[0] if item.size > 0 else None
                        except Exception as e:
                            validation_results['valid'] = False
                            validation_results['errors'].append(f"Error reading dataset '{key}': {str(e)}")
        
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Error opening file: {str(e)}")
        
        return validation_results