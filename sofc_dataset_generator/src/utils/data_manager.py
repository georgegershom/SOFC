"""
Data Management and Storage System

This module handles the export, storage, and management of paired warp-stress datasets
for ML training, including various data formats and compression options.
"""

import numpy as np
import pandas as pd
import h5py
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import hashlib
from datetime import datetime
import logging

from ..extraction.warp_extractor import WarpFieldData
from ..extraction.stress_extractor import StressFieldData


@dataclass
class DatasetSample:
    """Container for a single dataset sample with paired warp-stress data."""
    sample_id: str
    doe_parameters: Dict[str, Any]
    warp_data: WarpFieldData
    stress_data: StressFieldData
    metadata: Dict[str, Any]
    timestamp: str
    
    def get_hash(self) -> str:
        """Generate unique hash for this sample."""
        # Create hash from DOE parameters
        param_str = json.dumps(self.doe_parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()


@dataclass
class DatasetMetadata:
    """Metadata for the complete dataset."""
    dataset_name: str
    version: str
    creation_date: str
    n_samples: int
    parameter_ranges: Dict[str, Dict[str, float]]
    data_format: str
    compression: str
    file_size_mb: float
    description: str
    tags: List[str]


class SOFCDatasetManager:
    """Manages SOFC dataset creation, storage, and retrieval."""
    
    def __init__(self, output_directory: Union[str, Path]):
        """Initialize dataset manager.
        
        Args:
            output_directory: Directory to store dataset files
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.samples = []
        self.metadata = None
        
    def add_sample(self, sample: DatasetSample):
        """Add a sample to the dataset."""
        self.samples.append(sample)
        self.logger.info(f"Added sample {sample.sample_id} to dataset")
    
    def create_dataset_from_samples(self, dataset_name: str, 
                                  description: str = "",
                                  tags: List[str] = None) -> DatasetMetadata:
        """Create dataset from accumulated samples."""
        
        if not self.samples:
            raise ValueError("No samples available to create dataset")
        
        # Generate metadata
        metadata = self._generate_metadata(dataset_name, description, tags or [])
        self.metadata = metadata
        
        # Export dataset in multiple formats
        self._export_hdf5_dataset(dataset_name)
        self._export_npz_dataset(dataset_name)
        self._export_csv_metadata(dataset_name)
        self._export_json_metadata(dataset_name)
        
        self.logger.info(f"Created dataset '{dataset_name}' with {len(self.samples)} samples")
        return metadata
    
    def _generate_metadata(self, dataset_name: str, description: str, tags: List[str]) -> DatasetMetadata:
        """Generate dataset metadata."""
        
        # Calculate parameter ranges
        param_ranges = self._calculate_parameter_ranges()
        
        # Estimate file size (rough approximation)
        estimated_size = self._estimate_file_size()
        
        metadata = DatasetMetadata(
            dataset_name=dataset_name,
            version="1.0",
            creation_date=datetime.now().isoformat(),
            n_samples=len(self.samples),
            parameter_ranges=param_ranges,
            data_format="HDF5",
            compression="gzip",
            file_size_mb=estimated_size,
            description=description,
            tags=tags
        )
        
        return metadata
    
    def _calculate_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """Calculate parameter ranges across all samples."""
        
        if not self.samples:
            return {}
        
        # Collect all parameter values
        param_values = {}
        for sample in self.samples:
            for param_name, value in sample.doe_parameters.items():
                if isinstance(value, (int, float)):
                    if param_name not in param_values:
                        param_values[param_name] = []
                    param_values[param_name].append(value)
        
        # Calculate ranges
        param_ranges = {}
        for param_name, values in param_values.items():
            param_ranges[param_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return param_ranges
    
    def _estimate_file_size(self) -> float:
        """Estimate total file size in MB."""
        
        if not self.samples:
            return 0.0
        
        # Estimate size of one sample
        sample = self.samples[0]
        
        # Warp data size
        warp_size = (
            sample.warp_data.height_map_top.nbytes +
            sample.warp_data.height_map_bottom.nbytes +
            sample.warp_data.original_coordinates.nbytes +
            sample.warp_data.displacement_vectors.nbytes
        )
        
        # Stress data size
        stress_size = (
            sample.stress_data.stress_tensor.nbytes +
            sample.stress_data.voxelized_stress.nbytes +
            sample.stress_data.von_mises_stress.nbytes
        )
        
        # Total size per sample
        sample_size = warp_size + stress_size
        
        # Total dataset size (with compression factor ~0.3)
        total_size_mb = (sample_size * len(self.samples) * 0.3) / (1024 * 1024)
        
        return total_size_mb
    
    def _export_hdf5_dataset(self, dataset_name: str):
        """Export dataset to HDF5 format."""
        
        filepath = self.output_dir / f"{dataset_name}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            samples_group = f.create_group('samples')
            metadata_group = f.create_group('metadata')
            
            # Store each sample
            for i, sample in enumerate(self.samples):
                sample_group = samples_group.create_group(f'sample_{i:06d}')
                
                # DOE parameters
                param_group = sample_group.create_group('parameters')
                for key, value in sample.doe_parameters.items():
                    if isinstance(value, (int, float)):
                        param_group.attrs[key] = value
                    else:
                        param_group.attrs[key] = str(value)
                
                # Warp data
                warp_group = sample_group.create_group('warp_data')
                warp_group.create_dataset('height_map_top', data=sample.warp_data.height_map_top, 
                                        compression='gzip', compression_opts=9)
                warp_group.create_dataset('height_map_bottom', data=sample.warp_data.height_map_bottom,
                                        compression='gzip', compression_opts=9)
                warp_group.create_dataset('grid_x', data=sample.warp_data.grid_x,
                                        compression='gzip', compression_opts=9)
                warp_group.create_dataset('grid_y', data=sample.warp_data.grid_y,
                                        compression='gzip', compression_opts=9)
                warp_group.create_dataset('original_coordinates', data=sample.warp_data.original_coordinates,
                                        compression='gzip', compression_opts=9)
                warp_group.create_dataset('displacement_vectors', data=sample.warp_data.displacement_vectors,
                                        compression='gzip', compression_opts=9)
                warp_group.create_dataset('curvature_map', data=sample.warp_data.curvature_map,
                                        compression='gzip', compression_opts=9)
                
                # Warp metadata
                warp_group.attrs['grid_resolution'] = sample.warp_data.grid_resolution
                warp_group.attrs['bounds'] = sample.warp_data.bounds
                for key, value in sample.warp_data.statistics.items():
                    warp_group.attrs[f'stat_{key}'] = value
                
                # Stress data
                stress_group = sample_group.create_group('stress_data')
                stress_group.create_dataset('stress_tensor', data=sample.stress_data.stress_tensor,
                                          compression='gzip', compression_opts=9)
                stress_group.create_dataset('voxelized_stress', data=sample.stress_data.voxelized_stress,
                                          compression='gzip', compression_opts=9)
                stress_group.create_dataset('von_mises_stress', data=sample.stress_data.von_mises_stress,
                                          compression='gzip', compression_opts=9)
                stress_group.create_dataset('principal_stresses', data=sample.stress_data.principal_stresses,
                                          compression='gzip', compression_opts=9)
                stress_group.create_dataset('coordinates', data=sample.stress_data.coordinates,
                                          compression='gzip', compression_opts=9)
                
                # Stress metadata
                stress_group.attrs['voxel_resolution'] = sample.stress_data.voxel_resolution
                stress_group.attrs['bounds'] = sample.stress_data.bounds
                for key, value in sample.stress_data.statistics.items():
                    stress_group.attrs[f'stat_{key}'] = value
                
                # Sample metadata
                sample_group.attrs['sample_id'] = sample.sample_id
                sample_group.attrs['timestamp'] = sample.timestamp
                sample_group.attrs['hash'] = sample.get_hash()
            
            # Dataset metadata
            if self.metadata:
                metadata_group.attrs['dataset_name'] = self.metadata.dataset_name
                metadata_group.attrs['version'] = self.metadata.version
                metadata_group.attrs['creation_date'] = self.metadata.creation_date
                metadata_group.attrs['n_samples'] = self.metadata.n_samples
                metadata_group.attrs['description'] = self.metadata.description
                metadata_group.attrs['file_size_mb'] = self.metadata.file_size_mb
        
        self.logger.info(f"Exported HDF5 dataset to {filepath}")
    
    def _export_npz_dataset(self, dataset_name: str):
        """Export dataset to compressed NumPy format."""
        
        filepath = self.output_dir / f"{dataset_name}.npz"
        
        # Collect all data
        data_dict = {}
        
        # Stack warp data
        height_maps_top = np.stack([s.warp_data.height_map_top for s in self.samples])
        height_maps_bottom = np.stack([s.warp_data.height_map_bottom for s in self.samples])
        
        data_dict['height_maps_top'] = height_maps_top
        data_dict['height_maps_bottom'] = height_maps_bottom
        
        # Stack stress data
        stress_tensors = np.stack([s.stress_data.stress_tensor for s in self.samples])
        von_mises_stresses = np.stack([s.stress_data.von_mises_stress for s in self.samples])
        
        data_dict['stress_tensors'] = stress_tensors
        data_dict['von_mises_stresses'] = von_mises_stresses
        
        # DOE parameters as structured array
        param_names = list(self.samples[0].doe_parameters.keys())
        param_data = np.array([[s.doe_parameters.get(name, 0.0) for name in param_names] 
                              for s in self.samples])
        data_dict['doe_parameters'] = param_data
        data_dict['parameter_names'] = param_names
        
        # Sample IDs
        data_dict['sample_ids'] = [s.sample_id for s in self.samples]
        
        # Save compressed
        np.savez_compressed(filepath, **data_dict)
        
        self.logger.info(f"Exported NPZ dataset to {filepath}")
    
    def _export_csv_metadata(self, dataset_name: str):
        """Export sample metadata to CSV format."""
        
        filepath = self.output_dir / f"{dataset_name}_metadata.csv"
        
        # Create DataFrame with sample metadata
        rows = []
        for sample in self.samples:
            row = {
                'sample_id': sample.sample_id,
                'timestamp': sample.timestamp,
                'hash': sample.get_hash(),
            }
            
            # Add DOE parameters
            row.update(sample.doe_parameters)
            
            # Add summary statistics
            row.update({f'warp_{k}': v for k, v in sample.warp_data.statistics.items()})
            row.update({f'stress_{k}': v for k, v in sample.stress_data.statistics.items()})
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Exported CSV metadata to {filepath}")
    
    def _export_json_metadata(self, dataset_name: str):
        """Export dataset metadata to JSON format."""
        
        filepath = self.output_dir / f"{dataset_name}_info.json"
        
        if self.metadata:
            metadata_dict = asdict(self.metadata)
            
            with open(filepath, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
        
        self.logger.info(f"Exported JSON metadata to {filepath}")
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> List[DatasetSample]:
        """Load dataset from HDF5 file."""
        
        dataset_path = Path(dataset_path)
        samples = []
        
        with h5py.File(dataset_path, 'r') as f:
            samples_group = f['samples']
            
            for sample_key in samples_group.keys():
                sample_group = samples_group[sample_key]
                
                # Load DOE parameters
                param_group = sample_group['parameters']
                doe_parameters = dict(param_group.attrs)
                
                # Load warp data
                warp_group = sample_group['warp_data']
                warp_data = WarpFieldData(
                    original_coordinates=warp_group['original_coordinates'][:],
                    deformed_coordinates=warp_group['original_coordinates'][:] + warp_group['displacement_vectors'][:],
                    displacement_vectors=warp_group['displacement_vectors'][:],
                    grid_x=warp_group['grid_x'][:],
                    grid_y=warp_group['grid_y'][:],
                    height_map_top=warp_group['height_map_top'][:],
                    height_map_bottom=warp_group['height_map_bottom'][:],
                    warp_magnitude=np.linalg.norm(warp_group['displacement_vectors'][:], axis=1),
                    curvature_map=warp_group['curvature_map'][:],
                    gradient_map=np.zeros_like(warp_group['curvature_map'][:]),  # Placeholder
                    grid_resolution=warp_group.attrs['grid_resolution'],
                    bounds=tuple(warp_group.attrs['bounds']),
                    statistics={k.replace('stat_', ''): v for k, v in warp_group.attrs.items() if k.startswith('stat_')}
                )
                
                # Load stress data
                stress_group = sample_group['stress_data']
                stress_data = StressFieldData(
                    coordinates=stress_group['coordinates'][:],
                    stress_tensor=stress_group['stress_tensor'][:],
                    principal_stresses=stress_group['principal_stresses'][:],
                    von_mises_stress=stress_group['von_mises_stress'][:],
                    hydrostatic_stress=np.mean(stress_group['stress_tensor'][:, :3], axis=1),
                    voxel_grid_x=np.zeros((10, 10, 10)),  # Placeholder
                    voxel_grid_y=np.zeros((10, 10, 10)),  # Placeholder
                    voxel_grid_z=np.zeros((10, 10, 10)),  # Placeholder
                    voxelized_stress=stress_group['voxelized_stress'][:],
                    layer_stress_maps={},  # Placeholder
                    stress_invariants=np.zeros((len(stress_group['stress_tensor']), 3)),  # Placeholder
                    stress_gradients=np.zeros((10, 10, 10, 6, 3)),  # Placeholder
                    voxel_resolution=stress_group.attrs['voxel_resolution'],
                    bounds=tuple(stress_group.attrs['bounds']),
                    layer_info={},
                    statistics={k.replace('stat_', ''): v for k, v in stress_group.attrs.items() if k.startswith('stat_')}
                )
                
                # Create sample
                sample = DatasetSample(
                    sample_id=sample_group.attrs['sample_id'],
                    doe_parameters=doe_parameters,
                    warp_data=warp_data,
                    stress_data=stress_data,
                    metadata={},
                    timestamp=sample_group.attrs['timestamp']
                )
                
                samples.append(sample)
        
        self.logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
        return samples
    
    def create_ml_ready_dataset(self, dataset_name: str, 
                               feature_extraction_config: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """Create ML-ready dataset with extracted features."""
        
        if not self.samples:
            raise ValueError("No samples available")
        
        # Default feature extraction config
        if feature_extraction_config is None:
            feature_extraction_config = {
                'include_height_maps': True,
                'include_stress_tensors': True,
                'include_derived_quantities': True,
                'flatten_spatial_data': True,
                'normalize_features': True
            }
        
        # Extract features from all samples
        X_warp_list = []
        X_stress_list = []
        y_params_list = []
        
        for sample in self.samples:
            # Extract warp features (inputs)
            if feature_extraction_config.get('include_height_maps', True):
                warp_features = np.concatenate([
                    sample.warp_data.height_map_top.ravel(),
                    sample.warp_data.height_map_bottom.ravel(),
                    sample.warp_data.curvature_map.ravel()
                ])
                X_warp_list.append(warp_features)
            
            # Extract stress features (targets)
            if feature_extraction_config.get('include_stress_tensors', True):
                stress_features = sample.stress_data.stress_tensor.ravel()
                X_stress_list.append(stress_features)
            
            # Extract DOE parameters
            param_values = np.array([v for v in sample.doe_parameters.values() if isinstance(v, (int, float))])
            y_params_list.append(param_values)
        
        # Stack all features
        ml_dataset = {}
        
        if X_warp_list:
            ml_dataset['X_warp'] = np.stack(X_warp_list)
        
        if X_stress_list:
            ml_dataset['X_stress'] = np.stack(X_stress_list)
        
        if y_params_list:
            ml_dataset['y_parameters'] = np.stack(y_params_list)
        
        # Normalize if requested
        if feature_extraction_config.get('normalize_features', True):
            from sklearn.preprocessing import StandardScaler
            
            for key in ['X_warp', 'X_stress']:
                if key in ml_dataset:
                    scaler = StandardScaler()
                    ml_dataset[key] = scaler.fit_transform(ml_dataset[key])
                    ml_dataset[f'{key}_scaler'] = scaler
        
        # Save ML dataset
        ml_filepath = self.output_dir / f"{dataset_name}_ml_ready.npz"
        np.savez_compressed(ml_filepath, **{k: v for k, v in ml_dataset.items() 
                                          if not isinstance(v, StandardScaler)})
        
        # Save scalers separately
        scalers = {k: v for k, v in ml_dataset.items() if isinstance(v, StandardScaler)}
        if scalers:
            scaler_filepath = self.output_dir / f"{dataset_name}_scalers.pkl"
            with open(scaler_filepath, 'wb') as f:
                pickle.dump(scalers, f)
        
        self.logger.info(f"Created ML-ready dataset with shapes: {[(k, v.shape) for k, v in ml_dataset.items() if hasattr(v, 'shape')]}")
        
        return ml_dataset
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        
        if not self.samples:
            return {}
        
        stats = {
            'n_samples': len(self.samples),
            'parameter_ranges': self._calculate_parameter_ranges(),
            'warp_statistics': {},
            'stress_statistics': {},
            'data_sizes': {}
        }
        
        # Aggregate warp statistics
        warp_stats = [s.warp_data.statistics for s in self.samples]
        for key in warp_stats[0].keys():
            values = [ws[key] for ws in warp_stats]
            stats['warp_statistics'][key] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Aggregate stress statistics
        stress_stats = [s.stress_data.statistics for s in self.samples]
        for key in stress_stats[0].keys():
            values = [ss[key] for ss in stress_stats]
            stats['stress_statistics'][key] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Data size statistics
        sample = self.samples[0]
        stats['data_sizes'] = {
            'height_map_shape': sample.warp_data.height_map_top.shape,
            'stress_tensor_shape': sample.stress_data.stress_tensor.shape,
            'n_nodes': len(sample.warp_data.original_coordinates),
            'n_elements': len(sample.stress_data.stress_tensor)
        }
        
        return stats


def create_dataset_manager(output_directory: Union[str, Path]) -> SOFCDatasetManager:
    """Factory function to create dataset manager."""
    return SOFCDatasetManager(output_directory)


if __name__ == "__main__":
    # Test dataset manager
    manager = create_dataset_manager("test_output")
    print("Dataset manager created successfully!")
    print(f"Output directory: {manager.output_dir}")