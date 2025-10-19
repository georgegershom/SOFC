"""
HDF5 Data Exporter

Exports SOFC datasets to HDF5 format for efficient storage and ML training.
"""

import h5py
import numpy as np
from typing import Dict, List, Optional, Any
import json
from .ml_dataset import MLDataset, WarpStressPair


class HDF5Exporter:
    """Exports ML datasets to HDF5 format"""
    
    def __init__(self, filename: str, mode: str = 'w'):
        self.filename = filename
        self.mode = mode
    
    def export_dataset(self, dataset: MLDataset, compression: str = 'gzip', 
                      compression_opts: int = 9):
        """
        Export complete dataset to HDF5 file
        
        Args:
            dataset: MLDataset object to export
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (for gzip: 0-9)
        """
        with h5py.File(self.filename, self.mode) as f:
            # Create root group
            root = f.create_group('dataset')
            
            # Add metadata
            self._add_metadata(root, dataset)
            
            # Add samples
            samples_group = root.create_group('samples')
            for i, sample in enumerate(dataset.samples):
                self._add_sample(samples_group, sample, i, compression, compression_opts)
            
            # Add aggregated data for easy access
            self._add_aggregated_data(root, dataset, compression, compression_opts)
    
    def _add_metadata(self, group: h5py.Group, dataset: MLDataset):
        """Add dataset metadata to HDF5 group"""
        # Basic metadata
        group.attrs['dataset_name'] = dataset.dataset_name
        group.attrs['creation_date'] = dataset.creation_date
        group.attrs['description'] = dataset.description
        group.attrs['n_samples'] = dataset.n_samples
        
        # Statistics
        if dataset.statistics:
            stats_group = group.create_group('statistics')
            self._add_dict_to_group(stats_group, dataset.statistics)
        
        # Configuration
        if dataset.config:
            config_group = group.create_group('config')
            self._add_dict_to_group(config_group, dataset.config)
    
    def _add_sample(self, group: h5py.Group, sample: WarpStressPair, index: int,
                   compression: str, compression_opts: int):
        """Add individual sample to HDF5 group"""
        sample_group = group.create_group(f'sample_{index:06d}')
        
        # Sample metadata
        sample_group.attrs['sample_id'] = sample.sample_id
        sample_group.attrs['warp_magnitude'] = sample.warp_magnitude
        sample_group.attrs['max_stress'] = sample.max_stress
        sample_group.attrs['simulation_time'] = sample.simulation_time
        
        # Manufacturing parameters
        if sample.manufacturing_params:
            params_group = sample_group.create_group('manufacturing_params')
            self._add_dict_to_group(params_group, sample.manufacturing_params)
        
        # Mesh info
        if sample.mesh_info:
            mesh_group = sample_group.create_group('mesh_info')
            self._add_dict_to_group(mesh_group, sample.mesh_info)
        
        # Warp data
        warp_group = sample_group.create_group('warp')
        warp_group.create_dataset('points', data=sample.warp_points, 
                                compression=compression, compression_opts=compression_opts)
        warp_group.create_dataset('displacements', data=sample.warp_displacements,
                                compression=compression, compression_opts=compression_opts)
        
        if sample.warp_height_map is not None:
            warp_group.create_dataset('height_map', data=sample.warp_height_map,
                                    compression=compression, compression_opts=compression_opts)
            warp_group.create_dataset('x_coords', data=sample.warp_x_coords,
                                    compression=compression, compression_opts=compression_opts)
            warp_group.create_dataset('y_coords', data=sample.warp_y_coords,
                                    compression=compression, compression_opts=compression_opts)
        
        # Stress data
        stress_group = sample_group.create_group('stress')
        stress_group.create_dataset('tensor', data=sample.stress_tensor,
                                  compression=compression, compression_opts=compression_opts)
        stress_group.create_dataset('von_mises', data=sample.stress_von_mises,
                                  compression=compression, compression_opts=compression_opts)
        stress_group.create_dataset('principal', data=sample.stress_principal,
                                  compression=compression, compression_opts=compression_opts)
        stress_group.create_dataset('element_centers', data=sample.stress_element_centers,
                                  compression=compression, compression_opts=compression_opts)
        
        if sample.surface_stress_map is not None:
            stress_group.create_dataset('surface_map', data=sample.surface_stress_map,
                                      compression=compression, compression_opts=compression_opts)
            stress_group.create_dataset('surface_x_coords', data=sample.surface_x_coords,
                                      compression=compression, compression_opts=compression_opts)
            stress_group.create_dataset('surface_y_coords', data=sample.surface_y_coords,
                                      compression=compression, compression_opts=compression_opts)
    
    def _add_aggregated_data(self, group: h5py.Group, dataset: MLDataset,
                           compression: str, compression_opts: int):
        """Add aggregated data for easy ML training access"""
        aggregated_group = group.create_group('aggregated')
        
        # Feature matrices
        features_group = aggregated_group.create_group('features')
        
        # Height map features
        height_maps = []
        for sample in dataset.samples:
            if sample.warp_height_map is not None:
                height_maps.append(sample.warp_height_map.flatten())
        
        if height_maps:
            height_maps_array = np.array(height_maps)
            features_group.create_dataset('height_maps', data=height_maps_array,
                                        compression=compression, compression_opts=compression_opts)
        
        # Point cloud features
        point_clouds = []
        for sample in dataset.samples:
            point_clouds.append(sample.warp_points.flatten())
        
        point_clouds_array = np.array(point_clouds)
        features_group.create_dataset('point_clouds', data=point_clouds_array,
                                    compression=compression, compression_opts=compression_opts)
        
        # Target matrices
        targets_group = aggregated_group.create_group('targets')
        
        # Surface stress maps
        stress_maps = []
        for sample in dataset.samples:
            if sample.surface_stress_map is not None:
                stress_maps.append(sample.surface_stress_map.flatten())
        
        if stress_maps:
            stress_maps_array = np.array(stress_maps)
            targets_group.create_dataset('surface_stress_maps', data=stress_maps_array,
                                       compression=compression, compression_opts=compression_opts)
        
        # Element stress targets
        element_stresses = []
        for sample in dataset.samples:
            element_stresses.append(sample.stress_von_mises)
        
        element_stresses_array = np.array(element_stresses)
        targets_group.create_dataset('element_stresses', data=element_stresses_array,
                                   compression=compression, compression_opts=compression_opts)
        
        # Parameter matrix
        parameters = []
        for sample in dataset.samples:
            param_values = list(sample.manufacturing_params.values())
            parameters.append(param_values)
        
        parameters_array = np.array(parameters)
        aggregated_group.create_dataset('parameters', data=parameters_array,
                                      compression=compression, compression_opts=compression_opts)
        
        # Sample metadata
        metadata = []
        for sample in dataset.samples:
            metadata.append([
                sample.warp_magnitude,
                sample.max_stress,
                sample.simulation_time
            ])
        
        metadata_array = np.array(metadata)
        aggregated_group.create_dataset('metadata', data=metadata_array,
                                      compression=compression, compression_opts=compression_opts)
    
    def _add_dict_to_group(self, group: h5py.Group, data: Dict[str, Any]):
        """Add dictionary data to HDF5 group"""
        for key, value in data.items():
            if isinstance(value, dict):
                sub_group = group.create_group(key)
                self._add_dict_to_group(sub_group, value)
            elif isinstance(value, (int, float, str)):
                group.attrs[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                group.create_dataset(key, data=np.array(value))
            else:
                # Convert to string for complex types
                group.attrs[key] = str(value)
    
    def load_dataset(self) -> MLDataset:
        """Load dataset from HDF5 file"""
        with h5py.File(self.filename, 'r') as f:
            root = f['dataset']
            
            # Load metadata
            dataset_name = root.attrs['dataset_name']
            creation_date = root.attrs['creation_date']
            description = root.attrs['description']
            n_samples = root.attrs['n_samples']
            
            # Create dataset
            dataset = MLDataset(
                dataset_name=dataset_name,
                creation_date=creation_date,
                description=description,
                n_samples=n_samples
            )
            
            # Load statistics
            if 'statistics' in root:
                dataset.statistics = self._load_dict_from_group(root['statistics'])
            
            # Load configuration
            if 'config' in root:
                dataset.config = self._load_dict_from_group(root['config'])
            
            # Load samples
            samples_group = root['samples']
            for sample_name in samples_group.keys():
                sample = self._load_sample(samples_group[sample_name])
                dataset.add_sample(sample)
            
            return dataset
    
    def _load_sample(self, sample_group: h5py.Group) -> WarpStressPair:
        """Load individual sample from HDF5 group"""
        # Load metadata
        sample_id = sample_group.attrs['sample_id']
        warp_magnitude = sample_group.attrs['warp_magnitude']
        max_stress = sample_group.attrs['max_stress']
        simulation_time = sample_group.attrs['simulation_time']
        
        # Load manufacturing parameters
        manufacturing_params = {}
        if 'manufacturing_params' in sample_group:
            manufacturing_params = self._load_dict_from_group(sample_group['manufacturing_params'])
        
        # Load mesh info
        mesh_info = {}
        if 'mesh_info' in sample_group:
            mesh_info = self._load_dict_from_group(sample_group['mesh_info'])
        
        # Load warp data
        warp_group = sample_group['warp']
        warp_points = warp_group['points'][:]
        warp_displacements = warp_group['displacements'][:]
        
        warp_height_map = None
        warp_x_coords = None
        warp_y_coords = None
        if 'height_map' in warp_group:
            warp_height_map = warp_group['height_map'][:]
            warp_x_coords = warp_group['x_coords'][:]
            warp_y_coords = warp_group['y_coords'][:]
        
        # Load stress data
        stress_group = sample_group['stress']
        stress_tensor = stress_group['tensor'][:]
        stress_von_mises = stress_group['von_mises'][:]
        stress_principal = stress_group['principal'][:]
        stress_element_centers = stress_group['element_centers'][:]
        
        surface_stress_map = None
        surface_x_coords = None
        surface_y_coords = None
        if 'surface_map' in stress_group:
            surface_stress_map = stress_group['surface_map'][:]
            surface_x_coords = stress_group['surface_x_coords'][:]
            surface_y_coords = stress_group['surface_y_coords'][:]
        
        return WarpStressPair(
            sample_id=sample_id,
            manufacturing_params=manufacturing_params,
            warp_points=warp_points,
            warp_displacements=warp_displacements,
            warp_height_map=warp_height_map,
            warp_x_coords=warp_x_coords,
            warp_y_coords=warp_y_coords,
            stress_tensor=stress_tensor,
            stress_von_mises=stress_von_mises,
            stress_principal=stress_principal,
            stress_element_centers=stress_element_centers,
            surface_stress_map=surface_stress_map,
            surface_x_coords=surface_x_coords,
            surface_y_coords=surface_y_coords,
            warp_magnitude=warp_magnitude,
            max_stress=max_stress,
            simulation_time=simulation_time,
            mesh_info=mesh_info
        )
    
    def _load_dict_from_group(self, group: h5py.Group) -> Dict[str, Any]:
        """Load dictionary from HDF5 group"""
        data = {}
        
        # Load attributes
        for key, value in group.attrs.items():
            data[key] = value
        
        # Load datasets
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = self._load_dict_from_group(group[key])
            else:
                data[key] = group[key][:]
        
        return data


if __name__ == "__main__":
    # Example usage
    print("HDF5Exporter module loaded successfully")
    print("Use to export and load ML datasets in HDF5 format")