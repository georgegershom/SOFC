"""
Dataset Manager for SOFC Warp-Stress Paired Data

This module manages the storage, organization, and export of paired warp-stress datasets
for machine learning applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import h5py
import json
import pickle
from datetime import datetime
import uuid
import shutil
from tqdm import tqdm

from ..extraction.warp_extractor import WarpFieldData
from ..extraction.stress_extractor import StressFieldData


@dataclass
class DatasetSample:
    """Container for a single paired warp-stress sample."""
    sample_id: str
    doe_parameters: Dict[str, Any]
    warp_data: WarpFieldData
    stress_data: StressFieldData
    simulation_metadata: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DatasetMetadata:
    """Metadata for the complete dataset."""
    dataset_id: str
    dataset_name: str
    description: str
    n_samples: int
    creation_date: str
    version: str
    
    # Parameter space information
    parameter_ranges: Dict[str, Any]
    doe_method: str
    
    # Data format information
    warp_format: Dict[str, Any]
    stress_format: Dict[str, Any]
    
    # Statistics
    statistics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.creation_date is None:
            self.creation_date = datetime.now().isoformat()


class SOFCDatasetManager:
    """Manages SOFC warp-stress paired datasets."""
    
    def __init__(self, dataset_root: Union[str, Path]):
        """Initialize dataset manager.
        
        Args:
            dataset_root: Root directory for dataset storage
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Dataset structure
        self.samples = []
        self.metadata = None
        
        # File paths
        self.metadata_file = self.dataset_root / "metadata.json"
        self.samples_dir = self.dataset_root / "samples"
        self.processed_dir = self.dataset_root / "processed"
        self.exports_dir = self.dataset_root / "exports"
        
        # Create directories
        self.samples_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
    
    def create_dataset(self, dataset_name: str, description: str, 
                      parameter_ranges: Dict[str, Any], doe_method: str = "latin_hypercube") -> str:
        """Create a new dataset.
        
        Returns:
            Dataset ID
        """
        dataset_id = str(uuid.uuid4())
        
        self.metadata = DatasetMetadata(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            description=description,
            n_samples=0,
            creation_date=datetime.now().isoformat(),
            version="1.0",
            parameter_ranges=parameter_ranges,
            doe_method=doe_method,
            warp_format={
                "grid_resolution": None,
                "coordinate_system": "cartesian",
                "units": "meters"
            },
            stress_format={
                "voxel_resolution": None,
                "stress_components": ["xx", "yy", "zz", "xy", "xz", "yz"],
                "units": "pascals"
            }
        )
        
        self._save_metadata()
        return dataset_id
    
    def add_sample(self, doe_parameters: Dict[str, Any], 
                  warp_data: WarpFieldData, 
                  stress_data: StressFieldData,
                  simulation_metadata: Dict[str, Any]) -> str:
        """Add a new sample to the dataset.
        
        Returns:
            Sample ID
        """
        sample_id = f"sample_{len(self.samples):06d}"
        
        sample = DatasetSample(
            sample_id=sample_id,
            doe_parameters=doe_parameters,
            warp_data=warp_data,
            stress_data=stress_data,
            simulation_metadata=simulation_metadata
        )
        
        self.samples.append(sample)
        
        # Save sample to disk
        self._save_sample(sample)
        
        # Update metadata
        if self.metadata:
            self.metadata.n_samples = len(self.samples)
            
            # Update format information
            if warp_data.grid_resolution:
                self.metadata.warp_format["grid_resolution"] = warp_data.grid_resolution
            if stress_data.voxel_resolution:
                self.metadata.stress_format["voxel_resolution"] = stress_data.voxel_resolution
            
            self._save_metadata()
        
        return sample_id
    
    def _save_sample(self, sample: DatasetSample):
        """Save individual sample to disk."""
        sample_dir = self.samples_dir / sample.sample_id
        sample_dir.mkdir(exist_ok=True)
        
        # Save DOE parameters
        with open(sample_dir / "doe_parameters.json", 'w') as f:
            json.dump(sample.doe_parameters, f, indent=2)
        
        # Save simulation metadata
        with open(sample_dir / "simulation_metadata.json", 'w') as f:
            json.dump(sample.simulation_metadata, f, indent=2)
        
        # Save warp data
        self._save_warp_data(sample.warp_data, sample_dir / "warp_data.h5")
        
        # Save stress data
        self._save_stress_data(sample.stress_data, sample_dir / "stress_data.h5")
        
        # Save sample metadata
        sample_metadata = {
            "sample_id": sample.sample_id,
            "timestamp": sample.timestamp,
            "files": {
                "doe_parameters": "doe_parameters.json",
                "simulation_metadata": "simulation_metadata.json",
                "warp_data": "warp_data.h5",
                "stress_data": "stress_data.h5"
            }
        }
        
        with open(sample_dir / "sample_metadata.json", 'w') as f:
            json.dump(sample_metadata, f, indent=2)
    
    def _save_warp_data(self, warp_data: WarpFieldData, filepath: Path):
        """Save warp data to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            # Original 3D data
            f.create_dataset('original_coordinates', data=warp_data.original_coordinates)
            f.create_dataset('deformed_coordinates', data=warp_data.deformed_coordinates)
            f.create_dataset('displacement_field', data=warp_data.displacement_field)
            
            # Surface data
            f.create_dataset('top_surface_original', data=warp_data.top_surface_original)
            f.create_dataset('top_surface_deformed', data=warp_data.top_surface_deformed)
            f.create_dataset('bottom_surface_original', data=warp_data.bottom_surface_original)
            f.create_dataset('bottom_surface_deformed', data=warp_data.bottom_surface_deformed)
            
            # 2.5D height maps
            if warp_data.x_grid is not None:
                f.create_dataset('x_grid', data=warp_data.x_grid)
                f.create_dataset('y_grid', data=warp_data.y_grid)
                f.create_dataset('top_height_map', data=warp_data.top_height_map)
                f.create_dataset('bottom_height_map', data=warp_data.bottom_height_map)
                f.create_dataset('warp_height_map', data=warp_data.warp_height_map)
            
            # Metadata
            f.attrs['plate_dimensions'] = warp_data.plate_dimensions if warp_data.plate_dimensions else [0, 0]
            f.attrs['grid_resolution'] = warp_data.grid_resolution if warp_data.grid_resolution else [0, 0]
            f.attrs['max_warp'] = warp_data.max_warp
            f.attrs['rms_warp'] = warp_data.rms_warp
    
    def _save_stress_data(self, stress_data: StressFieldData, filepath: Path):
        """Save stress data to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            # Element and nodal data
            f.create_dataset('element_coordinates', data=stress_data.element_coordinates)
            f.create_dataset('stress_tensors', data=stress_data.stress_tensors)
            f.create_dataset('nodal_coordinates', data=stress_data.nodal_coordinates)
            f.create_dataset('nodal_stress_tensors', data=stress_data.nodal_stress_tensors)
            
            # Voxelized data
            if stress_data.voxelized_stress is not None:
                f.create_dataset('voxel_grid_x', data=stress_data.voxel_grid_x)
                f.create_dataset('voxel_grid_y', data=stress_data.voxel_grid_y)
                f.create_dataset('voxel_grid_z', data=stress_data.voxel_grid_z)
                f.create_dataset('voxelized_stress', data=stress_data.voxelized_stress)
            
            # Derived quantities
            f.create_dataset('von_mises_stress', data=stress_data.von_mises_stress)
            f.create_dataset('hydrostatic_stress', data=stress_data.hydrostatic_stress)
            f.create_dataset('principal_stresses', data=stress_data.principal_stresses)
            f.create_dataset('max_shear_stress', data=stress_data.max_shear_stress)
            
            # Metadata
            f.attrs['stress_units'] = stress_data.stress_units
            f.attrs['max_von_mises'] = stress_data.max_von_mises
            f.attrs['max_principal_stress'] = stress_data.max_principal_stress
            f.attrs['voxel_resolution'] = stress_data.voxel_resolution if stress_data.voxel_resolution else [0, 0, 0]
            
            if stress_data.volume_averaged_stress is not None:
                f.create_dataset('volume_averaged_stress', data=stress_data.volume_averaged_stress)
    
    def _save_metadata(self):
        """Save dataset metadata to file."""
        if self.metadata:
            with open(self.metadata_file, 'w') as f:
                json.dump(asdict(self.metadata), f, indent=2, default=str)
    
    def load_dataset(self, dataset_path: Optional[Union[str, Path]] = None):
        """Load existing dataset from disk."""
        if dataset_path:
            self.dataset_root = Path(dataset_path)
            self.metadata_file = self.dataset_root / "metadata.json"
            self.samples_dir = self.dataset_root / "samples"
        
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                self.metadata = DatasetMetadata(**metadata_dict)
        
        # Load samples (metadata only, data loaded on demand)
        self.samples = []
        if self.samples_dir.exists():
            for sample_dir in sorted(self.samples_dir.iterdir()):
                if sample_dir.is_dir():
                    sample_metadata_file = sample_dir / "sample_metadata.json"
                    if sample_metadata_file.exists():
                        with open(sample_metadata_file, 'r') as f:
                            sample_metadata = json.load(f)
                        
                        # Create placeholder sample (data loaded on demand)
                        sample = DatasetSample(
                            sample_id=sample_metadata["sample_id"],
                            doe_parameters={},  # Will be loaded on demand
                            warp_data=None,     # Will be loaded on demand
                            stress_data=None,   # Will be loaded on demand
                            simulation_metadata={},  # Will be loaded on demand
                            timestamp=sample_metadata["timestamp"]
                        )
                        self.samples.append(sample)
    
    def get_sample(self, sample_id: str) -> DatasetSample:
        """Load complete sample data from disk."""
        sample_dir = self.samples_dir / sample_id
        
        if not sample_dir.exists():
            raise ValueError(f"Sample {sample_id} not found")
        
        # Load DOE parameters
        with open(sample_dir / "doe_parameters.json", 'r') as f:
            doe_parameters = json.load(f)
        
        # Load simulation metadata
        with open(sample_dir / "simulation_metadata.json", 'r') as f:
            simulation_metadata = json.load(f)
        
        # Load warp data
        warp_data = self._load_warp_data(sample_dir / "warp_data.h5")
        
        # Load stress data
        stress_data = self._load_stress_data(sample_dir / "stress_data.h5")
        
        # Load sample metadata
        with open(sample_dir / "sample_metadata.json", 'r') as f:
            sample_metadata = json.load(f)
        
        return DatasetSample(
            sample_id=sample_id,
            doe_parameters=doe_parameters,
            warp_data=warp_data,
            stress_data=stress_data,
            simulation_metadata=simulation_metadata,
            timestamp=sample_metadata["timestamp"]
        )
    
    def _load_warp_data(self, filepath: Path) -> WarpFieldData:
        """Load warp data from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            warp_data = WarpFieldData(
                original_coordinates=f['original_coordinates'][:],
                deformed_coordinates=f['deformed_coordinates'][:],
                displacement_field=f['displacement_field'][:],
                top_surface_original=f['top_surface_original'][:],
                top_surface_deformed=f['top_surface_deformed'][:],
                bottom_surface_original=f['bottom_surface_original'][:],
                bottom_surface_deformed=f['bottom_surface_deformed'][:],
                plate_dimensions=tuple(f.attrs['plate_dimensions']),
                grid_resolution=tuple(f.attrs['grid_resolution']),
                max_warp=f.attrs['max_warp'],
                rms_warp=f.attrs['rms_warp']
            )
            
            # Load height maps if available
            if 'x_grid' in f:
                warp_data.x_grid = f['x_grid'][:]
                warp_data.y_grid = f['y_grid'][:]
                warp_data.top_height_map = f['top_height_map'][:]
                warp_data.bottom_height_map = f['bottom_height_map'][:]
                warp_data.warp_height_map = f['warp_height_map'][:]
        
        return warp_data
    
    def _load_stress_data(self, filepath: Path) -> StressFieldData:
        """Load stress data from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            stress_data = StressFieldData(
                element_coordinates=f['element_coordinates'][:],
                stress_tensors=f['stress_tensors'][:],
                nodal_coordinates=f['nodal_coordinates'][:],
                nodal_stress_tensors=f['nodal_stress_tensors'][:],
                von_mises_stress=f['von_mises_stress'][:],
                hydrostatic_stress=f['hydrostatic_stress'][:],
                principal_stresses=f['principal_stresses'][:],
                max_shear_stress=f['max_shear_stress'][:],
                stress_units=f.attrs['stress_units'].decode() if isinstance(f.attrs['stress_units'], bytes) else f.attrs['stress_units'],
                max_von_mises=f.attrs['max_von_mises'],
                max_principal_stress=f.attrs['max_principal_stress'],
                voxel_resolution=tuple(f.attrs['voxel_resolution'])
            )
            
            # Load voxelized data if available
            if 'voxelized_stress' in f:
                stress_data.voxel_grid_x = f['voxel_grid_x'][:]
                stress_data.voxel_grid_y = f['voxel_grid_y'][:]
                stress_data.voxel_grid_z = f['voxel_grid_z'][:]
                stress_data.voxelized_stress = f['voxelized_stress'][:]
            
            # Load volume-averaged stress if available
            if 'volume_averaged_stress' in f:
                stress_data.volume_averaged_stress = f['volume_averaged_stress'][:]
        
        return stress_data
    
    def calculate_dataset_statistics(self):
        """Calculate statistics for the entire dataset."""
        if not self.samples:
            return
        
        # Initialize statistics containers
        warp_stats = []
        stress_stats = []
        parameter_stats = {}
        
        print("Calculating dataset statistics...")
        for sample in tqdm(self.samples):
            # Load sample data
            full_sample = self.get_sample(sample.sample_id)
            
            # Warp statistics
            warp_stats.append({
                'max_warp': full_sample.warp_data.max_warp,
                'rms_warp': full_sample.warp_data.rms_warp
            })
            
            # Stress statistics
            stress_stats.append({
                'max_von_mises': full_sample.stress_data.max_von_mises,
                'max_principal_stress': full_sample.stress_data.max_principal_stress
            })
            
            # Parameter statistics
            for param_name, param_value in full_sample.doe_parameters.items():
                if param_name not in parameter_stats:
                    parameter_stats[param_name] = []
                parameter_stats[param_name].append(param_value)
        
        # Calculate summary statistics
        warp_df = pd.DataFrame(warp_stats)
        stress_df = pd.DataFrame(stress_stats)
        
        statistics = {
            'warp_statistics': {
                'max_warp': {
                    'min': float(warp_df['max_warp'].min()),
                    'max': float(warp_df['max_warp'].max()),
                    'mean': float(warp_df['max_warp'].mean()),
                    'std': float(warp_df['max_warp'].std())
                },
                'rms_warp': {
                    'min': float(warp_df['rms_warp'].min()),
                    'max': float(warp_df['rms_warp'].max()),
                    'mean': float(warp_df['rms_warp'].mean()),
                    'std': float(warp_df['rms_warp'].std())
                }
            },
            'stress_statistics': {
                'max_von_mises': {
                    'min': float(stress_df['max_von_mises'].min()),
                    'max': float(stress_df['max_von_mises'].max()),
                    'mean': float(stress_df['max_von_mises'].mean()),
                    'std': float(stress_df['max_von_mises'].std())
                },
                'max_principal_stress': {
                    'min': float(stress_df['max_principal_stress'].min()),
                    'max': float(stress_df['max_principal_stress'].max()),
                    'mean': float(stress_df['max_principal_stress'].mean()),
                    'std': float(stress_df['max_principal_stress'].std())
                }
            },
            'parameter_statistics': {}
        }
        
        # Parameter statistics
        for param_name, values in parameter_stats.items():
            if isinstance(values[0], (int, float)):
                param_array = np.array(values)
                statistics['parameter_statistics'][param_name] = {
                    'min': float(param_array.min()),
                    'max': float(param_array.max()),
                    'mean': float(param_array.mean()),
                    'std': float(param_array.std())
                }
            else:
                # Categorical parameter
                unique_values, counts = np.unique(values, return_counts=True)
                statistics['parameter_statistics'][param_name] = {
                    'unique_values': unique_values.tolist(),
                    'counts': counts.tolist()
                }
        
        # Update metadata
        if self.metadata:
            self.metadata.statistics = statistics
            self._save_metadata()
        
        return statistics
    
    def export_for_ml(self, export_format: str = "hdf5", 
                     include_raw_data: bool = True,
                     train_split: float = 0.8) -> Dict[str, Path]:
        """Export dataset in ML-ready format.
        
        Args:
            export_format: 'hdf5', 'numpy', or 'tensorflow'
            include_raw_data: Whether to include full resolution data
            train_split: Fraction of data for training
            
        Returns:
            Dictionary of exported file paths
        """
        export_dir = self.exports_dir / f"ml_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(exist_ok=True)
        
        # Split dataset
        n_samples = len(self.samples)
        n_train = int(n_samples * train_split)
        
        train_indices = np.random.choice(n_samples, n_train, replace=False)
        test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
        
        exported_files = {}
        
        if export_format.lower() == "hdf5":
            exported_files = self._export_hdf5_ml(export_dir, train_indices, test_indices, include_raw_data)
        elif export_format.lower() == "numpy":
            exported_files = self._export_numpy_ml(export_dir, train_indices, test_indices, include_raw_data)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Export metadata
        ml_metadata = {
            'dataset_info': asdict(self.metadata) if self.metadata else {},
            'export_info': {
                'export_date': datetime.now().isoformat(),
                'export_format': export_format,
                'include_raw_data': include_raw_data,
                'train_split': train_split,
                'n_train_samples': len(train_indices),
                'n_test_samples': len(test_indices)
            }
        }
        
        metadata_file = export_dir / "ml_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(ml_metadata, f, indent=2, default=str)
        
        exported_files['metadata'] = metadata_file
        
        return exported_files
    
    def _export_hdf5_ml(self, export_dir: Path, train_indices: np.ndarray, 
                       test_indices: np.ndarray, include_raw_data: bool) -> Dict[str, Path]:
        """Export to HDF5 format for ML."""
        
        train_file = export_dir / "train_data.h5"
        test_file = export_dir / "test_data.h5"
        
        # Export training data
        self._write_hdf5_split(train_file, train_indices, include_raw_data)
        
        # Export test data
        self._write_hdf5_split(test_file, test_indices, include_raw_data)
        
        return {
            'train_data': train_file,
            'test_data': test_file
        }
    
    def _write_hdf5_split(self, filepath: Path, indices: np.ndarray, include_raw_data: bool):
        """Write data split to HDF5 file."""
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            inputs_group = f.create_group('inputs')  # Warp data
            targets_group = f.create_group('targets')  # Stress data
            parameters_group = f.create_group('parameters')  # DOE parameters
            
            n_samples = len(indices)
            
            # Initialize arrays for batch processing
            warp_maps = []
            stress_voxels = []
            doe_params_list = []
            
            print(f"Processing {n_samples} samples for {filepath.name}...")
            for i, idx in enumerate(tqdm(indices)):
                sample = self.get_sample(self.samples[idx].sample_id)
                
                # Warp data (input)
                if sample.warp_data.warp_height_map is not None:
                    warp_maps.append(sample.warp_data.warp_height_map)
                
                # Stress data (target)
                if sample.stress_data.voxelized_stress is not None:
                    stress_voxels.append(sample.stress_data.voxelized_stress)
                
                # DOE parameters
                doe_params_list.append(sample.doe_parameters)
            
            # Save warp data
            if warp_maps:
                warp_array = np.array(warp_maps)
                inputs_group.create_dataset('warp_height_maps', data=warp_array)
            
            # Save stress data
            if stress_voxels:
                stress_array = np.array(stress_voxels)
                targets_group.create_dataset('stress_voxels', data=stress_array)
            
            # Save DOE parameters
            if doe_params_list:
                # Convert to structured array
                param_names = list(doe_params_list[0].keys())
                param_arrays = {}
                
                for param_name in param_names:
                    values = [params[param_name] for params in doe_params_list]
                    if isinstance(values[0], str):
                        # Categorical parameter
                        unique_vals = list(set(values))
                        encoded_values = [unique_vals.index(val) for val in values]
                        param_arrays[param_name] = np.array(encoded_values)
                        parameters_group.attrs[f'{param_name}_categories'] = unique_vals
                    else:
                        param_arrays[param_name] = np.array(values)
                
                for param_name, param_array in param_arrays.items():
                    parameters_group.create_dataset(param_name, data=param_array)
    
    def _export_numpy_ml(self, export_dir: Path, train_indices: np.ndarray, 
                        test_indices: np.ndarray, include_raw_data: bool) -> Dict[str, Path]:
        """Export to NumPy format for ML."""
        
        train_file = export_dir / "train_data.npz"
        test_file = export_dir / "test_data.npz"
        
        # Export training data
        self._write_numpy_split(train_file, train_indices, include_raw_data)
        
        # Export test data
        self._write_numpy_split(test_file, test_indices, include_raw_data)
        
        return {
            'train_data': train_file,
            'test_data': test_file
        }
    
    def _write_numpy_split(self, filepath: Path, indices: np.ndarray, include_raw_data: bool):
        """Write data split to NumPy file."""
        
        # Similar to HDF5 but save as .npz
        warp_maps = []
        stress_voxels = []
        doe_params_list = []
        
        print(f"Processing {len(indices)} samples for {filepath.name}...")
        for idx in tqdm(indices):
            sample = self.get_sample(self.samples[idx].sample_id)
            
            if sample.warp_data.warp_height_map is not None:
                warp_maps.append(sample.warp_data.warp_height_map)
            
            if sample.stress_data.voxelized_stress is not None:
                stress_voxels.append(sample.stress_data.voxelized_stress)
            
            doe_params_list.append(sample.doe_parameters)
        
        # Prepare data dictionary
        save_dict = {}
        
        if warp_maps:
            save_dict['warp_height_maps'] = np.array(warp_maps)
        
        if stress_voxels:
            save_dict['stress_voxels'] = np.array(stress_voxels)
        
        # Save parameters
        if doe_params_list:
            param_names = list(doe_params_list[0].keys())
            for param_name in param_names:
                values = [params[param_name] for params in doe_params_list]
                if isinstance(values[0], str):
                    # Encode categorical parameters
                    unique_vals = list(set(values))
                    encoded_values = [unique_vals.index(val) for val in values]
                    save_dict[param_name] = np.array(encoded_values)
                    save_dict[f'{param_name}_categories'] = np.array(unique_vals)
                else:
                    save_dict[param_name] = np.array(values)
        
        np.savez_compressed(filepath, **save_dict)
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive dataset summary."""
        summary = {
            'metadata': asdict(self.metadata) if self.metadata else {},
            'n_samples': len(self.samples),
            'storage_info': {
                'dataset_root': str(self.dataset_root),
                'total_size_mb': self._calculate_dataset_size(),
                'samples_dir': str(self.samples_dir),
                'processed_dir': str(self.processed_dir),
                'exports_dir': str(self.exports_dir)
            }
        }
        
        if self.metadata and self.metadata.statistics:
            summary['statistics'] = self.metadata.statistics
        
        return summary
    
    def _calculate_dataset_size(self) -> float:
        """Calculate total dataset size in MB."""
        total_size = 0
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                filepath = Path(root) / file
                total_size += filepath.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB


def create_dataset_manager(dataset_root: Union[str, Path]) -> SOFCDatasetManager:
    """Factory function to create dataset manager."""
    return SOFCDatasetManager(dataset_root)


if __name__ == "__main__":
    # Test dataset manager
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = create_dataset_manager(temp_dir)
        
        # Create test dataset
        dataset_id = manager.create_dataset(
            dataset_name="Test SOFC Dataset",
            description="Test dataset for development",
            parameter_ranges={"test_param": {"min": 0, "max": 1}},
            doe_method="latin_hypercube"
        )
        
        print(f"Created test dataset: {dataset_id}")
        print(f"Dataset summary: {manager.get_dataset_summary()}")