"""
ML Dataset Structure

Defines the data structures for ML training datasets containing paired
warp and stress field data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import h5py
import json
from datetime import datetime


@dataclass
class WarpStressPair:
    """Container for a single warp-stress pair"""
    # Sample metadata
    sample_id: str
    manufacturing_params: Dict[str, float]
    
    # Warp field data
    warp_points: np.ndarray  # [n_points, 3] - deformed coordinates
    warp_displacements: np.ndarray  # [n_points, 3] - displacement vectors
    warp_height_map: Optional[np.ndarray] = None  # [ny, nx] - height map
    warp_x_coords: Optional[np.ndarray] = None  # [nx] - x coordinates
    warp_y_coords: Optional[np.ndarray] = None  # [ny] - y coordinates
    
    # Stress field data
    stress_tensor: np.ndarray  # [n_elements, 6] - stress tensor components
    stress_von_mises: np.ndarray  # [n_elements] - Von Mises stress
    stress_principal: np.ndarray  # [n_elements, 3] - principal stresses
    stress_element_centers: np.ndarray  # [n_elements, 3] - element coordinates
    
    # Surface stress data (for easier ML training)
    surface_stress_map: Optional[np.ndarray] = None  # [ny, nx] - surface stress
    surface_x_coords: Optional[np.ndarray] = None  # [nx] - x coordinates
    surface_y_coords: Optional[np.ndarray] = None  # [ny] - y coordinates
    
    # Metadata
    warp_magnitude: float = 0.0
    max_stress: float = 0.0
    simulation_time: float = 0.0
    mesh_info: Dict[str, int] = field(default_factory=dict)
    
    def get_warp_features(self, feature_type: str = 'height_map') -> np.ndarray:
        """Extract features for ML training"""
        if feature_type == 'height_map' and self.warp_height_map is not None:
            return self.warp_height_map.flatten()
        elif feature_type == 'point_cloud':
            return self.warp_points.flatten()
        elif feature_type == 'displacements':
            return self.warp_displacements.flatten()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def get_stress_targets(self, target_type: str = 'surface_map') -> np.ndarray:
        """Extract targets for ML training"""
        if target_type == 'surface_map' and self.surface_stress_map is not None:
            return self.surface_stress_map.flatten()
        elif target_type == 'element_stress':
            return self.stress_von_mises
        elif target_type == 'principal_stress':
            return self.stress_principal.flatten()
        else:
            raise ValueError(f"Unknown target type: {target_type}")


@dataclass
class MLDataset:
    """Container for complete ML dataset"""
    # Dataset metadata
    dataset_name: str
    creation_date: str
    description: str
    n_samples: int
    
    # Data samples
    samples: List[WarpStressPair] = field(default_factory=list)
    
    # Dataset statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    def add_sample(self, sample: WarpStressPair):
        """Add a sample to the dataset"""
        self.samples.append(sample)
        self.n_samples = len(self.samples)
    
    def get_feature_matrix(self, feature_type: str = 'height_map') -> np.ndarray:
        """Get feature matrix for all samples"""
        features = []
        for sample in self.samples:
            features.append(sample.get_warp_features(feature_type))
        
        return np.array(features)
    
    def get_target_matrix(self, target_type: str = 'surface_map') -> np.ndarray:
        """Get target matrix for all samples"""
        targets = []
        for sample in self.samples:
            targets.append(sample.get_stress_targets(target_type))
        
        return np.array(targets)
    
    def get_parameter_matrix(self) -> np.ndarray:
        """Get manufacturing parameter matrix"""
        params = []
        for sample in self.samples:
            param_values = list(sample.manufacturing_params.values())
            params.append(param_values)
        
        return np.array(params)
    
    def calculate_statistics(self):
        """Calculate dataset statistics"""
        if not self.samples:
            return
        
        # Warp statistics
        warp_magnitudes = [sample.warp_magnitude for sample in self.samples]
        stress_values = [sample.max_stress for sample in self.samples]
        
        self.statistics = {
            'warp_magnitude': {
                'mean': float(np.mean(warp_magnitudes)),
                'std': float(np.std(warp_magnitudes)),
                'min': float(np.min(warp_magnitudes)),
                'max': float(np.max(warp_magnitudes))
            },
            'max_stress': {
                'mean': float(np.mean(stress_values)),
                'std': float(np.std(stress_values)),
                'min': float(np.min(stress_values)),
                'max': float(np.max(stress_values))
            },
            'parameter_ranges': self._calculate_parameter_ranges()
        }
    
    def _calculate_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """Calculate parameter ranges across all samples"""
        if not self.samples:
            return {}
        
        # Get all parameter names
        param_names = list(self.samples[0].manufacturing_params.keys())
        param_ranges = {}
        
        for param_name in param_names:
            values = [sample.manufacturing_params[param_name] for sample in self.samples]
            param_ranges[param_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return param_ranges
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_seed: int = 42) -> Tuple['MLDataset', 'MLDataset', 'MLDataset']:
        """Split dataset into train/validation/test sets"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        np.random.seed(random_seed)
        n_samples = len(self.samples)
        indices = np.random.permutation(n_samples)
        
        n_train = int(train_ratio * n_samples)
        n_val = int(val_ratio * n_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create split datasets
        train_dataset = MLDataset(
            dataset_name=f"{self.dataset_name}_train",
            creation_date=datetime.now().isoformat(),
            description=f"Training set from {self.dataset_name}",
            n_samples=len(train_indices)
        )
        
        val_dataset = MLDataset(
            dataset_name=f"{self.dataset_name}_val",
            creation_date=datetime.now().isoformat(),
            description=f"Validation set from {self.dataset_name}",
            n_samples=len(val_indices)
        )
        
        test_dataset = MLDataset(
            dataset_name=f"{self.dataset_name}_test",
            creation_date=datetime.now().isoformat(),
            description=f"Test set from {self.dataset_name}",
            n_samples=len(test_indices)
        )
        
        # Assign samples
        for idx in train_indices:
            train_dataset.add_sample(self.samples[idx])
        for idx in val_indices:
            val_dataset.add_sample(self.samples[idx])
        for idx in test_indices:
            test_dataset.add_sample(self.samples[idx])
        
        # Calculate statistics for each split
        train_dataset.calculate_statistics()
        val_dataset.calculate_statistics()
        test_dataset.calculate_statistics()
        
        return train_dataset, val_dataset, test_dataset
    
    def export_summary(self, filename: str):
        """Export dataset summary to JSON file"""
        summary = {
            'dataset_name': self.dataset_name,
            'creation_date': self.creation_date,
            'description': self.description,
            'n_samples': self.n_samples,
            'statistics': self.statistics,
            'config': self.config,
            'sample_info': {
                'warp_magnitude_range': [min(s.warp_magnitude for s in self.samples),
                                       max(s.warp_magnitude for s in self.samples)],
                'max_stress_range': [min(s.max_stress for s in self.samples),
                                   max(s.max_stress for s in self.samples)],
                'simulation_time_total': sum(s.simulation_time for s in self.samples)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Dataset summary exported to {filename}")
    
    def get_sample_by_id(self, sample_id: str) -> Optional[WarpStressPair]:
        """Get sample by ID"""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
    
    def filter_samples(self, filter_func) -> 'MLDataset':
        """Filter samples based on a function"""
        filtered_samples = [sample for sample in self.samples if filter_func(sample)]
        
        filtered_dataset = MLDataset(
            dataset_name=f"{self.dataset_name}_filtered",
            creation_date=datetime.now().isoformat(),
            description=f"Filtered dataset from {self.dataset_name}",
            n_samples=len(filtered_samples),
            samples=filtered_samples,
            config=self.config
        )
        
        filtered_dataset.calculate_statistics()
        return filtered_dataset


if __name__ == "__main__":
    # Example usage
    dataset = MLDataset(
        dataset_name="sofc_synthetic_dataset",
        creation_date=datetime.now().isoformat(),
        description="Synthetic SOFC dataset for ML training",
        n_samples=0
    )
    
    print("MLDataset module loaded successfully")
    print("Use to create and manage ML training datasets")