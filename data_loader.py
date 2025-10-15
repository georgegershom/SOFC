"""
Data loader utilities for SOFC warp and stress dataset.
Provides convenient functions for ML applications.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class SOFCDataLoader:
    """
    Data loader for SOFC warp and stress dataset.
    Provides convenient methods for loading and preprocessing data for ML.
    """
    
    def __init__(self, dataset_dir: str = "sofc_dataset"):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        dataset_dir: Path to the dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        
        # Load metadata
        metadata_file = self.dataset_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.n_samples = self.metadata['n_samples']
        self.nx = self.metadata['nx']
        self.ny = self.metadata['ny']
        self.nz = self.metadata['nz']
        
        print(f"Loaded dataset with {self.n_samples} samples")
        print(f"Spatial resolution: {self.nx}×{self.ny}×{self.nz}")
    
    def load_sample(self, sample_idx: int) -> Tuple[Dict, Dict, Dict]:
        """
        Load a single sample.
        
        Returns:
        --------
        warp_data: Dictionary with warp field arrays
        stress_data: Dictionary with stress field arrays
        params: Dictionary with manufacturing parameters
        """
        sample = self.metadata['samples'][sample_idx]
        
        # Load warp data
        warp_file = self.dataset_dir / sample['warp_file']
        warp_data = dict(np.load(warp_file))
        
        # Load stress data
        stress_file = self.dataset_dir / sample['stress_file']
        stress_data = dict(np.load(stress_file))
        
        # Load parameters
        params = sample['parameters']
        
        return warp_data, stress_data, params
    
    def get_warp_features(self, sample_idx: int, flatten: bool = True) -> np.ndarray:
        """
        Get warp field as feature vector.
        
        Parameters:
        -----------
        sample_idx: Sample index
        flatten: If True, return flattened array; else return 2D array
        
        Returns:
        --------
        warp_features: Warp field as feature vector
        """
        warp_data, _, _ = self.load_sample(sample_idx)
        
        # Use mean warp as feature
        warp = warp_data['warp_mean']
        
        if flatten:
            return warp.flatten()
        else:
            return warp
    
    def get_stress_targets(self, sample_idx: int, 
                          component: str = 'von_mises',
                          flatten: bool = True) -> np.ndarray:
        """
        Get stress field as target vector.
        
        Parameters:
        -----------
        sample_idx: Sample index
        component: Stress component ('von_mises', 'sigma_xx', 'sigma_yy', etc.)
        flatten: If True, return flattened array; else return 3D array
        
        Returns:
        --------
        stress_targets: Stress field as target vector
        """
        _, stress_data, _ = self.load_sample(sample_idx)
        
        if component == 'von_mises':
            # Compute von Mises stress
            sigma_xx = stress_data['sigma_xx']
            sigma_yy = stress_data['sigma_yy']
            sigma_zz = stress_data['sigma_zz']
            sigma_xy = stress_data['sigma_xy']
            sigma_yz = stress_data['sigma_yz']
            sigma_xz = stress_data['sigma_xz']
            
            stress = np.sqrt(0.5 * (
                (sigma_xx - sigma_yy)**2 +
                (sigma_yy - sigma_zz)**2 +
                (sigma_zz - sigma_xx)**2 +
                6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
            ))
        else:
            stress = stress_data[component]
        
        if flatten:
            return stress.flatten()
        else:
            return stress
    
    def get_parameter_features(self, sample_idx: int, 
                              normalize: bool = True) -> np.ndarray:
        """
        Get manufacturing parameters as feature vector.
        
        Parameters:
        -----------
        sample_idx: Sample index
        normalize: If True, normalize to [0, 1] based on parameter ranges
        
        Returns:
        --------
        param_features: Parameter vector
        """
        _, _, params = self.load_sample(sample_idx)
        
        # Convert to array
        param_array = np.array(list(params.values()))
        
        if normalize:
            # Parameter ranges (same as in generator)
            param_ranges = np.array([
                [1200, 1600],  # peak_sintering_temp
                [1, 10],       # heating_rate
                [1, 10],       # cooling_rate
                [1, 8],        # dwell_time
                [300, 800],    # anode_thickness
                [5, 30],       # electrolyte_thickness
                [20, 80],      # cathode_thickness
                [40, 60],      # anode_ni_content
                [6, 10],       # electrolyte_ysz_dopant
                [20, 45],      # cathode_porosity
                [50, 65],      # green_density
                [1, 5],        # binder_content
                [50, 150],     # plate_length
                [50, 150],     # plate_width
            ])
            
            # Normalize
            param_array = (param_array - param_ranges[:, 0]) / (param_ranges[:, 1] - param_ranges[:, 0])
        
        return param_array
    
    def load_all_samples(self, 
                        feature_type: str = 'warp',
                        target_type: str = 'von_mises',
                        include_params: bool = False,
                        normalize_stress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all samples as ML-ready arrays.
        
        Parameters:
        -----------
        feature_type: Type of features ('warp', 'params', 'both')
        target_type: Type of targets ('von_mises', 'sigma_xx', etc.)
        include_params: If True, include parameters in features
        normalize_stress: If True, normalize stress to [0, 1]
        
        Returns:
        --------
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples, n_targets)
        """
        print(f"Loading all {self.n_samples} samples...")
        
        X_list = []
        y_list = []
        
        for i in range(self.n_samples):
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i+1}/{self.n_samples}...")
            
            # Get features
            if feature_type == 'warp':
                features = self.get_warp_features(i, flatten=True)
            elif feature_type == 'params':
                features = self.get_parameter_features(i, normalize=True)
            elif feature_type == 'both':
                warp_feat = self.get_warp_features(i, flatten=True)
                param_feat = self.get_parameter_features(i, normalize=True)
                features = np.concatenate([warp_feat, param_feat])
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            # Get targets
            targets = self.get_stress_targets(i, component=target_type, flatten=True)
            
            X_list.append(features)
            y_list.append(targets)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Normalize stress if requested
        if normalize_stress:
            y_min = y.min()
            y_max = y.max()
            y = (y - y_min) / (y_max - y_min)
            print(f"Normalized stress range: [{y_min/1e6:.2f}, {y_max/1e6:.2f}] MPa")
        
        print(f"Dataset loaded: X shape = {X.shape}, y shape = {y.shape}")
        
        return X, y
    
    def get_train_test_split(self, 
                            test_size: float = 0.2,
                            random_state: int = 42,
                            **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test split of the dataset.
        
        Parameters:
        -----------
        test_size: Fraction of data for testing
        random_state: Random seed
        **kwargs: Additional arguments passed to load_all_samples
        
        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        # Load all data
        X, y = self.load_all_samples(**kwargs)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def compute_statistics(self) -> Dict:
        """Compute dataset statistics."""
        print("Computing dataset statistics...")
        
        stats = {
            'warp': {'min': [], 'max': [], 'mean': [], 'std': []},
            'stress': {'min': [], 'max': [], 'mean': [], 'std': []},
            'params': {}
        }
        
        # Collect statistics from all samples
        all_params = []
        
        for i in range(self.n_samples):
            warp_data, stress_data, params = self.load_sample(i)
            
            # Warp statistics
            warp = warp_data['warp_mean']
            stats['warp']['min'].append(warp.min())
            stats['warp']['max'].append(warp.max())
            stats['warp']['mean'].append(warp.mean())
            stats['warp']['std'].append(warp.std())
            
            # Stress statistics (von Mises)
            sigma_xx = stress_data['sigma_xx']
            sigma_yy = stress_data['sigma_yy']
            sigma_zz = stress_data['sigma_zz']
            sigma_xy = stress_data['sigma_xy']
            sigma_yz = stress_data['sigma_yz']
            sigma_xz = stress_data['sigma_xz']
            
            sigma_vm = np.sqrt(0.5 * (
                (sigma_xx - sigma_yy)**2 +
                (sigma_yy - sigma_zz)**2 +
                (sigma_zz - sigma_xx)**2 +
                6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
            ))
            
            stats['stress']['min'].append(sigma_vm.min() / 1e6)  # MPa
            stats['stress']['max'].append(sigma_vm.max() / 1e6)
            stats['stress']['mean'].append(sigma_vm.mean() / 1e6)
            stats['stress']['std'].append(sigma_vm.std() / 1e6)
            
            # Parameters
            all_params.append(list(params.values()))
        
        # Aggregate statistics
        for key in ['min', 'max', 'mean', 'std']:
            stats['warp'][key] = {
                'mean': np.mean(stats['warp'][key]),
                'std': np.std(stats['warp'][key]),
                'min': np.min(stats['warp'][key]),
                'max': np.max(stats['warp'][key])
            }
            stats['stress'][key] = {
                'mean': np.mean(stats['stress'][key]),
                'std': np.std(stats['stress'][key]),
                'min': np.min(stats['stress'][key]),
                'max': np.max(stats['stress'][key])
            }
        
        # Parameter statistics
        all_params = np.array(all_params)
        param_names = list(self.metadata['samples'][0]['parameters'].keys())
        
        for i, name in enumerate(param_names):
            stats['params'][name] = {
                'mean': float(all_params[:, i].mean()),
                'std': float(all_params[:, i].std()),
                'min': float(all_params[:, i].min()),
                'max': float(all_params[:, i].max())
            }
        
        return stats


def demo_usage():
    """Demonstrate data loader usage."""
    print("=" * 60)
    print("SOFC Data Loader Demo")
    print("=" * 60)
    
    # Initialize loader
    loader = SOFCDataLoader(dataset_dir="sofc_dataset")
    
    # Load a single sample
    print("\n1. Loading single sample (index 0)...")
    warp_data, stress_data, params = loader.load_sample(0)
    print(f"   Warp shape: {warp_data['warp_mean'].shape}")
    print(f"   Stress shape: {stress_data['sigma_xx'].shape}")
    print(f"   Parameters: {len(params)} values")
    
    # Get feature/target vectors
    print("\n2. Getting feature/target vectors...")
    warp_features = loader.get_warp_features(0)
    stress_targets = loader.get_stress_targets(0, component='von_mises')
    param_features = loader.get_parameter_features(0)
    print(f"   Warp features: {warp_features.shape}")
    print(f"   Stress targets: {stress_targets.shape}")
    print(f"   Param features: {param_features.shape}")
    
    # Load all samples
    print("\n3. Loading all samples...")
    X, y = loader.load_all_samples(feature_type='warp', 
                                   target_type='von_mises',
                                   normalize_stress=True)
    
    # Compute statistics
    print("\n4. Computing dataset statistics...")
    stats = loader.compute_statistics()
    print(f"   Warp range: [{stats['warp']['min']['min']:.4f}, "
          f"{stats['warp']['max']['max']:.4f}] mm")
    print(f"   Stress range: [{stats['stress']['min']['min']:.2f}, "
          f"{stats['stress']['max']['max']:.2f}] MPa")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def example_ml_workflow():
    """Example ML workflow using the dataset."""
    print("\n" + "=" * 60)
    print("Example ML Workflow")
    print("=" * 60)
    
    # Check if sklearn is available
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, r2_score
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return
    
    # Initialize loader
    loader = SOFCDataLoader(dataset_dir="sofc_dataset")
    
    # Load data
    print("\n1. Loading dataset...")
    X, y = loader.load_all_samples(feature_type='warp', 
                                   target_type='von_mises',
                                   normalize_stress=False)
    
    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Dimensionality reduction for features
    print("\n3. Applying PCA for dimensionality reduction...")
    pca_X = PCA(n_components=50)
    X_train_pca = pca_X.fit_transform(X_train)
    X_test_pca = pca_X.transform(X_test)
    print(f"   Reduced features from {X_train.shape[1]} to {X_train_pca.shape[1]}")
    print(f"   Explained variance: {pca_X.explained_variance_ratio_.sum():.3f}")
    
    # Dimensionality reduction for targets
    print("\n4. Applying PCA for stress field compression...")
    pca_y = PCA(n_components=20)
    y_train_pca = pca_y.fit_transform(y_train)
    y_test_pca = pca_y.transform(y_test)
    print(f"   Reduced targets from {y_train.shape[1]} to {y_train_pca.shape[1]}")
    print(f"   Explained variance: {pca_y.explained_variance_ratio_.sum():.3f}")
    
    # Train model
    print("\n5. Training Ridge regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train_pca, y_train_pca)
    
    # Predict
    print("\n6. Making predictions...")
    y_pred_pca = model.predict(X_test_pca)
    
    # Inverse transform to get full stress field
    y_pred = pca_y.inverse_transform(y_pred_pca)
    
    # Evaluate
    print("\n7. Evaluating model...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"   MSE: {mse:.2e} Pa²")
    print(f"   R² Score: {r2:.4f}")
    
    print("\n" + "=" * 60)
    print("ML workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demo
    demo_usage()
    
    # Run example ML workflow
    example_ml_workflow()
