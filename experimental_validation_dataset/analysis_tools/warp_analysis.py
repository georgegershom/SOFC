#!/usr/bin/env python3
"""
Warp Analysis Tool for SOFC Experimental Validation Dataset

This module provides tools for analyzing and processing warp field measurements
from various experimental techniques, including data fusion, noise filtering,
and quality assessment.

Author: Research Team
Date: 2024-01-15
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

class WarpFieldAnalyzer:
    """Analyzes warp field measurements from multiple techniques"""
    
    def __init__(self):
        self.techniques = ['laser_confocal', 'white_light_interferometry', 'structured_light']
    
    def load_measurement_data(self, sample_dir: str) -> Dict:
        """Load all warp measurement data for a sample"""
        measurements = {}
        
        for technique in self.techniques:
            file_path = Path(sample_dir) / 'warp_measurements' / f'{technique}.json'
            if file_path.exists():
                with open(file_path, 'r') as f:
                    measurements[technique] = json.load(f)
        
        return measurements
    
    def process_warp_data(self, measurement_data: Dict) -> Dict:
        """Process and clean warp measurement data"""
        processed_data = {}
        
        for technique, data in measurement_data.items():
            if technique not in self.techniques:
                continue
            
            # Extract coordinates and warp field
            X = np.array(data['coordinates']['X'])
            Y = np.array(data['coordinates']['Y'])
            Z = np.array(data['warp_field'])
            
            # Handle missing data (NaN values)
            if technique == 'structured_light':
                valid_mask = ~np.isnan(Z)
                X = X[valid_mask]
                Y = Y[valid_mask]
                Z = Z[valid_mask]
            
            # Apply noise filtering
            if technique == 'laser_confocal':
                Z = self._apply_gaussian_filter(Z, sigma=2.0)
            elif technique == 'white_light_interferometry':
                Z = self._apply_savitzky_golay_filter(Z, order=3, window=5)
            
            # Remove outliers
            Z = self._remove_outliers(X, Y, Z, threshold=3.0)
            
            processed_data[technique] = {
                'X': X,
                'Y': Y,
                'Z': Z,
                'resolution': data['resolution'],
                'uncertainty': data['uncertainty'],
                'technique': technique
            }
        
        return processed_data
    
    def fuse_measurements(self, processed_data: Dict) -> Dict:
        """Fuse measurements from multiple techniques using weighted average"""
        if len(processed_data) == 1:
            technique = list(processed_data.keys())[0]
            return processed_data[technique]
        
        # Create common grid
        all_X = np.concatenate([data['X'].flatten() for data in processed_data.values()])
        all_Y = np.concatenate([data['Y'].flatten() for data in processed_data.values()])
        
        x_min, x_max = np.min(all_X), np.max(all_X)
        y_min, y_max = np.min(all_Y), np.max(all_Y)
        
        # Use finest resolution available
        min_resolution = min(data['resolution']['lateral'] for data in processed_data.values())
        n_x = int((x_max - x_min) / min_resolution) + 1
        n_y = int((y_max - y_min) / min_resolution) + 1
        
        X_grid = np.linspace(x_min, x_max, n_x)
        Y_grid = np.linspace(y_min, y_max, n_y)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
        
        # Interpolate each measurement to common grid
        interpolated_data = {}
        for technique, data in processed_data.items():
            Z_interp = griddata(
                (data['X'].flatten(), data['Y'].flatten()),
                data['Z'].flatten(),
                (X_mesh, Y_mesh),
                method='linear',
                fill_value=np.nan
            )
            
            # Calculate weights based on uncertainty
            weight = 1.0 / (data['uncertainty'] ** 2)
            
            interpolated_data[technique] = {
                'X': X_mesh,
                'Y': Y_mesh,
                'Z': Z_interp,
                'weight': weight,
                'uncertainty': data['uncertainty']
            }
        
        # Fuse using weighted average
        total_weight = sum(data['weight'] for data in interpolated_data.values())
        Z_fused = np.zeros_like(X_mesh)
        uncertainty_fused = np.zeros_like(X_mesh)
        
        for data in interpolated_data.values():
            valid_mask = ~np.isnan(data['Z'])
            Z_fused[valid_mask] += data['Z'][valid_mask] * data['weight'] / total_weight
            uncertainty_fused[valid_mask] += (data['uncertainty'] * data['weight'] / total_weight) ** 2
        
        uncertainty_fused = np.sqrt(uncertainty_fused)
        
        return {
            'X': X_mesh,
            'Y': Y_mesh,
            'Z': Z_fused,
            'uncertainty': uncertainty_fused,
            'technique': 'fused',
            'resolution': {'lateral': min_resolution, 'vertical': min_resolution}
        }
    
    def calculate_warp_metrics(self, warp_data: Dict) -> Dict:
        """Calculate key warp field metrics"""
        X = warp_data['X']
        Y = warp_data['Y']
        Z = warp_data['Z']
        
        # Remove NaN values for calculations
        valid_mask = ~np.isnan(Z)
        Z_valid = Z[valid_mask]
        
        if len(Z_valid) == 0:
            return {'error': 'No valid data points'}
        
        # Basic statistics
        metrics = {
            'max_warp': float(np.max(Z_valid)),
            'min_warp': float(np.min(Z_valid)),
            'mean_warp': float(np.mean(Z_valid)),
            'std_warp': float(np.std(Z_valid)),
            'rms_warp': float(np.sqrt(np.mean(Z_valid**2))),
            'peak_to_valley': float(np.max(Z_valid) - np.min(Z_valid))
        }
        
        # Calculate curvature metrics
        if X.shape[0] > 2 and X.shape[1] > 2:
            curvature_metrics = self._calculate_curvature_metrics(X, Y, Z)
            metrics.update(curvature_metrics)
        
        # Calculate spatial frequency content
        frequency_metrics = self._calculate_frequency_metrics(X, Y, Z)
        metrics.update(frequency_metrics)
        
        return metrics
    
    def _apply_gaussian_filter(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian filter for noise reduction"""
        return ndimage.gaussian_filter(data, sigma=sigma)
    
    def _apply_savitzky_golay_filter(self, data: np.ndarray, order: int, window: int) -> np.ndarray:
        """Apply Savitzky-Golay filter for noise reduction"""
        from scipy.signal import savgol_filter
        return savgol_filter(data, window, order)
    
    def _remove_outliers(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                        threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using statistical threshold"""
        z_mean = np.mean(Z)
        z_std = np.std(Z)
        
        # Calculate z-scores
        z_scores = np.abs((Z - z_mean) / z_std)
        
        # Replace outliers with interpolated values
        outlier_mask = z_scores > threshold
        if np.any(outlier_mask):
            # Interpolate over outliers
            valid_mask = ~outlier_mask
            if np.any(valid_mask):
                Z_clean = griddata(
                    (X[valid_mask], Y[valid_mask]),
                    Z[valid_mask],
                    (X, Y),
                    method='linear',
                    fill_value=z_mean
                )
                return Z_clean
        
        return Z
    
    def _calculate_curvature_metrics(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict:
        """Calculate curvature-related metrics"""
        # Calculate gradients
        dZ_dx = np.gradient(Z, axis=1)
        dZ_dy = np.gradient(Z, axis=0)
        
        # Calculate second derivatives
        d2Z_dx2 = np.gradient(dZ_dx, axis=1)
        d2Z_dy2 = np.gradient(dZ_dy, axis=0)
        d2Z_dxdy = np.gradient(dZ_dx, axis=0)
        
        # Calculate mean curvature
        H = 0.5 * (d2Z_dx2 + d2Z_dy2)
        
        # Calculate Gaussian curvature
        K = d2Z_dx2 * d2Z_dy2 - d2Z_dxdy**2
        
        # Calculate principal curvatures
        k1 = H + np.sqrt(H**2 - K)
        k2 = H - np.sqrt(H**2 - K)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(k1) | np.isnan(k2))
        k1_valid = k1[valid_mask]
        k2_valid = k2[valid_mask]
        
        if len(k1_valid) == 0:
            return {}
        
        return {
            'mean_curvature': float(np.mean(H[valid_mask])),
            'gaussian_curvature': float(np.mean(K[valid_mask])),
            'max_principal_curvature': float(np.max(k1_valid)),
            'min_principal_curvature': float(np.min(k2_valid)),
            'curvature_std': float(np.std(H[valid_mask]))
        }
    
    def _calculate_frequency_metrics(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict:
        """Calculate spatial frequency content metrics"""
        # Remove NaN values
        valid_mask = ~np.isnan(Z)
        if not np.any(valid_mask):
            return {}
        
        Z_clean = Z.copy()
        Z_clean[~valid_mask] = 0
        
        # Calculate 2D FFT
        fft_2d = np.fft.fft2(Z_clean)
        power_spectrum = np.abs(fft_2d)**2
        
        # Calculate frequency axes
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
        
        freq_x = np.fft.fftfreq(X.shape[1], dx)
        freq_y = np.fft.fftfreq(X.shape[0], dy)
        
        # Calculate dominant frequency
        max_idx = np.unravel_index(np.argmax(power_spectrum[1:, 1:]), power_spectrum[1:, 1:].shape)
        dominant_freq_x = freq_x[max_idx[1] + 1]
        dominant_freq_y = freq_y[max_idx[0] + 1]
        dominant_freq = np.sqrt(dominant_freq_x**2 + dominant_freq_y**2)
        
        # Calculate frequency content metrics
        total_power = np.sum(power_spectrum)
        high_freq_power = np.sum(power_spectrum[power_spectrum > np.percentile(power_spectrum, 90)])
        
        return {
            'dominant_frequency': float(dominant_freq),
            'total_power': float(total_power),
            'high_frequency_ratio': float(high_freq_power / total_power),
            'frequency_std': float(np.std(power_spectrum))
        }
    
    def visualize_warp_field(self, warp_data: Dict, save_path: Optional[str] = None) -> None:
        """Create visualization of warp field"""
        X = warp_data['X']
        Y = warp_data['Y']
        Z = warp_data['Z']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 3D surface plot
        ax1 = axes[0, 0]
        if X.shape[0] > 1 and X.shape[1] > 1:
            im1 = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
            ax1.set_title('Warp Field Contour')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            plt.colorbar(im1, ax=ax1, label='Warp (μm)')
        
        # 3D surface plot
        ax2 = axes[0, 1]
        if X.shape[0] > 1 and X.shape[1] > 1:
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax2.set_title('3D Warp Surface')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
            ax2.set_zlabel('Warp (μm)')
        
        # Histogram of warp values
        ax3 = axes[1, 0]
        valid_Z = Z[~np.isnan(Z)]
        if len(valid_Z) > 0:
            ax3.hist(valid_Z, bins=50, alpha=0.7, edgecolor='black')
            ax3.set_title('Warp Value Distribution')
            ax3.set_xlabel('Warp (μm)')
            ax3.set_ylabel('Frequency')
        
        # Cross-section plots
        ax4 = axes[1, 1]
        if X.shape[0] > 1 and X.shape[1] > 1:
            center_y = X.shape[0] // 2
            center_x = X.shape[1] // 2
            
            ax4.plot(X[center_y, :], Z[center_y, :], 'b-', label='Y centerline')
            ax4.plot(Y[:, center_x], Z[:, center_x], 'r-', label='X centerline')
            ax4.set_title('Cross-section Profiles')
            ax4.set_xlabel('Position (mm)')
            ax4.set_ylabel('Warp (μm)')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def analyze_sample(sample_dir: str) -> Dict:
    """Analyze a complete sample's warp measurements"""
    analyzer = WarpFieldAnalyzer()
    
    # Load measurement data
    measurements = analyzer.load_measurement_data(sample_dir)
    
    if not measurements:
        return {'error': 'No measurement data found'}
    
    # Process data
    processed_data = analyzer.process_warp_data(measurements)
    
    # Fuse measurements
    fused_data = analyzer.fuse_measurements(processed_data)
    
    # Calculate metrics
    metrics = analyzer.calculate_warp_metrics(fused_data)
    
    # Create visualization
    viz_path = Path(sample_dir) / 'warp_analysis_visualization.png'
    analyzer.visualize_warp_field(fused_data, str(viz_path))
    
    # Compile results
    results = {
        'sample_id': Path(sample_dir).name,
        'measurements_available': list(measurements.keys()),
        'processed_measurements': list(processed_data.keys()),
        'fused_data': {
            'technique': fused_data['technique'],
            'resolution': fused_data['resolution'],
            'data_points': int(np.sum(~np.isnan(fused_data['Z'])))
        },
        'metrics': metrics,
        'visualization_path': str(viz_path)
    }
    
    return results

def main():
    """Main function to analyze all samples"""
    dataset_dir = Path('/workspace/experimental_validation_dataset/fabricated_samples')
    
    if not dataset_dir.exists():
        print("Dataset directory not found. Please run measurement_simulation.py first.")
        return
    
    all_results = []
    
    for sample_dir in dataset_dir.iterdir():
        if sample_dir.is_dir():
            print(f"Analyzing {sample_dir.name}...")
            results = analyze_sample(str(sample_dir))
            all_results.append(results)
            
            # Save individual results
            with open(sample_dir / 'warp_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save summary results
    summary = {
        'total_samples': len(all_results),
        'analysis_date': pd.Timestamp.now().isoformat(),
        'results': all_results
    }
    
    with open('/workspace/experimental_validation_dataset/warp_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete. Processed {len(all_results)} samples.")
    print("Results saved to warp_analysis_summary.json")

if __name__ == "__main__":
    main()