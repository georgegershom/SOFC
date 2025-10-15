"""
Warp Field Extraction Module

This module extracts warp fields from FEA simulation results and converts them
to various formats suitable for machine learning (2.5D height maps, point clouds, etc.).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pyvista as pv
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py


@dataclass
class WarpFieldData:
    """Container for warp field data in various formats."""
    # Original 3D data
    original_coordinates: np.ndarray  # Original node coordinates
    deformed_coordinates: np.ndarray  # Deformed node coordinates
    displacement_field: np.ndarray   # Displacement vectors
    
    # Surface data
    top_surface_original: np.ndarray
    top_surface_deformed: np.ndarray
    bottom_surface_original: np.ndarray
    bottom_surface_deformed: np.ndarray
    
    # 2.5D height maps
    top_height_map: Optional[np.ndarray] = None
    bottom_height_map: Optional[np.ndarray] = None
    warp_height_map: Optional[np.ndarray] = None  # Relative to flat reference
    
    # Grid information
    x_grid: Optional[np.ndarray] = None
    y_grid: Optional[np.ndarray] = None
    grid_resolution: Optional[Tuple[int, int]] = None
    
    # Metadata
    plate_dimensions: Tuple[float, float] = None  # (length, width)
    max_warp: float = 0.0
    rms_warp: float = 0.0


class WarpFieldExtractor:
    """Extracts and processes warp fields from FEA simulation results."""
    
    def __init__(self, grid_resolution: Tuple[int, int] = (64, 64)):
        """Initialize warp field extractor.
        
        Args:
            grid_resolution: Target resolution for 2.5D height maps (nx, ny)
        """
        self.grid_resolution = grid_resolution
        self.tolerance = 1e-10
    
    def extract_warp_field(self, mesh: pv.UnstructuredGrid, 
                          displacement_field: np.ndarray,
                          geometry_params: Dict[str, Any]) -> WarpFieldData:
        """Extract complete warp field data from FEA results.
        
        Args:
            mesh: FEA mesh
            displacement_field: Nodal displacement field [n_nodes, 3]
            geometry_params: Geometry parameters from DOE
            
        Returns:
            WarpFieldData containing all warp field representations
        """
        
        # Get original coordinates
        original_coords = mesh.points.copy()
        
        # Calculate deformed coordinates
        deformed_coords = original_coords + displacement_field
        
        # Extract plate dimensions
        bounds = mesh.bounds
        length = bounds[1] - bounds[0]  # x-direction
        width = bounds[3] - bounds[2]   # y-direction
        
        # Extract surface coordinates
        top_orig, top_def = self._extract_top_surface(mesh, original_coords, deformed_coords)
        bottom_orig, bottom_def = self._extract_bottom_surface(mesh, original_coords, deformed_coords)
        
        # Generate 2.5D height maps
        x_grid, y_grid, top_height_map = self._create_height_map(top_orig, top_def, bounds)
        _, _, bottom_height_map = self._create_height_map(bottom_orig, bottom_def, bounds)
        
        # Calculate warp relative to flat reference
        warp_height_map = self._calculate_relative_warp(top_height_map, bottom_height_map)
        
        # Calculate warp statistics
        max_warp = np.max(np.abs(warp_height_map))
        rms_warp = np.sqrt(np.mean(warp_height_map**2))
        
        # Create warp field data object
        warp_data = WarpFieldData(
            original_coordinates=original_coords,
            deformed_coordinates=deformed_coords,
            displacement_field=displacement_field,
            top_surface_original=top_orig,
            top_surface_deformed=top_def,
            bottom_surface_original=bottom_orig,
            bottom_surface_deformed=bottom_def,
            top_height_map=top_height_map,
            bottom_height_map=bottom_height_map,
            warp_height_map=warp_height_map,
            x_grid=x_grid,
            y_grid=y_grid,
            grid_resolution=self.grid_resolution,
            plate_dimensions=(length, width),
            max_warp=max_warp,
            rms_warp=rms_warp
        )
        
        return warp_data
    
    def _extract_top_surface(self, mesh: pv.UnstructuredGrid, 
                           original_coords: np.ndarray, 
                           deformed_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract top surface coordinates."""
        
        # Find nodes on top surface (maximum z-coordinate)
        z_coords = original_coords[:, 2]
        z_max = np.max(z_coords)
        
        # Nodes within tolerance of maximum z
        top_mask = np.abs(z_coords - z_max) < self.tolerance
        top_indices = np.where(top_mask)[0]
        
        top_original = original_coords[top_indices]
        top_deformed = deformed_coords[top_indices]
        
        return top_original, top_deformed
    
    def _extract_bottom_surface(self, mesh: pv.UnstructuredGrid, 
                              original_coords: np.ndarray, 
                              deformed_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract bottom surface coordinates."""
        
        # Find nodes on bottom surface (minimum z-coordinate)
        z_coords = original_coords[:, 2]
        z_min = np.min(z_coords)
        
        # Nodes within tolerance of minimum z
        bottom_mask = np.abs(z_coords - z_min) < self.tolerance
        bottom_indices = np.where(bottom_mask)[0]
        
        bottom_original = original_coords[bottom_indices]
        bottom_deformed = deformed_coords[bottom_indices]
        
        return bottom_original, bottom_deformed
    
    def _create_height_map(self, original_surface: np.ndarray, 
                          deformed_surface: np.ndarray, 
                          bounds: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create 2.5D height map from surface coordinates."""
        
        # Create regular grid
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        
        x_grid = np.linspace(x_min, x_max, self.grid_resolution[0])
        y_grid = np.linspace(y_min, y_max, self.grid_resolution[1])
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Extract surface points
        surface_xy = deformed_surface[:, :2]  # x, y coordinates
        surface_z = deformed_surface[:, 2]    # z coordinates (height)
        
        # Interpolate to regular grid
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Use different interpolation methods based on data density
        if len(surface_xy) > 1000:
            # Use griddata for large datasets
            height_map_flat = griddata(surface_xy, surface_z, grid_points, method='cubic', fill_value=np.nan)
        else:
            # Use RBF for smaller datasets
            try:
                rbf = RBFInterpolator(surface_xy, surface_z, kernel='thin_plate_spline')
                height_map_flat = rbf(grid_points)
            except:
                # Fallback to linear interpolation
                height_map_flat = griddata(surface_xy, surface_z, grid_points, method='linear', fill_value=np.nan)
        
        # Reshape to grid
        height_map = height_map_flat.reshape(self.grid_resolution)
        
        # Handle NaN values by nearest neighbor interpolation
        if np.any(np.isnan(height_map)):
            height_map = self._fill_nan_values(height_map, surface_xy, surface_z, X_grid, Y_grid)
        
        return X_grid, Y_grid, height_map
    
    def _fill_nan_values(self, height_map: np.ndarray, 
                        surface_xy: np.ndarray, 
                        surface_z: np.ndarray,
                        X_grid: np.ndarray, 
                        Y_grid: np.ndarray) -> np.ndarray:
        """Fill NaN values in height map using nearest neighbor."""
        
        # Find NaN locations
        nan_mask = np.isnan(height_map)
        
        if not np.any(nan_mask):
            return height_map
        
        # Create KDTree for nearest neighbor search
        tree = cKDTree(surface_xy)
        
        # Find nearest neighbors for NaN points
        nan_indices = np.where(nan_mask)
        nan_points = np.column_stack([X_grid[nan_indices], Y_grid[nan_indices]])
        
        distances, indices = tree.query(nan_points)
        
        # Fill NaN values with nearest neighbor values
        height_map[nan_indices] = surface_z[indices]
        
        return height_map
    
    def _calculate_relative_warp(self, top_height_map: np.ndarray, 
                               bottom_height_map: np.ndarray) -> np.ndarray:
        """Calculate warp relative to flat reference plate."""
        
        # Calculate plate thickness variation
        thickness_map = top_height_map - bottom_height_map
        
        # Calculate average thickness
        avg_thickness = np.mean(thickness_map)
        
        # Calculate mid-surface height
        mid_surface_height = bottom_height_map + thickness_map / 2
        
        # Calculate reference plane (best-fit plane)
        reference_height = self._fit_reference_plane(mid_surface_height)
        
        # Warp is deviation from reference plane
        warp_map = mid_surface_height - reference_height
        
        return warp_map
    
    def _fit_reference_plane(self, height_map: np.ndarray) -> np.ndarray:
        """Fit reference plane to height map."""
        
        # Create coordinate arrays
        ny, nx = height_map.shape
        x_coords = np.arange(nx)
        y_coords = np.arange(ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Flatten arrays
        x_flat = X.ravel()
        y_flat = Y.ravel()
        z_flat = height_map.ravel()
        
        # Remove NaN values
        valid_mask = ~np.isnan(z_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        
        # Fit plane: z = ax + by + c
        A = np.column_stack([x_valid, y_valid, np.ones(len(x_valid))])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
        
        # Calculate reference plane
        reference_plane = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
        
        return reference_plane
    
    def create_point_cloud(self, warp_data: WarpFieldData, surface: str = 'top') -> np.ndarray:
        """Create point cloud representation of surface.
        
        Args:
            warp_data: Warp field data
            surface: 'top', 'bottom', or 'both'
            
        Returns:
            Point cloud as [N, 3] array
        """
        
        if surface == 'top':
            return warp_data.top_surface_deformed
        elif surface == 'bottom':
            return warp_data.bottom_surface_deformed
        elif surface == 'both':
            return np.vstack([warp_data.top_surface_deformed, warp_data.bottom_surface_deformed])
        else:
            raise ValueError(f"Unknown surface type: {surface}")
    
    def create_digital_elevation_model(self, warp_data: WarpFieldData) -> Dict[str, np.ndarray]:
        """Create digital elevation model (DEM) representation."""
        
        dem_data = {
            'x_coordinates': warp_data.x_grid,
            'y_coordinates': warp_data.y_grid,
            'top_elevation': warp_data.top_height_map,
            'bottom_elevation': warp_data.bottom_height_map,
            'warp_elevation': warp_data.warp_height_map,
            'thickness_map': warp_data.top_height_map - warp_data.bottom_height_map
        }
        
        return dem_data
    
    def calculate_warp_metrics(self, warp_data: WarpFieldData) -> Dict[str, float]:
        """Calculate various warp characterization metrics."""
        
        warp_map = warp_data.warp_height_map
        
        metrics = {
            'max_warp': np.max(np.abs(warp_map)),
            'rms_warp': np.sqrt(np.mean(warp_map**2)),
            'peak_to_valley': np.max(warp_map) - np.min(warp_map),
            'std_warp': np.std(warp_map),
            'mean_warp': np.mean(warp_map),
            'warp_range': np.ptp(warp_map),  # Peak-to-peak
        }
        
        # Calculate curvature metrics
        if warp_map.shape[0] > 2 and warp_map.shape[1] > 2:
            curvature_metrics = self._calculate_curvature_metrics(warp_map)
            metrics.update(curvature_metrics)
        
        return metrics
    
    def _calculate_curvature_metrics(self, warp_map: np.ndarray) -> Dict[str, float]:
        """Calculate curvature-based metrics."""
        
        # Calculate gradients
        gy, gx = np.gradient(warp_map)
        
        # Calculate second derivatives
        gxx = np.gradient(gx, axis=1)
        gyy = np.gradient(gy, axis=0)
        gxy = np.gradient(gx, axis=0)
        
        # Mean curvature: H = (1 + gx²)gyy - 2gxgygxy + (1 + gy²)gxx / 2(1 + gx² + gy²)^(3/2)
        denominator = 2 * (1 + gx**2 + gy**2)**(3/2)
        numerator = (1 + gx**2) * gyy - 2 * gx * gy * gxy + (1 + gy**2) * gxx
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)
        mean_curvature = numerator / denominator
        
        # Gaussian curvature: K = (gxxgyy - gxy²) / (1 + gx² + gy²)²
        gaussian_curvature = (gxx * gyy - gxy**2) / (1 + gx**2 + gy**2)**2
        
        curvature_metrics = {
            'max_mean_curvature': np.max(np.abs(mean_curvature)),
            'rms_mean_curvature': np.sqrt(np.mean(mean_curvature**2)),
            'max_gaussian_curvature': np.max(np.abs(gaussian_curvature)),
            'rms_gaussian_curvature': np.sqrt(np.mean(gaussian_curvature**2)),
        }
        
        return curvature_metrics
    
    def visualize_warp_field(self, warp_data: WarpFieldData, 
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Create visualization of warp field."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top surface height map
        im1 = axes[0, 0].contourf(warp_data.x_grid, warp_data.y_grid, 
                                 warp_data.top_height_map, levels=20, cmap='viridis')
        axes[0, 0].set_title('Top Surface Height')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        plt.colorbar(im1, ax=axes[0, 0], label='Height (m)')
        
        # Bottom surface height map
        im2 = axes[0, 1].contourf(warp_data.x_grid, warp_data.y_grid, 
                                 warp_data.bottom_height_map, levels=20, cmap='viridis')
        axes[0, 1].set_title('Bottom Surface Height')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Y (m)')
        plt.colorbar(im2, ax=axes[0, 1], label='Height (m)')
        
        # Warp field (deviation from flat)
        im3 = axes[1, 0].contourf(warp_data.x_grid, warp_data.y_grid, 
                                 warp_data.warp_height_map, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('Warp Field (Deviation from Flat)')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        plt.colorbar(im3, ax=axes[1, 0], label='Warp (m)')
        
        # Thickness variation
        thickness_map = warp_data.top_height_map - warp_data.bottom_height_map
        im4 = axes[1, 1].contourf(warp_data.x_grid, warp_data.y_grid, 
                                 thickness_map, levels=20, cmap='plasma')
        axes[1, 1].set_title('Thickness Variation')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        plt.colorbar(im4, ax=axes[1, 1], label='Thickness (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_warp_data(self, warp_data: WarpFieldData, 
                        filepath: Union[str, Path], 
                        format: str = 'hdf5'):
        """Export warp field data to file."""
        
        filepath = Path(filepath)
        
        if format.lower() == 'hdf5':
            self._export_hdf5(warp_data, filepath.with_suffix('.h5'))
        elif format.lower() == 'numpy':
            self._export_numpy(warp_data, filepath.with_suffix('.npz'))
        elif format.lower() == 'csv':
            self._export_csv(warp_data, filepath.with_suffix('.csv'))
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_hdf5(self, warp_data: WarpFieldData, filepath: Path):
        """Export to HDF5 format."""
        
        with h5py.File(filepath, 'w') as f:
            # 3D data
            f.create_dataset('original_coordinates', data=warp_data.original_coordinates)
            f.create_dataset('deformed_coordinates', data=warp_data.deformed_coordinates)
            f.create_dataset('displacement_field', data=warp_data.displacement_field)
            
            # Surface data
            f.create_dataset('top_surface_original', data=warp_data.top_surface_original)
            f.create_dataset('top_surface_deformed', data=warp_data.top_surface_deformed)
            f.create_dataset('bottom_surface_original', data=warp_data.bottom_surface_original)
            f.create_dataset('bottom_surface_deformed', data=warp_data.bottom_surface_deformed)
            
            # 2.5D height maps
            f.create_dataset('x_grid', data=warp_data.x_grid)
            f.create_dataset('y_grid', data=warp_data.y_grid)
            f.create_dataset('top_height_map', data=warp_data.top_height_map)
            f.create_dataset('bottom_height_map', data=warp_data.bottom_height_map)
            f.create_dataset('warp_height_map', data=warp_data.warp_height_map)
            
            # Metadata
            f.attrs['plate_dimensions'] = warp_data.plate_dimensions
            f.attrs['grid_resolution'] = warp_data.grid_resolution
            f.attrs['max_warp'] = warp_data.max_warp
            f.attrs['rms_warp'] = warp_data.rms_warp
    
    def _export_numpy(self, warp_data: WarpFieldData, filepath: Path):
        """Export to NumPy format."""
        
        np.savez_compressed(
            filepath,
            original_coordinates=warp_data.original_coordinates,
            deformed_coordinates=warp_data.deformed_coordinates,
            displacement_field=warp_data.displacement_field,
            top_surface_original=warp_data.top_surface_original,
            top_surface_deformed=warp_data.top_surface_deformed,
            bottom_surface_original=warp_data.bottom_surface_original,
            bottom_surface_deformed=warp_data.bottom_surface_deformed,
            x_grid=warp_data.x_grid,
            y_grid=warp_data.y_grid,
            top_height_map=warp_data.top_height_map,
            bottom_height_map=warp_data.bottom_height_map,
            warp_height_map=warp_data.warp_height_map,
            plate_dimensions=warp_data.plate_dimensions,
            grid_resolution=warp_data.grid_resolution,
            max_warp=warp_data.max_warp,
            rms_warp=warp_data.rms_warp
        )
    
    def _export_csv(self, warp_data: WarpFieldData, filepath: Path):
        """Export height maps to CSV format."""
        
        # Flatten height maps and create DataFrame
        x_flat = warp_data.x_grid.ravel()
        y_flat = warp_data.y_grid.ravel()
        top_flat = warp_data.top_height_map.ravel()
        bottom_flat = warp_data.bottom_height_map.ravel()
        warp_flat = warp_data.warp_height_map.ravel()
        
        df = pd.DataFrame({
            'x': x_flat,
            'y': y_flat,
            'top_height': top_flat,
            'bottom_height': bottom_flat,
            'warp': warp_flat
        })
        
        df.to_csv(filepath, index=False)


def create_warp_extractor(grid_resolution: Tuple[int, int] = (64, 64)) -> WarpFieldExtractor:
    """Factory function to create warp field extractor."""
    return WarpFieldExtractor(grid_resolution)


if __name__ == "__main__":
    # Test warp field extraction
    print("Warp field extractor module loaded successfully!")
    
    extractor = create_warp_extractor(grid_resolution=(32, 32))
    print(f"Created warp extractor with grid resolution: {extractor.grid_resolution}")