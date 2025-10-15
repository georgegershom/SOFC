"""
Residual Stress Field Extraction Module

This module extracts residual stress fields from FEA simulation results and converts them
to various formats suitable for machine learning (3D voxelized fields, stress tensors, etc.).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pyvista as pv
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py


@dataclass
class StressFieldData:
    """Container for stress field data in various formats."""
    # Original element-based stress data
    element_coordinates: np.ndarray  # Element centroid coordinates
    stress_tensors: np.ndarray       # Full stress tensors [n_elements, 6] (xx, yy, zz, xy, xz, yz)
    
    # Nodal stress data (extrapolated/averaged)
    nodal_coordinates: np.ndarray    # Node coordinates
    nodal_stress_tensors: np.ndarray # Nodal stress tensors [n_nodes, 6]
    
    # 3D voxelized stress field
    voxel_grid_x: Optional[np.ndarray] = None
    voxel_grid_y: Optional[np.ndarray] = None
    voxel_grid_z: Optional[np.ndarray] = None
    voxelized_stress: Optional[np.ndarray] = None  # [nx, ny, nz, 6]
    voxel_resolution: Optional[Tuple[int, int, int]] = None
    
    # Stress invariants and derived quantities
    von_mises_stress: Optional[np.ndarray] = None
    hydrostatic_stress: Optional[np.ndarray] = None
    deviatoric_stress: Optional[np.ndarray] = None
    principal_stresses: Optional[np.ndarray] = None  # [n_points, 3]
    max_shear_stress: Optional[np.ndarray] = None
    
    # Layer-wise stress data
    layer_stress_data: Optional[Dict[str, np.ndarray]] = None
    
    # Metadata
    stress_units: str = "Pa"
    max_von_mises: float = 0.0
    max_principal_stress: float = 0.0
    volume_averaged_stress: Optional[np.ndarray] = None


class StressFieldExtractor:
    """Extracts and processes stress fields from FEA simulation results."""
    
    def __init__(self, voxel_resolution: Tuple[int, int, int] = (32, 32, 16)):
        """Initialize stress field extractor.
        
        Args:
            voxel_resolution: Target resolution for 3D voxelized stress field (nx, ny, nz)
        """
        self.voxel_resolution = voxel_resolution
        self.tolerance = 1e-10
    
    def extract_stress_field(self, mesh: pv.UnstructuredGrid, 
                           element_stress: np.ndarray,
                           geometry_params: Dict[str, Any]) -> StressFieldData:
        """Extract complete stress field data from FEA results.
        
        Args:
            mesh: FEA mesh
            element_stress: Element stress tensors [n_elements, 6]
            geometry_params: Geometry parameters from DOE
            
        Returns:
            StressFieldData containing all stress field representations
        """
        
        # Get element centroids
        element_coords = self._calculate_element_centroids(mesh)
        
        # Extrapolate stress to nodes
        nodal_coords = mesh.points.copy()
        nodal_stress = self._extrapolate_stress_to_nodes(mesh, element_stress)
        
        # Create 3D voxelized stress field
        voxel_x, voxel_y, voxel_z, voxelized_stress = self._create_voxelized_field(
            mesh, element_coords, element_stress
        )
        
        # Calculate stress invariants and derived quantities
        von_mises = self._calculate_von_mises_stress(element_stress)
        hydrostatic = self._calculate_hydrostatic_stress(element_stress)
        deviatoric = self._calculate_deviatoric_stress(element_stress)
        principal_stresses = self._calculate_principal_stresses(element_stress)
        max_shear = self._calculate_max_shear_stress(principal_stresses)
        
        # Calculate layer-wise stress data
        layer_stress = self._extract_layer_stress_data(mesh, element_coords, element_stress)
        
        # Calculate volume-averaged stress
        volume_avg_stress = self._calculate_volume_averaged_stress(mesh, element_stress)
        
        # Calculate statistics
        max_von_mises = np.max(von_mises)
        max_principal = np.max(np.abs(principal_stresses))
        
        # Create stress field data object
        stress_data = StressFieldData(
            element_coordinates=element_coords,
            stress_tensors=element_stress,
            nodal_coordinates=nodal_coords,
            nodal_stress_tensors=nodal_stress,
            voxel_grid_x=voxel_x,
            voxel_grid_y=voxel_y,
            voxel_grid_z=voxel_z,
            voxelized_stress=voxelized_stress,
            voxel_resolution=self.voxel_resolution,
            von_mises_stress=von_mises,
            hydrostatic_stress=hydrostatic,
            deviatoric_stress=deviatoric,
            principal_stresses=principal_stresses,
            max_shear_stress=max_shear,
            layer_stress_data=layer_stress,
            max_von_mises=max_von_mises,
            max_principal_stress=max_principal,
            volume_averaged_stress=volume_avg_stress
        )
        
        return stress_data
    
    def _calculate_element_centroids(self, mesh: pv.UnstructuredGrid) -> np.ndarray:
        """Calculate centroids of mesh elements."""
        
        centroids = []
        for i in range(mesh.n_cells):
            cell = mesh.get_cell(i)
            centroid = np.mean(cell.points, axis=0)
            centroids.append(centroid)
        
        return np.array(centroids)
    
    def _extrapolate_stress_to_nodes(self, mesh: pv.UnstructuredGrid, 
                                   element_stress: np.ndarray) -> np.ndarray:
        """Extrapolate element stress to nodes using averaging."""
        
        n_nodes = mesh.n_points
        n_stress_components = element_stress.shape[1]
        
        # Initialize nodal stress and weights
        nodal_stress = np.zeros((n_nodes, n_stress_components))
        nodal_weights = np.zeros(n_nodes)
        
        # Loop through elements and accumulate stress at nodes
        for i in range(mesh.n_cells):
            cell = mesh.get_cell(i)
            element_stress_tensor = element_stress[i]
            
            # Add element stress to all nodes of the element
            for node_id in cell.point_ids:
                nodal_stress[node_id] += element_stress_tensor
                nodal_weights[node_id] += 1.0
        
        # Average stress at nodes
        for i in range(n_nodes):
            if nodal_weights[i] > 0:
                nodal_stress[i] /= nodal_weights[i]
        
        return nodal_stress
    
    def _create_voxelized_field(self, mesh: pv.UnstructuredGrid, 
                              element_coords: np.ndarray, 
                              element_stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create 3D voxelized stress field."""
        
        # Get mesh bounds
        bounds = mesh.bounds
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        # Create regular voxel grid
        x_voxel = np.linspace(x_min, x_max, self.voxel_resolution[0])
        y_voxel = np.linspace(y_min, y_max, self.voxel_resolution[1])
        z_voxel = np.linspace(z_min, z_max, self.voxel_resolution[2])
        
        X_voxel, Y_voxel, Z_voxel = np.meshgrid(x_voxel, y_voxel, z_voxel, indexing='ij')
        
        # Interpolate stress to voxel grid
        voxel_points = np.column_stack([X_voxel.ravel(), Y_voxel.ravel(), Z_voxel.ravel()])
        
        # Initialize voxelized stress field
        n_stress_components = element_stress.shape[1]
        voxelized_stress = np.zeros((*self.voxel_resolution, n_stress_components))
        
        # Interpolate each stress component
        for comp in range(n_stress_components):
            stress_component = element_stress[:, comp]
            
            # Use nearest neighbor interpolation for robustness
            tree = cKDTree(element_coords)
            distances, indices = tree.query(voxel_points)
            
            # Interpolate using inverse distance weighting for close points
            interpolated_values = np.zeros(len(voxel_points))
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < self.tolerance:
                    # Exact match
                    interpolated_values[i] = stress_component[idx]
                else:
                    # Find multiple nearest neighbors for better interpolation
                    k_neighbors = min(8, len(element_coords))
                    neighbor_distances, neighbor_indices = tree.query(voxel_points[i], k=k_neighbors)
                    
                    # Inverse distance weighting
                    weights = 1.0 / (neighbor_distances + self.tolerance)
                    weights /= np.sum(weights)
                    
                    interpolated_values[i] = np.sum(weights * stress_component[neighbor_indices])
            
            # Reshape to voxel grid
            voxelized_stress[:, :, :, comp] = interpolated_values.reshape(self.voxel_resolution)
        
        return X_voxel, Y_voxel, Z_voxel, voxelized_stress
    
    def _calculate_von_mises_stress(self, stress_tensors: np.ndarray) -> np.ndarray:
        """Calculate von Mises stress from stress tensors."""
        
        # Extract stress components
        sxx = stress_tensors[:, 0]
        syy = stress_tensors[:, 1]
        szz = stress_tensors[:, 2]
        sxy = stress_tensors[:, 3]
        sxz = stress_tensors[:, 4] if stress_tensors.shape[1] > 4 else np.zeros_like(sxx)
        syz = stress_tensors[:, 5] if stress_tensors.shape[1] > 5 else np.zeros_like(sxx)
        
        # von Mises stress formula
        von_mises = np.sqrt(0.5 * (
            (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
            6 * (sxy**2 + sxz**2 + syz**2)
        ))
        
        return von_mises
    
    def _calculate_hydrostatic_stress(self, stress_tensors: np.ndarray) -> np.ndarray:
        """Calculate hydrostatic (mean) stress."""
        
        sxx = stress_tensors[:, 0]
        syy = stress_tensors[:, 1]
        szz = stress_tensors[:, 2]
        
        hydrostatic = (sxx + syy + szz) / 3.0
        
        return hydrostatic
    
    def _calculate_deviatoric_stress(self, stress_tensors: np.ndarray) -> np.ndarray:
        """Calculate deviatoric stress tensor."""
        
        hydrostatic = self._calculate_hydrostatic_stress(stress_tensors)
        
        # Deviatoric stress tensor
        deviatoric = stress_tensors.copy()
        deviatoric[:, 0] -= hydrostatic  # sxx - p
        deviatoric[:, 1] -= hydrostatic  # syy - p
        deviatoric[:, 2] -= hydrostatic  # szz - p
        # Shear components remain unchanged
        
        return deviatoric
    
    def _calculate_principal_stresses(self, stress_tensors: np.ndarray) -> np.ndarray:
        """Calculate principal stresses from stress tensors."""
        
        n_elements = stress_tensors.shape[0]
        principal_stresses = np.zeros((n_elements, 3))
        
        for i in range(n_elements):
            # Construct stress tensor matrix
            stress_matrix = np.array([
                [stress_tensors[i, 0], stress_tensors[i, 3], stress_tensors[i, 4] if stress_tensors.shape[1] > 4 else 0],
                [stress_tensors[i, 3], stress_tensors[i, 1], stress_tensors[i, 5] if stress_tensors.shape[1] > 5 else 0],
                [stress_tensors[i, 4] if stress_tensors.shape[1] > 4 else 0, 
                 stress_tensors[i, 5] if stress_tensors.shape[1] > 5 else 0, 
                 stress_tensors[i, 2]]
            ])
            
            # Calculate eigenvalues (principal stresses)
            eigenvalues = np.linalg.eigvals(stress_matrix)
            
            # Sort in descending order (σ1 ≥ σ2 ≥ σ3)
            principal_stresses[i] = np.sort(eigenvalues)[::-1]
        
        return principal_stresses
    
    def _calculate_max_shear_stress(self, principal_stresses: np.ndarray) -> np.ndarray:
        """Calculate maximum shear stress from principal stresses."""
        
        # Maximum shear stress = (σ1 - σ3) / 2
        max_shear = (principal_stresses[:, 0] - principal_stresses[:, 2]) / 2.0
        
        return max_shear
    
    def _extract_layer_stress_data(self, mesh: pv.UnstructuredGrid, 
                                 element_coords: np.ndarray, 
                                 element_stress: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract stress data for each layer."""
        
        layer_data = {}
        
        # Get material IDs if available
        if 'material_id' in mesh.cell_data:
            material_ids = mesh.cell_data['material_id']
            unique_materials = np.unique(material_ids)
            
            # Material ID to layer name mapping
            material_to_layer = {1: 'anode', 2: 'electrolyte', 3: 'cathode'}
            
            for mat_id in unique_materials:
                layer_name = material_to_layer.get(mat_id, f'layer_{mat_id}')
                
                # Find elements belonging to this layer
                layer_mask = material_ids == mat_id
                layer_indices = np.where(layer_mask)[0]
                
                # Extract layer stress data
                layer_coords = element_coords[layer_indices]
                layer_stress = element_stress[layer_indices]
                
                # Calculate layer statistics
                layer_von_mises = self._calculate_von_mises_stress(layer_stress)
                layer_principal = self._calculate_principal_stresses(layer_stress)
                
                layer_data[layer_name] = {
                    'coordinates': layer_coords,
                    'stress_tensors': layer_stress,
                    'von_mises': layer_von_mises,
                    'principal_stresses': layer_principal,
                    'max_von_mises': np.max(layer_von_mises),
                    'mean_von_mises': np.mean(layer_von_mises),
                    'volume_fraction': len(layer_indices) / len(element_stress)
                }
        
        return layer_data
    
    def _calculate_volume_averaged_stress(self, mesh: pv.UnstructuredGrid, 
                                        element_stress: np.ndarray) -> np.ndarray:
        """Calculate volume-averaged stress tensor."""
        
        # Get element volumes
        if 'volume' in mesh.cell_data:
            volumes = mesh.cell_data['volume']
        else:
            # Calculate volumes
            volumes = np.array([self._calculate_element_volume(mesh, i) for i in range(mesh.n_cells)])
        
        total_volume = np.sum(volumes)
        
        # Volume-weighted average
        volume_avg_stress = np.sum(element_stress * volumes[:, np.newaxis], axis=0) / total_volume
        
        return volume_avg_stress
    
    def _calculate_element_volume(self, mesh: pv.UnstructuredGrid, element_id: int) -> float:
        """Calculate volume of a single element."""
        
        cell = mesh.get_cell(element_id)
        
        # Simplified volume calculation for hexahedral elements
        if cell.type == 12:  # VTK_HEXAHEDRON
            points = cell.points
            if len(points) == 8:
                # Calculate edge vectors
                dx = np.abs(points[1, 0] - points[0, 0])
                dy = np.abs(points[3, 1] - points[0, 1])
                dz = np.abs(points[4, 2] - points[0, 2])
                return dx * dy * dz
        
        # Fallback: use PyVista's volume calculation if available
        try:
            return cell.volume
        except:
            return 1.0  # Default volume
    
    def calculate_stress_metrics(self, stress_data: StressFieldData) -> Dict[str, float]:
        """Calculate various stress characterization metrics."""
        
        metrics = {
            'max_von_mises': stress_data.max_von_mises,
            'mean_von_mises': np.mean(stress_data.von_mises_stress),
            'std_von_mises': np.std(stress_data.von_mises_stress),
            'max_principal_stress': stress_data.max_principal_stress,
            'max_hydrostatic': np.max(np.abs(stress_data.hydrostatic_stress)),
            'max_shear_stress': np.max(stress_data.max_shear_stress),
        }
        
        # Add layer-specific metrics
        if stress_data.layer_stress_data:
            for layer_name, layer_data in stress_data.layer_stress_data.items():
                metrics[f'{layer_name}_max_von_mises'] = layer_data['max_von_mises']
                metrics[f'{layer_name}_mean_von_mises'] = layer_data['mean_von_mises']
        
        # Add volume-averaged stress components
        if stress_data.volume_averaged_stress is not None:
            stress_components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
            for i, component in enumerate(stress_components[:len(stress_data.volume_averaged_stress)]):
                metrics[f'volume_avg_stress_{component}'] = stress_data.volume_averaged_stress[i]
        
        return metrics
    
    def visualize_stress_field(self, stress_data: StressFieldData, 
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Create visualization of stress field."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # von Mises stress distribution
        coords = stress_data.element_coordinates
        von_mises = stress_data.von_mises_stress
        
        # Create 2D projections for visualization
        # XY projection (top view)
        scatter1 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=von_mises, 
                                     cmap='viridis', s=1)
        axes[0, 0].set_title('von Mises Stress (Top View)')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Stress (Pa)')
        
        # XZ projection (side view)
        scatter2 = axes[0, 1].scatter(coords[:, 0], coords[:, 2], c=von_mises, 
                                     cmap='viridis', s=1)
        axes[0, 1].set_title('von Mises Stress (Side View)')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Z (m)')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Stress (Pa)')
        
        # Principal stress distribution
        max_principal = stress_data.principal_stresses[:, 0]
        scatter3 = axes[1, 0].scatter(coords[:, 0], coords[:, 1], c=max_principal, 
                                     cmap='RdBu_r', s=1)
        axes[1, 0].set_title('Maximum Principal Stress (Top View)')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        plt.colorbar(scatter3, ax=axes[1, 0], label='Stress (Pa)')
        
        # Hydrostatic stress distribution
        hydrostatic = stress_data.hydrostatic_stress
        scatter4 = axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=hydrostatic, 
                                     cmap='coolwarm', s=1)
        axes[1, 1].set_title('Hydrostatic Stress (Top View)')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        plt.colorbar(scatter4, ax=axes[1, 1], label='Stress (Pa)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_stress_data(self, stress_data: StressFieldData, 
                          filepath: Union[str, Path], 
                          format: str = 'hdf5'):
        """Export stress field data to file."""
        
        filepath = Path(filepath)
        
        if format.lower() == 'hdf5':
            self._export_hdf5(stress_data, filepath.with_suffix('.h5'))
        elif format.lower() == 'numpy':
            self._export_numpy(stress_data, filepath.with_suffix('.npz'))
        elif format.lower() == 'vtk':
            self._export_vtk(stress_data, filepath.with_suffix('.vtk'))
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_hdf5(self, stress_data: StressFieldData, filepath: Path):
        """Export to HDF5 format."""
        
        with h5py.File(filepath, 'w') as f:
            # Element data
            f.create_dataset('element_coordinates', data=stress_data.element_coordinates)
            f.create_dataset('stress_tensors', data=stress_data.stress_tensors)
            
            # Nodal data
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
            f.attrs['voxel_resolution'] = stress_data.voxel_resolution
            
            if stress_data.volume_averaged_stress is not None:
                f.create_dataset('volume_averaged_stress', data=stress_data.volume_averaged_stress)
    
    def _export_numpy(self, stress_data: StressFieldData, filepath: Path):
        """Export to NumPy format."""
        
        save_dict = {
            'element_coordinates': stress_data.element_coordinates,
            'stress_tensors': stress_data.stress_tensors,
            'nodal_coordinates': stress_data.nodal_coordinates,
            'nodal_stress_tensors': stress_data.nodal_stress_tensors,
            'von_mises_stress': stress_data.von_mises_stress,
            'hydrostatic_stress': stress_data.hydrostatic_stress,
            'principal_stresses': stress_data.principal_stresses,
            'max_shear_stress': stress_data.max_shear_stress,
            'max_von_mises': stress_data.max_von_mises,
            'max_principal_stress': stress_data.max_principal_stress,
        }
        
        if stress_data.voxelized_stress is not None:
            save_dict.update({
                'voxel_grid_x': stress_data.voxel_grid_x,
                'voxel_grid_y': stress_data.voxel_grid_y,
                'voxel_grid_z': stress_data.voxel_grid_z,
                'voxelized_stress': stress_data.voxelized_stress,
                'voxel_resolution': stress_data.voxel_resolution,
            })
        
        if stress_data.volume_averaged_stress is not None:
            save_dict['volume_averaged_stress'] = stress_data.volume_averaged_stress
        
        np.savez_compressed(filepath, **save_dict)
    
    def _export_vtk(self, stress_data: StressFieldData, filepath: Path):
        """Export to VTK format for visualization."""
        
        # Create point cloud from element coordinates
        points = stress_data.element_coordinates
        
        # Create PyVista point cloud
        point_cloud = pv.PolyData(points)
        
        # Add stress data
        point_cloud.point_data['von_mises_stress'] = stress_data.von_mises_stress
        point_cloud.point_data['hydrostatic_stress'] = stress_data.hydrostatic_stress
        point_cloud.point_data['max_shear_stress'] = stress_data.max_shear_stress
        point_cloud.point_data['stress_xx'] = stress_data.stress_tensors[:, 0]
        point_cloud.point_data['stress_yy'] = stress_data.stress_tensors[:, 1]
        point_cloud.point_data['stress_zz'] = stress_data.stress_tensors[:, 2]
        point_cloud.point_data['stress_xy'] = stress_data.stress_tensors[:, 3]
        
        if stress_data.stress_tensors.shape[1] > 4:
            point_cloud.point_data['stress_xz'] = stress_data.stress_tensors[:, 4]
        if stress_data.stress_tensors.shape[1] > 5:
            point_cloud.point_data['stress_yz'] = stress_data.stress_tensors[:, 5]
        
        # Save to VTK
        point_cloud.save(str(filepath))


def create_stress_extractor(voxel_resolution: Tuple[int, int, int] = (32, 32, 16)) -> StressFieldExtractor:
    """Factory function to create stress field extractor."""
    return StressFieldExtractor(voxel_resolution)


if __name__ == "__main__":
    # Test stress field extraction
    print("Stress field extractor module loaded successfully!")
    
    extractor = create_stress_extractor(voxel_resolution=(16, 16, 8))
    print(f"Created stress extractor with voxel resolution: {extractor.voxel_resolution}")