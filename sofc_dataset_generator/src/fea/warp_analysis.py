"""
Warp Analysis Module

Analyzes the deformed geometry of SOFC plates to extract warp field data
for use as input features in ML models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


@dataclass
class WarpField:
    """Container for warp field data"""
    # Surface coordinates
    surface_coords: np.ndarray  # [n_points, 3] - x, y, z coordinates
    
    # Deformed coordinates
    deformed_coords: np.ndarray  # [n_points, 3] - x', y', z' coordinates
    
    # Displacements
    displacements: np.ndarray  # [n_points, 3] - ux, uy, uz
    
    # Height map data
    height_map: Optional[np.ndarray] = None  # 2D height map
    height_map_x: Optional[np.ndarray] = None  # X coordinates for height map
    height_map_y: Optional[np.ndarray] = None  # Y coordinates for height map
    
    # Warp metrics
    max_warp: float = 0.0
    rms_warp: float = 0.0
    warp_area: float = 0.0
    
    def get_height_map(self, resolution: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate height map from surface coordinates"""
        if self.height_map is not None:
            return self.height_map, self.height_map_x, self.height_map_y
        
        # Get surface bounds
        x_min, x_max = np.min(self.surface_coords[:, 0]), np.max(self.surface_coords[:, 0])
        y_min, y_max = np.min(self.surface_coords[:, 1]), np.max(self.surface_coords[:, 1])
        
        # Create regular grid
        x_grid = np.linspace(x_min, x_max, resolution[0])
        y_grid = np.linspace(y_min, y_max, resolution[1])
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate Z coordinates
        points = self.surface_coords[:, :2]  # x, y coordinates
        values = self.surface_coords[:, 2]   # z coordinates
        
        Z_grid = griddata(points, values, (X_grid, Y_grid), method='cubic', fill_value=0.0)
        
        # Store for future use
        self.height_map = Z_grid
        self.height_map_x = x_grid
        self.height_map_y = y_grid
        
        return Z_grid, x_grid, y_grid
    
    def get_point_cloud(self) -> np.ndarray:
        """Get point cloud representation of deformed surface"""
        return self.deformed_coords
    
    def get_displacement_magnitude(self) -> np.ndarray:
        """Get displacement magnitude at each point"""
        return np.linalg.norm(self.displacements, axis=1)
    
    def get_z_displacement(self) -> np.ndarray:
        """Get Z-direction displacement (out-of-plane)"""
        return self.displacements[:, 2]


class WarpAnalyzer:
    """Analyzes warp patterns in SOFC plates"""
    
    def __init__(self, mesh: Dict, simulation_results):
        self.mesh = mesh
        self.results = simulation_results
        self.warp_fields = {}
    
    def analyze_surface_warp(self, surface: str = 'top') -> WarpField:
        """
        Analyze warp of a specific surface
        
        Args:
            surface: Surface to analyze ('top', 'bottom', 'electrolyte_top', 'electrolyte_bottom')
        """
        # Get surface nodes
        surface_nodes = self._get_surface_nodes(surface)
        
        # Get original coordinates
        original_coords = self.mesh['nodes'][surface_nodes]
        
        # Get displacements
        surface_displacements = self.results.displacements[surface_nodes]
        
        # Compute deformed coordinates
        deformed_coords = original_coords + surface_displacements
        
        # Create warp field
        warp_field = WarpField(
            surface_coords=original_coords,
            deformed_coords=deformed_coords,
            displacements=surface_displacements
        )
        
        # Compute warp metrics
        self._compute_warp_metrics(warp_field)
        
        # Store warp field
        self.warp_fields[surface] = warp_field
        
        return warp_field
    
    def _get_surface_nodes(self, surface: str) -> np.ndarray:
        """Get node indices for specific surface"""
        nodes = self.mesh['nodes']
        
        if surface == 'top':
            z_max = np.max(nodes[:, 2])
            return np.where(np.abs(nodes[:, 2] - z_max) < 1e-6)[0]
        
        elif surface == 'bottom':
            z_min = np.min(nodes[:, 2])
            return np.where(np.abs(nodes[:, 2] - z_min) < 1e-6)[0]
        
        elif surface == 'electrolyte_top':
            # Electrolyte top surface
            z_electrolyte_top = (self.mesh['mesh_parameters'].anode_thickness_mm + 
                               self.mesh['mesh_parameters'].electrolyte_thickness_mm)
            return np.where(np.abs(nodes[:, 2] - z_electrolyte_top) < 1e-6)[0]
        
        elif surface == 'electrolyte_bottom':
            # Electrolyte bottom surface
            z_electrolyte_bottom = self.mesh['mesh_parameters'].anode_thickness_mm
            return np.where(np.abs(nodes[:, 2] - z_electrolyte_bottom) < 1e-6)[0]
        
        else:
            raise ValueError(f"Unknown surface: {surface}")
    
    def _compute_warp_metrics(self, warp_field: WarpField):
        """Compute warp metrics"""
        # Maximum warp (maximum Z displacement)
        warp_field.max_warp = np.max(np.abs(warp_field.displacements[:, 2]))
        
        # RMS warp
        warp_field.rms_warp = np.sqrt(np.mean(warp_field.displacements[:, 2]**2))
        
        # Warp area (area with significant warp)
        significant_warp = np.abs(warp_field.displacements[:, 2]) > 0.1 * warp_field.max_warp
        warp_field.warp_area = np.sum(significant_warp) / len(significant_warp)
    
    def generate_height_map(self, surface: str = 'top', resolution: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate height map for surface"""
        if surface not in self.warp_fields:
            self.analyze_surface_warp(surface)
        
        return self.warp_fields[surface].get_height_map(resolution)
    
    def analyze_warp_patterns(self, surface: str = 'top') -> Dict[str, float]:
        """Analyze warp patterns and return quantitative metrics"""
        if surface not in self.warp_fields:
            self.analyze_surface_warp(surface)
        
        warp_field = self.warp_fields[surface]
        
        # Basic metrics
        metrics = {
            'max_warp': warp_field.max_warp,
            'rms_warp': warp_field.rms_warp,
            'warp_area_ratio': warp_field.warp_area
        }
        
        # Displacement statistics
        displacement_mag = warp_field.get_displacement_magnitude()
        metrics.update({
            'max_displacement': np.max(displacement_mag),
            'mean_displacement': np.mean(displacement_mag),
            'std_displacement': np.std(displacement_mag)
        })
        
        # Z-displacement statistics
        z_displacement = warp_field.get_z_displacement()
        metrics.update({
            'max_z_displacement': np.max(z_displacement),
            'min_z_displacement': np.min(z_displacement),
            'mean_z_displacement': np.mean(z_displacement),
            'std_z_displacement': np.std(z_displacement)
        })
        
        # Curvature analysis
        curvature_metrics = self._analyze_curvature(warp_field)
        metrics.update(curvature_metrics)
        
        return metrics
    
    def _analyze_curvature(self, warp_field: WarpField) -> Dict[str, float]:
        """Analyze surface curvature"""
        # Generate height map
        height_map, x_grid, y_grid = warp_field.get_height_map()
        
        # Compute gradients
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        
        # First derivatives
        dz_dx = np.gradient(height_map, dx, axis=1)
        dz_dy = np.gradient(height_map, dy, axis=0)
        
        # Second derivatives
        d2z_dx2 = np.gradient(dz_dx, dx, axis=1)
        d2z_dy2 = np.gradient(dz_dy, dy, axis=0)
        d2z_dxdy = np.gradient(dz_dx, dy, axis=0)
        
        # Mean curvature
        mean_curvature = 0.5 * (d2z_dx2 + d2z_dy2)
        
        # Gaussian curvature
        gaussian_curvature = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        
        # Principal curvatures
        trace = d2z_dx2 + d2z_dy2
        det = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        
        k1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det))
        k2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det))
        
        return {
            'max_mean_curvature': np.max(np.abs(mean_curvature)),
            'rms_mean_curvature': np.sqrt(np.mean(mean_curvature**2)),
            'max_gaussian_curvature': np.max(np.abs(gaussian_curvature)),
            'rms_gaussian_curvature': np.sqrt(np.mean(gaussian_curvature**2)),
            'max_principal_curvature': np.max(np.abs(k1)),
            'min_principal_curvature': np.min(np.abs(k2))
        }
    
    def visualize_warp(self, surface: str = 'top', save_path: Optional[str] = None):
        """Visualize warp field"""
        if surface not in self.warp_fields:
            self.analyze_surface_warp(surface)
        
        warp_field = self.warp_fields[surface]
        
        # Generate height map
        height_map, x_grid, y_grid = warp_field.get_height_map()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Height map
        im1 = axes[0, 0].contourf(x_grid, y_grid, height_map, levels=20, cmap='viridis')
        axes[0, 0].set_title(f'{surface.capitalize()} Surface Height Map')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0, 0], label='Height (mm)')
        
        # Displacement magnitude
        displacement_mag = warp_field.get_displacement_magnitude()
        scatter = axes[0, 1].scatter(warp_field.surface_coords[:, 0], 
                                   warp_field.surface_coords[:, 1], 
                                   c=displacement_mag, cmap='plasma', s=20)
        axes[0, 1].set_title(f'{surface.capitalize()} Displacement Magnitude')
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Displacement (mm)')
        
        # Z displacement
        z_displacement = warp_field.get_z_displacement()
        scatter2 = axes[1, 0].scatter(warp_field.surface_coords[:, 0], 
                                    warp_field.surface_coords[:, 1], 
                                    c=z_displacement, cmap='RdBu_r', s=20)
        axes[1, 0].set_title(f'{surface.capitalize()} Z Displacement')
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        plt.colorbar(scatter2, ax=axes[1, 0], label='Z Displacement (mm)')
        
        # 3D surface plot
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax3d.plot_surface(x_grid, y_grid, height_map, cmap='viridis', alpha=0.8)
        ax3d.set_title(f'{surface.capitalize()} 3D Surface')
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Height (mm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Warp visualization saved to {save_path}")
        
        plt.show()
    
    def export_warp_data(self, surface: str = 'top', filename: str = 'warp_data.npz'):
        """Export warp data to file"""
        if surface not in self.warp_fields:
            self.analyze_surface_warp(surface)
        
        warp_field = self.warp_fields[surface]
        
        # Prepare data
        data = {
            'surface_coords': warp_field.surface_coords,
            'deformed_coords': warp_field.deformed_coords,
            'displacements': warp_field.displacements,
            'max_warp': warp_field.max_warp,
            'rms_warp': warp_field.rms_warp,
            'warp_area': warp_field.warp_area
        }
        
        # Add height map if available
        if warp_field.height_map is not None:
            data['height_map'] = warp_field.height_map
            data['height_map_x'] = warp_field.height_map_x
            data['height_map_y'] = warp_field.height_map_y
        
        # Save
        np.savez(filename, **data)
        print(f"Warp data exported to {filename}")


if __name__ == "__main__":
    # Example usage
    from .mesh_generator import SOFCMeshGenerator, MeshParameters
    from .fea_solver import FEASolver
    from ..materials.sofc_materials import SOFCMaterials
    
    # Create mesh and run simulation
    mesh_params = MeshParameters()
    mesh_gen = SOFCMeshGenerator(mesh_params)
    mesh = mesh_gen.generate_mesh()
    
    materials = SOFCMaterials()
    material_props = {
        'electrolyte': materials.get_all_properties('8YSZ', 800.0),
        'anode': materials.get_all_properties('NiYSZ', 800.0),
        'cathode': materials.get_all_properties('LSM', 800.0),
        'interconnect': materials.get_all_properties('Crofer22APU', 800.0)
    }
    
    solver = FEASolver(mesh, material_props)
    n_nodes = len(mesh['nodes'])
    temperature_field = np.full(n_nodes, 800.0)
    
    results = solver.solve_thermo_mechanical(
        temperature_field=temperature_field,
        boundary_conditions={'bottom_fixed_z': 0.0},
        assembly_pressure=0.2
    )
    
    # Analyze warp
    warp_analyzer = WarpAnalyzer(mesh, results)
    
    # Analyze top surface
    top_warp = warp_analyzer.analyze_surface_warp('top')
    print(f"Top surface max warp: {top_warp.max_warp:.3f} mm")
    print(f"Top surface RMS warp: {top_warp.rms_warp:.3f} mm")
    
    # Generate height map
    height_map, x_grid, y_grid = top_warp.get_height_map()
    print(f"Height map shape: {height_map.shape}")
    
    # Analyze patterns
    metrics = warp_analyzer.analyze_warp_patterns('top')
    print("\nWarp metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Visualize
    warp_analyzer.visualize_warp('top', 'warp_analysis.png')