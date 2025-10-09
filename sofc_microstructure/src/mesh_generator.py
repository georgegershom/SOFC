"""
Computational Mesh Generation for SOFC Microstructures

This module converts 3D voxelated microstructures into high-quality computational
meshes suitable for finite element analysis, computational fluid dynamics, and
other numerical simulations.

Author: AI Assistant
Date: 2025-10-08
"""

import numpy as np
import scipy.ndimage as ndi
from skimage import measure, morphology
import trimesh
import pyvista as pv
import vtk
from typing import Tuple, Dict, List, Optional, Union
import os
import json
import warnings

warnings.filterwarnings('ignore')

class SOFCMeshGenerator:
    """
    Generator for computational meshes from SOFC microstructure data.
    Supports multiple mesh types and formats for different simulation needs.
    """
    
    def __init__(self, microstructure: np.ndarray, voxel_size: float, phases: Dict[str, int]):
        """
        Initialize the mesh generator.
        
        Parameters:
        -----------
        microstructure : np.ndarray
            3D microstructure array with phase labels
        voxel_size : float
            Size of each voxel in micrometers
        phases : dict
            Dictionary mapping phase names to phase IDs
        """
        self.microstructure = microstructure
        self.voxel_size = voxel_size
        self.phases = phases
        self.dimensions = microstructure.shape
        
        # Generated meshes storage
        self.surface_meshes = {}
        self.volume_meshes = {}
        self.interface_meshes = {}
        
        # Mesh quality metrics
        self.mesh_quality = {}
        
    def generate_surface_meshes(self, smooth_iterations: int = 2, 
                              decimation_factor: float = 0.1) -> Dict[str, trimesh.Trimesh]:
        """
        Generate surface meshes for each phase using marching cubes.
        
        Parameters:
        -----------
        smooth_iterations : int
            Number of smoothing iterations to apply
        decimation_factor : float
            Factor for mesh decimation (0.1 = reduce to 10% of original)
            
        Returns:
        --------
        dict
            Dictionary of phase names to trimesh objects
        """
        print("Generating surface meshes for each phase...")
        
        self.surface_meshes = {}
        
        for phase_name, phase_id in self.phases.items():
            if phase_name == 'pore':  # Skip pore phase for surface mesh
                continue
                
            print(f"  Processing {phase_name} phase...")
            
            # Extract phase mask
            phase_mask = (self.microstructure == phase_id).astype(float)
            
            if not np.any(phase_mask):
                print(f"    Warning: No voxels found for {phase_name}")
                continue
            
            # Smooth the mask
            smoothed_mask = phase_mask.copy()
            for _ in range(smooth_iterations):
                smoothed_mask = ndi.gaussian_filter(smoothed_mask, sigma=0.8)
            
            try:
                # Generate mesh using marching cubes
                vertices, faces, normals, values = measure.marching_cubes(
                    smoothed_mask, level=0.5, spacing=(self.voxel_size,) * 3
                )
                
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                                     vertex_normals=normals)
                
                # Apply mesh processing
                mesh = self._process_surface_mesh(mesh, decimation_factor)
                
                self.surface_meshes[phase_name] = mesh
                
                print(f"    Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
            except Exception as e:
                print(f"    Error generating mesh for {phase_name}: {e}")
                continue
        
        return self.surface_meshes
    
    def generate_volume_meshes(self, target_edge_length: float = None) -> Dict[str, pv.UnstructuredGrid]:
        """
        Generate volume meshes (tetrahedral) for each phase.
        
        Parameters:
        -----------
        target_edge_length : float
            Target edge length for mesh elements in micrometers
            
        Returns:
        --------
        dict
            Dictionary of phase names to PyVista UnstructuredGrid objects
        """
        if target_edge_length is None:
            target_edge_length = self.voxel_size * 2.0
        
        print("Generating volume meshes for each phase...")
        
        self.volume_meshes = {}
        
        for phase_name, phase_id in self.phases.items():
            if phase_name in ['pore', 'interface']:  # Skip these phases
                continue
                
            print(f"  Processing {phase_name} phase...")
            
            # Extract phase mask
            phase_mask = (self.microstructure == phase_id)
            
            if not np.any(phase_mask):
                continue
            
            try:
                # Generate volume mesh using voxel-to-tetrahedral conversion
                volume_mesh = self._generate_tetrahedral_mesh(phase_mask, target_edge_length)
                
                if volume_mesh is not None:
                    self.volume_meshes[phase_name] = volume_mesh
                    print(f"    Generated volume mesh: {volume_mesh.n_points} points, {volume_mesh.n_cells} cells")
                
            except Exception as e:
                print(f"    Error generating volume mesh for {phase_name}: {e}")
                continue
        
        return self.volume_meshes
    
    def generate_interface_mesh(self, phase1: str, phase2: str, 
                              mesh_resolution: float = None) -> Optional[trimesh.Trimesh]:
        """
        Generate high-quality mesh of the interface between two phases.
        
        Parameters:
        -----------
        phase1 : str
            Name of first phase
        phase2 : str
            Name of second phase
        mesh_resolution : float
            Target mesh resolution in micrometers
            
        Returns:
        --------
        trimesh.Trimesh or None
            Interface mesh
        """
        if mesh_resolution is None:
            mesh_resolution = self.voxel_size
        
        print(f"Generating interface mesh between {phase1} and {phase2}...")
        
        if phase1 not in self.phases or phase2 not in self.phases:
            print("Error: Invalid phase names")
            return None
        
        # Get phase masks
        mask1 = (self.microstructure == self.phases[phase1])
        mask2 = (self.microstructure == self.phases[phase2])
        
        # Find interface region
        interface_mask = self._find_interface_region(mask1, mask2)
        
        if not np.any(interface_mask):
            print("No interface found between phases")
            return None
        
        try:
            # Generate interface mesh
            vertices, faces, normals, values = measure.marching_cubes(
                interface_mask.astype(float), level=0.5, 
                spacing=(self.voxel_size,) * 3
            )
            
            # Create and process mesh
            interface_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                                           vertex_normals=normals)
            
            # Refine mesh to target resolution
            interface_mesh = self._refine_interface_mesh(interface_mesh, mesh_resolution)
            
            interface_key = f"{phase1}_{phase2}"
            self.interface_meshes[interface_key] = interface_mesh
            
            print(f"Generated interface mesh: {len(interface_mesh.vertices)} vertices, {len(interface_mesh.faces)} faces")
            
            return interface_mesh
            
        except Exception as e:
            print(f"Error generating interface mesh: {e}")
            return None
    
    def generate_pore_network_mesh(self, min_pore_size: float = None) -> Optional[pv.UnstructuredGrid]:
        """
        Generate a mesh representing the pore network for transport analysis.
        
        Parameters:
        -----------
        min_pore_size : float
            Minimum pore size to include in micrometers
            
        Returns:
        --------
        pv.UnstructuredGrid or None
            Pore network mesh
        """
        if min_pore_size is None:
            min_pore_size = self.voxel_size * 2
        
        print("Generating pore network mesh...")
        
        # Extract pore phase
        pore_mask = (self.microstructure == self.phases['pore'])
        
        if not np.any(pore_mask):
            print("No pore phase found")
            return None
        
        # Remove small isolated pores
        min_pore_voxels = int((min_pore_size / self.voxel_size) ** 3)
        cleaned_pores = morphology.remove_small_objects(pore_mask, min_size=min_pore_voxels)
        
        if not np.any(cleaned_pores):
            print("No pores above minimum size threshold")
            return None
        
        try:
            # Generate pore network mesh
            pore_mesh = self._generate_pore_network(cleaned_pores)
            
            if pore_mesh is not None:
                print(f"Generated pore network mesh: {pore_mesh.n_points} points, {pore_mesh.n_cells} cells")
            
            return pore_mesh
            
        except Exception as e:
            print(f"Error generating pore network mesh: {e}")
            return None
    
    def _process_surface_mesh(self, mesh: trimesh.Trimesh, 
                            decimation_factor: float) -> trimesh.Trimesh:
        """Process and clean surface mesh."""
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        # Remove duplicate vertices
        mesh.merge_vertices()
        
        # Smooth mesh (if method exists)
        try:
            mesh = mesh.smoothed()
        except AttributeError:
            # Alternative smoothing method or skip if not available
            pass
        
        # Decimate mesh if requested
        if decimation_factor < 1.0:
            target_faces = int(len(mesh.faces) * decimation_factor)
            mesh = mesh.simplify_quadric_decimation(target_faces)
        
        # Ensure mesh is watertight
        if not mesh.is_watertight:
            try:
                mesh.fill_holes()
            except:
                pass  # Continue even if hole filling fails
        
        return mesh
    
    def _generate_tetrahedral_mesh(self, phase_mask: np.ndarray, 
                                 target_edge_length: float) -> Optional[pv.UnstructuredGrid]:
        """Generate tetrahedral mesh from phase mask."""
        
        try:
            # Convert mask to surface mesh first
            vertices, faces, _, _ = measure.marching_cubes(
                phase_mask.astype(float), level=0.5, 
                spacing=(self.voxel_size,) * 3
            )
            
            # Create PyVista mesh from surface
            surface_mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces])
            
            # Generate tetrahedral mesh using PyVista's delaunay_3d
            # This is a simplified approach - for production use, consider TetGen or CGAL
            
            # Create bounding box points
            bounds = surface_mesh.bounds
            
            # Create a simple tetrahedral mesh within bounds
            # This is a placeholder - in practice, you'd use more sophisticated meshing
            
            # For now, create a structured tetrahedral mesh
            n_points_per_dim = max(10, int((bounds[1] - bounds[0]) / target_edge_length))
            
            x = np.linspace(bounds[0], bounds[1], n_points_per_dim)
            y = np.linspace(bounds[2], bounds[3], n_points_per_dim)
            z = np.linspace(bounds[4], bounds[5], n_points_per_dim)
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            # Filter points inside the phase
            inside_points = []
            for point in points:
                # Convert to voxel coordinates
                vox_x = int((point[0] / self.voxel_size))
                vox_y = int((point[1] / self.voxel_size))
                vox_z = int((point[2] / self.voxel_size))
                
                if (0 <= vox_x < self.dimensions[0] and
                    0 <= vox_y < self.dimensions[1] and
                    0 <= vox_z < self.dimensions[2] and
                    phase_mask[vox_x, vox_y, vox_z]):
                    inside_points.append(point)
            
            if len(inside_points) < 4:
                return None
            
            inside_points = np.array(inside_points)
            
            # Create PyVista point cloud and generate Delaunay triangulation
            point_cloud = pv.PolyData(inside_points)
            volume_mesh = point_cloud.delaunay_3d()
            
            return volume_mesh
            
        except Exception as e:
            print(f"Error in tetrahedral mesh generation: {e}")
            return None
    
    def _find_interface_region(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Find interface region between two phases."""
        
        # Dilate both masks slightly
        mask1_dilated = morphology.binary_dilation(mask1, morphology.ball(1))
        mask2_dilated = morphology.binary_dilation(mask2, morphology.ball(1))
        
        # Interface is where dilated regions overlap
        interface = mask1_dilated & mask2_dilated
        
        return interface
    
    def _refine_interface_mesh(self, mesh: trimesh.Trimesh, 
                             target_resolution: float) -> trimesh.Trimesh:
        """Refine interface mesh to target resolution."""
        
        # Calculate current edge lengths
        edges = mesh.edges_unique
        edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        
        mean_edge_length = np.mean(edge_lengths)
        
        # If mesh is too coarse, subdivide
        if mean_edge_length > target_resolution * 2:
            mesh = mesh.subdivide()
        
        # If mesh is too fine, decimate
        elif mean_edge_length < target_resolution * 0.5:
            target_faces = int(len(mesh.faces) * 0.5)
            mesh = mesh.simplify_quadric_decimation(target_faces)
        
        return mesh
    
    def _generate_pore_network(self, pore_mask: np.ndarray) -> Optional[pv.UnstructuredGrid]:
        """Generate pore network mesh for transport analysis."""
        
        try:
            # Use distance transform to find pore centers and radii
            distance_transform = ndi.distance_transform_edt(pore_mask)
            
            # Find local maxima as pore centers
            from scipy.ndimage import maximum_filter
            
            local_maxima = (distance_transform == maximum_filter(distance_transform, size=5))
            local_maxima = local_maxima & (distance_transform > 2)  # Minimum radius threshold
            
            pore_centers = np.array(np.where(local_maxima)).T
            pore_radii = distance_transform[local_maxima]
            
            if len(pore_centers) == 0:
                return None
            
            # Convert to physical coordinates
            pore_centers_physical = pore_centers * self.voxel_size
            pore_radii_physical = pore_radii * self.voxel_size
            
            # Create network connections using Delaunay triangulation
            from scipy.spatial import Delaunay
            
            tri = Delaunay(pore_centers_physical)
            
            # Create PyVista mesh
            points = pore_centers_physical
            
            # Create cells (tetrahedra from Delaunay triangulation)
            cells = []
            cell_types = []
            
            for simplex in tri.simplices:
                cells.extend([4] + simplex.tolist())  # 4 vertices per tetrahedron
                cell_types.append(vtk.VTK_TETRA)
            
            # Create unstructured grid
            pore_mesh = pv.UnstructuredGrid(cells, cell_types, points)
            
            # Add pore radii as point data
            pore_mesh.point_data['pore_radius'] = pore_radii_physical
            
            return pore_mesh
            
        except Exception as e:
            print(f"Error in pore network generation: {e}")
            return None
    
    def analyze_mesh_quality(self) -> Dict:
        """Analyze quality metrics for generated meshes."""
        
        print("Analyzing mesh quality...")
        
        quality_metrics = {}
        
        # Surface mesh quality
        for phase_name, mesh in self.surface_meshes.items():
            quality_metrics[f"{phase_name}_surface"] = self._analyze_surface_mesh_quality(mesh)
        
        # Volume mesh quality
        for phase_name, mesh in self.volume_meshes.items():
            quality_metrics[f"{phase_name}_volume"] = self._analyze_volume_mesh_quality(mesh)
        
        # Interface mesh quality
        for interface_name, mesh in self.interface_meshes.items():
            quality_metrics[f"{interface_name}_interface"] = self._analyze_surface_mesh_quality(mesh)
        
        self.mesh_quality = quality_metrics
        return quality_metrics
    
    def _analyze_surface_mesh_quality(self, mesh: trimesh.Trimesh) -> Dict:
        """Analyze quality of surface mesh."""
        
        quality = {}
        
        # Basic statistics
        quality['n_vertices'] = len(mesh.vertices)
        quality['n_faces'] = len(mesh.faces)
        quality['surface_area'] = mesh.area
        quality['volume'] = mesh.volume if mesh.is_watertight else 0.0
        
        # Edge length statistics
        edges = mesh.edges_unique
        edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        
        quality['edge_length_mean'] = np.mean(edge_lengths)
        quality['edge_length_std'] = np.std(edge_lengths)
        quality['edge_length_min'] = np.min(edge_lengths)
        quality['edge_length_max'] = np.max(edge_lengths)
        
        # Face quality (aspect ratios)
        face_areas = mesh.area_faces
        face_perimeters = []
        
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            perimeter = (np.linalg.norm(v1 - v0) + 
                        np.linalg.norm(v2 - v1) + 
                        np.linalg.norm(v0 - v2))
            face_perimeters.append(perimeter)
        
        face_perimeters = np.array(face_perimeters)
        
        # Shape regularity (4π * area / perimeter²)
        shape_regularity = 4 * np.pi * face_areas / (face_perimeters ** 2)
        
        quality['face_regularity_mean'] = np.mean(shape_regularity)
        quality['face_regularity_min'] = np.min(shape_regularity)
        
        # Mesh topology
        quality['is_watertight'] = mesh.is_watertight
        quality['is_winding_consistent'] = mesh.is_winding_consistent
        quality['euler_number'] = mesh.euler_number
        
        return quality
    
    def _analyze_volume_mesh_quality(self, mesh: pv.UnstructuredGrid) -> Dict:
        """Analyze quality of volume mesh."""
        
        quality = {}
        
        # Basic statistics
        quality['n_points'] = mesh.n_points
        quality['n_cells'] = mesh.n_cells
        quality['volume'] = mesh.volume
        
        # Cell quality metrics
        if mesh.n_cells > 0:
            # Calculate cell volumes and aspect ratios
            cell_volumes = []
            aspect_ratios = []
            
            for i in range(mesh.n_cells):
                cell = mesh.get_cell(i)
                
                if cell.type == vtk.VTK_TETRA:
                    # Tetrahedral cell
                    points = cell.points
                    
                    # Volume calculation for tetrahedron
                    v1 = points[1] - points[0]
                    v2 = points[2] - points[0]
                    v3 = points[3] - points[0]
                    
                    volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
                    cell_volumes.append(volume)
                    
                    # Aspect ratio (simplified)
                    edge_lengths = [
                        np.linalg.norm(points[1] - points[0]),
                        np.linalg.norm(points[2] - points[0]),
                        np.linalg.norm(points[3] - points[0]),
                        np.linalg.norm(points[2] - points[1]),
                        np.linalg.norm(points[3] - points[1]),
                        np.linalg.norm(points[3] - points[2])
                    ]
                    
                    aspect_ratio = max(edge_lengths) / min(edge_lengths)
                    aspect_ratios.append(aspect_ratio)
            
            if cell_volumes:
                quality['cell_volume_mean'] = np.mean(cell_volumes)
                quality['cell_volume_std'] = np.std(cell_volumes)
                quality['cell_volume_min'] = np.min(cell_volumes)
                quality['cell_volume_max'] = np.max(cell_volumes)
            
            if aspect_ratios:
                quality['aspect_ratio_mean'] = np.mean(aspect_ratios)
                quality['aspect_ratio_max'] = np.max(aspect_ratios)
                quality['well_shaped_cells_fraction'] = np.sum(np.array(aspect_ratios) < 3.0) / len(aspect_ratios)
        
        return quality
    
    def export_meshes(self, output_dir: str = "sofc_microstructure/data/meshes"):
        """Export all generated meshes in multiple formats."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting meshes to {output_dir}...")
        
        # Export surface meshes
        for phase_name, mesh in self.surface_meshes.items():
            # STL format (widely supported)
            stl_path = os.path.join(output_dir, f"{phase_name}_surface.stl")
            mesh.export(stl_path)
            
            # OBJ format
            obj_path = os.path.join(output_dir, f"{phase_name}_surface.obj")
            mesh.export(obj_path)
            
            # PLY format (with vertex colors if available)
            ply_path = os.path.join(output_dir, f"{phase_name}_surface.ply")
            mesh.export(ply_path)
            
            print(f"  Exported {phase_name} surface mesh")
        
        # Export volume meshes
        for phase_name, mesh in self.volume_meshes.items():
            # VTK format (native PyVista format)
            vtk_path = os.path.join(output_dir, f"{phase_name}_volume.vtk")
            mesh.save(vtk_path)
            
            # VTU format (unstructured grid)
            vtu_path = os.path.join(output_dir, f"{phase_name}_volume.vtu")
            mesh.save(vtu_path)
            
            print(f"  Exported {phase_name} volume mesh")
        
        # Export interface meshes
        for interface_name, mesh in self.interface_meshes.items():
            stl_path = os.path.join(output_dir, f"{interface_name}_interface.stl")
            mesh.export(stl_path)
            
            print(f"  Exported {interface_name} interface mesh")
        
        # Export mesh quality report
        if self.mesh_quality:
            quality_path = os.path.join(output_dir, "mesh_quality_report.json")
            with open(quality_path, 'w') as f:
                json.dump(self.mesh_quality, f, indent=2, default=str)
            
            print(f"  Exported mesh quality report")
        
        print("Mesh export completed!")
    
    def visualize_meshes(self, save_path: str = None):
        """Create visualization of generated meshes."""
        
        # Use PyVista for 3D visualization
        plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))
        
        # Plot 1: Surface meshes
        plotter.subplot(0, 0)
        colors = ['red', 'blue', 'green', 'yellow']
        
        for i, (phase_name, mesh) in enumerate(self.surface_meshes.items()):
            # Convert trimesh to pyvista
            pv_mesh = pv.PolyData(mesh.vertices, np.c_[np.full(len(mesh.faces), 3), mesh.faces])
            plotter.add_mesh(pv_mesh, color=colors[i % len(colors)], 
                           opacity=0.7, label=phase_name)
        
        plotter.add_title("Surface Meshes")
        plotter.add_legend()
        
        # Plot 2: Volume meshes
        plotter.subplot(0, 1)
        
        for i, (phase_name, mesh) in enumerate(self.volume_meshes.items()):
            plotter.add_mesh(mesh, color=colors[i % len(colors)], 
                           opacity=0.3, label=phase_name)
        
        plotter.add_title("Volume Meshes")
        plotter.add_legend()
        
        # Plot 3: Interface meshes
        plotter.subplot(1, 0)
        
        for i, (interface_name, mesh) in enumerate(self.interface_meshes.items()):
            pv_mesh = pv.PolyData(mesh.vertices, np.c_[np.full(len(mesh.faces), 3), mesh.faces])
            plotter.add_mesh(pv_mesh, color=colors[i % len(colors)], 
                           label=interface_name)
        
        plotter.add_title("Interface Meshes")
        plotter.add_legend()
        
        # Plot 4: Mesh quality summary
        plotter.subplot(1, 1)
        plotter.add_text("Mesh Quality Summary", font_size=16)
        
        if self.mesh_quality:
            text_content = []
            for mesh_name, quality in self.mesh_quality.items():
                if 'n_vertices' in quality:
                    text_content.append(f"{mesh_name}:")
                    text_content.append(f"  Vertices: {quality['n_vertices']}")
                    text_content.append(f"  Faces: {quality.get('n_faces', 'N/A')}")
                    if 'edge_length_mean' in quality:
                        text_content.append(f"  Avg Edge: {quality['edge_length_mean']:.3f}")
                    text_content.append("")
            
            full_text = "\n".join(text_content[:20])  # Limit text length
            plotter.add_text(full_text, position='upper_left', font_size=10)
        
        if save_path:
            plotter.screenshot(save_path)
            print(f"Mesh visualization saved to {save_path}")
        
        plotter.show()


# Example usage
if __name__ == "__main__":
    print("SOFC Mesh Generator module loaded successfully!")
    print("Use this module to generate computational meshes from microstructure data.")