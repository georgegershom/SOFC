"""
Mesh Generator for SOFC Geometry

This module creates 3D meshes for SOFC plates with multiple layers
(anode, electrolyte, cathode) for finite element analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import meshio
import pyvista as pv
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


@dataclass
class LayerGeometry:
    """Geometry definition for a single SOFC layer."""
    thickness: float  # m
    material_id: int
    name: str
    z_bottom: float = 0.0
    z_top: float = 0.0


@dataclass
class SOFCGeometry:
    """Complete SOFC geometry definition."""
    length: float  # m
    width: float  # m
    layers: List[LayerGeometry]
    total_thickness: float = 0.0


class SOFCMeshGenerator:
    """Generates 3D meshes for SOFC plates."""
    
    def __init__(self, mesh_resolution: float = 1e-3):
        """Initialize mesh generator.
        
        Args:
            mesh_resolution: Target element size in meters
        """
        self.mesh_resolution = mesh_resolution
        self.geometry = None
        self.mesh = None
    
    def create_geometry(self, doe_parameters: Dict[str, Any]) -> SOFCGeometry:
        """Create SOFC geometry from DOE parameters."""
        
        # Extract geometric parameters
        length = doe_parameters.get('geometry.length', 100e-3)  # Convert mm to m
        width = doe_parameters.get('geometry.width', 100e-3)
        
        anode_thickness = doe_parameters.get('geometry.anode_thickness', 500e-6)  # Convert μm to m
        electrolyte_thickness = doe_parameters.get('geometry.electrolyte_thickness', 15e-6)
        cathode_thickness = doe_parameters.get('geometry.cathode_thickness', 40e-6)
        
        # Create layer definitions
        layers = []
        z_current = 0.0
        
        # Anode layer (bottom)
        anode_layer = LayerGeometry(
            thickness=anode_thickness,
            material_id=1,
            name="anode",
            z_bottom=z_current,
            z_top=z_current + anode_thickness
        )
        layers.append(anode_layer)
        z_current += anode_thickness
        
        # Electrolyte layer (middle)
        electrolyte_layer = LayerGeometry(
            thickness=electrolyte_thickness,
            material_id=2,
            name="electrolyte",
            z_bottom=z_current,
            z_top=z_current + electrolyte_thickness
        )
        layers.append(electrolyte_layer)
        z_current += electrolyte_thickness
        
        # Cathode layer (top)
        cathode_layer = LayerGeometry(
            thickness=cathode_thickness,
            material_id=3,
            name="cathode",
            z_bottom=z_current,
            z_top=z_current + cathode_thickness
        )
        layers.append(cathode_layer)
        z_current += cathode_thickness
        
        # Create geometry object
        geometry = SOFCGeometry(
            length=length,
            width=width,
            layers=layers,
            total_thickness=z_current
        )
        
        self.geometry = geometry
        return geometry
    
    def generate_mesh(self, doe_parameters: Dict[str, Any]) -> pv.UnstructuredGrid:
        """Generate 3D mesh for SOFC geometry."""
        
        # Create geometry if not already done
        if self.geometry is None:
            self.create_geometry(doe_parameters)
        
        # Generate structured mesh
        mesh = self._generate_structured_mesh()
        
        # Add material IDs and other properties
        mesh = self._add_mesh_properties(mesh)
        
        self.mesh = mesh
        return mesh
    
    def _generate_structured_mesh(self) -> pv.UnstructuredGrid:
        """Generate structured hexahedral mesh."""
        
        # Calculate number of elements in each direction
        nx = max(2, int(self.geometry.length / self.mesh_resolution))
        ny = max(2, int(self.geometry.width / self.mesh_resolution))
        
        # Calculate number of elements in z-direction for each layer
        nz_layers = []
        for layer in self.geometry.layers:
            nz_layer = max(1, int(layer.thickness / self.mesh_resolution))
            nz_layers.append(nz_layer)
        
        total_nz = sum(nz_layers)
        
        # Create coordinate arrays
        x = np.linspace(0, self.geometry.length, nx + 1)
        y = np.linspace(0, self.geometry.width, ny + 1)
        
        # Create z coordinates with layer boundaries
        z_coords = [0.0]
        for i, layer in enumerate(self.geometry.layers):
            z_layer = np.linspace(layer.z_bottom, layer.z_top, nz_layers[i] + 1)[1:]
            z_coords.extend(z_layer)
        z = np.array(z_coords)
        
        # Create mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten coordinates
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Create hexahedral elements
        cells = []
        material_ids = []
        
        z_offset = 0
        for layer_idx, (layer, nz_layer) in enumerate(zip(self.geometry.layers, nz_layers)):
            for k in range(nz_layer):
                for j in range(ny):
                    for i in range(nx):
                        # Node indices for hexahedral element
                        n0 = (i + 0) + (j + 0) * (nx + 1) + (k + z_offset + 0) * (nx + 1) * (ny + 1)
                        n1 = (i + 1) + (j + 0) * (nx + 1) + (k + z_offset + 0) * (nx + 1) * (ny + 1)
                        n2 = (i + 1) + (j + 1) * (nx + 1) + (k + z_offset + 0) * (nx + 1) * (ny + 1)
                        n3 = (i + 0) + (j + 1) * (nx + 1) + (k + z_offset + 0) * (nx + 1) * (ny + 1)
                        n4 = (i + 0) + (j + 0) * (nx + 1) + (k + z_offset + 1) * (nx + 1) * (ny + 1)
                        n5 = (i + 1) + (j + 0) * (nx + 1) + (k + z_offset + 1) * (nx + 1) * (ny + 1)
                        n6 = (i + 1) + (j + 1) * (nx + 1) + (k + z_offset + 1) * (nx + 1) * (ny + 1)
                        n7 = (i + 0) + (j + 1) * (nx + 1) + (k + z_offset + 1) * (nx + 1) * (ny + 1)
                        
                        # Hexahedral element connectivity
                        cell = [8, n0, n1, n2, n3, n4, n5, n6, n7]
                        cells.append(cell)
                        material_ids.append(layer.material_id)
            
            z_offset += nz_layer
        
        # Convert to PyVista format
        cells_array = np.array(cells)
        cell_types = np.full(len(cells), 12)  # VTK_HEXAHEDRON = 12
        
        # Create unstructured grid
        mesh = pv.UnstructuredGrid(cells_array, cell_types, points)
        
        # Add material IDs as cell data
        mesh.cell_data['material_id'] = np.array(material_ids)
        
        return mesh
    
    def _add_mesh_properties(self, mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
        """Add additional properties to the mesh."""
        
        # Add layer names
        layer_names = []
        for material_id in mesh.cell_data['material_id']:
            for layer in self.geometry.layers:
                if layer.material_id == material_id:
                    layer_names.append(layer.name)
                    break
        
        mesh.cell_data['layer_name'] = layer_names
        
        # Add element volumes
        volumes = []
        for i in range(mesh.n_cells):
            cell = mesh.get_cell(i)
            volume = self._calculate_hex_volume(cell.points)
            volumes.append(volume)
        
        mesh.cell_data['volume'] = np.array(volumes)
        
        # Add coordinate information to points
        mesh.point_data['x'] = mesh.points[:, 0]
        mesh.point_data['y'] = mesh.points[:, 1]
        mesh.point_data['z'] = mesh.points[:, 2]
        
        # Add surface flags
        self._add_surface_flags(mesh)
        
        return mesh
    
    def _calculate_hex_volume(self, points: np.ndarray) -> float:
        """Calculate volume of hexahedral element."""
        # Simplified volume calculation for regular hexahedron
        # For irregular elements, use more sophisticated methods
        
        if len(points) != 8:
            return 0.0
        
        # Calculate edge vectors
        dx = np.abs(points[1, 0] - points[0, 0])
        dy = np.abs(points[3, 1] - points[0, 1])
        dz = np.abs(points[4, 2] - points[0, 2])
        
        return dx * dy * dz
    
    def _add_surface_flags(self, mesh: pv.UnstructuredGrid):
        """Add flags for surface nodes and elements."""
        
        # Find boundary nodes
        bounds = mesh.bounds
        tol = 1e-10
        
        # Point flags
        on_x_min = np.abs(mesh.points[:, 0] - bounds[0]) < tol
        on_x_max = np.abs(mesh.points[:, 0] - bounds[1]) < tol
        on_y_min = np.abs(mesh.points[:, 1] - bounds[2]) < tol
        on_y_max = np.abs(mesh.points[:, 1] - bounds[3]) < tol
        on_z_min = np.abs(mesh.points[:, 2] - bounds[4]) < tol
        on_z_max = np.abs(mesh.points[:, 2] - bounds[5]) < tol
        
        mesh.point_data['on_bottom'] = on_z_min
        mesh.point_data['on_top'] = on_z_max
        mesh.point_data['on_sides'] = on_x_min | on_x_max | on_y_min | on_y_max
        mesh.point_data['on_boundary'] = on_x_min | on_x_max | on_y_min | on_y_max | on_z_min | on_z_max
    
    def generate_tetrahedral_mesh(self, doe_parameters: Dict[str, Any]) -> pv.UnstructuredGrid:
        """Generate tetrahedral mesh (alternative to hexahedral)."""
        
        # Create geometry if not already done
        if self.geometry is None:
            self.create_geometry(doe_parameters)
        
        # Generate point cloud
        points = self._generate_point_cloud()
        
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # Convert to PyVista format
        cells = []
        for simplex in tri.simplices:
            cell = [4] + list(simplex)  # 4 = number of points in tetrahedron
            cells.append(cell)
        
        cells_array = np.array(cells)
        cell_types = np.full(len(tri.simplices), 10)  # VTK_TETRA = 10
        
        mesh = pv.UnstructuredGrid(cells_array, cell_types, points)
        
        # Add material IDs
        material_ids = self._assign_material_ids_tet(mesh)
        mesh.cell_data['material_id'] = material_ids
        
        return mesh
    
    def _generate_point_cloud(self) -> np.ndarray:
        """Generate point cloud for tetrahedral meshing."""
        
        # Calculate number of points in each direction
        nx = max(5, int(self.geometry.length / self.mesh_resolution))
        ny = max(5, int(self.geometry.width / self.mesh_resolution))
        
        points = []
        
        # Generate points for each layer
        for layer in self.geometry.layers:
            nz = max(2, int(layer.thickness / self.mesh_resolution))
            
            # Regular grid points
            x = np.linspace(0, self.geometry.length, nx)
            y = np.linspace(0, self.geometry.width, ny)
            z = np.linspace(layer.z_bottom, layer.z_top, nz)
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            layer_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            points.append(layer_points)
        
        return np.vstack(points)
    
    def _assign_material_ids_tet(self, mesh: pv.UnstructuredGrid) -> np.ndarray:
        """Assign material IDs to tetrahedral elements."""
        
        material_ids = np.zeros(mesh.n_cells, dtype=int)
        
        # Calculate element centroids
        centroids = []
        for i in range(mesh.n_cells):
            cell = mesh.get_cell(i)
            centroid = np.mean(cell.points, axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Assign material IDs based on z-coordinate
        for i, centroid in enumerate(centroids):
            z = centroid[2]
            
            for layer in self.geometry.layers:
                if layer.z_bottom <= z <= layer.z_top:
                    material_ids[i] = layer.material_id
                    break
        
        return material_ids
    
    def refine_mesh(self, mesh: pv.UnstructuredGrid, refinement_factor: int = 2) -> pv.UnstructuredGrid:
        """Refine mesh by subdividing elements."""
        
        # This is a simplified refinement - in practice, use proper mesh refinement algorithms
        # For now, return the original mesh
        print(f"Warning: Mesh refinement not fully implemented. Returning original mesh.")
        return mesh
    
    def export_mesh(self, filepath: Union[str, Path], format: str = "vtk"):
        """Export mesh to file."""
        
        if self.mesh is None:
            raise ValueError("No mesh to export. Generate mesh first.")
        
        filepath = Path(filepath)
        
        if format.lower() == "vtk":
            self.mesh.save(str(filepath.with_suffix('.vtk')))
        elif format.lower() == "vtu":
            self.mesh.save(str(filepath.with_suffix('.vtu')))
        elif format.lower() == "msh":
            # Convert to meshio format and save as Gmsh
            meshio_mesh = self._convert_to_meshio()
            meshio.write(str(filepath.with_suffix('.msh')), meshio_mesh)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _convert_to_meshio(self) -> meshio.Mesh:
        """Convert PyVista mesh to meshio format."""
        
        if self.mesh is None:
            raise ValueError("No mesh to convert.")
        
        # Extract cells by type
        cells = []
        cell_data = {}
        
        # Get hexahedral cells
        hex_cells = []
        for i in range(self.mesh.n_cells):
            cell = self.mesh.get_cell(i)
            if cell.type == 12:  # VTK_HEXAHEDRON
                hex_cells.append(cell.point_ids)
        
        if hex_cells:
            cells.append(("hexahedron", np.array(hex_cells)))
            cell_data["material_id"] = [self.mesh.cell_data["material_id"]]
        
        # Create meshio mesh
        meshio_mesh = meshio.Mesh(
            points=self.mesh.points,
            cells=cells,
            point_data={k: v for k, v in self.mesh.point_data.items()},
            cell_data=cell_data
        )
        
        return meshio_mesh
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get mesh quality statistics."""
        
        if self.mesh is None:
            return {}
        
        stats = {
            'n_points': self.mesh.n_points,
            'n_cells': self.mesh.n_cells,
            'bounds': self.mesh.bounds,
            'volume': self.mesh.volume if hasattr(self.mesh, 'volume') else 0.0,
        }
        
        # Element quality metrics
        if 'volume' in self.mesh.cell_data:
            volumes = self.mesh.cell_data['volume']
            stats['min_volume'] = np.min(volumes)
            stats['max_volume'] = np.max(volumes)
            stats['avg_volume'] = np.mean(volumes)
            stats['volume_ratio'] = np.max(volumes) / np.min(volumes) if np.min(volumes) > 0 else np.inf
        
        # Layer statistics
        if 'material_id' in self.mesh.cell_data:
            unique_materials, counts = np.unique(self.mesh.cell_data['material_id'], return_counts=True)
            stats['layers'] = dict(zip(unique_materials, counts))
        
        return stats


def create_mesh_generator(mesh_resolution: float = 1e-3) -> SOFCMeshGenerator:
    """Factory function to create mesh generator."""
    return SOFCMeshGenerator(mesh_resolution)


if __name__ == "__main__":
    # Test mesh generation
    doe_params = {
        'geometry.length': 100e-3,  # 100 mm
        'geometry.width': 100e-3,   # 100 mm
        'geometry.anode_thickness': 500e-6,      # 500 μm
        'geometry.electrolyte_thickness': 15e-6,  # 15 μm
        'geometry.cathode_thickness': 40e-6,      # 40 μm
    }
    
    mesh_gen = create_mesh_generator(mesh_resolution=2e-3)  # 2 mm elements
    
    print("Creating geometry...")
    geometry = mesh_gen.create_geometry(doe_params)
    print(f"Geometry: {geometry.length*1000:.1f} x {geometry.width*1000:.1f} x {geometry.total_thickness*1e6:.1f} mm³")
    
    print("Generating mesh...")
    mesh = mesh_gen.generate_mesh(doe_params)
    
    stats = mesh_gen.get_mesh_statistics()
    print(f"Mesh statistics: {stats}")
    
    print("Mesh generation completed successfully!")