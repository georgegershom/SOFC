"""
SOFC Mesh Generator

Generates structured hexahedral meshes for SOFC cells with appropriate
refinement for stress analysis and warp measurement.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import meshio
import pyvista as pv


@dataclass
class MeshParameters:
    """Parameters for mesh generation"""
    # Cell dimensions (mm)
    cell_length: float = 100.0
    cell_width: float = 100.0
    
    # Layer thicknesses (μm)
    electrolyte_thickness: float = 150.0
    anode_thickness: float = 300.0
    cathode_thickness: float = 50.0
    interconnect_thickness: float = 2000.0  # 2mm
    
    # Mesh density parameters
    elements_per_mm: float = 10.0  # Base mesh density
    electrolyte_elements_z: int = 8  # Elements through electrolyte thickness
    electrode_elements_z: int = 4   # Elements through electrode thickness
    interconnect_elements_z: int = 12  # Elements through interconnect thickness
    
    # Refinement parameters
    edge_refinement_factor: float = 2.0  # Refinement at edges
    interface_refinement_factor: float = 1.5  # Refinement at interfaces
    
    def __post_init__(self):
        """Convert thicknesses to mm for mesh generation"""
        self.electrolyte_thickness_mm = self.electrolyte_thickness / 1000.0
        self.anode_thickness_mm = self.anode_thickness / 1000.0
        self.cathode_thickness_mm = self.cathode_thickness / 1000.0
        self.interconnect_thickness_mm = self.interconnect_thickness / 1000.0


class SOFCMeshGenerator:
    """Generates structured hexahedral meshes for SOFC cells"""
    
    def __init__(self, mesh_params: MeshParameters):
        self.params = mesh_params
        self.mesh = None
        self.element_groups = {}
        self.node_groups = {}
    
    def generate_mesh(self) -> Dict:
        """Generate the complete SOFC mesh"""
        # Calculate mesh dimensions
        mesh_dims = self._calculate_mesh_dimensions()
        
        # Generate node coordinates
        nodes = self._generate_nodes(mesh_dims)
        
        # Generate element connectivity
        elements, element_groups = self._generate_elements(mesh_dims)
        
        # Create mesh data structure
        self.mesh = {
            'nodes': nodes,
            'elements': elements,
            'element_groups': element_groups,
            'mesh_parameters': self.params,
            'mesh_dimensions': mesh_dims
        }
        
        return self.mesh
    
    def _calculate_mesh_dimensions(self) -> Dict[str, int]:
        """Calculate mesh dimensions in each direction"""
        # X and Y directions (in-plane)
        nx = int(self.params.cell_length * self.params.elements_per_mm)
        ny = int(self.params.cell_width * self.params.elements_per_mm)
        
        # Z direction (through thickness)
        nz_electrolyte = self.params.electrolyte_elements_z
        nz_anode = self.params.electrode_elements_z
        nz_cathode = self.params.electrode_elements_z
        nz_interconnect = self.params.interconnect_elements_z
        
        nz_total = nz_anode + nz_electrolyte + nz_cathode + nz_interconnect
        
        return {
            'nx': nx,
            'ny': ny,
            'nz_total': nz_total,
            'nz_anode': nz_anode,
            'nz_electrolyte': nz_electrolyte,
            'nz_cathode': nz_cathode,
            'nz_interconnect': nz_interconnect
        }
    
    def _generate_nodes(self, mesh_dims: Dict[str, int]) -> np.ndarray:
        """Generate node coordinates"""
        nx, ny, nz_total = mesh_dims['nx'], mesh_dims['ny'], mesh_dims['nz_total']
        
        # Create coordinate arrays
        x = np.linspace(0, self.params.cell_length, nx + 1)
        y = np.linspace(0, self.params.cell_width, ny + 1)
        
        # Z coordinates with layer-specific spacing
        z_coords = self._generate_z_coordinates(mesh_dims)
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z_coords, indexing='ij')
        
        # Flatten and stack
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return nodes
    
    def _generate_z_coordinates(self, mesh_dims: Dict[str, int]) -> np.ndarray:
        """Generate Z coordinates with appropriate layer spacing"""
        z_coords = []
        z_current = 0.0
        
        # Anode layer
        nz_anode = mesh_dims['nz_anode']
        anode_z = np.linspace(z_current, z_current + self.params.anode_thickness_mm, nz_anode + 1)
        z_coords.extend(anode_z[:-1])  # Exclude last point to avoid duplication
        z_current += self.params.anode_thickness_mm
        
        # Electrolyte layer
        nz_electrolyte = mesh_dims['nz_electrolyte']
        electrolyte_z = np.linspace(z_current, z_current + self.params.electrolyte_thickness_mm, nz_electrolyte + 1)
        z_coords.extend(electrolyte_z[:-1])
        z_current += self.params.electrolyte_thickness_mm
        
        # Cathode layer
        nz_cathode = mesh_dims['nz_cathode']
        cathode_z = np.linspace(z_current, z_current + self.params.cathode_thickness_mm, nz_cathode + 1)
        z_coords.extend(cathode_z[:-1])
        z_current += self.params.cathode_thickness_mm
        
        # Interconnect layer
        nz_interconnect = mesh_dims['nz_interconnect']
        interconnect_z = np.linspace(z_current, z_current + self.params.interconnect_thickness_mm, nz_interconnect + 1)
        z_coords.extend(interconnect_z)
        
        return np.array(z_coords)
    
    def _generate_elements(self, mesh_dims: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """Generate hexahedral elements and element groups"""
        nx, ny, nz_total = mesh_dims['nx'], mesh_dims['ny'], mesh_dims['nz_total']
        
        elements = []
        element_groups = {
            'anode': [],
            'electrolyte': [],
            'cathode': [],
            'interconnect': []
        }
        
        element_id = 0
        
        # Generate elements layer by layer
        z_start = 0
        z_end = mesh_dims['nz_anode']
        
        # Anode elements
        for k in range(z_start, z_end):
            for j in range(ny):
                for i in range(nx):
                    # 8-node hexahedral element connectivity
                    elem_nodes = self._get_hex_element_nodes(i, j, k, nx, ny, nz_total)
                    elements.append(elem_nodes)
                    element_groups['anode'].append(element_id)
                    element_id += 1
        
        # Electrolyte elements
        z_start = z_end
        z_end += mesh_dims['nz_electrolyte']
        for k in range(z_start, z_end):
            for j in range(ny):
                for i in range(nx):
                    elem_nodes = self._get_hex_element_nodes(i, j, k, nx, ny, nz_total)
                    elements.append(elem_nodes)
                    element_groups['electrolyte'].append(element_id)
                    element_id += 1
        
        # Cathode elements
        z_start = z_end
        z_end += mesh_dims['nz_cathode']
        for k in range(z_start, z_end):
            for j in range(ny):
                for i in range(nx):
                    elem_nodes = self._get_hex_element_nodes(i, j, k, nx, ny, nz_total)
                    elements.append(elem_nodes)
                    element_groups['cathode'].append(element_id)
                    element_id += 1
        
        # Interconnect elements
        z_start = z_end
        z_end += mesh_dims['nz_interconnect']
        for k in range(z_start, z_end):
            for j in range(ny):
                for i in range(nx):
                    elem_nodes = self._get_hex_element_nodes(i, j, k, nx, ny, nz_total)
                    elements.append(elem_nodes)
                    element_groups['interconnect'].append(element_id)
                    element_id += 1
        
        return np.array(elements), element_groups
    
    def _get_hex_element_nodes(self, i: int, j: int, k: int, nx: int, ny: int, nz: int) -> List[int]:
        """Get 8-node hexahedral element node connectivity"""
        # Node numbering convention (VTK style)
        n0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
        n1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + (i + 1)
        n2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + (i + 1)
        n3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
        n4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
        n5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + (i + 1)
        n6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + (i + 1)
        n7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
        
        return [n0, n1, n2, n3, n4, n5, n6, n7]
    
    def get_surface_nodes(self, surface: str) -> np.ndarray:
        """Get node indices for specific surfaces"""
        if self.mesh is None:
            raise ValueError("Mesh not generated yet")
        
        nodes = self.mesh['nodes']
        nx, ny, nz_total = (self.mesh['mesh_dimensions']['nx'] + 1,
                           self.mesh['mesh_dimensions']['ny'] + 1,
                           self.mesh['mesh_dimensions']['nz_total'] + 1)
        
        surface_nodes = []
        
        if surface == 'top':
            # Top surface (interconnect top)
            z_max = np.max(nodes[:, 2])
            surface_nodes = np.where(np.abs(nodes[:, 2] - z_max) < 1e-6)[0]
        
        elif surface == 'bottom':
            # Bottom surface (anode bottom)
            z_min = np.min(nodes[:, 2])
            surface_nodes = np.where(np.abs(nodes[:, 2] - z_min) < 1e-6)[0]
        
        elif surface == 'electrolyte_top':
            # Electrolyte top surface
            z_electrolyte_top = (self.params.anode_thickness_mm + 
                               self.params.electrolyte_thickness_mm)
            surface_nodes = np.where(np.abs(nodes[:, 2] - z_electrolyte_top) < 1e-6)[0]
        
        elif surface == 'electrolyte_bottom':
            # Electrolyte bottom surface
            z_electrolyte_bottom = self.params.anode_thickness_mm
            surface_nodes = np.where(np.abs(nodes[:, 2] - z_electrolyte_bottom) < 1e-6)[0]
        
        return np.array(surface_nodes)
    
    def export_mesh(self, filename: str, format: str = 'vtk'):
        """Export mesh to file"""
        if self.mesh is None:
            raise ValueError("Mesh not generated yet")
        
        nodes = self.mesh['nodes']
        elements = self.mesh['elements']
        
        if format.lower() == 'vtk':
            # Create PyVista mesh
            mesh = pv.UnstructuredGrid()
            mesh.points = nodes
            
            # Add hexahedral cells
            hex_cells = np.column_stack([np.full(len(elements), 8), elements])
            mesh.cells = hex_cells.ravel()
            mesh.celltypes = np.full(len(elements), pv.CellType.HEXAHEDRON)
            
            # Add element groups as cell data
            for group_name, element_ids in self.mesh['element_groups'].items():
                group_array = np.zeros(len(elements))
                group_array[element_ids] = 1
                mesh.cell_data[group_name] = group_array
            
            mesh.save(filename)
        
        elif format.lower() == 'xdmf':
            # Export as XDMF for FEniCS/other FEA codes
            cells = [("hexahedron", elements)]
            meshio.write(filename, nodes, cells)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Mesh exported to {filename}")
    
    def get_mesh_statistics(self) -> Dict[str, int]:
        """Get mesh statistics"""
        if self.mesh is None:
            raise ValueError("Mesh not generated yet")
        
        return {
            'n_nodes': len(self.mesh['nodes']),
            'n_elements': len(self.mesh['elements']),
            'n_anode_elements': len(self.mesh['element_groups']['anode']),
            'n_electrolyte_elements': len(self.mesh['element_groups']['electrolyte']),
            'n_cathode_elements': len(self.mesh['element_groups']['cathode']),
            'n_interconnect_elements': len(self.mesh['element_groups']['interconnect'])
        }


if __name__ == "__main__":
    # Example usage
    mesh_params = MeshParameters(
        cell_length=100.0,
        cell_width=100.0,
        electrolyte_thickness=150.0,
        anode_thickness=300.0,
        cathode_thickness=50.0,
        interconnect_thickness=2000.0
    )
    
    mesh_gen = SOFCMeshGenerator(mesh_params)
    mesh = mesh_gen.generate_mesh()
    
    print("Mesh Statistics:")
    stats = mesh_gen.get_mesh_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export mesh
    mesh_gen.export_mesh('sofc_mesh.vtk')
    
    # Get surface nodes
    top_nodes = mesh_gen.get_surface_nodes('top')
    bottom_nodes = mesh_gen.get_surface_nodes('bottom')
    print(f"\nTop surface nodes: {len(top_nodes)}")
    print(f"Bottom surface nodes: {len(bottom_nodes)}")