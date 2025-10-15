"""Mesh generation and field interpolation utilities."""

import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import interpolate
from scipy.spatial import Delaunay
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False


class MeshGenerator:
    """Generate computational meshes for different fidelity levels."""
    
    def __init__(self, geometry: Dict):
        """Initialize with geometry parameters."""
        self.geometry = geometry
        self.length = geometry['length']
        self.width = geometry['width']
        
        # Convert thicknesses from μm to m
        self.anode_thickness = geometry['anode_thickness'] * 1e-6
        self.electrolyte_thickness = geometry['electrolyte_thickness'] * 1e-6
        self.cathode_thickness = geometry['cathode_thickness'] * 1e-6
        self.interconnect_thickness = geometry['interconnect_thickness'] * 1e-6
        
        self.total_thickness = (self.anode_thickness + 
                               self.electrolyte_thickness + 
                               self.cathode_thickness)
    
    def generate_1d_mesh(self, n_points: int = 100) -> Dict:
        """Generate 1D mesh through thickness."""
        mesh = {
            'coordinates': np.linspace(0, self.total_thickness, n_points),
            'n_points': n_points,
            'dimension': 1
        }
        
        # Identify layer boundaries
        boundaries = [0, 
                     self.anode_thickness,
                     self.anode_thickness + self.electrolyte_thickness,
                     self.total_thickness]
        
        # Assign material IDs
        material_ids = np.zeros(n_points, dtype=int)
        for i, x in enumerate(mesh['coordinates']):
            if x <= boundaries[1]:
                material_ids[i] = 0  # Anode
            elif x <= boundaries[2]:
                material_ids[i] = 1  # Electrolyte
            else:
                material_ids[i] = 2  # Cathode
        
        mesh['material_ids'] = material_ids
        mesh['boundaries'] = boundaries
        
        return mesh
    
    def generate_2d_mesh(self, nx: int = 50, ny: int = 20) -> Dict:
        """Generate 2D mesh (x-z plane)."""
        x = np.linspace(0, self.length, nx)
        z = np.linspace(0, self.total_thickness, ny)
        
        X, Z = np.meshgrid(x, z)
        
        mesh = {
            'X': X,
            'Z': Z,
            'nx': nx,
            'ny': ny,
            'dimension': 2
        }
        
        # Create material field
        material_field = np.zeros((ny, nx), dtype=int)
        for j in range(ny):
            z_coord = z[j]
            if z_coord <= self.anode_thickness:
                material_field[j, :] = 0  # Anode
            elif z_coord <= self.anode_thickness + self.electrolyte_thickness:
                material_field[j, :] = 1  # Electrolyte
            else:
                material_field[j, :] = 2  # Cathode
        
        mesh['material_field'] = material_field
        
        return mesh
    
    def generate_3d_mesh(self, nx: int = 30, ny: int = 30, nz: int = 15) -> Dict:
        """Generate 3D structured mesh."""
        x = np.linspace(0, self.length, nx)
        y = np.linspace(0, self.width, ny)
        z = np.linspace(0, self.total_thickness, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        mesh = {
            'X': X,
            'Y': Y,
            'Z': Z,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'dimension': 3,
            'n_nodes': nx * ny * nz,
            'n_elements': (nx-1) * (ny-1) * (nz-1)
        }
        
        # Create material field
        material_field = np.zeros((nx, ny, nz), dtype=int)
        for k in range(nz):
            z_coord = z[k]
            if z_coord <= self.anode_thickness:
                material_field[:, :, k] = 0  # Anode
            elif z_coord <= self.anode_thickness + self.electrolyte_thickness:
                material_field[:, :, k] = 1  # Electrolyte
            else:
                material_field[:, :, k] = 2  # Cathode
        
        mesh['material_field'] = material_field
        
        # Add channel geometry if needed
        if 'channel_width' in self.geometry:
            mesh['channels'] = self._add_channel_geometry(mesh, self.geometry)
        
        return mesh
    
    def _add_channel_geometry(self, mesh: Dict, geometry: Dict) -> Dict:
        """Add fuel/air channels to mesh."""
        channel_width = geometry['channel_width'] * 1e-3  # mm to m
        channel_height = geometry['channel_height'] * 1e-3
        rib_width = geometry['rib_width'] * 1e-3
        
        pitch = channel_width + rib_width
        n_channels = int(self.width / pitch)
        
        channels = {
            'fuel': [],
            'air': []
        }
        
        for i in range(n_channels):
            channel_start = i * pitch
            channel_end = channel_start + channel_width
            
            # Fuel channels at anode side
            channels['fuel'].append({
                'y_start': channel_start,
                'y_end': channel_end,
                'z': 0,
                'height': channel_height
            })
            
            # Air channels at cathode side
            channels['air'].append({
                'y_start': channel_start,
                'y_end': channel_end,
                'z': self.total_thickness,
                'height': channel_height
            })
        
        return channels
    
    def generate_microstructure_mesh(self, domain_size: float = 10e-6,
                                    n_particles: int = 100) -> Dict:
        """Generate representative volume element for microstructure."""
        # Random particle centers
        particle_centers = np.random.rand(n_particles, 3) * domain_size
        particle_radii = np.random.normal(1e-6, 0.2e-6, n_particles)
        particle_radii = np.clip(particle_radii, 0.5e-6, 2e-6)
        
        # Phase assignment (Ni, YSZ, pore)
        phases = np.random.choice([0, 1, 2], n_particles, p=[0.3, 0.4, 0.3])
        
        microstructure = {
            'domain_size': domain_size,
            'particle_centers': particle_centers,
            'particle_radii': particle_radii,
            'phases': phases,
            'n_particles': n_particles
        }
        
        return microstructure


class FieldInterpolator:
    """Interpolate fields between different mesh resolutions."""
    
    @staticmethod
    def interpolate_1d_to_3d(field_1d: np.ndarray, z_coords_1d: np.ndarray,
                            mesh_3d: Dict) -> np.ndarray:
        """Interpolate 1D field to 3D mesh."""
        # Create interpolator
        f_interp = interpolate.interp1d(z_coords_1d, field_1d, 
                                       kind='cubic', fill_value='extrapolate')
        
        # Get unique z coordinates from 3D mesh
        z_unique = np.unique(mesh_3d['Z'])
        
        # Interpolate to new z coordinates
        field_z = f_interp(z_unique)
        
        # Broadcast to 3D
        nx, ny, nz = mesh_3d['nx'], mesh_3d['ny'], mesh_3d['nz']
        field_3d = np.zeros((nx, ny, nz))
        
        for k, z_val in enumerate(z_unique):
            field_3d[:, :, k] = field_z[k]
        
        return field_3d
    
    @staticmethod
    def interpolate_2d_to_3d(field_2d: np.ndarray, mesh_2d: Dict,
                            mesh_3d: Dict) -> np.ndarray:
        """Interpolate 2D field to 3D mesh."""
        # Create 2D interpolator
        f_interp = interpolate.RegularGridInterpolator(
            (mesh_2d['X'][:, 0], mesh_2d['Z'][0, :]),
            field_2d.T,
            method='linear',
            fill_value=0
        )
        
        # Get 3D mesh points
        nx, ny, nz = mesh_3d['nx'], mesh_3d['ny'], mesh_3d['nz']
        field_3d = np.zeros((nx, ny, nz))
        
        # Interpolate for each y-slice
        for j in range(ny):
            points = np.column_stack((mesh_3d['X'][:, j, :].ravel(),
                                     mesh_3d['Z'][:, j, :].ravel()))
            values = f_interp(points)
            field_3d[:, j, :] = values.reshape(nx, nz)
        
        return field_3d
    
    @staticmethod
    def coarsen_field(field_fine: np.ndarray, factor: int = 2) -> np.ndarray:
        """Coarsen field by averaging."""
        if field_fine.ndim == 1:
            # 1D coarsening
            n_coarse = len(field_fine) // factor
            field_coarse = np.zeros(n_coarse)
            for i in range(n_coarse):
                field_coarse[i] = np.mean(field_fine[i*factor:(i+1)*factor])
            
        elif field_fine.ndim == 2:
            # 2D coarsening
            ny, nx = field_fine.shape
            ny_coarse = ny // factor
            nx_coarse = nx // factor
            field_coarse = np.zeros((ny_coarse, nx_coarse))
            
            for j in range(ny_coarse):
                for i in range(nx_coarse):
                    field_coarse[j, i] = np.mean(
                        field_fine[j*factor:(j+1)*factor, 
                                  i*factor:(i+1)*factor]
                    )
        
        elif field_fine.ndim == 3:
            # 3D coarsening
            nz, ny, nx = field_fine.shape
            nz_coarse = nz // factor
            ny_coarse = ny // factor
            nx_coarse = nx // factor
            field_coarse = np.zeros((nz_coarse, ny_coarse, nx_coarse))
            
            for k in range(nz_coarse):
                for j in range(ny_coarse):
                    for i in range(nx_coarse):
                        field_coarse[k, j, i] = np.mean(
                            field_fine[k*factor:(k+1)*factor,
                                      j*factor:(j+1)*factor,
                                      i*factor:(i+1)*factor]
                        )
        
        else:
            raise ValueError(f"Unsupported field dimension: {field_fine.ndim}")
        
        return field_coarse
    
    @staticmethod
    def add_spatial_variation(field_uniform: float, mesh: Dict,
                            variation_type: str = 'linear',
                            variation_scale: float = 0.1) -> np.ndarray:
        """Add spatial variation to uniform field."""
        
        if mesh['dimension'] == 1:
            n = mesh['n_points']
            field = np.ones(n) * field_uniform
            
            if variation_type == 'linear':
                field += np.linspace(0, variation_scale * field_uniform, n)
            elif variation_type == 'sinusoidal':
                field += variation_scale * field_uniform * np.sin(2 * np.pi * mesh['coordinates'] / mesh['coordinates'][-1])
            elif variation_type == 'random':
                field += np.random.normal(0, variation_scale * field_uniform, n)
        
        elif mesh['dimension'] == 2:
            ny, nx = mesh['X'].shape
            field = np.ones((ny, nx)) * field_uniform
            
            if variation_type == 'linear':
                field += variation_scale * field_uniform * (mesh['X'] / mesh['X'].max())
            elif variation_type == 'sinusoidal':
                field += variation_scale * field_uniform * np.sin(2 * np.pi * mesh['X'] / mesh['X'].max())
            elif variation_type == 'random':
                field += np.random.normal(0, variation_scale * field_uniform, (ny, nx))
        
        elif mesh['dimension'] == 3:
            nx, ny, nz = mesh['nx'], mesh['ny'], mesh['nz']
            field = np.ones((nx, ny, nz)) * field_uniform
            
            if variation_type == 'linear':
                field += variation_scale * field_uniform * (mesh['X'] / mesh['X'].max())
            elif variation_type == 'sinusoidal':
                field += variation_scale * field_uniform * np.sin(2 * np.pi * mesh['X'] / mesh['X'].max())
            elif variation_type == 'random':
                field += np.random.normal(0, variation_scale * field_uniform, (nx, ny, nz))
        
        return field


def export_to_vtk(mesh: Dict, fields: Dict, filename: str):
    """Export mesh and fields to VTK format for visualization."""
    if not MESHIO_AVAILABLE:
        print("Warning: meshio not installed, skipping VTK export")
        return
    if mesh['dimension'] == 3:
        # Create points
        points = np.column_stack((mesh['X'].ravel(),
                                 mesh['Y'].ravel(),
                                 mesh['Z'].ravel()))
        
        # Create cells (hexahedra for structured grid)
        nx, ny, nz = mesh['nx'], mesh['ny'], mesh['nz']
        cells = []
        
        for k in range(nz-1):
            for j in range(ny-1):
                for i in range(nx-1):
                    # Node indices for hexahedron
                    n0 = i + j*nx + k*nx*ny
                    n1 = (i+1) + j*nx + k*nx*ny
                    n2 = (i+1) + (j+1)*nx + k*nx*ny
                    n3 = i + (j+1)*nx + k*nx*ny
                    n4 = i + j*nx + (k+1)*nx*ny
                    n5 = (i+1) + j*nx + (k+1)*nx*ny
                    n6 = (i+1) + (j+1)*nx + (k+1)*nx*ny
                    n7 = i + (j+1)*nx + (k+1)*nx*ny
                    
                    cells.append(("hexahedron", [n0, n1, n2, n3, n4, n5, n6, n7]))
        
        # Prepare point data
        point_data = {}
        for name, field in fields.items():
            if field.shape == (nx, ny, nz):
                point_data[name] = field.ravel()
        
        # Write to file
        meshio.write_points_cells(
            filename,
            points,
            cells,
            point_data=point_data
        )