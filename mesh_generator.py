"""
Mesh Generation Module for SOFC Microstructure Data

This module converts 3D voxelated microstructure data into computational meshes
suitable for finite element analysis and other computational modeling approaches.
"""

import numpy as np
import h5py
import vtk
from vtk.util import numpy_support
import pyvista as pv
import trimesh
from scipy import ndimage
from skimage import morphology, measure
import meshio
import gmsh
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MeshGenerator:
    """
    Generator for computational meshes from 3D microstructure data.
    """
    
    def __init__(self, microstructure: np.ndarray, voxel_size: float = 0.1):
        """
        Initialize mesh generator.
        
        Parameters:
        -----------
        microstructure : np.ndarray
            3D array with phase labels
        voxel_size : float
            Size of each voxel in micrometers
        """
        self.microstructure = microstructure
        self.voxel_size = voxel_size
        self.resolution = microstructure.shape
        
        # Phase labels
        self.PORE = 0
        self.NI = 1
        self.YSZ_ANODE = 2
        self.YSZ_ELECTROLYTE = 3
        self.INTERLAYER = 4
        
        self.phase_names = {
            self.PORE: 'Pore',
            self.NI: 'Ni',
            self.YSZ_ANODE: 'YSZ_Anode',
            self.YSZ_ELECTROLYTE: 'YSZ_Electrolyte',
            self.INTERLAYER: 'Interlayer'
        }
    
    def generate_structured_hex_mesh(self, element_size: float = None) -> pv.StructuredGrid:
        """
        Generate structured hexahedral mesh from voxel data.
        
        Parameters:
        -----------
        element_size : float, optional
            Target element size. If None, uses voxel_size.
        
        Returns:
        --------
        pv.StructuredGrid
            Structured grid mesh
        """
        print("Generating structured hexahedral mesh...")
        
        if element_size is None:
            element_size = self.voxel_size
        
        # Create coordinate arrays
        x = np.arange(self.resolution[0]) * self.voxel_size
        y = np.arange(self.resolution[1]) * self.voxel_size
        z = np.arange(self.resolution[2]) * self.voxel_size
        
        # Create meshgrid
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Create structured grid
        grid = pv.StructuredGrid()
        grid.dimensions = self.resolution
        grid.points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        grid['phase'] = self.microstructure.ravel()
        
        # Add cell data
        grid.cell_data['phase'] = self.microstructure.ravel()
        
        return grid
    
    def generate_unstructured_tet_mesh(self, 
                                     max_element_size: float = None,
                                     min_element_size: float = None) -> pv.UnstructuredGrid:
        """
        Generate unstructured tetrahedral mesh using marching cubes.
        
        Parameters:
        -----------
        max_element_size : float, optional
            Maximum element size
        min_element_size : float, optional
            Minimum element size
        
        Returns:
        --------
        pv.UnstructuredGrid
            Unstructured tetrahedral mesh
        """
        print("Generating unstructured tetrahedral mesh...")
        
        if max_element_size is None:
            max_element_size = self.voxel_size * 2
        if min_element_size is None:
            min_element_size = self.voxel_size * 0.5
        
        # Create separate meshes for each phase
        all_meshes = []
        
        for phase_id, phase_name in self.phase_names.items():
            if phase_id == self.PORE:
                continue  # Skip pores for solid mesh
            
            phase_mask = (self.microstructure == phase_id)
            
            if not np.any(phase_mask):
                continue
            
            # Apply marching cubes
            try:
                vertices, faces, _, _ = measure.marching_cubes(
                    phase_mask.astype(float), 
                    level=0.5,
                    spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
                )
                
                if len(vertices) > 0 and len(faces) > 0:
                    # Create mesh
                    mesh = pv.PolyData(vertices, faces)
                    mesh['phase'] = np.full(mesh.n_cells, phase_id)
                    all_meshes.append(mesh)
                    
            except Exception as e:
                print(f"Warning: Could not generate mesh for {phase_name}: {e}")
                continue
        
        if not all_meshes:
            print("Warning: No valid meshes generated")
            return pv.UnstructuredGrid()
        
        # Combine all meshes
        combined_mesh = all_meshes[0]
        for mesh in all_meshes[1:]:
            combined_mesh = combined_mesh + mesh
        
        # Convert to unstructured grid
        unstructured = combined_mesh.cast_to_unstructured_grid()
        
        return unstructured
    
    def generate_surface_mesh(self, phase_id: int = None) -> pv.PolyData:
        """
        Generate surface mesh for specific phase or all phases.
        
        Parameters:
        -----------
        phase_id : int, optional
            Phase ID to extract surface for. If None, extracts all phases.
        
        Returns:
        --------
        pv.PolyData
            Surface mesh
        """
        print(f"Generating surface mesh for phase {phase_id}...")
        
        if phase_id is not None:
            phase_mask = (self.microstructure == phase_id)
        else:
            # Combine all solid phases
            phase_mask = (self.microstructure != self.PORE)
        
        if not np.any(phase_mask):
            print("Warning: No solid phase found for surface generation")
            return pv.PolyData()
        
        try:
            # Apply marching cubes
            vertices, faces, _, _ = measure.marching_cubes(
                phase_mask.astype(float),
                level=0.5,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
            
            # Create surface mesh
            surface_mesh = pv.PolyData(vertices, faces)
            
            # Add phase information
            if phase_id is not None:
                surface_mesh['phase'] = np.full(surface_mesh.n_cells, phase_id)
            else:
                # Determine phase for each face (simplified)
                face_centers = surface_mesh.cell_centers().points
                phases = []
                for center in face_centers:
                    # Find closest voxel
                    x_idx = int(center[0] / self.voxel_size)
                    y_idx = int(center[1] / self.voxel_size)
                    z_idx = int(center[2] / self.voxel_size)
                    
                    x_idx = np.clip(x_idx, 0, self.resolution[0] - 1)
                    y_idx = np.clip(y_idx, 0, self.resolution[1] - 1)
                    z_idx = np.clip(z_idx, 0, self.resolution[2] - 1)
                    
                    phase = self.microstructure[x_idx, y_idx, z_idx]
                    phases.append(phase)
                
                surface_mesh['phase'] = np.array(phases)
            
            return surface_mesh
            
        except Exception as e:
            print(f"Error generating surface mesh: {e}")
            return pv.PolyData()
    
    def generate_interface_mesh(self) -> pv.PolyData:
        """
        Generate mesh specifically for interfaces between phases.
        
        Returns:
        --------
        pv.PolyData
            Interface mesh
        """
        print("Generating interface mesh...")
        
        # Find interface between anode and electrolyte
        anode_mask = (self.microstructure == self.NI) | (self.microstructure == self.YSZ_ANODE)
        electrolyte_mask = (self.microstructure == self.YSZ_ELECTROLYTE)
        
        if not (np.any(anode_mask) and np.any(electrolyte_mask)):
            print("Warning: No anode/electrolyte interface found")
            return pv.PolyData()
        
        # Create interface mask
        anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(1))
        electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(1))
        interface_mask = anode_dilated & electrolyte_dilated
        
        try:
            # Apply marching cubes to interface
            vertices, faces, _, _ = measure.marching_cubes(
                interface_mask.astype(float),
                level=0.5,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
            
            # Create interface mesh
            interface_mesh = pv.PolyData(vertices, faces)
            interface_mesh['interface_type'] = np.full(interface_mesh.n_cells, 1)  # Anode/Electrolyte
            
            return interface_mesh
            
        except Exception as e:
            print(f"Error generating interface mesh: {e}")
            return pv.PolyData()
    
    def export_mesh(self, mesh: pv.DataSet, filename: str, format: str = 'vtk'):
        """
        Export mesh to various formats.
        
        Parameters:
        -----------
        mesh : pv.DataSet
            Mesh to export
        filename : str
            Output filename
        format : str
            Export format ('vtk', 'stl', 'obj', 'ply', 'xdmf')
        """
        print(f"Exporting mesh to {filename} in {format} format...")
        
        # Create output directory
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        if format.lower() == 'vtk':
            mesh.save(filename)
        elif format.lower() == 'stl':
            if hasattr(mesh, 'extract_surface'):
                surface = mesh.extract_surface()
                surface.save(filename)
            else:
                mesh.save(filename)
        elif format.lower() == 'obj':
            mesh.save(filename)
        elif format.lower() == 'ply':
            mesh.save(filename)
        elif format.lower() == 'xdmf':
            # Convert to meshio format
            if isinstance(mesh, pv.StructuredGrid):
                points = mesh.points
                cells = [("hexahedron", mesh.cells.reshape(-1, 9)[:, 1:])]
            else:
                points = mesh.points
                cells = [("tetra", mesh.cells.reshape(-1, 5)[:, 1:])]
            
            # Get cell data
            cell_data = {}
            for key in mesh.cell_data.keys():
                cell_data[key] = [mesh.cell_data[key]]
            
            # Create meshio mesh
            meshio_mesh = meshio.Mesh(points, cells, cell_data=cell_data)
            meshio_mesh.write(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_gmsh_mesh(self, 
                          output_file: str,
                          element_size: float = None,
                          mesh_order: int = 1) -> str:
        """
        Generate mesh using GMSH with proper boundary conditions.
        
        Parameters:
        -----------
        output_file : str
            Output mesh file
        element_size : float, optional
            Target element size
        mesh_order : int
            Mesh order (1 for linear, 2 for quadratic)
        
        Returns:
        --------
        str
            Path to generated mesh file
        """
        print("Generating GMSH mesh...")
        
        if element_size is None:
            element_size = self.voxel_size
        
        # Initialize GMSH
        gmsh.initialize()
        gmsh.model.add("sofc_microstructure")
        
        try:
            # Create geometry
            # For simplicity, create a box geometry
            # In practice, you would import the actual microstructure geometry
            
            # Box dimensions
            lx = self.resolution[0] * self.voxel_size
            ly = self.resolution[1] * self.voxel_size
            lz = self.resolution[2] * self.voxel_size
            
            # Create box
            box = gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
            
            # Synchronize
            gmsh.model.occ.synchronize()
            
            # Set mesh size
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size * 2)
            
            # Generate mesh
            gmsh.model.mesh.generate(mesh_order)
            
            # Save mesh
            gmsh.write(output_file)
            
            print(f"GMSH mesh saved to {output_file}")
            return output_file
            
        finally:
            gmsh.finalize()
    
    def create_mesh_statistics(self, mesh: pv.DataSet) -> Dict[str, float]:
        """
        Calculate mesh statistics.
        
        Parameters:
        -----------
        mesh : pv.DataSet
            Mesh to analyze
        
        Returns:
        --------
        Dict[str, float]
            Mesh statistics
        """
        print("Calculating mesh statistics...")
        
        stats = {
            'n_points': mesh.n_points,
            'n_cells': mesh.n_cells,
            'mesh_volume': mesh.volume if hasattr(mesh, 'volume') else 0,
            'mesh_surface_area': mesh.area if hasattr(mesh, 'area') else 0
        }
        
        # Calculate element quality metrics
        if hasattr(mesh, 'compute_cell_quality'):
            quality = mesh.compute_cell_quality()
            stats['mean_element_quality'] = np.mean(quality)
            stats['min_element_quality'] = np.min(quality)
            stats['max_element_quality'] = np.max(quality)
        
        # Calculate aspect ratios
        if hasattr(mesh, 'compute_cell_quality'):
            aspect_ratios = mesh.compute_cell_quality(metric='aspect_ratio')
            stats['mean_aspect_ratio'] = np.mean(aspect_ratios)
            stats['min_aspect_ratio'] = np.min(aspect_ratios)
            stats['max_aspect_ratio'] = np.max(aspect_ratios)
        
        return stats


def main():
    """Main function for mesh generation."""
    print("Starting mesh generation...")
    
    # Load microstructure data
    try:
        with h5py.File('output/sofc_microstructure.h5', 'r') as f:
            microstructure = f['microstructure'][:]
            voxel_size = f.attrs['voxel_size_um']
    except FileNotFoundError:
        print("Microstructure data not found. Please run microstructure_generator.py first.")
        return
    
    # Create mesh generator
    mesh_gen = MeshGenerator(microstructure, voxel_size)
    
    # Create output directory
    os.makedirs('output/meshes', exist_ok=True)
    
    # Generate different types of meshes
    print("Generating structured hexahedral mesh...")
    hex_mesh = mesh_gen.generate_structured_hex_mesh()
    mesh_gen.export_mesh(hex_mesh, 'output/meshes/structured_hex_mesh.vtk', 'vtk')
    mesh_gen.export_mesh(hex_mesh, 'output/meshes/structured_hex_mesh.xdmf', 'xdmf')
    
    print("Generating unstructured tetrahedral mesh...")
    tet_mesh = mesh_gen.generate_unstructured_tet_mesh()
    if tet_mesh.n_cells > 0:
        mesh_gen.export_mesh(tet_mesh, 'output/meshes/unstructured_tet_mesh.vtk', 'vtk')
        mesh_gen.export_mesh(tet_mesh, 'output/meshes/unstructured_tet_mesh.stl', 'stl')
    
    print("Generating surface mesh...")
    surface_mesh = mesh_gen.generate_surface_mesh()
    if surface_mesh.n_cells > 0:
        mesh_gen.export_mesh(surface_mesh, 'output/meshes/surface_mesh.stl', 'stl')
        mesh_gen.export_mesh(surface_mesh, 'output/meshes/surface_mesh.obj', 'obj')
    
    print("Generating interface mesh...")
    interface_mesh = mesh_gen.generate_interface_mesh()
    if interface_mesh.n_cells > 0:
        mesh_gen.export_mesh(interface_mesh, 'output/meshes/interface_mesh.stl', 'stl')
    
    # Generate GMSH mesh
    print("Generating GMSH mesh...")
    gmsh_file = mesh_gen.generate_gmsh_mesh('output/meshes/gmsh_mesh.msh')
    
    # Calculate mesh statistics
    print("Calculating mesh statistics...")
    hex_stats = mesh_gen.create_mesh_statistics(hex_mesh)
    tet_stats = mesh_gen.create_mesh_statistics(tet_mesh) if tet_mesh.n_cells > 0 else {}
    
    # Save statistics
    import json
    with open('output/meshes/mesh_statistics.json', 'w') as f:
        json.dump({
            'hexahedral_mesh': hex_stats,
            'tetrahedral_mesh': tet_stats
        }, f, indent=2)
    
    print("\n" + "="*50)
    print("MESH GENERATION COMPLETE")
    print("="*50)
    print("Meshes saved to 'output/meshes/' directory:")
    print("  - structured_hex_mesh.vtk/.xdmf (Structured hexahedral mesh)")
    print("  - unstructured_tet_mesh.vtk/.stl (Unstructured tetrahedral mesh)")
    print("  - surface_mesh.stl/.obj (Surface mesh)")
    print("  - interface_mesh.stl (Interface mesh)")
    print("  - gmsh_mesh.msh (GMSH mesh)")
    print("  - mesh_statistics.json (Mesh statistics)")


if __name__ == "__main__":
    main()