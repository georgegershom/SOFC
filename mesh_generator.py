#!/usr/bin/env python3
"""
SOFC Microstructure Mesh Generator
===================================
Generate computational meshes from voxelated microstructural data
for high-fidelity FEM/FVM simulations.

Author: SOFC Modeling Team
Date: 2025-10-08
"""

import numpy as np
import h5py
import trimesh
from skimage import measure, filters
import scipy.ndimage as ndi
import os
import json
from datetime import datetime


class MeshGenerator:
    """
    Generate high-quality computational meshes from voxel data.
    """
    
    def __init__(self, volume, voxel_size=0.5, phase_definitions=None):
        """
        Initialize mesh generator.
        
        Parameters:
        -----------
        volume : np.ndarray
            3D voxelated microstructure
        voxel_size : float
            Physical size of voxels in micrometers
        phase_definitions : dict
            Mapping of phase names to IDs
        """
        self.volume = volume
        self.voxel_size = voxel_size
        self.size = volume.shape
        
        # Default phase definitions
        if phase_definitions is None:
            self.phases = {
                'pore': 0,
                'nickel': 1,
                'ysz_composite': 2,
                'ysz_electrolyte': 3,
                'interlayer': 4
            }
        else:
            self.phases = phase_definitions
        
        self.meshes = {}
        self.interfaces = {}
    
    def smooth_volume(self, sigma=0.5):
        """
        Apply Gaussian smoothing to reduce staircase artifacts.
        """
        print("Applying volume smoothing...")
        smoothed = np.zeros_like(self.volume, dtype=float)
        
        for phase_id in np.unique(self.volume):
            phase_mask = (self.volume == phase_id).astype(float)
            smoothed_phase = filters.gaussian(phase_mask, sigma=sigma)
            smoothed += smoothed_phase * phase_id
        
        return smoothed
    
    def generate_surface_mesh(self, phase_name, smoothing=True, 
                             decimation_factor=0.5):
        """
        Generate surface mesh for a specific phase using marching cubes.
        
        Parameters:
        -----------
        phase_name : str
            Name of the phase to mesh
        smoothing : bool
            Apply smoothing to reduce artifacts
        decimation_factor : float
            Mesh decimation factor (0-1)
        """
        print(f"\nGenerating surface mesh for {phase_name}...")
        
        if phase_name not in self.phases:
            print(f"  Phase {phase_name} not found")
            return None
        
        phase_id = self.phases[phase_name]
        
        # Create binary mask for phase
        if smoothing:
            smoothed = self.smooth_volume()
            phase_mask = smoothed == phase_id
        else:
            phase_mask = self.volume == phase_id
        
        # Apply morphological operations to clean up
        phase_mask = ndi.binary_closing(phase_mask, iterations=1)
        phase_mask = ndi.binary_opening(phase_mask, iterations=1)
        
        # Generate mesh using marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                phase_mask.astype(float), 
                level=0.5,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
            
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, 
                                  vertex_normals=normals)
            
            # Clean up mesh
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Smooth mesh
            if smoothing:
                mesh = mesh.smoothed()
            
            # Simplify mesh if requested
            if decimation_factor < 1.0:
                target_faces = int(len(mesh.faces) * decimation_factor)
                mesh = mesh.simplify_quadric_decimation(target_faces)
            
            # Calculate mesh statistics
            stats = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'volume_um3': mesh.volume,
                'surface_area_um2': mesh.area,
                'bounds_um': mesh.bounds.tolist(),
                'watertight': mesh.is_watertight,
            }
            
            print(f"  Mesh generated:")
            print(f"    Vertices: {stats['vertices']:,}")
            print(f"    Faces: {stats['faces']:,}")
            print(f"    Volume: {stats['volume_um3']:.2f} μm³")
            print(f"    Surface area: {stats['surface_area_um2']:.2f} μm²")
            print(f"    Watertight: {stats['watertight']}")
            
            self.meshes[phase_name] = {
                'mesh': mesh,
                'stats': stats
            }
            
            return mesh
            
        except Exception as e:
            print(f"  Error generating mesh: {e}")
            return None
    
    def generate_interface_mesh(self, phase1_name, phase2_name):
        """
        Generate mesh for interface between two phases.
        """
        print(f"\nGenerating interface mesh between {phase1_name} and {phase2_name}...")
        
        if phase1_name not in self.phases or phase2_name not in self.phases:
            print("  One or both phases not found")
            return None
        
        phase1_id = self.phases[phase1_name]
        phase2_id = self.phases[phase2_name]
        
        # Find interface voxels
        phase1_mask = self.volume == phase1_id
        phase2_mask = self.volume == phase2_id
        
        # Dilate both phases and find intersection
        phase1_dilated = ndi.binary_dilation(phase1_mask)
        phase2_dilated = ndi.binary_dilation(phase2_mask)
        
        interface = phase1_dilated & phase2_dilated
        
        if not np.any(interface):
            print("  No interface found")
            return None
        
        # Generate mesh for interface region
        try:
            verts, faces, normals, values = measure.marching_cubes(
                interface.astype(float),
                level=0.5,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
            
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                  vertex_normals=normals)
            
            # Clean up
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            
            interface_name = f"{phase1_name}_{phase2_name}"
            
            stats = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'area_um2': mesh.area,
                'bounds_um': mesh.bounds.tolist(),
            }
            
            print(f"  Interface mesh generated:")
            print(f"    Vertices: {stats['vertices']:,}")
            print(f"    Faces: {stats['faces']:,}")
            print(f"    Interface area: {stats['area_um2']:.2f} μm²")
            
            self.interfaces[interface_name] = {
                'mesh': mesh,
                'stats': stats
            }
            
            return mesh
            
        except Exception as e:
            print(f"  Error generating interface mesh: {e}")
            return None
    
    def generate_volume_mesh(self, phase_name, max_element_size=2.0):
        """
        Generate tetrahedral volume mesh for FEM simulations.
        """
        print(f"\nGenerating volume mesh for {phase_name}...")
        
        if phase_name not in self.meshes:
            # First generate surface mesh
            surface_mesh = self.generate_surface_mesh(phase_name)
            if surface_mesh is None:
                return None
        
        mesh_data = self.meshes[phase_name]
        surface_mesh = mesh_data['mesh']
        
        # Check if mesh is watertight (required for volume meshing)
        if not surface_mesh.is_watertight:
            print("  Attempting to repair mesh...")
            surface_mesh.fill_holes()
            
            if not surface_mesh.is_watertight:
                print("  Warning: Mesh is not watertight, volume mesh may fail")
        
        try:
            # Generate tetrahedral mesh
            # Note: This is a simplified approach. For production use,
            # consider using GMSH or TetGen through their Python APIs
            
            print("  Generating tetrahedral elements...")
            
            # For demonstration, we'll create a simple voxel-based tet mesh
            # In practice, use proper tetrahedral mesh generation
            
            phase_id = self.phases[phase_name]
            phase_mask = self.volume == phase_id
            
            # Find all voxel centers that belong to this phase
            voxel_coords = np.where(phase_mask)
            n_voxels = len(voxel_coords[0])
            
            if n_voxels == 0:
                print("  No voxels found for this phase")
                return None
            
            # Create node positions at voxel centers
            nodes = np.zeros((n_voxels, 3))
            for i in range(n_voxels):
                nodes[i] = [
                    voxel_coords[0][i] * self.voxel_size,
                    voxel_coords[1][i] * self.voxel_size,
                    voxel_coords[2][i] * self.voxel_size
                ]
            
            print(f"  Volume mesh info:")
            print(f"    Nodes: {len(nodes):,}")
            print(f"    Approximate element size: {max_element_size} μm")
            
            # Store volume mesh data
            mesh_data['volume_mesh'] = {
                'nodes': nodes,
                'phase_mask': phase_mask,
                'element_size': max_element_size
            }
            
            return nodes
            
        except Exception as e:
            print(f"  Error generating volume mesh: {e}")
            return None
    
    def export_stl(self, phase_name, filename=None):
        """
        Export mesh to STL format.
        """
        if phase_name not in self.meshes:
            print(f"No mesh found for {phase_name}")
            return
        
        if filename is None:
            filename = f'output/meshes/{phase_name}.stl'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        mesh = self.meshes[phase_name]['mesh']
        mesh.export(filename)
        
        print(f"Exported {phase_name} mesh to {filename}")
    
    def export_obj(self, phase_name, filename=None):
        """
        Export mesh to OBJ format.
        """
        if phase_name not in self.meshes:
            print(f"No mesh found for {phase_name}")
            return
        
        if filename is None:
            filename = f'output/meshes/{phase_name}.obj'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        mesh = self.meshes[phase_name]['mesh']
        mesh.export(filename)
        
        print(f"Exported {phase_name} mesh to {filename}")
    
    def export_vtk_mesh(self, phase_name, filename=None):
        """
        Export mesh to VTK format for ParaView.
        """
        if phase_name not in self.meshes:
            print(f"No mesh found for {phase_name}")
            return
        
        if filename is None:
            filename = f'output/meshes/{phase_name}_mesh.vtk'
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        mesh = self.meshes[phase_name]['mesh']
        
        # Write VTK file
        with open(filename, 'w') as f:
            f.write('# vtk DataFile Version 3.0\n')
            f.write(f'{phase_name} mesh\n')
            f.write('ASCII\n')
            f.write('DATASET POLYDATA\n')
            
            # Write vertices
            n_verts = len(mesh.vertices)
            f.write(f'POINTS {n_verts} float\n')
            for vert in mesh.vertices:
                f.write(f'{vert[0]} {vert[1]} {vert[2]}\n')
            
            # Write faces
            n_faces = len(mesh.faces)
            n_entries = n_faces * 4  # 3 vertices per face + count
            f.write(f'POLYGONS {n_faces} {n_entries}\n')
            for face in mesh.faces:
                f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
        
        print(f"Exported {phase_name} mesh to {filename}")
    
    def export_all_meshes(self):
        """
        Export all generated meshes in multiple formats.
        """
        print("\nExporting all meshes...")
        
        for phase_name in self.meshes:
            self.export_stl(phase_name)
            self.export_obj(phase_name)
            self.export_vtk_mesh(phase_name)
        
        # Export metadata
        metadata = {
            'generated': datetime.now().isoformat(),
            'voxel_size_um': self.voxel_size,
            'volume_size_voxels': list(self.size),
            'phases': {}
        }
        
        for phase_name, data in self.meshes.items():
            metadata['phases'][phase_name] = data['stats']
        
        metadata_file = 'output/meshes/metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_file}")
    
    def generate_multiphase_mesh(self):
        """
        Generate meshes for all phases and interfaces.
        """
        print("\n" + "="*60)
        print("GENERATING MULTIPHASE COMPUTATIONAL MESH")
        print("="*60)
        
        # Generate meshes for main phases
        main_phases = ['pore', 'nickel', 'ysz_composite', 'ysz_electrolyte']
        
        for phase in main_phases:
            if phase in self.phases:
                self.generate_surface_mesh(phase, smoothing=True, 
                                         decimation_factor=0.7)
        
        # Generate interface meshes
        interface_pairs = [
            ('nickel', 'ysz_composite'),
            ('nickel', 'pore'),
            ('ysz_composite', 'pore'),
            ('ysz_composite', 'ysz_electrolyte')
        ]
        
        for phase1, phase2 in interface_pairs:
            if phase1 in self.phases and phase2 in self.phases:
                self.generate_interface_mesh(phase1, phase2)
        
        # Export all meshes
        self.export_all_meshes()
        
        print("\n" + "="*60)
        print("MESH GENERATION COMPLETE")
        print("="*60)
        
        return self.meshes


def main():
    """
    Generate meshes from microstructure data.
    """
    # Load microstructure data
    h5_file = 'output/microstructure.h5'
    
    if os.path.exists(h5_file):
        print(f"Loading microstructure from {h5_file}...")
        with h5py.File(h5_file, 'r') as f:
            volume = f['volume'][:]
            voxel_size = f['volume'].attrs['voxel_size_um']
    else:
        print("No microstructure found. Please run sofc_microstructure_generator.py first.")
        return
    
    # Create mesh generator
    generator = MeshGenerator(volume, voxel_size)
    
    # Generate all meshes
    meshes = generator.generate_multiphase_mesh()
    
    print("\nMeshes saved to output/meshes/")


if __name__ == '__main__':
    main()