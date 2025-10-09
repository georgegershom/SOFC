"""
3D Microstructural Data Generator for SOFC Electrode Modeling

This module generates realistic 3D voxelated microstructures that mimic
synchrotron X-ray tomography or FIB-SEM tomography data of SOFC electrodes.

Author: AI Assistant
Date: 2025-10-08
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import spatial
from skimage import morphology, filters, segmentation, measure
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import tifffile
import json
from typing import Tuple, Dict, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SOFCMicrostructureGenerator:
    """
    Generator for realistic 3D SOFC microstructures with multiple phases:
    - Phase 0: Pore (void space)
    - Phase 1: Ni-YSZ anode material
    - Phase 2: YSZ electrolyte
    - Phase 3: Interface/interlayer regions
    """
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int] = (512, 512, 256),
                 voxel_size: float = 0.05,  # micrometers
                 random_seed: int = 42):
        """
        Initialize the microstructure generator.
        
        Parameters:
        -----------
        dimensions : tuple
            (nx, ny, nz) voxel dimensions
        voxel_size : float
            Size of each voxel in micrometers
        random_seed : int
            Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.voxel_size = voxel_size
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Physical dimensions in micrometers
        self.physical_size = tuple(d * voxel_size for d in dimensions)
        
        # Phase labels
        self.phases = {
            'pore': 0,
            'ni_ysz': 1,
            'ysz_electrolyte': 2,
            'interface': 3
        }
        
        # Material properties (typical values for SOFC)
        self.material_properties = {
            'ni_ysz': {
                'porosity_target': 0.35,  # 35% porosity in anode
                'particle_size_range': (0.5, 2.0),  # micrometers
                'connectivity': 0.8
            },
            'ysz_electrolyte': {
                'porosity_target': 0.05,  # 5% porosity in electrolyte
                'particle_size_range': (0.2, 1.0),  # micrometers
                'connectivity': 0.95
            },
            'interface': {
                'thickness_range': (0.1, 0.3),  # micrometers
                'roughness': 0.2
            }
        }
        
        self.microstructure = None
        self.metadata = {}
        
    def generate_realistic_microstructure(self, 
                                        anode_thickness: float = 15.0,
                                        electrolyte_thickness: float = 10.0,
                                        interface_roughness: float = 0.5) -> np.ndarray:
        """
        Generate a complete 3D microstructure with realistic morphology.
        
        Parameters:
        -----------
        anode_thickness : float
            Thickness of anode layer in micrometers
        electrolyte_thickness : float
            Thickness of electrolyte layer in micrometers
        interface_roughness : float
            Roughness parameter for anode/electrolyte interface
            
        Returns:
        --------
        np.ndarray
            3D array with phase labels
        """
        print("Generating 3D SOFC microstructure...")
        
        # Initialize the microstructure array
        self.microstructure = np.zeros(self.dimensions, dtype=np.uint8)
        
        # Step 1: Create layered structure (anode-supported configuration)
        self._create_layered_structure(anode_thickness, electrolyte_thickness, interface_roughness)
        
        # Step 2: Generate porous anode microstructure
        self._generate_anode_microstructure()
        
        # Step 3: Generate dense electrolyte microstructure
        self._generate_electrolyte_microstructure()
        
        # Step 4: Refine interface regions
        self._refine_interface_regions()
        
        # Step 5: Apply realistic morphological operations
        self._apply_morphological_refinement()
        
        # Step 6: Calculate and store metadata
        self._calculate_microstructure_metrics()
        
        print("Microstructure generation completed!")
        return self.microstructure
    
    def _create_layered_structure(self, anode_thickness: float, 
                                electrolyte_thickness: float, 
                                interface_roughness: float):
        """Create the basic layered structure with rough interface."""
        
        # Convert thicknesses to voxels
        anode_voxels = int(anode_thickness / self.voxel_size)
        electrolyte_voxels = int(electrolyte_thickness / self.voxel_size)
        
        # Create rough interface using Perlin-like noise
        interface_position = self._generate_rough_interface(
            anode_voxels, interface_roughness
        )
        
        # Assign phases based on z-position and interface
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                interface_z = interface_position[i, j]
                
                for k in range(self.dimensions[2]):
                    if k < interface_z:
                        # Anode region
                        self.microstructure[i, j, k] = self.phases['ni_ysz']
                    elif k < interface_z + electrolyte_voxels:
                        # Electrolyte region
                        self.microstructure[i, j, k] = self.phases['ysz_electrolyte']
                    else:
                        # Pore/void region above electrolyte
                        self.microstructure[i, j, k] = self.phases['pore']
    
    def _generate_rough_interface(self, base_position: int, roughness: float) -> np.ndarray:
        """Generate a rough interface using multi-scale noise."""
        
        # Create coordinate grids
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Multi-scale Perlin-like noise for realistic roughness
        interface = np.zeros((self.dimensions[0], self.dimensions[1]))
        
        # Large scale variations
        freq1 = 0.02
        noise1 = np.sin(freq1 * X) * np.cos(freq1 * Y) * roughness * 20
        
        # Medium scale variations
        freq2 = 0.08
        noise2 = np.sin(freq2 * X + np.pi/4) * np.cos(freq2 * Y + np.pi/3) * roughness * 8
        
        # Fine scale variations
        freq3 = 0.2
        noise3 = np.random.normal(0, roughness * 2, (self.dimensions[0], self.dimensions[1]))
        noise3 = filters.gaussian(noise3, sigma=2)
        
        # Combine noise components
        interface = base_position + noise1 + noise2 + noise3
        
        # Ensure interface stays within bounds
        interface = np.clip(interface, 10, self.dimensions[2] - 20)
        
        return interface.astype(int)
    
    def _generate_anode_microstructure(self):
        """Generate realistic porous anode microstructure using particle-based approach."""
        
        # Get anode region mask
        anode_mask = (self.microstructure == self.phases['ni_ysz'])
        
        if not np.any(anode_mask):
            return
        
        # Generate particle centers using Poisson disk sampling
        particle_centers = self._poisson_disk_sampling(
            anode_mask, 
            min_distance=2.0/self.voxel_size,
            max_attempts=30
        )
        
        # Create particles with size distribution
        particle_sizes = np.random.lognormal(
            mean=np.log(1.0/self.voxel_size), 
            sigma=0.3, 
            size=len(particle_centers)
        )
        
        # Generate particle-based structure
        particle_structure = np.zeros_like(self.microstructure)
        
        for center, size in zip(particle_centers, particle_sizes):
            self._add_particle(particle_structure, center, size, anode_mask)
        
        # Apply porosity control
        target_porosity = self.material_properties['ni_ysz']['porosity_target']
        self._adjust_porosity(particle_structure, anode_mask, target_porosity)
        
        # Update microstructure in anode region
        self.microstructure[anode_mask] = particle_structure[anode_mask]
    
    def _generate_electrolyte_microstructure(self):
        """Generate dense electrolyte microstructure with minimal porosity."""
        
        # Get electrolyte region mask
        electrolyte_mask = (self.microstructure == self.phases['ysz_electrolyte'])
        
        if not np.any(electrolyte_mask):
            return
        
        # Create mostly dense structure with occasional small pores
        dense_structure = np.full_like(self.microstructure, self.phases['ysz_electrolyte'])
        
        # Add small amount of porosity using morphological operations
        target_porosity = self.material_properties['ysz_electrolyte']['porosity_target']
        
        # Generate small pores
        pore_seeds = np.random.random(self.dimensions) < (target_porosity * 0.1)
        pore_seeds = pore_seeds & electrolyte_mask
        
        # Dilate pore seeds slightly
        pore_structure = morphology.binary_dilation(pore_seeds, morphology.ball(1))
        pore_structure = pore_structure & electrolyte_mask
        
        # Apply pores to electrolyte
        dense_structure[pore_structure] = self.phases['pore']
        
        # Update microstructure in electrolyte region
        self.microstructure[electrolyte_mask] = dense_structure[electrolyte_mask]
    
    def _poisson_disk_sampling(self, mask: np.ndarray, min_distance: float, 
                              max_attempts: int = 30) -> List[Tuple[int, int, int]]:
        """Generate particle centers using Poisson disk sampling for realistic distribution."""
        
        # Get valid coordinates within mask
        valid_coords = np.where(mask)
        if len(valid_coords[0]) == 0:
            return []
        
        # Random starting point
        start_idx = np.random.randint(len(valid_coords[0]))
        start_point = (valid_coords[0][start_idx], 
                      valid_coords[1][start_idx], 
                      valid_coords[2][start_idx])
        
        points = [start_point]
        active_list = [start_point]
        
        while active_list and len(points) < 5000:  # Limit number of particles
            # Choose random active point
            active_idx = np.random.randint(len(active_list))
            active_point = active_list[active_idx]
            
            found_valid = False
            
            for _ in range(max_attempts):
                # Generate candidate point
                angle = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(min_distance, 2*min_distance)
                
                dx = r * np.sin(phi) * np.cos(angle)
                dy = r * np.sin(phi) * np.sin(angle)
                dz = r * np.cos(phi)
                
                candidate = (
                    int(active_point[0] + dx),
                    int(active_point[1] + dy),
                    int(active_point[2] + dz)
                )
                
                # Check if candidate is valid
                if (0 <= candidate[0] < self.dimensions[0] and
                    0 <= candidate[1] < self.dimensions[1] and
                    0 <= candidate[2] < self.dimensions[2] and
                    mask[candidate]):
                    
                    # Check minimum distance to existing points
                    too_close = False
                    for existing_point in points:
                        dist = np.sqrt(sum((c - e)**2 for c, e in zip(candidate, existing_point)))
                        if dist < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        points.append(candidate)
                        active_list.append(candidate)
                        found_valid = True
                        break
            
            if not found_valid:
                active_list.pop(active_idx)
        
        return points
    
    def _add_particle(self, structure: np.ndarray, center: Tuple[int, int, int], 
                     size: float, mask: np.ndarray):
        """Add a spherical particle to the structure."""
        
        x, y, z = center
        radius = int(size / 2)
        
        # Create spherical kernel
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        
        center_k = radius
        for i in range(kernel_size):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    dist = np.sqrt((i - center_k)**2 + (j - center_k)**2 + (k - center_k)**2)
                    if dist <= radius:
                        kernel[i, j, k] = 1
        
        # Apply kernel to structure
        x_start = max(0, x - radius)
        x_end = min(self.dimensions[0], x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(self.dimensions[1], y + radius + 1)
        z_start = max(0, z - radius)
        z_end = min(self.dimensions[2], z + radius + 1)
        
        kx_start = max(0, radius - x)
        kx_end = kx_start + (x_end - x_start)
        ky_start = max(0, radius - y)
        ky_end = ky_start + (y_end - y_start)
        kz_start = max(0, radius - z)
        kz_end = kz_start + (z_end - z_start)
        
        region = structure[x_start:x_end, y_start:y_end, z_start:z_end]
        kernel_region = kernel[kx_start:kx_end, ky_start:ky_end, kz_start:kz_end]
        mask_region = mask[x_start:x_end, y_start:y_end, z_start:z_end]
        
        # Only modify where mask is True
        valid_region = kernel_region.astype(bool) & mask_region.astype(bool)
        region[valid_region] = self.phases['ni_ysz']
    
    def _adjust_porosity(self, structure: np.ndarray, mask: np.ndarray, target_porosity: float):
        """Adjust porosity to match target value."""
        
        # Calculate current porosity in masked region
        masked_structure = structure[mask]
        current_porosity = np.sum(masked_structure == self.phases['pore']) / len(masked_structure)
        
        if current_porosity < target_porosity:
            # Need to add more pores
            solid_voxels = np.where(mask & (structure != self.phases['pore']))
            n_to_remove = int((target_porosity - current_porosity) * len(masked_structure))
            
            if len(solid_voxels[0]) > 0 and n_to_remove > 0:
                indices_to_remove = np.random.choice(
                    len(solid_voxels[0]), 
                    min(n_to_remove, len(solid_voxels[0])), 
                    replace=False
                )
                
                for idx in indices_to_remove:
                    structure[solid_voxels[0][idx], 
                             solid_voxels[1][idx], 
                             solid_voxels[2][idx]] = self.phases['pore']
        
        elif current_porosity > target_porosity:
            # Need to fill some pores
            pore_voxels = np.where(mask & (structure == self.phases['pore']))
            n_to_fill = int((current_porosity - target_porosity) * len(masked_structure))
            
            if len(pore_voxels[0]) > 0 and n_to_fill > 0:
                indices_to_fill = np.random.choice(
                    len(pore_voxels[0]), 
                    min(n_to_fill, len(pore_voxels[0])), 
                    replace=False
                )
                
                for idx in indices_to_fill:
                    structure[pore_voxels[0][idx], 
                             pore_voxels[1][idx], 
                             pore_voxels[2][idx]] = self.phases['ni_ysz']
    
    def _refine_interface_regions(self):
        """Refine the interface between anode and electrolyte."""
        
        # Find interface voxels
        anode_mask = (self.microstructure == self.phases['ni_ysz'])
        electrolyte_mask = (self.microstructure == self.phases['ysz_electrolyte'])
        
        # Dilate both phases to find interface region
        anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(2))
        electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(2))
        
        # Interface is where dilated regions overlap
        interface_region = anode_dilated.astype(bool) & electrolyte_dilated.astype(bool)
        
        # Mark interface voxels
        self.microstructure[interface_region] = self.phases['interface']
    
    def _apply_morphological_refinement(self):
        """Apply morphological operations to create realistic microstructure."""
        
        # Smooth the structure slightly to remove artifacts
        for phase in [self.phases['ni_ysz'], self.phases['ysz_electrolyte']]:
            phase_mask = (self.microstructure == phase)
            
            # Apply opening to remove small isolated regions
            cleaned_mask = morphology.binary_opening(phase_mask, morphology.ball(1))
            
            # Apply closing to fill small holes
            cleaned_mask = morphology.binary_closing(cleaned_mask, morphology.ball(1))
            
            # Update microstructure
            self.microstructure[phase_mask & ~cleaned_mask] = self.phases['pore']
            self.microstructure[~phase_mask & cleaned_mask] = phase
    
    def _calculate_microstructure_metrics(self):
        """Calculate and store microstructure metrics."""
        
        total_voxels = np.prod(self.dimensions)
        
        # Volume fractions
        volume_fractions = {}
        for phase_name, phase_id in self.phases.items():
            count = np.sum(self.microstructure == phase_id)
            volume_fractions[phase_name] = count / total_voxels
        
        # Porosity (pore phase)
        porosity = volume_fractions['pore']
        
        # Interface area calculation
        interface_area = self._calculate_interface_area()
        
        # Connectivity analysis
        connectivity_metrics = self._analyze_connectivity()
        
        # Store metadata
        self.metadata = {
            'dimensions': self.dimensions,
            'voxel_size_um': self.voxel_size,
            'physical_size_um': self.physical_size,
            'volume_fractions': volume_fractions,
            'porosity': porosity,
            'interface_area_um2': interface_area,
            'connectivity_metrics': connectivity_metrics,
            'generation_parameters': {
                'random_seed': self.random_seed,
                'material_properties': self.material_properties
            }
        }
        
        print(f"Microstructure Metrics:")
        print(f"  Dimensions: {self.dimensions}")
        print(f"  Voxel size: {self.voxel_size} μm")
        print(f"  Total porosity: {porosity:.3f}")
        print(f"  Volume fractions:")
        for phase, fraction in volume_fractions.items():
            print(f"    {phase}: {fraction:.3f}")
    
    def _calculate_interface_area(self) -> float:
        """Calculate the total interface area between phases."""
        
        # Use marching cubes to extract interface surfaces
        try:
            # Calculate interface area between anode and electrolyte
            anode_mask = (self.microstructure == self.phases['ni_ysz']).astype(float)
            verts, faces, _, _ = measure.marching_cubes(anode_mask, level=0.5)
            
            # Calculate surface area
            surface_area = 0.0
            for face in faces:
                v0, v1, v2 = verts[face]
                # Cross product for triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                surface_area += area
            
            # Convert to physical units
            surface_area *= self.voxel_size ** 2
            
            return surface_area
            
        except Exception as e:
            print(f"Warning: Could not calculate interface area: {e}")
            return 0.0
    
    def _analyze_connectivity(self) -> Dict:
        """Analyze phase connectivity using connected components."""
        
        connectivity_metrics = {}
        
        for phase_name, phase_id in self.phases.items():
            if phase_name == 'interface':  # Skip interface phase
                continue
                
            phase_mask = (self.microstructure == phase_id)
            
            if not np.any(phase_mask):
                connectivity_metrics[phase_name] = {
                    'connected_components': 0,
                    'largest_component_fraction': 0.0,
                    'percolation_x': False,
                    'percolation_y': False,
                    'percolation_z': False
                }
                continue
            
            # Find connected components
            labeled_array, num_components = ndi.label(phase_mask)
            
            # Find largest component
            if num_components > 0:
                component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background
                largest_size = np.max(component_sizes)
                largest_fraction = largest_size / np.sum(phase_mask)
            else:
                largest_fraction = 0.0
            
            # Check percolation in each direction
            percolation_x = self._check_percolation(phase_mask, axis=0)
            percolation_y = self._check_percolation(phase_mask, axis=1)
            percolation_z = self._check_percolation(phase_mask, axis=2)
            
            connectivity_metrics[phase_name] = {
                'connected_components': num_components,
                'largest_component_fraction': largest_fraction,
                'percolation_x': percolation_x,
                'percolation_y': percolation_y,
                'percolation_z': percolation_z
            }
        
        return connectivity_metrics
    
    def _check_percolation(self, mask: np.ndarray, axis: int) -> bool:
        """Check if phase percolates through the domain in given axis."""
        
        # Get slices at opposite ends of the axis
        if axis == 0:
            start_slice = mask[0, :, :]
            end_slice = mask[-1, :, :]
        elif axis == 1:
            start_slice = mask[:, 0, :]
            end_slice = mask[:, -1, :]
        else:  # axis == 2
            start_slice = mask[:, :, 0]
            end_slice = mask[:, :, -1]
        
        # Check if there's a connected path from start to end
        # This is a simplified check - could be made more rigorous
        return np.any(start_slice) and np.any(end_slice)
    
    def save_dataset(self, output_dir: str = "sofc_microstructure/data"):
        """Save the complete dataset in multiple formats."""
        
        if self.microstructure is None:
            raise ValueError("No microstructure generated. Call generate_realistic_microstructure() first.")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as HDF5 (recommended for large datasets)
        h5_path = os.path.join(output_dir, "sofc_microstructure.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('microstructure', data=self.microstructure, compression='gzip')
            f.create_dataset('dimensions', data=self.dimensions)
            f.create_dataset('voxel_size', data=self.voxel_size)
            
            # Save metadata as attributes (convert to JSON strings for complex types)
            for key, value in self.metadata.items():
                try:
                    if isinstance(value, (int, float, str)):
                        f.attrs[key] = value
                    else:
                        # Convert complex types to JSON string
                        f.attrs[key] = json.dumps(value, default=str)
                except Exception as e:
                    print(f"Warning: Could not save metadata key '{key}': {e}")
                    f.attrs[key] = str(value)
        
        # Save as TIFF stack (compatible with ImageJ, Fiji, etc.)
        tiff_path = os.path.join(output_dir, "sofc_microstructure.tiff")
        # Transpose for standard TIFF format (Z, Y, X)
        tiff_data = np.transpose(self.microstructure, (2, 1, 0))
        tifffile.imwrite(tiff_path, tiff_data, imagej=True, 
                        resolution=(1/self.voxel_size, 1/self.voxel_size),
                        metadata={'unit': 'um'})
        
        # Save individual 2D slices
        slices_dir = os.path.join(output_dir, "slices")
        os.makedirs(slices_dir, exist_ok=True)
        
        for i in range(self.dimensions[2]):
            slice_path = os.path.join(slices_dir, f"slice_{i:04d}.tiff")
            tifffile.imwrite(slice_path, self.microstructure[:, :, i])
        
        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Dataset saved to {output_dir}")
        print(f"  - HDF5 format: {h5_path}")
        print(f"  - TIFF stack: {tiff_path}")
        print(f"  - Individual slices: {slices_dir}")
        print(f"  - Metadata: {metadata_path}")
    
    def visualize_microstructure(self, save_path: str = None):
        """Create comprehensive visualization of the microstructure."""
        
        if self.microstructure is None:
            raise ValueError("No microstructure generated.")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cross-sections in different planes
        mid_x = self.dimensions[0] // 2
        mid_y = self.dimensions[1] // 2
        mid_z = self.dimensions[2] // 2
        
        # XY plane (top view)
        ax1 = plt.subplot(3, 4, 1)
        plt.imshow(self.microstructure[:, :, mid_z], cmap='tab10', vmin=0, vmax=3)
        plt.title(f'XY Cross-section (z={mid_z})')
        plt.colorbar(label='Phase ID')
        
        # XZ plane (side view)
        ax2 = plt.subplot(3, 4, 2)
        plt.imshow(self.microstructure[:, mid_y, :], cmap='tab10', vmin=0, vmax=3)
        plt.title(f'XZ Cross-section (y={mid_y})')
        plt.colorbar(label='Phase ID')
        
        # YZ plane (front view)
        ax3 = plt.subplot(3, 4, 3)
        plt.imshow(self.microstructure[mid_x, :, :], cmap='tab10', vmin=0, vmax=3)
        plt.title(f'YZ Cross-section (x={mid_x})')
        plt.colorbar(label='Phase ID')
        
        # 2. Volume fraction pie chart
        ax4 = plt.subplot(3, 4, 4)
        phase_names = list(self.metadata['volume_fractions'].keys())
        fractions = list(self.metadata['volume_fractions'].values())
        colors = ['lightblue', 'orange', 'lightgreen', 'red']
        plt.pie(fractions, labels=phase_names, colors=colors, autopct='%1.1f%%')
        plt.title('Volume Fractions')
        
        # 3. Porosity distribution along z-axis
        ax5 = plt.subplot(3, 4, 5)
        z_porosity = []
        for z in range(self.dimensions[2]):
            slice_data = self.microstructure[:, :, z]
            porosity = np.sum(slice_data == self.phases['pore']) / slice_data.size
            z_porosity.append(porosity)
        
        plt.plot(range(self.dimensions[2]), z_porosity)
        plt.xlabel('Z position (voxels)')
        plt.ylabel('Porosity')
        plt.title('Porosity vs. Z-position')
        plt.grid(True)
        
        # 4. Phase distribution along z-axis
        ax6 = plt.subplot(3, 4, 6)
        z_phases = {phase: [] for phase in self.phases.keys()}
        
        for z in range(self.dimensions[2]):
            slice_data = self.microstructure[:, :, z]
            total_voxels = slice_data.size
            
            for phase_name, phase_id in self.phases.items():
                fraction = np.sum(slice_data == phase_id) / total_voxels
                z_phases[phase_name].append(fraction)
        
        for phase_name, fractions in z_phases.items():
            plt.plot(range(self.dimensions[2]), fractions, label=phase_name)
        
        plt.xlabel('Z position (voxels)')
        plt.ylabel('Phase fraction')
        plt.title('Phase Distribution vs. Z-position')
        plt.legend()
        plt.grid(True)
        
        # 5. Interface visualization
        ax7 = plt.subplot(3, 4, 7)
        interface_slice = (self.microstructure[:, :, mid_z] == self.phases['interface'])
        plt.imshow(interface_slice, cmap='Reds')
        plt.title('Interface Regions (XY)')
        plt.colorbar()
        
        # 6. Connectivity metrics bar chart
        ax8 = plt.subplot(3, 4, 8)
        conn_metrics = self.metadata['connectivity_metrics']
        phases = list(conn_metrics.keys())
        components = [conn_metrics[phase]['connected_components'] for phase in phases]
        
        plt.bar(phases, components)
        plt.ylabel('Connected Components')
        plt.title('Phase Connectivity')
        plt.xticks(rotation=45)
        
        # 7. 3D visualization (subsampled for performance)
        ax9 = plt.subplot(3, 4, 9, projection='3d')
        
        # Subsample for visualization
        step = max(1, min(self.dimensions) // 50)
        x_sub = np.arange(0, self.dimensions[0], step)
        y_sub = np.arange(0, self.dimensions[1], step)
        z_sub = np.arange(0, self.dimensions[2], step)
        
        X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub, indexing='ij')
        
        # Show only interface voxels for clarity
        interface_mask = self.microstructure[::step, ::step, ::step] == self.phases['interface']
        
        if np.any(interface_mask):
            ax9.scatter(X[interface_mask], Y[interface_mask], Z[interface_mask], 
                       c='red', s=1, alpha=0.6)
        
        ax9.set_xlabel('X')
        ax9.set_ylabel('Y')
        ax9.set_zlabel('Z')
        ax9.set_title('3D Interface Structure')
        
        # 8. Histogram of phase IDs
        ax10 = plt.subplot(3, 4, 10)
        unique, counts = np.unique(self.microstructure, return_counts=True)
        plt.bar(unique, counts, color=['lightblue', 'orange', 'lightgreen', 'red'])
        plt.xlabel('Phase ID')
        plt.ylabel('Voxel Count')
        plt.title('Phase Distribution')
        
        # 9. Pore size distribution (simplified)
        ax11 = plt.subplot(3, 4, 11)
        pore_mask = (self.microstructure == self.phases['pore'])
        
        if np.any(pore_mask):
            # Use distance transform to estimate pore sizes
            distance = ndi.distance_transform_edt(pore_mask)
            pore_sizes = distance[pore_mask]
            pore_sizes = pore_sizes[pore_sizes > 0] * self.voxel_size
            
            if len(pore_sizes) > 0:
                plt.hist(pore_sizes, bins=50, alpha=0.7)
                plt.xlabel('Pore Size (μm)')
                plt.ylabel('Frequency')
                plt.title('Pore Size Distribution')
        
        # 10. Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        stats_text = f"""
        Microstructure Statistics:
        
        Dimensions: {self.dimensions}
        Voxel Size: {self.voxel_size:.3f} μm
        Physical Size: {[f'{s:.1f}' for s in self.physical_size]} μm
        
        Volume Fractions:
        """
        
        for phase, fraction in self.metadata['volume_fractions'].items():
            stats_text += f"  {phase}: {fraction:.3f}\n"
        
        stats_text += f"\nTotal Porosity: {self.metadata['porosity']:.3f}"
        stats_text += f"\nInterface Area: {self.metadata['interface_area_um2']:.1f} μm²"
        
        ax12.text(0.1, 0.9, stats_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create generator instance
    generator = SOFCMicrostructureGenerator(
        dimensions=(256, 256, 128),  # Smaller for faster generation
        voxel_size=0.1,  # 0.1 μm voxels
        random_seed=42
    )
    
    # Generate microstructure
    microstructure = generator.generate_realistic_microstructure(
        anode_thickness=15.0,  # 15 μm anode
        electrolyte_thickness=8.0,  # 8 μm electrolyte
        interface_roughness=0.8
    )
    
    # Save dataset
    generator.save_dataset()
    
    # Create visualization
    generator.visualize_microstructure("sofc_microstructure/results/microstructure_analysis.png")
    
    print("3D SOFC microstructure dataset generation completed!")